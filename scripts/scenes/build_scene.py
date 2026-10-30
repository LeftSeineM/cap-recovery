"""Initial states of a scene: sampled and checked per seed, stored for the expert to start from.

``uv run python scripts/scenes/build_scene.py 000000 --seeds 0:100 --gpus 0 1 2 3 [--jobs N] [--grid 5x5] [--images] [--camera NAME] [--output DIR]``
``uv run python scripts/scenes/build_scene.py 000100 --layouts 1:11 --seeds 0 [--grid 5x2] ...``

Select --robot NAME (default scene.ROBOT), or --robot all to build every ROBOTS entry.
For each layout and seed, makes the initial state as in ``table_1000/scenes/initial.py``
(the layout's arrangement: ``scene.initialize`` for layout 0, ``scene.LAYOUTS[n]['arrange']``
for layout n; penetration and settling checks, retried with sub-seeds) on CPU PhysX.
``--layouts`` takes numbers like ``--seeds`` (default 0); every layout is built for every seed.
The seeds of each layout are split into ``--jobs`` contiguous parts, one process each, spread
over ``--gpus`` or submitted through ``--launcher`` (see ``table_1000/common/batch.py``); each
seed's result is written when it is done, and seeds whose process stopped before theirs run
again (``--retries``), as do seeds that raised an error (the error may be the process's: GPU memory,
a lost device) and seeds without a valid state after an error in their process. Logs are in ``outputs/scenes/<scene>/<robot>/build/``; the processes' results go to
``<output>/<scene>/<robot>/staging/``, removed once merged. Into ``<output>/<scene>/<robot>/``
(``<output>`` = ``assets/scenes`` by default), replacing what an earlier run wrote:

- ``initial_states.hdf5``: the settled state of every seed with one, its ``seed`` and ``attempt``;
- ``initial.json``: the summary (seeds, valid and invalid, mean attempts, settling motion,
  robot contacts) and every attempt of every seed with its problems;
- ``initial_states.jpg`` with ``--grid NxM``: the first N × M valid states in N columns;
- ``initial/<seed>.jpg`` with ``--images``: every valid state at the camera's resolution.

That is layout 0; layout n goes to ``<output>/<scene>/<robot>/layouts/<n>/``, the same files (the summary
also with the layout's ``quality`` and ``description``). With ``--grid`` and several layouts,
``layouts.jpg`` shows the smallest valid seed of each, N per row.

Images come from ``--camera`` (default: the scene's ``IMAGE_VIEW``, else the robot's first
RGB camera; or another fixed view of the scene). When images are rendered into the default output, the
smallest valid seed's image also becomes ``asset_sources/scenes/<scene>/thumbnail.jpg``
(480 px wide), from layout 0 of the default robot. The exit status is non-zero when a seed has no result, an error or no valid state.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TILE_WIDTH, CAPTION, THUMBNAIL_WIDTH = 384, 18, 480
PRODUCTS = ('initial_states.hdf5', 'initial.json', 'initial_states.jpg', 'initial')


def grid_size(text):
    columns, rows = map(int, text.lower().split('x'))
    if columns < 1 or rows < 1:
        raise argparse.ArgumentTypeError('--grid takes NxM with positive N and M')
    return columns, rows


def main_camera(spec):
    """The robot's first RGB camera."""
    return next(name for name, camera in spec.cameras.items() if camera.sensor.kind == 'rgb')


def worker(args):
    """Initial states of ``args.seeds`` in layout ``args.layout`` in this process, into ``args.work``,
    per seed as it is done: ``states/<seed>.hdf5`` (a valid state), ``images/<seed>.jpg`` (all valid
    seeds with ``--images``, else the first ``--first`` valid ones), then ``records/<seed>.json``, the
    seed's result."""
    work = args.work.resolve()
    for name in ('states', 'images', 'records'):
        (work / name).mkdir(parents=True, exist_ok=True)
    render = args.images or args.first > 0
    from table_1000.common.batch import worker_gpu
    from table_1000.sim.isaaclab.app import start_app
    from table_1000.sim.isaaclab.render import RT2_ARGS
    app = start_app(gpu=worker_gpu(args.gpu), workers=4, cameras=render, extra_args=RT2_ARGS if render else ())
    failed = False
    try:
        from PIL import Image
        from table_1000.robots import load_spec
        from table_1000.scenes import load, robot_config, initial_dir
        from table_1000.common.batch import parse_seeds
        from table_1000.scenes.initial import InitializationFailed, initial_state, write_states
        from table_1000.sim.isaaclab.env import RobotEnv
        root, assets = args.root.resolve(), args.assets.resolve()
        scene = load(args.scene, 'scene', root)
        spec = load_spec(Path(robot_config(scene, args.robot)['asset']).name, root)
        camera = None
        kwargs = {}
        if render:
            from table_1000.sim.isaaclab import render as rendering
            rendering.configure()
            camera = args.camera or getattr(scene, 'IMAGE_VIEW', None) or main_camera(spec)
            if camera in scene.VIEWS:
                kwargs['views'] = {camera: scene.VIEWS[camera]}
            else:
                kwargs['cameras'] = {camera: spec.cameras[camera]}
        env = RobotEnv(scene, spec, assets, root=root, **kwargs)
        pictured = 0

        def save(record):
            # Written whole or not at all: the file marks the seed as done.
            path = work / 'records' / f'{record["seed"]}.json'
            path.with_suffix('.tmp').write_text(
                json.dumps({'robot': spec.name, 'camera': camera, 'record': record}, indent=1, default=float))
            os.replace(path.with_suffix('.tmp'), path)

        erred = None      # the last seed that raised an error in this process
        arrange = scene.LAYOUTS[args.layout]['arrange'] if args.layout else None
        for seed in parse_seeds(args.seeds):
            record = None
            try:
                _, record = initial_state(env, scene, assets, seed, initialize=arrange)
                write_states(work / 'states' / f'{seed}.hdf5',
                             {seed: (record['attempt'], to_numpy(env.env.scene.get_state(is_relative=True)))}, {})
                if args.images or pictured < args.first:
                    # RT2 denoises over time: the first render after a state change blends in the last one.
                    env.render([camera])
                    Image.fromarray(env.render([camera])[camera]['rgb'][0]).save(work / 'images' / f'{seed}.jpg',
                                                                                quality=90)
                    pictured += 1
                print(seed, 'attempt', record['attempt'], flush=True)
            except InitializationFailed as error:
                record = {'seed': seed, 'attempt': None, 'attempts': error.attempts}
                if erred is not None:
                    # The error may have broken the process: the seed runs again in a new one.
                    record['after_error'] = erred
                print(seed, 'no valid state', flush=True)
            except Exception:
                # Kept as this seed's result if it errs again when retried in a new process.
                traceback.print_exc()
                erred = seed
                (work / 'states' / f'{seed}.hdf5').unlink(missing_ok=True)
                record = {'seed': seed, 'attempt': None, 'attempts': record['attempts'] if record else [],
                          'error': traceback.format_exc().strip().splitlines()[-1]}
            save(record)
        env.close()
    except Exception:
        traceback.print_exc()
        failed = True
    app.close(exit_code=int(failed))


def to_numpy(tree):
    return {key: to_numpy(value) if isinstance(value, dict) else value.cpu().numpy() for key, value in tree.items()}


def grid(images, captions, columns, path):
    """``images`` (paths) captioned in a grid of ``columns``, each ``TILE_WIDTH`` wide."""
    from PIL import Image, ImageDraw
    first = Image.open(images[0])
    height = round(first.height * TILE_WIDTH / first.width)
    rows = -(-len(images) // columns)
    canvas = Image.new('RGB', (columns * TILE_WIDTH, rows * (height + CAPTION)), 'white')
    draw = ImageDraw.Draw(canvas)
    for index, (image, caption) in enumerate(zip(images, captions)):
        x, y = (index % columns) * TILE_WIDTH, (index // columns) * (height + CAPTION)
        canvas.paste(Image.open(image).convert('RGB').resize((TILE_WIDTH, height), Image.LANCZOS), (x, y + CAPTION))
        draw.text((x + 4, y + 3), caption, fill='black')
    canvas.save(path, quality=90)


def thumbnail(image, path):
    from PIL import Image
    picture = Image.open(image)
    picture.resize((THUMBNAIL_WIDTH, round(picture.height * THUMBNAIL_WIDTH / picture.width)),
                   Image.LANCZOS).save(path, quality=85)


def main(argv=None):
    from table_1000.common.batch import add_arguments, job_count, parse_seeds, process_name, run_batch, seed_range, seeds_record
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('scene')
    parser.add_argument('--robot', help='supported robot name, default scene.ROBOT; all builds every robot')
    parser.add_argument('--seeds', nargs='+', required=True, help='a:b (a to b - 1) and single seeds')
    parser.add_argument('--layouts', nargs='+', default=['0'],
                        help="layouts, like --seeds: 0 is scene.initialize, n is scene.LAYOUTS[n] (default 0)")
    add_arguments(parser)
    parser.add_argument('--grid', type=grid_size, help='NxM: the first N x M valid states of each layout in N columns')
    parser.add_argument('--images', action='store_true', help='one image per valid state')
    parser.add_argument('--camera', help="a robot camera or a view of the scene (default: the scene's IMAGE_VIEW, else the robot's first RGB camera)")
    parser.add_argument('--output', type=Path, help='results go to <output>/<scene>/<robot>/ (default: assets/scenes)')
    parser.add_argument('--root', type=Path, default=ROOT, help='repository root holding the scene and robot sources')
    parser.add_argument('--assets', type=Path, default=ROOT / 'assets')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--layout', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--work', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--first', type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker(args)
        return
    from table_1000.scenes import load, robot_config, initial_dir
    from table_1000.scenes.initial import read_states, summarize, write_states
    jobs = job_count(parser, args)
    seeds, layouts = parse_seeds(args.seeds), parse_seeds(args.layouts)
    if not seeds:
        parser.error('no seeds')
    root = args.root.resolve()
    scene = load(args.scene, root=root)
    if args.robot == 'all':
        failures = []
        original = list(sys.argv[1:] if argv is None else argv)
        index = next(i for i, argument in enumerate(original) if argument == '--robot' or argument.startswith('--robot='))
        for robot in scene.ROBOTS:
            command = original.copy()
            if command[index] == '--robot':
                command[index + 1] = robot
            else:
                command[index] = f'--robot={robot}'
            try:
                main(command)
            except SystemExit as error:
                failures.append(f'{robot}: {error}')
        if failures:
            raise SystemExit('; '.join(failures))
        return
    robot_config(scene, args.robot)
    args.robot = args.robot or scene.ROBOT
    defined = getattr(load(args.scene, 'scene', root), 'LAYOUTS', {}) if any(layouts) else {}
    unknown = [layout for layout in layouts if layout and layout not in defined]
    if unknown:
        parser.error(f'layouts {seed_range(unknown)} not in the scene\'s LAYOUTS ({seed_range(defined) or "none"})')
    output = initial_dir(args.output or args.assets / 'scenes', args.scene, args.robot).resolve()
    staging, logs = output / 'staging', root / 'outputs' / 'scenes' / args.scene / args.robot / 'build'

    def result(item):
        """The latest result of ``(layout, seed)`` and the process directory holding it, or ``(None, None)``."""
        for work in reversed(parts[item]):
            if (work / 'records' / f'{item[1]}.json').is_file():
                return json.loads((work / 'records' / f'{item[1]}.json').read_text()), work
        return None, None

    def rerun(item):
        done, _ = result(item)
        return not done or 'error' in done['record'] or 'after_error' in done['record']

    def named(layout, group):
        return (f'layout {layout} ' if layout else '') + f'seeds {seed_range(group)}'

    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(logs, ignore_errors=True)
    first = args.grid[0] * args.grid[1] if args.grid else 0
    parts = {}

    def command(part, index, attempt):
        # A part holds seeds of one layout (run_batch keys items by layout).
        layout, part_seeds = part[0][0], [seed for _, seed in part]
        name = process_name(f'part{index}', attempt)
        for item in part:
            parts.setdefault(item, []).append(staging / name)
        argv = [sys.executable, str(Path(__file__).resolve()), args.scene, '--worker', '--robot', args.robot, '--layout', str(layout),
                '--seeds', *seed_range(part_seeds).split(), '--work', str(staging / name), '--first', str(first),
                '--root', str(root), '--assets', str(args.assets.resolve())]
        argv += ['--images'] if args.images else []
        argv += ['--camera', args.camera] if args.camera else []
        return argv, logs / f'{name}.log', f'{args.scene} {name} {named(layout, part_seeds)}'

    started = time.perf_counter()
    items = [(layout, seed) for layout in layouts for seed in seeds]
    batch = run_batch(items, command=command, count=jobs, gpus=args.gpus, launcher=args.launcher,
                      retries=args.retries, missing=rerun, key=lambda item: item[0], root=root)
    missing, checkout = set(batch), batch.commit
    total_s = time.perf_counter() - started

    problems, overview = [], []
    for layout in layouts:
        records, states, images, robot, camera = {}, {}, {}, None, None
        for seed in seeds:
            done, work = result((layout, seed))
            if not done:
                continue
            robot, camera, records[seed] = done['robot'], done['camera'], done['record']
            if (work / 'states' / f'{seed}.hdf5').is_file():
                states.update(read_states(work / 'states' / f'{seed}.hdf5')[1])
            if (work / 'images' / f'{seed}.jpg').is_file():
                images[seed] = work / 'images' / f'{seed}.jpg'
        for seed in seeds:
            records.setdefault(seed, {'seed': seed, 'attempt': None, 'attempts': [], 'problems': ['process exited']})
        directory = output / 'layouts' / str(layout) if layout else output
        directory.mkdir(parents=True, exist_ok=True)
        for name in PRODUCTS:
            path = directory / name
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        write_states(directory / 'initial_states.hdf5', states,
                     {'scene': args.scene, 'robot': robot or '', 'physics': 'isaac-cpu', 'commit': checkout or '',
                      'layout': layout})
        about = {key: defined[layout][key] for key in ('quality', 'description')} if layout else {}
        summary = {'scene': args.scene, 'layout': layout, **about, 'robot': robot, 'physics': 'isaac-cpu',
                   **batch.record(directory), 'seeds': seeds_record(seeds), **summarize(records), 'camera': camera,
                   'total_s': total_s}
        (directory / 'initial.json').write_text(json.dumps({**summary, 'records': dict(sorted(records.items()))},
                                                           indent=1, default=float) + '\n')
        valid = sorted(seed for seed in images if records[seed]['attempt'] is not None)
        if args.grid and valid:
            shown = valid[:first]
            grid([images[seed] for seed in shown], [f'seed {seed}  attempt {records[seed]["attempt"]}' for seed in shown],
                 args.grid[0], directory / 'initial_states.jpg')
            overview.append((images[valid[0]], f'layout {layout}' + (f' ({about["quality"]})' if about else '')
                             + f'  seed {valid[0]}'))
        if args.images and valid:
            (directory / 'initial').mkdir()
            for seed in valid:
                shutil.copy(images[seed], directory / 'initial' / f'{seed}.jpg')
        if valid and args.output is None and layout == 0 and args.robot == scene.ROBOT:
            thumbnail(images[valid[0]], root / 'asset_sources' / 'scenes' / args.scene / 'thumbnail.jpg')
        print(json.dumps(summary, indent=1, default=float))
        lost = {seed for seed in seeds if (layout, seed) in missing}
        errors = {seed for seed, record in records.items() if 'error' in record}
        without = set(summary['without_state']) - lost
        problems += [f'{kind}: {named(layout, group)} (see {logs})'
                     for kind, group in (('no result', lost - errors), ('errors', errors)) if group]
        problems += [(f'layout {layout} ' if layout else '') + f'seeds without a valid state: {seed_range(without)}'] if without else []
    if len(overview) > 1:
        grid(*zip(*overview), args.grid[0], output / 'layouts.jpg')
    shutil.rmtree(staging, ignore_errors=True)
    if problems:
        raise SystemExit('; '.join(problems))


if __name__ == '__main__':
    main()
