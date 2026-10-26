"""Videos of robot acceptance cases, rendered from the trajectories ``check_robot.py`` recorded.

``uv run python scripts/robots/render_check.py x1pro --physics isaac-cpu --cases grasp_and_push --trials 3``

For each chosen case (``--cases``, default all with a trajectory; ``--trials`` picks ``grasp_and_push``
trials) of each robot and backend, reads ``<backend>/<case>/trajectory.hdf5`` from the results directory
of ``check_robot.py`` (same ``--output``) and writes one video per view next to it, ``<view>.mp4``:
``--views`` names fixed views of the check scene (``wide``, the default) and RGB robot cameras. Frame t
restores the recorded state before step t (25 fps, the case's name and time above it); nothing is
simulated. Processes as in ``check_robot.py``: ``--jobs`` over ``--gpus`` or through ``--launcher``, ``--retries``.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.robots.check_robot import BACKENDS, parse_unit, unit_dir, unit_name  # noqa: E402

# Renders of the first frame (RT2 denoises over time), as scripts/scenes/render.py.
FIRST_FRAME_RENDERS = 4


def worker(args):
    """Render this process's units."""
    from table_1000.common.batch import worker_gpu
    from table_1000.sim.isaaclab.app import start_app
    from table_1000.sim.isaaclab.render import RT2_ARGS
    app = start_app(gpu=worker_gpu(args.gpu), workers=4, extra_args=RT2_ARGS)
    failed = False
    try:
        from PIL import Image
        from table_1000.robots import load_robot_test, load_spec
        from table_1000.robots.checks import VIEWS, check_scene
        from table_1000.sim import CONTROL_HZ
        from table_1000.sim.isaaclab import render, replay
        from table_1000.sim.isaaclab.env import RobotEnv
        from table_1000.sim.isaaclab.video import caption, encode, save_frame
        render.configure()
        root = args.root.resolve()
        robot = args.robots[0]
        spec = load_spec(robot, root)
        views = {name: VIEWS[name] for name in args.views if name in VIEWS}
        cameras = {name: spec.cameras[name] for name in args.views if name not in VIEWS}
        env = RobotEnv(check_scene(spec, load_robot_test(robot, root)), spec, args.assets.resolve(),
                       cameras=cameras, views=views, root=root)
        for case, trial in map(parse_unit, args.units):
            try:
                started = time.perf_counter()
                directory = unit_dir(args.dir, case, trial)
                file, episodes = replay.episodes(directory / 'trajectory.hdf5')
                episode = next(iter(episodes.values()))
                frames = directory / 'frames'
                shutil.rmtree(frames, ignore_errors=True)
                for view in args.views:
                    (frames / view).mkdir(parents=True)
                title = case if trial is None else f'{case}  trial {trial}'
                for t in range(episode.length):
                    replay.set_state(env, episode.state(t))
                    for _ in range(FIRST_FRAME_RENDERS - 1 if t == 0 else 0):
                        env.render([])
                    rendered = env.render(args.views)
                    for view in args.views:
                        image = caption(Image.fromarray(rendered[view]['rgb'][0]), title, f'{t / CONTROL_HZ:.2f} s')
                        save_frame(image, frames / view / f'{t:05d}.jpg')
                file.close()
                for view in args.views:
                    encode(frames / view, CONTROL_HZ, directory / f'{view}.mp4')
                shutil.rmtree(frames)
                print(unit_name(case, trial), episode.length, f'{time.perf_counter() - started:.0f} s', flush=True)
            except Exception:
                # A bad trajectory leaves the other units of the process to render.
                traceback.print_exc()
                failed = True
        env.close()
    except Exception:
        traceback.print_exc()
        failed = True
    app.close(exit_code=int(failed))


def main(argv=None):
    from table_1000.common.batch import add_arguments, job_count, process_name, run_batch
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('robots', nargs='+', help='robot names (asset_sources/robots/<name>)')
    parser.add_argument('--physics', nargs='+', choices=BACKENDS, required=True, help='backends the cases ran on')
    parser.add_argument('--cases', nargs='+', help='only these cases (default: every case with a trajectory)')
    parser.add_argument('--trials', type=int, nargs='+', help='only these grasp_and_push trials')
    parser.add_argument('--views', nargs='+', default=['wide'], help='fixed views (wide) and RGB robot cameras')
    add_arguments(parser)
    parser.add_argument('--output', type=Path, help='results in <output>/<robot>/ (default <assets>/robots/<robot>/check/)')
    parser.add_argument('--root', type=Path, default=ROOT, help='repository root holding the robot sources')
    parser.add_argument('--assets', type=Path, default=ROOT / 'assets', help='asset root holding the built robots')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--dir', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--units', nargs='+', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker(args)
        return
    jobs = job_count(parser, args)
    import h5py
    from table_1000.robots import load_spec
    from table_1000.robots.checks import VIEWS
    root, assets = args.root.resolve(), args.assets.resolve()
    directories, units, lengths = {}, [], {}
    for robot in args.robots:
        spec = load_spec(robot, root)
        rgb = [name for name, camera in spec.cameras.items() if camera.sensor.kind == 'rgb']
        unknown = [name for name in args.views if name not in VIEWS and name not in rgb]
        if unknown:
            parser.error(f'{", ".join(unknown)}: not a view ({", ".join(VIEWS)}) or RGB camera ({", ".join(rgb)}) of {robot}')
        directories[robot] = args.output.resolve() / robot if args.output else assets / 'robots' / robot / 'check'
        for backend in args.physics:
            for path in sorted((directories[robot] / backend).glob('**/trajectory.hdf5')):
                relative = path.parent.relative_to(directories[robot] / backend).parts
                case, trial = relative[0], int(relative[1][1:]) if len(relative) > 1 else None
                if (args.cases and case not in args.cases) or (trial is not None and args.trials
                                                               and trial not in args.trials):
                    continue
                unit = (robot, backend, case, trial)
                with h5py.File(path, 'r') as file:
                    lengths[unit] = int(file['data/demo_0'].attrs['num_samples'])
                units.append(unit)
    if not units:
        parser.error('no recorded trajectory of the chosen cases; run check_robot.py first')

    def command(group, index, attempt):
        robot, backend = group[0][:2]
        name = process_name(f'render{index}', attempt)
        log = directories[robot] / backend / 'logs' / name / 'run.log'
        argv = [sys.executable, str(Path(__file__).resolve()), robot, '--worker', '--physics', backend,
                '--dir', str(directories[robot] / backend), '--units', *(unit_name(*unit[2:]) for unit in group),
                '--views', *args.views, '--root', str(root), '--assets', str(assets)]
        return argv, log, f'{robot} {backend} {name}'

    def video(unit):
        robot, backend, case, trial = unit
        return unit_dir(directories[robot] / backend, case, trial) / f'{args.views[-1]}.mp4'

    for unit in units:
        for view in args.views:
            video(unit).with_name(f'{view}.mp4').unlink(missing_ok=True)
    started = time.perf_counter()
    missing = run_batch(units, key=lambda unit: unit[:2], cost=lambda unit: lengths[unit], count=jobs,
                        command=command, missing=lambda unit: not video(unit).exists(), gpus=args.gpus,
                        launcher=args.launcher, retries=args.retries)
    for robot in dict.fromkeys(unit[0] for unit in units):
        directory = directories[robot]
        path = directory / 'summary.json'
        summary = json.loads(path.read_text())
        summary['render'] = missing.record(directory, patch_name='render-dirty.patch')
        path.write_text(json.dumps(summary, indent=1, default=float) + '\n')
    for unit in units:
        if unit not in missing:
            print(*unit[:3], *([] if unit[3] is None else [unit[3]]), video(unit).parent, flush=True)
    print(f'total {time.perf_counter() - started:.0f} s', flush=True)
    if missing:
        raise SystemExit('no video: ' + ', '.join(' '.join(map(str, unit)) for unit in missing))


if __name__ == '__main__':
    main()
