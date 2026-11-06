"""Videos of a batch of trajectories, rendered from their recorded states.

``uv run python scripts/scenes/render.py 000000 --physics isaac-cpu --gpus 0 1 2 3 [--jobs N] --cameras NAME ... [--merge] [--captions] [--count N] [--seeds ...] [--failed] [--input DIR]``

Reads the batch ``<input>/<scene>/<robot>/<expert>/<backend>/`` (``<input>`` = ``data/scenes``,
written by ``run_expert.py``; ``--batch DIR``: that batch, e.g. a development run) and writes, per episode, one video per camera
``videos/<seed>/<camera>.mp4`` (``fails/`` for failed episodes), or with
``--merge`` the cameras in one grid ``videos/<seed>/merge.mp4``. ``--cameras``
names robot cameras (RGB) and fixed views of the scene. A video of an episode
of T steps has T frames at 25 fps: frame t is the state before step t, the
observation that ``actions[t]`` answers. ``--captions`` adds a band above the
frames: the subtask (and, for a failed episode, the failure) on the left, the
time on the right.

The episodes are the batch's succeeded ones, or with ``--failed`` its failed
ones; ``--seeds`` (as in ``build_scene.py``) picks episodes, succeeded or failed
(only failed ones with ``--failed``), and ``--count N`` keeps the first N by
seed. A run replaces the videos of its episodes and keeps the others. Every frame
restores the recorded state of that step; nothing is simulated. The episodes are
packed by length into ``--jobs`` processes, spread over ``--gpus`` or submitted
through ``--launcher`` (see ``table_1000/common/batch.py``); episodes whose process
stopped before their videos run again (``--retries``). Logs are in
``outputs/scenes/<scene>/render/<batch hash>/``; ``summary.json`` lists the seeds
with videos.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Renders of the first frame, which follows another episode's last state: RT2 denoises over
# time and blends in what it rendered before. Later frames follow a state 40 ms earlier,
# close enough for one render (outputs/scenes/000000/experiments/render-once-vs-twice).
FIRST_FRAME_RENDERS = 4


def choose(succeeded, failed, seeds=None, count=None, failures=False):
    """The episodes to render: the succeeded ones (the failed ones with ``failures``), or those of
    ``seeds`` among them (among all episodes without ``failures``); the first ``count``.
    Raises ``ValueError`` naming the requested seeds without such an episode."""
    pool = set(failed) if failures else set(succeeded) | set(failed)
    if seeds is None:
        chosen = sorted(failed if failures else succeeded)
    else:
        missing = sorted(set(seeds) - pool)
        if missing:
            raise ValueError(missing)
        chosen = sorted(seeds)
    return chosen[:count] if count is not None else chosen


def camera_sets(scene, spec, names):
    """``(robot cameras, fixed views)`` of ``names``; raises ``ValueError`` for an unknown or depth camera."""
    cameras, views = {}, {}
    for name in names:
        if name in scene.VIEWS:
            views[name] = scene.VIEWS[name]
        elif name in spec.cameras and spec.cameras[name].sensor.kind == 'rgb':
            cameras[name] = spec.cameras[name]
        else:
            rgb = [n for n, camera in spec.cameras.items() if camera.sensor.kind == 'rgb']
            raise ValueError(f'{name} is not an RGB camera of the robot ({", ".join(rgb)}) '
                             f'or a view of the scene ({", ".join(scene.VIEWS)})')
    return cameras, views


def check_cameras(scene, robot, root, names):
    """Raise ``ValueError`` unless ``names`` are RGB cameras of ``robot`` or views of ``scene``."""
    from table_1000.robots import load_spec
    from table_1000.scenes import load
    camera_sets(load(scene, 'scene', root), load_spec(robot, root), names)



def main(argv=None):
    from table_1000.common.batch import add_arguments, job_count, parse_seeds, seed_range
    from table_1000.common.batch_runtime import run as run_program
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('scene')
    parser.add_argument('--robot', help='default scene robot, ignored with --batch')
    parser.add_argument('--expert', help='default smallest expert number, ignored with --batch')
    parser.add_argument('--layout', type=int, default=0)
    parser.add_argument('--physics', required=True, help='the backend the batch was generated on')
    parser.add_argument('--cameras', nargs='+', required=True, help='robot cameras (RGB) and views of the scene')
    parser.add_argument('--merge', action='store_true', help='one video with the cameras in a grid')
    parser.add_argument('--captions', action='store_true', help='subtask, failure and time above the frames')
    parser.add_argument('--failed', action='store_true', help='the failed episodes instead of the succeeded ones')
    parser.add_argument('--seeds', nargs='+', help='a:b (a to b - 1) and single seeds')
    parser.add_argument('--count', type=int, help='the first N of the chosen episodes by seed')
    add_arguments(parser)
    parser.add_argument('--input', type=Path, default=ROOT / 'data' / 'scenes',
                        help='the batch in <input>/<scene>/<robot>/<expert>/<backend>/, videos written next to it')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--assets', type=Path, default=ROOT / 'assets')
    parser.add_argument('--batch', type=Path, help='this batch directory instead of <input>/<scene>/<robot>/<expert>/<backend>/')
    args = parser.parse_args(argv)
    import h5py
    from table_1000.scenes.recordings import MERGED, video_path, videos
    job_count(parser, args)
    if args.count is not None and args.count < 1:
        parser.error('--count must be positive')
    if len(set(args.cameras)) != len(args.cameras):
        parser.error('--cameras lists a camera twice')
    from table_1000.scenes import load_expert, data_dir
    if args.batch:
        directory = args.batch.resolve()
    else:
        expert = load_expert(args.scene, args.robot, args.expert, args.root)
        directory = data_dir(args.input, args.scene, expert.ROBOT, expert.NUMBER, args.physics, args.layout).resolve()
    trajectories = directory / 'trajectories.hdf5'
    if not trajectories.is_file():
        parser.error(f'no trajectories {trajectories}; run run_expert.py first')
    root, args.assets = args.root.resolve(), args.assets.resolve()
    with h5py.File(trajectories, 'r') as file:
        env_args = json.loads(file['data'].attrs['env_args'])
    robot = env_args['robot']
    if env_args['scene'] != args.scene:
        parser.error(f'{directory} holds scene {env_args["scene"]}, not {args.scene}')
    try:
        check_cameras(args.scene, robot, root, args.cameras)
    except ValueError as error:
        parser.error(str(error))
    summary = json.loads((directory / 'summary.json').read_text())

    def episodes(path):
        """``{seed: steps}`` of a trajectory file."""
        if not path.is_file():
            return {}
        with h5py.File(path, 'r') as file:
            return {int(name.split('_')[1]): int(group.attrs['num_samples']) for name, group in file['data'].items()}

    succeeded, failed = episodes(trajectories), episodes(directory / 'trajectories_failed.hdf5')
    try:
        seeds = choose(succeeded, failed, parse_seeds(args.seeds) if args.seeds else None, args.count, args.failed)
    except ValueError as error:
        kind = 'failed episode' if args.failed else 'episode'
        parser.error(f'seeds {seed_range(error.args[0])} have no {kind}; the batch covers '
                     f'{summary["seeds"]["range"]}, failed: {seed_range(failed) or "none"}')
    if not seeds:
        parser.error('no episodes to render')
    # Per batch directory: renders of different batches run at once without sharing it.
    staging = root / 'outputs' / 'scenes' / args.scene / 'render' / hashlib.sha1(str(directory).encode()).hexdigest()[:12]
    if not args.dry_run:
        shutil.rmtree(staging, ignore_errors=True)
    targets = {seed: [video_path(directory, seed, None if output == MERGED else output, failed=seed in failed)
                      for output in ([MERGED] if args.merge else args.cameras)] for seed in seeds}
    if not args.dry_run:
        for path in (path for paths in targets.values() for path in paths):
            path.unlink(missing_ok=True)

    lengths = {**succeeded, **failed}
    missing = run_program('table_1000.scenes.render_program:program', vars(args) | dict(
        batch=directory, robot=robot, batch_key=staging.name, units=[(seed, lengths[seed]) for seed in seeds]),
        root=root, output=staging)
    if args.dry_run:
        return
    summary['render'] = missing.record(directory, patch_name='render-dirty.patch')
    for key, kinds in videos(directory).items():
        summary[key] = {kind: seed_range(values) for kind, values in kinds.items()}
    (directory / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=float) + '\n')
    print('VIDEOS', json.dumps({key: summary[key] for key in ('videos', 'fails')}), flush=True)
    if missing:
        raise SystemExit(f'no video: seeds {seed_range([unit[0] for unit in missing])}; see {staging}')


if __name__ == '__main__':
    main()
