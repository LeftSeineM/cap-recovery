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
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.robots.check_robot import BACKENDS, unit_dir  # noqa: E402

# Renders of the first frame (RT2 denoises over time), as scripts/scenes/render.py.
FIRST_FRAME_RENDERS = 4


def main(argv=None):
    from table_1000.common.batch import add_arguments, job_count
    from table_1000.common.batch_runtime import run as run_program
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
    args = parser.parse_args(argv)
    job_count(parser, args)
    import h5py
    from table_1000.robots import load_spec
    from table_1000.robots.checks import VIEWS
    root, assets = args.root.resolve(), args.assets.resolve()
    args.assets = assets
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

    def video(unit):
        robot, backend, case, trial = unit
        return unit_dir(directories[robot] / backend, case, trial) / 'trajectory.hdf5'

    if not args.dry_run:
        for unit in units:
            for view in args.views:
                video(unit).with_name(f'{view}.mp4').unlink(missing_ok=True)
    started = time.perf_counter()
    missing = run_program('table_1000.robots.batch_program:renders', vars(args) | dict(units=units,
        directories=directories, lengths=[lengths[u] for u in units]), root=root,
        output=root / 'runs/robot-render' / str(time.time_ns()))
    if args.dry_run:
        return
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
