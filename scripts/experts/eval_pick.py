"""Pick-skill coverage on X1Pro: random trials or given placements, with optional videos.

``uv run python scripts/experts/eval_pick.py --robots x1pro x1pro_rm001 --scenarios open box side back corner --seeds 0 2 --trials 40 --launcher "srun --gres=gpu:1" --output outputs/grasp-exp/eval``
``uv run python scripts/experts/eval_pick.py --case cube1 0.62 0.37 0 standing --video --output ...``

A group is one robot (``--robots``) in one scenario (``--scenarios``: ``open`` the bare table,
``box`` the object in an open box, ``side``/``back``/``corner`` walls beside and behind it) with
one seed (``--seeds``); its ``--trials`` random trials cycle through the test objects
(``table_1000/experts/evals/pick_eval.py``), each placed by the seed and its index. ``--case NAME X Y
YAW POSE`` (world frame: table centre at the origin, robot on the -Y side) runs given placements
in the open scenario instead.

Every trial of every group is a work unit; the units of one robot and scenario share a process
kind (one environment); local concurrency follows ``--gpus``, or all jobs are submitted through
``--launcher PREFIX``, a job scheduler that picks its GPU and
sets ``CUDA_VISIBLE_DEVICES`` (e.g. ``--launcher "srun --gres=gpu:1"``, default auto from timing history). Each
trial writes ``<robot>/<scenario>-s<seed>/trials/<index>.json``; trials whose process died before
that run again (``--retries``, default 1). Each group gets ``summary.json`` (success by object,
failure reasons, unreachable placements) and ``trials.jsonl``; ``--output`` gets ``summary.json``
of all groups. ``--video`` records every trial as ``videos/<index>_<object>_<status>.mp4`` in its
group (a wide and a close view side by side).
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from table_1000.common.serialization import dumps

WIDE = {'position': (1.5, -1.45, 1.6), 'target': (0., -.1, .8), 'focal_length_mm': 18, 'resolution': (960, 720)}
SCENARIOS = ('open', 'box', 'side', 'back', 'corner')
CLOSE = {'position': (.5, .6, 1.2), 'target': (0., 0., .8), 'focal_length_mm': 24, 'resolution': (960, 720)}


def tasks(args, scenario, seed):
    """``[(index, name, placement)]`` of a group."""
    import numpy as np
    from table_1000.experts.evals.pick_eval import OBJECTS, sample
    if args.case:
        return [(index, name, {'x': float(x), 'y': float(y), 'yaw': float(yaw), 'pose': pose})
                for index, (name, x, y, yaw, pose) in enumerate(args.case)]
    names = args.objects or list(OBJECTS)
    return [(index, names[index % len(names)], sample(np.random.default_rng([seed, index]), names[index % len(names)],
                                                         scenario=scenario))
            for index in range(args.trials)]


def group_dir(output, robot, scenario, seed, case=False):
    return output / robot / ('cases' if case else f'{scenario}-s{seed}')


class Recorder:
    """Every step's frame of the env's views side by side, captioned, saved under ``frames``."""

    def __init__(self, env, views, frames):
        self.env, self.views, self.frames, self.count = env, list(views), Path(frames), 0
        shutil.rmtree(self.frames, ignore_errors=True)
        self.frames.mkdir(parents=True)

    def capture(self, text):
        from PIL import Image
        from table_1000.sim.isaaclab.video import caption, save_frame, tile
        rendered = self.env.render(self.views)
        images = [Image.fromarray(rendered[name]['rgb'][0]) for name in self.views]
        save_frame(caption(tile(images, images[0].height), text), self.frames / f'{self.count:05d}.jpg')
        self.count += 1


def trial_path(directory, index):
    return directory / 'trials' / f'{index:04d}.json'


def aim_close_view(env, placement):
    """Point the close camera at the object, from the side its walls leave open."""
    import torch
    side = 1. if placement['x'] >= 0 else -1.
    target = [placement['x'], placement['y'], .80]
    eye = [placement['x'] + side * .35, placement['y'] + .30, 1.05]
    obstacles = placement.get('obstacles', {})
    if obstacles:
        if 'wall_side' in obstacles:
            side = -1. if obstacles['wall_side'][0][3] > placement['x'] else 1.
        eye = [placement['x'] + side * .3, placement['y'] - .12, 1.15]
    env.env.scene['close'].set_world_poses_from_view(torch.tensor([eye], dtype=torch.float32),
                                                     torch.tensor([target], dtype=torch.float32))


def summarize(reports):
    import numpy as np
    by_object = {}
    for report in reports:
        entry = by_object.setdefault(f"{report['object']} {report['pose']}", {'trials': 0, 'success': 0, 'failures': {}})
        entry['trials'] += 1
        if report['status'] == 'success':
            entry['success'] += 1
        else:
            reason = report['status'] if report['status'] == 'slipped' else report['failure']['reason']
            entry['failures'][reason] = entry['failures'].get(reason, 0) + 1
    for entry in by_object.values():
        entry['rate'] = entry['success'] / entry['trials']
    success = [r for r in reports if r['status'] == 'success']
    # Placements without a feasible grasp: the skill fails them before moving.
    unreachable = [r for r in reports if r['status'] == 'failed' and r['failure']['reason'] == 'no reachable grasp']
    reachable = len(reports) - len(unreachable)
    return {'trials': len(reports), 'success': len(success), 'rate': len(success) / len(reports) if reports else None,
            'unreachable': len(unreachable),
            'reachable_rate': len(success) / reachable if reachable else None,
            'by_object': dict(sorted(by_object.items())),
            'max_slip_m': max([r['hold']['max_slip_m'] for r in success], default=None),
            'max_penetration_m': max([r['max_penetration_m'] for r in reports], default=None),
            # Trials whose fingers went more than 0.5 mm into the object.
            'finger_penetration_trials': sum(any('robot' in pair and r['object'] in pair for pair in r['penetration'])
                                             for r in reports),
            'mean_sim_s': float(np.mean([r['sim_s'] for r in success])) if success else None,
            'mean_wall_s': float(np.mean([r['wall_s'] for r in reports])) if reports else None}


def main(argv=None):
    from table_1000.common.batch import add_arguments, job_count
    from table_1000.common.batch_runtime import run as run_program
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--robots', nargs='+', default=['x1pro'], help='robot names (asset_sources/robots/<name>)')
    parser.add_argument('--scenarios', nargs='+', default=['open'], choices=SCENARIOS, help='obstacles around the object')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--trials', type=int, default=0, help='random trials per group')
    parser.add_argument('--objects', nargs='+', help='test objects of the random trials (default: all)')
    parser.add_argument('--select', type=int, nargs='+', help='run only these trial indices (to re-run or record some)')
    parser.add_argument('--case', nargs=5, action='append', metavar=('NAME', 'X', 'Y', 'YAW', 'POSE'))
    parser.add_argument('--gap', type=float, default=.12, help='robot base to table edge (m)')
    parser.add_argument('--lift', type=float, default=.6, help='starting torso lift (m)')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--random-pushes', type=int, default=8,
                        help='random push directions after the six axes (0: a quicker run)')
    parser.add_argument('--assets', type=Path, default=ROOT / 'assets', help='asset root holding robots/<name>')
    add_arguments(parser)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', action='store_true', help='record nested wall/CPU times and available instructions')
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    args.assets = args.assets.resolve()
    if args.case and (args.scenarios != ['open'] or len(args.seeds) > 1):
        parser.error('--case places objects in the open scenario, once')
    job_count(parser, args)
    units = [(robot, scenario, seed, index) for robot in args.robots for scenario in args.scenarios
             for seed in args.seeds for index, _, _ in tasks(args, scenario, seed)
             if args.select is None or index in args.select]
    if not units:
        parser.error('give --trials or --case')
    if not args.dry_run:
        shutil.rmtree(args.output, ignore_errors=True)
    def directory(unit):
        return group_dir(args.output, *unit[:3], bool(args.case))
    started = time.perf_counter()
    missing = run_program('table_1000.experts.evals.pick_program:program', vars(args) | dict(units=units),
                          root=ROOT, output=args.output)
    if args.dry_run:
        return
    groups = {}
    for unit in units:
        groups.setdefault(directory(unit), []).append(unit)
    overall = {**missing.record(args.output), 'gap': args.gap, 'lift': args.lift, 'random_pushes': args.random_pushes,
               'groups': {}}
    for path, members in groups.items():
        reports = [json.loads(trial_path(path, unit[3]).read_text()) for unit in members if unit not in missing]
        reports.sort(key=lambda report: report['index'])
        with open(path / 'trials.jsonl', 'w') as out:
            for report in reports:
                out.write(dumps(report) + '\n')
        robot, scenario, seed = members[0][:3]
        summary = {'robot': robot, 'scenario': scenario, 'seed': seed, **summarize(reports)}
        (path / 'summary.json').write_text(dumps(summary, indent=1) + '\n')
        overall['groups'][str(path.relative_to(args.output))] = {
            key: summary[key] for key in ('trials', 'success', 'unreachable', 'reachable_rate', 'rate')}
    overall['total_s'] = time.perf_counter() - started
    (args.output / 'summary.json').write_text(dumps(overall, indent=1) + '\n')
    for name, entry in overall['groups'].items():
        print(f"{name:28s} {entry['success']}/{entry['trials']}  unreachable {entry['unreachable']}  "
              f"reachable success {entry['reachable_rate'] if entry['reachable_rate'] is None else round(entry['reachable_rate'], 3)}")
    print(f"total {overall['total_s']:.0f} s")
    if missing:
        raise SystemExit(f'no report of {len(missing)} trials; see {args.output / "workers"}')


if __name__ == '__main__':
    main()
