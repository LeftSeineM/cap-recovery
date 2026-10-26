"""Robot acceptance in Isaac Sim: physical-sense cases, and comparison metrics on request.

``uv run python scripts/robots/check_robot.py x1pro x1pro_rm001 --physics isaac-cpu isaac-gpu --gpus 0 1 2``

The work units (each case of every robot and physics backend; each trial of ``grasp_and_push``, ``--trials``
default 0 to 12; ``parallel``, several GPU environments, on ``isaac-gpu`` only; ``metrics`` with
``--metrics``) are packed by their usual duration (``COST``) into ``--jobs`` processes that run at once:
spread over ``--gpus`` (default one per GPU), or each submitted through ``--launcher PREFIX``, a job
scheduler that picks its GPU and sets ``CUDA_VISIBLE_DEVICES`` (e.g. ``--launcher "srun --gres=gpu:1"``,
default 16 jobs). Units whose process died before writing their result run again (``--retries``, default
1); a case that ran and failed does not. ``--cases`` runs only those cases (``parallel`` and ``metrics``
by name too).

Results go to ``<assets>/robots/<robot>/check/`` (or ``<output>/<robot>/``): ``<backend>/<case>/`` (for
``grasp_and_push`` ``<backend>/grasp_and_push/t<trial>/``) holds the case's ``report.json`` and
``trajectory.hdf5``, the scene state at every step (videos: ``render_check.py``);
``<backend>/logs/`` the processes' logs; ``<backend>/report.json`` all cases of the backend;
``summary.json`` every case's status per backend. Results of cases not run this time are kept. The exit
status is 1 when a case does not pass or has no result. The robot's settings and cases come from
``asset_sources/robots/<name>/robot_test.py``; see ``table_1000/robots/checks.py``.
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

TRIALS = list(range(13))
BACKENDS = ('isaac-cpu', 'isaac-gpu')
# Usual duration of a unit on isaac-cpu (s; a grasp_and_push trial), for packing the processes. A process
# also spends about 80 s starting.
COST = {'keyframes': 5, 'sensors': 10, 'arm_hits_block': 20, 'arm_pushes_objects': 20, 'drop_on_robot': 120,
        'lift_into_shelf': 20, 'self_collision': 20, 'grasp_and_shake': 90, 'grasp_and_push': 300,
        'grasp_pressing_table': 140, 'base_into_wall': 10, 'base_full_turn': 20, 'drive_holding': 25,
        'joint_limits': 10, 'parallel': 40, 'metrics': 320}
GPU_COST = 1.5   # isaac-gpu over isaac-cpu


def unit_dir(backend_dir, case, trial):
    """Result directory of one case, or of one trial of ``grasp_and_push``."""
    return backend_dir / case / f't{trial}' if case == 'grasp_and_push' else backend_dir / case


def write_report(directory, report):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'report.json').write_text(json.dumps(report, indent=1, default=float))


def unit_name(case, trial):
    return case if trial is None else f'{case}:{trial}'


def parse_unit(name):
    case, _, trial = name.partition(':')
    return case, int(trial) if trial else None


def worker(args):
    """Run one process's units, each writing ``report.json`` and ``trajectory.hdf5`` to its directory."""
    from table_1000.common.batch import worker_gpu
    from table_1000.sim.isaaclab.app import start_app
    from table_1000.sim.isaaclab.render import RT2_ARGS
    started = time.perf_counter()
    backend = args.physics[0]
    device = 'cpu' if backend == 'isaac-cpu' else 'cuda:0'
    work = args.log.parent
    # Robot camera images (the sensors case) render with RTX Real-Time 2.0.
    app = start_app(gpu=worker_gpu(args.gpu), workers=4, device=device, extra_args=RT2_ARGS)
    print('startup_s', time.perf_counter() - started, flush=True)
    # parallel builds and drops environments of its own: before the checker's.
    units = sorted(map(parse_unit, args.units), key=lambda unit: unit[0] != 'parallel')
    failed, checker = False, None
    try:
        from table_1000.robots import load_robot_test, load_spec
        from table_1000.robots.checks import Checker
        from table_1000.robots.parallel import check_parallel
        from table_1000.sim.isaaclab import render
        from table_1000.sim.isaaclab.replay import write_episode
        render.configure()
        root, assets = args.root.resolve(), args.assets.resolve()
        robot = args.robots[0]
        spec, settings = load_spec(robot, root), load_robot_test(robot, root)
        for case, trial in units:
            directory = unit_dir(args.dir, case, trial)
            directory.mkdir(parents=True, exist_ok=True)
            case_started = time.perf_counter()
            try:
                if case == 'parallel':
                    result = check_parallel(spec, settings, assets, root)
                else:
                    if checker is None:
                        checker = Checker(spec, settings, work, assets, root, device=device)
                        print('load_s', checker.load_s, flush=True)
                    checker.output, checker.trials, checker.states = directory, (trial,), []
                    result = checker.run(case)
            except Exception:
                result = {'status': 'ERROR', 'error': traceback.format_exc()}
            result['wall_s'] = time.perf_counter() - case_started
            report = {'robot': robot, 'physics': backend, 'case': case, 'trial': trial,
                      'attempt': args.attempt, 'log': str(args.log), 'result': result}
            try:
                if checker is not None and checker.states and len(checker.states) > 1:
                    write_episode(directory / 'trajectory.hdf5', checker.states,
                                  {'robot': robot, 'physics': backend, 'case': case, 'trial': -1 if trial is None else trial})
                write_report(directory, report)
            except Exception:
                # The case's result still reaches the summary, as an error naming the failed write.
                result = report['result'] = {'status': 'ERROR', 'error': traceback.format_exc(), 'wall_s': result['wall_s']}
                write_report(directory, report)
            if checker is not None:
                checker.states = None
            failed |= case != 'metrics' and result.get('status') != 'PASS'
            print(case, trial, result.get('status'), f"{result['wall_s']:.0f} s", flush=True)
    except Exception:
        traceback.print_exc()
        failed = True
    if checker is not None:
        checker.env.close()
    # Closing ends the process: a failed case or run exits with status 1.
    app.close(exit_code=int(failed))


def merge(results):
    """``grasp_and_push`` from its trials: trials joined, a check passes when it passes in every trial."""
    merged = {'status': 'PASS' if all(r.get('status') == 'PASS' for r in results) else 'FAIL',
              'wall_s': sum(r.get('wall_s', 0.) for r in results), 'trials': {}, 'checks': {}}
    for result in results:
        merged['trials'].update(result.get('trials', {}))
        for key, value in result.get('checks', {}).items():
            merged['checks'][key] = merged['checks'].get(key, True) and value
        if 'error' in result:
            merged.setdefault('errors', []).append(result['error'])
    merged['trials'] = dict(sorted(merged['trials'].items(), key=lambda item: int(item[0])))
    return merged


def collect(directory, robot, ran=(), checkout=None, source=None):
    """Merge the case reports under ``directory`` (one robot) into ``<backend>/report.json`` and
    ``summary.json``; returns the summary. Results of units not in ``ran`` (``(backend, case, trial)``) are
    from earlier runs: listed under ``earlier`` per backend."""
    summary = {'robot': robot, 'commit': checkout, 'backends': {}, 'cases': {}}
    if source is not None:
        summary.update(source.record(directory))
    for backend in BACKENDS:
        cases = {}
        retried = {}
        for path in sorted((directory / backend).glob('*/report.json')):
            report = json.loads(path.read_text())
            cases[path.parent.name] = report['result']
            retried[path.parent.name] = report.get('attempt', 0)
        trials = [json.loads(path.read_text()) for path in sorted(
            (directory / backend / 'grasp_and_push').glob('t*/report.json'), key=lambda path: int(path.parent.name[1:]))]
        if trials:
            cases['grasp_and_push'] = merge([report['result'] for report in trials])
            retried['grasp_and_push'] = max(report.get('attempt', 0) for report in trials)
        if not cases:
            continue
        (directory / backend / 'report.json').write_text(json.dumps(
            {'robot': robot, 'physics': backend, 'cases': cases}, indent=1, default=float))
        graded = {name: result for name, result in cases.items() if name != 'metrics'}
        on_disk = [(path.parent.name, None) for path in (directory / backend).glob('*/report.json')] + [
            ('grasp_and_push', int(path.parent.name[1:])) for path in (directory / backend / 'grasp_and_push').glob('t*/report.json')]
        summary['backends'][backend] = {
            'passed': sum(r.get('status') == 'PASS' for r in graded.values()), 'cases': len(graded),
            'failed': sorted(name for name, r in graded.items() if r.get('status') != 'PASS'),
            'earlier': sorted(unit_name(case, trial) for case, trial in on_disk if (backend, case, trial) not in ran)}
        for name, result in graded.items():
            summary['cases'].setdefault(name, {})[backend] = {
                'status': result.get('status'), 'wall_s': result.get('wall_s'), 'retried': retried[name],
                'failed_checks': sorted(key for key, value in result.get('checks', {}).items() if not value)}
        if 'metrics' in cases:
            summary.setdefault('metrics', {})[backend] = cases['metrics']
    (directory / 'summary.json').write_text(json.dumps(summary, indent=1, default=float) + '\n')
    return summary


def main(argv=None):
    from table_1000.common.batch import add_arguments, job_count, process_name, run_batch
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('robots', nargs='+', help='robot names (asset_sources/robots/<name>)')
    parser.add_argument('--physics', nargs='+', choices=BACKENDS, required=True, help='physics backends to run')
    parser.add_argument('--cases', nargs='+', help='only these cases (default: every case of robot_test.py)')
    parser.add_argument('--trials', type=int, nargs='+', default=TRIALS,
                        help='grasp_and_push trials (0: at the work point; k > 0: random placement, seed k)')
    parser.add_argument('--metrics', action='store_true', help='also the comparison metrics')
    add_arguments(parser)
    parser.add_argument('--output', type=Path, help='results in <output>/<robot>/ (default <assets>/robots/<robot>/check/)')
    parser.add_argument('--root', type=Path, default=ROOT, help='repository root holding the robot sources')
    parser.add_argument('--assets', type=Path, default=ROOT / 'assets', help='asset root holding the built robots')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--dir', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--units', nargs='+', help=argparse.SUPPRESS)
    parser.add_argument('--log', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--attempt', type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker(args)
        return
    jobs = job_count(parser, args)
    from table_1000.robots import load_robot_test, load_spec
    from table_1000.robots.checks import select_cases
    root, assets = args.root.resolve(), args.assets.resolve()
    directories, units = {}, []
    for robot in args.robots:
        directories[robot] = args.output.resolve() / robot if args.output else assets / 'robots' / robot / 'check'
        known = [*select_cases(load_spec(robot, root), load_robot_test(robot, root)), 'parallel']
        unknown = [case for case in args.cases or [] if case not in [*known, 'metrics']]
        if unknown:
            parser.error(f'{", ".join(unknown)}: not a case of {robot} ({", ".join(known)}, metrics)')
        cases = args.cases or known
        if args.metrics and 'metrics' not in cases:
            cases = [*cases, 'metrics']
        for backend in args.physics:
            for case in cases:
                if case == 'parallel' and backend != 'isaac-gpu':
                    continue
                for trial in (args.trials if case == 'grasp_and_push' else [None]):
                    units.append((robot, backend, case, trial))
    if not units:
        parser.error('nothing to run')

    def directory(unit):
        robot, backend, case, trial = unit
        return unit_dir(directories[robot] / backend, case, trial)

    def command(group, index, attempt):
        robot, backend = group[0][:2]
        name = process_name(f'part{index}', attempt)
        log = directories[robot] / backend / 'logs' / name / 'run.log'
        argv = [sys.executable, str(Path(__file__).resolve()), robot, '--worker', '--physics', backend,
                '--dir', str(directories[robot] / backend), '--units', *(unit_name(*unit[2:]) for unit in group),
                '--log', str(log), '--attempt', str(attempt), '--root', str(root), '--assets', str(assets)]
        return argv, log, f'{robot} {backend} {name}'

    for unit in units:
        shutil.rmtree(directory(unit), ignore_errors=True)
    started = time.perf_counter()
    missing = run_batch(units, key=lambda unit: unit[:2],
                        cost=lambda unit: COST.get(unit[2], 30) * (GPU_COST if unit[1] == 'isaac-gpu' else 1),
                        count=jobs, command=command, missing=lambda unit: not (directory(unit) / 'report.json').exists(),
                        gpus=args.gpus, launcher=args.launcher, retries=args.retries)
    for unit in missing:
        # Its process exited before this unit's result, in every attempt.
        write_report(directory(unit), {'case': unit[2], 'trial': unit[3], 'attempt': args.retries, 'result': {
            'status': 'ERROR', 'error': f'no result after {args.retries + 1} attempts; see '
                                        f'{directories[unit[0]] / unit[1] / "logs"}'}})
    failing = []
    for robot, robot_dir in directories.items():
        summary = collect(robot_dir, robot, {unit[1:] for unit in units if unit[0] == robot}, source=missing)
        print(robot, {backend: f"{entry['passed']}/{entry['cases']}" for backend, entry in summary['backends'].items()},
              robot_dir / 'summary.json', flush=True)
        for backend, entry in summary['backends'].items():
            if entry['earlier']:
                print(f'  {backend}: {len(entry["earlier"])} results from earlier runs kept (summary.json: earlier)', flush=True)
        failing += [f'{robot} {backend} {name}' for backend, entry in summary['backends'].items()
                    for name in entry['failed']]
    print(f'total {time.perf_counter() - started:.0f} s', flush=True)
    if failing:
        raise SystemExit('failed: ' + ', '.join(failing))


if __name__ == '__main__':
    main()
