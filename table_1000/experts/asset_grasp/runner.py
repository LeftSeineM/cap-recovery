"""Asset-grasp workers and one shared batch/publication pipeline."""

from importlib.metadata import version
import json
from pathlib import Path
import runpy
import shutil
import sys
import time

from table_1000.common.batch import commit, run_batch
from table_1000.common.diagnostics import dump_stacks_every
from table_1000.experts.asset_grasp import (inventory, summarize, SHAKE_AMPLITUDE, SHAKE_SECONDS,
    SLIP_LIMIT, WORK_OFFSET, START_PROJECTION_TOLERANCE)
from table_1000.experts.asset_grasp.results import prepare, require_mode, publish, video_seed
from table_1000.experts.asset_grasp.stability import (TRANSPORT, ROTATION_LIMIT, TRANSLATION_LIMIT,
    MOTION_LIMITS, MOTION_TOLERANCE)
from table_1000.experts.grasps.checks import GAP_TOLERANCE
from table_1000.scenes import load
from table_1000.robots import load_robot_test
from table_1000.sim import PHYSICS_DT, CONTROL_HZ


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.part')
    temporary.write_text(json.dumps(value, indent=2, default=float, ensure_ascii=False) + '\n')
    temporary.replace(path)


def trial_path(output, asset, seed):
    return output / asset / 'trials' / f'{seed:04d}.json'


def worker(args, root):
    from table_1000.common.batch import Load, worker_gpu
    from table_1000.sim.isaaclab.app import start_app
    from table_1000.sim.isaaclab.render import RT2_ARGS
    device = 'cpu' if args.physics == 'isaac-cpu' else 'cuda:0'
    cameras = args.video or args.render_worker
    app = start_app(gpu=worker_gpu(args.gpu), workers=4, device=device, cameras=cameras,
                    extra_args=RT2_ARGS if cameras else ())
    evaluator = None
    try:
        from table_1000.experts.asset_grasp import Evaluator
        from table_1000.experts.asset_grasp.recording import Recorder, VIEWS, render
        from table_1000.sim.isaaclab.video import encode
        from table_1000.sim.isaaclab.render import configure
        if cameras:
            configure()
        for asset in args.asset:
            try:
                evaluator = Evaluator(asset, args.robot, args.assets, args.work, root, device,
                                      VIEWS if cameras else {}, diagnostic_shake=args.diagnostic_shake)
                for seed in args.seeds:
                    dump_stacks_every()
                    result = args.result or args.output / asset
                    if args.render_worker:
                        frames = args.work / f'frames-{seed:04d}'
                        render(evaluator, result / 'trajectories' / f'{seed:04d}.hdf5',
                               args.render_output or result / 'videos' / f'{seed:04d}.mp4', frames)
                        shutil.rmtree(frames)
                        continue
                    if trial_path(args.output, asset, seed).exists():
                        continue
                    frames = args.work / f'frames-{seed:04d}' if args.video else None
                    recorder = Recorder(evaluator.env, frames)
                    with Load() as load:
                        report = evaluator.trial(seed, args.center, args.region_size, recorder)
                    report['load'] = load.record()
                    report.update(robot=args.robot, physics=args.physics)
                    trajectory = result / 'trajectories' / f'{seed:04d}.hdf5'
                    recorder.save(trajectory, {'asset': asset, 'seed': seed, 'robot': args.robot,
                                               'physics': args.physics})
                    report['trajectory'] = str(trajectory.relative_to(result))
                    if args.video:
                        video = result / 'videos' / f'{seed:04d}.mp4'
                        video.parent.mkdir(parents=True, exist_ok=True)
                        encode(frames, 25, video)
                        report['video'] = str(video.relative_to(result))
                        shutil.rmtree(frames)
                    write_json(trial_path(args.output, asset, seed), report)
                    print(asset, seed, report['status'], report.get('failure', {}).get('reason', ''), flush=True)
            finally:
                if evaluator is not None:
                    evaluator.close()
                    evaluator = None
    finally:
        error = sys.exc_info()
        if error[0] is not None:
            import traceback
            traceback.print_exception(*error)
        app.close(exit_code=int(error[0] is not None))


def worker_command(root, assets, robot, physics, work, asset, seeds, **paths):
    command = [sys.executable, str(root / 'scripts/experts/eval_asset_grasp.py'), '--worker',
               '--asset', asset, '--robot', robot, '--physics', physics, '--assets', str(assets),
               '--work', str(work), '--seeds', *map(str, seeds)]
    for name, path in paths.items():
        command += ['--' + name.replace('_', '-'), str(path)]
    return command


def render_run(args, root, jobs):
    result = args.result.resolve()
    summary = json.loads((result / 'summary.json').read_text())
    output = args.output.resolve()
    if output.is_relative_to(args.assets.resolve()):
        raise ValueError('复核录像请写入 runs/ 或 outputs/；资产验收目录只允许正式运行发布')
    work = output.parent / (output.stem + '-work')
    output.unlink(missing_ok=True)

    def command(group, index, attempt):
        directory = work / f'attempt-{attempt}'
        cmd = worker_command(root, args.assets.resolve(), summary['robot'], summary['physics'],
                             directory, summary['asset'], [args.seed], result=result, render_output=output)
        cmd.append('--render-worker')
        return cmd, directory / 'run.log', f'render {summary["asset"]} seed {args.seed}'

    missing = run_batch([args.seed], command=command, missing=lambda seed: not output.exists(),
                        count=jobs, gpus=args.gpus, launcher=args.launcher, retries=args.retries)
    if missing:
        raise RuntimeError(f'no rendered video; see {work}')



def run(args, root, jobs, error):
    if args.output:
        error('--output 只供内部工作进程使用；开发运行请用 --dev [--name 标签]')
    if args.config is None and not args.asset:
        error('需要 --config（按配置测一个场景的资产）或 --asset（直接测指定资产）')
    if args.config is None and not args.robot:
        error('不用 --config 时需要 --robot')
    config = runpy.run_path(str(args.config)) if args.config else {'TRIALS': 20}
    args.trials = args.trials if args.trials is not None else config.get('TRIALS', 20)
    require_mode(dev=args.dev, asset=args.asset, seeds=args.seeds, video=args.video, label=args.name,
                 trials=args.trials, configured_trials=config.get('TRIALS', 20))
    if args.trials < 1 or args.region_size <= 0:
        error('--trials and --region-size must be positive')
    if args.seeds_per_worker is not None and args.seeds_per_worker < 1:
        error('--seeds-per-worker must be positive')
    args.robot = args.robot or config['ROBOT']
    if args.config:
        items = inventory(load(config['SCENE']), config.get('EXCLUDED_INSTANCES', {}))
        configured = config.get('ASSETS', [asset for asset, item in items.items() if not item['excluded']])
        selected = args.asset or configured
        if not selected or set(selected) - items.keys():
            error('select at least one asset from the configured scene')
    else:
        items, selected = {}, sorted(set(args.asset))
        missing_assets = [asset for asset in selected if not (args.assets / asset).is_dir()]
        if missing_assets:
            error(f'assets not built under {args.assets}: {", ".join(missing_assets)}')
    seeds = sorted(set(args.seeds if args.seeds is not None else range(args.trials)))
    if not seeds or min(seeds) < 0:
        error('seeds must be non-negative')
    checkout = commit(root)
    run, output = prepare(root, dev=args.dev, label=args.name, checkout=checkout)
    assets = args.assets.resolve()
    conditions = {'scene': config.get('SCENE'), 'robot': args.robot, 'physics': args.physics,
                  'versions': {package: version(package) for package in ('isaacsim', 'isaaclab', 'mplib')},
                  'run_name': run, 'dev': args.dev,
                  'config': str(args.config.resolve().relative_to(root)) if args.config else None,
                  'config_text': args.config.read_text() if args.config else None, 'selected_assets': selected,
                  'seeds': seeds, 'region_size_m': args.region_size,
                  'seeds_per_worker': args.seeds_per_worker,
                  'center_xy_m': args.center or [a + b for a, b in zip(load_robot_test(args.robot).WORK, WORK_OFFSET)],
                  'physics_dt_s': PHYSICS_DT, 'external_force': False,
                  'diagnostic_shake': args.diagnostic_shake,
                  'diagnostic_shake_amplitude_rad': SHAKE_AMPLITUDE, 'diagnostic_shake_segment_s': SHAKE_SECONDS,
                  'transport_envelope': TRANSPORT, 'motion_limits': MOTION_LIMITS,
                  'motion_tolerance': MOTION_TOLERANCE, 'rotation_limit_rad': float(ROTATION_LIMIT),
                  'translation_limit_m': TRANSLATION_LIMIT,
                  'slip_limit_m': SLIP_LIMIT, 'control_hz': CONTROL_HZ,
                  'candidate_policy': 'shared within-part order; gravity load selects current real-part heads; whole envelope fallback',
                  'candidate_identifiers': {
                      'single_target': 'raw closing/approach/slide/hand-turn geometry order, before angle/cost sorting',
                      'part_heads': 'original per-part full cost stream index; head blocking unchanged'},
                  'closure_gap_tolerance_m': GAP_TOLERANCE,
                  'closure_only_record': True,
                  'transport_preflight': 'fixed lift; mixed TCP translation and rotation; single local IK continuation; strongest collision-free complete arm stroke',
                  'transport_translation_axes': '24 horizontal directions; vertical or 45-degree upward diagonal preserving 50 mm height',
                  'transport_acceptance': 'measured holding only; TCP strength is informational and never excludes a trial',
                  'transport_joint_clearance': 'every sampled arm command keeps the existing 0.01 rad arrival tolerance from hard stops; otherwise choose another path or relocate',
                  'transport_companion': 'joint-bounded instantaneous acceleration allocation; zero-companion alternative',
                  'transport_time_law': 'TOPPRA outward stroke targeting 95% TCP envelope; stop then retrace',
                  'transport_rotation_axes': '25 unoriented TCP axes, both signs; strongest complete path',
                  'transport_relocation': {
                      'reference': 'work centre at nominal pick height with measured orientation; declared test keyframe fallback',
                      'planning': 'existing collision-checked RRT; preflight reference and measured arrival',
                      'grip_reference': 'unchanged through relocation and acceptance',
                      'arrival_tracking': 'record failure; require complete preflight and acceptance from measured pose'},
                  'transport_start_projection': {
                      'tolerance': START_PROJECTION_TOLERANCE,
                      'units': 'radians for revolute joints, metres for prismatic joints',
                      'origin': 'FK of the projected start; target joint limits remain unchanged'},
                  'excluded_instances': config.get('EXCLUDED_INSTANCES', {}), 'inventory': items}
    write_json(output / 'conditions.json', conditions)
    print('run:', output, flush=True)

    def missing(unit):
        return not trial_path(output, *unit).exists()

    def command(group, index, attempt):
        work = output / 'workers' / f'{index}-attempt{attempt}'
        asset = group[0][0]
        cmd = worker_command(root, assets, args.robot, args.physics, work, asset,
                             [seed for _, seed in group], output=output, region_size=args.region_size)
        if args.center:
            cmd += ['--center', *map(str, args.center)]
        if args.video:
            cmd.append('--video')
        if args.diagnostic_shake:
            cmd.append('--diagnostic-shake')
        return cmd, work / 'run.log', f'asset grasp part {index}'

    started = time.perf_counter()
    units = [(asset, seed) for asset in selected for seed in seeds]
    batch = run_batch(units, key=lambda unit: (unit[0], unit[1] // args.seeds_per_worker)
              if args.seeds_per_worker is not None else unit[0], count=jobs, command=command, missing=missing,
              gpus=args.gpus, launcher=args.launcher, retries=args.retries, root=root)
    conditions.update(batch.record(output))
    write_json(output / 'conditions.json', conditions)
    overall = {**conditions, 'assets': {}, 'wall_s': time.perf_counter() - started}
    trials, summaries = {}, {}
    for asset in selected:
        paths = [trial_path(output, asset, seed) for seed in seeds]
        reports = [json.loads(path.read_text()) for path in paths if path.exists()]
        trials[asset] = reports
        summary = summaries[asset] = {**conditions, 'asset': asset, **summarize(reports, len(seeds))}
        write_json(output / asset / 'summary.json', summary)
        overall['assets'][asset] = {key: value for key, value in summary.items() if key not in conditions}
        overall['assets'][asset]['results'] = str((output / asset).relative_to(root))
    if not args.dev:
        # A grasp failure is valid; a missing/error trajectory cannot replace acceptance results.
        for asset, reports in trials.items():
            if len(reports) != len(seeds):
                raise RuntimeError(f'{asset}: missing reports')
            for report in reports:
                if report['status'] == 'error':
                    raise RuntimeError(f'{asset} seed {report["seed"]}: {report["failure"]}')
                if not (output / asset / report['trajectory']).is_file():
                    raise FileNotFoundError(output / asset / report['trajectory'])
        chosen = {asset: video_seed(reports) for asset, reports in trials.items()}

        def render_command(group, index, attempt):
            asset = group[0]
            work = output / 'workers' / f'render-{index}-attempt{attempt}'
            cmd = worker_command(root, assets, args.robot, args.physics, work, asset,
                                 [chosen[asset]], output=output)
            cmd.append('--render-worker')
            return cmd, work / 'run.log', f'acceptance video {asset}'

        missing_videos = run_batch(selected, key=lambda asset: asset, count=jobs, command=render_command,
                                  missing=lambda asset: not (output / asset / 'videos' / f'{chosen[asset]:04d}.mp4').exists(),
                                  gpus=args.gpus, launcher=args.launcher, retries=args.retries, root=root, source=batch)
        if missing_videos:
            raise RuntimeError(f'missing acceptance videos: {missing_videos}')
        if conditions['commit'] != checkout or 'dirty_patch' in conditions:
            raise RuntimeError('正式批次冻结来源不是启动时的干净提交，未发布验收结果')
        for asset in selected:
            summary_path = output / asset / 'summary.json'
            summary = summaries[asset]
            summary['video'] = {'seed': chosen[asset], 'path': f'videos/{chosen[asset]:04d}.mp4',
                                'selection': 'first_failure_else_smallest_seed', 'rendered_from_trajectory': True}
            write_json(summary_path, summary)
            destination = publish(output / asset, assets, asset, args.robot, args.physics)
            overall['assets'][asset]['results'] = str(destination.relative_to(assets))
    write_json(output / 'summary.json', overall)
    lines = ['# Asset grasp results', '', '| Asset | Picked | First stable / assessed | Not evaluated | Rate | Failures |',
             '| --- | ---: | ---: | ---: | ---: | --- |']
    for asset, entry in overall['assets'].items():
        reasons = ', '.join(f'{k}: {v}' for k, v in entry['failures'].items())
        rate = f"{entry['rate']:.0%}" if entry['rate'] is not None else 'unassessed'
        lines.append(f"| {asset} | {entry['picked']}/{len(seeds)} | {entry['first_attempt_success']}/{entry['assessed']} | {entry['not_evaluated']} | {rate} | {reasons} |")

    (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    if any(r['status'] != 'PASS' for r in overall['assets'].values()):
        raise SystemExit(1)
