"""Asset-grasp workers and one shared batch/publication pipeline."""

from importlib.metadata import version
import json

from table_1000.common.serialization import dumps
import runpy
import shutil
import time

from table_1000.common.batch import commit, parse_seeds
from table_1000.common.batch_runtime import run as run_program
from table_1000.experts.evals.asset_grasp import inventory, summarize, SHAKE_AMPLITUDE, SHAKE_SECONDS, SLIP_LIMIT, WORK_OFFSET, START_PROJECTION_TOLERANCE
from table_1000.experts.evals.asset_grasp.results import require_mode, publish, video_seed
from table_1000.experts.evals.asset_grasp.stability import TRANSPORT, ROTATION_LIMIT, TRANSLATION_LIMIT, MOTION_LIMITS, MOTION_TOLERANCE
from table_1000.experts.skills.hold.checks import GAP_TOLERANCE
from table_1000.scenes import load, batch_ref
from table_1000.robots import load_robot_test
from table_1000.sim import PHYSICS_DT, CONTROL_HZ


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.part')
    temporary.write_text(dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def trial_path(output, asset, seed):
    return output / asset / 'trials' / f'{seed:04d}.json'


def render_run(args, root):
    args.assets = args.assets.resolve()
    result = args.result.resolve()
    summary = json.loads((result / 'summary.json').read_text())
    output = args.output.resolve()
    if output.is_relative_to(args.assets.resolve()):
        raise ValueError('复核录像请写入 runs/ 或 outputs/；资产验收目录只允许正式运行发布')
    work = output.parent / (output.stem + '-work')
    if not args.dry_run:
        output.unlink(missing_ok=True)
        shutil.rmtree(work, ignore_errors=True)
    missing = run_program('table_1000.experts.evals.asset_grasp.render_program:program',
        vars(args) | dict(robot=summary['robot'], physics=summary['physics'], units=[
            (summary['asset'], args.seed, str(result / 'trajectories' / f'{args.seed:04d}.hdf5'), str(output))]),
        root=root, output=work)
    if missing:
        raise RuntimeError(f'no rendered video; see {work}')


def run(args, root, error):
    args.assets = args.assets.resolve()
    if args.config is None and not args.items:
        error('需要 --config（按配置测一个场景的资产）或 --items（直接测指定资产）')
    if args.config is None and not args.robot:
        error('不用 --config 时需要 --robot')
    config = runpy.run_path(str(args.config)) if args.config else {'TRIALS': 20}
    args.trials = args.trials if args.trials is not None else config.get('TRIALS', 20)
    require_mode(dev=args.dev, asset=args.items, seeds=args.seeds, video=args.video, label=args.name,
                 trials=args.trials, configured_trials=config.get('TRIALS', 20))
    if args.trials < 1 or args.region_size <= 0:
        error('--trials and --region-size must be positive')
    if args.seeds_per_worker is not None and args.seeds_per_worker < 1:
        error('--seeds-per-worker must be positive')
    args.robot = args.robot or config['ROBOT']
    if args.config:
        items = inventory(load(config['SCENE']), config.get('EXCLUDED_INSTANCES', {}))
        configured = config.get('ASSETS', [asset for asset, item in items.items() if not item['excluded']])
        selected = args.items or configured
        if not selected or set(selected) - items.keys():
            error('select at least one asset from the configured scene')
    else:
        items, selected = {}, sorted(set(args.items))
        missing_assets = [asset for asset in selected if not (args.assets / asset).is_dir()]
        if missing_assets:
            error(f'assets not built under {args.assets}: {", ".join(missing_assets)}')
    seeds = parse_seeds(args.seeds) if args.seeds is not None else list(range(args.trials))
    if not seeds or min(seeds) < 0:
        error('seeds must be non-negative')
    checkout = commit(root)
    if args.output:
        run, output = args.output.name, args.output.resolve()
    else:
        from table_1000.common.batch import run_name
        run = run_name(checkout, args.name)
        output = root / ('runs/grasp' if args.dev else 'outputs/asset-grasp') / run
    units = [(asset, seed) for asset in selected for seed in seeds]
    options = vars(args) | dict(units=units)
    if args.dry_run:
        run_program('table_1000.experts.evals.asset_grasp.program:program', options, root=root, output=output)
        return
    if not args.dev and (checkout is None or checkout.endswith('+dirty')):
        raise ValueError('正式运行要求干净的 Git 提交；未提交的开发运行请加 --dev')
    if (output / 'conditions.json').exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
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
                  'collect_preferences': args.collect_preferences,
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

    started = time.perf_counter()
    batch = run_program('table_1000.experts.evals.asset_grasp.program:program', options, root=root, output=output)
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
        overall['assets'][asset]['results'] = batch_ref(output / asset, root, root)
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

        missing_videos = run_program('table_1000.experts.evals.asset_grasp.render_program:program',
            options | dict(units=[(asset, chosen[asset],
                str(output / asset / 'trajectories' / f'{chosen[asset]:04d}.hdf5'),
                str(output / asset / 'videos' / f'{chosen[asset]:04d}.mp4')) for asset in selected]),
            root=root, output=output / 'render', source=batch)
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
