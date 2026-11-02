"""Save failed episodes as checkpoints at an evaluated skill's entry.

Default: the innermost skill on the failure path; a failed step check selects its
last skill and runs the check after replay. --frame selects a frame by number,
resolving control frames to a concrete descendant. Development snapshots go to runs/expert;
--case NAME explicitly retains a selected case in data/cases/<scene>/<robot>/<expert>/<NAME>.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from table_1000.common.serialization import dumps


def main(argv=None):
    import h5py
    from table_1000.common.batch import commit, parse_seeds
    from table_1000.experts.generate import checkpoint_path, read_checkpoints, INITIAL_STATE
    from table_1000.experts.teleop import failed_step, start_checkpoint, trajectory_state
    from table_1000.scenes.initial import write_tree
    from table_1000.scenes import case_dir, dev_dir, batch_ref
    from table_1000.experts.identity import check_identity
    from table_1000.sim import CONTROL_HZ
    from scripts.scenes.run_expert import run_name
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('batch', type=Path)
    parser.add_argument('--seeds', nargs='+', help='a:b (a to b - 1) and single seeds (default: every failed one)')
    parser.add_argument('--frame', type=int, help='frame number (control frames resolve to their last concrete skill)')
    parser.add_argument('--name', help='label for the development snapshot batch')
    parser.add_argument('--case', help='save a selected durable case under data/cases with this name')
    parser.add_argument('--output', type=Path, help='output root (default runs/expert, or data/cases with --case)')
    args = parser.parse_args(argv)
    batch = args.batch.resolve()
    summary = json.loads((batch / 'summary.json').read_text())
    check_identity(summary, summary['robot'], summary['expert'], batch)
    if args.case:
        from table_1000.experts.audit.records import export_case
        selected = parse_seeds(args.seeds) if args.seeds else [
            int(path.stem) for path in (batch / 'episodes').glob('*.json')
            if json.loads(path.read_text())['status'] == 'failed']
        if len(selected) != 1:
            parser.error('--case needs exactly one failed seed; select it with --seeds')
        directory = case_dir(args.output or ROOT / 'data/cases', summary['scene'], summary['robot'], summary['expert'], args.case)
        export_case([batch], selected[0], directory)
        print('CASE', directory)
        return
    path = batch / 'trajectories_failed.hdf5'
    if not path.is_file():
        parser.error(f'{batch} has no failed trajectories')
    wanted = set(parse_seeds(args.seeds)) if args.seeds else None
    step_names = {}
    snapshots = defaultdict(dict)       # step k -> {seed: (attempt, state, start, record)}
    with h5py.File(path, 'r') as file:
        for name, group in file['data'].items():
            seed = int(name.split('_')[1])
            report_path = batch / 'episodes' / f'{seed}.json'
            if wanted is not None and seed not in wanted or not report_path.is_file():
                continue
            report = json.loads(report_path.read_text())
            step = failed_step(report)
            if step is None:
                continue
            length = int(group.attrs['num_samples'])
            if not report.get('call_trace'):
                parser.error('current calls schema required; convert old batches outside the recorder')
            from table_1000.experts.audit.trace import snapshot_state
            scene_state, snapshot = snapshot_state(batch, report, args.frame)
            step = snapshot['step']
            index = snapshot['frame']['enter'] - 1
            own = start_checkpoint(batch, summary, step)
            found = read_checkpoints(own)[1] if own.is_file() else {}
            if seed in found:
                start = found[seed][2]
            else:
                obs = group['obs']
                start = {'qpos': obs['qpos'][0], 'base_pose': obs['base_pose'][0]}
            record = {'batch': str(batch), 'sample': index, 'before_failure_s': (length - 1 - index) / CONTROL_HZ,
                      'failure': report.get('failure', {}).get('reason'), 'subtask': report.get('failure', {}).get('subtask')}
            record.update(frame=snapshot['frame']['id'], check=snapshot['check'])
            step_names[step] = INITIAL_STATE if step == 1 else report['step_names'][step - 2]
            snapshots[step][seed] = (int(report.get('attempt', 0)), scene_state, start, record)
    if not snapshots:
        parser.error('no failed seed with a recorded trajectory')
    checkout = commit(ROOT)
    run = run_name(checkout, '-'.join([summary['robot'], summary['expert'], 'snapshots', *([args.name] if args.name else [])]))
    directory = (case_dir(args.output or ROOT / 'data' / 'cases', summary['scene'],
                          summary['robot'], summary['expert'], args.case) if args.case else
                 dev_dir(args.output or ROOT / 'runs' / 'expert', summary['scene'],
                         summary['robot'], summary['expert'], run))
    directory.mkdir(parents=True)
    for step, seeds in sorted(snapshots.items()):
        out = checkpoint_path(directory, step - 1)
        out.parent.mkdir(exist_ok=True)
        with h5py.File(out, 'w') as file:
            file.attrs.update(scene=summary['scene'], robot=summary['robot'], expert=summary['expert'], physics=summary['physics'],
                              commit=str(summary.get('commit')), step=step - 1, name=step_names[step])
            for seed, (attempt, state, start, record) in sorted(seeds.items()):
                record['batch'] = batch_ref(batch, directory)
                if '_expert_frame' in state:
                    state = dict(state)
                    snapshot = json.loads(state['_expert_frame'])
                    snapshot['batch'] = batch_ref(batch, directory)
                    state['_expert_frame'] = dumps(snapshot).encode('utf-8')
                group = file.create_group(str(seed))
                group.attrs.update(seed=seed, attempt=attempt, at=0, snapshot=dumps(record))
                write_tree(group.create_group('state'), state)
                write_tree(group.create_group('start'), start)
    out = {**{key: summary[key] for key in ('expert', 'scope', 'layout', 'outside_scope')},
           'run': run, 'command': ' '.join(['failure_snapshots.py', *(argv if argv is not None else sys.argv[1:])]),
           'scene': summary['scene'], 'robot': summary.get('robot'), 'physics': summary['physics'], 'commit': checkout,
           'snapshots_of': batch_ref(batch, directory),
           'snapshots': {str(step): sorted(seeds) for step, seeds in sorted(snapshots.items())}}
    (directory / 'summary.json').write_text(dumps(out, indent=1, ensure_ascii=False) + '\n')
    print('SNAPSHOTS', directory, dumps(out['snapshots']))


if __name__ == '__main__':
    main()
