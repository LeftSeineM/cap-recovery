"""Join the episodes of batches run step by step into whole episodes, one per seed.

``uv run python scripts/scenes/join_steps.py BATCH ... [--output DIR]``

The batches (directories ``run_expert.py`` wrote, in order) each run some steps of
the expert (``--steps k[-m]``): the first from the initial states, every other from
the checkpoints of the one before (``--from``). A seed's episode in a batch is cut
where the next batch's start state was kept (the checkpoint of its step ``k - 1``)
and continued by its episode there, also when the episode failed after that
checkpoint (the next batch ran the step again); the chain ends where the seed has no
such continuation. The joined episode keeps the first's initial state and attributes,
the subtasks and steps shifted to their place, ``status`` and ``failure`` of the last
(``failed`` with reason ``not run further`` when it succeeded before the last batch),
and ``parts``: each part's batch, commit, steps, first step in the joined episode and
length.

The joined episodes are the scene's batch like one run of the whole expert: ``DIR``
(default the scene's data, ``data/scenes/<scene>/<backend>/``) is replaced by ``trajectories.hdf5`` and ``trajectories_failed.hdf5`` (by the
last part's status) and a ``summary.json``; ``render.py <scene> --physics <backend>``
renders them. ``episodes/<seed>.json`` is the last part's report placed in the joined episode
(control steps shifted, ``status`` and ``failure`` as the joined episode's, ``parts``, the
summed ``wall_s``). The last part's call and solve records are copied alongside,
with ``call_trace_offset`` mapping their control steps and ``call_trace_initial_state``
keeping that part's own initial state for entry at local step zero. Earlier parts'
records and later repeats stay in their source batches. ``summary.json`` counts
the joined reports as ``run_expert.py`` does, with the commits
joined. Into the scene's data (no ``--output``) only a join of every seed through every step
goes.
"""

import argparse
import json
import shlex
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def episodes(batch):
    """``{seed: (file, name, status)}`` of a batch's trajectory files (open h5py files)."""
    import h5py
    found = {}
    for kind in ('trajectories', 'trajectories_failed'):
        path = batch / f'{kind}.hdf5'
        if path.is_file():
            file = h5py.File(path, 'r')
            for name in file['data']:
                found[int(name.split('_')[1])] = (file, name, file['data'][name].attrs['status'])
    return found


def step_count(batch):
    """How many steps the expert of a batch has (from an episode report), None without reports."""
    report = next((batch / 'episodes').glob('*.json'), None) if (batch / 'episodes').is_dir() else None
    return len(json.loads(report.read_text()).get('step_names') or []) if report else None


def shifted(report, offset, length):
    """An episode report of a part that starts at control step ``offset`` of the joined episode."""
    report = dict(report)
    report['events'] = [{**event, 'step': event['step'] + offset} for event in report.get('events', [])]
    report['plans'] = [{**plan, 'step': plan['step'] + offset} for plan in report.get('plans', [])]
    report['subtasks'] = [{**entry, 'start': entry['start'] + offset,
                           'end': None if entry['end'] is None else entry['end'] + offset}
                          for entry in report.get('subtasks', [])]
    report['steps'] = [{**entry, 'at': entry['at'] + offset} for entry in report.get('steps') or []]
    report['control_steps'] = offset + length
    return report


def carry_traces(batch, report, group, output, offset):
    """Keep the last part's numerical records and map its frames into the joined trajectory."""
    import h5py
    for key in ('call_trace', 'solve_trace'):
        if report.get(key):
            target = output / report[key]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(batch / report[key], target)
    if report.get('call_trace'):
        # Resuming a checkpoint can change the scene before its first control step.
        # Keep that initial state instead of substituting the previous part's last sample.
        initial = Path(report['call_trace']).with_name(f"{report['seed']}-initial.hdf5")
        if report.get('call_trace_initial_state'):
            shutil.copyfile(batch / report['call_trace_initial_state'], output / initial)
        else:
            with h5py.File(output / initial, 'w') as file:
                group.file.copy(group['initial_state'], file, 'initial_state')
        report['call_trace_initial_state'] = str(initial)
        report['call_trace_offset'] = report.get('call_trace_offset', 0) + offset


def main(argv=None):
    import h5py
    from table_1000.common.batch import seeds_record
    from table_1000.experts.generate import summarize
    from table_1000.scenes.recordings import clear_batch, concatenate, decoded_attrs
    from table_1000.sim import CONTROL_HZ
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('batches', type=Path, nargs='+', help='batch directories, in step order')
    parser.add_argument('--output', type=Path,
                        help="the joined batch directory, replaced (default: the scene's data, "
                             "data/scenes/<scene>/<backend>/)")
    args = parser.parse_args(argv)
    summaries = []
    for index, batch in enumerate(args.batches):
        if not (batch / 'summary.json').is_file():
            parser.error(f'{batch} has no summary.json')
        summary = json.loads((batch / 'summary.json').read_text())
        if index and Path(summary.get('from') or '').resolve() != args.batches[index - 1].resolve():
            parser.error(f'{batch} did not start from {args.batches[index - 1]}')
        summaries.append(summary)
    first = summaries[0]
    default = args.output is None
    args.output = args.output or ROOT / 'data' / 'scenes' / first['scene'] / first['physics']
    if (first.get('steps_run') or [1])[0] != 1:
        parser.error(f'{args.batches[0]} does not start from the initial states (step 1)')
    output = Path(args.output).resolve()
    for batch in map(Path.resolve, args.batches):
        if batch.is_relative_to(output) or output.is_relative_to(batch):
            parser.error(f'{args.output} would replace {batch}, one of the batches joined; give another --output')
    # The scene's data comes from every seed: batches of some seeds go elsewhere.
    partial = [batch for batch, summary in zip(args.batches, summaries)
               if any(token == '--seeds' or token.startswith('--seeds=') for token in shlex.split(summary.get('command') or ''))]
    if default and (partial or first.get('from')):
        parser.error(f'{(partial or args.batches)[0]} may not cover every seed of the scene, so the join is not its '
                     f'data; give --output')
    # ... and every step: a join that stops early is a development batch.
    last_run, count = summaries[-1].get('steps_run'), step_count(args.batches[-1])
    if default and last_run and last_run[1] != count:
        parser.error(f'the batches run steps 1-{last_run[1]} of {count}, not the scene\'s data; give --output')
    if any((summary.get('steps_run') or [1])[0] == 1 for summary in summaries[1:]):
        parser.error('only the first batch starts at step 1')
    found = [episodes(batch) for batch in args.batches]
    if not found[0]:
        parser.error(f'{args.batches[0]} has no episodes')
    # Where each batch started: the step before its first, cut out of the previous part there.
    starts = [(summary.get('steps_run') or [1])[0] for summary in summaries]
    # Each seed's chain of (batch, episode group, status, length): a part goes on in the next batch
    # when that batch ran the seed from the part's checkpoint (also after a failure later in the part).
    chains = {}
    for seed in sorted(found[0]):
        chain, index = [], 0
        while True:
            file, name, status = found[index][seed]
            group = file['data'][name]
            done = decoded_attrs(group).get('steps', [])
            cut = None
            if index + 1 < len(found) and seed in found[index + 1]:
                cut = next((entry['at'] for entry in done if entry['step'] == starts[index + 1] - 1), None)
            if cut is None:
                chain.append((index, group, status, int(group.attrs['num_samples']), done))
                break
            chain.append((index, group, status, cut, done))
            index += 1
        chains[seed] = chain
    clear_batch(args.output)
    (args.output / 'episodes').mkdir()
    reports = []
    files = {kind: h5py.File(args.output / f'{kind}.hdf5', 'w') for kind in ('trajectories', 'trajectories_failed')}
    counts = {}
    try:
        env_file = next(entry[0] for batch_episodes in found for entry in batch_episodes.values())
        for out_file in files.values():
            out_file.create_group('data').attrs.update(dict(env_file['data'].attrs))
        for seed, chain in chains.items():
            sources, subtasks, steps, records, offset = [], [], [], [], 0
            for index, group, _, length, done in chain:
                sources.append((group, length))
                subtasks += [{**entry, 'start': entry['start'] + offset,
                              'end': None if entry['end'] is None else min(entry['end'], length) + offset}
                             for entry in decoded_attrs(group).get('subtasks', []) if entry['start'] < length]
                steps += [{**entry, 'at': entry['at'] + offset} for entry in done if entry['at'] <= length]
                records.append({'batch': str(args.batches[index].resolve()), 'commit': summaries[index].get('commit'),
                                'steps_run': summaries[index].get('steps_run'), 'start': offset, 'length': length})
                offset += length
            index, last, status, length = chain[-1][:4]
            if status == 'success' and index + 1 < len(found):
                status = 'not run further'
            out = concatenate(sources, files['trajectories' if status == 'success' else 'trajectories_failed'],
                              f'demo_{seed}')
            for key in ('status', 'failure'):
                if key in last.attrs:
                    out.attrs[key] = last.attrs[key]
                elif key in out.attrs:
                    del out.attrs[key]
            if status == 'not run further':
                status = 'failed'
                out.attrs['failure'] = json.dumps({'subtask': None, 'reason': 'not run further'})
            out.attrs['status'] = status       # 'error' (the expert's code) stays apart from 'failed'
            out.attrs['success'] = status == 'success'
            out.attrs['subtasks'] = json.dumps(subtasks)
            out.attrs['steps'] = json.dumps(steps)
            out.attrs['parts'] = json.dumps(records)
            counts[status] = counts.get(status, 0) + 1
            # The last part's report, placed in the joined episode, for the tools that read reports.
            report = shifted(json.loads((args.batches[index] / 'episodes' / f'{seed}.json').read_text()),
                             records[-1]['start'], length)
            carry_traces(args.batches[index], report, last, args.output, records[-1]['start'])
            report.update(status=status, parts=records, duration_s=offset / CONTROL_HZ,
                          wall_s=sum(json.loads((args.batches[part] / 'episodes' / f'{seed}.json').read_text())['wall_s']
                                     for part, *_ in chain))
            if 'failure' in out.attrs:
                report['failure'] = {**report.get('failure', {}), **json.loads(out.attrs['failure'])}
            else:
                report.pop('failure', None)
            reports.append(report)
            (args.output / 'episodes' / f'{seed}.json').write_text(json.dumps(report, indent=1, default=float))
        for out_file in files.values():
            data = out_file['data']
            data.attrs['total'] = sum(int(data[name].attrs['num_samples']) for name in data)
    finally:
        for file in files.values():
            file.close()
        for batch_episodes in found:
            for file in {entry[0] for entry in batch_episodes.values()}:
                file.close()
    commits = sorted({str(summary.get('commit')) for summary in summaries})
    summary = {'scene': summaries[0].get('scene'), 'robot': summaries[0].get('robot'),
               'physics': summaries[0].get('physics'), 'joined': [str(batch.resolve()) for batch in args.batches],
               'commit': commits[0] if len(commits) == 1 else None, 'commits': commits,
               **summarize(reports, sorted(chains)), 'failed': counts.get('failed', 0), 'error': counts.get('error', 0),
               'steps': summaries[-1].get('steps'), 'videos': {}, 'fails': {}}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False) + '\n')
    print('JOINED', args.output, json.dumps(counts))


if __name__ == '__main__':
    main()
