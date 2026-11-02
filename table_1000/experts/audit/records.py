"""Read current search records, aggregate failures, and export portable cases/media."""
from collections import Counter
import json
from pathlib import Path

from table_1000.common.serialization import dumps
import shutil

from .trace import decode, failure_state
from table_1000.scenes import batch_ref, rebase_batch_refs


from .codec import load_document as load


def expert_calls(tree):
    """Yield skills directly called by expert code, with their enclosing step.

    Callback-only control frames are transparent. A subtask inside an ordinary
    skill does not turn that skill's internal adaptation into an expert call.
    """
    frames = {frame['id']: frame for frame in tree['frames']}
    for call in tree['frames']:
        if call['kind'] != 'skill' or call['control']:
            continue
        owner = frames.get(call['parent'])
        while owner is not None and (owner['kind'] not in ('skill', 'step') or owner['control']):
            owner = frames.get(owner['parent'])
        if owner is not None and owner['kind'] == 'step':
            yield call, owner


def caught_planning(tree):
    """Planning exceptions escaping a whole skill and caught by its expert caller."""
    events = {event['id']: event for event in tree['failures']}
    calls = list(expert_calls(tree))
    for call, step in calls:
        error = call.get('exception', {})
        cause = events.get(call.get('failure_id'))
        if error.get('kind') != 'planning' or cause is None or cause['kind'] != 'planning':
            continue
        # Propagation of the same exception out of the step is abandonment.
        if step.get('exception') and step.get('failure_id') == call['failure_id']:
            continue
        # A closed step (successful or failing later), or a subsequent direct
        # skill invocation, proves that expert code resumed after this exception.
        continuation = next((other for other, owner in calls
                             if owner['id'] == step['id'] and other['id'] > call['id']), None)
        if step['exit'] is None and continuation is None:
            continue
        yield call, cause, continuation


def search_costs(tree):
    """Per-physical-node search work, independent of failure classification.

    Layer times are inclusive: candidate work can contain solver work, and an
    exhausted decision can contain both. They must not be added across layers.
    """
    nodes = {}
    def add(branch, at, frame, layer, reason, event, step):
        node = nodes.setdefault((branch, at), dict(branch=branch, at=at, solver_coverage=tree['solver_coverage'], layers={}))
        row = node['layers'].setdefault(layer, dict(count=0, cpu_s=0., wall_s=0., reasons={}, frames=[], steps=[], timed_count=0))
        row['count'] += 1
        row['reasons'][reason] = row['reasons'].get(reason, 0) + 1
        if frame not in row['frames']:
            row['frames'].append(frame)
        if step not in row['steps']:
            row['steps'].append(step)
        if 'cpu_s' in event and 'wall_s' in event:
            row['timed_count'] += 1
        for cost in ('cpu_s', 'wall_s'):
            row[cost] += event.get(cost, 0.)
    for solve in tree['solves']:
        add(solve['branch'], solve['at'], solve['frame'], 'solver_calls',
            f"{solve['name']}: {solve['status']}", solve, solve['step'])
    for decision in tree['decisions']:
        for event in decision['events']:
            if event['outcome'] == 'rejected':
                add(event['branch'], event['at'], decision['frame'], 'candidate_rejections',
                    f"{event['stage']}: {event.get('reason', event.get('status', event['stage']))}", event, decision['step'])
        if decision.get('empty', False):
            add(decision['branch'], decision['exit'], decision['frame'], 'decision_exhaustions',
                decision['name'], decision, decision['step'])
    return list(nodes.values())


def planning_rollbacks(tree):
    """Rollbacks triggered by planning with no failed physical check to own them."""
    events = {event['id']: event for event in tree['failures']}
    claimed, result = set(), []
    for outcome in tree['outcomes']:
        if outcome['kind'] != 'rollback':
            continue
        checks = {i for i in outcome['discarded'] if i in events and events[i]['kind'] == 'check'} - claimed
        if not checks and events.get(outcome['failure_id'], {}).get('kind') == 'planning':
            result.append(dict(outcome, failure=events[outcome['failure_id']]))
        claimed.update(checks)
    return result


def classify(tree):
    """Derive failure categories from immutable observations and search outcomes.

    Only whole skill planning failures caught by expert code are a1. A rollback
    counts its triggering check once, with other discarded checks as components;
    abandonment similarly owns its checks. Only interrupted checks have no enclosing outcome. This function
    neither edits the tree nor reads simulator state or files.
    """
    observations = {(entry['id'], entry['branch']): entry for entry in tree.get('checks', [])}
    events = [dict(observations.get((event.get('check'), event['branch']), {}), **event)
              for event in tree['failures']]
    checks = {event['id']: event for event in events if event['kind'] == 'check'}
    claimed, result = set(), []

    def close(event, category, outcome=None, components=()):
        value = dict(event, category=category, end=None, recovery=None, components=list(components))
        if outcome is not None:
            value['end'] = outcome['at']
            end_clock = outcome.get('observed_clock')
            if end_clock is not None and 'cost_start' in value:
                value.update(zip(('cpu_s', 'wall_s'),
                                 (max(0., end - start) for start, end in zip(value['cost_start'], end_clock))))
            value['outcome_id'] = outcome['id']
        return value

    for call, cause, continuation in caught_planning(tree):
        value = close(cause, 'a1')
        value.update(frame=call['id'], call_branch=call['branch'], origin_frame=cause['frame'], origin_reason=cause['reason'],
                     reason=call['exception']['reason'], legacy_error=call['exception']['reason'],
                     at=call['exit'], action_start=call['enter'], skill=call['name'],
                     continuation=continuation['id'] if continuation else None,
                     cpu_s=call.get('cpu_s', 0.), wall_s=call.get('wall_s', 0.))
        # The call's entry clock is factual, so the whole failed invocation can
        # be timed even if its final candidate solve consumed almost no time.
        if 'cost_start' in call:
            value['cost_start'] = call['cost_start']
        result.append(value)
    order = {'rollback': 0, 'step_abandoned': 1, 'step_passed': 2}
    for outcome in sorted(tree['outcomes'], key=lambda item: order[item['kind']]):
        if outcome['kind'] == 'rollback':
            discarded = [checks[i] for i in outcome['discarded'] if i in checks and i not in claimed]
            if not discarded:
                continue
            trigger = next((f for f in discarded if f['id'] == outcome['failure_id']), discarded[-1])
            value = close(trigger, 'a3', outcome, [f for f in discarded if f is not trigger])
            value['cost_start'] = outcome['cost_start']
            if outcome.get('observed_clock') is not None:
                value.update(zip(('cpu_s', 'wall_s'),
                    (max(0., end - start) for start, end in zip(value['cost_start'], outcome['observed_clock']))))
            value['recovery'] = dict(kind='rollback', **outcome['restore'], control=outcome['control'])
            value['trigger_failure_id'] = outcome['failure_id']
            result.append(value)
            claimed.update(f['id'] for f in discarded)
        elif outcome['kind'] == 'step_abandoned':
            components = [f for f in checks.values() if f['step'] == outcome['step'] and f['id'] not in claimed]
            value = dict(outcome, id=f"b{outcome['id']}", frame=outcome['frame'], category='b',
                         criterion_id='run_steps.terminal', family_id='step.outcome',
                         legacy_error=outcome['reason'], end=outcome['at'], recovery=None,
                         cpu_s=0., wall_s=0., components=components)
            result.append(value)
            claimed.update(f['id'] for f in components)
        elif outcome['kind'] == 'step_passed':
            for event in checks.values():
                if event['step'] != outcome['step'] or event['id'] in claimed:
                    continue
                # The surviving prefix remains on the completed path when a
                # descendant retry resumes it; discarded suffixes were claimed.
                if outcome['branch'] != event['branch'] and not outcome['branch'].startswith(event['branch'] + '/'):
                    continue
                value = close(event, 'a2', outcome)
                value['recovery'] = dict(kind='step_passed', at=outcome['at'], branch=outcome['branch'])
                result.append(value)
                claimed.add(event['id'])
    result.extend(close(event, 'pending') for event in checks.values() if event['id'] not in claimed)
    return result


def documents(batch, seed=None, repeat=None):
    """Yield main and archived trees; joined parts retain their own namespaces."""
    batch = Path(batch).resolve()
    for path in sorted((batch / 'episodes').glob('*.json')):
        report = json.loads(path.read_text())
        if seed is not None and report['seed'] != seed:
            continue
        parts = report.get('trace_parts')
        if parts is not None:
            for part in parts:
                yield from documents(batch / part['directory'], report['seed'], repeat)
            continue
        if repeat is not None and report.get('repeat', 0) != repeat:
            continue
        report = dict(report, recording_root=str(batch))
        yield batch, report, load(batch / report['call_trace'])
        for archive in report.get('retry_archives', []):
            directory = batch / archive['batch']
            record = json.loads((directory / 'episodes' / f"{report['seed']}.json").read_text())
            yield directory, dict(record, recording_root=str(batch), archive=archive), load(directory / record['call_trace'])


def failures(batches, seed=None):
    """One record per failure, even when a retry archive also contains its prefix."""
    found = {}
    for batch in map(Path, batches):
        for source, report, tree in documents(batch, seed):
            archive = report.get('archive')
            if archive is None:
                classified = classify(tree)
            else:
                # The main tree retains all facts and outcomes, including the
                # discarded suffixes. An earlier archive only supplies frames
                # and media; its incomplete prefix must not classify a check a
                # second time before a later rollback owns it as a component.
                classified = [entry for entry in found.values()
                              if entry['report']['recording_root'] == report['recording_root']
                              and entry['seed'] == report['seed']
                              and entry['report'].get('repeat', 0) == report.get('repeat', 0)]
            for failure in classified:
                key = (str(batch.resolve()), report['seed'], report.get('repeat', 0), failure['id'],
                       # Joined source parts each number their own failures.
                       report['recording_root'], failure['branch'], failure['frame'])
                entry = {**failure, 'batch': str(source), 'seed': report['seed'], 'report': report, 'tree': tree}
                frame = next((f for f in tree['frames'] if f['id'] == failure['frame']
                              and f['branch'] == failure.get('call_branch', failure['branch'])), None)
                matches = (archive is not None and archive['failure_id'] == failure.get('trigger_failure_id', failure['id'])
                           and archive['branch'] == failure['branch'])
                rank = 3 if matches else 2 if frame is not None else 0 if archive else 1
                # An a3 belongs to exactly one discarded branch, not to another
                # archive that happens to retain the same prefix.
                entry['_rank'] = rank
                if key not in found or rank > found[key]['_rank']:
                    found[key] = entry
    return list(found.values())


def physical_start(failure):
    """A rollback discards its entire callback, even if its last solve failed without moving."""
    recovery = failure['recovery']
    if recovery and recovery['kind'] == 'rollback':
        return recovery['at']
    return failure.get('action_start', failure.get('start', failure['at']))


def statistics(batches):
    entries = failures(batches)
    rows = {category: dict(count=0, cpu_s=0., wall_s=0.) for category in ('a1', 'a2', 'a3', 'b', 'pending')}
    rejections = Counter()
    search_layers, planning_retries, coverage = {}, [], Counter()
    clips = []
    denominator = dict(cpu_s=0., wall_s=0.)
    seen = set()
    for batch in map(Path, batches):
        for source, report, tree in documents(batch):
            key = (str(source), report['seed'], report.get('repeat', 0))
            if report.get('archive') is not None or key in seen:
                continue
            seen.add(key)
            coverage[tree['solver_coverage']] += 1
            for node in search_costs(tree):
                for layer, value in node['layers'].items():
                    row = search_layers.setdefault(layer, dict(count=0, timed_count=0, cpu_s=0., wall_s=0., reasons=Counter()))
                    for key in ('count', 'timed_count', 'cpu_s', 'wall_s'):
                        row[key] += value[key]
                    row['reasons'].update(value['reasons'])
            planning_retries.extend(dict(batch=str(source), seed=report['seed'], **item)
                                    for item in planning_rollbacks(tree))
            # Top-level calls partition episode execution. Retries' discarded
            # costs are included in their still-active enclosing control call.
            for frame in tree['frames']:
                if frame['parent'] is None:
                    for cost in denominator:
                        denominator[cost] += frame.get(cost, 0.)
    timelines = {}
    for entry in entries:
        identity = (entry['report']['recording_root'], entry['seed'], entry['report'].get('repeat', 0))
        for axis, cost in enumerate(denominator):
            if entry['category'] != 'b' and entry[cost] > 0 and 'cost_start' in entry:
                start = entry['cost_start'][axis]
                timelines.setdefault((identity, cost), {}).setdefault(entry['category'], []).append((start, start + entry[cost]))
    for entry in entries:
        row = rows[entry['category']]
        row['count'] += 1
        # b is an outcome, not another disjoint timed solve/recovery interval.
        if entry['category'] == 'a1':
            rejections[f"{entry.get('stage') or entry.get('criterion_id', 'candidate')}: {entry['reason']}"] += 1
        if entry['category'] in ('a2', 'a3', 'pending'):
            source = Path(entry['batch'])
            trajectory_path = str(trajectory(source, entry['report']).relative_to(source))
            clips.append(dict(batch=str(source), seed=entry['seed'], repeat=entry['report'].get('repeat', 0),
                              failure_id=entry['id'], category=entry['category'], report=entry['report'],
                              trajectory=trajectory_path, branch=entry['branch'],
                              control_start=physical_start(entry),
                              failure=entry['at'], control_end=entry['end'] if entry['end'] is not None else entry['report']['control_steps'],
                              state_index_offset=-1, recovery=entry['recovery']))
    def union(intervals):
        result = []
        for start, end in sorted(intervals):
            if result and start <= result[-1][1]:
                result[-1] = (result[-1][0], max(end, result[-1][1]))
            else:
                result.append((start, end))
        return result
    for (_, cost), categories in timelines.items():
        covered = []
        for category in ('a1', 'a2', 'a3', 'pending'):
            before = sum(end - start for start, end in covered)
            combined = union([*covered, *categories.get(category, [])])
            rows[category][cost] += sum(end - start for start, end in combined) - before
            covered = combined
    for category, row in rows.items():
        for cost in denominator:
            row[cost + '_fraction'] = row[cost] / denominator[cost] if denominator[cost] else 0.
    return dict(categories=rows, total_execution=denominator, rejection_counts=dict(rejections), clips=clips,
                search_costs=search_layers, solver_coverage=dict(coverage), planning_rollbacks=planning_retries,
                cost_note='b is terminal outcome; pending is separate; caught whole-skill planning calls and physical recovery intervals are timed; search layer costs are inclusive')


def trajectory(batch, report):
    """The execution's own recording, or None for an unrecorded repeat."""
    if report.get('trajectory'):
        return Path(batch) / report['trajectory']
    # Retry archives are independent batches with their own main recording.
    if report.get('repeat', 0) and not report.get('from_retry'):
        return None
    return Path(batch) / ('trajectories.hdf5' if report['status'] == 'success' else 'trajectories_failed.hdf5')


def state_at(group, at):
    """Numeric tree before control ``at``; states[at-1] is after control at."""
    import h5py
    def read(node, index=None):
        return {name: read(value, index) if isinstance(value, h5py.Group)
                else value[()] if index is None else value[index:index + 1] for name, value in node.items()}
    return read(group['initial_state']) if at == 0 else read(group['states'], at - 1)


def media_batch(batch, report, spans, output):
    """Export node stills and action clips as a normal render.py batch, without simulating."""
    import h5py
    import numpy as np
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    def write(group, states):
        for key, value in states[0].items():
            if isinstance(value, dict):
                write(group.create_group(key), [state[key] for state in states])
            else:
                group.create_dataset(key, data=np.concatenate([state[key] for state in states]))
    with h5py.File(output / 'trajectories.hdf5', 'w') as target:
        data = target.create_group('data')
        for index, span in enumerate(spans):
            source = Path(span.get('batch', batch))
            own_report = span.get('report', report)
            path = trajectory(source, own_report)
            if path is None:
                raise ValueError(f"repeat {own_report['repeat']} has no recorded trajectory: {source}")
            with h5py.File(path, 'r') as original:
                data.attrs.update(dict(original['data'].attrs))
                group = original[f"data/demo_{own_report['seed']}"]
                start, end = span['start'], span['end']
                if not 0 <= start <= end <= int(group.attrs['num_samples']):
                    raise ValueError(f'invalid span {start}:{end} for {source}')
                # Render.py renders before each sample, so append a duplicate
                # endpoint to include the final physical state in the clip.
                states = [state_at(group, at) for at in range(start, end + 1)]
                states.append(states[-1])
                out = data.create_group(f'demo_{index}')
                out.attrs.update(seed=index, status='success', num_samples=len(states) - 1,
                                 source=batch_ref(source, output), source_seed=own_report['seed'], source_start=start, source_end=end)
                write(out.create_group('initial_state'), states[:1])
                write(out.create_group('states'), states[1:])
        data.attrs['total'] = sum(group.attrs['num_samples'] for group in data.values())
    summary = json.loads((Path(batch) / 'summary.json').read_text())
    summary = rebase_batch_refs(summary, batch, output)
    summary.update(seeds={'range': f'0:{len(spans)}'}, videos={}, fails={}, media_spans=[
        {k: batch_ref(v, output) if k == 'batch' else v for k, v in span.items() if k != 'report'} for span in spans])
    (output / 'summary.json').write_text(dumps(summary, indent=1, ensure_ascii=False) + '\n')
    return output


def export_case(batches, seed, output, failure_id=None, *, repeat=0):
    """One summary.json authority: root checkpoint, tree, and every supplied attempt."""
    import h5py
    import numpy as np
    from table_1000.scenes.initial import write_tree
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sources = []
    chosen = None
    for index, batch in enumerate(map(Path, batches)):
        summary = json.loads((batch / 'summary.json').read_text())
        name = f'{seed}-{repeat}' if repeat else str(seed)
        report = json.loads((batch / 'episodes' / f'{name}.json').read_text())
        tree = load(batch / report['call_trace'])
        destination = output / 'sources' / str(index)
        destination.mkdir(parents=True)
        for report_path in (batch / 'episodes').glob('*.json'):
            own_report = json.loads(report_path.read_text())
            if own_report['seed'] != seed:
                continue
            for relative in (Path('episodes') / report_path.name, Path(own_report['call_trace'])):
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(batch / relative, target)
            (destination / 'episodes' / report_path.name).write_text(
                dumps(rebase_batch_refs(own_report, batch, destination), indent=1))
            if own_report.get('trajectory'):
                target = destination / own_report['trajectory']
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(batch / own_report['trajectory'], target)
            for archive in own_report.get('retry_archives', []):
                target = destination / archive['batch']
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(batch / archive['batch'], target)
        if report.get('trace_parts'):
            shutil.copytree(batch / 'trace-parts', destination / 'trace-parts')
        (destination / 'summary.json').write_text(dumps(rebase_batch_refs(summary, batch, destination), indent=1))
        for filename in ('trajectories.hdf5', 'trajectories_failed.hdf5'):
            if (batch / filename).is_file():
                with h5py.File(batch / filename) as source_file, h5py.File(destination / filename, 'w') as target_file:
                    data = target_file.create_group('data')
                    data.attrs.update(dict(source_file['data'].attrs))
                    name = f'demo_{seed}'
                    if name in source_file['data']:
                        source_file.copy(source_file['data'][name], data, name)
        sources.append(dict(directory=str(destination.relative_to(output)), batch=batch_ref(batch, output),
                            commit=summary.get('commit'),
                            **{'from': rebase_batch_refs({'from': summary.get('from')}, batch, output)['from']},
                            tree=str(Path('sources') / str(index) / report['call_trace']),
                            attempts=report.get('retry_archives', [])))
        if chosen is None:
            if failure_id is not None:
                owner = next(((source, own_report, own_tree) for source, own_report, own_tree in documents(batch, seed)
                             if own_report.get('repeat', 0) == report.get('repeat', 0)
                             and any(f['id'] == failure_id and any(frame['id'] == f['frame']
                                 and frame['branch'] == f['branch'] for frame in own_tree['frames'])
                                    for f in own_tree['failures'])), None)
                if owner is None:
                    raise ValueError(f'planning event {failure_id} not found for seed {seed}, repeat {repeat}')
                state, snapshot = failure_state(*owner[:2], failure_id)
                category = next((f['category'] for f in classify(owner[2]) if f['id'] == failure_id), None)
                step, root = snapshot['step'], snapshot['frame']['id']
                root_tree = str(Path('sources') / str(index) / owner[0].relative_to(batch.resolve())
                                / owner[1]['call_trace'])
            else:
                root_frame = next(f for f in reversed(tree['frames']) if f['kind'] == 'step')
                step, root = root_frame['arguments']['index'], root_frame['id']
                state = decode(root_frame['state']['physical'])
                root_tree = sources[-1]['tree']
            chosen = summary, report, state, step, root, root_tree
    summary, report, state, step, root, root_tree = chosen
    if '_expert_frame' in state:
        state = dict(state)
        snapshot = json.loads(state['_expert_frame'])
        snapshot['batch'] = batch_ref(snapshot['batch'], output)
        state['_expert_frame'] = dumps(snapshot).encode('utf-8')
    checkpoint = f'checkpoints/step{step - 1}.hdf5'
    (output / 'checkpoints').mkdir()
    with h5py.File(output / checkpoint, 'w') as file:
        file.attrs.update(scene=summary['scene'], robot=summary['robot'], expert=summary['expert'],
                          physics=summary['physics'], step=step - 1,
                          name='initial state' if step == 1 else report['step_names'][step - 2],
                          commit=str(summary.get('commit')))
        group = file.create_group(str(seed))
        group.attrs.update(seed=seed, attempt=report['attempt'], at=0)
        write_tree(group.create_group('state'), state)
        # Independent solve checkpoints already carry held commands. Start is
        # the recorded episode baseline, rather than a fabricated zero pose.
        start = {k: np.asarray(v) for k, v in report['start'].items()}
        write_tree(group.create_group('start'), start)
    result = {k: summary[k] for k in ('scene', 'robot', 'expert', 'scope', 'layout', 'outside_scope', 'physics')}
    result.update(commit=summary.get('commit'), steps_run=[step, step], case=dict(version=1,
        category=category if failure_id is not None else 'b', kind='planning' if failure_id is not None else 'step',
        seed=seed, repeat=repeat,
        root=dict(checkpoint=checkpoint, frame=root, tree=root_tree),
        failure_id=failure_id, sources=sources))
    (output / 'summary.json').write_text(dumps(result, indent=1, ensure_ascii=False) + '\n')
    return result
