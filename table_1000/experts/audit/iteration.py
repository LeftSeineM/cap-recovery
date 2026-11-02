"""Index an expert line's development batches into per-step search forests, and render media.

A line is one directory runs/expert/<scene>/<robot>/<NNNN or NNNN-legacy-MMDD>/
of batches (``table_1000.scenes.dev_batches``). Nothing is linked or edited
there: derived files (the index, per-batch graph caches, media) live in the
line's ``.iteration/`` directory and are rebuilt when a batch changes.

Step names identify actions across expert revisions (numbers can move). A name
the final script no longer has is a deprecated step of its own, numbered by
neither; its likely replacement is read from the next script revision. Control
coordinates always remain source-local.
Categories (a1/a2/a3/b), search costs and planning rollbacks come from ``records``,
which derives them from recorded facts; the viewer never classifies. No recording or simulation code runs here.
"""
from collections import Counter, defaultdict
from datetime import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from table_1000.common.serialization import dumps
from table_1000.scenes import resolve_batch_ref
from . import codec, records

CACHE = '.iteration'
GRAPH_FORMAT = 5  # bump when _batch_graph's output changes, so cached graphs rebuild
INDEX_FORMAT = 4  # likewise for line_index's output


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    write_text(path, dumps(value, ensure_ascii=False, indent=1) + '\n')


def write_text(path, value):
    """Atomically publish a file while an existing browser may be reading it."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name+'.', delete=False) as stream:
        stream.write(value)
        temporary = Path(stream.name)
    temporary.replace(path)


def _summary_failure(failure):
    keys = ('id', 'category', 'stage', 'reason', 'criterion_id', 'measurements',
            'at', 'end', 'start', 'recovery', 'cpu_s', 'wall_s', 'task_outcome', 'components')
    return {k: _compact(failure[k]) for k in keys if k in failure}


def _compact(value):
    """Keep readable arguments, referencing the source for dense geometric arrays."""
    if isinstance(value, dict):
        return {k: _compact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 16:
            return dict(items=len(value), preview=[_compact(v) for v in value[:3]],
                        detail='dense values retained in source call trace')
        return [_compact(v) for v in value]
    return value


def _summarize_checks(checks):
    """Aggregate repeated control-step guards, retaining last values and ranges."""
    groups = {}
    for check in checks:
        values = check.get('values') or {}
        identity = tuple((k, v) for k, v in values.items() if isinstance(v, str))
        key = (check.get('criterion_id'), check.get('result'), check.get('branch'), identity)
        item = groups.setdefault(key, dict(criterion_id=key[0], result=key[1], count=0,
            first_at=check.get('at'), last_at=check.get('at'), numeric_ranges={}))
        item.update(last_at=check.get('at'), last_id=check.get('id'),
                    measurements=_compact(check.get('measurements', [])), values=_compact(values))
        item['count'] += check.get('count', 1)
        for name, value in values.items():
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                lo, hi = item['numeric_ranges'].get(name, [value, value])
                item['numeric_ranges'][name] = [min(lo, value), max(hi, value)]
    return list(groups.values())


def _report_file(source, seed, repeat):
    repeated = Path(f'episodes/{seed}-{repeat}.json')
    return str(repeated if repeat and (Path(source)/repeated).is_file() else Path(f'episodes/{seed}.json'))


def _keys(names):
    """Step identities of a script: its names, with ``#k`` on the k-th of a repeated name."""
    seen = Counter(names)
    counted = Counter()
    keys = []
    for name in names:
        counted[name] += 1
        keys.append(f'{name}#{counted[name]}' if seen[name] > 1 else name)
    return keys


def _step_key(frame, names):
    """The identity of a step frame within its own script ``names``."""
    return _keys(names)[frame['arguments']['index'] - 1] if frame['name'] in names else frame['name']


def _replacements(name, revisions):
    """Final-script steps that took a deprecated step's place in the next revision without it.

    ``revisions`` are step-name lists in time order. The candidates lie between
    the nearest neighbours both revisions share; ``certain`` when exactly one.
    """
    for before, after in zip(revisions, revisions[1:]):
        if name not in before or name in after:
            continue
        at = before.index(name)
        previous = next((n for n in reversed(before[:at]) if n in after), None)
        following = next((n for n in before[at + 1:] if n in after), None)
        low = after.index(previous) + 1 if previous else 0
        high = after.index(following) if following else len(after)
        candidates = [n for n in after[low:high] if n not in before]
        return dict(candidates=candidates, certain=len(candidates) == 1, previous=previous)
    return dict(candidates=[], certain=False, previous=None)


def _matching_prefixes(step, seed, *, base=Path()):
    """Label byte-identical two-second state prefixes without merging attempts."""
    import h5py
    for item in [*step['attempts'], *step['nodes'], *step['edges']]:
        item.pop('prefix_note', None)
    groups = defaultdict(list)
    nodes = {n['id']: n for n in step['nodes']}
    for number, attempt in enumerate(step['attempts'], 1):
        node = nodes[attempt['entry']]
        source = base/node['batch']
        report = read(source/node['report_file'])
        if report['control_steps'] - node['at'] < 50:
            continue
        path = records.trajectory(source, report)
        if path is None:
            continue
        with h5py.File(path) as file:
            digest = hashlib.sha256()
            def sample(name, value):
                if isinstance(value, h5py.Dataset):
                    digest.update(name.encode())
                    digest.update(value[node['at']:node['at'] + 50].tobytes())
            file[f'data/demo_{seed}/states'].visititems(sample)
        groups[attempt['parent_checkpoint'], digest.hexdigest()].append((number, attempt))
    for (checkpoint, _), matches in groups.items():
        if len(matches) < 2:
            continue
        label = '、'.join(str(n) for n, _ in matches)
        origin = '同一父检查点' if checkpoint else '同一初态'
        note = f'尝试 {label} 从{origin}出发，前 50 控制帧（2 s）状态逐值一致；各尝试仍分别保留。'
        for _, attempt in matches:
            attempt['prefix_note'] = note
            prefix = attempt['entry'].split('/r')[0] + f'/r{attempt["repeat"]}/'
            for item in [*step['nodes'], *step['edges']]:
                if item['id'].startswith(prefix):
                    item['prefix_note'] = note


def _batch_graph(batch, seed):
    """Physical graph in audit.view's node/edge format, with source-step ownership.

    Equal control numbers in separate repeats/rollback branches are different
    states. a1 stays on the owning step even at a shared step boundary. Edges
    partition executed control intervals; their cost is the measured enclosing
    call's cost, explicitly labelled rather than invented per-frame timing.
    """
    docs = list(records.documents(batch, seed))
    names = docs[0][1]['step_names']
    failures = records.failures([batch], seed)
    branch_sources = {}
    branch_limits = {}
    branch_starts = {}
    restores = defaultdict(list)
    source_order = {}
    for source, report, _ in docs:
        for archive in report.get('retry_archives', []):
            branch_sources[report.get('repeat', 0), archive['branch']] = source/archive['batch']
            if 'control' in archive and 'retry' in archive:
                archived = next((t for s, _, t in docs if s == source/archive['batch']), None)
                if archived is not None:
                    frames = {f['id']: f for f in archived['frames']}
                    ancestors, control = set(), frames[archive['control']]
                    while control is not None:
                        ancestors.add(control['id'])
                        control = frames.get(control['parent'])
                    r = report.get('repeat', 0)
                    source_order[str(source/archive['batch'])] = len(restores[r])
                    restores[r].append(dict(child=f'{archive["branch"]}/{archive["control"]}-{archive["retry"]}',
                                            frames=frames, ancestors=ancestors))
    for source, report, tree in docs:
        repeat = report.get('repeat', 0)
        if 'archive' not in report:
            for outcome in tree['outcomes']:
                if outcome['kind'] == 'rollback':
                    branch_starts[repeat, outcome['restore']['branch']] = outcome['restore']['at']
            for archive in report.get('retry_archives', []):
                child = f'{archive["branch"]}/{archive["control"]}-{archive["retry"]}'
                if (repeat, child) not in branch_sources:
                    branch_limits[repeat, child] = report['control_steps']
        for frame in tree['frames']:
            key = repeat, frame['branch']
            branch_starts[key] = min(branch_starts.get(key, frame['enter']), frame['enter'])
            if key not in branch_sources or source == branch_sources[key]:
                branch_limits[key] = report['control_steps']

    def physical_branch(repeat, branch, at):
        # An enclosing step/call can enter before a RetryState and exit after
        # it. Its exit coordinate belongs to the later physical branch.
        if at <= branch_limits.get((repeat, branch), at):
            return branch
        candidates = [b for (r, b), end in branch_limits.items() if r == repeat
            and b.startswith(branch + '/') and branch_starts[r, b] <= at <= end]
        if not candidates:
            raise ValueError(f'{batch}: no recorded physical branch for {branch}@{at}')
        return max(candidates, key=len)

    def exit_branch(source, repeat, frame):
        # Open enclosing calls keep their entry branch after restoring inside
        # them. Use the archived control ancestry, not numeric frame ranges:
        # a later exit can fit numerically inside the discarded trajectory.
        branch = frame['branch']
        prior = restores[repeat][:source_order.get(source, len(restores[repeat]))]
        for restore in prior:
            original = restore['frames'].get(frame['id'])
            if (frame['id'] in restore['ancestors'] and original is not None
                    and all(original[k] == frame[k] for k in ('name', 'enter', 'branch'))):
                branch = restore['child']
        return branch
    own, step_frames = {}, {}
    for source, report, tree in docs:
        source = str(source)
        frames = {f['id']: f for f in tree['frames']}
        repeat = report.get('repeat', 0)
        for f in tree['frames']:
            root = f
            while root['kind'] != 'step' and root['parent'] is not None:
                root = frames[root['parent']]
            if root['kind'] == 'step':
                number = _step_key(root, names)
                own[source, repeat, f['id']] = number
                step_frames[source, repeat, number] = root
    groups = defaultdict(lambda: dict(nodes={}, edges=[], attempts=[], failures=[]))
    # Archives contain prefix copies. Failures are deduplicated by records.
    failure_groups = defaultdict(list)
    for failure in failures:
        source = failure['batch']
        repeat = failure['report'].get('repeat', 0)
        number = own.get((source, repeat, failure['frame']), own.get((source, repeat, failure.get('root'))))
        if number is None:
            raise ValueError(f'{source}: failure {failure["id"]} at frame {failure["frame"]} is outside every step')
        failure_groups[number].append(failure)
    for number, owned in failure_groups.items():
        # The page counts failures from this list: an edge is colored by every
        # failure overlapping it, so edges and failures are not one to one.
        groups[number]['failures'] = [dict(_summary_failure(f), batch=str(f['batch']), repeat=f['report'].get('repeat', 0))
                                      for f in owned]
    for source_path, report, tree in docs:
        source = str(source_path)
        repeat = report.get('repeat', 0)
        numbers = {n for (s, r, _), n in own.items() if s == source and r == repeat}
        for number in numbers:
            group = groups[number]
            prefix = f'{batch.name}/r{repeat}/'
            def node(branch, at):
                branch = physical_branch(repeat, branch, at)
                key = f'{prefix}{branch}@{at}'
                group['nodes'].setdefault(key, dict(id=key, branch=prefix + branch, at=at,
                    batch=str(source_path), repeat=repeat, a1=[], checks=[],
                    report_file=_report_file(source, seed, repeat)))
                return group['nodes'][key]
            fs = [f for f in tree['frames'] if own.get((source, repeat, f['id'])) == number]
            concrete = [f for f in fs if f['exit'] is not None and (f['concrete'] or f['kind'] == 'step')]
            for f in concrete:
                node(f['branch'], f['enter'])
                node(exit_branch(source, repeat, f), f['exit'])
            local = [f for f in failure_groups[number] if f['batch'] == source and f['report'].get('repeat', 0) == repeat]
            for failure in local:
                point = node(failure['branch'], failure['at'])
                item = _summary_failure(failure)
                if failure['category'] == 'a1':
                    point['a1'].append(item)
                else:
                    point['checks'].append(item)
                if failure['end'] is not None and (failure.get('recovery') or {}).get('kind') != 'rollback':
                    node(failure['branch'], failure['end'])
            # Archives repeat a prefix of the main record's facts; only the main record is read.
            if 'archive' not in report:
                planning = {p['id'] for p in records.planning_rollbacks(tree)}
                for outcome in tree['outcomes']:
                    if outcome['kind'] != 'rollback' or own.get((source, repeat, outcome['step'])) != number:
                        continue
                    restore = outcome['restore']
                    # The a3 is read from whichever record (often the archive) owns its frames.
                    owned = [f for f in failure_groups[number] if f['category'] == 'a3' and f.get('outcome_id') == outcome['id']
                             and f['report'].get('repeat', 0) == repeat]
                    trigger = 'a3' if owned else 'planning' if outcome['id'] in planning else 'rollback'
                    group['edges'].append(dict(id=f'{prefix}rollback-{outcome["id"]}',
                        source=node(outcome['branch'], outcome['at'])['id'], target=node(restore['branch'], restore['at'])['id'],
                        name='RetryState.restore', category='rollback', trigger=trigger, reason=outcome.get('reason'),
                        start=restore['at'], end=restore['at'], failures=[_summary_failure(f) for f in owned]))
                for cost in records.search_costs(tree):
                    frames = [frame for layer in cost['layers'].values() for frame in layer['frames']]
                    if any(own.get((source, repeat, frame)) == number for frame in frames):
                        node(cost['branch'], cost['at'])['search'] = cost['layers']
            for branch in {n['branch'].removeprefix(prefix) for n in group['nodes'].values()
                           if n['branch'].startswith(prefix)}:
                points = sorted(n['at'] for n in group['nodes'].values() if n['branch'] == prefix + branch)
                for start, end in zip(points, points[1:]):
                    covering = [f for f in concrete if (f['branch'] == branch or branch.startswith(f['branch'] + '/'))
                                and f['enter'] <= start and f['exit'] >= end]
                    if not covering:
                        continue
                    frame = min(covering, key=lambda f: f['exit'] - f['enter'])
                    relevant = [f for f in local if f['branch'] == branch
                        and f['category'] in ('a2', 'a3', 'pending')
                        and records.physical_start(f) < end
                        and (f['end'] if f['end'] is not None else report['control_steps']) > start]
                    category = max((f['category'] for f in relevant),
                                   key=lambda c: ('pending', 'a2', 'a3').index(c), default='success')
                    group['edges'].append(dict(id=f'{prefix}{branch}:{start}-{end}',
                        source=node(branch, start)['id'], target=node(branch, end)['id'],
                        name=frame['name'], category=category, start=start, end=end,
                        batch=source, repeat=repeat, arguments=_compact(frame['arguments']),
                        report_file=_report_file(source, seed, repeat),
                        cpu_s=frame.get('cpu_s'), wall_s=frame.get('wall_s'),
                        timing_scope=dict(name=frame['name'], start=frame['enter'], end=frame['exit'],
                                          note='measured enclosing call; overlapping edges do not have disjoint costs'),
                        failures=[_summary_failure(f) for f in relevant],
                        checks=_summarize_checks(c for c in tree.get('checks', [])
                                if start <= c.get('at', -1) <= end and c.get('frame') in {f['id'] for f in fs})))
            if '/attempts/' not in source:
                root = step_frames[source, repeat, number]
                completed = next((s for s in report.get('steps', []) if s['step'] == root['arguments']['index']), None)
                preceding = source_path/'checkpoints'/f'step{root["arguments"]["index"] - 1}.hdf5'
                group['attempts'].append(dict(batch=str(batch), commit=read(batch/'summary.json').get('commit'),
                    repeat=repeat, source_step=root['arguments']['index'], name=root['name'],
                    result=root.get('outcome') or ('success' if completed else report['status']),
                    episode_status=report['status'], start=root['enter'], end=root['exit'],
                    cpu_s=(completed or root).get('cpu_s'), wall_s=(completed or root).get('wall_s'),
                    parent_checkpoint=str(preceding) if preceding.is_file() else report.get('from'),
                    batch_parent_checkpoint=report.get('from'), failure=report.get('failure'),
                    entry=node(root['branch'], root['enter'])['id'],
                    calls=_call_tree(tree['frames'], root)))
    for number, group in groups.items():
        group['nodes'] = list(group['nodes'].values())
        group['edges'] = list({e['id']: e for e in group['edges']}.values())
        # The final call tree retains discarded branches, but its trajectory has
        # been rewound. Their control coordinates belong to the archive, even
        # when those numbers also fit within the final (different) trajectory.
        for item in [*group['nodes'], *group['edges']]:
            if 'repeat' not in item:
                continue
            repeat = item['repeat']
            branch = item.get('branch', item.get('source', '').rsplit('@', 1)[0])
            branch = branch.removeprefix(f'{batch.name}/r{repeat}/')
            if (repeat, branch) in branch_sources:
                item['batch'] = str(branch_sources[repeat, branch])
                item['report_file'] = f'episodes/{seed}.json'
    return groups


MAIN_ARGUMENTS = ('obj', 'arm', 'to', 'container', 'region', 'part', 'surface', 'target')


def _unwrapped(arguments):
    """A frame's arguments with an action wrapper (``{'$': 'Place', 'value': {...}}``) opened."""
    arguments = dict(arguments) if isinstance(arguments, dict) else {}
    action = arguments.get('action')
    if isinstance(action, dict) and isinstance(action.get('value'), dict):
        arguments = {**action['value'], **{k: v for k, v in arguments.items() if k != 'action'}}
    return arguments


def _callback(value):
    return isinstance(value, dict) and value.get('$') == 'callback'


def _call_tree(frames, root):
    """The recorded calls under a step frame: kind, name, interval, main arguments, and for a
    callback argument the concrete value a call below it received under the same name."""
    children = defaultdict(list)
    for frame in frames:
        children[frame['parent']].append(frame)
    def descendants(frame):
        for child in children[frame['id']]:
            yield child
            yield from descendants(child)
    def build(frame):
        arguments = _unwrapped(frame['arguments'])
        main = {k: arguments[k] for k in MAIN_ARGUMENTS if arguments.get(k) is not None}
        if isinstance(arguments.get('actions'), list):
            main['actions'] = _command_text(frame)
        resolved = {}
        for key, value in main.items():
            if _callback(value):
                found = next((_unwrapped(d['arguments'])[key] for d in descendants(frame)
                              if _unwrapped(d['arguments']).get(key) is not None and not _callback(_unwrapped(d['arguments'])[key])), None)
                if found is not None:
                    resolved[key] = _compact(found)
        return dict(id=frame['id'], kind=frame['kind'], name=frame['name'], enter=frame['enter'], exit=frame['exit'],
                    branch=frame['branch'], control=bool(frame.get('control')), main=_compact(main), resolved=resolved,
                    arguments=_compact(frame['arguments']), children=[build(c) for c in children[frame['id']]])
    return build(root)


def _command_text(frame):
    """A chain's actions in one line: each action and its object."""
    def one(action):
        value = action.get('value') if isinstance(action, dict) else None
        target = value.get('obj') if isinstance(value, dict) else None
        name = action.get('$', '?') if isinstance(action, dict) else str(action)
        return f'{name}({target})' if isinstance(target, str) else name
    return ' → '.join(one(a) for a in frame['arguments']['actions'])


def _ordered(steps, keys, revisions):
    """Final steps in order, each deprecated one just before its (likely) replacement."""
    final = [steps[key] for key in keys]
    positions = {key: i for i, key in enumerate(keys)}
    placed = []
    for key in sorted(k for k in steps if k not in positions):
        step = steps[key]
        step['replacement'] = _replacements(key, revisions)
        step['source_steps'] = sorted({a['source_step'] for a in step['attempts']})
        candidates = [positions[n] for n in step['replacement']['candidates'] if n in positions]
        previous = step['replacement']['previous']
        at = min(candidates) if candidates else positions[previous] + 1 if previous in positions else len(final)
        placed.append((at, key))
    ordered = list(final)
    for at, key in sorted(placed, reverse=True):
        ordered.insert(at, steps[key])
    return ordered


def batches(line):
    """A line's runs in time order (names start with the time), and its final joined batch.

    A joined batch only splices recorded parts; it is the developer's final
    chain, not a run of its own. Joined names need not carry a time, so the
    final one covers the most steps, and among those ends on the newest part.
    """
    runs, joined = [], []
    for path in sorted(p.parent for p in Path(line).glob('*/summary.json')):
        summary = read(path/'summary.json')
        if 'joined' in summary:
            joined.append(((len(summary['steps']['rows']), Path(summary['joined'][-1]).name), path))
        else:
            runs.append(path)
    return runs, max(joined)[1] if joined else None


def latest(line):
    """Start time (YYYYMMDD-HHMMSS) of the line's newest batch, read from batch names; None without any."""
    times = [m.group(0) for p in Path(line).glob('*/summary.json') if (m := re.match(r'\d{8}-\d{6}', p.parent.name))]
    return max(times, default=None)


def newest_calls(line):
    """The newest run's first calls document, which shows how the line is recorded (None: no calls)."""
    for run in reversed(batches(line)[0]):
        calls = sorted((run/'calls').glob('*.json.gz'))
        if calls:
            return calls[0]
    return None


def recorded_facts(calls):
    """Whether a calls document has the search-tree facts (#114); earlier lines cannot be browsed."""
    with gzip.open(calls, 'rt') as stream:
        return json.load(stream).get('version') == codec.SCHEMA


def _chain(runs, joined, seed):
    """``{batch: (first, last, repeat)}``: the batch whose recording each step of the final trajectory uses.

    A joined batch lists its parts; otherwise the newest run that reached the
    last step is followed up its ``from`` parents. A part covers its first step
    up to the step before the next part's first.
    """
    if joined is not None:
        parts = [resolve_batch_ref(p, joined) for p in read(joined/'summary.json')['joined']]
    else:
        parts, current = [], None
        for run in reversed(runs):
            summary = read(run/'summary.json')
            report = run/'episodes'/f'{seed}.json'
            last = (summary.get('steps_run') or [None, None])[1]
            if report.is_file() and read(report)['status'] == 'success' and last in (None, len(read(report)['step_names'])):
                current = run
                break
        while current is not None and current in runs:
            parts.insert(0, current)
            source = read(current/'summary.json').get('from')
            current = resolve_batch_ref(source, current) if source else None
            while current is not None and not (current/'summary.json').is_file():
                current = current.parent  # a checkpoint file inside its batch
    starts = [(read(p/'summary.json').get('steps_run') or [1, None])[0] for p in parts]
    return {p: (first, (starts[i + 1] - 1) if i + 1 < len(parts) else None, 0)
            for i, (p, first) in enumerate(zip(parts, starts))}


def _signature(batch):
    """Changes whenever a batch's summary, reports or call traces are rewritten."""
    files = [batch/'summary.json', *sorted((batch/'episodes').glob('*.json')), *sorted((batch/'calls').glob('*'))]
    return [[str(f.relative_to(batch)), f.stat().st_mtime_ns, f.stat().st_size] for f in files if f.is_file()]


def _relative(value, line):
    """Source paths in a cached graph, relative to the line (it may be renamed to a legacy round)."""
    if isinstance(value, dict):
        return {k: _relative(v, line) for k, v in value.items()}
    if isinstance(value, list):
        return [_relative(v, line) for v in value]
    if isinstance(value, str) and value.startswith(str(line) + '/'):
        return value[len(str(line)) + 1:]
    return value


def _graph(batch, line, seed):
    """A batch's per-step graph, cached in the line's .iteration/graphs/ until the batch changes."""
    cache = line/CACHE/'graphs'/f'{batch.name}.json'
    signature = _signature(batch)
    if cache.is_file():
        cached = read(cache)
        if cached['signature'] == signature and cached.get('format') == GRAPH_FORMAT:
            return cached
    report = read(batch/'episodes'/f'{seed}.json')
    summary = read(batch/'summary.json')
    value = dict(signature=signature, format=GRAPH_FORMAT, step_names=report['step_names'], commit=summary.get('commit'),
                 command=summary.get('command'),
                 groups=_relative(dict(_batch_graph(batch, seed)), str(line)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    write(cache, value)
    return value


def line_index(line, seed=0, progress=None):
    """Index a line's batches into steps (see the module docstring) and save .iteration/index.json."""
    line = Path(line).resolve()
    runs, joined = batches(line)
    chain = _chain(runs, joined, seed)
    reports = [(run, run/'episodes'/f'{seed}.json') for run in runs]
    # Runs from before #114 have no search-tree facts: they are counted, not read.
    traces = {run: run/read(report).get('call_trace', 'calls/missing') for run, report in reports if report.is_file()}
    earlier = [run for run, trace in traces.items() if not trace.is_file() or not recorded_facts(trace)]
    recorded = [run for run, report in reports if report.is_file() and run not in earlier]
    final_source = joined or (recorded[-1] if recorded else None)
    names = read(final_source/'episodes'/f'{seed}.json')['step_names'] if final_source else []
    keys = _keys(names)
    steps = {key: dict(step=i + 1, name=name, deprecated=False, attempts=[], nodes=[], edges=[], failures=[])
             for i, (key, name) in enumerate(zip(keys, names))}
    revisions = []
    for number, run in enumerate(recorded):
        if progress:
            progress(number, len(recorded), run.name)
        graph = _graph(run, line, seed)
        revisions.append(_keys(graph['step_names']))
        span = chain.get(run)
        for key, group in graph['groups'].items():
            step = steps.setdefault(key, dict(step=None, name=key.split('#')[0], deprecated=True,
                                              attempts=[], nodes=[], edges=[], failures=[]))
            step['nodes'].extend(group['nodes'])
            step['failures'].extend(group['failures'])
            step['edges'].extend(group['edges'])
            for attempt in group['attempts']:
                attempt['batch'] = run.name
                attempt['commit'] = graph['commit']
                attempt['run_command'] = graph['command']
                attempt['selected'] = bool(span and span[2] == attempt['repeat'] and span[0] <= attempt['source_step']
                                           and (span[1] is None or attempt['source_step'] <= span[1]))
                step['attempts'].append(attempt)
    for step in steps.values():
        step['attempts'].sort(key=lambda a: (a['batch'], a['repeat']))
        _matching_prefixes(step, seed, base=line)
        for node in step['nodes']:
            node['a1_count'] = len(node['a1'])
            buckets = {}
            for failure in node['a1']:
                key = (failure.get('stage'), failure.get('reason'), failure.get('criterion_id'))
                bucket = buckets.setdefault(key, dict(stage=key[0], reason=key[1], criterion_id=key[2], count=0, example_ids=[]))
                bucket['count'] += 1
                if len(bucket['example_ids']) < 5:
                    bucket['example_ids'].append(failure['id'])
            node['a1'] = list(buckets.values())
    summary = read((final_source or runs[-1])/'summary.json') if runs else {}
    value = dict(version=INDEX_FORMAT, line=line.name, sources=sources(line), scene=summary.get('scene'), robot=summary.get('robot'),
                 expert=summary.get('expert'), layout=summary.get('layout'), seed=seed,
                 joined=joined.name if joined else None, chain=[p.name for p in chain],
                 unrecorded=[run.name for run, report in reports if not report.is_file()],
                 before_facts=[run.name for run in earlier],
                 steps=_ordered(steps, keys, revisions))
    value['media'] = media_plan(line, value)
    write(line/CACHE/'index.json', value)
    return value


def sources(line):
    """Signatures of every batch an index of line reads; a change means rebuild."""
    runs, joined = batches(line)
    return {run.name: _signature(run) for run in [*runs, *([joined] if joined else [])]}


def stale(line):
    """Whether the line's cached index is missing or was built from other batch files."""
    index = Path(line)/CACHE/'index.json'
    if not index.is_file():
        return True
    cached = read(index)
    return cached.get('version') != INDEX_FORMAT or cached['sources'] != sources(line)


RESULTS = {'success': '通过', 'b': 'b', 'pending': '中断', 'error': '出错'}


def forest(step):
    """One tree per outer attempt (batch × repeat) of a step, drawn from its physical graph.

    A RetryState branch starts in the state it restored, which is its ancestor
    branch's state at that control frame: the two are one node, so a retry
    forks there as a sibling, and a restore without motion in between leaves no
    edge (the node counts it as search overhead). The discarded branch keeps
    its motion and ends in a node marked with what triggered the rollback.
    """
    trees = []
    for attempt in step['attempts']:
        prefix = attempt['entry'].rsplit('@', 1)[0]
        prefix = prefix[:prefix.index(f'/r{attempt["repeat"]}/') + len(f'/r{attempt["repeat"]}/')]
        nodes = {n['id']: n for n in step['nodes'] if n['id'].startswith(prefix)}
        at = {}
        for node in nodes.values():
            at.setdefault(node['branch'], {})[node['at']] = node['id']
        canonical = {}
        for branch in sorted(at, key=lambda b: b.count('/')):
            start = min(at[branch])
            parent, fork = branch, None
            while fork is None and parent.count('/') > prefix.count('/'):
                parent = parent.rsplit('/', 1)[0]
                fork = at.get(parent, {}).get(start)
            for frame, identity in at[branch].items():
                canonical[identity] = canonical.get(fork, fork) if fork and frame == start else identity
        merged = {}
        for identity, node in nodes.items():
            target = merged.setdefault(canonical[identity], dict(
                id=canonical[identity], at=node['at'], branch=node['branch'].removeprefix(prefix), batch=node['batch'],
                media=node.get('media'), media_ready=node.get('media_ready', False),
                media_unavailable=node.get('media_unavailable'), a1=[], a1_count=0,
                checks=[], retries=[], members=[], search={}))
            target['members'].append(identity)
            target['a1_count'] += node.get('a1_count', 0)
            target['a1'] += node.get('a1', [])
            target['checks'] += node.get('checks', [])
            # Retried branches searched again from the same state: their layers add up per layer.
            for layer, row in node.get('search', {}).items():
                total = target['search'].setdefault(layer, dict(count=0, cpu_s=0., wall_s=0., reasons={}))
                for cost in ('count', 'cpu_s', 'wall_s'):
                    total[cost] += row[cost]
                for reason, count in row['reasons'].items():
                    total['reasons'][reason] = total['reasons'].get(reason, 0) + count
        edges, rollbacks = [], {}
        for edge in step['edges']:
            if edge['source'] not in nodes:
                continue
            source, target = canonical[edge['source']], canonical[edge['target']]
            if edge['category'] == 'rollback':
                trigger = edge['trigger']
                rollbacks[edge['source']] = trigger, target
                merged[target]['retries'].append(dict(trigger=trigger, moved=source != target,
                                                      failures=edge.get('failures', [])))
            elif source != target and not any(edge['start'] < frame < edge['end']
                                              for frame in at[nodes[edge['source']]['branch']]):
                # A coarser edge from an archive read before its branch's later
                # states were known spans finer ones; the finer edges remain.
                edges.append(dict(edge, source=source, target=target))
        # The discarded branch ends where its last motion did; a restore from
        # the fork itself (no motion) has no node of its own to mark.
        for source, (trigger, fork) in rollbacks.items():
            branch = nodes[source]['branch']
            last = canonical[at[branch][max(at[branch])]]
            if last != fork:
                merged[last]['terminal'] = trigger
        children = {}
        for edge in edges:
            children.setdefault(edge['source'], []).append(edge)
        root = canonical[attempt['entry']]
        # The step's exit: the deepest surviving branch's node at the step's end frame.
        ends = [n for n in merged.values() if n['at'] == attempt['end'] and not n.get('terminal')]
        exit_node = max(ends, key=lambda n: n['branch'].count('/'), default=None)
        chain = []
        if attempt['selected'] and exit_node:
            parents = {e['target']: e for e in edges}
            current = exit_node['id']
            while current in parents:
                chain.append(parents[current]['id'])
                current = parents[current]['source']
            chain.reverse()
        # Whether a step that did not pass was abandoned (b) is the reader's call.
        result = attempt['result']
        if result != 'success' and any(c['category'] == 'b' for n in merged.values() for c in n['checks']):
            result = 'b'
        batch = attempt['batch'].removeprefix('batches/')
        started = re.match(r'^(\d{8})-(\d{6})', batch)
        failures = [f for f in step['failures'] if f['repeat'] == attempt['repeat']
                    and Path(f['batch']).parts[0] == attempt['batch']]
        trees.append(dict(id=prefix.rstrip('/'), batch=attempt['batch'], repeat=attempt['repeat'],
            calls=attempt.get('calls'), run_command=attempt.get('run_command'),
            failure_counts=dict(Counter(f['category'] for f in failures)), frames=attempt['end'] - attempt['start'],
            commit=attempt.get('commit'), source_step=attempt['source_step'], selected=attempt['selected'],
            started=datetime.strptime(''.join(started.groups()), '%Y%m%d%H%M%S').isoformat() if started else None,
            result=result, result_label=RESULTS.get(result, result), step_outcome=attempt['result'],
            cpu_s=attempt.get('cpu_s'), wall_s=attempt.get('wall_s'), parent_checkpoint=attempt.get('parent_checkpoint'),
            prefix_note=attempt.get('prefix_note'), root=root, exit=exit_node and exit_node['id'], chain=chain,
            nodes=list(merged.values()), edges=edges))
    trees.sort(key=lambda t: (t['started'] or '', t['batch'], t['repeat']))
    return trees


def media_plan(line, index, *, stride=5):
    """Name every physical node's still and moving edge's preview (``item['media']``, relative to
    the line's .iteration/); returns all spans. Names depend only on the recorded source and
    interval, so equal states share a file and re-indexing never orphans published media."""
    spans, recordings = {}, {}
    for step in index['steps']:
        label = f's{step["step"]}' if step['step'] else step['name']
        for kind in ('nodes', 'edges'):
            for item in step[kind]:
                if kind == 'edges' and (item['category'] == 'rollback' or item['end'] <= item['start']):
                    continue
                source = Path(line)/item['batch']
                report_file = item.get('report_file') or _report_file(source, index['seed'], item.get('repeat', 0))
                report_key = (source, report_file)
                if report_key not in recordings:
                    recordings[report_key] = records.trajectory(source, read(source/report_file))
                if recordings[report_key] is None:
                    item['media'] = None
                    item['media_unavailable'] = '未录制轨迹'
                    continue
                start, end = (item['at'], item['at']) if kind == 'nodes' else (item['start'], item['end'])
                key = hashlib.sha256(f'{item["batch"]}:{item.get("repeat", 0)}:{start}:{end}:{stride}'.encode()).hexdigest()[:20]
                item['media'] = f'media/{key}.{"jpg" if kind == "nodes" else "mp4"}'
                spans.setdefault(key, dict(key=key, kind='still' if kind == 'nodes' else 'clip', batch=item['batch'],
                    repeat=item.get('repeat', 0), start=start, end=end, path=item['media'], stride=stride,
                    step=label, report_file=item.get('report_file')))
    return list(spans.values())


def prepare_media(line, spans, output, *, seed=0):
    """Sample recorded states of ``spans`` (from media_plan) into one render.py batch, without
    stepping physics.

    A clip includes both endpoints; its sparse frames render at 25 fps and
    ``finish_media`` restores their original control-time spacing in the preview.
    """
    import h5py
    import numpy as np
    line, output = Path(line).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    index = read(line/CACHE/'index.json')
    def write_samples(initial_out, states_out, initial, states, points):
        # Read a dataset once per span, rather than traversing the HDF5 tree for
        # every control sample. The source-local indices are strictly increasing.
        indices = np.array([at - 1 for at in points if at > 0], dtype=int)
        for key, value in states.items():
            if isinstance(value, h5py.Group):
                write_samples(initial_out.create_group(key), states_out.create_group(key),
                              initial[key], value, points)
            else:
                sampled = value[indices] if len(indices) else value[:0]
                if points[0] == 0:
                    sampled = np.concatenate([initial[key][()], sampled])
                initial_out.create_dataset(key, data=sampled[:1])
                states_out.create_dataset(key, data=np.concatenate([sampled[1:], sampled[-1:]]))
    available = []
    for span in spans:
        source = line/span['batch']
        report = read(source/(span.get('report_file') or _report_file(source, seed, span['repeat'])))
        path = records.trajectory(source, report)
        if path is not None:
            available.append((dict(span), path))
    spans = [span for span, _ in available]
    with h5py.File(output/'trajectories.hdf5', 'w') as target:
        data = target.create_group('data')
        for number, (span, path) in enumerate(available):
            source = line/span['batch']
            with h5py.File(path) as original:
                data.attrs.update(dict(original['data'].attrs))
                group = original[f'data/demo_{seed}']
                start, end = span['start'], span['end']
                if not 0 <= start <= end <= group.attrs['num_samples']:
                    raise ValueError(f'{source}: span {start}:{end} exceeds recorded trajectory')
                points = list(range(start, end + 1, span['stride']))
                if points[-1] != end:
                    points.append(end)
                out = data.create_group(f'demo_{number}')
                out.attrs.update(seed=number, status='success', num_samples=len(points),
                    source=str(source), source_start=start, source_end=end,
                    subtasks=dumps([dict(instruction=f'{span["step"]}: source frames {start}–{end}', start=0, end=len(points))]))
                write_samples(out.create_group('initial_state'), out.create_group('states'),
                              group['initial_state'], group['states'], points)
                span.update(seed=number, controls=points)
        data.attrs['total'] = sum(g.attrs['num_samples'] for g in data.values())
    write(output/'summary.json', dict(scene=index['scene'], robot=index['robot'], expert=index['expert'],
        physics='isaac-cpu', seeds=dict(range=f'0:{len(spans)}', count=len(spans)), videos={}, fails={}))
    write(output/'media.json', dict(cache=str(line/CACHE), spans=spans, source_control_hz=25,
                                    note='state-only sparse replay; no physics advanced'))
    return spans


def _preview_filter(timing=''):
    """Arrange render.py's captioned 2×2 grid into top + two stacked wrists."""
    return (f'[0:v]{timing}split=4[t0][l0][r0][c0];'
        '[t0]crop=iw/2:(ih-34)/2:0:34[t];'
        '[l0]crop=iw/2:(ih-34)/2:iw/2:34,scale=320:240[l];'
        '[r0]crop=iw/2:(ih-34)/2:0:34+(ih-34)/2,scale=320:240[r];'
        '[l][r]vstack=inputs=2[w];[t][w]hstack=inputs=2[panels];'
        '[c0]crop=iw:34:0:0,scale=960:26[c];[c][panels]vstack=inputs=2[preview]')


def _still_filter():
    """A node's three views (top, left wrist, right wrist) side by side at full panel size."""
    return ('[0:v]split=3[t0][l0][r0];'
        '[t0]crop=iw/2:(ih-34)/2:0:34[t];'
        '[l0]crop=iw/2:(ih-34)/2:iw/2:34[l];'
        '[r0]crop=iw/2:(ih-34)/2:0:34+(ih-34)/2[r];'
        '[t][l][r]hstack=inputs=3[preview]')


def finish_media(batch, *, ready_only=False, overwrite=False, kinds=('still', 'clip')):
    """Publish compact media; ready_only exposes completed clips during rendering."""
    batch = Path(batch)
    manifest = read(batch/'media.json')
    cache = Path(manifest['cache'])
    completed = 0
    for span in manifest['spans']:
        if span['kind'] not in kinds:
            continue
        source = batch/f'videos/{span["seed"]}/merge.mp4'
        target = cache/span['path']
        if target.is_file() and not overwrite:
            continue
        if ready_only and not source.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.stem+'.',
                                         suffix=target.suffix, delete=False) as stream:
            temporary = Path(stream.name)
        command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-threads', '1', '-i', str(source)]
        if span['kind'] == 'still':
            command += ['-filter_complex', _still_filter(), '-map', '[preview]',
                        '-frames:v', '1', '-q:v', '3', '-threads', '1', str(temporary)]
        else:
            # Render.py writes at 25 fps. Timestamp source control samples rather
            # than changing playback speed of the underlying physical motion.
            duration = (span['end']-span['start'])/25
            timing = f'settb=1/1000,setpts=min(N*{span["stride"]}/25\\,{duration})/TB,'
            command += ['-filter_complex', _preview_filter(timing), '-map', '[preview]',
                        '-fps_mode', 'vfr', '-an', '-c:v', 'libx264', '-threads', '1', '-preset', 'fast', '-crf', '25',
                        '-movflags', '+faststart', str(temporary)]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            temporary.unlink(missing_ok=True)
            if ready_only:
                # The renderer may have created merge.mp4 but not written its
                # final metadata yet. A later publication pass retries it.
                continue
            raise
        temporary.replace(target)
        completed += 1
    return completed
