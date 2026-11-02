"""Offline physical-state search graph; media is generated only on request."""
from pathlib import Path

from table_1000.common.serialization import dumps

from .records import documents, failures as classified_failures, media_batch, physical_start, search_costs


def graph(batch, seed, repeat=None):
    nodes, edges, failures = {}, {}, {}
    records = list(documents(batch, seed, repeat))
    archives = {(report['recording_root'], report.get('repeat', 0),
                 report['archive']['failure_id'], report['archive']['branch']): (source, report)
                for source, report, _ in records if 'archive' in report}
    namespaces = list(dict.fromkeys((report['recording_root'], report.get('repeat', 0))
                                    for source, report, _ in records))
    scoped = []
    for source, report, tree in records:
        identity = (report['recording_root'], report.get('repeat', 0))
        prefix = f'part{namespaces.index(identity)}/' if len(namespaces) > 1 else ''
        tree = {**tree, **{key: [dict(item, branch=prefix + item['branch']) for item in tree[key]]
                          for key in ('frames',)}}
        scoped.append((source, report, tree))
    for source, report, tree in scoped:
        frames = tree['frames']
        concrete = [f for f in frames if f['exit'] is not None
                    and (f['kind'] == 'step' or (f['concrete'] and f['exit'] > f['enter']))]
        for frame in concrete:
            for at in (frame['enter'], frame['exit']):
                key = f"{frame['branch']}@{at}"
                nodes.setdefault(key, dict(id=key, branch=frame['branch'], at=at, batch=str(source), report=report, a1=[]))
    for source, report, tree in records:
        if 'archive' in report:
            continue
        identity = (report['recording_root'], report.get('repeat', 0))
        prefix = f'part{namespaces.index(identity)}/' if len(namespaces) > 1 else ''
        for cost in search_costs(tree):
            branch = prefix + cost['branch']
            key = f"{branch}@{cost['at']}"
            node = nodes.setdefault(key, dict(id=key, branch=branch, at=cost['at'], batch=str(source), report=report, a1=[]))
            node['search_costs'] = cost['layers']
            node['solver_coverage'] = cost['solver_coverage']
    for failure in classified_failures([batch], seed):
        report = failure['report']
        if repeat is not None and report.get('repeat', 0) != repeat:
            continue
        identity = (report['recording_root'], report.get('repeat', 0))
        prefix = f'part{namespaces.index(identity)}/' if len(namespaces) > 1 else ''
        recovery = failure['recovery']
        if recovery:
            recovery = dict(recovery, branch=prefix + recovery['branch'])
        value = dict(failure, branch=prefix + failure['branch'], recovery=recovery)
        failures[(value['branch'], value['id'], value['frame'])] = value
    for failure in failures.values():
        key = f"{failure['branch']}@{failure['at']}"
        node = nodes.setdefault(key, dict(id=key, branch=failure['branch'], at=failure['at'],
                                         batch=failure['batch'], report=failure['report'], a1=[]))
        if failure['category'] == 'a1':
            node['a1'].append({k: v for k, v in failure.items() if k not in ('state', 'batch', 'report', 'tree', '_rank')})
        elif failure['end'] is not None:
            end_key = f"{failure['branch']}@{failure['end']}"
            nodes.setdefault(end_key, dict(id=end_key, branch=failure['branch'], at=failure['end'],
                                           batch=failure['batch'], report=failure['report'], a1=[]))
    for source, report, tree in scoped:
        concrete = [f for f in tree['frames'] if (f['concrete'] or f['kind'] == 'step')
                    and f['exit'] is not None]
        for branch in {f['branch'] for f in concrete}:
            points = sorted(n['at'] for n in nodes.values() if n['branch'] == branch)
            for start, end in zip(points, points[1:]):
                covering = [f for f in concrete if f['branch'] == branch and f['enter'] <= start and f['exit'] >= end]
                if not covering:
                    continue
                frame = min(covering, key=lambda f: f['exit'] - f['enter'])
                key = (branch, start, end)
                relevant = [failure['category'] for failure in failures.values()
                    if (failure['branch'] == branch and failure['category'] in ('a2', 'a3', 'pending')
                            and physical_start(failure) < end
                            and (failure['end'] if failure['end'] is not None else report['control_steps']) > start)]
                category = max(relevant, key=lambda item: ('pending', 'a2', 'a3').index(item), default='success')
                edges[key] = dict(id=f'{branch}:{start}-{end}', source=f'{branch}@{start}', target=f'{branch}@{end}',
                                  name=frame['name'], category=category, start=start, end=end,
                                  batch=str(source), report=report, arguments=frame['arguments'])
    # Rollbacks are physical-tree facts even when their trigger was planning
    # rather than a failed physical check (a3).
    for source, report, tree in records:
        identity = (report['recording_root'], report.get('repeat', 0))
        prefix = f'part{namespaces.index(identity)}/' if len(namespaces) > 1 else ''
        for outcome in tree['outcomes']:
            if outcome['kind'] != 'rollback':
                continue
            start = outcome['restore']['at']
            branch, target_branch = prefix + outcome['branch'], prefix + outcome['restore']['branch']
            source_key, target_key = f"{branch}@{outcome['at']}", f'{target_branch}@{start}'
            owner, own_report = archives.get((*identity, outcome['failure_id'], outcome['branch']), (source, report))
            # A control-only rollback may have no failure marker or subsequent
            # action. Its two physical endpoints are still facts in the tree.
            for key, whose, at in ((source_key, branch, outcome['at']), (target_key, target_branch, start)):
                nodes.setdefault(key, dict(id=key, branch=whose, at=at, batch=str(owner), report=own_report, a1=[]))
            edges[(branch, outcome['id'], 'rollback')] = dict(id=f"{prefix}rollback-{outcome['id']}",
                source=source_key, target=target_key, name='RetryState.restore', category='rollback',
                start=start, end=start, batch=str(owner), report=own_report)
    return dict(nodes=list(nodes.values()), edges=list(edges.values()), seed=seed, repeat=repeat)


def export_view(batch, seed, output, *, media=False, repeat=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    value = graph(batch, seed, repeat)
    spans = []
    for node in value['nodes']:
        node['media'] = len(spans)
        spans.append(dict(start=node['at'], end=node['at'], batch=node['batch'], report=node['report']))
    for edge in value['edges']:
        edge['media'] = len(spans)
        spans.append(dict(start=edge['start'], end=edge['end'], batch=edge['batch'], report=edge['report']))
    if media:
        report = value['nodes'][0]['report']
        media_batch(batch, report, spans, output / 'media')
    for item in [*value['nodes'], *value['edges']]:
        item.pop('report')
        item.pop('batch')
    data = dumps(value, ensure_ascii=False).replace('<', '\\u003c')
    template = Path(__file__).with_name('search.html').read_text()
    (output / 'index.html').write_text(template.replace('SEARCH_DATA', data))
    (output / 'graph.json').write_text(dumps(value, indent=1, ensure_ascii=False))
    return value
