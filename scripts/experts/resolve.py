"""Re-solve expert numerical problems with MPlib, without starting Isaac.

The first pass replays the recorded MPlib random stream prefix. Later passes
use independent seeds. Every call starts in its recorded measured world: rates
measure numerical solves, not physical execution or end-to-end skill success.
"""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from table_1000.common.serialization import dump, dumps


def selection(document, frame=None, plan=None, decision=None):
    operations = [op for op in document['operations'] if op['parent'] is None]
    if frame is not None:
        operations = [op for op in operations if op['frame'] == frame]
    if decision is not None:
        descendants = {decision}
        for item in document['decisions']:
            if item['parent'] in descendants:
                descendants.add(item['id'])
        operations = [op for op in operations if op['decision'] in descendants]
    if plan is not None:
        operations = [op for op in operations if op['id'] == plan]
    return [op['id'] for op in operations]


def aggregate(document, runs):
    by_id = {op['id']: op for op in document['operations']}
    totals = []
    for number in sorted({item['operation'] for run in runs for item in run}):
        found = [item for run in runs for item in run if item['operation'] == number]
        op = by_id[number]
        totals.append(dict(operation=number, kind=op['kind'], frame=op['frame'], decision=op['decision'],
                           candidate=op['candidate'], stage=op['stage'], original=op['result']['status'],
                           original_reason=op['result'].get('reason', op['result']['status']),
                           successes=sum(item['result']['status'] == 'Success' for item in found), count=len(found),
                           statuses=dict(Counter(item['result']['status'] for item in found)),
                           reasons=dict(Counter(item['result'].get('reason', item['result']['status']) for item in found)),
                           first_same=found[0]['same_status'] and found[0]['same_reason'], diagnostic=op.get('diagnostic')))
    return totals


def candidate_comparison(document, operations):
    """Compact candidate verdicts with references to every numerical gate's comparison."""
    groups = {}
    for item in document['decisions']:
        for event in item['events']:
            key = (item['id'], event['candidate'])
            group = groups.setdefault(key, dict(decision=item['id'], name=item['name'], candidate=event['candidate'],
                                                verdicts=[], tests=[]))
            if event['outcome']:
                group['verdicts'].append({name: event[name] for name in ('stage', 'outcome')})
    for operation in operations:
        key = (operation['decision'], operation['candidate'])
        group = groups.setdefault(key, dict(decision=operation['decision'], candidate=operation['candidate'],
                                            verdicts=[], tests=[]))
        group['tests'].append(operation)
    return list(groups.values())


def decision_verdicts(document, root):
    descendants, verdicts = {root}, {}
    for item in document['decisions']:
        if item['parent'] in descendants:
            descendants.add(item['id'])
        if item['id'] not in descendants:
            continue
        for event in item['events']:
            if event['outcome']:
                verdicts.setdefault((item['name'], event['candidate']), []).append(
                    dict(stage=event['stage'], outcome=event['outcome']))
    return verdicts


def compare_decisions(original, root, replayed):
    before, after = decision_verdicts(original, root), decision_verdicts(replayed, 0)
    keys = dict.fromkeys([*before, *after])
    return [dict(name=name, candidate=candidate, original=before.get((name, candidate), []),
                 replayed=after.get((name, candidate), []),
                 same=before.get((name, candidate)) == after.get((name, candidate))) for name, candidate in keys]


def last_decision(document, frame=None):
    names = ('Grasping.reach', 'Placing.put_down', 'Motion.push', 'Handing.handoffs', 'Base.base_for')
    choices = [item for item in document['decisions'] if item['name'] in names
               and (frame is None or item['frame'] == frame)]
    # A base search is a child of the failed skill, but the final grasp/place search is
    # the selection whose failure the skill reports.
    return choices[-1]['id']


def main(argv=None):
    from table_1000.common.batch import Load, commit, parse_seeds, run_name
    from table_1000.experts.audit.problems import Offline, load
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('batch', type=Path)
    parser.add_argument('--seeds', nargs='+', required=True, help='seed numbers or a:b')
    parser.add_argument('--repeat', type=int, default=0, help='recorded episode repeat suffix')
    parser.add_argument('--frame', type=int, help='skill frame number')
    parser.add_argument('--plan', type=int, help='operation number (motion, plan_rrt or plan_screw)')
    parser.add_argument('--decision', type=int, help='decision number and its descendants')
    parser.add_argument('--redecide', nargs='?', const='last', help='rerun a decision (number or last) with ideal following')
    parser.add_argument('--runs', type=int, default=10, help='original stream once, then independent random streams')
    parser.add_argument('--output', type=Path, help='new experiment directory (default: '
                        'runs/resolve/<scene>/<date>-<time>-<commit>[-<name>]/)')
    parser.add_argument('--name', help='label for the default output directory')
    args = parser.parse_args(argv)
    if args.output is None:
        scene = json.loads((args.batch / 'summary.json').read_text())['scene']
        args.output = ROOT / 'runs' / 'resolve' / scene / run_name(commit(ROOT), args.name)
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    for seed in parse_seeds(args.seeds):
        name = f'{seed}-{args.repeat}' if args.repeat else str(seed)
        episode = json.loads((args.batch / 'episodes' / f'{name}.json').read_text())
        if episode.get('solve_trace') is None:
            parser.error(f'{name}: no solve_trace was saved; rerun run_expert.py with --problems all to collect it. '
                         'For a failed skill, create a frame snapshot with scripts/experts/failure_snapshots.py '
                         'and rerun with --from <snapshot batch> --steps k --problems all')
        document = load(args.batch / episode['solve_trace'])
        number = (last_decision(document, args.frame) if args.redecide == 'last' else
                  int(args.redecide) if args.redecide else None)
        selected = selection(document, args.frame, args.plan, args.decision if args.decision is not None else number)
        if number is not None and args.plan is None and args.decision is None:
            plans = [op['id'] for op in document['operations'] if op['parent'] is None and op['kind'] != 'ik'
                     and (args.frame is None or op['frame'] == args.frame)]
            if plans:
                selected = sorted(set([*selected, plans[-1]]))
        started = time.perf_counter()
        print(f'{name}: solving {len(selected)} recorded problems, {args.runs} runs', flush=True)
        with Load() as load_record:
            resolver = Offline(document, args.output / f'planning-{name}')
            runs = []
            for index in range(args.runs):
                runs.append(resolver.replay(selected, None if index == 0 else seed * 1000003 + index))
                print(f'{name}: problems {index + 1}/{args.runs}', flush=True)
            report = dict(seed=seed, source=episode.get('replay'), selected=selected, operations=aggregate(document, runs), runs=runs,
                          solve_trace=episode['solve_trace'],
                          decisions=[{key: item[key] for key in ('id', 'parent', 'frame', 'name')} for item in document['decisions']])
            report['candidates'] = candidate_comparison(document, report['operations'])
            if args.redecide:
                decisions = []
                for index in range(args.runs):
                    outcome = resolver.decide(number, None if index == 0 else seed * 1000003 + index)
                    detail = outcome.pop('problems')
                    outcome['candidates'] = compare_decisions(document, number, detail)
                    with gzip.open(args.output / f'{name}-decision-{number}-{index}.json.gz', 'wt') as file:
                        dump(detail, file, separators=(',', ':'))
                    decisions.append(outcome)
                    print(f'{name}: decision {number} {index + 1}/{args.runs}', flush=True)
                report['redecide'] = dict(decision=number, successes=sum(item['success'] for item in decisions),
                                          count=len(decisions), runs=decisions,
                                          original_result=document['decisions'][number].get('result'),
                                          original_exception=document['decisions'][number].get('exception'))
        report.update(load=load_record.record(), wall_s=time.perf_counter() - started)
        with gzip.open(args.output / f'{name}.json.gz', 'wt') as file:
            dump(report, file, ensure_ascii=False, separators=(',', ':'))
        brief = {key: report[key] for key in ('seed', 'source', 'selected', 'operations', 'load', 'wall_s')}
        if 'redecide' in report:
            brief['redecide'] = {key: value for key, value in report['redecide'].items() if key != 'runs'}
        reports.append(brief)
        print(f'{name}: {len(selected)} problems, {args.runs} runs', flush=True)
    (args.output / 'summary.json').write_text(dumps(reports, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
