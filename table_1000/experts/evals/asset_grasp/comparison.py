"""Compare first-attempt acceptance counts and export grasp preference data.

uv run python scripts/benchmarks/asset_grasp.py compare --baseline runs/grasp/000103/unified-regression-02 --run runs/grasp/RUN --output outputs/asset-grasp/comparison
Different acceptance protocols are reported side by side, not treated as a
controlled measurement of the policy change alone. Thin-object categories are
reported separately; this classification never changes the grasp policy.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from table_1000.common.serialization import dumps

import numpy as np

from table_1000.common.batch import load_record
from table_1000.experts.evals.asset_grasp import acceptance_report, summarize

THIN = {'paper_label', 'receipt', 'plastic_scrap', 'wire_scrap', 'zip_bag', 'lab_notebook'}


def quantiles(values):
    values = list(values)
    return {'n': len(values), 'mean': float(np.mean(values)), **dict(zip(('p50', 'p90', 'max'),
        map(float, np.quantile(values, [.5, .9, 1.]))))} if values else None


def timing(reports):
    first = [report.get('first_attempt_timing', report) for report in reports]
    return {'trial_wall_s': quantiles(report['wall_s'] for report in reports),
            'first_attempt_wall_s': quantiles(report['wall_s'] for report in first),
            'first_attempt_cpu_s': quantiles(report['cpu_s'] for report in first),
            'preference_wall_s': quantiles(report['wall_s'] - initial['wall_s']
                                           for report, initial in zip(reports, first)),
            'preference_cpu_s': quantiles(report['cpu_s'] - initial['cpu_s']
                                          for report, initial in zip(reports, first)),
            'first_sim_s': quantiles(report['sim_s'] for report in first),
            'cpu_s': quantiles(report['cpu_s'] for report in reports),
            'note': 'trial wall and CPU include all preference attempts; first_attempt_timing records the first attempt separately.'}


def expert_comparison(before, after):
    pairs, records = {}, {}
    for label, root in [('before', before), ('after', after)]:
        summary = json.loads((root / 'summary.json').read_text())
        episodes = [json.loads(path.read_text()) for path in sorted((root / 'episodes').glob('*.json'))]
        if len(episodes) != summary['seeds']['count']:
            raise ValueError(f'incomplete expert run: {root}')
        pairs[label] = {row['seed']: row for row in episodes}
        subtasks = {}
        for episode in episodes:
            for subtask, elapsed in zip(episode['subtasks'], episode['subtask_wall_s'], strict=True):
                subtasks.setdefault(subtask['instruction'], []).append(elapsed)
        records[label] = {'run': str(root), 'commit': summary['commit'],
            'succeeded': summary['succeeded'], 'timing': {
                'wall_s': quantiles(row['wall_s'] for row in episodes),
                'cpu_s': quantiles(row['cpu_s'] for row in episodes),
                'subtask_wall_s': {name: quantiles(values) for name, values in subtasks.items()}},
            'load': summary['load']}
    if pairs['before'].keys() != pairs['after'].keys():
        raise ValueError('expert seeds disagree')
    rows = []
    for seed, old in sorted(pairs['before'].items()):
        new = pairs['after'][seed]
        rows.append({'seed': seed, 'old_status': old['status'], 'new_status': new['status'],
            'old_wall_s': old['wall_s'], 'new_wall_s': new['wall_s'],
            'wall_ratio': new['wall_s'] / old['wall_s'],
            'old_failure': old.get('failure'), 'new_failure': new.get('failure')})
    return {**records, 'rows': rows}


def compare(baseline, run):
    old = json.loads((baseline / 'summary.json').read_text())
    new = json.loads((run / 'summary.json').read_text())
    rows, preferences, loads, old_loads, peaks = [], [], [], [], {}
    for asset, summary in new['assets'].items():
        previous = old['assets'][asset]
        old_reports = [acceptance_report(json.loads(path.read_text())) for path in sorted((baseline / asset / 'trials').glob('*.json'))]
        if len(old_reports) != previous['expected']:
            raise ValueError(f'{asset}: incomplete baseline trials')
        reports = [acceptance_report(json.loads(path.read_text())) for path in sorted((run / asset / 'trials').glob('*.json'))]
        if len(reports) != summary['expected']:
            raise ValueError(f'{asset}: incomplete trials ({len(reports)}/{summary["expected"]})')
        # Older summaries may exclude measured trials for their motion strength.
        summary = {**summary, **summarize(reports, len(reports))}
        old_by_seed = {report['seed']: report for report in old_reports}
        seeds = {report['seed'] for report in reports}
        if not seeds <= old_by_seed.keys():
            raise ValueError(f'{asset}: baseline is missing requested seeds')
        old_reports = [old_by_seed[seed] for seed in sorted(seeds)]
        old_loads.extend({'load': report['load'], 'wall_s': report['wall_s']} for report in old_reports)
        old_counts = summarize(old_reports, len(old_reports))
        new_counts = summarize(reports, len(reports))
        old_first = old_counts['first_attempt_success']
        old_failures = Counter(report.get('failure', {}).get('reason', report['status'])
                               for report in old_reports
                               if report['status'] not in ('success', 'not_evaluated'))
        changes = [{'seed': report['seed'],
            'old_status': old_by_seed[report['seed']]['status'], 'new_status': report['status'],
            'old_reason': old_by_seed[report['seed']].get('failure', {}).get('reason', 'success'),
            'new_reason': report.get('failure', {}).get('reason', 'success')}
            for report in reports]
        for report in reports:
            if not (run / asset / report['trajectory']).is_file():
                raise FileNotFoundError(f'{asset} seed {report["seed"]}: trajectory')
            loads.append({'load': report['load'], 'wall_s': report['wall_s']})
            for key, value in report.get('hold', {}).get('motion_peaks', {}).items():
                peaks.setdefault(key, []).append(value)
        stability = Counter()
        preflight = Counter()
        for report in reports:
            plan = next((event for event in report.get('events', [])
                         if event['note'] == 'transport preflight'), None)
            if plan is not None:
                if plan['selected'] is None:
                    preflight['all_choices_unreachable'] += 1
                else:
                    preflight[f'selected_choice_{len(plan["choices"])}'] += 1
            hold = report.get('hold', {})
            if not hold:
                continue
            stability['executed'] += 1
            stability['dropped'] += bool(hold['dropped'])
            stability['translation_exceeded'] += int(hold['max_translation_m'] > .005)
            stability['rotation_exceeded'] += int(hold['max_rotation_rad'] > float(np.radians(5)))
            diagnostic = report.get('diagnostic_shake')
            if diagnostic:
                stability['diagnostic_executed'] += 1
                stability['diagnostic_held'] += bool(diagnostic['held'])
        rows.append({'asset': asset, 'thin': asset.split('/')[1] in THIN,
                     'old_first': old_first, 'old_expected': old_counts['expected'],
                     'old_assessed': old_counts['assessed'],
                     'old_not_evaluated': old_counts['not_evaluated'],
                     'old_failures': dict(old_failures),
                     'new_first': summary['first_attempt_success'], 'new_expected': summary['expected'],
                     'new_assessed': new_counts['assessed'],
                     'not_evaluated': new_counts['not_evaluated'],
                     'not_evaluated_reasons': summary.get('not_evaluated_reasons', {}),
                     'closure_check': summary.get('closure_check'),
                     'timing': {'baseline': timing(old_reports), 'current': timing(reports)},
                     'trial_changes': changes,
                     'transport_preflight': dict(preflight),
                     'picked': summary['picked'], 'failures': summary['failures'], 'stability': dict(stability)})
        for family, entry in summary['preference_data']['by_grasp'].items():
            preferences.append({'asset': asset, 'family': family, **entry})
    totals = {}
    for key, thin in [('non_thin', False), ('thin', True)]:
        group = [row for row in rows if row['thin'] == thin]
        totals[key] = {'assets': len(group), **{field: sum(row[field] for row in group)
                         for field in ('old_first', 'old_expected', 'old_assessed', 'old_not_evaluated',
                                       'new_first', 'new_expected',
                                       'new_assessed', 'not_evaluated')}}
        for field in ('old_failures', 'failures', 'not_evaluated_reasons'):
            counts = Counter()
            for row in group:
                counts.update(row[field])
            totals[key][field] = dict(counts)
        signals = [row['closure_check'] for row in group if row['closure_check'] is not None]
        if signals:
            counts = Counter()
            by_reason = {}
            for signal in signals:
                counts.update(signal['table'])
                for reason, table in signal['by_failure_reason'].items():
                    by_reason.setdefault(reason, Counter()).update(table)
            totals[key]['closure_check'] = {'table': dict(counts),
                'unmeasured': sum(signal['unmeasured'] for signal in signals),
                'no_check': sum(signal['no_check'] for signal in signals),
                'by_failure_reason': {reason: dict(table) for reason, table in by_reason.items()}}
    return {'baseline': str(baseline), 'run': str(run), 'commit': new['commit'], 'rows': rows,
            'totals': totals, 'preference_data': preferences, 'load': load_record(loads),
            'baseline_load': load_record(old_loads),
            'motion_peaks': {key: dict(zip(('p50', 'p99', 'max'), map(float, np.quantile(values, [.5, .99, 1.]))))
                             for key, values in peaks.items()},
            'comparison_limit': 'Inspect conditions.json in both runs: closure handling, candidate geometry and motion reachability can change independently of stability thresholds; count differences do not isolate one policy effect.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='output directory for comparison.json/md and preferences.csv')
    parser.add_argument('--expert-before', type=Path)
    parser.add_argument('--expert-after', type=Path)
    args = parser.parse_args(argv)
    if bool(args.expert_before) != bool(args.expert_after):
        parser.error('--expert-before and --expert-after must be supplied together')
    result = compare(args.baseline, args.run)
    if args.expert_before:
        result['expert'] = expert_comparison(args.expert_before, args.expert_after)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'comparison.json').write_text(dumps(result, indent=2, ensure_ascii=False) + '\n')
    with (args.output / 'preferences.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=['asset', 'family', 'trials', 'success', 'rate'])
        writer.writeheader()
        writer.writerows(result['preference_data'])
    lines = ['# 首次抓稳逐资产对照', '',
             '对照两批 conditions.json 核查协议；两侧均只扣除未执行持物测试的试次；强度仅作信息，分别保留请求数量。', '',
             '按相同 seed 的首次实际状态配对。', '']
    for title, thin in [('非薄物体', False), ('薄物体（#55，仅单列）', True)]:
        lines.extend([f'## {title}', '', '| 资产 | 旧首次 / 已评估 | 新首次 / 已评估 | 请求数 旧/新 | 未评估 旧/新 | 抓起 | 失败原因 |',
                      '| --- | ---: | ---: | ---: | ---: | ---: | --- |'])
        for row in result['rows']:
            if row['thin'] != thin:
                continue
            failures = ', '.join(f'{key}: {value}' for key, value in row['failures'].items())
            lines.append(f'| {row["asset"]} | {row["old_first"]}/{row["old_assessed"]} | '
                         f'{row["new_first"]}/{row["new_assessed"]} | {row["old_expected"]}/{row["new_expected"]} | {row["old_not_evaluated"]}/{row["not_evaluated"]} | {row["picked"]}/{row["new_expected"]} | {failures} |')
        lines.append('')
    lines.extend(['握法偏好数据单列在 `preferences.csv`；它覆盖主尝试及换部位后的尝试，不是验收通过率。', ''])
    (args.output / 'comparison.md').write_text('\n'.join(lines))
    signals = []
    for row in result['rows']:
        if row['closure_check']:
            for disagreement in row['closure_check']['disagreements']:
                signals.append({'asset': row['asset'], **disagreement})
    (args.output / 'closure-disagreements.json').write_text(dumps(signals, indent=2, ensure_ascii=False) + '\n')
    lines = ['# 首次闭合信号与实际结果', '',
             '实际结果只用首次抬升/运输；未完成实际测量单列，不当作不稳。夹实信号不是对运输稳固性的保证。', '',
             '| 资产 | 判夹实且稳 | 判未夹实但稳（误判） | 判夹实但失败（漏判） | 判未夹实且失败 | 未测量 | 未检查 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    columns = ['predicted_held_actual_stable', 'predicted_unheld_actual_stable',
               'predicted_held_actual_failed', 'predicted_unheld_actual_failed']
    for row in result['rows']:
        signal = row['closure_check']
        if signal:
            counts = [signal['table'][key] for key in columns]
            lines.append('| ' + row['asset'] + ' | ' + ' | '.join(map(str, [*counts, signal['unmeasured'], signal['no_check']])) + ' |')
    lines.append('')
    (args.output / 'closure-checks.md').write_text('\n'.join(lines))
    lines = ['# 耗时算术均值（秒）', '',
             '逐次 trial 包括偏好搜索；首次尝试另列。全部 seed（含失败与长尾）等权。两批共享机器负载不同，这些是观测墙钟，不是控制负载的微基准。', '',
             '| 资产 | 旧 CPU | 新 CPU | 旧 wall | 新 wall |',
             '| --- | ---: | ---: | ---: | ---: |']
    for row in result['rows']:
        old, new = row['timing']['baseline'], row['timing']['current']
        values = [old['cpu_s']['mean'], new['cpu_s']['mean'],
                  old['trial_wall_s']['mean'], new['trial_wall_s']['mean']]
        lines.append('| ' + row['asset'] + ' | ' + ' | '.join(f'{value:.2f}' for value in values) + ' |')
    if 'expert' in result:
        lines.extend(['', '## 专家逐 seed 墙钟', '', '| seed | 旧状态 | 新状态 | 旧秒 | 新秒 | 比值 |',
                      '| ---: | --- | --- | ---: | ---: | ---: |'])
        for row in result['expert']['rows']:
            lines.append(f'| {row["seed"]} | {row["old_status"]} | {row["new_status"]} | '
                         f'{row["old_wall_s"]:.2f} | {row["new_wall_s"]:.2f} | {row["wall_ratio"]:.2f} |')
    (args.output / 'timing.md').write_text('\n'.join(lines) + '\n')
    print(dumps(result['totals'], ensure_ascii=False))


if __name__ == '__main__':
    main()
