"""How much expert episodes moved the torso and the base, from their reports.

``uv run python scripts/experts/resource_stats.py data/scenes/000002/x1pro_rm001/0000/isaac-cpu [more batches ...]``

Counts, in each batch's ``episodes/*.json``, the ``torso moved`` notes (moves of the torso lift and
their travel, m) and the ``base relocated`` notes (moves of the base, their travel, m, and turn,
deg) the skills make. Prints per batch the mean per episode over all episodes and over the
succeeded ones, and the counts per subtask.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def episode_stats(report):
    """``({quantity: total}, {subtask: {quantity: total}})`` of one episode report."""
    totals, subtasks = defaultdict(float), defaultdict(lambda: defaultdict(float))
    for event in report['events']:
        if event['note'] == 'torso moved':
            values = {'torso moves': 1, 'torso travel m': event['travel']}
        elif event['note'] == 'base relocated':
            values = {'base moves': 1, 'base travel m': event['travel'], 'base turn deg': np.degrees(event['turn'])}
        else:
            continue
        for key, value in values.items():
            totals[key] += value
            subtasks[event['subtask']][key] += value
    return totals, subtasks


QUANTITIES = ('torso moves', 'torso travel m', 'base moves', 'base travel m', 'base turn deg')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('batches', nargs='+', type=Path, help='directories holding episodes/*.json')
    args = parser.parse_args(argv)
    for batch in args.batches:
        # Each seed's first run (not the other runs of --repeats).
        reports = [json.loads(path.read_text()) for path in sorted((batch / 'episodes').glob('*.json'))
                   if '-' not in path.stem]
        stats = [(report['status'] == 'success', *episode_stats(report)) for report in reports]
        succeeded = sum(ok for ok, _, _ in stats)
        print(f'== {batch}: {len(reports)} episodes, {succeeded} succeeded')
        print(f'{"mean per episode":<48}' + ''.join(f'{q:>16}' for q in QUANTITIES))
        for label, chosen in (('all', stats), ('succeeded', [s for s in stats if s[0]])):
            if chosen:
                print(f'{label:<48}' + ''.join(f'{np.mean([t[q] for _, t, _ in chosen]):>16.2f}' for q in QUANTITIES))
        per_subtask = defaultdict(lambda: defaultdict(float))
        for _, _, subtasks in stats:
            for subtask, values in subtasks.items():
                for key, value in values.items():
                    per_subtask[subtask][key] += value / len(stats)
        for subtask, values in per_subtask.items():
            print(f'  {str(subtask)[:46]:<46}' + ''.join(f'{values[q]:>16.2f}' for q in QUANTITIES))


if __name__ == '__main__':
    main()
