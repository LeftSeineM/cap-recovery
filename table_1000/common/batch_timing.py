"""Latest unit, environment-load and worker-start timings, shared across worktrees."""
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, median
import tempfile


def timing_path(root, key):
    return (Path(root) / 'runs/.batch-timing').joinpath(*map(str, key)).with_suffix('.json')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        json.dump(value, stream)
    Path(stream.name).replace(path)


def record(root, output, unit, report):
    """Atomically replace this unit's last measurement, including completed physical failures."""
    fields = ('wall_s', 'cpu_s', 'save_wall_s', 'save_cpu_s', 'status', 'commit')
    value = {name: report[name] for name in fields if name in report}
    value.update(report.get('unit_timing', {}))
    value.update(batch=str(output), written_at=datetime.now(timezone.utc).isoformat())
    write(timing_path(root, unit.timing_key), value)
    provenance = {name: value[name] for name in ('status', 'commit', 'batch', 'written_at') if name in value}
    for field, prefix, cpu in (('asset_load_s', 'loads', 'asset_load_cpu_s'),
                               ('scene_load_s', 'loads', 'scene_load_cpu_s'),
                               ('app_start_s', 'startup', 'app_cpu_s')):
        if field in report:
            measurement = provenance | dict(wall_s=report[field])
            if cpu in report:
                measurement['cpu_s'] = report[cpu]
            write(timing_path(root, (prefix, *unit.load_key)), measurement)


class Timings:
    """Exact unit → other seeds' mean → requested-batch median → declared estimate.

    Per-job estimates add one app start plus the load of each distinct load_key.
    Unit estimates add any separate save_wall_s; app/load work is reported separately.
    """
    def __init__(self, root, units):
        keys = {u.timing_key[:-1] for u in units}
        families = {key: [json.loads(p.read_text()) for p in timing_path(root, (*key, '0')).parent.glob('*.json')]
                    for key in keys}
        records = {u.key: json.loads(timing_path(root, u.timing_key).read_text())
                   for u in units if timing_path(root, u.timing_key).exists()}
        cost = lambda row: row['wall_s'] + row.get('save_wall_s', 0.)
        known = [cost(row) for row in records.values() if row['wall_s'] is not None]
        self.costs = {}
        for unit in units:
            row = records.get(unit.key)
            peers = [cost(r) for r in families[unit.timing_key[:-1]] if r['wall_s'] is not None]
            self.costs[unit.key] = cost(row) if row and row['wall_s'] is not None else mean(peers) if peers else median(known) if known else unit.estimate_s
        self.coverage = len(records), len(units)
        self.has_history = any(families.values())
        self._loads = {u.key: u.load_key for u in units}
        self.loads = {}
        starts = []
        for key in dict.fromkeys(self._loads.values()):
            path = timing_path(root, ('loads', *key))
            self.loads[key] = json.loads(path.read_text())['wall_s'] if path.exists() else 0.
            path = timing_path(root, ('startup', *key))
            if path.exists():
                starts.append(json.loads(path.read_text())['wall_s'])
        self.startup = median(starts) if starts else 60.
        self.labels = {u.key: '/'.join(map(str, u.load_key)) for u in units}

    def load_key(self, unit):
        return self._loads[unit]

    def wall(self, units):
        return self.startup + sum(self.costs[u] for u in units) + sum(self.loads[k] for k in {self.load_key(u) for u in units})

    def announce(self, groups, labels=None):
        print(f'BATCH {len(groups)} jobs; estimated longest {max(map(self.wall, groups), default=0.):.0f} s; '
              f'history {self.coverage[0]}/{self.coverage[1]} units', flush=True)
        labels = labels or self.labels
        for label in dict.fromkeys(labels[group[0]] for group in groups):
            selected = [group for group in groups if labels[group[0]] == label]
            print(f'BATCH {label}: {len(selected)} jobs; estimated longest {max(map(self.wall, selected)):.0f} s', flush=True)
