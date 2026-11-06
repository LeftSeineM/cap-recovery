"""Manifest consumers used by controller tests with simulated worker outputs."""
import json
from pathlib import Path


def manifest(command):
    return json.loads(Path(command[command.index('table_1000.common.batch_runtime') + 1]).read_text())


def publish(data, unit, report):
    target = Path(data['output']) / unit['result']
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(unit['data'] | report | dict(job=Path(data['work']).name, commit=data['commit'], worker_attempt=data['attempt'])))
    return target


def expert_job(command):
    data = manifest(command)
    return data, Path(data['work']), [unit['data']['seed'] for unit in data['units']]


def expert_result(data, work, seed, report):
    (work / 'episodes').mkdir(parents=True, exist_ok=True)
    (work / 'episodes' / f'{seed}.json').write_text(json.dumps(report))
    unit = next(u for u in data['units'] if u['data']['seed'] == seed)
    publish(data, unit, report)
