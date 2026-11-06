"""One controller and worker loop for batch programs; units own their simulation state."""
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
import importlib
import json
import os
from pathlib import Path
import sys
import time
import tomllib
import traceback
from typing import Iterable, Hashable, Protocol

from . import batch
from .batch_timing import Timings, write as _write, record as _record


@dataclass(frozen=True)
class Unit:
    """Stable work identity, inputs, result path and timing/load identities.

    result is batch-relative, or absolute when an existing domain owns its result directory.
    Lower priority runs first inside each packed job; it does not create a partition.
    """
    key: tuple
    data: dict
    result: str
    timing_key: tuple
    load_key: tuple
    estimate_s: float = 300.
    priority: int = 0


@dataclass
class Context:
    """Worker-local state; the program owns loading, resetting and ExitStack cleanup."""
    root: Path
    output: Path
    options: dict
    gpu: int
    work: Path
    stack: ExitStack
    loaded: object = None
    previous: object = None


@dataclass
class UnitResult:
    """Domain report and optional whole-unit timing, separate from episode metrics."""
    report: dict
    value: object = None
    timing: dict | None = None


class Program(Protocol):
    def units(self, options: dict) -> Iterable[Unit]: ...
    def pack_key(self, unit: Unit) -> Hashable: ...
    def run_unit(self, context: Context, unit: Unit) -> UnitResult: ...


@dataclass(frozen=True)
class WorkerCommand:
    """External worker, e.g. a main checkout. argv accepts {manifest}, {output}; local workers also receive --gpu."""
    python: str
    cwd: str
    argv: tuple[str, ...]


@dataclass
class BatchConfig:
    max_jobs: int = 100
    auto: list = field(default_factory=list)
    launcher: str | None = None

    def jobs(self, requested, total, units):
        """Auto: floor(total/600) >=20 gives at most 100 jobs; otherwise 1–20
        jobs from floor(total/300). Total includes app startup and loads.
        Configuration overrides tiers and the final cap.
        """
        if requested != 'auto':
            return min(self.max_jobs, units, max(1, requested))
        count = 1
        for tier in self.auto:
            count = int(total // tier['job_seconds'])
            if count >= tier.get('min_jobs', 0):
                count = min(count, tier.get('max_jobs', self.max_jobs))
                break
        return min(self.max_jobs, units, max(1, count))


def load_config(overrides=None, path=None):
    root = Path(__file__).resolve().parents[2]
    with (root / 'configs/batch.toml').open('rb') as stream:
        values = tomllib.load(stream)
    machine = Path(path or Path.home() / '.config/table-1000/batch.toml')
    if machine.exists():
        with machine.open('rb') as stream:
            values.update(tomllib.load(stream))
    values.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return BatchConfig(**values)


def _json(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    return value


def _tuple(value):
    return tuple(map(_tuple, value)) if isinstance(value, list) else value


def _unit(data):
    return Unit(**(data | {k: _tuple(data[k]) for k in ('key', 'timing_key', 'load_key')}))


def _program(name):
    module, attribute = name.split(':')
    return getattr(importlib.import_module(module), attribute)


def _pending(output, unit):
    path = output / unit.result
    if not path.exists():
        return True
    report = json.loads(path.read_text())
    return report['status'].lower() == 'error'


def run(program, options, *, root, output, timing=None, worker_command=None, source=None):
    """Pack once per round, submit every launcher job, and preserve local GPU concurrency."""
    root, output = Path(root).resolve(), Path(output).resolve()
    options = _json(dict(options))
    explicit_jobs = options.get('jobs', 'auto')
    config = load_config({'launcher': options.get('launcher'),
                          'max_jobs': explicit_jobs if explicit_jobs != 'auto' else None})
    implementation = _program(program)
    units = list(implementation.units(options))
    by_key = {u.key: u for u in units}
    timing = timing or Timings(root, units)
    count = config.jobs(options.get('jobs', 'auto'), timing.wall(by_key), len(units))
    key = lambda k: implementation.pack_key(by_key[k])
    costs = set(timing.costs.values())
    cost = timing.costs.__getitem__ if len(costs) > 1 else None
    history = timing if timing.has_history else None
    groups = batch.plan(list(by_key), key, cost, count, timing=history)
    for group in groups:
        group.sort(key=lambda k: by_key[k].priority)
    timing.announce(groups, {k: str(key(k)) for k in by_key})
    document = dict(program=program, root=str(root), output=str(output), options=options,
        units=[asdict(u) for u in units], jobs=len(groups), requested_jobs=count,
        history=list(timing.coverage), estimated_longest_s=max(map(timing.wall, groups), default=0.),
        groups=[dict(units=g, estimated_wall_s=timing.wall(g)) for g in groups], rounds=[])
    # A dry run reserves no name and writes no conditions/results; a subsequent run may reuse it.
    _write(output / 'plan.json', _json(document))
    if options.get('dry_run'):
        return batch.BatchResult([], batch.commit(root))
    output.mkdir(parents=True, exist_ok=True)
    pending = [k for k in by_key if _pending(output, by_key[k])]
    with batch._checkout(root, source) as (checkout, revision, patch):
        for attempt in range(options.get('retries', 1) + 1):
            if not pending:
                break
            current = batch.plan(pending, key, cost, count, timing=history)
            for group in current:
                group.sort(key=lambda k: by_key[k].priority)
            jobs = []
            for index, group in enumerate(current):
                work = output / 'workers' / f'{index}-attempt{attempt}'
                manifest = work / 'units.json'
                _write(manifest, _json(dict(program=program, options=options, output=str(output),
                    root=str(checkout), work=str(work), commit=revision,
                    attempt=attempt, units=[asdict(by_key[k]) for k in group])))
                argv = [sys.executable, '-m', 'table_1000.common.batch_runtime', str(manifest)]
                if worker_command is not None:
                    argv = ['env', '-C', worker_command.cwd, f'PYTHONPATH={worker_command.cwd}', worker_command.python,
                            *(a.format(manifest=str(manifest), output=str(output)) for a in worker_command.argv)]
                else:
                    argv = batch._pinned_command(argv, root, checkout)
                jobs.append((argv, work / 'run.log', f'{program} {index} attempt {attempt}'))
            started = time.perf_counter()
            if config.launcher:
                batch.launch(jobs, config.launcher)
            else:
                batch.run_jobs(jobs, options.get('gpus') or [0])
            # External workers own their execution; publish their completed reports through the same index.
            if worker_command is not None:
                for k in pending:
                    unit = by_key[k]
                    if (output / unit.result).exists():
                        _record(root, output, unit, json.loads((output / unit.result).read_text()))
            pending = [k for k in pending if _pending(output, by_key[k])]
            document['rounds'].append(dict(attempt=attempt, jobs=len(jobs), missing=pending,
                                            wall_s=time.perf_counter() - started))
            document.update(commit=revision, missing=pending)
            _write(output / 'plan.json', _json(document))
        result = batch.BatchResult(pending, revision, patch)
        document.update(result.record(output), finished=time.time())
        _write(output / 'plan.json', _json(document))
    return result


def work(manifest, *, gpu=None):
    data = json.loads(Path(manifest).read_text())
    from .serialization import dumps
    with ExitStack() as stack:
        context = Context(Path(data['root']), Path(data['output']), data['options'],
            batch.worker_gpu(gpu), Path(data['work']), stack)
        program = _program(data['program'])
        for index, raw in enumerate(data['units']):
            unit = _unit(raw)
            if not _pending(context.output, unit):
                continue
            wall, cpu, started = time.perf_counter(), time.process_time(), time.time()
            try:
                result = program.run_unit(context, unit)
                report = result.report
                if result.timing is not None:
                    report['unit_timing'] = result.timing
                context.previous = result.value
            except Exception as error:
                report = dict(unit.data, status='error', success=False, failure={'reason': f'{type(error).__name__}: {error}'}, error=traceback.format_exc())
                (context.work / f'error-{index}.txt').write_text(traceback.format_exc())
            report = dict(unit.data) | report
            report.setdefault('wall_s', time.perf_counter() - wall)
            report.setdefault('cpu_s', time.process_time() - cpu)
            report.update(commit=data['commit'], worker_attempt=data['attempt'], worker_pid=os.getpid(), worker_unit_index=index,
                          unit_started=started, unit_finished=time.time(), job=context.work.name)
            path = context.output / unit.result
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.part')
            temporary.write_text(dumps(report, ensure_ascii=False, indent=2) + chr(10))
            temporary.replace(path)
            _record(context.root, context.output, unit, report)
            print('UNIT', unit.key, report['status'], flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    work(args.manifest, gpu=args.gpu)
