"""Batch runs of the scripts: work units packed into processes, seeds, the commit they ran from and the load.

A batch script's work units (seeds, cases, asset backends) are packed into ``--jobs`` processes
(:func:`plan`) that run at once: spread evenly over the local ``--gpus``, or each submitted through
``--launcher``, a job scheduler's command prefix such as ``srun --gres=gpu:1`` that picks the GPU.
Units without a result (their process stopped first) run again (``--retries``); see
:func:`run_batch`. Seeds are written as in ``--seeds``: ``a:b`` (``a`` to ``b - 1``) and single
seeds, e.g. ``0:100`` or ``20 64``.

Git batches use one private detached worktree for all rounds, with the current Python environment
and shared assets/data/runs/outputs. Uncommitted files are frozen as a binary patch, including
untracked files not ignored by Git. Non-Git sources run in place. The returned :class:`BatchResult`
records the submitted commit and can save the patch alongside a batch's summary.
"""

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

# Processes at once through a launcher, unless --jobs says otherwise.
LAUNCHED_JOBS = 16
LOAD_PERIOD = 30.   # s between samples of the load (see Load)
SHARED = ('assets', 'data', 'runs', 'outputs')


def _run_processes(jobs, count, run):
    """Wait for jobs; on interruption stop running process groups before joining their threads."""
    from concurrent.futures import ThreadPoolExecutor
    processes, lock, stopping = [], threading.Lock(), threading.Event()

    def start(argv, **kwargs):
        with lock:
            if stopping.is_set():
                return 143
            process = subprocess.Popen(argv, start_new_session=True, **kwargs)
            processes.append(process)
        return process.wait()

    previous = signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    pool = ThreadPoolExecutor(count)
    try:
        return list(pool.map(lambda job: run(job, start), jobs))
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        with lock:
            stopping.set()
            running = [process for process in processes if process.poll() is None]
            for process in running:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:  # It finished between poll and killpg.
                    pass
        for process in running:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    finally:
        pool.shutdown(wait=True)
        signal.signal(signal.SIGTERM, previous)


def add_arguments(parser):
    """``--jobs``, ``--gpus``, ``--launcher`` and ``--retries`` of a batch script, and its workers' ``--gpu``."""
    parser.add_argument('--jobs', type=int, help=f'processes at once (default: one per --gpus; {LAUNCHED_JOBS} with --launcher)')
    parser.add_argument('--gpus', type=int, nargs='+', help='local GPUs the processes are spread over (default 0); '
                        'not with --launcher')
    parser.add_argument('--launcher', help='command prefix of a job scheduler that runs a process on a free GPU it sets '
                        'CUDA_VISIBLE_DEVICES to; instead of --gpus')
    parser.add_argument('--retries', type=int, default=1, help='reruns of work without a result, e.g. its process stopped first')
    parser.add_argument('--gpu', type=int, help=argparse.SUPPRESS)


def job_count(parser, args):
    """The processes at once of :func:`add_arguments`' options; sets the default ``args.gpus``."""
    if args.launcher and args.gpus:
        parser.error('--gpus runs locally; with --launcher the scheduler picks the GPUs')
    if args.jobs is not None and args.jobs < 1:
        parser.error('--jobs must be positive')
    args.gpus = args.gpus or [0]
    return args.jobs or (LAUNCHED_JOBS if args.launcher else len(args.gpus))


def parse_seeds(values):
    """Seeds of ``--seeds`` values, sorted without duplicates."""
    result = set()
    for value in values:
        if ':' in value:
            start, stop = map(int, value.split(':'))
            result.update(range(start, stop))
        else:
            result.add(int(value))
    return sorted(result)


def seed_range(seeds):
    """The shortest ``--seeds`` text of ``seeds``: ``0:100``, ``20 64``, ``0:50 60:70``."""
    seeds, parts, index = sorted(seeds), [], 0
    while index < len(seeds):
        stop = index
        while stop + 1 < len(seeds) and seeds[stop + 1] == seeds[stop] + 1:
            stop += 1
        parts.append(str(seeds[index]) if stop == index else f'{seeds[index]}:{seeds[stop] + 1}')
        index = stop + 1
    return ' '.join(parts)


def seeds_record(seeds):
    """``{'range', 'count'}`` of a batch's seeds, as summaries report them."""
    return {'range': seed_range(seeds), 'count': len(seeds)}


def split(values, count):
    """``values`` in ``count`` contiguous parts of nearly equal size (no empty parts)."""
    count = max(1, min(count, len(values)))
    size, extra = divmod(len(values), count)
    parts, start = [], 0
    for index in range(count):
        stop = start + size + (index < extra)
        parts.append(values[start:stop])
        start = stop
    return parts


def run_jobs(jobs, gpus):
    """Run ``(command, log, label)`` jobs, one at a time per entry of ``gpus`` (a GPU listed twice runs two),
    each command given ``--gpu``; returns the failed labels."""
    import queue
    free = queue.Queue()
    for gpu in gpus:
        free.put(gpu)

    def run(job, start):
        command, log, label = job
        gpu = free.get()
        try:
            print('START', label, 'GPU', gpu, flush=True)
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('w') as stream:
                code = start([*command, '--gpu', str(gpu)], stdout=stream, stderr=subprocess.STDOUT)
            print('DONE' if code == 0 else 'FAILED', label, log, flush=True)
            return code
        finally:
            free.put(gpu)

    codes = _run_processes(jobs, len(gpus), run)
    return [job[2] for job, code in zip(jobs, codes) if code]


def launch(jobs, launcher, limit):
    """Run ``(command, log, label)`` jobs, ``limit`` at a time, each as ``<launcher> bash -c '<command> > <log> 2>&1'``;
    returns the failed labels. The launcher waits for a free GPU, sets ``CUDA_VISIBLE_DEVICES`` to it and returns
    the command's status; the command gets no ``--gpu`` and reads its GPU with :func:`worker_gpu`."""
    def run(job, start):
        command, log, label = job
        log.parent.mkdir(parents=True, exist_ok=True)
        line = f'{launcher} bash -c {shlex.quote(f"{shlex.join(map(str, command))} > {shlex.quote(str(log))} 2>&1")}'
        print('SUBMIT', label, flush=True)
        code = start(line, shell=True)
        print('DONE' if code == 0 else 'FAILED', label, log, flush=True)
        return code

    # Launcher clients cancel their remote jobs when stopped (e.g. srun, gq submit --wait).
    codes = _run_processes(jobs, min(limit, len(jobs)), run)
    return [job[2] for job, code in zip(jobs, codes) if code]


def pack(items, cost, count):
    """``items`` in at most ``count`` groups of nearly equal total ``cost(item)``: the costliest first, each
    onto the lightest group. No group is empty."""
    groups = [[] for _ in range(max(1, min(count, len(items))))]
    totals = [0.] * len(groups)
    for item in sorted(items, key=cost, reverse=True):
        index = totals.index(min(totals))
        groups[index].append(item)
        totals[index] += cost(item)
    return groups


def plan(items, key, cost, count):
    """Groups of ``items`` for ``count`` processes: items of one ``key(item)`` share a process kind (e.g. one
    robot and physics backend) and are packed by :func:`pack`, or without ``cost`` (all alike) split into
    contiguous runs; each key gets processes in proportion to its total cost, at least one and at most one
    per item. With more keys than ``count``, there are more groups than ``count``: :func:`run_batch` runs
    ``count`` of them at a time."""
    weight = cost or (lambda item: 1)
    by_key = {}
    for item in items:
        by_key.setdefault(key(item), []).append(item)
    totals = {k: sum(map(weight, values)) for k, values in by_key.items()}
    counts = {k: max(1, round(count * total / sum(totals.values()))) for k, total in totals.items()}
    while sum(counts.values()) > count and any(c > 1 for c in counts.values()):
        k = min((k for k in counts if counts[k] > 1), key=lambda k: totals[k] / counts[k])
        counts[k] -= 1
    counts = {k: min(c, len(by_key[k])) for k, c in counts.items()}
    while sum(counts.values()) < count:
        open_keys = [k for k in counts if counts[k] < len(by_key[k])]
        if not open_keys:
            break
        counts[max(open_keys, key=lambda k: totals[k] / counts[k])] += 1
    return [group for k, values in by_key.items()
            for group in (pack(values, cost, counts[k]) if cost else split(values, counts[k]))]


class BatchResult(list):
    """Missing items, the submitted ``commit`` and its optional binary ``patch`` (bytes)."""

    def __init__(self, missing, checkout, patch=None):
        super().__init__(missing)
        self.commit = checkout
        self.patch = patch

    def record(self, directory, *, patch_name='dirty.patch'):
        """Save the patch in a batch directory; return ``commit`` and optional relative ``dirty_patch``
        fields for its summary. Applying the patch to HEAD restores the submitted working files."""
        record = {'commit': self.commit}
        if self.patch is not None:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / patch_name).write_bytes(self.patch)
            record['dirty_patch'] = patch_name
        return record


def _dirty_patch(root, revision, directory):
    """Diff HEAD against working files, including untracked files, using a private Git index.

    Git handles binary files, modes, deletions and symlinks. The source index, working files,
    branches and stash stack are untouched; ignored files are excluded unless already tracked.
    """
    with tempfile.TemporaryDirectory(prefix='dirty-index-', dir=directory) as temporary:
        env = {**os.environ, 'GIT_INDEX_FILE': str(Path(temporary) / 'index')}
        git = ['git', '-C', str(root)]
        subprocess.run([*git, 'read-tree', revision], env=env, check=True)
        subprocess.run([*git, 'add', '--all', '--', '.'], env=env, check=True)
        return subprocess.run([*git, 'diff', '--cached', '--binary', '--no-ext-diff', '--no-textconv', revision, '--'],
                              env=env, check=True, capture_output=True).stdout


@contextmanager
def _checkout(root):
    """Yield a unique frozen checkout, revision and patch; non-Git sources yield their own path."""
    checkout = commit(root)
    path = patch = None
    previous = signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        if checkout:
            directory = root / '.local' / 'batch-checkouts'
            directory.mkdir(parents=True, exist_ok=True)
            revision = checkout.removesuffix('+dirty')
            if checkout.endswith('+dirty'):
                patch = _dirty_patch(root, revision, directory)
            path = Path(tempfile.mkdtemp(prefix=f'{checkout}-', dir=directory))
            print('CHECKOUT', checkout, path, flush=True)
            subprocess.run(['git', '-C', str(root), 'worktree', 'add', '--detach', str(path), revision], check=True)
            if patch:
                subprocess.run(['git', '-C', str(path), 'apply', '--binary'], input=patch, check=True)
            for name in SHARED:
                (root / name).mkdir(exist_ok=True)
                (path / name).symlink_to(root / name, target_is_directory=True)
        yield path or root, checkout, patch
    finally:
        unwinding = sys.exc_info()[0] is not None
        try:
            if path is not None:
                removed = subprocess.run(['git', '-C', str(root), 'worktree', 'remove', '--force', str(path)],
                                         capture_output=True)
                if removed.returncode:
                    # A failed/interrupted add may leave only mkdtemp's unregistered directory.
                    if (path / '.git').exists():
                        removed.check_returncode()
                    if path.exists():
                        shutil.rmtree(path)
        except BaseException as error:
            if not unwinding:
                raise
            print('CHECKOUT cleanup failed:', error, file=sys.stderr, flush=True)
        finally:
            signal.signal(signal.SIGTERM, previous)


def _pinned_command(argv, root, checkout):
    """Redirect source paths, retaining the interpreter and shared paths; override editable imports."""
    def redirect(value):
        value = str(value)
        path = Path(value)
        if not path.is_absolute():
            # Command arguments like seeds and labels are not paths. Existing relative file paths
            # (e.g. scripts/scenes/run_expert.py) are interpreted from the submitting directory.
            if not path.exists() or ('/' not in value and not path.is_file()):
                return value
            path = path.absolute()
        if not path.is_relative_to(root):
            return value
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in (*SHARED, '.venv', '.local'):
            return str(path)
        return str(checkout / relative)

    # Do not resolve/rewrite argv[0]: it is the original environment's Python, often in .venv/.
    pythonpath = os.pathsep.join(filter(None, (str(checkout), os.environ.get('PYTHONPATH'))))
    return ['env', '-C', str(checkout), f'PYTHONPATH={pythonpath}', str(argv[0]), *map(redirect, argv[1:])]


def run_batch(items, *, command, missing, count, gpus, launcher=None, retries=1, key=lambda item: None, cost=None,
              root=None):
    """Run ``items`` (work units) as processes (:func:`plan`), at most ``count`` at a time, on ``gpus``
    (spread evenly, see :func:`run_jobs`) or each through ``launcher`` (see :func:`launch`).
    ``command(group, index, attempt)`` gives a process's ``(argv, log, label)``. Items still ``missing(item)``
    after a round (their process died, was killed or failed to start before writing them, or a result that
    ``missing`` counts as not one, such as an error that may be the process's) run again, up to ``retries``
    more rounds, in as many processes as before at most. Returns a :class:`BatchResult` of missing items.

    A Git ``root`` (default: this package's repository) is checked out once before calling ``command``;
    all rounds use its scripts, source paths and imports. Its ignored assets/data/runs/outputs remain
    shared with ``root``. Dirty roots include working files and nonignored untracked files; non-Git
    roots run in place. Reports use the result's ``record(directory)`` to save its patch and commit
    rather than inspecting the mutable original checkout after the workers finish."""
    pending = list(items)
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    with _checkout(root) as (checkout, revision, patch):
        for attempt in range(retries + 1):
            if not pending:
                break
            if attempt:
                print(f'RETRY {len(pending)} without a result (attempt {attempt})', flush=True)
            jobs = [command(group, index, attempt) for index, group in enumerate(plan(pending, key, cost, count))]
            if checkout != root:
                jobs = [(_pinned_command(argv, root, checkout), log, label) for argv, log, label in jobs]
            if launcher:
                launch(jobs, launcher, count)
            else:
                run_jobs(jobs, [gpus[index % len(gpus)] for index in range(min(count, len(jobs)))])
            pending = [item for item in pending if missing(item)]
    return BatchResult(pending, revision, patch)


def process_name(stem, attempt):
    """Name of a process (its log, its work directory) in round ``attempt`` of :func:`run_batch`."""
    return stem + (f'-retry{attempt}' if attempt else '')


def worker_gpu(gpu):
    """The physical GPU of a worker process: ``gpu`` when given (``--gpu``), else the one a launcher made visible."""
    if gpu is not None:
        return gpu
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    if not visible.isdigit():
        raise SystemExit(f'CUDA_VISIBLE_DEVICES={visible}: a GPU index is needed (not a UUID or MIG id)')
    return int(visible)


def run_name(checkout, label=None):
    """A development run's name: ``<date>-<time>-<commit>[-<label>]``, ``nocommit`` outside a checkout."""
    parts = [time.strftime('%Y%m%d-%H%M%S'), (checkout or 'nocommit').replace('+', '-')]
    return '-'.join(parts + ([label] if label else []))


def commit(root):
    """The checked-out commit of ``root``, ``+dirty`` with uncommitted changes; None unless ``root``
    is a checkout's top (a copy of the sources inside another checkout is not that checkout)."""
    try:
        top = subprocess.run(['git', '-C', str(root), 'rev-parse', '--show-toplevel'],
                             capture_output=True, text=True, check=True).stdout.strip()
        if os.path.realpath(top) != os.path.realpath(root):
            return None
        head = subprocess.run(['git', '-C', str(root), 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', '--no-optional-locks', '-C', str(root), 'status', '--porcelain',
                                '--untracked-files=normal'],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return head + ('+dirty' if dirty else '')


class Load:
    """The machine's load while a block runs: ``host``, ``cores`` (the machine's, as the load average is)
    and ``samples``, one at the start, every ``period`` seconds (from a daemon thread) and at the
    end: ``{'t', 'load_1m', 'cpu_s'}``, seconds since the start, the 1-minute load average and the
    CPU seconds this process used since the sample before. When the machine is busy the process
    waits for cores: its CPU seconds fall behind the wall seconds."""

    def __init__(self, period=LOAD_PERIOD):
        self.period, self.host, self.cores = period, socket.gethostname(), os.cpu_count()

    def __enter__(self):
        self.samples, self.started, self.cpu = [], time.perf_counter(), time.process_time()
        self.sample()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def run(self):
        while not self.stop.wait(self.period):
            self.sample()

    def sample(self):
        cpu = time.process_time()
        self.samples.append({'t': time.perf_counter() - self.started, 'load_1m': os.getloadavg()[0],
                             'cpu_s': cpu - self.cpu})
        self.cpu = cpu

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.sample()

    def record(self):
        """``{'host', 'cores', 'samples'}`` for a report."""
        return {'host': self.host, 'cores': self.cores, 'samples': self.samples}


def load_record(records, quantiles=(.1, .5, .9, 1.)):
    """Over the ``load`` of reports (:meth:`Load.record`) and their ``wall_s``: quantiles of the load
    per core over every sample and of each report's CPU seconds per wall second, and the hosts."""
    import numpy as np
    records = [(report['load'], report['wall_s']) for report in records if report.get('load')]
    if not records:
        return None
    per_core = [sample['load_1m'] / load['cores'] for load, _ in records for sample in load['samples']]
    cpu_per_wall = [sum(sample['cpu_s'] for sample in load['samples']) / wall for load, wall in records if wall > 0]
    named = lambda values: {f'p{round(q * 100)}': float(np.quantile(values, q)) for q in quantiles} if values else None
    return {'hosts': sorted({load['host'] for load, _ in records}), 'load_per_core': named(per_core),
            'cpu_per_wall': named(cpu_per_wall)}
