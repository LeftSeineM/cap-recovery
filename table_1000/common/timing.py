"""Optional wall/CPU profiling of nested operations, including lazy candidate generation.

``wall_s`` includes measured children; ``self_s`` excludes them and can be summed across
categories. Native-library collision checks inside IK or RRT remain in their parent's time.
Generator calls measure each resume only, never the work a consumer does between yields.

In an expert report (``run_expert.py --profile``): ``wall_profile`` covers the whole run, a
step's field of the same name is cumulative up to it, and ``subtask_profiles`` matches
``subtasks``. Each category has ``calls``, ``wall_s`` and ``self_s``, and the calling thread's
``thread_cpu_s`` / ``self_thread_cpu_s`` by the same rule; where Linux allows, also user-space
``instructions`` / ``self_instructions`` (``profile_counters`` says why not otherwise).
Thread CPU leaves out library worker threads and Isaac's background threads, which only the
process ``cpu_s`` counts. ``physics_step`` (action conversion, physics, recording) is
multithreaded: it has wall time and the whole process's ``process_cpu_s`` over it, not to be added
to other categories' thread CPU. ``profile_totals`` gives the recorded span's totals. Thread CPU
varies far less with load than wall time, so compare categories by it.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from inspect import isgeneratorfunction
import ctypes
import os
import platform
import struct
from time import perf_counter, process_time, thread_time


_active = ContextVar('wall_profile', default=None)


class Instructions:
    """Linux retired user instructions on the calling thread, without inherited workers."""

    def __init__(self):
        syscall = {('Linux', 'x86_64'): 298, ('Linux', 'aarch64'): 241}[
            platform.system(), platform.machine()]
        # perf_event_attr: PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS;
        # exclude kernel/hypervisor, leave enabled, no inheritance or multiplexed group.
        attr = ctypes.create_string_buffer(128)
        struct.pack_into('IIQQQQQ', attr, 0, 0, 128, 1, 0, 0, 0, (1 << 2) | (1 << 5) | (1 << 6))
        libc = ctypes.CDLL(None, use_errno=True)
        fd = libc.syscall(syscall, ctypes.byref(attr), 0, -1, -1, 8)  # CLOEXEC
        if fd < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        self.fd = fd

    def read(self):
        return struct.unpack('Q', os.read(self.fd, 8))[0]

    def close(self):
        os.close(self.fd)


@contextmanager
def measure(name):
    """Measure a block only while a profile is recording."""
    profile = _active.get()
    if profile is None:
        yield
    else:
        with profile.span(name):
            yield


class WallProfile:
    def __init__(self):
        self.totals = {}
        self.stack = []
        self.instructions = None
        self.elapsed = {}
        self.excluded = [0., 0., 0.]
        self.counters = {'cpu': 'calling thread; process for physics_step',
                         'instructions': {'available': False, 'reason': 'profiling disabled'}}

    def _sample(self):
        return [perf_counter(), thread_time(), self.instructions.read() if self.instructions else 0]

    @contextmanager
    def record(self, enabled=True):
        if not enabled:
            yield self
            return
        try:
            self.instructions = Instructions()
        except (OSError, KeyError) as error:
            self.counters['instructions'] = {'available': False, 'reason': str(error),
                                             'errno': getattr(error, 'errno', None)}
        else:
            self.counters['instructions'] = {'available': True, 'scope': 'calling thread, user space'}
        try:
            started = self._sample()
            token = _active.set(self)
            try:
                yield self
            finally:
                _active.reset(token)
                ended = self._sample()
                self.elapsed = {'wall_s': ended[0] - started[0], 'thread_cpu_s': ended[1] - started[1],
                                'process_spans_thread_cpu_s': self.excluded[1]}
                if self.instructions:
                    self.elapsed['instructions'] = ended[2] - started[2]
                    self.elapsed['process_spans_instructions'] = self.excluded[2]
        finally:
            if self.instructions:
                self.instructions.close()
                self.instructions = None

    @contextmanager
    def span(self, name):
        """CPU self time subtracts children even when a child publishes process CPU instead.

        Process CPU is CPU used by all process threads during this interval; it can include
        unrelated background work and is not additive with calling-thread categories.
        """
        process_cpu = name == 'physics_step'
        started = self._sample()
        process_started = process_time() if process_cpu else None
        frame = [0., 0., 0.]
        self.stack.append(frame)
        try:
            yield
        finally:
            ended = self._sample()
            elapsed = [b - a for a, b in zip(started, ended)]
            process_elapsed = process_time() - process_started if process_cpu else None
            self.stack.pop()
            row = self.totals.setdefault(name, {'calls': 0})
            row['calls'] += 1
            pairs = [('wall_s', 'self_s', 0)]
            if not process_cpu:
                pairs.append(('thread_cpu_s', 'self_thread_cpu_s', 1))
                if self.instructions:
                    pairs.append(('instructions', 'self_instructions', 2))
            else:
                row['process_cpu_s'] = row.get('process_cpu_s', 0.) + process_elapsed
                for i, value in enumerate(elapsed):
                    self.excluded[i] += value - frame[i]
            for inclusive, own, index in pairs:
                row[inclusive] = row.get(inclusive, 0) + elapsed[index]
                row[own] = row.get(own, 0) + elapsed[index] - frame[index]
            if self.stack:
                for i, value in enumerate(elapsed):
                    self.stack[-1][i] += value

    def snapshot(self):
        return {name: dict(row) for name, row in self.totals.items()}

    def since(self, before):
        return {name: {key: value - before.get(name, {}).get(key, 0) for key, value in row.items()}
                for name, row in self.totals.items()}


def timed(name):
    """Measure a function (or each resume of a generator) while a profile is recording."""
    def decorate(function):
        if isgeneratorfunction(function):
            @wraps(function)
            def generator(*args, **kwargs):
                source = function(*args, **kwargs)
                try:
                    while True:
                        profile = _active.get()
                        try:
                            if profile is None:
                                value = next(source)
                            else:
                                with profile.span(name):
                                    value = next(source)
                        except StopIteration:
                            return
                        yield value
                finally:
                    source.close()
            return generator

        @wraps(function)
        def measured(*args, **kwargs):
            profile = _active.get()
            if profile is None:
                return function(*args, **kwargs)
            with profile.span(name):
                return function(*args, **kwargs)
        return measured
    return decorate
