"""Accounting must survive nesting, lazy consumers and failed searches."""

import pytest

from table_1000.common import timing
from table_1000.common.timing import WallProfile, measure, timed


@pytest.fixture(autouse=True)
def no_hardware_counter(monkeypatch):
    def denied():
        raise PermissionError(13, 'Permission denied')
    monkeypatch.setattr(timing, 'Instructions', denied)


def test_nested_exception_and_disabled(monkeypatch):
    clock = iter([0., 0., 1., 3., 5., 10., 12., 12.])
    monkeypatch.setattr(timing, 'perf_counter', lambda: next(clock))
    cpu = iter([0., 0., 1., 2., 3., 4., 5., 5.])
    monkeypatch.setattr(timing, 'thread_time', lambda: next(cpu))

    def child():
        with measure('child'):
            raise ValueError('unreachable')

    @timed('parent')
    def parent():
        with pytest.raises(ValueError):
            child()

    profile = WallProfile()
    with profile.record():
        parent()
        before = profile.snapshot()
        with profile.span('child'):
            pass
    assert before['parent'] == {'calls': 1, 'wall_s': 5., 'self_s': 3.,
                                'thread_cpu_s': 3., 'self_thread_cpu_s': 2.}
    assert before['child'] == {'calls': 1, 'wall_s': 2., 'self_s': 2.,
                               'thread_cpu_s': 1., 'self_thread_cpu_s': 1.}
    assert profile.since(before)['child'] == before['child']
    assert profile.counters['instructions']['errno'] == 13
    # The recording context is reset, and disabled profiling does not read the clock.
    parent()
    with profile.record(False):
        parent()
    assert profile.snapshot()['parent'] == before['parent']


def test_lazy_candidates_exclude_consumer(monkeypatch):
    clock = iter([0., 0., 1., 20., 22., 40., 40., 50.])
    monkeypatch.setattr(timing, 'perf_counter', lambda: next(clock))
    cpu = iter([0., 0., 1., 10., 12., 30., 30., 40.])
    monkeypatch.setattr(timing, 'thread_time', lambda: next(cpu))

    @timed('candidates')
    def candidates():
        yield 1
        yield 2

    profile = WallProfile()
    with profile.record():
        assert list(candidates()) == [1, 2]
    assert profile.snapshot()['candidates'] == {'calls': 3, 'wall_s': 3., 'self_s': 3.,
                                               'thread_cpu_s': 3., 'self_thread_cpu_s': 3.}


def test_instructions_and_process_span_subtract_from_thread_parent(monkeypatch):
    class Counter:
        closed = False

        def __init__(self):
            self.values = iter([0, 10, 20, 50, 80, 100])

        def read(self):
            return next(self.values)

        def close(self):
            self.closed = True

    counter = Counter()
    monkeypatch.setattr(timing, 'Instructions', lambda: counter)
    samples = iter([0., 1., 2., 5., 8., 10.])
    monkeypatch.setattr(timing, 'perf_counter', lambda: next(samples))
    cpu = iter([0., 1., 2., 3., 4., 5.])
    monkeypatch.setattr(timing, 'thread_time', lambda: next(cpu))
    process = iter([10., 16.])
    monkeypatch.setattr(timing, 'process_time', lambda: next(process))
    profile = WallProfile()
    with profile.record():
        with measure('parent'):
            with measure('physics_step'):
                pass
    rows = profile.snapshot()
    parent, physics = rows['parent'], rows['physics_step']
    assert parent['self_s'] == 4.
    assert parent['self_thread_cpu_s'] == 2.
    assert parent['instructions'] == 70
    assert parent['self_instructions'] == 40
    assert physics == {'calls': 1, 'wall_s': 3., 'self_s': 3., 'process_cpu_s': 6.}
    assert profile.elapsed == {'wall_s': 10., 'thread_cpu_s': 5., 'instructions': 100,
                               'process_spans_thread_cpu_s': 1., 'process_spans_instructions': 30.}
    assert counter.closed


def test_disabled_profile_never_initializes_or_reads_counters(monkeypatch):
    def unexpected():
        pytest.fail('disabled profiling read a counter')
    for name in ('Instructions', 'perf_counter', 'thread_time', 'process_time'):
        monkeypatch.setattr(timing, name, unexpected)

    @timed('work')
    def work():
        with measure('physics_step'):
            return 42

    profile = WallProfile()
    with profile.record(False):
        assert work() == 42
    assert profile.snapshot() == {}


@pytest.mark.parametrize('fail_after', [0, 1])
def test_failed_counter_read_closes_fd_and_restores_context(monkeypatch, fail_after):
    class Counter:
        closed = False

        def read(self):
            nonlocal fail_after
            if fail_after == 0:
                raise OSError('counter cannot be scheduled')
            fail_after -= 1
            return 0

        def close(self):
            self.closed = True

    counter = Counter()
    monkeypatch.setattr(timing, 'Instructions', lambda: counter)
    with pytest.raises(OSError, match='cannot be scheduled'):
        with WallProfile().record():
            pass
    assert counter.closed
    assert timing._active.get() is None
