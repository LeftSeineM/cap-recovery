"""Exercise actual native pools, including episode entry and exception restoration."""

import json
import time

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from table_1000.experts.generate import Generator
from table_1000.experts.evals.pick_eval import Evaluator
from table_1000.experts.runtime import single_threaded_blas


def blas_pools():
    return [pool for pool in threadpool_info() if pool['user_api'] == 'blas']


def test_episode_records_single_threaded_blas_and_restores(tmp_path):
    generator = object.__new__(Generator)
    generator.output = tmp_path
    (tmp_path / 'episodes').mkdir()

    def episode(seed, attempt, state, start, repeat):
        assert blas_pools() and all(pool['num_threads'] == 1 for pool in blas_pools())
        np.testing.assert_allclose(np.eye(3) @ np.ones(3), np.ones(3))
        return dict(seed=seed, repeat=repeat, status='success'), time.perf_counter()

    generator.episode = episode
    with threadpool_limits(limits=32, user_api='blas'):
        before = threadpool_info()
        for repeat in (0, 1):
            report = generator.run(0, 0, {}, repeat=repeat)
            pools = [pool for pool in report['native_math_threadpools'] if pool['user_api'] == 'blas']
            assert pools and all(pool['num_threads'] == 1 for pool in pools)
            path = tmp_path / 'episodes' / ('0-1.json' if repeat else '0.json')
            assert json.loads(path.read_text())['native_math_threadpools'] == report['native_math_threadpools']
            assert threadpool_info() == before


@pytest.mark.parametrize('entry', ['scene', 'pick'])
def test_episode_entry_limits_blas_before_reset_and_restores_on_error(entry):
    class ResetFailed(Exception):
        pass

    def reset(*args):
        assert blas_pools() and all(pool['num_threads'] == 1 for pool in blas_pools())
        raise ResetFailed

    if entry == 'scene':
        runner = object.__new__(Generator)
        runner.episode = reset
        run = lambda: runner.run(0, 0, {})
    else:
        from types import SimpleNamespace
        runner = object.__new__(Evaluator)
        runner.env = SimpleNamespace(reset=reset)
        runner.scene = SimpleNamespace(placement=lambda **kwargs: kwargs)
        run = lambda: runner.trial('cube1', dict(x=0., y=0., yaw=0., pose='standing'), 0)
    with threadpool_limits(limits=32, user_api='blas'):
        before = threadpool_info()
        with pytest.raises(ResetFailed):
            run()
        assert threadpool_info() == before


def test_blas_policy_preserves_openmp():
    before = [pool for pool in threadpool_info() if pool['user_api'] == 'openmp']
    assert before
    with single_threaded_blas():
        assert [pool for pool in threadpool_info() if pool['user_api'] == 'openmp'] == before
