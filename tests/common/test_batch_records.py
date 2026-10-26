"""What a batch records of where it ran: the commit, and the machine's load."""

import shutil
import subprocess
import time

import pytest

from table_1000.common import batch


def git(*args, cwd):
    subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True)


@pytest.mark.skipif(shutil.which('git') is None, reason='needs git')
def test_commit_only_of_a_checkouts_top(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    git('init', '-q', cwd=repo)
    (repo / 'a.py').write_text('a = 1\n')
    git('add', 'a.py', cwd=repo)
    git('-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'a', cwd=repo)
    head = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert batch.commit(repo) == head
    (repo / 'new.py').write_text('new = 1\n')
    index = (repo / '.git/index').read_bytes()
    assert batch.commit(repo) == head + '+dirty'
    assert (repo / '.git/index').read_bytes() == index
    (repo / 'new.py').unlink()
    (repo / 'a.py').write_text('a = 2\n')
    assert batch.commit(repo) == head + '+dirty'
    # A copy of the sources inside the checkout (as under outputs/) is not that commit.
    copy = repo / 'outputs' / 'snapshot'
    copy.mkdir(parents=True)
    shutil.copy(repo / 'a.py', copy)
    assert batch.commit(copy) is None
    assert batch.commit(tmp_path / 'elsewhere') is None


def test_load_samples_at_start_period_and_end():
    with batch.Load(period=.05) as load:
        end = time.process_time() + .2
        while time.process_time() < end:
            pass
    record = load.record()
    assert record['cores'] >= 1 and record['host']
    samples = record['samples']
    assert len(samples) >= 4 and samples[0]['t'] == pytest.approx(0., abs=.01)
    assert all(a['t'] <= b['t'] for a, b in zip(samples, samples[1:]))
    assert sum(sample['cpu_s'] for sample in samples) == pytest.approx(.2, abs=.1)


def test_load_record_quantiles():
    reports = [{'wall_s': 10., 'load': {'host': 'a', 'cores': 4, 'samples': [{'t': 0., 'load_1m': 2., 'cpu_s': 0.},
                                                                             {'t': 10., 'load_1m': 4., 'cpu_s': 20.}]}},
               {'wall_s': 10., 'load': {'host': 'b', 'cores': 2, 'samples': [{'t': 0., 'load_1m': 2., 'cpu_s': 5.}]}},
               {'wall_s': 1.}]
    record = batch.load_record(reports)
    assert record['hosts'] == ['a', 'b']
    assert record['load_per_core']['p100'] == 1. and record['load_per_core']['p50'] == 1.
    assert record['cpu_per_wall']['p100'] == 2. and record['cpu_per_wall']['p10'] == pytest.approx(.65)
    assert batch.load_record([{'wall_s': 1.}]) is None
