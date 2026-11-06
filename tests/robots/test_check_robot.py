"""check_robot.py without a simulator: units packed into processes, reruns of missing results, the summary."""

import json
from tests.common.batch_worker import manifest, publish

from scripts.robots import check_robot
from table_1000.common import batch
from table_1000.common.batch import pack, plan


def test_pack_balances_cost():
    groups = pack([5, 1, 4, 2, 3], lambda item: item, 2)
    assert sorted(map(sum, groups)) == [7, 8]
    assert pack([1], lambda item: item, 4) == [[1]]
    groups = plan([('a', 9), ('a', 9), ('b', 1)], lambda item: item[0], lambda item: item[1], 2)
    assert sorted(len(group) for group in groups) == [1, 2] and all(len({i[0] for i in g}) == 1 for g in groups)
    # A key never gets more processes than items; the rest go to the others.
    groups = plan([('a', 1000)] + [('b', 10)] * 10, lambda item: item[0], lambda item: item[1], 6)
    assert sorted(len(group) for group in groups) == [1, 2, 2, 2, 2, 2]


def test_units_retries_and_summary(tmp_path, monkeypatch):
    rounds = []

    def run_jobs(jobs, gpus):
        rounds.append((len(jobs), gpus))
        for command, _, label in jobs:
            data = manifest(command)
            for unit in data['units']:
                case, trial = unit['data']['case'], unit['data']['trial']
                attempt = data['attempt']
                if case == 'grasp_and_push' and trial == 2 and attempt == 0:
                    continue
                status = 'FAIL' if case == 'joint_limits' else 'PASS'
                result = {'status': status, 'checks': {'held': status == 'PASS'}, 'wall_s': 1.}
                if case == 'grasp_and_push':
                    result['trials'] = {str(trial): {}}
                publish(data, unit, dict(status=status, result=result))
        return []

    monkeypatch.setattr(batch, 'run_jobs', run_jobs)
    try:
        check_robot.main(['x1pro', '--physics', 'isaac-cpu', '--cases', 'keyframes', 'joint_limits',
                          'grasp_and_push', 'parallel', '--trials', '0', '1', '2', '--gpus', '0', '1',
                          '--jobs', '4', '--output', str(tmp_path)])
    except SystemExit as error:
        assert 'joint_limits' in str(error) and 'grasp_and_push' not in str(error)
    assert rounds[0] == (4, [0, 1])                 # four jobs, with two local workers
    assert len(rounds) == 2                               # one more round for the units without a result
    summary = json.loads((tmp_path / 'x1pro' / 'summary.json').read_text())
    assert summary['backends']['isaac-cpu'] == {'passed': 2, 'cases': 3, 'failed': ['joint_limits'], 'earlier': []}
    assert summary['cases']['grasp_and_push']['isaac-cpu']['retried'] == 1
    assert summary['cases']['joint_limits']['isaac-cpu']['retried'] == 0     # failed, but not rerun
    report = json.loads((tmp_path / 'x1pro' / 'isaac-cpu' / 'report.json').read_text())
    assert list(report['cases']['grasp_and_push']['trials']) == ['0', '1', '2']
    from contextlib import ExitStack
    from types import SimpleNamespace
    from table_1000.common import batch_runtime as runtime
    from table_1000.robots.batch_program import checks
    options = dict(units=[('x1pro', 'isaac-gpu', case, None) for case in ('keyframes', 'parallel')],
                   directories={'x1pro': str(tmp_path / 'paired')}, assets=str(tmp_path / 'assets'),
                   jobs=1, retries=0, gpus=[0])
    observed = []
    def ordered_jobs(jobs, gpus):
        assert len(jobs) == 1
        data = manifest(jobs[0][0])
        assert [unit['data']['case'] for unit in data['units']] == ['parallel', 'keyframes']
        for unit in data['units']:
            publish(data, unit, dict(status='PASS', result=dict(status='PASS')))
        return []
    monkeypatch.setattr(batch, 'run_jobs', ordered_jobs)
    runtime.run('table_1000.robots.batch_program:checks', options, root=tmp_path, output=tmp_path / 'batch')
    checker = SimpleNamespace(env=SimpleNamespace(close=lambda: observed.append('checker_close')),
        states=[], run=lambda case: observed.append(case) or dict(status='PASS'))
    def parallel(*args):
        observed.extend(['parallel', 'parallel_close'])
        return dict(status='PASS')
    monkeypatch.setattr('table_1000.robots.parallel.check_parallel', parallel)
    monkeypatch.setattr('table_1000.robots.checks.Checker', lambda *args, **kwargs:
        observed.append('checker_open') or checker)
    with ExitStack() as stack:
        context = runtime.Context(tmp_path, tmp_path, options, 0, tmp_path, stack, loaded=dict(
            spec=None, settings=None, checker=None, device='cuda:0', app_start_s=0., setup_s=0.))
        for unit in sorted(checks.units(options), key=lambda u: u.priority):
            checks.run_unit(context, unit)
    assert observed == ['parallel', 'parallel_close', 'checker_open', 'keyframes']


def test_trajectory_round_trip(tmp_path):
    import numpy as np
    from table_1000.sim.isaaclab import replay
    states = [{'articulation': {'robot': {'joint_position': np.full((1, 3), t, np.float32)}},
               'rigid_object': {'cube': {'root_pose': np.full((1, 7), t, np.float32)}}} for t in range(4)]
    replay.write_episode(tmp_path / 'trajectory.hdf5', states, {'robot': 'x1pro', 'case': 'keyframes', 'trial': -1})
    file, episodes = replay.episodes(tmp_path / 'trajectory.hdf5')
    episode = episodes['demo_0']
    assert episode.length == 3 and episode.attrs['case'] == 'keyframes'
    assert float(episode.state(0)['articulation']['robot']['joint_position'][0, 0]) == 0
    assert float(episode.state(3)['rigid_object']['cube']['root_pose'][0, 0]) == 3
    file.close()
