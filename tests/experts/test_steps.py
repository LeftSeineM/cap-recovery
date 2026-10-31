"""Steps of an expert without a simulator: which steps run, checkpoints by step, per-step rates, joined episodes."""

import json
import types

import h5py
import numpy as np
import pytest

from table_1000.experts.context import ExpertFailed, Step, run_steps
from table_1000.experts.generate import INITIAL_STATE, merge_checkpoints, read_checkpoints, step_rates, write_checkpoints
from table_1000.common.batch import seeds_record


class FakeContext:
    def __init__(self, only=None):
        self.only, self.current, self.step_names, self.ran, self.kept = only, None, None, [], []
        self.env = types.SimpleNamespace(steps=0)

    def checkpoint(self, index, name):
        self.kept.append(index)

    def fail(self, reason):
        raise ExpertFailed(self.current, reason)


def steps(failing=None):
    def check(name):
        def run(ctx):
            if name == failing:
                ctx.fail('not done')
        return run
    def run(ctx, name):
        ctx.ran.append(name)
        ctx.env.steps += 10
    return [Step(name, lambda ctx, name=name: run(ctx, name), check(name)) for name in 'abc']


def test_run_steps_runs_the_chosen_steps_and_keeps_a_checkpoint_after_each():
    ctx = FakeContext(only=(2, 3))
    run_steps(ctx, steps())
    assert ctx.ran == ['b', 'c'] and ctx.kept == [2, 3] and ctx.step_names == ['a', 'b', 'c']


def test_a_failed_check_is_the_steps_and_keeps_no_checkpoint():
    ctx = FakeContext()
    with pytest.raises(ExpertFailed) as error:
        run_steps(ctx, steps(failing='b'))
    assert error.value.subtask == 'b' and ctx.kept == [1]


def test_run_steps_rejects_steps_the_expert_does_not_have():
    with pytest.raises(ValueError):
        run_steps(FakeContext(only=(2, 4)), steps())


def state(value):
    return {'articulation': {'robot': {'joint_position': np.full((1, 3), value)}}}


def test_checkpoints_are_merged_by_step(tmp_path):
    start = {'qpos': np.zeros(3), 'base_pose': np.array([0., 0., 1.])}
    for seed, done in ((0, 2), (1, 1)):
        write_checkpoints(tmp_path / f'{seed}.hdf5', seed, 3, start,
                          [{'step': k, 'name': f'step {k}', 'at': 10 * k, 'state': state(seed + k)}
                           for k in range(1, done + 1)], {'robot': 'r', 'expert': '0000'})
    assert merge_checkpoints(tmp_path.glob('*.hdf5'), tmp_path / 'batch', {'scene': 's'}) == [1, 2]
    attrs, states = read_checkpoints(tmp_path / 'batch' / 'checkpoints' / 'step2.hdf5')
    assert attrs['scene'] == 's' and attrs['step'] == 2 and attrs['name'] == 'step 2'
    assert list(states) == [0]
    attempt, scene_state, robot = states[0]
    assert attempt == 3 and scene_state['articulation']['robot']['joint_position'][0, 0] == 2.
    assert robot['base_pose'][2] == 1.
    assert sorted(read_checkpoints(tmp_path / 'batch' / 'checkpoints' / 'step1.hdf5')[1]) == [0, 1]



def test_the_initial_state_is_step_0(tmp_path):
    start = {'qpos': np.zeros(3), 'base_pose': np.zeros(3)}
    write_checkpoints(tmp_path / '0.hdf5', 0, 0, start, [{'step': 0, 'name': INITIAL_STATE, 'at': 0, 'state': state(5)},
                                                        {'step': 1, 'name': 'step 1', 'at': 10, 'state': state(6)}], {'robot': 'r', 'expert': '0000'})
    assert merge_checkpoints([tmp_path / '0.hdf5'], tmp_path / 'batch', {'scene': 's'}) == [0, 1]
    attrs, states = read_checkpoints(tmp_path / 'batch' / 'checkpoints' / 'step0.hdf5')
    assert attrs['step'] == 0 and attrs['name'] == INITIAL_STATE
    assert states[0][1]['articulation']['robot']['joint_position'][0, 0] == 5.


def report(seed, done, names='abc', run=None):
    return {'seed': seed, 'step_names': list(names), 'steps_run': run,
            'steps': [{'step': k, 'name': names[k - 1], 'at': k} for k in done]}


def test_step_rates_count_from_the_seeds_the_first_step_started_in():
    reports = [report(0, [1, 2, 3]), report(1, [1, 2]), report(2, [1]), report(3, [])]
    rates = step_rates(reports, [0, 1, 2, 3])
    assert rates['seeds'] == 4
    assert [(row['started'], row['completed']) for row in rates['rows']] == [(4, 3), (3, 2), (2, 1)]
    assert [row['progressive_rate'] for row in rates['rows']] == [.75, .5, .25]
    # Steps 2-3 again from the step 1 checkpoints (seeds 0-2) of that batch.
    resumed = step_rates([report(0, [2, 3], run=[2, 3]), report(1, [2, 3], run=[2, 3]), report(2, [], run=[2, 3])],
                         [0, 1, 2], parent={'steps': rates})
    assert [(row['step'], row['started'], row['completed']) for row in resumed['rows']] == [(1, 4, 3), (2, 3, 2),
                                                                                             (3, 2, 2)]
    assert resumed['rows'][-1]['progressive_rate'] == .5
    assert step_rates([{'seed': 0, 'step_names': None}], [0]) is None


def write_batch(directory, episodes, summary):
    """A batch directory: ``episodes`` ``{seed: (status, actions, attrs)}``."""
    directory.mkdir(parents=True)
    for kind, wanted in (('trajectories', 'success'), ('trajectories_failed', 'failed')):
        with h5py.File(directory / f'{kind}.hdf5', 'w') as file:
            data = file.create_group('data')
            data.attrs['env_args'] = json.dumps({'robot': 'r'})
            for seed, (status, actions, attrs) in episodes.items():
                if (status == 'success') != (wanted == 'success'):
                    continue
                group = data.create_group(f'demo_{seed}')
                group.create_dataset('actions', data=np.asarray(actions, dtype=float)[:, None])
                group.create_group('initial_state').create_dataset('x', data=[[actions[0] - 1.]])
                group.create_group('states').create_dataset('x', data=np.asarray(actions, dtype=float)[:, None])
                group.attrs.update(num_samples=len(actions), status=status,
                                   **{key: json.dumps(value) for key, value in attrs.items()})
    (directory / 'episodes').mkdir()
    for seed, (status, actions, attrs) in episodes.items():
        report = {'seed': seed, 'status': status, 'step_names': ['a', 'b', 'c'], 'steps': attrs['steps'],
                  'subtasks': attrs['subtasks'], 'events': [{'step': 1, 'subtask': None, 'note': 'n'}], 'plans': [],
                  'control_steps': len(actions), 'duration_s': len(actions) / 25, 'wall_s': 10.}
        if 'failure' in attrs:
            report['failure'] = attrs['failure']
        (directory / 'episodes' / f'{seed}.json').write_text(json.dumps(report))
    (directory / 'summary.json').write_text(json.dumps({'robot': 'test', 'expert': '0000', 'scope': {0: None}, 'layout': 0,
                                                     'seeds': seeds_record(episodes),
                                                     'outside_scope': False, **summary}))


def test_join_cuts_each_part_at_the_checkpoint_the_next_started_from(tmp_path, monkeypatch):
    from scripts.scenes import join_steps
    main = join_steps.main
    first, second = tmp_path / 'scene' / 'isaac-cpu' / 'steps1-2', tmp_path / 'scene' / 'isaac-cpu' / 'steps2-3'
    # Steps 1-2 ran in the first batch; the second ran step 2-3 again from the step 1 checkpoint (at 2).
    write_batch(first, {
        0: ('success', [1, 2, 3, 4], {'steps': [{'step': 1, 'at': 2}, {'step': 2, 'at': 4}],
                                      'subtasks': [{'instruction': 'a', 'start': 0, 'end': 2},
                                                   {'instruction': 'b', 'start': 2, 'end': 4}]}),
        1: ('failed', [1, 2], {'steps': [], 'subtasks': [], 'failure': {'subtask': 'a', 'reason': 'r'}}),
        # Failed in step 2 after its step-1 checkpoint: the second batch ran step 2 again.
        2: ('failed', [1, 2, 3], {'steps': [{'step': 1, 'at': 2}], 'subtasks': [],
                                  'failure': {'subtask': 'b', 'reason': 'slipped'}}),
        3: ('error', [1, 2], {'steps': [], 'subtasks': [], 'failure': {'subtask': 'a', 'reason': 'error'}})},
        {'scene': 'scene', 'physics': 'isaac-cpu', 'steps_run': [1, 2], 'from': None, 'commit': 'c1',
         'command': 'run_expert.py scene --dev --steps 1-2 --seeds 0:4'})
    write_batch(second, {
        0: ('success', [7, 8, 9], {'steps': [{'step': 2, 'at': 1}, {'step': 3, 'at': 3}],
                                   'subtasks': [{'instruction': 'b', 'start': 0, 'end': 3}]}),
        2: ('success', [5, 6], {'steps': [{'step': 2, 'at': 1}, {'step': 3, 'at': 2}], 'subtasks': []})},
        {'scene': 'scene', 'physics': 'isaac-cpu', 'steps_run': [2, 3], 'from': str(first), 'commit': 'c2'})
    joined = tmp_path / 'data' / 'scene' / 'isaac-cpu'
    (joined / 'episodes').mkdir(parents=True)     # an earlier batch's, replaced
    (joined / 'episodes' / '9.json').write_text('{}')
    main([str(first), str(second), '--output', str(joined)])
    assert not (joined / 'episodes' / '9.json').exists() and (first / 'summary.json').is_file()
    with h5py.File(joined / 'trajectories.hdf5', 'r') as file:
        group = file['data/demo_0']
        assert group['actions'][:, 0].tolist() == [1, 2, 7, 8, 9]
        assert group['states/x'][:, 0].tolist() == [1, 2, 7, 8, 9] and group.attrs['num_samples'] == 5
        assert [entry['at'] for entry in json.loads(group.attrs['steps'])] == [2, 3, 5]
        assert [(e['instruction'], e['start'], e['end']) for e in json.loads(group.attrs['subtasks'])] == [
            ('a', 0, 2), ('b', 2, 5)]
        assert [part['commit'] for part in json.loads(group.attrs['parts'])] == ['c1', 'c2']
        assert file['data/demo_2']['actions'][:, 0].tolist() == [1, 2, 5, 6]
    with h5py.File(joined / 'trajectories_failed.hdf5', 'r') as file:
        assert json.loads(file['data/demo_1'].attrs['failure'])['reason'] == 'r'
        assert file['data/demo_3'].attrs['status'] == 'error'
    summary = json.loads((joined / 'summary.json').read_text())
    assert (summary['succeeded'], summary['failed'], summary['error']) == (2, 1, 1)
    assert summary['failures'] == {'a: r': [1], 'a: error': [3]} and summary['commits'] == ['c1', 'c2']
    # Each seed's report is its last part's, placed in the joined episode (the second part starts at 2).
    report = json.loads((joined / 'episodes' / '0.json').read_text())
    assert report['events'][0]['step'] == 3 and [entry['at'] for entry in report['steps']] == [3, 5]
    assert report['control_steps'] == 5 and report['wall_s'] == 20. and len(report['parts']) == 2
    assert summary['seeds']['count'] == 4
    with pytest.raises(SystemExit):     # the first batch must start from the initial states
        main([str(second), '--output', str(tmp_path / 'scene' / 'other')])
    data = tmp_path / 'data' / 'scenes' / 'scene' / 'test' / '0000' / 'isaac-cpu'
    monkeypatch.setattr(join_steps, 'ROOT', tmp_path)
    initial = tmp_path / 'assets' / 'scenes' / 'scene' / 'test'
    initial.mkdir(parents=True)
    with h5py.File(initial / 'initial_states.hdf5', 'w') as file:
        for seed in range(4):
            file.create_group(str(seed))
    with pytest.raises(SystemExit):     # steps 1-3 of 3 only: the first batch alone is not the data
        main([str(first)])
    main([str(first), str(second)])     # the scene's data by default
    assert json.loads((data / 'summary.json').read_text())['succeeded'] == 2
    other_layout = data / 'layouts' / '2' / 'summary.json'
    other_layout.parent.mkdir(parents=True)
    other_layout.write_text('{"layout": 2}')
    main([str(first), str(second)])
    assert other_layout.read_text() == '{"layout": 2}'
    with pytest.raises(SystemExit):     # nor replace a batch it joins, or one inside it
        main([str(first), str(second), '--output', str(second)])
    with pytest.raises(SystemExit):
        main([str(first), str(second), '--output', str(tmp_path / 'scene' / 'isaac-cpu')])
    assert (first / 'summary.json').is_file()
    second_summary = json.loads((second / 'summary.json').read_text())
    second_summary['outside_scope'] = True
    (second / 'summary.json').write_text(json.dumps(second_summary))
    outside = tmp_path / 'outside'
    main([str(first), str(second), '--output', str(outside)])
    assert json.loads((outside / 'summary.json').read_text())['outside_scope'] is True
    with pytest.raises(SystemExit):
        main([str(first), str(second)])


@pytest.mark.parametrize('scope, layout, initial_seeds', [([0, 1], 0, [0]), (None, 2, [0, 1])])
def test_partial_seed_join_requires_output(tmp_path, monkeypatch, capsys, scope, layout, initial_seeds):
    from scripts.scenes import join_steps
    from table_1000.scenes import data_dir, initial_dir

    monkeypatch.setattr(join_steps, 'ROOT', tmp_path)
    initial = initial_dir(tmp_path / 'assets' / 'scenes', 'scene', 'test', layout)
    initial.mkdir(parents=True)
    with h5py.File(initial / 'initial_states.hdf5', 'w') as file:
        for seed in initial_seeds:
            file.create_group(str(seed))
    batch = tmp_path / 'steps1-3'
    write_batch(batch, {0: ('success', [1, 2, 3], {'steps': [{'step': 3, 'at': 3}], 'subtasks': []})},
                {'scene': 'scene', 'physics': 'isaac-cpu', 'steps_run': [1, 3], 'scope': {layout: scope},
                 'layout': layout, 'command': 'run_expert.py scene --dev --steps 1-3'})
    formal = data_dir(tmp_path / 'data' / 'scenes', 'scene', 'test', '0000', 'isaac-cpu', layout)
    formal.mkdir(parents=True)
    previous = formal / 'summary.json'
    previous.write_text('{"previous": true}')
    with pytest.raises(SystemExit):
        join_steps.main([str(batch)])
    assert 'missing seeds 1' in capsys.readouterr().err
    assert previous.read_text() == '{"previous": true}'
    output = tmp_path / 'partial-join'
    join_steps.main([str(batch), '--output', str(output)])
    assert json.loads((output / 'summary.json').read_text())['seeds'] == {'range': '0', 'count': 1}


def test_a_failure_case_is_the_last_run_of_its_subtask():
    from scripts.experts.failure_cases import case, failing_start
    report = {'seed': 3, 'duration_s': 1., 'wall_s': 2., 'control_steps': 50,
              'failure': {'subtask': 'pick', 'reason': 'no reachable grasp'},
              'subtasks': [{'instruction': 'pick', 'start': 0, 'end': 10}, {'instruction': 'place', 'start': 10, 'end': 20},
                           {'instruction': 'pick', 'start': 20, 'end': None}],
              'events': [{'step': 5, 'subtask': 'pick', 'note': 'next grasp', 'reason': 'old'},
                         {'step': 25, 'subtask': 'pick', 'note': 'next grasp', 'reason': 'goal in collision',
                          'goal_collisions': [['finger', 'drawer', 'drawer']]}]}
    assert failing_start(report) == 20
    numbers = case(report)
    assert numbers['rejections'] == {'goal in collision': 1} and numbers['collisions'] == {'finger - drawer': 1}
    # A step's check failed: where that step began.
    check = {**report, 'failure': {'subtask': 'step 2', 'reason': 'r'}, 'steps': [{'step': 1, 'at': 18}]}
    assert failing_start(check) == 18 and failing_start({**check, 'steps': []}) == 0


def test_the_steps_of_a_resumed_run_keep_the_seed_s_start():
    from table_1000.experts.context import Context
    env = types.SimpleNamespace(spec=types.SimpleNamespace(base=None, name='test'), steps=0,
                                obs={'agent': {'qpos': np.ones((1, 2)), 'base_pose': np.zeros((1, 3))}}, joints=['a', 'b'])
    scene = types.SimpleNamespace(OBJECTS={}, ROBOT='test', ROBOTS={'test': {'position': [0., 0., .5]}})
    model = types.SimpleNamespace(set_state=lambda *args: None)
    start = {'qpos': np.zeros(2), 'base_pose': np.zeros(3)}
    assert Context(env, model, scene, '.', None, start=start).start is start
    assert Context(env, model, scene, '.', None).start['qpos'].tolist() == [1., 1.]


def test_successful_step_teleop_keeps_source_batch_and_can_be_joined(tmp_path):
    from scripts.experts.teleop import save_attempt
    from scripts.scenes.join_steps import main as join
    from table_1000.experts.teleop import start_checkpoint

    source, output = tmp_path / 'source-batch', tmp_path / 'teleop-result'
    write_batch(source, {4: ('success', [1, 2], {'steps': [{'step': 1, 'at': 2}], 'subtasks': []})},
                {'scene': 'scene', 'robot': 'r', 'expert': '0000', 'physics': 'isaac-cpu', 'steps_run': [1, 1]})
    write_batch(output, {4: ('success', [3, 4], {'steps': [{'step': 2, 'at': 2}], 'subtasks': []})}, {})
    (source / 'checkpoints').mkdir()
    (output / 'checkpoints').mkdir()
    start = {'qpos': np.zeros(3), 'base_pose': np.zeros(3)}
    write_checkpoints(source / 'checkpoints/4.hdf5', 4, 0, start,
                      [{'step': 1, 'name': 'a', 'at': 2, 'state': state(2)}], {'robot': 'r', 'expert': '0000'})
    merge_checkpoints([source / 'checkpoints/4.hdf5'], source, {})
    (source / 'checkpoints/4.hdf5').unlink()
    write_checkpoints(output / 'checkpoints/4.hdf5', 4, 0, start,
                      [{'step': 2, 'name': 'b', 'at': 2, 'state': state(4)}], {'robot': 'r', 'expert': '0000'})
    # The case can come from a later failed batch; step starts at its source's checkpoint.
    case = {'batch': str(tmp_path / 'failed-case'), 'start': str(source / 'checkpoints/step1.hdf5')}
    episode = json.loads((output / 'episodes/4.json').read_text())
    episode['steps_run'] = [2, 2]
    summary = save_attempt(output, episode, case, {'start': {'kind': 'step'}, 'verdict': 'done'},
                           {'scene': 'scene', 'robot': 'r', 'expert': '0000', 'physics': 'isaac-cpu', 'commit': 'abc',
                            'scope': {0: None}, 'layout': 0, 'outside_scope': False, 'steps_run': [2, 2]})
    assert summary['from'] == str(source)
    assert start_checkpoint(output, summary, 2) == source / 'checkpoints/step1.hdf5'
    assert start_checkpoint(output, summary, 2).is_file()
    assert not (output / 'checkpoints/4.hdf5').exists()
    assert list(read_checkpoints(output / 'checkpoints/step2.hdf5')[1]) == [4]
    joined = tmp_path / 'joined'
    join([str(source), str(output), '--output', str(joined)])
    with h5py.File(joined / 'trajectories.hdf5', 'r') as file:
        assert file['data/demo_4/actions'][:, 0].tolist() == [1, 2, 3, 4]


@pytest.mark.parametrize('enter, expected', [(0, 19), (1, 20)])
def test_joined_failure_keeps_traces_entry_state_and_offline_problems(tmp_path, enter, expected):
    import gzip
    from scripts.experts.failure_snapshots import main as snapshots
    from scripts.experts.resolve import main as resolve
    from scripts.scenes.join_steps import main as join
    from table_1000.experts.audit.problems import Problems
    from table_1000.experts.solver import MotionSolver
    from table_1000.experts.audit.trace import CallTrace, frame, read_trace, snapshot_state
    from table_1000.robots import load_spec
    from table_1000.robots.kinematics import RobotModel

    first, last, joined = (tmp_path / name for name in ('first', 'last', 'joined'))
    write_batch(first, {4: ('success', [1, 2, 3], {'steps': [{'step': 1, 'at': 2}], 'subtasks': []})},
                {'scene': 'scene', 'robot': 'x1pro_rm001', 'physics': 'isaac-cpu', 'steps_run': [1, 1]})
    write_batch(last, {4: ('failed', [20, 21], {'steps': [], 'subtasks': [],
                                             'failure': {'subtask': 'pick', 'reason': 'no grasp'}})},
                {'scene': 'scene', 'robot': 'x1pro_rm001', 'physics': 'isaac-cpu',
                 'steps_run': [2, 2], 'from': str(first)})
    for batch, kind in ((first, 'trajectories'), (last, 'trajectories_failed')):
        with h5py.File(batch / f'{kind}.hdf5', 'a') as file:
            file['data'].attrs['env_args'] = json.dumps({'action_layout': {'gripper': [0, 1]}})
            length = file['data/demo_4'].attrs['num_samples']
            file.create_dataset('data/demo_4/obs/qpos', data=np.zeros((length, 2)))
            file.create_dataset('data/demo_4/obs/base_pose', data=np.zeros((length, 3)))
    model = RobotModel(load_spec('x1pro_rm001'), tmp_path / 'planning')
    model.set_state(model.spec.keyframe('ready'))
    ctx = types.SimpleNamespace(model=model, spec=model.spec, steps=enter, current='pick', _skills=None,
                                rng=np.random.default_rng(4), scene=types.SimpleNamespace(OBJECTS={}),
                                env=types.SimpleNamespace(targets=lambda: {'gripper': np.array([[.03]])}))
    ctx.solvers = {'left': MotionSolver(ctx, 'left')}
    ctx.trace = CallTrace(ctx, 4, 0)
    record = ctx.trace.problems = Problems(ctx)
    record.install()
    try:
        with pytest.raises(ExpertFailed):
            with frame(ctx, 'step', 'b', {'index': 2}):
                with frame(ctx, 'skill', 'pick', {'arm': 'left'}, concrete=True):
                    assert model.ik('left', model.tcp('left', world=True), world=True)[0] == 'Success'
                    ctx.steps = 2
                    raise ExpertFailed('pick', 'no grasp')
    finally:
        record.close()
    path = last / 'episodes/4.json'
    report = json.loads(path.read_text())
    report.update(steps_run=[2, 2], call_trace='calls/4.json.gz', solve_trace='problems/4.json.gz', failure_frame=1)
    path.write_text(json.dumps(report))
    ctx.trace.save(last / report['call_trace'])
    record.save(last / report['solve_trace'])
    join([str(first), str(last), '--output', str(joined)])
    # A joined batch stays usable after its sources are moved, and after being joined again.
    first.rename(tmp_path / 'first-moved')
    last.rename(tmp_path / 'last-moved')
    again = tmp_path / 'joined-again'
    join([str(joined), '--output', str(again)])
    joined.rename(tmp_path / 'joined-moved')
    report = json.loads((again / 'episodes/4.json').read_text())
    assert (again / report['call_trace']).is_file() and (again / report['solve_trace']).is_file()
    assert report['call_trace_offset'] == 2
    trace = read_trace(again, report)
    assert trace['frames'][1]['enter'] == 2 + enter and trace['frames'][1]['exit'] == 4
    state, snapshot = snapshot_state(again, report)
    assert state['x'][0, 0] == expected
    assert snapshot['frame']['id'] == 1 and snapshot['step'] == 2
    assert state['targets']['gripper'][0, 0] == .03
    snapshots([str(again), '--output', str(tmp_path / 'snapshots')])
    (checkpoint,) = (tmp_path / 'snapshots').glob('scene/*/checkpoints/step1.hdf5')
    saved = read_checkpoints(checkpoint)[1][4][1]
    assert saved['x'][0, 0] == expected
    assert json.loads(saved['_expert_frame'])['frame']['enter'] == 2 + enter
    resolve([str(again), '--seeds', '4', '--frame', '1', '--runs', '1', '--output', str(tmp_path / 'resolve')])
    with gzip.open(tmp_path / 'resolve/4.json.gz', 'rt') as file:
        result = json.load(file)
    assert result['selected'] == [0]
    assert result['operations'][0]['frame'] == 1 and result['operations'][0]['successes'] == 1
