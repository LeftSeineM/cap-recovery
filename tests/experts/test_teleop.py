"""Teleoperating cases without a simulator: the cases of batches, drags in a view, the operated step."""

import json
import types

import h5py

import numpy as np

from table_1000.experts.context import ExpertFailed, Step
from table_1000.experts.teleop import (HumanExpert, Orbit, cases, check, drag, failed_step, trajectory_start,
                                      trajectory_state, view_camera)

NAMES = ['open', 'move', 'close']


def report(seed, done, status='failed', steps_run=None, reason='no grasp'):
    return {'seed': seed, 'status': status, 'step_names': NAMES, 'steps_run': steps_run,
            'steps': [{'step': step, 'name': NAMES[step - 1], 'at': 10 * step} for step in done],
            **({'failure': {'subtask': 'x', 'reason': reason}} if status != 'success' else {})}


def write(batch, summary, reports):
    (batch / 'episodes').mkdir(parents=True)
    (batch / 'summary.json').write_text(json.dumps(summary))
    for name, entry in reports.items():
        (batch / 'episodes' / f'{name}.json').write_text(json.dumps(entry))


def write_trajectory(path, seed, length):
    """A failed trajectory file: the pen's pose and a two-group action (``arm``, ``gripper``) per step."""
    with h5py.File(path, 'w') as file:
        data = file.create_group('data')
        data.attrs['env_args'] = json.dumps({'action_layout': {'arm': [0, 2], 'gripper': [2, 3]}})
        group = data.create_group(f'demo_{seed}')
        group.attrs['num_samples'] = length
        group.create_dataset('states/rigid_object/pen/root_pose', data=np.arange(length * 7.).reshape(length, 7))
        group.create_dataset('actions', data=np.arange(length * 3.).reshape(length, 3))
        group.create_dataset('obs/qpos', data=np.ones((length, 2)))
        group.create_dataset('obs/base_pose', data=np.zeros((length, 3)))


def test_the_failed_step_is_the_first_not_completed():
    assert failed_step(report(0, [1])) == 2
    assert failed_step(report(0, [], steps_run=[2, 3])) == 2
    assert failed_step(report(0, [1, 2, 3], status='success')) is None


def test_cases_come_by_success_rate_and_start_before_the_failed_step(tmp_path):
    whole = tmp_path / 'expert' / '000002' / 'a-whole'
    write(whole, {'run': 'a-whole', 'scene': '000002'},
          {'0': report(0, [1, 2, 3], status='success'), '1': report(1, [1])})
    repeated = tmp_path / 'expert' / '000002' / 'b-repeats'
    write(repeated, {'run': 'b-repeats', 'scene': '000002', 'steps_run': [2, 2], 'from': str(whole),
                     'repeats': {'count': 3, 'succeeded': {'4': 1, '5': 2, '6': 3}}},
          {'4': report(4, [], steps_run=[2, 2]), '4-1': report(4, [2], status='success', steps_run=[2, 2]),
           '5': report(5, [2], status='success', steps_run=[2, 2]), '5-2': report(5, [], steps_run=[2, 2], reason='drop'),
           '6': report(6, [2], status='success', steps_run=[2, 2])})
    found = cases(tmp_path / 'expert', '000002')
    assert [(case['run'], case['seed'], case['successes']) for case in found] == \
        [('b-repeats', 4, 1), ('b-repeats', 5, 2), ('a-whole', 1, None)]
    # A repeated seed whose first run succeeded is a case from the run that failed.
    assert found[1]['failure'] == 'drop' and found[1]['rate'] == 2 / 3
    # The batch's first step starts from the checkpoints of the batch it ran from; a later one from its own.
    assert found[0]['start'] == str(whole / 'checkpoints' / 'step1.hdf5')
    assert found[2]['start'] == str(whole / 'checkpoints' / 'step1.hdf5') and found[2]['step'] == 2


def test_a_drag_moves_in_the_image_plane_at_the_point_depth():
    view = {'position': (0., -2., 1.), 'target': (0., 0., 1.), 'focal_length_mm': 20.955, 'resolution': (100, 50)}
    camera = view_camera(view)          # fx = 100 px, looking along +y
    point = np.array([0., 0., 1.])      # 2 m ahead
    np.testing.assert_allclose(drag(camera, point, 10., 0.), [.2, 0., 0.], atol=1e-12)     # right: +x
    np.testing.assert_allclose(drag(camera, point, 0., 10.), [0., 0., -.2], atol=1e-12)    # down: -z
    assert drag(camera, point, 0., 0., wheel=1.)[1] < 0.                                   # toward the camera


def test_the_operated_step_replaces_the_expert_step_unchecked():
    calls = []
    expert = types.SimpleNamespace(STEPS=[Step('a', lambda ctx: calls.append('a'), lambda ctx: None),
                                          Step('b', lambda ctx: calls.append('b'), lambda ctx: ctx.fail('no'))])
    human = HumanExpert(expert, 2, lambda ctx, step: calls.append(('operated', step.name)))
    assert human.STEPS[0] is expert.STEPS[0]
    assert human.STEPS[1].check is None
    human.STEPS[1].run(None)
    assert calls == [('operated', 'b')]


def test_a_check_reports_its_verdict_and_keeps_the_subtask():
    class Ctx:
        current = 'take the pen'

        def fail(self, reason):
            raise ExpertFailed(self.current, reason)
    ctx = Ctx()
    assert check(ctx, Step('b', None, lambda ctx: ctx.fail('pen not on the table'))) == \
        {'passed': False, 'reason': 'pen not on the table'}
    assert check(ctx, Step('b', None, lambda ctx: None)) == {'passed': True, 'reason': None}
    assert ctx.current == 'take the pen'


def test_an_outside_view_turns_about_its_target_pans_and_zooms():
    view = {'position': (0., -2., 1.), 'target': (0., 0., 1.), 'focal_length_mm': 20.955, 'resolution': (100, 50)}
    orbit = Orbit(view)
    orbit.move('orbit', np.pi / 2 / .005, 0.)       # dragged right a quarter turn: the camera swings to its left
    np.testing.assert_allclose(orbit.view['position'], (-2., 0., 1.), atol=1e-9)
    orbit.move('zoom', 0., 1.)                      # closer, same target
    assert np.linalg.norm(np.subtract(orbit.view['position'], orbit.view['target'])) < 2.
    target = np.array(orbit.view['target'])
    orbit.move('pan', 10., 0.)                      # dragging right moves the scene right: the target left
    assert np.linalg.norm(np.array(orbit.view['target']) - target) > 0.
    assert view['position'] == (0., -2., 1.)        # the scene's view is left as it was


def test_a_recorded_failure_gives_the_state_at_the_failure(tmp_path):
    batch = tmp_path / 'batch'
    write(batch, {'scene': '000002'}, {'3': report(3, [1]), '4': report(4, [1])})
    (batch / 'episodes' / '3.json').write_text(json.dumps({**report(3, [1]), 'steps': [
        {'step': 1, 'name': 'open', 'at': 4}]}))
    write_trajectory(batch / 'trajectories_failed.hdf5', 3, 6)
    found = {case['seed']: case for case in cases(tmp_path / 'none', '000002', [batch])}
    assert found[3]['failure_at'] == 6 and found[3]['step_at'] == 4
    assert 'trajectory' not in found[4]          # not recorded
    state = trajectory_state(found[3]['trajectory'], 3, found[3]['failure_at'] - 1)
    np.testing.assert_array_equal(state['rigid_object']['pen']['root_pose'], [np.arange(35., 42.)])
    # What the action terms held there: that step's action, by group (a gripper squeezing: below its gap).
    np.testing.assert_array_equal(state['targets']['arm'], [[15., 16.]])
    np.testing.assert_array_equal(state['targets']['gripper'], [[17.]])
    assert trajectory_start(found[3]['trajectory'], 3)['qpos'].shape == (1, 2)
    assert found[3]['run'].endswith('/batch')


def test_failure_snapshots_are_the_checkpoints_the_failed_steps_start_from(tmp_path):
    from scripts.experts.failure_snapshots import main
    from table_1000.experts.generate import read_checkpoints
    batch = tmp_path / 'batch'
    write(batch, {'scene': '000002', 'physics': 'isaac-cpu', 'commit': 'abc'},
          {'3': {**report(3, [1]), 'steps': [{'step': 1, 'name': 'open', 'at': 4}]}, '5': report(5, [1, 2, 3], status='success')})
    write_trajectory(batch / 'trajectories_failed.hdf5', 3, 30)
    main([str(batch), '--before', '1', '--output', str(tmp_path / 'runs')])
    (snapshots,) = (tmp_path / 'runs' / '000002').iterdir()
    attrs, found = read_checkpoints(snapshots / 'checkpoints' / 'step1.hdf5')   # seed 3 failed in step 2
    assert attrs['physics'] == 'isaac-cpu' and list(found) == [3]
    # 1 s (25 control steps) before the last sample, not before the step began (sample 4).
    np.testing.assert_array_equal(found[3][1]['rigid_object']['pen']['root_pose'], [np.arange(4 * 7., 5 * 7.)])
    np.testing.assert_array_equal(found[3][1]['targets']['gripper'], [[14.]])
    assert json.loads((snapshots / 'summary.json').read_text())['snapshots'] == {'2': [3]}


def test_step_start_operates_first_ik_without_a_solver(tmp_path, monkeypatch):
    from table_1000.experts.context import Context
    from table_1000.experts.generate import Generator
    from table_1000.experts.teleop import Operator, operate
    from table_1000.robots import load_spec
    from table_1000.robots.kinematics import RobotModel

    model = RobotModel(load_spec('x1pro_rm001'), tmp_path / 'planning')
    model.set_state(model.spec.keyframe('ready'))
    ready = model.q.copy()
    env = types.SimpleNamespace(spec=model.spec, joints=model.joints, steps=0,
                                contacts=types.SimpleNamespace(phase=None, intervals=lambda: []),
                                privileged=lambda: {'joints': {}, 'poses': {}})
    def reset(state):
        env.steps = 0
        env.obs = {'agent': {'qpos': ready[None].copy(), 'base_pose': np.zeros((1, 3)),
                             'tcp': {arm: model.tcp(arm, world=True)[None] for arm in model.spec.arms}}}
    env.reset_to_state = reset
    env.qpos = lambda joints: env.obs['agent']['qpos'][:, [model.index[name] for name in joints]]
    env.targets = lambda: {f'{arm}_gripper': np.array([[.08]]) for arm in model.spec.arms}
    actions = []
    def advance(action):
        env.steps += 1
        actions.append(action)
        return env.obs
    env.step = advance
    operator, log, seen = Operator(), {}, []
    def show(ctx, status, targets, arm):
        assert ctx.solvers == {}
        seen.append(status)
        operator.command('done')
    # Keep the simulation state provider out of this test; use real Context/operate/RobotModel IK.
    monkeypatch.setattr(Context, 'checkpoint', lambda ctx, *args: None)
    generator = object.__new__(Generator)
    generator.name, generator.output, generator.assets = 'fake', tmp_path, tmp_path
    generator.steps, generator.source, generator.problems = (2, 2), 'checkpoint', 'failed'
    generator.skill_attempts = 3
    generator.env, generator.model, generator.spec = env, model, model.spec
    generator.env_args = {}
    generator.scene = types.SimpleNamespace(OBJECTS={}, ROBOT={'position': [0., 0., 0.]})
    expert = types.SimpleNamespace(STEPS=[Step('first', None), Step('move', None)])
    generator.expert = HumanExpert(expert, 2, lambda ctx, step: operate(ctx, step, operator, show, {}, log))
    generator.recording = types.SimpleNamespace(begin=lambda *args: None, finish=lambda *args: None)
    (tmp_path / 'episodes').mkdir()
    result = generator.run(4, 2, {}, start={'qpos': ready, 'base_pose': np.zeros(3)})
    assert result['status'] == 'success', result.get('failure')
    assert len(seen) == 1 and seen[0]['unreachable'] == []
    assert all(arm in actions[-1] for arm in model.spec.arms)
    assert result['solve_trace'] is None and not (tmp_path / 'problems').exists()
    assert (tmp_path / result['call_trace']).is_file()


def test_resolve_outputs_stay_outside_the_teleop_case_scan(tmp_path, monkeypatch):
    import gzip
    from scripts.experts import resolve
    from table_1000.experts import problems

    batch = tmp_path / 'runs/expert/000002/failed'
    episode = {**report(4, [1]), 'solve_trace': 'problems/4.json.gz'}
    write(batch, {'scene': '000002'}, {'4': episode})
    (batch / 'problems').mkdir()
    with gzip.open(batch / episode['solve_trace'], 'wt') as file:
        json.dump({'operations': [], 'decisions': []}, file)
    monkeypatch.setattr(resolve, 'ROOT', tmp_path)
    monkeypatch.setattr(problems, 'Offline', lambda *args: types.SimpleNamespace(replay=lambda *args: []))
    resolve.main([str(batch), '--seeds', '4', '--runs', '1', '--name', 'check'])
    (output,) = (tmp_path / 'runs/resolve/000002').iterdir()
    assert output.name.endswith('-nocommit-check')
    assert isinstance(json.loads((output / 'summary.json').read_text()), list)
    assert [(case['seed'], case['batch']) for case in cases(tmp_path / 'runs/expert', '000002')] == [(4, str(batch))]
