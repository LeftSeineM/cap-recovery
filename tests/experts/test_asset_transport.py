import types

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from table_1000.experts.asset_grasp.stability import TRANSPORT, classify, is_dropped, metrics, motion
from table_1000.experts.grasps import Box


@pytest.fixture(autouse=True)
def geometric_preflight(monkeypatch):
    # These fake IK models test reachability and execution boundaries. The real
    # TOPPRA/FK constraints have their own regression below.
    import table_1000.experts.asset_grasp as evaluation
    def stroke(model, arm, result, excursion, **kwargs):
        rows = np.asarray(result['position'])
        lift = np.asarray(result['lift'])
        return {**result, 'position': np.vstack([rows, rows[-2::-1]]),
                'lift': np.vstack([lift, lift[-2::-1]]),
                'duration': 2 * (len(rows) - 1) / 25,
                'command_motion_peaks': dict(linear_acceleration_m_s2=1.0, angular_acceleration_rad_s2=2.9),
                'constraint_ratios': {}}
    monkeypatch.setattr(evaluation, 'retime_stroke', stroke)
    monkeypatch.setattr(evaluation, 'coupled_twist', lambda *a, **kw: np.zeros(3))
    original_plan = evaluation.transport_plan
    def plan(model, *args, **kwargs):
        if not hasattr(model, 'check_path'):
            model.check_path = lambda *args, **kwargs: (0, [])
        return original_plan(model, *args, **kwargs)
    monkeypatch.setattr(evaluation, 'transport_plan', plan)
    monkeypatch.setattr(evaluation.Evaluator, 'transport_start',
                        lambda evaluator, ctx, arm: (evaluator.model.q.copy(), ctx.tcp(arm).copy()))


def test_rotation_about_grip_point_is_separate_from_far_vertex_slip():
    reference = np.eye(4)
    reference[:3, 3] = [.2, 0., 0.]
    rotation = np.eye(4)
    rotation[:3, :3] = Rotation.from_euler('z', 4, degrees=True).as_matrix()
    result = metrics(reference, rotation @ reference, Box((.4, .04, .02)).points())
    assert result['translation_m'] < 1e-12
    assert result['rotation_rad'] == pytest.approx(np.radians(4))
    assert result['vertex_slip_m'] > .005
    assert classify([dict(result, dropped=False)])['held']


@pytest.mark.parametrize('translation,angle,dropped,held', [
    (.0049, 4.9, False, True), (.0051, 0, False, False),
    (0, 5.1, False, False), (0, 0, True, False)])
def test_both_limits_and_drop_are_independent(translation, angle, dropped, held):
    row = {'translation_m': translation, 'rotation_rad': np.radians(angle),
           'vertex_slip_m': .03, 'dropped': dropped}
    zero = dict(row, translation_m=0., rotation_rad=0., dropped=False)
    assert classify([zero, row, zero])['held'] == held
    assert not classify([zero], completed=False)['held']


def test_dropped_requires_lost_finger_contact_and_large_separation():
    assert not is_dropped(.001, .06, True)  # rotation about an actual gripping point
    assert not is_dropped(.004, 0., False)  # brief loss of contact is not yet a drop
    assert is_dropped(.051, 0., False)
    assert is_dropped(.001, .051, False)


def test_measured_derivatives():
    t = np.arange(10) / 25
    poses = np.tile(np.eye(4), (len(t), 1, 1))
    poses[:, 0, 3] = .5 * 2 * t ** 2
    poses[:, :3, :3] = Rotation.from_rotvec(np.c_[t * 0, t * 0, t * .7]).as_matrix()
    measured = motion(poses, 1 / 25)
    assert np.allclose(measured['linear_acceleration_m_s2'], 2.)
    assert np.allclose(measured['angular_velocity_rad_s'], .7)
    assert measured['angular_acceleration_rad_s2'].max() < 1e-12


def test_transport_does_not_reset_command_to_lagging_measurement_at_segment_boundary():
    from table_1000.experts.asset_grasp import Evaluator
    evaluator = Evaluator.__new__(Evaluator)
    model = types.SimpleNamespace(q=np.array([.1, .1]), limits=np.array([[-3., 3.], [-3., 3.]]),
             arm_indices=lambda arm: [0], lift_indices=lambda: [1])
    model.ik = lambda arm, pose, start, **kw: ('Success', np.array([
        pose[1, 3] + .25 + Rotation.from_matrix(pose[:3, :3]).as_rotvec()[0], start[1]]))
    evaluator.model = model
    evaluator.spec = types.SimpleNamespace(arms={'right': types.SimpleNamespace(finger_links=['l', 'r'])})
    tcp = np.eye(4)
    tcp[1, 3] = -.15
    ctx = types.SimpleNamespace(observers=[], steps=0, note=lambda *a, **k: None,
          tcp=lambda arm: tcp.copy(), poses=lambda: {'object:body': np.eye(4)},
          geometry={'object': types.SimpleNamespace(points=lambda **kw: Box((.02,) * 3).points())},
          env=types.SimpleNamespace(contacts=types.SimpleNamespace(surface_contacts={}, phase='start')))
    phase, commands = {}, []
    def step(action=None):
        ctx.steps += 1
        if action:
            commands.append((phase['text'], action['right'][0, 0]))
        model.q[:] = .1  # the simulated measurement lags the commanded endpoint
        for observer in ctx.observers:
            observer(ctx)
    ctx.step = step
    ctx.hold = lambda seconds: [step() for _ in range(round(seconds * 25))]
    evaluator.transport(ctx, types.SimpleNamespace(body='object:body'), 'right', phase)
    i = next(i for i, (segment, _) in enumerate(commands) if segment == 'transport rotation')
    expected = 2 * TRANSPORT['angular_velocity_rad_s'] ** 2 / TRANSPORT['angular_acceleration_rad_s2'] / 32
    assert commands[i][1] - commands[i - 1][1] == pytest.approx(expected)
    assert not ctx.observers


def test_transport_rejects_local_ik_failure_instead_of_switching_joint_branch():
    from table_1000.experts.asset_grasp import Evaluator
    evaluator = Evaluator.__new__(Evaluator)
    model = types.SimpleNamespace(q=np.zeros(2), limits=np.array([[-3., 3.], [-3., 3.]]),
              arm_indices=lambda arm: [0], lift_indices=lambda: [1])
    # Multistart can solve the pose on a distant branch, but cannot continue
    # the previous command locally. No such command may reach the actuator.
    model.ik = lambda *args, **kw: ('IK Failed', None) if kw['attempts'] == 1 else ('Success', np.ones(2))
    evaluator.model = model
    evaluator.transport_reference = lambda arm, ctx: (model.q.copy(), np.eye(4), None, {})
    evaluator.spec = types.SimpleNamespace(arms={'right': types.SimpleNamespace(finger_links=['l', 'r'])})
    def fail(reason, **details):
        raise RuntimeError(reason)
    commands = []
    ctx = types.SimpleNamespace(observers=[], steps=0, fail=fail, note=lambda *a, **k: None,
          tcp=lambda arm: np.eye(4), poses=lambda: {'object:body': np.eye(4)},
          geometry={'object': types.SimpleNamespace(points=lambda **kw: Box((.02,) * 3).points())},
          env=types.SimpleNamespace(contacts=types.SimpleNamespace(surface_contacts={}, phase='start')),
          step=lambda action: commands.append(action))
    with pytest.raises(RuntimeError, match='transport motion unreachable'):
        evaluator.transport(ctx, types.SimpleNamespace(body='object:body'), 'right', {})
    assert commands == [] and ctx.observers == []


def test_preflight_reverses_direction_and_checks_up_before_any_execution():
    from table_1000.experts.asset_grasp import transport_plan
    origin = np.eye(4)
    origin[1, 3] = -.12
    calls = []
    model = types.SimpleNamespace(q=np.zeros(2), limits=np.array([[-3., 3.], [-3., 3.]]),
              arm_indices=lambda arm: [0], lift_indices=lambda: [1])
    def ik(arm, pose, start, **kwargs):
        assert kwargs['attempts'] == 1
        calls.append(pose.copy())
        if pose[1, 3] > -.09:  # toward the midline reaches the local joint limit
            return 'IK Failed', None
        return 'Success', np.array([pose[1, 3] + .12, pose[2, 3]])
    model.ik = ik
    commands, plan = transport_plan(model, 'right', origin)
    assert plan['choices'][0]['rejected']['segment'] == 'lateral'
    assert plan['selected']['horizontal_direction'] == [0., -1., 0.]
    assert {row[0] for row in commands} == {'lateral', 'up', 'rotation'}
    assert min(pose[1, 3] for pose in calls) == pytest.approx(-.17)
    assert max(pose[2, 3] for pose in calls) == pytest.approx(.05)
    assert np.array_equal(model.q, [0, 0])


def test_preflight_discards_whole_choice_when_later_segment_fails():
    from table_1000.experts.asset_grasp import transport_plan
    model = types.SimpleNamespace(q=np.zeros(2), limits=np.array([[-3., 3.], [-3., 3.]]),
              arm_indices=lambda arm: [0], lift_indices=lambda: [1])
    calls = []
    def ik(arm, pose, start, **kwargs):
        calls.append(pose.copy())
        return ('IK Failed', None) if pose[2, 3] > .03 else ('Success', start.copy())
    model.ik = ik
    commands, plan = transport_plan(model, 'right', np.eye(4))
    assert commands is None and plan['selected'] is None
    assert len(plan['choices']) == 58
    assert plan['choices'][-1]['rejected']['segment'] == 'up'


def test_preflight_projects_small_start_overshoot_and_its_tcp_without_changing_limits():
    from table_1000.experts.asset_grasp import transport_plan
    measured = np.array([3.0002, 0.])
    limits = np.array([[-3., 3.], [-3., 3.]])
    origin = np.eye(4)
    origin[0, 3] = 3.0002
    def tcp(arm, q, world):
        assert world and np.array_equal(q, [3., 0.])
        result = np.eye(4)
        result[0, 3] = q[0]
        return result
    model = types.SimpleNamespace(q=measured.copy(), limits=limits.copy(), tcp=tcp,
        arm_indices=lambda arm: [0], lift_indices=lambda: [1])
    def ik(arm, pose, start, **kwargs):
        assert np.all(start >= limits[:, 0]) and np.all(start <= limits[:, 1])
        # A target outside the strict range remains a failure. The measured
        # origin would fail; the projected origin permits a full lateral pulse.
        return ('Success', start.copy()) if pose[0, 3] <= 3. else ('IK Failed', None)
    model.ik = ik
    commands, plan = transport_plan(model, 'right', origin)
    assert commands is None and plan['rejected']['reason'] == 'transport start lacks joint tracking clearance'
    assert plan['start_projection']['max_correction'] == pytest.approx(.0002)
    np.testing.assert_array_equal(model.q, measured)
    np.testing.assert_array_equal(model.limits, limits)
    np.testing.assert_array_equal(origin[:3, 3], [3.0002, 0., 0.])


def test_preflight_does_not_project_a_start_outside_the_tracking_budget():
    from table_1000.experts.asset_grasp import transport_plan
    model = types.SimpleNamespace(q=np.array([3.002, 0.]), limits=np.array([[-3., 3.], [-3., 3.]]),
        arm_indices=lambda arm: [0], lift_indices=lambda: [1],
        ik=lambda *a, **k: pytest.fail('an invalid measured start must be rejected before IK'))
    commands, plan = transport_plan(model, 'right', np.eye(4))
    assert commands is None and not plan['choices']
    assert plan['rejected'] == 'measured start exceeds projection tolerance'


def test_preflight_rejects_interior_hard_stop_despite_safe_start():
    from table_1000.experts.asset_grasp import transport_plan
    model = types.SimpleNamespace(q=np.zeros(2), limits=np.array([[-1.01, 1.01], [-3., 3.]]),
        arm_indices=lambda arm: [0], lift_indices=lambda: [1],
        ik=lambda arm, pose, start, **kw: ('Success', np.array([
            Rotation.from_matrix(pose[:3, :3]).magnitude(), 0.])))
    commands, plan = transport_plan(model, 'right', np.eye(4))
    assert commands is None and plan['rejected']['segment'] == 'rotation'
    assert plan['joint_clearance'] == [1.01]
    rotations = [c for c in plan['choices'] if c['segment'] == 'rotation']
    assert rotations and all(c['rejected']['reason'] == 'transport path lacks joint tracking clearance'
                             for c in rotations)
    assert all(0. < min(c['rejected']['clearance']) < .01 for c in rotations)


def test_transport_reference_preserves_other_arm_and_closed_fingers(monkeypatch):
    import table_1000.experts.asset_grasp as evaluation
    evaluator = evaluation.Evaluator.__new__(evaluation.Evaluator)
    measured = np.array([1., 2., 3., 4., 5.])
    evaluator.model = types.SimpleNamespace(q=measured.copy(), arm_indices=lambda arm: [0, 1],
        lift_indices=lambda: [2], vector=lambda value: np.array(value),
        tcp=lambda arm, q, **kw: np.eye(4))
    evaluator.spec = types.SimpleNamespace(keyframe=lambda key: [10., 20., 30., 40., 50.])
    evaluator.settings = types.SimpleNamespace(ROBOT={'keyframe': 'ready'})
    starts = []
    monkeypatch.setattr(evaluation, 'transport_plan', lambda model, arm, origin, start: (starts.append(start.copy()) or [], {}))
    goal, _, _, _ = evaluator.transport_reference('right')
    np.testing.assert_array_equal(goal, [10., 20., 30., 4., 5.])
    np.testing.assert_array_equal(starts[0], goal)
    np.testing.assert_array_equal(evaluator.model.q, measured)


@pytest.mark.parametrize('slip,planning_failure', [(0., False), (.006, False), (.06, True), (0., 'tracking')])
def test_transport_relocation_checks_grip_without_resetting_reference(monkeypatch, slip, planning_failure):
    import table_1000.experts.asset_grasp as evaluation
    from table_1000.experts.solver import PlanningFailed
    evaluator = evaluation.Evaluator.__new__(evaluation.Evaluator)
    evaluator.model = types.SimpleNamespace(q=np.zeros(2), joints=['arm', 'lift'])
    evaluator.spec = types.SimpleNamespace(arms={'right': types.SimpleNamespace(finger_links=['l', 'r'])})
    goal = np.array([.1, .2])
    evaluator.transport_reference = lambda arm, ctx: (goal, np.eye(4), [('ready',)], {'selected': 'reference'})
    commands = [('lateral', np.array([.01]), np.array([.02]))] * 4
    preflights = iter([(None, {'choices': [1, 2, 3, 4], 'selected': None}),
                      (commands, {'selected': 'relocated'})])
    monkeypatch.setattr(evaluation, 'transport_plan', lambda *args, **kwargs: next(preflights))
    tcp, obj = np.eye(4), np.eye(4)
    calls, actions = [], []
    def fail(reason, **details):
        raise AssertionError(reason)
    ctx = types.SimpleNamespace(observers=[], steps=0, fail=fail, note=lambda *a, **k: None,
        tcp=lambda arm: tcp.copy(), poses=lambda: {'object:body': obj.copy()},
        geometry={'object': types.SimpleNamespace(points=lambda **kw: Box((.02,) * 3).points())},
        env=types.SimpleNamespace(contacts=types.SimpleNamespace(surface_contacts={}, phase='start')))
    def step(action=None):
        ctx.steps += 1
        actions.append(action)
        for observer in ctx.observers:
            observer(ctx)
    def relocate(pose, *, goals, lift, retime):
        calls.append((pose, goals, lift))
        for _ in range(3):
            tcp[0, 3] += .01
            obj[0, 3] = tcp[0, 3] + slip
            step()
        if planning_failure:
            raise PlanningFailed(None, 'tracking' if planning_failure == 'tracking' else 'RRTConnect failed')
    ctx.solver = lambda arm: types.SimpleNamespace(move_to_pose_with_RRTConnect=relocate)
    ctx.step, ctx.hold = step, lambda seconds: [step() for _ in range(round(seconds * 25))]
    result = evaluator.transport(ctx, types.SimpleNamespace(body='object:body'), 'right', {})
    assert calls[0][0] is None and calls[0][2] is True
    np.testing.assert_array_equal(calls[0][1][0], goal)
    assert result['completed'] == (slip == 0.)
    assert result['held'] == (slip == 0.)
    assert result['dropped'] == (slip > .05)
    assert result['max_translation_m'] == pytest.approx(slip)
    assert any(row['segment'] == 'transport relocation' for row in result['trace'])
    assert sum(action is not None for action in actions) == (4 if slip == 0. else 0)
    assert not ctx.observers


@pytest.mark.parametrize('status', ['Success', 'IK Failed'])
def test_transport_reference_prefers_nominal_work_height_then_declared_pose(monkeypatch, status):
    import table_1000.experts.asset_grasp as evaluation
    from table_1000.experts.skills.common import IK_ATTEMPTS
    from table_1000.experts.skills.grasping import LIFT
    evaluator = evaluation.Evaluator.__new__(evaluation.Evaluator)
    evaluator.center = np.array([.5, -.15])
    evaluator.settings = types.SimpleNamespace(ROBOT={'keyframe': 'ready'})
    evaluator.spec = types.SimpleNamespace(keyframe=lambda key: [10., 20., 30.])
    start, goal = np.array([.1, .2, .3]), np.array([.4, .5, .3])
    tcp, grasp_pose = np.eye(4), np.eye(4)
    tcp[:3, :3] = Rotation.from_euler('z', .3).as_matrix()
    grasp_pose[2, 3] = .9
    calls, origins = [], []
    def ik(arm, target, seed, **kwargs):
        calls.append(kwargs)
        np.testing.assert_allclose(target[:3, :3], tcp[:3, :3])
        np.testing.assert_allclose(target[:3, 3], [.5, -.15, .9 + LIFT])
        np.testing.assert_array_equal(seed, start)
        return status, goal if status == 'Success' else None
    evaluator.model = types.SimpleNamespace(q=start.copy(), ik=ik, vector=lambda q: np.array(q),
        arm_indices=lambda arm: [0], lift_indices=lambda: [1], tcp=lambda *a, **k: np.eye(4))
    ctx = types.SimpleNamespace(tcp=lambda arm: tcp.copy(), solver=lambda arm: types.SimpleNamespace(sync=lambda: None),
        skills=lambda: types.SimpleNamespace(held={'right': {'grasp': {'pose': grasp_pose}}}))
    def preflight(model, arm, origin, start):
        origins.append(origin.copy())
        return [('complete motion',)], {'selected': True}
    monkeypatch.setattr(evaluation, 'transport_plan', preflight)
    selected, _, commands, plan = evaluator.transport_reference('right', ctx)
    assert calls == [dict(world=True, attempts=IK_ATTEMPTS, lift=True)]
    np.testing.assert_array_equal(selected, goal if status == 'Success' else [10., 20., .3])
    assert commands and plan['references'][0]['ik_status'] == status
    assert len(plan['references']) == (1 if status == 'Success' else 2)


def test_retimed_ik_jitter_respects_joint_and_tcp_envelopes():
    import mplib
    from table_1000.experts.asset_grasp.transport import retime
    from table_1000.experts.asset_grasp.stability import MOTION_LIMITS
    planner = types.SimpleNamespace(joint_vel_limits=np.array([.3, 1.]),
                                    joint_acc_limits=np.array([.6, 2.]))
    planner.TOPP = types.MethodType(mplib.Planner.TOPP, planner)
    def tcp(arm, q, world):
        pose = np.eye(4)
        pose[0, 3] = q[0]
        pose[:3, :3] = Rotation.from_rotvec([0., 0., q[1]]).as_matrix()
        return pose
    model = types.SimpleNamespace(q=np.zeros(2), limits=np.array([[-2., 2.], [-2., 2.]]),
        arm_indices=lambda arm: [1], lift_indices=lambda: [0], planners={'right': planner}, tcp=tcp)
    path = np.array([0., .01, .025, .04, .05, .04, .026, .01, 0.])
    original = {'position': (.2 * path + np.array([0., .01, -.01, .02, 0., -.01, .01, 0., 0.]))[:, None],
                'lift': path[:, None]}
    start = np.array([-.01, .02])
    result = retime(model, 'right', original, start=start)
    rows = np.c_[result['lift'], result['position']]
    np.testing.assert_allclose(rows[0], start, atol=1e-10)
    np.testing.assert_allclose(rows[-1], 0., atol=1e-10)
    assert rows[:, 0].max() >= .049  # Preserve the stroke.
    assert result['duration'] > (len(path) - 1) / 25
    np.testing.assert_array_equal(model.q, np.zeros(2))
    measured = motion(np.asarray([tcp('right', q, True) for q in rows]), 1 / 25)
    assert all(measured[key].max() <= limit for key, limit in MOTION_LIMITS.items())


def test_motion_report_records_boundary_spike_without_acceptance_gate():
    from table_1000.experts.asset_grasp.stability import motion_report
    trace = []
    for x, segment in [(0., 'relocation'), (0., 'relocation'), (.02, 'lateral'), (.02, 'lateral')]:
        pose = np.eye(4)
        pose[0, 3] = x
        trace.append({'tcp_pose': pose.tolist(), 'segment': segment})
    report = motion_report(trace, 1 / 25)
    assert report['motion_peaks']['linear_acceleration_m_s2'] > TRANSPORT['linear_acceleration_m_s2']
    assert set(report['segment_motion_peaks']) == {'relocation', 'lateral'}


def test_transport_entry_keeps_loaded_drive_targets():
    from table_1000.experts.asset_grasp import Evaluator
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.model = types.SimpleNamespace(q=np.array([.09, .19, .3]),
        arm_indices=lambda arm: [0], lift_indices=lambda: [1])
    ctx = types.SimpleNamespace(env=types.SimpleNamespace(
        targets=lambda: {'right': np.array([[.1]]), 'lift': np.array([[.2]])}))
    np.testing.assert_array_equal(evaluator.commanded_q(ctx, 'right'), [.1, .2, .3])
    np.testing.assert_array_equal(evaluator.model.q, [.09, .19, .3])


def test_transport_preflight_rejects_carried_object_collision_and_tries_other_wrist_side():
    from table_1000.experts.asset_grasp import transport_plan
    model = types.SimpleNamespace(q=np.zeros(2), limits=np.array([[-3., 3.], [-3., 3.]]),
        arm_indices=lambda arm: [0], lift_indices=lambda: [1],
        ik=lambda arm, pose, start, **kw: ('Success', np.array([
            Rotation.from_matrix(pose[:3, :3]).as_rotvec()[0], start[1]])))
    paths = []
    def check(arm, rows, lift):
        paths.append(rows)
        return len(rows), [('lift_link', 'object:case', 'case')] if np.max(rows) > .5 else []
    model.check_path = check
    commands, plan = transport_plan(model, 'right', np.eye(4))
    assert commands and plan['selected']['wrist_sign'] == -1.
    assert any(c.get('rejected', {}).get('reason') == 'transport path collision' for c in plan['choices'])
    assert any(np.max(p) > .5 for p in paths) and any(np.min(p) < -.5 for p in paths)


@pytest.mark.parametrize('angular', [False, True])
def test_toppra_stroke_uses_whole_arm_and_keeps_lift_fixed(angular):
    from table_1000.experts.asset_grasp.transport import retime_stroke
    from table_1000.experts.asset_grasp.stability import MOTION_LIMITS
    # Two joints contribute equally; neither alone reaches the TCP envelope.
    scale = 1. if angular else .5
    def tcp(arm, q, world):
        pose = np.eye(4)
        if angular:
            pose[:3, :3] = Rotation.from_rotvec([0., 0., scale * (q[1] + q[2])]).as_matrix()
        else:
            pose[2, 3] = scale * (q[1] + q[2])
        return pose
    planner = types.SimpleNamespace(joint_vel_limits=np.array([.1, .8, .8]),
                                    joint_acc_limits=np.array([.2, 1.5, 1.5]))
    model = types.SimpleNamespace(q=np.array([.3, 0., 0.]), joints=['lift', 'shoulder', 'elbow'],
        limits=np.array([[0., 1.], [-2., 2.], [-2., 2.]]),
        arm_indices=lambda arm: [1, 2], lift_indices=lambda: [0], planners={'right': planner}, tcp=tcp)
    excursion = 1. if angular else .05
    coordinate = np.linspace(0., excursion, 33)
    path = {'position': np.c_[coordinate, coordinate] / (2 * scale),
            'lift': np.full((33, 1), .3)}
    result = retime_stroke(model, 'right', path, excursion, angular=angular)
    np.testing.assert_allclose(result['lift'], .3)
    np.testing.assert_allclose(result['position'][[0, -1]], 0., atol=1e-12)
    np.testing.assert_allclose(result['position'].max(axis=0), excursion / (2 * scale))
    assert max(result['constraint_ratios']['joint_velocity']) <= 1.000001
    assert max(result['constraint_ratios']['joint_acceleration']) <= 1.000001
    assert max(result['constraint_ratios']['tcp'].values()) <= 1.000001
    key = 'angular_acceleration_rad_s2' if angular else 'linear_acceleration_m_s2'
    assert result['command_motion_peaks'][key] / MOTION_LIMITS[key] >= .9
    np.testing.assert_array_equal(model.q, [.3, 0., 0.])


def test_coupled_twist_uses_whole_arm_without_freeing_lift():
    from table_1000.experts.asset_grasp.transport import coupled_twist
    initial = np.array([.4, 0., 0.])
    def tcp(arm, q, **kwargs):
        pose = np.eye(4)
        pose[0, 3] = .2 * (q[1] + q[2])
        pose[:3, :3] = Rotation.from_rotvec([q[1], q[2], 0.]).as_matrix()
        assert q[0] == initial[0]
        return pose
    model = types.SimpleNamespace(q=initial.copy(), arm_indices=lambda arm: [1, 2],
             lift_indices=lambda: [0], tcp=tcp,
             planners={'right': types.SimpleNamespace(joint_acc_limits=np.array([.2, 1.5, 1.5]))})
    companion = coupled_twist(model, 'right', initial, np.array([1., 0., 0.]))
    np.testing.assert_allclose(companion, [2.5, 2.5, 0.], atol=1e-4)
    np.testing.assert_array_equal(model.q, initial)
