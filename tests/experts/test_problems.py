"""Numerical solve records and reconstruction, without Isaac."""
from types import SimpleNamespace

import numpy as np

from table_1000.experts.problems import Offline, Problems, decision, load, mark
from table_1000.experts.solver import MotionSolver
from table_1000.experts.trace import decode
from table_1000.robots import load_spec
from table_1000.robots.kinematics import RobotModel


def test_flat_put_down_records_and_restores_native_release_hulls(tmp_path):
    """Problems recording must not abort flat Place before its first motion."""
    from table_1000.experts.skills.flat_release import FlatReleaseGeometry
    from table_1000.experts.skills.placing import Placing, convex_keepout
    ctx = SimpleNamespace(spec=SimpleNamespace(name='fake'), trace=SimpleNamespace(stack=[4]))
    record = Problems(ctx)
    ctx.trace.problems = record
    points = np.array([[x, y, z] for x in (.015, .025) for y in (-.01, .01) for z in (0., .2)])
    wall = convex_keepout(points)
    flat = FlatReleaseGeometry([('drawer:wall', wall, (points.min(0), points.max(0)))], .05,
                               ceiling=.09)
    def frames(gap):
        pose = np.eye(4)
        pose[0, 3] = gap
        return {'finger': pose}
    hand = SimpleNamespace(shapes={'finger': [(np.eye(4), 'box', (.01, .01, .01), True)]}, frames=frames)
    # Also exercise a populated, native-object cache: identity keys are local
    # to this process and must be rebuilt, rather than included in the record.
    assert flat.minimum(hand, np.eye(4), np.eye(4), 0., .04, .08) is None
    assert flat.channels
    skill = object.__new__(Placing)
    skill.ctx, skill.hands = ctx, {'left': hand}
    skill.solvers = {'left': SimpleNamespace(finger_gap=lambda: 0.)}
    skill.held = {'left': {'grasp': {'open_gap': .04}}}
    options = [(np.eye(4), np.zeros(3))]
    result = skill.put_down('left', options, np.eye(4), .01, .08, 'flat', flat)
    assert not result['done'] and result['blocked_by'] == ['drawer:wall']
    path = tmp_path / 'flat-problems.json.gz'
    record.save(path)
    restored = decode(load(path)['decisions'][0]['arguments'])['flat_geometry']
    assert restored.ceiling == flat.ceiling
    assert not restored.channels and not restored.heights
    restored_wall = restored.obstacles[0][1]
    np.testing.assert_array_equal(restored_wall.pose.to_transformation_matrix(), wall.pose.to_transformation_matrix())
    original_shape = wall.shapes[0].get_collision_geometry()
    restored_shape = restored_wall.shapes[0].get_collision_geometry()
    np.testing.assert_array_equal(restored_shape.get_vertices(), original_shape.get_vertices())
    np.testing.assert_array_equal(restored_shape.get_faces(), original_shape.get_faces())
    assert restored.minimum(hand, np.eye(4), np.eye(4), 0., .04, .08) is None
    skill.ctx.trace.problems = None
    assert skill.put_down('left', options, np.eye(4), .01, .08, 'flat', restored) == result


def numerical_context(tmp_path):
    model = RobotModel(load_spec('x1pro_rm001'), tmp_path / 'recording')
    model.set_state(model.spec.keyframe('ready'))
    ctx = SimpleNamespace(model=model, spec=model.spec, solvers={}, steps=0, trace=SimpleNamespace(stack=[7]))
    for arm in model.spec.arms:
        ctx.solvers[arm] = MotionSolver(ctx, arm)
    ctx.trace.problems = Problems(ctx)
    return ctx


def test_world_roundtrip_preserves_collision_geometry_attachment_contacts_and_frame(tmp_path):
    ctx = numerical_context(tmp_path)
    solver = ctx.solvers['left']
    base = np.eye(4)
    base[:3, 3] = [.2, -.3, .1]
    ctx.model.base = base
    solver.world.set_frame(np.linalg.inv(base))
    position = np.eye(4)
    position[:3, 3] = [.8, .2, .7]
    solver.add_box('support', [.2, .3, .4], position)
    solver.add_box('held', [.01, .02, .03], position)
    solver.attach_at('held', np.eye(4))
    solver.world.set_padding(.01)
    solver.allow(['support'])
    borrowed = frozenset(('held', 'right_arm_gripper_left_link'))
    overlapping = frozenset(('support', 'right_arm_gripper_left_link'))
    solver.world.allow(['support'], ['right_arm_gripper_left_link'])
    solver.world.set_shared_permissions({borrowed, overlapping})
    record = ctx.trace.problems
    with record.operation('ik', 'left', dict(target=solver.tcp(), world=True)) as entry:
        entry['result'] = {'status': 'Success'}
    path = tmp_path / 'problems.json.gz'
    record.save(path)
    document = load(path)
    resolver = Offline(document, tmp_path / 'offline')
    restored = resolver.restore(document['operations'][0])
    np.testing.assert_array_equal(resolver.model.q, ctx.model.q)
    np.testing.assert_array_equal(resolver.model.base, base)
    assert restored.world.padding == .01
    assert restored.world.object_padding == {'support': .01, 'held': 0.}
    assert restored.world.permissions == solver.world.permissions
    assert restored.world.shared_permissions == {borrowed, overlapping}
    restored.world.set_shared_permissions(())
    assert borrowed not in restored.world.permissions
    assert overlapping in restored.world.permissions
    restored = resolver.restore(document['operations'][0])
    assert restored.planner.planning_world.is_object_attached('held')
    np.testing.assert_array_equal(restored.attachments['held'], np.eye(4))
    np.testing.assert_array_equal(restored.world.poses['support'], position)
    np.testing.assert_array_equal(restored.world.frame, np.linalg.inv(base))
    assert restored.world.objects['support'].raw[0][1] == 'box'
    np.testing.assert_array_equal(restored.world.objects['support'].raw[0][2], [.2, .3, .4])
    # Restoring again must clear attachments and contact allowances, including link/link pairs.
    restored.world.allow(['left_arm_link1'], ['right_arm_link1'])
    resolver.restore(document['operations'][0])
    assert restored.world.permissions == solver.world.permissions


def test_observer_keeps_rng_stream_nested_calls_and_shared_worlds(tmp_path):
    import mplib.pymp
    ctx = numerical_context(tmp_path)
    model, record = ctx.model, ctx.trace.problems
    arm = 'left'
    target = model.tcp(arm, world=True)
    unreachable = target.copy()
    unreachable[:3, 3] = [100., 100., 100.]
    mplib.pymp.set_global_seed(123)
    model.ik(arm, unreachable, world=True)
    expected = model.ik(arm, target, world=True)
    record.install()
    record.events.append({'seed': 123, 'frame': 7})
    mplib.pymp.set_global_seed(123)
    try:
        model.ik(arm, unreachable, world=True)
        actual = model.ik(arm, target, world=True)
        model.ik(arm, target, world=True)
        model.plan_rrt(arm, target, world=True)
    finally:
        record.close()
    assert expected[0] == actual[0] == 'Success'
    np.testing.assert_array_equal(expected[1], actual[1])
    assert len(record.worlds) <= 3
    assert len(record.operations) == 5
    assert record.operations[-1]['parent'] == 3
    assert record.operations[0]['diagnostic']['without_collision'] != 'Success'
    resolver = Offline(record.document, tmp_path / 'offline')
    results = resolver.replay([0, 1, 2, 3])
    assert all(item['same_status'] for item in results)
    np.testing.assert_allclose(decode(results[1]['result']['q']), actual[1])


def test_generator_decisions_release_scope_at_yield_and_keep_candidate_ancestry():
    ctx = SimpleNamespace(spec=SimpleNamespace(name='fake'), trace=SimpleNamespace(stack=[4]))
    record = Problems(ctx)
    ctx.trace.problems = record
    class Skill:
        def __init__(self):
            self.ctx = ctx
        @decision
        def options(self, skip):
            mark(self, 'grasp IK', 'left/0/False', pose=np.eye(4))
            yield 3
        @decision
        def reach(self):
            for option in self.options({('left', 1)}):
                assert record.decisions[record.stack[-1]]['name'].endswith('.reach')
                mark(self, 'plan', option)
                return option
    assert Skill().reach() == 3
    assert record.stack == []
    assert [item['parent'] for item in record.decisions] == [None, 0]
    assert record.decisions[1]['events'][0]['candidate'] == 'left/0/False'
    assert record.decisions[0]['events'][0]['stage'] == 'plan'


def test_redecide_uses_expert_placement_filters_and_records_unvisited_candidates(tmp_path, monkeypatch):
    """An originally unreachable release stays rejected; the full search reports failure."""
    from table_1000.experts.skills import Skills
    ctx = numerical_context(tmp_path)
    ctx.poses = lambda: {}
    ctx.rng = np.random.default_rng(3)
    skill = object.__new__(Skills)
    skill.ctx, skill.model, skill.solvers = ctx, ctx.model, ctx.solvers
    skill.held, skill.reaches = {}, {}
    skill.home, skill.start_base = ctx.model.q.copy(), np.zeros(3)
    ctx._skills = skill
    placed = np.eye(4)
    placed[:3, 3] = [100., 100., 100.]
    ctx.trace.problems.events.append({'seed': 19, 'frame': 7})
    ctx.trace.problems.install()
    try:
        original = skill.put_down('left', [(placed, placed[:3, 3])], np.eye(4), .01, .08)
    finally:
        ctx.trace.problems.close()
    assert not original['done'] and original['ik_checked'] == 2
    document = ctx.trace.problems.document
    offline = Offline(document, tmp_path / 'offline')
    import mplib.pymp
    set_seed, seeds = mplib.pymp.set_global_seed, []
    def reseed(value):
        seeds.append(value)
        set_seed(value)
    monkeypatch.setattr(mplib.pymp, 'set_global_seed', reseed)
    replayed = offline.decide(0, seed=20)
    assert not replayed['success']
    assert replayed['result']['ik_checked'] == 2
    assert len(replayed['problems']['operations']) == 2
    assert all(op['stage'] == 'release IK' for op in replayed['problems']['operations'])
    another = offline.decide(0, seed=21)
    assert seeds == [replayed['mplib_seed'], another['mplib_seed']]
    assert seeds[0] != seeds[1]  # Adjacent seeds must not collapse when making them odd.


def test_interrupted_follow_records_measured_error_and_propagates_failure():
    import pytest
    from table_1000.experts.context import ExpertFailed
    record = SimpleNamespace(operations=[{}])
    error = ExpertFailed('move', 'unexpected contact')
    def step(action):
        raise error
    solver = object.__new__(MotionSolver)
    solver.ctx = SimpleNamespace(step=step, trace=SimpleNamespace(problems=record))
    solver.arm = SimpleNamespace(name='left')
    solver.qpos = lambda lift=False: np.array([.5])
    solver.log = [{'problem': 0}]
    with pytest.raises(ExpertFailed) as caught:
        solver.follow({'position': np.array([[1.], [2.]])})
    assert caught.value is error
    execution = record.operations[0]['execution']
    assert execution['max_tracking_error_rad'] == .5
    assert execution['final_error_rad'] == 1.5
    assert execution['wall_s'] >= 0.
