"""AABB rejection preserves exact FCL results as the live world changes."""

from types import SimpleNamespace

import numpy as np
import pytest

from table_1000.planning.broadphase import BroadphaseWorld
from table_1000.planning.world import PlanningScene, box_object, pose


@pytest.fixture
def arm_model(tmp_path):
    import mplib
    from test_model_world import robot
    from table_1000.robots.planning_model import write_srdf
    source = robot(tmp_path)
    write_srdf(source, tmp_path / 'robot.srdf')
    planner = mplib.Planner(str(source), 'l3', srdf=str(tmp_path / 'robot.srdf'))
    planner.planning_world = BroadphaseWorld(planner.planning_world)
    q = np.array([.2, .2, -.3])
    planner.robot.set_qpos(q, True)
    from table_1000.planning.query_pairs import QueryPairs
    return SimpleNamespace(planners={'arm': planner}, q=q, limits=np.asarray(planner.joint_limits),
                           reduction=QueryPairs(planner, source),
                           spec=SimpleNamespace(arms={'arm': SimpleNamespace(finger_links=('l3',))}))


def pairs(results):
    return sorted((r.collision_type, r.object_name1, r.object_name2,
                   r.link_name1, r.link_name2, r.res.num_contacts()) for r in results)


def test_random_states_attachments_and_live_permissions(arm_model):
    planner = arm_model.planners['arm']
    fast = planner.planning_world
    native = fast.world
    ready = arm_model.q.copy()
    names = ['broadphase.a', 'broadphase.b']
    try:
        for i, name in enumerate(names):
            frame = np.eye(4)
            frame[:3, 3] = [.25 + i * .15, 0., .3]
            obj, _ = box_object(name, (.15, .15, .15), frame)
            native.add_object(obj.fcl(frame))
        rng = np.random.default_rng(43)
        for mode in range(4):
            if mode == 1:
                native.attach_object(names[0], planner.robot.get_name(), planner.move_group_link_id,
                                     pose(np.eye(4)), list(arm_model.spec.arms['arm'].finger_links))
            elif mode == 2:
                native.get_allowed_collision_matrix().set_entry(names[0], names[1], True)
            elif mode == 3:
                native.detach_object(names[0], False)
            for _ in range(20):
                q = rng.uniform(arm_model.limits[:, 0], arm_model.limits[:, 1])
                planner.robot.set_qpos(q, True)
                assert pairs(fast.check_collision()) == pairs(native.check_collision())
    finally:
        for name in names:
            native.remove_object(name)
        planner.robot.set_qpos(ready, True)


def test_external_object_replacement_and_pose_changes(arm_model):
    planner = arm_model.planners['arm']
    fast = planner.planning_world
    native = fast.world
    name = 'broadphase.live'
    try:
        for distance in (0., 4., 1e-7, 4.):
            frame = np.eye(4)
            frame[:3, 3] = [distance, 0., .2]
            obj, _ = box_object(name, (.3, .3, .3), frame)
            native.add_object(obj.fcl(frame))
            assert pairs(fast.check_collision()) == pairs(native.check_collision())
    finally:
        native.remove_object(name)


def test_explicit_requests_are_delegated():
    calls = []
    native = SimpleNamespace(get_planned_articulations=lambda: [],
                             check_collision=lambda *args: calls.append(args) or ['native'])
    fast = BroadphaseWorld(native)
    request = object()
    assert fast.check_collision(request) == ['native']
    assert calls == [(request,)]


def test_incremental_geometry_and_pair_caches(arm_model, monkeypatch):
    from unittest.mock import Mock
    planner = arm_model.planners['arm']
    fast = planner.planning_world
    native = fast.world
    scene = PlanningScene(planner)
    frame = np.eye(4)
    frame[:3, 3] = [.3, 0., .3]
    scene.add(*box_object('moving', (.15, .15, .15), frame))
    scene.add(*box_object('fixed', (.2, .2, .2), frame))
    build_bounds = Mock(wraps=fast._shape_bounds)
    build_pairs = Mock(wraps=fast._prepare_pairs)
    monkeypatch.setattr(fast, '_shape_bounds', build_bounds)
    monkeypatch.setattr(fast, '_prepare_pairs', build_pairs)

    def check():
        assert pairs(fast.check_collision()) == pairs(native.check_collision())

    check()
    initial_bounds = build_bounds.call_count
    initial_indices = fast.base_indices
    assert build_pairs.call_count == 1
    fixed_bounds = fast.bounds['object', 'fixed']
    for distance in (4., 0., 1e-7):
        moved = frame.copy()
        moved[0, 3] = distance
        scene.set_pose('moving', moved)
        check()
        assert fast.bounds['object', 'fixed'] is fixed_bounds
        assert build_bounds.call_count == initial_bounds
        assert build_pairs.call_count == 1

    scene.set_padding(.01)
    check()
    assert build_bounds.call_count == initial_bounds + 2
    assert build_pairs.call_count == 1
    for padding in (0., .01, 0.):
        scene.set_padding(padding)
        check()
    assert build_bounds.call_count == initial_bounds + 2
    assert build_pairs.call_count == 1

    for allowed in (True, False):
        scene.allow(['moving'], ['l1', 'l2', 'l3'], allowed)
        check()
    assert build_bounds.call_count == initial_bounds + 2
    assert build_pairs.call_count == 1

    for name in ('moving', 'fixed'):
        native.attach_object(name, planner.robot.get_name(), planner.move_group_link_id,
                             pose(np.eye(4)), ['l3'])
        check()
        assert fast.base_indices is initial_indices
    assert build_bounds.call_count == initial_bounds + 2
    assert build_pairs.call_count == 3
    native.detach_object('fixed', False)
    check()
    assert fast.base_indices is initial_indices
    scene.set_pose('moving', frame)  # Scene update also handles direct attachment removal.
    check()
    assert build_bounds.call_count == initial_bounds + 2
    assert build_pairs.call_count == 5

    before_add = fast.base_indices
    scene.add(*box_object('added', (.11, .12, .13), frame))
    check()
    assert build_bounds.call_count == initial_bounds + 3
    assert build_pairs.call_count == 6
    assert fast.base_indices is not before_add
    native.remove_object('added')
    check()
    assert ('object', 'added') not in fast.bounds
    assert build_bounds.call_count == initial_bounds + 3
    assert build_pairs.call_count == 7
    scene.clear()
    check()
    assert build_pairs.call_count == 8


def test_replacement_reuses_only_identical_geometry_and_shape_poses(arm_model):
    planner = arm_model.planners['arm']
    fast = planner.planning_world
    from mplib.collision_detection import fcl
    shape = fcl.Box(.12, .12, .12)
    frame = np.eye(4)
    frame[:3, 3] = [0., 0., .2]
    for offset, geom in ((0., shape), (3., shape), (0., shape), (0., fcl.Box(.9, .9, .9))):
        relative = np.eye(4)
        relative[0, 3] = offset
        obj = fcl.FCLObject('replace', pose(frame), [fcl.CollisionObject(geom)], [pose(relative)])
        fast.world.add_object(obj)
        assert pairs(fast.check_collision()) == pairs(fast.world.check_collision())
    fast.world.remove_object('replace')
    scene = PlanningScene(planner)
    frame[0, 3] = 3.
    for width in (.1, 8., .1):
        scene.add(*box_object('replace', (width, width, width), frame))
        actual = fast.check_collision()
        assert pairs(actual) == pairs(fast.world.check_collision())
        assert any(r.object_name2 == 'replace' for r in actual) == (width == 8.)
    scene.clear()


@pytest.mark.parametrize('fixed_prismatic', [False, True])
@pytest.mark.parametrize('held', [False, True])
def test_native_query_reduction_preserves_every_collision_and_restores_acm(arm_model, fixed_prismatic, held):
    from mplib.planning.ompl import FixedJoint
    planner = arm_model.planners['arm']
    native = planner.planning_world.world
    names = ['query.near', 'query.far', 'query.held']
    for name, position in zip(names, ([.3, 0., .4], [8., 8., 8.], [.2, 0., .5])):
        frame = np.eye(4)
        frame[:3, 3] = position
        obj, _ = box_object(name, (.12, .12, .12), frame)
        native.add_object(obj.fcl(frame))
    try:
        if held:
            native.attach_object(names[2], planner.robot.get_name(), planner.move_group_link_id,
                                 pose(np.eye(4)), ['l3'])
        matrix = native.get_allowed_collision_matrix()
        matrix.set_entry('base', names[1], False)
        matrix.set_default_entry(names[0], False)
        before = str(matrix)
        rng = np.random.default_rng(21)
        states = rng.uniform(arm_model.limits[:, 0], arm_model.limits[:, 1], (40, 3))
        fixed = {FixedJoint(0, 1, float(arm_model.q[1]))} if fixed_prismatic else set()
        if fixed_prismatic:
            states[:, 1] = arm_model.q[1]
        expected = []
        for q in states:
            planner.robot.set_qpos(q, True)
            expected.append(pairs(native.check_collision()))
        with arm_model.reduction.reduced(arm_model.q, fixed) as removed:
            assert removed
            assert all(second in names for _, second, _ in removed)
            for q, expected_pairs in zip(states, expected):
                planner.robot.set_qpos(q, True)
                assert pairs(native.check_collision()) == expected_pairs
        assert str(matrix) == before
        with pytest.raises(RuntimeError, match='stop'):
            with arm_model.reduction.reduced(arm_model.q, fixed):
                raise RuntimeError('stop')
        assert str(matrix) == before
    finally:
        for name in names:
            native.remove_object(name)
