"""Kinematics and self-collision planning of a fixed single arm (no simulator)."""

import numpy as np
import pytest

from table_1000.robots.kinematics import pose_error


def free_samples(model, arm, count, seed=0):
    rng = np.random.default_rng(seed)
    indices = model.arm_indices(arm)
    low, high = model.limits[indices].T
    samples = []
    while len(samples) < count:
        q = model.q.copy()
        q[indices] = rng.uniform(low, high)
        if not model.self_collisions(q):
            samples.append(q)
    return samples


def test_joints_and_keyframes(arm_model):
    spec = arm_model.spec
    assert sorted(arm_model.joints) == sorted(spec.body_joints)
    assert spec.base is None and spec.keyframe_names == ['zero', 'ready']
    for name in spec.keyframe_names:
        assert arm_model.self_collisions(arm_model.vector(spec.keyframe(name))) == []


def test_ik_round_trip(arm_model):
    """IK reaches the TCP poses of random collision-free configurations."""
    samples = free_samples(arm_model, 'arm', 40, seed=2)
    solved = 0
    for q in samples:
        target = arm_model.tcp('arm', q)
        status, solution = arm_model.ik('arm', target)
        if status == 'Success':
            solved += 1
            position, rotation = pose_error(arm_model.tcp('arm', solution), target)
            assert position < 1e-3 and rotation < 1e-2
    assert solved / len(samples) >= .9


def test_chain_ik_keeps_fingers_and_matches_full_model(arm_model):
    """Removing unrelated branches keeps FK, masked joints and convergent solutions."""
    from table_1000.planning.world import pose
    solver = arm_model.planners['arm'].pinocchio_model
    assert len(solver.indices) == len(arm_model.arm_indices('arm')) < len(arm_model.joints)
    mask = np.ones(len(arm_model.joints), dtype=bool)
    mask[arm_model.arm_indices('arm')] = False
    for q in free_samples(arm_model, 'arm', 12, seed=19):
        target = pose(arm_model.tcp('arm', q))
        start = q.copy()
        start[~mask] += .01
        full, full_ok, _ = solver.full.compute_IK_CLIK(solver.full_link, target, start, mask)
        chain, chain_ok, _ = solver.compute_IK_CLIK(solver.full_link, target, start, mask)
        assert full_ok and chain_ok
        np.testing.assert_array_equal(chain[mask], start[mask])
        np.testing.assert_allclose(chain, full, atol=1e-7)


def test_world_targets_follow_the_base_pose(arm_model):
    base = np.eye(4)
    base[:3, 3] = [1., 2., .7]
    local = arm_model.tcp('arm')
    arm_model.set_state(base=base)
    try:
        np.testing.assert_allclose(arm_model.tcp('arm', world=True), base @ local)
    finally:
        arm_model.set_state(base=np.eye(4))


def test_rrt_moves_only_the_arm(arm_model):
    goal = free_samples(arm_model, 'arm', 1, seed=3)[0]
    result = arm_model.plan_rrt('arm', None, goals=[goal])
    assert result['status'] == 'Success'
    assert result['position'].shape[1] == 6
    np.testing.assert_allclose(result['position'][-1], goal[arm_model.arm_indices('arm')], atol=1e-3)
    for row in result['position']:
        assert arm_model.self_collisions(arm_model.vector(dict(zip(result['joints'], row)))) == []


def test_native_rrt_fixed_lift_returns_full_move_group_rows(arm_model, arm_root, tmp_path):
    """Native OMPL restores fixed dimensions before the shortcut checks the world."""
    from dataclasses import replace
    from types import SimpleNamespace
    import xml.etree.ElementTree as ET
    from table_1000.robots.kinematics import RobotModel
    root = ET.parse(arm_root / 'robot.urdf').getroot()
    root.find("joint[@name='joint1']/parent").set('link', 'lift_link')
    ET.SubElement(root, 'link', name='lift_link')
    joint = ET.SubElement(root, 'joint', name='lift', type='prismatic')
    ET.SubElement(joint, 'parent', link='base_link')
    ET.SubElement(joint, 'child', link='lift_link')
    ET.SubElement(joint, 'axis', xyz='0 0 1')
    ET.SubElement(joint, 'limit', lower='0', upper='.4', effort='50', velocity='.2')
    ET.ElementTree(root).write(tmp_path / 'robot.urdf')
    spec = replace(arm_model.spec, lift=('lift',))
    model = RobotModel(spec, tmp_path / 'planning', root=tmp_path)
    model.set_state({**spec.keyframe('ready'), 'lift': .12})
    goal = free_samples(model, 'arm', 1, seed=3)[0]
    adapter = model.planners['arm'].planner
    world, checked = adapter.world, []

    def set_qpos(q):
        checked.append(np.array(q))
        world.set_qpos_all(q)

    adapter.world = SimpleNamespace(set_qpos_all=set_qpos, check_collision=world.check_collision)
    result = model.plan_rrt('arm', None, goals=[goal])
    assert result['status'] == 'Success' and checked
    assert all(len(q) == 7 and q[0] == pytest.approx(.12) for q in checked)
    assert result['position'].shape[1] == 6


def test_screw_is_straight(arm_model):
    start = arm_model.tcp('arm')
    target = start.copy()
    target[:3, 3] += [.05, 0., -.04]
    result = arm_model.plan_screw('arm', target)
    assert result['status'] == 'Success'
    direction = (target[:3, 3] - start[:3, 3]) / np.linalg.norm(target[:3, 3] - start[:3, 3])
    deviations = []
    for row in result['position']:
        point = arm_model.tcp('arm', arm_model.vector(dict(zip(result['joints'], row))))[:3, 3] - start[:3, 3]
        deviations.append(np.linalg.norm(point - (point @ direction) * direction))
    assert max(deviations) < 5e-3


def test_look_at_needs_a_head(arm_model):
    with pytest.raises(Exception):
        arm_model.look_at(np.zeros(3), 'link6')


def test_check_path_catches_a_collision_between_waypoints(arm_model):
    """Two collision-free waypoints whose straight joint path sweeps the hand through an obstacle."""
    from table_1000.planning.world import PlanningScene, box_object
    middle = arm_model.vector(arm_model.spec.keyframe('ready'))
    joint = arm_model.arm_indices('arm')[0]
    start, end = middle.copy(), middle.copy()
    start[joint], end[joint] = -1., 1.
    scene = PlanningScene(arm_model.planners['arm'])
    scene.add(*box_object('obstacle', (.025, .025, .025), arm_model.fk('link6', middle)))
    try:
        indices = arm_model.arm_indices('arm')
        assert arm_model.check_path('arm', [start[indices]])[1] == []
        assert arm_model.check_path('arm', [end[indices]])[1] == []
        checks, pairs = arm_model.check_path('arm', [start[indices], end[indices]])
        assert checks > 2 and any('obstacle' in pair for pair in pairs)
    finally:
        scene.clear()


def test_path_contact_permission_expires_for_each_pair_separately():
    from types import SimpleNamespace
    from table_1000.robots.kinematics import RobotModel
    state = {}
    first = ('finger_left', 'drawer', 'front')
    second = ('finger_right', 'drawer', 'front')

    def collisions():
        x = state['q'][0]
        pairs = {second} if x < .7 else set()
        if x < .1 or .3 < x < .5:
            pairs.add(first)
        return [SimpleNamespace(link_name1=a, object_name2=b, link_name2=c) for a, b, c in pairs]

    world = SimpleNamespace(set_qpos_all=lambda q: state.update(q=q), check_collision=collisions)
    model = SimpleNamespace(q=np.zeros(1), lift_indices=lambda: [],
                            planners={'arm': SimpleNamespace(planning_world=world,
                                       robot=SimpleNamespace(set_qpos=lambda *a: None))})
    checks, pairs = RobotModel.check_path(model, 'arm', [[0.], [1.]])
    assert checks > 2 and first in pairs
