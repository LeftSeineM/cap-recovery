"""Local base maneuvers must clear the edge, including the initial turn."""
from types import SimpleNamespace

import numpy as np
import pytest

from table_1000.experts.skills.base import Base, drive, drive_path
from table_1000.experts.context import ExpertFailed


def robot_at_edge():
    skill = Base()
    skill.model = SimpleNamespace(base=np.eye(4), q=np.array([.3]))
    skill.ctx = SimpleNamespace(poses=lambda: {})
    synced = []
    skill.solvers = {'arm': SimpleNamespace(sync=lambda: synced.append(True))}

    def put_base(pose, world):
        skill.pose = np.asarray(pose)
        skill.model.base[:2, 3] = skill.pose[:2]

    def clear(q):
        # A rectangular body beside an infinite tabletop edge: it fits while
        # facing the edge, but its corner sweeps into it if it turns in place.
        np.testing.assert_array_equal(q, [.3])
        x, y, yaw = skill.pose
        c, s = np.cos(yaw), np.sin(yaw)
        corners = np.array([[-.28, -.20], [-.28, .20], [.28, -.20], [.28, .20]])
        world = corners @ np.array([[c, s], [-s, c]]) + [x, y]
        return world[:, 1].max() < -.51

    skill.put_base, skill.clear = put_base, clear
    return skill, synced


def test_back_away_before_turning_without_an_extra_clearance_parameter():
    skill, synced = robot_at_edge()
    start, goal = np.array([0., -.8, np.pi / 2]), np.array([.3, -1.03, .54])
    initial = skill.model.base.copy()
    skill.put_base(start, {})
    assert skill.clear(skill.model.q)
    skill.model.base = initial.copy()
    assert any(not (skill.put_base(pose, {}) or skill.clear(skill.model.q)) for pose in drive_path(start, goal))
    skill.model.base = initial.copy()
    route = skill.base_route(start, goal)
    assert len(route) == 2
    np.testing.assert_allclose(route[0], [0., goal[1], start[2]], atol=1e-12)
    np.testing.assert_allclose(drive(start, route[0]), [0., -.23, 0.], atol=1e-12)
    previous = start
    for waypoint in route:
        for pose in drive_path(previous, waypoint):
            skill.put_base(pose, {})
            assert skill.clear(skill.model.q)
        previous = waypoint
    assert synced


def test_direct_clear_route_stays_direct_and_restores_the_planning_base():
    skill, synced = robot_at_edge()
    start, goal = np.array([0., -1.2, np.pi / 2]), np.array([.3, -1.3, .54])
    initial = skill.model.base.copy()
    route = skill.base_route(start, goal)
    assert len(route) == 1
    np.testing.assert_array_equal(route[0], goal)
    np.testing.assert_array_equal(skill.model.base, initial)
    assert synced == [True]


def test_no_route_does_not_remove_a_collision_and_restores_the_base():
    skill, synced = robot_at_edge()
    skill.clear = lambda q: False
    initial = skill.model.base.copy()
    assert skill.base_route([0., -.8, np.pi / 2], [.3, -1.03, .54]) is None
    np.testing.assert_array_equal(skill.model.base, initial)
    assert synced == [True]


def test_route_query_restores_the_world_when_collision_query_raises():
    skill, synced = robot_at_edge()
    initial = skill.model.base.copy()
    def broken(q):
        raise RuntimeError('collision query failed')
    skill.clear = broken
    with pytest.raises(RuntimeError, match='collision query failed'):
        skill.base_route([0., -.8, np.pi / 2], [.3, -1.03, .54])
    np.testing.assert_array_equal(skill.model.base, initial)
    assert synced == [True]


def test_make_room_does_not_drive_when_stowing_still_leaves_both_routes_blocked():
    skill = Base()
    actions, stowed = [], []
    def fail(reason, **details):
        raise ExpertFailed('make room', reason, **details)
    skill.ctx = SimpleNamespace(spec=SimpleNamespace(base=True), note=lambda *a, **kw: None, fail=fail)
    skill.base_for = lambda *a, **kw: np.array([.3, -.2, .5])
    skill.base_pose = lambda: np.zeros(3)
    skill.base_route = lambda *a: None
    skill.ready = lambda: stowed.append(True)
    skill.move_base = lambda *a: actions.append(a)
    with pytest.raises(ExpertFailed, match='no clear base route'):
        skill.make_room([np.eye(4)])
    assert stowed == [True]
    assert actions == []
