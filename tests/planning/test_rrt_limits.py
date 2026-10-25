"""A collision-free TOPP spline can still cross a URDF joint-position limit."""
from types import SimpleNamespace
import json

from mplib import Planner
import numpy as np

from table_1000.planning.rrt import parameterize


def wall(rows):
    for start, end in zip(rows[:-1], rows[1:]):
        count = max(1, int(np.ceil(np.abs(end - start).max() / .01)))
        points = np.linspace(start, end, count + 1)
        if np.any((points[:, 0] > .2) & (points[:, 0] < .8) & (points[:, 1] < .9)):
            return [('hand', 'wall', 'wall')]
    return []


def test_collision_free_spline_exceeding_a_joint_limit_keeps_the_original_segments():
    planner = SimpleNamespace(
        joint_vel_limits=np.ones(2), joint_acc_limits=np.ones(2),
        joint_limits=np.array([[2., 3.], [-1., 2.], [4., 5.], [0., 1.]]),
        move_group_joint_indices=[1, 3])
    planner.TOPP = lambda path, step: Planner.TOPP(planner, path, step)
    raw = np.array([[0., .5], [0., 1.], [1., 1.], [1., .5]])
    spline = planner.TOPP(raw, .04)[1]
    assert not wall(spline)
    assert spline[:, 1].max() > 1.

    result = parameterize(planner, raw, .04, wall)

    assert result['status'] == 'Success'
    assert result['timing'] == 'linear segments'
    assert not wall(result['position'])
    limits = planner.joint_limits[planner.move_group_joint_indices]
    assert np.all(result['position'] >= limits[:, 0])
    assert np.all(result['position'] <= limits[:, 1])
    np.testing.assert_allclose(result['position'][[0, -1]], raw[[0, -1]], atol=1e-10)


def test_a_raw_path_outside_joint_limits_is_not_timed_or_collision_checked():
    def forbidden(*args):
        raise AssertionError('an invalid raw path was used')
    planner = SimpleNamespace(TOPP=forbidden, joint_limits=np.array([[0., 1.], [0., 1.]]),
                              move_group_joint_indices=[0, 1])
    result = parameterize(planner, [[0., 0.], [1., 1.1]], .04, forbidden)
    assert result['status'] == result['reason'] == 'joint_limit'
    assert result['stage'] == 'raw'
    json.dumps(result)
