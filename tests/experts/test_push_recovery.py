"""A reachable contact does not imply that the following push can be planned."""
from types import SimpleNamespace

import numpy as np
import pytest

from table_1000.experts.skills.motion import Motion
from table_1000.experts.solver import PlanningFailed


def test_push_tries_another_contact_and_counts_executed_travel_only_once():
    body = np.eye(4)
    contacts, moves, allowances, ends = [], [], [], []
    solver = SimpleNamespace(
        world=SimpleNamespace(objects={}), close_gripper=lambda: None,
        move_to_pose_with_RRTConnect=lambda *a, **kw: None,
        allow=lambda names, value=True: allowances.append(value),
        move_to_pose_with_screw=lambda goal: contacts.append(goal.copy()),
        tcp=lambda: np.eye(4),
        first_reachable=lambda options, move: move(options[0]))
    motion = Motion()
    motion.ctx = SimpleNamespace(poses=lambda: {'part:body': body.copy()},
        spec=SimpleNamespace(arms={'left': SimpleNamespace(finger_targets=lambda gap: {})}, base=None),
        note=lambda *a, **kw: None)
    def ik(arm, goal, start, **kw):
        if kw.get('check_collision') is False:
            ends.append(goal[0, 3])
            # The original end (0.1 m) is reachable; anything further is not.
            return ('Success' if goal[0, 3] <= .1 + 1e-9 else 'IK Failed', None)
        return 'Success', np.zeros(1)

    motion.model = SimpleNamespace(vector=lambda _: np.zeros(1), ik=ik)
    motion.arms, motion.held = ['left'], {}
    motion.solvers = {'left': solver}
    motion.hands = {'left': SimpleNamespace(tip=lambda: 0.)}
    motion.shoulder = lambda _: np.zeros(3)
    motion.stow = lambda: False

    def push(arm, displacement, drag):
        moves.append(displacement.copy())
        body[0, 3] += .04 if len(moves) == 1 else displacement[0]
        if len(moves) == 1:
            raise PlanningFailed('push', 'joint_limit')

    motion.move_by_displacement = push
    target = SimpleNamespace(body='part:body', obj='part', offset=np.eye(4), names=['part:face'],
                             shape=SimpleNamespace(points=lambda: np.zeros((1, 3))))

    result = motion.push(target, [1., 0., 0.], .1)

    assert result['moved'] == pytest.approx(.1)
    np.testing.assert_allclose(moves, [[.1, 0., 0.], [.06, 0., 0.]])
    np.testing.assert_allclose(contacts[1][:3, 3], [.04, 0., 0.])
    np.testing.assert_allclose(ends[:2], [.1, .1])
    assert allowances[:4] == [True, False, True, False]
