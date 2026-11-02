"""A reachable contact does not imply that the following push can be planned."""
from table_1000.experts.world import geometry as world_geometry

from types import SimpleNamespace

import numpy as np
import pytest

from table_1000.experts.skills.push.motion import Motion
from table_1000.experts.solver import PlanningFailed


def test_push_tries_another_contact_and_counts_executed_travel_only_once():
    body = np.eye(4)
    contacts, moves, allowances, ends, order = [], [], [], [], []
    solver = SimpleNamespace(
        world=SimpleNamespace(objects={}), close_gripper=lambda: None,
        move_to_pose_with_RRTConnect=lambda *a, **kw: order.append(('plan', None)),
        allow=lambda names, value=True: allowances.append(value),
        move_to_pose_with_screw=lambda goal: contacts.append(goal.copy()),
        tcp=lambda: np.eye(4),
        first_reachable=lambda options, move: move(options[0]))
    motion = Motion()
    motion.braced, motion.brace_seed = False, None
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
    motion.arms, motion.held = ['left', 'right'], {}
    motion.solvers = {'left': solver}
    motion.hands = {arm: SimpleNamespace(tip=lambda: 0.) for arm in motion.arms}
    motion.shoulder = lambda arm: np.zeros(3) if arm == 'left' else np.full(3, 10.)
    motion.stow = lambda arms=None, **kw: order.append(('stow', arms)) or False

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
    assert order[:4] == [('stow', ['right']), ('plan', None),
                         ('stow', ['right']), ('plan', None)]


def test_braced_unreachable_push_keeps_the_support_and_base_fixed():
    from table_1000.experts.context import Unreachable
    motion = Motion()
    motion.braced, motion.brace_seed = True, None
    motion.arms, motion.held = ['left'], {}
    def fail(reason, failure, *, kind, **details):
        raise failure(None, reason, **details)
    motion.ctx = SimpleNamespace(poses=lambda: {'part:body': np.eye(4)},
        spec=SimpleNamespace(base=object(), arms={'left': SimpleNamespace(finger_targets=lambda _: {})}),
        fail=fail, note=lambda *a, **kw: None)
    motion.model = SimpleNamespace(vector=lambda _: np.zeros(1),
                                   ik=lambda *a, **kw: ('IK Failed', None))
    motion.hands = {'left': SimpleNamespace(tip=lambda: 0.)}
    motion.solvers = {'left': object()}
    motion.shoulder = lambda _: np.zeros(3)
    motion.stow = lambda *a, **kw: False
    motion.make_room = lambda *a, **kw: pytest.fail('base moved during a loaded brace')
    target = SimpleNamespace(body='part:body', obj='part', offset=np.eye(4), names=['part:face'],
                             shape=SimpleNamespace(points=lambda: np.zeros((1, 3))))
    with pytest.raises(Unreachable, match='no reachable push'):
        motion.push(target, [1., 0., 0.], .1)


def test_dragging_an_object_does_not_carry_its_lower_support(monkeypatch):
    from table_1000.experts.skills.push import motion as module
    from table_1000.experts.skills import semantic
    from table_1000.experts.skills.common import SUPPORT_DISTANCE
    monkeypatch.setattr(module.fcl, 'distance', lambda *a: SimpleNamespace(min_distance=0.))
    heights = {'book': 0., 'payload': .03}
    monkeypatch.setattr(semantic, 'supported_inside', lambda ctx, obj, *args: obj == 'payload')
    monkeypatch.setattr(world_geometry, 'object_points',
                        lambda ctx, obj: np.array([[0., 0., heights[obj]]]))
    pieces = {name: SimpleNamespace(body=body, fcl=lambda _: object()) for name, body in
              [('target:body', 'target:body'), ('book:cover', 'book:body'), ('payload:body', 'payload:body')]}
    world = SimpleNamespace(objects=pieces, poses={n: np.eye(4) for n in pieces},
                            planner=SimpleNamespace(planning_world=SimpleNamespace(is_object_attached=lambda _: False)))
    skill = Motion()
    skill.solvers = {'left': SimpleNamespace(world=world, sync=lambda: None)}
    skill.ctx = SimpleNamespace(poses=lambda: {'target:body': np.eye(4)},
         geometry={'target': SimpleNamespace(points=lambda *a: np.array([[0., 0., .02]]))})
    own, riders, contacts = skill.dragged('left', ('target', 'body'))
    assert own == ['target:body']
    assert riders == ['payload:body']
    assert 'book:cover' in contacts
