"""A candidate's valid pad overlap must survive the actual closing motion."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from table_1000.experts.context import ExpertFailed
from table_1000.experts.grasps import MIN_HOLD_AREA
from table_1000.experts.skills.grasping import Grasping, MAX_REGRASPS


def grasping():
    skill = Grasping()
    events, closed, attached, retracted = [], [], [], []
    poses = {'obj:body': np.eye(4)}
    def close():
        closed.append(1)
        poses['obj:body'] = np.eye(4)
        poses['obj:body'][0, 3] = .01 * len(closed)
    def fail(reason, **details):
        raise ExpertFailed('grasp', reason, **details)
    skill.ctx = SimpleNamespace(poses=lambda: poses, note=lambda name, **values: events.append((name, values)),
                                fail=fail, spec=SimpleNamespace(base=False))
    skill.solvers = {'left': SimpleNamespace(close_gripper=close, finger_gap=lambda: .02, tcp=lambda: np.eye(4),
        open_gripper=lambda gap: None, first_reachable=lambda poses: retracted.append(poses),
        allow=lambda names, allowed: None, attach=lambda name, pose: attached.append(name),
        world=SimpleNamespace(objects={'part': SimpleNamespace(body='obj:body')}))}
    skill.arms, skill.held, skill.log, skill.hands = ['left'], {}, [], {'left': object()}
    candidate = SimpleNamespace(pose=np.eye(4), open_gap=.05, width=.02, backoff=0., cost=0.,
                                info={'contact_gap': .02, 'held_mm2': 40.})
    calls = []
    def reach(target, arms, skip, *args):
        calls.append(len(skip))
        skip.add(('left', len(calls)))
        return 'left', candidate, []
    skill.reach = reach
    target = SimpleNamespace(obj='obj', body='obj:body', names=['part'], offset=np.eye(4), shape=object())
    return skill, target, events, closed, attached, retracted, calls


def test_closed_object_displacement_rejects_an_otherwise_accepted_gap():
    skill, target, events, closed, attached, retracted, calls = grasping()
    with patch('table_1000.experts.skills.grasping._held_area', side_effect=[0., MIN_HOLD_AREA]) as area:
        skill.grasp(target)
    assert len(closed) == 2 and len(retracted) == 1 and calls == [0, 1]
    assert attached == ['part'] and skill.held['left']['target'] is target
    assert events[0][0] == 'missed grasp' and events[0][1]['actual_held_mm2'] == 0.
    assert [call.args[1][0, 3] for call in area.call_args_list] == [.01, .02]


def test_closed_overlap_keeps_the_existing_regrasp_limit():
    skill, target, events, closed, attached, retracted, calls = grasping()
    with patch('table_1000.experts.skills.grasping._held_area', return_value=0.):
        with pytest.raises(ExpertFailed, match='object not held'):
            skill.grasp(target)
    assert len(closed) == MAX_REGRASPS + 1 and not attached and not skill.held
