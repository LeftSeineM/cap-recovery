"""Controlled physical proof can accept zero predicted area and reject actual slips."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from table_1000.experts.context import ExpertFailed
from table_1000.experts.skills.hold.pick import Grasping, FOLLOW_TOLERANCE


def experiment(*, gain=1., contacts=True, lost_after_probe=False):
    skill = Grasping()
    body, tcp = np.eye(4), np.eye(4)
    target = SimpleNamespace(obj='rod', body='rod:body')
    events, released, moves = [], [], []
    def fail(reason, **details):
        raise ExpertFailed('pick', reason, **details)
    skill.ctx = SimpleNamespace(poses=lambda: {'rod:body': body.copy()},
        note=lambda note, **kw: events.append((note, kw)), hold=lambda seconds: None, fail=fail)
    solver = SimpleNamespace(tcp=lambda: tcp.copy(), finger_gap=lambda: .012,
        attachments={}, attach=lambda *a: None, allow=lambda *a: None,
        world=SimpleNamespace(objects={}, allow=lambda *a: None))
    skill.solvers = {'left': solver}
    skill.held = {'left': {'target': target, 'supports': ['table'], 'carried': [],
                           'grasp': {'open_gap': .03}}}
    def move(arm, delta):
        moves.append(delta)
        tcp[:3, 3] += delta
        if not lost_after_probe or len(moves) == 1:
            body[:3, 3] += np.asarray(delta) * gain
    def pads(*args):
        touching = contacts and not (lost_after_probe and len(moves) > 1)
        return {'held': touching, 'pad_contacts': {'left': [1] if touching else [],
                                                  'right': [1] if touching else []}, 'reasons': []}
    skill.move_by_displacement = move
    skill.pad_closure = pads
    skill.release = lambda arm: (released.append(arm), skill.held.pop(arm))
    result = {'arm': 'left', 'grasp': {}, 'closure_check': {'held': True,
                'actual_held_area_m2': 0., 'area_held': False}}
    return skill, target, result, body, tcp, moves, released, events


def test_opposing_real_contacts_and_zero_predicted_area_pass_actual_lift():
    skill, target, result, body, tcp, moves, released, _ = experiment()
    skill.lift_grasp(target, result)
    assert [d[2] for d in moves] == pytest.approx([.02, .08])
    assert result['rise'] == pytest.approx(.1)
    assert result['lift_check']['held']
    assert result['closure_check']['actual_held_area_m2'] == 0.
    assert 'verified_hold' in skill.held['left'] and not released


@pytest.mark.parametrize('kwargs', [dict(gain=0.), dict(gain=.2),
                                    dict(contacts=False), dict(lost_after_probe=True)])
def test_real_empty_grip_or_slip_fails_and_cannot_return_a_held_object(kwargs):
    skill, target, result, body, tcp, moves, released, events = experiment(**kwargs)
    with pytest.raises(ExpertFailed, match='object not lifted|object slipped'):
        skill.lift_grasp(target, result)
    assert released == ['left'] and not skill.held
    assert any(note == 'missed grasp' for note, _ in events)
    if kwargs != dict(lost_after_probe=True):
        assert len(moves) == 1  # stop after the 20 mm proof, before transport


def test_hold_follows_material_at_the_grip_instead_of_the_far_end():
    skill, target, result, body, tcp, *_ = experiment()
    reference = np.eye(4)
    reference[0, 3] = .1
    # A long body turns around the TCP while its far end moves 10 cm.
    current = np.eye(4)
    current[:3, :3] = Rotation.from_euler('y', 70., degrees=True).as_matrix()
    current[:3, 3] = current[:3, :3] @ reference[:3, 3]
    body[:] = current
    assert skill.check_hold(target, 'left', reference)['held']
    body[1, 3] += FOLLOW_TOLERANCE + .001
    assert not skill.check_hold(target, 'left', reference)['held']


def test_actual_collision_surface_penetration_is_still_a_hard_failure():
    skill, target, result, *_ = experiment()
    skill.pad_closure = lambda *a: {'pad_contacts': {'left': [1], 'right': [1]},
                                  'reasons': ['closed past collision surfaces']}
    with pytest.raises(ExpertFailed, match='object slipped during lift'):
        skill.lift_grasp(target, result)
    assert not skill.held
