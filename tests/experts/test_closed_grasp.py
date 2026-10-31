"""A candidate's valid pad overlap must survive the actual closing motion."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from table_1000.experts.context import ExpertFailed
from table_1000.experts.skills.common import Target
from table_1000.experts.skills.hold.pick import Grasping, MAX_REGRASPS


def grasping():
    skill = Grasping()
    skill.constrained = lambda target: False
    skill.dragged = lambda *args: ([], [], [])
    events, closed, attached, retracted = [], [], [], []
    poses = {'obj:body': np.eye(4)}
    def close():
        closed.append(1)
        poses['obj:body'] = np.eye(4)
        poses['obj:body'][0, 3] = .01 * len(closed)
    def fail(reason, **details):
        raise ExpertFailed('grasp', reason, **details)
    skill.ctx = SimpleNamespace(poses=lambda: poses, note=lambda name, **values: events.append((name, values)),
                                fail=fail, geometry={}, spec=SimpleNamespace(base=False))
    skill.solvers = {'left': SimpleNamespace(close_gripper=close, finger_gap=lambda: .02, tcp=lambda: np.eye(4), carry=lambda *args: None,
        open_gripper=lambda gap: None, first_reachable=lambda poses: retracted.append(poses),
        allow=lambda names, allowed: None, attach=lambda name, pose: attached.append(name),
        world=SimpleNamespace(objects={'part': SimpleNamespace(body='obj:body')}, permissions=set(), allow=lambda *args: None))}
    skill.arms, skill.held, skill.log, skill.hands = ['left'], {}, [], {'left': object()}
    candidate = SimpleNamespace(pose=np.eye(4), open_gap=.05, width=.02, backoff=0., cost=0.,
                                info={'contact_gap': .02, 'held_mm2': 40.}, target=None, choice=None)
    calls = []
    def reach(target, arms, skip, *args, **kwargs):
        calls.append(len(skip))
        skip.add(('left', len(calls)))
        return 'left', candidate, []
    skill.reach = reach
    target = Target('obj', ['part'], 'obj:body', object(), np.eye(4), np.empty((0, 3)))
    return skill, target, events, closed, attached, retracted, calls


def test_closed_object_displacement_rejects_an_otherwise_accepted_gap():
    skill, target, events, closed, attached, retracted, calls = grasping()
    measured = []
    def check(*args):
        measured.append(skill.ctx.poses()['obj:body'][0, 3])
        return {'held': len(measured) == 2}
    with patch.object(skill, 'check_closure', side_effect=check):
        skill.grasp(target)
    assert len(closed) == 2 and len(retracted) == 1 and calls == [0, 1]
    assert attached == ['part'] and skill.held['left']['target'] is target
    missed = [values for name, values in events if name == 'missed grasp']
    assert missed[0]['closure']['held'] is False
    assert measured == [.01, .02]


def test_closed_overlap_keeps_the_existing_regrasp_limit():
    skill, target, events, closed, attached, retracted, calls = grasping()
    with patch.object(skill, 'check_closure', return_value={'held': False}):
        with pytest.raises(ExpertFailed, match='object not held'):
            skill.grasp(target)
    assert len(closed) == MAX_REGRASPS + 1 and not attached and not skill.held


def test_pick_plans_with_the_measured_post_lift_hold_and_payload():
    from scipy.spatial.transform import Rotation
    skill, target, *_ = grasping()
    poses = skill.ctx.poses()
    poses['rider:body'] = np.eye(4)
    solver = skill.solvers['left']
    solver.attachments = {'part': np.eye(4), 'rider': np.eye(4)}
    solver.world.objects['rider'] = SimpleNamespace(body='rider:body')
    solver.attach = lambda name, pose: solver.attachments.__setitem__(name, pose.copy())
    skill.thin_geometry = lambda target: None
    skill.grasp = lambda *args, **kwargs: {'arm': 'left'}
    skill.held['left'] = {'target': target}
    skill.ctx.hold = lambda seconds: None
    def lift(arm, displacement):
        measured = np.eye(4)
        measured[:3, :3] = Rotation.from_euler('y', 56., degrees=True).as_matrix()
        measured[:3, 3] = [.03, 0., .1]
        poses['obj:body'] = measured
        poses['rider:body'] = measured.copy()
        poses['rider:body'][1, 3] += .02
    skill.move_by_displacement = lift
    result = skill.pick(target)
    np.testing.assert_allclose(solver.attachments['part'], result['tcp_object'])
    np.testing.assert_allclose(solver.attachments['part'], poses['obj:body'])
    np.testing.assert_allclose(solver.attachments['rider'], poses['rider:body'])


@pytest.mark.parametrize('selection', ['rim', 'container part'])
def test_free_body_carries_adjacent_parts_without_revoking_their_finger_contact(selection):
    skill, target, events, closed, attached, retracted, calls = grasping()
    target.rim = selection == 'rim'
    target.container = selection == 'container part'
    skill.ctx.geometry = {'obj': object()}
    skill.dragged = lambda *args: (['obj:cap'], ['rider:body'], ['table'])
    solver = skill.solvers['left']
    solver.world.objects.update({'obj:cap': SimpleNamespace(body='obj:body'),
                                 'rider:body': SimpleNamespace(body='rider:body')})
    candidate = SimpleNamespace(pose=np.eye(4), open_gap=.05, width=.02, backoff=0., cost=0.,
                               info={'contact_gap': .02, 'held_mm2': 40.}, target=None, choice=None)
    skill.rejected = dict.fromkeys(('grasp IK', 'pre-grasp IK', 'then', 'need'), 0)
    skill.reach = lambda *args, **kwargs: ('left', candidate, ['table', 'obj:cap', 'rider:body'])
    carried, permissions = [], []
    solver.carry = lambda names, contacts: carried.append((names, contacts))
    solver.world.allow = lambda names, contacts: permissions.append((names, contacts))
    with patch.object(skill, 'check_closure', return_value={'held': True}), \
            patch('table_1000.experts.skills.semantic.supported_inside', return_value=True):
        result = skill.grasp(target)
    assert result['supports'] == ['table']
    assert skill.held['left']['supports'] == ['table']
    assert [names for names, _ in carried] == [['obj:cap'], ['rider:body']]
    assert permissions[-1] == (['rider:body'], ['obj:cap', 'rider:body'])


def test_ordinary_part_does_not_carry_its_external_support():
    skill, target, events, closed, attached, retracted, calls = grasping()
    skill.ctx.geometry = {'obj': object()}
    skill.dragged = lambda *args: ([], ['cabinet:drawer'], ['table'])
    with patch.object(skill, 'check_closure', return_value={'held': True}):
        skill.grasp(target)
    assert skill.held['left']['riders'] == []


def test_part_handover_previews_the_complete_carried_asset():
    from table_1000.experts.skills.hold.handover import Handing
    from table_1000.experts.skills.hold.frames import Box
    skill = Handing()
    target = Target('pen', ['pen:barrel'], 'pen:body', Box((.01, .01, .1)), np.eye(4),
                    Box((.01, .01, .1)).points())
    skill.held = {'right': {'target': target, 'supports': [], 'grasp': {}, 'carried': ['pen:cap'],
                            'riders': [], 'carry_contacts': [], 'carry_permissions': []}}
    skill.holding = lambda obj: 'right'
    calls = []
    world = SimpleNamespace(allow=lambda *args: calls.append(args))
    skill.solvers = {'right': None, 'left': SimpleNamespace(world=world)}
    skill.ctx = SimpleNamespace(spec=SimpleNamespace(arms={'right': SimpleNamespace(finger_links=['finger'])}))
    skill.hand_over = lambda chosen, *args: chosen
    chosen = skill.handover('pen', to='left')
    assert chosen.names == ['pen:barrel', 'pen:cap']
    assert target.names == ['pen:barrel']
    assert calls == [(['pen:barrel', 'pen:cap'], ['finger']),
                     (['pen:barrel', 'pen:cap'], ['finger'], False)]
    skill.held['right']['riders'] = ['pencil:body']
    with pytest.raises(ValueError, match='loaded container handover.*keep the current arm'):
        skill.handover('pen', to='left')




@pytest.mark.parametrize('approach_planned', [False, True])
def test_failed_relocated_grasp_approach_tries_only_one_other_stance(approach_planned):
    from table_1000.experts.context import Unreachable
    skill, target, events, closed, attached, retracted, calls = grasping()
    skill.ctx.spec.base = True
    def fail(reason, failure=ExpertFailed, **details):
        raise failure('grasp', reason, **details)
    skill.ctx.fail = fail
    skill.braced = False
    skill.stow = lambda *a, **kw: False
    skill.rejected = dict.fromkeys(('grasp IK', 'pre-grasp IK', 'then', 'need'), 0)
    skill.grasp_paths = lambda *args: {'left': [[np.eye(4)]]}
    skill.obstacles = lambda *args: ([], ['table'])
    moved = []
    skill.make_room = lambda *args, **kw: moved.append(kw)
    def reach(*args, **kwargs):
        if approach_planned and moved:
            args[2].add(('left', 0))
        return None
    skill.reach = reach
    with pytest.raises(ExpertFailed, match='no reachable grasp'):
        skill.grasp(target)
    assert len(moved) == (2 if approach_planned else 1)
    assert all(m['relocate'] for m in moved)
    assert not closed and not attached
