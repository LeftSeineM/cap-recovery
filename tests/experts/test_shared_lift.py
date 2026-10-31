"""Shared-lift plans clear the other free arm before the approach, not on collision."""
from types import SimpleNamespace
import numpy as np
from table_1000.experts.solver import MotionSolver


def test_shared_lift_stows_other_arm_before_depart_and_plan():
    events=[]
    solver=object.__new__(MotionSolver);solver.arm=SimpleNamespace(name='left')
    solver.lock_lift = False
    skill=SimpleNamespace(arms=['left','right'],stow=lambda arms,**kw:events.append(('stow',arms)))
    solver.ctx=SimpleNamespace(_skills=skill);solver.model=SimpleNamespace(q=np.array([.1]),lift_indices=lambda:[0])
    solver.depart=lambda:events.append(('depart',None))
    solver._rrt=lambda *args:events.append(('plan',args))
    goals=[np.array([.3])]
    solver.move_to_pose_with_RRTConnect(None,lift=True,goals=goals)
    assert [e[0]for e in events]==['stow','depart','plan']
    assert events[0][1]==['right'] and events[2][1][-1]is goals
    events.clear();solver.move_to_pose_with_RRTConnect(None,lift=False,goals=goals)
    assert [e[0]for e in events]==['depart','plan']


def test_small_shared_lift_plans_without_retracting_the_other_arm():
    events = []
    solver = object.__new__(MotionSolver)
    solver.arm = SimpleNamespace(name='left')
    solver.lock_lift = False
    solver.ctx = SimpleNamespace(_skills=SimpleNamespace(
        arms=['left', 'right'], stow=lambda *a, **kw: events.append('stow')))
    solver.model = SimpleNamespace(q=np.array([.1]), lift_indices=lambda: [0])
    solver.depart = lambda: events.append('depart')
    solver._rrt = lambda *a: events.append('plan')
    solver.move_to_pose_with_RRTConnect(None, lift=True, goals=[np.array([.2])], local_lift=True)
    assert events == ['depart', 'plan']
    events.clear()
    solver.move_to_pose_with_RRTConnect(None, lift=True, goals=[np.array([.2])])
    assert events == ['stow', 'depart', 'plan']


def test_blocked_small_shared_lift_retracts_then_replans():
    from table_1000.experts.solver import PlanningFailed
    events = []
    solver = object.__new__(MotionSolver)
    solver.arm = SimpleNamespace(name='left')
    solver.lock_lift = False
    def stow(arms, **kw):
        assert arms == ['right']
        events.append('stow')
        return True
    solver.ctx = SimpleNamespace(_skills=SimpleNamespace(arms=['left', 'right'], stow=stow))
    solver.model = SimpleNamespace(q=np.array([.1]), lift_indices=lambda: [0])
    solver.depart = lambda: events.append('depart')
    def plan(*args):
        events.append('plan')
        if events.count('plan') == 1:
            raise PlanningFailed('move', 'path collision')
    solver._rrt = plan
    solver.move_to_pose_with_RRTConnect(None, lift=True, goals=[np.array([.2])], local_lift=True)
    assert events == ['depart', 'plan', 'stow', 'plan']


def test_contact_preview_keeps_ready_phase_except_after_missed_grasp():
    from table_1000.experts.skills.hold.pick import Grasping
    skill = Grasping()
    skill.ctx = SimpleNamespace()
    skill.braced = False
    skill.centre = lambda target: np.zeros(3)
    stows = []
    skill.stow = lambda **kw: stows.append(kw['why'])
    assert list(skill.options(object(), [], set())) == []
    assert stows == ['shared torso lift']
    stows.clear()
    assert list(skill.options(object(), [], set(), local=True)) == []
    assert stows == []
    skill.braced = True
    assert list(skill.options(object(), [], set())) == []
    assert stows == []
