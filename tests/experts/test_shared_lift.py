"""Shared-lift plans clear the other free arm before the approach, not on collision."""
from types import SimpleNamespace
import numpy as np
from table_1000.experts.solver import MotionSolver


def test_shared_lift_stows_other_arm_before_depart_and_plan():
    events=[]
    solver=object.__new__(MotionSolver);solver.arm=SimpleNamespace(name='left')
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
