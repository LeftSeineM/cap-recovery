"""An executed partial move must not be counted twice by torso fallback."""
from types import SimpleNamespace
import numpy as np
import pytest
from table_1000.experts.skills.motion import Motion
from table_1000.experts.solver import PlanningFailed

@pytest.mark.parametrize('moved,upper,expected_lift',[(.04,1.,.26),(.1,1.,.2),(.02,.25,.25)])
def test_torso_fallback_only_completes_remaining_rise(moved,upper,expected_lift):
    pose=np.eye(4);pose[2,3]=.8
    state=dict(pose=pose,lift=.2,calls=0,steps=[])
    goals=[]
    def screw(goal,**kwargs):
        state['calls']+=1;goals.append(goal.copy())
        if state['calls']==1:
            state['pose'][2,3]+=moved
            raise PlanningFailed('test','tracking')
        state['pose']=goal.copy()
    def step(action):
        target=float(action['lift'][0,0]);state['pose'][2,3]+=target-state['lift'];state['lift']=target;state['steps'].append(target)
    solver=SimpleNamespace(tcp=lambda:state['pose'].copy(),move_to_pose_with_screw=screw)
    ctx=SimpleNamespace(env=SimpleNamespace(qpos=lambda _:np.array([[state['lift']]])),
        spec=SimpleNamespace(lift=['lift'],lift_velocity_limit=1.),step=step,hold=lambda _:None,note=lambda *a,**kw:None)
    motion=Motion();motion.solvers={'left':solver};motion.ctx=ctx;motion.model=SimpleNamespace(limits=np.array([[0.,upper]]),lift_indices=lambda:[0]);motion.held={};motion.stow=lambda _:False;motion.in_the_way=lambda *a:[]
    motion.braced = False
    assert motion.straight('left',np.array([0.,0.,.1]))==pytest.approx(1.)
    assert state['lift']==pytest.approx(expected_lift)
    assert state['pose'][2,3]==pytest.approx(.9)
    assert all(goal[2,3]==pytest.approx(.9)for goal in goals)
