"""A nearest release IK branch may have no above pose at its torso height."""
from types import SimpleNamespace
import numpy as np
from table_1000.experts.skills.placing import Placing


def test_need_and_place_keep_a_release_branch_that_can_reach_above():
    calls=[]
    def ik(arm,pose,q,**options):
        calls.append(options)
        if options.get('closest') is False:return 'Success',[np.array([0.,0.]),np.array([.5,0.])]
        if q[0]==0.:return 'IK Failed',None
        return 'Success',np.array([.5,1.])
    skill=object.__new__(Placing)
    skill.model=SimpleNamespace(q=np.zeros(2),ik=ik)
    skill.ctx=SimpleNamespace(trace=None)
    assert skill.in_reach_down('arm',np.eye(4),np.eye(4),.01,.08)
    goals=skill.down_goals('arm',np.eye(4),.08,True,20)
    np.testing.assert_allclose(goals,[[.5,1.]])
    assert calls[0]['attempts']==5 and calls[3]['attempts']==20
    assert all(not option.get('lift',False) for option in (calls[1],calls[2],calls[4],calls[5]))
