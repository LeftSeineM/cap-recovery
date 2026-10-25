"""Even a sub-control-period movement must retain its final configuration."""
from types import SimpleNamespace
import numpy as np
from mplib import Planner
from table_1000.planning.rrt import parameterize


def test_short_timed_motion_keeps_both_endpoints():
    planner=SimpleNamespace(joint_vel_limits=np.ones(2),joint_acc_limits=np.ones(2),
                            joint_limits=np.array([[-2., 2.], [-2., 2.]]), move_group_joint_indices=[0, 1])
    planner.TOPP=lambda path,step:Planner.TOPP(planner,path,step)
    path=np.array([[0.,0.],[1e-4,0.]])
    def valid(rows):
        assert len(rows)>=2, 'a nonzero motion lost its endpoint'
        return []
    result=parameterize(planner,path,.04,valid)
    assert result['status']=='Success'
    np.testing.assert_allclose(result['position'][[0,-1]],path,atol=1e-10)
