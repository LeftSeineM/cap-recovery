"""The timed motion must stay on a validated route around a corner."""
from types import SimpleNamespace
import numpy as np
from mplib import Planner
from table_1000.planning.rrt import parameterize


def corner_check(rows):
    rows=np.asarray(rows)
    for start,end in zip(rows[:-1],rows[1:]):
        n=max(1,int(np.ceil(np.max(np.abs(end-start))/.01)))
        for q in np.linspace(start,end,n+1):
            if not (abs(q[0])<1e-8 or abs(q[1]-1.)<1e-8):
                return [('hand','wall','wall')]
    return []


def test_timing_preserves_the_checked_corner_and_joint_limits():
    planner=SimpleNamespace(joint_vel_limits=np.ones(2),joint_acc_limits=np.ones(2),
                            joint_limits=np.array([[-2., 2.], [-2., 2.]]), move_group_joint_indices=[0, 1])
    planner.TOPP=lambda path,step:Planner.TOPP(planner,path,step)
    raw=np.array([[0.,0.],[0.,1.],[1.,1.]])
    assert corner_check(raw)==[]
    assert corner_check(planner.TOPP(raw,.04)[1])
    result=parameterize(planner,raw,.04,corner_check)
    assert result['status']=='Success'
    assert corner_check(result['position'])==[]
    np.testing.assert_allclose(result['position'][[0,-1]],raw[[0,-1]],atol=1e-8)
    assert np.all(np.diff(result['time'])>0)
    assert np.max(np.abs(result['velocity']))<=1.+1e-4
    assert np.max(np.abs(result['acceleration']))<=1.+1e-4


def test_an_unsafe_raw_route_is_rejected_before_timing():
    def forbidden(*args):raise AssertionError('unsafe geometry was timed')
    result=parameterize(SimpleNamespace(TOPP=forbidden, joint_limits=np.array([[-2., 2.], [-2., 2.]]),
                                         move_group_joint_indices=[0, 1]),np.array([[0.,0.],[1.,1.]]),.04,corner_check)
    assert result['status']=='path collision'
    assert result['collisions']==[('hand','wall','wall')]
