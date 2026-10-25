"""A stance must admit the continuous action, not just its endpoints."""
from types import SimpleNamespace
import numpy as np
from table_1000.experts.skills.base import Base


def model_for(screw):
    return SimpleNamespace(q=np.zeros(2), limits=np.array([[-2.,2.],[-2.,2.]]),
                           arm_indices=lambda arm:np.array([0,1]),
                           ik=lambda *a,**kw:('Success',[np.zeros(2)]),plan_screw=screw)


def test_reachable_endpoints_do_not_accept_a_disconnected_straight_action():
    skill=Base()
    skill.model=model_for(lambda *a,**kw:{'status':'joint limit'})
    assert skill.path_margin('arm',[np.eye(4),np.eye(4)])==-np.inf


def test_margin_covers_the_interior_of_the_motion():
    skill=Base()
    skill.model=model_for(lambda *a,**kw:{'status':'Success','position':np.array([[0.,0.],[1.9,0.],[0.,0.]])})
    assert np.isclose(skill.path_margin('arm',[np.eye(4),np.eye(4)]),.1)


def test_free_transfer_does_not_require_a_straight_connection():
    def screw(*a,**kw):
        raise AssertionError('a free transfer was screened as a straight motion')
    skill=Base();skill.model=model_for(screw)
    assert skill.path_margin('arm',{'poses':[np.eye(4),np.eye(4)],'linear':[False,False]})==2.
