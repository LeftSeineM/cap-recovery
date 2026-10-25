"""A ready pose is the robot's declaration; lift and held arms stay independent."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from table_1000.experts.skills import Skills


def test_ready_uses_declared_arm_joints_without_resetting_shared_lift():
    names=['lift','left','right'];initial=np.array([.2,0.,0.])
    model=SimpleNamespace(index=dict(zip(names,range(3))),vector=lambda q:np.array([q[n]for n in names]))
    world=SimpleNamespace(objects={'table':object()})
    ctx=SimpleNamespace(model=model,spec=SimpleNamespace(arms={a:SimpleNamespace(joints=[a])for a in names[1:]},
        keyframes={'ready':{'lift':.8,'left':.4,'right':-.4}},base=None),
        env=SimpleNamespace(joints=names),start={'qpos':initial},scene=SimpleNamespace(OBJECTS={}),
        solver=lambda arm:SimpleNamespace(world=world))
    with patch('table_1000.experts.skills.Hand'):
        skills=Skills(ctx)
    np.testing.assert_allclose(skills.home,[.2,.4,-.4])
    assert np.array_equal(initial,[.2,0.,0.])
