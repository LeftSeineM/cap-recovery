"""A ready pose is the robot's declaration; lift and held arms stay independent."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from table_1000.experts.skills import Skills


def test_ready_uses_declared_arm_joints_without_resetting_shared_lift():
    names=['lift','left','right'];initial=np.array([.2,0.,0.])
    model=SimpleNamespace(index=dict(zip(names,range(3))),vector=lambda q:np.array([q[n]for n in names]))
    world=SimpleNamespace(objects={'table':object()})
    ctx=SimpleNamespace(model=model,spec=SimpleNamespace(arms={a:SimpleNamespace(joints=[a])for a in names[1:]},
        keyframes={'ready':{'lift':.8,'left':.4,'right':-.4}},base=None),
        env=SimpleNamespace(joints=names, gripper_drive=lambda arm: None),start={'qpos':initial},scene=SimpleNamespace(OBJECTS={}),
        solver=lambda arm:SimpleNamespace(world=world))
    with patch('table_1000.experts.skills.Hand'):
        skills=Skills(ctx)
    np.testing.assert_allclose(skills.home,[.2,.4,-.4])
    assert skills.braced is False and skills.brace_seed is None
    assert np.array_equal(initial,[.2,0.,0.])


@pytest.mark.parametrize('blocked', [False, True])
def test_ready_keeps_the_release_gap_until_the_arm_has_withdrawn(blocked):
    from table_1000.experts.skills.base import Base
    from table_1000.experts.solver import PlanningFailed
    safe_to_open = False
    openings = []
    def move(*args, **kwargs):
        nonlocal safe_to_open
        if blocked:
            raise PlanningFailed(None, 'path collision')
        safe_to_open = True
    def open_gripper():
        assert safe_to_open, 'opening in place would sweep the released object'
        openings.append(True)
    skill = Base()
    skill.home = np.zeros(1)
    skill.arms, skill.held = ['right'], {}
    skill.model = SimpleNamespace(q=np.ones(1), arm_indices=lambda arm: [0])
    skill.solvers = {'right': SimpleNamespace(move_to_pose_with_RRTConnect=move, open_gripper=open_gripper)}
    skill.ctx = SimpleNamespace(note=lambda *args, **kwargs: None)
    skill.ready()
    assert openings == ([] if blocked else [True])


def test_ready_opens_the_empty_hand_already_at_home():
    from table_1000.experts.skills.base import Base
    openings = []
    skill = Base()
    skill.home = np.zeros(1)
    skill.arms, skill.held = ['right'], {}
    skill.model = SimpleNamespace(q=np.zeros(1), arm_indices=lambda arm: [0])
    skill.solvers = {'right': SimpleNamespace(
        move_to_pose_with_RRTConnect=lambda *a, **kw: pytest.fail('unnecessary arm motion'),
        open_gripper=lambda: openings.append(True))}
    skill.ready()
    assert openings == [True]
