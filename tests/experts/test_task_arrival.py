"""Joint lag is diagnostic; the stopped task result and physical constraints decide."""
from types import SimpleNamespace

import numpy as np
import pytest

from table_1000.experts.solver import MotionSolver, PlanningFailed
from table_1000.experts.context import ExpertFailed


def lagging(*, goal=.01, collision=False, joint_done=False, payload=True):
    solver = MotionSolver.__new__(MotionSolver)
    actions, notes = [], []
    solver.arm = SimpleNamespace(name='left', joints=['j'])
    solver.qpos = lambda lift=False: np.zeros(1)
    def tcp(arm=None, q=None, world=False):
        frame = np.eye(4)
        frame[0, 3] = goal if q is not None else 0.
        return frame
    solver.model = SimpleNamespace(q=np.zeros(1), arm_indices=lambda arm: [0], tcp=tcp,
                                    collisions=lambda *args: [('hand', 'wall', 'wall')] if collision else [])
    solver.tcp = tcp
    solver.sync = lambda: None
    solver.ctx = SimpleNamespace(current='move', step=lambda action=None: actions.append(action),
        note=lambda note, **kw: notes.append((note, kw)), trace=None)
    solver.log, solver.timings = [{}], {'executing_s': 0.}
    solver.constraint_arrival = (lambda: True) if joint_done else None
    released = []
    if not payload:
        solver.ctx._skills = SimpleNamespace(held={'left': {'target': object()}},
            constrained=lambda target: False, check_hold=lambda *args: {'held': False},
            release=lambda arm: (released.append(arm), solver.ctx._skills.held.pop(arm)))
    return solver, notes, actions, released


def test_joint_error_above_old_limit_passes_visible_task_endpoint():
    solver, notes, actions, _ = lagging()
    assert solver.follow({'position': np.array([[.0169]])})
    assert notes[-1][1]['final_error_rad'] == pytest.approx(.0169)
    assert notes[-1][1]['position_error_m'] == pytest.approx(.01)
    assert notes[-1][1]['task_reached']
    # After observing lag the active command is stopped at the measured pose.
    assert actions[-1]['left'][0, 0] == 0.


def test_measured_drawer_opening_can_satisfy_the_action_after_joint_lag():
    solver, notes, *_ = lagging(goal=.05, joint_done=True)
    assert solver.follow({'position': np.array([[.0169]])})
    assert notes[-1][1]['joint_result_reached']


@pytest.mark.parametrize('options,reason', [
    (dict(goal=.10), 'task space goal not reached'),
    (dict(collision=True), 'collision'),
    (dict(payload=False), 'object lost while stopping'),
])
def test_blockage_collision_or_lost_payload_never_becomes_arrival(options, reason):
    solver, notes, actions, released = lagging(**options)
    expected = ExpertFailed if options.get('payload') is False else PlanningFailed
    with pytest.raises(expected, match=reason) as error:
        solver.follow({'position': np.array([[.0169]])})
    assert notes
    if options.get('payload') is False:
        assert released == ['left']
        assert type(error.value) is ExpertFailed


def test_actual_loss_aborts_straight_without_empty_hand_motion_alternatives():
    from table_1000.experts.skills.push.motion import Motion
    solver, notes, actions, released = lagging(payload=False)
    solver.move_to_pose_with_screw = lambda pose: solver.follow({'position': np.array([[.0169]])})
    skill = Motion()
    skill.solvers, skill.ctx = {'left': solver}, solver.ctx
    skill.in_the_way = lambda *args: pytest.fail('lost grip cannot select another motion')
    with pytest.raises(ExpertFailed, match='object lost while stopping') as error:
        skill.straight('left', (0., 0., .02))
    assert type(error.value) is ExpertFailed
    assert released == ['left'] and not solver.ctx._skills.held
    assert any(note == 'missed grasp' for note, _ in notes)


def test_drawer_arrival_uses_measured_opening_and_retains_the_requested_goal(monkeypatch):
    from table_1000.experts.skills.drawer import joints
    from table_1000.experts.skills.push.motion import Motion
    gap = [0.]
    joint = SimpleNamespace(kind='SLIDER', child='drawer', limits=(0., .173),
                            direction=lambda pose: np.array([1., 0., 0.]))
    monkeypatch.setattr(joints.Joint, 'of', lambda *args: joint)
    monkeypatch.setattr(joints, 'opening', lambda *args: gap[0])
    skill = Motion()
    skill.ctx = SimpleNamespace(geometry={'cab': SimpleNamespace(joints={'slide': None})},
                                pose=lambda *args: np.eye(4))
    arrival = skill.drawer_arrival(('cab', 'drawer'), np.array([.173, 0., 0.]))
    gap[0] = .1724
    assert arrival()
    gap[0] = .1
    assert not arrival()  # the callback did not adopt the latest opening as a new goal
    gap[0] = .1724
    closed = skill.drawer_arrival(('cab', 'drawer'), np.array([-.18, 0., 0.]))
    gap[0] = .015
    assert closed()
    gap[0] = .025
    assert not closed()
