"""A hold in one arm must not invalidate the other arm's planning world."""
from types import SimpleNamespace

import mplib
import numpy as np

from table_1000.experts.solver import MotionSolver
from table_1000.experts.audit.trace import planner_state, restore
from table_1000.planning.world import PlanningScene, box_object
from table_1000.robots.planning_model import write_planning_urdf, write_srdf
from tests.planning.test_model_world import robot


def test_other_arm_sees_only_declared_hold_contacts_and_revokes_them_on_release(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    urdf = write_planning_urdf(robot(source), tmp_path / 'planning' / 'robot.urdf',
                               ['j1', 'j2', 'j3'], {}, ('l3', 'tcp', (0., 0., 0.), (0., 0., 0.)))
    srdf = tmp_path / 'robot.srdf'
    write_srdf(urdf, srdf)
    ctx = SimpleNamespace(model=SimpleNamespace(base=np.eye(4)), solvers={}, poses=lambda: {},
                          _skills=SimpleNamespace(held={'right': {}}))
    transform = np.eye(4)
    transform[:3, 3] = [.4, 0., .4]  # Native l3 collision geometry at q=0.
    for arm, finger in [('left', 'l1'), ('right', 'l3')]:
        solver = MotionSolver.__new__(MotionSolver)
        solver.ctx, solver.model = ctx, ctx.model
        solver.arm = SimpleNamespace(name=arm, finger_links=(finger,), tcp_link=finger)
        solver.planner = mplib.Planner(str(urdf), 'l3', srdf=str(srdf), use_convex=True)
        solver.planner.robot.set_qpos(np.zeros(3), True)
        solver.world = PlanningScene(solver.planner)
        solver.attachments = {}
        ctx.solvers[arm] = solver
        for name in ('held', 'unheld', 'intruder'):
            solver.world.add(*box_object(name, (.04, .04, .04), transform))
    left, right = ctx.solvers['left'], ctx.solvers['right']
    right.allow(['held', 'unheld'])
    right.world.allow(['held'], ['l2'])  # Not a finger contact: do not borrow it.
    right.attach_at('held', np.eye(4))

    def contacts():
        return {(c.link_name1, c.object_name2) for c in left.planner.planning_world.check_collision()}

    assert ('l3', 'held') in contacts()
    left.sync()
    assert ('l3', 'held') not in contacts()
    assert ('l3', 'unheld') in contacts()
    assert ('l3', 'intruder') in contacts()
    assert left.world.shared_permissions == {frozenset(('held', 'l3'))}
    # Pose replacement must preserve, but not take ownership of, the borrowed pair.
    left.world.set_pose('held', transform)
    assert ('l3', 'held') not in contacts()
    ctx._skills.held.clear()
    left.sync()
    assert ('l3', 'held') in contacts()
    assert not left.world.shared_permissions

    # A local declaration made while borrowing survives removal of its lender.
    ctx._skills.held['right'] = {}
    left.sync()
    left.world.allow(['held'], ['l3'])
    ctx.scene = SimpleNamespace(OBJECTS={})
    ctx.skills = lambda: ctx._skills
    ctx.solver = lambda arm: ctx.solvers[arm]
    ctx.env = SimpleNamespace(contacts=SimpleNamespace(phase=None))
    saved = planner_state(ctx)
    restore(ctx, dict(held={'right': {}}, reaches={}, planners=saved, current='next item'), restore_rng=False)
    assert left.world.shared_permissions == {frozenset(('held', 'l3'))}
    ctx._skills.held.clear()
    left.sync()
    assert ('l3', 'held') not in contacts()
    left.world.allow(['held'], ['l3'], False)
    assert ('l3', 'held') in contacts()

    # Conversely, revoking a local declaration cannot revoke the still-live hold.
    ctx._skills.held['right'] = {}
    left.sync()
    left.world.allow(['held'], ['l3'])
    left.world.allow(['held'], ['l3'], False)
    assert ('l3', 'held') not in contacts()
    right.allow(['held'], False)
    left.sync()
    assert ('l3', 'held') in contacts()
