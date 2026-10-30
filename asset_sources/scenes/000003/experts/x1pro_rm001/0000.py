"""Open the bottom drawer, carry the table pen into it, and close it using shared skills."""

ROBOT = 'x1pro_rm001'
SCOPE = {0: None}
SOURCE = 'hand'

import numpy as np

from table_1000.experts import skills
from table_1000.experts.context import Step, run_steps
from table_1000.experts.skills import semantic

DRAWER = 'cabinet.drawer3.slide'
OPENING = .16
PICK_TILT = np.radians(35.)
PICK_ANGLES = np.radians([15., 180.])


def pick_approach(ctx):
    """Lean above the pen about its axis toward the robot, independent of its table yaw."""
    pen = ctx.pose('pen', 'body')
    across = np.cross([0., 0., 1.], pen[:3, 0])
    across /= np.linalg.norm(across)
    if across @ (pen[:3, 3] - ctx.base_transform()[:3, 3]) < 0.:
        across = -across
    return np.cos(PICK_TILT) * np.array([0., 0., -1.]) + np.sin(PICK_TILT) * across


def open_drawer(ctx):
    with ctx.subtask('open the bottom drawer of the cabinet'):
        semantic.open(ctx, 'cabinet', DRAWER, OPENING)


def put_pen_inside(ctx):
    if not semantic.is_open(ctx, 'cabinet', DRAWER, OPENING):
        open_drawer(ctx)
    skills.chain(ctx, [skills.Pick('pen', approach=pick_approach(ctx), max_angle=PICK_ANGLES,
                                   instruction='pick up the pen from the table'),
                       skills.Place('pen', semantic.inside_region('pen', 'cabinet', 'cabinet.drawer3'),
                                    instruction='put the pen into the bottom drawer')])


def close_drawer(ctx):
    with ctx.subtask('close the bottom drawer'):
        semantic.close(ctx, 'cabinet', DRAWER)


def pen_inside(ctx):
    semantic.require(ctx, semantic.inside(ctx, 'pen', 'cabinet', 'cabinet.drawer3'), 'pen not in the drawer')


def closed_with_pen(ctx):
    semantic.require(ctx, semantic.is_closed(ctx, 'cabinet', DRAWER), 'drawer not closed')
    pen_inside(ctx)


# Pick checks that the pen was lifted; pick and place share one step so that
# checkpoints between steps never leave an object in hand (as in 000002).
STEPS = [Step('open the bottom drawer', open_drawer,
              lambda ctx: semantic.require(ctx, semantic.is_open(ctx, 'cabinet', DRAWER, OPENING), 'drawer not opened')),
         Step('move the pen from the table to the bottom drawer', put_pen_inside, pen_inside),
         Step('close the bottom drawer', close_drawer, closed_with_pen)]


def solve(ctx):
    run_steps(ctx, STEPS)
