"""Store the desoldering wick and clear a place on the electronics bench.

Open the middle drawer, put the wick from the table into it, close it, then
move the wireless mouse to a clear part of the table. Intended for layout 3
(the work area); semantic skills derive grasps, regions and pull directions
from the live geometry.
"""

ROBOT = 'x1pro'
SCOPE = {3: None}
SOURCE = 'hand'

from table_1000.experts import skills
from table_1000.experts.context import Step, run_steps
from table_1000.experts.skills import semantic

DRAWER = 'tool_chest.drawer2.slide'
BODY = 'tool_chest.drawer2'
OPENING = .16
ASIDE = (-.12, .02)
CLEAR = .06


def subtask(instruction, action):
    def run(ctx):
        with ctx.subtask(instruction):
            action(ctx)
    return run


def clear(ctx):
    return [semantic.body_box(ctx, 'chest', 'tool_chest.housing', CLEAR),
            semantic.travel(ctx, 'chest', DRAWER, CLEAR)]


def store_wick(ctx):
    if not semantic.is_open(ctx, 'chest', DRAWER, OPENING):
        subtask('open the middle drawer of the tool chest',
                lambda ctx: semantic.open(ctx, 'chest', DRAWER, OPENING))(ctx)
    skills.chain(ctx, [
        skills.Pick('wick', instruction='pick up the desoldering wick from the table'),
        skills.Place('wick', semantic.inside_region('wick', 'chest', BODY),
                     instruction='put the desoldering wick into the middle drawer'),
    ])


def move_mouse(ctx):
    skills.chain(ctx, [
        skills.Pick('wireless_mouse', instruction='pick up the wireless mouse from the table'),
        skills.Place('wireless_mouse', semantic.on_region(
            'wireless_mouse', 'table', near=(*ASIDE, ctx.scene.TABLE['height']), keep_out=clear(ctx)),
            instruction='put the wireless mouse on the clear part of the table'),
    ])


def stored(ctx):
    semantic.require(ctx, semantic.is_closed(ctx, 'chest', DRAWER), 'drawer not closed')
    semantic.require(ctx, semantic.inside(ctx, 'wick', 'chest', BODY), 'wick not in the drawer')


def finished(ctx):
    stored(ctx)
    semantic.require(ctx, semantic.on(ctx, 'wireless_mouse', 'table', keep_out=clear(ctx)),
                     'wireless mouse not on the table')


STEPS = [
    Step('open the middle drawer', subtask('open the middle drawer of the tool chest',
         lambda ctx: semantic.open(ctx, 'chest', DRAWER, OPENING)),
         lambda ctx: semantic.require(ctx, semantic.is_open(ctx, 'chest', DRAWER, OPENING), 'drawer not opened')),
    Step('store the desoldering wick in the middle drawer', store_wick,
         lambda ctx: semantic.require(ctx, semantic.inside(ctx, 'wick', 'chest', BODY), 'wick not in the drawer')),
    Step('close the middle drawer', subtask('close the middle drawer of the tool chest',
         lambda ctx: semantic.close(ctx, 'chest', DRAWER)), stored),
    Step('move the wireless mouse to a clear part of the table', move_mouse, finished),
]


def solve(ctx):
    run_steps(ctx, STEPS)
