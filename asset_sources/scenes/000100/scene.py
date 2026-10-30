"""Scene 000100: cluttered office desk for X1Pro and RM001, 38 objects.

Layouts 1–10 arrange them, five tidy and five not.

World frame: origin on the floor below the table centre, +Z up. The robot stands
in front of the table's -Y edge facing +Y. X1Pro starts 0.29 m behind it (its torso
reaches 0.26 m ahead of the base); with its torso lift and the gripper tilted up to 60 degrees from
straight down it reaches the whole width of the table from the front edge to about
y = +0.10. Containers stand in a
back row inside that reach: the three-drawer cabinet on the right (drawers toward
the robot), the file basket on the left, two pen holders and three glass jars
between them. RM001 starts farther back for its head camera, then advances to the
working edge; see ROBOTS for its placement and reach checks. The rest lies in the
front area, y from -0.47 to -0.10.

``initialize`` makes a messy desk: containers jittered in the back row (the small
jars may stand anywhere), pens partly in the holders and partly strewn, notebooks
dropped at random so that some overlap, clips and magnets partly in the jars, the
top drawer sometimes open with a few small things in it, everything else strewn over
the front area. ``LAYOUTS`` are arrangements of the same objects for judging
tidiness; each has a short description of how it is arranged.
"""

import math

import numpy as np

from table_1000.scenes.arrange import (ACROSS, drawer_contents, footprint_area, into, stand, stand_in, strew)
from table_1000.scenes.layout import SamplingFailed, yaw_pose

TABLE = {'size': (1.36, 1.02), 'height': .755, 'top_thickness': .03, 'leg_size': .05,
         'center': (0., 0.), 'friction': (.5, .4)}

ROBOTS = {'x1pro': {
    'position': (-.03, -.80, 0.),
    'yaw': math.pi / 2,
    # Torso raised and head tilted down so the head camera sees the whole desk over the folded arms.
    'joints': {'lift_joint': .70, 'head_pitch_joint': .60},
}}
# RM001 uses the README ready lift (0.45 m), as in 000002/000003 with 5 cm
# extra torso clearance. With the calibrated head camera, y=-1.30 and pitch=0.10
# put all four corners of the 1.36 x 1.02 m tabletop inside the image. The mobile
# base starts 0.50 m behind x1pro to see the whole desktop, then advances about
# 0.50 m to the front working edge for manipulation. Front/back container IK is
# checked after this advance, as for the existing mobile-base scenes.
ROBOTS['x1pro_rm001'] = {'position': (-.03, -1.30, 0.), 'yaw': math.pi / 2,
                          'joints': {'lift_joint': .45, 'head_pitch_joint': .10}}
ROBOT = 'x1pro_rm001'

CONTAINERS = {
    'cabinet': 'cabinet/000000',            # three drawers, 0.29 m cube
    'basket': 'file_basket/000000',         # 0.33 x 0.24 x 0.10
    'holder': 'pen_holder/000000',          # three compartments
    'cup': 'pen_holder/000001',
    'jar': 'storage_box/000000',            # glass, 0.15 across
    'lidded_jar': 'storage_box/000001',
    'dish': 'storage_box/000002',           # glass, 0.055 across
}
NOTEBOOKS = {'notebook_a6': 'notebook/000000', 'notebook_a4': 'notebook/000002',
             'notebook_a5': 'notebook/000004', 'notebook_slim': 'notebook/000005'}
# Long pens (0.15-0.18 m) stand safely only in the 0.13 m cup, short ones also in the 0.10 m holder.
LONG_PENS = {'pen_black': 'pen/000000', 'pen_white': 'pen/000001', 'marker_blue': 'pen/000003',
             'gel_pen': 'pen/000004'}
SHORT_PENS = {'highlighter_yellow': 'pen/000007', 'highlighter_pink': 'pen/000008', 'pencil': 'pen/000010',
              'marker_black': 'pen/000011'}
PENS = {**LONG_PENS, **SHORT_PENS}
CLIPS = {'clip_large': 'binder_clip/000002', 'clip_blue': 'binder_clip/000000',
         'clip_pink': 'binder_clip/000004', 'clip_small': 'binder_clip/000006'}
MAGNETS = {'magnet_orange': 'magnet/000000', 'magnet_red': 'magnet/000002'}
SMALL = {'eraser': 'eraser/000000', 'usb_drive': 'usb_drive/000000', 'staples': 'staple/000000'}
TOOLS = {'board_eraser': 'board_eraser/000000', 'glue_bottle': 'glue/000000', 'glue_stick': 'glue/000001',
         'tape': 'tape/000000', 'stapler': 'stapler/000000', 'remote': 'remote_control/000000',
         'screwdriver': 'screwdriver/000000', 'knife': 'utility_knife/000000', 'snack': 'distractor_food/000000',
         'charger': 'power_plug/000000'}

OBJECTS = {name: {'asset': f'objects/{asset}'} for group in (CONTAINERS, NOTEBOOKS, PENS, CLIPS, MAGNETS, SMALL, TOOLS)
           for name, asset in group.items()}

VIEWS = {
    'top': {'position': (0., -.30, 2.05), 'target': (0., -.17, .755), 'focal_length_mm': 18,
            'resolution': (1280, 960)},
    'wide': {'position': (1.2, -1.35, 1.65), 'target': (0., -.12, .78), 'focal_length_mm': 20,
             'resolution': (960, 540)},
}
IMAGE_VIEW = 'top'      # the whole desk, the top of the cabinet too, without the robot's chest

FRONT = ((-.45, -.47), (.45, -.10))          # where loose things lie on a messy desk
FLAT = ['table', *NOTEBOOKS]                  # what strewn things may land on


# -- helpers -------------------------------------------------------------------------
def into_holder(layout, pens, holder, lean=(4., 12.)):
    """Stand ``pens`` in ``holder`` (the three compartments of ``holder`` or the round ``cup``)."""
    if holder == 'holder':
        spots = [(-.04, 0.), (0., 0.), (.04, 0.), (-.04, .015), (0., -.015), (.04, .015)]
    else:
        spots = [(-.02, -.02), (.02, .02), (-.02, .02), (.02, -.02), (0., 0.)]
    stand_in(layout, pens, holder, spots, lean)


def back_row(layout, jitter=0., turn=0.):
    """Cabinet, basket and the two pen holders in the back row; ``jitter`` (m) and ``turn`` (rad)
    randomize them."""
    def spot(x, y):
        return ((x - jitter, y - jitter), (x + jitter, y + jitter))

    yaw = (-turn, turn) if turn else 0.
    layout.place('cabinet', spot(.38, .03), yaw=yaw)
    layout.place('basket', spot(-.40, .01), yaw=yaw)
    layout.place('holder', spot(-.17, .04), yaw=yaw)
    layout.place('cup', spot(.14, .05), yaw=yaw)


BASKET_FLOOR = (.146, .101)     # half extents (m) of the file basket's floor between its walls


def basket_notebooks(layout, names=NOTEBOOKS):
    """Notebooks into the file basket: those that fit lie flat in one stack against its back wall,
    the largest at the bottom, long sides along the basket; one wider than the floor (the A4,
    0.21 m on a 0.20 m floor) on a slant over them, its foot against the front wall."""
    flat, slanted = [], []
    for name in sorted(names, key=lambda name: footprint_area(layout, name), reverse=True):
        width = np.sort(np.ptp(layout.geometry[name].points(), axis=0)[:2])
        (flat if (width < 2 * np.array(BASKET_FLOOR)[::-1] - .004).all() else slanted).append(name)
    for name in flat:
        size = np.ptp(layout.geometry[name].points(), axis=0)
        across = min(size[:2])
        layout.drop(name, (0., BASKET_FLOOR[1] - across / 2 - .003), math.pi / 2 if size[1] > size[0] else 0.,
                    frame='basket')
    for name in slanted:
        slant_in_basket(layout, name)


def slant_in_basket(layout, name):
    """Lay notebook ``name`` across the basket at the least slant (from 20 degrees up) at which it
    comes down with its foot on the floor at the front wall, its far end on what lies behind or
    over the back rim. Its long side runs front to back, the end with a full edge down."""
    geometry = layout.geometry[name]
    flat = geometry.stable_rotation(np.eye(3))
    size = np.ptp(geometry.points() @ flat.T, axis=0)
    if size[0] > size[1]:
        flat = yaw_pose(0., 0., 0., math.pi / 2)[:3, :3] @ flat
    points = geometry.points() @ flat.T
    if edge_width(points * (1., -1., 1.)) > edge_width(points):     # an A4's bookmark hangs out of one end
        flat = yaw_pose(0., 0., 0., math.pi)[:3, :3] @ flat
    floor = layout.poses['basket'][2, 3] + .003
    for degrees in range(20, 62, 2):
        up = math.radians(degrees)       # about X: the far (+Y) end up
        rotation = np.array([[1., 0., 0.], [0., math.cos(up), -math.sin(up)], [0., math.sin(up), math.cos(up)]]) @ flat
        front = (geometry.points() @ rotation.T)[:, 1].min()      # its foot's front edge
        try:
            _, support = layout.drop(name, (0., -BASKET_FLOOR[1] + .002 - front), rotation=rotation, frame='basket')
        except SamplingFailed:
            continue
        if support == 'basket' and abs(layout.world_points(name)[:, 2].min() - floor) < .001:
            return
        layout.remove(name)
    raise SamplingFailed(f'no slant for {name} in the basket')


def edge_width(points, depth=.002):
    """Width along X of the points within ``depth`` of the -Y end of ``points``."""
    near = points[points[:, 1] < points[:, 1].min() + depth]
    return np.ptp(near[:, 0])


def notebook_stack(layout, names, xy, yaw=0., on=None):
    """``names`` stacked squarely at ``xy`` (in ``on``'s frame, default the world), the largest at
    the bottom."""
    for name in sorted(names, key=lambda name: footprint_area(layout, name), reverse=True):
        layout.drop(name, xy, yaw, frame=on)


def scatter_front(layout, names, region=FRONT):
    """Drop ``names`` at random over ``region``, on the table or on notebooks."""
    strew(layout, names, region, onto=FLAT)


# -- the messy desk ----------------------------------------------------------------------
def initialize(layout):
    rng = layout.rng
    layout.free_landing = True      # a pen may fall out of its holder, a glue stick tip over
    back_row(layout, jitter=.025, turn=math.radians(12))
    layout.place('jar', ((-.04, -.02), (0., .08)) if rng.random() < .6 else FRONT, yaw=(-math.pi, math.pi))
    for name in ('lidded_jar', 'dish'):
        layout.place(name, ((-.30, -.40), (.30, .12)), yaw=(-math.pi, math.pi))

    into_holder(layout, list(rng.choice(list(SHORT_PENS), rng.integers(0, 4), replace=False)), 'holder',
                lean=(4., 20.))
    into_holder(layout, list(rng.choice(list(PENS), rng.integers(0, 4), replace=False)), 'cup', lean=(4., 20.))

    if rng.random() < .5:
        drawer_contents(layout, 1, list(rng.choice([*SMALL, *MAGNETS], 2, replace=False)), rng.uniform(.06, .12))
    jarred = [name for name in [*CLIPS, *MAGNETS] if name not in layout.poses and rng.random() < .35]
    if rng.random() < .7:
        into(layout, jarred, 'jar', .03)
    else:
        into(layout, jarred[:2], 'dish', .008)

    notebooks = list(rng.permutation(list(NOTEBOOKS)))
    if rng.random() < .4 and notebooks[0] != 'notebook_a4':
        into(layout, notebooks[:1], 'basket', .03)
        notebooks = notebooks[1:]
    scatter_front(layout, notebooks)
    scatter_front(layout, [name for name in OBJECTS if name not in layout.poses])


# -- layouts ---------------------------------------------------------------------------------
def tidy_back_row(layout):
    back_row(layout)
    layout.place('jar', ((-.02, .05), (-.02, .05)))


def messy_back_row(layout):
    """The back row as left by someone who did not care: containers shifted and turned well off square."""
    back_row(layout, jitter=.06, turn=math.radians(35))
    layout.place('jar', ((-.08, -.06), (.04, .10)))


def pens_in_holders(layout):
    into_holder(layout, list(SHORT_PENS), 'holder')
    into_holder(layout, list(LONG_PENS), 'cup')


def stored_pens_and_notebooks(layout):
    tidy_back_row(layout)
    pens_in_holders(layout)
    basket_notebooks(layout)


def sorted_by_kind(layout):
    stored_pens_and_notebooks(layout)
    drawer_contents(layout, 1, [*SMALL, *MAGNETS])
    drawer_contents(layout, 2, [*CLIPS, 'tape', 'glue_bottle', 'glue_stick'], yaw=(-.2, .2),
                    lying=['glue_bottle', 'glue_stick'])
    drawer_contents(layout, 3, ['screwdriver', 'knife', 'remote', 'charger', 'snack'], yaw=(-.2, .2))
    layout.row(['stapler', 'board_eraser', 'lidded_jar', 'dish'], (-.20, -.17), gap=.04)


def cabinet_left(layout):
    stand(layout, {'cabinet': (-.38, .03), 'cup': (-.14, .05), 'jar': (.02, .05), 'holder': (.17, .04),
                   'basket': (.40, .01)})
    pens_in_holders(layout)
    basket_notebooks(layout)
    into(layout, [*CLIPS, *MAGNETS], 'jar', .03)
    drawer_contents(layout, 1, [*SMALL, 'tape', 'glue_bottle', 'glue_stick'], yaw=(-.2, .2),
                    lying=['glue_bottle', 'glue_stick'])
    drawer_contents(layout, 2, ['stapler', 'board_eraser', 'remote'], yaw=(-.2, .2))
    drawer_contents(layout, 3, ['screwdriver', 'knife', 'snack', 'charger'], yaw=(-.2, .2))
    layout.row(['lidded_jar', 'dish'], (-.03, -.12), gap=.04)


def ready_to_work(layout):
    tidy_back_row(layout)
    pens_in_holders(layout)
    notebook_stack(layout, NOTEBOOKS, (-.38, -.28))
    layout.row(['stapler', 'tape'], (-.22, -.30), yaw=ACROSS, gap=.03)
    into(layout, [*CLIPS, *MAGNETS], 'jar', .03)
    # Keep the board eraser with the small tools: beside both jars it crowds the exposed drawer floor.
    drawer_contents(layout, 1, [*SMALL, 'knife', 'glue_stick', 'board_eraser'], yaw=(-.2, .2), lying=['glue_stick'])
    drawer_contents(layout, 2, ['remote', 'charger', 'snack', 'glue_bottle'], yaw=(-.2, .2), lying=['glue_bottle'])
    drawer_contents(layout, 3, ['screwdriver', 'lidded_jar', 'dish'], yaw=(-.2, .2))


def in_columns(layout):
    stand(layout, {'cabinet': (.38, -.08)})
    layout.place('basket', ((-.42, -.07), (-.42, -.07)), yaw=math.pi / 2)
    layout.row(['cup', 'holder', 'jar'], (-.13, .10), direction=-math.pi / 2, yaw=math.pi / 2, gap=.03)
    pens_in_holders(layout)
    basket_notebooks(layout)
    into(layout, [*CLIPS, *MAGNETS], 'jar', .03)
    drawer_contents(layout, 1, [*SMALL, 'knife', 'glue_stick'], yaw=(-.2, .2), lying=['glue_stick'])
    drawer_contents(layout, 2, ['remote', 'charger', 'snack', 'glue_bottle'], yaw=(-.2, .2), lying=['glue_bottle'])
    drawer_contents(layout, 3, ['screwdriver', 'board_eraser'], yaw=(-.2, .2))
    layout.row(['lidded_jar', 'dish', 'tape', 'stapler'], (.08, .08), direction=-math.pi / 2, yaw=math.pi / 2, gap=.04)


def all_stored(layout):
    stand(layout, {'jar': (-.48, .04), 'holder': (-.30, .04), 'cup': (-.17, .05), 'basket': (.06, .01),
                   'cabinet': (.38, .03)})
    pens_in_holders(layout)
    basket_notebooks(layout)
    into(layout, list(CLIPS), 'jar', .03)
    drawer_contents(layout, 1, [*MAGNETS, *SMALL, 'knife', 'glue_stick'], yaw=(-.2, .2), lying=['glue_stick'])
    drawer_contents(layout, 2, ['tape', 'stapler', 'glue_bottle', 'board_eraser', 'remote'], yaw=(-.2, .2),
                    lying=['glue_bottle'])
    drawer_contents(layout, 3, ['lidded_jar', 'dish', 'screwdriver', 'charger', 'snack'], yaw=(-.2, .2))


def pile(layout):
    messy_back_row(layout)
    for name in ('lidded_jar', 'dish'):
        layout.place(name, ((-.10, -.40), (.10, -.15)), yaw=(-math.pi, math.pi))
    layout.free_landing = True      # a heap shifts as it settles

    # Largest first, as things end up in a heap: notebooks at the bottom, small things on top.
    for name in sorted((name for name in OBJECTS if name not in layout.poses), 
                       key=lambda name: footprint_area(layout, name), reverse=True):
        layout.scatter(name, ((-.18, -.40), (.18, -.16)), yaw=(-math.pi, math.pi))


def wrong_places(layout):
    messy_back_row(layout)
    notebook_stack(layout, NOTEBOOKS, (0., 0.), on='cabinet')
    into(layout, ['pen_black', 'marker_blue', 'highlighter_pink', 'snack', 'charger', 'tape'], 'basket', .07)
    into_holder(layout, ['screwdriver', 'knife', 'glue_stick'], 'cup')
    into(layout, [*CLIPS, *MAGNETS], 'holder', .02)
    into(layout, ['usb_drive', 'staples'], 'jar', .03)
    scatter_front(layout, [name for name in OBJECTS if name not in layout.poses], region=((-.45, -.47), (.15, -.10)))


def half_done(layout):
    messy_back_row(layout)
    into_holder(layout, list(SHORT_PENS), 'holder')
    notebook_stack(layout, NOTEBOOKS, (-.45, -.30))
    layout.row(['screwdriver', 'knife', 'board_eraser'], (-.25, -.38), yaw=ACROSS, gap=.03)
    scatter_front(layout, [name for name in OBJECTS if name not in layout.poses], region=((-.05, -.47), (.45, -.10)))


def left_open(layout):
    layout.free_landing = True      # notebooks dropped askew slide off the basket
    messy_back_row(layout)
    into_holder(layout, list(SHORT_PENS), 'holder', lean=(4., 20.))
    into_holder(layout, list(LONG_PENS), 'cup', lean=(4., 20.))
    for name in sorted(NOTEBOOKS, key=lambda name: footprint_area(layout, name)):
        layout.scatter(name, ((-.04, -.03), (.04, .03)), yaw=(-.4, .4), frame='basket', onto=['basket', *NOTEBOOKS])
    # From the bottom up: an open drawer is in the way of filling the one below.
    drawer_contents(layout, 3, ['knife', 'screwdriver'], opening=.16, yaw=(-.2, .2))
    drawer_contents(layout, 2, [*CLIPS, 'tape'], opening=.12)
    drawer_contents(layout, 1, [*SMALL, *MAGNETS], opening=.15)
    scatter_front(layout, [name for name in OBJECTS if name not in layout.poses], region=((-.30, -.40), (.10, -.12)))


def knocked_over(layout):
    layout.free_landing = True      # spilled pens roll
    turn = (-math.radians(35), math.radians(35))
    layout.place('cabinet', ((.32, -.03), (.44, .09)), yaw=turn)
    layout.place('basket', ((-.46, -.05), (-.34, .07)), yaw=turn)
    layout.place('jar', ((-.08, -.06), (.04, .10)))
    # Both pen holders on their sides, their open ends toward the robot, the pens spilled out.
    toward_robot = yaw_pose(0., 0., 0., -math.pi / 2)[:3, :3]
    for holder, x in (('holder', -.17), ('cup', .14)):
        turned = yaw_pose(0., 0., 0., layout.rng.uniform(-.5, .5))[:3, :3] @ toward_robot
        rotation = layout.geometry[holder].stable_rotation(turned @ np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]))
        layout.drop(holder, (x + layout.rng.uniform(-.03, .03), -.02), rotation=rotation)
    for pens, x in ((SHORT_PENS, -.17), (LONG_PENS, .14)):
        for pen in pens:
            layout.scatter(pen, ((x - .10, -.30), (x + .10, -.12)), yaw=(-math.pi, math.pi), onto='table')
    basket_notebooks(layout)
    into(layout, [*CLIPS, *MAGNETS], 'jar', .03)
    drawer_contents(layout, 1, list(SMALL))
    drawer_contents(layout, 2, ['remote', 'charger', 'snack', 'tape', 'stapler'], yaw=(-.2, .2))
    drawer_contents(layout, 3, ['screwdriver', 'knife', 'board_eraser', 'glue_bottle', 'glue_stick'], yaw=(-.2, .2),
                    lying=['glue_bottle', 'glue_stick'])
    layout.row(['lidded_jar', 'dish'], (.30, -.30), gap=.04)


NOTEBOOKS_IN_BASKET = 'notebooks in the basket (three flat in a stack, the A4 too wide to lie flat slanted over them)'
LAYOUTS = {
    1: {
        'quality': 'good', 'arrange': sorted_by_kind,
        'description': f'Sorted by kind: {NOTEBOOKS_IN_BASKET}; pens standing in the two holders; eraser, USB '
                       'drive, staples and magnets in the top drawer, clips, tape and glue in the middle one, '
                       'screwdriver, knife, remote, charger and snack in the bottom one (all closed); only the '
                       'stapler, board eraser and the two small jars in one row on the desk.'},
    2: {
        'quality': 'good', 'arrange': cabinet_left,
        'description': f'The back row mirrored, cabinet on the left and basket on the right: {NOTEBOOKS_IN_BASKET}; '
                       'pens in the two holders; clips and magnets in the large glass jar between them, the two '
                       'small jars in front of it; everything else in the three drawers (closed).'},
    3: {
        'quality': 'good', 'arrange': ready_to_work,
        'description': 'Ready to work: the notebooks squarely stacked on the front left of the desk, the stapler '
                       'and tape lined up beside them; the basket empty; pens in the holders; clips and magnets in '
                       'the large glass jar; everything else, the two small jars too, in the three drawers '
                       '(closed).'},
    4: {
        'quality': 'good', 'arrange': in_columns,
        'description': f'In three columns front to back: the basket turned lengthwise on the left, '
                       f'{NOTEBOOKS_IN_BASKET}; the pen cup, the pen holder and the large glass jar (clips and '
                       'magnets) one behind another, the two small jars, tape and stapler in a column beside them; '
                       'the cabinet alone on the right; everything else in the three drawers (closed).'},
    5: {
        'quality': 'good', 'arrange': all_stored,
        'description': f'Everything stored: the basket in the middle next to the cabinet, {NOTEBOOKS_IN_BASKET}; '
                       'the large glass jar (clips) and the pen holders with their pens at the left end of the back '
                       'row; everything else, the two small jars too, in the three drawers (closed); nothing else on '
                       'the desk.'},
    6: {
        'quality': 'bad', 'arrange': pile,
        'description': 'Everything swept into one heap in the middle of the desk.'},
    7: {
        'quality': 'bad', 'arrange': wrong_places,
        'description': 'Things in the wrong places: notebooks on top of the cabinet, pens mixed with the snack '
                       'and charger in the basket, tools in the pen cup, clips and magnets in the pen holder, '
                       'the rest strewn over the left of the desk.'},
    8: {
        'quality': 'bad', 'arrange': half_done,
        'description': 'Only half tidied: notebooks stacked and a few pens and tools in order on the left, the '
                       'right half of the desk still strewn with everything else.'},
    9: {
        'quality': 'bad', 'arrange': left_open,
        'description': 'Put away carelessly: the three drawers left open with things in them, the notebooks '
                       'dropped askew on top of the basket, the remaining tools, glue and jars strewn over the '
                       'middle of the desk.'},
    10: {
        'quality': 'bad', 'arrange': knocked_over,
        'description': 'Tidy except that both pen holders lie knocked over on their sides with the pens spilled '
                       'over the desk in front of them.'},
}
