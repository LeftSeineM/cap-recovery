"""Scene 000103: X1Pro at the electronics bench of the reference photo, 55 objects; layout 0 is the photo,
layouts 1–10 arrange the same objects, five tidy and five not.

Recreates a photo of a cramped hobby electronics bench (reference.jpg: "Electronics workbench" by James
Bastow, CC BY-SA 2.0, https://commons.wikimedia.org/wiki/File:Electronics_workbench.jpg) with the objects
it shows; scene 000101 is a freer arrangement of a larger set of the same kind of objects.

World frame: origin on the floor below the table centre, +Z up. The robot stands in front of the
table's -Y edge facing +Y and reaches from the front edge to about y = +0.10. The back of the bench is
the same in every layout: a backboard with parts drawers and four pegs along the back edge, the
monitor, the oscilloscope, the soldering station with the iron stand and the instrument box in front
of it, the bench power supply on the left and the desk lamp on the right reaching over the bench.
The red tool chest (three drawers) stands at the front right turned to face the robot, the waste bin
(not in the photo; big enough for the empty bottle) on the floor off the bench's front right corner.

``initialize`` is the photo: the keyboard, the logbook under the big multimeter, calculators, boards,
leads and tools piled up in the middle, litter among them, the cutters on a peg, the iron in its stand;
each seed shifts and turns the loose things a little. ``LAYOUTS`` are arrangements of the same objects
for judging tidiness; the fixtures at the back stay where they are.
"""

import math

import numpy as np

from table_1000.scenes.arrange import UPRIGHT, drawer_contents, footprint_area, into
from table_1000.scenes.layout import PENETRATION_LIMIT, SamplingFailed, yaw_pose

TABLE = {'size': (1.36, 1.02), 'height': .755, 'top_thickness': .03, 'leg_size': .05,
         'center': (0., 0.), 'friction': (.5, .4)}

ROBOTS = {'x1pro': {
    'position': (-.03, -.80, 0.),
    'yaw': math.pi / 2,
    'joints': {'lift_joint': .70, 'head_pitch_joint': .60},
}}
ROBOT = 'x1pro'

BACKGROUND = {'backboard': 'bench_backboard/000000', 'monitor': 'monitor/000000', 'psu': 'bench_power_supply/000000',
              'scope': 'oscilloscope/000000', 'box': 'instrument_box/000000', 'lamp': 'desk_lamp/000000',
              'station': 'soldering_station/000000', 'holder': 'iron_holder/000000'}
CONTAINERS = {'chest': 'tool_chest/000000', 'bin': 'trash_bin/000001'}
TOOLS = {'meter': 'multimeter/000000', 'pocket_meter': 'multimeter/000001', 'pliers': 'pliers/000000',
         'cutters': 'pliers/000001', 'hands': 'helping_hands/000000', 'driver_set': 'screwdriver_set/000000'}
SOLDERING = {'iron': 'soldering_iron/000000', 'spool': 'solder_spool/000000', 'flux': 'flux_tin/000000',
             'wick': 'desolder_wick/000000'}
GEAR = {'board': 'circuit_board/000000', 'mini_board': 'circuit_board/000001', 'lcd': 'lcd_module/000000',
        'adapter': 'power_adapter/000000', 'heat_sink': 'heat_sink/000000', 'connector': 'db_adapter/000000',
        'scope_probe': 'scope_probe/000000', 'lead_red': 'clip_lead/000000', 'lead_black': 'clip_lead/000001',
        'pen': 'pen/000012', 'mouse': 'computer_mouse/000000'}
CABLES = {'cable': 'cable_coil/000000', 'network_cable': 'cable_coil/000001'}
DESK = {'keyboard': 'keyboard/000000', 'wireless_mouse': 'computer_mouse/000001', 'logbook': 'lab_notebook/000000',
        'calculator': 'calculator/000000', 'graphing_calculator': 'calculator/000001'}
LITTER = {'label_1': 'paper_label/000000', 'label_2': 'paper_label/000001', 'label_3': 'paper_label/000002',
          'label_4': 'paper_label/000003', 'label_5': 'paper_label/000000', 'label_6': 'paper_label/000001',
          'label_7': 'paper_label/000002', 'label_8': 'paper_label/000000', 'label_9': 'paper_label/000003',
          'label_10': 'paper_label/000001', 'wire_end': 'wire_scrap/000000', 'wire_bit': 'wire_scrap/000001',
          'offcut': 'plastic_scrap/000000', 'parts_bag': 'zip_bag/000000', 'receipt': 'receipt/000000',
          'empty_can': 'drink_can/000001'}
EMPTY_BOTTLE = {'empty_bottle': 'water_bottle/000001'}

OBJECTS = {name: {'asset': f'objects/{asset}'}
           for group in (BACKGROUND, CONTAINERS, TOOLS, SOLDERING, GEAR, CABLES, DESK, LITTER, EMPTY_BOTTLE)
           for name, asset in group.items()}

VIEWS = {
    # From above the robot's head: the whole bench, the shelf over it and the floor on both sides, where the
    # bin stands.
    'top': {'position': (0., -.55, 2.7), 'target': (0., .15, .45), 'focal_length_mm': 15,
            'resolution': (1280, 960)},
    # The photo's camera: a person seated at the bench looking over it at the back wall.
    'photo': {'position': (.02, -.83, 1.28), 'target': (.03, .04, .80), 'focal_length_mm': 13.,
              'resolution': (1280, 969)},
    'wide': {'position': (1.55, -1.75, 1.75), 'target': (.20, .05, .70), 'focal_length_mm': 16,
             'resolution': (1280, 960)},
}
IMAGE_VIEW = 'top'

# The fixtures at the back, as in the photo: (x, y, yaw degrees), in this order.
FIXED = [('backboard', (0., .43, 0.)), ('monitor', (-.46, .26, 0.)), ('psu', (-.38, .01, 4.)),
         ('scope', (-.07, .25, 0.)), ('box', (.565, .215, 87.)), ('station', (.21, .22, -5.)),
         ('holder', (.37, .24, -90.))]
LAMP = (.60, -.03, -80.)        # stood last: its arm reaches over the bench
CHEST = (.45, -.24, -45.)       # front right, turned to face the robot
BIN = (.85, -.62, 0.)           # on the floor off the bench's front right corner, in sight from above
BOTTLE_IN_BIN = 0.                # yaw of the empty bottle lying in the bin: along its long side
# The photo, by name: (x, y, yaw degrees), dropped in this order so that what rests on something
# comes after it.
PHOTO = [
    ('empty_bottle', (.64, .31, 0.)),       # on the instrument box beside the can
    ('logbook', (.15, -.02, -8.)),
    ('meter', (.10, .02, 85.)),
    ('keyboard', (-.27, -.24, 13.)),
    ('graphing_calculator', (-.60, -.36, -70.)),
    ('board', (-.20, -.02, -8.)),
    ('adapter', (-.60, -.05, 85.)),
    ('connector', (-.12, -.06, 30.)),
    ('pocket_meter', (.02, -.13, 80.)),
    ('mouse', (.16, -.15, -30.)),
    ('hands', (.24, -.36, 200.)),
    ('driver_set', (.05, -.30, 75.)),
    ('heat_sink', (.18, -.25, 70.)),
    ('pliers', (.15, -.43, -40.)),
    ('scope_probe', (.00, -.40, -12.)),
    ('mini_board', (-.07, -.46, 10.)),
    ('wick', (-.13, -.43, 0.)),
    ('lead_red', (.08, -.27, 25.)),
    ('lead_black', (.12, -.36, -10.)),
    ('parts_bag', (.11, -.20, 20.)),
    ('label_3', (-.03, -.22, 5.)),
    ('label_1', (.26, -.40, 10.)),          # the slips at the front right
    ('label_2', (.30, -.39, -35.)),
    ('label_4', (.29, -.46, 80.)),
    ('label_5', (.33, -.43, 25.)),
    ('label_6', (.27, -.44, -60.)),
    ('label_7', (.35, -.47, 5.)),
    ('label_8', (.31, -.48, -20.)),
    ('label_9', (.37, -.35, 40.)),
    ('label_10', (.33, -.46, 100.)),
    ('wire_end', (-.20, -.44, 60.)),
    ('wire_bit', (.21, -.30, 20.)),
    ('offcut', (-.40, -.46, 0.)),
    ('receipt', (.45, -.24, 25.)),          # on the chest
    ('pen', (.16, -.01, 60.)),
    ('lcd', (-.40, .02, 0.)),               # on the power supply
    ('spool', (.18, .22, 0.)),              # on the station
    ('flux', (.25, .19, 0.)),
    ('network_cable', (.56, .14, 0.)),      # on the instrument box
    ('wireless_mouse', (.56, .14, 15.)),
    ('empty_can', (.56, .27, 30.)),
    ('calculator', (-.60, -.36, -70.)),     # on the graphing calculator
    ('cable', (-.12, -.22, 0.)),
]
JITTER = .01, math.radians(6.)       # how far each seed shifts (m) and turns (rad) the loose things of the photo

FRONT = ((-.60, -.47), (.20, -.10))     # the free front of the bench, left of the chest
FLAT = ['table', 'logbook', 'calculator', 'graphing_calculator', 'keyboard', 'driver_set']
# The iron resting in its stand, in the stand's frame: tip down in the spiral, handle rising toward +X.
IRON_IN_HOLDER = ((.0633, .019, .0773), (0., -.21695, 0., .97618))      # position, quaternion w x y z
DRAWERS = ((1, ('scope_probe', 'heat_sink', 'pen', 'wick')),
           (2, ('adapter', 'board', 'connector', 'lcd', 'mini_board')),
           (3, ('lead_red', 'lead_black', 'mouse')))


# -- fixtures and containers -------------------------------------------------------------
def fixtures(layout, lamp=True):
    for name, (x, y, yaw) in FIXED:
        layout.place(name, ((x, y), (x, y)), yaw=math.radians(yaw))
    if lamp:
        stand_lamp(layout)


def stand_lamp(layout):
    x, y, yaw = LAMP
    layout.place('lamp', ((x, y), (x, y)), yaw=math.radians(yaw))


def chest_at(layout, at=CHEST):
    x, y, yaw = at
    layout.place('chest', ((x, y), (x, y)), yaw=math.radians(yaw))


def on_floor(layout, name, xy, yaw=0., rotation=None):
    """``name`` resting on the floor at ``xy``: upright turned by ``yaw``, or in ``rotation`` (3x3) (on its side)."""
    if rotation is None:
        transform = layout.resting_pose(name, xy, yaw, on='floor')
    else:
        transform = yaw_pose(*xy, 0., 0.)
        transform[:3, :3] = rotation
        transform[2, 3] = -layout.geometry[name].lowest(rotation)
    if layout.penetration(name, transform) > PENETRATION_LIMIT:
        raise SamplingFailed(f'{name} does not fit on the floor at {np.round(xy, 3).tolist()}')
    layout.set_pose(name, transform, on='floor')


def bin_at(layout, at=BIN):
    x, y, yaw = at
    on_floor(layout, 'bin', (x, y), math.radians(yaw))


def lying(layout, name, yaw=None):
    """A resting orientation of ``name`` on its side (a knocked over can or bottle), turned by ``yaw``
    (random when ``None``)."""
    yaw = layout.rng.uniform(-math.pi, math.pi) if yaw is None else yaw
    side = yaw_pose(0., 0., 0., yaw)[:3, :3] @ UPRIGHT.T
    return layout.geometry[name].stable_rotation(side)


def quaternion_matrix(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def iron_in_holder(layout):
    """The iron resting tip down in the spiral of its stand (lifted clear of any contact)."""
    position, quaternion = IRON_IN_HOLDER
    local = np.eye(4)
    local[:3, :3], local[:3, 3] = quaternion_matrix(*quaternion), position
    transform = layout.poses['holder'] @ local
    for _ in range(10):
        if layout.penetration('iron', transform) <= PENETRATION_LIMIT:
            layout.set_pose('iron', transform, 'holder', loose=True)
            return
        transform[2, 3] += .001
    raise SamplingFailed('the iron does not fit its stand')


def hang(layout, name, peg):
    """Hang pliers ``name`` jaws up on peg ``peg`` (0 at the left), flat against the wall, the peg between
    the handles: lowered from high in the handles' gap until it rests on the peg."""
    board, geometry = layout.poses['backboard'], layout.geometry['backboard']
    points = geometry.points('bench_backboard.body.pegs')
    xs = np.sort(points[:, 0])
    groups = np.split(xs, np.flatnonzero(np.diff(xs) > .02) + 1)       # the pegs, left to right
    wall_y = geometry.bounds('bench_backboard.body.panel')[0][1]
    centre = board @ [groups[peg].mean(), wall_y, points[:, 2].mean(), 1.]
    transform = np.eye(4)
    transform[:3, :3] = board[:3, :3] @ np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]]).T    # x up, z out
    along, free = -.06, None        # the peg's place along the pliers (x), starting deep between the handles
    while True:
        transform[:3, 3] = centre[:3] - transform[:3, :3] @ [along, 0., 0.] + board[:3, :3] @ [0., -.0005, 0.]
        if layout.penetration(name, transform) > PENETRATION_LIMIT:
            break
        free, along = transform.copy(), along + .0005
    if free is None:
        raise SamplingFailed(f'{name} does not hang on peg {peg}')
    layout.set_pose(name, free, on='backboard', loose=True)     # swings until a handle meets the peg


def on_station(layout, names=('spool', 'flux')):
    """The solder reel and the flux tin side by side on top of the soldering station."""
    for name, xy in zip(names, ((-.032, 0.), (.035, .01))):
        layout.drop(name, xy, 0., frame='station')


def soldering(layout):
    iron_in_holder(layout)
    on_station(layout)


def hung_tools(layout):
    hang(layout, 'pliers', 0)
    hang(layout, 'cutters', 1)


def stored_gear(layout, drawers=DRAWERS):
    """Gear into the chest's drawers (closed): ``(drawer, names)`` pairs."""
    for drawer, names in drawers:
        drawer_contents(layout, drawer, list(names), yaw=(-.15, .15), cabinet='chest')


def on_chest(layout, names=('network_cable', 'cable', 'driver_set')):
    """The two coiled cables, one on the other, and the screwdriver set beside them, on the chest."""
    spots = {'network_cable': (-.07, 0.), 'cable': (-.07, 0.), 'driver_set': (.10, 0.)}
    for name in names:
        layout.drop(name, spots[name], math.pi / 2 if name == 'driver_set' else 0., frame='chest')


def binned(layout, at=BIN):
    """The bin on the floor with the empty bottle lying at the bottom and the rest of the litter on it."""
    bin_at(layout, at)
    layout.drop('empty_bottle', (0., 0.), rotation=lying(layout, 'empty_bottle', BOTTLE_IN_BIN), frame='bin')
    for name in LITTER:
        layout.scatter(name, ((-.07, -.05), (.07, .05)), yaw=(-math.pi, math.pi), frame='bin',
                       onto=['bin', 'empty_bottle', *LITTER])


def computer(layout, keyboard=(-.10, -.41), mouse=(.20, -.42)):
    layout.place('keyboard', (keyboard, keyboard))
    layout.place('wireless_mouse', (mouse, mouse))


def desk_stack(layout, at=(-.50, -.37)):
    """The logbook with the two calculators side by side on it."""
    layout.drop('logbook', at, 0.)
    layout.drop('calculator', (at[0], at[1] + .045), 0.)
    layout.drop('graphing_calculator', (at[0], at[1] - .045), 0.)


def tidy_desk(layout, mirror=False):
    """The desk things squared up: the calculators on the logbook at the front left, the two meters
    side by side behind it, the keyboard and wireless mouse at the front, the third hand in the middle;
    ``mirror``: left and right swapped (the chest on the left)."""
    if mirror:
        desk_stack(layout, at=(.50, -.37))
        layout.row(['meter', 'pocket_meter'], (.27, -.215), gap=.03)
        computer(layout, keyboard=(.10, -.41), mouse=(-.20, -.42))
        layout.place('hands', ((-.15, -.20), (-.15, -.20)), yaw=math.pi / 2)
    else:
        desk_stack(layout, at=(-.50, -.37))
        layout.row(['meter', 'pocket_meter'], (-.62, -.215), gap=.03)
        computer(layout, keyboard=(-.10, -.41), mouse=(.20, -.42))
        layout.place('hands', ((.05, -.20), (.05, -.20)), yaw=math.pi / 2)


def remaining(layout, names=None):
    return [name for name in (names or OBJECTS) if name not in layout.poses]


def strewn(layout, names, region=FRONT, onto=FLAT):
    """``names``, largest first, dropped at random over ``region`` onto ``onto`` (the table or flat things;
    None: anything)."""
    for name in sorted(names, key=lambda name: footprint_area(layout, name), reverse=True):
        layout.scatter(name, region, yaw=(-math.pi, math.pi), onto=onto, tries=200)


def tidy_back(layout):
    """What every tidy layout shares: the fixtures, the iron in its stand with the reel and flux on the
    station, the pliers on the pegs."""
    fixtures(layout)
    soldering(layout)
    hung_tools(layout)


# -- layout 0: the photo -----------------------------------------------------------------
def initialize(layout):
    rng = layout.rng
    fixtures(layout, lamp=False)
    chest_at(layout)
    shift, turn = JITTER
    for name, (x, y, yaw) in PHOTO:
        layout.drop(name, (x + rng.uniform(-shift, shift), y + rng.uniform(-shift, shift)),
                    math.radians(yaw) + rng.uniform(-turn, turn))
    iron_in_holder(layout)
    hang(layout, 'cutters', 1)
    bin_at(layout)
    stand_lamp(layout)


# -- layouts -----------------------------------------------------------------------------
def sorted_by_kind(layout):
    tidy_back(layout)
    chest_at(layout)
    stored_gear(layout)
    on_chest(layout)
    binned(layout)
    tidy_desk(layout)


def mirrored(layout):
    tidy_back(layout)
    chest_at(layout, (-.49, -.35, 0.))
    stored_gear(layout)
    on_chest(layout)
    binned(layout, at=(-.85, -.62, 0.))
    tidy_desk(layout, mirror=True)


def work_area(layout):
    tidy_back(layout)
    chest_at(layout)
    stored_gear(layout, ((1, ('scope_probe', 'heat_sink', 'pen')), (2, ('adapter', 'connector')), (3, ('mouse',))))
    on_chest(layout)
    binned(layout)
    # The job laid out in the middle: the dev board, the display and the small board side by side with
    # the wick in front and the third hand beside them, the clip leads below; the meters at the left, the
    # keyboard pushed to the front left with the wireless mouse, the logbook with the calculators beside.
    layout.place('board', ((.05, -.25), (.05, -.25)))
    layout.place('lcd', ((.13, -.25), (.13, -.25)))
    layout.place('mini_board', ((.19, -.25), (.19, -.25)))
    layout.place('wick', ((.05, -.16), (.05, -.16)))
    layout.place('hands', ((.16, -.15), (.16, -.15)), yaw=math.pi / 2)
    layout.place('lead_red', ((.00, -.40), (.00, -.40)))
    layout.place('lead_black', ((.15, -.33), (.15, -.33)))
    layout.row(['meter', 'pocket_meter'], (-.62, -.215), gap=.03)
    layout.place('keyboard', ((-.40, -.42), (-.40, -.42)))
    layout.place('wireless_mouse', ((-.12, -.47), (-.12, -.47)))
    desk_stack(layout, at=(-.14, -.245))


def in_rows(layout):
    fixtures(layout)
    soldering(layout)
    chest_at(layout)
    stored_gear(layout)
    on_chest(layout)
    binned(layout)
    # Not hung but lined up on the bench, everything in two straight rows: the meters, tools, third hand
    # and mouse at the back, the logbook, calculators and keyboard at the front.
    layout.row(['meter', 'pocket_meter'], (-.62, -.225), yaw=math.pi / 2, gap=.03)
    layout.row(['pliers', 'cutters', 'hands', 'wireless_mouse'], (-.25, -.225), yaw=math.pi / 2, gap=.03)
    layout.row(['logbook', 'calculator', 'graphing_calculator', 'keyboard'], (-.65, -.41),
               yaw=[0., math.pi / 2, math.pi / 2, 0.], gap=.02)


def all_stored(layout):
    tidy_back(layout)
    chest_at(layout)
    stored_gear(layout, ((1, ('scope_probe', 'heat_sink', 'pen', 'wick')),
                         (2, ('adapter', 'board', 'connector', 'lcd', 'mini_board')),
                         (3, ('lead_red', 'lead_black', 'mouse'))))
    on_chest(layout)
    binned(layout)
    # What stays out at the left and along the front edge, the middle of the bench clear: the logbook
    # with the calculators on it, the keyboard and the mouse in a row, the meters and the third hand
    # behind the logbook.
    layout.row(['logbook', 'keyboard', 'wireless_mouse'], (-.62, -.41), gap=.025)
    layout.drop('calculator', (layout.poses['logbook'][0, 3], -.365), 0.)
    layout.drop('graphing_calculator', (layout.poses['logbook'][0, 3], -.455), 0.)
    layout.row(['meter', 'pocket_meter'], (-.62, -.215), gap=.03)
    layout.place('hands', ((-.20, -.23), (-.20, -.23)), yaw=math.pi / 2)


def pile(layout):
    layout.free_landing = True      # a heap shifts as it settles
    fixtures(layout)
    chest_at(layout)
    bin_at(layout)
    on_floor(layout, 'empty_bottle', (BIN[0] + .22, BIN[1]))
    for name in sorted(remaining(layout), key=lambda name: footprint_area(layout, name), reverse=True):
        layout.scatter(name, ((-.32, -.40), (.12, -.14)), yaw=(-math.pi, math.pi), tries=200)


def wrong_places(layout):
    layout.free_landing = True
    fixtures(layout)
    chest_at(layout)
    bin_at(layout)
    # The pliers, a clip lead, the pen and the connector thrown in the bin; litter, the iron and the mouse
    # on top of the chest; the reel and flux on the power supply; the rest strewn over the bench.
    into(layout, ['pliers', 'lead_red', 'pen', 'connector'], 'bin', (.07, .05))
    for name in ('iron', 'label_1', 'label_2', 'offcut', 'parts_bag', 'mouse'):
        layout.scatter(name, ((-.10, -.06), (.10, .06)), yaw=(-math.pi, math.pi), frame='chest',
                       onto=['chest', *LITTER, 'iron'])
    layout.drop('spool', (-.04, -.03), 0., frame='psu')
    layout.drop('flux', (.04, -.03), 0., frame='psu')
    layout.place('keyboard', ((-.35, -.43), (-.05, -.30)), yaw=(-.5, .5))
    strewn(layout, [name for name in remaining(layout) if name != 'empty_bottle'])
    layout.place('empty_bottle', ((-.55, -.45), (.15, -.12)), yaw=(-math.pi, math.pi))


def half_done(layout):
    layout.free_landing = True
    fixtures(layout)
    soldering(layout)
    hang(layout, 'cutters', 1)
    chest_at(layout)
    bin_at(layout)
    into(layout, ['label_1', 'label_2', 'receipt'], 'bin', (.07, .05))
    # The left half done: the logbook and calculators, the keyboard, the meters in a row; the right half and
    # the chest still strewn with everything else.
    desk_stack(layout)
    layout.row(['meter', 'pocket_meter'], (-.62, -.215), gap=.03)
    layout.place('keyboard', ((-.14, -.41), (-.14, -.41)))
    strewn(layout, [name for name in remaining(layout) if name != 'empty_bottle'], region=((-.25, -.47), (.22, -.10)),
           onto=None)      # crowded: things land on each other
    on_floor(layout, 'empty_bottle', (BIN[0] + .22, BIN[1]))


def left_open(layout):
    layout.free_landing = True
    fixtures(layout)
    hung_tools(layout)
    chest_at(layout)
    binned(layout)
    # Put away carelessly: the three drawers left open with things tossed in, the iron and the rest strewn.
    # Bottom drawer first: an open drawer above covers the one below.
    drawer_contents(layout, 3, ['lead_red', 'mouse'], opening=.16, yaw=(-.3, .3), cabinet='chest')
    drawer_contents(layout, 2, ['board', 'heat_sink', 'connector'], opening=.12, cabinet='chest')
    drawer_contents(layout, 1, ['scope_probe', 'pen', 'spool'], opening=.15, cabinet='chest')
    layout.place('keyboard', ((-.20, -.42), (-.10, -.38)), yaw=(-.3, .3))
    strewn(layout, remaining(layout))


def knocked_over(layout):
    layout.free_landing = True      # spilled litter rolls
    tidy_back(layout)
    chest_at(layout)
    stored_gear(layout)
    on_chest(layout)
    tidy_desk(layout)
    # Tidy, except that the bin lies on its side on the floor with the litter and the empty bottle
    # spilled around it.
    toward_robot = yaw_pose(0., 0., 0., -math.pi / 2 + layout.rng.uniform(-.5, .5))[:3, :3]
    on_floor(layout, 'bin', BIN[:2], rotation=layout.geometry['bin'].stable_rotation(toward_robot @ UPRIGHT))
    for name in [*LITTER, 'empty_bottle']:
        rotation = lying(layout, name) if name in ('empty_can', 'empty_bottle') else None
        for _ in range(50):
            xy = (layout.rng.uniform(.62, 1.10), layout.rng.uniform(-.85, -.56))
            try:
                on_floor(layout, name, xy, layout.rng.uniform(-math.pi, math.pi), rotation)
                break
            except SamplingFailed:
                continue
        else:
            raise SamplingFailed(f'no room on the floor for {name}')


LAYOUTS = {
    1: {
        'quality': 'good', 'arrange': sorted_by_kind,
        'description': 'Sorted by kind: the pliers and cutters hanging on the pegs, the iron in its stand with '
                       'the solder reel and flux tin on the station; probe, heat sink, pen and wick in the top '
                       'drawer, the boards, display, adapter and connector in the middle one, the clip leads and '
                       'the mouse in the bottom one (all closed); the two coiled cables and the screwdriver set on '
                       'the chest; all litter and the empty bottle in the bin; the keyboard and wireless '
                       'mouse squared up at the front, the calculators on the logbook at the front left, the two '
                       'meters side by side behind them and the third hand in the middle.'},
    2: {
        'quality': 'good', 'arrange': mirrored,
        'description': 'Mirrored: the tool chest at the front left facing the robot and the bin on the floor to '
                       'the left; everything stored as when sorted, the keyboard and mouse at the front, the '
                       'calculators on the logbook at the front right, the meters and the third hand behind them.'},
    3: {
        'quality': 'good', 'arrange': work_area,
        'description': 'Ready to work on the boards: the dev board, the display and the small board side by side '
                       'in the middle with the wick in front and the third hand behind them, the clip leads below, '
                       'the two meters at hand on the left; the keyboard pushed to the front left, the logbook with '
                       'the calculators at the left edge; the iron in its stand, pliers on the pegs, the rest of '
                       'the gear in the drawers (closed), all litter in the bin.'},
    4: {
        'quality': 'good', 'arrange': in_rows,
        'description': 'In two straight rows across the bench: pliers, cutters, third hand and the two meters at '
                       'the back, the logbook, the two calculators, the keyboard and the mouse at the front; the '
                       'iron in its stand, gear in the drawers (closed), cables and screwdriver set on the chest, '
                       'all litter in the bin.'},
    5: {
        'quality': 'good', 'arrange': all_stored,
        'description': 'Everything put away: the pliers on the pegs, the iron in its stand, all the gear in the '
                       'drawers (closed), cables and screwdriver set on the chest, all litter in the '
                       'bin; only the logbook with the calculators on it, the keyboard and the mouse in a row at '
                       'the front edge, the two meters and the third hand behind the logbook at the left, the '
                       'middle of the bench completely clear.'},
    6: {
        'quality': 'bad', 'arrange': pile,
        'description': 'Everything that was on the bench, litter included, swept into one heap in the middle; the '
                       'empty bottle standing by the bin.'},
    7: {
        'quality': 'bad', 'arrange': wrong_places,
        'description': 'Things in the wrong places: the pliers, a clip lead, the pen and the connector in the bin, '
                       'litter, the iron and the mouse on top of the chest, the solder reel and flux on the power '
                       'supply, the rest strewn over the bench, the empty bottle among it.'},
    8: {
        'quality': 'bad', 'arrange': half_done,
        'description': 'Only half tidied: the iron in its stand, the cutters on a peg, the logbook with the '
                       'calculators, the keyboard and the meters in order on the left, a few labels in the bin; '
                       'the middle and right of the bench still strewn with everything else, litter included.'},
    9: {
        'quality': 'bad', 'arrange': left_open,
        'description': 'Put away carelessly: the three drawers left open with things tossed in, the iron and the '
                       'rest strewn over the bench, although the pliers hang on their pegs and the litter is in '
                       'the bin.'},
    10: {
        'quality': 'bad', 'arrange': knocked_over,
        'description': 'Tidy as when sorted, except that the bin lies knocked over on the floor with the litter '
                       'and the empty bottle spilled around it.'},
}
