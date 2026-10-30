"""Scene 000101: X1Pro at an electronics repair workbench, 62 objects; layouts 1–10 arrange them, five tidy and five not.

Modelled after a photo of a cramped hobby electronics bench (reference.jpg: "Electronics workbench" by
James Bastow, CC BY-SA 2.0, https://commons.wikimedia.org/wiki/File:Electronics_workbench.jpg):
multimeters, pliers and cutters, a soldering station with the iron in its stand, a third hand, solder,
flux and desoldering wick, test leads and a scope probe, a keyboard, calculators, a dev board,
batteries, drinks, and the litter of a working bench: cut wire ends, paper label strips, an empty
parts bag, a receipt, an empty can and an empty bottle.

World frame: origin on the floor below the table centre, +Z up. The robot stands in front of the
table's -Y edge facing +Y, as in scene 000100, and reaches the whole width of the table from the
front edge to about y = +0.10. The bench's fixtures stand in a back row inside that reach: the tool
tray on the left, the parts organiser (six open compartments), the tool cup, the soldering station
with the iron stand in front of it, the third hand, and the three-drawer tool chest on the right.
The waste bin stands at the front left. The rest lies in the front area, y from -0.47 to -0.10.

``initialize`` makes a messy bench: fixtures jittered and turned in the back row, the iron stand
and the bin anywhere in front, the iron in its stand or lying about, a few tools left in the tray,
a few small parts in the parts box, the drinks standing or knocked over, the top drawer sometimes
open with something in it, the keyboard askew and everything else, litter included, strewn over
the front area. ``LAYOUTS`` are arrangements of the same objects for judging tidiness.
"""

import math

import numpy as np

from table_1000.scenes.arrange import ACROSS, UPRIGHT, drawer_contents, footprint_area, into, stand, stand_in, strew
from table_1000.scenes.layout import PENETRATION_LIMIT, SamplingFailed, yaw_pose

TABLE = {'size': (1.36, 1.02), 'height': .755, 'top_thickness': .03, 'leg_size': .05,
         'center': (0., 0.), 'friction': (.5, .4)}

ROBOTS = {'x1pro': {
    'position': (-.03, -.80, 0.),
    'yaw': math.pi / 2,
    'joints': {'lift_joint': .70, 'head_pitch_joint': .60},
}}
ROBOT = 'x1pro'

CONTAINERS = {
    'chest': 'tool_chest/000000',           # three drawers, 0.36 x 0.24 x 0.23
    'tray': 'tool_tray/000000',             # open, 0.33 x 0.22 x 0.06
    'parts': 'parts_box/000000',            # six open compartments, 0.21 x 0.13 x 0.04
    'cup': 'tool_cup/000000',               # 0.085 across, 0.12 tall
    'bin': 'trash_bin/000000',              # open, 0.14 x 0.11 x 0.12
    'station': 'soldering_station/000000',  # 0.14 x 0.13 x 0.09, flat top
    'holder': 'iron_holder/000000',         # spiral iron stand, 0.11 x 0.075
}
FIXTURES = {'hands': 'helping_hands/000000'}
TOOLS = {'meter': 'multimeter/000000', 'pocket_meter': 'multimeter/000001', 'pliers': 'pliers/000000',
         'cutters': 'pliers/000001'}
STANDING = {'screwdriver_blue': 'precision_screwdriver/000000', 'screwdriver_red': 'precision_screwdriver/000001',
            'probe_red': 'test_probe/000000', 'probe_black': 'test_probe/000001'}
BATTERIES = {f'battery_{i}': 'battery/000000' for i in range(1, 7)}
SMALL = {'mini_board': 'circuit_board/000001', 'insulating_tape': 'electrical_tape/000000',
         'lcd': 'lcd_module/000000', 'wick': 'desolder_wick/000000', **BATTERIES}
SOLDERING = {'iron': 'soldering_iron/000000', 'spool': 'solder_spool/000000', 'flux': 'flux_tin/000000',
             'tip_cleaner': 'tip_cleaner/000000'}
GEAR = {'board': 'circuit_board/000000', 'mouse': 'computer_mouse/000000', 'adapter': 'power_adapter/000000',
        'stripper': 'wire_stripper/000000', 'tweezers': 'tweezers/000000', 'heat_sink': 'heat_sink/000000',
        'connector': 'db_adapter/000000', 'scope_probe': 'scope_probe/000000', 'lead_red': 'clip_lead/000000',
        'lead_black': 'clip_lead/000001', 'pen': 'pen/000012', 'driver_set': 'screwdriver_set/000000'}
CABLES = {'cable': 'cable_coil/000000', 'network_cable': 'cable_coil/000001'}
DESK = {'calculator': 'calculator/000000', 'graphing_calculator': 'calculator/000001',
        'logbook': 'lab_notebook/000000', 'keyboard': 'keyboard/000000', 'wireless_mouse': 'computer_mouse/000001'}
DRINKS = {'can': 'drink_can/000000', 'bottle': 'water_bottle/000000'}
LITTER = {'label_1': 'paper_label/000000', 'label_2': 'paper_label/000001', 'label_3': 'paper_label/000002',
          'label_4': 'paper_label/000003', 'wire_end': 'wire_scrap/000000', 'wire_bit': 'wire_scrap/000001',
          'offcut': 'plastic_scrap/000000', 'parts_bag': 'zip_bag/000000', 'receipt': 'receipt/000000',
          'empty_can': 'drink_can/000001'}
EMPTY_BOTTLE = {'empty_bottle': 'water_bottle/000001'}      # too tall for the bin: it stands beside it
TRASH = {**LITTER, **EMPTY_BOTTLE}

OBJECTS = {name: {'asset': f'objects/{asset}'}
           for group in (CONTAINERS, FIXTURES, TOOLS, STANDING, SMALL, SOLDERING, GEAR, CABLES, DESK, DRINKS, TRASH)
           for name, asset in group.items()}

VIEWS = {
    'top': {'position': (0., -.30, 2.05), 'target': (0., -.17, .755), 'focal_length_mm': 18,
            'resolution': (1280, 960)},
    'wide': {'position': (1.2, -1.35, 1.65), 'target': (0., -.12, .78), 'focal_length_mm': 20,
             'resolution': (960, 540)},
}
IMAGE_VIEW = 'top'

FRONT = ((-.58, -.47), (.58, -.10))          # where loose things lie on a messy bench
FLAT = ['table', 'logbook', 'calculator', 'graphing_calculator', 'keyboard', 'driver_set']   # what strewn things may land on

# Back row positions (x, y) of the fixtures, left to right.
BACK_ROW = {'tray': (-.50, .00), 'parts': (-.215, .05), 'cup': (-.06, .06), 'station': (.08, .10),
            'hands': (.235, .09), 'chest': (.48, .04)}
HOLDER = (.08, -.035)           # in front of the station, the iron's handle toward the robot
BIN = (-.57, -.37)              # front left corner
# Spots in the tray's frame for the tools lying side by side across it, their long sides front to back.
TRAY_SPOTS = {'meter': (-.112, 0.), 'pocket_meter': (-.025, 0.), 'pliers': (.045, 0.), 'cutters': (.105, 0.)}
# Centres of the parts box's compartments, three columns by two rows, in its frame.
COMPARTMENTS = [(x, y) for y in (.0315, -.0315) for x in (-.069, 0., .069)]
CUP_SPOTS = [(-.017, -.017), (.017, .017), (-.017, .017), (.017, -.017)]
# The iron resting in its stand, in the stand's frame: tip down in the spiral, handle rising toward +X
# (from the stand's own test: outputs/scenes/000101/experiments/assets/iron_in_holder_check.py).
IRON_IN_HOLDER = ((.0633, .019, .0773), (0., -.21695, 0., .97618))      # position, quaternion w x y z
# Each drawer's contents when the gear is put away, top drawer first.
DRAWERS = ((1, ('scope_probe', 'heat_sink', 'tweezers', 'pen')),
           (2, ('stripper', 'adapter', 'board', 'connector')),
           (3, ('lead_red', 'lead_black', 'mouse')))


# -- helpers -------------------------------------------------------------------------
def quaternion_matrix(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def back_row(layout, spots=None, jitter=0., turn=0.):
    """The fixtures at ``spots`` (default :data:`BACK_ROW`); ``jitter`` (m) and ``turn`` (rad) randomize them."""
    for name, (x, y) in (spots or BACK_ROW).items():
        facing = math.pi / 2 if name == 'hands' else 0.        # the third hand stands sideways, lens to the left
        layout.place(name, ((x - jitter, y - jitter), (x + jitter, y + jitter)), yaw=(facing - turn, facing + turn))


def holder_at(layout, xy=HOLDER, yaw=-math.pi / 2):
    layout.place('holder', (xy, xy), yaw=yaw)


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


def on_station(layout, names=('spool', 'flux')):
    """The solder reel and the flux tin side by side on top of the soldering station."""
    for name, xy in zip(names, ((-.032, 0.), (.035, .01))):
        layout.drop(name, xy, 0., frame='station')


def soldering_corner(layout, holder=HOLDER, cleaner=(.175, -.06)):
    """Iron in its stand in front of the station, reel and flux on the station, tip cleaner beside the stand."""
    holder_at(layout, holder)
    iron_in_holder(layout)
    on_station(layout)
    layout.place('tip_cleaner', (cleaner, cleaner))


def tools_in_tray(layout, names=tuple(TOOLS), yaw=ACROSS):
    """The meters and pliers lying side by side in the tray, long sides front to back."""
    for name in names:
        layout.drop(name, TRAY_SPOTS[name], yaw, frame='tray')


def in_compartments(layout, groups):
    """Each group of names into the next compartment of the parts box, lying side by side along its X."""
    for group, (x, y) in zip(groups, COMPARTMENTS):
        for name, dy in zip(group, (np.arange(len(group)) - (len(group) - 1) / 2) * .018):
            layout.drop(name, (x, y + dy), 0., frame='parts')


def boxed_small(layout):
    in_compartments(layout, [['battery_1', 'battery_2', 'battery_3'], ['battery_4', 'battery_5', 'battery_6'],
                             ['mini_board'], ['insulating_tape'], ['lcd'], ['wick']])


def standing_in_cup(layout, names=tuple(STANDING), lean=(4., 12.)):
    stand_in(layout, list(names), 'cup', CUP_SPOTS, lean)


def stored_gear(layout, drawers=DRAWERS):
    """Gear into the chest's drawers (closed): ``(drawer, names)`` pairs."""
    for drawer, names in drawers:
        drawer_contents(layout, drawer, list(names), yaw=(-.15, .15), cabinet='chest')


def on_chest(layout):
    """The two coiled cables, one on the other, and the screwdriver set side by side on the chest."""
    layout.drop('network_cable', (-.07, 0.), 0., frame='chest')
    layout.drop('cable', (-.07, 0.), 0., frame='chest')
    layout.drop('driver_set', (.10, 0.), ACROSS, frame='chest')


def binned(layout, at=BIN, bottle=(-.60, -.205)):
    """The litter in the bin, the empty can at the bottom; the empty bottle standing beside it."""
    layout.place('bin', (at, at))
    layout.drop('empty_can', (0., 0.), 0., frame='bin')
    for name in LITTER:
        if name != 'empty_can':
            layout.scatter(name, ((-.035, -.025), (.035, .025)), yaw=(-math.pi, math.pi), frame='bin',
                           onto=['bin', *LITTER])
    layout.place('empty_bottle', (bottle, bottle))


def computer(layout, keyboard=(.02, -.40), mouse=(.33, -.40)):
    layout.place('keyboard', (keyboard, keyboard))
    layout.place('wireless_mouse', (mouse, mouse))


def desk_stack(layout, at=(-.33, -.355)):
    """The logbook with the two calculators side by side on it."""
    layout.drop('logbook', at, 0.)
    layout.drop('calculator', (at[0], at[1] + .045), 0.)
    layout.drop('graphing_calculator', (at[0], at[1] - .045), 0.)


def drinks_at(layout, start=(.47, -.22)):
    layout.row(['bottle', 'can'], start, gap=.03)


def scatter_front(layout, names, region=FRONT):
    strew(layout, names, region, onto=FLAT)


def lying(layout, name):
    """A random resting orientation of ``name`` on its side (a knocked over can or bottle)."""
    side = yaw_pose(0., 0., 0., layout.rng.uniform(-math.pi, math.pi))[:3, :3] @ UPRIGHT.T
    return layout.geometry[name].stable_rotation(side)


def remaining(layout, names=None):
    return [name for name in (names or OBJECTS) if name not in layout.poses]


def tidy_bench(layout):
    """The parts every tidy state shares: tools in the tray, small parts boxed, the cup filled."""
    tools_in_tray(layout)
    boxed_small(layout)
    standing_in_cup(layout)


# -- the messy bench -------------------------------------------------------------------
def initialize(layout):
    rng = layout.rng
    layout.free_landing = True      # batteries, pens and the knocked over drinks roll
    back_row(layout, jitter=.03, turn=math.radians(15))
    layout.place('holder', ((-.05, -.20), (.30, -.04)), yaw=(-math.pi, math.pi))
    if rng.random() < .5:
        iron_in_holder(layout)
    layout.place('bin', ((-.60, -.42), (.60, -.30)), yaw=(-.4, .4))
    in_bin = [name for name in LITTER if rng.random() < .2]
    into(layout, in_bin, 'bin', (.035, .025))
    on_station(layout, [name for name in ('spool', 'flux') if rng.random() < .4])
    in_tray = [name for name in TOOLS if rng.random() < .3]
    tools_in_tray(layout, in_tray, yaw=ACROSS + rng.uniform(-.15, .15))
    standing_in_cup(layout, list(rng.choice(list(STANDING), rng.integers(0, 4), replace=False)), lean=(4., 20.))
    boxed = [name for name in SMALL if rng.random() < .3]
    in_compartments(layout, [[name] for name in rng.permutation(boxed)])
    if rng.random() < .4:
        candidates = remaining(layout, ['adapter', 'tweezers', 'heat_sink', 'connector', 'pen', 'mini_board'])
        drawer_contents(layout, 1, list(rng.choice(candidates, 2, replace=False)), rng.uniform(.06, .12),
                        cabinet='chest')
    layout.place('keyboard', ((-.30, -.43), (.30, -.32)), yaw=(-.35, .35))
    for name in [*DRINKS, 'empty_bottle']:
        if name in layout.poses:
            continue
        if rng.random() < .25:
            layout.scatter(name, FRONT, rotation=lying(layout, name), onto='table')
        else:
            layout.place(name, ((-.50, -.45), (.50, -.12)), yaw=(-math.pi, math.pi))
    # Largest first, so the small things can still find room among (or on) the big ones.
    for name in sorted(remaining(layout), key=lambda name: footprint_area(layout, name), reverse=True):
        layout.scatter(name, FRONT, yaw=(-math.pi, math.pi), onto=FLAT, tries=200)


# -- layouts ---------------------------------------------------------------------------------
def messy_back_row(layout):
    back_row(layout, jitter=.05, turn=math.radians(30))


def sorted_by_kind(layout):
    back_row(layout)
    tidy_bench(layout)
    soldering_corner(layout)
    stored_gear(layout)
    on_chest(layout)
    binned(layout)
    computer(layout)
    desk_stack(layout)
    drinks_at(layout)


def ready_to_solder(layout):
    back_row(layout)
    tools_in_tray(layout, ['meter', 'cutters'])
    in_compartments(layout, [['battery_1', 'battery_2', 'battery_3'], ['battery_4', 'battery_5', 'battery_6'],
                             ['mini_board'], ['insulating_tape']])
    standing_in_cup(layout)
    soldering_corner(layout)
    # The job laid out in front of the stand: the board in the middle with the display beside it, wick,
    # tweezers, pliers, pocket meter and the two clip leads at hand; the keyboard pushed to the right.
    layout.place('board', ((-.07, -.26), (-.07, -.26)))
    layout.place('lcd', ((.03, -.26), (.03, -.26)))
    layout.place('wick', ((-.07, -.16), (-.07, -.16)))
    layout.place('tweezers', ((-.15, -.26), (-.15, -.26)), yaw=ACROSS)
    layout.row(['pocket_meter', 'pliers'], (-.46, -.20), yaw=ACROSS, gap=.03)
    layout.row(['lead_red', 'lead_black'], (-.08, -.33), direction=-math.pi / 2, yaw=math.pi / 2, gap=.02)
    stored_gear(layout, ((1, ('scope_probe', 'heat_sink', 'pen')), (2, ('stripper', 'adapter', 'connector')),
                         (3, ('mouse',))))
    on_chest(layout)
    binned(layout)
    computer(layout, keyboard=(.25, -.40), mouse=(.25, -.27))
    desk_stack(layout, at=(-.36, -.385))
    drinks_at(layout, start=(.47, -.22))


def chest_left(layout):
    back_row(layout, {'chest': (-.48, .04), 'hands': (-.235, .09), 'station': (-.08, .10), 'cup': (.06, .06),
                      'parts': (.215, .05), 'tray': (.50, .00)})
    tidy_bench(layout)
    soldering_corner(layout, holder=(-.08, -.035), cleaner=(.02, -.06))
    stored_gear(layout)
    on_chest(layout)
    binned(layout, at=(.57, -.37), bottle=(.60, -.205))
    computer(layout, keyboard=(-.05, -.40), mouse=(-.36, -.40))
    desk_stack(layout, at=(.33, -.355))
    layout.row(['can', 'bottle'], (-.55, -.24), gap=.03)


def in_columns(layout):
    back_row(layout, {'chest': (.49, .04), 'parts': (-.18, .07), 'cup': (-.02, .07), 'station': (.12, .10),
                      'hands': (.235, .09)})
    layout.place('tray', ((-.52, -.10), (-.52, -.10)), yaw=math.pi / 2)
    tidy_bench(layout)       # the tray turned: the tools across it now lie along the world X
    soldering_corner(layout, holder=(.12, -.035), cleaner=(.22, -.06))
    stored_gear(layout)
    on_chest(layout)
    binned(layout, at=(-.30, -.38), bottle=(-.30, -.25))
    layout.row(['logbook', 'calculator', 'graphing_calculator'], (-.06, -.04), direction=-math.pi / 2,
               yaw=math.pi / 2, gap=.02)
    computer(layout, keyboard=(.30, -.40), mouse=(.30, -.27))
    layout.row(['bottle', 'can'], (.43, -.22), gap=.03)


def all_stored(layout):
    back_row(layout)
    tidy_bench(layout)
    soldering_corner(layout)
    stored_gear(layout)
    on_chest(layout)
    binned(layout)
    # What stays out lined up in one row along the front edge: the logbook with the calculators on it,
    # the keyboard and mouse, the drinks.
    end = layout.row(['logbook', 'keyboard', 'wireless_mouse'], (-.46, -.40), gap=.025)
    layout.row(['bottle', 'can'], (end[0], -.40), gap=.025)
    layout.drop('calculator', (layout.poses['logbook'][0, 3], -.355), 0.)
    layout.drop('graphing_calculator', (layout.poses['logbook'][0, 3], -.445), 0.)


def pile(layout):
    messy_back_row(layout)
    layout.place('holder', ((.02, -.08), (.18, -.04)), yaw=(-math.pi, math.pi))
    layout.place('bin', ((-.60, -.42), (-.50, -.30)))
    # The drinks and the empty bottle standing behind the heap, everything else swept into it.
    for name in [*DRINKS, 'empty_bottle', 'empty_can']:
        layout.place(name, ((.48, -.40), (.60, -.16)), yaw=(-math.pi, math.pi))
    layout.free_landing = True      # a heap shifts as it settles
    for name in sorted(remaining(layout), key=lambda name: footprint_area(layout, name), reverse=True):
        layout.scatter(name, ((-.34, -.39), (.30, -.18)), yaw=(-math.pi, math.pi))


def wrong_places(layout):
    messy_back_row(layout)
    holder_at(layout, (.10, -.06), yaw=-math.pi / 2 + .4)
    layout.place('bin', ((-.60, -.42), (-.50, -.30)))
    layout.free_landing = True
    # Drinks and the iron among the tools in the tray, litter in the parts box and the tool cup,
    # batteries and leads in the bin, the board on the chest, the pen stuck in the iron stand.
    layout.drop('bottle', (-.06, 0.), rotation=lying(layout, 'bottle'), frame='tray')
    layout.drop('can', (.10, 0.), frame='tray')
    layout.drop('iron', (-.04, .07), 0., frame='tray')
    for name in ('label_1', 'label_2', 'wire_end', 'offcut', 'parts_bag'):
        layout.scatter(name, ((-.08, -.04), (.08, .04)), yaw=(-math.pi, math.pi), frame='parts',
                       onto=['parts', *LITTER])
    for name in ('label_3', 'label_4', 'wire_bit'):
        layout.scatter(name, ((-.012, -.012), (.012, .012)), yaw=(-math.pi, math.pi), frame='cup')
    into(layout, [*list(BATTERIES)[:4], 'lead_red'], 'bin', (.03, .02))
    layout.drop('board', (0., 0.), 0., frame='chest')
    layout.drop('tweezers', (.03, 0.), 0., frame='chest')
    layout.place('keyboard', ((-.30, -.43), (.30, -.32)), yaw=(-.35, .35))
    scatter_front(layout, sorted(remaining(layout), key=lambda name: footprint_area(layout, name), reverse=True),
                  region=((-.48, -.47), (.55, -.12)))


def half_done(layout):
    messy_back_row(layout)
    holder_at(layout, (.08, -.06))
    iron_in_holder(layout)
    layout.place('bin', ((-.57, -.37), (-.57, -.37)))
    layout.free_landing = True
    tools_in_tray(layout)
    standing_in_cup(layout, ['screwdriver_blue', 'screwdriver_red'])
    in_compartments(layout, [['battery_1', 'battery_2', 'battery_3'], ['mini_board']])
    on_chest(layout)
    into(layout, ['label_1', 'label_2', 'receipt'], 'bin', (.035, .025))
    computer(layout, keyboard=(-.25, -.40), mouse=(-.25, -.27))
    layout.drop('logbook', (-.42, -.20), 0.)
    layout.drop('calculator', (-.42, -.20), 0.)
    scatter_front(layout, remaining(layout), region=((-.08, -.47), (.55, -.12)))


def left_open(layout):
    layout.free_landing = True
    messy_back_row(layout)
    holder_at(layout, (.08, -.045))
    binned(layout)
    for name in TOOLS:      # dropped into the tray any which way
        layout.scatter(name, ((-.08, -.04), (.08, .04)), yaw=(-math.pi, math.pi), frame='tray', onto=['tray', *TOOLS])
    for name in BATTERIES:
        layout.scatter(name, ((-.07, -.035), (.07, .035)), yaw=(-math.pi, math.pi), frame='parts')
    drawer_contents(layout, 3, ['board', 'mouse', 'lead_red'], opening=.16, yaw=(-.3, .3), cabinet='chest')
    drawer_contents(layout, 2, ['spool', 'adapter', 'insulating_tape', 'heat_sink'], opening=.12, cabinet='chest')
    drawer_contents(layout, 1, ['calculator', 'mini_board', 'flux'], opening=.15, cabinet='chest')
    layout.place('keyboard', ((-.05, -.42), (.05, -.38)), yaw=(-.3, .3))
    scatter_front(layout, remaining(layout), region=((-.42, -.47), (.20, -.12)))


def knocked_over(layout):
    layout.free_landing = True      # spilled litter, batteries, probes and the can roll
    back_row(layout)
    tools_in_tray(layout)
    soldering_corner(layout)
    stored_gear(layout)
    on_chest(layout)
    computer(layout)
    desk_stack(layout)
    in_compartments(layout, [['mini_board'], ['insulating_tape'], ['lcd'], ['wick']])
    # The bin knocked over with its litter spilled, the tool cup on its side with the screwdrivers and
    # probes spilled, the drinks knocked over, the batteries rolled out over the bench.
    toward_robot = yaw_pose(0., 0., 0., -math.pi / 2 + layout.rng.uniform(-.5, .5))[:3, :3]
    layout.drop('bin', (-.57, -.30), rotation=layout.geometry['bin'].stable_rotation(toward_robot @ UPRIGHT))
    for name in TRASH:
        rotation = lying(layout, name) if name in ('empty_can', 'empty_bottle') else None
        layout.scatter(name, ((-.62, -.47), (-.40, -.16)), yaw=(-math.pi, math.pi), rotation=rotation)
    layout.drop('cup', (-.06 + layout.rng.uniform(-.03, .03), .0),
                rotation=layout.geometry['cup'].stable_rotation(toward_robot @ UPRIGHT))
    for name in STANDING:
        layout.scatter(name, ((-.20, -.30), (.00, -.12)), yaw=(-math.pi, math.pi), onto='table')
    for name in DRINKS:
        layout.scatter(name, ((.30, -.33), (.55, -.15)), rotation=lying(layout, name), onto='table')
    for name in BATTERIES:
        layout.scatter(name, ((-.25, -.25), (.25, -.12)), yaw=(-math.pi, math.pi), onto='table')


LAYOUTS = {
    1: {
        'quality': 'good', 'arrange': sorted_by_kind,
        'description': 'Sorted by kind: meters and pliers side by side in the tool tray; batteries, the small '
                       'boards, tape and desoldering wick in the parts box compartments; screwdrivers and probes '
                       'standing in the tool cup; the iron in its stand in front of the soldering station with the '
                       'solder reel and flux tin on the station and the tip cleaner beside it; the remaining gear in '
                       'the three drawers (closed), the coiled cables and the screwdriver set on the chest; all '
                       'litter in the bin with the empty bottle beside it; keyboard and mouse squared up at the '
                       'front, the calculators on the logbook at the front left, the two drinks at the front right.'},
    2: {
        'quality': 'good', 'arrange': chest_left,
        'description': 'The bench mirrored: the tool chest and third hand on the left, the soldering station and '
                       'iron stand in the middle, the tool tray on the right; everything stored as when sorted, the '
                       'bin and empty bottle at the front right, the calculators on the logbook in front of the tray, '
                       'the keyboard and mouse at the front left of the middle, the drinks at the front left.'},
    3: {
        'quality': 'good', 'arrange': ready_to_solder,
        'description': 'Ready to work on the board: the iron in its stand, reel and flux on the station; the board '
                       'and the display module in front of the stand with the wick, tweezers, pliers, pocket meter '
                       'and the two clip leads neatly at hand; the keyboard pushed to the right; everything else '
                       'stored, the litter in the bin.'},
    4: {
        'quality': 'good', 'arrange': in_columns,
        'description': 'In columns: the tool tray turned lengthwise along the left edge with the meters and pliers, '
                       'the bin and empty bottle in front of the parts box, logbook and calculators in a column down '
                       'the middle, the keyboard and mouse at the front right, the soldering corner and the chest in '
                       'the back; gear in the drawers (closed).'},
    5: {
        'quality': 'good', 'arrange': all_stored,
        'description': 'Everything put away and the rest lined up along the front edge: tools in the tray, small '
                       'parts in the parts box, screwdrivers and probes in the cup, the iron in its stand, gear in '
                       'the drawers (closed), cables and the screwdriver set on the chest, all litter in the bin; '
                       'the logbook with the calculators on it, the keyboard, the mouse and the two drinks in one '
                       'row at the front, the middle of the bench completely clear.'},
    6: {
        'quality': 'bad', 'arrange': pile,
        'description': 'Everything, litter included, swept into one heap in the middle of the bench, the drinks '
                       'and the empty can and bottle standing beside it.'},
    7: {
        'quality': 'bad', 'arrange': wrong_places,
        'description': 'Things in the wrong places: the drinks and the iron in the tool tray, litter in the parts '
                       'box and the tool cup, batteries and a clip lead in the bin, the board and tweezers on top of '
                       'the chest, the rest strewn over the bench.'},
    8: {
        'quality': 'bad', 'arrange': half_done,
        'description': 'Only half tidied: the iron in its stand, the meters and pliers in the tray, the '
                       'screwdrivers in the cup, a few batteries boxed, the cables on the chest, a few labels in the '
                       'bin, keyboard and logbook on the left; the right half of the bench still strewn with '
                       'everything else, litter included.'},
    9: {
        'quality': 'bad', 'arrange': left_open,
        'description': 'Put away carelessly: tools tossed into the tray in a jumble, batteries thrown into the '
                       'parts box, the three drawers left open with things in them, the iron and the rest strewn '
                       'over the middle of the bench although the litter is in the bin.'},
    10: {
        'quality': 'bad', 'arrange': knocked_over,
        'description': 'Tidy except that the bin lies knocked over with its litter and the empty bottle spilled, the '
                       'tool cup lies on its side with the screwdrivers and probes spilled, both drinks lie on their '
                       'sides and the batteries have rolled over the bench.'},
}
