"""Scene 000102: X1Pro at a dressing table, 30 objects; layouts 1–10 arrange them, five tidy and five not.

Modelled after a photo of a vanity table (reference.jpg: "Vanity table adorned with perfumes,
cosmetics, and a laptop in a cozy room setting" by Shixart1985, CC BY 2.0,
https://commons.wikimedia.org/wiki/File:Vanity_table_adorned_with_perfumes,_cosmetics,_and_a_laptop_in_a_cozy_room_setting.jpg):
a white tray of perfume and skincare bottles, a closed laptop and a glass vase of flowers.

World frame: origin on the floor below the table centre, +Z up. The robot stands in front of the
table's -Y edge facing +Y, as in scene 000100, and reaches the whole width of the table from the
front edge to about y = +0.10. In the back row stand the vanity tray on the left, the vase, the
brush cup and the tissue box in the middle and the three-drawer acrylic organiser on the right. The rest lies in the front area,
y from -0.47 to -0.10.

``initialize`` makes a messy dressing table: the back row jittered and turned, a few bottles left in
the tray, the others standing or lying anywhere in front, brushes partly in the cup, lipsticks and
nail polish scattered, the laptop somewhere in front, the top drawer sometimes open. ``LAYOUTS`` are arrangements of the same objects for judging tidiness.
"""

import math

from table_1000.scenes.arrange import (UPRIGHT, drawer_contents, footprint_area, line_up_in, stand, stand_in,
                                       strew)
from table_1000.scenes.layout import yaw_pose

TABLE = {'size': (1.36, 1.02), 'height': .755, 'top_thickness': .03, 'leg_size': .05,
         'center': (0., 0.), 'friction': (.5, .4)}

ROBOTS = {'x1pro': {
    'position': (-.03, -.80, 0.),
    'yaw': math.pi / 2,
    'joints': {'lift_joint': .70, 'head_pitch_joint': .60},
}}
ROBOT = 'x1pro'

CONTAINERS = {
    'tray': 'vanity_tray/000000',           # 0.36 x 0.23 x 0.055, open
    'brush_cup': 'cosmetic_cup/000000',     # clear, 0.08 across, 0.11 tall
    'organizer': 'makeup_organizer/000000',  # three drawers (inside 43, 61, 83 mm high), 0.30 x 0.20 x 0.23
    'trinket_dish': 'trinket_dish/000000',  # ceramic, 0.09 across
    'pad_jar': 'cotton_pad_jar/000000',     # lidded, 0.07 across, 0.095 tall
}
PERFUMES = {'perfume_blue': 'perfume_bottle/000000', 'perfume_amber': 'perfume_bottle/000001',
            'perfume_purple': 'perfume_bottle/000002', 'perfume_clear': 'perfume_bottle/000003'}
SKINCARE = {'lotion': 'lotion_bottle/000000', 'deodorant': 'deodorant/000000', 'serum': 'serum_bottle/000000',
            'oil': 'oil_bottle/000000'}
JARS = {'cream_white': 'cream_jar/000000', 'cream_pink': 'cream_jar/000001'}
LIPSTICKS = {'lipstick_gold': 'lipstick/000000', 'lipstick_black': 'lipstick/000001', 'lipstick_rose': 'lipstick/000002'}
POLISH = {'polish_red': 'nail_polish/000000', 'polish_nude': 'nail_polish/000001', 'polish_plum': 'nail_polish/000002'}
BRUSHES = {'powder_brush': 'makeup_brush/000000', 'eye_brush': 'makeup_brush/000001'}
HAIR = {'hairbrush': 'hairbrush/000000', 'comb': 'comb/000000'}
DECOR = {'vase': 'flower_vase/000000', 'diffuser': 'reed_diffuser/000000', 'tissues': 'tissue_box/000000'}
FLAT = {'laptop': 'laptop/000000', 'compact': 'compact_mirror/000000'}

OBJECTS = {name: {'asset': f'objects/{asset}'}
           for group in (CONTAINERS, PERFUMES, SKINCARE, JARS, LIPSTICKS, POLISH, BRUSHES, HAIR, DECOR, FLAT)
           for name, asset in group.items()}

VIEWS = {
    'top': {'position': (0., -.30, 2.05), 'target': (0., -.17, .755), 'focal_length_mm': 18,
            'resolution': (1280, 960)},
    'wide': {'position': (1.2, -1.35, 1.65), 'target': (0., -.12, .78), 'focal_length_mm': 20,
             'resolution': (960, 540)},
}
IMAGE_VIEW = 'top'

FRONT = ((-.45, -.47), (.45, -.10))
ON = ['table', 'laptop']                    # what strewn things may land on
BOTTLES = {**PERFUMES, **SKINCARE}
SMALL = {**LIPSTICKS, **POLISH}             # too tall to stand in the top drawer: they lie in it
CUP_SPOTS = [(-.015, -.015), (.015, .015), (-.015, .015), (.015, -.015)]
# The tray's floor between its walls: x within +-0.172, y within +-0.107 in its frame.
TRAY_INNER = (.172, .107)
BACK_ROW = {'tray': (-.36, .00), 'vase': (-.10, .06), 'brush_cup': (.06, .06), 'tissues': (.165, .06),
            'organizer': (.40, .04)}


# -- helpers -------------------------------------------------------------------------
def back_row(layout, jitter=0., turn=0., spots=BACK_ROW):
    """The back row at ``spots``; ``jitter`` (m) and ``turn`` (rad) randomize it."""
    def spot(x, y):
        return ((x - jitter, y - jitter), (x + jitter, y + jitter))

    yaw = (-turn, turn) if turn else 0.
    for name, xy in spots.items():
        layout.place(name, spot(*xy), yaw=yaw)


def tray_rows(layout, back, front, gap=.008):
    """Bottles standing in the tray in two rows along it: ``back`` (the tall ones) against its back
    wall, ``front`` in front of them."""
    hx, hy = TRAY_INNER
    line_up_in(layout, back, 'tray', (-hx + .004, hy - .045), gap=gap)
    line_up_in(layout, front, 'tray', (-hx + .004, -hy + .045), gap=gap)


def brushes_in_cup(layout, lean=(4., 12.)):
    stand_in(layout, list(BRUSHES), 'brush_cup', CUP_SPOTS, lean)


def drawers(layout, contents, opening=0.):
    """``{drawer: names}`` into the organiser's drawers; lipsticks and nail polish lie."""
    for drawer, names in contents.items():
        drawer_contents(layout, drawer, list(names), opening, yaw=(-.1, .1), lying=list(SMALL), cabinet='organizer')


def upright(layout, names, region):
    """Stand ``names`` upright at random in ``region`` (world x, y), each turned at random."""
    for name in names:
        layout.place(name, region, yaw=(-math.pi, math.pi))


def lying(layout, name):
    """A random resting orientation of ``name`` on its side (a knocked over bottle)."""
    side = yaw_pose(0., 0., 0., layout.rng.uniform(-math.pi, math.pi))[:3, :3] @ UPRIGHT.T
    return layout.geometry[name].stable_rotation(side)


def scatter_front(layout, names, region=FRONT):
    strew(layout, names, region, onto=ON)


def remaining(layout):
    return [name for name in OBJECTS if name not in layout.poses]


# -- the messy dressing table ----------------------------------------------------------------
def initialize(layout):
    rng = layout.rng
    layout.free_landing = True      # lying bottles and lipsticks roll a little
    back_row(layout, jitter=.03, turn=math.radians(15))
    layout.place('laptop', ((-.30, -.36), (.30, -.22)), yaw=(-.6, .6))
    for name in [name for name in BOTTLES if rng.random() < .35]:
        layout.scatter(name, ((-.13, -.07), (.13, .07)), yaw=(-math.pi, math.pi), frame='tray', onto='tray')
    stand_in(layout, [name for name in BRUSHES if rng.random() < .4], 'brush_cup', CUP_SPOTS, (4., 20.))
    if rng.random() < .4:
        drawers(layout, {1: rng.choice([*SMALL, 'compact'], 3, replace=False)}, rng.uniform(.05, .10))
    for name in [name for name in BOTTLES if name not in layout.poses]:
        if rng.random() < .3:
            layout.scatter(name, FRONT, rotation=lying(layout, name), onto='table')
        else:
            layout.place(name, ((-.45, -.45), (.45, -.12)), yaw=(-math.pi, math.pi))
    upright(layout, ['diffuser', 'pad_jar', 'trinket_dish', *JARS], ((-.45, -.45), (.45, -.12)))
    scatter_front(layout, remaining(layout))


# -- layouts ---------------------------------------------------------------------------------
def messy_back_row(layout):
    back_row(layout, jitter=.06, turn=math.radians(35))


def sorted_by_kind(layout):
    back_row(layout)
    tray_rows(layout, ['lotion', 'deodorant', 'oil', 'serum', 'perfume_purple', 'diffuser'],
              ['perfume_blue', 'perfume_amber', 'perfume_clear', 'cream_white', 'cream_pink'])
    brushes_in_cup(layout)
    drawers(layout, {1: [*LIPSTICKS, *POLISH], 2: ['comb', 'compact'], 3: ['hairbrush']})
    layout.row(['pad_jar', 'trinket_dish'], (.06, -.01), direction=-math.pi / 2, gap=.03)
    layout.place('laptop', ((-.12, -.33), (-.12, -.33)))


def work_mode(layout):
    """Laptop in the middle front for work, the beauty things all pushed back and stored."""
    back_row(layout, spots={**BACK_ROW, 'vase': (-.47, -.24)})
    tray_rows(layout, ['lotion', 'oil', 'serum', 'deodorant', 'diffuser'],
              ['perfume_blue', 'perfume_amber', 'perfume_purple', 'perfume_clear', 'pad_jar'])
    brushes_in_cup(layout)
    drawers(layout, {1: [*LIPSTICKS, *POLISH], 2: ['comb', 'compact', 'cream_pink'], 3: ['cream_white', 'trinket_dish']})
    layout.drop('hairbrush', (0., 0.), 0., frame='organizer')       # on top of the organiser
    layout.place('laptop', ((.02, -.30), (.02, -.30)))


def mirrored(layout):
    back_row(layout, spots={'organizer': (-.38, .03), 'brush_cup': (-.17, .06), 'vase': (-.03, .06),
                            'tissues': (.10, .06), 'tray': (.36, .00)})
    tray_rows(layout, ['perfume_purple', 'perfume_blue', 'perfume_amber', 'perfume_clear', 'diffuser'],
              ['lotion', 'deodorant', 'oil', 'serum'])
    brushes_in_cup(layout)
    drawers(layout, {1: [*POLISH, *LIPSTICKS], 2: ['comb', 'compact'], 3: ['hairbrush']})
    layout.row(['cream_white', 'cream_pink', 'pad_jar', 'trinket_dish'], (-.02, -.20), gap=.03)
    layout.place('laptop', ((.30, -.38), (.30, -.38)))


def on_display(layout):
    """Perfumes on show in a row at the back of the desk; the tray turned lengthwise holds the skincare."""
    stand(layout, {'organizer': (.38, -.08), 'vase': (.15, .06)})
    layout.place('tray', ((-.44, -.10), (-.44, -.10)), yaw=math.pi / 2)
    line_up_in(layout, ['lotion', 'deodorant', 'oil', 'serum', 'cream_white'], 'tray',
               (-TRAY_INNER[0] + .004, .045), gap=.008)
    line_up_in(layout, ['cream_pink', 'pad_jar', 'trinket_dish', 'diffuser'], 'tray',
               (-TRAY_INNER[0] + .004, -.045), gap=.008)
    layout.row(['perfume_purple', 'perfume_blue', 'perfume_amber', 'perfume_clear'], (-.27, .06), gap=.03)
    layout.place('brush_cup', ((.02, -.12), (.02, -.12)))
    brushes_in_cup(layout)
    layout.row(list(LIPSTICKS), (.08, -.12), gap=.02)
    layout.place('tissues', ((-.20, -.12), (-.20, -.12)))
    drawers(layout, {1: [*POLISH, 'compact'], 2: ['comb'], 3: ['hairbrush']})
    layout.place('laptop', ((-.05, -.33), (-.05, -.33)))


def all_stored(layout):
    back_row(layout)
    tray_rows(layout, ['lotion', 'oil', 'serum', 'perfume_purple', 'diffuser', 'pad_jar'],
              ['perfume_blue', 'perfume_amber', 'perfume_clear', 'cream_white', 'cream_pink'], gap=.006)
    brushes_in_cup(layout)
    drawers(layout, {1: [*LIPSTICKS, *POLISH], 2: ['comb', 'compact', 'trinket_dish'], 3: ['hairbrush']})
    drawer_contents(layout, 3, ['deodorant'], yaw=(-.1, .1), lying=['deodorant'], cabinet='organizer')
    layout.drop('laptop', (0., 0.), 0., frame='organizer')      # put away on top of the organiser


def pile(layout):
    messy_back_row(layout)
    layout.free_landing = True
    for name in sorted(remaining(layout), key=lambda name: footprint_area(layout, name), reverse=True):
        layout.scatter(name, ((-.18, -.40), (.18, -.16)), yaw=(-math.pi, math.pi))


def wrong_places(layout):
    messy_back_row(layout)
    layout.free_landing = True
    # Hair things and nail polish in the tray, a serum bottle and a lipstick in the brush cup, the brushes
    # shut in a drawer, the laptop on top of the organiser.
    for name in ('hairbrush', 'comb', *POLISH):
        layout.scatter(name, ((-.10, -.05), (.10, .05)), yaw=(-math.pi, math.pi), frame='tray',
                       onto=['tray', 'hairbrush', 'comb'])
    stand_in(layout, ['serum', 'lipstick_gold'], 'brush_cup', CUP_SPOTS[:2], (4., 15.))
    drawers(layout, {1: list(BRUSHES)})
    layout.drop('laptop', (0., 0.), 0., frame='organizer')
    upright(layout, [name for name in BOTTLES if name not in layout.poses], ((-.45, -.45), (.45, -.12)))
    scatter_front(layout, remaining(layout))


def half_done(layout):
    messy_back_row(layout)
    layout.free_landing = True
    tray_rows(layout, ['lotion', 'deodorant', 'oil'], ['perfume_blue', 'perfume_amber'])
    brushes_in_cup(layout)
    layout.place('laptop', ((-.32, -.32), (-.32, -.32)))
    region = ((-.10, -.45), (.45, -.12))
    upright(layout, ['perfume_purple', 'perfume_clear', 'serum'], region)
    scatter_front(layout, remaining(layout), region=region)


def knocked_over(layout):
    layout.free_landing = True
    back_row(layout)
    tray_rows(layout, ['lotion', 'oil', 'perfume_purple'], ['perfume_blue', 'perfume_clear', 'cream_white'])
    brushes_in_cup(layout)
    drawers(layout, {1: [*LIPSTICKS, *POLISH], 2: ['comb', 'compact'], 3: ['hairbrush']})
    layout.place('laptop', ((-.12, -.33), (-.12, -.33)))
    # A sweep of the arm: these bottles and jars lie toppled across the front right.
    for name in ('deodorant', 'serum', 'perfume_amber', 'diffuser', 'cream_pink', 'pad_jar', 'trinket_dish'):
        layout.scatter(name, ((.08, -.42), (.42, -.12)), rotation=lying(layout, name), onto='table')


def left_open(layout):
    layout.free_landing = True
    messy_back_row(layout)
    for name in BOTTLES:      # put in the tray any which way, some lying
        rotation = lying(layout, name) if layout.rng.random() < .4 else None
        layout.scatter(name, ((-.13, -.07), (.13, .07)), yaw=(-math.pi, math.pi), rotation=rotation, frame='tray',
                       onto=['tray', *BOTTLES])
    drawers(layout, {3: [*JARS, 'trinket_dish']}, opening=.13)
    drawers(layout, {2: ['comb', 'compact']}, opening=.10)
    drawers(layout, {1: [*LIPSTICKS, *POLISH]}, opening=.12)
    scatter_front(layout, remaining(layout), region=((-.30, -.42), (.10, -.12)))


LAYOUTS = {
    1: {
        'quality': 'good', 'arrange': sorted_by_kind,
        'description': 'Sorted by kind: the bottles standing in two rows in the tray, tall skincare, a perfume '
                       'and the diffuser at the back, perfumes and creams in front; brushes in the cup; lipsticks '
                       'and nail polish in the top drawer, comb and compact in the middle one, the hairbrush in '
                       'the bottom one (closed); pad jar and trinket dish in a column in front of the cup; the '
                       'laptop at the front.'},
    2: {
        'quality': 'good', 'arrange': mirrored,
        'description': 'The back row mirrored, organiser on the left, tray on the right with the perfumes and '
                       'diffuser in its back row and the skincare in front; brushes in the cup; the creams, pad '
                       'jar and dish in one row on the desk; small things in the drawers (closed); laptop at the '
                       'front right.'},
    3: {
        'quality': 'good', 'arrange': work_mode,
        'description': 'Cleared for work: the laptop alone in the middle front; every bottle, the diffuser and the '
                       'pad jar in the tray, brushes in the cup, all small things, creams and the dish in the three '
                       'drawers (closed), the hairbrush on top of the organiser; the vase moved to the front left '
                       'corner.'},
    4: {
        'quality': 'good', 'arrange': on_display,
        'description': 'Perfumes on show: the four perfumes in a row at the back of the desk; the tray turned '
                       'lengthwise on the left holding the skincare and jars; tissue box, brush cup and the three '
                       'lipsticks in a row in the middle; the laptop at the front; the rest in the drawers '
                       '(closed).'},
    5: {
        'quality': 'good', 'arrange': all_stored,
        'description': 'Everything stored: all bottles, the diffuser, creams and pad jar in the tray, brushes in '
                       'the cup, the small things, dish, hairbrush and deodorant in the drawers (closed), the laptop '
                       'put away on top of the organiser; the front of the table completely clear.'},
    6: {
        'quality': 'bad', 'arrange': pile,
        'description': 'Everything swept into one heap in the middle of the table.'},
    7: {
        'quality': 'bad', 'arrange': wrong_places,
        'description': 'Things in the wrong places: hairbrush, comb and nail polish in the tray, a serum bottle '
                       'and a lipstick in the brush cup, the brushes shut in the drawer, the laptop on top of the '
                       'organiser, the bottles standing about the front of the table.'},
    8: {
        'quality': 'bad', 'arrange': half_done,
        'description': 'Only half tidied: a few bottles in the tray, brushes in the cup, the laptop on the left; '
                       'the right half of the table still strewn with everything else.'},
    9: {
        'quality': 'bad', 'arrange': left_open,
        'description': 'Put away carelessly: bottles tossed into the tray, some lying, the three drawers left '
                       'open with things in them, the rest strewn over the middle of the table.'},
    10: {
        'quality': 'bad', 'arrange': knocked_over,
        'description': 'Mostly tidy, but a sweep of the arm has knocked over several bottles, the diffuser and '
                       'the jars, which lie on their sides across the front right.'},
}
