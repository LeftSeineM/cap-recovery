"""Scene 000002: X1Pro (pivoting RM001 grippers) at a table with a three-drawer cabinet; a pen in its top drawer.

World frame: origin on the floor below the table centre, +Z up. The robot
stands in front of the table's -Y edge facing +Y, its centre ``GAP`` from the
edge (the chassis reaches under the table). The cabinet stands anywhere in the arms' reach (either arm), turned
to face the robot within +-15 degrees; the pen lies in the closed top drawer, in
the part that comes out when the drawer opens, along the drawer's width.
"""

import math

import numpy as np

from table_1000.scenes.layout import yaw_pose

TABLE = {'size': (1.36, 1.02), 'height': .755, 'top_thickness': .03, 'leg_size': .05,
         'center': (0., 0.), 'friction': (.5, .4)}

GAP = .29               # m from the base origin (the robot's centre) to the table edge

ROBOTS = {'x1pro_rm001': {
    'position': (0., -TABLE['size'][1] / 2 - GAP, 0.),
    'yaw': math.pi / 2,
    # Arms in the zero pose (folded), torso raised so the folded hands clear the table top.
    'joints': {
        'lift_joint': .40,
        'head_pitch_joint': .15,   # head camera looks at the whole table
    },
}}
ROBOT = 'x1pro_rm001'

OBJECTS = {
    'cabinet': {'asset': 'objects/cabinet/000000'},
    'pen': {'asset': 'objects/pen/000002'},
}

VIEWS = {
    'wide': {'position': (1.45, -1.55, 1.7), 'target': (0., -.2, .85), 'focal_length_mm': 20,
             'resolution': (960, 540)},
    'close': {'position': (.9, -.95, 1.35), 'target': (0., -.05, .85), 'focal_length_mm': 22,
              'resolution': (960, 540)},
}

# Cabinet origin: CABINET_AHEAD in front of the robot's centre and up to CABINET_SIDE across. The
# top drawer pulled out keeps clear of the folded hands, and the pen in it stays within the reach
# of a grasp from above (an IK sweep: about 0.68 m ahead of the centre); the bottom drawer pulled
# out keeps clear of the chest (about 0.48 m ahead at its height). Across, the base moves to where
# the arms reach.
CABINET_AHEAD, CABINET_SIDE = (.80, .86), .35
CABINET_TURN = math.radians(15)
# Pen in the top drawer (drawer frame = cabinet frame while closed): its centre over this region
# of the drawer floor, turned from the drawer's width by up to PEN_TURN, either way round. The
# region is the middle of the part that comes out, so a hand closing across the pen fits between
# the drawer's front wall and the housing.
# Released this far above the floor: exactly touching, a thin barrel against the floor can read as
# a deep penetration.
DROP = .001
TOP_FLOOR, PEN_REGION, PEN_TURN = 'cabinet.drawer1.bottom', ((-.03, -.06), (.03, -.035)), math.radians(10)


def initialize(layout):
    """Cabinet anywhere in reach, facing the robot within +-15 degrees; pen in the top drawer."""
    rng = layout.rng
    rx, ry = ROBOTS[ROBOT]['position'][:2]
    x, y = rx + rng.uniform(-CABINET_SIDE, CABINET_SIDE), ry + rng.uniform(*CABINET_AHEAD)
    face = math.atan2(y - ry, x - rx) - math.pi / 2
    layout.place('cabinet', region=((x, y), (x, y)), yaw=(face - CABINET_TURN, face + CABINET_TURN))
    cabinet = layout.poses['cabinet']
    floor = float(layout.geometry['cabinet'].bounds(TOP_FLOOR)[1][2])
    (px0, py0), (px1, py1) = PEN_REGION
    local = yaw_pose(rng.uniform(px0, px1), rng.uniform(py0, py1), 0.,
                     rng.uniform(-PEN_TURN, PEN_TURN) + rng.choice([0., math.pi]))
    pen = cabinet @ local
    pen[:3, :3] = layout.geometry['pen'].stable_rotation(pen[:3, :3])
    centre = layout.geometry['pen'].points() @ pen[:3, :3].T
    pen[:2, 3] -= (centre[:, :2].min(0) + centre[:, :2].max(0)) / 2   # the pen's middle, not its origin
    pen[2, 3] = cabinet[2, 3] + floor + DROP - layout.geometry['pen'].lowest(pen[:3, :3])
    layout.set_pose('pen', pen, on='cabinet')
