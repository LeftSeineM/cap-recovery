"""Scene 000000: X1Pro at a table with a three-drawer cabinet and a pen.

World frame: origin on the floor below the table centre, +Z up. The robot
stands in front of the table's -Y edge facing +Y. The cabinet's drawers pull
toward the robot (cabinet -Y = world -Y, turned by up to 15 degrees); the pen
lies on the table near the robot's right hand, in any direction.
"""

import math

TABLE = {'size': (1.36, 1.02), 'height': .755, 'top_thickness': .03, 'leg_size': .05,
         'center': (0., 0.), 'friction': (.5, .4)}

ROBOTS = {'x1pro': {
    'position': (-.03, -.80, 0.),
    'yaw': math.pi / 2,
    # Arms in the zero pose (folded, the real robot's usual start), torso raised
    # so the folded hands clear the table top; joints not listed are at zero.
    'joints': {
        'lift_joint': .40,
        'head_pitch_joint': .30,   # head camera looks at the workspace
    },
}}
ROBOT = 'x1pro'

OBJECTS = {
    'cabinet': {'asset': 'objects/cabinet/000000'},
    'pen': {'asset': 'objects/pen/000002'},
}

# Fixed views for the review videos, side by side; the initial states are pictured by the head camera.
VIEWS = {
    'wide': {'position': (1.45, -1.25, 1.6), 'target': (.05, -.25, .85), 'focal_length_mm': 22,
             'resolution': (960, 540)},
    'close': {'position': (.78, -.62, 1.25), 'target': (.17, -.24, .82), 'focal_length_mm': 26,
              'resolution': (960, 540)},
}


CABINET_YAW = math.radians(15)


def initialize(layout):
    """Cabinet on the table right of centre, drawers toward the robot, turned by up to +-15 degrees;
    pen in front of the robot's right hand in any direction, clear of the drawers' pull-out space.
    Origins vary by +-5 cm."""
    layout.place('cabinet', region=((.25, .01), (.35, .11)), yaw=(-CABINET_YAW, CABINET_YAW))
    layout.place('pen', region=((-.09, -.41), (.01, -.31)), yaw=(-math.pi, math.pi))
