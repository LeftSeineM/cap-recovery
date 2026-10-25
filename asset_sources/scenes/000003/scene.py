"""A pen anywhere along the working table edge, and a front-facing drawer cabinet.

The robot starts at the middle of the -Y edge. Both objects span the full
usable width; the pen stays in the front reach band. The cabinet's complete
drawer sweep stays over the table, facing this edge, and clear of the pen.
These are layout constraints, not an IK or navigation success filter.
"""

import math

import numpy as np

from table_1000.scenes.layout import PENETRATION_LIMIT, SamplingFailed, yaw_pose

TABLE = {'size': (1.36, 1.02), 'height': .755, 'top_thickness': .03, 'leg_size': .05,
         'center': (0., 0.), 'friction': (.5, .4)}
ROBOT = {'asset': 'robots/x1pro_rm001', 'position': (0., -.80, 0.), 'yaw': math.pi / 2,
         'joints': {'lift_joint': .40, 'head_pitch_joint': .15}}
OBJECTS = {'cabinet': {'asset': 'objects/cabinet/000000'}, 'pen': {'asset': 'objects/pen/000002'}}
VIEWS = {'wide': {'position': (1.55, -1.65, 1.8), 'target': (0., -.15, .85),
                  'focal_length_mm': 20, 'resolution': (960, 540)}}

DRAWER = 'cabinet.drawer3.slide'
CABINET_Y = (0., .06)
CABINET_TURN = math.radians(30)
TABLE_MARGIN, PEN_MARGIN = .01, .04
WORK_DEPTH = .26     # the whole pen lies in the first 26 cm from the working edge


def drawer_sweep(layout):
    """World collision vertices of the cabinet with the bottom drawer closed and fully open."""
    geometry, frame = layout.geometry['cabinet'], layout.poses['cabinet']
    lower, upper = geometry.joints[DRAWER]['limits']
    return np.vstack([hull.vertices for value in (lower, upper)
                      for hull in geometry.hulls(geometry.body_poses(frame, {DRAWER: value}))])


def initialize(layout):
    """Sample the whole usable width, keeping all operations on the robot's side of the table."""
    rng, geometry = layout.rng, layout.geometry['cabinet']
    centre, half = np.asarray(TABLE['center']), np.asarray(TABLE['size']) / 2
    # The orientation-dependent complete sweep, rather than a hand-written
    # cabinet size, determines the usable width at this yaw.
    rotation = yaw_pose(0., 0., 0., rng.uniform(-CABINET_TURN, CABINET_TURN))[:3, :3]
    lower, upper = geometry.joints[DRAWER]['limits']
    points = np.vstack([h.vertices for value in (lower, upper)
                        for h in geometry.hulls(geometry.body_poses(np.eye(4), {DRAWER: value}))]) @ rotation.T
    low, high = points[:, :2].min(0), points[:, :2].max(0)
    x = rng.uniform(centre[0] - half[0] + TABLE_MARGIN - low[0],
                    centre[0] + half[0] - TABLE_MARGIN - high[0])
    cabinet = layout.resting_pose('cabinet', (x, rng.uniform(*CABINET_Y)),
                                  math.atan2(rotation[1, 0], rotation[0, 0]))
    layout.set_pose('cabinet', cabinet)
    swept = drawer_sweep(layout)[:, :2]
    if np.any(swept.min(0) < centre - half + TABLE_MARGIN) or np.any(swept.max(0) > centre + half - TABLE_MARGIN):
        raise SamplingFailed('cabinet drawer sweep would leave the tabletop')
    # A conservative projected sweep also keeps the pen out of the drawer's
    # future path. No base must go behind the table to reach either target.
    blocked_low, blocked_high = swept.min(0) - TABLE_MARGIN, swept.max(0) + TABLE_MARGIN
    front = centre[1] - half[1]
    for _ in range(50):
        pen = layout.resting_pose('pen', (0., 0.), rng.uniform(-math.pi, math.pi))
        outline = layout.geometry['pen'].points() @ pen[:3, :3].T
        low, high = outline[:, :2].min(0), outline[:, :2].max(0)
        minimum = np.array([centre[0] - half[0] + PEN_MARGIN, front + PEN_MARGIN]) - low
        maximum = np.array([centre[0] + half[0] - PEN_MARGIN, front + WORK_DEPTH]) - high
        if np.any(minimum > maximum):
            raise SamplingFailed('pen does not fit in the front reach band')
        pen[:2, 3] = rng.uniform(minimum, maximum)
        pen_low, pen_high = low + pen[:2, 3], high + pen[:2, 3]
        if np.all(pen_high >= blocked_low) and np.all(blocked_high >= pen_low):
            continue
        if layout.penetration('pen', pen) <= PENETRATION_LIMIT:
            layout.set_pose('pen', pen)
            return
    raise SamplingFailed('no pen position clear of the drawer sweep')
