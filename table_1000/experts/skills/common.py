"""What the skills share: candidate and IK budgets, the grasped target, planning failure details."""

from dataclasses import dataclass

import numpy as np

from table_1000.experts.context import Unreachable
from table_1000.experts.grasps import Box
from mplib.collision_detection import fcl
from table_1000.planning.world import PlanningObject, geometry, pose as fcl_pose

APPROACH = .05          # m, pre-grasp distance behind the grasp along the approach
CANDIDATE_PLANS = 2     # RRTConnect tries (3 s, then 6 s) per candidate before the next is taken
IK_ATTEMPTS = 20        # IK restarts per pose
SUPPORT_DISTANCE = .003 # m, planning objects this close to the grasped one support it
ROOM_PLACES = 10        # first placements (grasps, per arm) the base moves to reach when none is in reach
RETREATS = (APPROACH, .03)   # m, back-off after releasing, tried in order


class Unmet(Unreachable):
    """What the next action needs of this one (a :class:`~table_1000.experts.skills.chain.InHand`)
    is not in reach: some candidates were, but none that meets it."""


def upright_need(need):
    """Whether the next action asks for an upright insertion hold."""
    recipe = (need.recipe or {}) if need is not None else {}
    return recipe.get('origin', {}).get('orientation') == 'upright' or recipe.get('hanging', False)


def collisions(error):
    """The collisions a :class:`PlanningFailed` reports (of its path, start or goal)."""
    plan = error.details.get('plan', {})
    return {key: plan[key] for key in ('collisions', 'start_collisions', 'goal_collisions') if plan.get(key)}


def normalized(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def planning_object(name, key, shape, offset=np.eye(4)):
    """Planning object ``name`` of ``shape`` moving with body ``key``."""
    if isinstance(shape, Box):
        entry = (np.asarray(offset, dtype=float), 'box', tuple(shape.size))
    else:
        entry = (np.asarray(offset, dtype=float), 'cylinder', (shape.radius, shape.length))
    return PlanningObject(name, key, [entry], padded=False)


@dataclass
class Target:
    """What a grasp holds: planning objects ``names``, shaped (for the grasp) by ``shape`` at
    ``offset`` in the frame of body ``body`` (a key of ``ctx.poses()``); ``points`` are the whole
    held object's collision points in that frame (for placing it)."""
    obj: str
    names: list
    body: str
    shape: object
    offset: np.ndarray
    points: np.ndarray
    fallback: bool = False  # a whole-object envelope used only when real parts are not legal
    rim: bool = False
    container: bool = False
    enclosed: bool = False


def convex_keepout(points):
    """FCL convex keep-out from world vertices, built around its local centre.

    Hull construction in world coordinates can merge nearly coplanar faces
    inconsistently after a distant object's settled float32 transform. A rigid
    translation and a 1 nm grid remove that numeric degeneracy; 2 nm padding
    covers the maximum snap error. This is far below insertion clearance.
    """
    points = np.asarray(points, dtype=float)
    centre = (points.min(0) + points.max(0)) / 2
    frame = np.eye(4)
    frame[:3, 3] = centre
    identity = fcl_pose(np.eye(4))
    local = np.unique(np.round(points - centre, 9), axis=0)
    return fcl.FCLObject('obstacle', fcl_pose(frame),
                         [fcl.CollisionObject(geometry('convex', local, padding=2e-9))], [identity])
