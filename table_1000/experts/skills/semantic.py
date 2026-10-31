"""Semantic skills: open and close a jointed part, put an object on a surface or into a container.

Each skill takes what to do in the terms of the scene (an object, its joint, a
surface) and works out the rest from the assets' collision geometry and joints:

- **open** / **close** -- the joint's axis gives the direction its child body moves
  (a slider's positive travel: a drawer pulled out); the handle is the part named
  ``handle`` of that body. ``open`` grasps the handle straight at it (the angle from
  straight relaxed through ``OPEN_ANGLES``), moves it along the axis and lets go; where
  the hand cannot follow it all the way it slides it part way and grasps again
  (``PULLS`` in all), letting go first when a pull fails. ``close`` pushes the handle
  back with the closed fingertips.
- **place_on** / **place_inside** -- the region to put an object down in is found by
  casting rays straight down over the support (the table, or a body or part of an
  object) at the live poses of everything in the scene: the cells whose first hit is
  an upward face of the support bound the region (on a surface the highest such face,
  in a container the lowest: its floor, not its rims); whatever is hit first inside it
  instead (an object standing on the table, the cabinet over the part of a drawer still
  inside it) is kept out by the collision box of its body, unless it is higher than the
  object sweeps coming down. :func:`~table_1000.experts.skills.place` then puts the
  object down there, near a given point (within ``NEAR`` of it).
- **predicates** -- :func:`~table_1000.experts.skills.drawer.joints.opening`,
  :func:`~table_1000.experts.skills.drawer.joints.is_open`,
  :func:`~table_1000.experts.skills.drawer.joints.is_closed`, :func:`on`,
  :func:`inside`: what the skills check when they are done, and what an expert's
  ``Step.check`` asks (:func:`require`).

Each skill runs under :func:`~table_1000.experts.skills.retry`. Joints are named as in
the asset's ``model.json`` (``cabinet.drawer1.slide``). Revolute joints (doors) are
not handled yet: their handle moves along an arc.
"""

from table_1000.experts.skills.drawer import joints as drawer_joints
from table_1000.experts.skills.place import regions
from table_1000.experts.world import geometry as world_geometry

from contextlib import nullcontext

import numpy as np

from table_1000.experts import skills
from table_1000.experts.context import ExpertFailed

PUSH_PAST = .003        # m pushed beyond the closed end, so that the part surely reaches it
# The most a handle grasp may turn from straight at it, relaxed in turn while none is in reach:
# tilted, pivoting fingers pinching the handle's ends slip off at 30 degrees.
OPEN_ANGLES = np.radians([10., 20.])
PULLS, PULL_LEAST = 3, .25  # grasps a part is opened with at most; the least part of the rest one moves
ON_TOLERANCE = .03      # m an object's lowest point may be above the surface it is on


def require(ctx, ok, reason, **details):
    """Fail the current subtask with ``reason`` unless ``ok``."""
    if not ok:
        ctx.fail(reason, **details)


# -- drawer actions -----------------------------------------------------------------------------


def _open(ctx, obj, name, amount, *, brace=False):
    """Implementation of :func:`open`, also used by the replayable brace skill."""
    j = drawer_joints.slider(ctx, obj, name)
    from table_1000.experts.skills.drawer.bracing import brace as support, stability
    if brace and (ctx.skills().held or len(ctx.skills().arms) < 2):
        raise ValueError('brace=True needs both hands empty; place held objects first, or use brace=False')

    def attempt():
        for _ in range(PULLS):
            if drawer_joints.is_open(ctx, obj, name, amount):
                break
            direction = j.direction(ctx.pose(obj, j.child))
            housing = ctx.pose(obj, j.parent)[:3, 3]
            s = ctx.skills() if brace else None
            active = min(s.arms, key=lambda a: np.linalg.norm(s.shoulder(a) - ctx.pose(obj, j.child)[:3, 3])) if brace else None
            with support(ctx, obj, j, active, amount) if brace else nullcontext() as selected:
                active = selected if brace else active
                move = direction * (amount - drawer_joints.opening(ctx, obj, name))
                if brace:
                    held = s.grasp(s.target(obj, j.handle), arm=active, then=move, approach=-direction,
                                   max_angle=OPEN_ANGLES[-1], base=False)
                else:
                    held = skills.grasp(ctx, obj, j.handle, then=move, approach=-direction, max_angle=OPEN_ANGLES)
                try:
                    move = direction * (amount - drawer_joints.opening(ctx, obj, name))
                    moved = (s.move_by_displacement(held['arm'], move, drag=(obj, j.child), least=PULL_LEAST) if brace else
                             skills.move_by_displacement(ctx, held['arm'], *move, drag=(obj, j.child), least=PULL_LEAST))
                except ExpertFailed:
                    if brace:
                        s.release(held['arm'])
                    else:
                        skills.release(ctx, held['arm'])
                    raise
                ctx.note('pulled', joint=name, opening=drawer_joints.opening(ctx, obj, name), moved=moved,
                         parent_shift=float(np.linalg.norm(ctx.pose(obj, j.parent)[:3, 3] - housing)),
                         finger_gap=ctx.solver(held['arm']).finger_gap())
                if brace:
                    s.release(held['arm'])
                else:
                    skills.release(ctx, held['arm'])
        require(ctx, drawer_joints.is_open(ctx, obj, name, amount), 'not opened', joint=name, opening=drawer_joints.opening(ctx, obj, name))

    def checked():
        with stability(ctx, obj, j) if brace else nullcontext():
            attempt()

    skills.retry(ctx, checked)


def _close(ctx, obj, name, *, brace=False):
    """Implementation of :func:`close`, also used by the replayable brace skill."""
    j = drawer_joints.slider(ctx, obj, name)
    from table_1000.experts.skills.drawer.bracing import brace as support, stability
    if brace and (ctx.skills().held or len(ctx.skills().arms) < 2):
        raise ValueError('brace=True needs both hands empty; place held objects first, or use brace=False')

    def attempt():
        housing = ctx.pose(obj, j.parent)[:3, 3]
        s = ctx.skills() if brace else None
        active = min(s.arms, key=lambda a: np.linalg.norm(s.shoulder(a) - ctx.pose(obj, j.child)[:3, 3])) if brace else None
        with support(ctx, obj, j, active) if brace else nullcontext() as selected:
            active = selected if brace else active
            direction = -j.direction(ctx.pose(obj, j.child))
            distance = drawer_joints.opening(ctx, obj, name) + PUSH_PAST
            pushed = (s.push(s.target(obj, j.handle), direction, distance, arm=active) if brace else
                      skills.push(ctx, obj, j.handle, direction=direction, distance=distance))
        ctx.note('pushed', joint=name, opening=drawer_joints.opening(ctx, obj, name), arm=pushed['arm'], tilt=pushed['tilt'],
                 parent_shift=float(np.linalg.norm(ctx.pose(obj, j.parent)[:3, 3] - housing)))
        require(ctx, drawer_joints.is_closed(ctx, obj, name), 'not closed', joint=name, opening=drawer_joints.opening(ctx, obj, name))

    def checked():
        with stability(ctx, obj, j) if brace else nullcontext():
            attempt()

    skills.retry(ctx, checked)


def open(ctx, obj, name, amount, *, brace=False):
    """Open joint ``name`` of ``obj`` by ``amount`` (m) by its handle; nothing when it already is.

    ``brace=True`` presses the housing roof with the other empty hand during
    each pull, then retracts it. Both hands must be empty at entry; otherwise
    ``ValueError`` asks the expert to place the held object first. Housing
    translation <= 5 mm, rotation <= 2 degrees, and other sliders' absolute
    openings <= 5 mm are checked every control step, including approach and
    retreat. A violation fails the skill. The opt-in route locks base and torso
    during support; no reachable loaded roof contact raises ``Unreachable``.
    Clear the roof/reposition first, or use ``brace=False`` to omit these
    stability checks. Metrics are in ``drawer stability``
    notes, including failed attempts. No asset/drive parameters are changed.
    """
    return skills.braced_drawer(ctx, obj, name, amount) if brace else _open(ctx, obj, name, amount)


def close(ctx, obj, name, *, brace=False):
    """Close joint ``name`` of ``obj``: push its handle back with the closed fingertips (no grasp
    needed, and the handle of an open drawer is too close to its body to grasp straight on).

    ``brace=True`` supports the housing roof with the opposite empty hand and
    checks <= 5 mm housing translation, <= 2 degree rotation and <= 5 mm other
    sliders' openings at every control step. Requirements, failures, logging
    and locked torso/base are as in :func:`open`; default behavior is unchanged.
    """
    return skills.braced_drawer(ctx, obj, name) if brace else _close(ctx, obj, name)


# -- predicates ----------------------------------------------------------------------------------
def on(ctx, obj, surface, part=None, keep_out=(), tolerance=ON_TOLERANCE):
    """Whether ``obj`` rests on ``surface`` (the table, an object, or its body or ``part``): straight
    below its centre the first thing is that surface, at most ``tolerance`` below its lowest point;
    its centre out of the ``keep_out`` rectangles."""
    points = world_geometry.object_points(ctx, obj)
    middle = world_geometry.centre(points)
    shapes = world_geometry.scene_shapes(ctx, exclude=(obj,))
    first, distance, _ = world_geometry.cast(shapes, [middle], [0., 0., -1.])
    if first[0] < 0 or not world_geometry.selects(shapes[first[0]], surface, part):
        return False
    if points[:, 2].min() - (middle[2] - distance[0]) > tolerance:
        return False
    for frame, low, high in keep_out:
        local = (np.linalg.inv(frame) @ np.r_[middle, 1.])[:2]
        if np.all((np.asarray(low)[:2] <= local) & (local <= np.asarray(high)[:2])):
            return False
    return True


def inside(ctx, obj, container, part=None):
    """Whether ``obj``'s centre is inside ``container`` (or its body or ``part``): rays from it
    sideways both ways along the container body's x and y axes and straight down all hit it."""
    shapes = [shape for shape in world_geometry.scene_shapes(ctx, exclude=(obj,)) if world_geometry.selects(shape, container, part)]
    middle = world_geometry.centre(world_geometry.object_points(ctx, obj))
    return _inside_point(shapes, middle)


def _inside_point(shapes, middle):
    """Five enclosing rays at a world point, along the container body's axes."""
    axes = shapes[0].frame[:3, :3]
    for direction in (axes[:, 0], -axes[:, 0], axes[:, 1], -axes[:, 1], -axes[:, 2]):
        if world_geometry.cast(shapes, [middle], direction)[0][0] < 0:
            return False
    return True


def supported_inside(ctx, obj, container):
    """Whether collision material immerses in the cavity and touches an inward
    floor/wall face within the existing support distance. Unlike ``inside``,
    contents may protrude above the mouth; touching an exterior wall is not support.
    """
    from mplib.collision_detection import fcl
    from table_1000.experts.skills.common import SUPPORT_DISTANCE, convex_keepout
    from table_1000.experts.skills.hold.sections import Edge, exposed_points
    shapes = world_geometry.scene_shapes(ctx)
    walls = [s for s in shapes if s.obj == container]
    normal = walls[0].frame[:3, 2]
    mouth = max(float((s.points @ normal).max()) for s in walls)
    below = Edge(normal * mouth, -normal, np.zeros(3), 0.)
    request = fcl.DistanceRequest(enable_nearest_points=True, enable_signed_distance=True)
    for item in (s for s in shapes if s.obj == obj):
        immersed = exposed_points(item.points, below)
        if not len(immersed):
            continue
        middle = immersed.mean(axis=0)
        if (not _inside_point(walls, middle) or any(
                np.all(w.planes[:, :3] @ middle + w.planes[:, 3] <= 0.) for w in walls)):
            continue
        material = convex_keepout(item.points)
        for wall in walls:
            contact = fcl.distance(convex_keepout(wall.points), material, request)
            if contact.min_distance >= SUPPORT_DISTANCE:
                continue
            # The nearest wall face must point into this immersed material,
            # not out toward an adjacent object on the other side of a wall.
            face = wall.planes[np.argmax(wall.planes[:, :3] @ contact.nearest_points + wall.planes[:, 3])]
            if face[:3] @ middle + face[3] > 0.:
                return True
    return False


def upright_inside(ctx, obj, container, part=None, max_tilt=np.radians(30.)):
    """Whether a slender body is contained and stands within ``max_tilt`` radians
    of the container's up axis. Use this same predicate before deciding to move
    a pen and after settling it. An existing correct pen need not be removed
    and reinserted. The longest body-frame collision extent defines its axis.
    Its lower end must penetrate at least a quarter of its length or half
    the container height, whichever is smaller. Enclosing rays test a point
    in that immersed segment: the pen's centre may protrude above the mouth.
    """
    target = ctx.skills().target(obj)
    size = np.ptp(target.points, axis=0)
    order = np.argsort(size)
    if size[order[-1]] < 2 * size[order[-2]]:
        raise ValueError('upright_inside requires a slender collision body; use inside for ordinary objects')
    shapes = [s for s in world_geometry.scene_shapes(ctx) if world_geometry.selects(s, container, part)]
    if not shapes:
        raise ValueError('upright_inside requires a container with collision geometry')
    if not 0. <= max_tilt < np.pi / 2:
        raise ValueError('max_tilt must be radians in [0, pi/2)')
    pose = ctx.poses()[target.body]
    axis = pose[:3, order[-1]]
    normal = shapes[0].frame[:3, 2]
    if abs(float(axis @ normal)) < np.cos(max_tilt):
        return False
    points = target.points @ pose[:3, :3].T + pose[:3, 3]
    bounds = np.concatenate([s.points for s in shapes]) @ normal
    bottom, mouth = float((points @ normal).min()), float(bounds.max())
    if bottom < float(bounds.min()) - world_geometry.LEVEL:
        return False
    depth = mouth - bottom
    required = min(float(size[order[-1]]) / 4, float(np.ptp(bounds)) / 2)
    if depth < required:
        return False
    middle = world_geometry.centre(points)
    middle += normal * (bottom + min(depth, float(np.ptp(points @ normal))) / 2 - middle @ normal)
    return _inside_point(shapes, middle)


# -- placing -------------------------------------------------------------------------------------


def put(ctx, obj, where, near, **options):
    region, keep_out = where(ctx)
    skills.place(ctx, obj, region, near=near, keep_out=keep_out, **options)


def place_on(ctx, obj, surface, part=None, near=None, keep_out=()):
    """Put the held ``obj`` down where :func:`on_region` says, nearest ``near`` (default the
    region's centre); it must rest on ``surface`` after."""
    where = regions.on_region(obj, surface, part, near, keep_out)
    skills.retry(ctx, lambda: put(ctx, obj, where, near))
    require(ctx, on(ctx, obj, surface, part, keep_out), f'not on the {surface}', obj=obj,
            position=world_geometry.centre(world_geometry.object_points(ctx, obj)).tolist())


def place_inside(ctx, obj, container, part=None, near=None):
    """Put the held ``obj`` down where :func:`inside_region` says; it must be inside after."""
    where = regions.inside_region(obj, container, part, near)
    skills.retry(ctx, lambda: put(ctx, obj, where, near))
    require(ctx, inside(ctx, obj, container, part), f'not in the {part or container}', obj=obj,
            position=world_geometry.centre(world_geometry.object_points(ctx, obj)).tolist())
