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
- **predicates** -- :func:`opening`, :func:`is_open`, :func:`is_closed`, :func:`on`,
  :func:`inside`: what the skills check when they are done, and what an expert's
  ``Step.check`` asks (:func:`require`).

Each skill runs under :func:`~table_1000.experts.skills.retry`. Joints are named as in
the asset's ``model.json`` (``cabinet.drawer1.slide``). Revolute joints (doors) are
not handled yet: their handle moves along an arc.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull

from table_1000.experts import skills
from table_1000.experts.context import ExpertFailed
from table_1000.experts.grasps import translated
from table_1000.experts.skills.common import normalized
from table_1000.experts.skills.placing import PLACE_ABOVE, PLACE_CLEARANCE

OPEN_TOLERANCE = .02    # m short of the asked opening that still counts as open
CLOSED = .005           # m from the closed end that counts as closed
PUSH_PAST = .003        # m pushed beyond the closed end, so that the part surely reaches it
# The most a handle grasp may turn from straight at it, relaxed in turn while none is in reach:
# tilted, pivoting fingers pinching the handle's ends slip off at 30 degrees.
OPEN_ANGLES = np.radians([10., 20.])
PULLS, PULL_LEAST = 3, .25  # grasps a part is opened with at most; the least part of the rest one moves
SUPPORT_STEP = .01      # m between the rays cast down over a support
UP = np.cos(np.radians(10.))  # least cosine between an upward face and the vertical
LEVEL = .005            # m within which support hits share a height
NEAR = .30              # m about a given near point that a region is cut down to (fewer spots to rank)
ON_TOLERANCE = .03      # m an object's lowest point may be above the surface it is on
# m an object put on a surface keeps inside its edges: room for the fingers that pick it up again
# (a pen 1 cm from the table's edge was grasped with the edge between the fingers)
EDGE = .04


def require(ctx, ok, reason, **details):
    """Fail the current subtask with ``reason`` unless ``ok``."""
    if not ok:
        ctx.fail(reason, **details)


# -- joints --------------------------------------------------------------------------------------
@dataclass
class Joint:
    """Joint ``name`` of an asset (``model.json``): ``kind`` (``SLIDER``, ``HINGE``, ...) between
    bodies ``parent`` and ``child``; unit ``axis`` (positive motion) in the parent's frame,
    ``child_axis`` in the child's; ``limits``; ``handle`` the child's part named ``handle`` (None
    without one)."""
    name: str
    kind: str
    parent: str
    child: str
    axis: np.ndarray
    limits: tuple
    child_axis: np.ndarray
    handle: str = None

    @classmethod
    def of(cls, geometry, name):
        entry = geometry.joints[name]
        parent = entry['body1']
        # model.json gives the joint frame in the asset frame.
        frame = np.linalg.inv(geometry.rest[parent]) @ np.array(entry['matrix'], dtype=float)
        axis = normalized(frame[:3, 0])
        handles = [part for _, part in geometry.parts(entry['body2']) if part.rsplit('.', 1)[-1].startswith('handle')]
        child_axis = geometry.rest[entry['body2']][:3, :3].T @ geometry.rest[parent][:3, :3] @ axis
        return cls(name, entry['type'], parent, entry['body2'], axis, tuple(entry['limits']), child_axis,
                   handles[0] if handles else None)

    def direction(self, child_pose):
        """World direction of positive motion with the child body at ``child_pose`` (a drawer
        half out may sit a little tilted in its housing; it moves along its own frame)."""
        return np.asarray(child_pose)[:3, :3] @ self.child_axis


def joint(ctx, obj, name):
    return Joint.of(ctx.geometry[obj], name)


def sim_joint(name):
    """The simulation's name of asset joint ``name`` (``cabinet_drawer1_slide``)."""
    return name.replace('.', '_')


def opening(ctx, obj, name):
    """How far joint ``name`` of ``obj`` is from its closed (lower) limit."""
    return ctx.joint(obj, sim_joint(name)) - Joint.of(ctx.geometry[obj], name).limits[0]


def is_open(ctx, obj, name, amount, tolerance=OPEN_TOLERANCE):
    return opening(ctx, obj, name) >= amount - tolerance


def is_closed(ctx, obj, name, tolerance=CLOSED):
    return opening(ctx, obj, name) <= tolerance


def slider(ctx, obj, name):
    j = joint(ctx, obj, name)
    if j.kind != 'SLIDER':
        raise NotImplementedError(f'{name}: only sliders are opened and closed so far, not {j.kind}')
    return j


def open(ctx, obj, name, amount):
    """Open joint ``name`` of ``obj`` by ``amount`` (m) by its handle; nothing when it already is."""
    j = slider(ctx, obj, name)

    def attempt():
        for _ in range(PULLS):
            if is_open(ctx, obj, name, amount):
                break
            direction = j.direction(ctx.pose(obj, j.child))
            held = skills.grasp(ctx, obj, j.handle, then=direction * (amount - opening(ctx, obj, name)),
                                approach=-direction, max_angle=OPEN_ANGLES)
            # Closing on the handle straight on may have pushed the part in a little.
            move = direction * (amount - opening(ctx, obj, name))
            housing = ctx.pose(obj, j.parent)[:3, 3]
            try:
                moved = skills.move_by_displacement(ctx, held['arm'], *move, drag=(obj, j.child), least=PULL_LEAST)
            except ExpertFailed:
                skills.release(ctx, held['arm'])    # so that it can be tried again
                raise
            ctx.note('pulled', joint=name, opening=opening(ctx, obj, name), moved=moved,
                     parent_shift=float(np.linalg.norm(ctx.pose(obj, j.parent)[:3, 3] - housing)),
                     finger_gap=ctx.solver(held['arm']).finger_gap())
            skills.release(ctx, held['arm'])
        require(ctx, is_open(ctx, obj, name, amount), 'not opened', joint=name, opening=opening(ctx, obj, name))

    skills.retry(ctx, attempt)


def close(ctx, obj, name):
    """Close joint ``name`` of ``obj``: push its handle back with the closed fingertips (no grasp
    needed, and the handle of an open drawer is too close to its body to grasp straight on)."""
    j = slider(ctx, obj, name)

    def attempt():
        housing = ctx.pose(obj, j.parent)[:3, 3]
        pushed = skills.push(ctx, obj, j.handle, direction=-j.direction(ctx.pose(obj, j.child)),
                             distance=opening(ctx, obj, name) + PUSH_PAST)
        ctx.note('pushed', joint=name, opening=opening(ctx, obj, name), arm=pushed['arm'], tilt=pushed['tilt'],
                 parent_shift=float(np.linalg.norm(ctx.pose(obj, j.parent)[:3, 3] - housing)))
        require(ctx, is_closed(ctx, obj, name), 'not closed', joint=name, opening=opening(ctx, obj, name))

    skills.retry(ctx, attempt)


def travel(ctx, obj, name, margin=0.):
    """Keep-out rectangle ``(frame, low, high)`` of the space joint ``name``'s child body sweeps from
    where it is to fully open (a drawer's pull-out space), grown by ``margin``."""
    j = slider(ctx, obj, name)
    geometry, frame = ctx.geometry[obj], ctx.pose(obj, j.child)
    low, high = geometry.bounds(j.child, j.child)
    sweep = j.child_axis * (j.limits[1] - j.limits[0] - opening(ctx, obj, name))
    return frame, np.minimum(low, low + sweep) - margin, np.maximum(high, high + sweep) + margin


def body_box(ctx, obj, body, margin=0.):
    """Keep-out rectangle ``(frame, low, high)`` of ``obj``'s body ``body`` (its collision box in its
    frame), grown by ``margin``."""
    low, high = ctx.geometry[obj].bounds(body, body)
    return ctx.pose(obj, body), low - margin, high + margin


# -- collision shapes and rays -------------------------------------------------------------------
@dataclass
class Shape:
    """A convex collision shape in the world: of ``part`` of body ``body`` of object ``obj``, the
    body at world ``frame``; ``points`` its vertices (world)."""
    obj: str
    body: str
    part: str
    frame: np.ndarray
    points: np.ndarray

    def __post_init__(self):
        self.planes = ConvexHull(self.points).equations     # n . x + d <= 0 inside


def asset_shapes(name, geometry, body_poses):
    """The :class:`Shape` of asset ``name`` with its bodies at world ``body_poses`` (``{body: 4x4}``)."""
    shapes = []
    for body, part, transform, kind, data in geometry.shapes:
        local = data if kind == 'convex' else np.vstack([np.eye(3) * data, -np.eye(3) * data])
        pose = np.asarray(body_poses[body]) @ transform
        shapes.append(Shape(name, body, part, np.asarray(body_poses[body]), local @ pose[:3, :3].T + pose[:3, 3]))
    return shapes


def table_shape(table):
    """The :class:`Shape` of the top of a scene's ``TABLE``."""
    frame = np.eye(4)
    frame[:2, 3], frame[2, 3] = table['center'], table['height']
    half = np.array([table['size'][0] / 2, table['size'][1] / 2])
    corners = np.array([[x, y, z] for x in (-half[0], half[0]) for y in (-half[1], half[1])
                        for z in (-table['top_thickness'], 0.)])
    return Shape('table', 'table', 'table', frame, corners + frame[:3, 3])


def scene_shapes(ctx, exclude=()):
    """Every collision shape of the scene at its live pose (the table's top too), but of ``exclude``."""
    poses = ctx.poses()
    shapes = [table_shape(ctx.scene.TABLE)] if 'table' not in exclude and hasattr(ctx.scene, 'TABLE') else []
    for name in ctx.scene.OBJECTS:
        if name not in exclude:
            geometry = ctx.geometry[name]
            shapes += asset_shapes(name, geometry, {body: poses[f'{name}:{body}'] for body in geometry.bodies})
    return shapes


def cast(shapes, origins, direction):
    """``(index, distance, normal)`` of the first of ``shapes`` each ray from ``origins`` (n x 3)
    along ``direction`` hits (index -1 and distance inf where it hits none), the normal that of
    the face it enters."""
    origins = np.atleast_2d(np.asarray(origins, dtype=float))
    direction = normalized(direction)
    first = np.full(len(origins), -1)
    distance = np.full(len(origins), np.inf)
    normal = np.zeros((len(origins), 3))
    for index, shape in enumerate(shapes):
        n, d = shape.planes[:, :3], shape.planes[:, 3]
        side = origins @ n.T + d            # > 0 outside the plane
        rate = n @ direction                # < 0 entering through it
        with np.errstate(divide='ignore', invalid='ignore'):
            t = -side / rate
        enter = np.where(rate < -1e-12, t, -np.inf)
        leave = np.where(rate > 1e-12, t, np.inf)
        parallel_out = (np.abs(rate) <= 1e-12) & (side > 1e-9)    # grazing a face counts as a hit
        t_in, t_out = enter.max(1), leave.min(1)
        hit = (t_in <= t_out) & (t_out >= 0) & ~parallel_out.any(1)
        t_in = np.maximum(t_in, 0.)
        closer = hit & (t_in < distance)
        first[closer], distance[closer] = index, t_in[closer]
        normal[closer] = n[enter[closer].argmax(1)]
    return first, distance, normal


def selects(shape, obj, part=None):
    """Whether ``shape`` belongs to object ``obj`` and, if given, its body or part ``part``."""
    return shape.obj == obj and (part is None or shape.body == part or shape.part == part
                                 or shape.part.startswith(part + '.'))


def support_region(shapes, obj, part=None, floor=False, headroom=np.inf, step=SUPPORT_STEP):
    """``(region, keep_out)`` to put something down on ``obj`` (its body or ``part``) among
    ``shapes`` (the scene, but what is put down): ``region`` the rectangle ``(frame, low, high)`` of
    the cells whose rays cast straight down hit an upward face of it first, at the highest height
    they share (``floor``: the lowest, a container's floor rather than its rims; the frame that of
    its body, raised to that height), None where there is none; ``keep_out`` the collision
    boxes, each in its body frame, of the bodies (of the support's own object: the parts) hit first
    inside it instead. Cells outside the actual support (such as a round floor's corners)
    remain excluded at the ray-grid resolution. What is put down comes straight down from
    ``headroom`` above that height: what
    is all higher (a handle over the edge of a drawer's floor) neither hides it nor is kept out; the
    hand above is checked against it when placing. See :meth:`~table_1000.experts.skills.Skills.place`."""
    support = [shape for shape in shapes if selects(shape, obj, part)]
    frame = support[0].frame
    local = np.linalg.inv(frame)

    def heights(shape):
        return (shape.points @ local[:3, :3].T + local[:3, 3])[:, 2]

    points = np.vstack([shape.points for shape in support]) @ local[:3, :3].T + local[:3, 3]
    low, high = points.min(0), points.max(0)
    xs, ys = np.arange(low[0] + step / 2, high[0], step), np.arange(low[1] + step / 2, high[1], step)
    grid = np.array([[x, y] for x in xs for y in ys])
    top = max(max(float(heights(shape).max()) for shape in shapes), high[2]) + .1
    origins = (frame @ np.c_[grid, np.full(len(grid), top), np.ones(len(grid))].T).T[:, :3]

    def hits(among):
        first, distance, normal = cast(among, origins, -frame[:3, 2])
        on_support = np.array([index >= 0 and selects(among[index], obj, part) for index in first])
        return first, top - distance, on_support & (normal @ frame[:3, 2] > UP)

    first, height, up = hits(shapes)
    if not up.any():
        return None, []
    levels = np.round(height[up] / LEVEL)
    level = levels.min() if floor else levels.max()
    surface_z = float(np.median(height[up & (np.round(height / LEVEL) == level)]))
    every = shapes
    shapes = [shape for shape in shapes if heights(shape).min() < surface_z + headroom]
    first, height, up = hits(shapes)
    cells = up & (np.round(height / LEVEL) == level)
    # The region's edges halfway between the cells on it and those off it.
    low, high = grid[cells].min(0) - step / 2, grid[cells].max(0) + step / 2
    surface = frame @ translated(np.eye(4), [0., 0., surface_z])
    inside = np.all((grid >= low) & (grid <= high), axis=1)
    keep_out = {}
    for index in set(first[inside & ~cells & (first >= 0)]):
        shape = shapes[index]
        # Below the support plane never intersects the descent column. A table
        # under a round container must not exclude its entire floor in XY.
        if heights(shape).max() <= surface_z:
            continue
        own = shape.obj == obj
        key = (shape.obj, shape.body, shape.part if own else None)
        if key not in keep_out:
            members = [s for s in every if s.obj == shape.obj and s.body == shape.body
                       and (not own or s.part == shape.part)
                       and heights(s).max() > surface_z and heights(s).min() < surface_z + headroom]
            body = np.linalg.inv(shape.frame)
            box = np.vstack([s.points for s in members]) @ body[:3, :3].T + body[:3, 3]
            keep_out[key] = (shape.frame, box.min(0), box.max(0))
    # A rectangle bounds the sampled support; do not turn a circular floor or
    # a gap into support simply because the first hit lies below its plane.
    unsupported = inside & ~cells & (height < surface_z - LEVEL)
    gaps = [(surface, xy - step / 2, xy + step / 2) for xy in grid[unsupported]]
    return (surface, low, high), [*keep_out.values(), *gaps]


def object_points(ctx, obj):
    """World collision points of ``obj`` at its live pose."""
    return np.vstack([shape.points for shape in scene_shapes(ctx) if shape.obj == obj])


def centre(points):
    return (points.min(0) + points.max(0)) / 2


# -- predicates ----------------------------------------------------------------------------------
def on(ctx, obj, surface, part=None, keep_out=(), tolerance=ON_TOLERANCE):
    """Whether ``obj`` rests on ``surface`` (the table, an object, or its body or ``part``): straight
    below its centre the first thing is that surface, at most ``tolerance`` below its lowest point;
    its centre out of the ``keep_out`` rectangles."""
    points = object_points(ctx, obj)
    middle = centre(points)
    shapes = scene_shapes(ctx, exclude=(obj,))
    first, distance, _ = cast(shapes, [middle], [0., 0., -1.])
    if first[0] < 0 or not selects(shapes[first[0]], surface, part):
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
    shapes = [shape for shape in scene_shapes(ctx, exclude=(obj,)) if selects(shape, container, part)]
    middle = centre(object_points(ctx, obj))
    axes = shapes[0].frame[:3, :3]
    for direction in (axes[:, 0], -axes[:, 0], axes[:, 1], -axes[:, 1], -axes[:, 2]):
        if cast(shapes, [middle], direction)[0][0] < 0:
            return False
    return True


def headroom(ctx, obj):
    """How high over a support the held ``obj`` sweeps as :func:`~table_1000.experts.skills.place`
    brings it down: its height, the clearance and the height it comes down from."""
    return float(np.ptp(object_points(ctx, obj)[:, 2])) + PLACE_CLEARANCE + PLACE_ABOVE


def around(region, near):
    """``region`` cut down to within NEAR of world point ``near`` (all of it without one), or of
    the region's point nearest it when ``near`` is outside."""
    if near is None:
        return region
    frame, low, high = region
    point = np.clip((np.linalg.inv(frame) @ np.r_[np.asarray(near, dtype=float)[:3], 1.])[:2], low, high)
    return frame, np.maximum(low, point - NEAR), np.minimum(high, point + NEAR)


def inset(region, margin):
    """``region`` shrunk by ``margin`` on every side, down to its middle line where it is narrower."""
    frame, low, high = region
    margin = np.minimum(margin, (np.asarray(high) - low) / 2)
    return frame, np.asarray(low) + margin, np.asarray(high) - margin


# -- placing -------------------------------------------------------------------------------------
def on_region(obj, surface, part=None, near=None, keep_out=()):
    """``where(ctx) -> (region, keep_out)``: where on ``surface`` (the table, an object, or its body
    or ``part``) to put ``obj`` down in the state then, out of ``keep_out`` rectangles besides what
    stands there and EDGE inside the surface's edges, within NEAR of ``near`` (world point) if given;
    for :func:`~table_1000.experts.skills.place` (or a :class:`~table_1000.experts.skills.Place`).
    See :func:`support_region`."""
    def where(ctx):
        region, blocked = support_region(scene_shapes(ctx, exclude=(obj,)), surface, part, headroom=headroom(ctx, obj))
        require(ctx, region is not None, 'no support', surface=surface, part=part)
        return around(inset(region, EDGE), near), [*blocked, *keep_out]
    return where


def inside_region(obj, container, part=None, near=None):
    """``where(ctx) -> (region, keep_out)``: where inside ``container`` (or its body or ``part``: a
    drawer) to put ``obj`` down in the state then, on its floor where nothing covers it (the part of
    a drawer out of its cabinet); as :func:`on_region`."""
    def where(ctx):
        region, blocked = support_region(scene_shapes(ctx, exclude=(obj,)), container, part, floor=True,
                                         headroom=headroom(ctx, obj))
        require(ctx, region is not None, 'no support', container=container, part=part)
        return around(region, near), blocked
    return where


def put(ctx, obj, where, near):
    region, keep_out = where(ctx)
    skills.place(ctx, obj, region, near=near, keep_out=keep_out)


def place_on(ctx, obj, surface, part=None, near=None, keep_out=()):
    """Put the held ``obj`` down where :func:`on_region` says, nearest ``near`` (default the
    region's centre); it must rest on ``surface`` after."""
    where = on_region(obj, surface, part, near, keep_out)
    skills.retry(ctx, lambda: put(ctx, obj, where, near))
    require(ctx, on(ctx, obj, surface, part, keep_out), f'not on the {surface}', obj=obj,
            position=centre(object_points(ctx, obj)).tolist())


def place_inside(ctx, obj, container, part=None, near=None):
    """Put the held ``obj`` down where :func:`inside_region` says; it must be inside after."""
    where = inside_region(obj, container, part, near)
    skills.retry(ctx, lambda: put(ctx, obj, where, near))
    require(ctx, inside(ctx, obj, container, part), f'not in the {part or container}', obj=obj,
            position=centre(object_points(ctx, obj)).tolist())
