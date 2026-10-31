"""Live scene collision geometry and support queries."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull

from table_1000.experts.skills.common import normalized
from table_1000.experts.skills.hold.frames import translated


SUPPORT_STEP = .01      # m between the rays cast down over a support


UP = np.cos(np.radians(10.))  # least cosine between an upward face and the vertical


LEVEL = .005            # m within which support hits share a height


def body_box(ctx, obj, body, margin=0.):
    """Keep-out rectangle ``(frame, low, high)`` of ``obj``'s body ``body`` (its collision box in its
    frame), grown by ``margin``."""
    low, high = ctx.geometry[obj].bounds(body, body)
    return ctx.pose(obj, body), low - margin, high + margin


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


def support_region(shapes, obj, part=None, floor=False, headroom=np.inf, step=SUPPORT_STEP, *, top_envelope=False):
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
    hand above is checked against it when placing. ``top_envelope`` retains all upward-face
    cells at their highest sampled height, for a slightly tilted flat stack support; the
    default selects a shared level. See :meth:`~table_1000.experts.skills.Skills.place`."""
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
    surface_z = (float(height[up].max()) if top_envelope else
                 float(np.median(height[up & (np.round(height / LEVEL) == level)])))
    every = shapes
    shapes = [shape for shape in shapes if heights(shape).min() < surface_z + headroom]
    first, height, up = hits(shapes)
    cells = up if top_envelope else up & (np.round(height / LEVEL) == level)
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
