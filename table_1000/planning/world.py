"""Planning-world collision objects for built assets and scene primitives.

Asset collision geometry comes from the built ``model.json``, the same export
the USD physics layer is authored over: every collision node of a rigid body
becomes a convex hull (or sphere) placed relative to that body. Shapes are
grouped into named planning objects so contact permissions can be granted per
part (for example only a drawer handle), and each object is posed from the live
pose of its body.
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


def pose(transform):
    """``mplib.Pose`` of a 4x4 transform."""
    from mplib import Pose
    transform = np.asarray(transform, dtype=float)
    return Pose(transform[:3, 3], quaternion(transform[:3, :3]))


def quaternion(m):
    """``(w, x, y, z)`` of a rotation matrix (Shepperd's method; scipy's is slow for one matrix)."""
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.:
        s = 2. * np.sqrt(trace + 1.)
        return np.array([s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2. * np.sqrt(1. + m[i, i] - m[j, j] - m[k, k])
    q = np.empty(4)
    q[0] = (m[k, j] - m[j, k]) / s
    q[1 + i] = s / 4
    q[1 + j] = (m[j, i] + m[i, j]) / s
    q[1 + k] = (m[k, i] + m[i, k]) / s
    return q


def model_shapes(model):
    """``[(body, part, T_body_shape, kind, data)]`` for every collision node.

    ``part`` is the collision node's parent (the named part); ``kind`` is
    ``'sphere'`` with a radius or ``'convex'`` with hull vertices in the shape
    frame. ``T_rest`` of each body in the asset frame is returned as well.
    """
    model = json.loads(Path(model).read_text()) if not isinstance(model, dict) else model
    worlds, owners, rest, shapes = {}, {}, {}, []
    for node in model['nodes']:
        name, parent = node['name'], node['parent']
        worlds[name] = worlds.get(parent, np.eye(4)) @ np.array(node['matrix'], dtype=float)
        owners[name] = name if node['rigid'] else owners.get(parent)
        if node['rigid']:
            rest[name] = worlds[name]
        if node.get('role') != 'collision':
            continue
        body = owners[name]
        transform = np.linalg.inv(worlds[body]) @ worlds[name]
        if node.get('shape') == 'SPHERE':
            shapes.append((body, parent, transform, 'sphere', float(node['radius'])))
        else:
            shapes.append((body, parent, transform, 'convex', np.asarray(node['vertices'], dtype=float)))
    return shapes, rest


def grow(vertices, padding):
    """Convex-hull vertices grown by ``padding`` per axis about their box centre."""
    vertices = np.asarray(vertices, dtype=float)
    low, high = vertices.min(0), vertices.max(0)
    centre, half = (low + high) / 2, np.maximum((high - low) / 2, 1e-9)
    return centre + (vertices - centre) * (half + padding) / half


# Collision checks cost about linearly in hull vertices; a planning hull keeps at most this many.
HULL_VERTICES = 64


def simplify_hull(vertices, limit=HULL_VERTICES):
    """Vertices of a convex hull of at most ``limit`` of ``vertices`` (greedily the farthest outside
    the hull so far), grown to cover the ones left out."""
    from scipy.spatial import ConvexHull
    vertices = np.unique(np.asarray(vertices, dtype=float), axis=0)
    if len(vertices) <= limit:
        return vertices
    directions = np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if x or y or z])
    chosen = sorted(set(np.argmax(vertices @ directions.T, axis=0).tolist()))
    while True:
        planes = ConvexHull(vertices[chosen]).equations
        outside = (vertices @ planes[:, :3].T + planes[:, 3]).max(1)
        if len(chosen) >= limit or outside.max() <= 1e-4:
            break
        chosen.append(int(np.argmax(outside)))
    return grow(vertices[chosen], max(float(outside.max()), 0.))


def geometry(kind, data, padding=0.):
    """FCL shape grown by ``padding`` on every side.

    Convex hulls grow per axis of their own frame about their box centre, which
    is exact for the box-like parts of these assets.
    """
    from mplib.collision_detection import fcl
    if kind == 'sphere':
        return fcl.Sphere(data + padding)
    if kind == 'box':
        return fcl.Box(*(np.asarray(data) + 2 * padding))
    if kind == 'cylinder':    # (radius, length) along the shape frame's Z
        return fcl.Cylinder(data[0] + padding, data[1] + 2 * padding)
    hull = trimesh.Trimesh(simplify_hull(data)).convex_hull
    vertices = grow(hull.vertices, padding) if padding else np.asarray(hull.vertices, dtype=float)
    return fcl.Convex(vertices, np.asarray(hull.faces, dtype=np.int32))


class PlanningObject:
    """Named group of shapes that moves with one body.

    ``body`` is the key of that body's live pose (``None`` for static objects).
    ``padded`` says whether the object is grown by the scene's padding.
    """

    def __init__(self, name, body, shapes, padded=True):
        self.name, self.body, self.raw, self.padded = name, body, shapes, padded
        self.cache = {}

    def fcl(self, world_pose, padding=0.):
        from mplib.collision_detection import fcl
        padding = padding if self.padded else 0.
        if padding not in self.cache:
            self.cache[padding] = [(transform, geometry(kind, data, padding)) for transform, kind, data in self.raw]
        shapes = self.cache[padding]
        objects = [fcl.CollisionObject(geom) for _, geom in shapes]
        return fcl.FCLObject(self.name, pose(world_pose), objects, [pose(t) for t, _ in shapes])


def asset_objects(model, prefix, group, padded=lambda name: True, merged=lambda name: False):
    """Planning objects of one asset instance.

    ``group(body, part)`` returns ``(object_name, frame_body)``; a shape on a
    body other than ``frame_body`` is placed by the bodies' rest transforms, so
    several bodies of a jointed but static assembly can form one object.
    Object names are prefixed with ``prefix``; ``padded(name)`` says whether
    the scene's padding applies to the object, ``merged(name)`` whether its
    shapes are replaced by their common convex hull (for a compact, convex
    assembly with many small parts).
    """
    shapes, rest = model_shapes(model)
    grouped = {}
    for body, part, transform, kind, data in shapes:
        name, frame = group(body, part)
        relative = np.linalg.inv(rest[frame]) @ rest[body] @ transform
        grouped.setdefault((prefix + name, frame), []).append((relative, kind, data))
    for key, entries in grouped.items():
        if merged(key[0][len(prefix):]):
            points = []
            for transform, kind, data in entries:
                local = data if kind == 'convex' else np.vstack([np.zeros(3), np.eye(3) * data, -np.eye(3) * data])
                points.append(local @ transform[:3, :3].T + transform[:3, 3])
            grouped[key] = [(np.eye(4), 'convex', np.vstack(points))]
    return [PlanningObject(name, prefix + frame, entries, padded(name)) for (name, frame), entries in grouped.items()]


def box_object(name, size, transform):
    """A static box centred on ``transform``: ``(object, transform)``."""
    return PlanningObject(name, None, [(np.eye(4), 'box', tuple(size))]), np.asarray(transform, dtype=float)


class PlanningScene:
    """Keeps a planner's world objects in sync with body poses.

    Poses are given in the world frame and stored in the planner's frame,
    ``frame`` being the planner-from-world transform. Padded objects are grown
    by ``padding``, which :meth:`set_padding` changes for later plans.
    """

    def __init__(self, planner, frame=np.eye(4)):
        self.planner, self.objects, self.frame = planner, {}, np.asarray(frame, dtype=float)
        self.padding, self.poses, self.object_padding = 0., {}, {}
        # Re-adding an object drops its matrix entries, so permissions are kept here.
        self.permissions = set()

    def add(self, obj, world_pose):
        self.objects[obj.name] = obj
        self.set_pose(obj.name, world_pose)

    def set_padding(self, padding):
        if padding != self.padding:
            self.padding = padding
            for name, world_pose in self.poses.items():
                if not self.planner.planning_world.is_object_attached(name):
                    self.set_pose(name, world_pose)

    def set_pose(self, name, world_pose):
        world = self.planner.planning_world
        if world.has_object(name):
            world.remove_object(name)
        self.poses[name] = np.asarray(world_pose, dtype=float)
        self.object_padding[name] = self.padding if self.objects[name].padded else 0.
        world.add_object(self.objects[name].fcl(self.frame @ world_pose, self.padding))
        others = sorted(other for pair in self.permissions if name in pair for other in pair if other != name)
        if others:
            world.get_allowed_collision_matrix().set_entry(name, others, True)

    def clear(self):
        """Detach and remove every object this scene added."""
        world = self.planner.planning_world
        for name in self.objects:
            if world.is_object_attached(name):
                world.detach_object(name, False)
            if world.has_object(name):
                world.remove_object(name)
        self.objects, self.poses, self.permissions, self.object_padding = {}, {}, set(), {}

    def set_frame(self, frame):
        """The planner's frame from the world (the robot's base moved): objects re-placed in it."""
        self.frame = np.asarray(frame, dtype=float)
        for name, world_pose in self.poses.items():
            if not self.planner.planning_world.is_object_attached(name):
                self.set_pose(name, world_pose)

    def update(self, body_poses):
        """``body_poses`` maps a body key (``prefix + body``) to its world 4x4."""
        for name, obj in self.objects.items():
            if obj.body in body_poses and not self.planner.planning_world.is_object_attached(name):
                self.set_pose(name, body_poses[obj.body])

    def allow(self, names, others, allowed=True):
        """Grant or revoke contact between planning objects and robot links or other objects."""
        pairs = {frozenset((name, other)) for name in names for other in others}
        if allowed:
            self.permissions |= pairs
        else:
            self.permissions -= pairs
        acm = self.planner.planning_world.get_allowed_collision_matrix()
        acm.set_entry(list(names), list(others), allowed)
