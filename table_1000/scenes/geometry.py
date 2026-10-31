"""Geometry of a built asset from its ``model.json``: rigid bodies, parts, collision shapes and joints.

The physics layer of every asset is authored over the same export. Its build
report maps original bodies and their local frames to the surviving physics
bodies, so these are the shapes the simulator collides. Bodies and parts carry their
``table1000:name`` (``cabinet.drawer3``, ``cabinet.drawer3.handle``); poses are
in the asset frame at the built (rest) configuration unless a body frame is
named. Used by scene initialization (resting heights, penetration) and by
experts (part boxes, joint axes).
"""

import json
from pathlib import Path
import re

import numpy as np

from table_1000.planning.world import model_shapes


def simulator_name(name):
    """The USD (and simulator) name of a model name: ``cabinet.drawer3.slide`` -> ``cabinet_drawer3_slide``."""
    return re.sub(r'\W', '_', name)


def transform_points(transform, points):
    return np.asarray(points) @ np.asarray(transform)[:3, :3].T + np.asarray(transform)[:3, 3]


class AssetGeometry:
    def __init__(self, model, centers_of_mass=None, body_mapping=None):
        model = json.loads(Path(model).read_text()) if not isinstance(model, dict) else model
        self.max_drop_m = float(model.get('max_drop_m', .15))
        if not np.isfinite(self.max_drop_m) or self.max_drop_m < 0.:
            raise ValueError('max_drop_m must be a finite nonnegative distance')
        self.shapes, self.rest = model_shapes(model)
        self.joints = {joint['name']: joint for joint in model.get('joints', [])}
        if body_mapping is not None:
            rest = self.rest
            self.rest = {entry['body']: rest[body] @ np.linalg.inv(entry['matrix'])
                         for body, entry in body_mapping.items()}
            self.shapes = [(body_mapping[body]['body'], part,
                            np.asarray(body_mapping[body]['matrix']) @ transform, kind, data)
                           for body, part, transform, kind, data in self.shapes]
            self.joints = {name: {**joint, 'body1': body_mapping[joint['body1']]['body'],
                                 'body2': body_mapping[joint['body2']]['body']}
                           for name, joint in self.joints.items()
                           if body_mapping[joint['body1']]['body'] != body_mapping[joint['body2']]['body']}
        self._hulls = None      # collision hulls in their shape frames, built once
        self._candidate_surfaces = {}  # body/offset -> candidate-frame contact surface
        self._grasp_surfaces = {}      # body -> measured closure surface
        # A one-body asset's authored centre of mass (``{body: point in its frame}``), in the asset
        # frame; None: uniform density.
        self.center_of_mass = None
        if len(self.rest) == 1:
            (body,) = self.rest
            if body in (centers_of_mass or {}):
                self.center_of_mass = transform_points(self.rest[body], [centers_of_mass[body]])[0]

    @classmethod
    def of(cls, assets, asset):
        """Geometry of the built asset ``asset`` (path under ``assets``), with the centres of mass its
        physics layer authors (``physics_build.json``). Merged builds require its
        ``body_mapping``; rebuild old products with ``scripts/assets/build_physics.py``.
        Part names stay model names; body names and frames match live physics."""
        report = Path(assets) / asset / 'physics_build.json'
        report = json.loads(report.read_text()) if report.is_file() else {}
        geometry = cls(Path(assets) / asset / 'model.json', report.get('centers_of_mass'), report.get('body_mapping'))
        if 'body_mapping' not in report and report.get('rigid_bodies', len(geometry.bodies)) != len(geometry.bodies):
            raise ValueError(f'{asset}: physics_build.json lacks body_mapping for merged rigid bodies; '
                             'rebuild this asset with scripts/assets/build_physics.py --output <private objects root> '
                             f'--assets {Path(asset).relative_to("objects")}')
        return geometry

    @property
    def bodies(self):
        return list(self.rest)

    def _selected(self, part):
        return [shape for shape in self.shapes
                if part is None or shape[1] == part or shape[1].startswith(part + '.') or shape[0] == part]

    def parts(self, part=None):
        """``[(body, part)]`` of ``part`` (a body, a part and its sub-parts, or everything), in model order."""
        return list(dict.fromkeys((shape[0], shape[1]) for shape in self._selected(part)))

    def points(self, part=None, frame=None):
        """Collision vertices of ``part`` (a body, a part and its sub-parts, or everything) in the
        frame of body ``frame`` (default: the asset frame), at rest. Spheres contribute the six
        points of their axis-aligned extent."""
        to_frame = np.eye(4) if frame is None else np.linalg.inv(self.rest[frame])
        points = []
        for body, _, transform, kind, data in self._selected(part):
            local = data if kind == 'convex' else np.vstack([np.eye(3) * data, -np.eye(3) * data])
            points.append(transform_points(to_frame @ self.rest[body] @ transform, local))
        if not points:
            raise KeyError(f'no collision geometry for {part!r}')
        return np.vstack(points)

    def bounds(self, part=None, frame=None):
        """``(low, high)`` corners of ``part``'s collision box, axis-aligned in ``frame``."""
        points = self.points(part, frame)
        return points.min(0), points.max(0)

    def stable_rotation(self, rotation):
        """The resting orientation of the asset on a flat support nearest to ``rotation`` (3x3), up
        to a turn about the vertical: one of the stable poses of its collision hull (uniform
        density), turned to best match ``rotation``. A pen lying across a vertex of its polygonal
        barrel rolls onto the nearest face. An authored centre of mass replaces the
        uniform density's (a desk lamp's heavy base keeps it upright)."""
        import trimesh
        if not hasattr(self, '_stable'):
            hull = trimesh.convex.convex_hull(self.points())
            poses = trimesh.poses.compute_stable_poses(hull, center_mass=self.center_of_mass)[0]
            self._stable = [transform[:3, :3] for transform in poses]
        best = None
        for stable in self._stable:
            difference = np.asarray(rotation) @ stable.T
            yaw = np.arctan2(difference[1, 0] - difference[0, 1], difference[0, 0] + difference[1, 1])
            c, s = np.cos(yaw), np.sin(yaw)
            turned = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) @ stable
            error = np.arccos(np.clip((np.trace(np.asarray(rotation).T @ turned) - 1) / 2, -1., 1.))
            if best is None or error < best[0]:
                best = (error, turned)
        return best[1]

    def solid(self, part, tolerance=1.2, narrow=.02):
        """Whether ``part``'s collision shapes may be replaced by their common convex hull for
        planning: together they are about convex (the hull at most ``tolerance`` times their summed
        volume), or too thin for anything to get into their hollows (narrower than ``narrow``, as a
        ring or a tube of a pen cap)."""
        import trimesh
        shapes, volumes = [], []
        for body, _, transform, kind, data in self._selected(part):
            local = data if kind == 'convex' else np.vstack([np.eye(3) * data, -np.eye(3) * data])
            points = transform_points(self.rest[body] @ transform, local)
            shapes.append(points)
            volumes.append(4 / 3 * np.pi * data ** 3 if kind == 'sphere' else trimesh.convex.convex_hull(points).volume)
        if len(shapes) == 1:
            return True
        hull = trimesh.convex.convex_hull(np.vstack(shapes))
        return hull.volume <= tolerance * sum(volumes) or min(hull.bounding_box_oriented.primitive.extents) < narrow

    def lowest(self, rotation):
        """Lowest collision point (z, asset frame origin at 0) of the asset turned by ``rotation`` (3x3)."""
        return float((self.points() @ np.asarray(rotation).T)[:, 2].min())

    def joint_axis(self, joint):
        """``(body1, origin, axis)`` of ``joint`` (its ``model.json`` name): the frame body and, in it,
        the joint origin and the unit direction of positive motion (a slider's travel)."""
        entry = self.joints[joint]
        matrix = np.array(entry['matrix'], dtype=float)
        return entry['body1'], matrix[:3, 3], matrix[:3, 0] / np.linalg.norm(matrix[:3, 0])

    def joint(self, name):
        """The ``model.json`` entry of joint ``name``: its model name (``cabinet.drawer3.slide``) or
        the simulator's (``cabinet_drawer3_slide``)."""
        if name in self.joints:
            return self.joints[name]
        matches = [entry for model_name, entry in self.joints.items() if simulator_name(model_name) == name]
        if len(matches) != 1:
            raise KeyError(f'no joint {name!r}')
        return matches[0]

    def body_poses(self, transform, joints=None):
        """``{body: world pose}`` of every body with the asset frame at ``transform`` and the joints at
        ``joints`` (``{name: value}``, m or rad; others at rest). A joint moves its second body
        relative to its first: a slider along the joint frame's X, a hinge about its Z."""
        poses = dict(self.rest)
        for name, value in (joints or {}).items():
            entry = self.joint(name)
            frame = np.array(entry['matrix'], dtype=float)
            moved = poses[entry['body1']] @ np.linalg.inv(self.rest[entry['body1']]) @ frame
            motion = np.eye(4)
            if entry['type'] == 'SLIDER':
                motion[0, 3] = value
            else:
                c, s = np.cos(value), np.sin(value)
                motion[:2, :2] = [[c, -s], [s, c]]
            poses[entry['body2']] = moved @ motion @ np.linalg.inv(frame) @ self.rest[entry['body2']]
        return {body: np.asarray(transform) @ pose for body, pose in poses.items()}

    def hulls(self, body_poses):
        """:class:`Hull` of every collision shape posed by ``body_poses`` (``{body: world pose}``)."""
        if self._hulls is None:
            self._hulls = [Hull.of(data if kind == 'convex' else sphere_points(data))
                           for _, _, _, kind, data in self.shapes]
        return [hull.posed(body_poses[body] @ transform)
                for (body, _, transform, _, _), hull in zip(self.shapes, self._hulls)]


class Hull:
    """A convex polyhedron: its vertices and the axes that can separate it from another one (face
    normals and edge directions), in one frame, with its axis-aligned bounds.

    Overlap is decided by the separating axis theorem, which is exact for convex polyhedra:
    FCL's convex queries misjudge the thin hulls of these assets both ways (wire loops several
    millimetres apart reported colliding, overlapping parts reported apart).
    """

    def __init__(self, vertices, normals, edges):
        self.vertices, self.normals, self.edges = vertices, normals, edges
        self.low, self.high = vertices.min(0), vertices.max(0)

    @classmethod
    def of(cls, points):
        from scipy.spatial import ConvexHull
        points = np.asarray(points, dtype=float)
        hull = ConvexHull(points, qhull_options='QJ')
        normals = unique_directions(hull.equations[:, :3])
        edges = np.vstack([points[hull.simplices[:, i]] - points[hull.simplices[:, (i + 1) % 3]] for i in range(3)])
        edges = unique_directions(edges[np.linalg.norm(edges, axis=1) > 1e-9])
        return cls(points[hull.vertices], normals, edges)

    def posed(self, transform):
        rotation, translation = np.asarray(transform)[:3, :3], np.asarray(transform)[:3, 3]
        return Hull(self.vertices @ rotation.T + translation, self.normals @ rotation.T, self.edges @ rotation.T)


def unique_directions(vectors, decimals=6):
    """Unit vectors of ``vectors`` with opposite and repeated directions removed."""
    units = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    units *= np.where(units[np.arange(len(units)), np.abs(units).argmax(1)] < 0, -1., 1.)[:, None]
    return np.unique(units.round(decimals), axis=0)


def sphere_points(radius):
    """Vertices of an icosphere (42 points) of ``radius`` around the origin."""
    import trimesh
    return trimesh.creation.icosphere(subdivisions=1, radius=radius).vertices


def overlap(a, b):
    """Depth (m) by which hulls ``a`` and ``b`` overlap along their least overlapping axis; zero or
    negative when they only touch or are apart."""
    if (a.low > b.high).any() or (b.low > a.high).any():
        return -float(max((a.low - b.high).max(), (b.low - a.high).max()))
    crossed = np.cross(a.edges[:, None], b.edges[None]).reshape(-1, 3)
    lengths = np.linalg.norm(crossed, axis=1)
    axes = np.vstack([a.normals, b.normals, crossed[lengths > 1e-6] / lengths[lengths > 1e-6, None]])
    pa, pb = a.vertices @ axes.T, b.vertices @ axes.T
    return float(np.minimum(pa.max(0) - pb.min(0), pb.max(0) - pa.min(0)).min())


def touching_pairs(hulls_a, hulls_b):
    """Pairs of a hull of each list whose bounding boxes overlap: the only ones that can."""
    if not hulls_a or not hulls_b:
        return []
    low_a, high_a = np.array([a.low for a in hulls_a]), np.array([a.high for a in hulls_a])
    low_b, high_b = np.array([b.low for b in hulls_b]), np.array([b.high for b in hulls_b])
    near = ((low_a[:, None] <= high_b[None]) & (low_b[None] <= high_a[:, None])).all(-1)
    return [(hulls_a[i], hulls_b[j]) for i, j in zip(*np.nonzero(near))]


def penetration(hulls_a, hulls_b):
    """Deepest penetration (m, 0 when apart or only touching) between two lists of hulls."""
    return max([0., *(overlap(a, b) for a, b in touching_pairs(hulls_a, hulls_b))])


def colliding(hulls_a, hulls_b):
    """Whether any hull of one list overlaps one of the other."""
    return any(overlap(a, b) > 0 for a, b in touching_pairs(hulls_a, hulls_b))


def box_hull(size, transform):
    """:class:`Hull` of a box of ``size`` centred on ``transform``."""
    half = np.asarray(size, dtype=float) / 2
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half
    return Hull.of(corners).posed(transform)
