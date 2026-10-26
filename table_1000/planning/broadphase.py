"""Conservative AABB rejection before the existing exact FCL collision queries."""

import numpy as np


class BroadphaseWorld:
    """Accelerate Python queries on a single-articulation MPlib world.

    The native world remains authoritative and is still used by native RRT.
    Geometry stays unchanged. Object poses, attachments and collision permissions
    are read live; cached local bounds depend only on immutable shape geometry.
    Other world configurations and explicit requests use the native implementation.
    """

    def __init__(self, world):
        self.world = world
        self.signature = None
        self.geometry = {}
        self.local = {}
        self.bounds = {}
        self.pair_signature = None
        self.base_signature = None
        self.versions = {}

    def __getattr__(self, name):
        return getattr(self.world, name)

    def cache_object(self, obj, geometry_key):
        """Record a scene-owned replacement with an immutable geometry version."""
        if geometry_key not in self.versions:
            self.versions[geometry_key] = self._local_bounds(obj)
        self.bounds['object', obj.name] = (obj, *self.versions[geometry_key])

    def _geometry_bounds(self, geom):
        from mplib.collision_detection import fcl
        key = id(geom)
        if key not in self.geometry:
            if isinstance(geom, (fcl.Convex, fcl.BVHModel)):
                points = np.asarray(geom.get_vertices()).reshape(-1, 3)
                low, high = points.min(0), points.max(0)
            elif isinstance(geom, fcl.Box):
                high = np.asarray(geom.side).ravel() / 2
                low = -high
            elif isinstance(geom, fcl.Sphere):
                high = np.full(3, geom.radius)
                low = -high
            elif isinstance(geom, fcl.Cylinder):
                high = np.array([geom.radius, geom.radius, geom.lz / 2])
                low = -high
            else:
                geom.compute_local_aabb()
                centre = np.asarray(geom.aabb_center).ravel()
                low, high = centre - geom.aabb_radius, centre + geom.aabb_radius
            self.geometry[key] = (geom, (low + high) / 2, (high - low) / 2)
        return self.geometry[key][1:]

    def _local_bounds(self, obj):
        # FCLObject replacements can share geometry (pose updates, or returning
        # to an earlier padding). Relative shape poses are part of the key too.
        shapes = [shape.get_collision_geometry() for shape in obj.shapes]
        frames = [relative.to_transformation_matrix() for relative in obj.shape_poses]
        key = tuple((id(shape), frame.tobytes()) for shape, frame in zip(shapes, frames))
        if key not in self.local:
            self.local[key] = (shapes, self._shape_bounds(shapes, frames))
        return self.local[key][1]

    def _shape_bounds(self, shapes, frames):
        lows, highs = [], []
        for shape, frame in zip(shapes, frames):
            centre, half = self._geometry_bounds(shape)
            if not np.all(np.isfinite(half)):
                return np.zeros(3), np.full(3, np.inf)
            centre = frame[:3, :3] @ centre + frame[:3, 3]
            half = np.abs(frame[:3, :3]) @ half
            lows.append(centre - half)
            highs.append(centre + half)
        if not lows:
            return np.zeros(3), np.full(3, -np.inf)
        low, high = np.min(lows, axis=0), np.max(highs, axis=0)
        return (low + high) / 2, (high - low) / 2

    def _prepare(self, robot, links, objects, attached, self_pairs):
        signature = (id(robot), tuple(map(id, links)), tuple(map(id, objects)),
                     tuple(attached), tuple(map(tuple, self_pairs)))
        if signature == self.signature:
            return
        self.objects = links + objects
        bounds = {}
        for kind, group in (('link', links), ('object', objects)):
            for obj in group:
                key = kind, obj.name
                previous = self.bounds.get(key)
                bounds[key] = (previous if previous is not None and previous[0] is obj
                               else (obj, *self._local_bounds(obj)))
        self.bounds = bounds
        self.centres = np.asarray([entry[1] for entry in bounds.values()])
        self.halves = np.asarray([entry[2] for entry in bounds.values()])
        self.signature = signature
        # Pair topology is independent of object identity and geometry/padding.
        # Permissions are consulted live at the exact query, so ACM edits need
        # neither geometry work nor pair reconstruction.
        pair_signature = (id(robot), tuple(obj.name for obj in links), tuple(obj.name for obj in objects),
                          tuple(attached), tuple(map(tuple, self_pairs)))
        if pair_signature == self.pair_signature:
            return
        self._prepare_pairs(robot, links, attached, self_pairs)
        self.pair_signature = pair_signature

    def _prepare_pairs(self, robot, links, attached, self_pairs):
        name = robot.get_name()
        signature = (id(robot), tuple(obj.name for obj in self.objects), tuple(map(tuple, self_pairs)), len(links))

        def entry(i, j, kind, first, second):
            return kind, first, second, self.objects[i].name, self.objects[j].name

        if signature != self.base_signature:
            # Every link/object pair exists regardless of attachment. Keep this
            # large block while only the held-object pairs and result types vary.
            robot_objects = np.column_stack((
                np.repeat(np.arange(len(links)), len(attached)),
                np.tile(np.arange(len(links), len(self.objects)), len(links))))
            self.base_indices = np.concatenate((np.asarray(self_pairs, dtype=int).reshape(-1, 2), robot_objects))
            self.base_entries = [entry(i, j, 'self', name, name) for i, j in self_pairs]
            self.base_entries += [entry(i, len(links) + j,
                                        'self_attached' if held else 'self_sceneobject', name, obj.name)
                                  for i in range(len(links))
                                  for j, (obj, held) in enumerate(zip(self.objects[len(links):], attached))]
            self.base_signature = signature
        else:
            for j, (previous, held) in enumerate(zip(self.attached, attached)):
                if previous != held:
                    obj = self.objects[len(links) + j]
                    for i in range(len(links)):
                        index = len(self_pairs) + i * len(attached) + j
                        self.base_entries[index] = entry(i, len(links) + j,
                                                         'self_attached' if held else 'self_sceneobject', name, obj.name)
        self.attached = tuple(attached)
        extra, entries = [], []
        held = [len(links) + i for i, value in enumerate(attached) if value]
        scene = [len(links) + i for i, value in enumerate(attached) if not value]
        for at, i in enumerate(held):
            for j in held[:at]:
                extra.append((i, j))
                entries.append(entry(i, j, 'attached_attached', self.objects[i].name, self.objects[j].name))
            for j in scene:
                extra.append((i, j))
                entries.append(entry(i, j, 'attached_sceneobject', self.objects[i].name, self.objects[j].name))
        self.indices = (np.concatenate((self.base_indices, np.asarray(extra, dtype=int).reshape(-1, 2)))
                        if extra else self.base_indices)
        self.entries = self.base_entries + entries if entries else self.base_entries

    def check_collision(self, request=None):
        from mplib.collision_detection import AllowedCollision, WorldCollisionResult, fcl
        robots = self.world.get_planned_articulations()
        if request is not None:
            return self.world.check_collision(request)
        if len(robots) != 1 or len(self.world.get_articulation_names()) != 1:
            return self.world.check_collision()
        robot = robots[0]
        model = robot.get_fcl_model()
        links = model.get_collision_objects()
        objects = [self.world.get_object(name) for name in self.world.get_object_names()]
        attached = [self.world.is_object_attached(obj.name) for obj in objects]
        for obj, held in zip(objects, attached):
            if held:
                self.world.get_attached_object(obj.name).update_pose()
        self._prepare(robot, links, objects, attached, model.get_collision_pairs())
        if not len(self.indices):
            return []
        frames = np.array([obj.pose.to_transformation_matrix() for obj in self.objects])
        centres = np.einsum('nij,nj->ni', frames[:, :3, :3], self.centres) + frames[:, :3, 3]
        finite = np.all(np.isfinite(self.halves), axis=1)
        halves = np.full_like(self.halves, np.inf)
        halves[finite] = np.einsum('nij,nj->ni', np.abs(frames[finite, :3, :3]), self.halves[finite])
        # Cover roundoff and the default FCL GJK tolerance at touching boundaries.
        halves += 1e-6 + 8 * np.finfo(float).eps * np.abs(centres)
        low, high = centres - halves, centres + halves
        empty = np.any(self.halves < 0, axis=1)
        low[empty], high[empty] = np.inf, -np.inf
        i, j = self.indices.T
        candidates = np.flatnonzero(np.all((low[i] <= high[j]) & (low[j] <= high[i]), axis=1))
        matrix = self.world.get_allowed_collision_matrix()
        result, query = [], fcl.CollisionRequest()
        for index in candidates:
            first, second = self.indices[index]
            obj1, obj2 = self.objects[first], self.objects[second]
            if matrix.get_allowed_collision(obj1.name, obj2.name) not in (None, AllowedCollision.NEVER):
                continue
            collision = fcl.collide(obj1, obj2, query)
            if collision.is_collision():
                result.append(WorldCollisionResult(collision, *self.entries[index]))
        return result
