"""Closure checks against authored collision surfaces, including curved and hollow objects."""

from collections import OrderedDict

import numpy as np
from scipy.spatial import ConvexHull
from table_1000.experts.grasps._surface_query import intervals, possible_pieces

from table_1000.experts.grasps import CONTACT_TOLERANCE, _close

# Two contacting surfaces may each penetrate CONTACT_TOLERANCE; Hand's FK
# cache and the closure bisection each resolve at most 0.5 mm of total gap.
GAP_TOLERANCE = 2 * CONTACT_TOLERANCE + .0005 + .0005
PARALLEL_PLANE_TOLERANCE = 1e-10  # the existing ray/parallel-halfspace tolerance


class CollisionSurface:
    """Nearest actual convex surface along each pad ray, retaining hollow gaps.

    A finger inside a ring can touch the nearby arc: the opposite arc behind
    that finger must not be mistaken for penetration of a filled bounding box.
    """

    def __init__(self, shapes):
        self.pieces, self.vertices, bounds = [], [], []
        self._authored_shapes = list(shapes)
        for transform, kind, data in self._authored_shapes:
            if kind == 'sphere':
                self.pieces.append((transform, kind, data))
            elif kind == 'convex':
                # Identical halfspaces from coplanar triangles impose the same
                # constraint. Keep them once, without rounding their coefficients.
                equations = ConvexHull(data).equations
                _, first = np.unique(equations, axis=0, return_index=True)
                self.pieces.append((transform, kind, equations[np.sort(first)]))
            else:
                raise ValueError(f'unsupported asset collision: {kind}')
            points = data if kind == 'convex' else np.vstack([np.eye(3) * data, -np.eye(3) * data])
            vertices = points @ transform[:3, :3].T + transform[:3, 3]
            self.vertices.append(vertices)
            if kind == 'sphere':
                low, high = transform[:3, 3] - data, transform[:3, 3] + data
            else:
                # Scale about an interior point: each face margin m_i grows
                # by at least the existing parallel-plane tolerance when
                # s=1+epsilon/min(m_i). This encloses the tolerance-expanded
                # polyhedron as well as its original vertices.
                center = np.mean(points, axis=0)
                margin = -np.max(equations[:, :3] @ center + equations[:, 3])
                if margin <= 0.:
                    low, high = np.full(3, -np.inf), np.full(3, np.inf)
                else:
                    center = center @ transform[:3, :3].T + transform[:3, 3]
                    outer = center + (vertices - center) * (1. + PARALLEL_PLANE_TOLERANCE / margin)
                    low, high = outer.min(0), outer.max(0)
            bounds.append((np.nextafter(low, -np.inf), np.nextafter(high, np.inf)))
        self._ray_bounds = np.asarray(bounds)
        # Closing-span views share immutable pieces. Their ray directions are
        # identical at successive back-offs, so share one exact-direction plan
        # too. Keep only the last plan, rather than retaining every wrist pose.
        self._query_cache = [None, None]
        self._ray_orders = {}
        self._interval_cache = OrderedDict()

    def points(self):
        return np.vstack(self.vertices)

    def closing_span(self, axis, low, high):
        """A query view excluding surfaces wholly beyond the finger closing sweep.

        Intervals that enter the span keep their original endpoints. The span
        never cuts a solid into artificial opposing faces or changes collision
        clearance. Other views of this body's surfaces remain unchanged.
        """
        result = CollisionSurface.__new__(CollisionSurface)
        result.__dict__.update(self.__dict__)
        result._closing_span = np.asarray(axis), low, high
        return result

    def collision_objects(self, enclosing):
        """Authored colliders not already covered by a candidate's clearance shape.

        The candidate box/cylinder remains conservative. Add the other parts
        of the same body too: naming several fins as one part must not let an
        open finger enter a neighbouring fin. Covered pieces need no duplicate
        FCL check, especially for a whole-body expert target.
        """
        from mplib.collision_detection import fcl
        from table_1000.experts.grasps import Box
        from table_1000.planning.world import geometry, pose
        key = repr(enclosing)
        if not hasattr(self, '_collision_objects'):
            self._collision_objects = {}
        if key not in self._collision_objects:
            objects, poses = [], []
            for (transform, kind, data), vertices in zip(self._authored_shapes, self.vertices):
                if isinstance(enclosing, Box):
                    low, high = (transform[:3, 3] - data, transform[:3, 3] + data) if kind == 'sphere' \
                        else (vertices.min(0), vertices.max(0))
                    covered = np.all(low >= -np.asarray(enclosing.size) / 2) \
                        and np.all(high <= np.asarray(enclosing.size) / 2)
                elif kind == 'sphere':
                    covered = np.linalg.norm(transform[:2, 3]) + data <= enclosing.radius \
                        and abs(transform[2, 3]) + data <= enclosing.length / 2
                else:
                    covered = np.all(np.linalg.norm(vertices[:, :2], axis=1) <= enclosing.radius) \
                        and np.all(np.abs(vertices[:, 2]) <= enclosing.length / 2)
                if covered:
                    continue
                # Use the authored local vertices directly. A rigid round
                # trip through candidate coordinates can perturb a thin face
                # enough to make Trimesh produce a non-watertight hull.
                objects.append(fcl.CollisionObject(geometry(kind, data)))
                poses.append(pose(transform))
            self._collision_objects[key] = objects, poses
        return self._collision_objects[key]

    def normals(self, points):
        """Outward normals at boundary points, without filling gaps between pieces."""
        distance = np.full(len(points), np.inf)
        normals = np.zeros_like(points)
        for transform, kind, data in self.pieces:
            local = (points - transform[:3, 3]) @ transform[:3, :3]
            if kind == 'sphere':
                radius = np.linalg.norm(local, axis=1)
                error, normal = np.abs(radius - data), local / radius[:, None]
            else:
                planes = local @ data[:, :3].T + data[:, 3]
                face = planes.argmax(axis=1)
                error = np.abs(planes[np.arange(len(points)), face])
                normal = data[face, :3]
            closer = error < distance
            normals[closer] = normal[closer] @ transform[:3, :3].T
            distance[closer] = error[closer]
        return normals

    def _query(self, direction):
        """Ray plan for these pieces only; independent of origins and closing span."""
        key = (direction.dtype.str, direction.tobytes())
        if not hasattr(self, '_query_cache'):
            # Contact subsets and individual-section views have different
            # pieces and must own their cache, unlike closing-span views.
            self._query_cache = [None, None]
        if self._query_cache[0] != key:
            transforms, groups, plan = [], {}, []
            for transform, kind, data in self.pieces:
                transform_key = (transform.dtype.str, transform.tobytes())
                if transform_key not in groups:
                    groups[transform_key] = len(transforms)
                    transforms.append(transform)
                vector = transform[:3, :3].T @ direction
                if kind == 'convex':
                    query = data[:, :3] @ vector
                else:
                    query = None
                plan.append((groups[transform_key], kind, data, vector, query))
            groups, starts, planes, slopes, sphere_indices = [], [0], [], [], []
            sphere_count = 0
            for group, kind, data, vector, query in plan:
                groups.append(group)
                if kind == 'convex':
                    # Parallel faces can reject a ray without dividing. Check
                    # them first; all original constraints and coefficients
                    # remain, and min/max endpoints are order independent.
                    order = np.argsort(np.abs(query) >= 1e-12, kind='stable')
                    planes.extend(data[order])
                    slopes.extend(query[order])
                    sphere_indices.append(-1)
                else:
                    sphere_indices.append(sphere_count)
                    sphere_count += 1
                starts.append(len(planes))
            compiled = (np.asarray(groups), np.asarray(starts),
                np.asarray(planes).reshape(-1, 4), np.asarray(slopes), np.asarray(sphere_indices))
            self._query_cache[:] = key, (plan, transforms, compiled)
        return self._query_cache[1]

    def interval(self, origins, direction):
        if len(origins) <= 64:
            return self._interval(origins, direction)
        direction = np.asarray(direction)
        span = getattr(self, '_closing_span', None)
        span_key = None if span is None else (span[0].dtype.str, span[0].tobytes(), span[1], span[2])
        key = (origins.dtype.str, origins.shape, origins.strides, origins.tobytes(),
               direction.dtype.str, direction.shape, direction.tobytes(), span_key)
        if not hasattr(self, '_interval_cache'):
            self._interval_cache = OrderedDict()
        cached = self._interval_cache.get(key)
        if cached is not None:
            self._interval_cache.move_to_end(key)
            return cached[0].copy(), cached[1].copy()
        low, high = self._interval(origins, direction)
        # Closure's last gap is queried again for interior pad support and
        # resistance. Keep a bounded exact-input cache, separate for different
        # piece sets, and return independent writable endpoint arrays.
        self._interval_cache[key] = low.copy(), high.copy()
        if len(self._interval_cache) > 24:
            self._interval_cache.popitem(last=False)
        return low, high

    def _interval(self, origins, direction):
        low, high = np.full(len(origins), np.inf), np.full(len(origins), -np.inf)
        if not len(origins) or not self.pieces:
            return low, high
        direction = np.asarray(direction)
        axis, minimum, maximum = getattr(self, '_closing_span', (np.zeros(3), -np.inf, np.inf))
        if not hasattr(self, '_ray_bounds'):
            self._ray_bounds = np.tile([[-np.inf] * 3, [np.inf] * 3], (len(self.pieces), 1, 1))
        possible = possible_pieces(origins, direction, self._ray_bounds, axis,
                                   minimum, maximum, hasattr(self, '_closing_span'))
        if not possible.any():
            return low, high
        plan, transforms, compiled = self._query(direction)
        # Authored convex pieces often share a frame. Transform the same rays
        # once, using exactly the original arithmetic for each shared frame.
        # A conservative bounds miss above needs neither these transforms nor
        # a halfspace query; its endpoints are already the empty intervals.
        needed = np.zeros(len(transforms), dtype=bool)
        needed[compiled[0][possible]] = True
        locals = np.empty((len(transforms), len(origins), 3))
        for group in np.flatnonzero(needed):
            transform = transforms[group]
            locals[group] = (origins - transform[:3, 3]) @ transform[:3, :3]
        # Spheres keep their original NumPy quadratic arithmetic. Convex
        # halfspaces use the compiled loop, avoiding Python dispatch per piece.
        spheres = []
        for index, (group, kind, data, vector, query) in enumerate(plan):
            if kind != 'sphere':
                continue
            if not possible[index]:
                spheres.append((np.full(len(origins), np.inf), np.full(len(origins), -np.inf)))
                continue
            local = locals[group]
            b = local @ vector
            disc = b * b - (local * local).sum(1) + data ** 2
            root = np.sqrt(np.maximum(disc, 0.))
            spheres.append((np.where(disc >= 0, -b - root, np.inf),
                            np.where(disc >= 0, -b + root, -np.inf)))
        # Samples on the hand are randomly ordered. A fixed spatial ordering
        # makes small ray packets useful; it changes traversal only, and the
        # kernel writes every endpoint back at its original sample index.
        # Any permutation is valid even for a different point set of the same
        # size, so span/contact views can safely share this bounded cache.
        if not hasattr(self, '_ray_orders'):
            self._ray_orders = {}
        if len(origins) not in self._ray_orders:
            from scipy.spatial import cKDTree
            if len(self._ray_orders) >= 8:
                self._ray_orders.pop(next(iter(self._ray_orders)))
            self._ray_orders[len(origins)] = cKDTree(origins, leafsize=16).indices.copy() \
                if len(origins) > 64 else np.arange(len(origins))
        return intervals(locals, *compiled,
            np.asarray(spheres).reshape(-1, 2, len(origins)), possible, origins @ axis,
            direction @ axis, minimum, maximum, hasattr(self, '_closing_span'),
            PARALLEL_PLANE_TOLERANCE, self._ray_orders[len(origins)])

    def at_contacts(self, points):
        """Keep the convex patches that the measured pads actually touch.

        A ring's opposite arc is not part of the held arc's contact opening.
        Selecting by points, rather than part name, also works when many
        collision pieces share one authored part.
        """
        result = CollisionSurface.__new__(CollisionSurface)
        result.pieces, result.vertices, indices = [], [], []
        for index, ((transform, kind, data), vertices) in enumerate(zip(self.pieces, self.vertices)):
            local = (points - transform[:3, 3]) @ transform[:3, :3]
            if kind == 'sphere':
                near = np.linalg.norm(local, axis=1) <= data + CONTACT_TOLERANCE
            else:
                near = (np.einsum('ij,kj->ik', local, data[:, :3]) + data[:, 3]
                        <= CONTACT_TOLERANCE).all(axis=1)
            if near.any():
                result.pieces.append((transform, kind, data))
                result.vertices.append(vertices)
                indices.append(index)
        if hasattr(self, '_ray_bounds'):
            result._ray_bounds = self._ray_bounds[indices]
        if hasattr(self, '_ray_orders'):
            result._ray_orders = self._ray_orders
        return result

    def wide_sections(self, origins, direction, tolerance):
        """Union of each convex patch's full-width sections, keeping gaps empty.

        The distance between two touched pieces is not either piece's
        thickness: rigid assemblies can be pinched on different components.
        """
        found = np.zeros(len(origins), dtype=bool)
        for index, (piece, vertices) in enumerate(zip(self.pieces, self.vertices)):
            component = CollisionSurface.__new__(CollisionSurface)
            component.pieces, component.vertices = [piece], [vertices]
            if hasattr(self, '_ray_bounds'):
                component._ray_bounds = self._ray_bounds[index:index + 1]
            if hasattr(self, '_ray_orders'):
                component._ray_orders = self._ray_orders
            low, high = component.interval(origins, direction)
            found |= high - low >= np.ptp(vertices @ direction) - tolerance
        return found


def pad_contacts(contacts, finger_links, body):
    """Per-finger pad collider contacts with the held body; rigid link alone is insufficient."""
    found = {}
    for link in finger_links:
        rows = []
        for row in contacts:
            for index, label in enumerate(row['bodies']):
                other = row['bodies'][1 - index]
                if label == 'robot.' + link and (other == body or other.startswith(body + '.')):
                    path = row['colliders'][index]
                    if link in row.get('pad_links', []) or any('pad' in part.lower() for part in path.split('/')):
                        rows.append(row)
        found[link] = rows
    return found


def closure(hand, surface, body_pose, tcp, gap, open_gap, contacts, body):
    """Check measured pad contacts and actual-surface gap, with a geometric error budget."""
    pads = pad_contacts(contacts, hand.side.finger_links, body)
    points = [point for rows in pads.values() for row in rows for point in row.get('points', [])]
    if points:
        local = (np.asarray(points) - body_pose[:3, 3]) @ body_pose[:3, :3]
        surface = surface.at_contacts(local)
    # Actual contacts, not another sampling-area test, establish pad support.
    predicted = _close(surface, body_pose, tcp, hand, hand.max_gap, check_area=False)
    expected = None if predicted is None else predicted[0]
    alignment = hand.pad_alignment(gap)
    reasons = []
    if not all(pads.values()):
        reasons.append('missing opposing pad contact')
    if expected is None:
        reasons.append('no opposing collision surfaces')
    elif gap < expected - GAP_TOLERANCE:
        reasons.append('closed past collision surfaces')
    if alignment['edge_only']:
        reasons.append('pad edge only')
    return {'held': not reasons, 'reasons': reasons, 'expected_gap_m': expected,
            'measured_gap_m': gap, 'gap_tolerance_m': GAP_TOLERANCE,
            'pad_contacts': pads, **alignment}
