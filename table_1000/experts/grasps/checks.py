"""Closure checks against authored collision surfaces, including curved and hollow objects."""

import numpy as np
from scipy.spatial import ConvexHull
from table_1000.experts.grasps._surface_query import intervals

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
        for transform, kind, data in shapes:
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
            for (transform, kind, data), vertices in zip(self.pieces, self.vertices):
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
                objects.append(fcl.CollisionObject(geometry(kind, data if kind == 'sphere' else vertices)))
                poses.append(pose(transform if kind == 'sphere' else np.eye(4)))
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
        low, high = np.full(len(origins), np.inf), np.full(len(origins), -np.inf)
        if not len(origins) or not self.pieces:
            return low, high
        direction = np.asarray(direction)
        possible = np.ones(len(self.pieces), dtype=bool)
        if len(origins) and len(self.pieces) and hasattr(self, '_ray_bounds'):
            # Any ray origin lies in this box. A translated ray can hit a
            # piece only if the line through zero hits piece_box-origin_box.
            bound_low = self._ray_bounds[:, 0] - origins.max(0)
            bound_high = self._ray_bounds[:, 1] - origins.min(0)
            parallel = direction == 0.
            possible = ((bound_low[:, parallel] <= 0.) & (bound_high[:, parallel] >= 0.)).all(1)
            if (~parallel).any():
                a, b = bound_low[:, ~parallel] / direction[~parallel], bound_high[:, ~parallel] / direction[~parallel]
                enter = np.nextafter(np.minimum(a, b), -np.inf).max(1)
                leave = np.nextafter(np.maximum(a, b), np.inf).min(1)
                possible &= leave >= enter
            if hasattr(self, '_closing_span'):
                axis, minimum, maximum = self._closing_span
                finite = np.isfinite(self._ray_bounds).all(axis=(1, 2))
                bounds = self._ray_bounds[finite]
                lower = np.where(axis >= 0., bounds[:, 0], bounds[:, 1]) @ axis
                upper = np.where(axis >= 0., bounds[:, 1], bounds[:, 0]) @ axis
                possible[finite] &= (np.nextafter(upper, np.inf) >= minimum) \
                    & (np.nextafter(lower, -np.inf) <= maximum)
        if not possible.any():
            return low, high
        plan, transforms, compiled = self._query(direction)
        # Authored convex pieces often share a frame. Transform the same rays
        # once, using exactly the original arithmetic for each shared frame.
        # A conservative bounds miss above needs neither these transforms nor
        # a halfspace query; its endpoints are already the empty intervals.
        locals = [(origins - transform[:3, 3]) @ transform[:3, :3]
                  for transform in transforms]
        # Spheres keep their original NumPy quadratic arithmetic. Convex
        # halfspaces use the compiled loop, avoiding Python dispatch per piece.
        spheres = []
        for group, kind, data, vector, query in plan:
            if kind != 'sphere':
                continue
            local = locals[group]
            b = local @ vector
            disc = b * b - (local * local).sum(1) + data ** 2
            root = np.sqrt(np.maximum(disc, 0.))
            spheres.append((np.where(disc >= 0, -b - root, np.inf),
                            np.where(disc >= 0, -b + root, -np.inf)))
        axis, minimum, maximum = getattr(self, '_closing_span', (np.zeros(3), -np.inf, np.inf))
        return intervals(np.asarray(locals), *compiled,
            np.asarray(spheres).reshape(-1, 2, len(origins)), possible, origins @ axis,
            direction @ axis, minimum, maximum, hasattr(self, '_closing_span'),
            PARALLEL_PLANE_TOLERANCE)

    def at_contacts(self, points):
        """Keep the convex patches that the measured pads actually touch.

        A ring's opposite arc is not part of the held arc's contact opening.
        Selecting by points, rather than part name, also works when many
        collision pieces share one authored part.
        """
        result = CollisionSurface.__new__(CollisionSurface)
        result.pieces, result.vertices = [], []
        for (transform, kind, data), vertices in zip(self.pieces, self.vertices):
            local = (points - transform[:3, 3]) @ transform[:3, :3]
            if kind == 'sphere':
                near = np.linalg.norm(local, axis=1) <= data + CONTACT_TOLERANCE
            else:
                near = (np.einsum('ij,kj->ik', local, data[:, :3]) + data[:, 3]
                        <= CONTACT_TOLERANCE).all(axis=1)
            if near.any():
                result.pieces.append((transform, kind, data))
                result.vertices.append(vertices)
        return result

    def wide_sections(self, origins, direction, tolerance):
        """Union of each convex patch's full-width sections, keeping gaps empty.

        The distance between two touched pieces is not either piece's
        thickness: rigid assemblies can be pinched on different components.
        """
        found = np.zeros(len(origins), dtype=bool)
        for piece, vertices in zip(self.pieces, self.vertices):
            component = CollisionSurface.__new__(CollisionSurface)
            component.pieces, component.vertices = [piece], [vertices]
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
