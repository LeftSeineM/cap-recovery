"""Shared object/part/convex-piece enumeration for skills and asset grasp acceptance."""

import numpy as np

def oriented_shape(points):
    """Fit in the geometry's oriented box, so a diagonal wire keeps its true width."""
    from table_1000.experts.skills.hold.frames import fitted, oriented_bounds
    to_box, _ = oriented_bounds(points)
    local = np.asarray(points) @ to_box[:3, :3].T + to_box[:3, 3]
    shape, offset = fitted(local)
    return shape, np.linalg.inv(to_box) @ offset


def targets(skill, obj="object", part=None, shape=None, offset=None, body=None):
    """Enumerate the existing asset-test order for a named object.

    Explicit part, shape, offset or body selects exactly that target. Otherwise
    keep the solid whole, volume-sorted solid parts and authored convex pieces,
    followed by a hollow whole envelope. ``target_groups`` compares real part
    heads before trying the whole envelope, preserving first-attempt ordering.
    Upright geometric containers add wall-top sections before ordinary parts;
    a closed lid skips blocked interior sections and prefers the exterior whole,
    retaining ordinary exterior parts when that envelope cannot be grasped.
    """
    if part is not None or shape is not None or offset is not None or body is not None or obj not in skill.ctx.geometry:
        yield skill.target(obj, part, shape, offset, body)
        return
    geometry = skill.ctx.geometry[obj]
    def target(part=None):
        result = skill.target(obj, part)
        # The aggregate envelope is tried only after authored part candidates.
        # This applies to solid bodies too: a broad envelope can hide a stable
        # part choice even when it is itself collision-valid.
        if part is None:
            result.fallback = True
        result.shape, result.offset = oriented_shape(geometry.points(part, result.body.split(':', 1)[1]))
        return result
    whole = target()
    solid = geometry.solid(None, narrow=0.)
    if solid:
        yield whole
    parts = [target(name) for body, name in geometry.parts()
             if f'{obj}:{body}' == whole.body and geometry.solid(name, narrow=0.)]
    # A folded sheet or a ring has no solid part-level box. Its authored convex
    # collision pieces provide actual contact surfaces, without asset-specific points.
    for body, name, transform, kind, data in geometry.shapes:
        if f'{obj}:{body}' != whole.body or geometry.solid(name, narrow=0.) or kind != 'convex':
            continue
        shape, offset = oriented_shape(data)
        piece = skill.target(obj, name, shape=shape, offset=transform @ offset)
        parts.append(piece)
    parts.sort(key=lambda target: -float(np.prod(np.ptp(target.shape.points(), axis=0))))
    # A hollow body's tall wall envelope centres the hand halfway down a
    # closed wall. Its upper contact band is the reachable material at the
    # rim. Generate sections from actual convex collision geometry, without
    # authored grasp points or asset names. Solid-asset ordering is unchanged.
    if not solid and hasattr(skill, 'hands'):
        sections = rim_sections(skill, obj, whole, geometry)
        for candidate in [*sections, *parts]:
            candidate.container = whole.container
        parts = sections + parts
        # A lid removes inner-wall sections, not ordinary exterior parts.
        # Keep those as a fallback when the whole envelope has no grasp.
    for target in parts:
        if not solid or target.names != whole.names:
            yield target
    if not solid:
        yield whole


def target_groups(candidates):
    """Real parts first, or the exterior envelope first for a closed body."""
    phases = ('whole', 'rim', 'part') if any(target.fallback and target.enclosed for target in candidates) \
        else ('rim', 'part', 'whole')
    return [group for phase in phases
            if (group := [target for target in candidates
                          if ('rim' if target.rim else
                              'whole' if target.fallback else 'part') == phase])]

def rim_sections(skill, obj, whole, geometry):
    """Actual wall tops of an upright hollow body, in a pad-depth contact band.

    Mark the whole as a container when enclosed empty horizontal sections
    exist. A lid above all sampled cavities marks it enclosed and returns no
    inner-wall targets. Geometry and authored material remain unchanged.
    """
    from table_1000.experts.skills.hold.frames import Box
    from table_1000.experts.skills.hold.sections import Edge, exposed_points
    # Narrow cavities already use the geometry's solid planning envelope.
    # Keep their ordinary target ordering and whole-body expert frame.
    if geometry.solid(None):
        return []
    hand = skill.hands[skill.arms[0]]
    pose = skill.ctx.poses()[whole.body]
    body = whole.body.split(':', 1)[1]
    pieces = [(name, data @ (pose @ transform)[:3, :3].T + (pose @ transform)[:3, 3])
              for owner, name, transform, kind, data in geometry.shapes
              if owner == body and kind == 'convex']
    if not pieces:
        return []
    all_points = np.vstack([points for _, points in pieces])
    extent = np.ptp(all_points, axis=0)
    # Upright containers have an unclamped footprint and thin upright walls.
    # Bent sheets and ordinary broad plates retain their existing route.
    if extent[2] <= min(extent[:2]) / 3:
        return []
    top = all_points[:, 2].max()
    # Require a horizontal empty section enclosed on both axes. This is a
    # geometric container test, excluding forks, coils and decorative slats.
    from table_1000.experts.skills.hold.checks import CollisionSurface
    surface = CollisionSurface([(np.eye(4), 'convex', points) for _, points in pieces])
    center = (all_points.min(0) + all_points.max(0)) / 2
    depth = min(float(np.ptp(points[pads, 0])) for _, points, pads in hand.inner(0.))
    center[2] = top - min(2 * depth, extent[2] / 2)
    planes = [piece[2] for piece in surface.pieces]
    def cavity(point):
        if any(np.all(point @ equations[:, :3].T + equations[:, 3] <= 1e-10) for equations in planes):
            return False
        for axis in np.eye(3)[:2]:
            sides = set()
            for equations in planes:
                distance = equations[:, :3] @ point + equations[:, 3]
                slope = equations[:, :3] @ axis
                if np.any(distance[abs(slope) < 1e-12] > 1e-10):
                    continue
                low = np.max(-distance[slope < -1e-12] / slope[slope < -1e-12], initial=-np.inf)
                high = np.min(-distance[slope > 1e-12] / slope[slope > 1e-12], initial=np.inf)
                if low <= high:
                    sides.add(-1 if high < 0. else 1 if low > 0. else 0)
            if not {-1, 1}.issubset(sides):
                return False
        return True
    cavities = [point for z in (0., .2) for x in (0., -.3, .3) for y in (0., -.3, .3)
                if cavity(point := center + extent * [x, y, -z])]
    if not cavities:
        return []
    whole.container = True
    # Retain an open cavity only if a vertical finger can enter from above.
    # A closed top still admits an exterior whole-body pinch.
    def blocked_above(point):
        for equations in planes:
            distance = equations[:, :3] @ point + equations[:, 3]
            slope = equations[:, 2]
            if np.any(distance[abs(slope) < 1e-12] > 1e-10):
                continue
            low = np.max(-distance[slope < -1e-12] / slope[slope < -1e-12], initial=-np.inf)
            high = np.min(-distance[slope > 1e-12] / slope[slope > 1e-12], initial=np.inf)
            if high >= max(low, 1e-10):
                return True
        return False
    if all(blocked_above(point) for point in cavities):
        whole.enclosed = True
        return []
    plane = Edge(np.array([0., 0., top - depth]), np.array([0., 0., 1.]), np.array([1., 0., 0.]), 0.)
    result = []
    for name, points in pieces:
        size = np.ptp(points, axis=0)
        if points[:, 2].max() < top - depth or size[2] <= depth:
            continue
        local_size = np.sort(np.linalg.svd(points - points.mean(0), compute_uv=False))
        if local_size[0] >= local_size[1] / 2:
            continue
        # A rim section is upright: its frame comes directly from the wall's
        # horizontal thin axis and gravity, avoiding an unconstrained hull
        # orientation search on a clipped sliver.
        _, _, axes = np.linalg.svd(points[:, :2] - points[:, :2].mean(0))
        normal = np.r_[axes[-1], 0.]
        vertical = np.array([0., 0., 1.])
        rotation = np.column_stack([np.cross(normal, vertical), normal, vertical])
        # A thick border above a thin wall is part of this pinch too. Include
        # the adjacent actual material rather than opening to wall thickness
        # and then rejecting the border as an unexpected obstruction.
        local_wall = points @ rotation
        low_wall, high_wall = local_wall.min(0), local_wall.max(0)
        band, names = [points], [name]
        for other_name, other in pieces:
            local_other = other @ rotation
            low_other, high_other = local_other.min(0), local_other.max(0)
            if (low_other[1] >= low_wall[1] - hand.hold_margin * 2
                    and high_other[1] <= high_wall[1] + hand.hold_margin * 2
                    and low_other[0] >= low_wall[0] - depth
                    and high_other[0] <= high_wall[0] + depth):
                band.append(other)
                names.append(other_name)
        clipped = np.vstack([exposed_points(material, plane) for material in band])
        if len(clipped) < 4:
            continue
        local = clipped @ rotation
        low, high = local.min(0), local.max(0)
        world_frame = np.eye(4)
        world_frame[:3, :3] = rotation
        world_frame[:3, 3] = rotation @ ((low + high) / 2)
        shape, offset = Box(tuple(high - low)), np.linalg.inv(pose) @ world_frame
        section = skill.target(obj, name, shape=shape, offset=offset)
        section.names = list(dict.fromkeys(f'{obj}:{part}' for part in names))
        section.rim = True
        result.append(section)
    return result
