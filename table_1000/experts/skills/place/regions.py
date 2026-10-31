"""Placement footprints, orientations and scene-derived target regions."""

from table_1000.experts.skills import semantic
from table_1000.experts.world import geometry as world_geometry

import numpy as np
from scipy.spatial.transform import Rotation

from table_1000.experts.skills.hold.frames import translated
from table_1000.experts.skills.place import strict


NEAR = .30              # m about a given near point that a region is cut down to (fewer spots to rank)


# m an object put on a surface keeps inside its edges: room for the fingers that pick it up again
# (a pen 1 cm from the table's edge was grasped with the edge between the fingers)
EDGE = .04


def headroom(ctx, obj):
    """How high over a support the held ``obj`` sweeps as :func:`~table_1000.experts.skills.place`
    brings it down: its height, the clearance and the height it comes down from."""
    return float(np.ptp(world_geometry.object_points(ctx, obj)[:, 2])) + strict.PLACE_CLEARANCE + strict.PLACE_ABOVE


def around(region, near):
    """``region`` cut down to within NEAR of world point ``near`` (all of it without one), or of
    the region's point nearest it when ``near`` is outside."""
    if near is None:
        return region
    frame, low, high = region[:3]
    point = np.clip((np.linalg.inv(frame) @ np.r_[np.asarray(near, dtype=float)[:3], 1.])[:2], low, high)
    return (frame, np.maximum(low, point - NEAR), np.minimum(high, point + NEAR), *region[3:])


def inset(region, margin):
    """``region`` shrunk by ``margin`` on every side, down to its middle line where it is narrower."""
    frame, low, high = region
    margin = np.minimum(margin, (np.asarray(high) - low) / 2)
    return frame, np.asarray(low) + margin, np.asarray(high) - margin


def on_region(obj, surface, part=None, near=None, keep_out=()):
    """``where(ctx) -> (region, keep_out)``: where on ``surface`` (the table, an object, or its body
    or ``part``) to put ``obj`` down in the state then, out of ``keep_out`` rectangles besides what
    stands there and EDGE inside the surface's edges, within NEAR of ``near`` (world point) if given;
    for :func:`~table_1000.experts.skills.place` (or a :class:`~table_1000.experts.skills.Place`).
    See :func:`support_region`."""
    def where(ctx):
        region, blocked = world_geometry.support_region(world_geometry.scene_shapes(ctx, exclude=(obj,)), surface, part, headroom=headroom(ctx, obj))
        semantic.require(ctx, region is not None, 'no support', surface=surface, part=part)
        return around(inset(region, EDGE), near), [*blocked, *keep_out]
    return where


def inside_region(obj, container, part=None, near=None, *, support=None, orientation=None,
                  top_tolerance=None):
    """``where(ctx) -> (region, keep_out)``: where inside ``container`` (or its body or ``part``: a
    drawer) to put ``obj`` down in the state then, on its floor where nothing covers it.
    ``support`` names an object already inside the container: use its live top
    surface instead of the floor for stacking (largest first). The support must
    satisfy :func:`inside`; an empty/unreachable surface fails with ``no support``.
    With ``orientation='flat'``, the full footprint stays inside the container,
    while its measured centre of mass projects inside the lower object's top.
    A small overhang is allowed; collision checks still exclude walls and other
    objects. Use :func:`on` and containment after settling to reject a slide/tip.
    Re-evaluated before picking and placing by ``Place``; no fixed layer height.
    ``orientation`` matches ``Place``: for upright/flat placement the floor or
    support is found without unrelated objects hiding it; actual convex scene
    shapes above it become 3D keep-outs (including existing pens and walls).
    ``top_tolerance`` optionally bounds the object's resting top by the live
    container geometry's upper edge plus this tolerance. This limits only the
    final pose, not the descent corridor. Omit it for contents such as pens
    that may protrude above the rim. For a part inside a larger container,
    overhead parts exclude the covered floor and bound flat release below
    the measured opening.
    Use ``Place(obj, inside_region(obj, container, support=previous, orientation='flat'),
    orientation='flat')``."""
    def where(ctx):
        if support == obj or support == container:
            raise ValueError('support must name another object already inside the container; '
                             'omit support to place on its floor')
        if support is not None:
            semantic.require(ctx, semantic.inside(ctx, support, container, part), 'stack support not inside container',
                    support=support, container=container)
        shapes = world_geometry.scene_shapes(ctx, exclude=(obj,))
        owner, piece = support or container, None if support else part
        region, blocked = world_geometry.support_region([s for s in shapes if world_geometry.selects(s, owner, piece)]
                                         if orientation else shapes, owner, piece, floor=support is None,
                                         headroom=headroom(ctx, obj))
        semantic.require(ctx, region is not None, 'no support', container=container, part=part)
        stack = None
        if support is not None and orientation == 'flat':
            container_shapes = [s for s in shapes if world_geometry.selects(s, container, part)]
            floor, _ = world_geometry.support_region(container_shapes, container, part, floor=True)
            semantic.require(ctx, floor is not None, 'no container floor', container=container)
            # Cast the live support top along the container's up, even if a
            # notebook was placed with its authored body Z pointing down.
            aligned = [world_geometry.Shape(s.obj, s.body, s.part, floor[0], s.points)
                       for s in shapes if s.obj == support]
            top, _ = world_geometry.support_region(aligned, support, top_envelope=True)
            semantic.require(ctx, top is not None, 'no support', support=support)
            # The lower book can settle slightly tilted. Put the next layer
            # above its highest collision point rather than through the median
            # sampled face; it then settles onto the real top without overlap.
            inv = np.linalg.inv(top[0])
            rise = max((s.points @ inv[:3, :3].T + inv[:3, 3])[:, 2].max() for s in aligned)
            top = (translated(top[0], top[0][:3, 2] * rise), *top[1:])
            target = ctx.skills().target(obj)
            com = (np.linalg.inv(ctx.poses()[target.body])
                   @ np.r_[ctx.env.center_of_mass(target.body)[0], 1.])[:3]
            stack = dict(surface=top, centre_of_mass=com)
            region = (top[0], floor[1], floor[2])
        if orientation:
            if orientation not in ('upright', 'flat', 'insert'):
                raise ValueError("orientation must be None, 'upright', 'flat' or 'insert'")
            if support is not None and orientation in ('upright', 'insert'):
                raise ValueError("insertion with a stack support is unsupported; omit support or use orientation='flat'")
            local = np.linalg.inv(region[0])
            blocked = [s.points for s in shapes
                       if ((s.points @ local[:3, :3].T + local[:3, 3])[:, 2].max() > 1e-6)]
        region = (*region, stack) if stack is not None else region
        if orientation == 'flat':
            live = dict(flat_container=container, part=part, support=support)
            region = (*region[:3], {**(stack or {}), **live})
        if orientation == 'upright':
            region = (*region, dict(upright_container=container, part=part))
        if top_tolerance is not None:
            local = np.linalg.inv(region[0])
            points = np.vstack([s.points for s in shapes if world_geometry.selects(s, container, part)])
            maximum = float((points @ local[:3, :3].T + local[:3, 3])[:, 2].max())
            metadata = region[3] if len(region) > 3 else {}
            metadata = {**metadata, 'maximum_height': maximum + top_tolerance,
                        'top_tolerance': top_tolerance}
            ceilings = []
            for shape in shapes:
                if shape.obj != container or world_geometry.selects(shape, container, part):
                    continue
                points = shape.points @ local[:3, :3].T + local[:3, 3]
                low, high = points.min(0), points.max(0)
                if (low[2] > 1e-6 and np.all(high[:2] >= region[1])
                        and np.all(low[:2] <= region[2])):
                    # Use only the pulled-out floor: a released object above
                    # the housing would land on its shelf or roof instead.
                    blocked.append((region[0], low[:2], high[:2]))
                    ceilings.append(float(low[2]))
            if ceilings:
                metadata['maximum_release_top'] = min(ceilings)
            region = (*region[:3], metadata)
        if support is None:
            metadata = region[3] if len(region) > 3 else {}
            region = (*region[:3], {**metadata, 'release_container': container,
                                  'release_part': part, 'release_support': support})
        return around(region, near), blocked
    return where


def overlaps(outline, low, high):
    """Whether the convex polygon ``outline`` (vertices in order) overlaps the rectangle
    ``low``-``high``: no separating axis among the rectangle's and the polygon's edge normals (a
    long, thin footprint may cross a corner with no vertex inside)."""
    if np.any(outline.max(0) < low) or np.any(outline.min(0) > high):
        return False
    corners = np.array([low, [high[0], low[1]], high, [low[0], high[1]]])
    for edge in np.roll(outline, -1, axis=0) - outline:
        normal = np.array([-edge[1], edge[0]])
        a, b = outline @ normal, corners @ normal
        if a.max() < b.min() or b.max() < a.min():
            return False
    return True


def orientations(points, current, frame, orientation=None):
    """Seed rotations for yaw search, derived from collision extents in the body frame.

    None preserves tilt. ``'upright'`` aligns either end of the longest axis
    with the support normal (length at least twice either width); ``'flat'``
    aligns either face of the shortest axis (both widths at least twice the
    thickness). Unsupported modes/shapes raise ValueError: omit orientation
    for ordinary placement. ``'insert'`` tries either end of every body axis
    for a small object through a narrow mouth. No asset names/layout poses are used.
    """
    if orientation is None:
        return [current[:3, :3]]
    if orientation not in ('upright', 'flat', 'insert'):
        raise ValueError("orientation must be None, 'upright', 'flat' or 'insert'")
    size = np.ptp(points, axis=0)
    order = np.argsort(size)
    upright = orientation == 'upright'
    if orientation != 'insert' and (size[order[2]] < 2 * size[order[1]] if upright else size[order[1]] < 2 * size[order[0]]):
        raise ValueError(f'{orientation} requires a slender or planar collision shape respectively; '
                         'omit orientation for ordinary placement')
    axes = np.eye(3) if orientation == 'insert' else [np.eye(3)[order[2] if upright else order[0]]]
    seeds = [Rotation.align_vectors([sign * frame[:3, 2]], [current[:3, :3] @ axis])[0].as_matrix()
             @ current[:3, :3] for axis in axes for sign in (1., -1.)]
    if orientation == 'flat':
        # A narrow rectangular container may fit only close to its own axes.
        # A yaw grid starting at the held yaw can miss that whole feasible band.
        seeds += [Rotation.align_vectors([sign * frame[:3, 2], frame[:3, 0]],
                                         [current[:3, order[0]], current[:3, order[2]]])[0].as_matrix()
                  @ current[:3, :3] for sign in (1., -1.)]
    return seeds
