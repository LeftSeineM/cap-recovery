"""Geometry for thin-object preparation, independent of asset names and simulation."""

from dataclasses import dataclass
from collections import OrderedDict
from functools import lru_cache
from weakref import WeakKeyDictionary

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from table_1000.experts.grasps import Box, SUPPORT_CLEARANCE, oriented_bounds


def box_extents(points):
    return np.sort(oriented_bounds(np.asarray(points))[1])


def pad_height(hand, gap):
    """Height above a touching fingertip needed for an interior pad contact patch."""
    count = int(np.ceil(hand.min_pad_area / hand.pad_spacing ** 2))
    heights = []
    for _, points, pads in hand.inner(gap):
        depth = points[pads, 0]
        interior = np.sort(depth[depth <= depth.max() - hand.hold_margin])[::-1]
        heights.append(hand.tip(gap) - interior[count - 1] + SUPPORT_CLEARANCE)
    return max(heights)


@dataclass(frozen=True)
class ThinGeometry:
    kind: str
    thickness: float
    contact_height: float
    contact_band: float


def classify(points, convex_pieces, hand):
    """Sheets have one small local extent; filaments have two.

    Volume weighting ignores tiny decorative collision pieces. Local geometry
    admits bent sheets; the whole-body bound excludes the thin walls of a tall
    container. The scale is the hand's interior contact patch and edge margins.
    """
    extent = box_extents(points)
    height = pad_height(hand, min(float(extent[1]), hand.max_gap))
    band = max(height, 2 * hand.hold_margin)
    pieces = [(box_extents(p), ConvexHull(p).volume) for p in convex_pieces]
    if not pieces:
        return None  # sphere-only geometry has no sheet or filament patch
    pieces.sort(key=lambda p: p[0][0])
    total = sum(volume for _, volume in pieces)
    median = pieces[np.searchsorted(np.cumsum([volume for _, volume in pieces]), total / 2)][0]
    thickness = float(median[0])
    # A broad plate can be thick enough for an ordinary pad pinch, but its
    # two in-plane dimensions exceed the opening. It needs the same exposed
    # edge as a sheet. Use the whole-body extent as well as material thickness
    # so the thin wall of a tall hollow container cannot select this route.
    plate = (extent[1] > hand.max_gap and extent[0] < hand.max_gap
             and extent[0] <= band + 2 * thickness and thickness < hand.max_gap)
    if plate:
        return ThinGeometry('plate', float(extent[0]), height, band)
    if thickness >= band or extent[0] > band + 2 * thickness:
        return None
    filament_volume = sum(volume for size, volume in pieces if size[1] < band)
    return ThinGeometry('filament' if filament_volume >= total / 2 else 'sheet', thickness, height, band)


@dataclass(frozen=True)
class Edge:
    point: np.ndarray
    outward: np.ndarray
    tangent: np.ndarray
    half_length: float

    def distances(self, points):
        return (np.asarray(points) - self.point) @ self.outward


def table_edges(table, robot_position):
    """Table edges ordered by distance from the robot, each pointing out of support."""
    center = np.array([*table['center'], table['height']], dtype=float)
    half = np.asarray(table['size']) / 2
    edges = []
    for axis in range(2):
        for sign in (-1., 1.):
            outward = np.eye(3)[axis] * sign
            point = center + outward * half[axis]
            edges.append(Edge(point, outward, np.cross([0., 0., 1.], outward), float(half[1 - axis])))
    robot = np.asarray(robot_position)[:2]
    return sorted(edges, key=lambda edge: np.linalg.norm(edge.point[:2] - robot))


def overhang_depth(hand, transverse_span, area=None):
    """Interior contact area plus margins at the finger tip and object edge."""
    pad_span = min(float(np.ptp(points[pads, 2])) for _, points, pads in hand.inner(0.))
    area = hand.min_pad_area if area is None else area
    return 2 * hand.hold_margin + area / min(pad_span, transverse_span)


def supported_overhangs(minimum, maximum, step, coordinate_scale=1., preferred=None):
    """Search the whole supported interval, including its incomplete final step.

    A short body's only table-clear grasp may lie in that final interval.
    Keep the upper endpoint inside support by floating-point roundoff only;
    physical clearance and contact area are checked by the normal planner.
    """
    upper = maximum - 8 * np.spacing(max(1., coordinate_scale))
    if upper < minimum:
        return np.empty(0)
    samples = np.r_[np.arange(minimum, upper, step), upper]
    if preferred is None:
        return samples
    # Reobserving a drag changes the target pose, not the need to start again
    # at the smallest exposed sliver. Revalidate the last useful contact patch
    # first, then nearby depths, retaining every original sample as fallback.
    preferred = float(np.clip(preferred, minimum, upper))
    return samples[np.argsort(abs(samples - preferred), kind='stable')]


def landing(points, center_of_mass, edge, overhang, along=None):
    """Translation exposing ``overhang`` while keeping the COM and footprint supported.

    None means the requested patch would tip the body or extend past a table
    corner. ``along`` requests the COM coordinate along the edge tangent;
    footprint limits clamp it before reachability and obstacle checks.
    """
    points = np.asarray(points)
    shift = edge.outward * (overhang - edge.distances(points).max())
    coordinates = (points + shift - edge.point) @ edge.tangent
    low, high = -edge.half_length - coordinates.min(), edge.half_length - coordinates.max()
    if low > high:
        return None
    desired = 0. if along is None else along - (np.asarray(center_of_mass) + shift - edge.point) @ edge.tangent
    shift += edge.tangent * np.clip(desired, low, high)
    if edge.distances(np.asarray(center_of_mass) + shift) >= 0:
        return None
    return shift


def _swept_obstacles(convex_pieces, displacement, obstacles):
    """Yield obstacles hit by continuous translation of the actual convex union.

    Each piece sweeps its own hull; empty space between pieces stays empty.
    ``obstacles`` contains world-frame FCL objects, without the support table.
    """
    from mplib.collision_detection import fcl
    from table_1000.planning.world import geometry, pose
    objects = [fcl.CollisionObject(geometry('convex', np.vstack([piece, piece + displacement])))
               for piece in convex_pieces]
    swept = fcl.FCLObject('thin target sweep', pose(np.eye(4)), objects,
                          [pose(np.eye(4)) for _ in objects])
    yield from (name for name, obstacle in obstacles
                if fcl.collide(swept, obstacle, fcl.CollisionRequest()).is_collision())


def swept_obstacles(convex_pieces, displacement, obstacles):
    """All blockers of the continuous translation of the actual convex union."""
    return list(_swept_obstacles(convex_pieces, displacement, obstacles))


def swept_obstacle(convex_pieces, displacement, obstacles):
    """First blocker of the actual convex translation; see :func:`swept_obstacles`."""
    return next(_swept_obstacles(convex_pieces, displacement, obstacles), None)


def exposed_points(points, edge):
    """Clip a convex collision polytope to the unsupported side of a table edge."""
    points = np.asarray(points, dtype=float)
    distances = edge.distances(points)
    outside = points[distances >= 0.]
    if not len(outside):
        return np.empty((0, 3))
    hull = ConvexHull(points)
    segments = {tuple(sorted((int(a), int(b)))) for face in hull.simplices
                for a, b in zip(face, np.roll(face, -1))}
    intersections = []
    for a, b in segments:
        if distances[a] * distances[b] < 0:
            intersections.append(points[a] + (points[b] - points[a]) * distances[a] / (distances[a] - distances[b]))
    return np.unique(np.vstack([outside, *[p[None] for p in intersections]]).round(12), axis=0)


def support_polygon(convex_pieces, table, contact_tolerance):
    """Hull of collision material in the existing table contact layer.

    A bent body's airborne outline cannot provide support. Clip its low
    material first, then the table footprint; rigidly connected contacts
    together carry a resultant inside their convex hull.
    """
    plane = Edge(np.array([0., 0., table['height'] + contact_tolerance]),
                 np.array([0., 0., -1.]), np.array([1., 0., 0.]), 0.)
    low = [exposed_points(piece, plane)[:, :2] for piece in convex_pieces]
    points = np.vstack(low) if low else np.empty((0, 2))
    if len(points) < 3:
        return np.empty((0, 2))
    try:
        polygon = points[ConvexHull(points).vertices]
    except QhullError:
        return np.empty((0, 2))
    for edge in table_edges(table, (0., 0., 0.)):
        clipped = []
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            da, db = (np.array([a, b]) - edge.point[:2]) @ edge.outward[:2]
            if da <= 0.:
                clipped.append(a)
            if da * db < 0.:
                clipped.append(a + (b - a) * da / (da - db))
        polygon = np.asarray(clipped).reshape(-1, 2)
        if len(polygon) < 3:
            return np.empty((0, 2))
    return polygon


def supported_point(point, polygon):
    """A downward load must lie strictly inside the support polygon."""
    if len(polygon) < 3:
        return False
    sides = np.roll(polygon, -1, axis=0) - polygon
    delta = np.asarray(point)[:2] - polygon
    distance = (sides[:, 0] * delta[:, 1] - sides[:, 1] * delta[:, 0]) / np.linalg.norm(sides, axis=1)
    return bool((distance > 0.).all())


@lru_cache(maxsize=64)
def _section_surface(pieces):
    from table_1000.experts.grasps.checks import CollisionSurface
    return CollisionSurface([(np.eye(4), 'convex', np.asarray(vertices)) for vertices in pieces])


_INTERVALS = OrderedDict()
_CLOSURES = OrderedDict()
_SURFACES = WeakKeyDictionary()


def _array_key(value):
    return value.dtype.str, value.shape, value.strides, value.tobytes()


class ExposedSection:
    """Exposed collision material with vertical closing and horizontal approaches.

    Retains its actual convex cross-section for the common grasp candidate and
    pad-area checks; a rotated sheet's clipped triangular corner stays triangular.
    """
    def __init__(self, vertices, pieces=None):
        self.vertices = np.asarray(vertices).tolist()
        self.size = np.ptp(vertices, axis=0).tolist()
        self.pieces = [np.asarray(piece).tolist() for piece in (pieces if pieces is not None else [vertices])]

    def points(self):
        return np.asarray(self.vertices)

    @property
    def _surface(self):
        # Collision material is immutable for this observed section. Keeping
        # its surface also avoids rebuilding an evicted hull during a preview.
        # Keep it outside vars(self), which is also the public shape diagnostic.
        if self not in _SURFACES:
            _SURFACES[self] = _section_surface(tuple(tuple(map(tuple, piece)) for piece in self.pieces))
        return _SURFACES[self]

    def collision_objects(self, enclosing):
        # This section's centred clearance box encloses all its clipped pieces.
        return ([], []) if enclosing is self else self._surface.collision_objects(enclosing)

    def normals(self, points):
        return self._surface.normals(points)

    def closing_span(self, axis, low, high):
        # Retain exact-section closure caching when even the tolerance-expanded
        # collider bounds fit inside the finger sweep; otherwise keep its filter.
        bounds = self._surface._ray_bounds
        lower = np.where(axis >= 0., bounds[:, 0], bounds[:, 1]) @ axis
        upper = np.where(axis >= 0., bounds[:, 1], bounds[:, 0]) @ axis
        if np.all(lower >= low) and np.all(upper <= high):
            return self
        return self._surface.closing_span(axis, low, high)

    def interval(self, origins, direction):
        direction = np.asarray(direction)
        key = self._surface, _array_key(origins), _array_key(direction)
        if key in _INTERVALS:
            low, high = _INTERVALS.pop(key)
            _INTERVALS[key] = low, high
            return low.copy(), high.copy()
        low, high = self._surface.interval(origins, direction)
        _INTERVALS[key] = low.copy(), high.copy()
        if len(_INTERVALS) > 64:
            _INTERVALS.popitem(last=False)
        return low, high

    def _close_cached(self, transform, tcp, hand, open_gap, iterations, check_area, *, contact_layer=False):
        from table_1000.experts.grasps import _close_uncached
        # A whole mode can visit more than 1024 closures before another mode
        # revisits the same section. Keep their working sets separate: unrelated
        # depths must not evict a section's reusable fingertip/pad closures.
        surface = self._surface
        cache = _CLOSURES.pop(surface) if surface in _CLOSURES else OrderedDict()
        _CLOSURES[surface] = cache
        if len(_CLOSURES) > 64:
            _CLOSURES.popitem(last=False)
        # Hold margin and drive force do not enter the closure calculation.
        # Tip and interior-pad modes can therefore reuse this result while
        # keeping their later area checks and candidate scores independent.
        key = (hand._contact_geometry_key, hand.max_gap, hand.min_pad_area, hand.pad_spacing,
               _array_key(transform), _array_key(tcp), _array_key(np.asarray(open_gap)),
               iterations, check_area, contact_layer)
        if key in cache:
            result = cache.pop(key)
            cache[key] = result
            return None if result is None else (result[0], result[1].copy())
        result = _close_uncached(self, transform, tcp, hand, open_gap, iterations, check_area, contact_layer=contact_layer)
        cache[key] = None if result is None else (result[0], result[1].copy())
        if len(cache) > 1024:
            cache.popitem(last=False)
        return result

    def fcl_shape(self):
        # Conservative clearance envelope. Clipping bent patches can leave
        # nearly coincident vertices; rebuilding that hull in FCL needlessly
        # loses watertightness. Contact rays still use the true clipped union.
        return 'box', self.size

    def closings(self, step):
        closing, width, basis, slides = list(Box(tuple(self.size)).closings(step))[1]
        # A short union's box centre can be empty. Preserve its original
        # samples and also seed each actual convex fragment's interior.
        slides = list(slides)
        seen = {tuple(np.round(slide, 12)) for slide in slides}
        for piece in self.pieces:
            slide = np.asarray(piece).mean(0)
            key = tuple(np.round(slide, 12))
            if key not in seen:
                slides.append(slide)
                seen.add(key)
        yield closing, width, basis, slides


def exposed_section(points, edge, convex_pieces=None):
    """``(shape, world frame)`` of an exposed collision patch; X points onto the table."""
    clipped = exposed_points(points, edge)
    if len(clipped) < 4 or edge.distances(clipped).max() <= 0:
        return None
    frame = np.eye(4)
    frame[:3, :3] = np.column_stack([-edge.outward, [0., 0., 1.], edge.tangent])
    local = clipped @ frame[:3, :3]
    center = (local.min(0) + local.max(0)) / 2
    frame[:3, 3] = frame[:3, :3] @ center
    pieces = None
    if convex_pieces is not None:
        pieces = [exposed_points(piece, edge) for piece in convex_pieces if edge.distances(piece).max() > 0]
        pieces = [(piece - frame[:3, 3]) @ frame[:3, :3] for piece in pieces if len(piece) >= 4]
    return ExposedSection(local - center, pieces), frame
