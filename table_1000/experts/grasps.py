"""Grasp frames, after ManiSkill3's motion-planning utilities (``build_grasp_pose``,
``compute_grasp_info_by_obb``).

TCP frames follow the robots' convention (see ``asset_sources/robots/*/robot.py``):
+X is the approach direction, +Y the direction in which the fingers open (the
closing axis), +Z completes the frame. Boxes are ``(T_world_box, size)``, for
example from :meth:`~table_1000.experts.context.Context.part_box`.
"""

from dataclasses import dataclass
import heapq

import numpy as np

from table_1000.common.timing import timed


def normalized(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def build_grasp_pose(approaching, closing, center):
    """World TCP pose approaching along ``approaching`` with the fingers closing along ``closing``
    (made orthogonal to it), the TCP at ``center``."""
    approaching = normalized(approaching)
    closing = normalized(closing - (closing @ approaching) * approaching)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack([approaching, closing, np.cross(approaching, closing)])
    transform[:3, 3] = center
    return transform


def compute_grasp_info_by_obb(box, approaching=(0., 0., -1.), target_closing=None, depth=0., ortho=True):
    """ManiSkill's grasp of an oriented box: approach along the box axis closest to ``approaching``,
    close across the shorter of the other two (of nearly equal ones, the one closest to
    ``target_closing``), the TCP ``depth`` into the box from the face it approaches (at most to its
    centre). Returns ``{'approaching', 'closing', 'center', 'extents'}``."""
    transform, extents = np.asarray(box[0], dtype=float), np.asarray(box[1], dtype=float)
    approaching = normalized(approaching)
    angles = approaching @ transform[:3, :3]
    order = np.argsort(np.abs(angles))
    ind0 = order[-1]
    shorter = np.argsort(extents[order[:-1]])
    ind1, ind2 = order[:-1][shorter[0]], order[:-1][shorter[1]]
    if target_closing is not None and extents[ind1] / extents[ind2] > .99:
        if abs(np.dot(target_closing, transform[:3, ind1])) < abs(np.dot(target_closing, transform[:3, ind2])):
            ind1, ind2 = ind2, ind1
    closing = transform[:3, ind1]
    if target_closing is not None and np.dot(target_closing, closing) < 0:
        closing = -closing
    extents = extents[[ind0, ind1, ind2]]
    half = extents[0] / 2
    center = transform[:3, 3] + approaching * (-half + min(depth, half))
    if ortho:
        closing = normalized(closing - (approaching @ closing) * approaching)
    return {'approaching': approaching, 'closing': closing, 'center': center, 'extents': extents}


def standoff(tcp, distance):
    """``tcp`` moved back along its approach axis by ``distance``."""
    result = np.array(tcp, dtype=float)
    result[:3, 3] -= result[:3, 0] * distance
    return result


def translated(transform, offset):
    result = np.array(transform, dtype=float)
    result[:3, 3] += offset
    return result


# -- grasps of convex primitives ---------------------------------------------------------
# A two-finger grasp of a convex object squeezes it across a closing direction
# along which it has two parallel (or round) faces; the hand may approach from
# any direction perpendicular to that one. :func:`grasp_candidates` enumerates
# such grasps of a box or a cylinder, places the TCP as deep as the pads, the
# palm and the table allow, and orders them by preference.

@dataclass(frozen=True)
class Box:
    """A box of ``size`` centred on its frame."""
    size: tuple

    def points(self):
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
        return signs * np.asarray(self.size) / 2

    def interval(self, origins, direction):
        """``(low, high)``: the lines ``origins + t direction`` (object frame: ``[N, 3]`` points, a unit
        direction) are inside the box for ``low <= t <= high`` (``low > high``: they miss it)."""
        half = np.asarray(self.size, dtype=float) / 2
        origins = np.atleast_2d(origins)
        low, high = np.full(len(origins), -np.inf), np.full(len(origins), np.inf)
        for axis in range(3):
            o, d, h = origins[:, axis], direction[axis], half[axis]
            if abs(d) < 1e-12:
                high = np.where(np.abs(o) > h, -np.inf, high)
                continue
            t0, t1 = (-h - o) / d, (h - o) / d
            low, high = np.maximum(low, np.minimum(t0, t1)), np.minimum(high, np.maximum(t0, t1))
        return low, high

    def fcl_shape(self):
        return 'box', tuple(self.size)

    def closings(self, step):
        """``(closing, width, approach basis, slides)`` in the object frame: one per face pair."""
        size, axes = np.asarray(self.size, dtype=float), np.eye(3)
        for k in range(3):
            i, j = [m for m in range(3) if m != k]
            yield axes[k], size[k], (axes[i], axes[j]), [axes[m] * t for m in (i, j) for t in _slides(size[m])]


@dataclass(frozen=True)
class Cylinder:
    """A cylinder of ``radius`` and ``length`` along its frame's Z, centred on it."""
    radius: float
    length: float

    def points(self, count=24):
        angles = np.linspace(0., 2 * np.pi, count, endpoint=False)
        ring = np.c_[np.cos(angles), np.sin(angles), np.zeros(count)] * self.radius
        return np.vstack([ring + [0., 0., -self.length / 2], ring + [0., 0., self.length / 2]])

    def interval(self, origins, direction):
        """``(low, high)``: the lines ``origins + t direction`` (object frame: ``[N, 3]`` points, a unit
        direction) are inside the cylinder for ``low <= t <= high`` (``low > high``: they miss it)."""
        origins = np.atleast_2d(origins)
        (ox, oy, oz), (dx, dy, dz) = origins.T, direction
        a = dx * dx + dy * dy
        c = ox * ox + oy * oy - self.radius ** 2
        if a < 1e-12:
            low = np.where(c > 0, np.inf, -np.inf)
            high = -low
        else:
            b = 2 * (ox * dx + oy * dy)
            root = np.sqrt(np.maximum(b * b - 4 * a * c, 0.))
            inside = b * b - 4 * a * c > 0
            low = np.where(inside, (-b - root) / (2 * a), np.inf)
            high = np.where(inside, (-b + root) / (2 * a), -np.inf)
        half = self.length / 2
        if abs(dz) < 1e-12:
            high = np.where(np.abs(oz) > half, -np.inf, high)
        else:
            t0, t1 = (-half - oz) / dz, (half - oz) / dz
            low, high = np.maximum(low, np.minimum(t0, t1)), np.minimum(high, np.maximum(t0, t1))
        return low, high

    def fcl_shape(self):
        return 'cylinder', (self.radius, self.length)

    def closings(self, step, tilts=np.radians([0., 15., 30.])):
        """Across the axis in every direction (``step`` apart), straight or, on a cylinder shorter than
        its diameter, tilted toward the axis by ``tilts`` (squeezing the rim obliquely; on a longer
        one flat pads tilted to the side would touch it at a point only), approaching from any
        direction square to the closing; and across the end faces."""
        axis = np.array([0., 0., 1.])
        if self.length > 2 * self.radius:
            tilts = tilts[:1]
        slides = [axis * t for t in _slides(self.length)]
        for phi in np.arange(0., np.pi, step):
            radial = np.array([np.cos(phi), np.sin(phi), 0.])
            tangent = np.cross(axis, radial)
            for tilt in tilts:
                for sign in ((1.,) if tilt == 0 else (1., -1.)):
                    closing = np.cos(tilt) * radial + sign * np.sin(tilt) * axis
                    width = min(2 * self.radius / np.cos(tilt), self.length / max(np.sin(tilt), 1e-9))
                    yield closing, width, (tangent, np.cross(closing, tangent)), slides
        for phi in np.arange(0., np.pi, step):
            yield axis, self.length, (np.array([np.cos(phi), np.sin(phi), 0.]),), [np.zeros(3)]


# A part is taken for a cylinder when its collision vertices, seen along its longest extent, lie all
# round it: no gap between their directions wider than ROUND_GAP (a polygonal barrel, not a square),
# and it is as wide as the cylinder across every direction, to within ROUND_ERROR (not a rounded
# square, a flat handle or a curved shell, which a cylinder grasp would take for wider than they are).
ROUND_GAP = np.radians(50.)
ROUND_ERROR = .002      # m
ROUND_DIRECTIONS = 36


def fitted(points):
    """``(shape, offset)`` for collision vertices ``points`` (``[N, 3]``, in a body frame): a
    :class:`Cylinder` along their longest extent where they lie all round it (see ``ROUND_GAP``),
    else their axis-aligned :class:`Box`; ``offset`` places the shape in the body frame."""
    low, high = points.min(0), points.max(0)
    offset = np.eye(4)
    offset[:3, 3] = (low + high) / 2
    size = high - low
    axis = int(np.argmax(size))
    across = [i for i in range(3) if i != axis]
    local = points[:, across] - offset[across, 3]
    radius = np.linalg.norm(local, axis=1)
    if size[axis] > max(size[across]) and np.all(radius > 1e-6):
        angles = np.sort(np.arctan2(local[:, 1], local[:, 0]))
        gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
        turns = np.linspace(0., np.pi, ROUND_DIRECTIONS, endpoint=False)
        widths = np.ptp(local @ np.c_[np.cos(turns), np.sin(turns)].T, axis=0)
        if gaps.max() <= ROUND_GAP and 2 * radius.max() - widths.min() <= ROUND_ERROR:
            # The cylinder's Z along the longest extent.
            offset[:3, :3] = _along(axis)
            return Cylinder(float(radius.max()), float(size[axis])), offset
    return Box(tuple(size)), offset


def _along(axis):
    """A rotation taking Z to coordinate axis ``axis`` (a right-handed frame)."""
    return np.eye(3)[:, [(axis + 1) % 3, (axis + 2) % 3, axis]]


def _slides(extent, edge=.025):
    """Contact centres along an object side of ``extent``: the middle, and on a long side
    (more than 6 cm) also halfway to positions ``edge`` from its ends."""
    if extent <= .06:
        return [0.]
    offset = (extent / 2 - edge) / 2
    return [0., -offset, offset]




class Hand:
    """The hand of one arm in its TCP frame (+X approach, +Y opening): collision shapes of the TCP
    link and the fingers, posed for a pad gap (the fingers parallel or pivoting, see
    :class:`~table_1000.robots.spec.ArmSpec`). Each finger's surfaces facing the other finger are
    sampled every ``PAD_SPACING``; those of shapes named ``*pad*`` in the URDF are its pads (all
    of them when it has none)."""

    def __init__(self, model, arm):
        import xml.etree.ElementTree as ET
        self.model, self.arm, self.side = model, arm, model.spec.arms[arm]
        self.max_gap = self.side.max_gap
        urdf = model.root / model.spec.urdf
        links = {link.get('name'): link for link in ET.parse(urdf).getroot().findall('link')}
        self.cache = {}
        # link -> [(T_link_shape, kind, data, pad)]
        self.shapes = {link: [_urdf_shape(collision, urdf.parent) for collision in links[link].findall('collision')]
                       for link in (self.side.tcp[0], *self.side.finger_links)}
        self.geometry = {}
        # Collision meshes are independent of finger opening.  Keep one FCL
        # geometry per URDF shape and only rebuild the cheap pose wrappers for
        # each opening.  This avoids repeating convex-hull construction while
        # preserving the original shape order and collision tests.
        self.collision_geometry = {}
        reference = self.frames(min(.04, self.max_gap))
        self.fingers = []   # (opening sign, inner surface points in the finger link's frame, pad mask)
        for link in self.side.finger_links:
            shapes = self.shapes[link]
            padded = any(shape[3] for shape in shapes)
            frame = reference[link]
            vertices = np.vstack([_vertices(shape) @ frame[:3, :3].T + frame[:3, 3] for shape in shapes])
            sign = float(np.sign(vertices[:, 1].mean()))
            inward = frame[:3, :3].T @ [0., -sign, 0.]
            points = [_facing(shape, inward) for shape in shapes]
            pads = [np.full(len(p), shape[3] or not padded) for p, shape in zip(points, shapes)]
            self.fingers.append((sign, np.vstack(points), np.concatenate(pads)))

    def frames(self, gap):
        """``{link: T_tcp_link}`` with the fingers at ``gap`` (on a 0.5 mm grid)."""
        key = int(round(float(np.clip(gap, 0., self.max_gap)) / 5e-4))
        if key not in self.cache:
            q = self.model.vector(self.side.finger_targets(key * 5e-4))
            tcp = np.linalg.inv(self.model.tcp(self.arm, q))
            self.cache[key] = {link: tcp @ self.model.fk(link, q) for link in self.shapes}
        return self.cache[key]

    def tip(self, gap=0.):
        """How far ahead of the TCP (along +X) the fingers reach at ``gap``."""
        frames = self.frames(gap)
        return max(float((_vertices(shape) @ frames[link][:3, :3].T + frames[link][:3, 3])[:, 0].max())
                   for link in self.side.finger_links for shape in self.shapes[link])

    def radius(self):
        """How far from the TCP any part of the hand reaches, at any gap."""
        if not hasattr(self, '_radius'):
            self._radius = max(float(np.linalg.norm(_vertices(shape) @ frames[link][:3, :3].T + frames[link][:3, 3], axis=1).max())
                               for frames in (self.frames(0.), self.frames(self.max_gap))
                               for link in self.shapes for shape in self.shapes[link])
        return self._radius

    def inner(self, gap):
        """``[(opening sign, [N, 3] inner surface points in the TCP frame, pad mask)]`` per finger at ``gap``."""
        frames = self.frames(gap)
        return [(sign, points @ frames[link][:3, :3].T + frames[link][:3, 3], pads)
                for link, (sign, points, pads) in zip(self.side.finger_links, self.fingers)]

    def fcl(self, tcp, gap, fingers=True):
        """FCL object of the hand at world TCP pose ``tcp`` with the fingers at ``gap`` (only the
        palm, the TCP link, for ``fingers=False``). Its collision objects are made once per gap and
        reused: it is valid until the next call with the same gap (check it before making another)."""
        from mplib.collision_detection import fcl
        from table_1000.planning.world import geometry, pose
        key = (int(round(float(np.clip(gap, 0., self.max_gap)) / 5e-4)), fingers)
        if key not in self.geometry:
            parts = []
            for link, frame in self.frames(gap).items():
                if link in self.side.finger_links and not fingers:
                    continue
                for index, (transform, kind, data, _) in enumerate(self.shapes[link]):
                    key_shape = (link, index)
                    if key_shape not in self.collision_geometry:
                        self.collision_geometry[key_shape] = geometry(kind, data)
                    parts.append((fcl.CollisionObject(self.collision_geometry[key_shape]), pose(frame @ transform)))
            self.geometry[key] = [part for part, _ in parts], [part_pose for _, part_pose in parts]
        objects, poses = self.geometry[key]
        return fcl.FCLObject('hand', pose(tcp), objects, poses)


def _urdf_shape(collision, directory):
    """``(T_link_shape, kind, data, pad)`` of a URDF collision (a box or a convex mesh)."""
    from scipy.spatial.transform import Rotation
    import trimesh
    transform = np.eye(4)
    origin = collision.find('origin')
    if origin is not None:
        transform[:3, :3] = Rotation.from_euler('xyz', [float(v) for v in origin.get('rpy', '0 0 0').split()]).as_matrix()
        transform[:3, 3] = [float(v) for v in origin.get('xyz', '0 0 0').split()]
    geometry = collision.find('geometry')
    pad = 'pad' in (collision.get('name') or '')
    if geometry.find('box') is not None:
        return transform, 'box', tuple(float(v) for v in geometry.find('box').get('size').split()), pad
    mesh = trimesh.load(directory / geometry.find('mesh').get('filename'), force='mesh')
    return transform, 'convex', np.asarray(mesh.convex_hull.vertices), pad


def _mesh(shape):
    import trimesh
    transform, kind, data, _ = shape
    if kind == 'box':
        return trimesh.creation.box(extents=data, transform=transform)
    return trimesh.Trimesh(data @ transform[:3, :3].T + transform[:3, 3]).convex_hull


def _vertices(shape):
    return np.asarray(_mesh(shape).vertices)


def _facing(shape, direction, spacing=None, cosine=.7):
    """Points every ``spacing`` (PAD_SPACING) on the faces of ``shape`` facing ``direction`` (link frame)."""
    import trimesh
    mesh = _mesh(shape)
    faces = np.flatnonzero(mesh.face_normals @ direction > cosine)
    if not len(faces):
        return np.zeros((0, 3))
    part = mesh.submesh([faces], append=True)
    count = max(1, int(part.area / (spacing or PAD_SPACING) ** 2))
    return np.asarray(trimesh.sample.sample_surface_even(part, count, seed=0)[0])


@dataclass
class Grasp:
    """One grasp: world TCP ``pose``, pad gap to open to, object ``width`` across the pads, the
    TCP's back-off from the contact centre and the preference ``cost``."""
    pose: np.ndarray
    open_gap: float
    width: float
    backoff: float
    cost: float
    info: dict


# Opening beyond the object's width, total. Pad surfaces are sampled every PAD_SPACING; a pad
# point squeezes the object when it is within CONTACT_TOLERANCE of its surface as the fingers
# close, and a grasp needs MIN_PAD_AREA of such pad surface on each finger.
OPEN_MARGIN, PAD_SPACING, CONTACT_TOLERANCE, MIN_PAD_AREA = .03, .0015, .001, 15e-6
ELONGATED = 3., .08     # longer than this times its next width and this (m): not grasped by its ends
# The TCP backs off from the contact centre in BACKOFF_STEP steps until the hand is clear.
BACKOFF_STEP, MAX_BACKOFF = .002, .06
# The object's widest section across the closing (where the pads squeeze it: a face of a box, a line
# along a cylinder) must lie on the pads at least HOLD_MARGIN from their tip end over MIN_HOLD_AREA
# (m^2; less: held by the pad's edge, pushed out of the hand by a few newtons); grasps holding it
# over less than HOLD_AREA cost more (see grasp_candidates). Sampled every HOLD_SPACING, within
# HOLD_TOLERANCE of the widest.
HOLD_MARGIN, MIN_HOLD_AREA, HOLD_AREA, HOLD_SPACING, HOLD_TOLERANCE = .002, 5e-6, 60e-6, 5e-4, 5e-4
# Clearance (m) the hand keeps from a support (the surface under the object); negative: the overlap
# allowed, so the fingers may touch the table and grasp small objects low.
SUPPORT_CLEARANCE = -.0005


def lengthwise(closings):
    """The width of ``closings`` (as :meth:`Box.closings` gives them) along the length of a long,
    thin shape: more than ``ELONGATED[1]`` (m) and ``ELONGATED[0]`` times its next width (a pen, not a
    handle pinched by its ends); None for a shape that is not."""
    widths = sorted({round(float(width), 6) for _, width, _, _ in closings})
    long = len(widths) > 1 and widths[-1] > ELONGATED[0] * widths[-2] and widths[-1] > ELONGATED[1]
    return widths[-1] if long else None


@timed('grasp_candidates')
def grasp_candidates(shape, transform, hand, obstacles, preferred, center_of_mass=None, approach_distance=.05,
                     step=np.radians(15), max_up=.05, cost_offset=40., cost_backoff=10., cost_hold=.5 / 30e-6):
    """Grasps of ``shape`` (:class:`Box` or :class:`Cylinder`) at world ``transform`` by ``hand``
    (:class:`Hand`), cheapest first, generated as they are taken.

    Approach directions are ``step`` apart around each closing direction, none rising more than
    ``max_up`` (z of the unit approach), none closing along the length of a long, thin shape (see
    :func:`lengthwise`: a pen squeezed by its ends); both hand turns about the approach are listed. The TCP
    starts at the contact centre and backs off along the approach until the open hand is clear
    of the object and of ``obstacles`` (``[(FCL object, clearance to keep)]``), also
    ``approach_distance`` further back (the pre-grasp), and the hand closed on the object
    touches it only with the fingers; a grasp whose pads then squeeze too little of the object, or
    hold its widest section by their edge only, is dropped (see :func:`_placed`). The cost is the
    angle (rad) from the ``preferred`` approach direction, plus ``cost_offset`` per metre between
    the centre of the squeezing pad area and ``center_of_mass`` (default: the shape's centre),
    ``cost_backoff`` per metre of back-off and ``cost_hold`` per m^2 the held section falls short of
    HOLD_AREA.
    """
    from mplib.collision_detection import fcl
    from table_1000.planning.world import geometry, pose
    transform = np.asarray(transform, dtype=float)
    rotation, origin = transform[:3, :3], transform[:3, 3]
    com = origin if center_of_mass is None else np.asarray(center_of_mass, dtype=float)
    target = fcl.FCLObject('target', pose(transform), [fcl.CollisionObject(geometry(*shape.fcl_shape()))], [pose(np.eye(4))])
    # Every hand checked has its TCP at a contact centre inside the shape, backed off by at most
    # MAX_BACKOFF and the approach: obstacles farther from the shape than that and the hand's
    # radius cannot touch it, and are left out of the checks.
    reach = hand.radius() + MAX_BACKOFF + approach_distance
    obstacles = [(obstacle, clearance) for obstacle, clearance in obstacles
                 if fcl.distance(target, obstacle).min_distance < reach + max(clearance, 0.)]
    preferred = normalized(preferred)
    # Every (closing, approach, slide, hand turn), by angle from the preferred approach: the angle is
    # a lower bound of the cost, so a placed grasp is yielded once no later one can be cheaper.
    choices = []
    closings = list(shape.closings(step))
    length = lengthwise(closings)
    for closing, width, basis, slides in closings:
        open_gap = min(width + OPEN_MARGIN, hand.max_gap)
        if width > open_gap - .01 or length is not None and width >= length:
            continue
        closing = rotation @ closing
        approaches = [rotation @ (np.cos(t) * basis[0] + np.sin(t) * basis[1]) for t in np.arange(0., 2 * np.pi, step)] \
            if len(basis) == 2 else [rotation @ (s * basis[0]) for s in (1., -1.)]
        for approach in approaches:
            if approach[2] > max_up:
                continue
            angle = float(np.arccos(np.clip(approach @ preferred, -1., 1.)))
            for slide in slides:
                for flip in (1., -1.):
                    choices.append((angle, len(choices), approach, flip * closing, slide, width, open_gap))
    choices.sort(key=lambda choice: choice[:2])
    placed_grasps, closest = [], None
    for angle, order, approach, closing, slide, width, open_gap in choices:
        while placed_grasps and placed_grasps[0][0] <= angle:
            yield heapq.heappop(placed_grasps)[2]
        center = origin + rotation @ slide
        placed = _placed(shape, transform, target, build_grasp_pose(approach, closing, center), hand, open_gap,
                         obstacles, approach_distance)
        if placed is None:
            continue
        tcp, backoff, gap, contact, held = placed
        cost = angle + cost_offset * float(np.linalg.norm(contact - com)) + cost_backoff * backoff \
            + cost_hold * max(0., HOLD_AREA - held)
        # Choices are placed by angle: the first that places comes closest of all to the preferred approach.
        closest = angle if closest is None else closest
        heapq.heappush(placed_grasps, (cost, order, Grasp(tcp, open_gap, float(width), backoff, cost, {
            'approach_deg': float(np.degrees(angle)), 'slide': float(np.linalg.norm(slide)), 'contact_gap': gap,
            'held_mm2': held * 1e6, 'closest_deg': float(np.degrees(closest))})))
    while placed_grasps:
        yield heapq.heappop(placed_grasps)[2]


def _placed(shape, transform, target, pose, hand, open_gap, obstacles, approach_distance):
    """``(tcp, backoff, contact gap, contact centre, held area)`` of the first back-off from ``pose``
    (the TCP at the contact centre) at which the hand is clear (see :func:`grasp_candidates`), or
    None when the pads squeeze too little of the object, or hold too little of its widest section,
    there."""
    # Beyond this back-off the pads, open, no longer reach along the approach as far as the object.
    reach = max(points[pads, 0].max() for _, points, pads in hand.inner(open_gap))
    nearest = float(((shape.points() @ transform[:3, :3].T + transform[:3, 3] - pose[:3, 3]) @ pose[:3, 0]).min())
    for backoff in np.arange(0., min(MAX_BACKOFF, reach - nearest) + 1e-9, BACKOFF_STEP):
        tcp = standoff(pose, backoff)
        # The open hand there and at the pre-grasp, one after the other (see Hand.fcl).
        if any(_collides(body, target, 0.) or any(_collides(body, obstacle, clearance) for obstacle, clearance in obstacles)
               for body in (hand.fcl(at, open_gap) for at in (tcp, standoff(tcp, approach_distance)))):
            continue
        closed = _close(shape, transform, tcp, hand, open_gap)
        if closed is None:
            return None     # backing off further only squeezes less of the object
        gap, contact = closed
        closed_hand = hand.fcl(tcp, gap)
        if _collides(hand.fcl(tcp, gap, fingers=False), target, 0.) or any(
                _collides(closed_hand, obstacle, clearance) for obstacle, clearance in obstacles):
            continue
        held = _held_area(shape, transform, tcp, hand, gap)
        if held < MIN_HOLD_AREA:
            return None     # backing off further only holds less
        return tcp, float(backoff), gap, contact, held
    return None


def _held_area(shape, transform, tcp, hand, gap):
    """Area (m^2, projected on the pads' plane) of the object's widest section across the closing
    that lies on the pads at least HOLD_MARGIN from their tip end, the hand at world ``tcp`` with
    the fingers at ``gap``."""
    rotation, origin = transform[:3, :3], transform[:3, 3]
    direction = rotation.T @ tcp[:3, 1]
    widest = float(np.ptp(shape.points() @ direction))
    pads = np.vstack([points[mask] for _, points, mask in hand.inner(gap)])
    xs = np.arange(pads[:, 0].min(), pads[:, 0].max() - HOLD_MARGIN, HOLD_SPACING)
    zs = np.arange(pads[:, 2].min(), pads[:, 2].max(), HOLD_SPACING)
    if not len(xs) or not len(zs):
        return 0.
    x, z = (grid.reshape(-1, 1) for grid in np.meshgrid(xs, zs, indexing='ij'))
    low, high = shape.interval((tcp[:3, 3] + x * tcp[:3, 0] + z * tcp[:3, 2] - origin) @ rotation, direction)
    return float(np.count_nonzero(high - low >= widest - HOLD_TOLERANCE)) * HOLD_SPACING ** 2


def _close(shape, transform, tcp, hand, open_gap, iterations=9):
    """Close the hand at world ``tcp`` on the object (free to slide along the closing to stay between the
    fingers): ``(gap, contact centre)`` where the fingers' inner surfaces first both touch it, or None
    when they touch it already open, never touch it, or touch it with less than MIN_PAD_AREA of pad
    each."""
    rotation, origin = transform[:3, :3], transform[:3, 3]
    direction = rotation.T @ tcp[:3, 1]

    def reach(gap):
        """Per finger: inner surface points (world), pad mask and how far the object reaches past each
        point along the finger's opening."""
        fingers = []
        for sign, points, pads in hand.inner(gap):
            world = points @ tcp[:3, :3].T + tcp[:3, 3]
            low, high = shape.interval((world - origin) @ rotation, direction)
            past = np.where(high >= low, high if sign > 0 else -low, -np.inf)
            fingers.append((sign, world, pads, past))
        return fingers

    def overlap(fingers):
        return sum(float(past.max()) for _, _, _, past in fingers)

    if overlap(reach(open_gap)) >= 0. or overlap(reach(0.)) < 0.:
        return None
    low, high = 0., open_gap
    for _ in range(iterations):
        middle = (low + high) / 2
        low, high = (middle, high) if overlap(reach(middle)) >= 0. else (low, middle)
    fingers = reach(low)
    # The object slides along the opening (+Y) by ``shift`` until both fingers press on it.
    reaches = {sign: float(past.max()) for sign, _, _, past in fingers}
    shift = (reaches[-1.] - reaches[1.]) / 2
    touching = [world[pads & (past + sign * shift >= -CONTACT_TOLERANCE)] for sign, world, pads, past in fingers]
    if min(len(points) for points in touching) < MIN_PAD_AREA / PAD_SPACING ** 2:
        return None
    return float(low), np.vstack(touching).mean(axis=0)


def _collides(first, second, clearance):
    """Whether two FCL objects are closer than ``clearance`` (m); a negative ``clearance`` is the
    overlap allowed (0: whether they overlap)."""
    from mplib.collision_detection import fcl
    if clearance > 0.:
        return fcl.distance(first, second).min_distance < clearance
    result = fcl.collide(first, second, fcl.CollisionRequest(num_max_contacts=4, enable_contact=clearance < 0.))
    if not result.is_collision():
        return False
    return clearance == 0. or max(contact.penetration_depth for contact in result.get_contacts()) > -clearance
