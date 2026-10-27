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
from threadpoolctl import ThreadpoolController

from table_1000.common.timing import timed

# Contact rays have three coordinates. Kit can reset native BLAS pools to 32
# threads despite startup environment settings; these small products lose to
# dispatch and spinning workers. Scope this to contact geometry, then restore
# the caller's pool sizes. PhysX and OpenMP thread counts are untouched.
_CONTACT_MATH = ThreadpoolController()


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

    def __init__(self, model, arm, *, pad_spacing=None, min_pad_area=None,
                 hold_margin=None, min_hold_area=None, opening_clearance=.01, cost_width=0., drive=None):
        import xml.etree.ElementTree as ET
        self.model, self.arm, self.side = model, arm, model.spec.arms[arm]
        self.max_gap = self.side.max_gap
        self.pad_spacing = PAD_SPACING if pad_spacing is None else pad_spacing
        self.min_pad_area = MIN_PAD_AREA if min_pad_area is None else min_pad_area
        self.hold_margin = HOLD_MARGIN if hold_margin is None else hold_margin
        self.min_hold_area = MIN_HOLD_AREA if min_hold_area is None else min_hold_area
        self.opening_clearance = opening_clearance
        self.cost_width = cost_width
        self.drive = drive
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
            points = [_facing(shape, inward, self.pad_spacing) for shape in shapes]
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

    def pad_alignment(self, gap):
        """Flat-pad contact band for a pivoted finger facing a flat object side.

        Tilting a pad by theta leaves at most 2*CONTACT_TOLERANCE/sin(theta)
        along its depth in the contact layer. A band narrower than the two
        existing HOLD_MARGIN edge clearances cannot provide an interior grip.
        """
        pivot = self.side.finger_pivot
        theta = 0. if pivot is None else abs(next(iter(self.side.finger_targets(gap).values())))
        depth = min(float(np.ptp(points[pads, 0])) for _, points, pads in self.inner(gap))
        band = depth if abs(np.sin(theta)) < 1e-12 else min(depth, 2 * CONTACT_TOLERANCE / abs(np.sin(theta)))
        return {'pad_angle_rad': 2 * theta, 'contact_band_m': band,
                'edge_only': band < 2 * self.hold_margin,
                'pad_alignment_cost': 1 - band / depth if depth else 1.}

    def moment_radius(self):
        """Equivalent circular radius of the existing desired pad holding area."""
        return float(np.sqrt(HOLD_AREA / np.pi))

    def normal_force(self, gap):
        """Steady closing force from the existing drives, summed over both fingers."""
        if self.drive is None:
            raise ValueError('grasp gravity scoring requires measured drive stiffness and effort limits')
        return pinch_normal_force(self.side, gap, *self.drive)

    def on_pad(self, link, points):
        """GPU reports expose body names and points, not collider names: test URDF pad solids."""
        from scipy.spatial import ConvexHull
        shapes = self.shapes[link]
        explicit = any(shape[3] for shape in shapes)
        for transform, kind, data, pad in shapes:
            if explicit and not pad:
                continue
            local = (np.asarray(points) - transform[:3, 3]) @ transform[:3, :3]
            if kind == 'box':
                inside = (np.abs(local) <= np.asarray(data) / 2 + CONTACT_TOLERANCE).all(axis=1)
            else:
                equations = ConvexHull(data).equations
                inside = (local @ equations[:, :3].T + equations[:, 3] <= CONTACT_TOLERANCE).all(axis=1)
            if inside.any():
                return True
        return False

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
        if not hasattr(self, '_inner_cache'):
            self._inner_cache = {}
        key = id(frames)
        if key not in self._inner_cache:
            self._inner_cache[key] = [(sign, points @ frames[link][:3, :3].T + frames[link][:3, 3], pads)
                for link, (sign, points, pads) in zip(self.side.finger_links, self.fingers)]
        return self._inner_cache[key]

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
    target: object = None  # assigned by the skill when merging alternative parts
    choice: int | None = None  # geometry enumeration, before angle/cost ordering


def merge_part_heads(streams, key):
    """Select among each part's current head without reordering within a part.

    Unlike a globally sorted merge, a part's next physical load may be lower
    than its current head: expose it only after that head has been considered.
    """
    streams = [iter(stream) for stream in streams]
    heap, indices, closest = [], [0] * len(streams), np.inf
    def advance(number):
        nonlocal closest
        grasp = next(streams[number], None)
        if grasp is not None:
            closest = min(closest, grasp.info['closest_deg'])
            heapq.heappush(heap, (key(number, grasp), number, grasp))
    for number in range(len(streams)):
        advance(number)
    while heap:
        _, number, grasp = heapq.heappop(heap)
        grasp.info['closest_deg'] = closest
        yield number, indices[number], grasp
        indices[number] += 1
        advance(number)


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


def gravity_lever(contact, center_of_mass, closing):
    """Gravity torque about the unit closing line, per unit weight (metres)."""
    return float(abs(np.dot(np.cross(np.asarray(center_of_mass) - contact, [0., 0., -1.]), closing)))


def pinch_normal_force(side, gap, stiffness, effort_limits):
    """Virtual work: N=d(angular drive work)/d(finger opening), with unchanged drive gains."""
    angles = np.array(list(side.finger_targets(gap).values()))
    closed = np.array(list(side.finger_targets(0.).values()))
    effort = np.minimum(np.asarray(stiffness) * np.abs(angles - closed), effort_limits)
    lever = 1.
    if side.finger_pivot is not None:
        _, (x, y) = side.finger_pivot
        angle = float(angles[0] / side.finger_signs[0])
        lever = abs(x * np.cos(angle) - y * np.sin(angle))
    return float(effort.sum() / lever)


@timed('grasp_candidates')
def grasp_candidates(shape, transform, hand, obstacles, preferred, center_of_mass=None, approach_distance=.05,
                     step=np.radians(15), max_up=.05, cost_offset=40., cost_backoff=10., cost_hold=.5 / 30e-6,
                     mass_center=None, weight=0., contact_surface=None, max_angle=None, _placement_cache=None,
                     _first_angle=False, reference_rotation=None):
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
    HOLD_AREA, plus ``hand.cost_width`` per metre of closing width (zero by default),
    the original gravity moment about the gripping line normalized by predicted
    closing force times the desired-area radius, and loss of parallel pad contact band.
    Actual contact-patch load compares part heads in the skill, without changing
    this within-part order. The
    optional ``mass_center`` and ``weight`` (N) describe a freely suspended body;
    their absence retains geometric centering for joint-constrained manipulation.
    ``contact_surface``, when given, supplies the actual held-body collision
    surfaces in the shape frame for closure and interior holding area; the
    fitted shape still supplies candidate axes and conservative clearance.
    ``max_angle`` (rad) excludes approaches outside the consumer's existing
    limit before contact computation. The remaining cost order and closest
    valid angle are the same as filtering the full stream afterwards.
    With ``reference_rotation`` (the measured TCP rotation), the angular cost
    also accounts for wrist roll: project its closing axis onto the preferred
    approach plane and measure the full rotation to that frame. Approach
    limits and physical contact tests retain their meanings.
    """
    from mplib.collision_detection import fcl
    from table_1000.planning.world import geometry, pose
    transform = np.asarray(transform, dtype=float)
    rotation, origin = transform[:3, :3], transform[:3, 3]
    com = origin if center_of_mass is None else np.asarray(center_of_mass, dtype=float)
    mass_center = com if mass_center is None else np.asarray(mass_center, dtype=float)
    extra, poses = contact_surface.collision_objects(shape) if contact_surface is not None else ([], [])
    target = fcl.FCLObject('target', pose(transform), [fcl.CollisionObject(geometry(*shape.fcl_shape())), *extra],
                          [pose(np.eye(4)), *poses])
    # Every hand checked has its TCP at a contact centre inside the shape, backed off by at most
    # MAX_BACKOFF and the approach: obstacles farther from the shape than that and the hand's
    # radius cannot touch it, and are left out of the checks.
    reach = hand.radius() + MAX_BACKOFF + approach_distance
    obstacles = [(obstacle, clearance) for obstacle, clearance in obstacles
                 if fcl.distance(target, obstacle).min_distance < reach + max(clearance, 0.)]
    preferred = normalized(preferred)
    preferred_frame = None
    if reference_rotation is not None:
        reference_rotation = np.asarray(reference_rotation)
        closing = reference_rotation[:, 1] - (reference_rotation[:, 1] @ preferred) * preferred
        if not np.any(closing):
            closing = np.cross(reference_rotation[:, 2], preferred)
        preferred_frame = build_grasp_pose(preferred, closing, origin)[:3, :3]
    # Every (closing, approach, slide, hand turn), by angle from the preferred approach: the angle is
    # a lower bound of the cost, so a placed grasp is yielded once no later one can be cheaper.
    choices = []
    closings = list(shape.closings(step))
    length = lengthwise(closings)
    for closing, width, basis, slides in closings:
        open_gap = min(width + OPEN_MARGIN, hand.max_gap)
        if width > open_gap - hand.opening_clearance or length is not None and width >= length:
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
    if max_angle is not None:
        choices = [choice for choice in choices
                   if float(np.degrees(choice[0])) <= np.degrees(max_angle) + 1e-6]

    def place(choice):
        _, order, approach, closing, slide, _, open_gap = choice
        if _placement_cache is not None and order in _placement_cache:
            return _placement_cache[order]
        center = origin + rotation @ slide
        placed = _placed(shape, transform, target, build_grasp_pose(approach, closing, center), hand, open_gap,
                         obstacles, approach_distance, contact_surface)
        if _placement_cache is not None:
            _placement_cache[order] = placed
        return placed

    placed_grasps, closest = [], None
    bounds = {}
    if preferred_frame is not None and not _first_angle:
        # The closest *approach* defines the torso phase, independently of
        # wrist-roll preference. Probe it in approach order and reuse its
        # placements when traversing by the tighter full-rotation bound.
        _placement_cache = {} if _placement_cache is None else _placement_cache
        for choice in choices:
            placed = place(choice)
            if placed is not None and not hand.pad_alignment(placed[2])['edge_only'] \
                    and not (weight and hand.normal_force(placed[2]) <= 0.):
                closest = choice[0]
                break
        if closest is None:
            return
        for angle, order, approach, closing, *_ in choices:
            frame = build_grasp_pose(approach, closing, origin)[:3, :3]
            bounds[order] = max(angle, float(np.arccos(np.clip(
                (np.trace(preferred_frame.T @ frame) - 1.) / 2., -1., 1.))))
        choices.sort(key=lambda choice: (bounds[choice[1]], choice[1]))

    for choice in choices:
        angle, order, approach, closing, slide, width, open_gap = choice
        bound = bounds.get(order, angle)
        while placed_grasps and placed_grasps[0][0] <= bound:
            yield heapq.heappop(placed_grasps)[2]
        # The angle probe and the replayed full stream share fixed geometry.
        # Their raw choice ids are identical, so reuse both accepted and rejected
        # placements without changing either stream's cost order or retry ids.
        placed = place(choice)
        if placed is None:
            continue
        tcp, backoff, gap, contact, held = placed
        alignment = hand.pad_alignment(gap)
        if alignment['edge_only']:
            continue
        # Use the *existing* actuator's predicted steady normal force. Dividing
        # by weight instead would assume clamping force scales with object mass
        # and over-rank marginal small patches on very light objects.
        lever = gravity_lever(contact, mass_center, closing)
        force = hand.normal_force(gap) if weight else None
        if weight and force <= 0.:
            continue
        moment_cost = weight * lever / (force * hand.moment_radius()) if weight else 0.
        angular_cost = angle if preferred_frame is None else max(angle, float(np.arccos(np.clip(
            (np.trace(preferred_frame.T @ tcp[:3, :3]) - 1.) / 2., -1., 1.))))
        cost = angular_cost + cost_offset * float(np.linalg.norm(contact - com)) + moment_cost \
            + alignment['pad_alignment_cost'] + cost_backoff * backoff + hand.cost_width * width \
            + cost_hold * max(0., HOLD_AREA - held)
        # Approach-ordered placement, or the preceding probe for wrist costs,
        # determines the closest valid approach independently of cost order.
        closest = angle if closest is None else closest
        grasp = Grasp(tcp, open_gap, float(width), backoff, cost, {
            'approach_deg': float(np.degrees(angle)), 'slide': float(np.linalg.norm(slide)), 'contact_gap': gap,
            'held_mm2': held * 1e6, 'closest_deg': float(np.degrees(closest)),
            'com_distance_m': float(np.linalg.norm(contact - mass_center)), 'gravity_lever_m': lever,
            'weight_N': weight, 'gravity_moment_Nm': weight * lever, 'predicted_normal_force_N': force,
            'moment_radius_m': hand.moment_radius(),
            'gravity_moment_cost': moment_cost, **alignment}, choice=order)
        if preferred_frame is not None:
            grasp.info['orientation_deg'] = float(np.degrees(angular_cost))
        if _first_angle:
            # Only the closest valid angle is needed to define the held-lift
            # phase. Its cheapest grasp is found by the normal cost stream.
            yield grasp
            return
        heapq.heappush(placed_grasps, (cost, order, grasp))
    while placed_grasps:
        yield heapq.heappop(placed_grasps)[2]


@_CONTACT_MATH.wrap(limits=1, user_api='blas')
def _placed(shape, transform, target, pose, hand, open_gap, obstacles, approach_distance, contact_surface=None):
    """``(tcp, backoff, contact gap, contact centre, held area)`` of the first back-off from ``pose``
    (the TCP at the contact centre) at which the hand is clear (see :func:`grasp_candidates`), or
    None when the pads squeeze too little of the object, or hold too little of its widest section,
    there."""
    # Beyond this back-off the pads, open, no longer reach along the approach as far as the object.
    reach = max(points[pads, 0].max() for _, points, pads in hand.inner(open_gap))
    nearest = float(((shape.points() @ transform[:3, :3].T + transform[:3, 3] - pose[:3, 3]) @ pose[:3, 0]).min())
    compound = contact_surface is not None and len(contact_surface.pieces) > 1
    for backoff in np.arange(0., min(MAX_BACKOFF, reach - nearest) + 1e-9, BACKOFF_STEP):
        tcp = standoff(pose, backoff)
        # The open hand there and at the pre-grasp, one after the other (see Hand.fcl).
        if any(_collides(body, target, 0.) or any(_collides(body, obstacle, clearance) for obstacle, clearance in obstacles)
               for body in (hand.fcl(at, open_gap) for at in (tcp, standoff(tcp, approach_distance)))):
            continue
        surface = shape if contact_surface is None else _closing_surface(contact_surface, transform, tcp, hand, open_gap)
        closed = _close(surface, transform, tcp, hand, open_gap,
                        contact_layer=contact_surface is not None)
        if closed is None:
            if compound:
                continue  # a different component can supply opposing faces later
            return None     # a single convex section only gets smaller
        gap, contact = closed
        closed_hand = hand.fcl(tcp, gap)
        if _collides(hand.fcl(tcp, gap, fingers=False), target, 0.) or any(
                _collides(closed_hand, obstacle, clearance) for obstacle, clearance in obstacles):
            continue
        held = _held_area(shape, transform, tcp, hand, gap) if contact_surface is None \
            else _contact_area(surface, transform, tcp, hand, gap,
                               section=isinstance(shape, Cylinder))
        if held < hand.min_hold_area:
            if compound:
                continue  # contact can move from an edge onto another component
            return None     # a single convex section only gets smaller
        return tcp, float(backoff), gap, contact, held
    return None


def _closing_surface(surface, transform, tcp, hand, open_gap):
    """Keep actual surfaces that can enter the finite finger closing sweep.

    A free object can centre after a finger touches it, but a remote component
    outside the whole opening cannot translate itself into an empty hand.
    Bound the inward surface samples at every existing 0.5 mm FK cache gap,
    including intermediate pivot positions, then project their enclosing box
    on the closing axis. The existing contact tolerance expands that span.
    """
    key = int(round(float(open_gap) / .0005))
    if not hasattr(hand, '_closing_bounds'):
        hand._closing_bounds = {}
    if key not in hand._closing_bounds:
        points = np.vstack([points for index in range(key + 1)
                            for _, points, _ in hand.inner(index * .0005)])
        low, high = points.min(0), points.max(0)
        hand._closing_bounds[key] = np.array([[x, y, z] for x in (low[0], high[0])
                                            for y in (low[1], high[1]) for z in (low[2], high[2])])
    world = hand._closing_bounds[key] @ tcp[:3, :3].T + tcp[:3, 3]
    points = (world - transform[:3, 3]) @ transform[:3, :3]
    axis = transform[:3, :3].T @ tcp[:3, 1]
    coordinate = points @ axis
    return surface.closing_span(axis, float(coordinate.min() - CONTACT_TOLERANCE),
                                float(coordinate.max() + CONTACT_TOLERANCE))


@_CONTACT_MATH.wrap(limits=1, user_api='blas')
def contact_resistance(shape, transform, tcp, hand, gap, *, open_gap=None):
    """Finite contact-patch torsion radius and normal-to-closing force redirection.

    Uniform pressure gives a torsional moment scale N*mean(radius). Each pad
    sample represents a square cell, integrated with four-point quadrature.
    A normal tilted by theta needs tan(theta) tangential force per unit normal
    force to redirect its resultant along the closing line. This consumes
    friction rather than providing torsional reserve. No material is changed
    and this geometric scale is not a calibrated friction-capacity guarantee.
    """
    if open_gap is not None:
        shape = _closing_surface(shape, transform, tcp, hand, open_gap)
    rotation, origin = transform[:3, :3], transform[:3, 3]
    direction = rotation.T @ tcp[:3, 1]
    fingers = []
    for sign, points, pads in hand.inner(gap):
        world = points @ tcp[:3, :3].T + tcp[:3, 3]
        low, high = shape.interval((world - origin) @ rotation, direction)
        past = np.where(high >= low, high if sign > 0 else -low, -np.inf)
        interior = pads & (points[:, 0] < points[pads, 0].max() - hand.hold_margin)
        fingers.append((sign, world, past, interior))
    reaches = {sign: float(past.max()) for sign, _, past, _ in fingers}
    shift = (reaches[-1.] - reaches[1.]) / 2
    radii, cosines = [], []
    cell = np.array([[x, z] for x in (-1., 1.) for z in (-1., 1.)]) * hand.pad_spacing / np.sqrt(12.)
    for sign, world, past, interior in fingers:
        touching = interior & (past + sign * shift >= -CONTACT_TOLERANCE)
        points = world[touching] + (sign * past[touching])[:, None] * tcp[:3, 1]
        if not len(points):
            return {'contact_radius_m': 0., 'contact_normal_cosine': 0., 'normal_redirection_cost': np.inf}
        normals = shape.normals((points - origin) @ rotation)
        cosines.append(float(np.mean(np.abs(normals @ direction))))
        local = (points - tcp[:3, 3]) @ tcp[:3, :3]
        patch = local[:, [0, 2]]
        centered = patch - patch.mean(axis=0)
        radii.append(float(np.linalg.norm(centered[:, None, :] + cell, axis=2).mean()))
    cosine = float(np.clip(min(cosines), 0., 1.))
    return {'contact_radius_m': min(radii), 'contact_normal_cosine': cosine,
            'normal_redirection_cost': np.sqrt(1. - cosine ** 2) / cosine if cosine else np.inf}


def _contact_area(shape, transform, tcp, hand, gap, *, section=True):
    """Smaller opposing contact patch inside the existing pad-tip margin.

    A curved/hollow part's globally widest section need not be the section
    held by the fingers. Retain the insertion-depth check on the convex
    pieces actually touched for a round candidate: a tangent patch near the
    pad tip alone does not establish that its central section is behind the
    tip margin. A box candidate retains the actual interior contact layer:
    a supported flat face need not span its whole collision component.
    """
    rotation, origin = transform[:3, :3], transform[:3, 3]
    direction = rotation.T @ tcp[:3, 1]
    fingers = []
    for sign, points, pads in hand.inner(gap):
        world = points @ tcp[:3, :3].T + tcp[:3, 3]
        low, high = shape.interval((world - origin) @ rotation, direction)
        past = np.where(high >= low, high if sign > 0 else -low, -np.inf)
        interior = pads & (points[:, 0] < points[pads, 0].max() - hand.hold_margin)
        fingers.append((sign, world, past, pads, interior))
    reaches = {sign: float(past.max()) for sign, _, past, _, _ in fingers}
    shift = (reaches[-1.] - reaches[1.]) / 2
    area = min(np.count_nonzero(interior & (past + sign * shift >= -CONTACT_TOLERANCE))
               for sign, _, past, _, interior in fingers) * hand.pad_spacing ** 2
    if not section:
        return area
    if not area:
        return 0.
    contacts = np.vstack([world[pads & (past + sign * shift >= -CONTACT_TOLERANCE)]
                          for sign, world, past, pads, _ in fingers])
    # _close lets the object centre itself along the closing direction.
    local = (contacts - shift * tcp[:3, 1] - origin) @ rotation
    patch = shape.at_contacts(local)
    if not patch.pieces:
        return 0.
    shifted = np.array(transform, copy=True)
    shifted[:3, 3] += shift * tcp[:3, 1]
    return min(area, _held_area(patch, shifted, tcp, hand, gap, local=True))


def _held_area(shape, transform, tcp, hand, gap, *, local=False):
    """Area (m^2, projected on the pads' plane) of the object's widest section across the closing
    that lies on the pads at least HOLD_MARGIN from their tip end, the hand at world ``tcp`` with
    the fingers at ``gap``."""
    rotation, origin = transform[:3, :3], transform[:3, 3]
    direction = rotation.T @ tcp[:3, 1]
    pads = np.vstack([points[mask] for _, points, mask in hand.inner(gap)])
    xs = np.arange(pads[:, 0].min(), pads[:, 0].max() - hand.hold_margin, HOLD_SPACING)
    zs = np.arange(pads[:, 2].min(), pads[:, 2].max(), HOLD_SPACING)
    if not len(xs) or not len(zs):
        return 0.
    x, z = (grid.reshape(-1, 1) for grid in np.meshgrid(xs, zs, indexing='ij'))
    origins = (tcp[:3, 3] + x * tcp[:3, 0] + z * tcp[:3, 2] - origin) @ rotation
    if local:
        supported = shape.wide_sections(origins, direction, HOLD_TOLERANCE)
    else:
        widest = float(np.ptp(shape.points() @ direction))
        low, high = shape.interval(origins, direction)
        supported = high - low >= widest - HOLD_TOLERANCE
    return float(np.count_nonzero(supported)) * HOLD_SPACING ** 2


def _close(shape, transform, tcp, hand, open_gap, iterations=9, check_area=True, *, contact_layer=False):
    """Close the hand at world ``tcp`` on the object (free to slide along the closing to stay between the
    fingers): ``(gap, contact centre)`` where the fingers' inner surfaces first both touch it, or None
    when they touch it already open, never touch it, or touch it with less than MIN_PAD_AREA of pad
    each. Candidate prediction may use ``contact_layer`` to establish pad
    support within the existing CONTACT_TOLERANCE on each side after a small
    edge first touches. The measured expert signal retains the default.
    """
    rotation, origin = transform[:3, :3], transform[:3, 3]
    direction = rotation.T @ tcp[:3, 1]
    reached = {}

    def reach(gap):
        """Per finger: inner surface points (world), pad mask and how far the object reaches past each
        point along the finger's opening."""
        inner = hand.inner(gap)
        key = id(inner)
        if key in reached:
            return reached[key][1]
        fingers = []
        for sign, points, pads in inner:
            world = points @ tcp[:3, :3].T + tcp[:3, 3]
            low, high = shape.interval((world - origin) @ rotation, direction)
            past = np.where(high >= low, high if sign > 0 else -low, -np.inf)
            fingers.append((sign, world, pads, past))
        # Retain the source list too: mock or alternate hands may return a new
        # list on every call, whose id must not be recycled within this search.
        reached[key] = inner, fingers
        return fingers

    def overlap(fingers):
        return sum(float(past.max()) for _, _, _, past in fingers)

    # Compound objects often leave a back-off entirely outside every piece.
    # Test the fully closed hand first: an empty closing needs no open-hand
    # rays. Both endpoint predicates and the subsequent search stay identical.
    if overlap(reach(0.)) < 0. or overlap(reach(open_gap)) >= 0.:
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
    enough = lambda points: min(len(side) for side in points) >= hand.min_pad_area / hand.pad_spacing ** 2
    if check_area and not enough(touching):
        if not contact_layer:
            return None
        # First contact can be a small edge ahead of a broad supporting face.
        # Equal centring gives each side overlap/2 penetration. Search only
        # that existing contact layer, on Hand.frames' existing 0.5 mm grid.
        for candidate_gap in np.r_[np.arange(low - .0005, 0., -.0005), 0.]:
            candidate_fingers = reach(float(candidate_gap))
            if overlap(candidate_fingers) > 2 * CONTACT_TOLERANCE:
                break
            reaches = {sign: float(past.max()) for sign, _, _, past in candidate_fingers}
            shift = (reaches[-1.] - reaches[1.]) / 2
            candidate_touching = [world[pads & (past + sign * shift >= -CONTACT_TOLERANCE)]
                                  for sign, world, pads, past in candidate_fingers]
            if enough(candidate_touching):
                low, touching = float(candidate_gap), candidate_touching
                break
        if not enough(touching):
            return None
    joined = np.vstack(touching)
    return float(low), joined.mean(axis=0) if len(joined) else tcp[:3, 3].copy()


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
