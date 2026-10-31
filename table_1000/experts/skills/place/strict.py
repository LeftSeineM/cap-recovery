"""Place: the held object put down in a region of a surface."""

from table_1000.experts.skills.place import regions

import itertools
from dataclasses import replace
from functools import cached_property

from table_1000.common.timing import timed

import numpy as np
from mplib.collision_detection import fcl
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from table_1000.experts.context import ExpertFailed, Unreachable, remaining_wall
from table_1000.experts.skills.hold.frames import MIN_HOLD_AREA, OPEN_MARGIN, _close, _held_area, _vertices, translated
from table_1000.experts.skills.chain import InHand
from table_1000.experts.skills.place.flat import FlatReleaseGeometry, FLAT_LEVEL, FLAT_STEP
from table_1000.experts.skills.place.low import LowRelease
from table_1000.planning.world import geometry, pose as fcl_pose
from table_1000.experts.skills.common import CANDIDATE_PLANS, IK_ATTEMPTS, RETREATS, ROOM_PLACES, collisions, convex_keepout
from table_1000.experts.solver import PlanningFailed
from table_1000.experts.audit.problems import decision, mark

PLACE_ABOVE = .08       # m, place comes down from this high above the release pose
PLACE_CLEARANCE = .01   # m between the held object and the surface at the release
MAX_PLACES = 8          # placements whose motion place plans
MAX_PLACE_IK = 200      # placements place checks by IK
PLACE_STEP = .02        # m between the positions place samples in its region
PLACE_TURN_STEP = 15.   # deg between the turns place samples
PLACE_TURNS = 8         # turns place tries per position, the least first
INSERT_MARGIN = .002   # m clearance in narrow insertion slots
PLACE_MARGIN = .01      # m the footprint keeps inside the region and off its keep-out areas
# What a place needs of the hold (see Placing.need): NEED_REACHED of NEED_PLACES of its placements,
# spread over the MAX_PLACE_IK nearest (those place checks), in reach by IK (NEED_IK_ATTEMPTS restarts)
# at the release and above it (more than one: the object may turn a little in the hand as it closes),
# per hold (arm, and the object in the hand rounded to NEED_ROUND m / rad).
NEED_PLACES, NEED_REACHED, NEED_IK_ATTEMPTS, NEED_ROUND = 24, 3, 5, (.02, .1)


class Placing(LowRelease):
    """The place skill of :class:`~table_1000.experts.skills.Skills`."""

    @cached_property
    def _insertion_channels(self):
        return {}

    def insertion_blockers(self, arm, release, above, gap, obstacles):
        """Scene parts intersecting the fingers/palm's full vertical insertion channel."""
        remaining_wall(getattr(self, 'ctx', None))
        hand = self.hands[arm]
        delta = release[:3, :3].T @ [0., 0., above]
        key = (arm, round(gap, 6), tuple(delta.round(9)))
        if key not in self._insertion_channels:
            frames, channels = hand.frames(gap), []
            for link, shapes in hand.shapes.items():
                for shape in shapes:
                    remaining_wall(getattr(self, 'ctx', None))
                    vertices = _vertices(shape)
                    frame = frames[link]
                    points = vertices @ frame[:3, :3].T + frame[:3, 3]
                    swept = np.vstack([points, points + delta])
                    channels.append((convex_keepout(swept), swept))
            self._insertion_channels[key] = channels
        blocked = set()
        bounds = np.asarray([bounds for _, _, bounds in obstacles]).reshape(-1, 2, 3)
        for channel, swept in self._insertion_channels[key]:
            remaining_wall(getattr(self, 'ctx', None))
            points = swept @ release[:3, :3].T + release[:3, 3]
            low, high = points.min(0), points.max(0)
            nearby = np.flatnonzero(np.all(high >= bounds[:, 0], axis=1)
                                    & np.all(low <= bounds[:, 1], axis=1))
            if len(nearby):
                posed = fcl.FCLObject('hand channel', fcl_pose(release @ channel.pose.to_transformation_matrix()),
                                     channel.shapes, channel.shape_poses)
                for index in nearby:
                    remaining_wall(getattr(self, 'ctx', None))
                    name, obstacle, _ = obstacles[index]
                    if fcl.distance(posed, obstacle).min_distance < 0.:
                        blocked.add(name)
        return sorted(blocked)

    def insertion_obstacles(self, target):
        """Named live scene geometry, excluding the carried body."""
        # semantic imports placement constants, so this dependency stays local.
        from table_1000.experts.world.geometry import scene_shapes
        obstacles = []
        for s in scene_shapes(self.ctx, exclude=(target.obj,)):
            remaining_wall(self.ctx)
            obstacles.append((f'{s.obj}:{s.part}', convex_keepout(s.points),
                              (s.points.min(0), s.points.max(0))))
        remaining_wall(self.ctx)
        return obstacles

    def insertion_occupancy(self, arm, tcp_object, gap, options, above, obstacles):
        """Rank full 3D hand channels against existing contents before contact and IK.

        Bounds supply a nonnegative preference; exact swept-hull checks still
        reject collisions later. No object is moved for ranking.
        """
        remaining_wall(getattr(self, 'ctx', None))
        hand = self.hands[arm]
        frames = hand.frames(gap)
        if not hasattr(hand, '_insertion_points'):
            hand._insertion_points = {}
        key = id(frames)
        if key not in hand._insertion_points:
            points = np.vstack([_vertices(shape) @ frames[link][:3, :3].T + frames[link][:3, 3]
                                for link, shapes in hand.shapes.items() for shape in shapes])
            # Hand.frames already quantizes and retains each FK frame. Retain
            # its identity too, so alternate hands cannot recycle a frame id.
            hand._insertion_points[key] = frames, points
        points = hand._insertion_points[key][1]
        bounds = np.asarray([bounds for _, _, bounds in obstacles]).reshape(-1, 2, 3)
        scores = []
        for placed, _ in options[:NEED_REACHED]:
            release = placed @ np.linalg.inv(tcp_object)
            world = points @ release[:3, :3].T + release[:3, 3]
            low, high = world.min(0), world.max(0) + [0., 0., above]
            extent = np.maximum(high - low, INSERT_MARGIN)
            overlap = np.maximum(0., np.minimum(high, bounds[:, 1]) - np.maximum(low, bounds[:, 0])) / extent
            amounts = np.min(overlap, axis=1)[np.all(overlap > 0., axis=1)]
            # Keep Python's original left-to-right scalar sum, including its
            # floating-point accumulation order; only bounds arithmetic batches.
            scores.append(sum(float(amount) for amount in amounts))
        return min(scores, default=0.)

    def check_insertion_depth(self, target, container, part, clearance):
        """Do not release a slipped object whose measured end is still above the live floor."""
        # semantic imports placement constants, so this dependency stays local.
        from table_1000.experts.world.geometry import scene_shapes, support_region
        ctx = self.ctx
        floor, _ = support_region(scene_shapes(ctx, exclude=(target.obj,)), container, part, floor=True)
        if floor is None:
            ctx.fail('container floor unavailable before release', obj=target.obj, container=container, recoverable=True)
        pose = ctx.poses()[target.body]
        points = target.points @ pose[:3, :3].T + pose[:3, 3]
        gap = float(((points - floor[0][:3, 3]) @ floor[0][:3, 2]).min())
        ctx.note('insertion depth measured', obj=target.obj, container=container, floor_gap_m=gap)
        if gap > clearance + INSERT_MARGIN:
            ctx.fail('object above container floor before release; pick it again before placing',
                     obj=target.obj, container=container, floor_gap_m=gap, recoverable=True)

    def flat_release_geometry(self, target, region):
        """Live obstacles and measured wall height, without asset-specific constants."""
        # semantic imports placement constants, so this dependency stays local.
        from table_1000.experts.world.geometry import scene_shapes, selects
        every = scene_shapes(self.ctx)
        shapes = [shape for shape in every if shape.obj != target.obj]
        local = np.linalg.inv(self.ctx.poses()[target.body])
        released = [(f'{shape.obj}:{shape.part}', shape.points @ local[:3, :3].T + local[:3, 3])
                    for shape in every if shape.obj == target.obj]
        metadata = region[3] if len(region) > 3 else {}
        container, part = metadata.get('flat_container'), metadata.get('part')
        wall = [shape.points for shape in shapes if shape.obj == container
                and (metadata.get('top_tolerance') is None or selects(shape, container, part))]
        normal = np.asarray(region[0])[:3, 2]
        height = float(np.ptp(np.vstack(wall) @ normal)) if wall else 0.
        obstacles = [(f'{shape.obj}:{shape.part}', convex_keepout(shape.points),
                      (shape.points.min(0), shape.points.max(0))) for shape in shapes]
        ceiling = metadata.get('maximum_release_top')
        if ceiling is not None:
            ceiling -= float(np.min(np.ptp(target.points, axis=0)))
        return FlatReleaseGeometry(obstacles, height, released, backoff=RETREATS[0], ceiling=ceiling)

    def flat_release_height(self, geometry, *args, **kwargs):
        height = geometry.minimum(*args, **kwargs,
                                  check_budget=lambda: remaining_wall(getattr(self, 'ctx', None)))
        return None if height is None else min(height + FLAT_STEP, geometry.maximum)

    def centre_flat_options(self, target, region, options):
        """Prefer the measured COM nearest the live support's centre."""
        frame, low, high = region[3]['surface'] if len(region) > 3 and 'surface' in region[3] else region[:3]
        centre = (np.asarray(low)[:2] + np.asarray(high)[:2]) / 2
        inv = np.linalg.inv(frame)
        local_com = (np.linalg.inv(self.ctx.poses()[target.body])
                     @ np.r_[self.ctx.env.center_of_mass(target.body)[0], 1.])[:3]
        return sorted(options, key=lambda option: np.linalg.norm(
            (inv @ option[0] @ np.r_[local_com, 1.])[:2] - centre))

    def flat_open_gap(self, target, arm, tcp_object):
        closing = target.offset[:3, :3].T @ np.linalg.inv(tcp_object)[:3, 1]
        width = float(np.ptp(target.shape.points() @ closing))
        return min(width + OPEN_MARGIN, self.hands[arm].max_gap)

    def check_flat_release(self, target, region, arm, above):
        """Reobserve minimum finger clearance, level and COM before opening."""
        ctx = self.ctx
        if len(region) > 3 and 'flat_container' in region[3]:
            # semantic imports placement constants, so this dependency stays local.
            from table_1000.experts.skills.place.regions import inside_region
            support = region[3]
            region, _ = inside_region(target.obj, support['flat_container'], support['part'],
                                     support=support['support'], orientation='flat',
                                     top_tolerance=support.get('top_tolerance'))(ctx)
        frame = np.asarray(region[0])
        current = ctx.poses()[target.body]
        points = target.points @ current[:3, :3].T + current[:3, 3]
        gap = float(((points - frame[:3, 3]) @ frame[:3, 2]).min())
        axis = int(np.argmin(np.ptp(target.points, axis=0)))
        tilt = float(np.arccos(np.clip(abs(current[:3, axis] @ frame[:3, 2]), 0., 1.)))
        surface, low, high = region[3]['surface'] if len(region) > 3 and 'surface' in region[3] else region[:3]
        com = (np.linalg.inv(surface) @ np.r_[ctx.env.center_of_mass(target.body)[0], 1.])[:2]
        supported = bool(np.all(com >= np.asarray(low)[:2] + INSERT_MARGIN)
                         and np.all(com <= np.asarray(high)[:2] - INSERT_MARGIN))
        measured = np.linalg.inv(self.solvers[arm].tcp()) @ current
        resting = current.copy(); resting[:3, 3] -= frame[:3, 2] * gap
        geometry = self.flat_release_geometry(target, region)
        arguments = (self.hands[arm], resting, measured, self.solvers[arm].finger_gap(),
                     self.held[arm]['grasp']['open_gap'], above)
        check_budget = lambda: remaining_wall(ctx)
        minimum = geometry.minimum(*arguments, check_budget=check_budget)
        clear = geometry.minimum(*arguments, heights=(gap,), check_budget=check_budget) is not None
        ctx.note('flat release measured', obj=target.obj, support_gap_m=gap, minimum_m=minimum,
                 maximum_m=geometry.maximum, tilt_deg=float(np.degrees(tilt)), com_inside=supported,
                 clearance_verified=clear)
        if (minimum is None or not clear or gap > geometry.maximum + 1e-6
                or abs(gap - minimum) > 2 * FLAT_STEP or tilt > FLAT_LEVEL or not supported):
            ctx.fail('flat release violates measured clearance, level or support; pick again with a compatible hold',
                     Unreachable, obj=target.obj, support_gap_m=gap, minimum_m=minimum,
                     maximum_m=geometry.maximum, tilt_deg=float(np.degrees(tilt)), com_inside=supported,
                     clearance_verified=clear,
                     blocked_by=sorted(geometry.blocked_by) if minimum is None else [], recoverable=True)

    @decision
    @timed('placements')
    def placements(self, target, current, region, near, keep_out, step, margin, orientation=None, sweep=0., turns=PLACE_TURNS):
        """``[(object pose at rest on the surface, spot)]`` in ``region`` (see :meth:`place`), the
        footprint at least ``margin`` inside it and out of ``keep_out``, nearest ``near`` first, then
        the least turned from ``current``. Positive ``sweep`` also screens the
        object's vertical descent corridor against convex keep-outs."""
        remaining_wall(getattr(self, 'ctx', None))
        frame, low, high = (np.asarray(item, dtype=float) for item in region[:3])
        support = region[3] if len(region) > 3 and 'surface' in region[3] else None
        maximum_height = region[3].get('maximum_height') if len(region) > 3 else None
        local = np.linalg.inv(frame)
        boxes, solids = [], []
        for blocked in keep_out:
            remaining_wall(getattr(self, 'ctx', None))
            if isinstance(blocked, np.ndarray):
                solids.append(blocked)
            else:
                f, l, h = blocked
                boxes.append((np.linalg.inv(np.asarray(f, dtype=float)), np.asarray(l, dtype=float)[:2] - margin,
                              np.asarray(h, dtype=float)[:2] + margin))
        if solids:
            identity = fcl_pose(np.eye(4))
            obstacles = []
            for points in solids:
                remaining_wall(getattr(self, 'ctx', None))
                obstacles.append(convex_keepout(points))
            # Bound the actual FCL hulls: simplification can grow them beyond
            # the source vertices. Include stabilization and transform rounding.
            obstacle_points = [np.asarray(obj.shapes[0].get_collision_geometry().get_vertices())
                               + obj.pose.to_transformation_matrix()[:3, 3] for obj in obstacles]
            bounds = np.asarray([(p.min(0) - 1e-7, p.max(0) + 1e-7) for p in obstacle_points])
            carried = geometry('convex', target.points)
            carried_points = np.asarray(carried.get_vertices())
            def blocked_by(obj, points):
                low, high = points.min(0) - 1e-7, points.max(0) + 1e-7
                nearby = np.flatnonzero(np.all(high + max(margin, 0.) >= bounds[:, 0], axis=1)
                                        & np.all(low - max(margin, 0.) <= bounds[:, 1], axis=1))
                for index in nearby:
                    remaining_wall(getattr(self, 'ctx', None))
                    if fcl.distance(obj, obstacles[index]).min_distance < margin:
                        return True
                return False
        options = []
        for seed, turn in itertools.product(regions.orientations(target.points, current, frame, orientation),
                                            np.radians(np.arange(0., 360., PLACE_TURN_STEP))):
            remaining_wall(getattr(self, 'ctx', None))
            # Turned about the surface normal, in the region frame.
            rotation = frame[:3, :3] @ Rotation.from_euler('z', turn).as_matrix() @ frame[:3, :3].T @ seed
            points = (target.points @ rotation.T) @ local[:3, :3].T     # in the region frame, about the origin
            # Bound the resting object by the measured rim, retaining open descent above it.
            if maximum_height is not None and np.ptp(points[:, 2]) > maximum_height:
                continue
            centre = (points[:, :2].min(0) + points[:, :2].max(0)) / 2
            spread = np.ptp(points[:, :2], axis=0) / 2
            hull = points[ConvexHull(points[:, :2]).vertices]    # the footprint's outline
            if solids and sweep:
                outline = hull[:, :2] - centre
                swept = np.vstack([np.c_[outline, np.full(len(outline), z)]
                                    for z in (0., float(np.ptp(points[:, 2]) + sweep))])
                corridor_centre = (swept.min(0) + swept.max(0)) / 2
                corridor = convex_keepout(swept)
                corridor_points = np.asarray(corridor.shapes[0].get_collision_geometry().get_vertices()) + corridor_centre
            first, last = low[:2] + spread + margin, high[:2] - spread - margin
            if np.any(first > last + 1e-9):
                continue
            for x in np.arange(first[0], last[0] + 1e-9, step):
                for y in np.arange(first[1], last[1] + 1e-9, step):
                    remaining_wall(getattr(self, 'ctx', None))
                    spot = frame @ [x, y, 0., 1.]
                    footprint = hull[:, :2] - centre + [x, y]
                    world = (frame @ np.c_[footprint, np.zeros(len(footprint)), np.ones(len(footprint))].T).T
                    if any(regions.overlaps((inv @ world.T).T[:, :2], low_, high_) for inv, low_, high_ in boxes):
                        continue
                    pose = np.eye(4)
                    pose[:3, :3] = rotation
                    pose[:3, 3] = frame[:3, :3] @ np.r_[[x, y] - centre, -points[:, 2].min()] + frame[:3, 3]
                    if support is not None:
                        com = pose[:3, :3] @ support['centre_of_mass'] + pose[:3, 3]
                        support_frame, support_low, support_high = support['surface']
                        com = (np.linalg.inv(support_frame) @ np.r_[com, 1.])[:2]
                        if not np.all((com >= np.asarray(support_low)[:2] + margin)
                                      & (com <= np.asarray(support_high)[:2] - margin)):
                            continue
                    if solids:
                        obj = fcl.FCLObject('placed', fcl_pose(pose), [fcl.CollisionObject(carried)], [identity])
                        if blocked_by(obj, carried_points @ rotation.T + pose[:3, 3]):
                            continue
                        if sweep:
                            # A footprint prism conservatively covers the entire
                            # carried body while translating along the normal.
                            # Keep its vertices local to avoid almost-coplanar
                            # duplicate vertices from compound collision parts.
                            corridor_pose = np.array(frame)
                            corridor_pose[:3, 3] = spot[:3]
                            # The stabilized hull carries its own local centre.
                            corridor_pose[:3, 3] += frame[:3, :3] @ corridor_centre
                            obj = fcl.FCLObject('descent', fcl_pose(corridor_pose),
                                                corridor.shapes, corridor.shape_poses)
                            if blocked_by(obj, corridor_points @ frame[:3, :3].T + spot[:3]):
                                continue
                    turned = Rotation.from_matrix(rotation @ current[:3, :3].T).magnitude()
                    options.append((float(np.linalg.norm((spot[:3] - near)[:2])), turned, pose, spot[:3]))
        # Nearest first, at most PLACE_TURNS of the least turned per position: IK checks spread over
        # positions rather than spending themselves on the turns of a few.
        remaining_wall(getattr(self, 'ctx', None))
        options.sort(key=lambda option: (round(option[0] / step), option[1]))
        kept, counts = [], {}
        for distance, turned, pose, spot in options:
            remaining_wall(getattr(self, 'ctx', None))
            key = tuple(np.round(spot[:2] / step).astype(int))
            if orientation:
                normal = frame[:3, 2] @ pose[:3, :3]
                axis = int(np.argmax(np.abs(normal)))
                key += (axis, int(np.sign(normal[axis])))
            counts[key] = counts.get(key, 0) + 1
            if counts[key] <= turns:
                kept.append((pose, spot))
        if orientation:
            # A bounded IK prefix must see either end/face, even when one is
            # initially nearer the held rotation. Keep distance bands ordered
            # and yaw choices ordered within each face, alternating the faces.
            bands = {}
            for pose, spot in kept:
                remaining_wall(getattr(self, 'ctx', None))
                normal = frame[:3, 2] @ pose[:3, :3]
                axis = int(np.argmax(np.abs(normal)))
                face = (axis, int(np.sign(normal[axis])))
                band = round(float(np.linalg.norm((spot - near)[:2])) / step)
                bands.setdefault(band, {}).setdefault(face, []).append((pose, spot))
            kept = [option for faces in bands.values()
                    for row in itertools.zip_longest(*faces.values())
                    for option in row if option is not None]
        remaining_wall(getattr(self, 'ctx', None))
        return kept

    @decision
    def put_down(self, arm, options, tcp_object, clearance, above, orientation=None, flat_geometry=None):
        """Move the held object down at the first of ``options`` (``(object pose, spot)``) in reach
        (IK above and at the release) whose motion plans; ``{'done', 'spot', 'ik_checked',
        'planned', 'last', ...collisions}``."""
        result = {'done': False, 'spot': None, 'ik_checked': 0, 'planned': 0, 'last': 'no placement in reach'}
        obstacles, channels = [], {}
        if orientation in ('upright', 'insert'):
            obstacles = self.insertion_obstacles(self.held[arm]['target'])
            gap = self.held[arm]['grasp']['open_gap']
            result['blocked_by'] = []
        flat_options = None
        if orientation == 'flat':
            solver, held = self.solvers[arm], self.held[arm]
            flat_options = []
            for index, (placed, spot) in enumerate(options):
                remaining_wall(getattr(self, 'ctx', None))
                height = self.flat_release_height(flat_geometry, self.hands[arm], placed, tcp_object,
                                                  solver.finger_gap(), held['grasp']['open_gap'], above)
                if height is not None:
                    flat_options.append((height, index, placed, spot))
            flat_options.sort(key=lambda candidate: candidate[0])
            result['blocked_by'] = sorted(flat_geometry.blocked_by) if not flat_options else []
            result['minimum_release_m'] = min((candidate[0] for candidate in flat_options), default=None)
            result['maximum_release_m'] = max((candidate[0] for candidate in flat_options), default=None)
        # The torso lift where it is first, then moving too.
        planned = set()
        for lift in (False, True):
            checked = 0
            candidates = ([(index, placed, spot, height) for height, index, placed, spot in flat_options]
                          if flat_options is not None else
                          [(index, placed, spot, clearance) for index, (placed, spot) in enumerate(options)])
            for index, placed, spot, height in candidates:
                remaining_wall(getattr(self, 'ctx', None))
                if checked >= MAX_PLACE_IK or result['planned'] >= MAX_PLACES:
                    break
                if index in planned:
                    continue
                mark(self, 'candidate', f'{index}/{lift}', placed=placed, spot=spot, lift=lift)
                if obstacles:
                    if index not in channels:
                        release = translated(placed, [0., 0., clearance]) @ np.linalg.inv(tcp_object)
                        channels[index] = self.insertion_blockers(arm, release, above, gap, obstacles)
                    if channels[index]:
                        result['blocked_by'] = sorted(set(result['blocked_by']) | set(channels[index]))
                        continue
                count = result['planned']
                arguments = (orientation,) if orientation else ()
                checked += 1
                if self.lower(arm, placed, spot, tcp_object, height, above, lift, result, *arguments):
                    result['release_clearance_m'] = height
                    return result
                if result['planned'] > count:
                    planned.add(index)
        return result

    @decision
    def lower(self, arm, placed, spot, tcp_object, clearance, above, lift, result, orientation=None):
        """Down at one of :meth:`put_down`'s options if it is in reach (``lift``: the torso lift
        moving too) and its motion plans; whether it is, counted in ``result``.
        Oriented motion checks measured pad contact when projected coverage is low;
        a lost object fails with an instruction to pick it again."""
        ctx, solver = self.ctx, self.solvers[arm]
        result['ik_checked'] += 1
        def measured_hold():
            target = self.held[arm]['target']
            poses, tcp = ctx.poses(), solver.tcp()
            area = _held_area(target.shape, poses[target.body] @ target.offset,
                              tcp, self.hands[arm], solver.finger_gap())
            if area < MIN_HOLD_AREA:
                # A tilted end hold can have no widest projected section on
                # the pads while both pads still touch the actual surface.
                contact = self.pad_closure(target, arm, self.held[arm]['grasp']['open_gap'])
                if not contact['held']:
                    ctx.fail('object not held during oriented placement; pick it again before placing',
                             obj=target.obj, arm=arm, held_area_m2=area, recoverable=True, closure=contact)
                ctx.note('oriented hold confirmed by pad contacts', obj=target.obj, arm=arm, held_area_m2=area)
            for name in target.names:
                solver.attach(name, poses[solver.world.objects[name].body])
            return np.linalg.inv(tcp) @ poses[target.body]
        if orientation:
            tcp_object = measured_hold()
        release = translated(placed, [0., 0., clearance]) @ np.linalg.inv(tcp_object)
        # Where it is let go (the hand and arm low by a wall), then above it at that torso height:
        # the way down moves the arm alone.
        goals = self.down_goals(arm, release, above, lift, IK_ATTEMPTS)
        if not goals:
            return False
        mark(self, 'plan')
        result['planned'] += 1
        stage = 'above'
        try:
            solver.move_to_pose_with_RRTConnect(None, lift=True, goals=goals, attempts=CANDIDATE_PLANS)
            if orientation:
                # Transport can settle an end hold. Reobserve above the mouth
                # before committing to its collision-checked straight descent.
                release = translated(placed, [0., 0., clearance]) @ np.linalg.inv(measured_hold())
                stage = 'align above'
                solver.first_reachable([False, True], lambda moving: solver.move_to_pose_with_screw(
                    translated(release, [0., 0., above]), lift=moving))
                release = translated(placed, [0., 0., clearance]) @ np.linalg.inv(measured_hold())
            stage = 'descent'
            if orientation:
                solver.first_reachable([False, True], lambda moving: solver.move_to_pose_with_screw(release, lift=moving))
            else:
                solver.move_to_pose_with_screw(release)
            mark(self, 'plan', outcome='accepted')
            result.update(done=True, spot=spot)
            return True
        except PlanningFailed as error:
            mark(self, 'plan', outcome='rejected', reason=error.reason)
            result.update(last=error.reason, **collisions(error))
            ctx.note('next placement', arm=arm, spot=spot.tolist(), reason=error.reason, stage=stage, **collisions(error))
            return False

    def place_settings(self, target, region, near, clearance, above, step, margin, orientation):
        """The same geometric defaults for hold lookahead and actual placement."""
        size = np.sort(np.ptp(target.points, axis=0))
        step = (min(PLACE_STEP, size[0] / 2) if orientation in ('upright', 'insert') else PLACE_STEP) if step is None else step
        margin = (INSERT_MARGIN if orientation in ('upright', 'insert') else PLACE_MARGIN) if margin is None else margin
        if orientation in ('upright', 'insert'):
            above = max(above, float(size[-1]))
            clearance = min(clearance, INSERT_MARGIN)
        elif orientation == 'flat':
            clearance = min(clearance, INSERT_MARGIN)
        frame, low, high = (np.asarray(item, dtype=float) for item in region[:3])
        if near is None:
            near = frame @ np.r_[(low[:2] + high[:2]) / 2, 0., 1.]
        return near, clearance, above, step, margin, size

    @decision
    def need(self, obj, region, near=None, keep_out=(), clearance=PLACE_CLEARANCE, above=PLACE_ABOVE,
             step=None, margin=None, orientation=None, turns=PLACE_TURNS):
        """What :meth:`place` with these arguments needs of the hand that will hold ``obj`` (not held
        yet, at its pose now): an :class:`~table_1000.experts.skills.chain.InHand` accepting the holds
        from which ``NEED_REACHED`` of its placements are in reach (IK at the release and above it),
        and the hands at the first of them (the TCP at the release and above it, per hold)."""
        ctx = self.ctx
        target = self.target(obj)
        current = ctx.poses()[target.body]
        near, clearance, above, step, margin, size = self.place_settings(
            target, region, near, clearance, above, step, margin, orientation)
        frame, low, high = (np.asarray(item, dtype=float) for item in region[:3])
        options = self.placements(target, current, region, np.asarray(near, dtype=float)[:3], keep_out, step, margin, orientation,
                                  sweep=above + clearance + size[-1] / 4 if orientation in ('upright', 'insert') else 0., turns=turns)
        if orientation == 'flat':
            options = self.centre_flat_options(target, region, options)
        options = options[:MAX_PLACE_IK]       # those place checks
        options = [options[i] for i in np.unique(np.linspace(0, len(options) - 1, NEED_PLACES).astype(int))] if options else []
        return self.restore_need({'obj': obj, 'target': target, 'options': options, 'clearance': clearance,
                                  'above': above, 'arms': [arm for arm in self.arms if arm not in self.held],
                                  'verdicts': [], 'origin': {'obj': obj, 'region': region, 'near': near, 'keep_out': keep_out,
                                                            'clearance': clearance, 'above': above, 'step': step, 'margin': margin,
                                                            'orientation': orientation, 'turns': turns}})

    def restore_need(self, recipe):
        """Rebuild a Place requirement with its evaluated candidates and cached verdicts."""
        if recipe.get('gravity_release'):
            return self.low_release_need(InHand(recipe['obj'], None, recipe['arms'], recipe=recipe))
        ctx = self.ctx
        obj, target, options = recipe['obj'], recipe['target'], recipe['options']
        clearance, above = recipe['clearance'], recipe['above']
        verdicts, obj_prefix = dict(recipe['verdicts']), f'{obj}:'
        obstacles, insertion_obstacles = [], []
        occupied = []
        flat_geometry = (self.flat_release_geometry(target, recipe['origin']['region'])
                         if recipe['origin'].get('orientation') == 'flat' else None)
        if recipe['origin'].get('orientation') in ('upright', 'insert'):
            insertion_obstacles = self.insertion_obstacles(target)
            container = recipe['origin']['region'][3].get('upright_container') if len(recipe['origin']['region']) > 3 else None
            occupied = [item for item in insertion_obstacles if item[0].split(':', 1)[0] not in ('table', container)]
        if recipe['origin'].get('orientation'):
            obstacles = [convex_keepout(points)
                         for points in recipe['origin'].get('keep_out', ()) if isinstance(points, np.ndarray)]

        @decision
        @timed('placement_need')
        def accepts(self, arm, tcp_object):
            rotation = Rotation.from_matrix(tcp_object[:3, :3]).as_rotvec()
            # Per hold and per robot state (the base and joints, as a base move or a stow leaves them).
            rounding = (.002, .025) if recipe['origin'].get('orientation') else NEED_ROUND
            key = (arm, *np.round(tcp_object[:3, 3] / rounding[0]).astype(int), *np.round(rotation / rounding[1]).astype(int),
                   self.model.base.round(3).tobytes(), self.model.q.round(2).tobytes())
            mark(self, 'cache', cached=key in verdicts, tcp_object=tcp_object, options=options, origin=recipe['origin'])
            if key not in verdicts:
                # Held so (its parts carried with the hand), as it will be when it is put down.
                solver, poses = self.solvers[arm], ctx.poses()
                body = np.linalg.inv(poses[target.body])
                attached = solver.planner.planning_world.is_object_attached
                # Those already carried (asked once picked) stay as they are held.
                parts = [name for name, obj in solver.world.objects.items()
                         if name.startswith(obj_prefix) and not attached(name)]
                for name in parts:
                    solver.attach_at(name, tcp_object @ body @ poses[solver.world.objects[name].body])
                try:
                    seed = self.model.q
                    if recipe['origin'].get('orientation'):
                        closed = _close(target.shape, target.offset, np.linalg.inv(tcp_object),
                                        self.hands[arm], self.hands[arm].max_gap)
                        if closed is None:
                            verdicts[key] = False
                            recipe['verdicts'] = list(verdicts.items())
                            return False
                        seed = self.model.vector(ctx.spec.arms[arm].finger_targets(closed[0]))
                    def reachable(placed):
                        remaining_wall(ctx)
                        if obstacles and flat_geometry is None:
                            release = translated(placed, [0., 0., clearance]) @ np.linalg.inv(tcp_object)
                            hand = self.hands[arm].fcl(release, min(closed[0] + .01, self.hands[arm].max_gap))
                            if any(fcl.distance(hand, obstacle).min_distance < 0. for obstacle in obstacles):
                                return False
                        if recipe['origin'].get('orientation'):
                            height = (self.flat_release_height(flat_geometry, self.hands[arm], placed,
                                                               tcp_object, closed[0],
                                                               self.flat_open_gap(target, arm, tcp_object), above)
                                      if flat_geometry is not None else clearance)
                            if height is None or not self.in_reach_down(arm, placed, tcp_object, height, above, q=seed):
                                return False
                            if recipe['origin']['orientation'] in ('upright', 'insert'):
                                release = translated(placed, [0., 0., clearance]) @ np.linalg.inv(tcp_object)
                                blocked = self.insertion_blockers(arm, release, above,
                                                                  min(closed[0] + .01, self.hands[arm].max_gap),
                                                                  insertion_obstacles)
                                if blocked:
                                    previous = set(recipe.get('blocked_by', ()))
                                    if set(blocked) - previous:
                                        ctx.note('insertion channel blocked', obj=obj, blocked_by=blocked)
                                    recipe['blocked_by'] = sorted(previous | set(blocked))
                                    return False
                            return True
                        return self.in_reach_down(arm, placed, tcp_object, clearance, above)
                    reached = (reachable(placed) for placed, _ in options)
                    required = 1 if recipe['origin'].get('orientation') else min(NEED_REACHED, len(options)) or 1     # a small region may have fewer
                    verdicts[key] = sum(itertools.islice((1 for ok in reached if ok), required)) >= required
                    recipe['verdicts'] = list(verdicts.items())
                finally:
                    for name in parts:
                        solver.detach(name)
                    solver.world.update(poses)
            mark(self, 'need', outcome='accepted' if verdicts[key] else 'rejected')
            return verdicts[key]
        def hands(arm, tcp_object):
            # Down to each of the first NEED_REACHED of those placements, one path each: release, then above.
            closed = (_close(target.shape, target.offset, np.linalg.inv(tcp_object), self.hands[arm],
                             self.hands[arm].max_gap) if flat_geometry is not None else None)
            if flat_geometry is not None and closed is None:
                return []
            releases = []
            for placed, _ in options[:NEED_REACHED]:
                height = (self.flat_release_height(flat_geometry, self.hands[arm], placed, tcp_object,
                                                   closed[0], self.flat_open_gap(target, arm, tcp_object), above)
                          if flat_geometry is not None and closed is not None else clearance)
                if height is not None:
                    releases.append(translated(placed, [0., 0., height]) @ np.linalg.inv(tcp_object))
            return [[release, translated(release, [0., 0., above])] for release in releases]
        # The placement options and occupied bounds are this need's snapshot.
        # Grasp phases replay identical holds at different robot stances; this
        # geometric preference depends on the hold, not the base or joint seed.
        occupancy = {}
        def rank(arm, hold, gap):
            remaining_wall(ctx)
            hold = np.asarray(hold)
            key = (arm, hold.dtype.str, hold.shape, hold.tobytes(), float(gap))
            if key not in occupancy:
                occupancy[key] = self.insertion_occupancy(arm, hold, gap, options, above, occupied)
            return occupancy[key]
        rank = rank if occupied else None
        need = InHand(obj, lambda arm, tcp_object: accepts(self, arm, tcp_object), recipe['arms'], hands, recipe, rank)
        ctx.requirements.append(need)
        return need

    @decision
    def in_reach_down(self, arm, placed, tcp_object, clearance, above, q=None):
        """Whether ``arm`` holding the object as ``tcp_object`` has IK at its release at ``placed``
        (the torso lift moving too) and above it."""
        release = translated(placed, [0., 0., clearance]) @ np.linalg.inv(tcp_object)
        return bool(self.down_goals(arm, release, above, True, NEED_IK_ATTEMPTS, first=True, q=q))

    @decision
    def down_goals(self, arm, release, above, lift, attempts, first=False, q=None):
        """Above configurations with a release branch at the same torso height."""
        remaining_wall(getattr(self, 'ctx', None))
        mark(self, 'release IK')
        status, releases = self.model.ik(arm, release, self.model.q if q is None else q, world=True, lift=lift,
                                         attempts=attempts, closest=False)
        if status != 'Success':
            mark(self, 'release IK', outcome='rejected', status=status)
            return []
        goals = []
        for branch, q_release in enumerate(sorted(releases, key=lambda q: np.linalg.norm(q - self.model.q))):
            remaining_wall(getattr(self, 'ctx', None))
            mark(self, 'above IK', str(branch))
            status, q_above = self.model.ik(arm, translated(release, [0., 0., above]), q_release,
                                           world=True, attempts=attempts)
            mark(self, 'above IK', outcome='accepted' if status == 'Success' else 'rejected', status=status)
            if status == 'Success':
                goals.append(q_above)
                if first:
                    break
        return goals

    def place(self, obj, region, arm=None, near=None, keep_out=(), clearance=PLACE_CLEARANCE, above=PLACE_ABOVE,
              step=None, margin=None, orientation=None, turns=PLACE_TURNS):
        """Put the held ``obj`` down in ``region`` and release it; see the module docstring.
        ``region`` is ``(frame, low, high)``: the rectangle ``low``-``high`` (x, y) on the surface
        z = 0 of the world 4x4 ``frame``; ``keep_out`` are such rectangles (any height) the
        footprint stays out of; a world Nx3 numpy array instead describes one
        convex keep-out, checked in 3D with ``margin`` clearance (used by oriented
        semantic regions so a leaning pen does not block its entire XY projection).
        A semantic flat-stack region may add a fourth entry with ``surface``
        (the live support rectangle) and body-local ``centre_of_mass``: the
        footprint stays in the region while its COM stays inside that support.
        Its optional ``maximum_height`` bounds only the object's resting top
        above the region plane; the approach and descent remain unchanged.
        ``near`` the world point to place it nearest (default the
        region's centre). ``orientation``: None preserves tilt; ``'upright'`` stands
        a slender object's longest axis along the normal, ``'flat'`` rests a planar
        object's shortest axis along it (either end/face down); ``'insert'`` tries
        each body axis for small objects through a narrow mouth. See :func:`orientations`.
        Upright descent starts at least one object length above release; increase
        ``above`` for deeper containers. Use ``chain([Pick(obj), Place(obj, region,
        orientation='upright')])`` to choose a hold compatible with insertion.
        turns bounds the yaw samples retained per position and body end/face;
        the existing yaw grid covers a full turn. Upright defaults: step at most half the narrowest extent and 2 mm margin
        (others: PLACE_STEP/PLACE_MARGIN); explicit step/margin override.
        Insertion releases within INSERT_MARGIN of the live floor; raised
        release candidates are excluded. Low projected pad area is accepted
        only with measured opposing pad contact; a lost hold raises a
        recoverable error. Withdrawal is vertical. Returns ``{'arm', 'spot', 'object'}``."""
        ctx = self.ctx
        arm = arm or self.holding(obj)
        if arm is None or arm not in self.held or self.held[arm]['target'].obj != obj:
            raise ValueError(f'{obj!r} is not held')
        solver, target = self.solvers[arm], self.held[arm]['target']
        if orientation:
            # A grasp may select one collision part. The entire asset travels
            # with it, and release must detach the same measured set of parts.
            names = [name for name in solver.world.objects if name.startswith(f'{obj}:')]
            if names:
                target = replace(target, names=names)
                self.held[arm]['target'] = target
        current = ctx.poses()[target.body]
        tcp_object = np.linalg.inv(solver.tcp()) @ current
        if orientation:
            # Closing/lifting can settle the object in the pads. Goals and the
            # carried collision geometry must use the same measured hold.
            poses = ctx.poses()
            for name in target.names:
                solver.attach(name, poses[solver.world.objects[name].body])
        near, clearance, above, step, margin, size = self.place_settings(
            target, region, near, clearance, above, step, margin, orientation)
        frame, low, high = (np.asarray(item, dtype=float) for item in region[:3])
        options = self.placements(target, current, region, np.asarray(near, dtype=float)[:3], keep_out, step, margin, orientation,
                                  sweep=above + clearance + size[-1] / 4 if orientation in ('upright', 'insert') else 0., turns=turns)
        if not options:
            if orientation != 'flat' and len(region) > 3 and region[3].get('release_container'):
                return self.low_release(obj, region, arm, np.asarray(near)[:3], keep_out, step, margin,
                                        'no room', orientation)
            ctx.fail('no room', region=[frame.tolist(), low.tolist(), high.tolist()])
        # IK first (cheap) over many, then motion for those that have it. If
        # reach or transfer plans fail, try one alternative base stance before
        # giving up: a release IK alone does not imply a clear transfer path.
        flat_geometry = self.flat_release_geometry(target, region) if orientation == 'flat' else None
        if flat_geometry is not None:
            options = self.centre_flat_options(target, region, options)
        result = self.put_down(arm, options, tcp_object, clearance, above, orientation, flat_geometry)
        if not result['done'] and self.stow():     # none in reach or planned with the other arm out
            result = self.put_down(arm, options, tcp_object, clearance, above, orientation, flat_geometry)
        raised_for_approach = False
        if not result['done']:
            # Moving a low, broad payload around the table can be blocked even
            # though its high destination is reachable. Raise its lowest live
            # point to the same approach plane place already uses, then retry.
            current = ctx.poses()[target.body]
            bottom = (target.points @ current[:3, :3].T + current[:3, 3])[:, 2].min()
            rise = float(frame[2, 3] + above - bottom)
            if rise > clearance:
                raised_for_approach = True
                ctx.note('raise held object to placement approach plane', obj=obj, rise=rise)
                self.move_by_displacement(arm, (0., 0., rise))
                current = ctx.poses()[target.body]
                tcp_object = np.linalg.inv(solver.tcp()) @ current
                options = self.placements(target, current, region, np.asarray(near, dtype=float)[:3], keep_out, step, margin, orientation,
                                          sweep=above + clearance + size[-1] / 4 if orientation in ('upright', 'insert') else 0., turns=turns)
                if flat_geometry is not None:
                    options = self.centre_flat_options(target, region, options)
                result = self.put_down(arm, options, tcp_object, clearance, above, orientation, flat_geometry)
        # A high approach is useful on open surfaces but can put the wrist or
        # torso into a drawer front/shelf. Once the payload has been raised and
        # the ordinary transfer failed, retry the same geometric releases with
        # progressively lower approach heights before relocating the base.
        if raised_for_approach and not result['done']:
            for candidate in sorted({max(clearance, min(above, .04)),
                                     max(clearance, min(above, .02))}, reverse=True):
                if candidate >= above - 1e-9:
                    continue
                ctx.note('retry lower placement approach', obj=obj, above=candidate)
                result = self.put_down(arm, options, tcp_object, clearance, candidate,
                                       orientation, flat_geometry)
                above = candidate
                if result['done']:
                    break
        if not result['done'] and ctx.spec.base:
            if orientation:
                tcp_object = np.linalg.inv(solver.tcp()) @ ctx.poses()[target.body]
            releases = []
            for placed, _ in options[:ROOM_PLACES]:
                height = (self.flat_release_height(flat_geometry, self.hands[arm], placed, tcp_object,
                                                   solver.finger_gap(), self.held[arm]['grasp']['open_gap'], above)
                          if flat_geometry is not None else clearance)
                if height is not None:
                    releases.append(translated(placed, [0., 0., height]) @ np.linalg.inv(tcp_object))
            if releases:
                self.make_room([[release, translated(release, [0., 0., above])] for release in releases], [arm], need=1,
                               touch=[], relocate=bool(orientation) or result['planned'] > 0)
            current = ctx.poses()[target.body]
            options = self.placements(target, current, region, np.asarray(near, dtype=float)[:3], keep_out, step, margin, orientation,
                                      sweep=above + clearance + size[-1] / 4 if orientation in ('upright', 'insert') else 0., turns=turns)
            if flat_geometry is not None:
                options = self.centre_flat_options(target, region, options)
            result = self.put_down(arm, options, tcp_object, clearance, above, orientation, flat_geometry)
        if not result['done']:
            if orientation != 'flat' and len(region) > 3 and region[3].get('release_container'):
                return self.low_release(obj, region, arm, np.asarray(near)[:3], keep_out, step, margin,
                                        result['last'], orientation)
            if orientation == 'flat':
                ctx.fail('no flat placement at minimum collision-free opening height; pick again with a compatible hold',
                         Unreachable, arm=arm, maximum_m=flat_geometry.maximum,
                         recoverable=True, options=len(options),
                         **{k: v for k, v in result.items() if k != 'done'})
            ctx.fail('no placement', Unreachable if result['planned'] == 0 else ExpertFailed, arm=arm,
                     options=len(options), **{k: v for k, v in result.items() if k != 'done'})
        spot = result['spot']
        if orientation == 'upright' and len(region) > 3 and 'upright_container' in region[3]:
            self.check_insertion_depth(target, region[3]['upright_container'], region[3]['part'], clearance)
        if orientation == 'flat':
            self.check_flat_release(target, region, arm, above)
        if orientation == 'flat':
            self.release(arm, upward=above, backoff=RETREATS[0])
        elif orientation in ('upright', 'insert'):
            self.release(arm, upward=above)
        else:
            self.release(arm)
        ctx.hold(.3)
        if orientation == 'upright' and len(region) > 3 and 'upright_container' in region[3]:
            # semantic imports placement constants, so this dependency stays local.
            from table_1000.experts.skills.semantic import upright_inside
            container = region[3]['upright_container']
            if not upright_inside(ctx, obj, container, region[3]['part']):
                ctx.fail('object not upright inside after release; pick it again before placing',
                         obj=obj, container=container, recoverable=True)
        result_out = {'arm': arm, 'spot': spot.tolist(), 'object': ctx.poses()[target.body].tolist()}
        if orientation == 'flat':
            result_out['release_clearance_m'] = result['release_clearance_m']
        return result_out
