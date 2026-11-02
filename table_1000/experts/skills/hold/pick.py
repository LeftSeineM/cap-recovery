"""Grasp and pick: candidates of a target, their IK and motion, the base moved where none is in reach."""

import copy
import time

import numpy as np
from mplib.collision_detection import fcl

from table_1000.experts.context import DT, Unreachable, remaining_wall
from table_1000.experts.skills.hold.frames import MIN_HOLD_AREA, SUPPORT_CLEARANCE, _closing_surface, _grasp_area, contact_resistance, fitted, grasp_candidates, merge_part_heads, standoff, translated
from table_1000.experts.skills.hold.checks import CollisionSurface, closure
from table_1000.planning.world import geometry, pose
from table_1000.experts.skills.common import (APPROACH, CANDIDATE_PLANS, IK_ATTEMPTS, ROOM_PLACES, SUPPORT_DISTANCE,
                                              Target, Unmet, collisions, normalized, planning_object, upright_need)
from table_1000.experts.solver import PlanningFailed
from table_1000.experts.audit.problems import decision, mark

LIFT = .10              # m, pick's lift
HELD_TOLERANCE = .005   # m the closed gap may fall short of the planned contact gap
MAX_REGRASPS = 2
MAX_IK = 60             # candidates checked by IK per arm and choice
MAX_PLANS = 6           # candidates whose motion is tried per grasp attempt, with the torso lift held and moving
# With the torso lift held, only grasps approaching within HELD_SLACK (deg) of the closest a candidate
# of either arm comes to the approach: a worse grasp is not worth sparing the torso.
HELD_SLACK = 15.
MIN_RISE = .05          # m the object must rise in pick's full lift
PROBE_LIFT = .02        # m: expose a missed grip before committing to transport
FOLLOW_TOLERANCE = .01  # m: held material may settle within a pad, not remain on its support
WHY_LIMIT = 20          # rejected grasps per choice whose collisions are counted for the report


def lifting(then):
    """Whether the move ``then`` after a grasp is straight up (a pick's lift, not a drag)."""
    then = np.asarray(then, dtype=float)
    return np.linalg.norm(then[:2]) < 1e-9 and then[2] > 0.


class Replayed:
    """An iterator's items, made as first asked for and replayed from the first on each pass."""

    def __init__(self, items):
        self.items, self.made = iter(items), []

    def __iter__(self):
        yield from self.made
        for item in self.items:
            self.made.append(item)
            yield item


class Grasping:
    """The grasp skills of :class:`~table_1000.experts.skills.Skills`."""

    def _grasp_placement_cache(self, shape, transform, hand, obstacles, approach_distance,
                               contact_surface, *, step=np.radians(15), max_up=.05,
                               end_grasps=False, edge_grasps=False):
        """Cache placement geometry for identical world inputs across grasp phases.

        Preferred directions and occupancy callbacks stay outside the key: they
        affect ordering and costs, not the contact result for a raw choice.
        Obstacle poses and clearances invalidate entries after a world update.
        """
        def array_key(value):
            array = np.asarray(value)
            return array.dtype.str, array.shape, array.tobytes()

        def obstacle_key(obstacle, clearance):
            pose = obstacle.pose.to_transformation_matrix()
            return obstacle.name, array_key(pose), float(clearance), len(obstacle.shapes)

        key = (type(shape), repr(shape.fcl_shape()), array_key(transform),
               (id(getattr(hand, '_contact_geometry_key', hand)),
                float(hand.max_gap), float(hand.opening_clearance), float(hand.pad_spacing),
                float(hand.min_pad_area), float(hand.min_hold_area), float(hand.hold_margin)),
               tuple(obstacle_key(obstacle, clearance) for obstacle, clearance in obstacles),
               contact_surface, float(approach_distance),
               float(step), float(max_up), bool(end_grasps), bool(edge_grasps))
        return self._grasp_placement_caches.setdefault(key, {})

    # -- what is grasped -----------------------------------------------------------------
    def target(self, obj, part=None, shape=None, offset=None, body=None):
        """The :class:`Target` of scene object ``obj`` (an asset, or a primitive with ``shape``):
        ``part`` (a part or body name, default all of it), shaped in the frame of ``body`` (default
        the first body of the part) by a cylinder where it is round, else its collision box (see
        :func:`~table_1000.experts.skills.hold.frames.fitted`), or by ``shape`` at ``offset`` there."""
        ctx = self.ctx
        offset = np.eye(4) if offset is None else np.asarray(offset, dtype=float)
        if obj not in ctx.scene.OBJECTS:
            if shape is None:
                raise ValueError(f'{obj!r} is not an asset of the scene: give its shape')
            for solver in self.solvers.values():
                if obj not in solver.world.objects:
                    solver.world.add(planning_object(obj, obj, shape, offset), ctx.poses()[obj])
            points = shape.points() @ offset[:3, :3].T + offset[:3, 3]
            return Target(obj, [obj], obj, shape, offset, points)
        geometry = ctx.geometry[obj]
        parts = geometry.parts(part)
        if not parts:
            raise KeyError(f'{obj} has no part {part!r}')
        body = body or parts[0][0]
        if shape is None:
            shape, offset = fitted(geometry.points(part, body))
        return Target(obj, [f'{obj}:{name}' for _, name in parts], f'{obj}:{body}', shape, offset,
                      geometry.points(None, body))

    def holding(self, obj):
        """The arm holding ``obj``, or None."""
        return next((arm for arm, held in self.held.items() if held['target'].obj == obj), None)

    # -- choice ------------------------------------------------------------------------
    def unprepared_flat_thin(self, target):
        """Ordinary grasping needs no thin-object preparation; ThinPicking overrides this."""
        return False

    def candidate_hand(self, target, arm):
        if target.container:
            hand = copy.copy(self.hands[arm])
            hand.opening_clearance = .002
            return hand
        return self.hands[arm]

    def centre(self, target):
        """World centre of ``target``'s shape."""
        return (self.ctx.poses()[target.body] @ target.offset)[:3, 3]

    def grasp_center_of_mass(self, target, point):
        """Use the held body's measured COM, including when grasping a small part."""
        if self.constrained(target):
            return point  # rails/hinges support weight; the grasp acts on the chosen handle
        if target.obj not in self.ctx.env.objects:
            return self.centre(target) if point is None else point  # an exact primitive target
        return self.ctx.env.center_of_mass(target.body)[0] if point is None else point

    def constrained(self, target):
        """A non-root articulation body is manipulated through its joint constraint.

        The root of a freely carried articulated object (for example a pen with
        a moving cap) still needs the free-body gravity score.
        """
        structure = self.ctx.env.objects.get(target.obj)
        return structure is not None and structure.articulated and target.body.split(':', 1)[1] != structure.root

    def grasp_weight(self, target):
        return 0. if self.constrained(target) or target.obj not in self.ctx.env.objects \
            else 9.81 * float(self.ctx.env.body_mass(target.body)[0])

    def contact_surface(self, target):
        """All stopping surfaces of the held body, in the candidate shape frame."""
        if target.obj not in self.ctx.geometry or self.constrained(target):
            return None
        geometry = self.ctx.geometry[target.obj]
        key = (target.body, target.offset.tobytes())
        if key not in geometry._candidate_surfaces:
            to_shape = np.linalg.inv(target.offset)
            geometry._candidate_surfaces[key] = CollisionSurface([
                (to_shape @ transform, kind, data) for body, name, transform, kind, data in geometry.shapes
                if body == target.body.split(':', 1)[1]])
        return geometry._candidate_surfaces[key]

    def part_load(self, arm, target, grasp):
        """Gravity torsion divided by the actual patch's projected N*r scale.

        This compares current part heads only. It never changes the original
        within-part cost or adds a second normal-force redirection demand.
        """
        surface = self.contact_surface(target)
        if surface is None or not grasp.info.get('weight_N', 0.):
            return 0.
        transform = self.ctx.poses()[target.body] @ target.offset
        support = contact_resistance(surface, transform, grasp.pose, self.hands[arm], grasp.info['contact_gap'],
                                     open_gap=grasp.open_gap)
        scale = grasp.info['predicted_normal_force_N'] * support['contact_radius_m'] * support['contact_normal_cosine']
        cost = grasp.info['gravity_moment_Nm'] / scale if scale > 0. else np.inf
        grasp.info.update(support, part_load_cost=cost, predicted_torsion_scale_Nm=scale)
        self.ctx.note('part head compared', arm=arm, parts=target.names,
                      pose=grasp.pose.tolist(), target_offset=target.offset.tolist(),
                      within_part_cost=grasp.cost, **grasp.info)
        return cost

    def pad_closure(self, target, arm, open_gap):
        """Measured opposing pad contacts and surface gap, without candidate area gates."""
        if target.obj in self.ctx.scene.OBJECTS:
            geometry = self.ctx.geometry[target.obj]
            body = target.body.split(':', 1)[1]
            # The whole held body can stop a finger, not just the selected part.
            if body not in geometry._grasp_surfaces:
                geometry._grasp_surfaces[body] = CollisionSurface(
                    [(transform, kind, data) for name, _, transform, kind, data in geometry.shapes if name == body])
            surface = geometry._grasp_surfaces[body]
        else:
            surface = CollisionSurface([(target.offset, 'convex', target.shape.points())])
        solver = self.solvers[arm]
        hand = self.hands[arm]
        contacts = []
        for row in self.ctx.env.contacts.surface_contacts.values():
            row = dict(row)
            row['pad_links'] = []
            for link in hand.side.finger_links:
                if 'robot.' + link not in row['bodies']:
                    continue
                explicit = any(shape[3] for shape in hand.shapes[link])
                if row.get('source') == 'tensor' or not explicit:
                    pose = self.ctx.env.link_pose(link)[0]
                    local = (np.asarray(row['points']) - pose[:3, 3]) @ pose[:3, :3]
                    if hand.on_pad(link, local):
                        row['pad_links'].append(link)
            contacts.append(row)
        body_pose, tcp, gap = self.ctx.poses()[target.body], self.ctx.tcp(arm), solver.finger_gap()
        return closure(hand, surface, body_pose, tcp, gap, open_gap, contacts, target.body)

    def check_closure(self, target, arm, grasp, *, kind='check'):
        checked = self.pad_closure(target, arm, grasp.open_gap)
        solver = self.solvers[arm]
        body_pose, tcp, gap = self.ctx.poses()[target.body], self.ctx.tcp(arm), solver.finger_gap()
        # Finite geometric coverage and planned gap rank/describe the grip.
        # They cannot overrule actual following during the controlled lift.
        actual_held = self.measured_held_area(target, arm, body_pose @ target.offset, tcp, gap,
                                             open_gap=grasp.open_gap)
        planned_gap_held = bool(gap >= grasp.info['contact_gap'] - HELD_TOLERANCE)
        area_held = bool(actual_held >= MIN_HOLD_AREA)
        checked.update(planned_contact_gap_m=grasp.info['contact_gap'], held_gap_tolerance_m=HELD_TOLERANCE,
                       actual_held_area_m2=float(actual_held), min_hold_area_m2=MIN_HOLD_AREA,
                       planned_gap_held=planned_gap_held, area_held=area_held)
        from table_1000.experts.audit.search import check, measurement
        check(self.ctx, 'Grasping.check_closure', checked['held'], kind=kind, family_id='grasp.closure',
              measurements=[measurement('finger_gap', gap, 'm', '>=', grasp.info['contact_gap'] - HELD_TOLERANCE,
                                        planned_gap_held, 'observed.finger_gap'),
                            measurement('held_area', actual_held, 'm2', '>=', MIN_HOLD_AREA,
                                        area_held, 'observed.geometry')],
              closure=checked, predicted_grasp=grasp.info)
        return checked

    def measured_held_area(self, target, arm, transform, tcp, gap, *, open_gap=None):
        """Measure the real contact layer within the candidate's finite closing sweep."""
        surface = self.contact_surface(target)
        if open_gap is None:
            held = getattr(self, 'held', {}).get(arm)
            if held and held['target'].body == target.body:
                open_gap = held['grasp']['open_gap']
        if surface is not None and open_gap is not None:
            surface = _closing_surface(surface, transform, tcp, self.hands[arm], open_gap)
        return _grasp_area(target.shape, transform, tcp, self.hands[arm], gap,
                           surface=surface)

    def shoulder(self, arm):
        return self.model.fk(self.ctx.spec.arms[arm].joints[1].replace('joint', 'link'), world=True)[:3, 3]

    def obstacles(self, arm, target, transform):
        """``([(FCL object, clearance to keep)], supports)``: the objects of ``arm``'s planning world
        other than ``target``'s (its shape at ``transform``), and the names of those supporting it."""
        self.solvers[arm].sync()
        world = self.solvers[arm].world
        shape = fcl.FCLObject('target', pose(transform), [fcl.CollisionObject(geometry(*target.shape.fcl_shape()))],
                              [pose(np.eye(4))])
        obstacles, supports = [], []
        for name, obj in world.objects.items():
            if name in target.names or world.planner.planning_world.is_object_attached(name):
                continue
            body = obj.fcl(world.poses[name])
            support = fcl.distance(shape, body).min_distance < SUPPORT_DISTANCE
            if support:
                supports.append(name)
            obstacles.append((body, SUPPORT_CLEARANCE if support else 0.))
        return obstacles, supports

    @decision
    def options(self, target, arms, skip, point=None, approach=None, then=None, max_angle=None, tried=None,
                need=None, mass_center=None, candidate_budget=None, ik_seed=None, geometric=None, local=False):
        """Feasible ``(arm, grasp, supports, q_grasp, q_pre)``, in the order tried: those of either
        arm (the nearer first) with the torso lift where it is (also through ``then``), then those
        moving it too; ``skip`` holds the ids of candidates already used, ``tried`` (``{lift: count}``)
        those whose motion was tried, up to ``MAX_PLANS`` each; ``need`` (an
        :class:`~table_1000.experts.skills.chain.InHand`) keeps those holding the object as it asks.
        ``self.rejected`` counts the candidates dropped, by reason.
        ``candidate_budget(arm, lift)`` charges each preparation IK across calls.
        ``ik_seed`` continues the selected arm branch, retaining other live joints.
        ``geometric`` reuses per-arm supports and candidate streams; IK and
        departure checks still run here.
        """
        targets = list(target) if isinstance(target, (list, tuple)) else [target]
        tried = {False: 0, True: 0} if tried is None else tried
        self.rejected = {'grasp IK': 0, 'pre-grasp IK': 0, 'then': 0, 'need': 0, 'collisions': {}}
        centre = self.centre(targets[0])
        then = None if then is None else np.asarray(then, dtype=float)
        upward = then is not None and lifting(then)
        arms = sorted(arms, key=lambda arm: np.linalg.norm((self.shoulder(arm) - centre)[:2]))
        found, single = {}, {}
        for arm in arms:
            if geometric is not None:
                supports, candidates = geometric[arm]
                stream = Replayed(candidates)
                head = next(iter(stream), None)
                found[arm] = [stream], [(targets[0], supports)], [head.info['closest_deg'] if head is not None else np.inf]
                continue
            streams = []
            metadata = []
            angles = []
            for target in targets:
                transform = self.ctx.poses()[target.body] @ target.offset
                upright = upright_need(need)
                preferred = approach if approach is not None else normalized(
                    transform[:3, 3] - self.shoulder(arm)) + [0., 0., -1.]
                if upright and approach is None:
                    preferred = np.array([0., 0., -1.])
                obstacles, supports = self.obstacles(arm, target, transform)
                surface = self.contact_surface(target)
                # The measured wrist is a preference for a declared next hold
                # at this stance. Unguided fallback keeps physical ordering;
                # grasp_paths likewise describes a future, different stance.
                arguments = dict(check_budget=lambda: remaining_wall(self.ctx),
                                 center_of_mass=point, mass_center=self.grasp_center_of_mass(target, mass_center),
                                 weight=self.grasp_weight(target), approach_distance=APPROACH,
                                 contact_surface=surface,
                                 reference_rotation=self.model.tcp(arm, world=True)[:3, :3]
                                                    if surface is not None and need is not None and not upright else None)
                if need is not None and (need.recipe or {}).get('origin', {}).get('orientation') == 'flat':
                    arguments['edge_grasps'] = True
                if upright:
                    arguments['end_grasps'] = True
                    arguments['first_feasible'] = True
                    if need.rank is not None:
                        body = self.ctx.poses()[target.body]
                        arguments['occupancy_cost'] = lambda tcp, gap, a=arm, b=body: need.rank(a, np.linalg.inv(tcp) @ b, gap)
                arguments['_placement_cache'] = self._grasp_placement_cache(
                    target.shape, transform, self.candidate_hand(target, arm), obstacles,
                    arguments['approach_distance'], surface,
                    end_grasps=arguments.get('end_grasps', False), edge_grasps=arguments.get('edge_grasps', False))
                inputs = target.shape, transform, self.candidate_hand(target, arm), obstacles, preferred
                # As with a single target, the phase needs only the closest
                # valid approach. Cost-sorted part heads can be expensive and
                # are needed only when this arm/phase is actually traversed.
                # Share placements with the full stream, whose candidate and
                # retry ordering remains unchanged.
                probe = next(grasp_candidates(*inputs, **arguments, max_angle=max_angle,
                                               _first_angle=True), None)
                angles.append(probe.info['closest_deg'] if probe is not None else np.inf)
                if len(targets) == 1:
                    single[arm] = inputs, arguments, probe
                    streams.append(())
                    metadata.append((target, supports))
                    continue
                # Keep original per-part retry ids when a phase has candidates.
                candidates = iter(()) if probe is None else grasp_candidates(*inputs, **arguments)
                streams.append(Replayed(candidates))
                metadata.append((target, supports))
            found[arm] = streams, metadata, angles
        # The approach any candidate of either arm comes closest to (each knows it from its first).
        closest = min((angle for _, _, angles in found.values() for angle in angles if np.isfinite(angle)),
                      default=0.)
        for lift in ((False,) if self.braced else (False, True)):
            if lift and not local:
                # Contact preparation needs the same ready configuration as
                # its movable-lift IK preview. Only a missed-grasp retreat
                # keeps the current arm branch for local joint planning.
                self.stow(why='shared torso lift')
            limit = np.degrees(max_angle) if max_angle is not None else np.inf
            # A brace has no movable-lift pass to try the remaining angles.
            if not lift and not self.braced and not local:
                limit = min(limit, closest + HELD_SLACK)
            for arm in arms:
                solver, side = self.solvers[arm], self.ctx.spec.arms[arm]
                streams, metadata, angles = found[arm]
                # A part whose closest valid approach is outside this phase
                # cannot yield a usable candidate. Skip that whole empty phase,
                # retaining its original stream for the movable-lift pass.
                if len(targets) == 1 and geometric is None:
                    inputs, arguments, probe = single[arm]
                    # A single stream can discard this phase's excluded
                    # angles before contact placement. Doing this to part
                    # heads would expose later heads early and reorder them.
                    active = [grasp_candidates(*inputs, **arguments, max_angle=np.radians(limit))
                              if probe is not None and angles[0] <= limit + 1e-6 else ()]
                else:
                    active = [stream if angle <= limit + 1e-6 else ()
                              for stream, angle in zip(streams, angles)]
                candidates = merge_part_heads(active,
                    lambda number, grasp: self.part_load(arm, metadata[number][0], grasp)
                                          if len(streams) > 1 else 0.)
                checked = 0
                for number, index, grasp in candidates:
                    remaining_wall(self.ctx)
                    target, supports = metadata[number]
                    geometry_choice = grasp.choice
                    identifier_index = geometry_choice if len(targets) == 1 and geometry_choice is not None else index
                    mark(self, 'candidate', f'{arm}/{number}/{identifier_index}/{lift}', grasp=grasp, lift=lift)
                    if tried[lift] >= MAX_PLANS:
                        mark(self, 'budget', outcome='not checked')
                        break
                    identifier = (arm, identifier_index) if len(targets) == 1 else (arm, number, index)
                    if identifier in skip:
                        mark(self, 'already planned', outcome='not checked')
                        continue
                    if checked >= MAX_IK:
                        mark(self, 'IK budget', outcome='not checked', checked=checked, limit=limit)
                        break
                    if grasp.info['approach_deg'] > limit + 1e-6:
                        mark(self, 'approach', outcome='not checked', limit=limit)
                        continue  # total cost order is not monotonic in approach angle
                    if candidate_budget is not None:
                        candidate_budget(arm, lift)
                    checked += 1
                    start = self.model.vector(side.finger_targets(grasp.open_gap))
                    seed = ik_seed if ik_seed is not None else self.brace_seed
                    if seed is not None:
                        moving = self.model.arm_indices(arm) + (self.model.lift_indices() if lift else [])
                        start[moving] = seed[moving]
                    solver.sync()
                    # The fingers touch the object and come closer to its supports than the padding.
                    solver.allow([*target.names, *supports])
                    mark(self, 'grasp IK')
                    status, q_grasp = self.model.ik(arm, grasp.pose, start, world=True, lift=lift, attempts=IK_ATTEMPTS)
                    solver.allow([*target.names, *supports], False)
                    if status != 'Success':
                        mark(self, 'grasp IK', outcome='rejected', status=status)
                        self.rejected['grasp IK'] += 1
                        if self.rejected['grasp IK'] <= WHY_LIMIT:
                            self.why(arm, grasp.pose, start, [*target.names, *supports])
                        continue
                    mark(self, 'pre-grasp IK')
                    status, q_pre = self.model.ik(arm, standoff(grasp.pose, APPROACH), q_grasp, world=True,
                                                  attempts=IK_ATTEMPTS)
                    if status != 'Success':
                        mark(self, 'pre-grasp IK', outcome='rejected', status=status)
                        self.rejected['pre-grasp IK'] += 1
                        continue
                    if then is not None:
                        mark(self, 'then')
                        end = translated(grasp.pose, then)
                        status, q_end = self.model.ik(arm, end, q_grasp, world=True, lift=lift, attempts=IK_ATTEMPTS)
                        if status == 'Success' and not upward:
                            # All the way along, as the move goes: an arm folding into itself half way (a
                            # drawer pulled out towards the body) makes it stop there. (A lift up has the
                            # torso lift rise where the arm alone stops.) Then, as the hand lets go there,
                            # straight back out the way it came in: a hand that pulled a drawer up to the
                            # body may have no room left to leave it. (The dragged part stays where it is in
                            # the planning world: this checks the arm's kinematics and the body, not the part.)
                            solver.allow([*target.names, *supports])
                            status = self.model.plan_screw(arm, end, q_grasp, world=True, time_step=DT,
                                                           lift=lift)['status']
                            if status == 'Success':
                                status = self.model.plan_screw(arm, standoff(end, APPROACH), q_end, world=True,
                                                               time_step=DT, lift=lift)['status']
                            solver.allow([*target.names, *supports], False)
                        # Close to the body the hand may not reach higher; then the torso must rise that far.
                        headroom = self.model.limits[self.model.lift_indices(), 1] - q_grasp[self.model.lift_indices()]
                        if status != 'Success' and not (lift and upward and np.min(headroom) >= then[2]):
                            mark(self, 'then', outcome='rejected', status=status)
                            self.rejected['then'] += 1
                            continue
                    mark(self, 'need')
                    if need is not None and not need.accepts(
                            arm, np.linalg.inv(grasp.pose) @ self.ctx.poses()[target.body]):
                        mark(self, 'need', outcome='rejected')
                        self.rejected['need'] += 1
                        continue
                    mark(self, 'screen', outcome='accepted')
                    skip.add(identifier)
                    grasp.target = target
                    yield arm, grasp, supports, q_grasp, q_pre, lift

    @decision
    def why(self, arm, pose, start, allowed):
        """Count, in ``self.rejected['collisions']``, what an IK solution of ``pose`` that ignores
        collisions would collide with (``'no IK'`` when there is none)."""
        status, q = self.model.ik(arm, pose, start, world=True, lift=True, attempts=IK_ATTEMPTS, check_collision=False)
        if status != 'Success':
            keys = ['no IK']
        else:
            solver = self.solvers[arm]
            solver.allow(allowed)
            keys = sorted({f'{link} / {obj}:{obj_link}'
                           for link, obj, obj_link in self.model.collisions(arm, q)}) or ['none']
            solver.allow(allowed, False)
        counts = self.rejected['collisions']
        for key in keys:
            counts[key] = counts.get(key, 0) + 1

    # -- skills --------------------------------------------------------------------------
    def grasp(self, target, arm=None, point=None, approach=None, then=None, max_angle=None, need=None, base=True,
              closure_only_record=False, close=None):
        """Grasp ``target`` (:class:`Target`, or a list ranked together) with ``arm`` (default either); see the module docstring.
        ``need``: only grasps holding it as that :class:`~table_1000.experts.skills.chain.InHand` asks
        (:class:`Unmet` where no other is in reach); ``base``: whether the base may move.
        ``closure_only_record`` defers physical proof to the caller's measured lift (pick and asset tests).
        Free-body closure is provisional until a controlled lift; joint-constrained handles
        use actual opposing pad contact without a vertical probe.
        A flat, unprepared thin body on the table raises ``ValueError``: use :meth:`pick`, or
        :meth:`slide_to_edge` before :meth:`grasp`; overhanging and already held bodies are allowed.
        Returns ``{'arm', 'grasp', 'supports', 'finger_gap', 'regrasps', 'choice_s', 'tcp_object', 'closure_check'}``."""
        ctx = self.ctx
        alternatives = target
        targets = target if isinstance(target, (list, tuple)) else [target]
        if close is None and any(self.unprepared_flat_thin(item) for item in targets):
            raise ValueError('grasp cannot prepare a flat thin object on the table; use pick, '
                             'or call slide_to_edge before grasp')
        target = target[0] if isinstance(target, (list, tuple)) else target
        arms = [arm] if arm else [a for a in self.arms if a not in self.held]
        unmet = need is not None and not set(arms) <= set(need.arms)     # an arm it leaves out
        if need is not None:
            arms = [a for a in arms if a in need.arms]
        skip, regrasps, started, roomed, stowed = set(), 0, time.perf_counter(), False, False
        rerouted = False
        while True:
            reached = self.reach(alternatives, arms, skip, point, approach, then, max_angle, need,
                                 **({'local': True} if regrasps else {}))
            # Candidates, none in reach with the other arm out: again with it in, before the base moves;
            # first the arms but the nearest (which may then reach from where it is), then all.
            # (None at all within max_angle: the next, wider one is tried instead.)
            considered = skip or any(self.rejected[key] for key in ('grasp IK', 'pre-grasp IK', 'then', 'need'))
            if reached is None and considered and not stowed:
                stowed = True
                nearest = min(self.arms, key=lambda a: np.linalg.norm((self.shoulder(a) - self.centre(target))[:2]))
                for out in ([a for a in self.arms if a != nearest], None):
                    if reached is None and self.stow(out, why=self.rejected):
                        skip = set()
                        reached = self.reach(alternatives, arms, skip, point, approach, then, max_angle, need)
            if reached is None and not roomed and ctx.spec.base and base:
                # None in reach from here: the base moves to reach the first candidates, then again.
                roomed = True
                paths = self.grasp_paths(alternatives, arms, point, approach, then, max_angle, need)
                if any(paths.values()):
                    # The fingers touch what options lets them: the target and its supports.
                    options = list(alternatives) if isinstance(alternatives, (list, tuple)) else [alternatives]
                    touch = []
                    for item in options:
                        _, supports = self.obstacles(arms[0], item, ctx.poses()[item.body] @ item.offset)
                        touch.extend([*item.names, *supports])
                    self.make_room(paths, need=1, touch=list(dict.fromkeys(touch)), relocate=True)
                    skip = set()
                    reached = self.reach(alternatives, arms, skip, point, approach, then, max_angle, need)
            if (reached is None and roomed and skip and not rerouted
                    and ctx.spec.base and base and not self.braced):
                # A stance selected for endpoint IK can still have no arm
                # approach path. As for placing, try one alternative stance
                # after actual approach plans failed; do not retry unchanged
                # geometry when no candidate ever passed the IK screens.
                rerouted = True
                paths = self.grasp_paths(alternatives, arms, point, approach, then, max_angle, need)
                if any(paths.values()):
                    touch = []
                    for item in targets:
                        _, supports = self.obstacles(arms[0], item, ctx.poses()[item.body] @ item.offset)
                        touch.extend([*item.names, *supports])
                    ctx.note('try another base stance after failed grasp approach', obj=target.obj)
                    self.make_room(paths, need=1, touch=list(dict.fromkeys(touch)), relocate=True)
                    skip = set()
                    reached = self.reach(alternatives, arms, skip, point, approach, then, max_angle, need)
            if reached is None:
                # With a need and the base held, only this stance failed: Unmet, for the chain to adapt (the base moves).
                unmet = unmet or need is not None and (not base or self.rejected['need'])
                blocked = (need.recipe or {}).get('blocked_by', ()) if need is not None else ()
                release_limit = ((need.recipe or {}).get('origin', {}).get('release_limit_m')
                                 if need is not None else None)
                ctx.fail('no reachable grasp', Unmet if unmet else Unreachable, kind='planning', target=target.names,
                         candidates_tried=len(skip), rejected=self.rejected,
                         **({'maximum_flat_release_m': release_limit, 'recoverable': True,
                             'recovery': 'pick again with a hold compatible with low flat release'}
                            if release_limit is not None else {}),
                         **({'blocked_by': blocked, 'recoverable': True} if blocked else {}))
            arm, grasp, supports = reached
            target = grasp.target or target
            solver = self.solvers[arm]
            if close is None:
                solver.close_gripper()
            else:
                close(target, arm, grasp)
            gap = solver.finger_gap()
            checked = self.check_closure(target, arm, grasp, kind='record' if closure_only_record else 'check')
            ctx.note('closure checked', arm=arm, only_record=closure_only_record, **checked)
            if checked['held'] or closure_only_record or not self.constrained(target):
                break
            # Missed: back out, re-observe the object and choose again.
            ctx.note('missed grasp', arm=arm, finger_gap=gap, width=grasp.width, pose=grasp.pose.tolist(),
                     closure=checked, **grasp.info)
            solver.open_gripper(grasp.open_gap)
            solver.first_reachable([standoff(solver.tcp(), APPROACH), translated(solver.tcp(), [0., 0., APPROACH])])
            solver.allow([*target.names, *supports], False)
            regrasps += 1
            if regrasps > MAX_REGRASPS:
                ctx.fail('object not held', kind='check', finger_gap=gap, width=grasp.width)
        poses = ctx.poses()
        # A free body moves as one rigid collision assembly. Joint-constrained
        # bodies continue to use Motion.dragged for their separate riders.
        carried, riders, contacts, carry_permissions = [], [], [], []
        free = target.obj in ctx.geometry and not self.constrained(target)
        if free:
            previous_permissions = set(solver.world.permissions)
            # dragged inspects unattached source pieces. Discover contents
            # before attaching even a whole-container grasp.
            carried, riders, contacts = self.dragged(arm, (target.obj, target.body.split(':', 1)[1]))
            # A free asset's root can carry authored child bodies (pen caps
            # and lid detents). Whole-object grasps already moved all of
            # these planning pieces; a part grasp must preserve that model.
            own = [name for name in solver.world.objects if name.startswith(target.obj + ':')]
            carried = [name for name in own if name not in target.names]
            if not (target.rim or target.container):
                # Ordinary parts may rest in a drawer: that supporting
                # cabinet must not become a carried payload. Only the
                # geometrically identified container carries touching contents,
                # including when an ordinary authored wall part is grasped.
                riders, contacts = [], []
            else:
                # semantic imports this package's public skills; defer to avoid the cycle.
                from table_1000.experts.skills import semantic
                contents = {name.split(':', 1)[0] for name in riders}
                contents = {name for name in contents if semantic.supported_inside(ctx, name, target.obj)}
                riders = [name for name in riders if name.split(':', 1)[0] in contents]
        for name in target.names:
            solver.attach(name, poses[solver.world.objects[name].body])
        if free:
            solver.carry(carried, target.names + supports + riders + contacts)
            solver.carry(riders, target.names + carried + contacts)
            # MPlib attachment recreates ACM entries. Restore the scene's
            # authored intra-asset permissions after all pieces are attached.
            solver.world.allow(own, own)
            solver.world.allow(riders, own + riders)
            carry_permissions = [list(pair) for pair in solver.world.permissions - previous_permissions]
            # Adjacent pieces of the held body were approach supports. Once
            # attached they remain deliberate finger contacts, rather than
            # being revoked with the table after the first lift.
            supports = [name for name in supports if name not in carried + riders]
        chosen = {'pose': grasp.pose.tolist(), 'open_gap': grasp.open_gap, 'width': grasp.width,
                  'backoff': grasp.backoff, 'cost': grasp.cost, **grasp.info}
        chosen['geometry_choice'] = getattr(grasp, 'choice', None)
        self.held[arm] = {'target': target, 'supports': supports, 'grasp': chosen, 'carried': carried,
                          'riders': riders, 'carry_contacts': contacts, 'carry_permissions': carry_permissions}
        ctx.note('grasped', arm=arm, target=target.names[0], approach=grasp.pose[:3, 0].tolist(),
                 closing=grasp.pose[:3, 1].tolist(), finger_gap=solver.finger_gap(),
                 closure=checked, **grasp.info)
        result = {'arm': arm, 'grasp': chosen, 'supports': supports, 'finger_gap': solver.finger_gap(),
                  'regrasps': regrasps, 'choice_s': time.perf_counter() - started,
                  'tcp_object': (np.linalg.inv(solver.tcp()) @ poses[target.body]).tolist(),
                  'closure_check': checked}
        self.log.append(result)
        if not closure_only_record and not self.constrained(target):
            self.lift_grasp(target, result, PROBE_LIFT)
        return result

    def grasp_paths(self, target, arms, point=None, approach=None, then=None, max_angle=None, need=None):
        """``{arm: paths}``: the world TCP poses of the first ``ROOM_PLACES`` grasp candidates of each
        arm, each with its pre-grasp before it (and the ``then`` move after it): what the base moves to
        reach; with a ``need`` that says where the hand takes the object next (``InHand.hands``), a path
        per such place after it, so that the base stands where both are in reach."""
        targets = list(target) if isinstance(target, (list, tuple)) else [target]
        paths = {arm: [] for arm in arms}
        def candidates(arm, target):
            transform = self.ctx.poses()[target.body] @ target.offset
            upright = upright_need(need)
            preferred = approach if approach is not None else normalized(
                transform[:3, 3] - self.shoulder(arm)) + [0., 0., -1.]
            if upright and approach is None:
                preferred = np.array([0., 0., -1.])
            obstacles, _ = self.obstacles(arm, target, transform)
            surface = self.contact_surface(target)
            arguments = {'check_budget': lambda: remaining_wall(self.ctx),
                         'edge_grasps': need is not None and (need.recipe or {}).get('origin', {}).get('orientation') == 'flat'}
            arguments['_placement_cache'] = self._grasp_placement_cache(
                target.shape, transform, self.hands[arm], obstacles, APPROACH, surface,
                end_grasps=upright, edge_grasps=arguments['edge_grasps'])
            if upright and need.rank is not None:
                body = self.ctx.poses()[target.body]
                arguments['occupancy_cost'] = lambda tcp, gap: need.rank(arm, np.linalg.inv(tcp) @ body, gap)
            for grasp in grasp_candidates(target.shape, transform, self.hands[arm], obstacles, preferred,
                    center_of_mass=point, mass_center=self.grasp_center_of_mass(target, None),
                    weight=self.grasp_weight(target), approach_distance=APPROACH,
                    contact_surface=surface, max_angle=max_angle, end_grasps=upright,
                    first_feasible=upright, **arguments):
                yield grasp
        for arm in arms:
            ordered = merge_part_heads([candidates(arm, item) for item in targets],
                lambda number, grasp: self.part_load(arm, targets[number], grasp) if len(targets) > 1 else 0.)
            count = 0
            for number, _, grasp in ordered:
                target = targets[number]
                if max_angle is not None and grasp.info['approach_deg'] > np.degrees(max_angle) + 1e-6:
                    continue
                if count >= ROOM_PLACES:
                    break
                count += 1
                path = [standoff(grasp.pose, APPROACH), grasp.pose]
                if then is not None:
                    path.append(translated(grasp.pose, then))
                    if not lifting(then):
                        path.append(standoff(path[-1], APPROACH))   # letting go there (see options)
                if need is not None and need.hands is not None:
                    if arm in need.arms:
                        body = self.ctx.poses()[target.body]
                        paths[arm] += [dict(poses=path + hands,
                                               linear=[False] + [True] * (len(path) - 1) + [False] + [True] * (len(hands) - 1))
                                      for hands in need.hands(arm, np.linalg.inv(grasp.pose) @ body)]
                    continue
                paths[arm].append(path)
        return paths

    @decision
    def reach(self, target, arms, skip, point, approach, then, max_angle=None, need=None, local=False):
        """Move to the first grasp of :meth:`options` whose motion plans, fingers open around the
        object; ``(arm, grasp, supports)``, or None when no candidate is left."""
        ctx, tried = self.ctx, {False: 0, True: 0}
        for arm, grasp, supports, q_grasp, q_pre, lift in self.options(target, arms, skip, point, approach, then,
                                                                      max_angle, tried, need, local=local):
            mark(self, 'plan', f'{arm}/{tried[lift]}/{lift}', grasp=grasp)
            tried[lift] += 1
            solver = self.solvers[arm]
            selected = grasp.target or target
            try:
                solver.open_gripper(grasp.open_gap)
                # With the torso lift held, one try each: where they fail, the lift is likely needed.
                solver.move_to_pose_with_RRTConnect(None, lift=True, goals=[q_pre],
                                                    attempts=CANDIDATE_PLANS if lift else 1,
                                                    **({'local_lift': True} if local else {}))
                solver.allow([*selected.names, *supports])
                solver.move_to_pose_with_screw(grasp.pose)
                mark(self, 'plan', outcome='accepted')
                return arm, grasp, supports
            except PlanningFailed as error:
                solver.allow([*selected.names, *supports], False)
                mark(self, 'plan', outcome='rejected', reason=error.reason)
                ctx.note('next grasp', arm=arm, cost=grasp.cost, lift=lift, reason=error.reason, **collisions(error))
                if not lift and error.reason.startswith('RRTConnect'):
                    # No way there at this torso height: the other candidates held so share the narrow
                    # passage (a hand from high down to a low handle); on to those moving the lift.
                    tried[False] = MAX_PLANS
        return None

    def pick(self, target, arm=None, point=None, approach=None, height=LIFT, max_angle=None, need=None, base=True):
        """:meth:`grasp` with the lift checked, then up by ``height``; the object must have risen.
        The thin-object route raises ``ValueError`` for non-default ``point``, ``approach`` or
        ``max_angle``: remove them so that preparation can choose its own grasp geometry.
        """
        ctx = self.ctx
        alternatives = target
        target = target[0] if isinstance(target, (list, tuple)) else target
        profile = self.thin_geometry(target)
        if profile is None:
            result = self.grasp(alternatives, arm, point, approach, then=(0., 0., height), max_angle=max_angle, need=need,
                                base=base, closure_only_record=True)
        elif profile.kind == 'plate':
            # Authored narrow features can already offer a legal grasp (a
            # bookmark/spine). Keep that route before moving a broad plate.
            try:
                result = self.grasp(alternatives, arm, point, approach, then=(0., 0., height),
                                    max_angle=max_angle, need=need, base=base, closure_only_record=True,
                                    close=lambda selected, arm, grasp: self.solvers[arm].close_gripper())
            except Unreachable as error:
                # A chain's requested next hold has not been met at this stance.
                # Let chain adapt the base/hand before attempting edge preparation.
                if isinstance(error, Unmet) or error.reason != 'no reachable grasp':
                    raise
                if point is not None or approach is not None or max_angle is not None:
                    raise ValueError('plate edge preparation chooses its own grasp geometry; '
                                     'remove point, approach and max_angle') from error
                target, result = self.grasp_thin(target, arm=arm, height=height, need=need, base=base, closure_only_record=True)
        else:
            if point is not None or approach is not None or max_angle is not None:
                raise ValueError('the thin-object pick route does not support point, approach or max_angle '
                                 'because preparation chooses its own grasp geometry; remove these arguments')
            target, result = self.grasp_thin(target, arm=arm, height=height, need=need, base=base, closure_only_record=True)
        target = self.held[result['arm']]['target']
        if 'preparation' in result:
            self.prepare_thin_lift(target, result)
        self.lift_grasp(target, result, height)
        return result

    def check_hold(self, target, arm, reference=None):
        """Physical contact and material following, independent of projected held area.

        Track the material point initially at the TCP, so a pen may pivot about
        its grip without its far end being mistaken for translational slip.
        """
        solver = self.solvers[arm]
        relative = np.linalg.inv(solver.tcp()) @ self.ctx.poses()[target.body]
        reference = self.held[arm].get('verified_hold') if reference is None else reference
        following = 0. if reference is None else float(np.linalg.norm(
            (relative @ np.linalg.inv(reference))[:3, 3]))
        checked = self.pad_closure(target, arm, self.held[arm]['grasp']['open_gap'])
        contacts = bool(checked['pad_contacts']) and all(checked['pad_contacts'].values())
        held = (contacts and 'closed past collision surfaces' not in checked['reasons']
                and following <= FOLLOW_TOLERANCE)
        checked.update(held=bool(held), following_error_m=following,
                       following_tolerance_m=FOLLOW_TOLERANCE)
        self.ctx.note('hold measured', obj=target.obj, arm=arm, **checked)
        return checked

    def lift_grasp(self, target, result, height=LIFT):
        """Shared pick/asset-test lift: a 20 mm probe, then the remaining departure.

        A 100 mm pickup retains the existing 50 mm minimum rise. The short
        probe requires half its travel in body rise plus two real pad contacts
        and <=10 mm held-point slip after settling. Failed proof releases the
        provisional planning attachment and never returns an empty-hand pick.
        """
        ctx, arm = self.ctx, result['arm']
        if height <= 0.:
            raise ValueError('a grasp proof needs a positive lift')
        solver = self.solvers[arm]
        before = ctx.poses()[target.body].copy()
        reference = np.linalg.inv(solver.tcp()) @ before
        probe = min(PROBE_LIFT, height)
        requested = 0.
        supports = list(self.held[arm]['supports'])
        for distance in (probe, height - probe):
            if distance <= 0.:
                continue
            self.move_by_displacement(arm, (0., 0., distance))
            ctx.hold(.2)
            after = ctx.poses()[target.body]
            rise = float(after[2, 3] - before[2, 3])
            checked = self.check_hold(target, arm, reference)
            requested += distance
            minimum = min(MIN_RISE, requested / 2)
            result.update(rise=rise, lift_check=checked)
            ctx.note('grasp lift measured', obj=target.obj, arm=arm, rise_m=rise,
                     minimum_rise_m=minimum, requested_m=requested, held=checked['held'])
            if rise < minimum or not checked['held']:
                ctx.note('missed grasp', arm=arm, closure=result.get('closure_check'),
                         rise_m=rise, lift_check=checked)
                # Detach/open even when the withdrawal itself cannot be planned.
                from table_1000.experts.context import ExpertFailed
                try:
                    self.release(arm)
                except ExpertFailed as error:
                    ctx.note('missed grasp release failed', reason=error.reason, details=error.details)
                ctx.fail('object not lifted' if rise < minimum else 'object slipped during lift', kind='check',
                         rise=rise, finger_gap=solver.finger_gap(), arm=arm,
                         grasp=result['grasp'], object_before=before.tolist(), lift_check=checked)
            if requested < height:
                # The original full departure allowed these same supports.
                solver.allow(supports)
                self.held[arm]['supports'] = supports
                if self.held[arm]['carried']:
                    solver.world.allow(self.held[arm]['carried'], supports)
        poses = ctx.poses()
        for name in list(solver.attachments):
            solver.attach(name, poses[solver.world.objects[name].body])
        self.held[arm]['verified_hold'] = np.linalg.inv(solver.tcp()) @ poses[target.body]
        result.update(finger_gap=solver.finger_gap(),
                      tcp_object=self.held[arm]['verified_hold'].tolist())
        return result
