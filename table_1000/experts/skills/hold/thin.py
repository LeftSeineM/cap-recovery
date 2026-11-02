"""Pressure-drag to a table edge and pinch the exposed collision geometry."""

from dataclasses import replace
import copy
import heapq
import itertools
import math

import numpy as np

from table_1000.experts.context import DT, ExpertFailed, Unreachable, remaining_wall, steps
from table_1000.experts.skills.hold.checks import CollisionSurface
from table_1000.experts.skills.hold.frames import BACKOFF_STEP, CONTACT_TOLERANCE, HOLD_AREA, _grasp_area, _vertices, build_grasp_pose, grasp_candidates, translated
from table_1000.experts.skills.common import APPROACH, IK_ATTEMPTS
from table_1000.experts.skills.hold.pick import MAX_IK, MAX_REGRASPS, PROBE_LIFT
from table_1000.scenes.geometry import transform_points
from table_1000.experts.solver import ARRIVAL_TOLERANCE, GRIPPER_TIME, PlanningFailed
from table_1000.experts.skills.push.pressure import PressureLoad, correction_path, material_motion, same_pressure_contact
from table_1000.experts.skills.hold.sections import ExposedSection, classify, exposed_section, landing, overhang_depth, supported_overhangs, support_polygon, supported_point, swept_obstacle, swept_obstacles, table_edges


class PreparationSearch:
    """One ordinary IK allowance per table edge, arm and torso choice.

    Depth samples, lateral fallback and reobservations belong to the same
    preparation. Giving each one a fresh allowance multiplies otherwise
    finite loops into hours of IK without executing a contact motion.
    """

    def __init__(self, ctx, arms, edges):
        self.ctx = ctx
        self.limit = MAX_IK * len(arms) * len(edges) * 2
        self.checked = 0
        self.mode = None
        self.by_mode = {}
        self.pressure_retries = 0
        self.empty_pressures = []
        self.failed_pressures = []
        self.counts = {arm: {'fixed': 0, 'lift': 0} for arm in arms}

    def __call__(self, arm, lift):
        if self.checked >= self.limit:
            # Exhausting a resource allowance does not prove geometric
            # unreachability; do not restart the same search via a fallback.
            self.ctx.fail('thin preparation search exhausted', kind='planning', candidates=self.checked,
                          limit=self.limit, by_arm=self.counts, by_mode=self.by_mode)
        self.checked += 1
        self.by_mode[self.mode] = self.by_mode.get(self.mode, 0) + 1
        self.counts[arm]['lift' if lift else 'fixed'] += 1


class ThinPicking:
    def contact_surface(self, target):
        """An exposed section already represents the observed material beyond the edge."""
        if isinstance(target.shape, ExposedSection):
            return target.shape
        return super().contact_surface(target)

    def candidate_hand(self, target, arm):
        hand = super().candidate_hand(target, arm)
        if not getattr(self, '_fingertip_candidates', False):
            return hand
        tip = copy.copy(hand)
        # This skill deliberately contacts the fingertip. Interior-pad grasps
        # retain their normal 2 mm end margin, and measured closure still uses
        # the original hand and all its checks.
        tip.hold_margin = 0.
        return tip

    def pinch_thin(self, target, arm=None, height=.1, need=None, base=True, closure_only_record=False):
        """Pinch at the finger ends with ordinary table clearance and closure checks."""
        self.ctx.note('thin fingertip pinch', target=target.names)
        self._fingertip_candidates = True
        try:
            result = self.grasp(target, arm=arm, approach=(0., 0., -1.), then=(0., 0., height),
                                need=need, base=base, close=self.close_thin,
                                closure_only_record=True)
            if not closure_only_record:
                self.lift_grasp(target, result, PROBE_LIFT)
            return target, result
        finally:
            self._fingertip_candidates = False

    def thin_geometry(self, target):
        """Classify an unconstrained rigid body by collision geometry and this hand."""
        if target.obj not in self.ctx.geometry or self.constrained(target):
            return None
        geometry = self.ctx.geometry[target.obj]
        body = target.body.split(':', 1)[1]
        if not hasattr(geometry, '_thin_profiles'):
            geometry._thin_profiles = {}
        key = (body, self.ctx.spec.name, self.arms[0])
        if key in geometry._thin_profiles:
            return geometry._thin_profiles[key]
        pieces = [data @ transform[:3, :3].T + transform[:3, 3]
                  for name, _, transform, kind, data in geometry.shapes if name == body and kind == 'convex']
        result = classify(geometry.points(frame=body), pieces, self.hands[self.arms[0]])
        geometry._thin_profiles[key] = result
        return result

    def unprepared_flat_thin(self, target):
        """Whether a classified thin body rests flat on the table without a graspable overhang."""
        thin = self.thin_geometry(target)
        if thin is None or self.holding(target.obj) is not None:
            return False
        ctx = self.ctx
        table = ctx.scene.TABLE
        pose = ctx.poses()[target.body]
        world = transform_points(pose, target.points)
        height = table['height']
        if (world[:, 2].min() > height + CONTACT_TOLERANCE
                or world[:, 2].max() < height - CONTACT_TOLERANCE
                or np.ptp(world[:, 2]) > thin.contact_band + 2 * thin.thickness):
            return False

        geometry = ctx.geometry[target.obj]
        body = target.body.split(':', 1)[1]
        pieces = []
        for name, _, local, kind, data in geometry.shapes:
            if name == body and kind == 'convex':
                transform = pose @ local
                pieces.append(data @ transform[:3, :3].T + transform[:3, 3])
        center = ctx.env.center_of_mass(target.body)[0]
        polygon = support_polygon(pieces, table, CONTACT_TOLERANCE)
        if not supported_point(center, polygon):
            return False

        edges = table_edges(table, ctx.base_transform()[:3, 3])
        for arm in self.arms:
            # The preparation also accepts fingertip grips with no interior
            # tip margin; use that same minimum rather than its soft area goal.
            hand = copy.copy(self.hands[arm])
            hand.hold_margin = 0.
            for edge in edges:
                span = float(np.ptp(world @ edge.tangent))
                if span <= 0.:
                    continue
                required = overhang_depth(hand, span) + CONTACT_TOLERANCE
                if edge.distances(world).max() >= required:
                    return False
        return True

    def close_thin(self, target, arm, grasp):
        """Wait for opposing pad contact instead of ending during free jaw travel.

        The deadline follows the unchanged drive's overdamped free-closing
        response, including effort saturation. It is a failure bound; actual
        contact and the common surface check decide completion.
        """
        solver = self.solvers[arm]
        robot = self.ctx.env.robot
        side = self.ctx.spec.arms[arm]
        ids = [robot.joint_names.index(name) for name in side.fingers]
        stiffness = robot.data.joint_stiffness[0, ids].cpu().numpy()
        damping = robot.data.joint_damping[0, ids].cpu().numpy()
        effort = robot.data.joint_effort_limits[0, ids].cpu().numpy()
        actual = self.ctx.env.qpos(side.fingers)[0]
        goal = np.array(list(side.finger_targets(0.).values()))
        # The existing geometry cache resolves total pad opening to 0.5 mm.
        tolerance = np.array(list(side.finger_targets(.0005).values())) - goal
        deadline = GRIPPER_TIME
        for k, d, limit, error, resolution in zip(stiffness, damping, effort, abs(actual - goal), abs(tolerance)):
            if k <= 0:
                continue  # passive mimic joint
            if error <= resolution:
                continue
            saturated = limit / k
            seconds = d / k * max(0., math.log(min(error, saturated) / max(resolution, 1e-12)))
            seconds += max(0., error - saturated) * d / limit
            deadline = max(deadline, seconds + DT)
        checked = None
        for step in range(steps(deadline)):
            self.ctx.step({f'{arm}_gripper': [[0.]]})
            checked = self.check_closure(target, arm, grasp, kind='record')
            # PhysX reports speculative contacts before the surfaces touch.
            # Those contacts suffice for the ordinary diagnostic, but must not
            # end a thin pinch while the jaws are still travelling freely.
            touching = all(any(row['separation_m'] <= 0. for row in rows)
                           for rows in checked['pad_contacts'].values())
            if (step + 1) * DT >= GRIPPER_TIME and checked['held'] and touching:
                break
        self.ctx.note('thin closure completed', arm=arm, seconds=(step + 1) * DT,
                      deadline_s=deadline, held=checked['held'], gap_m=solver.finger_gap())

    def slide_to_edge(self, target, arm=None, height=.1):
        """Reobserve and repress solid material after every finite drag segment.

        Travel and solid patch size determine each round's finite motion budget.
        Existing preparation retries use observed remaining travel and retain
        the selected edge. A shared candidate allowance also bounds searches
        before any motion. Every segment checks support and swept volume.
        """
        budget = None
        segments = []
        prepared = None
        retries = 0
        arms = [arm] if arm else [name for name in self.arms if name not in self.held]
        edges = table_edges(self.ctx.scene.TABLE, self.ctx.base_transform()[:3, 3])
        search = PreparationSearch(self.ctx, arms, edges)
        self.ctx.note('thin preparation search', candidate_limit=search.limit)
        while True:
            result = self._slide_segment(target, arm, height, prepared, search=search)
            prepared = result
            segments.append({key: value for key, value in result.items() if key != 'edge'})
            if budget is None:
                budget = math.ceil(result['remaining_command_m'] / result['budget_span_m']) + 1
            # An area bound alone is insufficient; observed yaw and slip can
            # also make the old planned depth obsolete. Check the real grip.
            if (result['overhang_m'] >= result['required_overhang_m']
                    and self._edge_grasp_available(target, result, height)):
                result['segments'] = segments
                return result
            self.ctx.note('thin drag reobserve', segment=len(segments), budget=budget,
                          overhang_m=result['overhang_m'], required_m=result['planned_overhang_m'])
            if len(segments) >= budget:
                if retries >= MAX_REGRASPS:
                    self.ctx.fail('insufficient thin object overhang', kind='check', required_m=result['planned_overhang_m'],
                                  measured_m=result['overhang_m'], segments=segments)
                retries += 1
                # Slip accumulates over a long stroke. A retry budgets the
                # observed remaining travel and retains the committed edge.
                budget += max(1, math.ceil(result['remaining_command_m'] / result['budget_span_m']))
                self.ctx.note('thin preparation retry', attempt=retries, budget=budget,
                              reason='insufficient thin object overhang')

    def pressure_geometry(self, target, pose):
        """Body collision material and physical pressure patches in one observed pose."""
        body = target.body.split(':', 1)[1]
        pieces = [(frame, kind, data) for name, _, frame, kind, data in self.ctx.geometry[target.obj].shapes
                  if name == body]
        patches, convex = [], []
        for frame, kind, data in pieces:
            placed = pose @ frame
            if kind == 'sphere':
                vertices = np.asarray(list(itertools.product((-data, data), repeat=3)))
                convex.append(transform_points(placed, vertices))
            elif kind == 'convex':
                points = transform_points(placed, data)
                convex.append(points)
                patches.append((points.mean(0), points))
        return CollisionSurface(pieces), patches, convex

    def _slide_segment(self, target, arm=None, height=.1, prepared=None, search=None):
        """Press the top collision surface and drag to a reachable supported edge landing.

        Contact depth is half the existing 1 mm contact layer. The common
        planners allow contact only for the fingers, target and support table.
        Returns the measured overhang and the selected edge; no attachment or
        external force moves the object.
        """
        ctx = self.ctx
        if prepared is not None and (prepared.get('repeated_empty_pressure')
                                     or self.thin_geometry(target).kind == 'plate'):
            # An unchanged body does not need a whole new mode/depth preview.
            # Recheck the committed landing with all ordinary grasp checks.
            candidate = prepared['arm']
            hand = self.hands[candidate]
            if prepared['fingertip']:
                hand = copy.copy(hand)
                hand.hold_margin = 0.
            return self._recheck_pressure_landing(target, candidate, prepared['edge'], hand,
                np.asarray(prepared['landing_com_m']), prepared['required_overhang_m'],
                prepared['planned_overhang_m'],
                (prepared['grasp_approach'], prepared['grasp_joints']), prepared, search,
                prepared['fingertip'], prepared['landing_rule'], [], height)
        pose = ctx.poses()[target.body]
        surface, patches, convex = self.pressure_geometry(target, pose)
        world = transform_points(pose, target.points)
        center = ctx.env.center_of_mass(target.body)[0]
        patches.sort(key=lambda item: np.linalg.norm(item[0][:2] - center[:2]))
        arms = [arm] if arm else [name for name in self.arms if name not in self.held]
        edges = table_edges(ctx.scene.TABLE, ctx.base_transform()[:3, 3])
        if prepared is not None:
            arms, edges = [prepared['arm']], [prepared['edge']]
        rejected = []
        plate = self.thin_geometry(target).kind == 'plate'
        if plate:
            solver = self.solvers[arms[0]]
            solver.sync()
            live = ctx.poses()
            sweep_obstacles = [(name, obj.fcl(live.get(obj.body, solver.world.poses[name])))
                               for name, obj in solver.world.objects.items()
                               if obj.body != target.body and name != 'table']
        # Keep the existing edge and arm order. Within each, preview all
        # modes before spending IK, ranked by the existing soft area target,
        # original candidate cost, then physical drag distance. Measure area
        # with the same interior pad margin for both modes: permitting a tip
        # contact must not make its edge area look like a better interior grip.
        serial = itertools.count()
        for edge, candidate in itertools.product(edges, arms):
            queue, preview_counts = [], {}
            for fingertip, rule in itertools.product((False, True), (None, 'shoulder')):
                mode = ('tip' if fingertip else 'pad') + '/' + (rule or 'nearest')
                counts = preview_counts[mode] = {'depths': 0, 'supported': 0, 'geometric': 0, 'blocked': 0}
                hand = self.hands[candidate]
                if fingertip:
                    hand = copy.copy(hand)
                    hand.hold_margin = 0.
                along = (prepared['landing_along_m'] if prepared is not None and rule is None else
                         float((self.shoulder(candidate) - edge.point) @ edge.tangent) if rule else None)
                transverse = float(np.ptp(world @ edge.tangent))
                depth = overhang_depth(hand, transverse)
                maximum = edge.distances(world).max() - edge.distances(center)
                preferred = np.asarray(prepared['grasp_approach']) if prepared is not None else -edge.outward
                seed = np.asarray(prepared['grasp_joints']) if prepared is not None else None
                desired = (prepared['planned_overhang_m'] if prepared is not None else
                           overhang_depth(hand, transverse, area=HOLD_AREA) + CONTACT_TOLERANCE)
                for exposed in supported_overhangs(depth + CONTACT_TOLERANCE, maximum,
                        BACKOFF_STEP, np.abs(world).max(), preferred=desired):
                    counts['depths'] += 1
                    shift = landing(world, center, edge, exposed, along)
                    if shift is None:
                        continue
                    polygon = support_polygon([piece + shift for piece in convex], ctx.scene.TABLE,
                                              CONTACT_TOLERANCE)
                    if not supported_point(center + shift, polygon):
                        continue
                    counts['supported'] += 1
                    # A blocked body sweep cannot be repaired by another pad
                    # orientation. Prune it before expensive closure previews.
                    if plate:
                        blockers = swept_obstacles(convex, shift, sweep_obstacles)
                        if blockers:
                            counts['blocked'] += 1
                            for obstacle in blockers:
                                blocked = {'stage': 'target drag sweep', 'obstacle': obstacle,
                                           'edge': edge.point.tolist(), 'arm': candidate}
                                if blocked not in rejected:
                                    rejected.append(blocked)
                            continue
                    section, frame = exposed_section(world + shift, edge, [piece + shift for piece in convex])
                    virtual = replace(target, shape=section, offset=np.linalg.inv(pose) @ frame)
                    transform = pose @ virtual.offset
                    obstacles, supports = self.obstacles(candidate, virtual, transform)
                    stream = grasp_candidates(section, transform, hand, obstacles, preferred,
                        mass_center=center + shift, weight=self.grasp_weight(target),
                        approach_distance=APPROACH, contact_surface=section,
                        _placement_cache=self._grasp_placement_cache(
                            section, transform, hand, obstacles, APPROACH, section),
                        check_budget=lambda: remaining_wall(ctx))
                    first = next(stream, None)
                    if first is None:
                        continue
                    counts['geometric'] += 1
                    entry = (edge, candidate, fingertip, rule, hand, depth, float(exposed), shift,
                             polygon, virtual, supports, preferred, seed, stream)
                    quality = _grasp_area(section, transform, first.pose, self.hands[candidate],
                                          first.info['contact_gap'], surface=section)
                    heapq.heappush(queue, (max(0., HOLD_AREA - quality),
                                          first.cost, float(np.linalg.norm(shift)), next(serial),
                                          first, entry, 0))
            ctx.note('thin preparation preview', arm=candidate, edge_point=edge.point.tolist(),
                     landings=len(queue), candidates=search.checked, by_mode=preview_counts)
            while queue:
                _, _, _, _, grasp, entry, checked = heapq.heappop(queue)
                (edge, candidate, fingertip, rule, hand, depth, exposed, shift, polygon,
                 virtual, supports, preferred, seed, stream) = entry
                search.mode = ('tip' if fingertip else 'pad') + '/' + (rule or 'nearest')
                if plate:
                    # Preparation moves the object to a future edge pose.
                    # The ordinary final grasp can relocate the mobile base;
                    # rejecting it here using only the current fixed base
                    # would discard that supported route before any drag.
                    # Pressure IK and every executed path remain checked.
                    option = candidate, grasp, supports, self.model.q.copy(), None
                else:
                    option = next(self.options(virtual, [candidate], set(), approach=preferred,
                        then=(0., 0., height), mass_center=center + shift, candidate_budget=search,
                        ik_seed=seed, geometric={candidate: (supports, (grasp,))}), None)
                if option is not None:
                    hint = option[1].pose[:3, 0].tolist(), option[3].tolist()
                    result = self._drag_landing(target, candidate, edge, hand, shift, depth, exposed,
                        option[1].info['held_mm2'] * 1e-6, hint, prepared, search, patches, convex,
                        surface, pose, center, polygon, fingertip, rule, rejected, height=height)
                    if result is not None:
                        return result
                    # Pressure feasibility is independent of this landing's
                    # grip orientation. Do not repeat the same pressure RRT
                    # for each alternative grip at this landing.
                    continue
                # A failed head yields to all other modes before this stream's
                # next candidate. Retain the ordinary per-stream MAX_IK too.
                if checked + 1 < MAX_IK:
                    following = next(stream, None)
                    if following is not None:
                        quality = _grasp_area(virtual.shape, pose @ virtual.offset, following.pose,
                                              self.hands[candidate], following.info['contact_gap'],
                                              surface=virtual.shape)
                        heapq.heappush(queue, (max(0., HOLD_AREA - quality),
                                              following.cost, float(np.linalg.norm(shift)), next(serial),
                                              following, entry, checked + 1))
            if prepared is not None:
                hand = self.hands[candidate]
                if prepared['fingertip']:
                    hand = copy.copy(hand)
                    hand.hold_margin = 0.
                depth = overhang_depth(hand, transverse)
                depths = supported_overhangs(depth + CONTACT_TOLERANCE, maximum,
                                             BACKOFF_STEP, np.abs(world).max())
                if len(depths):
                    exposed = min(prepared['planned_overhang_m'], float(depths[-1]))
                    shift = landing(world, center, edge, exposed, prepared['landing_along_m'])
                    if shift is not None:
                        polygon = support_polygon([piece + shift for piece in convex], ctx.scene.TABLE,
                                                  CONTACT_TOLERANCE)
                        if supported_point(center + shift, polygon):
                            ctx.note('thin retained drag landing', arm=candidate,
                                      planned_overhang_m=exposed, grip_rejected=self.rejected)
                            result = self._drag_landing(target, candidate, edge, hand, shift, depth,
                                exposed, prepared['predicted_hold_area_mm2'] * 1e-6,
                                (prepared['grasp_approach'], prepared['grasp_joints']),
                                prepared, search, patches, convex, surface, pose, center, polygon,
                                prepared['fingertip'], prepared['landing_rule'], rejected, height=height)
                            if result is not None:
                                return result
        ctx.fail('no reachable pressure drag', Unreachable, kind='planning', rejected=rejected)

    def _drag_landing(self, target, candidate, edge, hand, shift, depth, planned_depth,
                      best_area, grasp_hint, prepared, search, patches, convex, surface,
                      pose, center, polygon, fingertip, rule, rejected, height=.1):
        ctx = self.ctx
        solver = self.solvers[candidate]
        solver.sync()
        world_objects = solver.world
        live = ctx.poses()
        obstacles = [(name, obj.fcl(live.get(obj.body, world_objects.poses[name])))
                     for name, obj in world_objects.objects.items()
                     if obj.body != target.body and name != 'table']
        obstacle = swept_obstacle(convex, shift, obstacles)
        if obstacle is not None:
            rejected.append({'stage': 'target drag sweep', 'obstacle': obstacle,
                             'edge': edge.point.tolist(), 'arm': candidate})
            return None
        touched = [*target.names, 'table']
        supported = [item for item in patches if supported_point(item[0] + shift, polygon)]
        for patch, points in patches:
            if supported_point(patch + shift, polygon):
                continue
            # A convex-piece vertex mean is not a support-aware pressure
            # point. Use the centroid of this piece's supported footprint;
            # the vertical collision ray below still checks solid material.
            footprint = support_polygon([points + shift], ctx.scene.TABLE, CONTACT_TOLERANCE)
            if len(footprint) >= 3:
                point = patch.copy()
                point[:2] = footprint.mean(0) - shift[:2]
                if supported_point(point + shift, polygon):
                    supported.append((point, points))
        supported.sort(key=lambda item: np.linalg.norm(item[0][:2] - center[:2]))
        for (patch, patch_points), flip in itertools.product(supported, (1., -1.)):
            distance = float(np.linalg.norm(shift))
            direction = shift / distance if distance else edge.outward
            budget_span = float(np.ptp(patch_points @ direction)) / 2
            if budget_span <= 0.:
                raise ValueError('pressure patch has no extent along drag')
            command = shift.copy()
            if self.thin_geometry(target).kind == 'plate' and distance > budget_span:
                command *= budget_span / distance
            above = patch.copy()
            if self.thin_geometry(target).kind == 'plate':
                above[2] = (transform_points(pose, target.points))[:, 2].max() + CONTACT_TOLERANCE
            origin = (above - pose[:3, 3]) @ pose[:3, :3]
            _, high = surface.interval(origin[None], pose[:3, :3].T @ [0., 0., 1.])
            if not np.isfinite(high[0]):
                continue
            top = above[2] + high[0]
            press = build_grasp_pose([0., 0., -1.], flip * edge.tangent,
                                     [patch[0], patch[1], top + hand.tip() - CONTACT_TOLERANCE / 2])
            pressure_goal = translated(press, command)
            def equivalent(old):
                return (same_pressure_contact(old, candidate, flip, edge.outward, origin,
                                              pose, target.points, CONTACT_TOLERANCE)
                        and np.linalg.norm(old['goal'] - pressure_goal[:3, 3]) <= CONTACT_TOLERANCE)
            if search is not None and any(equivalent(old) for old in search.failed_pressures):
                continue
            previous_empty = search.empty_pressures[-1] if search.empty_pressures else None
            repeated_empty = previous_empty is not None and equivalent(previous_empty)
            corrective = repeated_empty and previous_empty['repeated']
            solver.allow(touched)
            start = self.model.vector(ctx.spec.arms[candidate].finger_targets(0.))
            status, q_pre = self.model.ik(candidate, translated(press, [0., 0., .05]), start,
                                          world=True, lift=True, attempts=IK_ATTEMPTS)
            if status != 'Success':
                rejected.append({'stage': 'press pre-grasp IK', 'status': status,
                                 'edge': edge.point.tolist(), 'arm': candidate, 'flip': flip})
                solver.allow(touched, False)
                continue
            contact_started = False
            try:
                solver.close_gripper()
                moving = self.model.arm_indices(candidate) + self.model.lift_indices()
                if np.max(np.abs(q_pre - self.model.q)[moving]) > ARRIVAL_TOLERANCE:
                    if prepared is None:
                        solver.move_to_pose_with_RRTConnect(None, lift=True, goals=[q_pre])
                    else:
                        # Adjacent pressure segments remain above the
                        # object. Reuse a straight local approach rather
                        # than parameterizing nearly identical RRT goals.
                        solver.move_to_pose_with_screw(translated(press, [0., 0., .05]), lift=True)
                # The free approach can reach a different redundant
                # branch than a standalone contact-pose IK. Preflight
                # press, drag and retreat from the measured joints.
                solver.sync()
                q = self.model.q.copy()
                reachable = True
                end = translated(press, command)
                for stage, destination in [('press path', press), ('drag path', end),
                                           ('drag retreat', translated(end, [0., 0., .05]))]:
                    line = self.model.plan_screw(candidate, destination, q, world=True,
                                                 time_step=DT, lift=True)
                    if line['status'] != 'Success':
                        rejected.append({'stage': stage, 'status': line['status'],
                                         'details': {key: line[key] for key in ('reason', 'collisions')
                                                     if key in line},
                                         'edge': edge.point.tolist(), 'arm': candidate, 'flip': flip})
                        reachable = False
                        break
                    q[self.model.arm_indices(candidate)] = line['position'][-1]
                    q[self.model.lift_indices()] = line['lift'][-1]
                if not reachable:
                    continue
                contact_started = True
                before = ctx.poses()[target.body].copy()
                before_com = ctx.env.center_of_mass(target.body)[0].copy()
                if self.thin_geometry(target).kind == 'plate':
                    feedback = self._plate_pressure_drag(target, candidate, press, end, direction, float(np.linalg.norm(command)))
                elif corrective:
                    feedback = self._drag_with_load(target, candidate, press, end, direction, distance)
                else:
                    feedback = self._observe_pressure_drag(target, candidate, press, end, direction)
                after = ctx.poses()[target.body].copy()
                self.move_by_displacement(candidate, [0., 0., .05])
                ctx.hold(.2)
                final = ctx.poses()[target.body]
                points = transform_points(final, target.points)
                if points[:, 2].max() < ctx.scene.TABLE['height'] - CONTACT_TOLERANCE:
                    ctx.fail('thin object lost support', kind='check', object_before=pose.tolist(),
                             object_after=final.tolist(), highest_point_m=float(points[:, 2].max()),
                             table_height_m=ctx.scene.TABLE['height'], command_m=command.tolist())
                progress = float((ctx.env.center_of_mass(target.body)[0] - before_com) @ direction)
                motion = material_motion(target.points, pose, final)
                empty = motion <= CONTACT_TOLERANCE
                if empty:
                    contact = {'arm': candidate, 'flip': flip, 'edge': edge.outward.copy(),
                               'origin': origin.copy(), 'pose': final.copy(),
                               'goal': pressure_goal[:3, 3].copy(), 'repeated': repeated_empty}
                    ctx.note('thin pressure no progress', arm=candidate, progress_m=progress,
                             material_motion_m=motion, repeated_contact=repeated_empty, corrective=corrective,
                             contact_local_m=origin.tolist(), flip=flip, feedback=feedback)
                    search.empty_pressures.append(contact)
                    if corrective:
                        search.pressure_retries += 1
                        search.failed_pressures.append(contact)
                        if search.pressure_retries >= MAX_REGRASPS:
                            ctx.fail('thin pressure repeated empty contact', kind='check',
                                     attempts=search.pressure_retries, feedback=feedback)
                        continue
                else:
                    search.empty_pressures.clear()
                measured = float(edge.distances(points).max())
                result = {'arm': candidate, 'edge': edge, 'required_overhang_m': depth,
                          'landing_along_m': float((center + shift - edge.point) @ edge.tangent),
                          'planned_overhang_m': planned_depth,
                          'grasp_approach': grasp_hint[0], 'grasp_joints': grasp_hint[1],
                          'fingertip': fingertip, 'landing_rule': rule,
                          'predicted_hold_area_mm2': best_area * 1e6,
                          'pressure_point_before_drag_m': patch.tolist(),
                          'pressure_point_planned_end_m': (patch + shift).tolist(),
                          'landing_com_m': (center + shift).tolist(),
                          'landing_support_polygon_xy_m': polygon.tolist(),
                          'overhang_m': measured,
                          'object_pose': final.tolist(),
                          'pressure_feedback': feedback, 'progress_m': progress,
                          'empty_pressure': empty, 'repeated_empty_pressure': empty and repeated_empty,
                          'material_motion_m': motion,
                          'command_m': command.tolist(),
                          'remaining_command_m': distance,
                          'budget_span_m': budget_span,
                          'search_candidates': search.checked if search is not None else None,
                          'search_candidate_limit': search.limit if search is not None else None,
                          'object_displacement_m': (after[:3, 3] - before[:3, 3]).tolist()}
                ctx.note('thin object dragged', **{key: value for key, value in result.items() if key != 'edge'},
                         edge_point=edge.point.tolist(), outward=edge.outward.tolist())
                return result
            except PlanningFailed as error:
                rejected.append({'stage': 'motion', 'reason': error.reason})
                # An executed contact motion must go through the normal
                # finite retry flow, rather than reuse stale geometry.
                if contact_started:
                    raise
            finally:
                solver.allow(touched, False)
        return None

    def _recheck_pressure_landing(self, target, arm, edge, hand, goal_com, depth,
                                  planned_depth, hint, prepared, search, fingertip, rule, rejected, height):
        ctx = self.ctx
        pose = ctx.poses()[target.body]
        center = ctx.env.center_of_mass(target.body)[0]
        shift = goal_com - center
        shift[2] = 0.
        surface, patches, convex = self.pressure_geometry(target, pose)
        world = transform_points(pose, target.points)
        polygon = support_polygon([piece + shift for piece in convex], ctx.scene.TABLE, CONTACT_TOLERANCE)
        if not supported_point(center + shift, polygon):
            ctx.fail('thin pressure landing lost support', kind='planning')
        section, frame = exposed_section(world + shift, edge, [piece + shift for piece in convex])
        virtual = replace(target, shape=section, offset=np.linalg.inv(pose) @ frame)
        transform = pose @ virtual.offset
        obstacles, supports = self.obstacles(arm, virtual, transform)
        stream = grasp_candidates(section, transform, hand, obstacles, np.asarray(hint[0]),
            mass_center=center + shift, weight=self.grasp_weight(target),
            approach_distance=APPROACH, contact_surface=section,
            _placement_cache=self._grasp_placement_cache(
                section, transform, hand, obstacles, APPROACH, section),
            check_budget=lambda: remaining_wall(ctx))
        ctx.note('thin pressure local landing recheck', arm=arm, retries=search.pressure_retries,
                 search_candidates=search.checked)
        if self.thin_geometry(target).kind == 'plate':
            # Keep the already selected mobile-base landing. Rechecking all
            # fixed-base IK branches before every small physical stroke
            # repeats work which the final grasp must do at its live pose.
            grasp = next(stream, None)
            option = None if grasp is None else (None, grasp, None, np.asarray(hint[1]))
        else:
            option = next(self.options(virtual, [arm], set(), approach=np.asarray(hint[0]),
                then=(0., 0., height), mass_center=center + shift, candidate_budget=search,
                ik_seed=np.asarray(hint[1]), geometric={arm: (supports, stream)}), None)
        if option is None:
            ctx.fail('thin pressure landing no longer reachable', kind='planning')
        hint = option[1].pose[:3, 0].tolist(), option[3].tolist()
        result = self._drag_landing(target, arm, edge, hand, shift, depth, planned_depth,
            option[1].info['held_mm2'] * 1e-6, hint, prepared, search, patches, convex,
            surface, pose, center, polygon, fingertip, rule, rejected, height=height)
        if result is None:
            ctx.fail('thin pressure has no new contact', kind='planning')
        return result

    def _plate_pressure_drag(self, target, arm, press, end, direction, distance):
        """Stop a plate press on measured bearing, retaining its loaded drive target.

        A thick rigid plate can bear the fingertip before the penetrated contact
        goal reaches the ordinary free-space joint tolerance. Pressure contact
        terminates this approach; the unchanged solver still checks every path.
        """
        ctx, solver = self.ctx, self.solvers[arm]
        contacts = ctx.env.contacts
        load = PressureLoad(target.body, ctx.spec.arms[arm].finger_links,
                            contacts.dt, self.grasp_weight(target), lambda: contacts.time)
        feedback = {'mode': 'plate measured bearing', 'samples': [], 'corrections_m': []}
        with contacts.observe_impulses(load.record):
            path = solver._plan('plate press', lambda: self.model.plan_screw(
                arm, press, world=True, time_step=DT, lift=True))
            def bearing():
                sample = load.read()
                feedback['samples'].append({'stage': 'press', 'step': ctx.steps, **sample})
                return not sample['bearing']
            completed = solver.follow(path, monitor=bearing)
            if completed:
                ctx.fail('plate pressure has no bearing', kind='check', feedback=feedback)
            before = ctx.env.center_of_mass(target.body)[0].copy()
            # Contact may start above the intended loaded reference. Complete
            # the checked press drive without waiting for free-space joint
            # arrival against the rigid plate.
            self._pressure_correction(arm, press, command_only=True)
            # A broad plate adds table friction under the fingertip load.
            # Establish a finite preload before a tangential stroke; merely
            # exceeding the object's weight can still let the finger slide.
            for _ in range(MAX_REGRASPS):
                load.read()
                ctx.step()
                sample = load.read()
                feedback['samples'].append({'stage': 'preload', 'step': ctx.steps, **sample})
                if sample['normal_impulse_ns'] >= 4 * load.minimum * load.substeps:
                    break
                press = press.copy()
                press[2, 3] -= CONTACT_TOLERANCE / 2
                self._pressure_correction(arm, press, command_only=True)
                feedback['corrections_m'].append(CONTACT_TOLERANCE / 2)
            start = self._commanded_pressure_start(arm)
            # Continue from the loaded joint reference, not the deflected
            # measured joints, so tangential motion retains the contact load.
            origin = self.model.tcp(arm, q=start, world=True)
            destination = translated(origin, direction * distance)
            for attempt in range(MAX_REGRASPS + 1):
                path = solver._plan('plate drag', lambda: self.model.plan_screw(
                    arm, destination, start, world=True, time_step=DT, lift=True))
                tcp_before = ctx.tcp(arm)[:3, 3].copy()
                com_before = ctx.env.center_of_mass(target.body)[0].copy()
                stop = None
                applied = 0
                def progress():
                    nonlocal stop, applied
                    applied += 1
                    along = float((ctx.env.center_of_mass(target.body)[0] - before) @ direction)
                    slip = float(((ctx.tcp(arm)[:3, 3] - tcp_before) -
                                  (ctx.env.center_of_mass(target.body)[0] - com_before)) @ direction)
                    sample = load.read()
                    feedback['samples'].append({'stage': 'drag', 'step': ctx.steps,
                                                'object_along_m': along, 'slip_m': slip, **sample})
                    stop = ('target reached' if along >= distance - CONTACT_TOLERANCE else
                            'sliding' if slip > .01 else
                            'command completed' if applied >= len(path['position']) else None)
                    return stop is None
                solver.follow(path, monitor=progress)
                if stop != 'sliding' or attempt == MAX_REGRASPS:
                    feedback['stop'] = stop
                    break
                # Correct load at the live contact, keeping the horizontal
                # goal. This is a deliberate pressure command, not a relaxed
                # collision or joint-arrival tolerance.
                correction = self.model.tcp(arm, self._commanded_pressure_start(arm), world=True)
                correction[2, 3] -= CONTACT_TOLERANCE / 2
                self._pressure_correction(arm, correction, command_only=True)
                feedback['corrections_m'].append(CONTACT_TOLERANCE / 2)
                start = self._commanded_pressure_start(arm)
                destination[2, 3] = correction[2, 3]
        return feedback

    def _observe_pressure_drag(self, target, arm, press, end, direction):
        """Observe a first contact without mistaking preload release for failure.

        A newly encountered contact has not demonstrated a repeated empty
        action. Keep its original path; only an unchanged material state at
        the same contact enables the corrective protocol on its next attempt.
        """
        ctx, solver = self.ctx, self.solvers[arm]
        contacts = ctx.env.contacts
        load = PressureLoad(target.body, ctx.spec.arms[arm].finger_links,
                            contacts.dt, self.grasp_weight(target), lambda: contacts.time)
        feedback = {'mode': 'first contact observation', 'samples': [], 'corrections_m': []}
        with ctx.env.contacts.observe_impulses(load.record):
            solver.move_to_pose_with_screw(press, lift=True)
            load.read()  # the approach, rather than a fresh held interval
            before = ctx.env.center_of_mass(target.body)[0].copy()
            before_tcp = ctx.tcp(arm)[:3, 3].copy()
            path = solver._plan('screw', lambda: self.model.plan_screw(
                arm, end, world=True, time_step=DT, lift=True))
            def observe():
                feedback['samples'].append({'stage': 'drag', 'step': ctx.steps, **load.read(),
                    'object_along_m': float((ctx.env.center_of_mass(target.body)[0] - before) @ direction),
                    'tcp_along_m': float((ctx.tcp(arm)[:3, 3] - before_tcp) @ direction)})
                return True
            solver.follow(path, monitor=observe)
        feedback['stop'] = 'path completed'
        return feedback

    def _commanded_pressure_start(self, arm):
        ctx, side = self.ctx, self.ctx.spec.arms[arm]
        start = self.model.q.copy()
        robot = ctx.env.robot
        for names, indices in ((side.joints, self.model.arm_indices(arm)),
                               (ctx.spec.lift, self.model.lift_indices())):
            ids = [robot.joint_names.index(name) for name in names]
            start[indices] = robot.data.joint_pos_target[0, ids].cpu().numpy()
        return start

    def _pressure_correction(self, arm, target, command_only=False):
        """IK and checked joint interpolation resolve a sub-mm loaded correction.

        The screw planner's existing 1 mm arrival tolerance cannot express a
        correction smaller than its arrival band. Keep that tolerance intact.
        """
        solver = self.solvers[arm]
        def plan():
            from table_1000.experts.solver import VALIDATION_STEP
            start = self._commanded_pressure_start(arm)
            status, goal = self.model.ik(arm, target, start, world=True, lift=True,
                                         attempts=IK_ATTEMPTS)
            if status != 'Success':
                return {'status': status}
            indices, lift_ids = self.model.arm_indices(arm), self.model.lift_indices()
            checks, collisions = self.model.check_path(arm, np.array([start[indices], goal[indices]]),
                VALIDATION_STEP, np.array([start[lift_ids], goal[lift_ids]]))
            if collisions:
                return {'status': 'path collision', 'collisions': collisions}
            group = lift_ids + indices
            result = correction_path(start[group], goal[group], solver.planner.joint_vel_limits,
                                     solver.planner.joint_acc_limits)
            result['checks'] = checks
            return self.model._arm_path(arm, result, lift=True)
        path = solver._plan('pressure correction', plan)
        applied = 0
        def command_sent():
            nonlocal applied
            applied += 1
            return applied < len(path['position'])
        solver.follow(path, monitor=command_sent if command_only else None)

    def _drag_with_load(self, target, arm, press, end, direction, distance):
        """Keep measured bearing and the applied joint reference through a drag."""
        ctx, solver = self.ctx, self.solvers[arm]
        side = ctx.spec.arms[arm]
        contacts = ctx.env.contacts
        load = PressureLoad(target.body, side.finger_links, contacts.dt,
                            self.grasp_weight(target), lambda: contacts.time)
        feedback = {'samples': [], 'corrections_m': [], 'stop': 'no bearing'}
        initial_com = ctx.env.center_of_mass(target.body)[0].copy()
        goal_com = initial_com + direction * distance
        current_press = press.copy()
        compensated = 0.
        with ctx.env.contacts.observe_impulses(load.record):
            solver.move_to_pose_with_screw(current_press, lift=True)
            for attempt in range(MAX_REGRASPS + 1):
                load.read()
                ctx.step()
                reading = load.read()
                feedback['samples'].append({'stage': 'press', 'step': ctx.steps, **reading})
                if not reading['bearing']:
                    error = max(0., float(ctx.tcp(arm)[2, 3] - current_press[2, 3]))
                    correction = min(error, CONTACT_TOLERANCE / 2 - compensated)
                    if correction <= 0.:
                        return feedback
                    current_press[2, 3] -= correction
                    compensated += correction
                    feedback['corrections_m'].append(correction)
                    self._pressure_correction(arm, current_press)
                    load.read()
                    ctx.step()
                    reading = load.read()
                    feedback['samples'].append({'stage': 'repress', 'step': ctx.steps, **reading})
                    if not reading['bearing']:
                        continue
                pose = ctx.poses()[target.body]
                before_com = ctx.env.center_of_mass(target.body)[0].copy()
                before_tcp = ctx.tcp(arm)[:3, 3].copy()
                anchor = load.point(ctx.env.contacts.surface_contacts.values())
                if anchor is None:
                    anchor = before_com
                local_anchor = (anchor - pose[:3, 3]) @ pose[:3, :3]
                previous_com = before_com.copy()
                remaining = float((goal_com - before_com) @ direction)
                if remaining <= CONTACT_TOLERANCE:
                    feedback['stop'] = 'target reached'
                    return feedback
                drag_goal = end.copy()
                drag_goal[2, 3] -= compensated
                start = self._commanded_pressure_start(arm)
                path = solver._plan('screw', lambda: self.model.plan_screw(
                    arm, drag_goal, start, world=True, time_step=DT, lift=True))
                load.read()

                def monitor():
                    nonlocal previous_com
                    reading = load.read()
                    com = ctx.env.center_of_mass(target.body)[0]
                    live = ctx.poses()[target.body]
                    along = float((com - before_com) @ direction)
                    material = live[:3, :3] @ local_anchor + live[:3, 3]
                    material_along = float((material - anchor) @ direction)
                    tcp_along = float((ctx.tcp(arm)[:3, 3] - before_tcp) @ direction)
                    # Use observed control-interval motion. An instantaneous
                    # PhysX velocity spike is not a completed displacement.
                    predicted = along + float((com - previous_com) @ direction)
                    previous_com = com.copy()
                    stop = ('target reached' if predicted >= remaining - CONTACT_TOLERANCE else
                            'bearing lost' if not reading['bearing_now'] else
                            'object not following' if tcp_along - material_along > CONTACT_TOLERANCE else None)
                    feedback['samples'].append({'stage': 'drag', 'step': ctx.steps, **reading,
                        'object_along_m': along, 'material_along_m': material_along,
                        'tcp_along_m': tcp_along, 'predicted_object_along_m': predicted})
                    if stop is not None:
                        feedback['stop'] = stop
                    return stop is None

                if solver.follow(path, monitor=monitor):
                    feedback['stop'] = 'path completed'
                    return feedback
                if feedback['stop'] == 'target reached':
                    return feedback
                if attempt == MAX_REGRASPS:
                    return feedback
                # Repress at the live material point, before withdrawing or
                # rebuilding a landing. Retain the original absolute endpoint.
                live = ctx.poses()[target.body]
                material = live[:3, :3] @ local_anchor + live[:3, 3]
                current_press = self.model.tcp(arm, self._commanded_pressure_start(arm), world=True)
                current_press[:3, 3] += material - anchor - (ctx.tcp(arm)[:3, 3] - before_tcp)
                self._pressure_correction(arm, current_press)
                feedback['local_represses'] = attempt + 1
        return feedback

    def exposed_target(self, target, edge):
        """Observed exposed material of the whole held body's collision union."""
        pose = self.ctx.poses()[target.body]
        world = transform_points(pose, target.points)
        geometry = self.ctx.geometry[target.obj]
        body = target.body.split(':', 1)[1]
        pieces = [data @ (pose @ transform)[:3, :3].T + (pose @ transform)[:3, 3]
                  for name, _, transform, kind, data in geometry.shapes if name == body and kind == 'convex']
        shape, frame = exposed_section(world, edge, pieces)
        return replace(target, shape=shape, offset=np.linalg.inv(pose) @ frame)

    def _edge_grasp_available(self, target, prepared, height):
        exposed = self.exposed_target(target, prepared['edge'])
        previous = getattr(self, '_fingertip_candidates', False)
        self._fingertip_candidates = prepared['fingertip']
        try:
            if self.thin_geometry(target).kind == 'plate':
                # A supported plate need not be dragged all the way to the
                # arm's initial shoulder. Once real exposed material admits
                # a grip, ordinary grasp can relocate the mobile base.
                arm = prepared['arm']
                transform = self.ctx.poses()[target.body] @ exposed.offset
                obstacles, _ = self.obstacles(arm, exposed, transform)
                option = next(grasp_candidates(exposed.shape, transform, self.candidate_hand(exposed, arm),
                    obstacles, -prepared['edge'].outward, mass_center=self.grasp_center_of_mass(target, None),
                    weight=self.grasp_weight(target), approach_distance=APPROACH, contact_surface=exposed.shape,
                    _placement_cache=self._grasp_placement_cache(
                        exposed.shape, transform, self.candidate_hand(exposed, arm), obstacles, APPROACH, exposed.shape),
                    check_budget=lambda: remaining_wall(self.ctx)), None)
            else:
                option = next(self.options(exposed, [prepared['arm']], set(), approach=-prepared['edge'].outward,
                                           then=(0., 0., height)), None)
        finally:
            self._fingertip_candidates = previous
        available = option is not None
        self.ctx.note('thin measured edge grasp', available=available,
                      overhang_m=prepared['overhang_m'], rejected=self.rejected if not available else {})
        return available

    def grasp_thin(self, target, arm=None, height=.1, need=None, base=True, closure_only_record=False):
        """Prepare a thin target, then use the common grasp search and closure checks."""
        thin = self.thin_geometry(target)
        if thin is None:
            raise ValueError('grasp_thin requires a thin collision geometry')
        world = self.solvers[self.arms[0]].world
        target = replace(target, names=[name for name, obj in world.objects.items() if obj.body == target.body])
        self.ctx.note('thin object route', kind=thin.kind, thickness_m=thin.thickness,
                      contact_height_m=thin.contact_height, contact_band_m=thin.contact_band)
        if thin.kind == 'filament':
            try:
                return self.pinch_thin(target, arm, height, need, base, closure_only_record)
            except Unreachable as error:
                self.ctx.note('thin filament needs edge preparation', reason=error.reason, details=error.details)
        try:
            return self.grasp_at_edge(target, arm, height, need, base, closure_only_record)
        except Unreachable as error:
            self.ctx.note('thin edge route unreachable', reason=error.reason, details=error.details)
            if thin.kind == 'plate' and error.reason == 'no reachable pressure drag':
                blocked = error.details['rejected']
                if (isinstance(blocked, list)
                        and all(isinstance(item, dict) and item.get('stage') == 'target drag sweep'
                                for item in blocked)):
                    edges = []
                    for item in blocked:
                        edge = item['edge']
                        row = next((row for row in edges if row['edge'] == edge), None)
                        if row is None:
                            row = {'edge': edge, 'clear_objects': []}
                            edges.append(row)
                        obj = item['obstacle'].split(':', 1)[0]
                        if obj not in row['clear_objects']:
                            row['clear_objects'].append(obj)
                    self.ctx.fail('plate edge route blocked; clear the obstacles for one listed edge and retry pick',
                                  Unreachable, kind='planning', edges=edges, rejected=blocked)
            if thin.kind in ('filament', 'plate'):
                raise
        except ExpertFailed as error:
            if thin.kind != 'sheet' or error.reason != 'thin preparation search exhausted':
                raise
            # Retain edge-first ordering and its finite preparation allowance.
            # A checked fingertip hold is a distinct route, not a restart of
            # the exhausted pressure-drag search.
            self.ctx.note('thin sheet preparation exhausted; try fingertip pinch', details=error.details)
        return self.pinch_thin(target, arm, height, need, base, closure_only_record)

    def grasp_at_edge(self, target, arm, height, need, base, closure_only_record):
        """One edge preparation and grip, with finite retries of measured drag shortfall."""
        self.ctx.note('thin edge grasp', modes=['pad', 'tip'], landings=['nearest', 'shoulder'])
        prepared = self.slide_to_edge(target, arm, height)
        edge = prepared['edge']
        exposed = self.exposed_target(target, edge)
        if arm is None and self.thin_geometry(target).kind == 'plate':
            # Retain the pressure arm's selected edge branch. A second arm
            # reaching across it can close successfully but leave no room for
            # the supported extraction; the mobile base can still relocate.
            arm = prepared['arm']
        previous = getattr(self, '_fingertip_candidates', False)
        self._fingertip_candidates = prepared['fingertip']
        try:
            result = self.grasp(exposed, arm=arm, approach=-edge.outward, then=(0., 0., height),
                                need=need, base=base, close=self.close_thin,
                                closure_only_record=True)
        finally:
            self._fingertip_candidates = previous
        result['preparation'] = {**{key: value for key, value in prepared.items() if key != 'edge'},
                                 'edge_point': edge.point.tolist(), 'edge_outward': edge.outward.tolist()}
        if not closure_only_record:
            self.prepare_thin_lift(exposed, result)
            self.lift_grasp(exposed, result, PROBE_LIFT)
        return exposed, result

    def prepare_thin_lift(self, target, result):
        """Keep table support during an optional extraction towards the COM-clear pose.

        Ordinary and direct fingertip grips have no edge preparation. An
        optional waypoint already within the existing screw resolution is
        omitted, so measured COM clearance is recorded rather than guaranteed.
        The following vertical lift still measures real body rise.
        """
        prepared = result.get('preparation')
        if prepared is None:
            return
        outward = np.asarray(prepared['edge_outward'])
        point = np.asarray(prepared['edge_point'])
        center = self.ctx.env.center_of_mass(target.body)[0]
        # The hand's support collision allowance is negative penetration; it
        # is not a COM target. Aim across the real edge by roundoff only.
        roundoff = 8 * np.spacing(max(1., np.abs(center).max(), np.abs(point).max()))
        required = max(0., -float((center - point) @ outward) + roundoff)
        distance = required if required > CONTACT_TOLERANCE else 0.
        arm = result['arm']
        solver = self.solvers[arm]
        supports = list(result['supports'])
        if distance:
            solver.world.allow(target.names, supports)
            try:
                try:
                    self.move_by_displacement(arm, outward * distance)
                except PlanningFailed as error:
                    if error.reason != 'joint_limit' and not (error.reason == 'collision'
                            and self.thin_geometry(target).kind == 'plate'):
                        raise
                    # Extraction is an optional waypoint. Both fixed-torso
                    # and free-torso straight plans failed before execution;
                    # retain this grip for the ordinary vertical departure.
                    self.ctx.note('thin extraction omitted', requested_m=distance,
                                  reason=error.reason, details=error.details)
                    distance = 0.
            finally:
                solver.world.allow(target.names, supports, False)
                # A straight move consumes its departure permissions. The
                # vertical lift must also be allowed to leave this support.
                solver.allow(supports)
                self.held[arm]['supports'] = supports
        actual = self.ctx.env.center_of_mass(target.body)[0]
        self.ctx.note('thin supported extraction', required_m=required, command_m=distance,
                      com_outside_m=float((actual - point) @ outward))
