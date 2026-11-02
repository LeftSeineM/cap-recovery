"""Straight moves (dragging a jointed part), push and release."""

from table_1000.experts.world import geometry as world_geometry

from itertools import combinations

import numpy as np
from mplib.collision_detection import fcl

from table_1000.experts.context import Unreachable, steps
from table_1000.experts.skills.hold.frames import standoff, translated
from table_1000.experts.skills.common import (CANDIDATE_PLANS, IK_ATTEMPTS, RETREATS, ROOM_PLACES, SUPPORT_DISTANCE,
                                              collisions, normalized)
from table_1000.experts.solver import PlanningFailed, note_torso
from table_1000.experts.audit.problems import decision, mark

PARTIALS = (.75, .5, .25)     # fractions of a straight move tried when all of it does not plan
SHORT_RETREAT = .02     # m back along the approach that takes the fingers off what they held
PUSH_GAP = .03          # m, push starts with the fingertips this far in front of the face
PUSH_TILTS = np.radians([0., 20., 40., 60.])   # push approaches, tilted down from the push direction


class DrawerClosed(Exception):
    """The supported drawer reached its measured closed end during a push."""


class Motion:
    """The straight-move, push and release skills of :class:`~table_1000.experts.skills.Skills`."""

    def move_by_displacement(self, arm, displacement, drag=None, least=1.):
        """The TCP straight by world ``displacement`` (or a part of it down to ``least``, see
        :meth:`straight`); ``drag``: ``(object, body)`` moved through a joint by the held part.
        Supports of the held object may be touched in this move only. Returns the fraction moved."""
        solver = self.solvers[arm]
        displacement = np.asarray(displacement, dtype=float)
        held = self.held.get(arm)
        supports = held['supports'] if held else []
        dragged, riders, contacts = self.dragged(arm, drag) if drag else ([], [], [])
        previous_arrival = getattr(solver, 'constraint_arrival', None)
        solver.constraint_arrival = self.drawer_arrival(drag, displacement) if drag else None
        rider_permissions = []
        if dragged:
            solver.carry(dragged, contacts + riders)
            solver.carry(riders, contacts)
            for first, second in combinations(riders, 2):
                if frozenset((first, second)) not in solver.world.permissions:
                    solver.world.allow([first], [second])
                    rider_permissions.append((first, second))
        try:
            return self.straight(arm, displacement, least)
        finally:
            solver.constraint_arrival = previous_arrival
            if dragged:
                solver.release(riders, contacts)
                solver.release(dragged, contacts + riders)
                for first, second in rider_permissions:
                    solver.world.allow([first], [second], False)
            if held:
                if held['carried']:
                    solver.world.allow(held['carried'], supports, False)
                solver.allow(supports, False)
                held['supports'] = []

    def drawer_arrival(self, drag, displacement):
        """Measured slider goal of this move, without requiring an exact hand pose."""
        from table_1000.experts.skills.drawer.joints import CLOSED, OPEN_TOLERANCE, Joint, opening
        obj, body = drag
        geometry = self.ctx.geometry[obj]
        for name in geometry.joints:
            joint = Joint.of(geometry, name)
            if joint.kind != 'SLIDER' or joint.child != body:
                continue
            delta = float(displacement @ joint.direction(self.ctx.pose(obj, body)))
            goal = float(np.clip(opening(self.ctx, obj, name) + delta, 0., joint.limits[1] - joint.limits[0]))
            return (lambda: opening(self.ctx, obj, name) >= goal - OPEN_TOLERANCE) if delta > 0. else (
                lambda: opening(self.ctx, obj, name) <= goal + CLOSED)
        return None

    def dragged(self, arm, drag):
        """``(own, riders, contacts)``: the planning objects of body ``drag`` (``(object, body)``) not
        yet held, every part of objects supported inside it, including stacks
        reached through contacts with other riders, and their nearby supports. External
        neighbours above the dragged body's bottom remain obstacles even within
        ``SUPPORT_DISTANCE``."""
        solver = self.solvers[arm]
        solver.sync()
        world, key = solver.world, f'{drag[0]}:{drag[1]}'
        attached = world.planner.planning_world.is_object_attached
        names = [name for name, obj in world.objects.items() if obj.body == key and not attached(name)]
        bodies = [world.objects[name].fcl(world.poses[name]) for name in names]
        def touching(names, bodies, skip):
            return [name for name, obj in world.objects.items()
                    if name not in names and not attached(name) and not name.startswith(skip) and any(
                        fcl.distance(body, obj.fcl(world.poses[name])).min_distance < SUPPORT_DISTANCE
                        for body in bodies)]

        # A supporting object below the dragged body is not its payload. In
        # particular, pushing an item atop a book must not attach that book
        # (and then collide it with every neighbour) to the pushing hand.
        from table_1000.experts.skills import semantic
        pose = self.ctx.poses()[key]
        points = self.ctx.geometry[drag[0]].points(drag[1], drag[1])
        bottom = (points @ pose[:3, :3].T + pose[:3, 3])[:, 2].min()
        frontier, objects, outside = names, set(), set()
        while frontier:
            candidates = {name.split(':')[0] for name in touching(frontier, bodies, drag[0] + ':')
                          if ':' in name} - objects - outside
            nearby = {obj for obj in candidates
                      if world_geometry.object_points(self.ctx, obj)[:, 2].min() >= bottom - SUPPORT_DISTANCE}
            added = {obj for obj in nearby if semantic.supported_inside(self.ctx, obj, drag[0], drag[1])}
            objects.update(added)
            outside.update(nearby - added)
            frontier = [name for name in world.objects
                        if ':' in name and name.split(':')[0] in added and not attached(name)]
            bodies = [world.objects[name].fcl(world.poses[name]) for name in frontier]
        riders = [name for name in world.objects
                  if ':' in name and name.split(':')[0] in objects and not attached(name)]
        carried = names + riders
        bodies = [world.objects[name].fcl(world.poses[name]) for name in carried]
        return names, riders, [name for name in touching(carried, bodies, drag[0] + ':')
                               if name.split(':')[0] not in objects | outside]

    def straight(self, arm, displacement, least=1.):
        """Straight by ``displacement`` with the arm, else (``least`` < 1) by the largest of
        ``PARTIALS`` of it down to ``least`` that plans, else with the arm and the torso lift; straight
        up out of their reach (the arm may hit the body close to it), the torso lift rises, then
        the arm does the rest, straight or along a planned path. Returns the fraction moved."""
        solver, ctx = self.solvers[arm], self.ctx
        start = solver.tcp()

        try:
            solver.move_to_pose_with_screw(translated(start, displacement))
            return 1.
        except PlanningFailed as error:
            failed = error
        # Into the other arm, out where it was left: that arm in, and again.
        if self.stow(self.in_the_way(arm, failed)):
            try:
                solver.move_to_pose_with_screw(translated(start, displacement))
                return 1.
            except PlanningFailed as error:
                failed = error
        # Six arm joints alone may run into a limit along the line: part of it, else all of it with
        # the torso lift as well (which follows its path less closely).
        options = [(f, False) for f in PARTIALS if f >= least] + [(1., True)]
        # (A limit or the body, an elbow by the torso, may also be hit only near the end of the line.)
        if failed.reason in ('joint_limit', 'no_progress', 'iteration_limit', 'collision'):
            for fraction, lift in options:
                try:
                    solver.move_to_pose_with_screw(translated(start, displacement * fraction), lift=lift)
                    if fraction < 1.:
                        ctx.note('moved part way', fraction=fraction, reason=failed.reason)
                    return fraction
                except PlanningFailed:
                    pass
        if self.braced or np.linalg.norm(displacement[:2]) > 1e-9 or displacement[2] <= 0.:
            raise failed
        straight = {'reason': failed.reason, 'collisions': failed.details.get('plan', {}).get('collisions')}
        rise = float(displacement[2])
        goal = translated(start, displacement)
        # A tracking failure can follow real motion. The torso only fills the
        # remaining rise toward the original absolute goal.
        remaining = float(goal[2, 3] - solver.tcp()[2, 3])
        if remaining < .005:
            return 1.
        lift = ctx.env.qpos(ctx.spec.lift)[0]
        upper = self.model.limits[self.model.lift_indices(), 1]
        target = np.minimum(lift + remaining, upper)
        ctx.note('rise with the torso', **straight, remaining=remaining, lift=lift.tolist(), target=target.tolist())
        count = steps(float(np.max(target - lift)) / ctx.spec.lift_velocity_limit)
        for k in range(1, count + 1):
            ctx.step({'lift': (lift + (target - lift) * k / count)[None]})
        ctx.hold(.3)
        note_torso(ctx, [lift, target])
        rest = float(goal[2, 3] - solver.tcp()[2, 3])
        if rest < .005:
            return 1.
        up = goal
        held = self.held.get(arm)
        if held:
            solver.allow(held['supports'], False)
        for move in (solver.move_to_pose_with_screw, lambda pose: solver.move_to_pose_with_RRTConnect(pose, lift=True)):
            try:
                move(up)
                return 1.
            except PlanningFailed as error:
                reason, collisions = error.reason, error.details.get('plan', {}).get('collisions')
        ctx.note('rise short', rest=rest, reason=reason, collisions=collisions)
        return float(np.clip((solver.tcp()[2, 3] - start[2, 3]) / rise, 0., 1.))

    def push_contacts(self, target, direction, arms):
        """Measured face contacts shared by pushing and coupled brace preflight."""
        transform = self.ctx.poses()[target.body] @ target.offset
        face = transform[:3, 3] + direction * float((target.shape.points() @ transform[:3, :3].T @ direction).min())
        level = normalized(direction - [0., 0., direction[2]])
        side = normalized(np.cross([0., 0., 1.], level))
        for arm in sorted(arms, key=lambda a: np.linalg.norm((self.shoulder(a) - face)[:2])):
            for tilt in PUSH_TILTS:
                approach = normalized(level * np.cos(tilt) - np.array([0., 0., 1.]) * np.sin(tilt))
                for closing in (side, normalized(np.cross(approach, side))):
                    pose = np.eye(4)
                    pose[:3, :3] = np.c_[approach, closing, np.cross(approach, closing)]
                    pose[:3, 3] = face - approach * self.hands[arm].tip()
                    yield arm, tilt, pose

    @decision
    def push(self, target, direction, distance, arm=None):
        """Push ``target`` (:class:`Target`) by ``distance`` along world ``direction``; see the
        module docstring. Returns ``{'arm', 'tilt', 'moved'}`` (how far its body went)."""
        ctx = self.ctx
        direction = normalized(np.asarray(direction, dtype=float))
        arms = [arm] if arm else [a for a in self.arms if a not in self.held]
        options = list(self.push_contacts(target, direction, arms))
        before = ctx.poses()[target.body][:3, 3]
        # Either arm with the torso lift where it is (all the way through the push), then moving it too;
        # none in reach with the other arm out: again with it in.
        tried, pushed, stowed, roomed = set(), False, False, False
        while not pushed:
            for lift, (index, (candidate, tilt, contact)) in ((lift, option) for lift in ((False,) if self.braced else (False, True))
                                                              for option in enumerate(options)):
                # A rejected candidate may already have moved the part. Continue
                # from its measured face and complete only the remaining travel.
                shift = ctx.poses()[target.body][:3, 3] - before
                contact = translated(contact, shift)
                remaining = distance - float(shift @ direction)
                mark(self, 'candidate', f'{index}/{lift}', arm=candidate, contact=contact, tilt=tilt, lift=lift)
                if index in tried:
                    mark(self, 'already planned', outcome='not checked')
                    continue
                solver = self.solvers[candidate]
                start = self.model.vector(ctx.spec.arms[candidate].finger_targets(0.))
                seed = self.brace_seed
                if seed is not None:
                    indices = self.model.arm_indices(candidate)
                    start[indices] = seed[indices]
                mark(self, 'pre-push IK')
                status, q_pre = self.model.ik(candidate, translated(contact, -direction * PUSH_GAP), start, world=True,
                                              lift=lift, attempts=IK_ATTEMPTS)
                if status != 'Success':
                    mark(self, 'pre-push IK', outcome='rejected', status=status)
                    continue
                mark(self, 'end IK')
                status, _ = self.model.ik(candidate, translated(contact, direction * remaining), q_pre, world=True,
                                          lift=lift, attempts=IK_ATTEMPTS, check_collision=False)
                if status != 'Success':
                    mark(self, 'end IK', outcome='rejected', status=status)
                    continue
                mark(self, 'plan')
                tried.add(index)
                # The fingertips push the part and may touch the rest of its body (a drawer's front).
                touched = [name for name, obj in solver.world.objects.items() if obj.body == target.body] or target.names
                try:
                    solver.close_gripper()
                    # Contact can obstruct the companion arm's later withdrawal.
                    # Preserve the old pre-contact clearance for pushing only.
                    self.stow([a for a in self.arms if a != candidate], why='push before contact')
                    solver.move_to_pose_with_RRTConnect(None, lift=True, goals=[q_pre], attempts=CANDIDATE_PLANS)
                    solver.allow(touched)
                    solver.move_to_pose_with_screw(contact)
                    if self.braced:
                        self._push_braced(candidate, target, direction * remaining)
                    else:
                        self.move_by_displacement(candidate, direction * remaining,
                                                  drag=(target.obj, target.body.split(':', 1)[1]))
                    mark(self, 'plan', outcome='accepted')
                    pushed = True
                    break
                except PlanningFailed as error:
                    mark(self, 'plan', outcome='rejected', reason=error.reason)
                    ctx.note('next push', arm=candidate, tilt=float(tilt), reason=error.reason, **collisions(error))
                finally:
                    solver.allow(touched, False)
            if not pushed:
                if not stowed and self.stow():
                    tried, stowed = set(), True
                elif not roomed and ctx.spec.base and not self.braced:
                    # None in reach from here: the base moves to reach the first contacts (and the push
                    # from them), then again.
                    roomed = True
                    paths = {}
                    for candidate, _, contact in options:
                        paths.setdefault(candidate, [])
                        if len(paths[candidate]) < ROOM_PLACES:
                            paths[candidate].append([translated(contact, -direction * PUSH_GAP), contact,
                                                     translated(contact, direction * distance)])
                    self.make_room(paths, need=1, touch=[name for name, obj in self.solvers[candidate].world.objects.items()
                                                         if obj.body == target.body] or target.names)
                    tried, stowed = set(), True
                else:
                    ctx.fail('no reachable push', Unreachable, kind='planning', target=target.names)
        moved = float((ctx.poses()[target.body][:3, 3] - before) @ direction)
        # Back off: straight back or up, else along a planned path. The push is done either way.
        tcp = solver.tcp()
        moves = [lambda pose=pose: solver.move_to_pose_with_screw(pose)
                 for pose in [translated(tcp, -direction * d) for d in RETREATS] + [translated(tcp, [0., 0., d]) for d in RETREATS]]
        moves.append(lambda: solver.move_to_pose_with_RRTConnect(translated(tcp, -direction * RETREATS[0]), lift=True))
        try:
            solver.first_reachable(moves, lambda move: move())
        except PlanningFailed as error:
            ctx.note('no retreat', arm=candidate, reason=error.reason, **collisions(error))
        return {'arm': candidate, 'tilt': float(tilt), 'moved': moved}

    def _push_braced(self, arm, target, displacement):
        """Stop pushing a supported drawer at its measured closed end."""
        # semantic imports skills, whose Motion mixin is still being initialized at module load.
        from table_1000.experts.skills.drawer.joints import CLOSED, Joint, opening

        ctx = self.ctx
        child = target.body.split(':', 1)[1]
        matches = [Joint.of(ctx.geometry[target.obj], name) for name in ctx.geometry[target.obj].joints
                   if Joint.of(ctx.geometry[target.obj], name).child == child]
        joint = next((joint for joint in matches if joint.kind == 'SLIDER'), None)
        if joint is None:
            raise ValueError('braced push requires a slider child; use an unbraced push')

        def closing(context):
            if opening(context, target.obj, joint.name) <= CLOSED / 2:
                raise DrawerClosed

        # Stop before pressing the housing through the slider stop.
        stopped = False
        ctx.observers.append(closing)
        try:
            self.move_by_displacement(arm, displacement, drag=(target.obj, child))
        except DrawerClosed:
            stopped = True
        finally:
            ctx.observers.remove(closing)
        if stopped:
            ctx.step({arm: self.solvers[arm].qpos()[None]})

    def release(self, arm, upward=0., backoff=0., settle=0.):
        """Open the fingers to the grasp's opening and back out along the approach (straight, else
        planned), or up. ``upward`` > 0 restricts withdrawal to a vertical screw
        of that distance, with the torso lift held first, then free (insertion).
        ``backoff`` first moves straight back by this distance, before going up
        on a flat release whose opening and both segments were screened.
        ``settle`` waits after opening/detaching, before observing withdrawal
        geometry, for an intentional gravity release to leave the fingers."""
        held = self.held.pop(arm)
        solver = self.solvers[arm]
        solver.open_gripper(held['grasp']['open_gap'])
        carried = held['carried']
        riders = held['riders']
        for name in dict.fromkeys([*held['target'].names, *carried, *riders]):
            solver.detach(name)
        for pair in held['carry_permissions']:
            solver.world.allow(pair[:1], pair[1:] or pair[:1], False)
        if settle:
            self.ctx.hold(settle)
        solver.sync()
        tcp = solver.tcp()
        # Straight out (with the torso lift too, if the arm alone runs into a limit or the body at either distance),
        # else out along a planned path, else out just clear of the fingers' grip and up, else up.
        def out_and_up():
            solver.move_to_pose_with_screw(standoff(tcp, SHORT_RETREAT))
            solver.first_reachable([translated(solver.tcp(), [0., 0., d]) for d in RETREATS])

        moves = [lambda d=d, lift=lift: solver.move_to_pose_with_screw(standoff(tcp, d), lift=lift)
                 for lift in (False, True) for d in RETREATS]
        moves.append(lambda: solver.move_to_pose_with_RRTConnect(standoff(tcp, RETREATS[0]), lift=True))
        moves.append(out_and_up)
        moves += [lambda d=d: solver.move_to_pose_with_screw(translated(tcp, [0., 0., d])) for d in RETREATS]
        if upward:
            moves = [lambda lift=lift: solver.move_to_pose_with_screw(translated(tcp, [0., 0., upward]), lift=lift)
                     for lift in (False, True)]
        try:
            if backoff:
                solver.first_reachable([False, True], lambda lift: solver.move_to_pose_with_screw(
                    standoff(tcp, backoff), lift=lift))
                backed = solver.tcp()
                moves = [lambda lift=lift: solver.move_to_pose_with_screw(
                    translated(backed, [0., 0., upward]), lift=lift) for lift in (False, True)]
                # A low shelf may leave the full withdrawal height at a torso
                # or wrist joint limit. Keep the requested height first, then
                # try shorter exits from the measured backoff pose; collision
                # checking still rejects an exit that does not clear the
                # opening.
                moves += [lambda height=upward * fraction, lift=lift:
                          solver.move_to_pose_with_screw(translated(backed, [0., 0., height]), lift=lift)
                          for fraction in (.5, .25) for lift in (False, True)]
            solver.first_reachable(moves, lambda move: move())
        finally:
            # Part grasps can grant approach contacts to adjacent pieces or
            # contents too. Their finger permissions end with this hold.
            solver.allow(list(dict.fromkeys([*held['target'].names, *carried, *riders, *held['supports']])), False)
