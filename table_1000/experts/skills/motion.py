"""Straight moves (dragging a jointed part), push and release."""

import numpy as np

from table_1000.experts.context import Unreachable, steps
from table_1000.experts.grasps import standoff, translated
from table_1000.experts.skills.common import (CANDIDATE_PLANS, IK_ATTEMPTS, RETREATS, ROOM_PLACES, SUPPORT_DISTANCE,
                                              collisions, normalized)
from table_1000.experts.solver import PlanningFailed, note_torso
from table_1000.experts.problems import decision, mark

PARTIALS = (.75, .5, .25)     # fractions of a straight move tried when all of it does not plan
SHORT_RETREAT = .02     # m back along the approach that takes the fingers off what they held
PUSH_GAP = .03          # m, push starts with the fingertips this far in front of the face
PUSH_TILTS = np.radians([0., 20., 40., 60.])   # push approaches, tilted down from the push direction


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
        if dragged:
            solver.carry(dragged, contacts + riders)
            solver.carry(riders, contacts)
        try:
            return self.straight(arm, displacement, least)
        finally:
            if dragged:
                solver.release(riders, contacts)
                solver.release(dragged, contacts + riders)
            if held:
                solver.allow(supports, False)
                held['supports'] = []

    def dragged(self, arm, drag):
        """``(own, riders, contacts)``: the planning objects of body ``drag`` (``(object, body)``) not
        yet held, every part of the other objects resting on or against it (a pen in a drawer rides
        along), and what they may touch besides each other and the rest of their objects: the
        objects within ``SUPPORT_DISTANCE`` of them."""
        from mplib.collision_detection import fcl
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

        objects = {name.split(':')[0] for name in touching(names, bodies, drag[0] + ':') if ':' in name}
        riders = [name for name in world.objects
                  if ':' in name and name.split(':')[0] in objects and not attached(name)]
        carried = names + riders
        bodies = [world.objects[name].fcl(world.poses[name]) for name in carried]
        return names, riders, [name for name in touching(carried, bodies, drag[0] + ':')
                               if name.split(':')[0] not in objects]

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
        if np.linalg.norm(displacement[:2]) > 1e-9 or displacement[2] <= 0.:
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

    @decision
    def push(self, target, direction, distance, arm=None):
        """Push ``target`` (:class:`Target`) by ``distance`` along world ``direction``; see the
        module docstring. Returns ``{'arm', 'tilt', 'moved'}`` (how far its body went)."""
        ctx = self.ctx
        direction = normalized(np.asarray(direction, dtype=float))
        transform = ctx.poses()[target.body] @ target.offset
        face = transform[:3, 3] + direction * float((target.shape.points() @ transform[:3, :3].T @ direction).min())
        level = normalized(direction - [0., 0., direction[2]])
        side = normalized(np.cross([0., 0., 1.], level))
        arms = [arm] if arm else [a for a in self.arms if a not in self.held]
        options = []
        for candidate in sorted(arms, key=lambda a: np.linalg.norm((self.shoulder(a) - face)[:2])):
            tip = self.hands[candidate].tip()
            for tilt in PUSH_TILTS:
                approach = normalized(level * np.cos(tilt) - np.array([0., 0., 1.]) * np.sin(tilt))
                for closing in (side, normalized(np.cross(approach, side))):
                    pose = np.eye(4)
                    pose[:3, :3] = np.c_[approach, closing, np.cross(approach, closing)]
                    pose[:3, 3] = face - approach * tip
                    options.append((candidate, tilt, pose))
        before = ctx.poses()[target.body][:3, 3]
        # Either arm with the torso lift where it is (all the way through the push), then moving it too;
        # none in reach with the other arm out: again with it in.
        tried, pushed, stowed, roomed = set(), False, False, False
        while not pushed:
            for lift, (index, (candidate, tilt, contact)) in ((lift, option) for lift in (False, True)
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
                    solver.move_to_pose_with_RRTConnect(None, lift=True, goals=[q_pre], attempts=CANDIDATE_PLANS)
                    solver.allow(touched)
                    solver.move_to_pose_with_screw(contact)
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
                elif not roomed and ctx.spec.base:
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
                    ctx.fail('no reachable push', Unreachable, target=target.names)
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

    def release(self, arm):
        """Open the fingers to the grasp's opening and back out along the approach (straight, else
        planned), or up."""
        held = self.held.pop(arm)
        solver = self.solvers[arm]
        solver.open_gripper(held['grasp']['open_gap'])
        for name in held['target'].names:
            solver.detach(name)
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
        try:
            solver.first_reachable(moves, lambda move: move())
        finally:
            solver.allow([*held['target'].names, *held['supports']], False)
