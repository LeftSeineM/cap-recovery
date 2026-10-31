"""Motion primitives of one arm, after ManiSkill3's ``TwoFingerGripperMotionPlanningSolver``.

Free-space moves plan with RRTConnect to an IK goal, contact moves with a
straight screw path (both MPlib, see :class:`~table_1000.robots.kinematics.RobotModel`),
and a planned path is sent as one joint-position action per control step. The
planning world holds the robot and whatever the expert adds (a table box,
asset objects from their collision geometry); objects follow their live poses
before every plan. Contact with an object is planned only where the expert
allows it (:meth:`MotionSolver.allow`), and held objects move with the hand
(:meth:`MotionSolver.attach`, :meth:`MotionSolver.carry`). A plan that cannot be
made raises :class:`PlanningFailed` with the reason (``IK Failed``,
``RRTConnect ...``, ``joint_limit``, ``collision``, ``path collision``,
``tracking``).
"""

from contextlib import nullcontext
import time

from table_1000.common.timing import timed

import numpy as np

from table_1000.experts.context import DT, ExpertFailed, remaining_wall, steps
from table_1000.planning.world import PlanningScene, asset_objects, box_object, pose

# Free-space (RRTConnect) plans keep this clearance from padded objects (its edges are checked
# coarsely); contact moves use the bare geometry. Objects the hand touches on purpose are never grown.
PADDING = .01
# RRTConnect checks its edges coarsely; its time-parameterized path is re-checked
# at this joint step and re-planned if any check collides.
VALIDATION_STEP = .01
ARRIVAL_TOLERANCE = .01   # rad, final joint error before a move counts as done
SETTLE_LIMIT = .5         # s of extra holding allowed to reach it
GRIPPER_TIME = .6         # s for the fingers to close or open
OPEN_GAP = .08            # m
GRIPPER_THERE = .002     # m from a commanded opening the fingers count as there
TORSO_STILL = .001        # m of torso lift travel below which a path leaves it where it was
LOCAL_LIFT = .15          # m: plan small shared-lift changes before retracting free arms
# Fingers starting within the padding of an object they are not meant to touch
# (a hand that just let go of a handle) back off
# straight first, along the approach or up, by the first of DEPARTS that leaves the padding.
DEPARTS = (.03, .06)      # m


class PlanningFailed(ExpertFailed):
    pass


def note_torso(ctx, path):
    """Note a move of the torso lift along ``path`` (rows of its joints) where it moved at all: its
    travel (m, summed over the path) and where it ended."""
    path = np.asarray(path, dtype=float)
    travel = float(np.abs(np.diff(path, axis=0)).sum())
    if travel > TORSO_STILL:
        ctx.note('torso moved', travel=travel, lift=path[-1].tolist())


class MotionSolver:
    def __init__(self, ctx, arm):
        self.ctx, self.model = ctx, ctx.model
        self.arm = ctx.spec.arms[arm]
        self.planner = self.model.planners[arm]
        # The planner works in the base frame (see table_1000.robots.kinematics).
        self.world = PlanningScene(self.planner, np.linalg.inv(self.model.base))
        self.log = []
        self.timings = {'planning_s': 0., 'executing_s': 0.}
        self.attachments = {}
        self.overlaps = []      # pairs allowed for the plan being made (see _plan)
        self.lock_lift = False  # a second hand is supporting an external body

    # -- planning world --------------------------------------------------------------
    def add_box(self, name, size, transform):
        """A static box (e.g. the table top) of ``size`` centred on ``transform``."""
        self.world.add(*box_object(name, size, transform))

    def add_table(self):
        """The table's top (``table``) and legs (``table.leg1`` ...), as the simulation builds them."""
        table = self.ctx.scene.TABLE
        (width, depth), (cx, cy), leg = table['size'], table['center'], table['leg_size']
        under = table['height'] - table['top_thickness']
        top = np.eye(4)
        top[:3, 3] = [cx, cy, table['height'] - table['top_thickness'] / 2]
        self.add_box('table', (width, depth, table['top_thickness']), top)
        for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
            post = np.eye(4)
            post[:3, 3] = [cx + sx * (width - leg) / 2, cy + sy * (depth - leg) / 2, under / 2]
            self.add_box(f'table.leg{i + 1}', (leg, leg, under), post)

    def add_object(self, name, group, padded=lambda part: True, merged=lambda part: False):
        """Planning objects of asset object ``name`` from its collision geometry: ``group(body, part)``
        returns ``(planning name, frame body)`` (see :func:`~table_1000.planning.world.asset_objects`);
        planning names get the prefix ``<name>:``."""
        objects = asset_objects(self.ctx.geometry[name], f'{name}:', group, padded, merged)
        poses = self.ctx.poses()
        for obj in objects:
            self.world.add(obj, poses[obj.body])
        return objects

    @timed('world_update')
    def sync(self):
        """Move planning objects to their live poses (and all of them where the base moved)."""
        # Each arm has a complete robot in its own world. The other hand's
        # declared hold contacts remain legitimate when this arm plans too.
        # Borrow only existing finger/attachment declarations; never suppress
        # contacts with unrelated objects or links, and revoke on release.
        shared = set()
        skills = getattr(self.ctx, '_skills', None)
        for arm in getattr(skills, 'held', {}):
            if arm == self.arm.name:
                continue
            other = self.ctx.solvers[arm]
            names = other.attachments.keys() & self.world.objects.keys()
            fingers = set(other.arm.finger_links)
            shared.update(pair for pair in other.world.permissions if pair & names and pair & fingers)
        self.world.set_shared_permissions(shared)
        frame = np.linalg.inv(self.model.base)
        poses = self.ctx.poses()
        if not np.allclose(frame, self.world.frame, atol=1e-6):
            self.world.set_frame(frame, poses)
        else:
            self.world.update(poses)

    def allow(self, names, allowed=True):
        """Permit (or revoke) planned contact between the fingers and planning objects ``names``."""
        self.world.allow(names, self.arm.finger_links, allowed)

    def attach(self, name, object_world):
        """Carry planning object ``name`` with the TCP at its measured pose relative to it;
        returns that relative pose."""
        relative = np.linalg.inv(self.tcp()) @ object_world
        self.attach_at(name, relative)
        return relative

    def attach_at(self, name, relative):
        """Carry planning object ``name`` with the TCP at pose ``relative`` to it."""
        self.planner.planning_world.attach_object(
            name, self.planner.robot.get_name(), self.planner.link_name_2_idx[self.arm.tcp_link], pose(relative),
            list(self.arm.finger_links))
        self.attachments[name] = np.asarray(relative).copy()
        self._restore_permissions(name)

    def detach(self, name):
        self.attachments.pop(name)
        self.planner.planning_world.detach_object(name, False)
        self._restore_permissions(name)

    def _restore_permissions(self, name):
        # MPlib recreates ACM entries when attaching or detaching. Keep the
        # scene's declarations until their owner revokes them.
        others = {other for pair in self.world.permissions if name in pair for other in pair if other != name}
        if others:
            self.planner.planning_world.get_allowed_collision_matrix().set_entry([name], sorted(others), True)

    def carry(self, names, contacts):
        """Attach planning objects moved through a joint (a drawer pulled by its handle) so they
        move with the TCP; each may touch ``contacts``."""
        poses = self.ctx.poses()
        for name in names:
            self.attach(name, poses[self.world.objects[name].body])
        self.world.allow(names, contacts)

    def release(self, names, contacts):
        for name in names:
            self.detach(name)
        self.world.allow(names, contacts, False)
        self.sync()

    # -- state ---------------------------------------------------------------------------
    def qpos(self, lift=False):
        return self.ctx.env.qpos([*self.arm.joints, *(self.ctx.spec.lift if lift else ())])[0]

    def tcp(self):
        return self.model.tcp(self.arm.name, world=True)

    def finger_gap(self):
        return self.arm.gap(self.ctx.env.qpos(self.arm.fingers)[0])

    # -- planning ------------------------------------------------------------------------
    @timed('motion_plan')
    def _plan(self, kind, function, goals=(), problem=None):
        remaining_wall(self.ctx)
        self.world.set_padding(PADDING if kind == 'rrt' else 0.)
        self.sync()
        # Find overlaps caused by padding at the endpoints. RRT uses the affected
        # objects' bare geometry throughout its search and checks the final route
        # without implicit contact exemptions. Straight moves retain their existing
        # robot-link padding exemptions (an elbow by the torso after a pull).
        robot = self.planner.robot.get_name()
        overlaps = [(link, other if name == robot else name)
                    for q in [self.model.q, *goals]
                    for link, name, other in self.model.collisions(self.arm.name, q)
                    if kind == 'rrt' or name == robot]
        overlaps = list(dict.fromkeys(overlaps))
        overlaps = [pair for pair in overlaps if frozenset(pair) not in self.world.permissions]
        if kind == 'rrt':
            # Near an endpoint, use the object's real geometry instead of ignoring
            # its collisions with a robot link along the whole path.
            names = {other for _, other in overlaps if other in self.world.objects}
            padding = self.world.padding
            self.world.padding = 0.
            for name in names:
                self.world.set_pose(name, self.world.poses[name])
            self.world.padding = padding
            overlaps = [pair for pair in overlaps if pair[1] not in names]
        self.overlaps = overlaps
        for link, other in overlaps:
            self.world.allow([link], [other])
        adapter = self.planner.planner
        before = adapter.before_simplify
        if kind == 'rrt':
            def prepare_simplify():
                # Native search needs temporary padding contacts; new shortcuts
                # must use the bare geometry and the permanent permissions only.
                for link, other in self.overlaps:
                    self.world.allow([link], [other], False)
                self.overlaps = []
                self.world.set_padding(0.)
            adapter.before_simplify = prepare_simplify
        started = time.perf_counter()
        record = getattr(getattr(self.ctx, 'trace', None), 'problems', None)
        capture = (record.operation('motion', self.arm.name, dict(kind=kind, start_allowed=overlaps, **problem))
                   if record else nullcontext(None))
        try:
            with capture as problem_entry:
                result = function()
                if problem_entry is not None:
                    from table_1000.experts.audit.problems import summary
                    problem_entry['result'] = summary(result)
        finally:
            adapter.before_simplify = before
            for link, other in self.overlaps:
                self.world.allow([link], [other], False)
            self.overlaps = []
            self.world.set_padding(0.)
        elapsed = time.perf_counter() - started
        self.timings['planning_s'] += elapsed
        entry = {'subtask': self.ctx.current, 'step': self.ctx.steps, 'kind': kind, 'status': result['status'],
                 'planning_s': elapsed}
        if overlaps:
            entry['start_allowed'] = overlaps
        self.log.append(entry)
        if record:
            entry['problem'] = problem_entry['id']
        if result['status'] != 'Success':
            entry.update({key: value for key, value in result.items()
                          if key not in ('position', 'velocity', 'acceleration', 'time')})
            raise PlanningFailed(self.ctx.current, result.get('reason', result['status']), plan=entry)
        entry['duration_s'] = float(result['duration'])
        entry['joint_length_rad'] = float(np.linalg.norm(np.diff(result['position'], axis=0), axis=1).sum())
        return result

    def check_geometry(self, rows, step=VALIDATION_STEP):
        for link, other in self.overlaps:
            self.world.allow([link], [other], False)
        self.overlaps = []
        self.world.set_padding(0.)
        count = len(self.model.spec.lift)
        return self.model.check_path(self.arm.name, np.asarray(rows)[:, count:], step,
                                     np.asarray(rows)[:, :count])[1]

    def _validated(self, result, step=VALIDATION_STEP):
        """Fail ``result`` if its path collides anywhere at VALIDATION_STEP."""
        if result['status'] != 'Success':
            return result
        # On the bare geometry, with nothing allowed for the start (check_path tolerates the
        # contacts the path starts in until it leaves them).
        for link, other in self.overlaps:
            self.world.allow([link], [other], False)
        self.overlaps = []
        self.world.set_padding(0.)
        checks, pairs = self.model.check_path(self.arm.name, result['position'], step, result.get('lift'))
        if pairs:
            return {'status': 'path collision', 'reason': 'path collision', 'collisions': pairs}
        return result

    def move_to_pose_with_RRTConnect(self, target, planning_time=3., attempts=3, lift=False, goals=None, retime=None,
                                    local_lift=False):
        """Plan to an IK goal of world TCP pose ``target`` (or to ``goals``, full configurations of
        the planning model) with RRTConnect and follow it; an approximate or colliding path is
        re-planned up to ``attempts`` times, each with twice the time of the one before. ``lift``
        ``retime`` optionally transforms the validated trajectory before execution.
        ``lift`` lets the torso lift move as well: where the goal leaves it where it is (or ``target`` has
        an IK solution that does), the plan holds it first, once, and frees it only if that fails.
        ``local_lift`` keeps the other arm out for small lift changes after a missed-grasp retreat;
        other approaches retain the ready posture used by contact preparation."""
        skill = getattr(self.ctx, '_skills', None)
        lift = lift and not self.lock_lift
        here = self.model.q[self.model.lift_indices()] if lift else None
        large = lift and goals is not None and all(
            np.max(np.abs(np.asarray(goal)[self.model.lift_indices()] - here)) > LOCAL_LIFT for goal in goals)
        clear_first = lift and (not local_lift or large)
        if clear_first and skill is not None:
            # The torso moves both arms. Clear its free arm before this arm
            # approaches contact, while an escape is still possible.
            skill.stow([arm for arm in skill.arms if arm != self.arm.name], why='shared torso lift')
        self.depart()
        free = [lift]
        if lift:
            here = self.model.q[self.model.lift_indices()]
            if goals is None or all(np.allclose(np.asarray(goal)[self.model.lift_indices()], here, atol=1e-4)
                                    for goal in goals):
                free = [False, True]
        timing = {'retime': retime} if retime is not None else {}
        for moving in free[:-1]:
            try:
                return self._rrt(target, planning_time, 1, moving, goals, **timing)
            except PlanningFailed as error:
                if not error.reason.startswith(('RRTConnect', 'path collision', 'IK Failed')):
                    raise
        try:
            self._rrt(target, planning_time, attempts, free[-1], goals, **timing)
        except PlanningFailed:
            # The ordinary full-robot collision/path checks also cover the
            # other arm as the torso moves. Retract only after that plan fails.
            if (not lift or clear_first or skill is None or not skill.stow(
                    [arm for arm in skill.arms if arm != self.arm.name], why='shared torso path blocked')):
                raise
            self._rrt(target, planning_time, attempts, free[-1], goals, **timing)

    @timed('departure_check')
    def touching(self):
        """``[(finger or TCP link, object)]`` within the free-space padding of each other now, not allowed."""
        self.world.set_padding(PADDING)
        self.sync()
        try:
            robot, hand = self.planner.robot.get_name(), {self.arm.tcp_link, *self.arm.finger_links}
            return [(link, name) for link, name, _ in self.model.collisions(self.arm.name, self.model.q)
                    if name != robot and link in hand and frozenset((link, name)) not in self.world.permissions]
        finally:
            self.world.set_padding(0.)

    def depart(self):
        """Back the hand straight off what it touches unasked for (see DEPARTS), where it does."""
        from table_1000.experts.skills.hold.frames import standoff, translated
        touched = self.touching()
        if not touched:
            return
        tcp = self.tcp()
        for target in [standoff(tcp, d) for d in DEPARTS] + [translated(tcp, [0., 0., d]) for d in DEPARTS]:
            try:
                self.move_to_pose_with_screw(target)
            except PlanningFailed:
                continue
            if not self.touching():
                self.ctx.note('departed', arm=self.arm.name, touched=touched)
                return

    def _rrt(self, target, planning_time, attempts, lift, goals, retime=None):
        lift = lift and not self.lock_lift
        for attempt in range(attempts):
            remaining = remaining_wall(self.ctx)
            budget = planning_time * 2 ** attempt
            if remaining is not None:
                budget = min(budget, remaining)
            try:
                result = self._plan('rrt', lambda: self._validated(self.model.plan_rrt(
                    self.arm.name, target, world=True, time_step=DT, planning_time=budget,
                    lift=lift, goals=goals, check=self.check_geometry)), goals=list(goals or []),
                    problem=dict(target=target, world=True, time_step=DT, planning_time=budget,
                                 lift=lift, goals=goals, attempt=attempt, attempts=attempts, validation_step=VALIDATION_STEP))
                break
            except PlanningFailed as error:
                if not error.reason.startswith(('RRTConnect', 'path collision')) or attempt == attempts - 1:
                    raise
        self.follow(retime(result) if retime is not None else result)

    def move_to_pose_with_screw(self, target, lift=False):
        """Move the TCP along a straight screw path to world pose ``target``; ``lift`` moves the
        torso lift as well."""
        lift = lift and not self.lock_lift
        self.follow(self._plan('screw', lambda: self.model.plan_screw(self.arm.name, target, world=True,
                                                                      time_step=DT, lift=lift),
                               problem=dict(target=target, world=True, time_step=DT, lift=lift, attempts=1)))

    def first_reachable(self, targets, move=None):
        """Move to the first of ``targets`` that plans (``move``: default the screw move); a plan
        failing for lack of a path tries the next, any other failure propagates."""
        move = move or self.move_to_pose_with_screw
        for index, target in enumerate(targets):
            try:
                return move(target)
            except PlanningFailed as error:
                if index == len(targets) - 1 or not error.reason.startswith(
                        ('IK Failed', 'joint_limit', 'collision', 'path collision', 'RRTConnect', 'no_progress',
                         'iteration_limit', 'time_parameterization')):
                    raise

    # -- execution -----------------------------------------------------------------------
    def follow(self, result, monitor=None):
        """Send the path one waypoint per control step (with the torso lift's, if it moves), then
        hold its end until arrival. An optional monitor stops before the next
        waypoint when it returns false, retaining the last applied command."""
        started = time.perf_counter()
        positions = np.asarray(result['position'])
        lift = result.get('lift')
        worst, held = 0., 0
        final = np.r_[positions[-1], lift[-1]] if lift is not None else positions[-1]
        try:
            for index, target in enumerate(positions):
                action = {self.arm.name: target[None]}
                if lift is not None:
                    action['lift'] = lift[index][None]
                self.ctx.step(action)
                worst = max(worst, float(np.abs(self.qpos() - target).max()))
                if monitor is not None and not monitor():
                    final = np.r_[target, lift[index]] if lift is not None else target
                    self.log[-1].update(max_tracking_error_rad=worst, stopped_waypoint=index)
                    self.timings['executing_s'] += time.perf_counter() - started
                    if lift is not None:
                        note_torso(self.ctx, lift[:index + 1])
                    return False
            if lift is not None:
                positions = np.c_[positions, lift]
            final, held = positions[-1], 0
            while np.abs(self.qpos(lift is not None) - final).max() > ARRIVAL_TOLERANCE:
                if held * DT >= SETTLE_LIMIT:
                    raise PlanningFailed(self.ctx.current, 'tracking',
                                         final_error_rad=float(np.abs(self.qpos(lift is not None) - final).max()))
                self.ctx.step()
                held += 1
                if monitor is not None and not monitor():
                    self.log[-1].update(max_tracking_error_rad=worst, stopped_waypoint=len(positions), settle_steps=held)
                    self.timings['executing_s'] += time.perf_counter() - started
                    if lift is not None:
                        note_torso(self.ctx, lift)
                    return False
            self.log[-1].update(max_tracking_error_rad=worst, settle_steps=held)
            self.timings['executing_s'] += time.perf_counter() - started
            if lift is not None:
                note_torso(self.ctx, lift)
            return True
        finally:
            measured = self.qpos(lift is not None)
            self._tracking(max(worst, float(np.abs(measured[:len(target)] - target).max())), held,
                           float(np.abs(measured - final).max()), time.perf_counter() - started)

    def _tracking(self, worst, held, final, elapsed):
        record = getattr(getattr(self.ctx, 'trace', None), 'problems', None)
        if record:
            record.operations[self.log[-1]['problem']]['execution'] = dict(
                max_tracking_error_rad=worst, settle_steps=held, final_error_rad=final, wall_s=elapsed)

    def gripper(self, gap, seconds=GRIPPER_TIME):
        """Command the pad gap ``gap`` for ``seconds``; opening to where the fingers are (within
        GRIPPER_THERE) takes no time."""
        gap = float(np.clip(gap, 0., self.arm.max_gap))
        if gap > 0. and abs(self.finger_gap() - gap) < GRIPPER_THERE:
            self.ctx.step({f'{self.arm.name}_gripper': [[gap]]})
            return
        for _ in range(steps(seconds)):
            self.ctx.step({f'{self.arm.name}_gripper': [[gap]]})

    def open_gripper(self, gap=OPEN_GAP):
        self.gripper(gap)

    def close_gripper(self):
        self.gripper(0.)
