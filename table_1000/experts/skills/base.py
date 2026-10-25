"""The base: driving it, and choosing where it stands so that poses are reached; the free arms back home."""

import os
import time

import numpy as np
from scipy.spatial.transform import Rotation

from table_1000.experts.context import steps
from table_1000.experts.skills.common import IK_ATTEMPTS, collisions
from table_1000.experts.solver import PlanningFailed
from table_1000.experts.problems import decision, mark

# The base drives at most BASE_SPEED (m/s) and turns at most BASE_TURN (rad/s), proportionally to the
# error (BASE_GAIN) until within BASE_TOLERANCE (m, rad), then settles for BASE_SETTLE s.
BASE_SPEED, BASE_TURN, BASE_GAIN = .15, .4, 2.
READY_TOLERANCE = .05   # rad from its start pose an arm counts as there
BASE_TOLERANCE, BASE_SETTLE = (.005, np.radians(1.)), .5
# make_room samples BASE_SAMPLES poses of the base BASE_DISTANCES (m) from the poses to reach, facing
# them within BASE_FACING (rad); in the order the base gets there (turn, drive straight, turn), at
# most BASE_CHECKS (BASE_CHECKS[1] while none reaches any) of those the robot stands clear at and its
# arms may reach, each at least
# BASE_SPREAD (m or rad) from those checked before, are checked by IK (BASE_IK_ATTEMPTS restarts,
# the torso lift free) for every joint BASE_MARGIN (rad) inside its limits (where none is, the first
# BASE_MARGIN_LOW inside them, with one pose to reach). The way there is
# checked every BASE_PATH_STEP (m, rad).
BASE_SAMPLES, BASE_CHECKS = 500, (20, 60)
BASE_DISTANCES, BASE_FACING = (.3, 1.), np.radians(60.)
BASE_MARGIN, BASE_MARGIN_LOW = .25, .05
BASE_IK_ATTEMPTS = 5
BASE_PATH_STEP = (.05, np.radians(10.))
BASE_SPREAD = (.05, np.radians(10.))
REACH_SAMPLES = 500     # random arm poses whose TCP distance from the shoulder bounds the arm's reach


def wrapped(angle):
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def drive(start, goal):
    """``(first turn, distance, last turn)`` taking the base from ``start`` to ``goal`` (x, y, yaw):
    turned in place, straight forward or back (whichever turns less), turned in place."""
    delta = np.asarray(goal[:2], dtype=float) - start[:2]
    distance = float(np.linalg.norm(delta))
    if distance < BASE_TOLERANCE[0]:
        return 0., 0., wrapped(goal[2] - start[2])
    first = wrapped(np.arctan2(delta[1], delta[0]) - start[2])
    if abs(first) > np.pi / 2:
        first, distance = wrapped(first + np.pi), -distance
    return first, distance, wrapped(goal[2] - start[2] - first)


def drive_time(start, goal):
    first, distance, last = drive(start, goal)
    return abs(distance) / BASE_SPEED + (abs(first) + abs(last)) / BASE_TURN


def drive_path(start, goal):
    """Base poses along :func:`drive` from ``start`` to ``goal``, ``BASE_PATH_STEP`` apart."""
    first, distance, last = drive(start, goal)
    x, y, yaw = start
    path = [(x, y, yaw + first * k / n) for n in [int(np.ceil(abs(first) / BASE_PATH_STEP[1]))] for k in range(1, n + 1)]
    yaw += first
    count = int(np.ceil(abs(distance) / BASE_PATH_STEP[0]))
    path += [(x + distance * k / count * np.cos(yaw), y + distance * k / count * np.sin(yaw), yaw)
             for k in range(1, count + 1)]
    x, y = (x + distance * np.cos(yaw), y + distance * np.sin(yaw))
    path += [(x, y, yaw + last * k / n) for n in [int(np.ceil(abs(last) / BASE_PATH_STEP[1]))] for k in range(1, n + 1)]
    return path


class Base:
    """The base skills of :class:`~table_1000.experts.skills.Skills` (and ``ready``)."""

    # -- the base --------------------------------------------------------------------------
    def base_pose(self):
        return np.asarray(self.ctx.env.obs['agent']['base_pose'][0], dtype=float)

    def move_base(self, forward=0., turn=0.):
        """Drive the base straight by ``forward`` (m along its heading, back when negative), then turn
        it in place by ``turn`` (rad); the arms keep their joints."""
        ctx = self.ctx
        x, y, yaw = self.base_pose()
        goal, heading = np.array([x, y]) + forward * np.array([np.cos(yaw), np.sin(yaw)]), yaw + turn
        limit = steps(abs(forward) / BASE_SPEED + abs(turn) / BASE_TURN + BASE_SETTLE)
        for _ in range(limit):
            x, y, now = self.base_pose()
            along = float((goal - [x, y]) @ [np.cos(now), np.sin(now)])
            if abs(along) < BASE_TOLERANCE[0]:
                break
            ctx.step({'base': [[np.clip(BASE_GAIN * along, -BASE_SPEED, BASE_SPEED),
                                np.clip(BASE_GAIN * wrapped(yaw - now), -BASE_TURN, BASE_TURN)]]})
        for _ in range(limit):
            error = wrapped(heading - self.base_pose()[2])
            if abs(error) < BASE_TOLERANCE[1]:
                break
            ctx.step({'base': [[0., np.clip(BASE_GAIN * error, -BASE_TURN, BASE_TURN)]]})
        ctx.step({'base': [[0., 0.]]})
        ctx.hold(BASE_SETTLE)
        for solver in self.solvers.values():
            solver.sync()
        ctx.note('base moved', forward=forward, turn=turn, pose=self.base_pose().tolist())

    @decision
    def base_for(self, poses, arms=None, need=None, path=False, touch=None):
        """World pose ``(x, y, yaw)`` of the base from where one of ``arms`` (default the free ones)
        reaches ``need`` (default all) of ``poses`` with each joint ``BASE_MARGIN`` inside its limits:
        where it stands if they are reached from there, else the sampled pose (see the constants)
        the base gets to soonest, the robot clear of the planning world there and on its way; else
        the one reaching most, with the widest margin (None when none reaches any). No poses: its start pose. Each of ``poses`` is a world TCP pose or a list of them
        along a straight move, each solved from the one before at the torso height of the first
        (``path``: ``poses`` are one such list); ``{arm: poses}`` gives each arm its own. The fingers
        may touch the planning objects ``touch`` there (default: any; see :meth:`reached`)."""
        model, ctx = self.model, self.ctx
        if not isinstance(poses, dict):
            arms = list(arms) if arms else [a for a in self.arms if a not in self.held]
            poses = dict.fromkeys(arms, poses)
        groups = {arm: [list(own)] if path else [[pose] if np.ndim(pose) == 2 else list(pose) for pose in own]
                  for arm, own in poses.items() if len(own)}
        if not groups:
            return self.start_base
        need = max(map(len, groups.values())) if need is None else need
        current = self.base_pose()
        # Sampled around the poses to reach, facing them.
        centre = np.mean([group[0][:2, 3] for own in groups.values() for group in own], axis=0)
        distance = ctx.rng.uniform(*BASE_DISTANCES, BASE_SAMPLES)
        bearing = ctx.rng.uniform(-np.pi, np.pi, BASE_SAMPLES)
        facing = ctx.rng.uniform(-BASE_FACING, BASE_FACING, BASE_SAMPLES)
        samples = np.c_[centre + distance[:, None] * np.c_[np.cos(bearing), np.sin(bearing)],
                        [wrapped(angle) for angle in bearing + np.pi + facing]]
        samples = sorted(samples, key=lambda pose: drive_time(current, pose))
        q = model.q.copy()          # the free arms in (see make_room)
        for arm in self.arms:
            if arm not in self.held:
                q[model.arm_indices(arm)] = self.home[model.arm_indices(arm)]
        actual, world_poses, started = model.base.copy(), ctx.poses(), time.perf_counter()
        chosen, best, fallback, most, checked = None, None, None, (-1, -np.inf), []
        spread = np.array(BASE_SPREAD)
        rejected = {'out of reach': 0, 'collision': 0, 'IK': 0, 'way': 0, 'collisions': {}}
        try:
            mark(self, 'samples', samples=[current, *samples], groups=groups, need=need)
            for index, pose in enumerate([current, *samples]):
                mark(self, 'candidate', index, pose=pose)
                if len(checked) >= BASE_CHECKS[most[0] <= 0]:
                    break
                if any(np.all(np.abs([*(pose[:2] - other[:2]), wrapped(pose[2] - other[2])]) < spread[[0, 0, 1]])
                       for other in checked):
                    continue
                self.put_base(pose, world_poses)
                if index and not self.in_reach(groups, need):
                    mark(self, 'out of reach', outcome='rejected')
                    rejected['out of reach'] += 1
                    continue
                if index and not self.clear(q):
                    mark(self, 'collision', outcome='rejected')
                    rejected['collision'] += 1
                    for pair in self.touching(q)[:3]:
                        key = f'{pair[0]} / {pair[1]}'
                        rejected['collisions'][key] = rejected['collisions'].get(key, 0) + 1
                    continue
                checked.append(pose)
                mark(self, 'IK')
                reached, margin = self.reached(groups, need, touch)
                if reached < need:
                    mark(self, 'IK', outcome='rejected', reached=reached, margin=margin)
                    rejected['IK'] += 1
                    if (reached, margin) > most:
                        best, most = pose, (reached, margin)
                    # Short of the margin but within BASE_MARGIN_LOW: the first such, kept in case.
                    if (index and fallback is None and need == 1 and margin >= BASE_MARGIN_LOW
                            and all(self.put_base(along, world_poses) or self.clear(q)
                                    for along in drive_path(current, pose))):
                        fallback = pose
                    continue
                if not index:
                    return current
                if all(self.put_base(along, world_poses) or self.clear(q) for along in drive_path(current, pose)):
                    mark(self, 'way', outcome='accepted')
                    chosen = pose
                    break
                mark(self, 'way', outcome='rejected')
                rejected['way'] += 1
        finally:
            model.base = actual
            for solver in self.solvers.values():
                solver.sync()
        if chosen is None:
            chosen = fallback
        ctx.note('base choice', checked=len(checked), rejected=rejected, most=list(most), choice_s=time.perf_counter() - started,
                 chosen=None if chosen is None else chosen.tolist(), best=None if best is None else np.asarray(best).tolist())
        return chosen if chosen is not None else best if most[0] > 0 else None

    def put_base(self, pose, world_poses):
        """The planning model's base at ``pose`` (x, y, yaw), the planning worlds following."""
        transform = self.model.base.copy()
        transform[:3, :3] = Rotation.from_euler('z', pose[2]).as_matrix()
        transform[:2, 3] = pose[:2]
        self.model.base = transform
        for solver in self.solvers.values():
            solver.world.set_frame(np.linalg.inv(transform))
            solver.world.update(world_poses)

    def touching(self, q):
        """``[(link, object, object link)]`` the robot at full configuration ``q`` touches in the
        planning worlds (what the arms hold included)."""
        return [pair for arm in list(self.held) or self.arms[:1] for pair in self.model.object_collisions(arm, q)]

    def clear(self, q):
        return not self.touching(q)

    def arm_reach(self, arm):
        """``(shoulder, radius)``: ``arm``'s shoulder in the root frame and the farthest its TCP gets from it."""
        if arm not in self.reaches:
            model, indices = self.model, self.model.arm_indices(arm)
            link = self.ctx.spec.arms[arm].joints[1].replace('joint', 'link')
            q = model.q.copy()
            radius = 0.
            for values in np.random.default_rng(0).uniform(*model.limits[indices].T, (REACH_SAMPLES, len(indices))):
                q[indices] = values
                radius = max(radius, float(np.linalg.norm(model.tcp(arm, q)[:3, 3] - model.fk(link, q)[:3, 3])))
            self.reaches[arm] = (model.fk(link)[:3, 3], radius)
        return self.reaches[arm]

    def in_reach(self, groups, need):
        """Whether ``need`` of an arm's ``groups`` (``{arm: groups}``) are within its reach (horizontally)."""
        for arm, own in groups.items():
            shoulder, radius = self.arm_reach(arm)
            shoulder = (self.model.base @ np.r_[shoulder, 1.])[:2]
            if sum(all(np.linalg.norm(pose[:2, 3] - shoulder) <= radius for pose in group) for group in own) >= need:
                return True
        return False

    def reached(self, groups, need, touch=None):
        """``(groups, margin)``: the most of an arm's ``groups`` (``{arm: groups}``) it reaches with
        ``BASE_MARGIN`` (see :meth:`path_margin`), and its widest margin; the fingers may touch the
        planning objects ``touch`` (default: any)."""
        most = (0, -np.inf)
        for arm, own in groups.items():
            # The hand reaching the poses touches what it reaches (a handle, an object to grasp), as
            # the skill moving it there lets it: no more, or the base stays where the skill cannot.
            world, fingers = self.solvers[arm].world, self.ctx.spec.arms[arm].finger_links
            touched = [(name, finger) for name in (world.objects if touch is None else touch) for finger in fingers
                       if frozenset((name, finger)) not in world.permissions]
            for name, finger in touched:
                world.allow([name], [finger])
            try:
                reached, margins = 0, []
                for group in own:
                    margins.append(self.path_margin(arm, group, attempts=BASE_IK_ATTEMPTS))
                    reached += margins[-1] >= BASE_MARGIN
                    if reached >= need:
                        return reached, max(margins)
                most = max(most, (reached, max(margins, default=-np.inf)))
            finally:
                for name, finger in touched:
                    world.allow([name], [finger], False)
        return most

    def path_margin(self, arm, poses, attempts=IK_ATTEMPTS):
        """Least distance (rad) of ``arm``'s joints to their limits through ``poses`` (world TCP), each
        solved from the one before (one branch) at the torso height of the first, as a straight move
        does; -inf where one has no IK (``attempts`` restarts each)."""
        model, indices = self.model, self.model.arm_indices(arm)
        q, margin = None, np.inf
        for index, pose in enumerate(poses):
            status, q = model.ik(arm, pose, q, world=True, lift=index == 0, attempts=attempts)
            if status != 'Success':
                return -np.inf
            margin = min(margin, float(np.min(np.minimum(q[indices] - model.limits[indices, 0],
                                                         model.limits[indices, 1] - q[indices]))))
        return margin

    def make_room(self, poses, arms=None, need=None, path=False, touch=None):
        """Move the base (see :meth:`base_for`) to where ``poses`` are reachable with a margin: the
        free arms in where they would sweep something on the way, turned in place, straight on,
        turned in place."""
        if not self.ctx.spec.base:
            return
        chosen = self.base_for(poses, arms, need, path, touch)
        self.ctx.note('room', chosen=None if chosen is None else [float(v) for v in chosen])
        if chosen is None:
            return
        before = self.base_pose()
        first, distance, last = drive(before, chosen)
        if abs(distance) < BASE_TOLERANCE[0] * 2 and abs(first) + abs(last) < BASE_TOLERANCE[1] * 2:
            return
        if not self.clear_way(before, chosen):
            self.ready()    # the free arms in, so that they sweep nothing as the base turns
        self.move_base(0., first)
        self.move_base(distance, last)
        self.note_relocated(before)

    def clear_way(self, start, goal):
        """Whether the robot as it is (the arms where they are) is clear of the planning world all
        the way from base pose ``start`` to ``goal``."""
        actual, world_poses = self.model.base.copy(), self.ctx.poses()
        try:
            return all(self.put_base(along, world_poses) or self.clear(self.model.q) for along in drive_path(start, goal))
        finally:
            self.model.base = actual
            for solver in self.solvers.values():
                solver.sync()

    def note_relocated(self, before):
        """Note a move of the base from ``before`` (x, y, yaw): how far it went and turned."""
        after = self.base_pose()
        self.ctx.note('base relocated', travel=float(np.linalg.norm(after[:2] - before[:2])),
                      turn=float(abs((after[2] - before[2] + np.pi) % (2 * np.pi) - np.pi)), pose=after.tolist())

    def stow(self, arms=None, why=None):
        """The free ``arms`` (default all) that are out back at their start pose (see :meth:`ready`),
        as a skill does before it moves the base when nothing is in reach as the arms are, or when a
        move runs into them; whether any was out."""
        out = [arm for arm in (self.arms if arms is None else arms) if arm not in self.held
               and np.abs(self.home - self.model.q)[self.model.arm_indices(arm)].max() >= READY_TOLERANCE]
        if out:
            self.ctx.note('arms stowed', arms=out, why=why)
            self.ready(out)
        return bool(out)

    def in_the_way(self, arm, error):
        """The free arms other than ``arm`` that the move failing with ``error`` ran into."""
        pairs = collisions(error).values()
        links = {name for found in pairs for pair in found for name in (pair[0], pair[-1])}
        return [other for other in self.arms if other != arm and other not in self.held
                and any(link.startswith(self.prefix(other)) for link in links)]

    def prefix(self, arm):
        """What the names of ``arm``'s joints and links start with (``left_arm_``)."""
        return os.path.commonprefix(self.ctx.spec.arms[arm].joints).removesuffix('joint')

    def ready(self, arms=None):
        """Back to the start pose with each of ``arms`` (default all) holding nothing (a planned
        move; skipped where it does not plan), the fingers open."""
        for arm in arms or self.arms:
            if arm in self.held:
                continue
            solver, indices = self.solvers[arm], self.model.arm_indices(arm)
            goal = self.model.q.copy()
            goal[indices] = self.home[indices]
            if np.abs(goal - self.model.q).max() < READY_TOLERANCE:
                continue
            try:
                solver.open_gripper()
                solver.move_to_pose_with_RRTConnect(None, goals=[goal])
            except PlanningFailed as error:
                self.ctx.note('not ready', arm=arm, reason=error.reason)
