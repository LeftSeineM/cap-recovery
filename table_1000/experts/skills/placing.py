"""Place: the held object put down in a region of a surface."""

import itertools

from table_1000.common.timing import timed

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from table_1000.experts.context import ExpertFailed, Unreachable
from table_1000.experts.grasps import translated
from table_1000.experts.skills.common import CANDIDATE_PLANS, IK_ATTEMPTS, ROOM_PLACES, collisions
from table_1000.experts.solver import PlanningFailed
from table_1000.experts.problems import decision, mark

PLACE_ABOVE = .08       # m, place comes down from this high above the release pose
PLACE_CLEARANCE = .01   # m between the held object and the surface at the release
MAX_PLACES = 8          # placements whose motion place plans
MAX_PLACE_IK = 200      # placements place checks by IK
PLACE_STEP = .02        # m between the positions place samples in its region
PLACE_TURN_STEP = 15.   # deg between the turns place samples
PLACE_TURNS = 8         # turns place tries per position, the least first
PLACE_MARGIN = .01      # m the footprint keeps inside the region and off its keep-out areas
# What a place needs of the hold (see Placing.need): NEED_REACHED of NEED_PLACES of its placements,
# spread over the MAX_PLACE_IK nearest (those place checks), in reach by IK (NEED_IK_ATTEMPTS restarts)
# at the release and above it (more than one: the object may turn a little in the hand as it closes),
# per hold (arm, and the object in the hand rounded to NEED_ROUND m / rad).
NEED_PLACES, NEED_REACHED, NEED_IK_ATTEMPTS, NEED_ROUND = 24, 3, 5, (.02, .1)


def overlaps(outline, low, high):
    """Whether the convex polygon ``outline`` (vertices in order) overlaps the rectangle
    ``low``-``high``: no separating axis among the rectangle's and the polygon's edge normals (a
    long, thin footprint may cross a corner with no vertex inside)."""
    if np.any(outline.max(0) < low) or np.any(outline.min(0) > high):
        return False
    corners = np.array([low, [high[0], low[1]], high, [low[0], high[1]]])
    for edge in np.roll(outline, -1, axis=0) - outline:
        normal = np.array([-edge[1], edge[0]])
        a, b = outline @ normal, corners @ normal
        if a.max() < b.min() or b.max() < a.min():
            return False
    return True


class Placing:
    """The place skill of :class:`~table_1000.experts.skills.Skills`."""

    @decision
    @timed('placements')
    def placements(self, target, current, region, near, keep_out, step, margin):
        """``[(object pose at rest on the surface, spot)]`` in ``region`` (see :meth:`place`), the
        footprint at least ``margin`` inside it and out of ``keep_out``, nearest ``near`` first, then
        the least turned from ``current``."""
        frame, low, high = (np.asarray(item, dtype=float) for item in region)
        local = np.linalg.inv(frame)
        boxes = [(np.linalg.inv(np.asarray(f, dtype=float)), np.asarray(l, dtype=float)[:2] - margin,
                  np.asarray(h, dtype=float)[:2] + margin) for f, l, h in keep_out]
        options = []
        for turn in np.radians(np.arange(0., 360., PLACE_TURN_STEP)):
            # Turned about the surface normal, in the region frame.
            rotation = frame[:3, :3] @ Rotation.from_euler('z', turn).as_matrix() @ frame[:3, :3].T @ current[:3, :3]
            points = (target.points @ rotation.T) @ local[:3, :3].T     # in the region frame, about the origin
            centre = (points[:, :2].min(0) + points[:, :2].max(0)) / 2
            spread = np.ptp(points[:, :2], axis=0) / 2
            hull = points[ConvexHull(points[:, :2]).vertices]    # the footprint's outline
            first, last = low[:2] + spread + margin, high[:2] - spread - margin
            if np.any(first > last + 1e-9):
                continue
            for x in np.arange(first[0], last[0] + 1e-9, step):
                for y in np.arange(first[1], last[1] + 1e-9, step):
                    spot = frame @ [x, y, 0., 1.]
                    footprint = hull[:, :2] - centre + [x, y]
                    world = (frame @ np.c_[footprint, np.zeros(len(footprint)), np.ones(len(footprint))].T).T
                    if any(overlaps((inv @ world.T).T[:, :2], low_, high_) for inv, low_, high_ in boxes):
                        continue
                    pose = np.eye(4)
                    pose[:3, :3] = rotation
                    pose[:3, 3] = frame[:3, :3] @ np.r_[[x, y] - centre, -points[:, 2].min()] + frame[:3, 3]
                    turned = Rotation.from_matrix(rotation @ current[:3, :3].T).magnitude()
                    options.append((float(np.linalg.norm((spot[:3] - near)[:2])), turned, pose, spot[:3]))
        # Nearest first, at most PLACE_TURNS of the least turned per position: IK checks spread over
        # positions rather than spending themselves on the turns of a few.
        options.sort(key=lambda option: (round(option[0] / step), option[1]))
        kept, counts = [], {}
        for distance, turned, pose, spot in options:
            key = tuple(np.round(spot[:2] / step).astype(int))
            counts[key] = counts.get(key, 0) + 1
            if counts[key] <= PLACE_TURNS:
                kept.append((pose, spot))
        return kept

    @decision
    def put_down(self, arm, options, tcp_object, clearance, above):
        """Move the held object down at the first of ``options`` (``(object pose, spot)``) in reach
        (IK above and at the release) whose motion plans; ``{'done', 'spot', 'ik_checked',
        'planned', 'last', ...collisions}``."""
        result = {'done': False, 'spot': None, 'ik_checked': 0, 'planned': 0, 'last': 'no placement in reach'}
        # The torso lift where it is first, then moving too.
        planned = set()
        for lift in (False, True):
            checked = 0
            for index, (placed, spot) in enumerate(options):
                if checked >= MAX_PLACE_IK or result['planned'] >= MAX_PLACES:
                    break
                if index in planned:
                    continue
                mark(self, 'candidate', f'{index}/{lift}', placed=placed, spot=spot, lift=lift)
                checked += 1
                count = result['planned']
                if self.lower(arm, placed, spot, tcp_object, clearance, above, lift, result):
                    return result
                if result['planned'] > count:
                    planned.add(index)
        return result

    @decision
    def lower(self, arm, placed, spot, tcp_object, clearance, above, lift, result):
        """Down at one of :meth:`put_down`'s options if it is in reach (``lift``: the torso lift
        moving too) and its motion plans; whether it is, counted in ``result``."""
        ctx, solver = self.ctx, self.solvers[arm]
        result['ik_checked'] += 1
        release = translated(placed, [0., 0., clearance]) @ np.linalg.inv(tcp_object)
        # Where it is let go (the hand and arm low by a wall), then above it at that torso height:
        # the way down moves the arm alone.
        goals = self.down_goals(arm, release, above, lift, IK_ATTEMPTS)
        if not goals:
            return False
        mark(self, 'plan')
        result['planned'] += 1
        try:
            solver.move_to_pose_with_RRTConnect(None, lift=True, goals=goals, attempts=CANDIDATE_PLANS)
            solver.move_to_pose_with_screw(release)
            mark(self, 'plan', outcome='accepted')
            result.update(done=True, spot=spot)
            return True
        except PlanningFailed as error:
            mark(self, 'plan', outcome='rejected', reason=error.reason)
            result.update(last=error.reason, **collisions(error))
            ctx.note('next placement', arm=arm, spot=spot.tolist(), reason=error.reason, **collisions(error))
            return False

    @decision
    def need(self, obj, region, near=None, keep_out=(), clearance=PLACE_CLEARANCE, above=PLACE_ABOVE,
             step=PLACE_STEP, margin=PLACE_MARGIN):
        """What :meth:`place` with these arguments needs of the hand that will hold ``obj`` (not held
        yet, at its pose now): an :class:`~table_1000.experts.skills.chain.InHand` accepting the holds
        from which ``NEED_REACHED`` of its placements are in reach (IK at the release and above it),
        and the hands at the first of them (the TCP at the release and above it, per hold)."""
        ctx = self.ctx
        target = self.target(obj)
        current = ctx.poses()[target.body]
        frame, low, high = (np.asarray(item, dtype=float) for item in region)
        if near is None:
            near = frame @ np.r_[(low[:2] + high[:2]) / 2, 0., 1.]
        options = self.placements(target, current, region, np.asarray(near, dtype=float)[:3], keep_out, step, margin)
        options = options[:MAX_PLACE_IK]       # those place checks
        options = [options[i] for i in np.unique(np.linspace(0, len(options) - 1, NEED_PLACES).astype(int))] if options else []
        return self.restore_need({'obj': obj, 'target': target, 'options': options, 'clearance': clearance,
                                  'above': above, 'arms': [arm for arm in self.arms if arm not in self.held],
                                  'verdicts': [], 'origin': {'obj': obj, 'region': region, 'near': near, 'keep_out': keep_out,
                                                            'clearance': clearance, 'above': above, 'step': step, 'margin': margin}})

    def restore_need(self, recipe):
        """Rebuild a Place requirement with its evaluated candidates and cached verdicts."""
        from table_1000.experts.skills.chain import InHand
        ctx = self.ctx
        obj, target, options = recipe['obj'], recipe['target'], recipe['options']
        clearance, above = recipe['clearance'], recipe['above']
        verdicts, obj_prefix = dict(recipe['verdicts']), f'{obj}:'

        @decision
        @timed('placement_need')
        def accepts(self, arm, tcp_object):
            rotation = Rotation.from_matrix(tcp_object[:3, :3]).as_rotvec()
            # Per hold and per robot state (the base and joints, as a base move or a stow leaves them).
            key = (arm, *np.round(tcp_object[:3, 3] / NEED_ROUND[0]).astype(int), *np.round(rotation / NEED_ROUND[1]).astype(int),
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
                    reached = (self.in_reach_down(arm, placed, tcp_object, clearance, above) for placed, _ in options)
                    required = min(NEED_REACHED, len(options)) or 1     # a small region may have fewer
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
            releases = [translated(placed, [0., 0., clearance]) @ np.linalg.inv(tcp_object) for placed, _ in options]
            return [[release, translated(release, [0., 0., above])] for release in releases[:NEED_REACHED]]
        need = InHand(obj, lambda arm, tcp_object: accepts(self, arm, tcp_object), recipe['arms'], hands, recipe)
        if not hasattr(ctx, 'requirements'):
            ctx.requirements = []
        ctx.requirements.append(need)
        return need

    @decision
    def in_reach_down(self, arm, placed, tcp_object, clearance, above):
        """Whether ``arm`` holding the object as ``tcp_object`` has IK at its release at ``placed``
        (the torso lift moving too) and above it."""
        release = translated(placed, [0., 0., clearance]) @ np.linalg.inv(tcp_object)
        return bool(self.down_goals(arm, release, above, True, NEED_IK_ATTEMPTS, first=True))

    @decision
    def down_goals(self, arm, release, above, lift, attempts, first=False):
        """Above configurations with a release branch at the same torso height."""
        mark(self, 'release IK')
        status, releases = self.model.ik(arm, release, self.model.q, world=True, lift=lift,
                                         attempts=attempts, closest=False)
        if status != 'Success':
            mark(self, 'release IK', outcome='rejected', status=status)
            return []
        goals = []
        for branch, q_release in enumerate(sorted(releases, key=lambda q: np.linalg.norm(q - self.model.q))):
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
              step=PLACE_STEP, margin=PLACE_MARGIN):
        """Put the held ``obj`` down in ``region`` and release it; see the module docstring.
        ``region`` is ``(frame, low, high)``: the rectangle ``low``-``high`` (x, y) on the surface
        z = 0 of the world 4x4 ``frame``; ``keep_out`` are such rectangles (any height) the
        footprint stays out of; ``near`` the world point to place it nearest (default the
        region's centre). Returns ``{'arm', 'spot', 'object'}`` (its pose after)."""
        ctx = self.ctx
        arm = arm or self.holding(obj)
        if arm is None or arm not in self.held or self.held[arm]['target'].obj != obj:
            raise ValueError(f'{obj!r} is not held')
        solver, target = self.solvers[arm], self.held[arm]['target']
        current = ctx.poses()[target.body]
        tcp_object = np.linalg.inv(solver.tcp()) @ current
        frame, low, high = (np.asarray(item, dtype=float) for item in region)
        if near is None:
            near = frame @ np.r_[(low[:2] + high[:2]) / 2, 0., 1.]
        options = self.placements(target, current, region, np.asarray(near, dtype=float)[:3], keep_out, step, margin)
        if not options:
            ctx.fail('no room', region=[frame.tolist(), low.tolist(), high.tolist()])
        # IK first (cheap) over many, then motion for those that have it. Where none is in reach,
        # the base moves to reach the first ones, and they are screened again.
        result = self.put_down(arm, options, tcp_object, clearance, above)
        if not result['done'] and self.stow():     # none in reach or planned with the other arm out
            result = self.put_down(arm, options, tcp_object, clearance, above)
        if result['planned'] == 0 and ctx.spec.base:
            releases = [translated(placed, [0., 0., clearance]) @ np.linalg.inv(tcp_object)
                        for placed, _ in options[:ROOM_PLACES]]
            self.make_room([[release, translated(release, [0., 0., above])] for release in releases], [arm], need=1,
                           touch=[])    # place lets the fingers touch nothing (the held object goes with them)
            current = ctx.poses()[target.body]
            options = self.placements(target, current, region, np.asarray(near, dtype=float)[:3], keep_out, step, margin)
            result = self.put_down(arm, options, tcp_object, clearance, above)
        if not result['done']:
            ctx.fail('no placement', Unreachable if result['planned'] == 0 else ExpertFailed, arm=arm,
                     options=len(options), **{k: v for k, v in result.items() if k != 'done'})
        spot = result['spot']
        self.release(arm)
        ctx.hold(.3)
        return {'arm': arm, 'spot': spot.tolist(), 'object': ctx.poses()[target.body].tolist()}
