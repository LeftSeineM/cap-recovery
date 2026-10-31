"""Physical fingertip support for a drawer's housing; no asset or drive overrides."""

from contextlib import contextmanager
import sys

import numpy as np

from table_1000.experts.context import ExpertFailed, Unreachable
from table_1000.experts.skills.hold.frames import build_grasp_pose, translated
from table_1000.experts.skills.push.pressure import PressureLoad
from table_1000.experts.skills.common import APPROACH
from table_1000.experts.solver import PlanningFailed


class Stability:
    """Peak housing translation/rotation and absolute opening of other sliders.

    Observe every control step, including approach and retreat. Relative to entry,
    translation must stay <= 5 mm and rotation <= 2 degrees; every other slider
    must stay <= 5 mm from its lower limit. An already open other drawer fails
    immediately: close it before using ``brace=True``. Peaks are emitted in a
    ``drawer stability`` note even on failure. These checks are opt-in together
    with bracing; unbraced calls omit these stability checks.
    """

    def __init__(self, ctx, obj, joint):
        from table_1000.experts.skills.drawer.joints import Joint
        self.ctx, self.obj, self.joint = ctx, obj, joint
        self.start = ctx.pose(obj, joint.parent).copy()
        self.failed = False
        self.others = [Joint.of(ctx.geometry[obj], name) for name, entry in ctx.geometry[obj].joints.items()
                       if name != joint.name and entry['type'] == 'SLIDER']
        self.peak = dict(translation_m=0., rotation_deg=0., other_openings_m={j.name: 0. for j in self.others})

    def sample(self, ctx, fail=True):
        from table_1000.experts.skills.drawer.joints import opening
        pose = ctx.pose(self.obj, self.joint.parent)
        shift = float(np.linalg.norm(pose[:3, 3] - self.start[:3, 3]))
        angle = float(np.degrees(np.arccos(np.clip((np.trace(pose[:3, :3] @ self.start[:3, :3].T) - 1) / 2, -1., 1.))))
        self.peak['translation_m'] = max(self.peak['translation_m'], shift)
        self.peak['rotation_deg'] = max(self.peak['rotation_deg'], angle)
        for joint in self.others:
            self.peak['other_openings_m'][joint.name] = max(self.peak['other_openings_m'][joint.name],
                                                          float(opening(ctx, self.obj, joint.name)))
        if fail and not self.failed and (shift > .005 or angle > 2. or any(v > .005 for v in self.peak['other_openings_m'].values())):
            self.failed = True
            ctx.fail('drawer housing stability exceeded (5 mm / 2 deg / other drawers 5 mm)', **self.peak)


def roof_contacts(ctx, obj, body):
    """Top-facing convex-piece centers, on actual housing material, not its box."""
    from table_1000.experts.world.geometry import asset_shapes, cast
    poses = {name: ctx.pose(obj, name) for name in ctx.geometry[obj].bodies}
    shapes = [shape for shape in asset_shapes(obj, ctx.geometry[obj], poses) if shape.body == body]
    high = max(shape.points[:, 2].max() for shape in shapes)
    points = []
    for shape in shapes:
        top = shape.points[shape.points[:, 2] >= high - .001]
        if len(top) < 3:
            continue
        low, upper = top[:, :2].min(0), top[:, :2].max(0)
        for x in np.linspace(low[0], upper[0], 5)[1:-1]:
            for y in np.linspace(low[1], upper[1], 5)[1:-1]:
                origin = np.array([[x, y, high + .01]])
                hit, distance, normal = cast(shapes, origin, [0., 0., -1.])
                if hit[0] >= 0 and normal[0, 2] > .98:
                    points.append(origin[0] - [0., 0., distance[0]])
    return points


@contextmanager
def brace(ctx, obj, joint, active, amount=None):
    """Choose a feasible assignment of manipulating/support hands; no unbraced fallback.

    Try both hand assignments at the current base before searching another
    base stance, the nearer active hand first. Failed setup retracts the hand. An error in
    the actual pull/push propagates immediately rather than repeating a motion.
    Both hands must be empty. See _brace and Stability for contact/limit checks.
    """
    arms = [active, *[a for a in ctx.skills().arms if a != active]]
    attempts = [(candidate, relocate) for relocate in (False, True) for candidate in arms]
    for number, (candidate, relocate) in enumerate(attempts):
        entered = False
        try:
            with _brace(ctx, obj, joint, candidate, amount, relocate=relocate):
                entered = True
                yield candidate
            return
        except ExpertFailed as error:
            if (entered or number == len(attempts) - 1 or
                    not (isinstance(error, (Unreachable, PlanningFailed)) or error.reason == 'no clear base route')):
                raise
            ctx.note('next brace assignment', active=candidate, reason='no coupled support')


@contextmanager
def _brace(ctx, obj, joint, active, amount=None, *, relocate=False):
    """Press the housing roof with the other empty hand during a drawer action.

    Called by ``semantic.open/close(..., brace=True)``. Both hands must be
    empty at entry. Closed fingertips press actual top collision material, advancing
    at most 6 mm to establish measured normal load >= 1 N. Losing that measured
    load during the drawer action fails immediately. No loaded contact or
    no collision-free approach raises ``Unreachable``; clear the roof or
    use an unbraced call explicitly. While supported, base/torso motion and
    stowing the support hand are disabled. The support hand retracts after the
    pull/push, also on failure. Contact is purely physical; no simulation attachment or drive override is used.
    """
    s = ctx.skills()
    arms = [a for a in s.arms if a != active and a not in s.held]
    if s.braced:
        raise ValueError('nested cabinet braces are unsupported; finish the current drawer action first')
    if not arms:
        raise ValueError('brace=True needs another empty hand; place the held object first, or use brace=False')
    arm = arms[0]
    s.stow(why='prepare two-hand cabinet support')
    solver = s.solvers[arm]
    names = [name for name, part in solver.world.objects.items() if part.body == f'{obj}:{joint.parent}']
    contacts = ctx.env.contacts
    load = PressureLoad(f'{obj}:{joint.parent}', ctx.spec.arms[arm].finger_links, contacts.dt,
                        1., lambda: contacts.time)
    target = s.target(obj, joint.handle)
    direction = joint.direction(ctx.pose(obj, joint.child))
    from table_1000.experts.skills.push.motion import PUSH_GAP
    from table_1000.experts.skills.drawer.joints import opening
    if amount is None:
        push = -direction
        pushes = [[translated(contact, -push * PUSH_GAP), contact,
                   translated(contact, push * (opening(ctx, obj, joint.name) + .003))]
                  for _, _, contact in s.push_contacts(target, push, [active])]
        paths = pushes
    else:
        paths = s.grasp_paths(target, [active], approach=-direction,
                              then=direction * max(0., amount - opening(ctx, obj, joint.name)),
                              max_angle=np.radians(20.))[active]
    options = []
    for point in roof_contacts(ctx, obj, joint.parent):
        forward = point - s.shoulder(arm)
        forward[2] = 0.
        forward /= np.linalg.norm(forward)
        closing = np.cross([0., 0., 1.], forward)
        for tilt in np.radians([45., 65., 25., 0.]):
            approach = forward * np.sin(tilt) - np.array([0., 0., 1.]) * np.cos(tilt)
            options.append((point, build_grasp_pose(approach, closing, point - approach * s.hands[arm].tip())))
    options.sort(key=lambda option: (option[1][2, 0], np.linalg.norm(option[0] - s.shoulder(arm))))
    chosen = {}
    def both_reached(groups, need, touch=None):
        chosen.clear()
        rejected = {}
        def reject(reason):
            rejected[reason] = rejected.get(reason, 0) + 1
        indices = s.model.arm_indices(active) + s.model.arm_indices(arm)
        for side_pose in groups[arm]:
            seed = s.model.vector(ctx.spec.arms[arm].finger_targets(0.))
            seed = s.model.vector(ctx.spec.arms[active].finger_targets(0. if amount is None else .06), seed)
            permissions = {a: set(other.world.permissions) for a, other in s.solvers.items()}
            solver.world.allow(names, ctx.spec.arms[arm].finger_links)
            s.solvers[active].world.allow(names, ctx.spec.arms[arm].finger_links)
            s.solvers[active].allow(target.names)
            try:
                status, q = s.model.ik(arm, side_pose[0], seed, world=True, lift=True, attempts=5)
                if status != 'Success':
                    reject('support IK')
                    continue
                status, q = s.model.ik(arm, side_pose[1], q, world=True, attempts=5)
                if status != 'Success':
                    reject('support IK')
                    continue
                q_contact = q.copy()
                status, q_pre = s.model.ik(arm, side_pose[0], q, world=True, attempts=5)
                if status != 'Success':
                    reject('support IK')
                    continue
                press = s.model.plan_screw(arm, translated(side_pose[1], side_pose[1][:3, 0] * .006),
                                          q_pre, world=True, time_step=.04)
                if press['status'] != 'Success':
                    reject('press '+press.get('reason',press['status']))
                    continue
                q_contact[s.model.arm_indices(arm)] = press['position'][-1]
                retreat = s.model.plan_screw(arm, side_pose[0], q_contact, world=True, time_step=.04)
                if retreat['status'] != 'Success':
                    reject('retreat '+retreat.get('reason',retreat['status']))
                    continue
                # The manipulating hand moves while the support hand presses,
                # so both preflights must use the loaded contact posture.
                q = q_contact
                actual, old_braced = s.model.q.copy(), s.braced
                s.model.q, s.braced = q, True
                try:
                    from table_1000.experts.skills.drawer.joints import opening
                    if amount is None:
                        feasible = None
                        s.solvers[active].allow([name for name, part in s.solvers[active].world.objects.items()
                                                 if part.body == target.body])
                        for path in pushes:
                            status, qa = s.model.ik(active, path[0], q, world=True, attempts=5)
                            if status == 'Success':
                                contact = s.model.plan_screw(active, path[1], qa, world=True, time_step=.04)
                                if contact['status'] != 'Success':
                                    reject('push approach '+contact.get('reason', contact['status']))
                                    continue
                                at_contact = qa.copy()
                                at_contact[s.model.arm_indices(active)] = contact['position'][-1]
                                s.model.q = at_contact
                                own, riders, touched = s.dragged(active, (obj, joint.child))
                                s.solvers[active].carry(own, touched + riders)
                                s.solvers[active].carry(riders, touched)
                                try:
                                    motion = s.model.plan_screw(active, path[-1], at_contact, world=True, time_step=.04)
                                finally:
                                    s.solvers[active].release(riders, touched)
                                    s.solvers[active].release(own, touched + riders)
                                    s.model.q = q
                                reject('push '+motion.get('reason',motion['status']))
                                if motion['status'] == 'Success':
                                    feasible = (None, None, None, qa)
                                    break
                    else:
                        candidates = s.options(target, [active], set(), approach=-direction,
                            then=direction * max(0., amount - opening(ctx, obj, joint.name)), max_angle=np.radians(20.))
                        feasible = next(candidates, None)
                        candidates.close()
                    if feasible is not None:
                        values = feasible[3][indices]
                        margin = float(np.min(np.minimum(values-s.model.limits[indices,0], s.model.limits[indices,1]-values)))
                        if margin >= .05:
                            chosen.update(pose=side_pose[1], pre=side_pose[0], q=q_pre.copy(), active_q=feasible[3].copy())
                            return 1, margin
                finally:
                    s.model.q, s.braced = actual, old_braced
            finally:
                for a, other in s.solvers.items():
                    for pair in other.world.permissions - permissions[a]:
                        first, second = sorted(pair)
                        other.world.allow([first], [second], False)
        ctx.note('brace stance rejected', counts=rejected)
        return 0, -np.inf
    groups = {active: paths, arm: [[translated(pose, -pose[:3, 0] * APPROACH), pose]
                                 for _, pose in options]}
    if not paths or not options:
        ctx.fail('no collision-free drawer/support candidates; clear the drawer approach and housing', Unreachable)
    if relocate and ctx.spec.base:
        s.make_room(groups, need=1, touch=names+target.names, evaluate=both_reached)
    both_reached(groups, 1)
    if not chosen:
        ctx.fail('no common reach for drawer and support; clear obstacles or use brace=False', Unreachable)
    point, pose = next(option for option in options if np.allclose(option[1], chosen['pose']))
    retreat_direction = None
    old_arms, old_braced = s.arms, s.braced
    old_seed = s.brace_seed
    old_locks = {a: other.lock_lift for a, other in s.solvers.items()}
    permissions = {a: set(other.world.permissions) for a, other in s.solvers.items()}
    try:
        for other in s.solvers.values():
            other.world.allow(names, ctx.spec.arms[arm].finger_links)
        ctx.note('brace approach', arm=arm, point=point.tolist(), approach=pose[:3, 0].tolist())
        solver.close_gripper()
        solver.move_to_pose_with_RRTConnect(None, lift=True, goals=[chosen['q']], attempts=2)
        retreat_direction = -pose[:3, 0] * APPROACH
        with contacts.observe_impulses(load.record):
            solver.move_to_pose_with_screw(pose)
            load.read()
            for depth in np.linspace(0., .006, 4):
                if depth:
                    solver.move_to_pose_with_screw(translated(pose, pose[:3, 0] * depth))
                load.read()
                ctx.hold(.08)
                sample = load.read()
                ctx.note('brace pressure', arm=arm, depth=float(depth), sample=sample)
                if sample['bearing_now']:
                    break
            else:
                ctx.fail('no loaded cabinet brace (1 N minimum); clear roof or use brace=False', Unreachable)
        ctx.note('cabinet braced', arm=arm, body=joint.parent, contact=point.tolist(),
                 preload_m=float(depth), load=sample)
        s.arms = [active]
        s.braced = True
        s.brace_seed = chosen['active_q']
        for other in s.solvers.values():
            other.lock_lift = True
        if amount is not None:
            # Loaded contact can occur before the preflight's maximum press.
            # Recheck that measured support posture before starting any pull;
            # an unreachable setup may still try the other hand assignment.
            candidates = s.options(target, [active], set(), approach=-direction,
                then=direction * max(0., amount - opening(ctx, obj, joint.name)), max_angle=np.radians(20.))
            try:
                feasible = next(candidates, None)
            finally:
                candidates.close()
            if feasible is None:
                ctx.fail('drawer grasp unreachable at measured brace contact', Unreachable)
            s.brace_seed = feasible[3].copy()
        lost = False
        def supported(context):
            nonlocal lost
            sample = load.read()
            if not sample['bearing_now'] and not lost:
                lost = True
                context.fail('cabinet brace lost its measured 1 N contact; reposition the support hand', load=sample)
        with contacts.observe_impulses(load.record):
            ctx.observers.append(supported)
            try:
                yield
            finally:
                ctx.observers.remove(supported)
        ctx.note('brace action load', arm=arm, contact_retained=not lost)
    finally:
        s.arms = old_arms
        s.braced = old_braced
        s.brace_seed = old_seed
        for a, other in s.solvers.items():
            other.lock_lift = old_locks[a]
        unwinding = sys.exc_info()[0] is not None
        for other in s.solvers.values():
            other.world.allow(target.names, ctx.spec.arms[active].finger_links)
        try:
            if retreat_direction is not None:
                solver.move_to_pose_with_screw(translated(solver.tcp(), retreat_direction))
        except ExpertFailed as error:
            ctx.note('brace retreat failed', arm=arm, reason=error.reason)
            if not unwinding:
                raise
        finally:
            for a, other in s.solvers.items():
                for pair in other.world.permissions - permissions[a]:
                    first, second = sorted(pair)
                    other.world.allow([first], [second], False)


@contextmanager
def stability(ctx, obj, joint):
    monitor = Stability(ctx, obj, joint)
    ctx.observers.append(monitor.sample)
    try:
        monitor.sample(ctx)
        yield monitor
    finally:
        ctx.observers.remove(monitor.sample)
        monitor.sample(ctx, fail=False)
        ctx.note('drawer stability', object=obj, joint=joint.name, **monitor.peak)
