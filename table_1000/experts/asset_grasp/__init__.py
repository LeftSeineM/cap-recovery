"""Isolated asset pick and wrist-shake evaluation using the existing grasp skill."""

from collections import Counter
import math
from pathlib import Path
import time
import types

import numpy as np

from table_1000.scenes.layout import Layout
from table_1000.sim import CONTROL_HZ
from table_1000.experts.asset_grasp.stability import (TRANSPORT, classify, is_dropped, metrics, motion_report, pulse)
from table_1000.experts.asset_grasp.transport import retime
from table_1000.experts.solver import PlanningFailed

REGION_SIZE = .10
WORK_OFFSET = (.05, 0.)  # keep the near edge of the square clear of the folded-arm workspace
SLIP_LIMIT = .005
SHAKE_AMPLITUDE = .4
SHAKE_SECONDS = .5
SHAKE_OFFSETS = (1, -1, 1, -1, 0)

# One tenth of the existing 0.01 joint tracking tolerance; only the measured
# preflight start may be projected, never an IK result or a joint limit.
START_PROJECTION_TOLERANCE = .001


def oriented_shape(points):
    """Fit in the geometry's oriented box, so a diagonal wire keeps its true width."""
    import trimesh
    from table_1000.experts.grasps import fitted
    to_box, _ = trimesh.bounds.oriented_bounds(points)
    local = np.asarray(points) @ to_box[:3, :3].T + to_box[:3, 3]
    shape, offset = fitted(local)
    return shape, np.linalg.inv(to_box) @ offset


def targets(skill):
    """Convex whole objects first; hollow assemblies use solid collision parts."""
    geometry = skill.ctx.geometry['object']
    def target(part=None):
        result = skill.target('object', part)
        # The aggregate envelope is tried only after authored part candidates.
        # This applies to solid bodies too: a broad envelope can hide a stable
        # part choice even when it is itself collision-valid.
        if part is None:
            result.fallback = True
        result.shape, result.offset = oriented_shape(geometry.points(part, result.body.split(':', 1)[1]))
        return result
    whole = target()
    solid = geometry.solid(None, narrow=0.)
    if solid:
        yield whole
    parts = [target(name) for body, name in geometry.parts()
             if f'object:{body}' == whole.body and geometry.solid(name, narrow=0.)]
    # A folded sheet or a ring has no solid part-level box. Its authored convex
    # collision pieces provide actual contact surfaces, without asset-specific points.
    for body, name, transform, kind, data in geometry.shapes:
        if f'object:{body}' != whole.body or geometry.solid(name, narrow=0.) or kind != 'convex':
            continue
        shape, offset = oriented_shape(data)
        piece = skill.target('object', name, shape=shape, offset=transform @ offset)
        parts.append(piece)
    parts.sort(key=lambda target: -float(np.prod(np.ptp(target.shape.points(), axis=0))))
    for target in parts:
        if not solid or target.names != whole.names:
            yield target
    if not solid:
        yield whole


def target_groups(candidates):
    """Real parts first; the whole-assembly envelope is a planning fallback."""
    return [group for fallback in (False, True)
            if (group := [target for target in candidates if getattr(target, 'fallback', False) == fallback])]


def pick(skill, report, candidates=None):
    """Choose among real part heads before the first physical grasp and lift."""
    from table_1000.experts.context import ExpertFailed
    ctx = skill.ctx
    report['target_attempts'] = []
    candidates = list(targets(skill) if candidates is None else candidates)
    if not candidates:
        ctx.fail('no reachable grasp', candidates_tried=0)
    attempt = None
    try:
        groups = target_groups(candidates)
        for number, group in enumerate(groups):
            target = group[0]
            attempt = {'shape': type(target.shape).__name__, 'parameters': vars(target.shape),
                       'parts': list(target.names), 'fallback': getattr(target, 'fallback', False)}
            report['target_attempts'].append(attempt)
            try:
                result = skill.grasp(group, approach=(0., 0., -1.), then=(0., 0., .10), base=False,
                                     closure_only_record=True)
                break
            except ExpertFailed as error:
                attempt['failure'] = {'reason': error.reason, 'details': error.details}
                if error.reason != 'no reachable grasp' or number == len(groups) - 1:
                    raise
        target = skill.held[result['arm']]['target']
        report['selected_target_index'] = next(i for i, item in enumerate(candidates) if item is target)
        attempt.update(shape=type(target.shape).__name__, parameters=vars(target.shape), parts=list(target.names))
        report['target'] = attempt
        report['pick'] = result
        solver = skill.solvers[result['arm']]
        attempt['closure'] = {'tcp_pose': ctx.tcp(result['arm']).tolist(),
                             'object_pose': ctx.poses()[target.body].tolist(),
                             'finger_gap_m': solver.finger_gap(),
                             'check': result['closure_check'],
                             'mass_kg': float(ctx.env.body_mass(target.body)[0]),
                             'center_of_mass_world_m': ctx.env.center_of_mass(target.body)[0].tolist(),
                             'surface_contacts': list(ctx.env.contacts.surface_contacts.values())}
        report['closure'] = attempt['closure']
        rest = [name for name, obj in solver.world.objects.items()
                if obj.body == target.body and name not in target.names]
        solver.carry(rest, target.names + result['supports'])
        before = ctx.poses()[target.body].copy()
        trace = []
        ctx.env.contacts.take_impulses()
        def observe(c):
            impulses = c.env.contacts.take_impulses()
            trace.append({'step': c.steps, 'tcp_pose': c.tcp(result['arm']).tolist(),
                'object_pose': c.poses()[target.body].tolist(), 'finger_gap_m': solver.finger_gap(),
                'surface_contacts': list(c.env.contacts.surface_contacts.values()),
                'normal_forces_n': [{'bodies': list(pair), 'force_n': impulse * CONTROL_HZ}
                                    for pair, impulse in impulses.items()]})
        ctx.observers.append(observe)
        try:
            skill.move_by_displacement(result['arm'], (0., 0., .10))
            ctx.hold(.2)
        finally:
            ctx.observers.remove(observe)
            attempt['lift_trace'] = trace
        rise = float(ctx.poses()[target.body][2, 3] - before[2, 3])
        attempt['lift'] = {'rise_m': rise, 'tcp_pose': ctx.tcp(result['arm']).tolist(),
                          'object_pose': ctx.poses()[target.body].tolist(),
                          'finger_gap_m': solver.finger_gap()}
        report['lift'] = attempt['lift']
        if rise < .05:
            solver.release(rest, target.names + result['supports'])
            skill.release(result['arm'])
            ctx.note('missed lift', rise=rise)
            ctx.fail('object not lifted', rise=rise, grasp=result['grasp'])
        result.update(rise=rise)
        return target, result
    except ExpertFailed as error:
        attempt['failure'] = {'reason': error.reason, 'details': error.details}
        raise


def inventory(source, excluded=()):
    """Deduplicate asset paths, retaining all instance names and explicit exclusions."""
    grouped = {}
    for name, spec in source.OBJECTS.items():
        grouped.setdefault(spec['asset'], []).append(name)
    return {asset: {'instances': names, 'excluded': all(name in excluded for name in names)}
            for asset, names in sorted(grouped.items())}


def scene(asset, spec, settings):
    return types.SimpleNamespace(
        TABLE=settings.TABLE, OBJECTS={'object': {'asset': asset}},
        ROBOT={'asset': f'robots/{spec.name}', 'position': settings.ROBOT['position'],
               'yaw': settings.ROBOT['yaw'],
               'joints': spec.keyframe(settings.ROBOT.get('keyframe', 'zero'))})


def placement(source, assets, seed, center, size=REGION_SIZE):
    rng = np.random.default_rng(seed)
    xy = np.asarray(center) + rng.uniform(-size / 2, size / 2, 2)
    yaw = float(rng.uniform(-math.pi, math.pi))
    layout = Layout(source, assets, rng)
    pose = layout.resting_pose('object', xy, yaw)
    return pose, {'xy_m': xy.tolist(), 'yaw_rad': yaw, 'pose': pose.tolist()}


def displacement(reference, current, points):
    """Largest collision-vertex displacement in the TCP frame (includes rotation)."""
    before = points @ reference[:3, :3].T + reference[:3, 3]
    after = points @ current[:3, :3].T + current[:3, 3]
    return float(np.linalg.norm(after - before, axis=1).max())


def grasp_family(report):
    """Coarse orientation family in the settled body's frame; exact poses stay in trials."""
    pick = report['pick']
    local = np.linalg.inv(report.get('settled_asset_pose', report['settled_pose'])) @ np.asarray(pick['grasp']['pose'])
    axes = [('+' if local[int(np.argmax(np.abs(local[:3, column]))), column] >= 0 else '-')
            + 'xyz'[int(np.argmax(np.abs(local[:3, column])))] for column in (0, 1)]
    parts = ','.join(report.get('target', {}).get('parts', ['whole']))
    return f"{parts} / approach {axes[0]} / close {axes[1]}"


def failure_reason(report):
    """Keep a missed closure distinct from an initially unreachable placement."""
    reason = report.get('failure', {}).get('reason', report['status'])
    if reason == 'no reachable grasp' and any(event['note'] == 'missed grasp'
                                               for event in report.get('events', [])):
        return 'object not held'
    return reason


def closure_statistics(reports):
    """First-attempt signal versus measured lift/transport; planning is unknown truth."""
    table = {f'predicted_{prediction}_actual_{actual}': 0
             for prediction in ('held', 'unheld') for actual in ('stable', 'failed')}
    unknown, missing, examples, by_reason = 0, 0, [], {}
    for report in reports:
        check = report.get('pick', {}).get('closure_check')
        if check is None:
            missing += 1
            continue
        stable = report['status'] == 'success'
        measured_failure = measured_stability(report)
        if not stable and not measured_failure:
            unknown += 1
            continue
        prediction = 'held' if check['held'] else 'unheld'
        actual = 'stable' if stable else 'failed'
        key = f'predicted_{prediction}_actual_{actual}'
        table[key] += 1
        reason = 'success' if stable else failure_reason(report)
        group = by_reason.setdefault(reason, Counter())
        group[key] += 1
        if check['held'] != stable:
            examples.append({'seed': report['seed'], 'predicted_held': check['held'], 'actual_stable': stable,
                             'failure_reason': reason, 'check_reasons': check['reasons'],
                             'parts': report.get('target', {}).get('parts', [])})
    return {'table': table, 'unmeasured': unknown, 'no_check': missing,
            'by_failure_reason': {key: dict(value) for key, value in by_reason.items()}, 'disagreements': examples}


def measured_stability(report):
    """An observed slip/drop is measured even if it interrupts relocation."""
    if report['status'] == 'invalid_motion':
        return False
    hold = report.get('hold', {})
    return report.get('lift', {}).get('rise_m', .05) < .05 or hold.get('completed', False) \
        or hold.get('held') is False or hold.get('dropped', False)


def summarize(reports, expected):
    success = sum(r['status'] == 'success' for r in reports)
    unassessed = Counter(failure_reason(r) for r in reports if r['status'] == 'not_evaluated')
    invalid = sum(r['status'] == 'invalid_motion' for r in reports)
    assessed = expected - sum(unassessed.values()) - invalid
    failures = Counter(failure_reason(r) for r in reports if r['status'] not in ('success', 'not_evaluated', 'invalid_motion'))
    if expected > len(reports):
        failures['missing report'] += expected - len(reports)
    by_grasp, not_evaluated = {}, Counter()
    attempts = [attempt for report in reports for attempt in [report, *report.get('preference_attempts', [])]]
    for report in attempts:
        if 'pick' not in report:
            continue
        key = grasp_family(report)
        hold = report.get('hold', {})
        if not measured_stability(report):
            # A planner/tracking failure after grasping did not measure hold
            # stability. Retain the family without calling it unstable.
            not_evaluated[key] += 1
            continue
        entry = by_grasp.setdefault(key, {'trials': 0, 'success': 0})
        entry['trials'] += 1
        entry['success'] += bool(hold.get('held'))
    for entry in by_grasp.values():
        entry['rate'] = entry['success'] / entry['trials']
    return {'expected': expected, 'reported': len(reports), 'success': success,
            'assessed': assessed, 'not_evaluated': sum(unassessed.values()), 'invalid_motion': invalid,
            'not_evaluated_reasons': dict(unassessed),
            'closure_check': closure_statistics(reports),
            'picked': sum(r.get('lift', {}).get('rise_m', 0.) >= .05 for r in reports),
            'first_attempt_success': success,
            'attempts': len(attempts), 'rate': success / assessed if assessed else None,
            'initial_unreachable': sum(failure_reason(r)
                                       == 'no reachable grasp' for r in reports),
            'failures': dict(failures), 'preference_data': {
                'by_grasp': by_grasp, 'not_evaluated_by_grasp': dict(not_evaluated)},
            'status': 'PASS' if success == assessed and assessed and not invalid else 'FAIL'}


def transport_plan(model, arm, origin, start=None):
    """Preflight complete equivalent pulses without moving or changing IK branches.

    Try the original transverse direction, its reverse, then the two orthogonal
    horizontal directions. Every choice has the same two 50 mm translation
    strokes, wrist stroke and timing; upward motion never approaches the table.
    """
    indices, lift_indices = model.arm_indices(arm), model.lift_indices()
    linear = pulse(TRANSPORT['linear_acceleration_m_s2'], stroke=REGION_SIZE / 2, hz=CONTROL_HZ)
    angular = pulse(TRANSPORT['angular_acceleration_rad_s2'],
                    speed=TRANSPORT['angular_velocity_rad_s'], hz=CONTROL_HZ)
    inward = -1. if origin[1, 3] > 0 else 1.
    choices = []
    measured = model.q.copy() if start is None else np.array(start, copy=True)
    projected = np.clip(measured, model.limits[:, 0], model.limits[:, 1])
    difference = np.abs(projected - measured)
    corrected = np.flatnonzero(difference)
    projection = {'tolerance': START_PROJECTION_TOLERANCE,
                  'max_correction': float(difference.max(initial=0.)),
                  'joint_indices': corrected.tolist(),
                  'measured': measured[corrected].tolist(),
                  'projected': projected[corrected].tolist()}
    info = {'choices': choices, 'selected': None, 'local_ik_attempts': 1,
            'start_projection': projection}
    if projection['max_correction'] > START_PROJECTION_TOLERANCE:
        info['rejected'] = 'measured start exceeds projection tolerance'
        return None, info
    if len(corrected):
        # IK already clips its seed. Keeping the measured, out-of-limit TCP
        # as the target origin would simply solve back to the invalid start.
        projection['measured_tcp'] = origin.tolist()
        origin = model.tcp(arm, q=projected, world=True)
        projection['projected_tcp'] = origin.tolist()
    directions = ([0., inward, 0.], [0., -inward, 0.], [1., 0., 0.], [-1., 0., 0.])
    for direction, sign in ((direction, sign) for direction in directions for sign in (1., -1.)):
        choice = {'horizontal_direction': direction, 'wrist_sign': sign}
        choices.append(choice)
        start, commands = projected.copy(), []
        for segment, axis in [('lateral', direction), ('up', [0., 0., 1.])]:
            for offset in linear:
                pose = origin.copy()
                pose[:3, 3] += offset * np.asarray(axis)
                status, next_q = model.ik(arm, pose, start, world=True, attempts=1,
                                          check_collision=False, lift=True)
                if status != 'Success':
                    choice['rejected'] = {'segment': segment, 'direction': axis,
                                          'offset_m': float(offset), 'status': status}
                    break
                start = next_q
                commands.append((segment, start[indices].copy(), start[lift_indices].copy()))
            if 'rejected' in choice:
                break
        if 'rejected' in choice:
            continue
        baseline, lift = start[indices].copy(), start[lift_indices].copy()
        lo, hi = model.limits[indices[-1]]
        if (hi - baseline[-1] if sign > 0 else baseline[-1] - lo) < angular.max():
            choice['rejected'] = {'segment': 'rotation', 'reason': 'wrist range insufficient'}
            continue
        for offset in angular:
            command = baseline.copy()
            command[-1] += sign * offset
            commands.append(('rotation', command, lift))
        # Keep the designed pulse time law and stretch only to joint/FK bounds.
        # Each segment begins/ends at rest; never reset it to lagging measured q.
        timed, previous = [], projected.copy()
        choice['timing'] = {}
        for segment in ('lateral', 'up', 'rotation'):
            rows = [(q, lift) for name, q, lift in commands if name == segment]
            path = {'position': np.vstack([previous[indices], *[q for q, _ in rows]]),
                    'lift': np.vstack([previous[lift_indices], *[lift for _, lift in rows]])}
            try:
                trajectory = retime(model, arm, path, profile=True)
            except PlanningFailed as error:
                choice['rejected'] = {'segment': segment, 'reason': error.reason, 'details': error.details}
                break
            timed.extend((segment, q, lift) for q, lift in zip(trajectory['position'][1:], trajectory['lift'][1:]))
            previous[indices], previous[lift_indices] = rows[-1]
            choice['timing'][segment] = {key: trajectory[key] for key in ('duration', 'command_motion_peaks')}
            if 'constraint_ratios' in trajectory:
                choice['timing'][segment]['constraint_ratios'] = trajectory['constraint_ratios']
        if 'rejected' in choice:
            continue
        _, collisions = model.check_path(arm, np.asarray([projected[indices], *[q for _, q, _ in timed]]),
                                         lift=np.asarray([projected[lift_indices], *[lift for _, _, lift in timed]]))
        if collisions:
            choice['rejected'] = {'reason': 'transport path collision', 'collisions': collisions}
            continue
        choice.update(selected=True, wrist_sign=sign)
        return timed, {**info, 'selected': choice, 'preflight_steps': len(timed)}
    return None, info


class Evaluator:
    def __init__(self, asset, robot, assets, work, root, device='cpu', views=None, diagnostic_shake=False):
        from table_1000.robots import load_robot_test, load_spec
        from table_1000.robots.kinematics import RobotModel
        from table_1000.sim.isaaclab.env import RobotEnv
        self.asset, self.assets = asset, Path(assets)
        self.diagnostic_shake = diagnostic_shake
        self.spec = load_spec(robot, root)
        self.settings = load_robot_test(robot, root)
        self.center = np.asarray(self.settings.WORK) + WORK_OFFSET
        self.scene = scene(asset, self.spec, self.settings)
        self.env = RobotEnv(self.scene, self.spec, assets, root=root, device=device,
                            views=views or {}, contact_watch=('robot', 'object'))
        self.model = RobotModel(self.spec, Path(work) / 'planning', root)

    def commanded_q(self, ctx, arm):
        """Continue held drive targets, preserving their static load error."""
        q = self.model.q.copy()
        targets = ctx.env.targets()
        q[self.model.arm_indices(arm)] = targets[arm][0]
        q[self.model.lift_indices()] = targets['lift'][0]
        return q

    def transport_start(self, ctx, arm):
        q = self.commanded_q(ctx, arm)
        return q, self.model.tcp(arm, q=q, world=True)

    def transport_reference(self, arm, ctx=None):
        """Find a carrying pose with room for the unchanged acceptance pulses.

        First keep the measured orientation at the configured work centre and
        nominal pick height. Use ordinary planning IK to obtain a branch, then
        require local continuation for the whole motion. The robot's declared
        test keyframe is the final reference. Only RRT may change branches.
        """
        choices = []
        if ctx is not None:
            from table_1000.experts.skills.common import IK_ATTEMPTS
            from table_1000.experts.skills.grasping import LIFT
            origin = ctx.tcp(arm).copy()
            origin[:2, 3] = self.center
            origin[2, 3] = np.asarray(ctx.skills().held[arm]['grasp']['pose'])[2, 3] + LIFT
            ctx.solver(arm).sync()
            status, goal = self.model.ik(arm, origin, self.model.q.copy(), world=True,
                                         attempts=IK_ATTEMPTS, lift=True)
            choice = {'kind': 'work centre at nominal pick height', 'ik_status': status,
                      'tcp': origin.tolist()}
            choices.append(choice)
            if status == 'Success':
                commands, plan = transport_plan(self.model, arm, origin, start=goal)
                choice['preflight'] = plan
                if commands is not None:
                    return goal, origin, commands, {'references': choices, **plan}
        indices = self.model.arm_indices(arm) + self.model.lift_indices()
        ready = self.model.vector(self.spec.keyframe(self.settings.ROBOT.get('keyframe', 'zero')))
        goal = self.model.q.copy()
        goal[indices] = ready[indices]
        origin = self.model.tcp(arm, q=goal, world=True)
        commands, plan = transport_plan(self.model, arm, origin, start=goal)
        choices.append({'kind': 'declared test keyframe', 'tcp': origin.tolist(), 'preflight': plan})
        return goal, origin, commands, {'references': choices, **plan}

    def transport(self, ctx, target, arm, phase):
        """Smooth TCP accelerations and wrist rotation derived from expert transport."""
        start, origin = self.transport_start(ctx, arm)
        commands, plan = transport_plan(self.model, arm, origin, start=start)
        ctx.note('transport preflight', **plan)
        points = ctx.geometry['object'].points(frame=target.body.split(':', 1)[1])
        reference = np.linalg.inv(ctx.tcp(arm)) @ ctx.poses()[target.body]
        initial_height = ctx.poses()[target.body][2, 3]
        trace = []

        def observe(c):
            relative = np.linalg.inv(c.tcp(arm)) @ c.poses()[target.body]
            row = metrics(reference, relative, points)
            touching = any(any(label == target.body or label.startswith(target.body + '.')
                               for label in contact['bodies'])
                           and any(label in ['robot.' + link for link in self.spec.arms[arm].finger_links]
                                   for label in contact['bodies'])
                           for contact in c.env.contacts.surface_contacts.values())
            row.update(step=c.steps, dropped=is_dropped(row['translation_m'],
                initial_height - c.poses()[target.body][2, 3], touching),
                tcp_pose=c.tcp(arm).tolist(), segment=phase['text'])
            trace.append(row)

        def result(completed=True):
            return {**classify(trace, completed=completed), **motion_report(trace, 1 / CONTROL_HZ),
                    'requested_envelope': TRANSPORT, 'preflight': plan}

        ctx.observers.append(observe)
        phase['text'] = ctx.env.contacts.phase = 'transport acceptance'
        observe(ctx)  # Include the measured entry pose before the first control step.
        try:
            if commands is None:
                goal, origin, reference_commands, reference_plan = self.transport_reference(arm, ctx)
                relocation = {'reference_tcp': origin.tolist(), 'goal': goal.tolist(),
                              'reference_preflight': reference_plan}
                plan['relocation'] = relocation
                if reference_commands is None:
                    ctx.fail('transport motion unreachable', arm=arm, preflight=plan)
                phase['text'] = 'transport relocation'
                ctx.note('transport relocation', **relocation)
                try:
                    # Reuse normal collision-checked RRT and its unchanged
                    # budgets. The observer spans departure, torso and arm
                    # motion; relocating never resets the grip reference.
                    ctx.solver(arm).move_to_pose_with_RRTConnect(None, goals=[goal], lift=True,
                        retime=lambda trajectory: retime(self.model, arm, trajectory, start=self.commanded_q(ctx, arm)))
                except PlanningFailed as error:
                    relocation['failure'] = {'reason': error.reason, 'details': error.details}
                    relocation['measured_q'] = self.model.q.tolist()
                    relocation['joints'] = list(self.model.joints)
                    if trace and not classify(trace)['held']:
                        return result(False)
                    if error.reason != 'tracking':
                        ctx.fail('transport motion unreachable', arm=arm, preflight=plan)
                    # Arrival is a reference for this test, not its criterion.
                    # Retain the tracking failure; the actual pose must still
                    # preflight and execute every unchanged acceptance stroke.
                if trace and not classify(trace)['held']:
                    return result(False)
                start, origin = self.transport_start(ctx, arm)
                commands, relocated_plan = transport_plan(self.model, arm, origin, start=start)
                relocation['actual_preflight'] = relocated_plan
                ctx.note('transport relocated preflight', **relocated_plan)
                if commands is None:
                    ctx.fail('transport motion unreachable', arm=arm, preflight=plan)
            # Execute only after the entire path has passed local continuation.
            # Keep the planned endpoint across segments, not lagging measured q.
            for segment, command, lift in commands:
                phase['text'] = 'transport rotation' if segment == 'rotation' else 'transport translation ' + segment
                ctx.step({arm: command[None], 'lift': lift[None]})
            phase['text'] = 'transport hold'
            ctx.hold(.5)
        finally:
            ctx.observers.remove(observe)
        return result()

    def shake(self, ctx, target, arm, phase):
        """Same wrist excursion as robot acceptance; monitor the entire motion."""
        indices = self.model.arm_indices(arm)
        baseline = self.model.q[indices].copy()
        wrist = indices[-1]
        lo, hi = self.model.limits[wrist]
        # Shift the oscillation centre inside the limits when the picked wrist is near an end.
        # This transition is monitored too; never skip the shake for a valid grasp.
        baseline[-1] = np.clip(baseline[-1], lo + SHAKE_AMPLITUDE, hi - SHAKE_AMPLITUDE)
        points = ctx.geometry['object'].points(frame=target.body.split(':', 1)[1])
        reference = np.linalg.inv(ctx.tcp(arm)) @ ctx.poses()[target.body]
        trace = []

        def observe(c):
            relative = np.linalg.inv(c.tcp(arm)) @ c.poses()[target.body]
            trace.append({'step': c.steps, 'slip_m': displacement(reference, relative, points),
                          'origin_drift_m': float(np.linalg.norm(relative[:3, 3] - reference[:3, 3])),
                          'wrist_rad': float(self.model.q[wrist])})

        ctx.observers.append(observe)
        phase['text'] = 'shake'
        ctx.env.contacts.phase = 'shake'
        try:
            for offset in SHAKE_OFFSETS:
                start = self.model.q[indices].copy()
                end = baseline.copy()
                end[-1] += offset * SHAKE_AMPLITUDE
                count = round(SHAKE_SECONDS * CONTROL_HZ)
                for step in range(1, count + 1):
                    ctx.step({arm: (start + (end - start) * step / count)[None]})
                if trace[-1]['slip_m'] > .03:
                    break
            ctx.hold(.5)
        finally:
            ctx.observers.remove(observe)
        worst = max(row['slip_m'] for row in trace)
        excursion = max(row['wrist_rad'] for row in trace) - min(row['wrist_rad'] for row in trace)
        completed = offset == SHAKE_OFFSETS[-1]
        return {'held': completed and worst <= SLIP_LIMIT and excursion >= 1.8 * SHAKE_AMPLITUDE,
                'completed': completed, 'max_slip_m': worst, 'wrist_excursion_rad': excursion,
                'segment_s': round(SHAKE_SECONDS * CONTROL_HZ) / CONTROL_HZ,
                'trace': trace}

    def trial(self, seed, center=None, size=REGION_SIZE, video=None):
        """Search geometric targets, resetting the same placement after a failed shake."""
        attempts, excluded = [], []
        while True:
            report = self._attempt(seed, center, size, video, excluded)
            attempts.append(report)
            print(f'{report.get("asset", "<asset>")} seed={seed} attempt={len(attempts)} {report["status"]} '
                  f'{report.get("failure", {}).get("reason", "")}', flush=True)
            if report['status'] in ('success', 'invalid_motion') or 'hold' not in report:
                break
            excluded.append(report['selected_target_index'])
            if len(excluded) >= report['target_count']:
                break
        result = attempts[0]
        result['first_attempt_timing'] = {key: result[key] for key in ('cpu_s', 'wall_s', 'sim_s')}
        result['preference_attempts'] = attempts[1:]
        for key in ('cpu_s', 'wall_s', 'sim_s'):
            result[key] = sum(attempt[key] for attempt in attempts)
        return result

    def _attempt(self, seed, center, size, video, excluded):
        import mplib.pymp
        from table_1000.experts.context import Context, ExpertFailed
        started = time.perf_counter()
        cpu_started = time.process_time()
        pose, sampled = placement(self.scene, self.assets, seed,
                                  self.center if center is None else center, size)
        self.env.reset({'object': pose})
        mplib.pymp.set_global_seed(seed + 1)
        ctx = Context(self.env, self.model, self.scene, self.assets, np.random.default_rng(seed))
        phase = {'text': 'settle'}
        if video:
            ctx.observers.append(lambda c: video.capture(
                f'{self.asset} seed={seed} excluded={excluded} {phase["text"]}'))
        report = {'asset': self.asset, 'seed': seed, 'placement': sampled,
                  'physics_dt_s': float(self.env.env.cfg.sim.dt), 'control_hz': CONTROL_HZ}
        try:
            self.env.contacts.phase = 'settle'
            ctx.hold(.5)
            skill = ctx.skills()
            report['candidate_policy'] = 'shared within-part order; gravity load selects current real-part heads; whole envelope fallback'
            body = skill.target('object').body
            settled = ctx.poses()[body]
            report['settled_pose'] = settled.tolist()
            report['settled_asset_pose'] = (settled @ np.linalg.inv(
                ctx.geometry['object'].rest[body.split(':', 1)[1]])).tolist()
            phase['text'] = 'pick'
            self.env.contacts.phase = 'pick'
            candidates = list(targets(skill))
            report['excluded_targets'], report['target_count'] = list(excluded), len(candidates)
            available = [item for i, item in enumerate(candidates) if i not in excluded]
            target, result = pick(skill, report, available)
            report['selected_target_index'] = next(i for i, item in enumerate(candidates) if item is target)
            settled = np.asarray(report['settled_pose'])
            grasp_local = np.linalg.inv(settled) @ np.asarray(result['grasp']['pose'])
            result['body_grasp_pose'] = grasp_local.tolist()
            report['pick'] = result
            result['family'] = grasp_family(report)
            report['hold'] = self.transport(ctx, target, result['arm'], phase)
            report['status'] = ('invalid_motion' if not report['hold']['motion_valid'] else
                                'success' if report['hold']['held'] else 'failed')
            if report['status'] == 'invalid_motion':
                report['failure'] = {'reason': 'transport motion exceeds envelope',
                                     'details': report['hold']['motion_violations']}
            elif not report['hold']['held']:
                reason = 'dropped' if report['hold']['dropped'] else 'unstable during transport'
                report['failure'] = {'reason': reason}
            if self.diagnostic_shake and report['status'] == 'success':
                report['diagnostic_shake'] = self.shake(ctx, target, result['arm'], phase)
        except ExpertFailed as error:
            reason = error.reason
            details = error.details
            if reason == 'no reachable grasp' and any(event['note'] == 'missed grasp' for event in ctx.events):
                # A failed closure can push a light object out of reach. That is
                # a physical grasp failure, not a placement without candidates.
                reason = 'object not held'
                details = {'subsequent_planning_failure': error.details}
            report.update(status='not_evaluated' if reason == 'transport motion unreachable' else 'failed',
                          failure={'reason': reason, 'details': details})
        except Exception:
            import traceback
            report.update(status='error', failure={'reason': 'execution error', 'traceback': traceback.format_exc()})
        finally:
            report['cpu_s'] = time.process_time() - cpu_started
            report['final_object_pose'] = self.env.object_pose('object')[0].tolist()
            report['events'] = ctx.events
            report['plans'] = [entry for solver in ctx.solvers.values() for entry in solver.log]
            report['contacts'] = self.env.contacts.intervals()
            report['sim_s'], report['wall_s'] = ctx.steps / CONTROL_HZ, time.perf_counter() - started
            for solver in ctx.solvers.values():
                solver.world.clear()
        return report

    def close(self):
        self.env.close()
