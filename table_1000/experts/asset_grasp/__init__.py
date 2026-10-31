"""First-attempt asset grasp test with the experts' own grasp skill.

One asset per environment: the expert policy picks it once, lifts it, then
runs three holding strokes (lateral, up, rotation) near the expert P99
transport envelope. Strokes keep the lift fixed and let the whole arm add
companion TCP motion, as experts do; the fixed direction sets, TOPPRA timing
and joint clearance live in :func:`transport_plan` and ``transport.py``.
Acceptance reads only the held-point slip, relative rotation and drops;
measured strength is recorded, never used to exclude a trial. The protocol
and result fields are specified in docs/specs/grasp-tests.md.
"""

from collections import Counter
import math
from pathlib import Path
import time
import types

import numpy as np
from scipy.spatial.transform import Rotation

from table_1000.scenes.layout import Layout
from table_1000.sim import CONTROL_HZ
from table_1000.experts.asset_grasp.stability import (TRANSPORT, classify, is_dropped, metrics, motion_report)
from table_1000.experts.asset_grasp.transport import coupled_twist, retime, retime_stroke
from table_1000.experts.solver import PlanningFailed
from table_1000.experts.skills.targets import targets, target_groups

REGION_SIZE = .10
WORK_OFFSET = (.05, 0.)  # keep the near edge of the square clear of the folded-arm workspace
SLIP_LIMIT = .005
SHAKE_AMPLITUDE = .4
SHAKE_SECONDS = .5
SHAKE_OFFSETS = (1, -1, 1, -1, 0)

# One tenth of the existing 0.01 joint tracking tolerance; only the measured
# preflight start may be projected, never an IK result or a joint limit.
START_PROJECTION_TOLERANCE = .001

def pick(skill, report, candidates=None, excluded=()):
    """Choose real part heads, keeping full-list IDs when excluding prior attempts."""
    from table_1000.experts.context import ExpertFailed
    ctx = skill.ctx
    report['target_attempts'] = []
    candidates = list(targets(skill) if candidates is None else candidates)
    available = [target for index, target in enumerate(candidates) if index not in excluded]
    if not available:
        ctx.fail('no reachable grasp', candidates_tried=0)
    attempt = None
    try:
        groups = target_groups(available)
        for number, group in enumerate(groups):
            target = group[0]
            source_target = None
            attempt = {'shape': type(target.shape).__name__, 'parameters': vars(target.shape),
                       'parts': list(target.names), 'fallback': getattr(target, 'fallback', False)}
            report['target_attempts'].append(attempt)
            try:
                if skill.thin_geometry(target) is None:
                    result = skill.grasp(group, approach=(0., 0., -1.), then=(0., 0., .10), base=False,
                                         closure_only_record=True)
                else:
                    source_target = target
                    target, result = skill.grasp_thin(target, height=.10, base=False, closure_only_record=True)
                break
            except ExpertFailed as error:
                attempt['failure'] = {'reason': error.reason, 'details': error.details}
                if error.reason != 'no reachable grasp' or number == len(groups) - 1:
                    raise
        target = skill.held[result['arm']]['target']
        report['selected_target_index'] = next(i for i, item in enumerate(candidates)
                                             if item is (source_target if source_target is not None else target))
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
            if 'preparation' in result:
                skill.prepare_thin_lift(target, result)
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
            try:
                skill.release(result['arm'])
            except ExpertFailed as error:
                attempt['release_failure'] = {'reason': error.reason, 'details': error.details}
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
    placement = settings.ROBOT
    return types.SimpleNamespace(
        TABLE=settings.TABLE, OBJECTS={'object': {'asset': asset}},
        ROBOT=spec.name, ROBOTS={spec.name: {'asset': f'robots/{spec.name}', 'position': placement['position'],
               'yaw': placement['yaw'],
               'joints': spec.keyframe(placement.get('keyframe', 'zero'))}})


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
    hold = report.get('hold', {})
    return report.get('lift', {}).get('rise_m', .05) < .05 or hold.get('completed', False) \
        or hold.get('held') is False or hold.get('dropped', False)


def acceptance_report(report):
    """Read historical results under the current holding-only acceptance rule."""
    hold = report.get('hold')
    if hold is not None and report['status'] not in ('success', 'failed', 'not_evaluated', 'error'):
        report = {**report, 'status': 'success' if hold['held'] else 'failed'}
        if hold['held']:
            report.pop('failure', None)
        else:
            report['failure'] = {'reason': 'dropped' if hold.get('dropped') else 'unstable during transport'}
    if failure_reason(report) == 'no reachable grasp':
        report = {**report, 'status': 'not_evaluated'}
    return report


def summarize(reports, expected):
    reports = [acceptance_report(report) for report in reports]
    success = sum(r['status'] == 'success' for r in reports)
    unassessed = Counter(failure_reason(r) for r in reports if r['status'] == 'not_evaluated')
    assessed = expected - sum(unassessed.values())
    failures = Counter(failure_reason(r) for r in reports if r['status'] not in ('success', 'not_evaluated'))
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
    lifted_transport = sum(lift_transport_success(report) for report in reports)
    return {'expected': expected, 'reported': len(reports), 'success': success,
            'lift_transport': {'success': lifted_transport, 'assessed': assessed,
                               'rate': lifted_transport / assessed if assessed else None,
                               'criterion': 'first lift >= 0.05 m above settled placement and complete transport without drop'},
            'stable_transport': {'success': success, 'assessed': assessed,
                                 'rate': success / assessed if assessed else None,
                                 'criterion': 'existing 5 mm / 5 degree stability limits'},
            'assessed': assessed, 'not_evaluated': sum(unassessed.values()),
            'not_evaluated_reasons': dict(unassessed),
            'closure_check': closure_statistics(reports),
            'picked': sum(r.get('lift', {}).get('rise_m', 0.) >= .05 for r in reports),
            'first_attempt_success': success,
            'attempts': len(attempts), 'rate': success / assessed if assessed else None,
            'initial_unreachable': sum(failure_reason(r)
                                       == 'no reachable grasp' for r in reports),
            'failures': dict(failures), 'preference_data': {
                'by_grasp': by_grasp, 'not_evaluated_by_grasp': dict(not_evaluated)},
            'status': 'PASS' if success == assessed and assessed else 'FAIL'}


def lift_transport_success(report):
    """First real lift and completed transport without a drop (#55 primary metric).

    The existing status and strict stability criterion remain independently
    available. A pickup from the floor after a failed preparation cannot
    replace lifting off the original table. Missing measurements never pass.
    """
    hold = report.get('hold', {})
    lift = report.get('lift', {})
    settled, lifted = report.get('settled_pose'), lift.get('object_pose')
    return (report.get('status') not in ('not_evaluated', 'error')
            and settled is not None and lifted is not None
            and lifted[2][3] - settled[2][3] >= .05
            and lift.get('rise_m', 0.) >= .05
            and bool(hold.get('completed')) and not hold.get('dropped', True))


def transport_plan(model, arm, origin, start=None):
    """Preflight fixed-lift arm strokes and retain the strongest feasible choice.

    Translation may rotate the TCP and rotation may translate it. TOPPRA
    times each outward stroke, which stops and exactly retraces its path. All
    choices retain the same excursion, joint and TCP limits. This selection
    is used only by asset tests, never by grasp candidates or expert motion.
    """
    indices, lift_indices = model.arm_indices(arm), model.lift_indices()
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
            'start_projection': projection, 'lift_locked': True}
    if projection['max_correction'] > START_PROJECTION_TOLERANCE:
        info['rejected'] = 'measured start exceeds projection tolerance'
        return None, info
    if len(corrected):
        # IK already clips its seed. Keeping the measured, out-of-limit TCP
        # as the target origin would simply solve back to the invalid start.
        projection['measured_tcp'] = origin.tolist()
        origin = model.tcp(arm, q=projected, world=True)
        projection['projected_tcp'] = origin.tolist()
    from table_1000.experts.solver import ARRIVAL_TOLERANCE
    clearance = np.minimum(projected[indices] - model.limits[indices, 0],
                           model.limits[indices, 1] - projected[indices])
    info['joint_clearance'] = clearance.tolist()
    if np.min(clearance) < ARRIVAL_TOLERANCE:
        info['rejected'] = {'reason': 'transport start lacks joint tracking clearance',
                            'joint_indices': np.asarray(indices)[clearance < ARRIVAL_TOLERANCE].tolist(),
                            'clearance': clearance.tolist(), 'required': ARRIVAL_TOLERANCE}
        return None, info
    # Each stroke returns to the same drive target at rest, so equivalent
    # directions can be preflighted independently without repeating IK work.
    # A fixed axis set keeps intensity selection independent of grasp ranking.
    rotation_axes = ([1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                     *([1., y, z] for y in (-1., 0., 1.) for z in (-1., 0., 1.) if y or z),
                     [0., 1., -1.], [0., 1., 1.])
    rotation_axes = (*rotation_axes, *([2., y, z] for y in (-1., 1.) for z in (-1., 1.)),
                     *([x, 2., z] for x in (-1., 1.) for z in (-1., 1.)),
                     *([x, y, 2.] for x in (-1., 1.) for y in (-1., 1.)))
    directions = ([0., inward, 0.], [0., -inward, 0.], [1., 0., 0.], [-1., 0., 0.],
                  # The circle skips the four axis directions already listed above.
                  *([np.cos(angle), np.sin(angle), 0.] for angle in np.arange(24) * np.pi / 12
                    if round(angle / (np.pi / 2), 6) % 1))
    upward = ([0., 0., 1.], *([x, y, 1.] for x in (-1., 1.) for y in (-1., 1.)))
    stroke = 2 * TRANSPORT['angular_velocity_rad_s'] ** 2 / TRANSPORT['angular_acceleration_rad_s2']
    selected, commands = {'timing': {}}, []
    for segment, axes, excursion in [('lateral', directions, REGION_SIZE / 2),
                                    ('up', upward, REGION_SIZE / 2),
                                    ('rotation', rotation_axes, stroke)]:
        best = None
        for raw_axis in axes:
            axis = np.asarray(raw_axis) / np.linalg.norm(raw_axis)
            distance = excursion / axis[2] if segment == 'up' else excursion
            for sign, mixed in ((sign, mixed) for sign in ((1., -1.) if segment == 'rotation' else (1.,))
                                for mixed in (True, False)):
                choice = {'segment': segment, 'axis': (sign * axis).tolist(), 'mixed': mixed}
                choices.append(choice)
                world_axis = origin[:3, :3] @ (axis * sign) if segment == 'rotation' else axis
                companion = (coupled_twist(model, arm, projected, world_axis, angular=segment == 'rotation')
                             if mixed else np.zeros(3))
                choice['companion_per_unit'] = companion.tolist()
                start, rows = projected.copy(), [projected.copy()]
                for offset in np.linspace(0., distance, 33)[1:]:
                    pose = origin.copy()
                    if segment == 'rotation':
                        pose[:3, :3] = origin[:3, :3] @ Rotation.from_rotvec(sign * axis * offset).as_matrix()
                        pose[:3, 3] += companion * offset
                    else:
                        pose[:3, 3] += offset * axis
                        pose[:3, :3] = Rotation.from_rotvec(companion * offset).as_matrix() @ origin[:3, :3]
                    status, next_q = model.ik(arm, pose, start, world=True, attempts=1,
                                              check_collision=False, lift=False)
                    if status != 'Success':
                        choice['rejected'] = {'segment': segment, 'offset': float(offset), 'status': status}
                        break
                    start = next_q
                    rows.append(start.copy())
                if 'rejected' in choice:
                    continue
                rows = np.asarray(rows)
                try:
                    trajectory = retime_stroke(model, arm, {'position': rows[:, indices], 'lift': rows[:, lift_indices]},
                                               distance, angular=segment == 'rotation')
                except PlanningFailed as error:
                    choice['rejected'] = {'segment': segment, 'reason': error.reason, 'details': error.details}
                    continue
                # An interior hard stop also truncates the physical velocity,
                # even when the start and sampled command remain in bounds.
                # Keep the same tracking budget along the complete stroke.
                path_clearance = np.minimum(trajectory['position'] - model.limits[indices, 0],
                                            model.limits[indices, 1] - trajectory['position']).min(axis=0)
                choice['joint_clearance'] = path_clearance.tolist()
                if np.min(path_clearance) < ARRIVAL_TOLERANCE:
                    choice['rejected'] = {'reason': 'transport path lacks joint tracking clearance',
                                          'joint_indices': np.asarray(indices)[path_clearance < ARRIVAL_TOLERANCE].tolist(),
                                          'clearance': path_clearance.tolist(), 'required': ARRIVAL_TOLERANCE}
                    continue
                _, collisions = model.check_path(arm, trajectory['position'], lift=trajectory['lift'])
                if collisions:
                    choice['rejected'] = {'reason': 'transport path collision', 'collisions': collisions}
                    continue
                timing = {key: trajectory[key] for key in ('duration', 'command_motion_peaks', 'constraint_ratios')}
                choice['timing'] = timing
                key = 'angular_acceleration_rad_s2' if segment == 'rotation' else 'linear_acceleration_m_s2'
                strength = timing['command_motion_peaks'][key] / TRANSPORT[key]
                if best is None or strength > best[0]:
                    best = strength, trajectory, choice, sign
            # Inspect every fixed direction: a barely feasible companion-heavy
            # path should not hide a stronger, better conditioned direction.
        if best is None:
            info['rejected'] = {'segment': segment, 'reason': 'no complete collision-free arm stroke'}
            return None, info
        _, trajectory, choice, sign = best
        choice['selected'] = True
        selected['timing'][segment] = choice['timing']
        selected.setdefault('paths', {})[segment] = {key: choice[key] for key in
                                                       ('axis', 'mixed', 'companion_per_unit')}
        if segment == 'lateral':
            selected['horizontal_direction'] = choice['axis']
        if segment == 'rotation':
            selected.update(wrist_sign=sign, rotation_axis_tcp=choice['axis'])
        commands.extend((segment, q, lift) for q, lift in zip(trajectory['position'][1:], trajectory['lift'][1:]))
    return commands, {**info, 'selected': selected, 'preflight_steps': len(commands)}


class Evaluator:
    def __init__(self, asset, robot, assets, work, root, device='cpu', views=None, diagnostic_shake=False,
                 collect_preferences=False):
        from table_1000.robots import load_robot_test, load_spec
        from table_1000.robots.kinematics import RobotModel
        from table_1000.sim.isaaclab.env import RobotEnv
        self.asset, self.assets = asset, Path(assets)
        self.diagnostic_shake = diagnostic_shake
        self.collect_preferences = collect_preferences
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
                from table_1000.experts.solver import ARRIVAL_TOLERANCE
                if commands is not None and min(plan.get('joint_clearance', [np.inf])) >= 2 * ARRIVAL_TOLERANCE:
                    return goal, origin, commands, {'references': choices, **plan}
                if commands is not None:
                    choice['rejected'] = 'reference needs arrival error plus transport tracking clearance'
        indices = self.model.arm_indices(arm) + self.model.lift_indices()
        ready = self.model.vector(self.spec.keyframe(self.settings.ROBOT.get('keyframe', 'zero')))
        goal = self.model.q.copy()
        goal[indices] = ready[indices]
        origin = self.model.tcp(arm, q=goal, world=True)
        commands, plan = transport_plan(self.model, arm, origin, start=goal)
        choices.append({'kind': 'declared test keyframe', 'tcp': origin.tolist(), 'preflight': plan})
        return goal, origin, commands, {'references': choices, **plan}

    def transport(self, ctx, target, arm, phase):
        """Arm-only TCP strokes under the measured expert transport envelope."""
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
        """Evaluate once; optionally reset failed holds to collect preference data."""
        attempts, excluded = [], []
        while True:
            report = self._attempt(seed, center, size, video, excluded)
            attempts.append(report)
            print(f'{report.get("asset", "<asset>")} seed={seed} attempt={len(attempts)} {report["status"]} '
                  f'{report.get("failure", {}).get("reason", "")}', flush=True)
            if not self.collect_preferences or report['status'] == 'success' or 'hold' not in report:
                break
            excluded.append(report['selected_target_index'])
            if len(excluded) >= report['target_count']:
                break
        result = attempts[0]
        result['first_attempt_timing'] = {key: result[key] for key in ('cpu_s', 'wall_s', 'sim_s')}
        result['preference_attempts'] = attempts[1:]
        result['preference_timing'] = {key: sum(attempt[key] for attempt in attempts[1:])
                                       for key in ('cpu_s', 'wall_s', 'sim_s')}
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
            target, result = pick(skill, report, candidates, excluded)
            settled = np.asarray(report['settled_pose'])
            grasp_local = np.linalg.inv(settled) @ np.asarray(result['grasp']['pose'])
            result['body_grasp_pose'] = grasp_local.tolist()
            report['pick'] = result
            result['family'] = grasp_family(report)
            report['hold'] = self.transport(ctx, target, result['arm'], phase)
            report['status'] = 'success' if report['hold']['held'] else 'failed'
            if not report['hold']['held']:
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
            report.update(status='not_evaluated' if reason in ('transport motion unreachable', 'no reachable grasp') else 'failed',
                          failure={'reason': reason, 'details': details})
        except Exception:
            import traceback
            report.update(status='error', failure={'reason': 'execution error', 'traceback': traceback.format_exc()})
        finally:
            report['acceptance'] = {'lift_transport': lift_transport_success(report),
                                    'stable_transport': report.get('status') == 'success'}
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
