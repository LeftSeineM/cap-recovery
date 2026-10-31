"""Time asset-test transport within joint and TCP limits.

Use TOPPRA for outward arm strokes and relocation paths, and verify sampled
joint/FK bounds at the control rate. Record active constraints so a joint-
limited segment is not reported as having reached the expert P99 intensity.
"""

import numpy as np
from scipy.interpolate import CubicSpline

from table_1000.sim import CONTROL_HZ
from table_1000.experts.evals.asset_grasp.stability import MOTION_LIMITS, MOTION_TOLERANCE, motion
from table_1000.experts.solver import PlanningFailed


def retime(model, arm, result, start=None):
    """Return a relocation path at exactly the control period; no state mutation.

    Keep the robot's TOPPRA parameterization and stretch it until sampled
    joint and FK bounds hold.
    """
    indices, lifts = model.arm_indices(arm), model.lift_indices()
    positions = np.asarray(result['position'])
    lift = np.asarray(result.get('lift', np.tile(model.q[lifts], (len(positions), 1))))
    path = np.c_[lift, positions]
    if start is not None:
        # A loaded position drive rests short of its target. Starting at the
        # measured q would step that target backwards and release its preload.
        path = path.copy()
        path[0] = start[lifts + indices]
    planner = model.planners[arm]
    # Consecutive duplicates carry no path information and can make TOPP singular.
    path = path[np.r_[True, np.any(np.diff(path, axis=0) != 0., axis=1)]]
    if len(path) < 2:
        return {**result, 'duration': (len(positions) - 1) / CONTROL_HZ, 'command_motion_peaks': {}}
    try:
        times, rows, _, _, duration = planner.TOPP(path, 1 / CONTROL_HZ)
    except RuntimeError as error:
        raise PlanningFailed(None, 'time_parameterization', error=str(error)) from error
    curve = CubicSpline(times / duration, rows, bc_type='clamped')
    count = max(len(positions) - 1, int(np.ceil(duration * CONTROL_HZ)), 2)
    limits = {key: value - MOTION_TOLERANCE[key] for key, value in MOTION_LIMITS.items()}
    while True:
        samples = np.linspace(0., 1., count + 1)
        rows = curve(samples)
        padded = np.vstack([rows[0], rows, rows[-1]])
        velocity = np.diff(padded, axis=0) * CONTROL_HZ
        acceleration = np.diff(velocity, axis=0) * CONTROL_HZ
        configurations = np.tile(model.q, (len(padded), 1))
        configurations[:, lifts + indices] = padded
        poses = np.asarray([model.tcp(arm, q=q, world=True) for q in configurations])
        peaks = {key: float(np.max(value)) for key, value in motion(poses, 1 / CONTROL_HZ).items()
                 if key in limits}
        joint_velocity = np.max(np.abs(velocity) / planner.joint_vel_limits, axis=0)
        joint_acceleration = np.max(np.abs(acceleration) / planner.joint_acc_limits, axis=0)
        constraints = {'joint_velocity': joint_velocity.tolist(), 'joint_acceleration': joint_acceleration.tolist(),
                       'tcp': {key: value / limits[key] for key, value in peaks.items()}}
        scale = max(np.max(joint_velocity), np.sqrt(np.max(joint_acceleration)),
                    *(value / limits[key] if 'velocity' in key else np.sqrt(value / limits[key])
                      for key, value in peaks.items()))
        if scale <= 1. + 1e-6:
            break
        count = max(count + 1, int(np.ceil(count * scale)))
    if np.any(rows < model.limits[lifts + indices, 0] - 1e-6) or np.any(rows > model.limits[lifts + indices, 1] + 1e-6):
        raise PlanningFailed(None, 'time_parameterization', error='transport path exceeds joint position limits')
    seconds = count / CONTROL_HZ
    return {**result, 'position': rows[:, len(lifts):], 'lift': rows[:, :len(lifts)],
            'time': samples * seconds, 'velocity': curve(samples, 1)[:, len(lifts):] / seconds,
            'acceleration': curve(samples, 2)[:, len(lifts):] / seconds ** 2,
            'duration': count / CONTROL_HZ, 'command_motion_peaks': peaks,
            'constraint_ratios': constraints}


def retime_stroke(model, arm, result, excursion, *, angular=False):
    """TOPPRA on an outward stroke; stop, then exactly retrace it.

    Append the Cartesian path coordinate as a virtual joint so the existing
    TOPPRA constraints bound its speed and acceleration alongside every real
    joint. FK checks also bound the other TCP components and interpolation
    error. The lift stays fixed throughout the locally solved arm path.
    """
    import toppra as ta
    from toppra import algorithm, constraint

    # Centre the requested 90%-100% band, leaving tracking room without
    # changing the measured envelope or its existing discrete tolerance.
    limits = {key: .95 * value for key, value in MOTION_LIMITS.items()}
    indices, lifts = model.arm_indices(arm), model.lift_indices()
    planner = model.planners[arm]
    path = np.c_[result['lift'], result['position']]
    coordinate = np.linspace(0., excursion, len(path))
    curve = ta.SplineInterpolator(np.linspace(0., 1., len(path)), np.c_[path, coordinate])
    velocity_key = 'angular_velocity_rad_s' if angular else 'linear_velocity_m_s'
    acceleration_key = 'angular_acceleration_rad_s2' if angular else 'linear_acceleration_m_s2'
    constraints = [
        constraint.JointVelocityConstraint(np.r_[planner.joint_vel_limits, limits[velocity_key]]),
        constraint.JointAccelerationConstraint(np.r_[planner.joint_acc_limits, limits[acceleration_key]])]
    trajectory = algorithm.TOPPRA(constraints, curve, parametrizer='ParametrizeConstAccel').compute_trajectory(0., 0.)
    if trajectory is None:
        raise PlanningFailed(None, 'time_parameterization', error='TOPPRA could not parameterize transport stroke')
    seconds = trajectory.duration
    while True:
        count = max(2, int(np.ceil(seconds * CONTROL_HZ)))
        # Sample the continuous time law on the real control grid and hold its
        # endpoint for the remaining fraction of a period. Uniformly stretching
        # short strokes to the next full step needlessly removes peak intensity.
        times = np.minimum(np.arange(count + 1) / CONTROL_HZ * trajectory.duration / seconds,
                           trajectory.duration)
        progress = trajectory(times)[:, -1] / excursion
        # Smooth acceleration switches over two control periods without
        # cutting either endpoint or changing the geometric IK path. Rest
        # padding preserves exact full excursion before retracing.
        progress = np.convolve(np.pad(progress, (2, 2), mode='edge'), [.25, .5, .25], mode='valid')
        outbound = curve(progress)[:, :-1]
        rows = np.vstack([outbound, outbound[-2::-1]])
        padded = np.vstack([rows[0], rows, rows[-1]])
        velocity = np.diff(padded, axis=0) * CONTROL_HZ
        acceleration = np.diff(velocity, axis=0) * CONTROL_HZ
        configurations = np.tile(model.q, (len(padded), 1))
        configurations[:, lifts + indices] = padded
        poses = np.asarray([model.tcp(arm, q=q, world=True) for q in configurations])
        peaks = {key: float(np.max(value)) for key, value in motion(poses, 1 / CONTROL_HZ).items()
                 if key in limits}
        joint_velocity = np.max(np.abs(velocity) / planner.joint_vel_limits, axis=0)
        joint_acceleration = np.max(np.abs(acceleration) / planner.joint_acc_limits, axis=0)
        scale = max(np.max(joint_velocity), np.sqrt(np.max(joint_acceleration)),
                    *(value / limits[key] if 'velocity' in key else np.sqrt(value / limits[key])
                      for key, value in peaks.items()))
        if scale <= 1. + 1e-6:
            break
        seconds = max(seconds + 1e-5, seconds * scale)
    if np.any(rows < model.limits[lifts + indices, 0] - 1e-6) or np.any(rows > model.limits[lifts + indices, 1] + 1e-6):
        raise PlanningFailed(None, 'time_parameterization', error='transport path exceeds joint position limits')
    names = [model.joints[i] for i in lifts + indices]
    constraint_ratios = {'joint_velocity': joint_velocity.tolist(), 'joint_acceleration': joint_acceleration.tolist(),
                         'tcp': {key: value / limits[key] for key, value in peaks.items()}, 'joints': names}
    return {**result, 'position': rows[:, len(lifts):], 'lift': rows[:, :len(lifts)],
            'duration': 2 * (len(outbound) - 1) / CONTROL_HZ, 'command_motion_peaks': peaks, 'constraint_ratios': constraint_ratios}


def coupled_twist(model, arm, q, direction, *, angular=False):
    """Allow the companion TCP component instead of cancelling whole-arm motion.

    Solve only the six-dimensional instantaneous acceleration allocation. The
    resulting short Cartesian screw still goes through the ordinary local IK,
    TOPPRA, full FK limits and held-object collision preflight before execution.
    """
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation

    indices = model.arm_indices(arm)
    origin = model.tcp(arm, q=q, world=True)
    columns = []
    for index in indices:
        offset = q.copy()
        offset[index] += 1e-5
        pose = model.tcp(arm, q=offset, world=True)
        columns.append(np.r_[(pose[:3, 3] - origin[:3, 3]) / 1e-5,
                             Rotation.from_matrix(pose[:3, :3] @ origin[:3, :3].T).as_rotvec() / 1e-5])
    jacobian = np.asarray(columns).T
    primary, companion = (jacobian[3:], jacobian[:3]) if angular else (jacobian[:3], jacobian[3:])
    _, _, basis = np.linalg.svd(np.asarray(direction)[None])
    perpendicular = basis[1:] @ primary
    left, singular, _ = np.linalg.svd(perpendicular, full_matrices=False)
    perpendicular = left[:, singular > 1e-8].T @ perpendicular
    limit = MOTION_LIMITS['linear_acceleration_m_s2' if angular else 'angular_acceleration_rad_s2']
    acceleration = model.planners[arm].joint_acc_limits[len(model.lift_indices()):]
    constraints = [{'type': 'ineq', 'fun': lambda value: limit ** 2
                    - np.sum((companion @ value) ** 2)}]
    if len(perpendicular):
        constraints.append({'type': 'eq', 'fun': lambda value: perpendicular @ value})
    result = minimize(lambda value: -direction @ primary @ value, np.zeros(len(indices)),
                      bounds=[(-float(value), float(value)) for value in acceleration], method='SLSQP',
                      constraints=constraints,
                      options={'ftol': 1e-10, 'maxiter': 100})
    strength = direction @ primary @ result.x
    if not result.success or strength <= 1e-6:
        return np.zeros(3)
    # Reach the requested primary load with the smallest companion load,
    # rather than spending surplus joint capacity on unnecessary rotation.
    primary_limit = MOTION_LIMITS['angular_acceleration_rad_s2' if angular else 'linear_acceleration_m_s2']
    target = min(strength, .95 * primary_limit)
    refined = minimize(lambda value: np.sum((companion @ value / limit) ** 2), result.x,
                       bounds=[(-float(value), float(value)) for value in acceleration], method='SLSQP',
                       constraints=[{'type': 'eq', 'fun': lambda value: primary @ value - direction * target}],
                       options={'ftol': 1e-10, 'maxiter': 100})
    if refined.success:
        return companion @ refined.x / target
    return companion @ result.x / strength
