"""Preserve transport pulse timing; use TOPPRA only for relocation paths.

Stretch a designed rest-to-rest pulse by the smallest feasible control-step
count under joint and FK limits. Record the active constraints so a joint-
limited segment is not reported as having reached the expert P99 intensity.
"""

import numpy as np
from scipy.interpolate import CubicSpline

from table_1000.sim import CONTROL_HZ
from table_1000.experts.asset_grasp.stability import MOTION_LIMITS, MOTION_TOLERANCE, motion
from table_1000.experts.solver import PlanningFailed


def retime(model, arm, result, start=None, *, profile=False):
    """Return a Solver.follow result at exactly the control period; no state mutation.

    With profile=True the input already samples the designed rest-to-rest TCP
    time law. Preserve that law rather than running a second time optimization
    over its out-and-back joint path. Increase duration only as required by
    sampled joint/FK bounds; the first feasible control-step count is used.
    Relocation paths without a time law retain the robot's TOPPRA parameterization.
    """
    indices, lifts = model.arm_indices(arm), model.lift_indices()
    positions = np.asarray(result['position'])
    lift = np.asarray(result.get('lift', np.tile(model.q[lifts], (len(positions), 1))))
    path = np.c_[lift, positions]
    if start is not None:
        # A loaded position drive rests short of its target. Starting at the
        # measured q would step that target backwards and release its preload.
        path = np.vstack([start[lifts + indices], path])
    planner = model.planners[arm]
    if profile:
        curve = CubicSpline(np.linspace(0., 1., len(path)), path, bc_type='clamped')
        count = len(path) - 1
        limits = MOTION_LIMITS
    else:
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
    lower, upper = count - 1, None
    constraints = {}
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
            upper = count
            if not profile or upper - lower <= 1:
                break
            count = (lower + upper) // 2
        else:
            lower = count
            count = ((lower + upper + 1) // 2 if upper is not None
                     else max(count + 1, int(np.ceil(count * scale))))
    if np.any(rows < model.limits[lifts + indices, 0] - 1e-6) or np.any(rows > model.limits[lifts + indices, 1] + 1e-6):
        raise PlanningFailed(None, 'time_parameterization', error='transport path exceeds joint position limits')
    seconds = count / CONTROL_HZ
    return {**result, 'position': rows[:, len(lifts):], 'lift': rows[:, :len(lifts)],
            'time': samples * seconds, 'velocity': curve(samples, 1)[:, len(lifts):] / seconds,
            'acceleration': curve(samples, 2)[:, len(lifts):] / seconds ** 2,
            'duration': count / CONTROL_HZ, 'command_motion_peaks': peaks,
            'constraint_ratios': constraints}
