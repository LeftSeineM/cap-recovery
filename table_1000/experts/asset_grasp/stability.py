"""Held-point translation and relative rotation, independent of object size."""

import numpy as np
from scipy.spatial.transform import Rotation

TRANSLATION_LIMIT = .005
ROTATION_LIMIT = np.radians(5.)
DROP_DISTANCE = .05

# P99 of 530 lifted rigid-object intervals in 264 successful 000002 episodes.
# Derivation and source intervals: docs/experiments/20261026-asset-grasp/README.md.
TRANSPORT = {'linear_acceleration_m_s2': 1.1286486148834252,
             'angular_velocity_rad_s': 1.2276920124649078,
             'angular_acceleration_rad_s2': 3.0087368790406877}

# A 50 mm raised-cosine stroke implies this existing peak translation speed.
MOTION_LIMITS = {**TRANSPORT, 'linear_velocity_m_s': np.sqrt(.025 * TRANSPORT['linear_acceleration_m_s2'])}
# One control period of phase error: Δv <= a*dt, Δa <= jerk*dt.
# Raised cosine has jerk_peak = a_peak*sqrt(a_peak/amplitude).
MOTION_TOLERANCE = {
    'linear_velocity_m_s': TRANSPORT['linear_acceleration_m_s2'] / 25,
    'linear_acceleration_m_s2': TRANSPORT['linear_acceleration_m_s2'] ** 1.5 / np.sqrt(.025) / 25,
    'angular_velocity_rad_s': TRANSPORT['angular_acceleration_rad_s2'] / 25,
    'angular_acceleration_rad_s2': TRANSPORT['angular_acceleration_rad_s2'] ** 2
        / TRANSPORT['angular_velocity_rad_s'] / 25}


def motion_report(trace, dt):
    """Per-segment TCP derivatives, including entry/exit samples at each boundary.

    Relocation is part of acceptance too. A full-trace check catches transitions
    even when a segment is shorter than the three samples needed for acceleration.
    """
    if len(trace) < 3:
        return {'motion_peaks': {}, 'segment_motion_peaks': {}, 'motion_valid': True,
                'motion_violations': {}}
    poses = np.asarray([row['tcp_pose'] for row in trace])
    def peaks(rows):
        return {key: float(np.max(value)) for key, value in motion(rows, dt).items() if key in MOTION_LIMITS}
    overall, segments = peaks(poses), {}
    for segment in dict.fromkeys(row['segment'] for row in trace):
        indices = [i for i, row in enumerate(trace) if row['segment'] == segment]
        segments[segment] = peaks(poses[max(0, indices[0] - 2):min(len(poses), indices[-1] + 3)])
    violations = {key: value for key, value in overall.items()
                  if value > MOTION_LIMITS[key] + MOTION_TOLERANCE[key]}
    return {'motion_peaks': overall, 'segment_motion_peaks': segments,
            'motion_valid': not violations, 'motion_violations': violations}


def is_dropped(translation, height_loss, finger_contact):
    """Loss of the hand plus 5 cm separation/fall; rotation while touching is not a drop."""
    return bool(not finger_contact and (translation > DROP_DISTANCE or height_loss > DROP_DISTANCE))


def metrics(reference, current, points):
    """Both transforms are T_tcp_body. Track the body point initially at TCP zero."""
    anchor = -reference[:3, :3].T @ reference[:3, 3]
    translation = np.linalg.norm(current[:3, :3] @ anchor + current[:3, 3])
    angle = Rotation.from_matrix(current[:3, :3] @ reference[:3, :3].T).magnitude()
    before = points @ reference[:3, :3].T + reference[:3, 3]
    after = points @ current[:3, :3].T + current[:3, 3]
    return {'translation_m': float(translation), 'rotation_rad': float(angle),
            'vertex_slip_m': float(np.linalg.norm(after - before, axis=1).max())}


def classify(trace, completed=True):
    """A transient failure stays a failure, even if the object returns to its pose."""
    maximum = {key: max(row[key] for row in trace)
               for key in ('translation_m', 'rotation_rad', 'vertex_slip_m')}
    dropped = any(row['dropped'] for row in trace)
    held = completed and not dropped and maximum['translation_m'] <= TRANSLATION_LIMIT \
        and maximum['rotation_rad'] <= ROTATION_LIMIT
    return {'held': bool(held), 'dropped': dropped, 'completed': completed,
            **{'max_' + key: value for key, value in maximum.items()}, 'trace': trace}


def pulse(acceleration, *, speed=None, stroke=None, hz=25):
    """Raised cosine, rest to rest. Round duration up so sampled limits never increase.

    x=A(1-cos(w*t)); a_peak=A*w², v_peak=A*w. Specify a speed
    envelope or a bounded stroke; the other follows without a tuned time.
    """
    amplitude = speed ** 2 / acceleration if speed is not None else stroke / 2
    seconds = 2 * np.pi * np.sqrt(amplitude / acceleration)
    count = int(np.ceil(seconds * hz))
    time = np.arange(1, count + 1) / hz
    return amplitude * (1 - np.cos(2 * np.pi * time / (count / hz)))


def motion(poses, dt):
    positions, rotations = poses[:, :3, 3], poses[:, :3, :3]
    velocity = np.diff(positions, axis=0) / dt
    angular = Rotation.from_matrix(rotations[1:] @ rotations[:-1].transpose(0, 2, 1)).as_rotvec() / dt
    return {'linear_velocity_m_s': np.linalg.norm(velocity, axis=1),
            'linear_acceleration_m_s2': np.linalg.norm(np.diff(velocity, axis=0) / dt, axis=1),
            'angular_velocity_rad_s': np.linalg.norm(angular, axis=1),
            'angular_acceleration_rad_s2': np.linalg.norm(np.diff(angular, axis=0) / dt, axis=1),
            'translation_excursion_m': np.linalg.norm(positions - positions[0], axis=1).max(),
            'rotation_excursion_rad': Rotation.from_matrix(rotations @ rotations[0].T).magnitude().max()}
