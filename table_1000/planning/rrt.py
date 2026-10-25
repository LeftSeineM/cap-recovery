"""RRT geometry with collision-checked shortcuts and path-preserving timing.

The OMPL call follows MPlib 0.2.1 ``Planner.plan_qpos`` (MIT, Copyright (c)
2021 Hao Su's Lab, UCSD). Its coarse shortcut checks and cubic TOPP spline
can leave a route that passed a finer collision check. Keep the raw route,
check every shortcut, and stop at corners when the spline cuts one or exceeds
a joint position limit.
"""

import numpy as np


def time_path(planner, path, time_step):
    """Retain both endpoints even when the move is shorter than one control period."""
    result = planner.TOPP(path, time_step)
    if len(result[0]) < 2:
        result = planner.TOPP(path, min(time_step, result[4] / 2))
    return result


def parameterize(planner, path, time_step, check, *, simplify=None):
    """Time a checked joint path; ``check(rows)`` returns its collision pairs."""
    path = np.asarray(path, dtype=float)
    limits = np.asarray(planner.joint_limits)[planner.move_group_joint_indices]

    def outside(rows):
        rows = np.asarray(rows)
        return np.flatnonzero(np.any(~np.isfinite(rows) | (rows < limits[:, 0]) | (rows > limits[:, 1]), axis=1))

    def invalid(rows, bad, stage):
        return dict(status='joint_limit', reason='joint_limit', stage=stage,
                    q=np.asarray(rows)[bad[0]].tolist(), limits=limits.tolist())

    bad = outside(path)
    if len(bad):
        return invalid(path, bad, 'raw')
    pairs = check(path)
    if pairs:
        return {'status': 'path collision', 'reason': 'path collision', 'collisions': pairs, 'stage': 'raw'}
    if simplify is not None:
        path = np.asarray(simplify(path), dtype=float)
    else:
        kept, index = [path[0]], 0
        while index < len(path) - 1:
            following = len(path) - 1
            while following > index + 1 and check(path[[index, following]]):
                following -= 1
            kept.append(path[following])
            index = following
        path = np.asarray(kept)
    if len(path) == 1 or np.array_equal(path[0], path[-1]) and len(path) == 2:
        zero = np.zeros_like(path[:1])
        return dict(status='Success', time=np.array([0.]), position=path[:1], velocity=zero,
                    acceleration=zero, duration=0., timing='stationary')
    try:
        timed = time_path(planner, path, time_step)
        mode = 'cubic'
        if len(outside(timed[1])) or check(timed[1]):
            # Two-point TOPP is linear in joint space. Each segment starts and ends
            # at rest, so joining them preserves the checked route and joint limits.
            parts, elapsed = [], 0.
            for start, end in zip(path[:-1], path[1:]):
                if np.array_equal(start, end):
                    continue
                times, positions, velocities, accelerations, duration = time_path(planner, np.array([start, end]), time_step)
                parts.append((times + elapsed, positions, velocities, accelerations))
                elapsed += duration
            timed = tuple(np.concatenate([part[k] if i == 0 else part[k][1:]
                                          for i, part in enumerate(parts)]) for k in range(4)) + (elapsed,)
            mode = 'linear segments'
    except RuntimeError:
        return {'status': 'time_parameterization', 'reason': 'time_parameterization'}
    bad = outside(timed[1])
    if len(bad):
        return invalid(timed[1], bad, 'timed')
    pairs = check(timed[1])
    if pairs:
        return {'status': 'path collision', 'reason': 'path collision', 'collisions': pairs, 'stage': 'timed'}
    times, positions, velocities, accelerations, duration = timed
    return dict(status='Success', time=times, position=positions, velocity=velocities,
                acceleration=accelerations, duration=duration, timing=mode, waypoints=len(path))


def plan_rrt(planner, start, goals, *, time_step, planning_time, fixed, check):
    from mplib.planning.ompl import FixedJoint
    planner.robot.set_qpos(start, True)
    indices = planner.move_group_joint_indices
    locked = {FixedJoint(0, index, start[index]) for index in fixed}
    status, path = planner.planner.plan(start[indices], [np.asarray(goal)[indices] for goal in goals],
                                       time=planning_time, range=.1, fixed_joints=locked, simplify=False)
    if status != 'Exact solution':
        return {'status': f'RRTConnect Failed. {status}'}
    return parameterize(planner, path, time_step, check, simplify=planner.planner.simplify_path)
