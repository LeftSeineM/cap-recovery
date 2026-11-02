import numpy as np
from table_1000.experts.skills.hold.frames import Box
from table_1000.experts.skills import semantic
from table_1000.experts.world import geometry as world_geometry
from table_1000.experts.skills.push.squaring import square_on_table as _square_common, yaw_error, push_contact, _payload_contacts


def square_on_table(ctx, obj, xy, yaw=None, *, translation_lever=0., **options):
    if translation_lever == 0.:
        return _square_common(ctx, obj, xy, yaw, **options)
    return _square_with_translation_lever(ctx, obj, xy, yaw, translation_lever=translation_lever, **options)


def _square_with_translation_lever(ctx, obj, xy, yaw=None, *, arm=None, max_rounds=12,
                    position_tolerance=.020, angle_tolerance=np.radians(5.),
                    max_push=.025, contact_height=.035, yaw_direction=(0., 1.),
                    translation_lever=0., check=None):
    """Push a supported object toward ``xy``/``yaw``, remeasuring each round.

    ``yaw=None`` ignores heading for round containers. Off-centre pushes turn
    the object; centre pushes correct translation. Contacts are on the measured
    footprint, ``contact_height`` above its bottom. ``translation_lever`` moves
    lateral translation contacts sideways to avoid an obstructed centre. Existing push planning and
    scene geometry remain unchanged; nothing is grasped or lifted. ``check``
    runs before and after every push (and may reject lost payload).
    Internal carried-payload contact permissions last only for that push.
    Exhausted rounds fail explicitly; small motion is recorded for diagnosis.
    """
    from table_1000.experts import skills

    goal = np.asarray(xy, dtype=float)
    turn_direction = np.array(yaw_direction, dtype=float, copy=True)
    budgets = np.asarray([position_tolerance, angle_tolerance, max_push, contact_height], dtype=float)
    if (goal.shape != (2,) or not np.all(np.isfinite(goal)) or max_rounds < 1
            or not np.all(np.isfinite(budgets)) or np.min(budgets) <= 0
            or turn_direction.shape != (2,) or not np.all(np.isfinite(turn_direction))
            or np.linalg.norm(turn_direction) < 1e-9
            or not np.isfinite(translation_lever)
            or (yaw is not None and not np.isfinite(yaw))):
        raise ValueError('finite goals, nonzero direction and positive budgets required')
    turn_direction /= np.linalg.norm(turn_direction)
    s = ctx.skills()
    semantic.require(ctx, not s.held, 'table squaring requires empty hands', kind='planning', obj=obj)
    geometry = ctx.geometry[obj]
    children = {joint['body2'] for joint in geometry.joints.values()}
    body = next(body for body in geometry.bodies if body not in children)
    history = []
    for iteration in range(max_rounds + 1):
        if check is not None:
            check(ctx)
        pose = ctx.pose(obj)
        delta = goal - pose[:2, 3]
        angle = 0. if yaw is None else yaw_error(yaw, np.arctan2(pose[1, 0], pose[0, 0]))
        error = dict(position_m=float(np.linalg.norm(delta)), yaw_rad=angle)
        ctx.note('table squaring measured', obj=obj, round=iteration, **error)
        points = world_geometry.object_points(ctx, obj)
        semantic.require(ctx, abs(points[:, 2].min() - ctx.scene.TABLE['height']) <= .01,
                         'table squaring needs table support', kind='check', obj=obj)
        if error['position_m'] <= position_tolerance and abs(angle) <= angle_tolerance:
            return dict(**error, pushes=history)
        semantic.require(ctx, iteration < max_rounds, 'table squaring did not converge', kind='check', obj=obj, **error)
        if abs(angle) > angle_tolerance:
            # Use the opposite face once the last turn has drifted past the
            # goal. Equal torque from opposite corners avoids walking a
            # loaded container across the table while correcting its yaw.
            direction = turn_direction * (1. if delta @ turn_direction >= -position_tolerance else -1.)
            side = np.array([-direction[1], direction[0]])
            span = np.ptp(points[:, :2] @ side)
            lever = -np.sign(angle) * .325 * span
            distance = min(max_push, max(.003, abs(angle) * abs(lever) * .7))
        else:
            # An axis at a time avoids glancing corner contacts on a box.
            axis = int(np.argmax(np.abs(delta)))
            direction = np.eye(2)[axis] * np.sign(delta[axis])
            # Only lateral motion needs the front corner to clear the jar.
            # Fore/aft centre contact avoids introducing an unnecessary yaw.
            lever = translation_lever if axis == 0 else 0.
            distance = min(max_push, abs(delta[axis]))
        contact = np.r_[push_contact(points, direction, lever),
                        points[:, 2].min() + contact_height]
        frame = ctx.poses()[obj + ':' + body]
        offset = np.eye(4)
        offset[:3, 3] = frame[:3, :3].T @ (contact - frame[:3, 3])
        before = pose.copy()
        # A tiny contact selector supplies only the push point; the existing
        # planner still uses every original collision part of the object.
        with _payload_contacts(s, obj, body):
            result = skills.retry(ctx, lambda: skills.push(
                ctx, obj, direction=np.r_[direction, 0.], distance=distance,
                arm=arm, shape=Box((1e-6, 1e-6, 1e-6)), offset=offset, body=body))
        if check is not None:
            check(ctx)
        after = ctx.pose(obj)
        shifted = float(np.linalg.norm(after[:2, 3] - before[:2, 3]))
        turned = abs(yaw_error(np.arctan2(after[1, 0], after[0, 0]),
                               np.arctan2(before[1, 0], before[0, 0])))
        history.append(dict(contact=contact.tolist(), direction=direction.tolist(),
                            distance=distance, shifted=shifted, turned=turned, result=result))
        ctx.note('table squaring progress', obj=obj, shifted_m=shifted, turned_rad=turned)
