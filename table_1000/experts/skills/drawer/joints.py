"""Asset joint descriptions, measured openings and slider travel geometry."""

from dataclasses import dataclass

import numpy as np

from table_1000.experts.skills.common import normalized


OPEN_TOLERANCE = .02    # m short of the asked opening that still counts as open


CLOSED = .020           # m from the closed end that counts as closed


@dataclass
class Joint:
    """Joint ``name`` of an asset (``model.json``): ``kind`` (``SLIDER``, ``HINGE``, ...) between
    bodies ``parent`` and ``child``; unit ``axis`` (positive motion) in the parent's frame,
    ``child_axis`` in the child's; ``limits``; ``handle`` the child's part named ``handle`` (None
    without one)."""
    name: str
    kind: str
    parent: str
    child: str
    axis: np.ndarray
    limits: tuple
    child_axis: np.ndarray
    handle: str = None

    @classmethod
    def of(cls, geometry, name):
        entry = geometry.joints[name]
        parent = entry['body1']
        # model.json gives the joint frame in the asset frame.
        frame = np.linalg.inv(geometry.rest[parent]) @ np.array(entry['matrix'], dtype=float)
        axis = normalized(frame[:3, 0])
        handles = [part for _, part in geometry.parts(entry['body2']) if part.rsplit('.', 1)[-1].startswith('handle')]
        child_axis = geometry.rest[entry['body2']][:3, :3].T @ geometry.rest[parent][:3, :3] @ axis
        return cls(name, entry['type'], parent, entry['body2'], axis, tuple(entry['limits']), child_axis,
                   handles[0] if handles else None)

    def direction(self, child_pose):
        """World direction of positive motion with the child body at ``child_pose`` (a drawer
        half out may sit a little tilted in its housing; it moves along its own frame)."""
        return np.asarray(child_pose)[:3, :3] @ self.child_axis


def joint(ctx, obj, name):
    return Joint.of(ctx.geometry[obj], name)


def sim_joint(name):
    """The simulation's name of asset joint ``name`` (``cabinet_drawer1_slide``)."""
    return name.replace('.', '_')


def opening(ctx, obj, name):
    """How far joint ``name`` of ``obj`` is from its closed (lower) limit."""
    return ctx.joint(obj, sim_joint(name)) - Joint.of(ctx.geometry[obj], name).limits[0]


def is_open(ctx, obj, name, amount, tolerance=OPEN_TOLERANCE):
    return opening(ctx, obj, name) >= amount - tolerance


def is_closed(ctx, obj, name, tolerance=CLOSED):
    return opening(ctx, obj, name) <= tolerance


def slider(ctx, obj, name):
    j = joint(ctx, obj, name)
    if j.kind != 'SLIDER':
        raise NotImplementedError(f'{name}: only sliders are opened and closed so far, not {j.kind}')
    return j


def travel(ctx, obj, name, margin=0.):
    """Keep-out rectangle ``(frame, low, high)`` of the space joint ``name``'s child body sweeps from
    where it is to fully open (a drawer's pull-out space), grown by ``margin``."""
    j = slider(ctx, obj, name)
    geometry, frame = ctx.geometry[obj], ctx.pose(obj, j.child)
    low, high = geometry.bounds(j.child, j.child)
    sweep = j.child_axis * (j.limits[1] - j.limits[0] - opening(ctx, obj, name))
    return frame, np.minimum(low, low + sweep) - margin, np.maximum(high, high + sweep) + margin
