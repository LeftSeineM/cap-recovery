"""Isaac Lab action terms of a :class:`~table_1000.robots.spec.RobotSpec` (import after ``start_app``).

One action per control step (25 Hz) becomes joint targets at every physics step
(250 Hz): Isaac Lab calls :meth:`process_actions` once per environment step and
:meth:`apply_actions` before each of its ``decimation`` physics steps.

========================  ===============================================  ==========
group                     action row                                       term
========================  ===============================================  ==========
``base``                  ``(v, w)`` forward speed (m/s), yaw rate (rad/s)  :class:`BaseVelocityAction`
``lift``, ``head``        joint positions (or deltas)                      :class:`JointTargetAction`
``<arm>``                 6 joint positions (or deltas)                    :class:`JointTargetAction`
``<arm>`` (``ee_delta``)  TCP pose delta ``(dx, dy, dz, rx, ry, rz)``      Isaac Lab's differential IK
``<arm>_gripper``         pad gap (m)                                      :class:`GripperGapAction`
========================  ===============================================  ==========

Joint targets move linearly from the previous to the new target across the
physics steps, with the matching velocity as feed-forward, so a 25 Hz command
stream reaches the drives without steps. ``ee_delta`` uses Isaac Lab's
:class:`~isaaclab.envs.mdp.actions.DifferentialInverseKinematicsAction` in
relative mode: the delta is applied to the measured TCP pose in the robot's root
frame (the world for a robot with a mobile base) and solved by damped least
squares at every physics step.
"""

from dataclasses import MISSING

import torch

from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from table_1000.sim import control


class JointTargetAction(ActionTerm):
    """Joint position targets (``delta``: added to the previous target), clipped to the joint limits."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.joint_ids, _ = self._asset.find_joints(list(cfg.joint_names), preserve_order=True)
        limits = self._asset.data.soft_joint_pos_limits[:, self.joint_ids]
        self.lower, self.upper = limits[..., 0], limits[..., 1]
        self.target = self._asset.data.joint_pos[:, self.joint_ids].clone()
        self.start = self.target.clone()
        self.raw = torch.zeros_like(self.target)
        self.substep = 0

    @property
    def action_dim(self):
        return len(self.joint_ids)

    @property
    def raw_actions(self):
        return self.raw

    @property
    def processed_actions(self):
        return self.target

    def hold_action(self):
        """The action that keeps the current target."""
        return torch.zeros_like(self.target) if self.cfg.delta else self.target.clone()

    def set_state(self, q, env_ids=slice(None)):
        """Targets of ``env_ids`` set to joint values ``q`` without motion (after a state write)."""
        self.target[env_ids] = q
        self.start[env_ids] = q

    def get_target(self):
        """The joint targets held (absolute, also in ``delta`` mode)."""
        return self.target.clone()

    def set_target(self, target):
        """Hold joint targets ``target``, reached from the current ones within the next step."""
        self.target = torch.clamp(target, self.lower, self.upper)

    def process_actions(self, actions):
        self.raw[:] = actions
        self.start = self.target.clone()
        self.target = torch.clamp(self.target + actions if self.cfg.delta else actions, self.lower, self.upper)
        self.substep = 0

    def apply_actions(self):
        self.substep += 1
        positions, velocities = control.interpolate(self.start, self.target, self.substep, self._env.cfg.decimation,
                                                    self._env.physics_dt)
        self._asset.set_joint_position_target(positions, joint_ids=self.joint_ids)
        self._asset.set_joint_velocity_target(velocities, joint_ids=self.joint_ids)

    def reset(self, env_ids=None):
        env_ids = slice(None) if env_ids is None else env_ids
        self.set_state(self._asset.data.joint_pos[env_ids][:, self.joint_ids], env_ids)


@configclass
class JointTargetActionCfg(ActionTermCfg):
    class_type: type = JointTargetAction
    joint_names: tuple = MISSING
    delta: bool = False


class GripperGapAction(ActionTerm):
    """Pad gap ``g`` (m) -> finger targets (:func:`table_1000.sim.control.finger_targets`); the drives
    close on their own and contact stops them at the object."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.joint_ids, _ = self._asset.find_joints(list(cfg.joint_names), preserve_order=True)
        self.signs = torch.tensor(cfg.signs, device=self.device, dtype=torch.float32)
        self.gap = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def action_dim(self):
        return 1

    @property
    def raw_actions(self):
        return self.gap

    @property
    def processed_actions(self):
        return control.finger_targets(self.gap, self.signs, self.cfg.max_gap, self.cfg.pivot)

    def hold_action(self):
        return self.gap.clone()

    def set_state(self, q, env_ids=slice(None)):
        """Gap of ``env_ids`` from finger positions ``q``."""
        self.gap[env_ids] = control.finger_gap(q, self.signs, self.cfg.pivot)

    def get_target(self):
        """The gap commanded: below the measured one while the fingers squeeze an object."""
        return self.gap.clone()

    def set_target(self, gap):
        self.gap[:] = torch.clamp(gap, 0., self.cfg.max_gap)

    def process_actions(self, actions):
        self.gap[:] = torch.clamp(actions, 0., self.cfg.max_gap)

    def apply_actions(self):
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self.joint_ids)
        self._asset.set_joint_velocity_target(torch.zeros_like(self.processed_actions), joint_ids=self.joint_ids)

    def reset(self, env_ids=None):
        env_ids = slice(None) if env_ids is None else env_ids
        self.set_state(self._asset.data.joint_pos[env_ids][:, self.joint_ids], env_ids)


@configclass
class GripperGapActionCfg(ActionTermCfg):
    class_type: type = GripperGapAction
    joint_names: tuple = MISSING
    signs: tuple = (1., -1.)
    max_gap: float = .1
    pivot: tuple = None


class BaseVelocityAction(ActionTerm):
    """``(v, w)`` -> targets of the virtual (x, y, yaw) base joints and the wheel speeds.

    The commanded speeds are clipped to ``max_speed`` and approached at no more
    than ``max_acceleration``; the virtual joints follow the integrated pose, so
    the base does not drift while it stands still. Before each command the
    integrated pose is kept within ``max_lag`` of the measured one, so a base
    blocked by an obstacle does not wind up a target far beyond it. The wheels
    only get speed targets: their drives are pure dampers, and a turning joint's
    position target is bounded (PhysX: ``[-2 pi, 2 pi]``).
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        base = cfg.base
        self.joint_ids, _ = self._asset.find_joints(list(base.virtual_joints), preserve_order=True)
        self.wheel_ids = self._asset.find_joints(list(base.wheels), preserve_order=True)[0] if base.wheels else []
        tensor = lambda value: torch.tensor(value, device=self.device, dtype=torch.float32)
        self.max_speed, self.max_acceleration, self.lag = tensor(base.max_speed), tensor(base.max_acceleration), tensor(base.max_lag)
        self.command = torch.zeros(self.num_envs, 2, device=self.device)
        self.speed = torch.zeros_like(self.command)
        self.pose = self._asset.data.joint_pos[:, self.joint_ids].clone()

    @property
    def action_dim(self):
        return 2

    @property
    def raw_actions(self):
        return self.command

    @property
    def processed_actions(self):
        return self.speed

    def hold_action(self):
        return self.command.clone()

    def set_state(self, pose, env_ids=slice(None)):
        """Base of ``env_ids`` at rest at ``[n, 3]`` pose (x, y, yaw)."""
        self.pose[env_ids] = pose
        self.speed[env_ids] = 0.
        self.command[env_ids] = 0.

    def process_actions(self, actions):
        self.pose = control.clamp_lag(self.pose, self._asset.data.joint_pos[:, self.joint_ids], self.lag)
        self.command[:] = torch.clamp(actions, -self.max_speed, self.max_speed)

    def get_target(self):
        """The commanded ``(v, w)``."""
        return self.command.clone()

    def set_target(self, command):
        """Command ``(v, w)``, taken as the speed reached (a restored base keeps moving as it did)."""
        self.command[:] = torch.clamp(command, -self.max_speed, self.max_speed)
        self.speed[:] = self.command

    def apply_actions(self):
        dt = self._env.physics_dt
        self.pose, self.speed, velocity = control.base_substep(self.pose, self.speed, self.command,
                                                               self.max_acceleration, dt)
        target = self.pose.clone()
        target[:, 2] = control.near_angle(target[:, 2], self._asset.data.joint_pos[:, self.joint_ids[2]])
        self._asset.set_joint_position_target(target, joint_ids=self.joint_ids)
        self._asset.set_joint_velocity_target(velocity, joint_ids=self.joint_ids)
        if len(self.wheel_ids) == 2:
            wheel_speed = control.wheel_speeds(self.speed, self.cfg.base.track, self.cfg.base.wheel_radius)
            self._asset.set_joint_velocity_target(wheel_speed, joint_ids=self.wheel_ids)

    def reset(self, env_ids=None):
        env_ids = slice(None) if env_ids is None else env_ids
        self.set_state(self._asset.data.joint_pos[env_ids][:, self.joint_ids], env_ids)


@configclass
class BaseVelocityActionCfg(ActionTermCfg):
    class_type: type = BaseVelocityAction
    base: object = MISSING


def actions_cfg(spec, modes=None, asset='robot'):
    """Action terms of ``spec``'s groups, in the order of the flat action vector.

    ``modes`` maps arm, lift and head groups to ``'pos'`` (default), ``'delta'``
    or (arms) ``'ee_delta'``.
    """
    modes = {'lift': 'pos', 'head': 'pos', **{arm: 'pos' for arm in spec.arms}, **(modes or {})}

    @configclass
    class ActionsCfg:
        pass

    cfg = ActionsCfg()

    def joints(names, mode):
        return JointTargetActionCfg(asset_name=asset, joint_names=tuple(names), delta=mode == 'delta')

    if spec.base:
        cfg.base = BaseVelocityActionCfg(asset_name=asset, base=spec.base)
    if spec.lift:
        cfg.lift = joints(spec.lift, modes['lift'])
    if spec.head:
        cfg.head = joints(spec.head, modes['head'])
    for name, arm in spec.arms.items():
        if modes[name] == 'ee_delta':
            setattr(cfg, name, DifferentialInverseKinematicsActionCfg(
                asset_name=asset, joint_names=list(arm.joints), body_name=arm.tcp[0],
                body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=tuple(arm.tcp[1])),
                controller=DifferentialIKControllerCfg(command_type='pose', use_relative_mode=True, ik_method='dls')))
        else:
            setattr(cfg, name, joints(arm.joints, modes[name]))
        setattr(cfg, f'{name}_gripper', GripperGapActionCfg(asset_name=asset, joint_names=tuple(arm.fingers),
                                                            signs=tuple(arm.finger_signs), max_gap=arm.max_gap,
                                                            pivot=arm.finger_pivot))
    return cfg
