"""Robot environment on Isaac Lab's ``ManagerBasedEnv`` (import after ``start_app``).

:func:`env_cfg` builds the Isaac Lab configuration from a scene source and a
:class:`~table_1000.robots.spec.RobotSpec`: the scene (:mod:`.scene`), one
action term per robot group (:mod:`.actions`), robot-observable observations and
a reset event that applies a layout. :class:`RobotEnv` wraps the resulting
environment with dict actions and NumPy observations; every array has a leading
environment axis. ``device='cpu'`` runs CPU PhysX; GPU PhysX needs only
``device='cuda:0'``. Acceptance scripts record backend-specific results.

Observation (ManiSkill's layout; only what the real robot measures):

- ``agent``: ``qpos``, ``qvel`` ``[N, J]`` in the order of :attr:`RobotEnv.joints`
  (body joints, then virtual base joints); ``base_pose`` ``[N, 3]`` world
  (x, y, yaw) of the root link; ``tcp`` ``{arm: [N, 4, 4]}`` world TCP poses
- ``sensor_param``: ``{camera: {'intrinsic': [N, 3, 3], 'cam2world': [N, 4, 4]}}``
  with the sensor's calibrated intrinsics and the optical frame's pose (OpenCV axes)
- ``time``: simulated seconds since the reset

Camera images are rendered on request (:meth:`RobotEnv.images`), not every step.
Simulator-only state -- object poses and velocities, contacts, rendering aids --
is reached through :meth:`RobotEnv.privileged` and the other methods, for experts,
checks and evaluation, never for a policy. ``recorders`` (Isaac Lab's
``RecorderManagerBaseCfg``, see :mod:`.recording`) record episodes to datasets.
"""

from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from table_1000.robots import ROOT
from table_1000.robots.cameras import mount_on_body
from table_1000.sim.isaaclab import scene as lab_scene
from table_1000.sim.isaaclab.actions import actions_cfg
from table_1000.sim.isaaclab.contacts import Contacts

from table_1000.sim import PHYSICS_DT as DT, SUBSTEPS as DECIMATION, control
MARKERS_PATH = '/World/Markers'


def matrix(position, quaternion_wxyz):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(np.asarray(quaternion_wxyz)[[1, 2, 3, 0]]).as_matrix()
    transform[:3, 3] = position
    return transform


def pose_tensor(transform, device):
    """``[7]`` (x, y, z, qw, qx, qy, qz) of a 4x4 transform."""
    position, rotation = lab_scene.pose(transform)
    return torch.tensor([*position, *rotation], dtype=torch.float32, device=device)


def transforms(pose):
    """``[N, 4, 4]`` torch transforms of ``[N, 7]`` poses (position, quaternion w x y z)."""
    from isaaclab.utils.math import matrix_from_quat
    out = torch.zeros(pose.shape[0], 4, 4, device=pose.device)
    out[:, :3, :3] = matrix_from_quat(pose[:, 3:7])
    out[:, :3, 3] = pose[:, :3]
    out[:, 3, 3] = 1.
    return out


# -- observation terms ------------------------------------------------------------
def joint_positions(env, asset_cfg):
    return env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]


def joint_velocities(env, asset_cfg):
    return env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]


def base_pose(env, asset_cfg, fixed):
    """World (x, y, yaw) of the root link, yaw in ``[-pi, pi)``: the virtual base joints, or the fixed placement."""
    if fixed is not None:
        return torch.tensor(fixed, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
    pose = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids].clone()
    pose[:, 2] = control.wrap_angle(pose[:, 2])
    return pose


def body_frame(env, asset_cfg, offset):
    """``[N, 4, 4]`` world pose of a frame fixed to a robot body (``offset``: 4x4 in the body)."""
    pose = env.scene[asset_cfg.name].data.body_link_pose_w[:, asset_cfg.body_ids[0]]
    return transforms(pose) @ torch.tensor(offset, dtype=torch.float32, device=env.device)


def intrinsic(env, matrix):
    return torch.tensor(matrix, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1, 1)


# -- reset event ---------------------------------------------------------------------
def apply_layout(env, env_ids):
    """Write the pending layout, object joints and robot joints (:meth:`RobotEnv.reset`) into the
    reset environments, everything at rest."""
    layout, robot_q, object_q = env.pending_reset
    set_joints(env.scene['robot'], robot_q, env_ids)
    origins = env.scene.env_origins[env_ids]

    def world(transform):
        pose = pose_tensor(transform, env.device).repeat(len(env_ids), 1)
        pose[:, :3] += origins
        return pose

    for name, transform in layout.items():
        structure = env.object_structures.get(name)
        if structure is None:
            place(env, env.scene[name], world(transform), env_ids)
        elif structure.articulated:
            # The articulation's root pose is its root body's.
            articulation = env.scene[name]
            articulation.write_root_pose_to_sim(world(transform @ structure.rest[structure.root]), env_ids=env_ids)
            articulation.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)
            set_joints(articulation, object_q.get(name, {}), env_ids)
        else:
            for body in structure.paths:
                rigid = env.scene[lab_scene.body_entity(name, body)]
                rigid.write_root_pose_to_sim(world(transform @ structure.rest[body]), env_ids=env_ids)
                rigid.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)


def set_joints(articulation, values, env_ids):
    """Joints of ``articulation`` at ``values`` (``{name: value}``; others at their defaults), at
    rest, with matching drive targets."""
    q = articulation.data.default_joint_pos[env_ids].clone()
    for name, value in values.items():
        q[:, articulation.joint_names.index(name)] = float(value)
    articulation.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
    articulation.set_joint_position_target(q, env_ids=env_ids)
    articulation.set_joint_velocity_target(torch.zeros_like(q), env_ids=env_ids)


def kinematic(shape):
    """Whether a scene shape is kinematic (posed, never moved by physics)."""
    return bool(shape.cfg.spawn.rigid_props.kinematic_enabled)


def place(env, shape, pose, env_ids):
    """Teleport ``shape`` of ``env_ids`` to ``[n, 7]`` world poses (position, quaternion w x y z), at rest.

    A kinematic body also gets the pose as its kinematic target -- otherwise the next physics step
    moves it back to its last target -- and as its USD transform, which PhysX does not write back
    for kinematic bodies and the renderer shows.
    """
    from isaaclab.sim.views import XformPrimView
    from isaaclab.utils.math import convert_quat
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    shape.write_root_pose_to_sim(pose, env_ids=env_ids)
    if not kinematic(shape):
        shape.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)
        return
    if env.device != 'cpu':
        # PhysX 5.1's GPU rigid view has no set_kinematic_targets implementation.
        # A reset teleports the body through set_transforms instead.
        return
    target = pose.clone()
    target[:, 3:] = convert_quat(target[:, 3:], to='xyzw')
    indices = torch.as_tensor(env_ids, dtype=torch.int32, device=env.device)
    shape.root_physx_view.set_kinematic_targets(target, indices)
    if shape.cfg.prim_path not in env.shape_views:
        env.shape_views[shape.cfg.prim_path] = XformPrimView(shape.cfg.prim_path, device=env.device)
    view = env.shape_views[shape.cfg.prim_path]
    view.set_world_poses(pose[:, :3], pose[:, 3:], indices=indices.tolist())


def env_cfg(scene, spec, assets, *, modes=None, cameras=None, depth=False, views=None, num_envs=1,
            device='cpu', root=None, recorders=None):
    """``ManagerBasedEnvCfg`` of ``spec`` in ``scene``. ``cameras``: robot cameras to render
    (``{name: CameraSpec}``, see :func:`~table_1000.robots.spec.override_cameras`); ``views``:
    fixed video views; ``modes``: arm, lift and head control modes (:func:`.actions.actions_cfg`)."""
    root = Path(root or ROOT)
    urdf = root / spec.urdf
    robot_q = dict(scene.ROBOT['joints'])
    if spec.base:
        robot_q.update(zip(spec.base.virtual_joints, scene_base(scene)))
    joints = [*spec.body_joints, *(spec.base.virtual_joints if spec.base else ())]

    @configclass
    class AgentCfg(ObservationGroupCfg):
        concatenate_terms = False

    @configclass
    class SensorParamCfg(ObservationGroupCfg):
        concatenate_terms = False

    robot = SceneEntityCfg('robot', joint_names=joints, preserve_order=True)
    agent = AgentCfg()
    agent.qpos = ObservationTermCfg(func=joint_positions, params={'asset_cfg': robot})
    agent.qvel = ObservationTermCfg(func=joint_velocities, params={'asset_cfg': robot})
    agent.base_pose = ObservationTermCfg(func=base_pose, params={
        'asset_cfg': SceneEntityCfg('robot', joint_names=list(spec.base.virtual_joints) if spec.base else [],
                                    preserve_order=True),
        'fixed': None if spec.base else list(scene_base(scene))})
    for name, arm in spec.arms.items():
        offset = np.eye(4)
        offset[:3, 3] = arm.tcp[1]
        setattr(agent, f'tcp_{name}', ObservationTermCfg(func=body_frame, params={
            'asset_cfg': SceneEntityCfg('robot', body_names=[arm.tcp[0]]), 'offset': offset.tolist()}))
    sensor_param = SensorParamCfg()
    for name, camera in (cameras or {}).items():
        link, transform = mount_on_body(urdf, camera.mount)
        fx, fy, cx, cy = camera.sensor.intrinsics
        setattr(sensor_param, f'{name}_intrinsic', ObservationTermCfg(
            func=intrinsic, params={'matrix': [[fx, 0., cx], [0., fy, cy], [0., 0., 1.]]}))
        setattr(sensor_param, f'{name}_cam2world', ObservationTermCfg(func=body_frame, params={
            'asset_cfg': SceneEntityCfg('robot', body_names=[link]), 'offset': transform.tolist()}))

    @configclass
    class ObservationsCfg:
        pass

    observations = ObservationsCfg()
    observations.agent = agent
    if cameras:
        observations.sensor_param = sensor_param

    @configclass
    class EventsCfg:
        layout = EventTermCfg(func=apply_layout, mode='reset')

    @configclass
    class Cfg(ManagerBasedEnvCfg):
        decimation = DECIMATION

    cfg = Cfg(scene=lab_scene.scene_cfg(scene, spec, assets, robot_q, num_envs=num_envs, cameras=cameras,
                                        depth=depth, views=views, root=root),
              observations=observations, actions=actions_cfg(spec, modes), events=EventsCfg())
    if recorders is not None:
        cfg.recorders = recorders
    cfg.sim = sim_utils.SimulationCfg(device=device, dt=DT,
                                      physx=sim_utils.PhysxCfg(solver_type=1, enable_ccd=device == 'cpu'),
                                      # Cameras render on request (RobotEnv.images), not inside steps.
                                      render_interval=10 ** 9, use_fabric=device != 'cpu')
    cfg.ui_window_class_type = None
    return cfg


def scene_base(scene):
    """``(x, y, yaw)`` of the robot in ``scene``."""
    position = scene.ROBOT['position']
    return float(position[0]), float(position[1]), float(scene.ROBOT['yaw'])


class RobotEnv:
    """A :class:`ManagerBasedEnv` of a robot in a scene, with dict actions and NumPy observations.

    An action is ``{group: [N, dim]}`` (see :mod:`.actions`); a group missing from
    it holds its previous target.
    """

    def __init__(self, scene, spec, assets, *, cameras=None, views=None, depth=False, contact_watch=('robot',),
                 **kwargs):
        self.scene_source, self.spec = scene, spec
        self.camera_specs = dict(cameras or {})
        # Isaac Lab's experience enables PhysX Fabric, which stops physics writing body poses
        # to USD; cameras under robot links then stay at the start pose. On CPU it is off.
        if kwargs.get('device', 'cpu') == 'cpu':
            from isaacsim.core.simulation_manager import SimulationManager
            SimulationManager.enable_fabric(enable=False)
        self.env = ManagerBasedEnv(env_cfg(scene, spec, assets, cameras=self.camera_specs, views=views, depth=depth,
                                           **kwargs))
        self.env.shape_views = {}       # USD views of kinematic shapes, by prim path (see place)
        self.objects = lab_scene.object_structures(scene, assets)
        self.env.object_structures = self.objects
        if self.env.device != 'cpu':
            self.env.sim.forward()
            for camera in self.env.scene.sensors.values():
                if hasattr(camera, '_view'):
                    # Lab 2.3's first Xform read calls Kit.update() and copies
                    # stale USD poses over GPU Fabric. Physics already owns the
                    # transforms; initialize the view against that Fabric state.
                    camera._view._initialize_fabric()
                    camera._view._fabric_usd_sync_done = True
        # The first physics step after the simulation starts restores kinematic bodies to their
        # authored poses, undoing the first reset's placement; it is taken here instead.
        self.env.sim.step(render=False)
        self.num_envs = self.env.num_envs
        self.robot = self.env.scene['robot']
        self.joints = [*spec.body_joints, *(spec.base.virtual_joints if spec.base else ())]
        self.shapes = list(getattr(scene, 'PRIMITIVES', {}))
        self.views = list(views or {})
        self.terms = self.env.action_manager._terms
        prefix = '/World/envs/env_0'
        labels = {f'{prefix}/Robot': 'robot', f'{prefix}/Table': 'table', lab_scene.GROUND_PATH: 'ground'}
        labels.update({f'{prefix}/{lab_scene.SHAPE_PREFIX}{name}': f'{name}:body' for name in self.shapes})
        # Object parts are labelled <object>:<body>.<part prim>, e.g. cabinet:cabinet.drawer3.handle.
        labels.update({f'{prefix}/{lab_scene.OBJECT_PREFIX}{name}{path}': f'{name}:{body}'
                       for name, structure in self.objects.items() for body, path in structure.paths.items()})
        self.contacts = Contacts(labels, contact_watch, DT, device=self.env.device)
        self.markers = None
        self.steps = 0
        self.timings = {'step_s': 0., 'render_s': 0.}

    @property
    def device(self):
        return self.env.device

    @property
    def time(self):
        """Simulated seconds since the reset."""
        return self.steps * DECIMATION * DT

    def initial_robot_q(self):
        """Scene joints (others at zero) and, for a mobile base, its pose as virtual joint values."""
        q = dict(zip(self.spec.base.virtual_joints, scene_base(self.scene_source))) if self.spec.base else {}
        q.update(self.scene_source.ROBOT['joints'])
        return q

    def reset(self, layout, robot_q=None, object_q=None):
        """Place the shapes and objects of ``layout`` (``{name: 4x4 world pose}``; an object's pose is
        its asset frame's), set the robot's joints (``{name: value}``, base included) and the objects'
        (``{object: {joint: value}}``; unlisted joints at their defaults)."""
        self.env.pending_reset = (layout, robot_q or self.initial_robot_q(), object_q or {})
        self.env.reset()
        self.env.sim.forward()
        self.steps = 0
        self.contacts.clear()
        return self.observe()

    def targets(self):
        """``{group: [N, d]}`` (NumPy) what each action term holds: joint targets, the gripper's
        commanded gap (below the measured one while it squeezes), the base's ``(v, w)``. With the
        scene state it is what :meth:`reset_to_state` restores; an action in ``'pos'`` mode is it."""
        return {name: term.get_target().cpu().numpy() for name, term in self.terms.items()
                if hasattr(term, 'get_target')}

    def reset_to_state(self, state):
        """Reset to a scene state (``InteractiveScene.get_state(is_relative=True)``, as NumPy or
        tensors with the environment axis): every body where it was, the action targets holding
        the restored joints, or ``state['targets']`` (:meth:`targets`) where the state has them."""
        def to_tensors(tree):
            return {key: to_tensors(value) if isinstance(value, dict) else torch.as_tensor(value, device=self.device)
                    for key, value in tree.items()}

        state = to_tensors(state)
        self.env.pending_reset = ({}, self.initial_robot_q(), {})
        self.env.reset_to(state, None, is_relative=True)
        # Lab resets the action terms before writing the state; they hold the joints written.
        for term in self.terms.values():
            term.reset()
        # A gripper holding an object squeezes it: its target is not where its fingers are.
        for name, target in state.get('targets', {}).items():
            self.terms[name].set_target(target.to(torch.float32))
        self.place_kinematic(state)
        self.env.sim.forward()
        self.steps = 0
        self.contacts.clear()
        return self.observe()

    def place_kinematic(self, state):
        """Pose the kinematic shapes as in ``state`` (tensors on the device): ``InteractiveScene.reset_to``
        leaves their kinematic targets and USD transforms."""
        rows = next(iter(state['rigid_object'].values()))['root_pose'].shape[0] if state['rigid_object'] else 0
        for name in self.shapes:
            shape = self.env.scene[name]
            if kinematic(shape):
                pose = state['rigid_object'][name]['root_pose'].clone()
                pose[:, :3] += self.env.scene.env_origins[:rows]
                place(self.env, shape, pose, list(range(rows)))

    def observe(self):
        obs = self.env.observation_manager.compute()
        agent = {key: value.cpu().numpy() for key, value in obs['agent'].items()}
        self.obs = {'agent': {'qpos': agent['qpos'], 'qvel': agent['qvel'], 'base_pose': agent['base_pose'],
                              'tcp': {arm: agent[f'tcp_{arm}'] for arm in self.spec.arms}},
                    'time': self.time}
        if 'sensor_param' in obs:
            params = {key: value.cpu().numpy() for key, value in obs['sensor_param'].items()}
            self.obs['sensor_param'] = {name: {'intrinsic': params[f'{name}_intrinsic'],
                                               'cam2world': params[f'{name}_cam2world']} for name in self.camera_specs}
        return self.obs

    def qpos(self, joints):
        """``[N, len(joints)]`` measured positions of the named joints."""
        return self.obs['agent']['qpos'][:, [self.joints.index(name) for name in joints]]

    @property
    def action_layout(self):
        """``{group: (start, stop)}`` of each action group in the flat action."""
        layout, start = {}, 0
        for name, term in self.terms.items():
            layout[name] = (start, start + term.action_dim)
            start += term.action_dim
        return layout

    def action(self, action):
        """The flat Isaac Lab action of a dict action; missing groups hold."""
        unknown = set(action) - set(self.terms)
        if unknown:
            raise KeyError(f'unknown action groups {sorted(unknown)}')
        rows = []
        for name, term in self.terms.items():
            if name in action:
                rows.append(torch.as_tensor(np.asarray(action[name], dtype=np.float32), device=self.device)
                            .reshape(self.num_envs, term.action_dim))
            elif hasattr(term, 'hold_action'):
                rows.append(term.hold_action())
            else:
                rows.append(torch.zeros(self.num_envs, term.action_dim, device=self.device))
        return torch.cat(rows, dim=1)

    def step(self, action=None):
        """Apply ``action`` for one control step; returns the observation."""
        import time
        started = time.perf_counter()
        self.env.step(self.action(action or {}))
        self.steps += 1
        self.timings['step_s'] += time.perf_counter() - started
        return self.observe()

    # -- privileged state (experts, checks, evaluation) ---------------------------------
    def privileged(self):
        """World state of the scene's shapes and object bodies, NumPy with the environment axis:

        - ``poses``, ``velocities``: ``{key: [N, 4, 4]}``, ``{key: [N, 6]}`` (linear and angular
          velocity of the centre of mass); a shape's key is its name, an object body's
          ``<object>:<body>`` (e.g. ``cabinet:cabinet.drawer3``)
        - ``joints``: ``{object: {joint: [N]}}`` of articulated objects
        """
        origins = self.env.scene.env_origins
        poses, velocities, joints = {}, {}, {}
        # On CPU, .numpy() of a PhysX data buffer shares its memory, which later steps overwrite:
        # what is returned is copied.

        def add(key, pose, velocity):
            pose = pose.clone()
            pose[:, :3] -= origins
            poses[key] = transforms(pose).cpu().numpy()
            velocities[key] = velocity.cpu().numpy().copy()

        for name in self.shapes:
            data = self.env.scene[name].data
            add(name, data.root_link_pose_w, data.root_com_vel_w)
        for name, structure in self.objects.items():
            if structure.articulated:
                articulation = self.env.scene[name]
                data = articulation.data
                for body, path in structure.paths.items():
                    index = articulation.body_names.index(path.rsplit('/', 1)[-1])
                    add(f'{name}:{body}', data.body_link_pose_w[:, index], data.body_com_vel_w[:, index])
                joints[name] = dict(zip(articulation.joint_names, data.joint_pos.T.cpu().numpy().copy()))
            else:
                for body in structure.paths:
                    data = self.env.scene[lab_scene.body_entity(name, body)].data
                    add(f'{name}:{body}', data.root_link_pose_w, data.root_com_vel_w)
        return {'poses': poses, 'velocities': velocities, 'joints': joints}

    def object_pose(self, name):
        """``[N, 4, 4]`` world pose of asset object ``name``'s frame (from its root or first body)."""
        structure = self.objects[name]
        body = structure.root or next(iter(structure.paths))
        pose = self.privileged()['poses'][f'{name}:{body}']
        return pose @ np.linalg.inv(structure.rest[body])

    def center_of_mass(self, key):
        """``[N, 3]`` world centre of mass of an object body (``<object>:<body>``)."""
        name, body = key.split(':', 1)
        structure = self.objects[name]
        if structure.articulated:
            articulation = self.env.scene[name]
            index = articulation.body_names.index(structure.paths[body].rsplit('/', 1)[-1])
            position = articulation.data.body_com_pos_w[:, index]
        else:
            position = self.env.scene[lab_scene.body_entity(name, body)].data.root_com_pos_w
        return (position - self.env.scene.env_origins).cpu().numpy()

    def set_shape_pose(self, name, transform):
        """Teleport a shape to world ``transform`` (every environment) and stop it."""
        pose = pose_tensor(transform, self.device).repeat(self.num_envs, 1)
        pose[:, :3] += self.env.scene.env_origins
        place(self.env, self.env.scene[name], pose, torch.arange(self.num_envs, device=self.device))

    def push_shape(self, name, force):
        """Push a shape's centre of mass with the world ``force`` (N, every environment) until changed. The
        force is held in the shape's frame, so it turns with the shape: call again each step to keep it along
        a world direction. A zero ``force`` stops pushing."""
        # The permanent composer, as the instantaneous one is cleared after each physics step of the
        # several in a control step. It converts with the orientation read on its first call after a
        # reset: resetting first converts with the current orientation.
        composer = self.env.scene[name].permanent_wrench_composer
        composer.reset()
        force = torch.as_tensor(force, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1, 1)
        if force.any():
            composer.set_forces_and_torques(forces=force, torques=torch.zeros_like(force), is_global=True)

    def link_pose(self, link):
        """``[N, 4, 4]`` world pose of a robot body as simulated."""
        index = self.robot.body_names.index(link)
        pose = self.robot.data.body_link_pose_w[:, index].clone()
        pose[:, :3] -= self.env.scene.env_origins
        return transforms(pose).cpu().numpy()

    # -- rendering -------------------------------------------------------------------------
    def render(self, names):
        """Render the current state and read cameras ``names``: ``{name: {'rgb'?, 'depth'?}}`` as NumPy
        ``[N, H, W, 3]`` uint8 and ``[N, H, W]`` metres (0 where nothing is hit)."""
        import time
        started = time.perf_counter()
        self.env.sim.render()
        if self.env.device != 'cpu':
            # Flush newly authored visual geometry before reading annotators.
            self.env.sim.render()
        out = {}
        for name in names:
            camera = self.env.scene[name]
            camera.update(dt=0., force_recompute=True)
            data = camera.data.output
            entry = {}
            if 'rgb' in data:
                entry['rgb'] = data['rgb'][..., :3].cpu().numpy().astype(np.uint8)
            if 'distance_to_image_plane' in data:
                depth = data['distance_to_image_plane'][..., 0]
                entry['depth'] = torch.nan_to_num(depth, nan=0., posinf=0., neginf=0.).cpu().numpy().astype(np.float32)
            out[name] = entry
        self.timings['render_s'] += time.perf_counter() - started
        return out

    def images(self, names=None):
        """Robot camera images ``{name: [N, H, W, 3] uint8 or [N, H, W] metres}`` at the current state
        (default: every robot camera), with the observation's ``sensor_param``."""
        names = list(self.camera_specs) if names is None else list(names)
        rendered = self.render(names)
        images = {name: entry.get('rgb', entry.get('depth')) for name, entry in rendered.items()}
        params = self.observe().get('sensor_param', {})
        return {'images': images, 'sensor_param': {name: params[name] for name in names}}

    def show_robot(self, visible=True):
        """Show or hide the robot in rendered images (physics is unchanged)."""
        from pxr import UsdGeom
        for path in sim_utils.find_matching_prim_paths('/World/envs/env_.*/Robot'):
            prim = UsdGeom.Imageable(self.env.sim.stage.GetPrimAtPath(path))
            prim.MakeVisible() if visible else prim.MakeInvisible()

    def set_markers(self, points, radius):
        """Show visual-only spheres (no collision, no mass) at world ``points`` (environment 0);
        earlier markers beyond them are hidden. They render in rgb and depth alike (Isaac Lab's
        visualization markers stay out of depth images), and being visual only, they change no
        physics."""
        from pxr import Gf, UsdGeom
        stage = self.env.sim.stage
        origin = self.env.scene.env_origins[0].cpu().numpy()
        count = self.markers or 0
        for i, point in enumerate(points):
            sphere = UsdGeom.Sphere.Define(stage, f'{MARKERS_PATH}/marker{i}')
            if i >= count:
                sphere.CreateDisplayColorAttr([Gf.Vec3f(.9, .2, .1)])
                UsdGeom.Xformable(sphere).AddTranslateOp()
            sphere.GetRadiusAttr().Set(float(radius))
            UsdGeom.Xformable(sphere).GetOrderedXformOps()[0].Set(Gf.Vec3d(*map(float, np.asarray(point) + origin)))
            UsdGeom.Imageable(sphere).MakeVisible()
        for i in range(len(points), count):
            UsdGeom.Imageable(stage.GetPrimAtPath(f'{MARKERS_PATH}/marker{i}')).MakeInvisible()
        self.markers = max(count, len(points))

    def close(self):
        """Close the environment and empty the stage, so the process can build another one."""
        import omni.usd
        self.contacts.close()
        self.env.close()
        omni.usd.get_context().new_stage()
