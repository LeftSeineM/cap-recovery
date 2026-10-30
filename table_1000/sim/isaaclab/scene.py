"""Isaac Lab scene configuration of a scene source: ground, lights, table, robot, shapes, cameras.

A scene source is a Python module (or namespace) with ``TABLE`` (a table built
from boxes), ``ROBOT`` (``asset``, base ``position`` and ``yaw``, starting
``joints``), optionally ``OBJECTS`` (``{name: {'asset': path under assets/}}``,
built assets) and ``PRIMITIVES``: simple rigid shapes for tests (``{name:
{'shape': 'box' | 'sphere' | 'cylinder', 'size': ..., 'mass': kg or None,
'friction'?, 'color'?}}``, posed by ``scene.placement()``); a primitive without
mass is kinematic, so it stays where it is put.

An asset object is referenced unchanged from its built USD (:func:`object_cfgs`):
an asset with an articulation root (a cabinet with drawers) becomes an Isaac Lab
articulation; one whose rigid bodies are joined without one (a pen whose cap is
held by spring-loaded balls) is spawned as a whole and each body becomes a rigid
object, so Isaac Lab's state reads and writes cover every body. Objects start
parked far above the scene; resets place them (:func:`table_1000.sim.isaaclab.env.apply_layout`).

Each environment's prims live under ``{ENV_REGEX_NS}`` (Isaac Lab's
``/World/envs/env_<i>``), all entities are declared here before the simulation
starts, and the environment changes them only through Isaac Lab's state and
target writes: the same scene runs on CPU and GPU PhysX. Robot bodies and shapes
report contacts (:mod:`table_1000.sim.isaaclab.contacts`).
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from table_1000.scenes import robot_config

import numpy as np
from scipy.spatial.transform import Rotation

from table_1000.assets import usd_entry

GROUND_PATH = '/World/defaultGroundPlane'
SHAPE_PREFIX = 'Shape_'   # shape prims are {ENV_REGEX_NS}/Shape_<name>
OBJECT_PREFIX = 'Object_'  # asset objects are {ENV_REGEX_NS}/Object_<name>
PARKED_HEIGHT = 20.        # m, objects before the first reset (one 2 m apart from the next)
SHAPE_CONTACT_OFFSET = .002


def quaternion(rotation):
    """``(w, x, y, z)`` of a 3x3 rotation matrix."""
    return tuple(Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]])


def pose(transform):
    """``(position, (w, x, y, z))`` of a 4x4 transform."""
    transform = np.asarray(transform, dtype=float)
    return tuple(transform[:3, 3]), quaternion(transform[:3, :3])


def robot_base(robot):
    transform = np.eye(4)
    c, s = np.cos(robot['yaw']), np.sin(robot['yaw'])
    transform[:3, :3] = [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]
    transform[:3, 3] = robot['position']
    return transform


def spawn_table(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """A table of five static boxes (top and legs) sharing one friction material."""
    from isaaclab.sim.utils.prims import clone
    return clone(_spawn_table)(prim_path, cfg, translation, orientation, **kwargs)


def _spawn_table(prim_path, cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim_utils
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade
    stage = sim_utils.get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    table = cfg.table
    (width, depth), height, thickness, leg = table['size'], table['height'], table['top_thickness'], table['leg_size']
    cx, cy = table['center']
    parts = {'top': ((cx, cy, height - thickness / 2), (width, depth, thickness))}
    for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
        parts[f'leg{i + 1}'] = ((cx + sx * (width - leg) / 2, cy + sy * (depth - leg) / 2, (height - thickness) / 2),
                                (leg, leg, height - thickness))
    material = UsdShade.Material.Define(stage, prim_path + '/PhysicsMaterial')
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(table['friction'][0])
    api.CreateDynamicFrictionAttr(table['friction'][1])
    api.CreateRestitutionAttr(0.)
    for name, (center, size) in parts.items():
        cube = UsdGeom.Cube.Define(stage, f'{prim_path}/{name}')
        cube.CreateSizeAttr(1.)
        cube.CreateDisplayColorAttr([Gf.Vec3f(.55, .42, .3)])
        xform = UsdGeom.Xformable(cube)
        xform.AddTranslateOp().Set(Gf.Vec3d(*center))
        xform.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim()).CreateContactOffsetAttr(SHAPE_CONTACT_OFFSET)
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material, materialPurpose='physics')
    sim_utils.standardize_xform_ops(root.GetPrim())
    return root.GetPrim()


def spawn_ground(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Isaac Lab's ground plane, not colliding with the robot: a robot with a mobile base is
    carried by its virtual joints, and its wheels must not drag on the floor."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.from_files import spawn_ground_plane
    from pxr import Usd, UsdPhysics
    prim = spawn_ground_plane(prim_path, cfg, translation, orientation, **kwargs)
    if cfg.filtered_root:
        stage = sim_utils.get_current_stage()
        bodies = [prim.GetPath() for root in sim_utils.find_matching_prim_paths(cfg.filtered_root)
                  for prim in Usd.PrimRange(stage.GetPrimAtPath(root)) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        for collider in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
            if collider.HasAPI(UsdPhysics.CollisionAPI):
                relation = UsdPhysics.FilteredPairsAPI.Apply(collider).CreateFilteredPairsRel()
                for body in bodies:
                    relation.AddTarget(body)
    return prim


def spawner_classes():
    """Configuration classes of the table and the ground (defined once Isaac Lab is importable)."""
    import isaaclab.sim as sim_utils
    from isaaclab.utils import configclass

    @configclass
    class TableCfg(sim_utils.SpawnerCfg):
        func: object = spawn_table
        table: dict = None

    @configclass
    class GroundCfg(sim_utils.GroundPlaneCfg):
        func: object = spawn_ground
        filtered_root: str = ''

    return TableCfg, GroundCfg


def with_ccd(spawn):
    """``spawn`` followed by enabling continuous collision detection on every spawned body
    (Isaac Lab's rigid body properties have no CCD switch)."""
    def spawn_with_ccd(prim_path, cfg, *args, **kwargs):
        import isaaclab.sim as sim_utils
        from pxr import PhysxSchema
        prim = spawn(prim_path, cfg, *args, **kwargs)
        for path in sim_utils.find_matching_prim_paths(prim_path):
            PhysxSchema.PhysxRigidBodyAPI.Apply(sim_utils.get_current_stage().GetPrimAtPath(path)).CreateEnableCCDAttr(True)
        return prim
    return spawn_with_ccd


def shape_cfg(spec, contacts=True):
    """Spawner of a primitive shape: dynamic with CCD, or kinematic without a mass."""
    import isaaclab.sim as sim_utils
    kinematic = spec.get('mass') is None
    common = dict(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic),
        mass_props=None if kinematic else sim_utils.MassPropertiesCfg(mass=float(spec['mass'])),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=SHAPE_CONTACT_OFFSET),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=spec.get('friction', (.6, .5))[0], dynamic_friction=spec.get('friction', (.6, .5))[1],
            restitution=0.),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(spec.get('color', (.2, .45, .8)))),
        activate_contact_sensors=contacts)
    shape, size = spec['shape'], spec['size']
    if shape == 'box':
        cfg = sim_utils.CuboidCfg(size=tuple(size), **common)
    elif shape == 'sphere':
        cfg = sim_utils.SphereCfg(radius=float(size), **common)
    elif shape == 'cylinder':
        # A 32-sided convex mesh: on PhysX's exact cylinder a cylinder lying on the table rolls by
        # itself (to 17 mm/s within a second), the mesh rests on one of its faces.
        cfg = sim_utils.MeshCylinderCfg(radius=float(size[0]), height=float(size[1]), axis='Z', **common)
    else:
        raise ValueError(f'unknown primitive shape {shape}')
    if not kinematic:
        cfg.func = with_ccd(cfg.func)
    return cfg


@dataclass(frozen=True)
class ObjectStructure:
    """Rigid bodies of a built asset, named as in its ``model.json`` (``table1000:name``).

    ``paths`` maps a body to its prim path below the asset's root prim, ``rest``
    to its pose in the asset frame as built. ``root`` is the articulation's root
    body, or None for bodies joined without an articulation.
    """
    entry: str
    paths: dict
    rest: dict
    root: str = None

    @property
    def articulated(self):
        return self.root is not None


@lru_cache
def object_structure(entry):
    """:class:`ObjectStructure` of the USD ``entry`` (opened with ``pxr``, after ``start_app``)."""
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.Open(str(entry))
    asset = stage.GetDefaultPrim()
    cache = UsdGeom.XformCache()
    inverse = np.linalg.inv(np.array(cache.GetLocalToWorldTransform(asset)).T)
    paths, rest, root = {}, {}, None
    for prim in Usd.PrimRange(asset):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        name = prim.GetCustomDataByKey('table1000:name')
        paths[name] = str(prim.GetPath())[len(str(asset.GetPath())):]
        rest[name] = inverse @ np.array(cache.GetLocalToWorldTransform(prim)).T
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            root = name
    return ObjectStructure(str(entry), paths, rest, root)


def object_structures(scene, assets):
    """``{name: ObjectStructure}`` of the scene's asset objects."""
    return {name: object_structure(str(usd_entry(Path(assets) / spec['asset'])))
            for name, spec in getattr(scene, 'OBJECTS', {}).items()}


def body_entity(name, body):
    """Scene entity name of the rigid body ``body`` of an unarticulated object."""
    return f'{name}__{body.replace(".", "_")}'


def object_cfgs(name, structure, index):
    """``{entity: cfg}`` of one asset object, parked ``index`` places above the scene."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    prim_path = f'{{ENV_REGEX_NS}}/{OBJECT_PREFIX}{name}'
    spawn = sim_utils.UsdFileCfg(usd_path=structure.entry)
    parked = (0., 0., PARKED_HEIGHT + 2. * index)
    if structure.articulated:
        # Joints without drives (sliding drawers) stay passive: gains come from the USD.
        return {name: ArticulationCfg(
            prim_path=prim_path, spawn=spawn, init_state=ArticulationCfg.InitialStateCfg(pos=parked),
            actuators={'joints': ImplicitActuatorCfg(joint_names_expr=['.*'], stiffness=None, damping=None)})}
    cfgs = {name: AssetBaseCfg(prim_path=prim_path, spawn=spawn, init_state=AssetBaseCfg.InitialStateCfg(pos=parked))}
    for body, path in structure.paths.items():
        cfgs[body_entity(name, body)] = RigidObjectCfg(prim_path=prim_path + path, spawn=None)
    return cfgs


def robot_cfg(scene, spec, assets, robot_q):
    """The robot as an Isaac Lab articulation from its built USD; drives keep the gains its
    ``physics.py`` authored. A robot with a mobile base is built at the world origin and moved by
    its virtual base joints; a fixed robot stands at the scene's base pose."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    position, rotation = pose(np.eye(4) if spec.base else robot_base(robot_config(scene, spec.name)))
    return ArticulationCfg(
        prim_path='{ENV_REGEX_NS}/Robot',
        spawn=sim_utils.UsdFileCfg(usd_path=str(usd_entry(Path(assets) / robot_config(scene, spec.name)['asset'])),
                                   activate_contact_sensors=True),
        init_state=ArticulationCfg.InitialStateCfg(pos=position, rot=rotation,
                                                   joint_pos={name: float(value) for name, value in robot_q.items()}),
        actuators={'drives': ImplicitActuatorCfg(joint_names_expr=['.*'], stiffness=None, damping=None)})


def scene_cfg(scene, spec, assets, robot_q, *, num_envs=1, cameras=None, depth=False, views=None, root=None):
    """``InteractiveSceneCfg`` of ``scene`` with ``spec``'s robot at joints ``robot_q``.

    ``cameras``: the robot cameras (``{name: CameraSpec}``, default none) to render;
    ``depth`` adds ground-truth depth to rgb streams. ``views``: fixed video views
    (:func:`~table_1000.sim.isaaclab.cameras.view_camera_cfg`). Shapes start at
    ``scene.placement()``, asset objects parked (see :func:`object_cfgs`).
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from table_1000.robots import ROOT
    from table_1000.sim.isaaclab.cameras import robot_camera_cfg, view_camera_cfg
    from table_1000.sim.isaaclab.render import light_cfgs
    TableCfg, GroundCfg = spawner_classes()
    cfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=5., replicate_physics=False)
    # Entities are created in this order; cameras and the ground filter need the robot first.
    cfg.robot = robot_cfg(scene, spec, assets, robot_q)
    cfg.ground = AssetBaseCfg(prim_path=GROUND_PATH, spawn=GroundCfg(
        filtered_root='/World/envs/env_.*/Robot' if spec.base else '',
        # Isaac Sim's default ground material, as before Isaac Lab.
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.5, dynamic_friction=.5, restitution=.8)))
    cfg.table = AssetBaseCfg(prim_path='{ENV_REGEX_NS}/Table', spawn=TableCfg(table=dict(scene.TABLE)))
    shapes = getattr(scene, 'PRIMITIVES', {})
    layout = scene.placement() if shapes else {}
    for name, shape in shapes.items():
        position, rotation = pose(layout[name])
        setattr(cfg, name, RigidObjectCfg(prim_path=f'{{ENV_REGEX_NS}}/{SHAPE_PREFIX}{name}', spawn=shape_cfg(shape),
                                          init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=rotation)))
    for index, (name, structure) in enumerate(object_structures(scene, assets).items()):
        for entity, entity_cfg in object_cfgs(name, structure, index).items():
            setattr(cfg, entity, entity_cfg)
    for name, light in light_cfgs().items():
        setattr(cfg, name, light)
    urdf = Path(root or ROOT) / spec.urdf
    for name, camera in (cameras or {}).items():
        camera_cfg = robot_camera_cfg(camera, urdf, '{ENV_REGEX_NS}/Robot', depth)
        camera_cfg.prim_path = camera_cfg.prim_path.replace('{name}', name)
        setattr(cfg, name, camera_cfg)
    for name, view in (views or {}).items():
        view_cfg = view_camera_cfg(view)
        view_cfg.prim_path = view_cfg.prim_path.replace('{name}', name)
        setattr(cfg, name, view_cfg)
    return cfg
