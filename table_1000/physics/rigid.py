"""Physics layer of an asset simulated as one rigid body: total mass and one contact material.

The assets in ``configs/rigid_physics.json`` are built with :func:`author` and their values there.

Every rigid body of the geometry layer becomes part of a single body at the
bodies' common parent prim (the asset root when they are siblings under it): the
original bodies lose ``RigidBodyAPI`` and their joints are deactivated, so caps,
lids, clip handles and buttons stay where they were modelled. The merged body
takes ``table1000:name`` of that parent, or ``object`` at the asset root. PhysX
derives centre of mass and inertia from the colliders.

PhysX has no rolling resistance: a pen on its round barrel rolls on until it
hits something. Round assets give ``angular_damping`` (1/s) as a stand-in.
"""

from pathlib import Path

CONTACT_OFFSET_M = .001
ROOT_NAME = 'object'


def author(geometry_path, output_path, mass_kg, friction, *, restitution=0., angular_damping=0., center_of_mass=None,
           position_iterations=16):
    """Write the physics layer over ``geometry_path`` into ``output_path``: ``mass_kg`` in total,
    ``friction`` ``(static, dynamic)`` on every collider, ``angular_damping`` in 1/s, and
    ``center_of_mass`` (m, in the body's frame) where the colliders' uniform density would put it
    wrong (light reeds or flowers over a heavy base). ``position_iterations`` controls the
    body's position solver (default 16); the velocity solver stays at 4 iterations."""
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    layer = Sdf.Layer.CreateNew(str(Path(output_path).resolve()))
    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().subLayerPaths = [layer.identifier, str(Path(geometry_path).resolve())]
    stage.SetEditTarget(layer)

    asset = stage.GetPrimAtPath('/Asset')
    bodies = [prim for prim in Usd.PrimRange(asset) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    root = bodies[0] if len(bodies) == 1 else common_parent(bodies)
    for body in bodies:
        if body != root:
            body.RemoveAPI(UsdPhysics.RigidBodyAPI)
    for prim in Usd.PrimRange(asset):
        if prim.GetCustomDataByKey('table1000:role') == 'joint':
            prim.SetActive(False)
    if len(bodies) > 1:
        UsdPhysics.RigidBodyAPI.Apply(root).CreateRigidBodyEnabledAttr(True)
        if not root.GetCustomDataByKey('table1000:name'):
            root.SetCustomDataByKey('table1000:name', ROOT_NAME)
        root.SetCustomDataByKey('table1000:role', 'rigid')

    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(mass_kg))
    if center_of_mass is not None:
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*center_of_mass))
    body = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    body.CreateLinearDampingAttr(0)
    body.CreateAngularDampingAttr(float(angular_damping))
    body.CreateSolverPositionIterationCountAttr(position_iterations)
    body.CreateSolverVelocityIterationCountAttr(4)

    UsdGeom.Scope.Define(stage, '/Asset/Materials')
    material = UsdShade.Material.Define(stage, '/Asset/Materials/surface')
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(float(friction[0]))
    api.CreateDynamicFrictionAttr(float(friction[1]))
    api.CreateRestitutionAttr(float(restitution))
    physx = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx.CreateFrictionCombineModeAttr('average')
    physx.CreateRestitutionCombineModeAttr('average')
    for prim in Usd.PrimRange(asset):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose='physics')
            collider = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collider.CreateContactOffsetAttr(CONTACT_OFFSET_M)
            collider.CreateRestOffsetAttr(0)
    layer.Save()
    return layer


def common_parent(prims):
    """The deepest prim that is an ancestor of every prim in ``prims``."""
    paths = [prim.GetPath() for prim in prims]
    parent = paths[0].GetParentPath()
    while not all(path.HasPrefix(parent) for path in paths):
        parent = parent.GetParentPath()
    return prims[0].GetStage().GetPrimAtPath(parent)


def body_mapping(stage, paths, model):
    """Map each model body to its surviving physics body and ``T_physics_model``.

    Read the composed physics stage so custom physics authors and configured
    merges share the same geometry contract. Matrices use column vectors, as
    in ``model.json``; original part names are never changed.
    """
    import numpy as np
    from pxr import UsdGeom, UsdPhysics
    cache = UsdGeom.XformCache()
    mapping = {}
    for node in model['nodes']:
        if not node['rigid']:
            continue
        original = stage.GetPrimAtPath(paths[node['name']])
        body = original
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if not body:
            raise ValueError(f"{node['name']}: no physics rigid body after composition")
        relative = cache.GetLocalToWorldTransform(original) * cache.GetLocalToWorldTransform(body).GetInverse()
        mapping[node['name']] = {'body': body.GetCustomDataByKey('table1000:name'),
                                 'matrix': np.asarray(relative).T.tolist()}
    return mapping
