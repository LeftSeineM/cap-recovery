"""Native spring/ball contact detent; no pen retention behavior is required.

Frozen mechanical baseline from Claude's contribution c1d058f and the native
research candidate saved as 67f6aee/519d6df. Masses and gains are engineering
proxies, not measured material calibration. Each 0.8 mm ball uses 1 g mass.
"""
from pathlib import Path

MASSES = {'body': .012, 'cap': .003, **{f'cap_ball{i}': .001 for i in range(1,4)}}


def author(geometry_path, output_path):
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
    layer = Sdf.Layer.CreateNew(str(Path(output_path).resolve()))
    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().subLayerPaths = [layer.identifier, str(Path(geometry_path).resolve())]
    stage.SetEditTarget(layer)
    UsdGeom.Scope.Define(stage, '/Asset/Materials')
    materials = {}
    for name, friction in [('plastic', (.35,.30)), ('steel', (.15,.15))]:
        material = UsdShade.Material.Define(stage, '/Asset/Materials/'+name)
        api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr(friction[0]); api.CreateDynamicFrictionAttr(friction[1])
        api.CreateRestitutionAttr(0)
        physx = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        physx.CreateFrictionCombineModeAttr('average'); physx.CreateRestitutionCombineModeAttr('average')
        materials[name] = material
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/Asset')):
        name = prim.GetCustomDataByKey('table1000:name')
        role = prim.GetCustomDataByKey('table1000:role')
        if role == 'rigid':
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(MASSES[name])
            body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            body.CreateLinearDampingAttr(0); body.CreateAngularDampingAttr(0)
            body.CreateSolverPositionIterationCountAttr(32); body.CreateSolverVelocityIterationCountAttr(8)
            # At 4 ms, a falling 0.8 mm detent ball can cross a thin shelf in
            # one step and become trapped below it by its spring constraint.
            body.CreateEnableCCDAttr(True)
            body.CreateSleepThresholdAttr(0); body.CreateStabilizationThresholdAttr(0)
            body.CreateMaxDepenetrationVelocityAttr(1)
        elif role == 'part':
            material = materials['steel' if name.startswith('cap_ball') else 'plastic']
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose='physics')
        elif role == 'collision':
            collider = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collider.CreateContactOffsetAttr(.0005); collider.CreateRestOffsetAttr(0)
        elif role == 'joint':
            drive = UsdPhysics.DriveAPI.Apply(prim,'linear')
            drive.CreateTypeAttr('force'); drive.CreateTargetPositionAttr(-.0003)
            drive.CreateStiffnessAttr(1750); drive.CreateDampingAttr(.3)
    layer.Save()
    return layer
