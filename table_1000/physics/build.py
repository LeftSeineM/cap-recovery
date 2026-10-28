"""Export Blender geometry and compose a native USD physics layer."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


RIGID_CONFIG = Path(__file__).resolve().parents[2] / 'configs' / 'rigid_physics.json'
RIGID_SCRIPT = """\"\"\"One rigid body, generated from configs/rigid_physics.json.\"\"\"

from table_1000.physics.rigid import author as author_rigid

VALUES = {values!r}


def author(geometry_path, output_path):
    author_rigid(geometry_path, output_path, **VALUES)
"""


def rigid_assets(config=RIGID_CONFIG):
    """``{asset path: values}`` of the assets built as one rigid body, from ``configs/rigid_physics.json``:
    ``mass_kg`` in total, ``friction`` ``[static, dynamic]``, optional ``angular_damping`` (1/s) and
    ``center_of_mass`` (m, body frame) and ``position_iterations`` (default 16); the keyword arguments
    of :func:`table_1000.physics.rigid.author`."""
    return json.loads(Path(config).read_text())


def author_physics(source, output, values=None):
    """Author ``physics.usda`` with the asset's own ``physics.py``, or without one, with a script made from
    ``values`` (its entry in ``configs/rigid_physics.json``); the script used is kept as ``physics.py``."""
    if (source / 'physics.py').is_file():
        shutil.copyfile(source / 'physics.py', output / 'physics.py')
    elif values is not None:
        (output / 'physics.py').write_text(RIGID_SCRIPT.format(values=values))
    else:
        raise FileNotFoundError(f'{source}: no physics.py and not in {RIGID_CONFIG}')
    spec = importlib.util.spec_from_file_location('asset_physics', output / 'physics.py')
    physics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(physics)
    physics.author(output / 'geometry.usdc', output / 'physics.usda')


def author_geometry(model, destination):
    import numpy as np
    from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdPhysics, UsdShade
    stage = Usd.Stage.CreateNew(str(destination))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, '/Asset')
    stage.SetDefaultPrim(root.GetPrim())

    def define_material(path, settings):
        """A UsdPreviewSurface material (plus OmniSurface for transmission) at ``path``."""
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, path + '/Shader')
        shader.CreateIdAttr('UsdPreviewSurface')
        for key, field, kind in [('diffuseColor', 'color', Sdf.ValueTypeNames.Color3f),
                                 ('roughness', 'roughness', Sdf.ValueTypeNames.Float),
                                 ('metallic', 'metallic', Sdf.ValueTypeNames.Float)]:
            value = Gf.Vec3f(*settings[field]) if field == 'color' else settings[field]
            shader.CreateInput(key, kind).Set(value)
        shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(settings['alpha'])
        shader.CreateInput('ior', Sdf.ValueTypeNames.Float).Set(settings['ior'])
        shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        if settings['transmission'] > 0:
            # PreviewSurface opacity is a cutout, not Principled transmission.
            # Isaac's built-in OmniSurface carries the separate refractive lobe.
            mdl = UsdShade.Shader.Define(stage, path + '/OmniSurface')
            mdl.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
            mdl.SetSourceAsset(Sdf.AssetPath('OmniSurface.mdl'), 'mdl')
            mdl.SetSourceAssetSubIdentifier('OmniSurface', 'mdl')
            for key, value, kind in [
                ('diffuse_reflection_weight', 1., Sdf.ValueTypeNames.Float),
                ('diffuse_reflection_color', Gf.Vec3f(*settings['color']), Sdf.ValueTypeNames.Color3f),
                ('metalness', settings['metallic'], Sdf.ValueTypeNames.Float),
                ('specular_reflection_roughness', settings['roughness'], Sdf.ValueTypeNames.Float),
                ('specular_reflection_ior', settings['ior'], Sdf.ValueTypeNames.Float),
                ('enable_specular_transmission', True, Sdf.ValueTypeNames.Bool),
                ('specular_transmission_weight', settings['transmission'], Sdf.ValueTypeNames.Float),
                # Cycles Principled tints each solid interface with sqrt(base).
                ('specular_transmission_color', Gf.Vec3f(*np.sqrt(np.clip(settings['color'], 0, 1))), Sdf.ValueTypeNames.Color3f),
                ('enable_opacity', settings['alpha'] < 1, Sdf.ValueTypeNames.Bool),
                ('geometry_opacity', settings['alpha'], Sdf.ValueTypeNames.Float),
            ]:
                mdl.CreateInput(key, kind).Set(value)
            mdl.CreateOutput('out', Sdf.ValueTypeNames.Token)
            for output in [material.CreateSurfaceOutput, material.CreateVolumeOutput,
                           material.CreateDisplacementOutput]:
                output('mdl').ConnectToSource(mdl.ConnectableAPI(), 'out')
        return material

    paths, worlds = {}, {}
    for node in model['nodes']:
        name, parent = node['name'], node['parent']
        local_name = name[len(parent) + 1:] if parent and name.startswith(parent + '.') else name
        path = paths.get(parent, '/Asset') + '/' + Tf.MakeValidIdentifier(local_name)
        if path in paths.values():
            raise ValueError(f'USD name collision: {name}')
        paths[name] = path
        matrix = np.array(node['matrix'])
        worlds[name] = worlds[parent] @ matrix if parent else matrix
        role = node.get('role')
        if role == 'collision' and node['shape'] == 'SPHERE':
            obj = UsdGeom.Sphere.Define(stage,path)
            obj.CreateRadiusAttr(node['radius'])
        elif role:
            obj = UsdGeom.Mesh.Define(stage, path)
            obj.CreatePointsAttr([Gf.Vec3f(*value) for value in node['vertices']])
            obj.CreateFaceVertexCountsAttr([len(face) for face in node['faces']])
            obj.CreateFaceVertexIndicesAttr([index for face in node['faces'] for index in face])
            obj.CreateSubdivisionSchemeAttr('none')
        else:
            obj = UsdGeom.Xform.Define(stage, path)
        prim = obj.GetPrim()
        prim.SetCustomDataByKey('table1000:name', name)
        prim.SetCustomDataByKey('table1000:role', 'rigid' if node['rigid'] else 'part' if node['part'] else role or 'organization')
        UsdGeom.Xformable(prim).AddTransformOp().Set(Gf.Matrix4d(*matrix.T.reshape(-1).tolist()))
        if node['rigid']:
            UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
        elif role == 'collision':
            obj.CreateVisibilityAttr('invisible')
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            if node['shape'] != 'SPHERE':
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('boundingCube' if node['shape'] == 'BOX' else 'convexHull')
        elif role == 'visual':
            obj.CreateNormalsAttr([Gf.Vec3f(*value) for value in node['normals']])
            obj.SetNormalsInterpolation('faceVarying')
            settings = node['material']
            obj.CreateDisplayColorAttr([Gf.Vec3f(*settings['color'])])
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            binding.Bind(define_material(path + '/Material', settings))
            if 'materials' in node:     # per-face materials: one face subset bound to each slot
                faces = np.array(node['face_materials'])
                for index, slot in enumerate(node['materials']):
                    members = np.flatnonzero(faces == index)
                    if len(members):
                        subset = binding.CreateMaterialBindSubset(f'material{index}', members.tolist())
                        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(
                            define_material(f'{path}/Material{index}', slot))
    for row in model['joints']:
        path = '/Asset/Joints/' + Tf.MakeValidIdentifier(row['name'])
        cls = UsdPhysics.PrismaticJoint if row['type'] == 'SLIDER' else UsdPhysics.RevoluteJoint
        joint = cls.Define(stage, path)
        joint.GetPrim().SetCustomDataByKey('table1000:name', row['name'])
        joint.GetPrim().SetCustomDataByKey('table1000:role', 'joint')
        joint.CreateBody0Rel().SetTargets([paths[row['body1']]])
        joint.CreateBody1Rel().SetTargets([paths[row['body2']]])
        joint.CreateAxisAttr('X' if row['type'] == 'SLIDER' else 'Z')
        for index, body in enumerate([row['body1'], row['body2']]):
            local = np.linalg.inv(worlds[body]) @ np.array(row['matrix'])
            transform = Gf.Transform(Gf.Matrix4d(*local.T.reshape(-1).tolist()))
            q = transform.GetRotation().GetQuat()
            getattr(joint, f'CreateLocalPos{index}Attr')(Gf.Vec3f(*transform.GetTranslation()))
            getattr(joint, f'CreateLocalRot{index}Attr')(Gf.Quatf(q.GetReal(), Gf.Vec3f(*q.GetImaginary())))
        factor = 1 if row['type'] == 'SLIDER' else 180 / np.pi
        if row['limits']:
            joint.CreateLowerLimitAttr(row['limits'][0] * factor)
            joint.CreateUpperLimitAttr(row['limits'][1] * factor)
        joint.CreateCollisionEnabledAttr(not row['disable_collisions'])
    stage.GetRootLayer().Save()
    return paths


def build(asset, source, output, values=None):
    from pxr import Sdf, Usd, UsdPhysics, UsdShade, UsdUtils
    from omni.kit.material.library import get_mdl_lib_paths
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        model_path = temporary / 'model.json'
        subprocess.run(['blender', '-b', '--python-exit-code', '1', '-P', str(Path(__file__).with_name('blender_model.py')),
                        '--', '--blend', str(asset / 'object.blend'), '--output', str(model_path)], check=True)
        model = json.loads(model_path.read_text())
        paths = author_geometry(model, output / 'geometry.usdc')
        author_physics(source, output, values)
        composed = Usd.Stage.CreateInMemory()
        composed.GetRootLayer().subLayerPaths = [str(output / 'physics.usda'), str(output / 'geometry.usdc')]
        composed.SetDefaultPrim(composed.GetPrimAtPath('/Asset'))
        scripts = [Sdf.AssetPath('behavior.py')] if (source / 'behavior.py').exists() else []
        # Standard USDZ contains the physical asset. Executable scripts remain
        # beside it, with their official component carried by an outer layer.
        flattened = composed.Flatten()
        packaged_stage = Usd.Stage.Open(flattened)
        root = packaged_stage.GetPrimAtPath('/Asset')
        root.RemoveProperty('omni:scripting:scripts')
        root.SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit([name for name in root.GetAppliedSchemas() if name != 'OmniScriptingAPI']))
        if any(slot['transmission'] > 0 for node in model['nodes']
               for slot in node.get('materials', [node['material']] if 'material' in node else [])):
            # MDL's runtime search path is separate from USD's asset resolver.
            mdl = next(Path(path) / 'OmniSurface.mdl' for path in get_mdl_lib_paths()
                       if (Path(path) / 'OmniSurface.mdl').is_file())
            UsdUtils.ModifyAssetPaths(flattened, lambda path: str(mdl) if path == 'OmniSurface.mdl' else path)
        flattened.Export(str(temporary / 'object.usdc'))
        if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(temporary / 'object.usdc')), str(output / 'object.usdz')):
            raise RuntimeError('USDZ packaging failed')
        if scripts:
            carrier = Usd.Stage.CreateNew(str(output / 'object.usda'))
            from pxr import UsdGeom
            root = UsdGeom.Xform.Define(carrier, '/Asset').GetPrim()
            root.GetReferences().AddReference('object.usdz')
            carrier.SetDefaultPrim(root)
            root.SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit(['OmniScriptingAPI']))
            root.CreateAttribute('omni:scripting:scripts', Sdf.ValueTypeNames.AssetArray).Set([Sdf.AssetPath(path.path) for path in scripts])
            for path in scripts:
                shutil.copyfile(source / path.path, output / path.path)
            carrier.GetRootLayer().Save()
        else:
            (output / 'object.usda').unlink(missing_ok=True)
            (output / 'behavior.py').unlink(missing_ok=True)
        packaged = Usd.Stage.Open(str(output / 'object.usdz'))
        rigid = [prim for prim in packaged.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        # The configured one-body assets merge their modelled bodies (table_1000.physics.rigid).
        assert len(rigid) == (1 if values is not None else sum(node['rigid'] for node in model['nodes']))
        assert all(UsdPhysics.MassAPI(prim).GetMassAttr().Get() > 0 for prim in rigid)
        # Authored centres of mass (body frame), for what the colliders' uniform density would put wrong.
        centers = {prim.GetCustomDataByKey('table1000:name'): list(UsdPhysics.MassAPI(prim).GetCenterOfMassAttr().Get())
                   for prim in rigid if UsdPhysics.MassAPI(prim).GetCenterOfMassAttr().HasAuthoredValue()}
        materials = {}
        for prim in packaged.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                material, rel = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
                assert material and material.GetPrim().HasAPI(UsdPhysics.MaterialAPI)
                materials[str(prim.GetPath())] = str(material.GetPath())
    for name in ['object.py','preview.json','metadata.json','README.md','three_quarter.jpg']:
        original=source/name
        if original.exists() and original != output/name:
            shutil.copyfile(original,output/name)
    if (source / 'physics_test.py').exists():
        shutil.copyfile(source / 'physics_test.py', output / 'physics_test.py')
    (output / 'model.json').write_text(json.dumps(model) + '\n')
    report = {'rigid_bodies': len(rigid), 'parts': sum(node['part'] for node in model['nodes']),
              'colliders': len(materials), 'physics_materials': materials, 'paths': paths,
              'centers_of_mass': centers,
              'geometry_layer': 'geometry.usdc', 'physics_source': 'physics.py', 'physics_layer': 'physics.usda',
              'entry': 'object.usda' if scripts else 'object.usdz',
              'behavior_scripts': [path.path for path in scripts],
              'build_wall_seconds': time.perf_counter() - started}
    (output / 'physics_build.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PHYSICAL_ASSET_BUILT', output / report['entry'], flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('asset_sources/objects'))
    parser.add_argument('--output', type=Path, default=Path('assets/objects'))
    parser.add_argument('--assets', type=Path, nargs='+',
                        help='Asset paths relative to the source/output roots (default: every asset with a physics.py '
                             'or in configs/rigid_physics.json)')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args(argv)
    source, output = args.source.resolve(), args.output.resolve()
    rigid = rigid_assets()
    references = args.assets if args.assets is not None else sorted(
        {path.parent.relative_to(source) for path in source.rglob('physics.py')} | set(map(Path, rigid)))
    from table_1000.common.path import configure_isaac_paths
    configure_isaac_paths()
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'active_gpu': args.gpu, 'physics_gpu': 0 if os.getenv('CUDA_VISIBLE_DEVICES') == str(args.gpu) else args.gpu,
                         'multi_gpu': False, 'create_new_stage': False,
                         'disable_viewport_updates': True, 'limit_cpu_threads': 4,
                         'extra_args': ['--enable', 'omni.kit.material.library']})
    try:
        for reference in references:
            build(output / reference, source / reference, output / reference, rigid.get(Path(reference).as_posix()))
    except BaseException:
        # Kit closes the process before Python can propagate an exception.
        import traceback
        import omni.kit.app
        traceback.print_exc()
        omni.kit.app.get_app().post_quit(1)
        raise
    finally:
        app.close(wait_for_replicator=False)
