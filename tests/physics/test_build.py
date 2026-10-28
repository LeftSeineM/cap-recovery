"""The build entry uses the author hook and preserves its source snapshot."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from table_1000.physics.build import author_geometry, author_physics, main


class PhysicsAuthorTest(unittest.TestCase):
    def test_source_generates_layer_and_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'source', Path(directory) / 'asset'
            source.mkdir()
            output.mkdir()
            script = "def author(geometry_path, output_path):\n    output_path.write_text(geometry_path.read_text() + ':physics')\n"
            (source / 'physics.py').write_text(script)
            (source / 'physics.usda').write_text('unused old layer')
            (output / 'geometry.usdc').write_text('geometry')
            author_physics(source, output)
            self.assertEqual((output / 'physics.usda').read_text(), 'geometry:physics')
            self.assertEqual((output / 'physics.py').read_bytes(), (source / 'physics.py').read_bytes())

    def test_configured_rigid_asset_gets_a_generated_script(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'source', Path(directory) / 'asset'
            source.mkdir()
            output.mkdir()
            values = {'mass_kg': .02, 'friction': (.4, .3), 'angular_damping': 2.}
            with patch('table_1000.physics.rigid.author') as author:
                author_physics(source, output, values)
            author.assert_called_once_with(output / 'geometry.usdc', output / 'physics.usda', **values)
            self.assertIn('configs/rigid_physics.json', (output / 'physics.py').read_text())
            with self.assertRaises(FileNotFoundError):
                author_physics(source, output)


class PhysicsBatchTest(unittest.TestCase):
    def test_discovery_and_selection_share_one_application(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'source', Path(directory) / 'assets'
            for reference in ['cabinet/000000', 'pen/000002']:
                path = source / reference
                path.mkdir(parents=True)
                (path / 'physics.py').write_text('')
            rigid = {'battery/000000': {'mass_kg': .02, 'friction': (.4, .3)}}
            for selected in [None, ['pen/000002', 'cabinet/000000']]:
                arguments = ['--source', str(source), '--output', str(output), '--gpu', '3']
                references = ['battery/000000', 'cabinet/000000', 'pen/000002']
                if selected:
                    arguments += ['--assets', *selected]
                    references = selected
                app = Mock()
                simulation_app = Mock(return_value=app)
                with self.subTest(selected=selected), patch.dict(sys.modules, {'isaacsim': SimpleNamespace(SimulationApp=simulation_app)}), \
                        patch('table_1000.common.path.configure_isaac_paths') as configure_paths, \
                        patch('table_1000.physics.build.rigid_assets', return_value=rigid), \
                        patch('table_1000.physics.build.build') as build:
                    main(arguments)
                    configure_paths.assert_called_once_with()
                    self.assertEqual(build.call_args_list, [call(output/ref, source/ref, output/ref, rigid.get(ref))
                                                            for ref in references])
                    simulation_app.assert_called_once()
                    self.assertEqual(simulation_app.call_args.args[0]['active_gpu'], 3)
                    app.close.assert_called_once_with(wait_for_replicator=False)


class RigidIterationsTest(unittest.TestCase):
    def test_default_and_configured_iterations_survive_layer_composition(self):
        try:
            from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics
        except ImportError:
            self.skipTest('requires the Isaac/OpenUSD Python environment')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            for label, values in [('default', {}), ('thin', {'position_iterations': 64})]:
                output = root / label
                output.mkdir()
                geometry = Usd.Stage.CreateNew(str(output / 'geometry.usdc'))
                UsdGeom.Xform.Define(geometry, '/Asset')
                body = UsdGeom.Xform.Define(geometry, '/Asset/body').GetPrim()
                UsdPhysics.RigidBodyAPI.Apply(body)
                collider = UsdGeom.Cube.Define(geometry, '/Asset/body/collider').GetPrim()
                UsdPhysics.CollisionAPI.Apply(collider)
                geometry.GetRootLayer().Save()
                author_physics(source, output, {'mass_kg': .001, 'friction': [.3, .25], **values})
                composed = Usd.Stage.CreateInMemory()
                composed.GetRootLayer().subLayerPaths = [str(output / 'physics.usda'), str(output / 'geometry.usdc')]
                # Reopen a flattened deliverable, as the packaged asset is consumed.
                composed.Flatten().Export(str(output / 'composed.usdc'))
                stage = Usd.Stage.Open(str(output / 'composed.usdc'))
                prim = stage.GetPrimAtPath('/Asset/body')
                api = PhysxSchema.PhysxRigidBodyAPI(prim)
                self.assertEqual(api.GetSolverPositionIterationCountAttr().Get(), values.get('position_iterations', 16))
                self.assertEqual(api.GetSolverVelocityIterationCountAttr().Get(), 4)
                self.assertAlmostEqual(UsdPhysics.MassAPI(prim).GetMassAttr().Get(), .001)
                self.assertEqual(api.GetLinearDampingAttr().Get(), 0)
                self.assertEqual(api.GetAngularDampingAttr().Get(), 0)


class VisualMaterialTest(unittest.TestCase):
    def test_transmission_is_independent_of_cutout_alpha(self):
        try:
            from pxr import Usd, UsdShade
        except ImportError:
            self.skipTest('requires the Isaac/OpenUSD Python environment')
        surfaces = {
            'smoke': {'transmission': .73, 'metallic': 0., 'alpha': 1.},
            'white': {'transmission': 0., 'metallic': 0., 'alpha': 1.},
            'metal': {'transmission': 0., 'metallic': 1., 'alpha': 1.},
            'cutout': {'transmission': 0., 'metallic': 0., 'alpha': .4},
        }
        nodes = [{'name': name, 'parent': None, 'matrix': [[int(i == j) for j in range(4)] for i in range(4)],
                  'rigid': False, 'part': False, 'role': 'visual',
                  'vertices': [(0, 0, 0), (1, 0, 0), (0, 1, 0)], 'faces': [(0, 1, 2)],
                  'normals': [(0, 0, 1)] * 3,
                  'material': {'color': (.5, .55, .6), 'roughness': .22, 'ior': 1.46, **surface}}
                 for name, surface in surfaces.items()]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'geometry.usda'
            author_geometry({'nodes': nodes, 'joints': []}, path)
            stage = Usd.Stage.Open(str(path))
            for name, surface in surfaces.items():
                material, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath('/Asset/' + name)).ComputeBoundMaterial()
                preview = material.ComputeSurfaceSource()[0]
                self.assertAlmostEqual(preview.GetInput('opacity').Get(), surface['alpha'])
                self.assertAlmostEqual(preview.GetInput('metallic').Get(), surface['metallic'])
                self.assertAlmostEqual(preview.GetInput('ior').Get(), 1.46)
                if surface['transmission']:
                    mdl = material.ComputeSurfaceSource('mdl')[0]
                    self.assertEqual(mdl.GetSourceAssetSubIdentifier('mdl'), 'OmniSurface')
                    self.assertAlmostEqual(mdl.GetInput('specular_transmission_weight').Get(), .73)
                    self.assertAlmostEqual(mdl.GetInput('specular_reflection_ior').Get(), 1.46)
                    for actual, base in zip(mdl.GetInput('specular_transmission_color').Get(), (.5, .55, .6)):
                        self.assertAlmostEqual(actual, base ** .5)
                    self.assertFalse(mdl.GetInput('enable_opacity').Get())
                else:
                    self.assertFalse(material.GetSurfaceOutput('mdl'))


if __name__ == '__main__':
    unittest.main()
