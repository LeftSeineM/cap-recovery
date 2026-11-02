"""Regression checks for geometric rejection, joint poses and sparse keyframes.

Run inside Blender; --assets-root needs cabinet/000000, pen/000000 and pen/000002.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import bpy
import numpy as np
from mathutils import Matrix, Vector
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from table_1000.modeling.geometry_checks import GeometryChecks, convex_data, sat_depth, sphere_convex_depth
from table_1000.modeling.preview import read_views, interpolate, run_worker
args = argparse.ArgumentParser()
args.add_argument('--assets-root',type=Path,required=True)
args.add_argument('--output',type=Path,required=True)
options = args.parse_args(sys.argv[sys.argv.index('--')+1:])
options.output.mkdir(parents=True,exist_ok=True)


class AcceptanceTests(unittest.TestCase):
    def load(self):
        bpy.ops.wm.open_mainfile(filepath=str(options.assets_root/'cabinet/000000/object.blend'))
        bpy.context.scene.rigidbody_world.enabled=False
        bpy.context.view_layer.update()

    def test_sparse_merge_and_interpolation(self):
        data={'move.mp4':{'fps':24,'frames':[
            {'frame':0,'camera':[1,-1,1,0,0,1],'joints':{'a':.1}},
            {'frame':10,'joints':{'b':.2}}, {'frame':20,'joints':{'a':0}}]}}
        with tempfile.TemporaryDirectory(dir=options.output) as tmp:
            p=Path(tmp)/'views.json';p.write_text(json.dumps(data))
            _,keys=read_views(p)['move.mp4']
            self.assertEqual(keys[1]['joints'],{'a':.1,'b':.2})
            self.assertEqual(interpolate(keys,5)['joints'],{'a':.1,'b':.1})
            self.assertEqual(interpolate(keys,20)['joints'],{'a':0,'b':.2})
            data['move.mp4']['frames'][1]['frame']=0;p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):read_views(p)

    def test_collision_and_filter(self):
        self.load();checks=GeometryChecks()
        self.assertFalse(checks.sample('closed',0,{}))
        first=bpy.data.objects['cabinet.drawer1.slide'].rigid_body_constraint.object2
        first.location.z-=.06
        bpy.context.view_layer.update()
        self.assertTrue(checks.sample('intentional_overlap',1,{}))
        checks.ignored={tuple(sorted((a.name,b.name))) for a in checks.bodies for b in checks.bodies if a!=b}
        self.assertFalse(checks.sample('explicitly_filtered',1,{}))

    def test_transform_sparse_slerp(self):
        data = {'move.mp4': {'fps': 24, 'frames': [
            {'frame': 0, 'camera': [1,-1,1,0,0,1]},
            {'frame': 10, 'transforms': {'cap': [1,0,0,0,0,0,2]}},
            {'frame': 20, 'transforms': {'body': [0,2,0,1,0,0,0]}},
            {'frame': 30, 'transforms': {'cap': [1,0,0,0,0,0,-1]}}]}}
        with tempfile.TemporaryDirectory(dir=options.output) as tmp:
            p = Path(tmp)/'views.json'; p.write_text(json.dumps(data))
            _, keys = read_views(p)['move.mp4']
            self.assertEqual(keys[2]['transforms']['cap'], [1,0,0,0,0,0,1])
            halfway = interpolate(keys,5)['transforms']['cap']
            self.assertAlmostEqual(halfway[0], .5)
            np.testing.assert_allclose(halfway[3:], [2**-.5,0,0,2**-.5])
            # Opposite quaternion signs encode the same rotation.
            np.testing.assert_allclose(interpolate(keys,25)['transforms']['cap'][3:], [0,0,0,1])

    def test_transform_subtree_saved_pivot_and_reset(self):
        path = options.assets_root/'pen/000002/object.blend'
        camera = [1,-1,1,0,0,1]
        cap_pose = [-.055,-.045,0,2**-.5,0,0,-2**-.5]
        views = {'moved.jpg': [{'frame':0, 'camera':camera, 'joints':{},
                               'transforms':{'body':[.2,0,0,1,0,0,0], 'cap':cap_pose}}],
                 'reset.jpg': [{'frame':0, 'camera':camera, 'joints':{}}]}
        initial = {}
        original_sample = GeometryChecks.sample
        def capture(checks, output, frame, values, transforms=None):
            body, cap = bpy.data.objects['body'], bpy.data.objects['cap']
            self.assertIsNone(body.parent); self.assertIsNone(cap.parent)
            for obj in [body, cap] + checks.visuals + checks.colliders:
                owner = obj.name if obj in (body, cap) else checks.owner[obj.name]
                delta = Matrix.Identity(4)
                if output == 'moved.jpg':
                    if owner == 'body':
                        delta = Matrix.Translation((.2,0,0))
                    else:
                        pivot = initial['cap'].translation
                        delta = (Matrix.Translation(pivot + Vector(cap_pose[:3]))
                                 @ Matrix.Rotation(-np.pi/2,4,'Z') @ Matrix.Translation(-pivot))
                np.testing.assert_allclose(obj.matrix_world, delta @ initial[obj.name], atol=1e-6)
            return original_sample(checks, output, frame, values, transforms)
        with tempfile.TemporaryDirectory(dir=options.output) as tmp, patch.object(GeometryChecks,'sample',capture):
            # A non-unit body frame must not rotate world-space axes.
            bpy.ops.wm.open_mainfile(filepath=str(path))
            bpy.context.scene.rigidbody_world.enabled = False
            frame = Matrix.Translation((.1,.2,.3)) @ Matrix.Rotation(np.pi/2,4,'Z')
            for name in ('body', 'cap', 'cap_ball1', 'cap_ball2', 'cap_ball3', 'cap.detent1', 'cap.detent2', 'cap.detent3'):
                obj = bpy.data.objects[name]
                obj.matrix_world = frame @ obj.matrix_world
            bpy.context.view_layer.update()
            initial.update({obj.name: obj.matrix_world.copy() for obj in bpy.context.scene.objects})
            saved_path = Path(tmp)/'world-frame.blend'
            bpy.ops.wm.save_as_mainfile(filepath=str(saved_path))
            run_worker(saved_path, views, Path(tmp)/'preview', True)
        # Rotating an inserted cap sideways must be seen by the collision check.
        overlapping = {'overlap.jpg': [{'frame':0, 'camera':camera, 'joints':{},
                                       'transforms':{'cap':[0,0,0,2**-.5,0,0,2**-.5]}}]}
        with tempfile.TemporaryDirectory(dir=options.output) as tmp:
            with self.assertRaisesRegex(ValueError, 'Collision coarse check failed'):
                run_worker(path, overlapping, Path(tmp), True)
            report = json.loads((Path(tmp)/'acceptance.json').read_text())
            self.assertTrue(report['failures'])

    def test_nested_semantic_parts_and_penetration(self):
        path = options.assets_root/'pen/000002/object.blend'
        camera = [1,-1,1,0,0,1]
        separated = {'part-pose.jpg': [{'frame':0, 'camera':camera, 'joints':{},
                                      'transforms':{'cap':[-.06,-.045,0,2**-.5,0,0,-2**-.5]}}]}
        with tempfile.TemporaryDirectory(dir=options.output) as tmp:
            run_worker(path, separated, Path(tmp), True)
            checks = GeometryChecks()
            self.assertEqual(len(checks.bodies), 5)
            self.assertEqual(len(checks.colliders), 79)
            self.assertEqual(checks.report['complexity']['visual_triangles'], 1132)
            for obj in checks.visuals + checks.colliders:
                self.assertTrue(obj.parent.get('semantic_part'))
                self.assertEqual(checks.owner[obj.name], obj.parent.parent.name)
            overlap = {'overlap.jpg': [{'frame':0, 'camera':camera, 'joints':{},
                                       'transforms':{'cap':[0,0,0,2**-.5,0,0,2**-.5]}}]}
            with self.assertRaisesRegex(ValueError, 'Collision coarse check failed'):
                run_worker(path, overlap, Path(tmp), True)
            self.assertTrue(json.loads((Path(tmp)/'acceptance.json').read_text())['failures'])

    def test_native_sphere_contacts_and_joint_child_preview(self):
        import itertools
        points=np.array(list(itertools.product([-1,1],repeat=3)),dtype=float)
        faces=[(0,1,3,2),(4,6,7,5),(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3)]
        self.assertAlmostEqual(sphere_convex_depth([1.4,0,0],.5,points,faces),.1,places=6)
        self.assertAlmostEqual(sphere_convex_depth([1.5,0,0],.5,points,faces),0,places=6)
        self.assertLess(sphere_convex_depth([1.4,1.4,0],.5,points,faces),0)
        self.assertAlmostEqual(sphere_convex_depth([0,0,0],.5,points,faces),1.5)
        path=options.assets_root/'pen/000002/object.blend'
        camera=[1,-1,1,0,0,1]
        with tempfile.TemporaryDirectory(dir=options.output) as tmp:
            run_worker(path,{'separated.jpg':[{'frame':0,'camera':camera,'joints':{},
                'transforms':{'cap':[-.06,0,0,1,0,0,0]}}]},Path(tmp),True)
            cap=bpy.data.objects['cap']
            for i in range(1,4):
                ball=bpy.data.objects[f'cap_ball{i}']
                self.assertAlmostEqual(ball.matrix_world.translation.x-cap.matrix_world.translation.x,-.0015,places=6)
            ball.location.x=.07;ball.location.y=0;ball.location.z=0
            bpy.context.view_layer.update()
            checks=GeometryChecks()
            self.assertTrue(checks.sample('intentional_ball_overlap',0,{}))
            # Rotate the display sphere so its tessellated x extent shrinks.
            # The analytic sphere still overlaps the flat barrel end by 0.21 mm.
            from mathutils import Quaternion
            ball.location=(.086+.0008-.00021,0,0)
            ball.rotation_mode='QUATERNION'
            ball.rotation_quaternion=Quaternion((0,1,0),.19)
            bpy.context.view_layer.update()
            self.assertTrue(checks.sample('rotated_sphere_overlap',0,{}))

    def test_containment_and_touch(self):
        normals=np.eye(3);edges=np.eye(3)
        import itertools
        a=np.array(list(itertools.product([-1,1],repeat=3)),dtype=float)
        b=a*.1
        self.assertGreater(sat_depth((a,normals,edges),(b,normals,edges)),1)
        self.assertAlmostEqual(sat_depth((a,normals,edges),(a+[2,0,0],normals,edges)),0)

    def test_open_and_concave_rejected(self):
        self.load();checks=GeometryChecks();o=checks.colliders[0]
        import bmesh
        bm=bmesh.new();bm.from_mesh(o.data);bm.faces.ensure_lookup_table()
        bmesh.ops.delete(bm,geom=[bm.faces[0]],context='FACES_ONLY');bm.to_mesh(o.data);bm.free()
        with self.assertRaises(ValueError):convex_data(o)
        self.load();checks=GeometryChecks()
        o=next(o for o in checks.colliders if o.rigid_body.collision_shape=='BOX')
        o.rigid_body.collision_shape='CONVEX_HULL'
        o.data.vertices[0].co=sum((v.co for v in o.data.vertices),Vector())/len(o.data.vertices)
        o.data.update()
        with self.assertRaises(ValueError):convex_data(o)

    def test_pose_and_invalid_joint(self):
        path=options.assets_root/'cabinet/000000/object.blend'
        with tempfile.TemporaryDirectory(dir=options.output) as tmp:
            entry=lambda q:{'pose.jpg':[{'frame':0,'camera':[1,-1,1,0,0,1],'joints':q}]}
            run_worker(path,entry({'cabinet.drawer1.slide':.08}),Path(tmp),True)
            body=bpy.data.objects['cabinet.drawer1.slide'].rigid_body_constraint.object2
            self.assertAlmostEqual(body.matrix_world.translation.y,-.213,places=5)
            self.assertAlmostEqual(body.matrix_world.translation.z,.237,places=5)
            other=bpy.data.objects['cabinet.drawer2.slide'].rigid_body_constraint.object2
            self.assertAlmostEqual(other.matrix_world.translation.y,-.133,places=5)
            self.assertAlmostEqual(other.matrix_world.translation.z,.146,places=5)
            for q in ({'unknown':0},{'cabinet.drawer1.slide':1}):
                with self.assertRaises(ValueError):run_worker(path,entry(q),Path(tmp),True)

    def test_complexity_totals(self):
        self.load()
        self.assertEqual(GeometryChecks().report['complexity'], {
            'visual_triangles': 588, 'colliders': 33,
            'primitive_colliders': 21, 'convex_colliders': 12,
            'convex_vertices': 160, 'convex_faces': 104,
        })

    def test_pen_button_and_refill_share_slider(self):
        path=options.assets_root/'pen/000000/object.blend'
        camera=[.82,-.82,.72,0,0,1]
        with tempfile.TemporaryDirectory(dir=options.output) as tmp:
            for values in ({'button_press':-.003}, {'button_press':0}):
                views={'pose.jpg':[{'frame':0,'camera':camera,'joints':values}]}
                run_worker(path,views,Path(tmp),True)
                checks=GeometryChecks()
                self.assertEqual(len(checks.bodies),2)
                self.assertEqual(len(checks.report['joints']),1)
                self.assertEqual(checks.ignored, {('pen.refill', 'pen.shell')})
                c=bpy.data.objects['button_press'].rigid_body_constraint
                self.assertEqual(c.type,'SLIDER')
                self.assertAlmostEqual(c.object2.matrix_world.translation.x,values['button_press'],places=6)
                self.assertAlmostEqual(c.object1.matrix_world.translation.x,0,places=6)
                for part in ('button', 'shaft', 'tip'):
                    node = bpy.data.objects['pen.refill.' + part]
                    self.assertTrue(node.get('semantic_part'))
                    self.assertEqual(node.parent, c.object2)
                    for obj in node.children:
                        self.assertEqual(checks.owner[obj.name], c.object2.name)


result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AcceptanceTests))
if not result.wasSuccessful():raise RuntimeError('Acceptance regression failed')
