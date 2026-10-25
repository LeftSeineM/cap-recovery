"""Planning URDF/SRDF derivation and planning-world objects from asset models.

Needs ``mplib`` (0.2.1) and ``trimesh`` in the running Python environment.
"""

from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from table_1000.robots.planning_model import write_planning_urdf, write_srdf
from table_1000.planning.world import PlanningScene, asset_objects, model_shapes


def robot(directory):
    """base -(revolute j1)- l1 -(fixed)- frame ; l1 -(prismatic j2)- l2 ; l2 -(revolute j3)- l3."""
    mesh = directory / 'meshes' / 'box.ply'
    mesh.parent.mkdir()
    trimesh.creation.box((.1, .1, .1)).export(mesh)
    collision = '<collision><geometry><mesh filename="meshes/box.ply"/></geometry></collision>'
    links = ''.join(f'<link name="{name}">{collision if name != "frame" else ""}</link>'
                    for name in ('base', 'l1', 'frame', 'l2', 'l3'))
    joints = ('<joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 .2"/>'
              '<axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="1" velocity="1"/></joint>'
              '<joint name="f" type="fixed"><parent link="l1"/><child link="frame"/><origin xyz=".1 0 0"/></joint>'
              '<joint name="j2" type="prismatic"><parent link="frame"/><child link="l2"/><origin xyz="0 0 .2"/>'
              '<axis xyz="0 0 1"/><limit lower="0" upper=".5" effort="1" velocity="1"/></joint>'
              '<joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/><origin xyz=".3 0 0"/>'
              '<axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="1" velocity="1"/></joint>')
    path = directory / 'robot.urdf'
    path.write_text(f'<robot name="r">{links}{joints}</robot>')
    return path


class PlanningModelTest(unittest.TestCase):
    def test_frozen_joints_become_fixed_at_their_value(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            source = robot(temporary / 'source' if (temporary / 'source').mkdir() is None else None)
            urdf = write_planning_urdf(source, temporary / 'plan' / 'robot.urdf', ['j1', 'j3'], {'j2': .25},
                                       ('l3', 'tcp', (.1, 0, 0), (0, 0, 0)))
            root = ET.parse(urdf).getroot()
            joints = {joint.get('name'): joint for joint in root.findall('joint')}
            self.assertEqual(joints['j2'].get('type'), 'fixed')
            np.testing.assert_allclose(np.fromstring(joints['j2'].find('origin').get('xyz'), sep=' '), [0, 0, .45])
            self.assertEqual(joints['j3'].get('type'), 'revolute')
            self.assertIn('tcp', {link.get('name') for link in root.findall('link')})
            self.assertTrue((temporary / 'plan' / 'meshes' / 'box.ply.convex.stl').is_file())
            with self.assertRaises(ValueError):
                write_planning_urdf(source, temporary / 'x' / 'robot.urdf', ['missing'], {}, ('l3', 't', (0, 0, 0), (0, 0, 0)))

    def test_srdf_disables_adjacent_bodies_through_empty_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            (temporary / 'source').mkdir()
            urdf = robot(temporary / 'source')
            pairs = write_srdf(urdf, temporary / 'robot.srdf', [('base', 'l3')])
            self.assertEqual(pairs, [('base', 'l1'), ('base', 'l3'), ('l1', 'l2'), ('l2', 'l3')])


MODEL = {'nodes': [
    {'name': 'box', 'parent': None, 'matrix': np.eye(4).tolist(), 'rigid': True},
    {'name': 'box.lid', 'parent': 'box', 'matrix': np.eye(4).tolist(), 'rigid': True},
    {'name': 'box.wall', 'parent': 'box', 'matrix': np.eye(4).tolist(), 'rigid': False},
    {'name': 'box.wall.collision', 'parent': 'box.wall', 'role': 'collision', 'shape': 'BOX', 'rigid': False,
     'matrix': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [.2, 0, 0, 1]],
     'vertices': (trimesh.creation.box((.1, .1, .1)).vertices).tolist()},
    {'name': 'box.lid.knob', 'parent': 'box.lid', 'rigid': False, 'matrix': np.eye(4).tolist()},
    {'name': 'box.lid.knob.collision', 'parent': 'box.lid.knob', 'role': 'collision', 'shape': 'SPHERE',
     'rigid': False, 'radius': .08, 'matrix': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, .3, 1]]},
]}
for node in MODEL['nodes']:
    node['matrix'] = np.asarray(node['matrix'], dtype=float).T.tolist() if node['name'].endswith('collision') else node['matrix']


class PlanningWorldTest(unittest.TestCase):
    def test_incremental_poses_padding_and_external_permissions(self):
        import mplib
        from unittest.mock import patch
        from table_1000.planning.world import box_object
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'source').mkdir()
            urdf = robot(directory / 'source')
            write_srdf(urdf, directory / 'robot.srdf')
            planner = mplib.Planner(str(urdf), 'l3', srdf=str(directory / 'robot.srdf'))
            scene = PlanningScene(planner)
            transform = np.eye(4)
            transform[:3, 3] = [.4, 0., .4]
            obj, _ = box_object('obstacle', (.04, .04, .04), transform)
            scene.add(obj, transform)
            acm = planner.planning_world.get_allowed_collision_matrix()
            acm.set_entry('obstacle', 'l3', True)
            with patch.object(obj, 'fcl', wraps=obj.fcl) as make:
                scene.set_pose('obstacle', transform.copy())
                self.assertEqual(make.call_count, 0)
                moved = transform.copy()
                moved[0, 3] += .001
                scene.set_pose('obstacle', moved)
                self.assertEqual(make.call_count, 1)
                # Replacement retains permissions even when another consumer owns them.
                self.assertTrue(acm.get_entry('obstacle', 'l3'))
                scene.set_padding(.01)
                self.assertEqual(make.call_count, 2)
                scene.set_padding(.01)
                self.assertEqual(make.call_count, 2)
                frame = np.eye(4)
                frame[0, 3] = .1
                scene.set_frame(frame)
                self.assertEqual(make.call_count, 3)
                np.testing.assert_allclose(planner.planning_world.get_object('obstacle').pose.to_transformation_matrix(),
                                           frame @ moved)
            scene.clear()

    def test_explicit_pose_update_detaches_and_refreshes_live_object(self):
        import mplib
        from table_1000.planning.world import box_object
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'source').mkdir()
            urdf = robot(directory / 'source')
            write_srdf(urdf, directory / 'robot.srdf')
            planner = mplib.Planner(str(urdf), 'l3', srdf=str(directory / 'robot.srdf'))
            scene = PlanningScene(planner)
            transform = np.eye(4)
            transform[:3, 3] = [5., 5., 5.]
            obj, _ = box_object('held', (.04, .04, .04), transform)
            scene.add(obj, transform)
            scene.allow(['held'], ['l3'])
            world = planner.planning_world
            world.attach_object('held', planner.robot.get_name(), planner.link_name_2_idx['l3'], ['l3'])
            self.assertTrue(world.is_object_attached('held'))
            scene.set_pose('held', transform)
            self.assertFalse(world.is_object_attached('held'))
            np.testing.assert_allclose(world.get_object('held').pose.to_transformation_matrix(), transform)
            self.assertTrue(world.get_allowed_collision_matrix().get_entry('held', 'l3'))
            scene.clear()

    def test_shapes_are_placed_relative_to_their_body(self):
        shapes, rest = model_shapes(MODEL)
        self.assertEqual(sorted(rest), ['box', 'box.lid'])
        by_part = {part: (body, transform, kind) for body, part, transform, kind, _ in shapes}
        self.assertEqual(by_part['box.wall'][0], 'box')
        np.testing.assert_allclose(by_part['box.wall'][1][:3, 3], [.2, 0, 0])
        self.assertEqual(by_part['box.lid.knob'][2], 'sphere')

    def test_merged_object_is_one_hull_covering_all_shapes(self):
        (merged,) = asset_objects(MODEL, '', lambda body, part: ('all', 'box'), merged=lambda name: True)
        ((transform, kind, points),) = merged.raw
        self.assertEqual(kind, 'convex')
        np.testing.assert_allclose(points.max(0), [.25, .08, .38])
        np.testing.assert_allclose(points.min(0), [-.08, -.08, -.05])

    def test_padding_grows_every_side(self):
        (wall,) = [o for o in asset_objects(MODEL, '', lambda body, part: (part, body)) if o.name == 'box.wall']
        wall.fcl(np.eye(4))
        wall.fcl(np.eye(4), .01)
        bare, padded = (np.asarray(wall.cache[p][0][1].get_vertices()) for p in (0., .01))
        np.testing.assert_allclose(padded.max(0), bare.max(0) + .01)
        np.testing.assert_allclose(padded.min(0), bare.min(0) - .01)

    def test_grouping_and_permissions_survive_pose_updates(self):
        import mplib
        objects = asset_objects(MODEL, 'a:', lambda body, part: (part, body))
        self.assertEqual(sorted(o.name for o in objects), ['a:box.lid.knob', 'a:box.wall'])
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / 'source').mkdir()
            urdf = robot(Path(temporary) / 'source')
            write_srdf(urdf, Path(temporary) / 'source' / 'robot.srdf')
            planner = mplib.Planner(str(urdf), 'l3', srdf=str(Path(temporary) / 'source' / 'robot.srdf'))
            scene = PlanningScene(planner)
            far = np.eye(4)
            far[:3, 3] = [5, 5, 5]
            for obj in objects:
                scene.add(obj, far)
            near = np.eye(4)
            near[:3, 3] = [.4, 0, .1]
            scene.update({'a:box.lid': near})
            # The knob sphere moved into l3 (at 0.4, 0, 0.4); the wall stayed away.
            self.assertTrue(planner.planning_world.check_collision())
            scene.allow(['a:box.lid.knob'], ['l2', 'l3'])
            scene.update({'a:box.lid': near})
            self.assertFalse(planner.planning_world.check_collision())
            # Simultaneous base/body motion must place each object from the new body pose,
            # while also moving static objects and retaining permissions.
            frame = np.eye(4)
            frame[0, 3] = -1.
            moved = near.copy()
            moved[0, 3] += 1.
            scene.set_frame(frame, {'a:box.lid': moved})
            np.testing.assert_allclose(scene.poses['a:box.lid.knob'], moved)
            self.assertFalse(planner.planning_world.check_collision())
            scene.allow(['a:box.lid.knob'], ['l2', 'l3'], False)
            self.assertTrue(planner.planning_world.check_collision())
            scene.set_frame(np.eye(4), {'a:box.lid': far})
            self.assertFalse(planner.planning_world.check_collision())


if __name__ == '__main__':
    unittest.main()
