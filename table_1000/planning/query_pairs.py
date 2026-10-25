"""Temporarily omit provably irrelevant scene pairs during a native RRT query."""

from contextlib import contextmanager
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from table_1000.planning.broadphase import BroadphaseWorld


class QueryPairs:
    """Preserve collision results while reducing native pair traversal.

    Static entities are omitted only after an exact non-collision check. Moving
    entities use a conservative swept AABB around their first moving joint;
    triangle inequalities bound all downstream translations and shape extents.
    A first prismatic joint extends the anchor over its complete joint range.
    Robot self-collision pairs are never changed.
    """

    def __init__(self, planner, urdf):
        self.planner = planner
        root = ET.parse(urdf).getroot()
        self.parents = {}
        for joint in root.findall('joint'):
            origin = joint.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
            rpy = np.fromstring(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
            axis = joint.find('axis')
            axis = np.fromstring(axis.get('xyz') if axis is not None else '1 0 0', sep=' ')
            axis = axis / np.linalg.norm(axis)
            self.parents[joint.find('child').get('link')] = dict(
                name=joint.get('name'), kind=joint.get('type'), parent=joint.find('parent').get('link'),
                xyz=xyz, rotation=Rotation.from_euler('xyz', rpy).as_matrix(), axis=axis)
        self.bounds = BroadphaseWorld(planner.planning_world)
        self.group = [planner.user_joint_names[i] for i in planner.move_group_joint_indices]
        self.indices = {name: i for i, name in enumerate(planner.user_joint_names)}

    def _chain(self, link):
        joints = []
        while link in self.parents:
            joint = self.parents[link]
            joints.append(joint)
            link = joint['parent']
        return joints[::-1]

    def _envelope(self, link, radius, free, q):
        chain = self._chain(link)
        first = next((i for i, joint in enumerate(chain) if joint['name'] in free), None)
        if first is None:
            return None
        own = chain[first]
        pinocchio = self.planner.pinocchio_model
        frame = pinocchio.get_link_pose(self.planner.link_name_2_idx[own['parent']]).to_transformation_matrix()
        anchor = frame[:3, :3] @ own['xyz'] + frame[:3, 3]
        low = high = anchor
        if own['kind'] == 'prismatic':
            limits = self.planner.joint_limits[self.indices[own['name']]]
            direction = frame[:3, :3] @ own['rotation'] @ own['axis']
            endpoints = anchor + limits[:, None] * direction
            low, high = endpoints.min(0), endpoints.max(0)
        for joint in chain[first + 1:]:
            radius += np.linalg.norm(joint['xyz'])
            if joint['kind'] == 'prismatic':
                i = self.indices[joint['name']]
                travel = np.max(np.abs(self.planner.joint_limits[i])) if joint['name'] in free else abs(q[i])
                radius += travel * np.linalg.norm(joint['axis'])
        return low - radius - 1e-6, high + radius + 1e-6

    def _radius(self, obj):
        centre, half = self.bounds._local_bounds(obj)
        return float(np.linalg.norm(np.abs(centre) + half))

    def _aabb(self, obj):
        centre, half = self.bounds._local_bounds(obj)
        frame = obj.pose.to_transformation_matrix()
        centre = frame[:3, :3] @ centre + frame[:3, 3]
        half = np.abs(frame[:3, :3]) @ half + 1e-6
        return centre - half, centre + half

    @contextmanager
    def reduced(self, start, fixed_joints=()):
        from mplib.collision_detection import AllowedCollision, fcl
        world = self.planner.planning_world
        world.set_qpos_all(np.asarray(start))
        q = self.planner.robot.get_qpos()
        self.planner.pinocchio_model.compute_forward_kinematics(q)
        locked = {joint.joint_idx for joint in fixed_joints if joint.articulation_idx == 0}
        free = {name for i, name in enumerate(self.group) if i not in locked}
        matrix = world.get_allowed_collision_matrix()
        objects = [world.get_object(name) for name in world.get_object_names()]
        scene = [obj for obj in objects if not world.is_object_attached(obj.name)]
        entities = [(obj, obj.name, 0.) for obj in self.planner.robot.get_fcl_model().get_collision_objects()]
        for obj in objects:
            if world.is_object_attached(obj.name):
                body = world.get_attached_object(obj.name)
                body.update_pose()
                link = next(name for name, index in self.planner.link_name_2_idx.items()
                            if index == body.get_attached_link_id())
                offset = np.linalg.norm(body.get_pose().to_transformation_matrix()[:3, 3])
                entities.append((obj, link, offset))
        changes = []
        try:
            boxes = {obj.name: self._aabb(obj) for obj in scene}
            for entity, link, offset in entities:
                envelope = self._envelope(link, offset + self._radius(entity), free, q)
                for obj in scene:
                    if matrix.get_allowed_collision(entity.name, obj.name) not in (None, AllowedCollision.NEVER):
                        continue
                    if envelope is None:
                        omit = not fcl.collide(entity, obj).is_collision()
                    else:
                        low, high = boxes[obj.name]
                        omit = np.any(high < envelope[0]) or np.any(low > envelope[1])
                    if omit:
                        changes.append((entity.name, obj.name, matrix.get_entry(entity.name, obj.name)))
                        matrix.set_entry(entity.name, obj.name, True)
            yield changes
        finally:
            for first, second, previous in reversed(changes):
                if previous is None:
                    matrix.remove_entry(first, second)
                else:
                    matrix.set_entry(first, second, previous == AllowedCollision.ALWAYS)
