"""Robot acceptance in the simulator: physical-sense cases and comparison metrics.

``scripts/robots/check_robot.py <name>`` runs, in a few Isaac processes, a check
scene (a table in front of the robot and a few primitive shapes, see
:func:`check_scene`) through:

- **cases** with pass/fail criteria, each starting from a reset (:attr:`Checker.LIBRARY`):
  ``keyframes``, ``sensors``, ``arm_hits_block``, ``arm_pushes_objects``,
  ``drop_on_robot``, ``lift_into_shelf``, ``self_collision``,
  ``grasp_and_shake``, ``grasp_and_push``, ``grasp_pressing_table``, ``base_into_wall``, ``base_full_turn``, ``drive_holding``, ``joint_limits``.
  Cases needing a part the robot lacks (cameras, a lift, a mobile base) do not
  apply to it;
- **metrics** recorded for comparison only: model/simulator consistency, IK,
  self-collision planning, joint tracking, base tracking and step cost.

The cases are generic; each robot's ``asset_sources/robots/<name>/robot_test.py``
sets them up: ``TABLE``, ``ROBOT`` (placement and starting keyframe), ``WORK``
(a point on the table top in front of the primary arm; targets are placed
relative to it and the table height), ``DROP_LINKS`` and ``CASES``. The primary
arm is the first arm of the robot description.

Common criteria of every case: robot contacts penetrate at most
``PENETRATION_LIMIT``; no robot joint moves faster than its URDF velocity
limit (with ``SPEED_MARGIN``) and no object moves horizontally faster than
``OBJECT_SPEED_LIMIT`` (nothing is flung; falling is not counted); blocked joints settle (``JITTER_LIMIT``). Unused
primitives are parked off to the side.
"""

from pathlib import Path
from table_1000.scenes import robot_config

import time
import types
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from table_1000.robots.cameras import direction, optical_pose, project
from table_1000.scenes import primitive_surface
from table_1000.sim import CONTROL_HZ, SUBSTEPS

PENETRATION_LIMIT = .002     # m, slow or resting contacts
# Objects dropped onto the robot hit it at about 2 m/s, 9 mm per 4 ms step; a
# few millimetres of transient overlap is the price of that step and accepted.
IMPACT_PENETRATION_LIMIT = .01
# Joint speed / URDF velocity limit. A finger knocked by an obstacle is back-driven
# a little past its limit; flung bodies go far beyond.
SPEED_MARGIN = 1.5
OBJECT_SPEED_LIMIT = 2.      # m/s, horizontal
# A body dropped onto the robot bounces off at a solver-dependent speed (PhysX
# resolves the step's overlap by pushing it out, up to about 3 m/s here, above
# the 2.2 m/s of the impact); a flung body is far faster.
DROP_SPEED_LIMIT = 4.
JITTER_LIMIT = 1e-3          # rad or m, oscillation std of a blocked joint over 0.5 s
HOLD_DRIFT_LIMIT = .005      # m, held object relative to the TCP
# A held object is pushed at its centre of mass in each world axis direction and
# PUSH_RANDOM random ones: ramped up over PUSH_RAMP s, held PUSH_HOLD s, released.
PUSH_FORCE, PUSH_RAMP, PUSH_HOLD, PUSH_RANDOM = 5., .2, .8, 8
# Grasp trials (``Checker.trials``): trial 0 grasps objects placed at the work point; trial k > 0 shifts them
# by up to GRASP_OFFSET and turns them by up to GRASP_YAW (deg), and the grasp follows the object's yaw with an
# error of up to GRASP_YAW_ERROR (deg), drawn with seed k.
GRASP_OFFSET, GRASP_YAW, GRASP_YAW_ERROR = .01, 10., 3.
PUSH_REPORT_ONLY = ('disc',)  # reported, not part of the pass condition
GRASP_PENETRATION_LIMIT = .001   # m, fingers in a held object
# Closing on an object with the fingers pressing on the table with each of PRESS_FORCES (N; 0: just touching),
# for approach tilts PRESS_TILTS (deg from horizontal); the hand is lowered in PRESS_STEP steps.
PRESS_FORCES, PRESS_TILTS, PRESS_STEP = (0., 5., 10.), (90., 80.), .0005
FINGER_LIMIT_MARGIN = .05   # of a finger joint's range, past its limits while closing on the table
CALM_SPEED, CALM_SHIFT = .5, .01   # m/s, m: the object is neither flung nor pushed away
TURN_RATE_MARGIN = 1.05      # measured yaw rate over the commanded one, through whole turns
TURN_ERROR_LIMIT = .02       # rad, turned angle against the command; m, translation while turning
KEYFRAME_DRIFT_LIMIT = .01   # rad, joint drift while holding a keyframe
# Robot cameras: markers in front of each camera, placed by its spec and the
# measured joints, must show up where the lens model projects them.
MARKER_DISTANCE = .4         # m, at most, and closer than 60 % of what is behind
MARKER_PIXELS = 8            # marker radius in the image
MARKER_GRID = (.2, .5, .8)   # image fractions of the 3 x 3 marker directions
MARKERS_PLACED = 5           # of the 9; the rest are blocked by the robot itself
PIXEL_LIMIT = 2.             # px, marker centre (rgb)
DEPTH_LIMIT = .003           # m, marker surface along the optical axis (depth)
CONTRAST_LIMIT = 5.          # grey-level standard deviation of a usable image
DT = 1 / CONTROL_HZ
# Fixed views of the check scene for videos (scripts/robots/render_check.py).
VIEWS = {'wide': {'position': (1.9, -1.7, 1.7), 'target': (.25, 0., .8), 'focal_length_mm': 20,
                  'resolution': (960, 540)}}

# Primitives; a missing mass makes the shape kinematic (it stays where it is put).
PRIMITIVES = {
    # Tall enough to stop the whole hand, however long its fingers: the joint-space sweep lifts the hand
    # mid-way, and the long fingers of X1Pro's rm001 gripper pass over a 0.14 m block with only the tips.
    'block': {'shape': 'box', 'size': (.06, .06, .30), 'color': (.8, .3, .2)},
    'box': {'shape': 'box', 'size': (.05, .05, .08), 'mass': .2},
    'can': {'shape': 'cylinder', 'size': (.025, .10), 'mass': .2, 'color': (.3, .7, .3)},
    'stick': {'shape': 'cylinder', 'size': (.007, .14), 'mass': .02, 'color': (.9, .8, .2)},
    'ball': {'shape': 'sphere', 'size': .03, 'mass': .1, 'color': (.9, .5, .1)},
    'cube': {'shape': 'box', 'size': (.06, .06, .06), 'mass': .3, 'color': (.6, .3, .8)},
    # Small objects of tabletop tasks, for grasp_and_push: a 1 cm steel cube, a 4 cm cube, a 2 x 2 x 10 cm
    # bar, a rod of 2 cm diameter and 10 cm, a disc of 3 cm diameter and 1 cm (plastic, 1000 kg/m^3).
    'cube1': {'shape': 'box', 'size': (.01, .01, .01), 'mass': .00785, 'color': (.75, .75, .78)},
    'cube4': {'shape': 'box', 'size': (.04, .04, .04), 'mass': .064, 'color': (.25, .55, .85)},
    'bar': {'shape': 'box', 'size': (.02, .02, .10), 'mass': .04, 'color': (.9, .7, .15)},
    'rod': {'shape': 'cylinder', 'size': (.01, .10), 'mass': .0314, 'color': (.3, .75, .35)},
    'disc': {'shape': 'cylinder', 'size': (.015, .01), 'mass': .00707, 'color': (.6, .35, .8)},
    'shelf': {'shape': 'box', 'size': (.12, .12, .02), 'color': (.5, .5, .5)},
    'wall': {'shape': 'box', 'size': (.1, 1.4, 1.), 'color': (.5, .5, .5)},
}


def translation(x, y, z, rotation=None):
    transform = np.eye(4)
    transform[:3, 3] = [x, y, z]
    if rotation is not None:
        transform[:3, :3] = rotation
    return transform


def resting_height(spec):
    """Height of a primitive's centre when it rests on a flat surface (cylinders upright)."""
    size = spec['size']
    return size if spec['shape'] == 'sphere' else (size[1] if spec['shape'] == 'cylinder' else size[2]) / 2


def parked(index):
    """Out of the way, resting on the floor."""
    spec = list(PRIMITIVES.values())[index]
    return translation(-3., -2. + .6 * index, resting_height(spec))


LIBRARY = ('keyframes', 'sensors', 'arm_hits_block', 'arm_pushes_objects', 'drop_on_robot', 'lift_into_shelf',
           'self_collision', 'grasp_and_shake', 'grasp_and_push', 'grasp_pressing_table', 'base_into_wall', 'base_full_turn', 'drive_holding', 'joint_limits')
# Parts a case needs; it does not apply to a robot without them.
NEEDS = {'sensors': 'cameras', 'lift_into_shelf': 'lift', 'base_into_wall': 'base', 'base_full_turn': 'base',
         'drive_holding': 'base'}


def save_marked(path, outputs, pixels):
    """The rgb image (else the depth image, white near, up to 2 m) with a cross at each pixel."""
    from PIL import Image, ImageDraw
    from table_1000.sim.isaaclab.cameras import depth_image
    image = Image.fromarray(outputs['rgb'][0] if 'rgb' in outputs else depth_image(outputs['depth'][0]))
    draw = ImageDraw.Draw(image)
    for x, y in pixels:
        draw.line((x - 12, y, x + 12, y), fill=(0, 255, 255))
        draw.line((x, y - 12, x, y + 12), fill=(0, 255, 255))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=90)


def marker_errors(seen, plain, expected):
    """Pixel distance from each ``expected`` marker centre to the centroid of the nearest
    region that came closer from depth image ``plain`` to ``seen``."""
    from scipy import ndimage
    closer = (seen > 0) & ((plain == 0) | (seen < plain - 1e-4))
    labels, count = ndimage.label(closer)
    if not count:
        return [float('inf')] * len(expected)
    centres = np.array(ndimage.center_of_mass(closer, labels, range(1, count + 1)))[:, ::-1] + .5
    return [float(np.linalg.norm(centres - pixel, axis=1).min()) for pixel in expected]


def select_cases(spec, settings):
    """The robot's ``CASES`` (default: every case of the library that applies to it)."""
    applies = lambda name: bool(getattr(spec, NEEDS.get(name, 'arms')))
    names = getattr(settings, 'CASES', None) or [name for name in LIBRARY if applies(name)]
    unknown = [name for name in names if name not in LIBRARY]
    unsupported = [name for name in names if name in LIBRARY and not applies(name)]
    if unknown or unsupported:
        raise ValueError(f'cases not in the library {unknown} or needing a missing part {unsupported}')
    return list(names)


def check_scene(spec, settings):
    """Scene source for :class:`~table_1000.sim.isaaclab.env.RobotEnv` from a robot's ``robot_test`` settings."""
    robot = settings.ROBOT
    scene = types.SimpleNamespace(
        TABLE=settings.TABLE, OBJECTS={}, PRIMITIVES=PRIMITIVES,
        ROBOT=spec.name, ROBOTS={spec.name: {'asset': f'robots/{spec.name}', 'position': robot['position'], 'yaw': robot['yaw'],
               'joints': spec.keyframe(robot.get('keyframe', 'zero'))}})

    def placement(**poses):
        layout = {name: parked(i) for i, name in enumerate(PRIMITIVES)}
        layout.update(poses)
        return layout

    scene.placement = placement
    return scene


def down(tilt_deg=90.):
    """TCP orientation with its approach axis (+X) tilted down by ``tilt_deg``."""
    return Rotation.from_euler('y', tilt_deg, degrees=True).as_matrix()


class VisualModel:
    """Visual meshes of the robot source, posed by the planning model's FK (for gap measurements)."""

    def __init__(self, spec, model, root):
        import trimesh
        urdf = Path(root) / spec.urdf
        self.model, self.parts = model, []
        for link in ET.parse(urdf).getroot().findall('link'):
            for visual in link.findall('visual'):
                mesh = visual.find('geometry/mesh')
                if mesh is None:
                    continue
                origin = visual.find('origin')
                transform = np.eye(4)
                if origin is not None:
                    transform[:3, :3] = Rotation.from_euler(
                        'xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
                    transform[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                self.parts.append((link.get('name'), transform,
                                   trimesh.load(urdf.parent / mesh.get('filename'), force='mesh')))

    def posed(self):
        import trimesh
        meshes = []
        for link, transform, mesh in self.parts:
            posed = mesh.copy()
            posed.apply_transform(self.model.fk(link, world=True) @ transform)
            meshes.append(posed)
        return trimesh.util.concatenate(meshes)

    def surface_below(self, point):
        """Highest visual surface point below ``point`` (world), at the model's current state; None if none."""
        hits, _, _ = self.posed().ray.intersects_location([point], [[0., 0., -1.]])
        return float(hits[:, 2].max()) if len(hits) else None

    def distance(self, points):
        """Smallest distance from ``points`` (world) to the visual surface."""
        _, distances, _ = self.posed().nearest.on_surface(np.asarray(points))
        return float(distances.min())


class Checker:
    def __init__(self, spec, settings, output, assets, root, device='cpu', trials=(0,)):
        """``output``: the process's working files (planning model); set ``self.output`` to a case's
        directory for its images. ``trials``: the grasp trials of ``grasp_and_push`` (see ``GRASP_OFFSET``).
        Set ``states`` to a list to record the scene state of every step into it."""
        from table_1000.robots.kinematics import RobotModel
        from table_1000.sim.isaaclab.env import RobotEnv
        self.spec, self.settings, self.output, self.root = spec, settings, Path(output), Path(root)
        self.arm = spec.primary_arm
        self.gripper = f'{self.arm}_gripper'
        self.trials = tuple(trials)
        self.top = settings.TABLE['height']
        self.scene = check_scene(spec, settings)
        started = time.perf_counter()
        self.env = RobotEnv(self.scene, spec, assets, cameras=spec.cameras, depth=True, root=root,
                            contact_watch=('robot', *PRIMITIVES), device=device)
        self.model = RobotModel(spec, self.output / 'planning')
        self.visual = VisualModel(spec, self.model, root)
        self.load_s = time.perf_counter() - started
        self.body_joints = [j for j in spec.body_joints]
        joints = ET.parse(Path(root) / spec.urdf).getroot().findall('joint')
        limits = {joint.get('name'): float(joint.find('limit').get('velocity'))
                  for joint in joints if joint.find('limit') is not None}
        self.child_link = {joint.get('name'): joint.find('child').get('link') for joint in joints}
        self.speed_limits = np.array([limits[j] for j in self.body_joints])
        self.case, self.past = '', []
        self.trace = {'joint_speed': 0., 'joint': None, 'object_speed': 0., 'object_speeds': {}}
        # A list to record into: the scene state before the first step, then after every step.
        self.states = None
        self.reaches = {}    # finger_reach by its arguments

    # -- driving ---------------------------------------------------------------
    def run(self, case):
        """Run ``case``. Its common checks cover every reset within it (a case
        may run several sub-cases, one per object or pose)."""
        self.case = case
        self.trace = {'joint_speed': 0., 'joint': None, 'object_speed': 0., 'object_speeds': {}}
        self.past = []
        self.env.contacts.clear()    # the previous case's contacts are not this case's
        return getattr(self, case)()

    def reset(self, robot=None, **poses):
        q = self.env.initial_robot_q()
        q.update(robot or {})
        self.past += self.env.contacts.intervals()
        obs = self.env.reset(self.scene.placement(**poses), q)
        self.model.set_state(dict(zip(self.env.joints, obs['agent']['qpos'][0])), self.base_transform())
        return obs

    def intervals(self):
        """Contact intervals of the case so far, over all its resets."""
        return self.past + self.env.contacts.intervals()

    def base_transform(self):
        x, y, yaw = self.env.obs['agent']['base_pose'][0]
        transform = translation(x, y, 0. if self.spec.base else robot_config(self.scene, self.spec.name)['position'][2])
        transform[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
        return transform

    def step(self, action=None, count=1):
        from table_1000.sim.isaaclab.replay import numpy_state
        for _ in range(count):
            if self.states == []:
                self.states.append(numpy_state(self.env))
            obs = self.env.step(action or {})
            if self.states is not None:
                self.states.append(numpy_state(self.env))
            ratio = np.abs(obs['agent']['qvel'][0, [self.env.joints.index(j) for j in self.body_joints]]) / self.speed_limits
            if ratio.max() > self.trace['joint_speed']:
                self.trace['joint_speed'], self.trace['joint'] = float(ratio.max()), self.body_joints[int(ratio.argmax())]
            velocities = self.env.privileged()['velocities']
            if velocities:
                # Horizontal: falling off the robot or the table is no fling.
                speeds = self.trace['object_speeds']
                for name, v in velocities.items():
                    speeds[name] = max(speeds.get(name, 0.), float(np.linalg.norm(v[0, :2])))
                self.trace['object_speed'] = max(self.trace['object_speed'], max(speeds.values()))
        # The planning model follows the measured joints, as on the real robot.
        self.model.set_state(dict(zip(self.env.joints, obs['agent']['qpos'][0])), self.base_transform())
        return obs

    def q(self, joints):
        return self.env.qpos(joints)[0]

    def arm_q(self, arm):
        return self.q(self.spec.arms[arm].joints)

    def ik(self, arm, target):
        status, q = self.model.ik(arm, target, world=True, attempts=40)
        if status != 'Success':
            raise RuntimeError(f'check pose unreachable for {arm}: {target[:3, 3]}')
        return q[self.model.arm_indices(arm)]

    def move_joints(self, arm, target, seconds):
        """Joint-space straight line to ``target`` (no planning: cases collide on purpose)."""
        start = self.arm_q(arm)
        count = max(1, round(seconds / DT))
        for k in range(1, count + 1):
            self.step({arm: (start + (target - start) * k / count)[None]})

    def move_line(self, arm, target):
        """Straight TCP line to the world ``target`` (self-collision checks only), at the planner's speed."""
        result = self.model.plan_screw(arm, target, world=True, time_step=DT)
        if result['status'] != 'Success':
            raise RuntimeError(f'line to {target[:3, 3]} not planned: {result.get("reason")}')
        for row in result['position']:
            self.step({arm: row[None]})

    def jitter(self, joints, seconds=.5):
        """Largest standard deviation of ``joints`` while holding for ``seconds``."""
        rows = []
        for _ in range(round(seconds / DT)):
            self.step()
            rows.append(self.q(joints))
        rows = np.array(rows)
        # Oscillation about a straight-line fit; a slow creep (a hand sliding
        # along an obstacle under saturated torque) is reported separately.
        t = np.arange(len(rows))
        residual = rows - np.polyval(np.polyfit(t, rows, 1), t[:, None]) if len(rows) > 2 else rows * 0.
        self.last_jitter = {'oscillation_std': dict(zip(joints, map(float, residual.std(axis=0)))),
                            'creep': dict(zip(joints, map(float, rows[-1] - rows[0])))}
        return float(residual.std(axis=0).max())

    def penetration(self, robot_only=True, sub_case=False):
        """Deepest contact of the case, or of the current sub-case since the last
        reset (robot pairs only by default)."""
        depth = 0.
        for entry in self.env.contacts.intervals() if sub_case else self.intervals():
            if not robot_only or any(side.startswith('robot') for side in entry['pair']):
                depth = max(depth, entry['max_penetration_m'])
        return depth

    def common(self, result, penetration_limit=PENETRATION_LIMIT, speed_limit=OBJECT_SPEED_LIMIT, speed_exempt=()):
        """Add the common checks to ``result``; objects of ``speed_exempt`` are left out of the speed check."""
        result['max_penetration_m'] = self.penetration()
        speed = (max((v for k, v in self.trace['object_speeds'].items() if k not in speed_exempt), default=0.)
                 if speed_exempt else self.trace['object_speed'])
        result.update(max_joint_speed_ratio=self.trace['joint_speed'], fastest_joint=self.trace['joint'],
                      max_object_speed=speed)
        checks = result.setdefault('checks', {})
        checks['penetration'] = result['max_penetration_m'] <= penetration_limit
        checks['joint_speed'] = result['max_joint_speed_ratio'] <= SPEED_MARGIN
        checks['object_speed'] = result['max_object_speed'] <= speed_limit
        checks = {key: bool(value) for key, value in checks.items()}
        result['checks'] = checks
        result['status'] = 'PASS' if all(checks.values()) else 'FAIL'
        deepest = {}
        for entry in self.intervals():
            pair = ' / '.join(entry['pair'])
            deepest[pair] = max(deepest.get(pair, 0.), entry['max_penetration_m'])
        result['contacts'] = dict(sorted(deepest.items(), key=lambda item: -item[1]))
        return result

    def finger_reach(self, arm, gap, steps=9):
        """Farthest point of ``arm``'s finger visual meshes past its TCP along the approach axis (+X), over
        finger gaps from ``gap`` down to closed (pivoting fingertips advance as they close)."""
        key = (arm, gap, steps)
        if key in self.reaches:
            return self.reaches[key]
        spec = self.spec.arms[arm]
        reach = -np.inf
        for value in np.linspace(gap, 0., steps):
            q = np.array(self.model.q, dtype=float)
            for joint, target in spec.finger_targets(value).items():
                q[self.model.index[joint]] = target
            inverse = np.linalg.inv(self.model.tcp(arm, q))
            for link, transform, mesh in self.visual.parts:
                if link in spec.finger_links:
                    points = (inverse @ self.model.fk(link, q) @ transform)[:3] @ np.c_[mesh.vertices, np.ones(len(mesh.vertices))].T
                    reach = max(reach, float(points[0].max()))
        self.reaches[key] = reach
        return reach

    def tcp(self, arm=None):
        return self.env.obs['agent']['tcp'][arm or self.arm][0]

    def pose_of(self, shape):
        """World pose of a check shape, as simulated."""
        return self.env.privileged()['poses'][shape][0]

    def held(self, body, arm=None):
        """Pose of ``body`` in the frame of the arm's TCP link, both as simulated."""
        link = self.env.link_pose(self.spec.arms[arm or self.arm].tcp[0])[0]
        return np.linalg.inv(link) @ self.pose_of(body)

    @staticmethod
    def work_point_in(settings, dx, dy, dz, rotation=None):
        """World pose ``(dx, dy)`` from ``settings.WORK`` and ``dz`` above its table top."""
        x, y = settings.WORK
        return translation(x + dx, y + dy, settings.TABLE['height'] + dz, rotation)

    def work_point(self, dx, dy, dz, rotation=None):
        return self.work_point_in(self.settings, dx, dy, dz, rotation)

    def on_work(self, dx, dy, height):
        """A primitive of ``height`` resting on the table at ``(dx, dy)`` from the work point."""
        return self.work_point(dx, dy, height / 2)

    # -- cases -------------------------------------------------------------------
    def arm_hits_block(self):
        """The hand sweeps sideways through a fixed block; it must stop, then reach the goal once freed."""
        arm = self.arm
        self.reset(block=self.on_work(0., .03, PRIMITIVES['block']['size'][2]))
        start, goal = self.ik(arm, self.work_point(0., -.21, .045, down(70))), self.ik(arm, self.work_point(0., .20, .045, down(70)))
        self.move_joints(self.arm, start, 2.)
        self.move_joints(self.arm, goal, 3.)
        blocked_error = float(np.abs(self.arm_q(self.arm) - goal).max())
        jitter = self.jitter(self.spec.arms[self.arm].joints)
        self.env.set_shape_pose('block', parked(0))
        self.step(count=round(1.5 / DT))
        result = {'blocked_error_rad': blocked_error, 'blocked_jitter_rad': jitter, 'blocked_motion': self.last_jitter,
                  'freed_error_rad': float(np.abs(self.arm_q(self.arm) - goal).max())}
        # The hand touched the block in the contact reports: the penetration checks see contacts.
        result['block_contact'] = any('block' in side for entry in self.intervals() for side in entry['pair'])
        result['checks'] = {'stopped_by_block': blocked_error > .05, 'settled': jitter <= JITTER_LIMIT,
                            'reaches_goal_when_freed': result['freed_error_rad'] <= .01,
                            'contact_reported': result['block_contact']}
        return self.common(result)

    def arm_pushes_objects(self):
        """The hand sweeps through a free box and can; they slide or tip, nothing is flung."""
        arm = self.arm
        self.reset(box=self.on_work(0., 0., .05), can=self.on_work(.02, .11, .10))
        start = self.ik(arm, self.work_point(0., -.21, .045, down(70)))
        before = {name: self.pose_of(name) for name in ('box', 'can')}
        self.move_joints(self.arm, start, 2.)
        self.move_line(arm, self.work_point(0., .25, .045, down(70)))
        self.step(count=round(1. / DT))
        after = {name: self.pose_of(name) for name in ('box', 'can')}
        moved = {name: float(np.linalg.norm(after[name][:2, 3] - before[name][:2, 3])) for name in ('box', 'can')}
        heights = {name: float(after[name][2, 3]) for name in ('box', 'can')}
        result = {'moved_m': moved, 'final_height_m': heights}
        result['checks'] = {'pushed': all(d > .01 for d in moved.values()),
                            'still_on_table': all(self.top < z < self.top + .1 for z in heights.values())}
        return self.common(result)

    def drop_on_robot(self):
        """A ball and a cube dropped onto each of ``DROP_LINKS`` rest on the visible surface or roll off."""
        targets = self.settings.DROP_LINKS
        results = {}
        for name, link in targets.items():
            for body in ('ball', 'cube'):
                self.reset()
                self.step(count=2)
                point = self.model.fk(link, world=True)[:3, 3]
                surface = self.visual.surface_below([point[0], point[1], 3.])
                if surface is None:
                    continue
                half = PRIMITIVES[body]['size'] if body == 'ball' else PRIMITIVES[body]['size'][2] / 2
                self.env.set_shape_pose(body, translation(point[0], point[1], surface + .25 + half))
                speeds, self.trace['object_speeds'] = self.trace['object_speeds'], {}
                self.step(count=round(2.5 / DT))
                speed_of_drop = max(self.trace['object_speeds'].values(), default=0.)
                for shape, speed in self.trace['object_speeds'].items():
                    speeds[shape] = max(speeds.get(shape, 0.), speed)
                self.trace['object_speeds'] = speeds
                pose = self.pose_of(body)
                speed = float(np.linalg.norm(self.env.privileged()['velocities'][body][0, :3]))
                resting_on_robot = pose[2, 3] > .1 and speed < .02
                gap = self.visual.distance(primitive_surface(PRIMITIVES[body], pose)) if resting_on_robot else None
                results[f'{name}/{body}'] = {'link': link, 'resting_on_robot': bool(resting_on_robot),
                                             'gap_to_visual_m': gap, 'penetration_m': self.penetration(sub_case=True),
                                             'max_object_speed': speed_of_drop}
        gaps = [r['gap_to_visual_m'] for r in results.values() if r['gap_to_visual_m'] is not None]
        result = {'drops': results, 'max_gap_m': max(gaps, default=0.)}
        # Resting more than 15 mm above the visible surface: the collision shell is too fat.
        result['checks'] = {'rests_on_visible_surface': result['max_gap_m'] <= .015}
        return self.common(result, IMPACT_PENETRATION_LIMIT, DROP_SPEED_LIMIT)

    def lift_into_shelf(self):
        """The torso rises into a fixed shelf above its top (the head, if any): it stops there, then lowers again."""
        self.reset()
        self.step(count=2)
        top_link = self.child_link[self.spec.head[-1] if self.spec.head else self.spec.lift[-1]]
        head = self.model.fk(top_link, world=True)[:3, 3]
        top = self.visual.surface_below([head[0], head[1], 3.])
        self.env.set_shape_pose('shelf', translation(head[0], head[1], top + .10))
        lift = self.spec.lift[0]
        start = self.q([lift])[0]
        self.step({'lift': [[start + .3]]}, count=round(2.5 / DT))
        blocked = self.q([lift])[0]
        jitter = self.jitter([lift])
        self.step({'lift': [[start]]}, count=round(2. / DT))
        result = {'lift_start_m': float(start), 'lift_blocked_m': float(blocked), 'blocked_jitter_m': jitter,
                  'lift_back_error_m': float(abs(self.q([lift])[0] - start))}
        result['checks'] = {'stopped_by_shelf': blocked < start + .25, 'settled': jitter <= JITTER_LIMIT,
                            'lowers_again': result['lift_back_error_m'] <= .005}
        return self.common(result)

    def self_collision(self):
        """The primary arm swings into the robot's own body: it stops at contact and comes back."""
        self.reset()
        self.step(count=2)
        arm = self.spec.arms[self.arm]
        ready = self.arm_q(self.arm)
        target, contact = self.colliding_target(self.arm, ready)
        collides = target is not None
        self.move_joints(self.arm, target, 3.)
        error = float(np.abs(self.arm_q(self.arm) - target).max())
        jitter = self.jitter(arm.joints)
        self.move_joints(self.arm, ready, 3.)
        self.step(count=round(1. / DT))
        result = {'model_contact': contact, 'target': target.tolist(), 'blocked_error_rad': error, 'blocked_jitter_rad': jitter,
                  'back_error_rad': float(np.abs(self.arm_q(self.arm) - ready).max()),
                  'self_contacts': sorted({' / '.join(e['pair']) for e in self.env.contacts.intervals()
                                           if all(s.startswith('robot') for s in e['pair'])})}
        result['checks'] = {'stopped_at_contact': error > .05, 'settled': jitter <= JITTER_LIMIT,
                            'comes_back': result['back_error_rad'] <= .01}
        return self.common(result)

    def colliding_target(self, arm, ready, beyond=.3):
        """Arm joints just past the first self-collision along one shoulder or elbow joint from ``ready``,
        still in collision there (a hand that only grazes a body on the way is passed by)."""
        indices = self.model.arm_indices(arm)
        for joint in (0, 1, 2, 3):
            for sign in (1., -1.):
                low, high = self.model.limits[indices[joint]]
                for angle in np.arange(.05, 3., .05):
                    q = ready.copy()
                    q[joint] = ready[joint] + sign * angle
                    if not low <= q[joint] <= high:
                        break
                    pairs = self.model.self_collisions(self.model.vector(dict(zip(self.spec.arms[arm].joints, q))))
                    if pairs:
                        q[joint] = np.clip(q[joint] + sign * beyond, low, high)
                        if self.model.self_collisions(self.model.vector(dict(zip(self.spec.arms[arm].joints, q)))):
                            return q, pairs
        raise RuntimeError('no self-collision reachable along single joints')

    def _grasp(self, body, pose, approach=.10, clearance=.004, yaw=0.):
        """Top-down grasp of ``body`` resting at ``pose``, the closing direction turned by ``yaw`` (deg) about
        the vertical from world Y; returns the closed finger gap."""
        self.reset(**{body: pose})
        self.step(count=round(.5 / DT))
        center = self.pose_of(body)[:3, 3]
        # TCP at mid-height (pivoting fingers close on a wide object above the TCP), fingertips
        # ``clearance`` above the table throughout the closing.
        z = max(center[2], self.top + self.finger_reach(self.arm, .08) + clearance)
        hand = Rotation.from_euler('z', yaw, degrees=True).as_matrix() @ down(90)
        grasp = translation(center[0], center[1], z, hand)
        self.step({self.gripper: [[.08]]}, count=round(.6 / DT))
        self.move_joints(self.arm, self.ik(self.arm, translation(*grasp[:3, 3] + [0., 0., approach], hand)), 2.5)
        self.move_joints(self.arm, self.ik(self.arm, grasp), 1.5)
        self.step({self.gripper: [[0.]]}, count=round(.8 / DT))
        arm = self.spec.arms[self.arm]
        return arm.gap(self.q(arm.fingers))

    def grasp_and_shake(self):
        """Grasp a box, a can and a thin stick, lift and shake each: no slip, no finger penetration."""
        objects = {'box': self.on_work(0., 0., .08), 'can': self.on_work(0., 0., .10),
                   # lying across the closing direction (world Y): its axis along world X
                   'stick': self.work_point(0., 0., .007, Rotation.from_euler('y', 90, degrees=True).as_matrix())}
        results = {}
        for body, pose in objects.items():
            gap = self._grasp(body, pose)
            lifted = self.ik(self.arm, translation(*self.tcp()[:3, 3] + [0., 0., .08], down(90)))
            self.move_joints(self.arm, lifted, 1.5)
            self.step(count=round(.5 / DT))
            relative = self.held(body)
            joint6 = self.spec.arms[self.arm].joints[-1]
            for sign in (1, -1, 1, -1):
                target = lifted.copy()
                target[-1] += sign * .4
                self.move_joints(self.arm, target, .5)
            self.move_joints(self.arm, lifted, .5)
            self.step(count=round(.5 / DT))
            drift = float(np.linalg.norm(self.held(body)[:3, 3] - relative[:3, 3]))
            held_z = self.pose_of(body)[2, 3]
            self.step({self.gripper: [[.08]]}, count=round(1.5 / DT))
            final = self.pose_of(body)[2, 3]
            results[body] = {'finger_gap_m': gap, 'drift_m': drift, 'falls_when_released': bool(final < held_z - .04),
                             'penetration_m': self.penetration(sub_case=True), 'joint6': joint6}
        result = {'objects': results}
        result['checks'] = {'held_without_slip': all(r['drift_m'] <= HOLD_DRIFT_LIMIT for r in results.values()),
                            'released': all(r['falls_when_released'] for r in results.values())}
        return self.common(result)

    def grasp_offset(self, trial):
        """``(dx, dy, object yaw, grasp yaw)`` of grasp trial ``trial`` (m, deg; see ``GRASP_OFFSET``)."""
        if trial == 0:
            return 0., 0., 0., 0.
        random = np.random.default_rng(trial)
        dx, dy = random.uniform(-GRASP_OFFSET, GRASP_OFFSET, 2)
        yaw = random.uniform(-GRASP_YAW, GRASP_YAW)
        return float(dx), float(dy), float(yaw), float(yaw + random.uniform(-GRASP_YAW_ERROR, GRASP_YAW_ERROR))

    def grasp_and_push(self):
        """For each grasp trial, grasp each small tabletop object, lift it and push it with PUSH_FORCE at its
        centre of mass in every direction in turn: it stays in the hand."""
        random = np.random.default_rng(0).normal(size=(PUSH_RANDOM, 3))
        directions = [sign * axis for axis in np.eye(3) for sign in (1., -1.)] + list(
            random / np.linalg.norm(random, axis=1, keepdims=True))
        ramp, hold = round(PUSH_RAMP / DT), round(PUSH_HOLD / DT)
        trials = {}
        for trial in self.trials:
            dx, dy, yaw, grasp_yaw = self.grasp_offset(trial)
            turn = Rotation.from_euler('z', yaw, degrees=True).as_matrix()
            lying = Rotation.from_euler('y', 90, degrees=True).as_matrix()   # long axis across the closing
            objects = {'cube1': self.work_point(dx, dy, .005, turn), 'cube4': self.work_point(dx, dy, .02, turn),
                       'bar': self.work_point(dx, dy, .01, turn @ lying),
                       # A lying PhysX cylinder turned about the vertical rolls away by itself: not turned.
                       'rod': self.work_point(dx, dy, .01, lying), 'disc': self.work_point(dx, dy, .005, turn)}
            results = {}
            for body, pose in objects.items():
                body_yaw = grasp_yaw - yaw if body == 'rod' else grasp_yaw
                # Fingertips 1 mm above the table: the pads reach down the sides of a 1 cm object.
                gap = self._grasp(body, pose, clearance=.001, yaw=body_yaw)
                lifted = self.ik(self.arm, translation(*self.tcp()[:3, 3] + [0., 0., .08],
                                                       Rotation.from_euler('z', body_yaw, degrees=True).as_matrix() @ down(90)))
                self.move_joints(self.arm, lifted, 1.5)
                self.step(count=round(.5 / DT))
                relative = self.held(body)
                drifts = []
                for direction in directions:
                    drift = 0.
                    for k in range(1, ramp + hold + 1):
                        self.env.push_shape(body, PUSH_FORCE * min(1., k / ramp) * direction)
                        self.step()
                        drift = max(drift, float(np.linalg.norm(self.held(body)[:3, 3] - relative[:3, 3])))
                    self.env.push_shape(body, np.zeros(3))
                    self.step(count=round(.3 / DT))
                    drifts.append(drift)
                    if drift > .03:    # dropped; pushing on would fling it
                        break
                results[body] = {'finger_gap_m': gap, 'drift_m': max(drifts), 'pushes': len(drifts),
                                 'held': max(drifts) <= HOLD_DRIFT_LIMIT and len(drifts) == len(directions),
                                 'drift_by_push_m': drifts, 'penetration_m': self.penetration(sub_case=True)}
            trials[trial] = {'offset': {'dx_m': dx, 'dy_m': dy, 'object_yaw_deg': yaw, 'grasp_yaw_deg': grasp_yaw},
                             'objects': results}
        result = {'force_n': PUSH_FORCE, 'directions': [d.tolist() for d in directions], 'trials': trials,
                  'report_only': list(PUSH_REPORT_ONLY)}
        result['checks'] = {'held_without_slip': all(r['held'] for t in trials.values()
                                                     for body, r in t['objects'].items() if body not in PUSH_REPORT_ONLY)}
        # A dropped report-only object is still pushed off the hand: its speed is not a fling of the robot's.
        dropped = {body for t in trials.values() for body, r in t['objects'].items()
                   if body in PUSH_REPORT_ONLY and not r['held']}
        return self.common(result, GRASP_PENETRATION_LIMIT, speed_exempt=dropped)

    def finger_table_force(self, steps, action=None):
        """Mean normal force (N) between the primary arm's fingers and the table over the next ``steps`` steps
        of ``action`` (default: the current targets)."""
        fingers = {f'robot.{link}' for link in self.spec.arms[self.arm].finger_links}
        self.env.contacts.take_impulses()
        self.step(action, count=steps)
        impulse = sum(value for pair, value in self.env.contacts.take_impulses().items()
                      if 'table' in pair and fingers & set(pair))
        return impulse / (steps * DT)

    def grasp_pressing_table(self):
        """Close on a 4 cm cube with the fingers on the table, just touching or pressing with each of
        PRESS_FORCES, for each approach tilt of PRESS_TILTS: the fingers stay within their limits and the cube
        is neither flung nor pushed away."""
        arm = self.spec.arms[self.arm]
        limits = {joint: self.model.limits[self.model.index[joint]] for joint in arm.fingers}
        open_gap = min(.07, arm.max_gap)
        runs = {}
        for tilt in PRESS_TILTS:
            for target in PRESS_FORCES:
                self.reset(cube4=self.on_work(0., 0., .04))
                self.step(count=round(.5 / DT))
                center = self.pose_of('cube4')[:3, 3]
                touch = max(center[2], self.top + self.finger_reach(self.arm, open_gap))
                self.step({self.gripper: [[open_gap]]}, count=round(.6 / DT))
                self.move_joints(self.arm, self.ik(self.arm, translation(*center[:2], touch + .1, down(tilt))), 2.5)
                # From 3 mm above the visual contact, down until the fingers touch or press with the target force.
                depth, force = -.003, 0.
                while depth < .02:
                    self.move_joints(self.arm, self.ik(self.arm, translation(*center[:2], touch - depth, down(tilt))), .3)
                    force = self.finger_table_force(5)
                    if force > 0. and force >= target:
                        break
                    depth += PRESS_STEP
                start = self.pose_of('cube4')[:3, 3].copy()
                speed, peak, q_min, q_max = 0., 0., np.full(len(arm.fingers), np.inf), np.full(len(arm.fingers), -np.inf)
                for _ in range(round(1. / DT)):
                    peak = max(peak, self.finger_table_force(1, {self.gripper: [[0.]]}))
                    speed = max(speed, float(np.linalg.norm(self.env.privileged()['velocities']['cube4'][0, :3])))
                    q = self.q(arm.fingers)
                    q_min, q_max = np.minimum(q_min, q), np.maximum(q_max, q)
                margin = {j: FINGER_LIMIT_MARGIN * (high - low) for j, (low, high) in limits.items()}
                within = all(limits[j][0] - margin[j] <= lo and hi <= limits[j][1] + margin[j]
                             for j, lo, hi in zip(arm.fingers, q_min, q_max))
                runs[f'tilt{tilt:g}_{target:g}n'] = {
                    'depth_below_visual_contact_m': depth, 'force_before_closing_n': force, 'peak_force_closing_n': peak,
                    'max_cube_speed': speed, 'cube_shift_m': float(np.linalg.norm(self.pose_of('cube4')[:3, 3] - start)),
                    'finger_range': [[float(lo), float(hi)] for lo, hi in zip(q_min, q_max)], 'fingers_within_limits': within,
                    'finger_gap_m': float(arm.gap(self.q(arm.fingers)))}
        result = {'runs': runs}
        # Touching is a force above zero; the depth limit reached in the air is not.
        result['checks'] = {'pressed': all(r['force_before_closing_n'] > 0. and r['force_before_closing_n'] >= f
                                           for r, f in zip(runs.values(), [f for _ in PRESS_TILTS for f in PRESS_FORCES])),
                            'fingers_within_limits': all(r['fingers_within_limits'] for r in runs.values()),
                            'cube_calm': all(r['max_cube_speed'] <= CALM_SPEED and r['cube_shift_m'] <= CALM_SHIFT
                                             for r in runs.values())}
        return self.common(result)

    def base_into_wall(self):
        """The base backs into a wall: it stops there without shoving or crossing it."""
        self.reset(wall=translation(-.65, 0., .5))
        self.step({'base': [[-.3, 0.]]}, count=round(3. / DT))
        x_blocked = float(self.env.obs['agent']['base_pose'][0, 0])
        jitter = self.jitter(list(self.spec.base.virtual_joints[:2]))
        self.step({'base': [[.3, 0.]]}, count=round(1.5 / DT))
        self.step({'base': [[0., 0.]]}, count=round(1. / DT))
        result = {'base_x_blocked_m': x_blocked, 'blocked_jitter_m': jitter,
                  'base_x_after_m': float(self.env.obs['agent']['base_pose'][0, 0])}
        result['checks'] = {'stopped_by_wall': x_blocked > -.5, 'settled': jitter <= JITTER_LIMIT,
                            'drives_away': result['base_x_after_m'] > x_blocked + .2}
        return self.common(result)

    def base_full_turn(self):
        """Turn in place at the full yaw rate for 1.25 turns, past every wrap of the yaw angle:
        the rate stays the commanded one and the base turns by the commanded angle."""
        self.reset()
        self.step({'base': [[-.4, 0.]]}, count=round(2. / DT))    # away from the table
        self.step({'base': [[0., 0.]]}, count=round(1. / DT))
        start = self.env.obs['agent']['base_pose'][0].copy()
        rate = self.spec.base.max_speed[1]
        ramp = rate / self.spec.base.max_acceleration[1]
        seconds = 2.5 * np.pi / rate
        yaw = self.env.joints.index(self.spec.base.virtual_joints[2])
        turned, fastest, previous = 0., 0., start[2]
        for _ in range(round(seconds / DT)):
            obs = self.step({'base': [[0., rate]]})
            fastest = max(fastest, abs(float(obs['agent']['qvel'][0, yaw])))
            current = obs['agent']['base_pose'][0, 2]
            turned += float(np.angle(np.exp(1j * (current - previous))))
            previous = current
        # Ramping up to the rate takes `ramp` s and loses half of it.
        expected = rate * (round(seconds / DT) * DT - ramp / 2)
        end = self.env.obs['agent']['base_pose'][0]
        result = {'turned_rad': turned, 'commanded_rad': expected, 'max_yaw_rate_rad_s': fastest,
                  'translation_m': float(np.linalg.norm(end[:2] - start[:2]))}
        result['checks'] = {'rate_kept': fastest <= rate * TURN_RATE_MARGIN,
                            'turned_as_commanded': abs(turned - expected) <= TURN_ERROR_LIMIT,
                            'turned_in_place': result['translation_m'] <= TURN_ERROR_LIMIT}
        return self.common(result)

    def drive_holding(self):
        """Hold a box, drive at full speed and stop: the box stays in the hand."""
        self._grasp('box', self.on_work(0., 0., .08))
        lifted = self.ik(self.arm, translation(*self.tcp()[:3, 3] + [0., 0., .08], down(90)))
        self.move_joints(self.arm, lifted, 1.5)
        self.step(count=round(.5 / DT))
        relative = self.held('box')

        def drift():
            return float(np.linalg.norm(self.held('box')[:3, 3] - relative[:3, 3]))

        phases = {}
        for name, command, seconds in (('drive', [-.5, 0.], 1.5), ('stop', [0., 0.], 1.), ('turn', [0., 1.], 1.),
                                       ('stop after turn', [0., 0.], 1.)):
            self.step({'base': [command]}, count=round(seconds / DT))
            phases[name] = drift()
            fingers = self.q(self.spec.arms[self.arm].fingers)
            phases[name + ' finger_centre'] = float(fingers.sum() / 2)
        arm = self.spec.arms[self.arm]
        sim = self.env.link_pose(arm.tcp[0])[0] @ translation(*arm.tcp[1])
        result = {'drift_m': max(v for k, v in phases.items() if 'finger' not in k), 'drift_by_phase_m': phases,
                  'model_vs_sim_tcp_m': _pose_error(sim, self.model.tcp(self.arm, world=True))[0],
                  'base_pose': self.env.obs['agent']['base_pose'][0].tolist()}
        result['checks'] = {'held_without_slip': result['drift_m'] <= HOLD_DRIFT_LIMIT}
        return self.common(result)

    def joint_limits(self):
        """Commands beyond the limits of the lift and head (the primary arm if the robot has neither) end at the
        limits, steady: the first joint of a group is driven past its upper limit, the next past its lower."""
        self.reset()
        self.step(count=2)
        limits = {name: self.model.limits[self.model.index[name]] for name in self.spec.body_joints}
        groups = {name: list(joints) for name, joints in (('lift', self.spec.lift), ('head', self.spec.head)) if joints}
        if not groups:
            groups = {self.arm: list(self.spec.arms[self.arm].joints[:2])}
        beyond, joints = {}, []
        for name, names in groups.items():
            row = list(self.q(self.spec.arms[name].joints)) if name in self.spec.arms else []
            targets = [limits[j][1] + .2 if i % 2 == 0 else limits[j][0] - .2 for i, j in enumerate(names)]
            beyond[name] = [targets + row[len(names):]]
            joints += names
        self.step(beyond, count=round(4. / DT))
        q = self.q(joints)
        jitter = self.jitter(joints)
        overshoot = max(max(limits[j][0] - v, v - limits[j][1]) for j, v in zip(joints, q))
        result = {'final': dict(zip(joints, map(float, q))), 'max_overshoot': float(overshoot), 'jitter': jitter}
        result['checks'] = {'within_limits': overshoot <= .01, 'settled': jitter <= JITTER_LIMIT,
                            'reaches_limits': all(min(abs(v - limits[j][0]), abs(v - limits[j][1])) < .02
                                                  for j, v in zip(joints, q))}
        return self.common(result)

    def keyframes(self):
        """Every keyframe (the zero pose included) is free of self-collision: no colliding pair in the
        planning model, and held for a second in simulation without contact between robot bodies or drift."""
        result, checks = {}, {}
        for name in self.spec.keyframe_names:
            q = self.spec.keyframe(name)
            pairs = [list(pair) for pair in self.model.self_collisions(self.model.vector(q))]
            self.reset(robot=q)
            start = self.q(self.body_joints)
            self.step(count=round(1. / DT))
            drift = float(np.abs(self.q(self.body_joints) - start).max())
            contacts = sorted({' / '.join(entry['pair']) for entry in self.env.contacts.intervals()
                               if all(side.startswith('robot') for side in entry['pair'])})
            result[name] = {'planning_pairs': pairs, 'robot_contacts': contacts, 'drift_rad': drift}
            checks[f'{name}_planning'] = not pairs
            checks[f'{name}_simulation'] = not contacts and drift <= KEYFRAME_DRIFT_LIMIT
        result['checks'] = checks
        return self.common(result)

    def sensor_poses(self):
        """The starting keyframe and the same with every head and arm joint turned by
        0.2 rad (away from its nearer limit), which moves every camera."""
        start = self.spec.keyframe(self.settings.ROBOT.get('keyframe', 'zero'))
        moved = dict(start)
        joints = [*self.spec.head, *(joint for arm in self.spec.arms.values() for joint in arm.joints)]
        for joint in joints:
            low, high = self.model.limits[self.model.index[joint]]
            moved[joint] = start[joint] + (.2 if high - start[joint] > start[joint] - low else -.2)
        return {'start': start, 'moved': moved}

    def markers(self, name, depth):
        """Marker centres in camera ``name``'s optical frame and their radius, aimed at a
        3 x 3 grid of pixels and kept in front of what ``depth`` (rendered without
        markers) shows there; directions blocked within a few centimetres are skipped."""
        sensor = self.spec.cameras[name].sensor
        width, height = sensor.resolution
        radius = MARKER_DISTANCE * MARKER_PIXELS / sensor.intrinsics[0]
        points = []
        for u in MARKER_GRID:
            for v in MARKER_GRID:
                pixel = np.array([u * width, v * height])
                window = 3 * MARKER_PIXELS
                patch = depth[max(int(pixel[1]) - window, 0):int(pixel[1]) + window,
                              max(int(pixel[0]) - window, 0):int(pixel[0]) + window]
                behind = patch[patch > 0].min() if (patch > 0).any() else np.inf
                distance = min(MARKER_DISTANCE, .6 * behind)
                if distance > 4 * radius:
                    points.append(direction(sensor, pixel) * distance)
        return np.array(points).reshape(-1, 3), radius

    def sensors(self):
        """Every robot camera, at the starting keyframe and with head and arms moved: the
        image has contrast and shows little of the robot itself; markers placed in front
        of the camera by its spec and the measured joints appear where its lens model
        projects them (rgb), at the depth it predicts (depth).

        Rendering does not advance physics, so images of one state differ only by what
        was shown: pixels that come closer when the markers appear are marker pixels,
        pixels that recede when the robot is hidden show the robot.
        """
        result, checks = {}, {}
        for pose, q in self.sensor_poses().items():
            self.reset(robot=q)
            self.step(count=round(.5 / DT))
            for name, camera in self.spec.cameras.items():
                sensor = camera.sensor
                self.env.set_markers([], 0.)
                plain = self.env.render([name])[name]
                points, radius = self.markers(name, plain['depth'][0])
                frame = optical_pose(self.model, camera.mount, self.root / self.spec.urdf)
                self.env.set_markers((frame[:3, :3] @ points.T).T + frame[:3, 3], radius)
                seen = self.env.render([name])[name]
                expected = project(sensor, points)
                save_marked(self.output / f'{name}_{pose}.jpg', seen, expected)
                seen = seen['depth'][0]
                entry = {'markers': len(points)}
                if sensor.kind == 'depth':
                    pixels = np.clip(np.round(expected - .5).astype(int), 0, [sensor.resolution[0] - 1, sensor.resolution[1] - 1])
                    # The ray through the centre meets the sphere radius short of it.
                    surface = points[:, 2] * (1 - radius / np.linalg.norm(points, axis=1))
                    errors = np.abs(seen[pixels[:, 1], pixels[:, 0]] - surface)
                    entry.update(depth_error_m=[round(float(e), 6) for e in errors],
                                 valid_pixels=float((plain['depth'][0] > 0).mean()))
                    passed = bool(len(errors) >= MARKERS_PLACED and errors.max() <= DEPTH_LIMIT and entry['valid_pixels'] > .5)
                else:
                    self.env.set_markers([], 0.)
                    self.env.show_robot(False)
                    bare = self.env.render([name])[name]['depth'][0]
                    self.env.show_robot(True)
                    depth = plain['depth'][0]
                    robot = (depth > 0) & ((bare == 0) | (bare > depth + 1e-3))
                    errors = marker_errors(seen, depth, expected)
                    entry.update(self_view=float(robot.mean()), contrast=float(plain['rgb'][0].mean(axis=2).std()),
                                 pixel_error=[round(e, 3) for e in errors])
                    passed = bool(len(errors) >= MARKERS_PLACED and max(errors) <= PIXEL_LIMIT
                                  and entry['self_view'] <= camera.self_view and entry['contrast'] >= CONTRAST_LIMIT)
                result.setdefault(name, {})[pose] = entry
                checks[f'{name}_{pose}'] = passed
        self.env.set_markers([], 0.)
        result['checks'] = checks
        return self.common(result)

    LIBRARY = LIBRARY

    def cases(self):
        return select_cases(self.spec, self.settings)

    # -- comparison metrics ---------------------------------------------------------
    def metrics(self, samples=100, seed=0):
        rng = np.random.default_rng(seed)
        model = self.model
        self.reset()
        self.step(count=2)
        ready = model.q.copy()
        out = {}
        for arm_name, arm in self.spec.arms.items():
            indices = model.arm_indices(arm_name)
            low, high = model.limits[indices].T
            free = []
            while len(free) < samples:
                q = ready.copy()
                q[indices] = rng.uniform(low, high)
                if not model.self_collisions(q):
                    free.append(q)
            # Model vs simulator TCP at the same joint values.
            # Measured joint values, since a random pose may meet the table and be pushed.
            errors = []
            for q in free[:30]:
                self.reset(robot=dict(zip(arm.joints, q[indices])))
                sim = self.env.link_pose(arm.tcp[0])[0] @ translation(*arm.tcp[1])
                position, rotation = _pose_error(sim, self.model.tcp(arm_name, world=True))
                errors.append((position, rotation))
            errors = np.array(errors)
            # IK on reachable poses.
            ik_times, solved, ik_errors = [], 0, []
            model.set_state(dict(zip(model.joints, ready)))
            for q in free:
                target = model.tcp(arm_name, q)
                started = time.perf_counter()
                status, solution = model.ik(arm_name, target)
                ik_times.append(time.perf_counter() - started)
                if status == 'Success':
                    solved += 1
                    ik_errors.append(_pose_error(model.tcp(arm_name, solution), target))
            # Self-collision planning between random free configurations.
            plan_times, planned = [], 0
            for start, goal in zip(free[:20], free[20:40]):
                model.set_state(dict(zip(model.joints, start)))
                started = time.perf_counter()
                result = model.plan_rrt(arm_name, None, goals=[goal], planning_time=2.)
                plan_times.append(time.perf_counter() - started)
                planned += result['status'] == 'Success'
            model.set_state(dict(zip(model.joints, ready)))
            out[arm_name] = {
                'fk_position_error_max_m': float(errors[:, 0].max()), 'fk_rotation_error_max_rad': float(errors[:, 1].max()),
                'ik_success_rate': solved / len(free), 'ik_time_median_s': float(np.median(ik_times)),
                'ik_position_error_max_m': float(max((e[0] for e in ik_errors), default=0.)),
                'plan_success_rate': planned / len(plan_times), 'plan_time_median_s': float(np.median(plan_times)),
                'plan_time_p95_s': float(np.percentile(plan_times, 95))}
        out['tracking'] = self.tracking_metrics()
        if self.spec.base:
            out['base'] = self.base_metrics()
        out['step_ms'] = self.step_cost()
        return out

    def tracking_metrics(self):
        """Arm following planned paths: worst error during the path and the settle time."""
        self.reset()
        self.step(count=2)
        targets = [self.work_point(0., -.10, .195, down(45)), self.work_point(.10, .10, .095, down(70)),
                   self.work_point(-.05, -.15, .145, down(90)), self.work_point(.05, -.05, .245, down(45))]
        worst, settle = [], []
        for target in targets:
            result = self.model.plan_rrt(self.arm, target, world=True)
            if result['status'] != 'Success':
                continue
            error = 0.
            for row in result['position']:
                self.step({self.arm: row[None]})
                error = max(error, float(np.abs(self.arm_q(self.arm) - row).max()))
            steps = 0
            while np.abs(self.arm_q(self.arm) - result['position'][-1]).max() > .002 and steps < 50:
                self.step()
                steps += 1
            worst.append(error)
            settle.append(steps * DT)
        return {'paths': len(worst), 'max_error_rad': float(max(worst, default=np.nan)),
                'settle_to_2mrad_s_max': float(max(settle, default=np.nan))}

    def base_metrics(self):
        """Speed tracking (backing away from the table) and an in-place turn in the open."""
        self.reset()
        self.step({'base': [[-.4, 0.]]}, count=round(2. / DT))
        velocity = self.env.obs['agent']['qvel'][0, [self.env.joints.index(j) for j in self.spec.base.virtual_joints[:2]]]
        speed = float(np.linalg.norm(velocity))
        self.step({'base': [[0., 0.]]}, count=round(1. / DT))
        start = self.env.obs['agent']['base_pose'][0].copy()
        self.step({'base': [[0., .8]]}, count=round(2. / DT))
        self.step({'base': [[0., 0.]]}, count=round(1. / DT))
        turned = self.env.obs['agent']['base_pose'][0]
        target = self.env.terms['base'].pose[0].cpu().numpy()
        return {'speed_at_0.4_m_s': speed, 'turn_translation_drift_m': float(np.linalg.norm(turned[:2] - start[:2])),
                'pose_error_after_stop_m': float(np.linalg.norm(turned[:2] - target[:2])),
                'yaw_error_after_stop_rad': float(abs(np.angle(np.exp(1j * (turned[2] - target[2])))))}

    def step_cost(self):
        """Physics ms per step: robot holding still, and moving its arm."""
        self.reset()
        before = self.env.timings['step_s']
        self.step(count=50)
        hold = (self.env.timings['step_s'] - before) / (50 * SUBSTEPS) * 1e3
        before = self.env.timings['step_s']
        self.move_joints(self.arm, self.ik(self.arm, self.work_point(0., -.10, .195, down(45))), 2.)
        moving = (self.env.timings['step_s'] - before) / (50 * SUBSTEPS) * 1e3
        return {'hold': hold, 'arm_moving': moving}


def _pose_error(a, b):
    from table_1000.robots.kinematics import pose_error
    return pose_error(a, b)
