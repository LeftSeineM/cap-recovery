"""X1Pro description (:data:`SPEC`, see table_1000/robots/spec.py).

TCP: a frame on ``<side>_arm_link6`` 0.145 m along the gripper axis, centred
between the finger pads; +X is the approach direction, +Y the direction in
which the left finger opens, +Z points to the wrist camera. The fingertips are
0.0185 m beyond the TCP.

Fingers are driven in metres: the left finger at ``+gap/2`` and the right at
``-gap/2`` gives a pad gap of ``gap`` (0 to 0.0975 m).
"""

from table_1000.robots.spec import ArmSpec, BaseSpec, CameraSpec, RobotSpec, Sensor

TCP_OFFSET = (.145, 0., 0.)
MAX_GAP = .0975
# Joint-space limits for time parameterization (rad/s, rad/s^2), well below
# the URDF velocity limits.
VELOCITY_LIMIT, ACCELERATION_LIMIT = .8, 1.5
# Planning hulls (except the fingers, which touch objects on purpose) are grown
# by this much, so plans keep the robot clear of itself and of objects.
PLANNING_PADDING = .002
# Bodies whose collision hulls overlap where the real robot does not collide:
# the head housing seated on the torso, the closed finger pads, and each arm's
# forearm (link4) resting on its upper arm (link2) when folded. The zero pose
# folds the arms fully (joint2 and joint3 at their limits), with the visual
# surfaces of link2 and link4 touching; the joint limits keep them from coming
# any closer. physics.py filters the same pairs.
SEATED = [('head_pitch_link', 'lift_link'),
          ('left_arm_gripper_left_link', 'left_arm_gripper_right_link'),
          ('right_arm_gripper_left_link', 'right_arm_gripper_right_link'),
          ('left_arm_link2', 'left_arm_link4'),
          ('right_arm_link2', 'right_arm_link4')]


def _arm(side):
    return ArmSpec(
        name=side, joints=tuple(f'{side}_arm_joint{i}' for i in range(1, 7)),
        tcp_link=f'{side}_tcp', tcp=(f'{side}_arm_link6', TCP_OFFSET),
        fingers=(f'{side}_arm_gripper_left_joint', f'{side}_arm_gripper_right_joint'),
        finger_links=(f'{side}_arm_gripper_left_link', f'{side}_arm_gripper_right_link'),
        max_gap=MAX_GAP, mount='lift_link', velocity_limit=VELOCITY_LIMIT, acceleration_limit=ACCELERATION_LIMIT)


# Arms raised in front of the torso, head looking at the work surface. The
# zero pose (every joint at 0: torso down, arms folded, head level) is the
# usual starting pose of the real robot.
READY = {'lift_joint': .40, 'head_pitch_joint': .30,
         **{f'{side}_arm_joint{i}': value for side in ('left', 'right')
            for i, value in zip((2, 3), (.4, -1.))}}

# 视角名与 robot-bridge 一致，安装在 URDF 光学坐标系上。
# 头部 DaBai DCW2 使用 2026-10-24 真机 color/depth camera_info 上报的参数；
# RGB 的 k3-k6 为零，以等价的 OpenCV k1,k2,p1,p2 表示，深度无畸变。
# 左右腕使用 2026-10-24 分别求解的原始 640×480 鱼眼内参，畸变顺序为 k1..k4。
# 留出 RMS 分别为 0.292/0.285 px；夹爪遮挡的未观测区域未验证。
DCW2_COLOR = Sensor((1280, 720), (718.66699219, 718.62670898, 641.63134766, 361.97146606),
                    distortion=(-.015393835492, .002529939404, -.000318315142, -.000307868962))
DCW2_DEPTH = Sensor((640, 400), (307.02694702, 307.02694702, 318.37670898, 195.97831726), kind='depth')
WRIST_FISHEYE = {
    'left': Sensor((640, 480), (280.06172223, 280.00303497, 318.72038243, 238.86527019), 'fisheye',
                   (-.018129471395, .001963860728, -.006322339487, .001676164700)),
    'right': Sensor((640, 480), (280.33196582, 280.39140119, 327.60904548, 245.44149876), 'fisheye',
                    (-.015410935420, -.004549747692, -.000553713973, -.000043040731)),
}
CAMERAS = {
    'face_view': CameraSpec(DCW2_COLOR, 'camera_head_front_color_optical_frame'),
    'face_depth': CameraSpec(DCW2_DEPTH, 'camera_head_front_depth_optical_frame'),
    **{f'{side}_wrist_view': CameraSpec(sensor, f'{side}_arm_gripper_camera_color_frame', self_view=.35)
       for side, sensor in WRIST_FISHEYE.items()},
}

SPEC = RobotSpec(
    name='x1pro', urdf='asset_sources/robots/x1pro/robot.urdf',
    arms={'right': _arm('right'), 'left': _arm('left')},
    # Wheels on base_link at (0, +-0.24, 0.086), 5 N m each (URDF): at most
    # 2 x 5 / 0.086 = 116 N of push and 116 x 0.24 = 28 N m of turn. Speed
    # limits are simulation values, not the real controller's.
    base=BaseSpec(wheels=('left_wheel_joint', 'right_wheel_joint'), wheel_radius=.086, track=.48,
                  max_speed=(.5, 1.), max_acceleration=(1., 2.), max_force=(116., 28.)),
    lift=('lift_joint',), head=('head_yaw_joint', 'head_pitch_joint'),
    keyframes={'ready': READY},
    cameras=CAMERAS,
    seated=tuple(SEATED),
    unpadded=tuple(f'{side}_arm_gripper_{finger}_link' for side in ('left', 'right') for finger in ('left', 'right')),
    planning_padding=PLANNING_PADDING)
