# X1Pro 双臂移动机器人（FX001 平动夹爪）

![preview](three_quarter.jpg)

带抬升柱、双 6 自由度手臂和平动二指夹爪的差速轮式机器人。换用转动二指夹爪 rm001 的同一台机器人见 [x1pro_rm001](../x1pro_rm001/)，它引用本目录的网格与 `.py` 文件。`robot.urdf` 来自真机控制容器中的 xacro 展开（底盘 dvt1、从臂 hy_1.0、夹爪 fx001_h_evt1），关节、限位、质量与惯量保持真机值。真机导出的原始 URDF 包（`ex001_render`，含 157.7 MB 外观 STL 与未采用的厂商碰撞网格，共 193 MB）不进入 Git：存档于 Hugging Face 私有数据集，版本记录在仓库根目录的 `assets.lock.json`，下载到已忽略的 `assets/robots/x1pro/raw/ex001_render/`。机器人构建只重写 `assets/robots/x1pro/` 下的 `import/` 与生成文件，不触及 `raw/`。

## 关节

| 关节 | 类型 | 限位 | 说明 |
| --- | --- | --- | --- |
| `lift_joint` | 滑动 | [0, 0.78] m | 抬升柱，正值向上 |
| `head_yaw_joint`、`head_pitch_joint` | 转动 | 见 URDF | 头部 |
| `{left,right}_arm_joint1..6` | 转动 | 见 URDF | 手臂 |
| `{left,right}_arm_gripper_{left,right}_joint` | 滑动 | 左指 [0, 0.04875] m，右指 [−0.04875, 0] m | 绝对值增大为张开 |
| `{left,right}_wheel_joint` | 连续转动 | — | 仅阻尼，按底盘速度随动 |
| `base_x_joint`、`base_y_joint`、`base_yaw_joint` | 滑动、滑动、连续转动 | — | 构建时加入的虚拟底盘关节（世界 x、y、偏航），不在 `robot.urdf` 中 |

真机夹爪由一个电机角 `*_arm_gripper`（0–5.717 rad）通过 mimic 带动两指，每指位移 = 0.00852694 m/rad × 电机角。这里把电机关节设为固定，左指按位移驱动，右指通过 PhysX mimic 关节跟随（右指 = −左指），控制时按上式换算。

机器人描述（关节分组、TCP、夹爪、底盘、关键姿态、相机）在 `robot.py`，验收设置（检查场景、工作点、用例）在 `robot_test.py`。关键姿态：零位（全部关节为 0，真机最常用的起始姿态：躯干最低、两臂折叠、头平视）与 `ready`（躯干 0.40 m、头俯 0.30 rad、两臂 joint2 = 0.4、joint3 = −1.0）。

## 相机

相机按 robot-bridge 的视角命名，内参和畸变定义在 [`robot.py`](robot.py)，由 `x1pro_rm001` 共用。内参单位为像素，适用于表中的原始分辨率：

| 视角 | 分辨率 | 模型 | `(fx, fy, cx, cy)` |
| --- | --- | --- | --- |
| `face_view` | 1280×720 | 带畸变针孔 | `(718.6670, 718.6267, 641.6313, 361.9715)` |
| `face_depth` | 640×400 | 无畸变针孔 | `(307.0269, 307.0269, 318.3767, 195.9783)` |
| `left_wrist_view` | 640×480 | OpenCV 鱼眼 | `(280.0617, 280.0030, 318.7204, 238.8653)` |
| `right_wrist_view` | 640×480 | OpenCV 鱼眼 | `(280.3320, 280.3914, 327.6090, 245.4415)` |

头部 Orbbec DaBai DCW2 使用 2026-10-24 从真机 `/camera_head_front/color/camera_info` 和 `/camera_head_front/depth/camera_info` 读取的内参和畸变。这些是驱动上报值；设备存储或配置中的底层来源尚未核实。RGB 的 `rational_polynomial` 系数中 k3–k6 均为零，使用等价的针孔 `(k1, k2, p1, p2)` 表示；深度使用自身内参。

左右腕无深度，使用 2026-10-24 分别求解的鱼眼内参和四项非零畸变 `(k1, k2, k3, k4)`；留出重投影 RMS 分别为 0.292、0.285 px。左腕另一次采集的独立验证 RMS 为 0.286 px。未观测或被夹爪遮挡的区域未验证，左右腕参数不能互相替代。

相机安装位置仍由各自 URDF 的光学坐标系给出。这次更新内参和畸变；头部 RGB–深度外参来源、TF 一致性及精确配准仍待确认，见 [#23](https://github.com/xucj98/table-1000/issues/23)。

指尖、相机等无质量坐标系在导入时并入所属刚体；需要时由 URDF 中对应固定关节的偏移计算，例如 `right_arm_gripper_left_tip_joint`。

## 几何与碰撞

- `meshes/visual/`：外观网格，由真机 STL 用二次误差简化，每个网格简化前后表面距离的 99 分位不超过 1 mm（共约 37 万面）。只按面数预算简化时，躯干外壳被压坏，误差约 19 mm。
- `meshes/collision/`：每个外观网格的少量凸块（CoACD，阈值 0.3；主体部件最多 4 块，传感器与轮子 1 块），导入为凸包碰撞。外观顶点越出碰撞凸块最多 0.19 mm（夹爪座），其余在 0.04 mm 以内。
- `arm_link6` 与夹爪电机坐标系位于夹爪座内部，不设碰撞。
- 指垫：每根手指的凸块内侧面不平行，持物时每指只有一个接触点，物体绕两点连线转动；右指的一块凸块还把内侧平面一直延伸到手掌，实物在指尖之后是收窄的。因此每根手指在指尖内侧加一个 25 × 18 × 3 mm 的盒形指垫（`prepare.py` 的 `PAD`），比凸块突出 0.5 mm，持物的接触落在指垫上，每指 4 个接触点。夹住物体时的指间距因此比物体宽 1 mm。

## 物理

`physics.py` 设置关节驱动（力驱动，增益按弧度或米给出）、articulation 求解迭代与自碰撞，以及接触材料：机身静/动摩擦 0.4/0.3，手指 1.0/0.8。

- **夹紧力**：手指驱动 5000 N/m、阻尼 60 N·s/m，力上限 20 N（URDF）。闭合目标越过物体 4 mm 即达上限，1 cm 与 4 cm 的物体夹紧力相同。两指耦合平分驱动力，每个指垫压 10 N。
- **两指耦合**：右指 mimic 左指，只驱动左指。两指各自驱动、都顶在 20 N 力上限时，沿开合方向没有回复力：持物原地转向时两指连同物体侧移了 13 mm；耦合后小于 0.1 mm。
- **指垫扭转摩擦**：接触片半径 6 mm（`torsionalPatchRadius`），否则物体可绕两指连线自由转动。
- **底盘**：虚拟关节刚度 1e5 N/m、8000 N·m/rad，推力上限 116 N、扭矩上限 28 N·m（两个轮电机各 5 N·m，轮半径 0.086 m，轮距 0.48 m）。挡住时停下，不会把障碍物硬推开。机器人与地面的碰撞关闭，轮子只随动。每个夹爪的两指之间、头与躯干之间、每只手臂的上臂（link2）与前臂（link4）之间不检测碰撞（规划模型也排除）：它们的凸块互相重叠，真机上由导轨、装配或关节限位保证不接触。零位时手臂折叠，joint2、joint3 都在限位上，前臂外壳贴在上臂上，两者不能更近。机身连杆不受重力，相当于理想的重力补偿控制，位置目标没有下垂；手中物体仍受重力。增益与摩擦是仿真取值，未对照真机控制器标定。

## 使用与修改

使用这台机器人只需本目录：构建出 `assets/robots/x1pro/robot.usda` 后即可在场景中引用，验收与遥控同样不需要原始包。

```bash
uv run python scripts/robots/build_robot.py x1pro --gpu 0
uv run python scripts/robots/check_robot.py x1pro --physics isaac-cpu --gpus 0
uv run python scripts/robots/teleop.py x1pro --output outputs/robots/x1pro-teleop --gpu 0 --port 8765
```

修改机器人时才需要原始包：`prepare.py` 写明所用的 URDF 配置、每个连杆的凸块数和关节改动，转换工具据此从原始包重新生成 `robot.urdf` 与 `meshes/`，再检查碰撞覆盖与目录大小；之后重新构建、验收并更新缩略图。

```bash
uv run python scripts/assets/remote.py pull robots/x1pro/raw/ex001_render
uv run python scripts/robots/prepare_robot.py x1pro
uv run python -m table_1000.robots.preview asset_sources/robots/x1pro --joints '{"lift_joint": 0.4, "head_pitch_joint": 0.3, "right_arm_joint2": 0.4, "right_arm_joint3": -1.0, "left_arm_joint2": 0.4, "left_arm_joint3": -1.0, "right_arm_gripper_left_joint": 0.03, "right_arm_gripper_right_joint": -0.03, "left_arm_gripper_left_joint": 0.03, "left_arm_gripper_right_joint": -0.03}'
```

缩略图左侧为外观网格，右侧为碰撞凸块，每个刚体一种颜色。构建约 4 s。构建后为 22 个机器人刚体加 3 个虚拟底盘连杆，86 个碰撞形状。零位与 `ready` 无自碰撞；检查场景中每物理步约 2.6 ms（静止）至 4.8 ms（手臂运动），关节跟踪误差小于 1 mrad。当前内参的真实图像验证见上文，更新参数后的 Isaac Sim `sensors` 渲染验收待重跑。转换约 10 分钟，只检查用 `--check`，约 10 s。缩略图需要 Blender。

