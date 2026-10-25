# preview.json

定义几何姿态与相机。顶层键为输出文件名：`.jpg` 对应静态图，`.mp4` 对应关键帧动画。

## 格式

```json
{
  "open_three_quarter.jpg": {
    "camera": [0.82, -0.82, 0.72, 0, 0, 1],
    "joints": {"drawer1_open": 0.173}
  },
  "open.mp4": {
    "fps": 24,
    "frames": [
      {"frame": 0, "camera": [0.82, -0.82, 0.72, 0, 0, 1]},
      {"frame": 24, "joints": {"drawer1_open": 0.173}},
      {"frame": 48, "joints": {"drawer1_open": 0}}
    ]
  }
}
```

| 字段 | 要求与默认值 |
| --- | --- |
| `camera` | `[x, y, z, upx, upy, upz]`；资产坐标系中物体指向相机的方向及画面向上参考方向，二者非零且不平行；数组保持单行 |
| `joints` | 关节名到位置的字典，滑动 m、转动 rad；未指定为 0，遵守模型限位 |
| `transforms` | 刚体根或变形归属根名称到 `[x, y, z, w, qx, qy, qz]` 的字典；未指定为 `[0, 0, 0, 1, 0, 0, 0]` |
| `controls` | 可选连续几何控制量字典；未指定为 0，单位及限位由 Scene 的 `geometry_controls` 声明，见[连续变形几何](modeling.md#第一阶段连续变形几何) |
| `frames` | 至少两个关键帧；`frame` 为整数，从 0 开始严格递增；首帧必须给出相机 |
| `fps` | 视频必填，1–120 的整数；也是几何粗测采样率 |

静态图必须给出 `camera`。相机采用正交投影、自动居中；视频使用整段共同画幅。

`transforms` 为相对保存姿态的增量：平移单位 m，平移与旋转轴均在资产坐标系中，旋转绕刚体根的保存原点；非零四元数标准化。每个采样姿态先复位，再摆关节，最后对指定刚体及其子树施加变换。

## 关键帧

省略字段继承上一关键帧。`joints` 按关节名合并，`transforms` 按刚体名合并且七值数组整体替换；空字典不复位。相机方向、关节位置和平移线性插值，旋转采用最短路径 SLERP。输出包含首尾帧，例如 0–48 共 49 帧。

`controls` 同样按名称合并、逐值线性插值；空字典继承，复位须显式写零。每个输出项独立从零开始。未知名称、非有限值及越限值报错。普通刚体资产不需要此字段，也不会执行源码回调。

连续袋面开合示例（对应模型须声明这些控制量）：

```json
{
  "open_close.mp4": {
    "fps": 24,
    "frames": [
      {"frame": 0, "camera": [0.82, -0.82, 0.72, 0, 0, 1]},
      {"frame": 24, "controls": {"bag.unzip": 0.324, "bag.bulge": 1}},
      {"frame": 48, "controls": {"bag.unzip": 0, "bag.bulge": 0}}
    ]
  }
}
```

每个采样姿态依次执行：恢复保存状态 → `apply_geometry_pose(controls)` 同步视觉、碰撞和关联刚体/关节锚点 → 原生关节运动 → `transforms`。粗测、共同画幅计算和实际渲染使用相同流程。回调执行后关节使用当前锚点，保证拉环随移动中的拉头转动。模型须有同目录的可信 `object.py` 快照；仅复制 `.blend` 不足以重现连续变形。

各输出项独立复位。静态分离姿态示例：

```json
{
  "separated_three_quarter.jpg": {
    "camera": [0.82, -0.82, 0.72, 0, 0, 1],
    "transforms": {"cap": [-0.055, -0.045, 0, 0.70710678, 0, 0, -0.70710678]}
  }
}
```

## 视图与输出

六视图：前 −Y、后 +Y、左 −X、右 +X、顶 +Z、底 −Z。四分之三方向为 `[0.82, -0.82, 0.72]`。向上参考通常为 +Z，顶视图为 +Y，底视图为 −Y。

每个资产提供六视图与四分之三图；关节资产提供端点姿态及零位、限位、往返运动视频；可分离部件提供分离姿态与运动视频。

JPG/MP4 为 960 × 480，左视觉、右碰撞；碰撞按刚体着色，无描边。预览不修改保存的模型。

存在连续变形几何时，右侧按归属着色，包含独立的变形归属；颜色不代表已定义柔性材料或物理约束。

## 自动检查与人工验收

几何粗测覆盖全部图片姿态和视频帧，检查不同刚体间的穿透，遵守禁碰关系；不检测帧间运动或求解动力学。

`acceptance.json`：

| 字段 | 含义 |
| --- | --- |
| `status`、`failures` | 自动通过：`passed`、`[]` |
| `samples` | 各图片与视频帧；通过时 `penetrations = 0` |
| `bodies`、`joints`、`ignored_body_pairs` | 刚体数、关节定义、禁碰对 |
| `complexity` | visual_triangles、colliders、primitive_colliders、convex_colliders、convex_vertices、convex_faces；仅报告总量 |
| `preview.status` | 完整渲染结束：`completed` |

同目录的 `summary.json` 是给人看的摘要：

| 字段 | 含义 |
| --- | --- |
| `status` | 同 `acceptance.json` |
| `rigid_bodies` | 刚体数 |
| `joints` | 关节数 |
| `complexity` | 同 `acceptance.json` |
| `poses_checked` | 检查的姿态数 |
| `penetrations`、`penetrating_poses` | 最深的 20 处穿透（图片或视频帧、刚体、深度 mm）与有穿透的姿态数；通过时为空和 0 |

连续变形资产另有 `deformables`（各归属的视觉/碰撞数量）、`geometry_controls`（控制量声明）、`deformation_scope`（能力边界），每个样本记录完整 `controls`；`bodies` 仍只统计真实刚体。动态碰撞在每个姿态重新校验并参与跨归属粗测。同一归属内部的袋面自碰撞、袋口封闭性、装物路径和装物后的闭合须由资产专项夹具验证，公共报告不覆盖这些检查。

交付要求自动检查通过、完整渲染完成，并人工查看图片与视频、核对[资产树与部件命名](modeling.md#部件命名与资产树)。
