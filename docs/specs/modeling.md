# object.py 与 object.blend

## 入口

`main(argv=None)`，参数 `--output DIR`。在 Blender Python 中执行，每个脚本生成一个资产的 `object.blend`。共用函数从 `table_1000.modeling.asset_builders` 导入，包路径由统一构建入口提供。

## 坐标与结构

- 一个资产一个 `.blend`；Blender 世界坐标系作为资产局部坐标系，原点与轴向按物体常识定义。
- 米制：`unit_settings.system = "METRIC"`、`scale_length = 1`；应用对象缩放和碰撞修改器。
- 每个刚体对应一棵明确标识的完整子树，整体允许树或森林。
- 每个刚体至少包含一个部件；部件包含一个或多个视觉或碰撞网格，允许仅视觉或仅碰撞部件，内部机构的刚体可以只有碰撞几何。视觉与碰撞不要求一一对应；有功能的空腔（装物内腔、可拔笔帽、胶带芯孔）与活动间隙必须保留。无功能的封闭腔（如被笔芯占满的笔管）碰撞用外轮廓实心凸体（`tube(..., solid_collision=True)`），细长薄片分块会让 PhysX 报出错误法向的深接触；同一最终刚体内被外壳完全包住的碰撞体删除，视觉不随之简化。改变碰撞分解时保留原质量、质心与惯量（`configs/rigid_physics.json` 的 `center_of_mass`、`diagonal_inertia`、`principal_axes`），并从运行时复核。

| 对象 | 要求 |
| --- | --- |
| 刚体根 | 零顶点 Mesh Object；`rigid_body_root = True`；`collision_shape = "COMPOUND"` |
| 部件节点 | 刚体根的直接子级；零顶点 Mesh Object；`semantic_part = True`；`collision_shape = "COMPOUND"`；不标记 `rigid_body_root` |
| 视觉对象 | `geometry_role = "visual"`；部件节点的子级；有网格面和材质；不设刚体；渲染可见 |
| 碰撞对象 | `geometry_role = "collision"`；部件节点的子级；设置刚体碰撞形状；`hide_render = True` |
| 碰撞形状 | 当前支持 `BOX`、`SPHERE`、显式闭合凸网格 `CONVEX_HULL`；`use_margin = True`、`collision_margin = 0` |
| Scene | `penetration_tolerance_m`，单位 m，默认 0.0002 |
| Scene（可选） | `max_drop_m`：专家放置时允许的松手落差上限，单位 m，缺省 0.15，0 表示不允许落差释放；导出到 `model.json`，不是物理参数 |

`SPHERE` 的显示网格以对象原点为球心、各顶点到球心距离相同；检测和物理导出使用解析球体。

## 部件命名与资产树

具有独立功能或材质的部件须独立命名，对象名称在该资产内全局唯一。采用点号分隔的完整名称，例如 `cabinet.drawer1.handle`；重复组件显式编号，不依赖 Blender 自动添加的 `.001` 后缀。

名称各段使用英文字母、数字和下划线。点号是命名分隔符，实际父子关系由 Object 的 `parent` 定义；资产前缀不要求创建共同根对象，森林仍然允许。

同一部件使用独立视觉与碰撞对象时，分别命名为 `cabinet.drawer1.handle.visual`、`cabinet.drawer1.handle.collision`；多个凸块使用 `.collision1`、`.collision2` 等。它们共同挂在 `cabinet.drawer1.handle` 部件节点下。部件的 Compound 聚合其碰撞几何，整个部件仍属于上层刚体，不拥有独立的质量、运动或关节。需要不同接触材料的区域须有独立碰撞对象。

对象 README 提供[资产树](README.md#readme-与缩略图)，标明刚体与部件。验收核对部件划分、完整名称、刚体归属及 README 与模型的一致性；自动检查不检查命名语义与 README 的一致性，由人工核对。

## 资产树导出

从 `object.blend` 的实际 `parent` 层级导出树或森林，刚体根标注 `[rigid]`，展示组织节点、刚体根和部件，两个空格缩进一级，子节点名称省略与父节点相同的前缀，按[名称压缩表示](README.md#名称压缩表示)写出。

## 关节

原生 Rigid Body Constraint，`preview_joint = True`，对象名作为关节名；`object1` 为上游刚体、`object2` 为运动刚体。

| 类型 | 运动轴 | 单位 | 限位字段 |
| --- | --- | --- | --- |
| `SLIDER` | 约束对象局部 X | m | `use_limit_lin_x`、`limit_lin_x_lower/upper` |
| `HINGE` | 约束对象局部 Z | rad | `use_limit_ang_z`、`limit_ang_z_lower/upper` |

保存装配姿态为零位；启用限位时须包含 0。关节连接无环，每个运动刚体最多一个上游关节。`disable_collisions` 定义两连接刚体之间的禁碰关系。

## 第一阶段连续变形几何

连续袋面等不能表达为刚体的几何，可使用 `asset_builders.deformable(name, root)` 创建独立归属节点。该节点为零顶点 Mesh，标记 `deformable_root = True`，不带刚体；其直接子级由 `part()` 创建，同样为非刚体零顶点 Mesh。视觉与碰撞网格直接挂在部件下，分别沿用 `geometry_role`、材质和碰撞形状规范。归属节点不能嵌套另一个独立刚体或变形归属节点。

资产树将其标为 `[deformable]`，单独统计，不计入刚体数。真实的拉头、拉环等仍按刚体层级建模，原生关节只连接真实刚体。连续袋面不因采用此表示获得物理质量、弹性、固定连接或受力拉链。

模型 Scene 的 `geometry_controls` 保存 JSON 字符串，例如：

```python
scene["geometry_controls"] = json.dumps({
    "bag.unzip": {"owner": "bag.film", "limits": [0, 0.324], "unit": "m"},
    "bag.bulge": {"owner": "bag.film", "limits": [0, 1], "unit": "1"},
})
```

每项只能包含 `owner`、`limits`、`unit`；归属须存在且每个变形归属至少有一个控制量。限位有限、递增且包含零，单位为 `m`、`rad` 或无量纲 `1`。保存姿态对应所有控制量为零。开口位置是几何控制量，不虚构连接袋面与拉头的 `SLIDER`。

同目录 `object.py` 另提供 `apply_geometry_pose(controls)`：接收完整控制量字典，根据绝对参数重算视觉及碰撞的显式顶点。预览会执行该源码快照，与构建一样只用于可信资产源码，依赖同版本 `table_1000` 包；不增加文件类型，不把 Python 驱动器或处理器嵌入模型。模块顶层不得构建/清空场景，生成仍由 `main(argv=None)` 入口执行。

回调契约：

- 视觉与碰撞使用同一组曲面参数或采样点；拓扑、对象集合、角色及父子关系固定，不使用修改器或形态键实现隐式变形。
- 每次从控制量确定完整姿态，重复执行结果相同；不依赖上次开合、帧号、外部文件或随机状态。
- 可移动关联的刚体根、变形归属根及关节锚点，使拉头和拉环轴心随袋口移动；不得改动刚体内部网格或任意组织/部件节点的变换。真实关节随后由公共预览器施加，回调不重复施加关节角度。
- 公共预览在每次调用前恢复归属根、关节锚点和变形网格的保存状态；开口控制归零须同时恢复视觉及碰撞。

检测器在每个采样姿态重新读取变形归属下碰撞网格的顶点、面法向、边轴及包围范围，重新验证闭合凸性，再与其他归属检查穿透；开合和复位都执行。仍不检测同一归属内部自碰撞，不证明柔性力学或帧间连续运动安全。此表示仅支持第一阶段；当前物理导出器明确拒绝此类模型，不能当作已完成第二阶段的资产交付。
