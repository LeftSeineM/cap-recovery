# physics.py 与物理资产

`physics.py` 是精细物理资产（抽屉、卡扣等需要专门调参和物理测试的资产）物理属性的权威源码，通过官方 `UsdPhysics`、`PhysxSchema` API 为几何层配置质量、接触材料和驱动。源码随 Git 提交；生成的 `physics.usda`、几何层及 USDZ 不提交，也不手工维护第二份配置。

## 构建接口

资产模块提供 `author(geometry_path, output_path)`，导入时不启动仿真。统一构建入口启动环境、导出几何，再调用该函数，将物理属性写入指定的 USDA 层并保存：

```text
object.blend ──导出──> geometry.usdc
physics.py ──读取几何并配置──> physics.usda
              两层组合与打包──> object.usdz
```

编辑时将几何层放在弱层、物理层放在强层，并把编辑目标设为物理层。函数不修改输入几何，不将网格复制进物理层。原生 USD API 为属性接口，不另定义物理 JSON 字段。

## 节点与名称

几何层保存视觉、碰撞、刚体划分，以及关节连接、轴、锚点、限位和禁碰关系。

- `/Asset` 为资产参考系；刚体根带 `RigidBodyAPI`，部件为普通 Xform，网格位于部件下。
- USD 保留实际层级，局部名称转换为合法 USD 标识符，同胞名称不得冲突。
- 导出节点的 `table1000:name` 保留 Blender 全局名称，`table1000:role` 标识刚体、部件、视觉或碰撞。脚本可按名称配置特定部件，或按角色批量设置碰撞属性，不逐个枚举凸块。

## 属性约定

| 内容 | 原生表达 |
| --- | --- |
| 总质量 | 刚体上的 `MassAPI`：`physics:mass` |
| 质心 | `physics:centerOfMass`，刚体局部坐标 |
| 主惯量与主轴 | `physics:diagonalInertia`、`physics:principalAxes` |
| 接触材料 | `MaterialAPI`：静摩擦、动摩擦、恢复系数 |
| 关节驱动 | `DriveAPI`：目标、刚度、阻尼、最大力和驱动类型 |
| PhysX 专用属性 | 对应 `PhysxSchema` API |

每个刚体明确给出总质量。默认只设置质量，由引擎根据碰撞形状估算质心和惯量；需要特殊质量分布时，一并显式给出质心、主惯量与主轴。这些属性只配置在刚体根；当前资产仅填写质量，不显式覆盖质心、惯量和主轴。

同一刚体可有多种接触材料。使用 `MaterialBindingAPI` 的 physics purpose 绑定到部件，由其碰撞网格继承；需要不同材料时拆分部件。视觉材质与接触材料分开，不用接触材料推断质量分布。

Stage 使用米、千克和 Z 向上。USD 原生转动关节/驱动角度使用度；Python 测试接口使用弧度，辅助工具负责转换。

## 单刚体资产

不需要机构行为的资产（大部分文具、容器）整体按一个刚体仿真，不写自己的 `physics.py`，在 [`configs/rigid_physics.json`](../../configs/rigid_physics.json) 中各占一行：

```json
{
  "battery/000000": {"mass_kg": 0.023, "friction": [0.4, 0.3], "angular_damping": 2.0},
  "desk_lamp/000000": {"mass_kg": 2.5, "friction": [0.6, 0.5], "center_of_mass": [0.0, -0.03, 0.04]}
}
```

构建时由这些数值生成调用 `table_1000.physics.rigid.author` 的 `physics.py`，写入产物目录作为快照。资产目录里有 `physics.py` 的即为精细物理资产，不列入该表。

- 几何层中的多个刚体合并成一个刚体，位于它们的公共父节点（都在资产根下时即资产根）；原有关节停用，笔帽、盒盖、长尾夹把手和按钮固定在建模时的位置。合并后的刚体沿用父节点的 `table1000:name`，资产根则名为 `object`。
- 质心和惯量由引擎按碰撞形状计算；所有碰撞形状使用同一接触材料，恢复系数 0。
- `friction` 为静摩擦、动摩擦；圆柱形的资产（笔、胶棒、胶带、螺丝刀）另给 `angular_damping`（1/s），代替仿真中没有的滚动阻力，否则在桌上一直滚。
- 碰撞形状的均匀密度给错重心时（花瓶里的花束、香薰瓶的藤条，体积大而实际很轻），另给 `center_of_mass`（m，刚体坐标系）。构建时把显式给出的质心写入 `physics_build.json` 的 `centers_of_mass`（`{刚体名: 质心}`），场景布局按它计算单刚体资产的稳定放置朝向（台灯的重底座让它正立）。
- PhysX 会把超过 64 个顶点的凸包悄悄简化，物体因此下沉或倾倒；每个碰撞凸包不超过 64 个顶点。
- 质量与摩擦是按实物估计的设计值，未经标定。

## 验证与交付

构建后重新打开 USDZ，核对刚体、碰撞、关节及材料绑定。仅改变源码表达时，比较组合后有效属性、Schema、关系和材料解析结果；等价则无需重复动力学测试。改变质量、惯量或其他物理参数时，重新验证受影响的测试。

产物目录保存 `physics.py` 快照和生成的 `physics.usda`。带行为资产的入口与分发见[行为规范](behaviors.md)。

参考：[Isaac 资产分层](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_setup/asset_structure.html)、[MassAPI](https://openusd.org/release/api/class_usd_physics_mass_a_p_i.html)、[DriveAPI](https://openusd.org/release/api/class_usd_physics_drive_a_p_i.html)。

## 视觉材质

导出保留 Blender 的颜色、粗糙度、金属度、IOR 和 Alpha；透射与 Alpha 分开处理。一个视觉网格可带多个材质槽，按面分配的材质导出为 USD 的面子集（`materialBind` 族），各自绑定。透射材质使用随 USDZ 打包的 OmniSurface 入口，运行时依赖 Isaac / Omniverse 标准 MDL 库；仅支持 USD Preview Surface 的查看器不能等价显示透射。
