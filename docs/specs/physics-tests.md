# physics_test.py

带盖笔与三层抽屉柜使用本规范的 Python 测试入口；其他资产的测试按需补充。

资产目录中的 `physics_test.py` 验证当前资产，允许加载该资产的多个实例。不同资产的组合测试不属于本阶段验收。测试脚本随 Git 保存，调用 `table_1000.physics.testing` 的公共工具及官方仿真 API，不另设测试配置格式。

验收包括两部分：人工检查每项测试的视频和曲线是否符合实验预期，以及在明确条件下测得的实时因子 RTF。

## 统一入口

脚本导出 `TESTS: dict[str, Callable[[TestContext], None]]`。键为测试名称，也是输出名称；每个函数完成一个实验。脚本导入时不启动仿真。

```python
from table_1000.physics.testing import TestContext


def pull(ctx: TestContext):
    pen = ctx.asset
    body = pen.body("body")
    cap = pen.body("cap")

    ctx.camera(position=(0.3, -0.5, 0.2), target=(0, 0, 0))
    ctx.observe_body(cap, fields=("position", "linear_velocity"))
    ctx.plots("pull.force.x", "asset.cap.position.x",
              "asset.cap.linear_velocity.x")

    with ctx.fixed(body):
        ctx.force("pull", cap, frame="world", keyframes=[
            {"time": 0, "force": (0, 0, 0)},
            {"time": 1, "force": (-3, 0, 0)},
            {"time": 1.2, "force": (0, 0, 0)},
        ])
        ctx.run(2)


TESTS = {"pull": pull}
```

示例参数仅演示接口，不规定资产的受力阈值。入口负责启动仿真、加载资产及官方行为组件、独立复位、采样和结果写入；测试函数不自行启动或关闭 Isaac。每个“资产 × 后端”在独立进程中运行，测试脚本中 `ctx.physics` 为 `cpu` 或 `gpu`；一项测试失败不影响其余测试。测试默认读取构建时复制到资产目录的 `physics_test.py`。

并行环境数默认 CPU 为 1、GPU 为 4，在同一场景中运行相同的实验。各环境接收相同的动作、夹持轨迹和限值，夹持按各自环境的状态反馈；记录、曲线和视频取第一个环境，测试脚本无需修改。接触临界或限力切换等敏感过程中，各环境可能分化成不同的结果，`result.json` 的 `environments` 记录其他环境与第一个环境的终态最大位置差。GPU 上第一个环境的结果也可能随环境数略有不同，比较时保持相同的环境数。

每个测试开始时，仅加载默认实例 `ctx.asset`，名称为 `asset`，恢复资产保存的初态，时间为零。不同测试以及计时重复之间不残留上一次的实例、约束或行为状态。

## 实例与公共工具

`ctx.asset_path` 为被测资产入口；`ctx.spawn(ctx.asset_path, name="pen_b", position=..., rotation=...)` 加载另一实例。返回的实例提供 `body(name)`、`joint(name)`，名称在该实例内解析；记录使用实例名前缀。实例随上下文统一释放，不手工导入行为。

公共工具覆盖下列反复使用的操作；特殊实验可直接访问官方 USD/Isaac 对象，不为其扩展新的配置语言。

| 操作 | 约定 |
| --- | --- |
| 初态设置 | 资产整体位姿、关节位置，以及指定刚体的世界位姿/速度；仅初始化时设置 |
| `ctx.fixed(body, position=None, rotation=None)` | 把刚体刚性锁在一个静止的世界位姿（默认为声明时位姿），不能移动；退出作用域时释放 |
| `ctx.grip(body, name=None)` | 理想的六轴夹持，用于让物体按轨迹运动、引导或测力，见下文 |
| 关节锁定与驱动 | 初始化前声明结构；运行时通过张量设置限位或驱动参数 |
| `ctx.force()`、`ctx.torque()` | 注册具名关键帧动作；同时记录实际施加的世界力/力矩及作用点 |
| `ctx.run(duration)`、`ctx.step()` | 推进指定时长或一个物理步；统一执行动作、行为、采样及到期渲染 |
| `ctx.observe_body()`、关节观测 | 记录指定状态，单位由物理量确定 |
| `ctx.observe()` | 注册名称到取值函数的自定义观测；自定义量需注明单位，不能按名称猜测单位 |
| `ctx.plots(*columns)` | 选择已登记的标量记录列，每列一个面板，共享时间轴 |
| `ctx.camera()`、地面工具 | 设置测试相机及地面；地面参数、夹具和相机不写回资产 |

所有资产、夹具、接触报告属性及物理结构须在首次 `run/step/ensure_initialized` 前声明。运行期间只写 PhysX 张量状态、目标、参数与外力，不能增删关节或写 USD 物理属性。相机与灯光等视觉结构不受此限制。

### 夹持

`ctx.grip(body)` 是一个理想的六轴夹持：通过关节直接连接被夹刚体，没有几何、不参与碰撞、不会打滑。它在初始化前声明，初始时以声明位姿夹住刚体；被夹刚体是 articulation 根时（如抽屉柜外壳），整棵 articulation 随之运动，其余关节保持自由。运动用 `move_to`（直线与定轴转动，起止平滑）、`follow`（跟踪给定轨迹）、`track`（每步反馈时保持给定位姿与速度），力控用 `limit`、`axes`，`release` / `grasp` 松开与重新夹住；各方法的参数见 [`Grip`](../../table_1000/sim/isaaclab/fixtures.py) 的说明。

- 位姿、速度、力与力矩都在世界坐标系中，位置是刚体原点，旋转为四元数 `(w, x, y, z)`。
- `duration` 必须给出，且为物理步长的整数倍。阻塞式调用推进仿真直到运动结束，期间动作、行为与采样照常执行。多段运动按顺序衔接，可以在两段之间读取实时状态计算下一段。`wait=False` 只排队并返回运动对象，用 `ctx.wait(*motions)` 让多个夹持同时运动。
- 增益由被夹 articulation 树的质量与惯量自动确定，并补偿其重力与轨迹加速度；物体以连续的真实速度跟随轨迹，静止夹持时不因重力下垂。所用增益写入结果的 `grips`。
- 每步记录 `<name>.target.position.*`、`<name>.force.*`、`<name>.torque.*`（夹持施加给被夹树的力与对夹持点的力矩）及 `<name>.error.position` / `<name>.error.angle`。力在 CPU 与 GPU 上都可用于测量；GPU 上的力矩为估算值，约偏低 5%。
- 位移控制测力（如以恒定速度拔帽并取力的峰值）用 `follow`；力控制（如一只逐渐用力的手）用 `limit` 配合远处的目标。上限针对合力：夹持保持的横向约束力也计入其中，只关心某一方向的推力时，用 `axes` 只保留该方向（如沿笔轴合帽）。已知外载荷作用下的实验仍用 `ctx.force()`。


### 初态与时间

Python 接口统一使用 SI 单位、弧度和 wxyz 四元数。默认重力为 `[0, 0, -9.81]`；地面显式创建，摩擦和恢复系数记录在报告中。

先恢复保存姿态和关节位置，再应用资产整体位姿，最后覆盖单独刚体的世界位姿。整体角速度按最终质心相对资产原点的位置计算各刚体线速度 `v + ω × r`，最后应用单体速度覆盖。有关节连接的初态不得违反连接及限位。

初态位置表示资产原点的位置，不是距地面的最低点高度。测试开始后不通过写入姿态模拟受力运动。

动作时间从本次实验初始化完成后计，单位为秒。关键帧首尾定义生效区间 `[首帧时间, 末帧时间)`，区间外停止施力。每个动作至少两帧，时间递增并对齐物理步；省略字段继承上一帧，数组整体替换。力、位置线性插值，旋转使用最短路径 SLERP。

力与力矩可选 `world` 或当前 `body` 坐标，作用点单独指定 `point_frame`；省略作用点时作用于当前质心。多个力/力矩相加。夹具释放保留当时位姿和速度，不复位刚体；临时约束不修改源资产。

## 视频、曲线与结果

按物理步记录状态，统一使用 sim time；动作列是该步实际施加的值，不是接触力或夹具反力。状态采样标明积分前/后时刻，视频与同一时间戳的姿态一致。

相机参数包括位置、目标、up、分辨率和 fps。每帧间隔包含整数个物理步，首尾帧均采集，视频按正常速度播放。轨迹重渲染须标为回放，成本不混入在线仿真耗时。

每个测试只生成一张 `plots.jpg`，按选择顺序上下排列；标签使用列名，单位由观测定义确定，长度可显示为 m 或 mm，CSV 保持 SI。不选择曲线时不出图。已有 CSV 可以独立重画，不重新仿真。

结果写入资产目录下的 `physics_test/`，与 `preview/` 并列，按物理后端分目录。测试 `pull` 的结果为：

```text
physics_test/
  summary.json        各后端的状态、关键指标与 RTF
  isaac-cpu/
    run.log           该后端的运行输出
    run-retry<N>.log  第 N 次重跑的运行输出（有重跑时）
    pull.mp4          完整视频
    pull/
      plots.jpg       关注曲线
      result.json     执行状态、关键指标、实测量、条件与性能
      trace.csv       每个物理步的全部记录列
  isaac-gpu/
    ...
```

重跑仅清理所选测试的结果。`result.json` 包含测试名、资产与软件版本、生效物理/渲染条件、环境数、实测量及性能数据。其中 `execution_status` 由执行器写入：测试函数正常结束为 `completed`；抛出异常（包括脚本中的断言失败）为 `failed`，此时 `error` 记录异常信息，不生成视频和曲线。脚本用普通断言检查明确要求，不需要另建断言语言。

测试用 `ctx.metric(name, value, unit=..., label=...)` 声明给人看的关键指标，如 `ctx.metric('seating_push', peak, unit='N', label='合帽力')`；单位按便于阅读选择（N、mm、deg 等），保存在 `result.json` 的 `metrics`。其余实测细节写入字典 `ctx.experiment_details`（SI 单位，键名带单位后缀，如 `peak_pull_N`），原样保存在 `result.json` 的 `experiment`。验收条件仍用断言检查，指标只记录数值。

`summary.json` 由各后端已有的 `result.json` 汇总生成，只列入当前 `physics_test.py` 中的测试；只重跑一个后端或部分测试时，其余结果照常列入：

| 字段 | 含义 |
| --- | --- |
| `asset` | 资产路径，如 `pen/000002` |
| `backends.<后端>` | `engine`（引擎与版本）、`environments`（并行环境数）、`completed`（完成的测试数）、`failed`（失败的测试名） |
| `tests.<测试>.status` | 各后端的 `completed` 或 `failed` |
| `tests.<测试>.error` | 失败时各后端异常的最后一行，完整信息见对应 `result.json` |
| `tests.<测试>.metrics.<指标>` | `label`、`unit`，以及各后端的数值（4 位有效数字），同一指标的不同后端在同一行 |
| `tests.<测试>.rtf` | 各后端的 `physics`（物理推进 RTF）与 `video`（含视频 RTF） |

人工验收看什么见[第 2 章](../tutorials/02-physics-assets.md)；检查通过后，把关键指标和性能更新到资产 README 的“物理实测”一节。

## RTF 基准

`RTF = simulated_seconds / wall_seconds`，无量纲，越大越快。RTF=1 表示与真实时间等速。本阶段规定测量条件，不设置所有资产通用的最低 RTF。

| 条件 | 标准基准 |
| --- | --- |
| 引擎 | Isaac Sim 5.1 / PhysX；记录实际完整版本 |
| 物理步长 | `0.004 s`（250 Hz），每次推进一个物理步，无隐藏子步 |
| 求解器 | CPU PhysX、TGS，位置/速度迭代 16/4；CCD 开启 |
| 环境及实例 | CPU 1 个环境，GPU 4 个并行环境；默认 1 个资产实例，多实例另列实际数量，不与单实例直接比较 |
| CPU | 参考 AMD EPYC 7542；Kit/PhysX 工作线程目标 4，OMP/BLAS 目标 1；记录生效线程数与 CPU 型号 |
| GPU | 单张 RTX 4090；CPU 参考使用其渲染，GPU 对照使用 CUDA 刚体求解；分别记录后端、设备、驱动及其他负载 |
| 视频 | RTX Real-Time 2.0（`RealTimePathTracing`），960 × 480、25 FPS，即每 10 个物理步一帧；关闭运动模糊，1 spp，CPU/GPU 每次采集执行 2 次不推进物理的 Isaac Lab 渲染，总反弹与镜面/透射反弹上限均为 12 |
| 灯光 | 统一使用 `DomeLight` 强度 900 和 `DistantLight` 强度 2500；太阳光 `RotateXYZ = (-40, 20, 0)` 度 |

接触裕量、材料、阻尼和相机采用当前测试与资产的实际值并记录；灯光使用上述统一配置。同一实验的比较保持这些条件一致。保存实际渲染模式及采样/降噪设置，不能只报告“RTX”。指定条件无法满足时报告偏差，不标为同条件基准。

GPU 对照使用场景最小位置/速度迭代 64/16，TGS 开启逐次施加外力；GPU PhysX 不支持 CCD，因此关闭并披露。笔的 GPU 跌落实验单独使用 PGS，关闭不受 PGS 支持的逐次施加外力：TGS 在钢珠的 maximal-coordinate 关节撞击时出现异常能量增长。该选择保持资产材料、关节参数、4 ms 步长和验收阈值不变，报告记录实际求解器；它不属于与 CPU 同求解设置的性能比较。

默认每项测试只运行一遍带视频的正式实验，报告两项 RTF，用于确认速度没有明显异常：

- **含视频 RTF**：从首帧采集开始到视频和 CSV 写完，包含动作、采样、渲染和编码。
- **物理推进 RTF**：同一遍运行中物理步与采样的耗时，不含相机采集。渲染与物理共用机器资源，这只是无渲染速度的估计。

`--benchmark` 执行标准测量，在资产 README 中记录性能时使用。它在正式实验之前另外测量：

1. **无渲染 RTF**：包含物理推进、资产行为、测试动作、状态读取与内存采样，关闭相机采集和渲染。先在独立场景执行一次完整实验（至少 1 s）作为预热，销毁后用全新场景测量。每轮累计至少 10 s sim time，测量 3 轮并报告各轮与中位数。短实验以独立复位重复累计，复位耗时单列且不计入积分区间，不把实验尾部任意延长为静止计时。无渲染运行与正式实验的终态差也写入报告，用于核对两次运行的一致性。
2. **准备成本**：进程启动、资产加载/初始化、首次渲染预热及复位单列，不混入稳态无渲染 RTF。

多环境时，RTF 按单个环境的 sim time 计算，另报告所有环境合计的 `environment_seconds_per_second`（RTF × 环境数）。测量区间内同时采样资源占用：本进程 CPU 占用（核数）、整机 CPU 利用率、所用显卡的 GPU 利用率和显存。GPU 数值为整卡数据，包含同卡其他进程；报告同时记录该卡上的计算进程数（含本进程）。

更细步长或不同硬件的结果作为独立对照，不外推标准 RTF；标准无渲染 RTF 由 `--benchmark` 单独测量，不由带渲染的耗时减去渲染耗时得到。

参考：[官方独立 Python 工作流](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/introduction/workflows.html)、[Isaac Lab 刚体示例](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/01_assets/run_rigid_object.html)。
