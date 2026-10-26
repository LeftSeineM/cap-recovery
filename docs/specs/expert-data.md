# 专家数据

规定专家运行产生的数据：批次、轨迹、报告、录像、检查点，以及用于失败分析的调用帧和求解记录。各文件的位置见[资产布局](README.md#资产布局)，专家程序见[专家规范](experts.md)。

## 批次

一次 `run_expert.py` 运行产生一个批次，目录格式相同：

- **场景的数据** `data/scenes/<场景>/<后端>/`：所有 seed 从初态跑完所有步骤（或由各步拼接而成），每次生成整体替换。
- **开发运行** `runs/expert/<场景>/<运行>/`（`--dev`）：`<运行>` 为 `<日期>-<时间>-<提交>[-<标签>]`，以后的运行不会替换它。只跑部分步骤或 seed 的运行都是开发运行。

后端命名与物理测试一致（`isaac-cpu`；场景生成尚未在 GPU 上验证）。`summary.json` 的 `commit` 是工作进程实际运行的提交，有未提交的改动时带 `+dirty`，这时 `dirty_patch` 为批次目录中保存这些改动的文件名，在该提交上应用即可还原代码；源码根目录不是一个检出时 `commit` 为 `null`（见[批量运行](../tutorials/README.md#批量运行的通用参数)）。

## 轨迹

**无图片、可重放**：直接使用 Isaac Lab 录制器的 HDF5，`trajectories.hdf5` 存成功的，`trajectories_failed.hdf5` 存失败的。每条轨迹 `data/demo_<seed>`：

| 内容 | 说明 |
| --- | --- |
| `initial_state` | 静置后的场景状态（`InteractiveScene.get_state`：每个 articulation 的根位姿、速度、关节位置与速度，每个 rigid object 的位姿与速度） |
| `actions` `[T, A]` | 每个控制步的动作；各动作组的区间见文件的 `env_args.action_layout` |
| `states` | 每步之后的场景状态 |
| `obs/*` | 每步之前的机器人观测（`qpos`、`qvel`、`base_pose`、`tcp_<arm>`） |
| 属性 | `scene`、`seed`、`attempt`、`instruction`、`subtasks`（每段的指令与起止步）、`status`、`failure`；由步骤组成的专家另有 `steps`（完成的步骤及其结束的控制步）和 `from`（开始用的检查点文件，从初态开始时为空）；拼接的轨迹另有 `parts` |

`env.reset_to(initial_state)` 后依次执行动作可在物理中重放；`scene.reset_to(states[t])` 精确恢复第 t 步，用于渲染。每条轨迹开始前用 seed 设置 MPlib 的随机数，RRTConnect 有规划时限，负载不同时路径仍可能不同：复现轨迹以录制文件为准。

一批轨迹由多个进程各自录制，结束时按 seed 顺序合并。带图片的数据集（如 LeRobot）可以由 `trajectories.hdf5` 逐步恢复状态、渲染机器人相机得到，目前不在生产流程中。

## 报告

每次运行一份 `episodes/<seed>.json`；同一起点的第 r 次重复（r = 1 … K − 1）为 `episodes/<seed>-<r>.json`，不录轨迹、不写检查点。报告包括：

- 子任务起止、每次规划、手指碰过的部件、换用的候选（`ctx.note`）、失败原因（附调用栈）、物体终态；
- 由步骤组成的专家：每步结束时的控制步、wall time 和进程的 CPU time（秒）（`steps` 的 `at`、`wall_s`、`cpu_s`），每个子任务的 wall time（`subtask_wall_s`）；
- 负载 `load`（见下文），以及调用帧 `call_trace`、求解记录 `solve_trace` 的相对路径；
- 有限重试丢弃的失败分支 `retry_archives`：每个分支一个相对路径 `attempts/<seed[-r]>/<尝试编号>/`，包括被外层重试再次丢弃的嵌套分支。分支是格式相同的完整批次（轨迹、检查点、调用帧、求解记录、报告），可以直接交给失败快照、离线重解和渲染工具；
- 开启 `--profile` 时的计时（见[计时](#计时)）。

手指以外的接触不统计：密集摆放时手臂碰到物体是正常的。

## 汇总 `summary.json`

批次的验收入口，不设硬性门槛：

| 字段 | 内容 |
| --- | --- |
| `commit`、`dirty_patch`、`run`、`command`、`from` | 提交、未提交改动的 patch（只在 `+dirty` 时有）、运行名、命令行、来源批次 |
| `seeds` | 这一批的 seed 区间与个数 |
| `success_rate` | 成功的 seed 占全部 seed 的比例 |
| `failures` | `{"子任务: 原因": [seed, ...]}`；原因来自专家的检查（`ctx.fail`）、规划失败或专家代码出错（`error`）；进程中途退出而没有结果的 seed 记为 `-: process exited` |
| `unreachable`、`unreachable_rate` | 什么都够不着（`Unreachable`）的 seed 及其比例 |
| `mean_duration_s`、`mean_wall_s` | 成功轨迹的平均 sim time 与每条的 wall time（秒） |
| `steps` | 由步骤组成的专家：每步的起始 seed 数、完成数、成功率（完成 / 起始）和逐步完成率（这一步及之前各步都完成的 seed 占第 1 步起始 seed 的比例，ManiSkill-HAB 的统计方式）；从检查点开始的批次沿用来源批次前几步的行和分母 |
| `repeats` | 有重复运行时：每个 seed 的成功次数（满分 K）、各失败原因在每个 seed 上出现的次数、全部运行的成功率，以及因进程退出而少跑的 seed（`short`） |
| `load` | 这一批的负载汇总，见下文 |
| `problems` | 求解记录的保存模式 |
| `skill_attempts` | `--skill-attempts` 的 K |
| `videos`、`fails` | 渲染后补上：按相机记下有录像的成功、失败 seed，合成录像记在 `merge` 下 |

批次的成功率、失败列表和逐步完成率都按每个 seed 的第一次运行计。`scripts/experts/step_times.py` 按步骤和子任务汇总 sim time、wall time 和规划耗时；`scripts/experts/resource_stats.py` 汇总每条轨迹躯干、底盘的移动次数与行程。

## 负载

机器忙时 wall time 会变长，所以每条轨迹记下运行时的负载，不要求在空闲时运行。

- `episodes/<seed>.json` 的 `load`：机器名 `host`、整台机器的核数 `cores`，以及开始、结束和其间每 30 s 一次的采样 `samples`（`t` 秒、1 分钟平均负载 `load_1m`、距上次采样进程的 CPU time `cpu_s`）；`cpu_s` 为整条轨迹进程的 CPU time。
- `summary.json` 的 `load`：`hosts`，所有采样的“负载 / 核数”与各条轨迹“CPU time / wall time”的分位数（`p10`、`p50`、`p90`、`p100`）。

CPU time 含仿真与规划的所有线程，可以多于 wall time；同一份代码在忙的机器上“CPU time / wall time”更低。

## 计时

`run_expert.py --profile`（默认关闭）在报告中按类别记录 wall time、线程 CPU time 等：抓取候选、IK、RRT、时间参数化、碰撞检查、路径验证、世界更新、底盘与交接搜索、控制步进。只有扣除子调用的 `self_s` 可以跨类别相加；各字段的含义见 [`timing.py`](../../table_1000/common/timing.py) 的说明。

## 录像

`render.py` 按状态逐步恢复并渲染指定的相机（机器人的 RGB 相机或场景的固定视角），不推进物理。

T 步的轨迹录像为 T 帧（25 fps，1 倍速），第 t 帧是第 t 步动作之前的状态，与 `obs[t]`、`actions[t]` 对应。成功的在 `videos/`，失败的在 `fails/`，每条轨迹一个目录，每个相机一条（`<seed>/<相机>.mp4`），或合成一条（`<seed>/merge.mp4`）。每次只写本次渲染的录像，其他录像保留。生成新批次时整个目录被替换，旧的录像随之删除。不带标注的单相机录像可以直接用来制作数据集。

## 检查点

由步骤组成的专家，每个批次按步骤存 `checkpoints/step<k>.hdf5`，从初态开始的批次另存 `step0.hdf5`（初态，名字为 `initial state`）。每个 seed 一组，属性 `seed`、`attempt`、`at`（这一步结束时的控制步）；内容：

- `state`：完整场景状态（格式同 `initial_state`：物体位姿与关节，机器人的底盘、躯干、手臂和夹爪）；
- `state/targets/<动作组>`：各动作项保持的指令（关节目标、夹爪的指令开口、底盘速度）。夹着东西的夹爪指令开口小于实际开口，靠这个差保持夹紧，恢复时一起写回；
- `start`：该 seed 回合开始时机器人的关节和底盘位姿。

文件属性记录场景、机器人、后端、提交、步骤序号和名字。上一步的代码改了以后，它的检查点仍可用于迭代下一步，但最终数据要重新生成。

## 拼接

`join_steps.py` 把逐步运行的各批按顺序接成完整轨迹，输出格式与一批轨迹相同：

- 每段在下一批开始用的检查点处截断，接上下一批同一 seed 的轨迹；`subtasks` 和 `steps` 平移到接好后的位置，`parts` 记录每段的批次、提交、步骤、起点和长度。
- 每个 seed 的报告取最后一段的报告并平移；`summary.json` 按同样的方式统计。
- 只有所有 seed 跑完所有步骤的拼接才默认写成场景的数据。
- 报告保留最后一段的 `call_trace`、`solve_trace`，对应文件复制到新批次；帧号和数值记录不改写，`call_trace_offset` 是原记录控制步到拼接轨迹的偏移。原帧 `enter = 0` 的入口状态另存为 `calls/<seed>-initial.hdf5`（`call_trace_initial_state`），恢复时取该段自己的初态。再次拼接会累加偏移并携带入口状态。
- 拼接不合并更早各段的调用帧、求解记录或 repeats；需要时读来源批次。

## 调用帧

每次运行（包括 repeats）保存 `calls/<seed>[-<r>].json.gz`，解压后是 JSON：`version`、`frames`、`failure_frame`（失败路径最内层帧）。帧有步骤、子任务、技能三种，帧序号在每次运行内从 0 开始，`id`、`parent` 表示层次。每帧记录：

- 名字、已求值的实际参数、进入和退出的控制步（`enter`、`exit`）；
- 入口子任务、`ctx.rng.bit_generator.state` 与派生的 MPlib 种子；
- 完整持物（手臂、Target 的构造数据、supports、grasp）、规划器挂载物体的相对位姿及允许接触，以及入口时规划世界的障碍物位姿和机器人配置；
- 结果或异常（原因、details、是否 `Unreachable`）。

retry、chain 和 Pick / Place / Handover action 也记录为控制帧；动作链传下的需求（`InHand`）以可重建的形式保存。场景状态共用录制的轨迹：入口 `enter = n` 的状态，n 为 0 时取 `initial_state`，否则取 `states[n-1]`，控制指令取帧的 `targets`。

`failure_snapshots.py` 从帧生成快照批次，快照写为新批次的 `checkpoints/step<k-1>.hdf5`，从它运行时直接调用该技能（恢复语义见[失败恢复](experts.md#失败恢复)）。报告的 `replay` 给出来源帧、原失败原因，以及失败帧和原因是否相同（`same_frame`、`same_reason`）。

## 求解记录

`run_expert.py --problems {failed,all,none}` 控制保存范围，默认 `failed`：运行中照常记录，结束为失败或程序错误时保存 `problems/<seed>[-<r>].json.gz`，报告的 `solve_trace` 指向它；成功时不写文件，`solve_trace` 为 null。`all` 保存所有运行；`none` 关闭记录及额外 IK 诊断。repeats 和帧重跑按每次运行自己的结果判断；`calls/` 在三种模式下都保存。人工遥操不记录求解问题，`solve_trace` 始终为 null。

记录包含技能帧内的决策树、候选和它们被淘汰的关卡（`stage`），以及每次 IK、直线规划、RRTConnect 的完整输入（机器人配置、目标、预算、当时的规划世界）与结果。以 `frame` 关联调用帧；结构见 [`problems.py`](../../table_1000/experts/problems.py) 的说明。

## 离线重解

`scripts/experts/resolve.py` 只用 MPlib、机器人资产和求解记录，不启动仿真；结果写到 `runs/resolve/<场景>/<运行>/`，运行名格式同开发运行，`summary.json` 是各 seed 重解结果的列表，不是专家批次。

- **重解问题**：按批次、seed、帧、决策或规划序号选取记录的问题，每次从记录的测量状态开始；第一次用原随机数，其余换随机数。报告每个问题的成功次数、状态分布和第一次是否与原记录相同。
- **重选候选**（`--redecide`）：用专家的候选枚举、IK、需求和规划规则重新执行一个决策，继续检查新通过候选的后续关卡，并与原判定对照。

两者衡量的是数值问题能否求解。运动按理想关节跟随更新，不计算接触、跟踪误差和步骤谓词，看不出物体是否实际抓住、放稳；RRTConnect 的时限仍受负载影响。
