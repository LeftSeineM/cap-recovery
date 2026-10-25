# 05 逐步编写专家

多步任务的专家（如场景 000002：开抽屉、取笔、关抽屉……）一次跑完整个流程时，后面步骤的失败要等前面都成功才看得到，每轮迭代都很慢。逐步编写的做法：先把第 1 步迭代可靠，用它成功的结束状态作为第 2 步的起始状态，再迭代第 2 步，依此类推。每一步的起始状态都来自真实执行，底盘位置、物体被放下的位置都和全流程一致，不需要手写。

前提：专家由步骤组成（`Step` 与 `run_steps`，见[场景与专家轨迹](../designs/asset-pipeline/scenes.md#步骤)）。步骤之间手里不拿东西，每步的 `check` 检查结束状态是否满足下一步的起始条件。

迭代时的运行都是开发运行（`--dev`）：每次写到新目录 `runs/expert/<场景>/<日期>-<时间>-<提交>[-<标签>]/`，不会覆盖以前的结果，`--name` 给运行加一个标签。不带 `--dev` 的运行生成场景的数据 `data/scenes/<场景>/<后端>/`，只能跑所有 seed 和所有步骤。下文的 `<第 1 步的批次>` 等指这样的运行目录。

## 1. 迭代第 1 步

```bash
uv run python scripts/scenes/run_expert.py 000002 --physics isaac-cpu --dev --name step1 --steps 1 --jobs 20 --launcher '<调度命令>'
```

- 先用约 20 个 seed 的初态（`--seeds 0:20`），改专家、重跑，直到成功率约 95%。每轮都是新目录，可以直接和上一轮对比。
- 失败按 `summary.json` 的 `failures` 和 `episodes/<seed>.json` 分析；录像用 `render.py 000002 --physics isaac-cpu --batch <第 1 步的批次>`。
- 达到目标后用 100 个 seed 跑一遍。这一批的 `checkpoints/step1.hdf5` 就是第 2 步的起始状态，约 95 个。

## 2. 下一步从检查点开始

```bash
uv run python scripts/scenes/run_expert.py 000002 --physics isaac-cpu --dev --name step2 --steps 2 \
    --from <第 1 步的批次> --jobs 20 --launcher '<调度命令>'
```

- 每个有第 1 步检查点的 seed 都从它开始：恢复状态、静置 0.2 s，只运行第 2 步。
- 迭代第 2 步时只改第 2 步用到的代码，同一份检查点可以反复使用。
- 每个批次都有各步的检查点，完整流程的批次也一样，所以也可以从任意一次完整运行的第 k − 1 步开始只跑第 k 步。从初态开始的批次另有 `step0.hdf5`（初态），`--steps 1 --from <批次>` 从它开始，与从初态开始相同。
- `summary.json` 的 `steps` 给出每步的成功率和逐步完成率（前面各步都成功才计入，分母是第 1 步的 seed 数），可以直接看出整条流程走到哪一步损失最多。

## 3. 什么时候合并步骤

逐步运行时，每一步只看到前一步成功的那部分状态。几步都可靠后（单步约 95% 以上），把它们合起来跑（`--steps 1-3`，或不带 `--steps` 跑完整流程），确认合起来的成功率约等于各步之积。明显偏低时，说明前一步虽然通过了 `check`，留下的状态对下一步不利（如底盘停的位置、物体放下的朝向），应该收紧前一步的 `check` 或修改前一步。

## 4. 改了共享技能后

`table_1000/experts/skills/` 的改动会影响所有步骤和场景。改动后：

1. 已经迭代完的步骤，用原来的检查点各跑一遍，确认成功率没有下降；
2. 场景 000000 跑完整流程（100 个 seed），和 main 比较。

检查点记录了生成它的提交（文件属性 `commit`）。上一步的代码改了以后，它的检查点仍可用于迭代下一步，但最终数据要重新逐步生成或跑完整流程。

## 5. 接成完整轨迹

```bash
uv run python scripts/scenes/join_steps.py <第 1 步的批次> <第 2 步的批次> ...
```

同一 seed 在各批的轨迹按顺序接起来，每段截到下一批开始用的检查点；每条轨迹的 `parts` 属性记录各段的来源批次与提交。接好的轨迹默认写成场景的数据 `data/scenes/000002/isaac-cpu/`（整个替换），和一次跑完整流程得到的批次格式相同、只是生产方式不同，下游不需要区分；用 `render.py 000002 --physics isaac-cpu` 渲染。

拼接报告携带最后一段的调用帧、求解记录及帧入口状态，可继续生成失败快照或离线重解；更早各段和后续 repeats 的记录仍从来源批次读取。帧下标通过报告中的 `call_trace_offset` 换算，`failure_snapshots.py` 自动处理段首初态与段内状态。

## 看失败的案例

```bash
uv run python scripts/experts/failure_cases.py <批次> [--seeds 4 44]
```

在 GPU 上运行。对批次里每个失败的 seed，渲染失败子任务开始时和轨迹结束时的场景，视角包括机器人头部相机、场景自带的视角，以及俯视、左、右、前四个视角，结果写到批次下的 `cases/<seed>/`。`case.json` 记录提交、失败原因、该子任务里规划放弃各个选项的原因，以及碰到的物体。

想确认同样的代码重跑是否还会这样失败，可以从前一步的检查点只重跑这些 seed：

```bash
uv run python scripts/scenes/run_expert.py 000002 --physics isaac-cpu --dev --name rerun --steps 2 \
    --from <批次> --seeds 4 44 --jobs 2 --launcher '<调度命令>'
```

从前一步的检查点重跑会重新执行整步。需要复现原来的技能调用时，从调用帧生成快照：

```bash
uv run python scripts/experts/failure_snapshots.py <批次> [--seeds 4 44] [--frame 帧序号] [--name 标签]
uv run python scripts/scenes/run_expert.py 000002 --physics isaac-cpu --dev --from <快照批次> --steps k \
    --seeds 4 44 --repeats 10 --jobs 2 --launcher '<调度命令>'
```

每条运行（包括 repeats 的每次）都有 `calls/<seed>[-<r>].json.gz`，`episodes` 报告的 `call_trace` 指向它。解压后是 JSON（`version`、`frames`、`failure_frame`），包含步骤、子任务、技能调用的帧树；帧序号在每次运行内从 0 开始，`id`、`parent` 表示调用关系，`enter`、`exit` 是 `ctx.steps`，`failure_frame` 指向失败路径最内层帧。技能帧包含已经求值的参数、进入时的持物与规划状态、随机数状态和结果或异常。retry 和 chain action 也记录，但其中的回调仅记录名字；它们实际发出的具体技能可以恢复。

默认选失败路径上最内层的具体技能。若是步骤的 check 失败，选该步最后一次具体技能，并在重跑后再执行原 check。`--frame` 指定帧序号，控制帧会定位到它最后一次具体技能调用。快照保存该技能**进入时**的场景、控制指令、持物、规划器挂载及允许接触、子任务和随机数，写入新批次的 `checkpoints/step<k-1>.hdf5`。

`run_expert --from <快照批次> --steps k` 恢复后直接执行这一技能，使用原参数；这次运行的结果就是该技能（以及需要时的 check）的结果。恢复也保留入口时规划世界所见的障碍物和机器人配置，不在调用前用实时场景刷新原 IK 的输入。报告的 `replay` 给出来源帧、原失败原因，以及失败帧、原因是否一致。它不会执行该步后续技能，因此技能成功表示这次技能重跑成功，不生成整步完成的检查点。帧恢复直接从入口执行，不增加静置控制步。遥操的 failure 起点对带调用轨迹的案例也从该帧入口恢复持物等状态，再交给人。

控制步与录制状态的对应关系：入口 `enter = n > 0` 对应 `states[n-1]`，`enter = 0` 对应 `initial_state`；控制指令取技能入口记录的 `targets`。旧批次没有调用轨迹时，仍可用 `--before N` 生成场景快照，从失败前 N 秒重新执行整步；它不恢复原技能决策。带调用轨迹的批次使用 `--frame`，不使用 `--before`。

一个案例由“批次、步骤、seed”确定：所有历史运行都在 `runs/expert/<场景>/` 下，目录名带提交，`summary.json` 记录命令行和来源批次（`from`），可以据此整理各次运行的失败。

## 同一检查点重复运行

每个技能帧进入时，MPlib 的全局随机种子由 seed、初态 attempt、帧序号和 repeat 派生并记录。帧快照的第 1 次重跑恢复原 `ctx.rng` 状态和 MPlib 种子；后续 repeats 换随机数。普通检查点沿用 seed 和起始步骤的随机流。RRTConnect 有规划时限，机器负载不同时路径可能不同，个别 seed 的结果可能不一致。

## 离线重解候选与规划

`run_expert.py` 默认 `--problems failed`：每次运行照常记录，失败或程序错误才保存 `problems/<seed>[-<r>].json.gz`，报告的 `solve_trace` 指向它；成功时不写文件，`solve_trace` 为 null。需要分析成功规划时用 `--problems all`；`--problems none` 关闭求解记录和额外 IK 诊断。repeats 和从帧重跑按每次结果分别判断，批次 `summary.json` 的 `problems` 记录模式。三种模式都保存 `calls/`。

有求解记录时，`operations` 给出 IK 和规划的序号，`frame` 对应 `calls` 中的技能帧；`decisions` 中的候选事件和 `stage` 说明它在抓取 IK、预抓取 IK、then、need、放置 release／above IK 或规划的哪一关淘汰。

```bash
# 只用 MPlib，不启动 Isaac；默认在 runs/resolve/<场景>/ 下新建带时间与提交的目录。
uv run python scripts/experts/resolve.py <批次> --seeds 40 --frame 32 --runs 10 \
    --name frame32
# 指定 operations 中的规划序号；--repeat 1 可选该 seed 的第 2 次运行记录。
uv run python scripts/experts/resolve.py <批次> --seeds 40 --plan 108 --runs 10 \
    --name plan108
# 重新运行该帧最后一个候选决策，继续检查新通过候选的后续关卡。
uv run python scripts/experts/resolve.py <批次> --seeds 40 --frame 32 --redecide --runs 10 \
    --name redecide32
```

离线产物默认写入 `runs/resolve/<场景>/<日期>-<时间>-<提交>[-<标签>]/`，`--name` 设置标签，`--output` 可指定新的目录。它不属于专家轨迹批次，不应放在遥操扫描的 `runs/expert/` 下。

`--decision 序号` 只重解该决策及其子决策的已记录问题；`--redecide 序号` 则重新执行这个决策（抓取 reach、放置 put_down、底盘 base_for、push 或 handoffs）。第一次按原种子和调用前缀重解，其余换随机数；调用前缀中的 RRT 仍受时限和负载影响。每个数值问题从自己的原测量状态开始，逐项报告成功次数、状态分布和第一次是否与原记录相同；候选重选另报整体成功次数、每个候选与原判定的对照，并保存每次的候选、筛选和规划记录。每个 seed 的详细结果为 gzip JSON，`summary.json` 汇总次数、原因和负载。重选按理想关节跟随更新模型，不能判断笔是否实际抓住、放稳或落进抽屉。

`solve_trace` 为 null 或旧批次只有技能帧时，需先重新采集。对失败案例按上一节生成帧快照，再加 `--problems all` 重跑，保留成功或失败的记录：

```bash
uv run python scripts/scenes/run_expert.py 000002 --physics isaac-cpu --dev --from <帧快照批次> \
    --steps k --problems all --jobs 2 --launcher '<调度命令>'
```

记录来自这次帧重跑；结合报告的 `replay.same_frame`、`replay.same_reason` 判断它与旧失败的关系。格式和语义见[离线求解问题](../designs/asset-pipeline/scenes.md#离线求解问题)。

## 从失败定位根因

完整运行、技能帧重跑、固定决策重选和单个规划重解回答不同的问题，成功次数要分别统计：

1. 固定代码和初态运行完整批次，先按失败子任务、原因和筛选关卡分类；保留平均墙钟与负载，不能只比较成功率。
2. 从失败技能入口重跑，检查换随机数后是否能过。技能内重新选底盘、IK 分支或抓取，会改变后续规划问题，因此这一步成功不能直接归因于 RRT 搜索运气。
3. 离线固定最后的决策或规划问题，再重复求解。逐候选比较 release／above IK、then、need 和路径碰撞；只改变一个输入做对照，验证原因。例如终点都有 IK，不代表同一分支能完成连续动作；一条几何路线无碰撞，也不代表简化或时间参数化后的轨迹仍无碰撞。
4. 修正技能的通用语义，先验证相关失败帧，再跑完整批次。帧成功不包含后续动作和物理检查，不能代替完整任务成功。
5. 最后在新生成的一组初态上验证，生成目录放在自己的 `runs/` 中，再用 `run_expert.py --input` 指向它，保留原资产中的初态。

`need` 是拿起前对握法的可达性筛选，实际握法会受到物理接触影响；要把它的假设握法、关节分支与放置时的实测握法分开比较。放置筛选与执行都检查已采样的 release 分支能否在同一躯干高度到达上方，不因最近分支失败就丢弃其他已求出的解。

自由空间路线在简化和时间参数化后仍须无碰撞，且每个控制点都在 URDF 关节位置限位内。几何安全的样条也可能越界；此时沿已验证的分段路线停靠拐角，不能把越界指令交给物理限位截住，否则实际机器人会离开规划路线。

`ready` 使用机器人声明的 ready 臂姿态，没有声明的关节沿用初态；它只收回空闲臂，不改变躯干高度。共享升降也会带动空闲手，因此抓取需要动升降时先收臂，避免初态中安全的折臂姿态在低处随升降扫过障碍物。

抓取规划失败后的 `room` 若仍选择当前站位，就没有改变接近条件。技能在这种让位请求中排除已失败的站位，另选可达且途中安全的位置；直接调用 `make_room` 时，当前位置满足请求仍可不动。决策记录中的 `previous stance` 表示候选因重复站位被排除。分析时把站位的 IK 可达性和从当前关节到接近位姿的路径可行性分开核对。

底盘先检查直接转向、直走、转向的局部路线；若起始转向会扫到桌沿，再检查沿当前朝向先走到目标投影处、随后转向目标的路线。两条路线都按完整机器人和持物检查，收回空闲臂后仍不通就失败，不能执行已被判为碰撞的路线。这是同一侧的局部让位，不承担绕障导航。

抓取会在闭合后用实测位姿复核原有的最小夹持区域条件；接近或闭合时物体移动，可能使预测的可夹持抓法变差。未达到条件时按原有上限退出重抓，pick 仍用物体实际升高检查抬起结果。分析弱抓法时应读取闭合当步的物体/TCP位姿和开度，异常后的清理动作可能已经改变夹爪状态。

## 人工遥操案例

专家解不了的案例，可以人工遥操完成这一步，看可行的做法，并留下轨迹：

```bash
uv run python scripts/experts/teleop.py 000002 --gpu 0 --port 8765
```

在有 GPU 的机器上运行（`--gpu` 选显存空闲的卡），用浏览器打开 `http://<机器>:8765/`（例如先 `ssh -L 8765:localhost:8765 <机器>`）。左侧列出 `runs/expert/000002/` 下所有开发运行里失败过的案例（批次、失败的步骤、seed），`--batches <批次> ...` 另外加入其他批次；用 `--repeats` 跑过的批次按成功次数排序，成功率最低的在前。每个案例有两个起点：

- step：失败那一步开始时的检查点；
- failure：带调用轨迹时从失败技能帧入口恢复；旧批次从失败轨迹的最后一个状态恢复（失败时录了轨迹的运行才有，`--repeats` 的第 2 次起不录）；页面上的“s before the failure”设成 N 时，从失败前 N 秒开始（不早于这一步开始），用来看失败前发生了什么。

选一个起点，仿真回到那个状态，由你来完成这一步：

- 画面：一个第三视角（右键拖动转动，中键或 Shift+右键拖动平移，滚轮缩放）和机器人的 RGB 相机；左键拖动任一画面，在它的成像平面内移动末端目标（红点），Shift+滚轮沿视线前后移动；
- 按钮选择控制哪只手、开合各自的夹爪（显示开口）；
- 键盘：Z/X、C/V、B/N 旋转末端，I/K、J/L、U/O 沿底盘坐标轴平移末端，W/S、A/D 移动和转动底盘，R/F 升降，方向键转头，Tab 切换手臂，空格开合夹爪，P 暂停，按住 Shift 加速（先点一下页面）；
- “Check”在当前状态上运行这一步的检查，只显示结果，不结束录制；
- “Done”或“Not done”按你的判断结束这一步。检查可能写得不对，所以结果以你的判断为准，结束时的检查结果另外记录，两者不一致的案例说明检查需要修改。

每次尝试是一个新批次 `runs/teleop/000002/<日期>-<时间>-<提交>-step<k>-seed<s>[-failure]/`，结构与 `run_expert.py --dev --steps k --from <批次>` 的结果相同：轨迹（Done 的在 `trajectories.hdf5`，否则在 `trajectories_failed.hdf5`）、`episodes/<seed>.json`（`teleop` 中记录案例、起点、每次检查和最终判断）、`summary.json`；Done 的还有第 k 步的检查点，后面的步骤可以 `--from` 它接着跑，`render.py --batch` 可以渲染录像。Ctrl+C 结束程序，进行中的那次尝试不保存。

人工遥操保留调用帧与场景轨迹，不记录每个控制步的 IK 数值问题，`solve_trace` 始终为 null。

## 在修复之后比较有限重试

先修复确定性根因，再用相同初态、相同源码和负载记录比较实际完整运行的结果：

```bash
python scripts/scenes/run_expert.py 000002 --physics isaac-cpu --dev --seeds 0:300 --skill-attempts 1 --name root-fixes-k1
python scripts/scenes/run_expert.py 000002 --physics isaac-cpu --dev --seeds 0:300 --skill-attempts 2 --name root-fixes-k2
python scripts/scenes/run_expert.py 000002 --physics isaac-cpu --dev --seeds 0:300 --skill-attempts 3 --name root-fixes-k3
```

K 默认3，包含第一次，只影响专家已有的 `retry` 回调；K=1关闭这层兜底。`Unreachable` 也可换随机数重试，`Unmet` 仍交给动作链适配，程序错误直接抛出。新尝试前恢复回调入口的场景、夹持指令、持物、规划世界和需求缓存，不从已经失败的物理状态继续。主轨迹保留连续的前缀与最后一次尝试；墙钟和 CPU 秒包含全部尝试。

查看 `episodes/<seed>.json` 的 `retry_archives` 可找到每条被丢弃的失败分支：其 `batch` 相对主批次，指向 `attempts/<seed[-r]>/<尝试编号>/`。这是可独立使用的失败批次，可以直接运行本教程的快照提取和离线重解命令。最终成功不代表未发生重试；嵌套失败分支也从此清单查找。`--problems failed` 为这些失败分支保存求解问题，成功主轨迹的 `solve_trace` 仍为 null。

重试范围由已有回调决定：`open`/`close` 的内部完成检查在回调内，`place_on`/`place_inside` 的放置后检查和 `Step.check` 在回调外。后两类检查失败会结束运行，K不会使专家重新拿起物体或重做整步。

清理动作之后，归档仍定位原来抛出异常的技能。若失败来自回调自己的检查，异常归属控制帧；快照只恢复其中的具体技能，不会恢复检查闭包。需要分别核对技能结果与回调检查；步骤的 `Step.check` 仍可按前面的规则重跑。

帧重放只执行原子的具体技能；要比较该技能自身的随机性，用 `--repeats`，不会隐式添加重试。成功率和耗时曲线须来自上面的完整运行，单帧成功率的乘积只能作为估计。最后再用现有 `build_scene.py` 生成一批独立初态到新的 `runs/` 目录，通过 `run_expert.py --input <新初态批次>` 验证，不能覆盖 `assets/scenes/`。
