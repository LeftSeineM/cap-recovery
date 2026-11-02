# Skill 层：专家、skill、primitive 与谓词库

## 决定

专家程序之下分为 skill、primitive、求解器和执行四层，各层共用一个谓词库：

```
专家       STEPS：每个 step 调用一次 skill（目标参数 + hints）
  │
skill      Put / Open / Close / PushTo：调度 primitive、联合规划、处理失败、验收目标谓词
  │
primitive  Move / Grasp / Carry / PlaceAt / Release / Push / Slide / Articulate / JointPush / Press / Drive / Torso
  │          每个对应一个数学问题：plan 只求解，execute 执行、等稳定、验收效果谓词
  │
求解器     候选生成、IK、RRTConnect、直线运动、底盘路线、碰撞检查
执行       控制步、跟踪、物理仿真

谓词库     各层共用的唯一判据定义
```

### 分层与边界

| 层 | 职责 | 输入 → 输出 | 只能调用 | 不能做 |
|---|---|---|---|---|
| 专家 | 任务分解与策略：step 的顺序、每个 step 的目标与提示 | `STEPS` | skill、谓词库 | 写站位、坐标、等待、时间预算、判据；调用 primitive 或求解器（临时越界除外） |
| skill | 调度 primitive，联合规划，处理失败，验收目标谓词 | 目标参数 + hints → `{ok, predicate, measured, reason, tried}` | primitive 的 `plan` / `execute`、谓词库 | 构造数学问题，直接调用求解器或发控制步 |
| primitive | 把意图写成一个数学问题（决策变量、约束、目标、参数），求解、执行、等稳定、验收效果谓词 | 意图 + 禁忌集 + 偏好 → `plan`：解与预测状态，或无解与原因；`execute`：`ok` 或 `mismatch` | 求解器、执行、世界模型、谓词库 | 重试；在规划阶段改变物理状态；自行决定挪底盘、收手等补救 |
| 求解器 | 数值求解 | 数学问题 → 解或失败原因 | 机器人模型、规划世界 | 涉及任务语义；执行 |
| 执行 | 控制步、跟踪、仿真 | 轨迹 → 实测状态 | — | 做决策 |
| 谓词库 | 唯一的判据定义 | 状态 → 真假与测量值 | 世界状态 | 在各层重复定义 |

新能力按以下顺序判断归属：
1. 需要新的决策变量或约束，就是新的 primitive；
2. 是已有 primitive 的新组合或新的失败处理，归 skill；
3. 出现新的目标谓词，就在谓词库中加谓词，并加一个新的 skill；
4. 只涉及某个场景、某个任务的顺序或目标，写在专家脚本里。

### Primitive

一个 primitive 对应一个能单独求解、执行后能单独判定的问题，以它所解的问题命名。求解方式相同并不意味着是同一个问题：Grasp 和 Move 都要算 IK 和路径，但 Grasp 有两指对夹的约束，所以分成两个 primitive。

- `plan(状态, 意图, 禁忌集, 偏好)` 只求解，不改变物理状态。它按偏好顺序给出解，每个解附带预测的执行后状态；把已用过或被排除的解加进禁忌集再调用，就得到下一个解。无解时说明原因，即哪个约束排除了全部候选：IK 不可达、与某物体碰撞、区域里没有空位、开口不够等。
- `execute(解)` 执行到终止条件，等物体和机器人稳定，然后检查效果谓词。全部成立返回 `ok`；否则返回 `mismatch`，并附上不成立的谓词、实测值和预测值。
- 前置条件不成立时不执行。
- primitive 自己不重试。参数只在 primitive 内定义一份，hints 作为目标函数中的偏好传入。
- 持续存在的状态（例如正在按压的手）会进入后续问题的世界模型。
- 每次执行记录一个帧，内容是规划问题、执行区间、终止原因和验收结果。搜索树上的一条边就是一次 primitive 执行。
- 失败类型由平台判定：还没执行就失败的是规划失败，执行后效果不成立的是物理失败。

| Primitive | 数学问题（决策变量 / 约束） | 前置条件 | 判据：执行后成立的效果谓词 |
|---|---|---|---|
| Move | 到给定位姿或关节配置的路径 / 无碰撞 | — | TcpAt(解) |
| Grasp | 抓取位姿、接近方向、开口，以及到达路径 / 两指能对夹、开口足够、无碰撞 | HandEmpty | Grasped(obj)：没有确认抓空 |
| Carry | 带着物体先竖直抬起，再到 PlaceAt 给出的接近位姿 / 带载无碰撞 | Grasped 或 Held | Held(obj) ∧ ObjAt(obj, 预测位姿) |
| PlaceAt | 区域内的静止位姿与释放位姿，以及从接近位姿到释放位姿的带载下降路径 / 落差不超上限、在区域内、无碰撞、满足要求的姿态 | Held(obj) | Held(obj) ∧ ObjAt(obj, 释放前的预测位姿) |
| Release | 结束这只手的接触，包括抓空或滑脱后的松爪：张开宽度或抬离距离，以及撤离路径 / 不碰周围 | — | HandEmpty ∧ ¬Pressing；接在 PlaceAt 之后时另加 Resting(obj) 和 PlaceAt 区域上的关系（Inside 或 OnTop，以及要求的姿态）；松开把手时另加 JointAt 不变；撤按压时另加 Unchanged(柜体)；恢复过程中松开物体时只检查 Unchanged |
| Push | 平面推动：接触点、方向、距离 / 推动方向能让物体稳定跟随 | — | ObjAt(obj, 预测位姿, 宽容差) |
| Slide | 压住顶面拖动：接触点、压入深度、路径、终点 / 摩擦足以带动，物体不被压翻 | — | ObjAt(obj, 终点)，例如伸出桌边 |
| Articulate | 握持关节运动：手抓着把手，末端轨迹随关节运动（平动走直线，转动走圆弧），构成闭链 / 整段 IK 可解、内力小 | Grasped(把手) | JointAt(joint, 目标) ∧ Grasped(把手) |
| JointPush | 非握持推关节：在可动部件表面选接触点，沿关节方向单侧推，转动件的接触点沿表面滑动 / 无碰撞、在限位内 | — | JointAt(joint, 目标) |
| Press | 固定表面上的按压位置与到达路径，到达后保持按住 / 接触面合适 | HandEmpty | Pressing(hand, surface) |
| Drive | 底盘路线 / 无碰撞，持物时碰撞检查包含手里的物体 | TorsoAt(≥ 安全高度)，没有手在按压 | BaseAt(站位)；持物时另加 Held(obj) |
| Torso | 升降 / 过程中无碰撞 | — | TorsoAt(高度) |
| 全部 | | | 另加 Unchanged(动作区域附近的物体、柜体与其他关节) |

只在执行后判断，不在动作前用预测量否决。有些结果当场验证不了，由下一个 primitive 的效果来判：抓得牢不牢看 Carry 的 Held 和 ObjAt，松手后有没有放好看 Release 的 Resting 和区域关系。

### 谓词库

primitive 的前置条件和效果、skill 的目标、任务终态都用同一套谓词，参数取自规划的解或专家给的目标参数。

| 层次 | 谓词 |
|---|---|
| 机器人 | TcpAt(pose)、BaseAt(pose)、TorsoAt(h)、GripperAt(gap)、HandEmpty(arm) |
| 接触与持握 | Grasped(obj)、Held(obj)、Pressing(hand, surface) |
| 物体状态 | Resting(obj)、ObjAt(obj, pose, tol)、Unchanged(物体集合) |
| 关系与关节 | Inside(obj, container[, part])、OnTop(obj, surface)、Clear(region)、Upright(obj)、Flat(obj)、JointAt(joint, q)、Open、Closed、Aligned(objs) |

容差初值：

| 谓词 | 容差 |
|---|---|
| TcpAt、BaseAt | 20 mm / 5° |
| TorsoAt | 5 mm |
| GripperAt（张开） | 3 mm |
| Grasped | 只有确认抓空才不成立：合爪合到底，或者目标按几何位置不在两指之间。接触报告缺失（例如一侧指垫没有接触报告）、与预期接触宽度的偏差、合爪过程中物体的位移都只记录，交给提起后的实际随动来判；抓得牢不牢由 Carry 的 Held + ObjAt 判定 |
| Held + 带载 ObjAt | Carry、PlaceAt 执行完后物体位姿与预测相差 ≤20 mm / 10°，且随手移动。容差内的滑动只记录，后续规划以实测握持为准 |
| 放置后的区域关系 | 静止后在目标区域内，放进容器时即满足收纳谓词。与规划落点的偏差和倾斜只记录 |
| Upright、Flat | 仅在专家要求姿态时检查，倾斜 ≤15° |
| HandEmpty（松手后） | 两侧指垫无接触，物体不随手移动 |
| Open、Closed | 开：≥ 目标 − 20 mm；关：≤ 20 mm |
| ObjAt（摆正目标） | 20 mm / 5°。单次 Push 的效果容差更宽 |
| Unchanged：柜体与其他抽屉 | 柜体 20 mm / 5°，其他抽屉 ≤20 mm |
| Unchanged：其他物体 | 位移 ≤20 mm、转角 ≤10°。超出后，落到桌下或倒下才算失败，其余只记录 |
| Resting | 线速度 <5 mm/s、角速度 <5°/s，连续 0.2 s；最多等 2 s |

### Skill

一个 skill 对应一种目标谓词，负责在各种物体和关节类型上让这个谓词成立。物体和关节类型的差异由 skill 内部选择不同策略来处理，例如平动关节用直线 Articulate，转动关节用圆弧 Articulate 或 JointPush；远距离搬运就在 Put 里多加一段 Drive，Drive 之前需要的升躯干由 skill 显式调度 Torso。

skill 成功结束时目标谓词成立，手是空的，也没有手在按压。失败时先回滚到这次调用的入口状态，再把失败交给专家，所以失败出口与入口状态相同，不会带着持物或按压进入下一个 step。

| Skill | 目标谓词 | 组成 |
|---|---|---|
| Put(obj, into / on, …) | Inside 或 OnTop，物体静止，手空 | 见下方序列 |
| Open(joint, amount) | Open | 需要时先 Press，再 Grasp(把手) → Articulate → Release，可以分几次接力 |
| Close(joint) | Closed | 需要时先 Press，再 JointPush |
| PushTo(obj, pose) | ObjAt，20 mm / 5° | 多次 Push，每次测量 |

Put 的序列长度由联合规划决定，方括号内为可选段：

```
[Torso → Drive]              空手走到抓取站位
[Slide]                      薄物体先拖到可抓位置
Grasp
Carry                        抬起
[Carry → Torso → Drive → Torso]
                             收到运输位姿，带着物体换站位
[Carry → Grasp(另一只手) → Release(原来的手)]
                             换手，之后从新手继续
Carry                        到 PlaceAt 的接近位姿
PlaceAt
Release
```

抓取站位与放置站位相同时就是最短的 Grasp → Carry → PlaceAt → Release。每段 Carry 都按 Held ∧ ObjAt 验收，Drive 之后的 Carry 按实测握持重算接近位姿。

Pick、Place、Handover、Navigate 是 Put 内部的阶段，不对专家开放。

联合规划：skill 先在预测状态上依次调用各 primitive 的 `plan`，每个 `plan` 从前一个解的预测状态出发。后一个无解时，它的无解原因（例如这种抓法放不进目标区域）回传给前一个，前一个把导致这个原因的那类解加进禁忌集，再给出下一个解。全部有解后才开始执行。

执行中的更新：每段 primitive 执行完，skill 用实测的握持、机器人和物体状态，检查剩余计划的前置条件和预测状态是否仍然成立。成立就继续执行；不成立（例如物体在手里滑动后，原来的放置解不再可行）就从当前状态重算剩余计划，即下表的 ②。

失败处理：

| 层 | 失败时 | 失败类别 |
|---|---|---|
| primitive | 返回无解（附原因）或 `mismatch` | — |
| skill ① 原地换方案 | 在当前状态下换方案，失败过的候选进入禁忌集 | 执行后失败、原地处理成功计 a2 |
| skill ② 从当前状态重新规划 | 已执行一部分后，后面无解或某个 primitive 返回 `mismatch`（例如物体在手里滑出容差）时，从当前状态另找路线（按实测握持重算、换站位、换手） | 由 `mismatch` 触发且原地成功计 a2；只因规划无解另找路线不算失败 |
| skill ③ 回滚（仅仿真） | 原地无法恢复或预算用完时，回到这次 skill 的入口，保留禁忌集，换没试过的方案 | 规划触发计 a1，检查触发计 a3 |
| 专家 | 失败到达专家时世界已回到这次 skill 的入口状态；这个 step 放弃，后续 step 从这里继续；专家在下一轮迭代中修改这个 step 的代码 | b |

预算：
- 每次 skill 调用最多执行 3 个方案；只在规划阶段被排除的候选不占预算。
- 最多回滚 2 次；失败退出时回到入口的那次不计入，也不另计 a1/a3。
- wall time 上限 20 min。

抓取换方案按以下顺序：
1. 同一抓取族里换候选，排除与失败接触点距离 <2 cm 且接近方向夹角 <20° 的候选；
2. 换抓取族：整体 → 部件（把手、侧壁、边缘，按资产标注）→ 薄物体路线；
3. 换站位；
4. 换手。

放置换方案按以下顺序：
1. 同一区域内换落点；
2. 换姿态（目标参数允许时）；
3. 低处释放；
4. 换站位；
5. 换手。

失败类别的定义和读时分类见[专家数据](../specs/expert-data.md#step-内失败)。

### 专家与 step

一个 step 只调用一次 skill。step 的检查就是这个 skill 的目标谓词，子任务标注由 skill 按阶段自动生成。专家能给的信息有两类：

| 类别 | 例子 | 性质 |
|---|---|---|
| 目标参数 | 容器和部件、要求的姿态（`orientation='upright'` 或 `'flat'`）、相对关系、开度、必须空出的区域（`keep_clear`） | 硬约束，写进目标谓词并在验收时检查；不写姿态就不检查姿态 |
| `hints` | 抓取部位、接近方向、手臂、`near`、放置朝向、可以放弃的避让区（`avoid`） | 软约束，作为目标函数中的偏好；没满足不算失败，但记录是否满足 |

```python
STEPS = [
    Step(put('notebook_a5', into=('cabinet', 'drawer2'), orientation='flat'),
         hints=dict(grasp_part='spine', approach='top', arm='right')),
    Step(put('pen_black', into='cup', orientation='upright')),
    Step(close(('cabinet', 'drawer1'))),
]
```

`keep_clear` 和 `avoid` 共用同一份区域定义和 Clear(region) 谓词：前者写进目标谓词并验收，后者只作偏好。现有的 `keep_out` 在放置候选过滤和验收中都是硬约束，迁移为 `keep_clear`。

hints 一律相对物体或资产来写，用部位名、功能点或物体坐标系里的方向，不写世界坐标。资产知识（可抓部位、把手、功能点、`max_drop_m`）写在资产里；方案预算和容差属于平台配置。

现有技能不够用时可以临时越界，但只能写在专家脚本里，并在每处标记 `# 越界(<应归的层>): <原因>`。一轮开发期间平台冻结。每轮结束后按这些标记汇总成 skill 缺口清单，下一轮补进平台，再删除脚本中对应的越界代码。

## 依据

- 000100 第二遍的两份专家脚本共约 5000 行，大部分在替 skill 层干活：
  - 手写放置区域 53 个函数；
  - 运行时替换技能方法 11 处、私有导入 6 处；
  - `ctx.hold` 39 处、自设时间预算 24 处、手写判据 159 处；
  - 捕获异常后自行补救 18 处。
- 20 条 a3 中只有 11 条是真实物理失败。其余是三类问题：判得过早（实提 2 cm）、动作前用预测量否决（松手前 4 mm 门槛）、只看接触报告。另有失败类型被专家手写错。
- 回滚重试会选回同一个抓取候选，技能内部的原地恢复（a2）为 0。抓起来以后才发现放不下的规划回滚（a1）两条线共 28 次。
- 每层的职责参照了同类工作：
  - BEHAVIOR 的 action primitives 与 BDDL 目标谓词一一对应，并区分前置条件错误和后置条件错误；
  - RoboCasa365 按基础技能组织原子任务；
  - RoboTwin 2.0 用资产标注的接触点和功能点表达抓取偏好；
  - MoveIt Task Constructor 把规划拆成 stage，由 Alternatives、Fallbacks 容器组织备选与回退；
  - PDDL 算子用前置条件和效果描述动作。

## 适用边界

- 谓词和 primitive 直接读仿真状态。按传感器能力限制决策可用的信息（#116）是单独的工作。
- 回滚只用于现在的开发环境。以后作为 robot harness 由 agent 调用、并记录全轨迹时，需要禁止回滚，失败处理和失败出口届时另定。
- 不做跨 skill 的前瞻：step 的顺序由专家安排，顺序排错记 b。
- 不做多个身体部件的并行运动（#44）。
- 现阶段的 skill 只覆盖目标谓词 Inside、OnTop、Open、Closed 和摆正。转动关节、按钮与旋钮（Toggle、Turn）、擦拭倾倒等状态改变类操作，需要新的谓词、skill 或 primitive。
- 推动的结果难以精确预测，所以 Push 的效果容差放宽，是否到达目标由 PushTo 判断。
- 容差和预算都是初值，按回归结果调整。

## 未采用的方案

| 方案 | 原因 |
|---|---|
| 按求解形式归并 primitive（Reach / MoveStraight / Gripper / Drive / Torso） | 不同的约束被混在一起，判据无从定义 |
| Pick、Place 对专家开放 | 拿着物体再去放的前后衔接要专家自己保证，这正是规划回滚的主要来源 |
| 专家手写 chain 串联取放 | 不写就没有前瞻，前瞻是否存在取决于专家的写法 |
| primitive 内部重试 | 换哪种方案需要 skill 的上下文，在 primitive 内重试会重复选同一个候选 |
| 原子判据与目标谓词分开定义 | 同一个结果会有两套判据；效果谓词本身就能表达原子判据 |
| step 带预写的备选（失败后平台改试下一个） | 失败被运行时回退掩盖，step 的结果取决于走了哪条路；专家应看记录后直接修改这个 step |
| 一个 step 内包含多个 skill 并由专家写 `check` | 搜索树、检查点和 b 的计数单位对不上，判据也会散落在专家脚本里 |

参考：
- [OmniGibson action primitives](https://behavior.stanford.edu/reference/action_primitives/symbolic_semantic_action_primitives.html)
- [RoboCasa365](https://arxiv.org/abs/2603.04356)
- [RoboTwin 2.0 API](https://robotwin-platform.github.io/doc/usage/API.html)
- [MoveIt Task Constructor](https://docs.picknik.ai/concepts/moveit_task_constructor/)
