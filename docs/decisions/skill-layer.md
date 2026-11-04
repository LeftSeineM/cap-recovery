# Skill 层：专家、skill、primitive 与谓词库

## 决定

专家程序之下分为 skill、primitive、solver 和执行四层，各层共用一个谓词库。依赖只向下，同层之间不互相调用。

| 层 | 做什么 | 不做什么 |
|---|---|---|
| 专家（如 `experts/<机器人>/<NNNN>.py` 中的 `STEPS`） | 任务分解：step 的顺序，每个 step 的目标、hints 和前瞻范围 | 写站位、坐标、等待、时间预算和判据；直接调用 primitive 或 solver |
| skill（如 Put、Open、Close、PushTo） | 按实测状态决定下一段做什么，组合 primitive，按前瞻范围规划，执行前检查与重接，验收目标谓词，记录搜索过程 | 构造几何或数学问题，发控制步，判定失败类别 |
| primitive（如 Grasp、Carry、PlaceAt、Release、Drive、JointPush） | 一个能单独求解、执行后能单独判定的问题：生成和排序自己的候选，规划，执行，等稳定，验收效果谓词 | 调用别的 primitive，自己重试，规划时改变物理状态 |
| solver（如 IK、碰撞检测、路径规划、夹爪几何、状态预测） | 数值计算：对象、附件和允许接触以参数传入，返回结果或失败原因 | 涉及任务语义 |
| 执行（如控制步、轨迹跟踪、物理仿真） | 按轨迹发控制并返回实测状态 | 做决策 |
| 谓词库（如 Grasped、Held、Inside、OnTop、Open、Closed、Resting、Unchanged） | 唯一的判据定义：给定实测状态，返回真假与测量值 | 在各层重复定义 |

判据：

- 判据只在谓词库中定义一份；primitive 的效果、skill 的目标和任务终态都用它，专家不写判据。
- 谓词只在实测状态上判定，以执行后的效果谓词为主。尽量不做执行前的检查，除非对比实验证实该检查的有效性，例如能够提前过滤掉50%的失败。

专家接口：

- 一个 step 只调用一次 skill。专家给三类信息：目标参数（硬约束，写进目标谓词并验收）、`hints`（偏好，相对物体或资产来写，不写世界坐标）、`lookahead`（`none`：只规划当前一段；`full`：执行前把剩余各段一起规划；`auto`：先 `none`，失败再 `full`）。
- skill 的行为默认值（例如是否扶柜）是平台默认，专家可以按场景覆盖。
- 现有技能不够用时可以在专家脚本里临时越界，每处标记 `# 越界(<应归的层>): <原因>`，下一轮补进平台后删除。

失败与记录：skill 失败时先回到调用入口，再把失败交给专家。skill 只记录事实（每次求解、检查、执行和实测谓词），a1、a2、a3、b 在读记录时推出，见[专家数据](../specs/expert-data.md#step-内失败)。

skill 内部如何编排 primitive（段、局部选项、搜索、缓存、预算）属于实现，写在 `table_1000/experts/skills/` 的代码和 docstring 里，不在本决议中约束。

## 验收与工作方法

- 以 `main` 上旧实现的行为为基线，先复刻、再改进。从旧实现迁移的机制，先在旧实现的运行记录里找到触发它的 seed，作为验收用例。
- 000002、000003 专家逐 step 验收：每个 step 20 个 seed，在同一入口 checkpoint 上与旧实现配对，成功率不低于旧实现，CPU time、sim time 均值不高于旧实现（口径见[实验与负载](../conventions/development.md#实验与负载)）。
- 资产抓取（000100、000103 的资产）首次抓稳成功率不低于旧实现。
- 专家脚本比旧实现短，不含站位、坐标、等待、预算和判据。
- 迭代按 step 进行，每轮实验控制在 10 min 内；skill层迭代需要修根因，不针对 seed 调参（见[开发原则](../conventions/development.md#修根因不调参)），专家层允许调参。

## 依据

- 000100 第二遍的两份专家脚本共约 5000 行，大部分在替 skill 层干活：手写判据 159 处、自设时间预算 24 处、运行时替换技能方法 11 处。
- 20 条物理回滚中只有 11 条是真实物理失败，其余来自判得过早、动作前用预测量否决、只看接触报告。
- 旧实现的许多行为只存在于代码里（放置对握法的约束、关抽屉只推不抓、松手后闭指撤离等），不以旧实现为基线逐 step 对照，迁移时就会丢失。

## 适用边界

- 谓词和 primitive 直接读仿真状态；按传感器能力限制决策可用的信息见 #116。
- 回到入口只用于开发环境。
- 不做跨 skill 的前瞻，step 的顺序由专家安排。
- 专家能要求的目标目前只有：放进或放到（Put：Inside、OnTop）、开关（Open、Close：Open、Closed）、推到指定位姿（PushTo：ObjAt）。TcpAt、Held 等谓词只作为 primitive 的效果和中间判据，不作为专家目标。关节只支持平动（抽屉），转动关节（门）还不支持。
- skill 内部编排的结构性重写另立决议。

## 未采用的方案

| 方案 | 原因 |
|---|---|
| Pick、Place 对专家开放，由专家手写 chain 串联 | 取放的衔接和前瞻要专家自己保证，这是规划回滚的主要来源 |
| primitive 内部重试 | 换哪种方案需要 skill 的上下文，在 primitive 内重试会重复选同一个候选 |
| 原子判据与目标谓词分开定义 | 同一个结果会有两套判据 |
| skill 在运行时记录失败类别 | 分类定义一改就要重跑；记录事实、读时分类可以随时改定义 |

参考：[MoveIt Task Constructor](https://docs.picknik.ai/concepts/moveit_task_constructor/)、[OmniGibson action primitives](https://behavior.stanford.edu/reference/action_primitives/symbolic_semantic_action_primitives.html)、[PDDLStream](https://github.com/caelan/pddlstream)。

## 附录：skill、primitive 与谓词清单

下面三张表列出现有的 skill、primitive 和谓词，具体参数与判定口径以代码为准。

### Skill

| Skill | 目标谓词 |
|---|---|
| Put(obj, into / on, …) | Inside 或 OnTop，物体静止，手空 |
| Open(joint, amount) | Open |
| Close(joint) | Closed |
| PushTo(obj, pose) | ObjAt，20 mm / 5° |

### Primitive

| Primitive | 数学问题（决策变量 / 约束） | 前置条件 | 判据：执行后成立的效果谓词 |
|---|---|---|---|
| Move | 到给定位姿或关节配置的路径 / 无碰撞 | — | TcpAt(解) |
| Grasp | 抓取位姿、接近方向、开口，以及到达路径 / 两指能对夹、开口足够、无碰撞 | HandEmpty | Grasped(obj) |
| Carry | 带着物体先竖直抬起，再到 PlaceAt 给出的接近位姿 / 带载无碰撞 | Grasped 或 Held | Held(obj) ∧ ObjAt(obj, 预测位姿) |
| PlaceAt | 区域内的静止位姿与释放位姿，以及从接近位姿到释放位姿的带载下降路径 / 落差不超上限、在区域内、无碰撞、满足要求的姿态 | Held(obj) | Held(obj) ∧ ObjAt(obj, 释放前的预测位姿) |
| Release | 结束这只手的接触，包括抓空或滑脱后的松爪：张开宽度或抬离距离，以及撤离路径 / 不碰周围 | — | HandEmpty ∧ ¬Pressing；接在 PlaceAt 之后时另加 Resting(obj) 和 PlaceAt 区域上的关系（Inside 或 OnTop，以及要求的姿态）；松开把手时另加 JointAt 不变；撤按压时另加 Unchanged(柜体)；恢复过程中松开物体时只检查 Unchanged |
| Push | 平面推动：接触点、方向、距离 / 推动方向能让物体稳定跟随 | — | ObjAt(obj, 预测位姿, 宽容差) |
| Slide | 压住顶面拖动：接触点、压入深度、路径、终点 / 摩擦足以带动，物体不被压翻 | — | ObjAt(obj, 终点)，例如伸出桌边 |
| Articulate | 握持关节运动：手抓着把手，末端轨迹随关节运动（平动走直线，转动走圆弧），构成闭链 / 整段 IK 可解、内力小 | Grasped(把手) | JointAt(joint, 目标) ∧ Grasped(把手) |
| JointPush | 非握持推关节：在可动部件表面选接触点，沿关节方向单侧推，转动件的接触点沿表面滑动 / 无碰撞、在限位内 | — | JointAt(joint, 目标) |
| Press | 固定表面上的按压位置与到达路径，到达后保持按住 / 接触面合适 | HandEmpty | Pressing(hand, surface) |
| Drive | 底盘路线 / 无碰撞，持物时碰撞检查包含手里的物体 | TorsoAt(≥ 安全高度)，没有手在按压 | BaseAt(站位)；持物时另加 Held(obj) |
| Torso | 升降 / 过程中无碰撞；持物升降时带载无碰撞 | — | TorsoAt(高度)；持物时另加 Held(obj) ∧ ObjAt(obj, 预测位姿) |
| 全部 | | | 另加：除声明可动者外全部 Unchanged，声明需维持的关系全部成立 |

PlaceAt 只选松手位置并带载下降到释放位姿，不松手；物体是否放好（Resting 与区域关系）由随后的 Release 判定。选松手位置和判定放置结果分在两个 primitive，失败时难以归因到松手位置。计划把二者合并为一个 Place primitive（决策变量为松手位姿、下降、张开与撤离，判据为手空、物体静止且在目标区域内），Release 只用于松开把手、撤按压、抓空后松爪和交接，另立决议。

### 谓词库

| 层次 | 谓词 |
|---|---|
| 机器人 | TcpAt(pose)、BaseAt(pose)、TorsoAt(h)、GripperAt(gap)、HandEmpty(arm) |
| 接触与持握 | Grasped(obj)、Held(obj)、Pressing(hand, surface) |
| 物体状态 | Resting(obj)、ObjAt(obj, pose, tol)、Unchanged(物体集合) |
| 关系与关节 | Inside(obj, container[, part])、OnTop(obj, surface)、Clear(region)、Upright(obj)、Flat(obj)、JointAt(joint, q)、Open、Closed、Aligned(objs) |
