# 规范

本目录规定 Table-1000 资产（对象、机器人、场景）与专家程序的文件、接口和格式，供编写代码和资产的 agent 阅读。命令与人工验收见[操作手册](../tutorials/README.md)。

| 规范 | 内容 |
| --- | --- |
| [建模](modeling.md) | `object.py`：坐标、刚体与部件、命名、关节 |
| [几何预览](preview.md) | `preview.json`：姿态、相机、关键帧与自动检查 |
| [物理属性](physics.md) | `physics.py` 与单刚体资产的物理配置 |
| [行为插件](behaviors.md) | `behavior.py`：原生物理表达不了的附加力或状态转换 |
| [物理测试](physics-tests.md) | `physics_test.py`：测试接口、结果与 RTF 基准 |
| [资产抓取测试](grasp-tests.md) | 首次抓稳、持物动作验收与结果格式 |
| [薄物体抓取](thin-grasp.md) | 片状与细线物体的分流、移边准备与验收指标 |
| [机器人](robots.md) | 机器人源码、碰撞与物理、机器人层、验收用例 |
| [场景](scenes.md) | `scene.py`：随机初始化、布局、初态检查 |
| [专家](experts.md) | `experts/<机器人>/<NNNN>.py`：文件与适用范围、分层、编写规则、步骤、重试与失败恢复 |
| [专家数据](expert-data.md) | 批次、轨迹、报告、检查点、调用帧与求解记录 |

本页说明每个文件放在哪里，以及资产共用的单位、名称表示和 README。

## 资产布局

顶层目录的用途见[仓库与目录](../conventions/repository.md#目录)。`assets/isaac/` 是 Isaac Sim 5.1 运行时取用的在线资产（地面等）的本地镜像，由 `scripts/assets/download_isaac_assets.py` 从 NVIDIA 资产服务器下载，许可同 Isaac Sim；目录存在时仿真从这里加载，不再联网。

同一资产在 `asset_sources/` 与 `assets/` 下使用相同的相对路径：对象 `objects/<category>/<六位编号>`、机器人 `robots/<名称>`、场景 `scenes/<六位编号>`；编号从 `000000` 开始。下面按资产类型列出每个文件，括号内为规范文档。

### 对象

```text
asset_sources/objects/cabinet/000000/
  object.py           建模入口：坐标系、刚体子树、视觉与碰撞、关节（建模）
  preview.json        几何姿态、相机、关键帧与预览输出（几何预览）
  physics.py          精细物理资产：质量、接触材料、驱动（物理属性）；单刚体资产不写，数值在 configs/rigid_physics.json
  physics_test.py     精细物理资产：物理实验的初态、动作与记录（物理测试）
  behavior.py         可选，随资产加载的附加力或状态转换（行为插件）
  metadata.json       UUID，见下文
  README.md           外观、尺寸、资产树、物理实测，见下文
  three_quarter.jpg   480 × 240 缩略图
assets/objects/cabinet/000000/
  object.blend        object.py 构建的模型
  geometry.usdc       导出的 USD 几何层
  physics.py          所用物理脚本的快照：资产自己的，或由 configs/rigid_physics.json 生成
  physics.usda        physics.py 生成的物理层
  object.usdz         加载入口；带行为时另有 object.usda 与行为依赖
  *.py、*.json、README.md、three_quarter.jpg    源码快照
  preview/            第一阶段验收：几何检查 acceptance.json、图片与视频
  physics_test/       第二阶段验收：summary.json，isaac-cpu/、isaac-gpu/ 各后端结果
  grasp_test/<机器人>/<后端>/    当前一份抓取验收结果，正式运行整体替换，不保留运行历史
    summary.json      首次抓稳验收、单列握法偏好数据、配置与协议、来源运行名及提交、选定录像 seed
    trials/<seed>.json             每个 seed 的报告
    trajectories/<seed>.hdf5       全部 seed 的场景状态，支持逐 seed 重渲染
    videos/<选定 seed>.mp4         每资产一条录像：首个失败 seed，否则 seed 0
```

### 机器人

```text
asset_sources/robots/x1pro/
  robot.urdf、meshes/ 转换后的 URDF 与视觉、碰撞网格（机器人）
  prepare.py          从厂商 URDF 包选取配置、划分碰撞
  physics.py          驱动、求解器与接触材料
  robot.py            机器人描述：关节、末端、相机
  robot_test.py       验收设置
  README.md
  three_quarter.jpg   1920 × 960 缩略图
asset_sources/robots/x1pro_rm001/    同一台机器人换 rm001 夹爪：只有夹爪网格，其余网格与 .py 引用 x1pro
assets/robots/x1pro/
  raw/                厂商原始 URDF 包，存档于 Hugging Face 数据集，版本见 assets.lock.json
  import/             Isaac Sim URDF 导入结果
  robot.usda          加载入口
  prepare.json、check.json、源码快照
```

### 场景

```text
asset_sources/scenes/000000/
  scene.py            桌子、支持的机器人与默认机器人、物体、录像视角、随机初始化（布局 0）、可选的其他布局（场景）
  experts/<机器人>/<NNNN>.py    专家程序，按机器人分目录、4 位编号，文件内声明适用范围（专家）；可以以后补
  experts/common.py   可选，本场景各专家共用的函数
  thumbnail.jpg       默认机器人最小有效 seed 初态的主相机图像，480 × 270
assets/scenes/000000/<机器人>/   build_scene.py 的产出，按机器人分开（--robot，all 为全部），与后端无关
  initial_states.hdf5 每个有效 seed 静置后的场景状态，专家轨迹从这里开始
  initial.json        初态检查：汇总与每次尝试的结果
  initial_states.jpg  可选，前 N × M 个有效初态的拼图
  initial/<seed>.jpg  可选，每个有效初态的相机原图
  layouts/<n>/        --layouts：布局 n 的同一套文件（initial.json 另记布局的好坏与说明）
  layouts.jpg         可选，多个布局时每个布局最小有效 seed 的拼图
data/scenes/000000/<机器人>/<专家编号>/<后端>/   场景的数据：一个专家的一批轨迹，run_expert.py 生成（范围内所有 seed、所有步骤），render.py 补上录像；layout n 在其 layouts/<n>/
  summary.json        验收入口：提交、seed、成功率、失败的 seed、耗时、机器负载、有录像的 seed
  trajectories.hdf5、trajectories_failed.hdf5    成功与失败的轨迹，每条 data/demo_<seed>
  episodes/<seed>.json                           每条轨迹的报告
  calls/<seed>[-<r>].json.gz                      每次运行的专家调用帧树（gzip JSON），r 为后续 repeat 序号
  problems/<seed>[-<r>].json.gz                   决策候选、筛选关卡与完整数值求解问题（gzip JSON）；默认只保存失败或程序错误，--problems all/none 改为全部/不保存
  attempts/<seed[-r]>/<尝试编号>/                 回滚丢弃的失败分支，保留同格式的完整批次；主报告 retry_archives 给出相对路径
  videos/<seed>/<相机>.mp4                       成功轨迹的录像，每个相机一条；--merge 时为 videos/<seed>/merge.mp4
  fails/                                         失败轨迹的录像，结构同 videos/
  checkpoints/step<k>.hdf5                       由步骤组成的专家：每个 seed 第 k 步结束时的场景状态，step0 为初态
runs/expert/000000/<机器人>/<专家编号>/<运行>/   一次开发运行（run_expert.py --dev）：结构同上，以后的运行不会替换它；
                                  <运行> = <日期>-<时间>-<提交>-<机器人>-<专家编号>-layout<n>[-<标签>]；
                                  上一轮开发线改名为 <专家编号>-legacy-<MMDD>/；.iteration/ 为浏览站点的派生文件
  cases/<seed>/                   可选，failure_cases.py 渲染的失败案例
data/cases/000000/<机器人>/<专家编号>/<用例名>/   长期用例：只含一个检查点的最小批次（summary.json、checkpoints/step<k>.hdf5），可直接作为 --from；failure_snapshots.py --case 写入
runs/resolve/000000/<运行>/       离线数值重解（scripts/experts/resolve.py），运行名同专家开发运行；
  summary.json                    JSON 列表，汇总各 seed 的重解结果，不是专家批次的验收入口
  <seed>[-<r>].json.gz             详细重解结果，planning-* 为临时规划模型
runs/teleop/000000/<运行>/        人工遥操一个案例的一步（scripts/experts/teleop.py）：结构同上，只有一个 seed；
                                  <运行> = <日期>-<时间>-<提交>-step<k>-seed<s>
```

```text
runs/grasp/<日期>-<时间>-<提交>[-<标签>]/   asset_grasp.py --dev [--name 标签]
  conditions.json、summary.json、summary.md  配置、协议、来源提交与批次汇总
  dirty.patch                    仅工作区有未提交改动时保存；不复制源码目录
  workers/                       工作进程日志
  objects/<category>/<编号>/     summary.json、trials/、trajectories/；仅要求时有 videos/
```

统一单位：m、kg、s、rad、N、N·m；四元数顺序为 `[w, qx, qy, qz]`。各文件另行注明坐标系。USD 原生属性遵循各 Schema 的单位，详见[物理属性](physics.md)。源码快照重建依赖同版本的 `table_1000` 包，共用模块不复制到资产目录。外部资源引用须注明版本、来源和许可，不使用私有绝对路径。

## 名称压缩表示

资产树与项目公共名称解析工具支持 `name1..N` 表示法：`cabinet.drawer1..3` 展开为 `cabinet.drawer1`、`cabinet.drawer2`、`cabinet.drawer3`，包含首尾。范围后可接名称后缀，例如 `cabinet.drawer1..3.handle`。

- 每个名称最多一个递增整数范围，不支持通配符或正则表达式；补零编号须保持相同宽度，例如 `collision00..15`。
- 配置展开后的名称必须存在且符合引用类型。同一字段内不得重复引用同一对象；需要不同配置时拆开范围单独填写。
- 资产树自动压缩仅合并同父级、名称仅连续显式编号不同、角色/碰撞类型/材质及归一化子树结构相同的对象；`.001` 自动后缀不参与归并。不按外观猜测同类部件。手写配置范围不要求各对象几何相同。

压缩只影响文字展示或配置书写，模型中的对象仍各自使用完整名称，实际层级不变。需要范围展开的脚本共用公共解析工具，不处理动作名、文件名等普通字符串。USDA 使用原生 USD 语法和实际 prim 路径，不直接支持此缩写。

## README 与缩略图

`README.md` 包含 `![preview](three_quarter.jpg)`、简短外观与尺寸说明、资产树及刚体数；有关节时说明类型、限位和方向。部件名称遵循[建模规范](modeling.md#部件命名与资产树)。

资产树用 `[rigid]` 标注刚体根，子节点可使用局部名称，重复组件可用[名称压缩表示](#名称压缩表示)，例如：

```text
cabinet
  housing [rigid]
    rubber_pad1..4
  drawer1..3 [rigid]
    handle
```

资产树展示所有刚体及其部件，省略部件下的视觉与碰撞网格；使用空格缩进表示实际父子层级，森林保留多个根节点。

带 `physics_test.py` 的资产另列“物理实测”表，记录各测试测得的关键物理量，如拔出力、合紧力、起滑角：每行写明物理量、CPU 与 GPU 实测值、来源测试及测量方式。数值取自 `physics_test/summary.json`，注明软件版本，并标明为未标定的仿真值；修改几何、`physics.py` 或测试后重新运行并更新。已知后端差异在表下说明或链接 issue。

同一节另列典型条件下的性能：CPU 后端 1 个环境、GPU 后端 4 个并行环境，分别给出无渲染和含视频的 RTF 范围，以及测量期间的本进程 CPU 占用和 GPU 利用率。数值取自 `--benchmark` 运行的 `result.json` 中的 `performance`，注明机器负载等条件，测量方法见 [RTF 基准](physics-tests.md#rtf-基准)。

第一阶段的[连续变形几何](modeling.md#第一阶段连续变形几何)另以 `[deformable]` 标注归属节点，README 单列其数量、控制量和仅几何预览的限制；不得计入刚体数或宣称物理导出已支持。仍使用上述五种第一阶段源文件，几何回调放在 `object.py` 中，随源码快照分发。

缩略图都是四分之三视角，左视觉、右碰撞（机器人的碰撞凸块按刚体着色）：对象 `three_quarter.jpg` 为 480 × 240，约 5–10 KB；机器人 `three_quarter.jpg` 为 1920 × 960（左右各 960 × 960），约 40 KB，数量少、结构复杂，用较大尺寸。场景缩略图 `thumbnail.jpg` 是最小有效 seed 初态的机器人主相机图像（X1Pro 为头部相机），480 × 270，约 10–20 KB。

## metadata.json

现有格式：

```json
{"uuid": "<object-uuid>"}
```

UUID 的用途、修改规则及资产版本/发布机制尚未确定；此处仅记录现有格式。
