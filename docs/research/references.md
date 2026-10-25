# 技术参考与路线选择

核查日期：2026-09-22。核查方法为官网、官方仓库文档与部分源码静态检查，尚未在新项目安装或运行。本文的推荐是项目判断，作者公布的性能不等于本项目已经复现。

## 1. 当前建议

先核查 **RoboDojo X1Pro 的机器人、相机、动作与策略桥接**，并把 **SimFoundry 作为单图资产/场景生成能力的候选参考**。W2 根据同一批照片的生成、编辑和交互结果选一个主执行后端；不预设 OmniGibson 与 RoboDojo 的 USD 场景可以直接互换。

组内同学用 GPT-6 根据一张照片搭场景，是值得复现的开发方式。需要把其中可重复的步骤收敛为全自动工具链，并测量每张照片的成功率、成本和失败类型。当前公开示例本身还不足以证明任意单张照片都能自动批量转换；这正是一期必须验证的交付，而不是可以由人工建模补足的隐含步骤。

## 2. SimFoundry

固定源码版本：[`9e34ebefcd020583fbb755a8b57268dce78eca26`](https://github.com/NVlabs/SimFoundry/tree/9e34ebefcd020583fbb755a8b57268dce78eca26)。

[官方项目页](https://research.nvidia.com/labs/gear/simfoundry/)展示从真实视频得到可交互场景、数字 cousin，以及 sim–real 评价和训练实验。对本项目最有价值的是资产/场景生成分阶段设计与物体级产物；公开展示的结果不能外推到 X1Pro 开放整理，也不能直接证明一期要求的单图生成、编辑和交互能力。

[pipeline 文档](https://github.com/NVlabs/SimFoundry/blob/9e34ebefcd020583fbb755a8b57268dce78eca26/scripts/pipeline/README.md)以视频/ZED 为主入口，后续阶段使用选定的 canonical frame。其现有入口若需要视频或 ZED，不能直接作为一期单图交付；只能作为候选组件或诊断对照，直到补齐真正的单图全自动入口。

根 README 的发布说明仍写数据生成/训练尚未开放，但当前源码已存在 [CuRobo demo generation](https://github.com/NVlabs/SimFoundry/blob/9e34ebefcd020583fbb755a8b57268dce78eca26/scripts/pipeline/C_application/stages/5_generate_demos.py)、teleop/replay 和 [policy evaluation](https://github.com/NVlabs/SimFoundry/blob/9e34ebefcd020583fbb755a8b57268dce78eca26/scripts/pipeline/C_application/stages/1_eval_policy_og_scene.py) 入口。应逐项验证可运行性，既不假定整链可用，也不能依据旧说明断言没有轨迹生成源码。现有策略路径包含 DROID 专用假设，X1Pro 仍需适配。

[物理参数生成代码](https://github.com/NVlabs/SimFoundry/blob/9e34ebefcd020583fbb755a8b57268dce78eca26/scripts/pipeline/A_reconstruction/stages/11_make_objects_sim_ready.py)使用模型推断质量、摩擦等参数，属于估计；稳定化通过不等于真实物理校准。[安装说明](https://github.com/NVlabs/SimFoundry/blob/9e34ebefcd020583fbb755a8b57268dce78eca26/docs/INSTALL.md)给出约 250GB 完整安装需求和低显存设置。默认 shape generation 的显存需求高于 24GB，4090 路径需验证 offload 与吞吐量；HF 权限及默认 Gemini/Vertex 服务也需单独核实，GPT API 额度不能直接替代这些依赖。

## 3. RoboDojo X1Pro

固定源码版本：[`564b0935ccece118b16aa786d6dc55ba304f3853`](https://github.com/ppppplus/robodojo-x1pro/tree/564b0935ccece118b16aa786d6dc55ba304f3853)。

[X1Pro 文档](https://github.com/ppppplus/robodojo-x1pro/blob/564b0935ccece118b16aa786d6dc55ba304f3853/x1pro/README.md)提供固定底座双臂、FX001/RM001 夹爪、机载相机和真实轨迹回放的适配。机器人 USD/mesh、上游资产和策略权重独立分发，需要检查现场机型和可取得性。

[照片工作站构建代码](https://github.com/ppppplus/robodojo-x1pro/blob/564b0935ccece118b16aa786d6dc55ba304f3853/x1pro/noodle_scene/noodle_expert/scene_builder.py)直接定义物体几何和位置；它是可借鉴的特定场景实现，尚不是通用照片上传到物理场景的产品化入口。[采集代码](https://github.com/ppppplus/robodojo-x1pro/blob/564b0935ccece118b16aa786d6dc55ba304f3853/x1pro/noodle_scene/noodle_expert/collect.py)包含估计的接触参数与简化对象；这些必须与真实测量对照。

[OpenPI bridge 文档](https://github.com/ppppplus/robodojo-x1pro/blob/564b0935ccece118b16aa786d6dc55ba304f3853/x1pro/noodle_scene/noodle_expert/README.md)给出三路 RGB、状态历史与 15Hz 控制等特定 checkpoint 合同。完成 rollout/replay 只代表指定动作帧执行完毕，不是任务成功。现场数据是否能直接使用该 checkpoint 待验证。

[验证记录](https://github.com/ppppplus/robodojo-x1pro/blob/564b0935ccece118b16aa786d6dc55ba304f3853/x1pro/VALIDATION.md)仍有独立环境启动/渲染待完成项。首两周必须实际跑通干净部署、新场景抓放和真实轨迹对齐，不能以源码存在代替验收。代码根许可与部分文档许可描述有差异，机器人和第三方资产的再分发条件分别核查。

## 4. TableVerse 对 scene 计数的启示

[TableVerse 论文](https://arxiv.org/abs/2607.21017)把 TableVerse-100K 描述为 100,000 个 unique、物理一致的 tabletop environments；[项目页](https://bytedance.github.io/TableVerse/)说明其从单张真实桌面图像恢复可交互场景，再生成 pick-and-place demonstrations。[数据卡](https://huggingface.co/datasets/ByteDance/TableVerse)进一步把数据按 `scenes/<scene_uid>/` 组织，每个 scene 目录包含资产、`scene.glb` 和 XML 场景文件，并报告约 100k environments、约 1M object instances 和 4.07TB 数据。

因此，本项目采用与它一致的基本计数直觉：**一个 scene 是一个固定的可交互环境包**，而不是一个随机 seed、一个 episode 或一条 trajectory。TableVerse 的公开摘要和数据卡没有给出“每个 scene 固定多少条 trajectory”的统一数字；它只说明环境与操作轨迹配对，不能据此推断每个 scene 有 100 条轨迹。

TableVerse 的 100k 主要证明自动化 Real2Sim 能够大量生产环境包；它不等于 100k 个独立的人类多终态整理场景，也不自动提供本项目要求的人类终态、Open 整理目标或 X1Pro 真机证据。Table-50 因而定义为 50 个 scene package，每个 scene 再生成 100 个受约束随机 initialization 和对应轨迹，另行报告人工目标和真机子集。

## 5. 被评模型

| 对象 | 已核查的依据 | 本项目需要验证 |
|---|---|---|
| GPT-5.6 | [官方模型页](https://developers.openai.com/api/docs/models/gpt-5.6-sol)列出 `gpt-5.6-sol`；可作为用户所指 GPT-5.6 的拟议固定名称 | 账户权限、固定 snapshot 或响应版本、图像/结构化输出、预算与限流 |
| GPT-6 | [官方模型页](https://developers.openai.com/api/docs/models/gpt-6-astra)列出 `gpt-6-astra` | 同上；不把当前 Codex 会话模型身份等同于用户 API 账户权限 |
| openpi | [官方仓库](https://github.com/Physical-Intelligence/openpi)提供 π 系列 checkpoint 与训练/推理；是框架名称 | 固定具体 π0/π0.5 等 checkpoint、X1Pro adapter 和训练来源 |
| OpenDM | [Dexmal 仓库](https://github.com/dexmal/opendm)提供 DM0.5 及推理/训练流程 | 固定 checkpoint；官方 ARX5/RoboDojo 设置不是 X1Pro 设置 |

openpi [官方 README](https://github.com/Physical-Intelligence/openpi)估计推理显存 >8GB、LoRA >22.5GB、全量微调 >70GB，并指出 PyTorch 与 JAX 路径的功能差异。4090 上的策略、渲染器和重建器不应默认并发驻留。A100 的 40/80GB 型号决定全量微调是否可行，本期优先最小适配。[远程推理说明](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md)可用于分离策略服务与机器人客户端，但必须测端到端时延。

OpenDM [机器人定义](https://github.com/dexmal/opendm/blob/main/opendm/constants/robot.py)本次核查未列 X1Pro。[推理接口](https://github.com/dexmal/opendm/blob/main/docs/en/dm05_inference.md)中的相机、状态、动作维数和归一化需与 checkpoint 匹配。[数据格式](https://github.com/dexmal/opendm/blob/main/docs/en/data.md)允许由未来 state 构造 action，本项目仍需保留真实 command/action/state 及各自时间戳，避免跟随延迟混淆监督。[LoRA 示例](https://github.com/dexmal/opendm/blob/main/docs/en/dm05_so101_lora_training.md)不能直接证明 X1Pro 单卡训练成本；W2 做小规模 profiling，再冻结预算。

## 6. W2 路线选择表

在同 2–3 个新场景上记录以下信息，再选主路径：

| 维度 | 必须回答的问题 |
|---|---|
| 场景生成输入 | 是否只用一张 RGB 照片？是否偷偷依赖深度、短视频、额外测量或人工建模？ |
| 运行时交互输入 | policy/遥操是否使用仿真渲染或真机传感器的实时观测和状态，而不是把来源照片当作执行观测？ |
| 编辑能力 | 是否能在编辑器直接修改、保存、重载场景；编辑过程中是否允许暂不满足物理约束？ |
| 机器人 | 实际 X1Pro 的夹爪、坐标、相机、频率和动作模式是否匹配？ |
| 物理 | reset 稳定吗？完整抓放有真实接触吗？是否存在瞬移或伪抓取？ |
| 产物 | 资产与轨迹能否版本化、重跑、记录失败并导出给模型？ |
| 成本 | GPU·h、显存、API 费用、人工修正分钟与成功率？ |
| 可持续性 | 资产/权重能否取得，部署是否可复现，许可和共享资源是否支持一期？ |

本文件保存项目长期需要的结论和一手链接；代码版本变化后需要重新核验。
