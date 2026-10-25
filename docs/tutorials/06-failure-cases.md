# 06 失败分析

专家批次里失败的 seed：看原因，判断是偶然还是结构性的，复现原来的失败，确认根因后修复。一个案例由“批次、步骤、seed”确定，开发运行都在 `runs/expert/<场景>/` 下；记录的格式见[专家数据](../specs/expert-data.md)。命令的其他参数用 `--help` 查看。

## 1. 看失败的案例

```bash
uv run python scripts/experts/failure_cases.py <批次>
```

为每个失败的 seed 渲染失败子任务开始时和轨迹结束时的多视角图片，写到批次下的 `cases/<seed>/`。**看什么**：图片能否说明原因（够不着、被挡住、没夹住）；`case.json` 中规划放弃各个选项的原因。录像用 `render.py --batch <批次> --failed`。

## 2. 偶然还是结构性的

```bash
uv run python scripts/scenes/run_expert.py 000002 --physics isaac-cpu --dev --steps 2 --from <批次> --seeds 4 44 --repeats 10
```

从前一步的检查点把失败的那一步重跑 10 次。**看什么**：`summary.json` 的 `repeats` 中每个 seed 的成功次数。几乎都成功的是偶然失败；几乎都失败的是结构性失败，要找根因。

## 3. 复现原来的失败

从检查点重跑时，失败之前的决策和随机数都变了，失败不一定再出现。要复现，从失败的那次技能调用开始：

```bash
uv run python scripts/experts/failure_snapshots.py <批次> --seeds 4 44
uv run python scripts/scenes/run_expert.py 000002 --physics isaac-cpu --dev --from <快照批次> --steps 2 --repeats 10
```

第 1 次用原来的随机数，后面换随机数。**看什么**：`episodes/<seed>.json` 的 `replay` 中 `same_frame`、`same_reason` 为真，说明在同一处、因同样的原因失败；`repeats` 给出换随机数后的成功次数。

## 4. 离线重解

```bash
uv run python scripts/experts/resolve.py <批次> --seeds 40 --frame 32 --redecide --runs 10
```

不启动仿真，用记录下的问题重新求解 IK 和规划，重新执行该帧最后一个候选决策。它不仿真抓取和放置。**看什么**：`runs/resolve/` 下 `summary.json` 汇总每个 seed 的成功次数与原因；每个候选在哪一关被淘汰、重选的每次结果在 `<seed>.json.gz` 的 `candidates`、`redecide.runs`。

它读报告 `solve_trace` 指向的求解记录。失败的运行默认就有；为 null 时（旧批次、`--problems none`），先按第 3 节生成快照，加 `--problems all` 重跑那一步补采，再对新批次离线重解。

## 5. 修复与验证

1. 每次只改一个因素，先在原失败的入口上配对重跑（原代码与修改各几次）；
2. 再跑完整批次，seed 数按成功率选取（见[开发原则](../conventions/development.md#seed-数随成功率增加)）；
3. 最后在没参与修复的一组新初态上验证（`build_scene.py --output` 写到 `runs/` 下，`run_expert.py --input` 指向它）。

修复之后可以用 `--skill-attempts 1/2/3` 各跑一批，比较有限重试的效果（见[有限重试](../specs/experts.md#有限重试)）。

## 解不了时

人工接手这一步，见[第 7 章](07-teleop.md)。
