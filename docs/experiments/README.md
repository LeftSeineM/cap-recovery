# 实验台账

记录研究结论和关键实验结果：性能优化前后、多轮迭代的对比、方案对照。规范和手册只写当前状态，跨版本的结果留在这里，作为论文材料和复现依据。

## 复现原则

每条结果写明产生它的 commit 和完整命令，检出这个 commit、运行这条命令即可复现。`runs/`、`outputs/` 中的文件可以随时清理，台账不依赖它们。

- 引用的 commit 必须属于某个 PR：PR 合并、关闭或未合并都可以，分支删除后仍可通过 `git fetch origin pull/<N>/head` 取回。不合入 `main` 的实验也开一个 PR（可以是草稿，结束后关闭）。只在本机或只在分支上、没有 PR 的 commit 不能写进台账，分支删除后它们可能被清理。
- `pull/<N>/head` 只指向 PR 最新的 commit，rebase 或 force push 后，原来的 commit 不再在它的历史里。所以分支上有 commit 写进台账以后，跟进 `main` 改用 `git merge main`。PR 用 squash 合并，分支上的 merge commit 不会进入 `main`。
- 运行报告带 `+dirty` 的结果不写进台账。
- 引用规则建立之前的结果，代码来源可能补不齐（报告记错了 commit、commit 未推送）。这类结果标为历史观察，与可复现的结果分开，写明依据，不作为复现依据。
- 规划有时限、wall time 受负载影响，重跑在统计意义上复现：写明 seed 范围与数量、硬件和负载；成功率尽量附 `--repeats` 的结果。
- 只写硬件型号与核数，不写内部机器名或地址：复现只需要同类硬件，不依赖某台机器。

## 条目

一个实验或一组相关实验一个条目，命名为 `<年月日>-<主题>`（如 `20261025-expert-wall-time`），日期是条目首次创建的日期。条目随产生结果的 PR 一起提交；同一主题的后续实验追加到同一条目，按时间排列，条目名不变。

- 只有表格时，条目是一个文件 `<年月日>-<主题>.md`。简单的折线图、柱状图可以用 Mermaid 的 `xychart-beta` 直接写在 md 里。
- 需要其他图时，条目是一个目录 `<年月日>-<主题>/`：`README.md` 写正文，结论和关键表格仍写在这里；`*.csv` 存画图用的数字（从批次 `summary.json` 中摘出的汇总值）；可选的 `plot.py` 从 csv 生成图，生成的 SVG 或 PNG 可以一起提交，在 README 中引用。图与 csv 不一致时以 csv 为准，重新生成图。
- 只放汇总后的数字，单个文件不超过几百 KB；轨迹、录像和 `summary.json` 原件不放，它们按 commit 和命令重新生成。

```markdown
# 专家 wall time 优化

关联：#43，PR #53

## 2026-10-24 规划世界去重

### 问题

一句话说明要回答什么。

### 结论

一两句结论。

### 结果

| 版本 | commit（PR） | 场景 | seed | 成功率 | 每条 wall time (s) | sim time (s) | CPU time / wall time | 负载 / 核数 p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 基线 | `<commit>`（main） | 000002 | 0:100 | … | … | … | … | … |
| 改动后 | `<commit>`（#53） | 000002 | 0:100 | … | … | … | … | … |

### 复现

    uv run python scripts/scenes/run_expert.py 000002 --physics isaac-cpu --dev --seeds 0:100 --jobs 20

### 环境

CPU 型号与核数、GPU、并发进程数。

### 备注

负载不可比、异常 seed 等需要注意的地方。
```

表中只列回答问题所需的指标。专家批次常用：成功率、`unreachable_rate`、每条轨迹的平均 sim time 与 wall time、CPU time / wall time 与负载分位数（均在批次的 `summary.json`）；物理测试常用：关键指标与 RTF（`physics_test/summary.json`）。

## 条目列表

- [20261026 笔 CPU CCD 回归](20261026-pen-ccd-regression.md)：薄板 12/12；000002 出现 1 条弱持握回退，000003 保持 19/20，000000 提高到 20/20。
- [20261025 场景 000002 专家迭代](20261025-expert-iteration-000002/README.md)：从 0 到 98% 的迭代历史，失败复现工具前后的对比。
