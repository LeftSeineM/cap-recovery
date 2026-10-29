# 实验台账

记录研究结论和关键实验结果：性能优化前后、多轮迭代的对比、方案对照。规范和手册只写当前状态，跨版本的结果留在这里，作为论文材料和复现依据。

台账不是开发时的硬性要求，需要时再从这些材料整理成条目，按下面的原则写。整理之前不随意清理相关的批次。

## 复现原则

每条结果写明产生它的 commit 和命令，在干净的检出上执行 `git checkout <commit> && bash -c '<命令>'` 即可复现。命令只用仓库内的相对路径，所需的资产和初态从该commit自行构建，不依赖 `assets/`、`runs/`、`outputs/` 中已有的文件。调度参数（`--jobs`、`--gpus`、`--launcher`）不影响结果，不写进命令。

- 引用的 commit 必须属于某个 PR，分支删除后仍可通过 `git fetch origin pull/<N>/head` 取回。只在本机或只在分支上、没有 PR 的 commit 不能写进台账，分支删除后它们可能被清理。
- `pull/<N>/head` 只指向 PR 最新的 commit，rebase 或 force push 后，原来的 commit 不再在它的历史里。所以分支上有 commit 写进台账以后，跟进 `main` 改用 `git merge main`。PR 用 squash 合并，分支上的 merge commit 不会进入 `main`。
- 运行报告带 `+dirty` 的结果不作为复现依据。
- 事后整理的结果，来源可能补不齐（报告记错了 commit、commit 未推送、候选在工作区里直接试、初态由未提交的代码构建）。commit 和命令根据能核实的填写，无法核实可不写。
- 规划有时限、wall time 受负载影响，重跑在统计意义上复现：写明 seed 范围与数量、硬件和负载；成功率尽量附 `--repeats` 的结果。
- 只写硬件型号与核数，不写内部机器名或地址：复现只需要同类硬件，不依赖某台机器。

## 条目

一个实验或一组相关实验一个条目，命名为 `<年月日>-<主题>`（如 `20261025-expert-wall-time`），日期是条目首次创建的日期。条目可以随产生结果的 PR 提交，也可以事后单独整理；同一主题的后续实验追加到同一条目，按时间排列，条目名不变。

- 只有表格时，条目是一个文件 `<年月日>-<主题>.md`。简单的折线图、柱状图可以用 Mermaid 的 `xychart-beta` 直接写在 md 里。
- 需要其他图时，条目是一个目录 `<年月日>-<主题>/`：`README.md` 写正文，结论和关键表格仍写在这里；`*.csv` 存画图用的数字（从批次 `summary.json` 中摘出的汇总值）；可选的 `plot.py` 从 csv 生成图，生成的 SVG 或 PNG 可以一起提交，在 README 中引用。图与 csv 不一致时以 csv 为准，重新生成图。
- 需要分析迭代过程时，条目可以附 `decisions.csv`，每个候选一行，被否定的也记：`candidate`（候选）、`mechanism`（假设的失败机制）、`evidence`（支持这个假设的失败案例与数据）、`change`（改了什么）、`result`（验证结果，引用结果表中的行）、`decision`（采用或否定，以及理由）、`tradeoff`（换来什么、代价是什么）。
- 只放汇总后的数字，单个文件不超过几百 KB；轨迹、录像和 `summary.json` 原件不放，它们按 commit 和命令重新生成。

```markdown
# 专家 wall time 优化

两三句话：这个条目解决什么问题，核心结论是什么，读者只看这一段也能知道条目在说什么。

关联：#43，PR #53

## 2026-10-24 规划世界去重

### 问题

一句话说明要回答什么。

### 结论

逐条写详细结论，附关键数据。

### 结果

| 版本 | commit（PR） | 场景 | seed | 成功率 | wall time 均值 (s) | CPU time 均值 (s) | 两版 CPU / wall 比 | 负载 / 核数均值 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 基线 | `<commit>`（main） | 000002 | 0:100 | … | … | … | … | … |
| 改动后 | `<commit>`（#53） | 000002 | 0:100 | … | … | … | … | … |

### 复现

    git checkout <commit>
    bash -c '<构建资产> && uv run python scripts/scenes/build_scene.py 000002 --seeds 0:100 && uv run python scripts/scenes/run_expert.py 000002 --physics isaac-cpu --dev --seeds 0:100'

### 环境

CPU 型号与核数、GPU、并发进程数。

### 备注

负载不可比、异常 seed 等需要注意的地方。
```

表中只列回答问题所需的指标。专家性能对比使用全部 seed 的 CPU time 和原始 wall time 算术均值及两版之比，负载 / 核数按 episode 均值等权汇总；逐条原值从 `episodes/<seed>.json` 读取。另列成功率、`unreachable_rate` 和耗时最多的案例；物理测试常用关键指标与 RTF（`physics_test/summary.json`）。

## 条目列表

- [20261027 专家 episode 单线程 BLAS](20261027-single-thread-blas.md)：Isaac 启动后 BLAS 按 32 线程运行，限制为 1 线程后进程 CPU time 降 48%–65%，wall time 不增加（#85）。
- [20261026 笔 000002 开启 CCD](20261026-pen-ccd-regression.md)：笔帽钢珠穿过抽屉薄底板、夹住抽屉的修复（#74），CPU time 约增加 3%，GPU 上 CCD 不生效。
- [20261026 资产抓取测试与专家耗时](20261026-asset-grasp/README.md)：同一套新测试下，一般尺寸资产首次抓稳从原策略的 74.5% 提高到 95.2%，首次抓取 CPU 30.7 → 132.1 s，#93 等性能优化后为 67.7 s，成功率不变；专家三个场景成功率相当；水瓶和辅助夹具仍慢；r1 和 #105 前 main 的消融见 #107。
- [20261025 场景 000002 专家迭代](20261025-expert-iteration-000002/README.md)：从 0 到 98% 的迭代历史，失败复现工具前后的对比。
- [20261024 专家 wall time 优化](20261024-expert-wall-time/README.md)：有界路径简化、碰撞粗筛、运动链 IK 等改进与逐项消融，000002 非 PhysX wall time −45.5%。
