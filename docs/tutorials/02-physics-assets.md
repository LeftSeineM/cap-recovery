# 02 物理资产与物理测试

为第 1 章的模型配置质量、摩擦和关节驱动，导出 USD 物理资产，用 CPU 和 GPU 两种物理后端测试。以[带盖笔 pen/000002](../../asset_sources/objects/pen/000002/) 和[三抽屉盒 cabinet/000000](../../asset_sources/objects/cabinet/000000/) 为例。

## 交付物

| 交付物 | 位置 | 内容 |
| --- | --- | --- |
| 物理源码（进 Git） | `asset_sources/objects/<类别>/<编号>/` | `physics.py`（质量、接触材料、关节摩擦与驱动，见[规范](../specs/physics.md)）、`physics_test.py`（物理测试，见[规范](../specs/physics-tests.md)）；README 的“物理实测”一节记录关键指标和性能 |
| 物理资产 | `assets/objects/<类别>/<编号>/` | `object.usdz`（仿真加载的资产）、`physics_build.json` 构建报告 |
| 测试结果 | 同上 `physics_test/` | `summary.json` 汇总；`<后端>/<测试>.mp4` 视频与 `<后端>/<测试>/plots.jpg` 曲线 |

每项测试是为这个资产专门设计的过程，例如笔的拔帽、合帽、跌落和倒置，盒子的抽屉推拉、起滑角。

## 验收

**`summary.json`**：每个后端的 `failed` 为空；每项测试的关键指标（带中文标签，CPU 与 GPU 并排）与 README“物理实测”的记录相符，例如笔的合帽力 CPU 约 3.7 N、GPU 约 2.1 N；`rtf.physics` 不明显低于 README 的记录（本例 CPU 约 0.3–1）。

**视频**：过程符合测试意图，没有穿透、抖动、飞出；有疑问时对照 `plots.jpg` 的曲线。

验收通过后，把关键指标和性能（加 `--benchmark` 测量）更新到 README。

## 命令

```bash
# 构建物理资产
uv run python scripts/assets/build_physics.py --assets pen/000002 cabinet/000000 --gpu 0
# 物理测试，本例约 5 分钟；跑完为每个资产打印一行 SUMMARY <路径>
uv run python scripts/assets/physics_test.py --assets pen/000002 cabinet/000000 --physics isaac-cpu isaac-gpu --gpus 0 1 2
```

其他参数用 `--help` 查看。

## 不通过时

修改 `physics.py` 或 `physics_test.py`，重新构建，再用 `--tests` 只重跑受影响的测试。测试默认运行构建时复制的 `physics_test.py`，改了测试要重新构建。
