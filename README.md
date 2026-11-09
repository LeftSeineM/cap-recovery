# Table-1000

Table-1000 是面向机器人桌面整理任务的 benchmark 与场景资产项目。

## 文档

| 文档 | 读者 | 内容 |
| --- | --- | --- |
| [研究方案](docs/research/proposal.md)、[一期计划](docs/research/plan.md)、[技术参考](docs/research/references.md)、[专家迭代](docs/research/expert-iteration.md) | 所有人 | 研究问题、交付物、计划 |
| [操作手册](docs/tutorials/README.md) | 人 | 命令、产出与人工验收 |
| [协作约定](docs/conventions/workflow.md)、[开发原则](docs/conventions/development.md)、[仓库与目录](docs/conventions/repository.md)、[文档写法](docs/conventions/writing.md) | 所有人与 agent | 怎样协作、开发和写文档 |
| [规范](docs/specs/README.md) | agent | 资产、场景与专家的文件、接口和格式 |
| [方案选择](docs/decisions/pen-cap-snap.md) | 所有人 | 重要方案的结论与依据 |
| [实验台账](docs/experiments/README.md) | 所有人 | 研究结论与关键实验结果，可按提交复现 |

[RoboDojo 麻将碰杠技能](table_1000/mahjong_cap/README.md)提供同向推牌与最右侧补牌的调用入口。

## 开发环境

在仓库根目录执行：

```bash
uv sync --locked
uv run python -c "import table_1000; print(table_1000.__file__)"
```

项目环境使用 Python 3.11（Isaac Sim 5.1 的要求）。核心包没有运行时第三方依赖；按需要安装的部分在 `pyproject.toml` 的可选依赖组中，版本由 `uv.lock` 锁定：

| 组 | 用途 | 主要内容 |
|---|---|---|
| `sim` | 仿真与资产物理测试 | Isaac Sim 5.1.0、Isaac Lab 2.3.2.post1、PyTorch 2.7.0（CUDA 12.8） |
| `robots` | 机器人运动学与规划 | MPlib、toppra、trimesh、rtree |
| `dev` | 资产与机器人开发 | Open3D、CoACD、huggingface_hub、pytest |

```bash
uv sync --locked --all-extras        # 或 --extra sim 等只装需要的组
```

`sim` 组约 19 GB，安装后通过 `uv run python` 使用项目环境中的 Isaac Sim，无需另外指定仿真解释器。Blender 使用其自带的 Python；策略使用各自的环境。

共享 uv 缓存时可在上述安装命令中添加 `--link-mode symlink`，避免复制环境中的大型依赖。物体构建、物理测试和机器人入口会自动从当前 Python 环境定位 Isaac Sim 资源，无需手动设置资源路径；显式设置的 `ISAAC_PATH`、`EXP_PATH`、`CARB_APP_PATH` 仍优先使用。

需要安装到已有的兼容 Python 环境时，可使用 `uv pip install -e .`。发行包名为 `table-1000`，导入名为 `table_1000`。

## 目录

顶层目录的用途和是否进 Git 见[仓库与目录](docs/conventions/repository.md#目录)。
