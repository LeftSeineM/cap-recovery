# 04 场景与专家数据

给验收场景和使用数据的人：一个场景交付初态、专家轨迹和录像。专家程序怎样编写和迭代见[第 5 章](05-step-by-step-expert.md)。以[场景 000000](../../asset_sources/scenes/000000/) 为例：X1Pro 打开三抽屉柜最下层的抽屉，把笔放进去，再关上。

![场景 000000 的初态，机器人头部相机](../../asset_sources/scenes/000000/thumbnail.jpg)

## 交付物

| 交付物 | 位置 | 内容 |
| --- | --- | --- |
| 场景源码（进 Git） | `asset_sources/scenes/<编号>/` | `scene.py`（场景与随机初始化，见[场景规范](../specs/scenes.md)）、`expert.py`（专家程序）、`thumbnail.jpg` |
| 初态 | `assets/scenes/<编号>/` | `initial.json` 检查报告、`initial_states.jpg` 拼图；有其他布局时另有 `layouts.jpg` |
| 专家轨迹 | `data/scenes/<编号>/<后端>/` | `summary.json` 汇总、`trajectories.hdf5` 成功的轨迹 |
| 录像 | 同上 `videos/<seed>/`、`fails/<seed>/` | 前 5 条成功轨迹和全部失败轨迹，四个视角合成一条并加标注 |

文件格式见[专家数据](../specs/expert-data.md)。

## 验收

### 初态

- `initial.json`：`invalid` 为 0；`mean_attempts` 是构建每个场景时尝试的次数，最小值为 1，越小越好；`settle` 中每个刚体静置时移动不超过 3 mm、2°；`robot_contacts` 为 0。
- `initial_states.jpg`：摆放合理，位置和朝向都有变化。
- `layouts.jpg`：好的布局整齐，彼此在看得见的地方明显不同；差的布局看得出没整理好（编号含义见[布局](../specs/scenes.md#布局)）。

### 专家轨迹

`summary.json`：

| 字段 | 典型值 | 看什么 |
| --- | --- | --- |
| `commit` | — | 不带 `+dirty` |
| `success_rate` | 约 0.95 | 失败的轨迹不进数据集 |
| `failures` | 几条 | 按“子任务: 原因”列出；集中出现的原因交给编写专家的人 |
| `mean_duration_s`、`mean_wall_s` | 约 28 s、60 s | sim time 与 wall time；wall time 随负载变化 |

### 录像

流程完整，笔在抽屉里，抽屉关好、手已撤开；没有可见穿模，夹持稳定，物体没有异常滑移或跳动。

## 命令

```bash
# 构建初态（有其他布局时加 --layouts 1:11）
uv run python scripts/scenes/build_scene.py 000000 --seeds 0:100 --gpus 0 1 2 3 --grid 5x5
# 生成轨迹，整体替换 data/ 下的目录；试跑用 --dev（见第 5 章）
uv run python scripts/scenes/run_expert.py 000000 --physics isaac-cpu --gpus 0 1 2 3 --jobs 8
# 渲染前 5 条成功轨迹；失败的加 --failed、去掉 --count
uv run python scripts/scenes/render.py 000000 --physics isaac-cpu --gpus 0 1 2 3 --jobs 8 \
    --cameras face_view left_wrist_view right_wrist_view wide --merge --captions --count 5
```

本例 100 个 seed 生成轨迹约 16 分钟。批量运行的通用参数见[批量运行](README.md#批量运行的通用参数)，其他参数用 `--help` 查看。

## 不通过时

摆放问题改 `scene.py`，从构建初态重来；专家问题见第 5、6 章，改完从生成轨迹重来。
