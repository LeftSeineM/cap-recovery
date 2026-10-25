# 07 人工遥操失败案例

专家解不了的案例，由人遥操完成这一步：看可行的做法，检验步骤的检查是否写对，并留下轨迹。

## 交付物

每次尝试是一个新批次 `runs/teleop/<场景>/<日期>-<时间>-<commit>-step<k>-seed<s>[-failure]/`，结构与开发运行相同（见[专家数据](../specs/expert-data.md)）：

- 轨迹：Done 的在 `trajectories.hdf5`，否则在 `trajectories_failed.hdf5`；
- `episodes/<seed>.json` 的 `teleop`：案例、起点、每次检查的结果和你的判断；
- Done 的批次带第 k 步的检查点，后面的步骤可以 `--from` 它接着跑，也可以用 `render.py --batch` 渲染录像。

## 操作与验收

浏览器打开页面后，左侧列出失败过的案例（批次、步骤、seed），跑过 `--repeats` 的按成功次数从低到高排列。每个案例有两个起点：

- step：失败那一步开始时的检查点；
- failure：失败的技能入口（旧批次为失败时的状态）；“s before the failure”设为 N 时从失败前 N 秒开始。

画面是一个第三视角和机器人的 RGB 相机：左键拖动画面在成像平面内移动末端目标，Shift+滚轮沿视线前后移动；右键拖动转视角，滚轮缩放。键盘：I/K、J/L、U/O 平移末端，Z/X、C/V、B/N 转动末端，W/S、A/D 底盘，R/F 升降，方向键转头，Tab 切换手臂，空格开合夹爪，P 暂停，按住 Shift 加速。

“Check”只显示这一步的检查结果；“Done”或“Not done”按你的判断结束这一步。

**看什么**：你的判断和结束时的检查结果是否一致。不一致说明检查写得不对，要改检查。

## 命令

```bash
uv run python scripts/experts/teleop.py 000002 --gpu 0 --port 8765
```

浏览器打开 `http://<机器>:8765/`（或先 `ssh -L 8765:localhost:8765 <机器>`）。其他批次用 `--batches` 加入。Ctrl+C 退出，进行中的那次尝试不保存。
