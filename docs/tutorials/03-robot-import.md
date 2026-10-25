# 03 机器人导入与验收

把厂商的 URDF 包变成仿真可用的机器人，并在仿真中验收行为。以双臂移动机器人 [X1Pro](../../asset_sources/robots/x1pro/) 为例：厂商包 193 MB，生成的源码不超过 10 MB。

![X1Pro：左为外观，右为碰撞凸块（每个刚体一种颜色）](../../asset_sources/robots/x1pro/three_quarter.jpg)

## 交付物

| 交付物 | 位置 | 内容 |
| --- | --- | --- |
| 机器人源码（进 Git） | `asset_sources/robots/x1pro/` | `robot.urdf` 与 `meshes/`（由厂商包生成）、`prepare.py`、`physics.py`、`robot.py`、`robot_test.py`、README 与缩略图，见[机器人资产规范](../specs/robots.md)。厂商原始包不进 Git，版本记录在 [assets.lock.json](../../assets.lock.json) |
| 机器人资产 | `assets/robots/x1pro/` | `robot.usda`；资产检查 `check.json`、`prepare.json` |
| 物理验收 | `assets/robots/x1pro/check/` | `summary.json` 汇总；每个用例的 `report.json`，渲染后的 `<视角>.mp4`；传感器用例的标记物图 |

## 验收

**资产检查**：`check.json` 中外观越出碰撞约 0.2 mm 以内；缩略图中外观没有破面、塌陷，碰撞覆盖完整，刚体划分合理。

**物理验收**：用例覆盖关键姿态、传感器、碰撞、夹持、底盘和持物行驶，条件和门槛见[规范](../specs/robots.md#验收)。

- `summary.json`：每个后端的 `failed` 为空；未通过的用例在 `failed_checks` 中列出检查项。
- 未通过或有疑问的用例看视频：过程符合用例意图，没有穿透、抖动、飞出。
- 传感器用例的标记物图 `<相机>_<姿态>.jpg`：十字落在标记物中心。

**键盘遥控**：各部分按按键运动，没有穿透和抖动。浏览器打开 `http://<服务器>:8765/` 后点击页面：W/S、A/D 底盘，R/F 升降，方向键转头，Tab 切换手臂，I/K、J/L、U/O 移动 TCP，Z/X、C/V、B/N 转动 TCP，空格开合夹爪，Backspace 复位，按住 Shift 加速。

## 命令

```bash
# 构建
uv run python scripts/robots/build_robot.py x1pro --gpu 0
# 资产检查（不需要原始包）
uv run python scripts/robots/prepare_robot.py x1pro --check
# 物理验收，约 45 分钟（16 个进程）
uv run python scripts/robots/check_robot.py x1pro --physics isaac-cpu isaac-gpu --gpus 0 1 2
# 渲染某个用例的视频
uv run python scripts/robots/render_check.py x1pro --physics isaac-cpu --cases grasp_and_push --trials 3 --gpus 0
# 键盘遥控
uv run python scripts/robots/teleop.py x1pro --output outputs/robots/x1pro-teleop --gpu 0 --port 8765
```

修改 `prepare.py` 或更新厂商数据时，先 `uv run hf auth login`、`scripts/assets/remote.py pull robots/x1pro/raw/ex001_render` 下载原始包，再 `prepare_robot.py x1pro` 重新生成源码。其他参数用 `--help` 查看。

## 不通过时

外观或碰撞问题改 `prepare.py` 后重新生成；驱动、材料或夹爪问题改 `physics.py` 后重新构建，再用 `--cases` 重跑受影响的用例。换夹爪的版本 [`x1pro_rm001`](../../asset_sources/robots/x1pro_rm001/) 命令相同，修改 x1pro 的网格后要重新生成它。
