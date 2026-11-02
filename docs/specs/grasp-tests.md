# 资产抓取测试

测试单个资产能否被专家第一次抓起并在持物动作中握稳。每个环境只放一个待测资产，用机器人验收的桌子和工作点，抓取候选、闭合检查和运动规划与专家共用。入口为 `scripts/experts/eval_asset_grasp.py`，实现见 [`evals/asset_grasp`](../../table_1000/experts/evals/asset_grasp/__init__.py)；共用的抓取策略见 [`skills/hold/frames.py`](../../table_1000/experts/skills/hold/frames.py) 与 [`skills/hold/pick.py`](../../table_1000/experts/skills/hold/pick.py)。

## 协议

- 资产由配置给出：`SCENE`、`ROBOT`、`TRIALS`（seed 为 `0..TRIALS-1`）、`EXCLUDED_INSTANCES`，可选 `ASSETS`。配置引用场景的资产清单并按资产路径去重，排除的实例逐个写明理由。单个资产也可用 `--asset` 和 `--robot` 直接测试。
- 资产原点在测试中心周围 10 × 10 cm 内均匀采样，偏航在 [-π, π) 均匀采样，竖直方向取碰撞几何的稳定支撑姿态。直接放在桌面，不用垫台。
- 每个 seed 只看第一次抓取：按专家策略选候选、闭合、抬升，实测上升不足 5 cm 即为失败。抬升与专家共用：先提 20 mm 确认两侧实际接触、物体随手，再提完其余高度。片状和细线物体走[薄物体路线](thin-grasp.md)，先移到桌边或指尖捏取。闭合检查只记录，不松手重抓。
- 抬起后依次做三段持物动作：横移 50 mm、上移 50 mm、转动约 1 rad，每段原路返回，最后保持 0.5 s。允许伴随运动（横移时转动、上移时斜向或转动、转动时平移），因为专家的持物运动本身是平移与转动同时进行。指令目标为强度上限的约 95%，同时满足机器人的关节速度和加速度上限。
- 强度上限取 000002 成功专家持物运动的 P99：线加速度 1.13 m/s²、角速度 1.23 rad/s、角加速度 3.01 rad/s²，推导见[台账](../experiments/20261026-asset-grasp/README.md#为什么)。
- 原地做不了持物动作时，先避碰转移到工作区中心的参考起点再做；转移途中的滑移和掉落同样计入。仍做不了时记为 `not_evaluated`。

## 验收

- 每个控制步读取物体相对实测 TCP 的位姿。初始握持点的最大位移不超过 5 mm、相对转角不超过 5° 才算握稳。失去手指接触，且握持点分离或下落超过 5 cm 的单列为掉落。
- 分母为全部试次减去 `not_evaluated`。抬升失败、持物失稳、执行错误和未产出报告都在分母内。
- 实测强度只作记录，不参与验收；个别段实测略超强度上限时照常计入，所以抓稳率是当前实际动作下的结果。

## 结果

正式运行要求干净的 commit，完整执行配置，全部 seed 齐全后整体替换各资产的验收目录（位置见[资产布局](README.md#资产布局)）；每资产保留一条录像，取首个失败 seed，没有失败则取 seed 0。部分资产、部分 seed 或带运行标签的属于开发运行，结果在 `runs/grasp/` 下，每次一个新目录。

每资产目录含 `summary.json`、`trials/<seed>.json`、`trajectories/<seed>.hdf5`，有录像时含 `videos/`；批次目录的 `conditions.json` 记录配置全文、协议参数和 commit。供 compare 和分析脚本读取的字段：

| 字段 | 含义 |
| --- | --- |
| `summary.first_attempt_success`（同 `success`） | 第一次抓起并通过持物动作的 seed 数 |
| `summary.lift_transport`、`stable_transport` | 两种口径的 `success`、`assessed`、`rate`：抬离原桌面并完成持物动作不掉落；与 `success` 相同的握稳口径。trial 的 `acceptance` 下有同名布尔值 |
| `summary.assessed`、`expected`、`not_evaluated` | 验收分母、请求数、未评估数 |
| `summary.closure_check` | 首次闭合信号（预测夹实/未夹实）× 实测结果（抓稳/失败）的四格表；没有实测结果的记为 `unmeasured`，未闭合的记为 `no_check` |
| `summary.preference_data` | 按部位和资产坐标系下的接近轴、闭合轴分组的实测稳定率。开启偏好补测（`conditions.json` 的 `collect_preferences`，默认关闭）时，第一次失败后换部位继续尝试的结果也进这里，不改变首次结果 |
| `trial.hold.motion_peaks`、`segment_motion_peaks` | 25 Hz 实测 TCP 差分得到的整段与各段峰值 |
| `trial.hold.preflight.selected.timing.<段>.constraint_ratios` | 指令的关节速度、加速度和 TCP 峰值与各自上限之比，用于判断强度受哪个约束限制；经过参考点转移的试次在 `trial.hold.preflight.relocation.actual_preflight.selected.timing` 下 |
| `trial.cpu_s`、`wall_s`、`load` | CPU time、wall time 与主机负载；含偏好补测，首次与补测分别在 `first_attempt_timing`、`preference_timing` |

`eval_asset_grasp.py compare` 按相同 seed 配对两批结果，两侧都按当前验收规则从 trial 重新判定。

## 已知限制

- 实测强度按 25 Hz 记录计算，看不到物理子步内的冲击峰值。约 3% 的动作段受关节上限限制，实测强度低于强度上限的 90%。
- 只检查被抓刚体相对 TCP 的运动，不适用于需要保持其他部件状态的多刚体机构。
- LCD 模块等带薄板边缘的组合体，候选可能优先夹住薄边；辅助夹具等偏心、薄底座的组合体可能绕夹持线转动后脱落。
- 扁平大件不在薄物体路线内，失败照常记录。
