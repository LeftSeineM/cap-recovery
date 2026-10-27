# 资产抓取验收

检验单个资产第一次抓起后，能否在持物动作中保持稳定。协议和字段见[抓取规范](../specs/grasp-tests.md)，结果位置见[资产布局](../specs/README.md#资产布局)。

## 交付物

| 文件 | 看什么 |
| --- | --- |
| 资产 `summary.json` | 首次抓稳数、已评估、未评估与动作无效、闭合信号四格表、偏好、来源提交、录像 seed |
| `trials/<seed>.json` | 第一次闭合、抬升、持物动作、转移起点与失败原因 |
| `trajectories/<seed>.hdf5` | 所有 seed 的状态，可重渲染任意 seed |
| `videos/<seed>.mp4` | 正式验收每资产一条录像，失败优先 |

## 验收

- 先看汇总和来源配置；开发批次可先看根目录 `summary.md`。未评估单列，薄物体单列。
- 看选定录像：确实离桌，完整执行持物动作；握持点位移≤5 mm、转角≤5°，掉落单列。可用 `--dev --diagnostic-shake` 追加剧烈晃动，结果在 `diagnostic_shake`。
- 失败 seed 看 trial 的 `closure`、`lift`、`hold.trace` 与 `events`。发生参考点转移时检查转移和抵达后的预检；其他 seed 可从轨迹重渲染。诊断工具见[抓取复核](09-grasp-review.md)。
- 对照 `hold.motion_peaks`、`hold.segment_motion_peaks` 与 `requested_envelope`。实测超出强度上限及离散容差时，trial 为 `invalid_motion`，单列排除验收分母，先修正跟踪后重测。

## 命令

测试对象用 `--config` 或 `--asset` 指定，没有默认值。配置文件 `configs/asset_grasp_<场景>.py` 给出场景、排除的实例及原因、机器人和试次数，测其他场景的资产时新建一个配置；`--asset` 不带配置时可测任意已构建资产，需同时给 `--robot`，试次数默认 20。使用已构建资产，从干净提交运行完整配置；部分测试和录像复核必须用开发模式。GPU 调度参数沿用[批量运行](README.md#批量运行的通用参数)。

~~~bash
# 正式验收，完整配置
uv run python scripts/experts/eval_asset_grasp.py --config configs/asset_grasp_000103.py --jobs 2 --gpus 0 1
# 配置中部分资产与 seed 的开发运行
uv run python scripts/experts/eval_asset_grasp.py --config configs/asset_grasp_000103.py --dev --name review --asset objects/soldering_iron/000000 --seeds 0 1 --video --gpus 0
# 不属于任何配置的已构建资产，需指定机器人
uv run python scripts/experts/eval_asset_grasp.py --dev --asset objects/pen/000002 --robot x1pro_rm001 --gpus 0
# 从记录重渲染；result 也可指向资产的 grasp_test/<机器人>/<后端>/
uv run python scripts/experts/eval_asset_grasp.py render --result runs/grasp/<运行名>/objects/soldering_iron/000000 --seed 1 --output outputs/asset-grasp/seed-1.mp4 --gpus 0
# GPU PhysX 对照开发运行
uv run python scripts/experts/eval_asset_grasp.py --config configs/asset_grasp_000103.py --dev --physics isaac-gpu --trials 3 --jobs 2 --gpus 0 1
# 逐资产对照与闭合四格表
uv run python scripts/experts/eval_asset_grasp.py compare --baseline runs/grasp/<旧运行> --run runs/grasp/<运行名> --output outputs/asset-grasp/comparison
~~~

对照输出 `comparison.json`、`comparison.md`、`closure-checks.md`、`preferences.csv`、`closure-disagreements.json` 和 `timing.md`。compare 使用首次实际状态；顶层保留首次报告，`preference_attempts` 仅为额外偏好尝试，耗时表给出完整 trial 的 CPU 与原始 wall 算术均值。可加 `--expert-before`、`--expert-after` 导出专家逐 seed 对照。

参数细节用 `--help`。候选耗时很长时，`--seeds-per-worker` 可拆小进程分片；已有结果的失败不自动重跑。开发结果每次写新目录，正式结果在所有报告、轨迹和录像齐全后整体发布。

## 不通过时

先区分初始不可达、没抬起、持物动作中失稳、动作无效和动作未评估，再查看相应接触、轨迹与规划日志。修改通用策略或测试动作后，用同一失败记录复核，再跑完整配置。LCD薄边、偏心薄底座及扁平大件的边界见[已知限制](../specs/grasp-tests.md#已知限制)。
