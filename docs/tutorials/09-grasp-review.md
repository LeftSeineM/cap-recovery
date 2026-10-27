# 抓取失败复核

先完成[资产验收](08-asset-grasp.md#验收)，再按失败阶段检查同一 seed 的 trial、轨迹和规划日志。字段含义与判定规则见[抓取规范](../specs/grasp-tests.md)，历史证据与性能对照见[实验台账](../experiments/20261026-asset-grasp/README.md)。

| 现象 | 复核文件与字段 |
| --- | --- |
| 没有合法握法 | `target_attempts`、候选拒绝原因和工作进程 `run.log`；区分几何、IK 与整段规划失败 |
| 闭合后没抬起 | `pick.closure_check`、`closure`、`target_attempts[].lift_trace`；看实际 collider、指腹开口、接触冲量及物体相对 TCP 的运动 |
| 持物动作中失稳 | `hold.trace` 的平移、转角、接触与 `segment`，结合录像排除主动松手；最大顶点位移不是验收平移 |
| 动作未评估 | `hold.preflight` 及 `transport preflight`、`transport relocation`、`transport relocated preflight` 事件，查看全部等价方向和参考点转移的原始规划错误 |
| 持物动作明显过猛 | 对照 `hold.motion_peaks` 与强度上限，再看轨迹中的关节/TCP 变化；位置可达不能证明命令速度合规 |
| 闭合信号误判 | `eval_asset_grasp.py compare` 的四格表、分歧 seed 和实际抬升/持物动作真值；未完成测量不算实测失败 |
| 运行变慢 | trial 或 episode 的 `cpu_s`、`wall_s`、`load` 和子任务耗时；带 `--profile` 的专家还保存 `wall_profile`，候选生成仅计迭代恢复时的计算 |

任意 seed 可通过[重渲染命令](08-asset-grasp.md#命令)复核；轨迹包含偏好尝试和复位跳转，先对齐 trial 中的阶段。没有记录的接触力不能从录像补造。法向冲量除以控制间隔得到的是平均法向力，不是完整摩擦力；末子步冲量与控制间隔平均值也不同。

性能比较使用同一组 seed 和普通开发批次，保留 CPU、原始墙钟及负载。注明归一公式；每核负载只能近似共享机器争用。剖析样本用于定位，不能替代普通批次验收，累计父子调用不能相加。

一次性调查代码留在本地忽略目录，不作为公共入口。台账引用的历史诊断在所列提交检出后运行，不用当前版本缺失的脚本去解释历史命令。
