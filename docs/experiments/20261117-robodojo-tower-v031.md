# RoboDojo 搭塔 v031

v031 的受保护撤离在一例开发轨迹与两例冻结首次复用轨迹完成原生终局。证据支持这三次运行；剩余失败和恢复分支分别保留，未合并成通用通过率。代码为本提交的 [`tower_cap`](../../table_1000/experts/robodojo_tower/tower_cap/)，逐文件哈希及归档哈希见 [`release.json`](../../table_1000/experts/robodojo_tower/release.json)。

| 来源 | 代码快照 | 分类 | E / T | 原生结束步 | 专家动作 | 结果 |
| --- | --- | --- | --- | --- | --- | --- |
| r1 / selector2 / layout15 | v029 | 开发基线 | 0 / 1 | 181，未终止 | 180 | 第二底座释放后撤手扰动，保护停止 |
| r1 / selector2 / layout15 | v031，本提交源码 | 开发后 | 0 / 1 | 956 | 955 | 原生 end=true、success=true |
| r2 / selector0 / layout2 | 同一 v031 字节 | 冻结首次复用 | 0 / 1 | 972 | 971 | 原生 end=true、success=true |
| r2 / selector0 / layout54 | 同一 v031 字节 | 冻结首次复用 | 0 / 1 | 1004 | 1003 | 原生 end=true、success=true |

基线 v029 与 v031 只在 `planner.py` 的受保护撤离分支不同：最小垂直净空从 6 mm 改为受检查的 18 mm，保留 30/45 mm 候选与原回原位规划；失败时不退回较小净空。第二底座撤离的实测位移从约 6.48 mm 降到记录精度内的 0 m，原 2 mm / 2° 结构保护保留。基线没有公共 Git commit，原冻结文件指纹由调用方原始归档保留；不把它写成该仓库的历史提交。

复用已有的 S0 动作发射、RoboDojo 原生判据、NumPy/SciPy 运动学与凸几何检查；v031 新增的是受保护撤离候选顺序与失败停止条件。三次完整成功均核验来源、动作/状态、三路录像完整解码、六张首末画面及独立交付回执。每次科学作业为一个既有专属 GPU、4 CPU、1050 原生总步数、1500 秒墙钟上限；未据此比较不同负载下的性能。

已知失败：r1/s2/l43 在 796 步发生盖块与上层板漂移；r2/s0/l26 在 887 步屋顶运输时发生支撑漂移。它们不计成功。v034 的支撑方向试验和稳定边界回退试验不属于该发布版本，也不计冻结迁移证据。

命令与输入边界见[操作说明](../tutorials/robodojo-tower-recovery.md)。本次公共复现检查为源码 SHA、27 文件包重建与语法检查；原始数据及外部环境由运行方持有，完整原生证据由索引哈希绑定，未声称可仅靠本仓库重跑原生实验。

layout54: the read-only boundary recording sidecar reported a missing import. It issued no extra native actions or intermediate restores; the frozen S1 and original native judge were unchanged.
