# RoboDojo 搭塔恢复边界

源码位于 [`robodojo_tower`](../../table_1000/experts/robodojo_tower/)。此目录为外部 RoboDojo backend 的固定源码快照，使用该环境的 `env/emit/last_command/output_directory` 接口；Table-1000 原有专家和资产接口保持各自语义。参数与调用方法见[操作说明](../tutorials/robodojo-tower-recovery.md)。

`release.json` 的 `python_files_sha256` 定义公开源码，`external_runtime_files_sha256` 定义调用方提供的部署数据。`package_fingerprint_sha256` 是所有 27 个文件的 SHA 字典按键排序、紧凑 JSON 编码后的 SHA256。`bundle.materialize` 先校验全部输入，再生成包；任一字节不同都不属于该验证版本。

运行证据分别记录开发后成功与冻结首次复用。冻结首次要求该 case 的原始输入在代码冻结后才进入开发流程；它不自动表示严格未见布局。恢复分支另记父轨迹、恢复边界、保留前缀、验证动作及实际后缀，不折算成冻结首次结果。

正式验收需要逐例原始来源、精确 E/T 接管、原生终局、动作与状态计数、完整录像和实际画面复核。对物理状态做写入恢复时必须保留边界标记及等价性缺口；读取数值一致不自动证明接触缓存、求解器历史或整段物理等价。

公共仓库只保存源码、用法和精简 SHA 证据索引。运行时几何、轨迹、录像、凭据和机器配置不进入 Git。原生实验依赖调用方保留的数据；公开索引不提供这些数据的下载或完整公开重放保证。
