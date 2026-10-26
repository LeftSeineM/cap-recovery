# Table-1000 agent 入口

开始工作前阅读 [README.md](README.md)、[协作约定](docs/conventions/workflow.md)、[开发原则](docs/conventions/development.md)、[仓库与目录](docs/conventions/repository.md)、[文档写法](docs/conventions/writing.md)，以及任务涉及的[规范](docs/specs/README.md)。

- 改动通过 PR 提交到 `main`，合并前完成审查和相应验证；合并需要项目负责人批准。新 issue 先与负责人确认再创建。
- 本地环境、凭据、机器配置和调试产物不进入公共仓库；资产定义放在 `asset_sources/`，生成或下载的资产放在已忽略的 `assets/`，开发运行放在已忽略的 `runs/`，日志和临时结果放在已忽略的 `outputs/`。
- 几何、层级、关节和物理属性由资产文件表示；仅在出现明确消费者需求时扩展 `metadata.json`，并同步规范和读取方。
- 改变行为或接口时同步更新规范与手册；研究结论和实验结果在需要时整理进[实验台账](docs/experiments/README.md)，正常开发不要求随时记录。
