# 接入 RoboDojo 麻将 CAP

交付物是[冻结v020包](../../table_1000/integrations/robodojo/mahjong_cap/README.md)、[速度恢复辅助函数](../../table_1000/integrations/robodojo/restore_velocity_readback.py)和[精简证据索引](../experiments/20261117-robodojo-mahjong/accepted-index.json)。策略模块及短先验与本地原生验证版本字节一致；示例几何仅规范化了资产路径元数据。

## 运行前提

在外部RoboDojo及其Isaac环境中安装本仓库，使`table_1000`可导入；worker需有NumPy，速度辅助函数另需PyTorch和所供`objstate`模块。本包未分发RoboDojo、资产、原始标注和原始轨迹；上游资源按其许可另行提供。现有Table-1000运行器没有接入此adapter。

1. 将权威标注的`rollout_success=false`与六份同源原件绑定，逐文件记录SHA。`E_frame`为空或画面与源状态无法核实时隔离；固定`T=max(1,E_frame-25)`并使用精确`objstate.step_idx`。
2. 原始前缀回放到T以建立控制和奖励历史，恢复序列化状态，保留原生counter、600总步限及奖励队列。剩余预算是600−T。原始视频帧号不作为NPZ行号。
3. 从实际安装资产提取并验证全部类别的碰撞边界和SHA，重绑包目录的`enabled_collision_bounds.json`。同时提供同作业`identity.json`、补牌碰撞等价证明及连续双臂指令；输入结构由对应模块的直接读取项定义。
4. 原生恢复时先写pose并完成USD mirror，再通过真实刚体API恢复速度。完成渲染同步后调用`restore_velocity_readback.restore_and_verify(scene_manager, data, k)`，另用独立recorder读回；速度最大误差须≤1e−6，不覆盖recorder缓存。该helper依赖所供`objstate.build_index/_type_of/_np`接口。接触求解器缓存与完整隐藏状态等价尚未验证。
5. 在同一worker调用包内`expert`；调用方动作入口负责每一步原生推进、预算和记录。停止后先完成进程清理，再做验收。此发布没有资源分配器、SSH、邮件发送器或原始恢复驱动器。

## 验收

- 原生`end_flag`和`success`同时为真，原始前缀与恢复状态读回通过，全部动作连续计入原600步。
- 三张匹配牌Z轴偏离朝上<30°，其余手牌Y轴偏离朝上<7°；补牌实际抬升、固定抓持、右手独立携带及最右支持释放有状态证明。
- 六路前缀/专家视频逐帧完整解码，检查首末画面；保存每步全状态、动作和失败机制，区分开发后成功与冻结首次新输入。

## 本机CPU检查

```bash
python -m unittest discover -s tests/integrations/robodojo -p test_mahjong_cap.py -v
python -m compileall -q table_1000/integrations/robodojo
```

这些检查覆盖打包一致性、入场稳定判定和参考路点几何，不替代外部RoboDojo原生验证。失败后查看原始终态、动作和实际姿态；按条件开发新能力，保持原判据和预算。背面朝上或非目标牌倾斜不能通过改标签或改接管点处理。
