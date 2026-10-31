# 平放薄物体抓取

专家 `pick` 与资产抓取测试共用薄物体路线。分类与悬出截面见 [`skills/hold/sections.py`](../../table_1000/experts/skills/hold/sections.py)，移边、闭合和抬升见 [`skills/hold/thin.py`](../../table_1000/experts/skills/hold/thin.py)，技能用法见 [`skills`](../../table_1000/experts/skills/__init__.py) 模块说明。

## 路线

- 自由单刚体按真实碰撞几何和当前手的指腹判为片状、细线或普通物体，不依赖资产名称、人工抓点或 metadata。
- 片状物体用指尖压住顶面拖到可达的桌边，撤回后重读位姿和悬出量，再水平接近、竖直夹取；细线和移边后质心无法留在桌上的小片用指尖捏取。
- 准备动作靠真实接触移动物体，只有手指可以接触目标和桌面；质量、摩擦、驱动、步长和共用抓取容差不由技能修改。
- 薄物体由 `pick` 自动分流，分流后自己选择抓取几何，传入 `point`、`approach` 或 `max_angle` 时报错。`grasp` 不做准备动作，对平放在桌上、没有可夹悬出的薄物体报错；先调用 `slide_to_edge` 再 `grasp` 可以分步完成。

## 验收

资产抓取测试对所有资产同时记录两个指标，字段见[资产抓取测试](grasp-tests.md#结果)：

- `lift_transport`：第一次抓取把物体抬离原桌面至少 5 cm，并完成持物动作不掉落。准备中落到地面后再捡起不算。薄物体以此为主指标。
- `stable_transport`：原有 5 mm / 5° 握稳判据，与 `success` 相同。

## 已知限制

- 移边准备的握法搜索有预算，耗尽时失败原因为 `thin preparation search exhausted`，可能漏掉可行候选。
- 计划时的支撑和可达不保证实际随动与稳定夹持；轻薄的碎片和细线成功率明显低于标签类片状物体。
- 极轻薄刚体在资产物理配置中提高位置迭代次数，见[物理属性](physics.md)。

## 参考

- [Hang 等，2019，Pre-Grasp Sliding Manipulation of Thin Objects](https://www.eng.yale.edu/grablab/pubs/Hang_RAL2019.pdf)：按支撑边缘组合滑动与侧向抓取。
