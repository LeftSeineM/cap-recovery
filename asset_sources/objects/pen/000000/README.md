# 黑色按动圆珠笔

![preview](three_quarter.jpg)

黑色漆面笔杆、银色鼻锥与握环，约 149 × 13 × 14 mm。

包括两个刚体组件：笔身；笔尖、笔芯和后部按钮构成的活动组件。滑动关节 `button_press` 限位 [−0.003, 0] m，负值使活动组件一起朝笔尖方向移动。

这是建模中的组件与演示关节；物理构建将两组件合为一个刚体，不模拟按动机构。笔管、鼻锥的视觉仍空心，碰撞使用原外轮廓实心凸体；收回状态下完全包在外壳内的 shaft、tip 只保留视觉，外露按钮和笔夹保留碰撞。原质量、质心与惯量显式保存在物理配置中。演示关节的连接体不互碰；以后实现真实按动机构时需拆成独立关节刚体。

```text
pen
  refill [rigid]
    button
    shaft
    tip
  shell [rigid]
    barrel
    clip
    clip_bridge
    grip_band1..4
    nose
```
