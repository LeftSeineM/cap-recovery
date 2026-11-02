# 蓝黑色按动笔

![preview](three_quarter.jpg)

蓝色笔杆、黑色握部/笔夹/按钮，收回时约148×10×11mm。

两个刚体组件：笔壳 `pen.shell`；按钮、笔芯与笔尖构成的活动组件 `pen.refill`。滑动关节 `button_press` 为 SLIDER，沿约束局部 X，限位[−0.003, 0]m，负值沿−X伸出笔尖。

这是建模中的组件与演示关节；物理构建将两组件合为一个刚体，不模拟按动机构。笔管、握部与后套的视觉仍空心，碰撞使用原外轮廓实心凸体；收回状态下完全包在外壳内的 shaft、tip 只保留视觉，外露按钮和笔夹保留碰撞。原质量、质心与惯量显式保存在物理配置中。演示关节的连接体不互碰；以后实现真实按动机构时需拆成独立关节刚体。

## 资产树

从当前 `object.blend` 实际父子关系导出，省略视觉/碰撞网格。部件随所属刚体运动。

```text
pen
  refill [rigid]
    button
    shaft
    tip
  shell [rigid]
    barrel
    clip
    collar
    grip
```
