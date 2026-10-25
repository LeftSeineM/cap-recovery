# behavior.py

`behavior.py` 为可选的资产行为源码，负责原生物理属性不能表达的附加力或状态转换。普通刚体及原生关节能够表达的行为无需脚本。

## 挂载与生命周期

构建生成的外层 `object.usda` 使用官方 Python Scripting 组件，在资产根节点上引用同目录的 `behavior.py`。行为继承当前 Isaac 版本提供的 `BehaviorScript`，由官方组件负责实例化、加载和销毁。应用入口统一启用所需扩展及脚本执行；场景和测试只加载资产，不逐个导入行为代码。

每个挂载实例独立保存状态。开始仿真时建立所需的物理步订阅，停止/复位时恢复行为状态，卸载时释放订阅及跨实例引用。`on_update` 是时间线更新，不等同于物理步；施力逻辑使用官方 PhysX 物理步回调，保证每个实际物理步只执行一次。

公共代码仅封装反复使用的状态访问、订阅清理等辅助功能，不另建脚本加载器、manifest 协议或通用装配接口注册器。兼容性、配对数量、发现条件、释放条件均属于具体行为；笔帽配对不能成为所有资产必须实现的接口。

## 行为规则

- 默认参数放在行为代码中；需要逐资产调整的参数作为 USDA 自定义属性，含义和单位由该行为说明。
- 当前实例的对象通过相对路径或保存的原始对象名定位，不硬编码场景实例路径。
- 同类资产可共享行为模块。跨实例作用由对应行为协调，确保一组内力只计算一次；不得预设只能与自身原配部件作用。
- 内部作用成对施力，并计入作用点力矩；脱离后相应保持力归零。
- 不通过写入姿态或隐藏子步伪造物理结果。测试夹具和外力属于测试脚本，不属于资产行为。
- 额外观测量由行为提供只读访问，测试可按需采样，不要求所有行为实现同一种配对状态。

## 分发

行为资产交付整个目录：`object.usda` 为加载入口，引用 `object.usdz` 并挂载旁置 `behavior.py`。物理属性与行为参数由 `physics.py` 写入生成的 `physics.usda`；脚本引用只写在生成的外层入口，不打入 USDZ。

```usda
#usda 1.0
(
    defaultPrim = "Asset"
)
def Xform "Asset" (
    prepend references = @object.usdz@
    prepend apiSchemas = ["OmniScriptingAPI"]
)
{
    uniform asset[] omni:scripting:scripts = [@behavior.py@]
}
```

Isaac Sim 5.1 的官方加载器不能执行 `archive.usdz[behavior.py]` 包内路径，因此脚本不打包进 USDZ。无行为资产可直接加载 `object.usdz`。引用使用相对路径；公共工具依赖同版本 `table_1000`，不逐资产复制。

验收包括独立目录加载、同资产多实例、跨实例行为、停止/重置/卸载，以及 GUI 与无界面应用的加载。验收用官方组件加载；无界面应用已验证，GUI 路径未验证，标为未支持。

参考：[Isaac 行为脚本](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/replicator_tutorials/tutorial_replicator_modular_scripting.html)、[官方脚本组件生命周期](https://docs.omniverse.nvidia.com/extensions/latest/ext_python-scripting-component/user_manual.html)。
