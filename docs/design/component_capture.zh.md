# 组件捕获

## 从已有对象捕获

`ti.aot.Component.from_owner(owner)` 从正常分配位置的 `ResourceScope` 声明获取资源、维度与值。`compile_graph` 录制正常 Graph 的顺序、绑定和初始化。只有显式参数的无状态入口也可以使用 `ti.aot.Component()`。

```python
component = ti.aot.Component.from_owner(owner)
root = component.compile_graph(owner.record_step, initializer=owner.record_initialize)
module = ti.aot.Module()
root.export(module, "step")
module.archive("component.tcm")
```

## 消费

`taichi_forge_runtime.aot.inspect_component(path, "step")` 可在创建设备前读取合同。用实际模型的配置选择兼容组件；加载时由 Runtime 建立实例资源。相容代码可以复用，有状态绑定由各实例拥有。按 Python 包导出的公共入口、消费时特化及设备支持范围见[定义包](../forge/package_aot.zh.md)和[独立 SDK](../forge/aot_sdk.zh.md)。
