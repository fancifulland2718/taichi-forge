# Kernel 与 Graph 快速开始

[English](quickstart.en.md) · [文档入口](index.zh.md)

安装 `taichi-forge` 后将以下内容保存为 Python 文件运行。示例使用 CPU，不要求 GPU 或可选 vendor 库。

```python
import taichi_forge as ti

ti.init(arch=ti.cpu)
values = ti.ndarray(dtype=ti.i32, shape=16)
values.fill(0)

@ti.kernel
def increment(x: ti.types.ndarray(dtype=ti.i32, ndim=1)):
    for i in x:
        x[i] += 1

argument = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1)
builder = ti.graph.GraphBuilder()
builder.dispatch(increment, argument)
graph = builder.compile()
bindings = graph.bind({"values": values})
try:
    graph.run(bindings)
    graph.run(bindings)
    ti.sync()
    assert (values.to_numpy() == 2).all()
finally:
    graph.close()
```

## 复用与所有权

编译 Graph 固定的是 dispatch 结构，不是缓冲区内容。分配与布局兼容时复用绑定，
原位修改数据会在后续运行中可见。更换资源集合时用 `graph.bind(...)` 发布新绑定；
结构变化时重新建立 definition。

`graph.run(...)` 提交工作，但并不通用地等待设备完成。需要完成凭据时，使用
`ticket = graph.submit(bindings)` 和 `ticket.wait()`。
不要覆盖仍在使用的输入，也不要把 Python 返回当作设备已完成；所有者应活到工作完成。

`graph.close()` 释放 Graph 拥有的状态，调用者的 `values` 仍由调用者拥有。
关闭后不能继续运行 Graph；`ti.reset()` 后应重建数组与 Graph。

## 下一步

- 核对已安装后端后可选择 `ti.cuda` 或 `ti.vulkan`。
  没有 device capture/replay 能力时，Graph 仍可能通过 ordinary 路径执行。
- Field/template 参数见 [Dense Field 绑定](dense_field_graph.zh.md)。
- 完成等待、控制流和诊断见 [Graph 执行](graph_runtime_optimization.zh.md)。
- 优化不变的语义 definition 见[完整 recipe 搜索](graph_recipe_integration.zh.md)。
  搜索是可选流程，不改变普通 compile/run 默认行为。
