# Kernel and Graph quick start

[中文](quickstart.zh.md) · [Documentation](index.en.md)

Save this as a Python file and run it after installing `taichi-forge`. It uses CPU
so the first example does not require a GPU or optional vendor libraries.

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

## Reuse and ownership

The compiled Graph fixes dispatch structure, not buffer contents. Reuse the
binding set while the allocation/layout stays compatible. In-place data changes
are visible on subsequent runs. Publish a new binding with `graph.bind(...)`
when changing the resource set; rebuild the definition for structural changes.

`graph.run(...)` submits work but is not a universal completion wait. Use
`ticket = graph.submit(bindings)` and `ticket.wait()` when you need a completion
ticket. Do not overwrite in-flight inputs or assume a Python return means the
device has finished. Keep owners alive until their work completes.

`graph.close()` releases Graph-owned state; caller-owned `values` remains yours.
Do not run a closed Graph. After `ti.reset()`, recreate the arrays and Graph.

## Next steps

- Select `ti.cuda` or `ti.vulkan` only after checking your installed backend.
  A Graph can execute ordinarily even when device capture/replay is unavailable.
- For Field/template arguments, see [dense Field bindings](dense_field_graph.en.md).
- For completion, control flow and diagnostics, see [Graph execution](graph_runtime_optimization.en.md).
- To optimize an unchanged semantic definition, see [complete recipe search](graph_recipe_integration.en.md).
  Search is optional and does not alter the ordinary compile/run default.
