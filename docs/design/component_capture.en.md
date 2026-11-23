# Component capture

## Capture from an existing object

`ti.aot.Component.from_owner(owner)` reads resource, dimension and value declarations made with `ResourceScope` at allocation. The `compile_graph` call records the order, bindings and initialization of the ordinary Graph. Stateless entries with explicit arguments can use `ti.aot.Component()`.

```python
component = ti.aot.Component.from_owner(owner)
root = component.compile_graph(owner.record_step, initializer=owner.record_initialize)
module = ti.aot.Module()
root.export(module, "step")
module.archive("component.tcm")
```

## Consumption

`taichi_forge_runtime.aot.inspect_component(path, "step")` reads the contract before device creation. Select a compatible component with the model's actual configuration. The runtime binds instance resources on load. Compatible code can be reused while each instance owns its mutable bindings. See [definition packages](../forge/package_aot.en.md) and the [standalone SDK](../forge/aot_sdk.en.md) for package-level entry points, consumer specialization and device support.
