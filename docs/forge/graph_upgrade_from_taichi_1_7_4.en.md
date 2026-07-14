# Graph Upgrade Notes from Taichi 1.7.4

This document describes the public behavior and compatibility boundary of Forge
graph support relative to vanilla Taichi 1.7.4. Backend architecture,
performance, memory policy, and diagnostics are maintained separately in
[Graph runtime and optimization](graph_runtime_optimization.en.md).

For exact signatures of Forge-only graph and native replay APIs, see
[Forge API reference](forge_api_reference.en.md).

## Public compatibility

Forge keeps the familiar graph-builder surface:

- `ti.graph.GraphBuilder`
- `GraphBuilder.dispatch(kernel, *args, template_args=None)`
- `GraphBuilder.create_sequential()`
- `GraphBuilder.append(sequential)`
- `GraphBuilder.compile()`
- `Graph.run(args)`
- `ti.graph.Arg(...)`

Ordinary kernel graphs continue to use the public CGraph model. AOT graph
serialization for ordinary kernel-dispatch graphs remains compatible with the
existing public model.

## What Forge adds

Forge adds backend-owned execution planning below the public API. This supports
faster replay paths and DSL-defined native algorithm replay without requiring a
new graph API.

The user-visible additions are:

- `GraphBuilder.dispatch()` and `Sequential.dispatch()` can use the
  keyword-only `template_args=` parameter to bind a data-oriented `self`, a
  Field, or another definition-time argument;
- runtime argument handling for scalar, matrix, ndarray, texture, and RW
  texture paths;
- stable ordinary-dispatch fallback when an optimized replay path is not
  supported;
- public `Graph.execution_stats()` execution/fallback/resource reports;
- explicit `kernel.grad` dispatch for manually managed gradient Graphs outside
  automatic-AD contexts;
- internal native replay for Forge-defined primitive sequences produced by the
  algorithm layer;
- `GraphBuilder.append_native(node, prewarm=False)` for Forge DSL-defined
  nodes such as `PrimitiveSequence`, `DeviceCheckResult`, and
  `DeviceMetricResult`.

## Definition and lifetime contracts

- `GraphBuilder.compile()` freezes the current dispatch and sequential
  definition. Later builder changes do not change an already compiled graph or
  its lazy AOT result.
- `Graph.run(args)` requires argument keys to exactly match the declared
  runtime arguments. Missing and unexpected keys raise `TaichiRuntimeError`
  before submission.
- One call on one `Graph` is a complete host invocation. Independent graphs
  remain independently submitable; this protection does not wait for GPU
  completion or add a default `ti.sync()`.
- `ti.reset()` invalidates graphs from the old runtime. Rebuild the builder
  and graph after reset.
- Dense Fields are definition-time bindings. Their contents may change, but
  replacing identity/layout or destroying their generation-qualified SNodeTree
  requires rebuilding the Graph. Numeric tree-id reuse does not revive it.
- Same-structure runtime resources may use backend replay; structural changes
  may recapture or fall back. Both paths preserve binding and execution
  semantics.
- Runtime graph safety does not replace an application-level snapshot, slot,
  or producer-consumer protocol for shared simulation and rendering data.
- New engine code should bind a `self`, Field, or another `ti.template()`
  argument through `template_args=`; the Field still does not enter each run's
  dictionary. Legacy adapters that write directly to the durable AOT plan keep
  exact runtime-argument discovery and strict missing/unexpected-key checks.
- `Graph.run()` is primal-only. Active `ti.ad.Tape()` and
  `ti.ad.FwdMode()` are rejected before submission instead of silently losing
  gradient/dual recording. An explicit `kernel.grad` Graph may be run manually
  outside those contexts.

The implementation details behind CUDA resource leases and dynamic patching,
Vulkan identity and deferred retirement, fixed replay capacity, failure
recovery, and `Graph.execution_stats()` are documented in
[Graph runtime and optimization](graph_runtime_optimization.en.md).

## Strict runtime-key migration

Definition-time template bindings must be removed from the runtime dictionary.
Code that kept passing them may have appeared to work with a permissive legacy
adapter, but Forge now rejects the extra key:

```python
builder.dispatch(
    solver.step,
    template_args={"self": solver, "state": solver.state},
)
graph = builder.compile()

graph.run({"state": solver.state})  # wrong: unexpected runtime argument
graph.run({})                       # correct for this zero-runtime-arg graph
```

Keep only declared `ti.graph.Arg` names in `Graph.run()`. This catches stale
engine adapters and misspelled parameters before backend submission.

## Heterogeneous multi-environment layout

Do not create one Graph per homogeneous environment. Use one block/Graph per
stable solver-layout-shape-feature signature and store homogeneous environments
on a leading Field axis. Separate heterogeneous blocks keep independent
writable Fields and submit independently. Simulation/render overlap still
requires slot/epoch-owned snapshot Fields; Graph does not add data-hazard
tracking.

## Native graph boundary

Native graph support is intentionally narrow:

- Supported: native nodes produced by Forge's own DSL/native algorithm layer.
- Not supported: arbitrary user native callbacks inside graph.
- Not supported: AOT serialization for graphs containing Forge native nodes;
  `ti.aot.Module.add_graph()` accepts ordinary kernel CGraphs only.
- Not promised: cross-backend execution inside one graph.
- Numeric-check result nodes replay only device-side native work. Result reads
  remain explicit through `to_int()`, `to_bool()`, `ok()`, or
  `to_float()`.

This keeps resource ownership and backend lifetime rules explicit.

## Suitable workloads

Graph is most useful when dispatch topology is stable and replayed many times:

- fixed-shape simulation substeps;
- repeated native primitive chains;
- rendering or staging chains with stable resource structure;
- AOT deployments that need a named graph entry point.

Graph is less useful when Python changes dispatch topology every frame, shapes
or resource structure change frequently, or one large kernel dominates launch
overhead.

## Difference from vanilla

Vanilla Taichi 1.7.4 primarily records ordinary kernel dispatches. Forge keeps
that model, then adds backend planning and native primitive replay under the
same public entry point. Unsupported optimized paths fall back instead of
changing graph semantics.
