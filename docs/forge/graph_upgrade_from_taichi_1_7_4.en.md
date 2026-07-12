# Graph Upgrade Notes from Taichi 1.7.4

This document describes the public behavior and compatibility boundary of Forge
graph support relative to vanilla Taichi 1.7.4.

For exact signatures of Forge-only graph and native replay APIs, see
[Forge API reference](forge_api_reference.en.md).

## Public Compatibility

Forge keeps the familiar graph-builder surface:

- `ti.graph.GraphBuilder`
- `GraphBuilder.dispatch(kernel, *args)`
- `GraphBuilder.create_sequential()`
- `GraphBuilder.append(sequential)`
- `GraphBuilder.compile()`
- `Graph.run(args)`
- `ti.graph.Arg(...)`

Ordinary kernel graphs continue to use the public CGraph model. AOT graph
serialization for ordinary kernel-dispatch graphs remains compatible with the
existing public model.

## What Forge Adds

Forge adds backend-owned execution planning below the public API. This is used
to support faster replay paths and DSL-defined native algorithm replay without
requiring users to learn a new graph API.

The user-visible additions are:

- Runtime argument handling for scalar, matrix, ndarray, texture, and RW texture
  paths.
- Stable fallback when a backend replay path is not supported.
- Internal native replay for Forge-defined primitive sequences produced by the
  algorithm layer.
- `GraphBuilder.append_native(node, prewarm=False)` for Forge DSL-defined native
  nodes such as `PrimitiveSequence`, `DeviceCheckResult`, and
  `DeviceMetricResult`.

## Definition, Argument, and Runtime Lifetime Contracts

- `GraphBuilder.compile()` freezes the current dispatch/sequential definition.
  Later builder changes, including mutation and reuse of the original
  `Sequential`, do not change the compiled graph's runtime or lazy AOT result.
- `Graph.run(args)` requires a dictionary whose keys exactly match the
  declared runtime arguments. Missing and unexpected keys raise
  `TaichiRuntimeError` before submission.
- One call on a `Graph` object is a complete host invocation: two Python
  callers cannot interleave its CGraph/native nodes. Independent graphs remain
  independently submitable; the guard does not wait for GPU completion or add
  a default `ti.sync()`.
- `ti.reset()` invalidates graphs from the old runtime. Rebuild the
  builder/graph after reset; invoking an old graph produces an explicit runtime
  error.
- CUDA replay uses a generation-qualified allocation identity plus complete
  ndarray metadata. A graph executable pins captured allocations until safe
  retirement, so ndarray deletion, GC, or allocation-slot reuse cannot bind an
  old captured address to a new resource. This does not replace an
  application-level producer-consumer or snapshot protocol.
- On CUDA, changing scalar or matrix values, or rebinding an ndarray with the
  same dtype, shape, element shape, and layout, patches stable graph-owned
  argument buffers and reuses the existing graph executable. It does not
  perform the old default-stream synchronization and full recapture for every
  A/B/A resource switch. Old allocation leases and host patch buffers retire
  behind CUDA events with a bounded in-flight budget.
- A structural ndarray change still recaptures safely. Texture arguments remain
  on the conservative fallback path until they have an equivalent lifetime
  owner. This optimization uses the dynamically loaded CUDA Driver API and
  adds no CUDA Toolkit header, CUDART, or CUDA-versioned wheel requirement.

## Native Graph Boundary

Native graph support is intentionally narrow:

- Supported: native nodes produced by Forge's own DSL/native algorithm layer.
- Not supported: arbitrary user native callbacks inside graph.
- Not supported: AOT serialization for graphs containing Forge native nodes;
  `ti.aot.Module.add_graph()` accepts ordinary kernel CGraphs only.
- Not promised: cross-backend graph execution in a single graph.
- Numeric-check result nodes replay only the device-side native primitive. Result
  reads are still explicit through `to_int()`, `to_bool()`, `ok()`, or
  `to_float()`.

This keeps resource ownership and backend lifetime rules explicit.

## Suitable Workloads

Graph is most useful when the dispatch topology is stable and the workload is
replayed many times:

- fixed-shape simulation substeps;
- repeated native primitive chains;
- rendering or staging chains where resources stay stable;
- AOT deployments that need a named graph entry point.

Graph is less useful when Python changes dispatch topology every frame, shapes
or resource bindings change frequently, or the workload is dominated by one
large kernel rather than launch/replay overhead.

## Difference from Vanilla

Vanilla Taichi 1.7.4 primarily records ordinary kernel dispatches. Forge keeps
that model, then adds backend planning and native primitive replay under the
same public entry point. Unsupported optimized paths fall back instead of
changing graph semantics.
