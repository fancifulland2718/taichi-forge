# Graph Upgrade Notes from Taichi 1.7.4

This document describes the public behavior and compatibility boundary of Forge
graph support relative to vanilla Taichi 1.7.4.

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

## Native Graph Boundary

Native graph support is intentionally narrow:

- Supported: native nodes produced by Forge's own DSL/native algorithm layer.
- Not supported: arbitrary user native callbacks inside graph.
- Not promised: cross-backend graph execution in a single graph.

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
