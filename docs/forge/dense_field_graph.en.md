# Dense Field Graph

> First available in `0.5.0`; see [release notes](release_notes.en.md).

Dense Field Graph first shipped in Taichi Forge `0.5.0`; this page describes
the published `0.6.2` release contract for compiling and
replaying kernels that either close over or receive dense `ti.field`, vector
fields, and matrix fields as runtime arguments. Static Field bindings and
runtime dense-storage bindings use the same public `ti.graph.GraphBuilder` API
and neither copies the Field payload.

This document is the public source of truth for Dense Field Graph support,
lifetime, concurrency, automatic differentiation, performance, and platform
status. General Graph architecture remains in
[Graph runtime and optimization](graph_runtime_optimization.en.md).

## Quick start

A closed-over Field is a definition-time dependency, not a runtime argument:

```python
import taichi_forge as ti

ti.init(arch=ti.vulkan)
state = ti.field(ti.f32, shape=1024)

@ti.kernel
def advance():
    for i in state:
        state[i] = state[i] * 0.99 + 0.01

builder = ti.graph.GraphBuilder()
builder.dispatch(advance)
graph = builder.compile()

graph.run({})
graph.run({})
```

The empty dictionary is intentional: a closed-over Field or a Field supplied
through `template_args` is a static dependency and does not need
`ArgKind.FIELD`.

Use the existing `ArgKind.NDARRAY` symbolic ABI when compatible Fields must be
replaceable between invocations. Graph automatically normalizes a canonical
compact dense Field into a runtime storage argument:

```python
@ti.kernel
def advance_runtime(
    state: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for i in state:
        state[i] = state[i] * 0.99 + 0.01

state_arg = ti.graph.Arg(
    ti.graph.ArgKind.NDARRAY, "state", ti.f32, ndim=1
)
builder = ti.graph.GraphBuilder()
builder.dispatch(advance_runtime, state_arg)
graph = builder.compile()

graph.run({"state": state})
```

The same runtime slot accepts a compatible `ti.ndarray` or an explicit
`ti.experimental.ndarray_view(field, slices=...)`. Dtype, logical ndim, and
vector/matrix element shape must exactly match the symbolic argument.
Qualified positive-stride padded fields and rank-preserving subviews bind
directly. Sparse, negative-stride, broadcast, overlapping, or otherwise
unsupported layouts fail explicitly without a shadow ndarray or implicit
staging.

For data-oriented kernels, bind `self` or another `ti.template()` parameter at
definition time:

```python
dt = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "dt", ti.f32)
builder.dispatch(
    solver.substep,
    dt,
    template_args={"self": solver},
)
graph = builder.compile()
graph.run({"dt": 1.0e-3})
```

## Supported dense layouts

| Area | Supported contract |
| --- | --- |
| Elements | Scalar, vector, and matrix Fields |
| Rank | 0-D, 1-D, 2-D, and higher dense shapes |
| Construction | `shape=...` and simple `root -> dense -> place` layouts |
| Placement | Same-node AOS-style and separate-node SOA-style placement |
| Ownership | One or multiple SNodeTrees in one Graph |
| Composition | Field-only, mixed runtime arguments, and mixed Forge-native segments |
| Runtime argument | Qualified compact or positive-stride, element-contiguous, unique dense mapping; full Field or rank-preserving subview |

Pointer, bitmasked, dynamic, hash, activation-list, and other sparse topology
behavior is outside this contract. Sparse support has separate backend and
lifetime requirements; it is not silently treated as dense.

## Binding and lifetime

A Field closed over by a kernel or bound through `template_args` has a mutable
payload but a static binding:

| May change between runs | Requires rebuilding the Graph |
| --- | --- |
| Field values | Static/closed-over Field identity |
| Compatible runtime dense Field identity and contents | SNodeTree generation of a static dependency |
| Runtime scalar/matrix values | Symbolic dtype, rank, element shape, or an incompatible layout |
| Runtime ndarray contents and compatible resource bindings | Shape, dtype, element shape, or layout |
| Slot-owned snapshot contents | Owning runtime after `ti.reset()` |

Forge records every referenced SNodeTree as an id-plus-generation dependency.
Destroying one of those trees makes the compiled Graph stale. Reusing the same
numeric tree id cannot redirect an old Graph to a new allocation. Rebuild the
Graph after replacing a static Field or its layout.

A runtime dense Field is not added to the Graph's static dependency set. Every
submission validates the descriptor's Program domain, SNodeTree id and
generation, layout fingerprint, dtype, rank, element shape, and byte range.
Passing the old Field after destroying its tree fails before enqueue; a new
compatible Field can bind to the same symbolic slot without rebuilding the
Graph. `ti.reset()` still invalidates the Graph and all views as a unit.

SNodeTree destruction is transactional with respect to registered Graph and
runtime objects. If retirement preparation fails, objects already prepared are
rolled back and the native tree remains live. After native destruction
succeeds, the Python wrapper becomes invalid even if later cache cleanup
raises. A wrapper retained across `ti.reset()` drops its old native references
without dereferencing the finalized Program.

Dense Field Graph does not copy or duplicate the Field payload. Diagnostic
`persistent_argument_bytes` also excludes opaque backend executables, command
buffers, descriptor pools, allocator high-water marks, and driver-retained
memory.

## Backend execution

| Backend | Dense Field Graph path | Important boundary |
| --- | --- | --- |
| CPU | Cached compiled dispatch plan | Preserves Graph semantics; it is not device-graph capture |
| CUDA | Driver-API capture and executable replay for compact internal storage | Exact binding replays in place and compatible internal allocation changes may patch; positive affine runtime arguments use ordinary fallback |
| Vulkan | Runtime-owned command recording and replay | Uses the bounded eight-slot in-flight policy; saturation may use ordinary dispatch instead of growing persistent driver resources |

An optimized path may fall back to ordinary dispatch, but it may not change
bindings, dispatch order, or results. Use `Graph.execution_stats()` to inspect
the actual path without adding a `ti.sync()`. Runtime dense Field arguments are
a runtime-bound JIT Graph contract; AOT Graph currently still requires an
owning Ndarray and does not accept borrowed dense storage arguments.

## Asynchronous simulation and rendering

The runtime protects its own launch, replay, queue, and lifetime state. It does
not infer application data hazards between independent Graphs or between a
simulation Graph and a renderer.

For asynchronous physics and rendering:

- use immutable, double-buffered, or slot/epoch-owned snapshot Fields;
- give different heterogeneous blocks disjoint writable Fields;
- share read-only Fields only under an explicit engine lifetime contract;
- publish a slot only after producer work for that slot is ordered;
- do not let simulation and rendering write the same Field concurrently.

One call to one compiled `Graph` is a complete host-side transaction and is
serialized by that Graph's lifecycle lock. Independent Graph objects remain
independently submitable, subject to backend queue/context synchronization.
The guard ends after host submission and does not add a default `ti.sync()`.

Construct and mutate a `GraphBuilder` from one coordinating thread. Compile
before handing immutable `Graph` objects to workers. Builder mutation and
`compile()` are not a concurrent construction API.

## Heterogeneous multi-environment organization

Use one block per stable solver/layout/shape/feature signature. Put homogeneous
environments inside that block on a leading Field axis:

```text
heterogeneous engine
  block A: solver/layout signature A -> Field[environment, ...]
  block B: solver/layout signature B -> Field[environment, ...]
  block C: solver/layout signature C -> Field[environment, ...]
```

This avoids one Graph per environment while preserving genuinely different
layouts between blocks. Each data-oriented owner remains a distinct kernel
specialization because its root bindings may differ. Forge does not merge
arbitrary closed-over owners by pointer or cache-key tricks. Compatible owners
can instead use the runtime dense-Field binding ABI; converting unrelated
closures into that ABI remains an explicit application choice rather than an
automatic cache optimization.

### Current 0.6.2 release boundary

Current Taichi Forge keeps the block model above: applications may own and
schedule independently compiled Graphs with different stable solver, layout,
shape, or feature signatures, while each block batches homogeneous
environments. Domain randomization stays inside a block when it preserves that
signature; a signature-changing environment belongs in another precompiled
Graph.

An `ArgKind.NDARRAY` slot may bind different but compatible runtime dense Fields
between invocations. Program, SNodeTree generation, layout fingerprint, and
byte range are validated when the binding plan is created or rebound; every
submission still reacquires generation-qualified ownership without copying the
payload. Closed-over and `template_args` Fields remain static bindings. Mixed
CGraph/native actions and structured control are available, but Forge does not
provide an automatic cross-block environment scheduler, structure-changing
Field hot rebinding, or a cross-device dependency planner.

Persistent offline cache and deliberate prewarming can reduce repeated compile
cost without weakening identity or lifetime checks.

## Automatic differentiation

`Graph.run()` is primal-only. It raises `TaichiRuntimeError` inside active
`ti.ad.Tape()` or `ti.ad.FwdMode()` contexts instead of silently omitting
gradients or dual propagation.

The boundary is also enforced across Python threads:

- Tape/FwdMode cannot enter while any Graph host submission is active;
- Graph cannot start while Tape/FwdMode setup is in progress;
- overlapping runtime-global AD contexts are rejected.

These checks cover host setup/submission only. They do not add a device wait or
serialize independent Graphs. An explicit `kernel.grad` object may be
dispatched to a separate Graph and run manually outside automatic-AD contexts.
Forge does not yet provide automatic primal/adjoint Graph pairing, reverse
Graph scheduling, or gradient contracts for Forge-native nodes.

## Diagnostics

`Graph.execution_stats()` returns the stable schema-v6 report. Relevant fields
include segment definitions, compiled task count, generation-qualified static
dependencies, pointer-free layout fingerprints, replay eligibility,
execution/fallback path, persistent argument bytes, immutable counter shapes,
and disabled-by-default host replay attribution. The snapshot is side-effect
free; use explicit submission telemetry for per-execution measurements.

Do not use private `_graph_stats` storage as an application API. GPU memory,
host RSS, graph/tree churn, and reset measurements are still needed because a
driver may retain resources that Python cannot enumerate.

## Performance and memory evidence

Measure after kernel and Graph warm-up, synchronize at identical boundaries,
compare results with direct dispatch, and report both median and trial range.
A relative range above 5% remains observational.

A Windows fresh-process test on 2026-07-14 used four heterogeneous blocks,
eight homogeneous environments per block, 256 base items, ten warmups, 200
rounds, and five CPU trials. Median throughput was 482.441 direct versus
673.679 Graph block invocations/s, or **+39.64%**. Direct and Graph ranges were
0.71% and 2.77%, so the result passed the 5% formal gate. Median steady RSS was
118.45 MiB direct and 119.16 MiB Graph.

Earlier matching tests measured **+270.71%** on Vulkan. CUDA measured a 16.27x
directional gain, but its trial ranges exceeded 5%, so that result remains
observational rather than a portable claim.

The cross-thread Graph/AD state machine adds no persistent per-Graph storage.
An intentionally empty CPU Graph microbenchmark measured a 127 ns/run median
host overhead versus an internal baseline without AD safety checks (5.29%,
1.73%/1.97% ranges). This is a worst-case percentage for near-zero work; the
representative four-block result above retained the established roughly 40%
Graph-over-direct gain. Moving this check into native code is not currently
justified by the absolute cost and added ABI complexity.

Graph is most useful for stable repeated dispatch topology. It is less useful
when topology or Field layout changes every frame, or when one large kernel
already dominates launch overhead. A fixed Vulkan replay capacity bounds
persistent resources; unbounded growth can consume much more driver memory
without repeatable throughput benefit.

## Compilation and startup

Dense Field Graph compilation includes Python specialization, backend kernel
compilation, Graph finalization, and first capture/record where applicable.
Do not include these phases in steady-state replay measurements.

Precompile stable specializations, use persistent offline cache, and prewarm a
bounded set of real block signatures. Avoid eagerly compiling every possible
domain-randomization combination. For phase-level interpretation and advanced
optimization trade-offs, see
[Compilation and advanced-optimization trade-offs](compilation_tradeoffs.en.md).

## Validation and Linux status

Windows validation covers CPU, CUDA, and Vulkan dense runtime paths, including
integer exactness, f32/f64 tolerances, AOS/SOA, multiple trees, mixed runtime
arguments, lifecycle invalidation, concurrent submission, automatic-AD
rejection, and explicit grad-kernel Graphs.

Linux code paths were kept platform-neutral, but Linux release claims still
require real GCC/Clang builds, CPU multi-block runs, CUDA Driver-only and
Toolkit-OFF zero-argument capture, Vulkan validation plus headless/headed
replay, sanitizer coverage, long churn, and allocator-specific RSS/VRAM/reset
measurements. Track these separately in
[Linux revalidation status](linux_revalidation.en.md).

## Related documents

- [Graph runtime and optimization](graph_runtime_optimization.en.md)
- [Graph compatibility and migration guide](graph_migration_guide.en.md)
- [Forge API reference](forge_api_reference.en.md)
- [Compilation and advanced-optimization trade-offs](compilation_tradeoffs.en.md)
- [Linux revalidation status](linux_revalidation.en.md)
