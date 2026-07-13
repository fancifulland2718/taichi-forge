# Graph Runtime and Optimization

This document is the public source of truth for Forge graph runtime architecture,
backend replay, performance policy, diagnostics, and validation. For migration
from vanilla Taichi 1.7.4, see
[Graph upgrade notes](graph_upgrade_from_taichi_1_7_4.en.md). For exact public
signatures, see [Forge API reference](forge_api_reference.en.md).

## Scope and invariants

Forge keeps Taichi's public graph-builder model. Backend optimization must not
change these contracts:

- `GraphBuilder.compile()` freezes the dispatch and sequential definition.
- `Graph.run(args)` accepts exactly the declared runtime argument keys.
- One run of one `Graph` is a complete host transaction. Its CGraph and native
  nodes cannot be interleaved by another caller.
- Independent graphs remain independently submitable. Runtime guards end after
  host submission and do not add a default `ti.sync()`.
- `ti.reset()` invalidates graphs owned by the previous runtime.
- An optimized backend path may fall back to ordinary dispatch, but may not
  silently change results, bindings, or execution order.

Runtime synchronization protects Forge-owned launch, replay, and resource
state. It does not make application-owned simulation and rendering data safe.
Asynchronous producers and renderers still need snapshots, slots, double
buffering, or another explicit ownership protocol.

## Runtime argument discovery and template adapters

Public applications should declare graph arguments through
`GraphBuilder.dispatch()` and pass exactly matching keys to `Graph.run()`. A
physics or rendering engine can use the keyword-only `template_args=` parameter
to bind a data-oriented `self`, a Field, or another `ti.template()` argument at
definition time:

```python
builder.dispatch(
    solver.step_kernel,
    slot_arg,
    template_args={"self": solver, "state": solver.state},
)
```

These objects participate in specialization but do not enter the `Graph.run()`
dictionary. Field contents may change between replays; replacing Field identity
or layout requires rebuilding the graph. An ndarray or texture compile exemplar
still has a symbolic Arg, and each run still receives the real runtime resource.

Forge treats the durable AOT plan as the source of truth for dispatch
definitions and incrementally records its real symbolic argument names. This
recovers the exact runtime key set even when a legacy adapter bypasses the
Python fast-path registration in public `GraphBuilder.dispatch()`. Validation
remains strict: missing keys and extra keys not declared by the AOT plan still
raise instead of being ignored. Direct access to `_aot_graph_plan` and the
native builder remains only for legacy compatibility; new engine code should
use `template_args=`.

## Backend execution model

| Backend | Graph execution | Main safety boundary | Replay resource policy |
| --- | --- | --- | --- |
| CPU | Cached JIT dispatch plan; no device graph capture | One compiled graph is a complete replay transaction; ordinary kernels are protected at the whole-kernel boundary | CPU scheduler and JIT state are runtime-owned |
| CUDA | CUDA Driver API capture and executable replay, with patch or recapture when bindings change | Capture/replay and direct submission are serialized at the native host-submission boundary | Captured allocations are generation-qualified and retained until ordered retirement |
| Vulkan | Runtime-owned command recording and replay | GFX recording and replay registry mutations are protected per host API call | Monotonic graph identity, deferred retirement, fixed eight-slot in-flight ring |

The CPU path preserves graph semantics and concurrency safety but does not
pretend to offer CUDA-style device graph launch. CUDA and Vulkan optimizations
are backend implementation details below the same public API.

## CUDA capture and replay

Each CUDA graph executable owns its capture stream, stable argument buffers,
resource signature, and retirement state. Kernel-module launches receive the
capture stream explicitly; capture-owned buffers are retired in that stream's
order.

On the host, a process-wide native submission transaction keeps one graph
capture/replay, or one complete multi-task kernel, contiguous with direct
driver-kernel submission on the shared primary context and default-stream
ordering domain. Python graph arguments remain per invocation and native
execution may release the GIL. The transaction ends after steady-state host
enqueue, so GPU execution remains asynchronous; initial capture or recapture
keeps only its required local synchronization.

The resource signature includes generation-qualified allocation identity, byte
span, dtype, shape, element shape, and layout. The executable holds allocation
leases while captured work can still reference them. Deleting an ndarray,
running Python GC, or reusing an allocator slot therefore cannot redirect an
old executable to a new allocation.

Changing scalar or matrix values, or rebinding an ndarray with the same
structure, patches graph-owned argument buffers and reuses the executable.
Structural ndarray changes recapture. Texture arguments conservatively use
ordinary dispatch until they have equivalent lifetime ownership. Old leases
and host patch payloads retire behind CUDA events with a bounded in-flight
budget.

Capture failures are classified rather than treated as one permanent state:

| Failure class | Policy |
| --- | --- |
| Unsupported argument or preflight condition | Use ordinary dispatch for that invocation |
| Unsupported graph structure or `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` | Disable capture for that fixed graph cache |
| Other non-fatal capture or instantiate failure | Retry after 1, 2, 4, 8, 16, then at most 32 ordinary invocations |
| Illegal address, assert, launch failure, or another context-fatal result | Raise immediately; do not launch the same invocation again |
| Exception while capture is active | End capture through the native guard, then propagate the exception |

This policy follows NVIDIA's
[Driver API result semantics](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html)
and [stream capture contract](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html).
The implementation dynamically loads the CUDA Driver API and adds no CUDA
Toolkit header, CUDART, or CUDA-versioned wheel dependency.
Windows builds enable standard MSVC exception unwinding (`/EHsc`) when the
embedding build has not selected another `/EH*` mode. Linux compiler flags are
unchanged.

## Vulkan replay and slot capacity

Vulkan replay uses a monotonic nonzero graph identity and explicit
runtime-local registration. It does not use a reusable JIT-cache host address
as identity. Destroying or clearing a cache requests retirement; command
buffers, descriptors, and completion semaphores remain owned until every
in-flight slot is ready. Later launches or synchronization collect completed
state without adding a retirement wait. Runtime shutdown closes registration
before device teardown.

Replay deliberately uses a fixed ring of eight in-flight slots. When all slots
are busy, the current invocation takes ordinary dispatch without waiting.
Local bounded-growth experiments removed rare saturation fallbacks but did not
produce repeatable median throughput gains. A 1024-graph churn sample with a
16-slot cap increased driver-reported Vulkan memory by about 2.55 GiB even
though host RSS and exact results remained stable. Driver-retained command,
descriptor, and semaphore pools can outlive host graph state.

Forge therefore does not expose slot capacity as a DSL option or grow it per
graph. Re-evaluate this policy with both
`tests/python/vulkan_graph_slot_bench.py` and the graph-retirement stress;
eliminating fallback counts alone is not a sufficient optimization result.

## Diagnostics

`Graph._graph_stats` is an internal, experimental snapshot for regression
tests and production investigations. Its underscore is intentional: field
names and compatibility are not a stable public API. On CUDA, the first read
enables detailed counters for later invocations of that graph; the normal path
does not pay detailed counter or logging overhead before opt-in. Vulkan reuses
lightweight registry counters already needed by replay.

The snapshot can distinguish capture or record, exact replay, patched replay,
recapture, ordinary fallback, structural rejection, transient failure, retry
backoff, capture exceptions, and Vulkan slot saturation. It also reports the
last path, fallback reason, and driver error where available.
`known_persistent_argument_bytes` is only a lower bound for Forge-visible
argument buffers. It excludes opaque graph executables, command buffers,
descriptor pools, allocator high-water marks, and other driver-retained
memory. Use GPU memory telemetry, host RSS, and graph-churn stress alongside
it.

## Performance and memory trade-offs

Graph is most useful when dispatch topology and resource structure stay stable
across many replays, such as fixed-shape simulation substeps, repeated native
primitive chains, and render or staging chains. It is less useful when Python
changes topology every frame, resource structure changes frequently, or one
large kernel dominates launch overhead.

Do not compare graph paths with compile warm-up included. Warm kernels and
graphs first, synchronize at the same measurement boundaries, report median
and tail latency, and record GPU memory before and after long replay and churn
samples. Check results against ordinary dispatch rather than judging only
throughput.

On one local Windows validation system, three synchronized CUDA samples of a
four-dispatch, 1,048,576-element graph produced replay medians of
0.0385/0.0446/0.0380 ms with reported GPU memory unchanged at 756 MiB. A
512-invocation Vulkan sample completed about 12.4k graphs/s with zero slot
fallback and 536-to-536 MiB reported GPU memory. These numbers are local
regression evidence, not portable performance promises.

## Native and AOT boundary

Graphs may contain native nodes produced by Forge's own DSL/native algorithm
layer. Arbitrary user native callbacks are not supported. AOT serialization
through `ti.aot.Module.add_graph()` accepts ordinary kernel CGraphs only, not
graphs containing Forge native nodes. One graph is not promised to mix
backends. Numeric-check result nodes replay device work; reading a result
remains explicit.

See [Native algorithms](native_algorithms.en.md) for primitive ownership and
result APIs.

## Validation and platform status

The focused validation set includes:

- `tests/python/test_graph.py` for public contracts, lifetime, replay, and
  diagnostics;
- `tests/python/cuda_graph_runtime_bench.py` and
  `tests/python/cuda_graph_dynamic_patch_bench.py` for CUDA replay;
- `tests/python/vulkan_graph_slot_bench.py` and
  `tests/python/vulkan_graph_retirement_stress.py` for Vulkan capacity and
  lifecycle;
- `tests/python/backend_async_runtime_stress.py` for CPU/CUDA/Vulkan
  cross-thread submission;
- `tests/python/ggui_vulkan_queue_concurrency_stress.py` for asynchronous
  producer and display submission;
- backend feature-split builds and native C++ safety tests.

Windows validation covers CPU, CUDA, and Vulkan runtime paths. Linux compiler
branches were kept platform-neutral, but real Linux build, driver, window
system, and long-stress results must be rechecked before making a Linux release
claim. See [Linux revalidation status](linux_revalidation.en.md) for the exact
remaining matrix.

## Related documents

- [Graph upgrade notes](graph_upgrade_from_taichi_1_7_4.en.md)
- [Forge API reference](forge_api_reference.en.md)
- [Native algorithms](native_algorithms.en.md)
- [Compilation and advanced-optimization trade-offs](compilation_tradeoffs.en.md)
- [Linux revalidation status](linux_revalidation.en.md)
