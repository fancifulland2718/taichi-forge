# Graph Runtime and Optimization

This document is the public source of truth for Forge graph runtime architecture,
backend replay, performance policy, diagnostics, and validation. For migration
from vanilla Taichi 1.7.4, see
[Graph compatibility and migration guide](graph_migration_guide.en.md). For exact public
signatures, see [Forge API reference](forge_api_reference.en.md).
The static-Field feature contract is maintained separately in
[Dense Field Graph](dense_field_graph.en.md).

The base Graph modernization and native-node replay model first shipped in
Forge 0.4.1. The stricter lifetime, backend replay, diagnostics, and Dense
Field Graph contracts described here are the current 0.5.x contract; they do
not reclassify the whole Graph API as a 0.5.0 addition.

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
- Destroying a referenced SNodeTree invalidates the Graph even if a later tree
  reuses the same numeric id.
- `Graph.run()` is primal-only and rejects active or concurrently entering
  Tape/FwdMode contexts rather than silently dropping automatic differentiation.
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

For a mixed `CGraph(a) -> native -> CGraph(b)`, the native node ends the
current CGraph segment. Forge recovers symbolic names only from AOT items added
to that segment, so `a` cannot contaminate the later `b` segment.
`Graph.run()` still validates the exact union of every segment's arguments. A
Field-only CGraph receives an empty argument map at the C++ execution layer,
and a native-only node receives no runtime dictionary.

Python flattens one invocation once and reuses its resource signature and
containers under the same Graph's per-Graph lock. The CompiledGraph binding
constructs a segment-local C++ `IValue` map from that segment's own
declarations; Python does not copy another dictionary per segment. This keeps
the backend semantics segment-local while preserving a zero-copy host path.
Legacy adapters that access underscored objects still work, but recovery reads
only AOT items added since the previous segment flush.

## Dense Field lifetime and heterogeneous blocks

Dense scalar, vector, and matrix Fields are supported as definition-time
bindings. Their contents may change; identity, layout, shape, dtype, element
shape, SNodeTree generation, and owning runtime may not be hot-rebound. Sparse
topologies remain outside this contract. Heterogeneous engines should group
homogeneous environments inside stable blocks and use explicit snapshot
ownership between asynchronous simulation and rendering.

The complete support matrix, lifecycle transaction, multi-environment layout,
AD boundary, performance evidence, and Linux status are maintained in
[Dense Field Graph](dense_field_graph.en.md).

## Backend execution model

| Backend | Graph execution | Main safety boundary | Replay resource policy |
| --- | --- | --- | --- |
| CPU | Cached JIT dispatch plan; no device graph capture | One compiled graph is a complete replay transaction; ordinary kernels are protected at the whole-kernel boundary | CPU scheduler and JIT state are runtime-owned |
| CUDA | CUDA Driver API capture and executable replay, with patch or recapture when bindings change | Capture/replay and direct submission are serialized at the native host-submission boundary | Captured allocations are generation-qualified and retained until ordered retirement |
| Vulkan | Runtime-owned command recording and replay | GFX recording and replay registry mutations are protected per host API call | Monotonic graph identity, deferred retirement, fixed eight-slot in-flight ring |

The CPU path preserves graph semantics and concurrency safety but does not
pretend to offer CUDA-style device graph launch. CUDA and Vulkan optimizations
are backend implementation details below the same public API.

## Structured control

`GraphBuilder.while_loop()`, `if_then_else()`, and `switch()` add
backend-neutral structured regions without introducing a solver-specific
Graph API. Each region is built from fixed `Sequential` definitions. Runtime
values may change between replays, but the argument schema, resource identity,
shape, dtype, and dispatch topology remain fixed.

A bounded iterative program uses a condition region and a body region:

```python
condition = builder.create_sequential()
condition.dispatch(
    evaluate_stop,
    residual_sq,
    initial_norm_sq,
    user_stop,
    predicate,
    status,
    atol,
    rtol,
)

body = builder.create_sequential()
body.append_native(operator.graph_action(direction, product))
body.dispatch(update_iteration, direction, product, counter, status)

builder.while_loop(
    condition,
    body,
    predicate=predicate,
    status=status,
    control_inputs=(residual_sq, initial_norm_sq, user_stop, atol, rtol),
    carried_state=(direction, product),
    counter=counter,
    max_iterations=128,
    lowering_mode="auto",
    name="iterative_program",
)
```

The condition kernel may combine any number of DSL-computed criteria, such as
absolute and relative tolerance, user cancellation, active work, or numerical
breakdown. It writes a one-element integer `predicate` ndarray; nonzero means
continue. An optional, distinct one-element integer `status` ndarray records
why execution stopped. Graph transports and reports that value but does not
assign solver meanings to status codes. The optional `counter` is the exact
logical iteration count. `max_iterations` is always a host-defined safety
bound and does not need to be encoded in a solver-specific condition.

`if_then_else()` selects one fixed branch from a predicate computed by its
condition region. `switch()` selects a zero-based fixed branch, or an optional
default, from a selector computed by its condition region. Branch schemas are
compiled before execution; Python callbacks cannot run inside a region.

Current lowering is explicit:

| Backend | Structured `while` | `if` / `switch` |
| --- | --- | --- |
| CPU | Exact `cpu_host_loop`; condition and body use cached compiled dispatch plans | Exact portable host control |
| CUDA | `auto` uses a native CUDA conditional Graph when the Driver API is at least 12.8 and the required symbols/lowering are available; otherwise exact portable replay | `auto` uses one native CUDA IF/SWITCH node when qualified; otherwise exact portable host control |
| Vulkan | Exact portable replay by default; optional masked chunk replay may reduce observations while reporting logical and executed iterations separately | Exact portable host control |

`ti.graph.structured_control_capabilities()` returns the active backend's
portable lowering and device-control qualification. The report intentionally
separates an RHI primitive from a complete structured runtime path: Vulkan has
indirect compute dispatch in its RHI, but does not claim device-controlled
`while`, `if`, or `switch` until predicate production, visibility,
zero-dispatch behavior, replay, and terminal observation are qualified as one
runtime contract.

`lowering_mode="portable"` forces the portable route.
`lowering_mode="native_required"` requires the qualified CUDA conditional
route and fails before execution when unavailable. Recordable provider actions
may enter a structured body only when their provider declares it safe; opaque
or unsupported providers fail closed.

Recordable providers may also declare private symbolic scratch bindings backed
by Graph temporary requirements. The Graph memory plan materializes one bounded
arena slot per in-flight invocation, resolves the private symbols before
submission, and keeps them out of `Graph.run()` / `Graph.submit()` arguments.
Bindings are reused for repeated execution of the same arena slot and rebound
when another asynchronous slot is selected. Providers must declare exact byte
and alignment requirements, return the complete declared symbol mapping, and
reject incompatible storage before backend work is submitted.

At the Graph root, consecutive ordinary CGraph segments and compatible
recordable-provider actions are lowered into one backend region. Fixed and
private temporary bindings are merged before compilation; conflicting bindings
fail explicitly. Structured regions inline the same provider dispatches only
when the provider has qualified the corresponding condition/body/branch role.

`Graph.control_flow_stats()` returns one immutable `GraphWhileReport` or
`GraphBranchReport` per structured region for the latest `run()`. Native CUDA
IF/SWITCH execution keeps `Graph.run()` fire-and-continue: selector readback and
report construction are deferred until `control_flow_stats()` is requested, so
that diagnostic call is the explicit synchronization point. While reports
include the selected lowering, logical and executed iterations, observation
boundaries, predicate/counter/status traces, terminal status, transfer bytes,
and native-upgrade reason. Portable structured regions use synchronous
`Graph.run()` and are rejected by `Graph.submit()` rather than being hidden
behind an asynchronous ticket. CUDA `while_loop`, `if_then_else`, and `switch`
regions declared with `lowering_mode='native_required'` may use `Graph.submit()`
when conditional Graph lowering is available. An ordered device setter reads
the predicate or selector before the conditional child, so the condition,
selected branch or bounded loop, and any explicit terminal
`GraphBuilder.observe()` snapshot are enqueued without a host control readback.
After asynchronous structured submission, read terminal state from
`ticket.observations()`; synchronous control-flow reports remain unavailable
for that submission.

Conditional-control metadata is uploaded asynchronously on the ordered default
stream and retained until the associated replay completes. The runtime keeps at
most two deferred replay batches; a third rapid submission waits for the oldest
batch instead of growing host staging and event state without bound. This
backpressure does not create a worker thread, an additional CUDA stream, or
device concurrency. `Graph.execution_stats()` exposes
`asynchronous_control_updates`, `deferred_replay_waits`, and
`peak_deferred_replay_batches` for qualification.

## Opt-in completion tickets

`Graph.run(args)` retains its established hot path and return contract.
Applications that need explicit asynchronous ownership can instead call
`ticket = Graph.submit(args)`. Submission validates the same exact runtime
arguments, serializes one Graph invocation at the same host boundary, and
publishes exactly one Program-local completion after all mixed CGraph and
native segments have been enqueued.

`ticket.done()` performs a nonblocking backend query; `ticket.wait()` waits for
that invocation rather than the whole device. Neither method inserts a default
`ti.sync()`. CPU completion is immediate. CUDA uses a Driver API event and
Vulkan uses a stream semaphore; a short GPU invocation is allowed to be ready
before its ticket is returned. Completion errors are sticky and surface from a
later `done()`, `wait()`, or runtime synchronization boundary.

Context-fatal CUDA errors and Vulkan device loss also become the Program's
immutable first fault. Once observed, later `Graph.run()`, `Graph.submit()`,
kernel, ticket-recording, synchronization, and Vulkan display submissions
fail fast instead of issuing more backend work. A failed Graph invocation is
never retried through ordinary dispatch. Stop producers and use `ti.reset()`
to retire the old Program; this does not promise recovery of a lost context or
device, so a real backend loss may require restarting the process. See
[Fatal backend errors and runtime reset](forge_api_reference.en.md#fatal-backend-errors-and-runtime-reset).

Pending runtime arguments are retained by the Program completion domain.
Graphs and Forge native workspaces are retained by the Python runtime owner
registry until the same completion becomes ready, even if the ticket is
dropped. Collection occurs on later submission, polling, synchronization, and
reset; the native completion queue is bounded so abandoned tickets cannot make
backend tracking grow without limit. This is a deliberately small completion
API: callbacks, `asyncio` adaptation, cross-Program ordering, and an explicit
Graph dependency scheduler remain out of scope.

## Bounded cooperative submission pacing

Applications with multiple asynchronous producers should combine completion
tickets with explicit admission pacing. Share a `ti.graph.SubmissionPacer`
across related `Graph.submit()` calls or CUDA/Vulkan batch-solve submissions to
bound backend invocations in flight, per-lane occupancy, and calls waiting for
admission. Admission occurs before backend enqueue, so the complete host launch
sequence of one paced invocation does not interleave with another. Invocations
already admitted to the backend remain asynchronous.

Scheduling is work-conserving round robin across lanes and FIFO within a lane.
Assign stable lanes to independent rhythms such as physics, rendering, and
streaming. Set `max_in_flight_per_lane` when one producer must not occupy every
slot. `on_saturation='wait'` applies backpressure. A real-time loop that cannot
block can use `on_saturation='raise'` and explicitly degrade or skip that frame
before any backend work has been submitted.

While a caller is blocked for capacity, the pacer uses that caller to poll all
in-flight completions with bounded adaptive backoff. This allows a later fast
invocation to free capacity without waiting for the oldest slow invocation and
does not require a persistent worker thread. The per-lane limit remains the
mechanism that reserves capacity against a high-rate producer; completion
polling does not preempt work already queued on the device.

The mechanism coordinates only calls sharing that pacer; ordinary kernels,
`Graph.run()`, and unpaced submissions remain outside its control.
`statistics()` exposes current and peak in-flight/queued counts, per-lane
grants and completions, rejections, backend failures, and admission wait time
for capacity and cadence validation. The pacer does not provide priorities,
deadlines, callbacks, or cross-Program dependencies. Applications needing
those policies should implement them above this admission boundary.

### Concurrency and resource budgeting

Host asynchrony is not a proxy for device parallelism. The public contract does
not guarantee that paced invocations receive independent CUDA streams or Vulkan
queues. Increasing `max_in_flight` first increases the number of invocations
allowed to queue and retain resources, not the number of available GPU cores.
An incomplete Graph may retain runtime-argument allocations, native
workspaces, replay command state, and a completion object. Mixing Graph and
solve work also adds plan-clone workspaces and operator numeric generations.

Pacer admission is based on invocation count and does not weight memory or
estimated GPU time. Its schema-v2 `statistics()["contract"]` identifies the
admission unit, lack of a device-concurrency guarantee, non-preemptive behavior,
and workspace, generation, and unpaced-submission exclusions. Graph
`execution_stats()` exposes `persistent_argument_bytes` and
`replay_slot_saturation_fallbacks`, but backend-driver command buffers,
descriptor pools, and allocator reservation still require backend profiling
and process-memory measurement.

Start with one invocation in flight. Increase the limit to two only when an
Nsight or equivalent trace demonstrates useful overlap between host enqueue or
wait and productive GPU work, while peak memory, p95/p99 latency, and replay
saturation remain within budget. Do not treat the runtime completion safety cap
as an application queue depth. Coarsen Graph work or increase batching before
creating one asynchronous ticket per small task.

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

Use the stable, frozen `Graph.execution_stats()` schema v1 report. It exposes
definition counts, compiled task count, segment-local runtime arguments,
generation-qualified static dependencies, a pointer-free layout fingerprint,
execution/fallback path, replay eligibility, persistent argument bytes, and
immutable per-segment counters. Application code should not read the internal
`Graph._graph_stats` cache.

The report distinguishes capture or record, exact replay, patched replay,
recapture, ordinary fallback, structural rejection, transient failure, retry
backoff, capture exceptions, native dispatch, and Vulkan slot saturation. The
first report opts in to detailed GPU counters for later executions. If work
ran before opt-in, `counters_complete=False` remains explicit for that runtime
epoch. Reading a report does not call `ti.sync()`.

Persistent argument bytes are only Forge-visible host/backend argument
storage. They exclude opaque graph executables, command buffers, descriptor
pools, allocator high-water marks, and other driver-retained memory. Use GPU
memory telemetry, host RSS, and graph/tree churn stress alongside the report.

## Numerical and automatic-differentiation contract

Replay changes host submission only; it does not change kernel arithmetic,
dispatch order, or Field dependencies. The release matrix requires exact
direct-versus-Graph equality for integer copy/gather/update without data
races. Normal f32 arithmetic uses `rtol=1e-5`; supported f64 paths use
`rtol=1e-12`. Floating atomic/reduction checks use an explicitly stated
tolerance because backend execution order may differ.

Graph is currently primal-only. Calling `Graph.run()` while `ti.ad.Tape()` or
`ti.ad.FwdMode()` is active or entering raises before submission. Conversely,
automatic AD cannot enter while a Graph host submission is active, and
overlapping runtime-global AD contexts are rejected. These guards add no device
wait. An explicit `kernel.grad` may be dispatched into its own Graph and run
manually outside automatic-AD contexts. Forge does not yet provide an immutable
primal/adjoint Graph pair, reverse dispatch ordering, or native-node gradient
executable contract.

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

`benchmarks/graph_structured_control_bench.py` measures preparation, first run,
steady wall time, control observations, and (where the backend profiler can see
the launches) device kernel time separately. On a local Windows RTX 5090
regression run with 262,144 f32 values and 16 iterations, CUDA native
conditional control had a 464.8 us steady median versus 1,436.6 us for forced
portable replay, a 67.6% reduction (3.09x). Control observations decreased from
17 batches / 204 bytes to 2 batches / 24 bytes. First conditional capture was
20.4 ms and remains preparation cost. The same uninstrumented probe measured
6,513.6 us on CPU host control and 4,375.4 us on Vulkan portable control; these
backend numbers describe the tested execution boundaries, not cross-device
performance promises.

Current Dense Field multi-block throughput, compile scaling, cache, RSS/VRAM,
and the Graph/AD guard microbenchmark are reported in
[Dense Field Graph](dense_field_graph.en.md). They are local regression evidence,
not portable performance promises; a relative trial range above 5% remains
observational.

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

- `tests/python/test_graph.py` for public contracts, lifetime, replay,
  structured control, and diagnostics;
- `tests/python/test_graph_iterative_qualification.py` for f32 PCG and
  nonsymmetric BiCGSTAB over the generic structured/provider contracts;
- `benchmarks/graph_structured_control_bench.py` for structured-control
  preparation, steady wall time, observation traffic, and kernel timing;
- `tests/python/test_graph_dense_field.py` for static Field binding, SNodeTree
  generation/lifetime, zero-argument replay, mixed segments, and concurrency;
- `tests/python/test_graph_dense_field_numerics.py` for integer exactness,
  f32/f64 tolerance, AOS/SOA layouts, multiple trees, primal-only AD rejection,
  and explicit grad-kernel Graphs;
- `benchmarks/graph_dense_field_multiblock_bench.py` for fresh-process
  1/2/4/8-block compilation, cache, throughput, fairness, RSS/VRAM, display,
  and reset reports;
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
claim. In particular, dense Field Graph still needs GCC/Clang builds, Linux CPU
multi-block runs, CUDA Driver-only/Toolkit-OFF zero-argument capture, Vulkan
validation/headless/headed replay, sanitizer coverage, and allocator-specific
RSS/VRAM/reset measurements. See
[Linux revalidation status](linux_revalidation.en.md) for the exact remaining
matrix.

## Related documents

- [Dense Field Graph](dense_field_graph.en.md)
- [Graph compatibility and migration guide](graph_migration_guide.en.md)
- [Forge API reference](forge_api_reference.en.md)
- [Native algorithms](native_algorithms.en.md)
- [Compilation and advanced-optimization trade-offs](compilation_tradeoffs.en.md)
- [Linux revalidation status](linux_revalidation.en.md)
