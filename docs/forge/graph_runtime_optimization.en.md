# Graph execution and optimization

[中文](graph_runtime_optimization.zh.md) · [Documentation](index.en.md)

Use Graphs to reuse a stable sequence of kernel and supported native operations.
This guide covers binding, completion, control flow, diagnosis and optimization.
Start with the [runnable example](quickstart.en.md); exact signatures are in the
[API reference](forge_api_reference.en.md). These are current source contracts;
an installed release may lack a newer optional capability.

## Build, bind and reuse

```python
# builder contains your dispatches; arrays match its declared runtime arguments.
graph = builder.compile()
bindings = graph.bind({"source": source, "output": output})
try:
    graph.run(bindings)
    ticket = graph.submit(bindings)
    ticket.wait()
finally:
    graph.close()
```

- `compile()` freezes dispatch structure. Later builder changes do not modify
  an existing Graph.
- `run()` accepts an exact argument dictionary or a `GraphBindingSet` from
  that Graph. Missing/extra keys are errors.
- `bind()` publishes scalar/matrix values and resource identities. Array
  contents remain live; permitted in-place updates are visible on the next run.
- Reuse bindings for fixed resources. Publish new bindings for replacements;
  rebuild the definition when shape, topology or semantic assumptions change.
- `GraphBindingSet.update()` / `replace()` publish a new version only after
  validation succeeds. A failed publication leaves the previous version intact.
- `run()` is not a universal device-completion wait. Do not overwrite
  in-flight inputs or consume GPU results on the host without completion.

Each Graph serializes its host invocation. That does not prevent data races
between independent Graphs or simulation/rendering users of the same storage.
Use application-owned slots, snapshots or a producer/consumer protocol.

## Runtime arguments and dense fields

Declare runtime inputs with `ti.graph.Arg`. Bind a data-oriented `self`,
a captured Field or another `ti.template()` parameter at definition time with
`dispatch(..., template_args={...})`; do not repeat it in the run dictionary.

Use an `ArgKind.NDARRAY` slot for a compatible dense Field or storage view that
must be replaceable between invocations. A view being constructible does not
mean every native consumer or capture route accepts it.

| Resource | Usage and cautions |
| --- | --- |
| Compact ndarray | Bind matching dtype, rank and element shape; retain the owner |
| Dense Field / positive-stride view | Check [layout and consumer support](storage_views.en.md); no implicit permission to alias writable arguments |
| Static/template Field | Data can change; replacing layout/tree requires rebuilding the Graph |
| Texture / RW texture | Match dimension, format and sampler contract; backend replay support is narrower than ordinary kernel execution |
| Acceleration structure | Use the declared AS argument type and a compatible live scene; see [ray APIs](forge_api_reference.en.md) |
| Managed external storage | Follow [interop ownership and synchronization](zero_copy_interop.en.md); zero-copy does not imply cross-device support |

Destroying a referenced tree or calling `ti.reset()` invalidates old Graphs
and bindings. A new object at the same address does not revive an old binding.

## Backend execution model

| Backend | Execution | Important distinction |
| --- | --- | --- |
| CPU | Cached compiled dispatches | Not GPU Graph capture |
| CUDA | Capture/replay for eligible work; ordinary execution where supported | New bindings can require patching or preparation; an explicit selected recipe cannot silently change physical strategy |
| Vulkan | Recorded command replay for eligible work; ordinary execution where supported | In-flight replay storage is bounded; saturation can use ordinary dispatch |

A top-level replay label does not mean every segment was captured. A Graph
can include recorded regions, ordinary kernels and root-ordered native calls.
Inspect per-segment execution information for the actual boundary.

Optional libraries do not automatically enable new algorithms. See
[hardware recording and search support](external_hardware_providers.en.md#recording-and-complete-recipe-search-are-separate-capabilities).

## Structured control

Build a condition and body from `builder.create_sequential()`, then call
`while_loop()`, `if_then_else()` or `switch()`. Conditions run as Graph work,
not Python callbacks. A while condition writes a one-element integer predicate:
nonzero means continue. Optional status records an application-defined reason;
an optional counter records logical iterations. Always provide a finite
`max_iterations` bound.

The following is a construction fragment; `evaluate_stop`, `update_state`
and the symbolic arguments are supplied by the application:

```python
condition = builder.create_sequential()
condition.dispatch(evaluate_stop, state, predicate)
body = builder.create_sequential()
body.dispatch(update_state, state)

builder.while_loop(
    condition,
    body,
    predicate=predicate,
    carried_state=(state,),
    counter=counter,
    max_iterations=32,
    lowering_mode="auto",
    name="iterate",
)
```

Sequential regions have one owner and form a tree. The current structured
depth limit is two. Do not reuse a mutable control node at multiple call sites
or form cycles. Recordable native actions can enter a body only where their
provider explicitly supports it.

| Mode | Meaning |
| --- | --- |
| `portable` | Force the portable control route |
| `auto` | Select a supported control implementation |
| `native_required` | Reject the graph if its backend control requirements cannot be met |

Query `ti.graph.structured_control_capabilities()` on the active runtime.
Important current boundaries:

- CPU uses exact host control over compiled dispatches.
- Eligible flat CUDA control uses conditional Graphs on supported Driver API
  12.8+ runtimes; older capture-capable runtimes can use bounded masked control.
  Masking preserves logical results but may still issue inactive tasks.
- Vulkan supports bounded native while replay, not native if/switch. A region
  is limited to eight chunks of at most 64 iterations each (512 total).
  Profiler/dispatch-cache configurations can prevent this replay route.
- The supported asynchronous depth-two GPU shape is an outer while containing
  one to eight ordered inner whiles, with eligible dispatch/action gaps.
  Each loop needs a separate counter; predicate, counter and optional status
  controls are distinct one-element i32 arrays from the same runtime.
  Per-loop budgets are 1–64. Other nested shapes can require portable parent
  control and may reject asynchronous submission.
- The default expanded nested route is limited to 4096 encoded actions.
  Complete-recipe search can offer compressed nested conditional execution on
  capable CUDA runtimes; its 4096 limit counts static dispatches, not the budget
  product. It supports eligible Taichi-kernel actions, not arbitrary vendor
  capture. This does not change ordinary `auto`.

Compressed control can help large budgets with early exit; highly active small
kernels can favor expanded control. Choose using actual workload measurements,
not a fixed speed threshold. `control_flow_stats()` and explicit trace/terminal
observation describe logical progress; logical early exit alone does not prove
that the device stopped issuing all encoded work.

## Opt-in completion tickets

`ticket = graph.submit(bindings)` returns a completion ticket for that
invocation. `ticket.done()` queries completion; `ticket.wait()` waits for it.
CPU work may already be complete, and short GPU work can finish before return.

Keep application resources valid until completion. Dropping a ticket is not a
cancellation or permission to overwrite its buffers. A completion or observation
failure must be surfaced; do not rerun side-effecting work through a fallback.

For related asynchronous producers, optionally share a `SubmissionPacer`:

```python
pacer = ti.graph.SubmissionPacer(2, max_in_flight_per_lane=1, max_queued=8)
first = graph_a.submit(bindings_a, pacer=pacer, lane="simulation")
second = graph_b.submit(bindings_b, pacer=pacer, lane="render")
first.wait()
second.wait()
```

Pacing bounds admitted invocations and queued callers; it does not guarantee
independent GPU streams, parallel execution, priorities or memory limits.
It affects only calls using that pacer. Start with a small queue and increase
it only when useful overlap offsets retained memory and latency.

## Close, reset and failure recovery

- Close Graph/materialization owners when finished. `Graph.close()` is
  idempotent; caller-owned inputs are not transferred to the Graph.
- For recipe execution, keep the materialized handle alive for all uses of
  its executor and close the handle/context afterward.
- Runtime reset invalidates execution plans, bindings and prepared resources;
  reconstruct them in the new runtime.
- A pre-submission unsupported optimized route may use a documented ordinary
  path. An explicitly selected recipe must preserve its execution contract.
- Once side-effecting work has been submitted, an error is not an instruction
  to run it again ordinarily.
- For context-fatal CUDA errors or Vulkan device loss, stop producers and
  retire the runtime. Reset does not guarantee device recovery; restart the
  process if necessary.

## Diagnostics

`graph.execution_stats()` returns a passive snapshot of execution and resources.
Use the public report, not underscored caches.

| Question | What to inspect |
| --- | --- |
| Did the graph replay? | Per-segment path, replay eligibility and fallback classification |
| Why ordinary or native-ordered work? | Segment boundaries and reasons, not only the aggregate label |
| Were counters collected? | `counters_complete`; uncollected zero values do not prove zero activity |
| What was retained? | Graph memory and deduplicated provider-memory reports |
| Where did time go? | Explicit ticket telemetry or a profiler, not an ordinary status query |
| How did control terminate? | Control-flow/terminal observation under the declared control contract |

Use `graph.submit(..., telemetry="summary")` or `"timestamps"` when that
invocation needs measurement. Reporting does not implicitly enable telemetry.
Kernel `task_manifest()`, Graph task manifests and dispatch labels help
correlate compiled work; they are not launch-parameter search APIs.

## Performance and memory trade-offs

Separate preparation, first execution and repeated execution. Measure the
complete useful step/frame, including required completion and publication.
Keep equivalent input state, tolerances and synchronization on both sides.
CUDA event spans can include idle gaps; host wall time is not device time.

For small gains, repeat measurements and alternate baseline/candidate order.
Use Nsight Systems to inspect launch/copy/wait boundaries and Nsight Compute
when kernel behavior needs explanation. Profiling is opt-in and can alter timing.

Distinguish these quantities:

- caller-owned arrays;
- Graph/plan requested persistent and temporary bytes;
- memory retained by multiple bindings or workspace lanes;
- allocator reservation/high-water state;
- opaque driver/vendor memory and measured process/device peaks.

Unknown memory is not zero. Do not add provider bytes already counted by an
owner twice. `memory.deferred_host_argument_bytes` is retained CPU upload
storage, not VRAM or cumulative transfer volume; older reports may mark it
unavailable. Requested Graph bytes alone are not a device peak.

## Offline modified-CompileIQ Graph recipe search

`builder.freeze()` creates the semantic definition. Use
`definition.search_recipes(...).run(evaluator)` to evaluate complete physical
alternatives. Ordinary `definition.compile()` explicitly uses the baseline;
search does not install a global selector or change runtime defaults.

Candidates depend on semantics, backend and provider set. Examples include map
fusion, memory staging, offload phases, traversal, scheduling, workspace and
structured control. An available provider need not produce a candidate for every
Graph. Hardware regions can require explicit providers and prepared operations.

For installation, an executable example, JSON/Markdown reports, checkpoint
resume and cross-process selection restoration, use
[Complete Graph recipe integration](graph_recipe_integration.en.md).
No raw block, workgroup, PTXAS or library-route axis is exposed to CompileIQ.

## Numerical and automatic-differentiation contract

Graph execution preserves the declared semantics, but different legal physical
recipes can change floating-point reduction order within the stated numerical
contract. The application owns its reference and tolerances; do not assume
bitwise equality or one universal f32/f64 tolerance.

Automatic Tape/FwdMode recording through `Graph.run()` is unsupported and
rejected. An explicitly dispatched `kernel.grad` Graph may run outside an
automatic AD context. Do not infer native-node gradients from primal execution.

## Native and AOT boundary

`GraphBuilder.append_native()` accepts supported Forge-produced actions, not
arbitrary Python callbacks or a universal native ABI. Root ordering does not
mean backend capture.

`ti.aot.Module.add_graph()` supports the documented ordinary kernel CGraph
subset. Complete recipe selection artifacts are not AOT binaries. JIT-only
native/control/fusion behavior must not be assumed serializable.

See [API reference](forge_api_reference.en.md),
[dense Field Graph](dense_field_graph.en.md),
[native algorithms](native_algorithms.en.md) and
[hardware providers](external_hardware_providers.en.md) for operation-specific limits.
