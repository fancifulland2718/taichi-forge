# CUDA execution domains (development)

[中文](cuda_execution_domains.zh.md)

The independent AOT consumer can bind a saved execution plan to a bounded CUDA
execution domain. Each domain owns a nonblocking stream, and each prepared
instance owns its executable, immutable arguments and compiler temporaries.
This API preserves the complete selected plan. Stream handles, priorities and
completion tickets are runtime choices; they are never serialized into AOT.

JIT Graphs can also be prepared as independent instances borrowing selected
compiled code and the current Program's dense Fields. This is part of F3.
Selected Canvas display has Windows CUDA/Vulkan qualification; cross-platform
windows and application FPS remain unqualified. Queue priority is a device
scheduling hint; it does not guarantee
concurrent kernels or priority preemption.

## Capability discovery

Before `ti.init()`, inspect `ti.runtime_contract_manifest()["features"]`. A
runtime-only consumer can use
`taichi_forge_runtime.runtime_contract_manifest()["features"]` without importing
the DSL or creating a device. These are native implementation declarations:

| Feature | Declared route |
| --- | --- |
| `cuda_execution_domains` | Bounded persistent queues, explicit priorities and exact ticket dependencies |
| `cuda_aot_execution_domains` | Complete saved execution-plan preparation and replay |
| `cuda_jit_graph_execution_domains` | Complete live JIT Graph preparation |
| `cuda_external_storage_execution_domains` | Managed external imports and selected acquire/release epochs |
| `cuda_canvas_execution_domains` | Selected Canvas copy/writer/empty/repeat and bounded prepared image pools |

Use `features.get(name, False)` for older packages. Missing declarations do not
become support because a version or commit looks newer. Device/driver support,
same-GPU sharing and each recipe's compiler/provider facts are checked at cold
creation/preparation. A true build declaration does not qualify Linux windows
or every provider. Existing bootstrap and private ABI 111 are unchanged; an
incompatible shim/native pair is rejected before C++ bindings are entered.

## Current JIT Graphs

```python
domain = ti.graph.execution_domain(capacity=2, instance_capacity=64, priority=0)
step = graph.prepare(arguments, execution_domain=domain)
step.initialize()
ticket = step.submit(wait_for=previous_tickets)
if ticket is not None:
    # Wait when consuming results or retiring resources, rather than at every
    # simulation/render handoff.
    ticket.wait()
step.close()    # release the host instance handle
domain.close()  # cold: drain and retire resources
```

`arguments` accepts a dictionary or this Graph's `GraphBindingSet`. Preparation
freezes its current values; later binding-set updates do not change an existing
instance. It consumes the complete physical dispatch, control and parallel
description, preserving each invocation's launch policy. It does not export an
artifact, recompile the frontend or copy Field roots. Separate preparations own
independent arguments, control buffers and declared private workspace. Shared
application ndarrays and Fields still require exact dependencies.

The current scope includes scalar/tensor JIT ndarrays, compact AOS
`ti.experimental.ndarray_view()` ranges over live dense Fields/ndarrays,
implicit dense Field access, ordinary and masked control, and selected CUDA parallel regions.
Existing ndarray `.grad` storage is borrowed for explicit adjoint kernels.
Automatic Tape/FwdMode is not supported; initialization/submission rejects that
context and concurrent AD entry. Closing the source Graph leaves prepared
instances usable. Field-tree destruction or Program reset retires dependent
instances before releasing code or storage.

Explicit and retained dense views keep their finite offsets and element shapes.
They do not copy roots or force application Fields into new ndarrays. The
borrowed AOT runtime exposes `runtime.import_dense_view(view)` for the same
storage boundary on CPU, CUDA and Vulkan. Destroying a Field tree revokes the
loan and all derived aliases after draining dependent executions; handle shells
remain closeable. Ndarray loans retain the allocation after source wrapper
collection. Import is cold and must run outside a borrowed submission scope.

Ordinary host/device transfers and CPU/CUDA pointer exports preserve the finite
view's offset and size. Mapping applies the offset when the underlying memory
is host-visible. Legacy CGraph arguments cannot represent offset views: use a
kernel or a complete execution plan, rather than a legacy `module.graph()`
launch. Legacy Vulkan memory export also rejects nonzero buffer-relative
offsets, since its offset field describes the VkDeviceMemory binding instead.
Both reject before submitting any work or exposing the parent allocation.

Synchronized CUDA external views can also be borrowed by JIT and loaded AOT
recipes. Finite aliases retain the original synchronization owner. Each complete
initialization or replay consumes one acquire/release epoch per external owner,
even when several kernels or aliases use it. Private initialization does not
consume a replay-only signal. If both phases use external storage, the producer
must rearm between them. Busy or rejected admission consumes no signal.
Ordinary AOT launch, memory copy/map/export and default prepared execution reject
these managed imports. Native resource factories that inspect external operands
during cold preparation are not qualified. Closing the external owner drains
dependent domains before revoking every loan and alias; this is a cold operation.
Actual CUDA/vendor memory destruction may synchronize unrelated work even after
the participating prefix has completed. For a pause while peers run, stop
admission and wait for the domain's latest completion while retaining its bounded
instances and image pool. Destroy them only at the cold teardown boundary.
Canvas stream selection and source completion are a separate interface boundary.

## Selected Canvas display domain

Bind the current Program's domain with `canvas.set_execution_domain(output)`.
Before starting simulation, call
`views = canvas.prepare_frame_pool(width, height, capacity=3)` and prepare the
complete writer Graph/AOT bindings for those physical views. Capacity is bounded
by the renderer's in-flight limit plus its last displayed image. Views returned
by preparation are binding identities; only the view returned by a current
`acquire_frame()` lease may be written. Do not prepare new recordings per frame.
The prepared pool never expands during admission: when every eligible image has
an in-flight reader, acquisition returns `None`. Idle repeat aliases are reused;
the last displayed image is retained. Resize requires closing the old output
domain, binding a new one and calling `prepare_frame_pool()` again before
preparing new writers. This cold reconfiguration retires old Vulkan readers.

```python
canvas.set_execution_domain(output)
views = canvas.prepare_frame_pool(width, height, capacity=3)
writers = {
    id(view.descriptor): graph.prepare({"pixels": view}, execution_domain=output)
    for view in views
}
for writer in writers.values():
    writer.initialize()

frame = canvas.acquire_frame(width, height)
if frame is not None:
    ticket = writers[id(frame.pixels.descriptor)].submit(wait_for=dependencies)
    if ticket is None:
        frame.cancel()
    else:
        canvas.submit_frame(frame, source_completion=ticket, track_source=True)
        window.show()
```

The supplied ticket must cover this frame's complete writer. Native validation
checks the live runtime, CUDA stream and released image owner. Source tracking
copies this exact ticket; it neither records the default queue's tail nor
includes later work on the selected queue. `frame.source_completion` covers
source reads, while `frame.completion` covers Vulkan consumption, not scanout.
Passing no writer to an empty frame submits only the external semaphore epoch
on the selected queue. Repeat uses the same queue and adds no pixel kernel.
Full admission returns busy without consuming a semaphore or sealing a writable
lease. Cancel itself submits no CUDA work; reuse performs any needed Vulkan
rearm.

Selected-domain `DisplayFrame.from_packed_u32_ndarray()` copies also accept
`source_completion` as an exact device predecessor. Their source ticket covers
the copy. Warm the finite source set before steady use; up to 64 source ndarray
leases per physical output image remain retained through cold retirement, so
discard/minimize and completion polling cannot free an in-flight source. Busy
does not add a source lease. Generic `set_image()` packing and texture/host
inputs are not selected-domain routes; use a prepared writer or packed buffer.

Resizing shared storage is a cold boundary: close the old output domain, create
and bind its replacement, and reprepare the new physical view bindings. An
attempt to replace live bound storage rejects before retiring it. Reset
invalidates old leases. Window destruction drains dependent CUDA/external use
before releasing the Vulkan device. Failed retirement retains the bounded
graphics closure instead of freeing storage that may still be in use.

Binding frames, multiple workspace lanes, dynamic recipe selection, provider
data without a live-loan description, strided/read-only or unsynchronized external views and unqualified
native routes reject during cold preparation without a default-stream fallback.
Existing `Graph.run()`/`submit()` contracts stay unchanged. The Python Program
owner retains the borrower used by `ti.graph.execution_domain()` so dropping
the last domain wrapper cannot accidentally trigger destructive runtime close.

## Prepare, initialize and submit

The source-free runtime also exposes `runtime.import_cuda_external_memory()`
for a finite linear CUDA buffer imported from a same-GPU external allocation.
Supply the allocation's OS handle, size and device UUID, a binary
`ready_for_cuda`/`ready_for_external` semaphore pair, dtype/shape and optional
byte offset. Original OS handles remain caller-owned on both success and
failure; the runtime imports private duplicates. Keep the producer device,
allocation and semaphore objects alive until the imported memory closes, and
separately drain the producer's own queued work. Import each physical pair once
and use execution-plan views for aliases. Ordinary host transfers and default
launches reject these managed buffers.

Each accepted complete recipe consumes one ready signal and publishes one
consumed signal. The producer must **enqueue** its signal before CUDA submit;
the signal may still be pending on device. Enqueue the external consumer wait
after CUDA submit returns. A host request placed in another thread's work queue
does not establish this ordering until that thread submits to Vulkan. No host
wait for GPU completion is required by this protocol.
See [CUDA external semaphore ordering](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html).
Current real-device qualification is Windows CUDA/Vulkan; the FD import path
does not constitute Linux window qualification.

Use the runtime package directly in a source-free consumer, or obtain the same
interface through `ti.aot.borrow_runtime()` when explicitly sharing a JIT owner.
Given a module exporting `step` and its argument dictionary:

```python
from taichi_forge_runtime.aot import Runtime

with Runtime("cuda") as runtime:
    module = runtime.load_module(module_path)
    domain = runtime.execution_domain(capacity=2, instance_capacity=64, priority=0)
    step = module.prepare_execution("step", arguments, execution_domain=domain)
    step.initialize()  # cold: publish initialization before enabling submit

    ticket = step.submit()
    if ticket is not None:
        # Other host work can run here. Waiting is an explicit caller decision.
        ticket.wait()
        ticket.close()
    step.close()      # release the host handle without waiting
    domain.close()    # cold destructive retirement
```

`initialize()` is required even without explicit initialization operations. It
publishes the exact default-stream prefix containing module/storage setup, then
executes any initializer in the selected domain. It never calls the runtime's
all-domain wait. Its event and initializer storage remain bounded by the cold
instance lifetime; completion does not immediately destroy driver objects.
Initialize all steady-state bindings before starting the application pipeline.

`submit(wait_for=[ticket, ...])` validates exact predecessor tickets, performs
bounded admission, and enqueues the complete plan. It returns a `Completion`
when accepted, or `None` when full. Full admission enqueues neither dependencies
nor partial work. Initialization is never implicit in `submit()` or the selected
domain's `run()`. `run()` discards the returned ticket and raises when full.

Tickets belong to one runtime generation. Dependency admission rejects a
foreign or reset owner.
`done()` queries and `wait()` blocks without acquiring the runtime submission
lock; a waiting thread does not lock out queries or peer-domain submission.
`domain.latest_completion()` snapshots the already accepted prefix. Work
submitted later cannot extend that ticket. Closing a ticket only releases its
host handle and does not cancel work.

## Ownership and supported state

Cold preparation uses persisted compiler facts to qualify internal state.
Qualified dense-field kernels, `real_func` calls, vector arguments and cross-task
temporaries use the existing compiler and AOT loader. BC, PTX and cubin retain
these facts. Unknown older artifacts and kernels using unisolated shared RNG,
runtime errors, sparse lists/allocators, printing, mesh state or unqualified
external calls are rejected for independent execution. Ordinary default-queue
APIs remain available under their existing contracts. Selected binding families,
storage rings and host-selected branches are not silently converted to another
execution route. Each native operation must admit the independent domain route.

The caller owns application data hazards. Domains do not infer that two array
views or dense fields alias. Explicit dependencies must order every overlapping
write/read or write/write, including snapshot overwrite and slot reuse.

`capacity` bounds pending submissions; `instance_capacity` bounds resident
recordings. A plan with an explicit initializer can retain both an initializer
and replay recording. Releasing a prepared handle keeps its code, arrays and
reverse dependency edges until domain retirement. Explicit module, memory or
native-resource destruction closes dependent domains before freeing their
storage. Runtime close/reset retires its domains. A failed retirement retains
the unsafe closure and reports the failure instead of freeing in-flight state.

`domain.release()` releases only the host handle; the runtime still owns its
bounded resources. `domain.stop()` permanently stops admission. For a resumable
pause, stop submitting on the host and retain the domain; wait only the exact
prefix needed by the application. Explicit `domain.close()` is destructive and
may encounter driver/provider synchronization during cleanup. It belongs at a
cold boundary, with external producers able to signal already submitted waits.

## Pipeline scheduling

An independent simulation queue does not remove actual data dependencies.
Express `simulation(n) → snapshot(n) → render(n)` with device tickets, also order
`snapshot(n) → simulation(n+1)` before live state is overwritten, and reuse a
snapshot slot only after its render ticket completes. Keep the host queues
bounded. When presentation is busy, retain the latest completed frame and drop
redundant presentation requests instead of blocking simulation admission.

Start with equal priorities and bound rendering separately from physics. If
display acquisition is busy, skip that presentation request and continue
eligible physics work. Keep only the newest eligible snapshot, respecting its
lease; reusing a slot still requires its exact render completion. Measure
physics and output progress independently before increasing physics priority.
On the qualified RTX 5090 synthetic ABBA matrix, high physics priority improved
its long-compute progress while reducing output progress and increasing the
time to finish both workloads. Short kernels were largely host limited.
Priority is an explicit policy choice, not an automatic throughput optimization.

CPU `wait()` and byte-buffer host transfers are explicit synchronization
boundaries. They are useful for cold initialization, final reads and shutdown;
placing them between every simulation and render submission serializes the
pipeline. Independent CUDA completion is distinct from Vulkan presentation and
screen scanout. The selected Canvas path is qualified with real Windows
CUDA/Vulkan frames and explicit pending-consumer gates. A 1000-frame three-slot
headed run has no hot CUDA synchronization, CUDA allocation/free, Vulkan explicit
host wait or shared-image growth; repeat launches no pixel kernel. Vulkan still
creates per-submit fence/semaphore/framebuffer metadata and driver submit/present
calls retain their own host cost. This is not a zero-cost presentation or
application throughput claim. The separate headless run checks every version;
the headed run checks actual presentation submission and the final pixel version.

The C equivalents are declared in `taichi_cuda.h`:
`ti_prepare_execution_plan_on_domain_ext`, `ti_initialize_prepared_ext`,
`ti_try_submit_prepared_ext`, and the execution-domain/completion handle APIs.
A successful submit returns an owned completion handle. A false result with no
C API error means busy; a false result with an error means rejection/failure.
Bindings reserve public ticket bookkeeping before device effects.
