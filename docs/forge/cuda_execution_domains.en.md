# CUDA and Vulkan execution domains (development)

[中文](cuda_execution_domains.zh.md)

The independent AOT consumer can bind a supported saved execution plan to a
bounded CUDA or Vulkan execution domain. CUDA owns a nonblocking stream; Vulkan
leases an additional device-created queue, selecting compute by default. Each prepared instance owns
its recordings, argument storage and compiler temporaries.
This API preserves the complete selected plan. Stream handles, priorities and
completion tickets are runtime choices; they are never serialized into AOT.

JIT Graphs can also be prepared as independent instances borrowing selected
compiled code and the current Program's dense Fields. This is part of F3.
Selected Canvas display has Windows CUDA/Vulkan qualification; cross-platform
windows and application FPS remain unqualified. Queue priority is a device
scheduling hint; it does not guarantee
concurrent kernels or priority preemption.

The shared preparation, submission, parameter, timing and retirement APIs cover
both backends. CUDA control/parallel recipes, external storage and Canvas retain
their existing scope. Vulkan accepts fixed buffer compute recipes and the image
routes with explicit ownership described below.

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
| `cuda_submission_parameters_u64` | Declared u64 scalar updates per submission for JIT/AOT independent instances |
| `cuda_completion_gpu_timing` | Preallocated per-submission sampling and nonblocking completion GPU timing |
| `cuda_execution_instance_retirement` | Explicit instance retirement and resident-budget reuse, preserving existing tickets |
| `vulkan_execution_domains` | Additional compute queues with exact timeline dependencies |
| `vulkan_aot_execution_domains` | Saved fixed buffer-only plan preparation and replay |
| `vulkan_jit_graph_execution_domains` | Selected JIT buffer-only dispatch/code loans |
| `vulkan_submission_parameters_u64` | Declared replay-only u64 updates on the selected Vulkan queue |
| `vulkan_completion_gpu_timing` | Preallocated timestamp queries and sparse execution-envelope samples |
| `vulkan_execution_instance_retirement` | Per-instance retirement and resident-budget reuse |
| `vulkan_execution_domain_shared_sampled_images` | Read-only sampling of same-Program managed color Textures across domains |
| `vulkan_execution_domain_private_graphics` | Instance-private color/depth attachments, offscreen drawing and compute consumers |
| `vulkan_execution_domain_live_image_writes` | Exclusive owned Texture loans for flat storage-kernel or described graphics plans |

Use `features.get(name, False)` for older packages. Missing declarations do not
become support because a version or commit looks newer. Device/driver support,
same-GPU sharing and each recipe's compiler/provider facts are checked at cold
creation/preparation. A true build declaration does not qualify Linux windows
or every provider. See [SDK component pairing](aot_sdk.en.md#component-pairing); an
incompatible shim/native pair is rejected before C++ bindings are entered.

## Vulkan queue and recipe boundary

Vulkan domains require enabled timeline semaphores and an unused additional
compute queue in the runtime's selected queue family. Queue zero remains the
ordinary runtime route. Each live domain leases one other queue; two domains
therefore require at least three created queues. Exhaustion fails at cold domain
creation without silently using queue zero. Imported devices must supply the
corresponding queue/timeline facts. Distinct queues need not have distinct GPU
execution engines, so queue availability is not a throughput guarantee.

The ordinary API works on Vulkan too:

```python
# JIT: ti.init(arch=ti.vulkan) has already selected the device.
domain = ti.graph.execution_domain(capacity=2, instance_capacity=2, gpu_timing=True)
step = graph.prepare(arguments, execution_domain=domain, mutable_parameters=names)
step.initialize()
ticket = step.submit(values=metadata, sample_gpu=True)
```

A source-free consumer uses `Runtime("vulkan", device_index=...)` and the same
`runtime.execution_domain()` / `module.prepare_execution()` methods. The device
index is explicit: runtime-only device zero and JIT automatic selection need not
choose the same GPU. AOT u64 kernels must declare `caps=["spirv_has_int64"]` when
exporting. Vulkan priorities are fixed at device creation; use `priority=0` here.

This route reuses selected compiled dispatches and private fixed workspaces.
Qualified public cases include scalar/tensor buffer kernels, explicit adjoint
kernels, initialization/replay, two independent instances, high-bit u64 updates,
sparse timing, completion sets and instance/reset retirement. The
`vulkan_execution_domain_while` feature declares reconstruction of selected device
while recipes, including chunks and nested control; the selected device must
support each strategy. Prefix, initial condition, loop chunks and suffix share
one queue submission. Parameter updates cover every physical dispatch copy, and
timing covers the complete compute sequence. Native buffer operations still
require a compatible fixed-recording factory.

`vulkan_execution_domain_private_images` supports private color images rebuilt
for each prepared instance. Ordinary compute segments accept sampled/storage
images and explicit color uploads. All phases share a fixed boundary layout;
first transitions, initialization and replay run on the selected domain. Resource
creation with initial pixels can still upload during cold preparation. Images
remain owned until instance retirement, with no per-submit layout lookup or
recording rebuild. `vulkan_execution_domain_image_control` additionally supports
these private images in selected device while conditions and bodies, including
chunked and nested recipes. Every recording restores the same boundary layout,
including zero rounds, early exits and exhausted budgets. Consecutive storage
uses also record a memory dependency; sampled/storage aliasing within one task
is rejected during prepare. AOT plans declare private images in `resources`;
see below for shared external Texture/native image loans. Windows tests cover two instances,
downstream explicit adjoints, failed preparation and retirement of pending work.

`vulkan_execution_domain_shared_sampled_images` supports managed color 2D
Textures from the same Program, device and compute queue family, currently with
one mip, one layer and flat kernel dispatch sequences. Multiple domains sample
the same image without copying pixels. Cold preparation retains its existing
resource lease and seals ordinary-queue writes and the shader-read transition
into a reusable timeline prefix. Each domain waits for that prefix before use.
Ordinary sampled reads remain allowed. Uploads, storage writes, layout changes
and Canvas copies requiring transfer-src layout are rejected during the loan.
Previously recorded writers are also rejected, even if they restore the read-only
layout. Permission lasts until instance retirement; completing one submission
does not restore write access. `step.close()` waits for that instance's accepted
work and releases its permission without waiting for other domains. Writes resume
after the last reader retires. Releasing a raw prepared handle does not retire its
instance: pending work and the domain retain permission.

Images with existing raw VkImage aliases cannot enter this route, and new aliases
of the same VkImage cannot be imported during a loan. External-memory aliasing,
external queue-family ownership transfers, shared-image control flow and multiple mips or
layers remain outside its scope. Windows tests cover pending readers on two
queues, retiring one reader, failed preparation, source-view deletion and Program
shutdown. This does not qualify full rendering-scene throughput.

`runtime.execution_domain(queue="graphics")` explicitly selects a graphics+compute
queue family. Identify a queue by both `info["queue_family_index"]` and
`info["queue_index"]`. When compute and graphics share a family, they share the
lease pool and cannot lease one queue twice. The default compute selection is
unchanged; it can also record admitted graphics if its actual family supports
graphics. CUDA rejects `queue="graphics"`.

Fixed flat offscreen plans reconstruct pipelines and color/depth attachments through
the existing native `vulkan_graphics` descriptions. Drawing, compute sampling and
explicit downstream gradients use the same domain. Recordings restore the
instance's image boundary layouts; attachments, pipelines and workspace remain
owned until instance retirement. Warm replay neither rebuilds them nor submits
to the default queue. Initial image data uploads remain cold preparation work.
Flat typed storage-kernel and described `VulkanGraphicsPipeline.record_pass()`
plans can also borrow existing owned 2D color/depth Textures with one mip/layer.
The same allocation remains visible after retirement; no target copy is made.
An instance that writes an image holds an exclusive loan for its entire lifetime:
ordinary reads/writes and other domain loans are rejected until it retires.
Pure sampled inputs retain shared-reader fanout. Cold preparation imports the
ordinary queue's exact preparation prefix; warm execution restores the boundary
layout on the selected domain. `release()` alone does not retire the loan.

Device-geometry `RasterPass` mesh, mesh_instance, particles and lines recordings
can use this route through explicit Graph admission and `queue="graphics"`.
Record creation retains the selected ordinary helper dispatches; preparation
borrows that code and rebuilds private VBO/normal/index scratch for each instance.
Camera/light and draw uniforms are frozen at `Graph.prepare()`. Later camera
changes require closing and preparing the instance again; ordinary recording
execution keeps its existing camera behavior. Call `initialize()` once before
submission. Immutable bytes use the same integrity checks as AOT and upload
synchronously through the default RHI compute queue during cold preparation.
`initialize()` still waits for the ordinary queue's preparation prefix. Warm
drawing and computation use the selected domain without frontend recompilation
or per-submission host geometry uploads.
Windows tests cover all four draw types and downstream sampled-computation
gradients; rasterization itself is not differentiable. Shared mutable geometry
still requires explicit ordering against its writers.

Passes with AS resources, display interop and graphics fork/join need their own
qualification. Raw image aliases, packed depth/stencil and borrowed-image
control flow remain excluded.

Flat AOT `ExecutionPlan` instances can reconstruct private native `vulkan_ray`
`triangle_scene` resources, or `triangle_blas` and `instance_tlas`, on compute
domains. Multiple TLAS instances within one preparation can reference a BLAS;
separate preparations own separate mutable AS and scratch. Factories allocate
and prepare build commands. Explicit `initialize()` executes BLAS then TLAS
builds on the selected domain. Native `refit`, `blas_build`, `instance_refit`,
`instance_update` and `query` can then replay there. Submission parameters can
drive preceding geometry kernels, and query outputs can feed computation and
its explicit adjoint; AS traversal itself is not differentiable. The build
capability is `vulkan_execution_domain_private_ray_resources`.

Private TLAS resources can also bind DSL `acceleration_structure` parameters in
the same flat plan, using compiled closest-hit or candidate-filtered/occlusion
queries. The capability is `vulkan_execution_domain_private_ray_kernels`.
Kernels and native operations share initialization and full BLAS dependency
checks, with no warm resource lookup or extra submission. Standalone and borrowed
runtimes consume this plan; ordinary JIT live AS loans remain a separate capability.

The same flat AOT plan can include `vulkan_ray_program.launch` with fully described
buffers, private sampled/storage images and private TLAS resources. The capability
is `vulkan_execution_domain_ray_programs`. Cold preparation packs the SBT; explicit
initialization uploads it on the selected domain. Operations created while binding
either initialization or replay participate in this ordering. Ordinary and sampled
recording variants share one retained SBT upload allocation, counted in instance
workspace bytes. Warm replay neither repacks nor uploads the SBT. Preparation checks
AS, image and buffer dependencies. Program outputs may feed later computation and
its explicit adjoint; ray tracing itself is not differentiable.

Graphics domains (`queue="graphics"`) can also consume private TLAS bindings in
vertex and fragment shaders within a flat AOT plan. The capability is
`vulkan_execution_domain_private_graphics_as`. Initialization, device transform
updates/refit, rasterization and subsequent computation stay on that domain.
Preparation validates the full BLAS closure and actual command-pool queue
capabilities; build-to-shader barriers follow the selected domain's ordering.
This does not rely on the ordinary runtime's cross-queue graphics semaphore.
The existing private attachment, completion and retirement contracts still apply.

Flat plans can also borrow public AS resources created by `native_resource()`
on the same runtime. The capability is `vulkan_execution_domain_native_ray_loans`.
Native queries, DSL kernels, ray programs and graphics shaders consume the original
AS and its full BLAS closure, without reconstructing a copy from seed geometry.
Preparation reserves the complete access set atomically: queries share reads,
while build/refit/update exclusively borrow the AS they actually write. Updating
a TLAS only reads its BLAS dependencies, so writers of two different TLAS resources
may share one BLAS, but cannot concurrently rebuild that BLAS. Existing ordinary
recordings check their permissions on submission and reject conflicts before
device effects.

Loans last until every accepted invocation of the instance retires. Closing one
reader does not release another reader's access. Returning a writer submits a
device dependency to the ordinary queue for subsequent access visibility. This
is a cold retirement cost; warm replay stays on the selected domain without
per-invocation resource-registry lookups. Standalone and borrowed runtimes support
these native-created AS resources. A borrowed runtime can also explicitly import
an original Vulkan AS from its Program:

```python
runtime = ti.aot.borrow_runtime()
scene = runtime.import_acceleration_structure(jit_scene)
```

The method accepts same-Program `TriangleScene`, `TriangleBLAS` and `InstanceTLAS`
objects and returns a `NativeResource` for the AOT bindings above. It retains the
original object without geometry uploads or AS reconstruction. TLAS imports keep
ordered BLAS inputs, including repeated references. Separate imports of the same
object satisfy resource alias constraints; identical geometry in different objects
does not. Importing alone grants no concurrent access: preparation acquires the loan.
Closing a source wrapper leaves retained imports alive. Closing an import still
used by an execution instance closes the affected domain through the existing
native-resource retirement path. A different domain using only another import
remains usable. Close instances before their imports to keep a domain available
for new preparations. Private input handles retire with their import; Program
reset retires every borrower.

For qualified flat selected Graphs, `Graph.prepare(..., execution_domain=domain)`
also borrows original AS resources referenced by native query/refit recordings
and DSL TLAS parameters. Native and kernel references to the same object share
one import per instance. Preparation borrows the selected compiled dispatches;
it does not recompile kernels or reconstruct geometry. Source Graph/AS wrappers
may close after preparation; the prepared instance retains its dependencies.
Closing the instance releases its loans before its imports, so ordinary queries
can observe refits made to the original AS. Program reset retires the instances.
Graphics/ray-program combinations with these JIT AS loans, parallel regions and
control-flow combinations still require separate qualification.

A selected recipe may contain only native operations: it needs no placeholder
kernel. Fixed ndarray bindings retained by a recording are live storage loans.
For example, persistent prepared sort recordings work on CUDA and Vulkan without
recompiling a kernel; per-instance execution workspace retains its existing policy.

Seed inputs retained for private reconstruction and AS dependencies remain valid
until instance retirement; closing them early invalidates the instance. Wrapping a public BLAS in a
private TLAS does not make it private: its loan still needs complete admission.
Micromaps and AS operations inside control flow remain outside this capability.
Ordinary synchronous resource creation keeps its contract.
Cold `initial_data` uploads still use the default queue and `initialize()` waits
for bootstrap; cold preparation is not fully independent of that queue. Warm
native queries and updates do not return to it.

Host selection, public parallel recipes and shared sparse/RNG/runtime-error
state remain outside this independent Vulkan route. Ordinary Graph execution keeps
its existing capabilities; CUDA Canvas and external-memory protocols are not
implicitly enabled on Vulkan domains.

Initialization imports the ordinary queue's submitted preparation prefix and
records instance initialization on the selected queue. Warm submission combines
dependencies, parameter upload and fixed commands into one `vkQueueSubmit`.
Busy admission changes none of these, and instance retirement waits for that
instance's accepted prefix rather than unrelated default-queue work.

Windows qualification uses an RTX 5090 with 16 created compute queues, including
controlled pending-work tests in both directions. A device lacking enough queues
does not qualify merely because the build reports the features. Linux, arbitrary
provider combinations and application throughput have not been qualified here.

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
step.close()    # cold: retire this instance and reclaim its resident budget
domain.close()  # cold: drain and retire resources
```

`arguments` accepts a dictionary or this Graph's `GraphBindingSet`. Preparation
freezes its current values; later binding-set updates do not change an existing
instance. It consumes the complete physical dispatch, control and parallel
description, preserving each invocation's launch policy. It does not export an
artifact, recompile the frontend or copy Field roots. Separate preparations own
independent arguments, control buffers and declared private workspace. Shared
application ndarrays and Fields still require exact dependencies.

The CUDA scope includes scalar/tensor JIT ndarrays, compact AOS
`ti.experimental.ndarray_view()` ranges over live dense Fields/ndarrays,
implicit dense Field access, ordinary and masked control, and selected CUDA parallel regions.
Existing ndarray `.grad` storage is borrowed for explicit adjoint kernels.
Automatic Tape/FwdMode is not supported; initialization/submission rejects that
context and concurrent AD entry. Closing the source Graph leaves prepared
instances usable. Field-tree destruction or Program reset retires dependent
instances before releasing code or storage.

On CUDA, fixed sequential Graphs also accept nested ArgPacks. Preparation freezes actual
device values into private pack storage; ndarray members retain their existing loan
contract. Sampled Texture arguments and packed members borrow each source image once,
without copying pixels. Initialize images first and explicitly order pixel updates
against sampling. Checks cover producer Graph/pack release and independent close in
two domains; other pack-bearing control recipes require separate qualification.

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

The independent AOT runtime also provides [Vulkan external buffer import](vulkan_external_buffers.en.md)
with native readiness and whole-buffer ownership on the selected domain. See its
separate initialization and public-entry boundaries; the CUDA completed-signal
rule does not apply to the Vulkan gate.

## Per-submission parameters

Declare mutable scalars during cold preparation. Other bindings stay frozen:

```python
names = ("epoch", "simulation_step", "timestamp_ns")
step = graph.prepare(arguments, execution_domain=domain, mutable_parameters=names)
step.initialize()
ticket = step.submit(values={"epoch": epoch, "simulation_step": step_id,
                             "timestamp_ns": timestamp}, wait_for=previous_tickets)
```

`module.prepare_execution(..., mutable_parameters=names)` has the same AOT
contract. This first implementation accepts replay-only `u64` kernel scalars.
Scalars also used in initialization, arrays and other types are rejected during
preparation. Names and count are caller-defined. Every submit requires all
declared values in `[0, 2**64)`. Mutating the original dictionary after acceptance
cannot change accepted work. Each instance owns preallocated staging and device
arguments. Uploads and the complete Graph use the selected queue, and
the completion covers both.
CUDA uploads use a small GPU kernel reading pinned storage to avoid cross-stream
head-of-line blocking on the DMA engine. Preparation requires CUDA unified
addressing; replay never allocates pinned memory.
Vulkan uses prepared staging and buffer copies with explicit transfer/compute
dependencies on its leased queue. No upload is redirected to the ordinary queue.

CUDA's mutable argument frame permits one in-flight submission per instance;
another returns `None`. Vulkan preallocates staging per instance and queue slot,
so accepted submissions of the same instance remain ordered on its queue without
overwriting earlier snapshots. Prepare independent instances for separate mutable
workspace and configure domain capacity as needed. Busy and unready calls enqueue no upload, consume no
external signal and allocate no submission sequence. Replay does not prepare,
allocate device parameter storage or synchronize globally. CUDA and the fixed
buffer-only Vulkan route share this contract.

## Completion GPU timing

Timing is off by default. Create the domain with `gpu_timing=True` to allocate
one timestamp pair per queue slot, then select submissions with `sample_gpu=True`:

```python
domain = ti.graph.execution_domain(capacity=2, gpu_timing=True)
step = graph.prepare(arguments, execution_domain=domain, mutable_parameters=names)
step.initialize()
ticket = step.submit(values=metadata, sample_gpu=sample_this_step)
if ticket is not None:
    measurement = ticket.gpu_timing
```

The result contains `status`, `scope`, `owner`, `sequence` and `duration_ns`.
Only `ready` has a duration; `not_sampled`, `pending`, `unsupported` and `failed`
return `None`. Reading performs a nonblocking poll without a submission lock.
The result belongs to that exact ticket and survives event reuse and domain close.

`execution_envelope` spans the complete execution, excluding predecessor waits,
external acquire, parameter uploads and external release. It includes gaps and
waits inside the selected Graph/providers, so it is not a sum of kernel times.
Sampling does not split the Graph or change its domain. AOT uses the same API.
This covers independent CUDA and Vulkan domains. Vulkan timestamps require support
in the selected queue family; unsupported samples report `unsupported` without a duration.
Existing ordinary Vulkan Graph timing retains its own contract.

## Selected Canvas display domain

Bind the current Program's CUDA domain with `canvas.set_execution_domain(output)`.
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

### Vulkan retained packed copy

A Vulkan Program can select an independent output domain with
`canvas.set_execution_domain(output)`. This path accepts same-Program 2D `u32`
ndarrays. Cold-prepare the fixed source set and display slots with
`canvas.prepare_frame_sources(frames, capacity=3)`. The caller chooses capacity;
at least two slots retain the last image while accepting an update. There is no
additional source-count or slot-count cap, but allocation, addressing and device
storage-buffer limits still apply. All sources must have the same shape.

```python
frame = ti.ui.DisplayFrame.from_packed_u32_ndarray(pixels)
canvas.set_execution_domain(output)
canvas.prepare_frame_sources([frame], capacity=3)  # Cold preparation

producer = writer.submit()  # The output domain or a same-Program peer domain
if producer is not None:
    display = canvas.submit_frame(
        frame, source_completion=producer, track_source=True, track_completion=True
    )
    if display is not None:
        copied = frame.source_completion
        window.show()
```

An accepted frame performs one device buffer copy on the selected output domain.
Graphics waits for that copy's exact timeline point. `copied.done()/wait()` covers
source-read completion: respect this dependency before reusing or overwriting
the source. This native ticket is not currently a public Graph `wait_for` input.
`display` covers graphics consumption, not presentation. Repeat uses the retained
display slot without another source copy. Busy returns `None`, enqueues no copy
and leaves `frame.source_completion` unchanged. Cancelling a pending display does
not cancel an accepted copy; its slot and source leases remain until completion.

The selected Vulkan domain supports prepared packed copies and the prepared
writable path below. Mixed geometry, texture/host input
and runtime offscreen attachments reject at their respective entry points.
Window destruction retires its copy instance without closing the entire output
domain. A window that has used only this selected Vulkan image path waits for its
own copy and graphics readers, without draining unrelated default-compute or output
tails. A window with any earlier general display work retains conservative device
draining. Visible windows still perform required swapchain/present retirement;
physical queue sharing can therefore impose additional waits. Readback, reconfiguration
and Program reset retain their existing synchronization. Missing retirement proof
retains the window resource closure instead of freeing potentially in-flight storage.
For changed source sets or shapes, close the old domain, bind a new one and
prepare again. This adds no pixel-processing kernel, but adds one device copy
compared with writing display storage directly; it is not an application frame-rate
claim.

### Vulkan prepared writable display slots

After selecting a Vulkan output domain, cold-prepare local display slots with
`canvas.prepare_frame_pool(width, height, capacity=3)` and prepare a Graph/AOT
writer for each returned physical view. Use the CUDA pool example above, but
always supply the written Vulkan frame's exact writer `source_completion`, even
when tracking is disabled. Writers must use that same output domain; ordinary
kernel/default launches cannot write these views. Each acquired lease accepts
one complete prepared recipe, which can contain multiple dispatches and aliases
of the slot. Wrong, later or stale completions reject before sealing and leave
the current lease available for retry.

This writes display storage directly, avoiding the extra packed-copy transfer.
Graphics waits for the exact writer; a slot can be reused only after its writer
and every graphics reader complete. Cancellation does not cancel accepted work.
Repeat submits no writer or empty marker. An unwritten frame may omit its
producer: the initial contents are cold-zeroed; later empty frames retain that
slot's previous contents. Empty submission still obeys output-domain capacity;
Busy does not consume its lease. The caller chooses capacity, with at least two
slots to retain the last image. Only allocation, addressing, device ranges and
shader indexing impose limits; there is no extra slot cap or hot pool growth.

Window close revokes only instances consuming these slots, leaving unrelated
instances on the domain usable. For resize, close the old domain, bind a new one
and prepare again. Views/tickets surviving reset retain retired safe objects,
not permission to write. Packed-copy and writable pools are separate cold modes;
storage and mode cannot change on a live bound domain. This capability does not
qualify the full downstream application GUI route or establish application frame-rate gains.

### Images with GUI widgets

A selected Vulkan display domain can include its window's GUI widgets. The first
`window.get_gui()` cold-prepares fonts and compatible draw resources and may wait
for the graphics queue; call it before the steady loop or controlled queue waits.
Repeated calls with the same resources preserve widgets already added this frame.
Resize or renderpass changes return to cold preparation. `repeat_frame()` reuses
the image; the application still describes widgets for each frame.
For a visible window, enqueue the image with `submit_frame()` or `repeat_frame()`
before adding that frame's widgets, then call `show()`.

Each GUI vertex/index slot retains the exact graphics reader from its submission.
Pending slots are neither overwritten nor grown. If no slot is available,
nonblocking `show()` returns `False` and drops the pending display frame. Blocking
readback waits for a required reader. Ordinary windows and existing ordinary
offscreen GUI share this rule. Geometry and runtime offscreen targets remain
restricted on selected domains; this does not qualify the full downstream application route.

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
    step.close()      # cold: retire this instance and reclaim its resident budget
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

Runtime domain registration grows during cold creation and reuses closed-domain
slots; it has no fixed domain-count limit. Creating streams and their storage
can still fail when backend or host resources are exhausted.

`capacity` bounds pending submissions; `instance_capacity` bounds resident logical
prepared instances. Initialization and replay share one instance reservation while
retaining their own recordings. A partially built instance retires its established
parts through the same owner. A capacity rejection, or failed construction with
proven cleanup before admission, removes only that failed instance; the domain and
resident peers stay usable. Failures after admission or without safe cleanup still
use cold domain retirement.

`step.close()` stops only this instance, waits for its last accepted submission,
retires its recordings and removes its storage/dependency edges. It returns the
resident instance reservation for another preparation; existing completion
tickets remain usable. Waiting does not hold the runtime submission lock.
Driver/provider destruction can still synchronize, so close belongs at a cold
boundary with external producers able to signal pending waits.

CUDA submission-parameter staging returns to a domain-owned pinned pool after its
last device read completes. Later preparations reuse idle mappings; an idle
mapping that is too small is replaced during cold preparation. The pool retains
its peak concurrent storage until domain close, avoiding context-wide unpinning
on each instance retirement. This does not change parameter upload or replay.

`step.release()` and garbage collection release only the host handle, keeping
code, arrays and reverse dependency edges until domain retirement. This path
does not reclaim the resident instance reservation. Explicit memory destruction
retires the instances that borrow that allocation before freeing its storage;
the domain and unrelated instances remain available. Explicit module or
native-resource destruction closes dependent domains. Runtime close/reset retires
its domains. A failed retirement retains
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

Backend creation/preparation APIs are declared in `taichi_cuda.h` and
`taichi_vulkan.h`, including `ti_prepare_execution_plan_on_domain_ext` and
`ti_prepare_execution_plan_on_vulkan_domain_ext`. Common initialization,
submission, completion and retirement APIs are declared in `taichi_core.h`.
A successful submit returns an owned completion handle. A false result with no
C API error means busy; a false result with an error means rejection/failure.
Bindings reserve public ticket bookkeeping before device effects.
