# Forge Low-Level Hardware Graphics Execution Plan

Status: in progress  
Target: 0.6.3 development branch  
Distribution constraint: no new official-wheel backend, CUDA/Vulkan-version,
or platform combination

## 1. Decision

Forge does not implement a renderer. Forge provides low-level hardware
resources, pipelines, and commands that renderers, physics engines, and
visualization systems can compose: images and attachments, samplers, graphics
pipelines, draws, buffer/image synchronization, acceleration-structure
build/refit/query, and their direct-execution and root-Graph ordering and
lifetime contracts.

Camera, light, material, particle-billboard, mesh-scene, PBR, visibility
policy, render-graph policy, and frame scheduling therefore remain outside
this interface. The existing `ti.hardware.raster.RasterPass` remains only a
compatibility and qualification convenience adapter. It is no longer the
target abstraction for Forge graphics and must not pull GGUI scene semantics
into the low-level API.

## 2. Automatic acceleration versus manual hardware interfaces

The API, report, and performance claims must distinguish these categories.

| Class | Initiator | Kernel-callable | Current or target examples |
| --- | --- | --- | --- |
| Compiler automatic lowering | A kernel explicitly contains a typed operation with hardware semantics; the backend selects an instruction | Yes | texture `fetch`/`sample_lod`, atomics, subgroups, shared memory |
| Domain API automatic provider selection | The user requests a domain operation; its implementation selects by backend and cost | No, unless a separate kernel intrinsic exists | `SparseMatrix @ ndarray` selecting cuSPARSE |
| Explicit kernel intrinsic | The kernel author explicitly requests a qualified hardware semantic | Yes | future typed cooperative matrix or inline Ray Query |
| Explicit hardware resource/command | The user creates resources, records, and submits native commands | No; Python/direct or Graph boundary only | graphics pipeline/draw, AS build/refit, batch Ray Query, cuFFT plan |

Ordinary field/ndarray loads, ordinary matrix multiplication, software
rasterization, and software ray tracing must never be silently rewritten to
these interfaces. Graph `admission="auto"` only determines whether an already
explicit command may enter a Graph; it does not automatically select it.

## 3. Layer placement

```text
application / renderer / physics engine
    camera, material, geometry policy, pass scheduling, cost model
                         |
ti.hardware resource + command layer
    Image/Attachment, Sampler, GraphicsPipeline, DrawRecording, AS resource
                         |
ti.graph NativeAction + runtime resource registry
    effects, runtime bindings, generation leases, replay, completion
                         |
Program / GfxRuntime queue bridge
    compute -> graphics -> compute ordering, image layout, barriers
                         |
RHI / Vulkan
    VkImage, VkSampler, graphics pipeline, render pass, draw, AS commands
```

Only the first and third categories in section 2 belong to kernel frontend and
code generation. Graphics pipeline/draw, AS build/refit, and batch query are
command-buffer operations and cannot be invoked inside a Taichi kernel. A
kernel may access an explicitly shared resource before or after the command;
Graph/native effects and the runtime queue bridge establish the ordering.

## 4. Official-wheel and dependency boundary

- D0 uses only the RHI, Vulkan loader, driver APIs, and embedded SPIR-V already
  present in the wheel. Runtime does not invoke `glslc`, DXC, NVRTC, or a
  Vulkan SDK, and links no new shared library.
- The public graphics pipeline accepts SPIR-V bytes or words. Shader
  compilation is an application or build step, not a wheel runtime dependency.
- D1 vendor libraries remain user-optional, dynamically probed, and
  operation-scoped in failure. They are eligible only when Forge need not
  produce CUDA/Vulkan-version-specific wheels.
- D2/D3 routes requiring SDK headers, versioned ABI compilation, or another
  wheel tag are limited to source builds, plugins, or application integration
  in this execution cycle.

## 5. P0 public interface slice

The first deliverable uses Vulkan-specific names to avoid a portability claim
for unimplemented CUDA and Metal backends:

```python
pipeline = ti.hardware.graphics.VulkanGraphicsPipeline(
    vertex_spirv=vertex_bytes,
    fragment_spirv=fragment_bytes,
    vertex_bindings=(
        ti.hardware.graphics.VertexBinding(0, stride=20),
    ),
    vertex_attributes=(
        ti.hardware.graphics.VertexAttribute(0, 0, ti.Format.rg32f, 0),
        ti.hardware.graphics.VertexAttribute(1, 0, ti.Format.rgb32f, 8),
    ),
    topology="triangles",
    depth_test=False,
)

recording = pipeline.record(
    color="color",
    vertex_buffers={0: "vertices"},
    draw=ti.hardware.graphics.Draw(element_count=3),
    clear_color=(0, 0, 0, 1),
)
recording.execute({"color": color_texture, "vertices": vertex_ndarray})

builder = ti.graph.GraphBuilder()
builder.append_native(recording, admission="auto")
graph = builder.compile()
graph.run({"color": color_texture, "vertices": vertex_ndarray})
```

P0 promises one color attachment, an optional depth attachment, one or more
vertex buffers, an optional `i32/u32` index buffer, direct/indexed/instanced
draws, explicit clear, viewport/scissor, and a final image layout. It contains
no scene, camera, material, shader reflection, shader compiler, descriptor
graph, indirect draw, or swapchain.

Every binding belongs to the current Program. Pipelines and recordings capture
the runtime generation; ndarray and texture leases survive until submission
completion. A closed pipeline, runtime reset, wrong backend/device,
out-of-range draw, missing attachment, non-four-byte SPIR-V, or incompatible
vertex/index usage fails before submission.

## 6. Queue, Graph, and image-layout contract

A Vulkan device may place compute and graphics in different queue families.
P0 must not assume they coincide:

1. flush the current runtime compute command list and obtain its completion
   semaphore;
2. make the graphics stream wait on that semaphore and record attachment
   transitions, the render pass, and draws;
3. obtain the graphics submission semaphore;
4. submit a compute-stream completion bridge that waits on graphics; and
5. order existing runtime completion, resource-lease retirement, and later
   kernels after that bridge.

Buffer and image allocations already use concurrent sharing when queue
families differ. The bridge performs no implicit host wait. Direct execution
and root-Graph replay both rerecord commands; neither may be described as a
Vulkan command-buffer cache or Graph fusion.

Attachments start with `undefined -> attachment` by default, making clear the
deterministic P0 semantic. P0 does not offer load/preserve. Color transitions
to the caller-declared `shader_read`, `shader_write`, or `transfer_src` final
layout. Depth remains pass-local by default; exposing it to a later kernel
requires a qualified depth-format and final-layout contract.

## 7. P1/P2 closure

### P1: image commands and samplers

- Land whole-color-image copy first with exact source/destination effects,
  runtime Texture leases, matching format/extent validation, and automatic
  root-Graph admission. Offset/region copy, buffer-to-image, image-to-buffer,
  blit, and raw transition interfaces remain deferred until bounds, format,
  and externally observable layout contracts can be closed without exposing
  unsafe backend state.
- Replace the empty RHI `ImageSamplerConfig` first with immutable min/mag
  filter and per-axis address state owned by a Vulkan sampler cache. The
  current one-mip texture contract keeps normalized coordinates; mipmap,
  anisotropy, and compare controls remain deferred until their image/resource,
  device-feature, and typed-operation contracts exist.
- Sampler configuration is texture-binding/resource semantics. `fetch()` does
  not filter; `sample_lod()` uses the explicit sampler. Buffer loads are never
  automatically changed to texture loads.
- Exact fetch has a stable negative performance result. P1 targets correct
  filtering/addressing/SDF/volume semantics and cache behavior, not a generic
  load-speedup claim.

### P1: generalized AS resources

- Split the current one-mesh, one-identity-instance `TriangleScene` into
  low-level BLAS and TLAS resources.
- Expose build/update flags, instance transform/mask/id, multiple BLAS
  instances, and barycentric/primitive/instance hit data.
- Build/refit/query stay command-scoped. Kernel-inline Ray Query requires a
  new AS argument type, effect/lifetime binding, typed IR, and SPIR-V lowering
  and remains an independent P2 item.

The implementation audit found that the current provider allocates private
geometry copies, one BLAS, one identity-instance TLAS, build/refit scratch,
and an embedded batch-query pipeline as one indivisible resource. A truthful
BLAS/TLAS split therefore cannot reuse `TriangleScene` handles as a public
low-level abstraction. It remains explicitly deferred until independently
owned BLAS/TLAS resources, instance updates, query descriptor binding, and
hit-schema tests can land together; the qualified `TriangleScene` provider is
retained unchanged.

### P2: explicit deferrals

- descriptor reflection/bindless, indirect/multi-draw, and mesh/task shaders;
- Vulkan cooperative matrix and kernel-inline Ray Query;
- the complete CUDA texture-object resource/ABI/lowering chain; and
- OptiX, DPX, public TMA/WGMMA, or any provider requiring new wheel variants.

## 8. Execution and commit boundaries

| Commit | Content | Qualification gate |
| --- | --- | --- |
| A | Bilingual plan, corrected RasterPass placement, catalog terminology | document parity and catalog tests |
| B | GfxRuntime cross-queue graphics submission bridge | queue-order, no-host-sync, completion/lifetime tests |
| C | Vulkan graphics pipeline/draw C++ resource and Python/Graph API | validation, real color/depth, direct + Graph, reset/device mismatch |
| D | Feasible minimum image/sampler slice | format/filter/address correctness, sampler cache/lifetime, unchanged wheel dependencies |
| E | Feasible AS generalization or explicit deferral record | build/refit/query correctness, memory report, dynamic-scene stability |
| F | RasterPass compatibility placement, API reference, release notes | existing compatibility tests and a new low-level example |
| G | Memory, performance, and stability qualification report | fresh-process AB/BA, route gate, raw artifact, retained negative results |

Each commit contains only its row's files and does not absorb pre-existing
user changes. If an implementation cannot meet correctness, lifetime, or
distribution gates, it remains planned with its missing chain documented;
Forge does not publish a partial public API.

## 9. Validation and completion criteria

- Correctness: known-triangle color/depth, indexed/instanced draw, viewport,
  clear, and visibility to kernels before/after Graph; no validation-layer
  errors.
- Lifetime: pipeline/recording/texture/ndarray close, runtime reset, Graph
  replay, and exception paths.
- Memory: requested bytes return after repeated create/close; driver-opaque
  state stays separate; long loops record RSS, device allocation, and
  in-flight-command high-water marks.
- Performance: equivalent software/hardware comparisons use fresh-process
  AB/BA and report CPU submit, GPU completion, queue-bridge cost, and repeated-
  draw amortization separately.
- Stability: CV, order drift, and cold/warm separation. No-speedup, regression,
  and cross-run instability remain results and cannot be hidden behind an
  inequivalent baseline.
- Distribution: official-wheel build switches, dynamic dependencies, and
  wheel tags are unchanged.

Completion means that an external renderer can issue one real hardware draw
using only low-level resources and commands, with auditable kernel/Graph
ordering, resource lifetime, memory, and distribution evidence. It does not
mean publishing a large enumeration of vendor instructions.
