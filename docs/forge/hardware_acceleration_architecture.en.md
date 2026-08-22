# Hardware Acceleration Architecture and Execution Plan

> Status: architecture contract for the Taichi Forge `0.6.3` development
> line. This document defines implementation and release boundaries. It is
> not a statement that every listed operation is already implemented or
> qualified.

Taichi Forge may use matrix engines, texture units, rasterizers, ray-tracing
hardware, and vendor GPU libraries without turning the distribution into a
matrix of CUDA, Vulkan SDK, OptiX, or vendor-specific wheels. The public API
describes stable operation semantics. Backend providers select instructions,
fixed-function commands, or external runtimes only after explicit capability
qualification.

The governing decision is:

> Optional dependency and hardware acceleration are orthogonal properties.

A provider can use dedicated hardware without adding a runtime dependency,
and a dynamically loaded vendor library does not by itself prove use of a
dedicated hardware unit.

## Goals and non-goals

The architecture has five goals:

1. expose a small set of stable operations useful to simulation, rendering,
   and physical solvers;
2. distinguish transparent compiler optimization from explicit hardware
   resources and executable operations;
3. preserve the existing driver-only CUDA plus Vulkan official wheel;
4. reuse Forge Graph, runtime-storage, generation, lifetime, and temporary
   ownership contracts; and
5. report the selected route precisely enough to qualify correctness and
   performance without inferring hardware use from an API name.

The first version does not:

- publish separate Forge wheels for CUDA toolkit, Vulkan SDK, OptiX ABI, GPU
  generation, or vendor;
- expose vendor instruction names such as WGMMA, TMA, `tcgen05`, or mesh
  shader as the stable semantic API;
- replace ordinary arithmetic with lower-precision matrix operations unless
  the user selected a compatible numeric contract;
- call a host library or dynamic loader from inside an arbitrary Taichi
  kernel;
- label an ordinary compute fallback as dedicated hardware acceleration; or
- turn ray traversal into a general collision-detection claim.

## Distribution boundary

Dependency tier describes deployment only. It does not classify the algorithm
or the hardware mechanism.

| Tier | Contract | Examples | Official-wheel policy |
| --- | --- | --- | --- |
| D0 `core` | Uses only the backend facilities already present in the official runtime. | PTX, SPIR-V, Vulkan extensions, CUDA Driver API, CUDA-Vulkan interop. | Supported by the existing wheel when the runtime device qualifies. |
| D1 `lazy_external` | The same wheel resolves an additional driver component or shared-library ABI at runtime. | OptiX, cuBLAS, cuSPARSE, cuSOLVER, cuFFT. | Allowed only when absence is harmless and loading is lazy and fault-isolated. |
| D2 `build_external` | Headers, templates, a device compiler, or an SDK-bound build are required. | CUB/CCCL and user-built SDK plugins. | Source-build, reference, or external-plugin use only; disabled in official wheel jobs. |
| D3 `wheel_variant` | Support requires another official Forge binary variant. | `forge-cu12`, `forge-cu13`, `forge-optix8`, or vendor-specific wheels. | Prohibited. |

The official runtime build must retain all of these invariants:

- the number and tags of official wheels do not increase because of a
  hardware provider;
- CUDA Toolkit, CUB reference, and CUPTI build switches remain disabled;
- no D1 vendor library becomes an import-time or link-time dependency of the
  Python package;
- missing D1 components do not break `import taichi_forge`, `ti.init()`, or a
  D0 operation; and
- wheel validation rejects an undeclared bundled vendor runtime.

D2 support can have dedicated source-build CI. That CI is evidence for the
reference or plugin path, not evidence that the official wheel contains it.

## Orthogonal provider classification

Every provider operation is described along four independent axes.

### Provider class

| Provider class | Meaning |
| --- | --- |
| `hardware_intrinsic` | Typed device operation lowered to PTX or SPIR-V, such as matrix MMA, atomic, or subgroup operations. |
| `fixed_function` | A graphics or traversal facility such as rasterization or qualified hardware ray traversal. |
| `vendor_hardware_runtime` | A vendor execution runtime whose contract is hardware-oriented, such as OptiX. |
| `vendor_algorithm` | A vendor algorithm library such as cuBLAS, cuSPARSE, cuSOLVER, cuFFT, or CUB. |
| `compute_native` | A Forge-owned optimized CUDA, Vulkan, or CPU implementation. |
| `compute_fallback` | A portable fallback that preserves the public semantic contract. |
| `runtime_interop` | A memory and synchronization bridge between runtimes. |

### Execution class and hardware qualification

`execution_class` describes what Forge submits:

```text
hardware_instruction
native_shader_operation
fixed_function
vendor_hardware_runtime
native_command
vendor_library
compute_kernel
```

`hardware_acceleration` describes the strength of the hardware claim:

```text
guaranteed
qualified
implementation_defined
none
```

For example, Vulkan acceleration-structure build is a `native_command`, but
the Vulkan contract does not guarantee a dedicated BVH-build unit. Its
hardware qualification is therefore `implementation_defined`. A Forge radix
sort remains a `compute_kernel` with `hardware_acceleration=none`, even when it
is much faster than a host implementation.

### Execution and Graph contract

Execution integration is reported per operation, not once per provider:

```text
execution_kind:
    kernel_intrinsic | native_command | external_library | compute_kernel

graph_support:
    inline | recordable | stream_capture | opaque | unsupported

stream_binding:
    runtime_ordered | current | explicit

workspace_ownership:
    none | caller_owned | provider_owned | graph_owned
```

An external library being generally compatible with CUDA Graph does not make
every plan, callback mode, or operation recordable.

## Automatic and explicit invocation

There are two separate choices: how the semantic operation is requested and
how a provider is selected.

| Invocation mode | User contract | Provider behavior | Examples |
| --- | --- | --- | --- |
| Transparent optimization | No new public call. Existing semantics must be preserved. | Compiler or provider selects a device mechanism automatically. | Grouped reductions, opt-in Vulkan list-generation ballot aggregation, qualified `cp.async`, and future mesh-shader specialization. |
| Explicit kernel semantic | The kernel calls a typed operation or compiler hint. | Code generation chooses an admitted intrinsic implementation. | Atomics, warp/subgroup operations, shared memory, block-local caching, texture sampling, future typed Matrix MMA, and future inline Vulkan ray query. |
| Explicit resource/executable | Python code creates a resource or executable and runs or records it. | Runtime manages native commands, stream order, workspace, and lifetime. | Current CUDA Matrix MMA, raster pass, AS build/refit, batch ray query, OptiX launch. |
| Optional algorithm provider | A domain API names `auto`, `builtin`, or a provider. | A D1 library is considered only after opt-in. | cuBLAS, cuSPARSE, cuSOLVER, cuFFT. |

The first Matrix MMA API is explicit. Forge must not silently replace an
ordinary matrix multiply with MMA until it can prove that the selected dtype,
accumulation, rounding, determinism, and error contract are compatible.

Transparent optimization can report the chosen mechanism for diagnostics, but
the implementation mechanism is not a compatibility promise.

### Existing core kernel routes

The catalog now records eight existing D0 route entries without creating new
public syntax. Six entries cover five explicit kernel semantic families:
CUDA/Vulkan atomics, CUDA warp operations, the implemented Vulkan subgroup
subset, CUDA/Vulkan `SharedArray`, and CUDA `ti.block_local`. They are inline
kernel lowering, not
Python native actions. The supported dtype and operation slice remains
backend- and capability-dependent; the catalog therefore reports
`hardware_acceleration=implementation_defined` instead of inferring one exact
instruction from the API name. The current `ti.block_local` qualification is
limited to supported gather/read-cache patterns. Sparse pointer-SNode
scatter/write-back is explicitly excluded because its existing CUDA
correctness test fails reproducibly.

Two routes are automatic and internal: compiler-recognized reductions can
aggregate within a CUDA block or Vulkan subgroup before publishing fewer
global atomics, and opt-in Vulkan list generation can reserve one contiguous
range per active subgroup via ballot/elect/broadcast. Unsupported reduction
patterns retain ordinary atomics; list generation retains the legacy
per-active-lane atomic path unless both the option and subgroup-ballot feature
are present. The explicit `ti.block_local` semantic can subsequently select the
automatic `internal.tile.async.cuda` specialization described below. Thus one
kernel can contain an explicit semantic while its final hardware mechanism is
chosen automatically.

The first qualified transparent specialization is
`internal.tile.async.cuda`. It does not add a kernel call. A CUDA kernel that
already requests the backend-neutral `ti.block_local` caching semantic may
lower the compiler-generated global-to-block-local prologue to PTX
`cp.async` only when all of these gates pass:

- the target is NVIDIA compute capability 8.0 or newer and PTX ISA 7.0 or
  newer;
- the struct-for block-local allocation is at least 8 KiB; and
- the IR site is a direct primitive 4-, 8-, or 16-byte global-to-BLS copy; and
- the block-local cache is read-only and has no write-back epilogue.

Every other site, including scatter/read-write BLS, retains the existing
synchronous load/store lowering. The
asynchronous group completes before the existing block barrier, so the body
observes the same block-local values. A Graph that calls such a compiled
kernel simply inherits its kernel PTX; this is not a separately recordable
native command. `ti.hardware.report()` reports the provider as `eligible`
before a qualifying kernel is compiled and `selected` afterward, together
with Program-generation lowering and copy-site counters. These counters are
diagnostics, not a stable instruction-selection promise. Reusing an offline
cached executable does not perform a new lowering and therefore need not
advance these current-Program counters.

The 8 KiB workload floor is evidence-based and intentionally narrow. On the
2026-08-23 qualification host (RTX 5090, compute capability 12.0, driver
610.62), an eight-f32-field 16-by-16 block-local workload produced three
fresh-process median pairs of 53.167--55.105 microseconds for synchronous
lowering and 45.333--45.810 microseconds for `cp.async` lowering. The
median-of-medians reduction was 15.1 percent; a four-field workload did not
clear the noise gate and therefore remains synchronous. This is route and
admission evidence for the bounded workload, not a general kernel-speed claim.
The instruction and completion requirements follow the
[NVIDIA PTX ISA asynchronous-copy contract](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-asynchronous-copy).

Vulkan mesh shaders remain `planned`. The repository currently has generated
Vulkan header declarations but no active `VK_EXT_mesh_shader` feature chain,
SPIR-V mesh-stage lowering, mesh pipeline construction, or
`vkCmdDrawMeshTasksEXT` command route. A device extension bit alone cannot
select this provider. The Vulkan specification requires both querying and
enabling `VkPhysicalDeviceMeshShaderFeaturesEXT`; see the
[ratified Vulkan specification](https://registry.khronos.org/vulkan/specs/latest-ratified/pdf/vkspec.pdf).

## Public API ownership

The `ti.hardware` module owns normalized hardware capability discovery and new
hardware resource/executable families. Existing domain APIs stay in their
current modules.

| Semantic family | Public owner | Scope |
| --- | --- | --- |
| Capability and provider report | `ti.hardware` | Python scope. |
| Matrix MMA | `ti.hardware.matrix` | The first CUDA slice is Python and Graph scoped; a typed kernel intrinsic remains planned. |
| Rasterization | `ti.hardware.raster` | Python and Graph scope. |
| Acceleration structures and batch ray query | `ti.hardware.ray` | Python and Graph scope; selected inline query operations may also be kernel scoped. |
| Texture and sampling | Existing `ti.Texture` and texture argument types | Kernel scope; reflected by `ti.hardware` reports without duplicating the API. |
| Dense, sparse, solver, and FFT algorithms | Existing `ti.linalg` or `ti.algorithms` entry points | Python and Graph scope, with provider selection on the domain operation. |
| CUDA-Vulkan memory and semaphore sharing | Existing `ti.interop` | Python and Graph resource scope. |

Hardware Capability schema v2 currently provides:

```python
ti.hardware.report()
ti.hardware.capability(operation)
ti.hardware.providers()
ti.hardware.operations()
ti.hardware.probe(provider)
```

`report()` reads only static contracts, compiled backends, and current runtime
facts; it does not load or enable optional libraries. `probe()` is a separate,
explicit D1 probe. The current cuBLAS, cuSPARSE, and cuSOLVER probes use a
transient native library handle, close it before returning an immutable
snapshot, and do not change enablement or selection. Static operation/provider
descriptors and resolved reports are immutable values. If a domain algorithm
has already loaded one of these libraries through its real lazy loader, a later
passive `report()` observes the cached loader/capability state and reports it as
`enabled/eligible`; observation itself never calls `load_*`.

The first qualified matrix slice is deliberately narrow:

```python
output = ti.ndarray(ti.f32, shape=(batch, 16, 16))
ti.hardware.matrix.mma_f16_f32(a_f16, b_f16, output)

recording = ti.hardware.matrix.CudaMatrixMmaRecording(batch)
builder = ti.graph.GraphBuilder()
builder.append_native(recording, admission="auto")
graph = builder.compile()
graph.run({"a": a_f16, "b": b_f16, "output": output})
```

This is a D0 CUDA Driver/PTX native command for compact row-major
`m16n16k16`, f16 inputs, f32 accumulation, and f32 output. One warp executes
one tile on NVIDIA compute capability 7.0 or newer. It does not require the
CUDA Toolkit runtime or a vendor algorithm package. The call is explicit;
ordinary `ti.Matrix` multiplication is never rewritten to this route, and
Graph `admission="auto"` only validates integration of the already explicit
recording.

## Kernel boundary

A hardware operation can be called inside a Taichi kernel only when all of the
following hold:

1. the operation has typed Taichi IR semantics;
2. target capability is known during compilation;
3. the backend can lower the operation to device code without a host callback;
4. resource arguments have a kernel ABI and generation-safe binding; and
5. unsupported combinations fail during compilation or graph admission.

Texture sampling already fits this model. A future Matrix MMA kernel API and
Vulkan ray query can fit it after opaque tile or acceleration-structure types
and their typed backend IR exist. The current CUDA Matrix MMA provider is a
native command between kernels, so calling it from a kernel fails closed.

Raster commands, acceleration-structure build/refit, OptiX launches, and
vendor library calls do not fit this model. They execute between kernels or as
Graph native actions. A Python provider registry is never queried by device
code.

OptiX and Vulkan may implement the same batch ray-query semantic operation,
but that does not imply that OptiX can be inlined into an arbitrary Taichi CUDA
kernel. Inline capability is reported separately from executable capability.

## Provider lifecycle and selection

Provider state is split into independent domains:

| State domain | Values |
| --- | --- |
| Discovery | `missing`, `present`, `incompatible`, `available` |
| Enablement | `disabled`, `enabled` |
| Selection | `not_considered`, `eligible`, `selected`, `rejected` |

Errors are not provider state. Reports carry `last_error` and
`failure_scope`, where scope is one of `invocation`, `plan`, `provider`, or
`runtime`. An out-of-memory error from one invocation must not permanently
mark the provider incompatible.

Provider selection follows these rules:

```text
provider="auto"
    Consider D0 providers and only those D1 providers that the application
    globally enabled.

provider="builtin"
    Exclude all D1 providers.

provider="<explicit-name>"
    Treat the call as an explicit opt-in. Fail if the provider is missing,
    disabled by policy, incompatible, or unqualified. Do not silently fall
    back.
```

Only `auto` can fall back, and the report must record every considered route
and the reason for rejection. Discovery does not imply enablement. Passive
reporting must not change future provider selection merely because a library
happened to be installed.

## Hardware Capability schema

Hardware Capability schema v2 is independent of the existing Primitive
Capability schema. Primitive operations may reference a hardware provider ID
later, but the primitive schema is not version-bumped merely to duplicate
deployment fields.

Each resolved operation report contains at least:

```text
identity:
    schema_version
    operation_id
    semantic_family
    backend
    implementation_status

deployment:
    dependency_tier
    dependency_name
    load_mode
    provider_abi
    provider_version

classification:
    provider_class
    execution_class
    hardware_acceleration
    activation_mode

execution:
    scope
    execution_kind
    graph_support
    stream_binding
    workspace_ownership
    resource_effects
    lifetime_policy
    update_policy

state:
    discovery
    enablement
    selection
    unavailable_reason
    last_error
    failure_scope

semantic_contract:
    dtypes
    shapes_or_tiles
    layouts
    numeric_contracts
    deterministic
    fallback_provider
    fallback_equivalent
```

Static descriptors define stable semantics. Native probes provide runtime
facts. Python code may normalize the result but must not infer an extension,
ABI, or hardware unit from vendor name or compute capability alone.

`activation_mode` is the non-negotiable automatic/manual boundary:

| Mode | Who initiates the work | Example |
| --- | --- | --- |
| `explicit_hardware_api` | The caller explicitly invokes a Python/Graph hardware command or resource. | `ti.hardware.linalg.gemm_f32`, `RasterPass`, batch Ray Query. |
| `explicit_kernel_intrinsic` | The kernel author explicitly writes an intrinsic or hint; lowering is inline. | atomics, `SharedArray`, texture sampling, `ti.block_local`. |
| `domain_api_auto_provider` | The caller requests a domain operation; that implementation selects the hardware provider. | CUDA `SparseMatrix @ ndarray`, `SparseSolver`. |
| `compiler_automatic` | Compiler/runtime recognition selects an optimization behind existing semantics. | grouped reduction, Vulkan listgen ballot, qualified `cp.async`. |

The mode says how a route is activated, not whether the eventual instruction
is guaranteed. `hardware_acceleration`, exact requirements, and fallback fields
remain separate qualification axes.

## Graph, RHI, resources, and lifetime

Hardware resources and executable operations extend the existing NativeAction
contracts. They do not create a parallel scheduler or ownership system.

A recordable native command must declare:

- public and derived runtime bindings;
- read, write, and synchronization effects;
- temporary requirements and workspace ownership;
- backend and structured-region eligibility;
- address stability and update policy;
- stream or queue synchronization domain;
- generation-bound lifetime leases; and
- whether the Graph owns a real backend command recording, a qualified stream
  capture, or only an opaque host operation.

A descriptive backend command count is not equivalent to an integrated Graph
action. Automatic Graph admission is allowed only after the provider exposes a
recordable action with an executable backend contract.

Resources become invalid after their owning runtime generation is reset. A
Graph compiled against one provider ABI, device, or resource generation must
not replay against another without an admitted rebind or rebuild.

### Current root-Graph composition boundary

Automatic-admissible backend commands preserve source order with adjacent
Taichi CGraph stages and other provider commands. Qualification now exercises
both a CUDA `cuBLAS -> kernel -> cuSPARSE -> kernel` chain and a Vulkan
`AS refit -> batch Ray Query` chain. The latter relies on the scene-generation
lease plus the provider's internal AS barriers; public vertex/ray/hit arrays
retain their declared effects. Closing a resource or resetting its runtime
invalidates the composed Graph before replay.

This ordering is not backend fusion. A backend-command append flushes the
ordinary CGraph builder and remains a distinct native-command node. Multiple
commands can therefore share one root Graph contract and one submission
lifetime while still producing segmented backend work. Manifests and
`backend_command_nodes` expose the exact count; documentation and benchmarks
must not report it as one CUDA Graph or one Vulkan command buffer. Structured
`Sequential` admission remains unsupported for these commands.

Provider caches also remain provider-owned. Replaying explicit cuSPARSE SpMV
reuses the matrix's one handle, descriptors, workspace, and successful optional
preprocess plan; Graph does not copy or separately budget those resources. The
matrix-generation lifetime lease is what keeps that state valid.

### Current M3 Vulkan buffer-command contract

`ti.graph.VulkanBufferCommand` and `VulkanBufferCommandRecording` provide the
first real D0 backend-command route. This is low-level RHI substrate for later
RasterPass and AS build/refit providers; it does not claim that those feature
providers already exist.

```python
command = ti.graph.VulkanBufferCommand
recording = ti.graph.VulkanBufferCommandRecording((
    command.fill_u32("destination", byte_count, 0),
    command.buffer_barrier("destination"),
    command.copy("destination", "source", byte_count),
    command.memory_barrier(),
))

# Explicit manual execution.
recording.execute({"source": source, "destination": destination})

# Automatic admission only decides whether this is a real backend command in
# the current Graph.
builder = ti.graph.GraphBuilder()
builder.append_native(recording, admission="auto")
graph = builder.compile()
graph.run({"source": source, "destination": destination})
```

Two meanings of "automatic" must remain separate. Graph
`admission="auto"` validates the executable contract, but it does not
automatically replace an ordinary kernel with this operation and does not make
it kernel-callable. Creating the recording, selecting commands, and declaring
barriers are explicit. A future feature provider may select this D0 route
inside its domain API only while preserving semantics and reporting the route.

The current qualification boundary is:

- Vulkan compute queue and the runtime-ordered stream only, with no new Vulkan
  SDK or runtime-package dependency;
- current-Program `ti.ndarray` bindings only, with four-byte-aligned fill/copy
  ranges;
- rejection before submission for same-allocation overlapping copies,
  out-of-bounds ranges, wrong backend/device, stale/reset generations, and
  recordings longer than 4096 commands;
- barriers are explicit recording semantics; the runtime does not infer extra
  provider barriers;
- workspace ownership is `none`, there is no host readback, and Graph
  submission retains ndarray leases through backend completion;
- replay mode is `rerecord`: each replay uses one native entry point to record
  the complete sequence into one runtime command list; this is not a claim of
  cached Vulkan command-buffer replay; and
- only root `GraphBuilder.append_native(...)` is qualified. Backend commands
  in a structured `Sequential` are rejected, and AOT serialization is outside
  this contract.

### Current M4 Texture/Sampler qualification boundary

Creating a `ti.Texture` and calling `sample_lod()` or `fetch()` are explicit
API choices. For Vulkan, the compiler automatically lowers those typed texture
operations to SPIR-V image/sampler instructions. This means an explicit
semantic request receives an automatic hardware implementation; ordinary
field or ndarray loads are not silently replaced with texture sampling, and
there is no software-sampling fallback.

The qualified slice is Vulkan 1D/2D/3D sampled textures whose `sample_lod()`
and `fetch()` operations return `vec4<f32>`. The default sampler is fixed to
linear filtering, repeat addressing, and normalized coordinates; filter,
address, anisotropy, and comparison controls are not public. Storage-image
load/store through `ti.types.rw_texture` additionally supports format-matched
`f32`, `i32`, and `u32` sampled types. Floating-point filtering is not claimed
bitwise deterministic across devices. Although CUDA GPUs have texture units,
the LLVM CUDA backend has no `TextureOpStmt` lowering, so that route remains
`planned` rather than being reported available merely because hardware exists.

### Current M4 Vulkan RasterPass qualification boundary

`ti.hardware.raster.RasterPass` is the first public fixed-function graphics
provider. The user explicitly creates a pass, declares camera/light/draw
state, and calls `record()` or `execute()`. Each execution uses the existing
D0 GGUI/RHI and the current Program's `GraphicsDevice` to create a Vulkan
graphics command list, render pass, and graphics pipeline, then runs hardware
rasterization, depth test/write, and color output. This is a real hardware
raster/depth/ROP route, not ordinary native CPU code replacing a software
rasterizer.

The qualified slice is deliberately narrow and fail-closed:

- Vulkan only, with a hidden-window 2D offscreen target and built-in GGUI
  shaders for meshes, mesh instances, particles, and lines;
- `VulkanRasterPassRecording` freezes resource bindings and draw topology but
  rereads new contents from the same field/ndarray objects; each replay
  rerecords the graphics command list;
- execution performs no host readback. `color_numpy()` and `depth_numpy()` are
  explicit synchronous observations, and one execution can be consumed by
  only one of them before another execution is required;
- it is not kernel-callable and never silently replaces a software renderer.
  Explicit root-Graph admission records one opaque segmented node, preserving
  ordering and lifetime but not fusing its GGUI helper dispatches. Automatic
  admission, structured regions, and AOT remain unsupported because current
  Scene VBO preparation contains helper kernels while color/depth are
  provider-owned targets that cannot yet provide exact bindings/effects; and
- the provider reuses only the Vulkan/GGUI D0 runtime and built-in shaders
  already present in official wheels. It adds no SDK, vendor package, or wheel
  variant.

### Current M6 Vulkan ray-query qualification boundary

`ti.hardware.ray.TriangleScene` is the first public acceleration-structure
resource. Creating it is an explicit Python operation: Forge copies one f32
triangle mesh into provider-owned buffers and records an update-enabled BLAS
followed by a one-instance identity TLAS build. `refit()` and `record_refit()`
replace only vertex positions through Vulkan UPDATE; `trace()` and `record()`
expose batch Ray Query as direct execution or a root-Graph backend command.
Scene construction is deliberately not Graph-recordable because resource
creation, sizing, scratch allocation, and lifetime ownership are setup rather
than replay work.

The route is qualified as fixed-function traversal rather than a generic
collision accelerator:

- it requires the complete Vulkan buffer-device-address, acceleration-
  structure, and Ray Query feature cluster and fails closed otherwise;
- shader compilation targets Vulkan 1.2/SPIR-V 1.4, but the resulting C array
  is embedded in the runtime. `glslc` is build-only and no SDK library becomes
  a wheel dependency;
- build-input, AS-storage, and scratch buffers use explicit usage and device-
  address contracts; BLAS-to-TLAS and build-to-query dependencies are recorded
  with Vulkan AS access and pipeline-stage barriers;
- replay rerecords one runtime-ordered compute command, retains the scene and
  bound ndarrays through submission, and rejects execution after scene close
  or runtime-generation change; and
- only indexed triangles with fixed vertex count and topology, vertex-only
  refit, one identity instance, closest opaque hit, and the documented f32
  ray/hit layouts are qualified. Topology-changing rebuilds, transforms,
  multi-instance/procedural geometry, indirect builds, serialization, and
  inline kernel query remain planned.

This is a manual hardware interface. No existing software ray tracer,
renderer, contact query, or ordinary Taichi kernel is automatically rewritten
to call it.

### Current optional cuBLAS GEMM qualification boundary

`ti.hardware.linalg.gemm_f32` is an explicit D1 vendor-library command. It
accepts three distinct compact row-major f32 ndarrays and computes
`C = alpha * A @ B + beta * C`. The implementation swaps operands and output
dimensions when calling column-major `cublasSgemm_v2`, binds host scalar mode
and the Program's default CUDA stream, and retains one handle until Program
finalization. Direct execution and root-Graph `rerecord` share this route.

`cublas_is_available()` performs the existing transient, side-effect-free
cuBLAS probe; `is_available()` remains its compatibility alias. Actual
execution separately lazy-loads the user's compatible library;
passive `report()` observes that loaded state but never initiates it. The
provider adds one stable ABI symbol to the existing dynamic table and adds no
Toolkit header, link dependency, bundled library, package dependency, build
switch, or wheel variant. Missing/incompatible providers fail at this command,
not CUDA initialization. The provider has no Graph-exclusive workspace.

This route does not rewrite `ti.Matrix` operations, kernels, linear operators,
or solvers. Batched/strided GEMM, mixed precision, tensor-core algorithm
selection, transposed layouts, in-place aliases, kernel calls, and AOT remain
outside this qualified slice.

### Domain-selected and explicit cuSPARSE routes

The hardware catalog now describes two pre-existing public routes instead of
leaving them mislabeled as internal foundations. They illustrate a distinct
selection boundary:

- the user explicitly requests a sparse domain operation;
- the `ti.linalg` implementation selects its backend provider automatically;
- no compiler pass rewrites an unrelated kernel.

On CUDA, `SparseMatrix @ ndarray` selects cuSPARSE for f32 scalar-CSR SpMV and
qualified fixed-block BSR SpMV. The matrix resource retains descriptors,
workspace, and optional preprocessing for repeated direct calls. This remains
the automatic domain route. Separately,
`ti.hardware.linalg.spmv_f32`/`CusparseSpmvRecording` exposes a manual direct or
root-Graph command over the same stored matrix. Its Graph action declares the
dense input/output effects, leases the matrix generation, re-records one
runtime-ordered provider command, and reuses all matrix-owned provider state.
It is not kernel-callable. `SparseSolver` selects both cuSPARSE and cuSOLVER,
owns analysis/factorization state and workspace, and remains direct-only. Its
f32 scalar-CSR LLT/LDLT modes use the implementation's sparse Cholesky route;
LU is host-assisted and includes explicit transfers.

Both libraries remain D1: real object construction/use lazy-loads the user's
compatible shared libraries, while passive reporting never does. Qualification
adds no dependency, linked or bundled library, build switch, or wheel variant.
The explicit command is Python/Graph orchestration around the existing native
provider entry and does not change the provider ABI or loader.

### Current M7 optional cuFFT qualification boundary

`ti.hardware.fft.CufftPlan1D` is the first new D1 vendor-algorithm provider.
`ti.hardware.fft.is_available()` delegates to the explicit transient probe:
it tries versioned shared-library candidates, verifies only the five basic C2C
symbols, queries the component version, and closes the handle before return.
Passive `ti.hardware.report()` never performs this probe. Creating a plan is
the separate enablement action and keeps the successful runtime loader alive;
later passive reports can then observe it.

The qualified slice is deliberately limited:

- one fixed-size, single-GPU, single-precision 1D C2C plan with compact
  `[real, imag]` pairs, an explicit batch count, and distinct input/output;
- forward and unnormalized inverse execution on the runtime default CUDA
  stream, with provider-owned workspace and no host readback;
- direct Python execution and root-Graph `rerecord` replay; plan close or
  runtime-generation change invalidates all recordings; and
- dynamic lookup of `cufftPlan1d`, `cufftDestroy`, `cufftSetStream`,
  `cufftExecC2C`, and `cufftGetVersion`, with no `cufft.h`, link dependency,
  bundled library, new build switch, or wheel variant.

The user's cuFFT installation may itself require compatible companion
components. Such failures remain plan/provider scoped. Forge does not expose
callbacks, LTO, multi-GPU, arbitrary strides, R2C/C2R, in-place execution,
kernel calls, or AOT in this slice, and does not automatically substitute
cuFFT for existing transforms or solvers.

## Cache boundaries

One universal cache key would invalidate too much portable work and still be
unsafe for native executables. Caches are separated as follows:

```text
semantic/codegen cache:
    provider_codegen_version
    target architecture
    required PTX or SPIR-V capabilities
    numeric contract

native executable/pipeline cache:
    provider ABI and version
    device identity
    driver-compatible fingerprint
    compile and pipeline options

runtime plan cache:
    runtime generation
    resource generation
    stream or graph binding identity
```

Updating a graphics driver must not automatically invalidate portable Python
frontend IR, while a driver-generated pipeline must never be reused with an
incompatible runtime.

## Capability roadmap and physics-engine ROI

The implementation order weights simulation usefulness, attainable speedup,
current Forge foundations, and qualification cost.

| Priority | Operation | Physics and rendering use | Deployment and scope |
| --- | --- | --- | --- |
| 1 | Vulkan raster pass | Replaces software rasterization for simulation visualization. | D0, explicit native executable. |
| 2 | cuBLAS/cuSPARSE/cuSOLVER provider normalization | Linear solves, sparse operators, and preconditioners; existing lazy loaders reduce implementation cost. | D1, domain algorithm operation. |
| 3 | Vulkan AS build/refit and ray query | Ray rendering, visibility, picking, and genuine ray-mesh queries; not general overlap or contact. | D0, resource plus native command or typed shader operation. |
| 4 | Texture and sampler qualification | Grid, SDF, volume, material, and rendering lookup. | D0, explicit kernel semantic. |
| 5 | Matrix MMA | Batched local FEM matrices, small blocks, dense batches, and block preconditioners under explicit numeric contracts. | D0; current slice is an explicit native command, with typed kernel semantics planned. |
| 6 | cuFFT | Spectral methods, convolution, and selected Poisson or fluid formulations. | D1, external-library plan and execution. |
| 7 | OptiX | High ray-rendering value on qualified NVIDIA RTX devices, with a narrower device and ABI range. | D1, vendor hardware executable. |
| 8 | Async tile and mesh-shader specialization | Dense tiled kernels and dynamic rendering geometry after public semantics are stable. | D0, transparent provider implementation. |

Sparse MMA, DPX, public TMA calls, and any D3 provider remain deferred.

### Closed audit of remaining provider candidates

The remaining planned entries were traced through frontend semantics, typed
IR/code generation, feature enablement, resource binding, command submission,
and packaging. They are deliberately not partial implementations:

| Candidate | Physics/rendering value | Missing chain in the current tree | Decision |
| --- | --- | --- | --- |
| CUDA texture/sampler | SDF, grid, volume, and material lookup. | LLVM/CUDA `Program` texture allocation/lifetime, CUDA array and texture-object upload, kernel argument ABI, and `TextureOpStmt` lowering are all absent; current `Texture` allocation exists only in the GFX Program. | Keep `planned`. This needs a complete CUDA texture resource family, not one intrinsic. |
| Vulkan cooperative matrix | Batched FEM element matrices, small-block solvers, and preconditioners. | No feature query/enablement, property enumeration, opaque tile type, typed IR, or SPIR-V load/mul-add/store lowering exists. The Vulkan specification requires enumerating device-supported M/N/K/type/scope tuples. | Keep `planned`; do not clone the CUDA `m16n16k16` contract. |
| Inline Vulkan Ray Query | Kernel-local visibility, picking, and specialized ray-mesh queries. | The batch provider has AS resources and an embedded standalone shader, but no kernel-visible AS argument, effect/lifetime binding, RayQuery IR/control, or SPIR-V lowering. | Keep `planned`; use the qualified direct/root-Graph batch query. |
| Vulkan mesh shader | Dynamic render geometry. | No extension feature chain, mesh/task shader codegen, mesh pipeline construction, or draw-mesh-tasks command recording exists. | Keep internal and `planned`; low physics ROI relative to the full RHI cost. |
| OptiX | NVIDIA ray rendering. | SDK-header-defined function-table ABI, license gate, device programs, module/program-group/pipeline/SBT, GAS/IAS, and lifetime contracts are absent. | User-built plugin/source-build candidate only; do not add an official wheel variant. |
| CUB/CCCL | Sort/scan/reduce primitives. | Header templates, a CUDA device compiler, and CUDART are build-time requirements. | Retain the existing D2 reference path; official wheels use Forge-owned primitives. |

The Vulkan cooperative-matrix and Ray Query requirements follow the
[Khronos Vulkan/SPIR-V specifications](https://registry.khronos.org/vulkan/specs/latest-ratified/pdf/vkspec.pdf).
OptiX initialization and its versioned function table remain defined by the
[NVIDIA OptiX SDK API](https://raytracing-docs.nvidia.com/optix9/api/OptiX_API_Reference.pdf),
whose SDK download also carries a separate license acceptance. Sparse MMA,
public TMA/WGMMA/DPX, and generation-specific instructions remain internal or
deferred until a portable physics operation and a measurable workload justify
them.

## Milestones

### M0: architecture contract

- publish equivalent English and Chinese architecture documents;
- freeze deployment, provider, execution, and hardware-qualification axes;
- freeze automatic versus explicit invocation and provider selection; and
- define commit and release gates without adding a public runtime API.

### M1: capability schema and read-only report

- implement Hardware Capability schema v2 independently of primitive schema;
- expose immutable descriptors and resolved native probe facts;
- keep optional-library probing explicit and side-effect-free with respect to
  enablement and selection; and
- cover discovery, incompatibility, and failure-scope tests.

### M2: distribution guardrails

- assert official runtime CMake switches in CI;
- audit wheel dynamic dependencies and bundled libraries;
- test import, initialization, and D0 execution without D1 components; and
- retain D2 reference builds in separate non-release jobs.

### M3: native command Graph/RHI substrate

- extend NativeAction with real backend command recording;
- reuse runtime bindings, effects, temporaries, leases, and update policy;
- define stream, queue, barrier, and workspace ownership; and
- qualify direct execution, Graph replay, reset, and device mismatch.

### M4: first high-ROI providers

- expose a Vulkan raster pass through the native-command substrate;
- normalize existing cuBLAS, cuSPARSE, and cuSOLVER loader reports and failure
  isolation; and
- qualify the existing texture and Vulkan sampler route while leaving CUDA
  texture-object support `planned` until it has a real lowering and tests.

### M5: matrix hardware

- first land a qualified explicit CUDA Driver/PTX native command for compact
  row-major `m16n16k16`, f16 inputs, f32 accumulation/output, direct execution,
  and root Graph replay;
- enumerate admitted tile, dtype, layout, scope, alignment, and accumulation
  contracts without silently rewriting ordinary matrix multiplication;
- keep opaque cooperative-matrix tile types, typed kernel IR, and Vulkan
  Cooperative Matrix lowering planned until they have independent route and
  correctness qualification; and
- keep async copy, TMA, WGMMA, and later generation mechanisms internal.

### M6: ray and acceleration structures

- add Vulkan BLAS/TLAS resource allocation and size queries;
- add build, update, refit, copy, scratch, and synchronization commands;
- add batch ray query and later qualified inline ray query; and
- introduce OptiX only after ABI and license gates pass.

### M7: optional vendor algorithms

- add a minimal single-GPU cuFFT plan and execution provider;
- exclude first-version callbacks, LTO, and multi-GPU paths that introduce
  additional NVRTC or nvJitLink version contracts; and
- keep CUB as a D2 reference or user-built plugin.

### M8: internal specialization

- select PTX `cp.async` only for the qualified compiler-generated
  global-to-BLS pattern and retain synchronous lowering outside admission;
- report per-Program eligibility and actual compiled-specialization selection;
- keep mesh shader fail-closed until feature enablement, SPIR-V codegen,
  pipeline construction, command recording, and workload qualification all
  exist; and
- do not expose the selected vendor mechanism as stable public syntax.

## Qualification and release gates

Every provider milestone must pass all applicable gates:

1. **distribution**: official wheel count, tags, build switches, and mandatory
   dependencies remain unchanged;
2. **route proof**: PTX/SPIR-V inspection, RHI command evidence, or provider
   trace proves the claimed execution path;
3. **correctness**: reference results cover dtype, bounds, NaN, precision,
   determinism, and resource state;
4. **failure isolation**: missing library, unsupported extension, incompatible
   ABI, wrong device, reset, and out-of-memory paths fail closed;
5. **Graph and lifetime**: direct execution and Graph replay agree, with stream,
   workspace, generation, and destruction tests;
6. **performance**: cold setup, plan or build, and steady-state execution are
   measured separately with correctness, route, and noise gates; and
7. **documentation**: English and Chinese support matrices state the exact
   operation, backend, provider, numeric contract, and qualification hardware.

An implementation is not publicly described as hardware accelerated merely
because a high-level API succeeded. `hardware_acceleration=qualified` requires
route evidence on an admitted device and driver.

## Commit boundaries

Implementation is delivered as reviewable, independently revertible commits.
The default boundaries are:

1. architecture documents and navigation only;
2. schema values, immutable descriptors, and schema tests;
3. official-wheel guardrails and their tests;
4. native command Graph/RHI contracts without a feature provider;
5. one provider family and its focused tests per commit;
6. provider qualification and performance evidence separately from mechanism
   implementation when the evidence is large; and
7. bilingual public API and release documentation after the support gate.

No commit should combine an unreviewed schema change, a new backend command,
an optional loader, and a public support claim. Mechanical formatting can be
included with the file it formats, but unrelated cleanup is excluded.
