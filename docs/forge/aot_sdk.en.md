# Standalone AOT SDK

[中文](aot_sdk.zh.md)

The 0.6.4 development build can install a separate C/C++ SDK. The producer uses
Forge Python to export modules; the consumer links `TaichiForge::c_api` and does
not import Python. CPU/CUDA still use the supplied LLVM runtime and compiler.
This is a full SDK, not a compiler-free runtime.

A Vulkan-only SDK can also be built without LLVM, CUDA or Python. Configure a
separate build directory with the following options, then build `taichi_c_api`
and install the `TaichiSDK` component as below:

```sh
cmake -S . -B BUILD_VULKAN -DCMAKE_BUILD_TYPE=Release \
  -DTI_WITH_LLVM=OFF -DTI_WITH_CUDA=OFF -DTI_WITH_CUDA_TOOLKIT=OFF \
  -DTI_WITH_PYTHON=OFF -DTI_WITH_SPLIT_PYTHON_RUNTIME=OFF \
  -DTI_WITH_OPENGL=OFF -DTI_WITH_METAL=OFF -DTI_WITH_GGUI=OFF \
  -DTI_WITH_VULKAN=ON -DTI_WITH_C_API=ON \
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=/absolute/path/BUILD_VULKAN/bin
```

Use an absolute output path belonging to that build to keep its binaries
separate from other configurations. The smaller SDK uses the same Vulkan AOT
artifacts and public C API. Windows x64 checks cover direct/ordinary Graph
forward and reverse execution, dense fields, fixed structure arrays, memory
ownership and module retirement. This build still contains parts of the Forge
compiler and needs Vulkan driver pipeline compilation. It provides no CPU/CUDA
runtime; it does not expand native Graph or sparse deployment support.

From an already configured matching Forge build:

```sh
cmake --build BUILD --target taichi_c_api
cmake --install BUILD --prefix SDK --component TaichiSDK
```

The explicit install component contains public C/C++ headers, the C API shared
library and import library where applicable, runtime bitcode, licenses, a CMake
package and `sdk_manifest.json`. It does not add another copy to normal wheel
installs. The manifest records enabled LLVM/CUDA/Vulkan features, LLVM version,
file sizes and SHA256 hashes. Its source revision is the revision at CMake
configuration time; hashes identify the actual installed bytes.

For an SDK that also hosts JIT, an opt-in build can expose the same C API from
the split `taichi_runtime` library. This avoids distributing a second complete
native core alongside the Python shim. In a matching Python-enabled build:

```sh
cmake -S . -B BUILD -DTI_WITH_PYTHON=ON \
  -DTI_WITH_SPLIT_PYTHON_RUNTIME=ON -DTI_WITH_RUNTIME_C_API=ON \
  -DTI_WITH_C_API=OFF
cmake --build BUILD --target taichi_runtime taichi_python
cmake --install BUILD --prefix SDK --component TaichiSDK
```

The consumer still links `TaichiForge::c_api`; the CMake package selects
`taichi_runtime` and the manifest records `native_library` and
`build_features.runtime_c_api`. Python is required to build the matching shim
and its private export contract, but the C/C++ consumer does not load Python.
`TI_WITH_RUNTIME_C_API` defaults to `OFF` for standalone CMake configuration;
the 0.6.4 runtime-wheel build enables it. Previously built wheels may lack the
C API. Keep the shim and native library paired.
Sharing this binary does not automatically share mutable runtime instances or
make arrays interchangeable between independently created runtimes. The AOT
artifact and native recording/provider support boundaries below still apply.

Optional cuDSS, cuSOLVERDn, cuBLASLt, cuTENSOR, cuSPARSELt, AmgX and VkFFT adapters have separate SDK components.
Enable their build targets, then build and install only the selected adapter:

```sh
cmake -S . -B BUILD -DTI_BUILD_BUNDLED_OPTIONAL_RUNTIME_PROVIDERS=ON
cmake --build BUILD --target taichi_forge_cutensor_provider
cmake --install BUILD --prefix SDK --component TaichiSDKProvider_cutensor
```

Replace `cutensor` with `cusparselt`, `amgx` or `cusolverdn` for those components.
The cuSOLVERDn component also installs `forge_cusolverdn_provider.h` for its
shared Cholesky execution API. Native plan execution and AOT reconstruction of
retained Cholesky recordings share this adapter. The vendor runtime remains external.

The `cub_source` component bridges an existing CUB source-addon C ABI 1 binary
to the same common provider protocol. It installs `forge_cub_source_provider.h`
and a small adapter, without the addon or CUDA Toolkit. Provide an explicit
addon library path; the common runtime reports its actual loaded path and
`sha256:<hex>` identity, CUB version and compiled CUDART version. Keep the addon
file immutable while loaded. Plans query caller-owned workspace once and retain
the runtime until all recorded/submitted work has retired. Windows tests cover
the six existing operations, empty inputs, identity, failures and retirement.
The native registry reconstructs `cub_source/plan` resources and
`cub_source/execute` operations. Frozen plans bind the addon hash, CUB/CUDART
versions, target code, declared driver requirements and exact workspace query;
prepare rejects mismatches before recording work. CPU/Vulkan do not register
these factories or load the configured addon.

Selected `CubSegmentedScanRecipeProvider` recipes export through
`Module.add_execution_plan`, retaining their i32/u32 modular-sum strategy and
frozen head bitset. The consumer recreates workspace; producer pointers and
scratch contents are not serialized. Export names the source/output arrays via
`source_bindings` as for other selected recipes. Windows CUDA checks cover all
six native primitives, zero-item select/scan, selected inclusive/exclusive scan,
empty segments, capacity tails, aliases, independent and shared bindings,
standalone/borrowed retirement, and native AOT WHILE zero/early/budget exits.
Actual downstream `kernel.grad` is checked for floating-point weights; integer
scan/sort itself has no automatic numerical derivative. The native operation
inventory does not add automatic export to the private `CubSourcePlan` interface.

The optional `cutlass_source` component bridges the existing CUTLASS matmul addon
C ABI 1 through the same provider owner and loaded-binary identity check. Its
`forge_cutlass_source_provider.h` describes direct and both split-K strategies,
compact f32 operands, transpose, alpha/beta and identity/ReLU. Workspace remains
caller-owned and the adapter retains the addon until plans retire. The old ABI
reports CUTLASS version but no CUDART version (reported as zero); deployment must
retain the addon's build manifest and declared dependencies. Windows checks cover
the three strategies, four transpose combinations, live beta feedback and failure
recovery. The external addon and CUDA Toolkit are not included in the adapter SDK.

Selected `CutlassMatmulRecipeProvider` recipes export through
`Module.add_execution_plan`. The native `cutlass_source/plan` resource freezes the
strategy, matrix semantics, binary/version/target identity and exact workspace;
`cutlass_source/execute` restores scalar f32 shape/alignment and physical alias
checks. Consumers recreate scratch and capture the existing addon kernels once,
without selecting a new strategy, compiling addon code or loading the producer's
cuBLASLt baseline. CPU/Vulkan do not register these CUDA factories. Windows CUDA
checks cover three selected strategies, transposes, beta/ReLU, standalone and
borrowed lifetimes, shared binding families, source Graph retirement and native
AOT WHILE exits. Actual downstream `kernel.grad` and explicitly recorded matrix
adjoints are checked; this does not add automatic differentiation to opaque addon
operations or claim every JIT control combination is supported.

For the cuBLASLt native plan API, replace `cutensor` with `cublaslt`. Its component
installs `forge_cublaslt_provider.h` and the adapter, with no vendor headers or
vendor binary linkage. The API supports compact f32 matmul, batched and transposed
operands, alpha/beta feedback, row-major or equivalent transposed column-major
layouts, and identity/ReLU epilogues. A plan either selects one cold heuristic
result or restores an explicit algorithm configuration and checks its exact
workspace requirement without searching. Plan information reports which path ran
and the algorithm's operand alignment requirements. Callers own device buffers
and workspace, activate the owning CUDA context, and retire all captured/in-flight
uses before destroying plans and their runtime. JIT plans, typed Graph capture,
and AOT consumption share this C API.

Ordinary `CublasLtMatmulPlan` recordings and their typed captures can be exported
with a Graph through `module.add_execution_plan("step", graph)`. Resource
`cublaslt/matmul_plan_f32` and operation `cublaslt/matmul_f32` (version 1) persist
the frozen configuration. Consumers resolve user-installed dependencies with
`runtime.configure_provider("cublaslt", adapter_path=..., library_path=...)`.
Preparation requires the same cuBLASLt version, checks the algorithm, layout and
exact workspace on the current device, and restores without a heuristic search.
Each prepared execution owns its reconstructed plan and workspace; repeated
calls to the same plan share one resource. Packages contain no device addresses,
vendor handles, producer DLL paths or unused workspace snapshots.

Validated CUDA coverage includes standalone/borrowed consumption, single/multiple
batches, transposed operands, both physical layouts, identity/ReLU, beta feedback,
and explicit matrix adjoints combined with actual `kernel.grad`. This does not
add automatic differentiation of vendor operations. CPU/Vulkan do not register
this execution factory. Ordinary compilation of `ti.linalg.record_matmul` and
selected matmul recipes also export their actual packing kernels, fused/separate
ReLU and private storage without selecting another algorithm. Even when the
vendor operation reads only private packed inputs, the original public operands'
exact shapes, f32 types and output-disjointness constraints remain enforced during
cold preparation. Compositions inside conditional or parallel regions still
require their own qualification.

cuDSS uses
its independent build option and matching 0.8.x SDK headers, without enabling
the other optional providers:

```sh
cmake -S . -B BUILD -DTI_BUILD_BUNDLED_CUDSS_PROVIDER=ON -DTI_CUDSS_ROOT_080=CUDSS_ROOT
cmake --build BUILD --target taichi_forge_cudss_provider_080
cmake --install BUILD --prefix SDK --component TaichiSDKProvider_cudss
```

VkFFT uses its own build option, VkFFT 1.3.4 sources and matching static Vulkan
shader-compiler libraries at build time:

```sh
cmake -S . -B BUILD -DTI_BUILD_VKFFT_PROVIDER=ON -DTI_VKFFT_ROOT=VKFFT_ROOT
cmake --build BUILD --target taichi_forge_vkfft_provider
cmake --install BUILD --prefix SDK --component TaichiSDKProvider_vkfft
```

The component includes upstream licenses and sources notices in `licenses/vkfft`.
VkFFT and its shader compiler are embedded in the adapter; consumers need the
Vulkan loader/driver and platform C/C++ runtime, without a separate vendor DLL,
Vulkan SDK or external shader compiler. Its manifest records these requirements
and all installed license files. The qualified Windows component is about 5 MB;
it remains optional and is not included by a default install.

OptiX provides separate components for ABI 93, 105 and 118, so an SDK can ship
only the adapter required by its deployment. Build with the existing pinned
OptiX headers and CUDA Toolkit configuration, then select the matching component:

```sh
cmake -S . -B BUILD -DTI_BUILD_BUNDLED_OPTIX_PROVIDERS=ON
cmake --build BUILD --target taichi_forge_optix_provider_118
cmake --install BUILD --prefix SDK --component TaichiSDKProvider_optix_abi118
```

Use `93` or `105` in both the target and component name for those variants. Each
component includes the common and OptiX C ABI headers, embedded device PTX and
matching license notices; the driver-provided OptiX/CUDA libraries remain external.
Consumers need neither OptiX SDK headers nor nvcc. Manifests are named
`optix_abi93.json`, `optix_abi105.json` and `optix_abi118.json`, with `provider_id`
remaining `optix` and a separate `variant` field. License directories are also
variant-specific. Components can coexist without replacing each other's inventory
or the base SDK manifest. Qualified Windows payloads are about 221 KiB each;
the existing runtime-wheel adapter selection is unchanged. Installing a component
does not select it automatically: pass its path to `runtime.configure_provider`.

Each ordinary component
installs an adapter into `bin/hardware_providers` on Windows or
`lib/hardware_providers` on Linux, the common provider C ABI header, the Forge
license and `share/taichi/providers/<provider>.json` with file sizes and SHA256
hashes. The base `sdk_manifest.json` remains separate. The CMake package exposes
`TaichiForge_PROVIDER_DIR` and `TaichiForge_PROVIDER_MANIFEST_DIR`. Compatible
vendor libraries and their dependencies remain application-managed. Installing
an adapter makes its existing C ABI available; AOT plan reconstruction additionally
requires a registered operation/resource schema and its qualified backend path.

OptiX adapters additionally expose `taichi_forge_optix_runtime_provider_query`,
specified in `taichi/optix/forge_optix_runtime_provider.h`; the original OptiX ABI 1
remains independently usable. Common sessions share the vendor-library reference
count with legacy contexts without creating GPU contexts, pipelines or scenes.
The embedded execution table keeps the original OptiX result codes, which must
not be interpreted as common provider results. Each scene/query owner must retain
its session through in-flight work and resource destruction; live leases in one
adapter require the same library path. The execution table identifies the exact
SDK/OptiX ABI; unknown driver runtime versions are zero, never the SDK version.

The optional domain execution table v2 adds the fixed-query extension declared in
`forge_optix_query.h`, also available through `taichi_forge_optix_fixed_query_api`.
It prepares opaque typed/first-accepted and compact occlusion queries, optionally
with live face tables, and uploads each query's immutable 32–64-byte parameters
once. The query retains its scene; callers retain arrays and retire all launches
before destruction. In-place AS refit is supported; replacement/rebuild requires
preparation again. Launches remain ordered and serialized through the owning
context: this extension does not declare CUDA stream-capture support. The legacy
provider ABI/table and common execution table v1 remain available. AOT query
preparation negotiates this extension once per context and uses it when advertised;
older adapters retain the existing ordered path. There is no per-launch capability
probe or host parameter upload on the prepared path, and no alpha/OMM expansion.
The adapter query, device face records and scene/context leases retire together.

`runtime.host_blob(data)` copies bytes or another buffer object into an immutable
public NativeResource for provider preparation. It uses the same representation
as artifact `host_blob` resources, without device allocation or transfer. The
source may change or be released after return. Empty blobs are valid; blobs cannot
serve as ndarray, texture or AS arguments. Closure and cross-runtime checks follow
the public resource contract; factories declare whether inputs remain needed after
construction.

Vulkan triangle queries can also be reconstructed through the common registry,
under provider `vulkan_ray`, version 1:

| operation | Kind and bindings | parameters |
| --- | --- | --- |
| `triangle_scene` | Resource: `vertices`, `indices` | `vertex_count`, `triangle_count` |
| `triangle_blas` | Resource: `vertices`, `indices`, optional `opacity_micromap` | `vertex_count`, `triangle_count`, optional `opaque` (default false with a micromap, true otherwise) |
| `opacity_micromap` | Resource: host blobs `data`, `descriptors`, optional `indices` | `triangle_count` |
| `instance_tlas` | Resource: `transforms`, `blas_0` ... `blas_(instance_count-1)` | `instance_count`, optional `custom_indices`, `visibility_masks`, `sbt_record_offsets` |
| `refit` | Operation: `scene`, `vertices` | Empty object |
| `blas_build` | Operation: `scene` (BLAS), `vertices`, `indices` | Empty object |
| `instance_refit` | Operation: `scene`, `transforms` | Empty object |
| `instance_update` | Operation: `scene` (TLAS) | Captured `transforms`, optional `custom_indices`, `visibility_masks`, `update` (default true) |
| `query` | Operation: `scene`, `rays`, `hits`, optional `hit_indices` | `ray_count` |

Declare the resource in an ExecutionPlan's `resources`, or create it with
`runtime.native_resource("vulkan_ray", "triangle_scene", parameters=..., bindings=...)`
and bind it across modules. Before allocating, check that
`runtime.native_operation_capabilities("vulkan_ray_query")` contains `vulkan_inline`
and `runtime.device_capabilities().get("spirv_has_ray_query", 0)` is nonzero.
No external provider DLL is needed.

Geometry uses f32 `[V,3]` vertices and i32 `[T,3]` indices. Rays use f32 `[N,8]`
containing origin.xyz, t_min, direction.xyz and t_max. Each port also accepts a
packed AoS vector ndarray with the corresponding width. Record counts must match
exactly. Callers provide finite geometry, indices in `[0,V)` and valid ray
intervals, following the JIT contract without per-frame host readback. The factory
copies geometry and completes AS construction on its owner's GFX queue before
returning; seed arrays can then close. Changing vertex/triangle counts or the
ordered instance BLAS references requires a new resource.
For `triangle_scene`, refit updates device vertices with the same count and
refreshes TLAS bounds.
Queries and refits record into the same native command sequence; replay performs
no registry lookup.

With `hit_indices`, f32 `[N,4]` holds t/u/v/0 and i32/u32 `[N,4]` holds
primitive/instance/custom/hit. This single scene uses zero instance/custom IDs.
Misses write `[-1,0,0,0]` and `[-1,-1,-1,0]`. Without `hit_indices`, legacy f32
`[N,4]` contains t/primitive/custom/hit, with `[-1,-1,-1,0]` for a miss.
Both variants accept both faces and return the closest hit in each ray's interval.
Adjacent subranges may share a buffer; preparation rejects overlapping inputs and
outputs, including duplicate imports of the same physical storage.

Closing a resource retires its dependent recordings; the runtime owns public
scenes shared across modules. Known requested allocations for private scenes
contribute to `workspace_bytes`; private arrays contribute to `resource_bytes`.
Public scenes are not counted again in every plan, and these figures are not
driver residency. Small Windows Vulkan checks cover standalone consumers,
borrowed owners, refit/miss reset, all four query variants, shared retirement and
downstream gradient kernels. AS traversal itself is not differentiated. Face
filtering and compact occlusion are not provided by this AOT factory yet.

Independent `triangle_blas` resources can be shared by several `instance_tlas`
resources and repeated within one TLAS. Bindings are in instance ordinal order;
primitive indices retain each BLAS's input triangle order. Optional metadata arrays
must have `instance_count` entries: custom IDs are 24-bit (default 0), masks are
8-bit (default 255). Optional `sbt_record_offsets` are 24-bit values, one per
instance, defaulting to zero. They remain frozen through host/device BUILD and
UPDATE; changing them requires a new TLAS. Batch/inline ray queries do not select
SBT records; preserving these values does not provide AOT ray-program execution.
Batch queries use ray mask 255. Transforms are packed f32
row-major affine matrices, either scalar `[I,12]` / `[I,3,4]`, or AoS `[I]` with
element shape `[3,4]`. As in JIT, the caller supplies finite, invertible transforms.
Initial construction packs device matrices and builds the TLAS once; it does not
read matrices back or first build an identity TLAS. Construction seeds may close
after successful return; the TLAS continues to retain its BLAS dependencies.

Use `refit(scene=blas, vertices=...)`, then `instance_refit(scene=tlas,
transforms=...)` for each affected TLAS, before querying it. Geometry refit alone
does not refresh dependent TLAS bounds. The two resources have distinct typed
contracts: queries accept a triangle scene or TLAS, geometry refit accepts a
triangle scene or BLAS, and instance refit accepts a TLAS. Closing a public BLAS
retires dependent TLAS recordings transitively; closing one TLAS leaves siblings
usable. The BLAS `opaque` build flag does not add material filtering: these batch
query shaders treat triangles as opaque and accept both faces.

Small Windows Vulkan checks cover shared/repeated BLAS, distinct instance/custom
IDs and masks, direct initial device build, GPU geometry/transform updates,
subrange matrices, typed/legacy hit and miss reset, cross-module retirement,
standalone/borrowed owners, and downstream gradient kernels. Raster graphics AS bindings
are not yet covered by these factories.

Vulkan `Module.add_kernel()` accepts `ti.types.acceleration_structure()` parameters
when the module declares `caps=["spirv_has_ray_query"]`. Pass a public
`instance_tlas` resource to `module.kernel(name)(tlas, ...)` or
`.prepare(tlas, ...)`; an ExecutionPlan kernel argument may also name a public or
private TLAS. These bind the same managed AS without copying geometry or creating
a hidden Program. Prepared commands reuse the existing queue and retirement
contract. A BLAS is not a valid kernel AS argument.

Kernels may use existing closest/any and filtered queries. Build the BLAS with
`opaque=False` when filtering each candidate; `respect_opacity=True` retains the
existing opaque semantics. Small standalone/borrowed Vulkan checks cover traversal
after candidate rejection, occlusion, intervals/reverse rays, GPU refit, hit-to-miss
reset, public/private bindings, close/reset and downstream gradients. AS traversal
has no automatic derivative. Both ordinary kernels and Graph execution sequences
can be exported; `kernel_template` remains unavailable. CPU/CUDA do not accept
Vulkan AS parameters.

For baked opacity micromaps, also check
`runtime.device_capabilities().get("vulkan_has_opacity_micromap", 0)` before
allocating. The `opacity_micromap` factory accepts immutable host blobs using the
same baked layout as `ti.hardware.ray.VulkanOpacityMicromap`: data bytes,
8-byte triangle descriptors, and optional signed i32 triangle indices. It supports
two-state/four-state formats and predefined indices -1 through -4. Existing
physical subdivision limits and descriptor/index bounds are checked during import.

```python
with runtime.host_blob(baked_data) as data, runtime.host_blob(baked_descriptors) as descriptors:
    omm = runtime.native_resource(
        "vulkan_ray", "opacity_micromap",
        parameters={"triangle_count": triangle_count},
        bindings={"data": data, "descriptors": descriptors},
    )
# Construction is complete; the seed blobs may close.
blas = runtime.native_resource(
    "vulkan_ray", "triangle_blas",
    parameters={"vertex_count": vertex_count, "triangle_count": triangle_count},
    bindings={"vertices": vertices, "indices": indices, "opacity_micromap": omm},
)
```

The BLAS retains the micromap, which must have the same triangle count. Explicit
`opaque=True` conflicts with a micromap and is rejected. Use AS kernel queries
with `respect_opacity=True` for opacity classification; the built-in batch `query`
operation still uses its opaque query shaders. Several BLAS may share one public
micromap; closing it retires dependent recordings transitively. Small Windows
Vulkan tests cover both formats, indexed/predefined states, filtered closest/any
queries, refit, miss reset, standalone/borrowed owners and downstream gradients.
These do not establish differentiable AS construction or traversal.

Explicit AOT plans also reconstruct programmable RT through provider
`vulkan_ray_program`, version 1. Check registration of
`vulkan_ray_program_pipeline` / `vulkan_ray_program_launch` and the independent
device capability `vulkan_has_ray_tracing_pipeline` before allocation. This is
separate from inline ray-query capability. No runtime shader compiler or external
provider DLL is required; the application supplies compiled SPIR-V.

| operation | Kind and bindings | parameters |
| --- | --- | --- |
| `pipeline` | Resource: host blobs `shader_0` ... | `shader_count`, `shaders`, `groups`; optional `max_trace_depth` (1), `allow_opacity_micromaps` (false) |
| `launch` | Operation: `pipeline`, host blob `payload`, `buffer_0` ..., `image_0` ..., `scene_0` ... | `dimensions`, `raygen`, `miss`, `hit`, optional `push_constants`; `buffer_count`/`buffers`, `image_count`/`images`, `scene_count`/`scenes` |

Pipeline `shaders` entries declare `stage` (`raygen`, `miss`, `closest_hit`,
`any_hit`) and optional `entry` (`main`). Each group declares `kind` (`raygen`,
`miss`, `triangles`) and shader indices `general`, `closest_hit`, `any_hit` as
applicable. Unused indices default to the native unused value. For example:

```python
with runtime.host_blob(shader_bytes) as code:
    pipeline = runtime.native_resource(
        "vulkan_ray_program", "pipeline",
        parameters={"shader_count": 1, "shaders": [{"stage": "raygen"}],
                    "groups": [{"kind": "raygen", "general": 0}]},
        bindings={"shader_0": code},
    )
```

Launch `dimensions` contains three positive uint32 values checked against device
limits. `raygen` is one record; `miss` and `hit` are record lists. Each record has
`group` plus optional byte `offset`/`size` (both default 0) into the immutable
`payload` blob. `push_constants` is another offset/size section. Payloads contain
scalar shader-ABI data; they do not establish ownership of encoded device addresses.

Each buffer entry declares `set`, `binding`, `kind` (`storage_buffer` or
`uniform_buffer`) and `access` (`read`, `write`, `read_write`); uniforms are
read-only. Image entries use `sampled_image` or `storage_image`, the same access
declaration, and optional storage-only `mip_level` (0). Scene entries declare
`set` and `binding`, and bind managed TLAS resources. Each count must equal its
metadata list length and numbered binding inventory, including zero counts.

For a managed Vulkan image that shaders will write, request `storage=True` in
`image/sampled` resource parameters; its default remains false and CUDA rejects
this option. The factory defaults to one mip; 2D Vulkan mip initialization is
described below. A launch cannot
sample and store the same image. Readable images need seed pixels or an earlier
unconditional upload; arbitrary shader writes alone do not prove full image
initialization. Layout transitions and RT/compute dependencies remain Forge-owned.

Pipeline creation consumes shader blobs synchronously, so public code blobs may
close afterward. Launch preparation constructs and initializes its SBT once on
the owner's queue. Replay reuses pipeline, descriptors and SBT; it performs no
registry lookup or payload upload. Closing a public pipeline/image/AS retires
dependent recordings. Reflection checks descriptor coverage, and preparation
checks physical limits, payload ranges and base instance SBT offsets; applications
remain responsible for shader memory bounds and data-dependent record indices.

Small Windows Vulkan tests cover standalone/borrowed consumers, real instance
SBT selection and any-hit filtering, hit/miss/refit, sampled-to-storage image
processing followed by ordinary sampling, downstream gradients and retirement.
The external shaders have no automatic derivative. Ray-program tests currently
use the base mip. Raster graphics resources have a separate qualification below.

Graph export accepts an initialized `launch.graph_recording()` in ordinary
Graphs and selected ordered Vulkan compute binding frames, including qualified
compute boundaries such as AS refit. The exporter preserves selected node order;
this ray-program coverage does not qualify mixed graphics queues or fork/join.
Supply fixed buffer and Texture owners under their Graph names in
`source_bindings`. Use the existing AS and image `resource_bindings` for consumer
seeds or public resources. For the pipeline, choose either a public resource name
or a port mapping `{"shader_0": shader0, ...}` containing the original immutable
`SpirvShader` artifacts, in their original first-use order (sorted raygen/miss/hit
groups, then closest-hit/any-hit, deduplicated by artifact identity). Symbolic
host-blob ports are also accepted. JIT pipelines need not retain code bytes just
to enable a later export.

Export verifies supplied artifacts against the source metadata. It stores an
`expected_pipeline` SHA256 contract in the launch and any reconstructed pipeline,
covering ordered shader stage/entry/code and groups, recursion depth and micromap
mode. Native creation computes the contract from actual bytes once; preparation
rejects incompatible public pipelines before recording. Group names and build
paths are not native ABI inputs. Explicit plans may omit this optional contract.
SBT scalar data and push constants share content-addressed host blobs. Code blobs
are likewise deduplicated within the archive; no producer driver handles are saved.

Image usage is collected from selected launch leases before resources are emitted.
An image sampled by an earlier kernel and written by a later RT launch is rebuilt
once with both usages. Sampled-image qualification and initialization
requirements remain in force. Small tests cover both ordinary and selected Graph
export, changed-code/group rejection, public/private owners and downstream gradients.

Declare a Graph AS source owner with `source_bindings={"acceleration": tlas}`;
`resource_bindings` uses the TLAS/BLAS public-owner or explicit-seed contracts below.
Fixed TLAS bindings are recognized automatically. Kernel queries and native refit
recordings share one resource; repeated parameters/fixed aliases reconstruct it
once. Public TLAS bindings retain the BLAS alias constraints. Consumer geometry and
transforms define the new AS; export stores no source handles and does not infer
current geometry. Small tests cover public/private Graphs, fixed aliases, invalid
mappings rejected before publication, consumer refit/filtering/downstream gradients,
and execution after producer retirement.

`blas_build` rebuilds an existing BLAS with vertices and indices of the original
counts; index contents may change. Refresh each dependent TLAS before querying.
`instance_update` captures host matrices as a JSON list of `I` rows, each with 12
finite f32 scalars, and optional metadata arrays using the same widths/defaults as
the resource factory. Preparation checks that `I` matches the bound TLAS. It
records UPDATE by default or BUILD with `update=False`, preserving BLAS order and
writing all captured matrices/masks/custom IDs. This matches Vulkan's host
`record_refit()` semantics: it replaces previously packed device matrices, rather
than only refreshing cached bounds. A subsequent `instance_refit` replaces matrices
while preserving those metadata values. No GPU data is read back.

Graphs containing `TriangleScene.record_refit()`, `record_typed()` and `record()`
can be exported automatically:

```python
module.add_execution_plan(
    "step", graph,
    resource_bindings={scene: {"vertices": "seed", "indices": "triangles"}},
)
# Or bind a public consumer scene without creating a private AS:
module.add_execution_plan("shared_step", graph, resource_bindings={scene: "scene"})
```

`TriangleBLAS.record_build()` / `record_refit()`, `InstanceTLAS.record_build()` /
`record_refit()` / `record_refit_transforms()`, and its typed/legacy batch query
recordings also export through the same path:

```python
module.add_execution_plan("step", graph, resource_bindings={
    blas: {"vertices": "seed", "indices": "triangles"},
    tlas: {"transforms": "initial_transforms"},
})
# Both resources public, or public BLAS with a privately reconstructed TLAS:
module.add_execution_plan("public_step", graph, resource_bindings={blas: "blas", tlas: "tlas"})
module.add_execution_plan("mixed_step", graph, resource_bindings={
    blas: "blas", tlas: {"transforms": "initial_transforms"},
})
```

Repeated source BLAS references reconstruct once. Version-17 resource-alias
constraints preserve shared BLAS relationships in public TLAS bindings; equal
geometry in a different BLAS is insufficient. A public TLAS dependency cannot also
be reconstructed privately in the same plan. A still-live source TLAS may export
its retained geometry after the original BLAS handle closes. Construction inputs
remain explicit; export does not snapshot live AS memory or retained source
arrays. Captured host matrices and metadata are taken from each immutable
recording, even if the source TLAS is changed later. Frozen nonzero instance SBT
offsets are preserved as construction metadata. Small
Windows tests cover private/public/mixed export, invalid mappings before kernel
publication, alias rejection before execution, retained-source geometry,
standalone/borrowed consumers, and downstream gradients. An additional JIT/AOT
comparison covers index winding rebuilds, host UPDATE/BUILD followed by device
matrix updates, metadata preservation, interval misses and replay after rejected
preparation. AS construction and traversal have no automatic derivatives.

For a private BLAS with a micromap, explicitly supply its immutable baked asset:

```python
resource_bindings = {
    blas: {"vertices": "seed", "indices": "triangles", "opacity_micromap": baked_micromap},
    tlas: {"transforms": "initial_transforms"},
}
module.add_execution_plan("step", graph, resource_bindings=resource_bindings)
```

The asset fingerprint must match the source BLAS. Export stores content-addressed
host blobs and reconstructs a shared asset object once per plan. Alternatively,
use a symbolic public micromap name at that port, or bind the whole BLAS/TLAS as
a public resource. Other construction ports continue to require symbolic names;
the exporter does not serialize source GPU handles or read live AS storage.

The first form requires seeds ready before preparation and reconstructs each
source scene once across repeated queries/refits. It serializes no producer
handles and does not infer current AS contents. The public form lets query-only
plans in other modules share the same scene and observe device refits without
rebuilding it. Small standalone/borrowed Vulkan fixtures qualify exported forward
and downstream backward kernels, consumption after producer retirement and shared
closure.

Explicit AOT plans can use `runtime.configure_provider("optix", adapter_path=...,
library_path=...)` with these version 1 native resources/operations, all under
provider `optix`:

| operation | Kind / bindings | parameters |
| --- | --- | --- |
| `context` | Resource, no bindings | Optional boolean `validation` |
| `triangle_scene` | Resource: `context`, `vertices`, `indices` | `vertex_count`, `triangle_count`, optional `allow_update` |
| `triangle_gas` | Shared geometry resource: same bindings and parameters as `triangle_scene` | Same as `triangle_scene` |
| `instance_scene` | Resource: `context`, `transforms`, `gas_0` through `gas_{instance_count-1}` | `instance_count`; optional `custom_indices`, `visibility_masks`, `opaque`, `allow_update` |
| `query` | Operation: `scene`, `rays`, `hits`; typed queries require `hit_indices`; optional `faces` | `ray_count`, optional `occlusion`, `any_hit`, `front_only` |
| `refit` | Operation: `scene`, `vertices` | Empty object |
| `instance_refit` | Operation: `scene`, optional `transforms` | Empty object |

The optional `optix/program` v1 resource reconstructs an immutable programmable
pipeline through the same context and adapter. Bind `context` and `module_0`
through `module_{module_count-1}` to host blobs containing nonempty PTX without
NUL bytes. Parameters are `module_count`, `module_sha256` (one lowercase SHA256
per blob), `groups`, `parameter_name`, `parameter_size`, `payload_count`,
`attribute_count`, `max_trace_depth`, and optional `allow_opacity_micromaps`.
Each group has a unique `name`, `kind` (0 raygen, 1 miss, 2 triangle hit),
`entry={module,name}` (nullable only for hit groups) and optional `any_hit`
with the same entry shape. Module indices and OptiX stage prefixes must match.
At least one raygen group is required. An optional `expected_program` digest
also checks the complete code/group/configuration contract before compilation.
PTX blobs are consumed during creation; the program retains its context.
Discover `native_resource_capabilities("optix_program")` and explicitly probe
the configured adapter with `probe_native_resource("optix_program")` before
allocation. This requires its existing programmable-pipeline extension, with no
new adapter ABI. CPU/Vulkan do not advertise this CUDA resource.
Standalone and borrowed runtimes qualify program creation/rollback/retirement.
Automatic program Graph export is separate from the explicit resources below.

`optix/prepared_launch` v1 binds `program`, `data` (an immutable host blob),
`array_N`, `image_N` and `scene_N`. Its parameters are:

- `expected_program`: the code/group/configuration digest of the bound program;
  `dimensions`: three positive uint32 values subject to OptiX launch limits.
- `array_count`, `image_count`, `scene_count`, and `array_layouts` with one entry
  per array: `dtype`, `shape`, `element_shape`, `record_bytes`, `alignment` and
  `access` (`read`, `write`, `read_write`). Exact shape/type/alignment are checked;
  exact-range aliases merge access, while overlapping distinct writable ranges
  are rejected.
- `parameters`, `raygen`, `miss`, `hit`: a parameter payload, one raygen record,
  and arrays of miss/hit records. Each payload specifies `offset`, `size` into
  `data` and `references`; SBT records additionally specify `group`. A reference
  has `kind` (`buffer`, `texture`, `scene`), binding `index` and byte `offset`
  within the payload. Its aligned 8-byte slot must initially be zero. Every
  bound array/image/scene must be referenced. Raygen/miss/hit group kinds and
  scene context ownership are checked before publication.

Preparation resolves these owner-bearing references, lets the adapter pack the
SBT, and completes its one initialization upload. The input blob can then close;
the launch retains its program, arrays, images and scenes. Device array contents,
image uploads and scene refits may change without rebuilding the packet. Changing
fixed scalar bytes or resource bindings creates another launch. Native operation
`optix/launch` v1 binds only `launch` with empty parameters. Discover
`optix_prepared_launch` as a resource and `optix_program_launch` as an operation;
both advertise `cuda_ordered`, not capture. Replay only calls the prepared launch,
with surrounding work and completion retirement handled by the existing owner.
Images require initial pixels or an unconditional preceding upload in the plan.

The program digest encoding is defined in
[`native_optix_program.cpp`](../../taichi/runtime/native_optix_program.cpp).
Explicit plans are qualified with adapter ABI variants 93/105/118 for instance
hit/miss, indirect SBT buffers/textures, refit, scalar rebinding, source-independent
consumption, pending retirement and downstream DSL gradients. This does not
differentiate user shaders, add nonzero IAS SBT-offset/micromap reconstruction,
or establish total driver pipeline residency. Reported native device payload
includes parameter/SBT storage; the adapter also retains a pinned host copy.

Bindings use compact scalar ndarrays: f32 vertices `[V,3]`, i32/u32 indices `[T,3]`,
and f32 rays `[N,8]` containing origin.xyz, t_min, direction.xyz, t_max. Typed output
is f32 `[N,4]` for t/u/v/0 and i32/u32 `[N,4]` for primitive/instance/custom/hit;
single-scene instance/custom IDs are zero. Misses write `[-1,0,0,0]` and
`[-1,-1,-1,0]`. Occlusion writes 0/1 to i32/u32 `[N]`, omits `hit_indices`, and
defaults `any_hit` to true, which cannot be disabled. Typed `any_hit=true` also
works without face rules or alpha masks; legacy adapters receive a cold-prepared
empty mask table, while the optional fixed-query adapter needs no such table.
Ordinary typed queries
default to the closest accepted hit.

These record ports also accept AoS vector ndarrays: shape `[count]` with element
shape `[width]`, using widths 3, 8, 4 or 1 as appropriate. Single-word ports
(face tables and occlusion) additionally accept scalar `[count,1]`. Storage must
be packed and match the exact record count; equal byte counts do not permit
flattened or differently shaped records. No layout conversion or copy is added.

In query version 1, omitted face rules accept both sides. `front_only=true` rejects backfaces by
winding and continues traversal. Alternatively, `faces` binds i32/u32 `[T]`, with
0 for two-sided and 1 for front-only; the two policies cannot be combined.
Topology and face rules are validated and frozen at preparation; changing them
requires another preparation. Refit updates vertices with the same topology and
requires finite device inputs, without per-frame host readback. Query outputs
cannot overlap inputs or each other. Preparation builds scenes/pipelines without
executing queries.

An `instance_scene` references `triangle_gas` resources from the same context;
several instance ordinals can reference one GAS. `query` accepts a triangle or
instance scene, not a bare GAS. Instance `transforms` are compact f32 row-major
affine 3x4 matrices: scalar `[I,12]` or `[I,3,4]`, or AoS shape `[I]` with element
shape `[3,4]`. These layouts apply to construction and device refit.
Optional metadata arrays have `I` entries:
`custom_indices` are unsigned 24-bit IDs (default 0), `visibility_masks` are
unsigned 8-bit masks (default 255), and `opaque` entries are booleans (default
false). Returned instance IDs are zero-based binding ordinals; custom IDs and
GAS-local primitive IDs remain distinct. Opaque instances still apply face
filtering. With instances, `faces` has `sum(T_i)` entries concatenated in instance
ordinal order, then each GAS's primitive order. Instances sharing a GAS can
therefore have different rules.

Query version 2 additionally binds live device tables. Set `version=2`, provide
`face_rules` in instance order (one entry for `triangle_scene`), and set
`face_table_count` to the number of `"table"` entries. Rules are `"two_sided"`,
`"front_only"` or `"table"`. Bind `faces_0`, `faces_1`, etc. in the order that
`"table"` entries occur; each is compact i32/u32 `[T_i]` for that instance's GAS
primitive order. For example, `face_rules=["front_only", "table", "two_sided"]`
uses `face_table_count=1` and only `faces_0`, for the second instance. Several table
ports may reference the same array when their primitive counts match.

Version 2 retains table addresses and reads their current values during traversal;
it neither snapshots them nor reads values back to the host. Values must be 0
(two-sided) or 1 (front-only) before each query, and can be produced by preceding
kernels in the same plan. Preparation checks layout, count and output aliases;
it does not require initialized table values. Rule modes and bindings stay fixed
until another preparation. The v1 `faces` and `front_only` parameters are not
accepted by v2; ray/output layouts, `ray_count`, `occlusion` and `any_hit` keep
their existing contracts. All-two-sided rules preserve the original query path.
Discover this capability before preparation using
`runtime.native_operation_capabilities("optix_query", version=2)`; the CUDA
factory reports `{"cuda_ordered"}`. Existing v1 plans retain frozen-table behavior.

Explicit query version 3 adds managed alpha images while preserving the v2 face
rules. `alpha_masks` has one entry per instance: `null` for no mask, or
`{"cutoff": 0.5, "channel": 0}` (both values have these defaults). Set `mask_count`
to the number of non-null entries and bind `uvs_0`/`image_0`, etc. in their order.
UVs are live compact f32 `[V_i,2]` or AoS float2 `[V_i]` in the GAS vertex order;
the image binding names a managed resource, never a raw texture handle. Alpha
uses the same frozen index table as the GAS. A mask on an `opaque=True` instance
is rejected during preparation. Face and alpha rules both have to accept a
candidate; rejecting it continues traversal for typed and compact occlusion
queries. The adapter must advertise face/alpha composition when both are used.

Create the image through the common native resource registry:

```python
image = {
    "kind": "native", "name": "alpha_image", "provider": "image",
    "operation": "sampled", "version": 1,
    "parameters": {"shape": [2, 4], "format": "r32f", "filter": "nearest",
                   "address_modes": ["clamp_to_edge", "clamp_to_edge"]},
    "bindings": {"pixels": "initial_alpha"},
}
upload = {
    "kind": "native", "provider": "image", "operation": "upload", "version": 1,
    "parameters": {}, "bindings": {"image": "alpha_image", "pixels": "next_alpha"},
}
```

The optional `pixels` resource input is copied once at preparation. Without it,
an unconditional `image/upload` must precede sampling; a write only inside one
branch or a zero-trip loop is insufficient. Uploads read the current device array
on the execution stream and can follow a kernel that generates texels. They do
not add a host readback or a whole-context fence on replay. A changed source array
alone does not change the image. Shape and sampler stay fixed until preparation
of a new resource. Images shared by queries are retained through execution
retirement, including independent prepared bindings and borrowed-runtime reset.

For OptiX, images must be single-level 2D `r32f` or `rgba32f`; channel is 0 or 0–3
respectively and cutoff is finite in `[0,1]`. Scalar pixel arrays use the logical
image shape, followed by a channel axis for RGBA; packed AoS channel vectors are
also accepted. CUDA preserves the existing Forge layout: logical `[X,Y]` is a
physical CUDA array of width `Y`, height `X`; the OptiX shader reverses the CUDA
coordinates so normalized UV `(u,v)` samples the first and second logical axes.
The default filter is `linear` and
address mode is `repeat`; `nearest`, `clamp_to_edge` and `mirrored_repeat` are
available. These defaults match managed JIT textures.

Discover `optix_query` version 3 (`cuda_ordered`) and `image_upload` version 1
(`cuda_capture`) before preparing. Opaque queries still use their prepared-query
fast path. Alpha uses the existing ordered adapter API; this does not qualify
OptiX stream capture. CUDA array residency is driver-private: packed texel bytes
are `product(shape) * channels * channel_bytes`, separate from reported query
workspace, which does not include image residency. Alpha recordings can export
through the image reconstruction contract below.

With matching private ABI83 components, explicit execution-plan kernels on CUDA or Vulkan
may bind sampled `Texture` arguments directly to `image/sampled` resource names:

```python
{"kind": "kernel", "entry": "sample", "arguments": ["alpha_image", "output"]}
```

The compiled parameter must be a sampled Texture of the same dimension. Image
owners stay alive through backend completion, including shared CUDA binding
frames. Upload-before-sample is checked over the entire control flow: an upload
only inside a conditional or zero-trip loop cannot initialize a later sample.
Resource creation/qualification is cold; replay uses the existing CUDA launch
packet without resolving Python texture names or inspecting texels.

For `cuda_immutable_argument_frames`, image-upload source addresses must remain
fixed across sibling bindings: CUDA cannot update the operands of captured
multidimensional/array copies. Source contents and other kernel arguments may
change. An incompatible sibling is rejected before CUDA Graph update and leaves
the original executable usable; use independent preparation for a new upload
source. This restriction follows the [CUDA Graph update rules](https://docs.nvidia.com/cuda/archive/12.2.2/cuda-c-programming-guide/index.html).

Windows CUDA checks cover normalized 1D, non-square 2D linear/nearest sampling,
3D axis order, seeded and GPU-updated images, captured and OptiX-ordered plans,
shared binding retirement, and real gradients of downstream numeric kernels.
This does not differentiate texture sampling. CUDA RWTexture, CUDA TextureCollection and
ordinary public C API image handles still require separate qualification.

Vulkan `texture_collection` kernel arguments can bind managed tables in explicit
`ExecutionPlan` objects. Declare the target indexing feature and minimum sampled
descriptor budget when exporting, for example:

```python
module = ti.aot.Module(caps=[
    "spirv_has_sampled_image_array_non_uniform_indexing",
    "max_sampled_texture_collection_size=3",
])
collection = dict(
    kind="native", name="table", provider="image", operation="collection", version=1,
    parameters=dict(dimensions=2, image_count=3),
    bindings=dict(image_0="second", image_1="first", image_2="second"),
)
```

Members are same-owner `image/sampled` resources. These `Module.add_kernel()`
artifacts require GFX metadata v5, so older loaders reject them explicitly.
Ordinary and RWTexture artifacts keep their previous minimum version. The
descriptor budget must cover all sampled bindings in one shader, including
other tables. Query `native_resource_capabilities("sampled_image_collection")`
before selecting this route; preparation checks the actual device limits.
Public tables can also be created with `runtime.native_resource("image",
"collection", ...)` and existing image `NativeResource` bindings. Slots retain
their order and may repeat owners; sizes and samplers may differ, but dimensions
must agree and formats must support floating/normalized sampling. Replacing
members requires a new table and preparation; uploading pixels preserves the
table. Collections own CPU metadata and image leases, without copying images.
Closing a member retires dependent tables and executions. Every member needs
initial pixels or an unconditional preceding upload; the application guarantees
valid GPU indices. Standalone and borrowed checks cover 2D fetch/sample_lod,
GPU updates, rebinding, retirement and downstream numeric gradients. Texture
sampling itself is not differentiated. Legacy CGraph and the kernel-template
interface remain unsupported.

Ordinary Graphs and selected Vulkan whole-graph immutable binding recipes can
reconstruct collections automatically. Supply the source `TextureCollection`
for each symbolic collection in `source_bindings`; private fixed tables come
from the Graph's existing bindings. Map unique members to pixel input ports or
public resource names in `resource_bindings`. Tables and individual image
arguments share one registry, allocating and initializing each repeated member
once. Preflight includes later RWTexture/attachment uses, preserving write usage
when sampling occurs first. Export never reads producer texture contents.
The whole collection may instead be public. Members also used by other kernels
or providers must retain consistent public identities; preparation checks aliases
through the corresponding `image_N` resource paths. A public table's shared
member cannot also be rebuilt privately. Checks cover private/public/mixed
members, sampling then storage writes then sampling, released initialization
inputs and retirement. Ordinary graphs include actual downstream reverse kernels;
selected graphs use a numeric reference. Segmented/fork-join recipes are not expanded.

Vulkan uses the same `image/sampled` and `image/upload` descriptions, backed by
the owner's existing GFX runtime. Discover `image_upload` version 1 with
`vulkan_inline` composition. Creation and initial pixel upload occur during
preparation; recorded uploads and sampling share the existing command list,
layout transitions and retirement. There is no extra warm host copy or wait.
Kernel images support single-level 1D/2D/3D, plus 2D Vulkan mip chains described
below. Vulkan also accepts extended sampler state; CUDA retains basic sampling.
Qualified Graphs use the automatic reconstruction contract below.

Pixel layout follows each backend's existing texture ABI. Vulkan image X is
contiguous: a packed scalar texel at `(x,y,z)` has index `(z*Y+y)*X+x`;
CUDA's managed logical ndarray axes use the reversed physical mapping described
above. A `[X,Y,Z]` seed names the packed storage extent, not an implicit
transpose. Windows Vulkan checks cover normalized 1D, non-square 2D linear/nearest
sampling, 3D coordinates, generated uploads, missing/conditional initialization,
real downstream numeric gradients and standalone/borrowed retirement.

Vulkan `depth32f` uploads use the depth aspect and request graphics-capable
recording through the same native-operation requirements. Initial uploads use
the existing ordered GFX graphics path; replay records the complete sequence on
that queue without a per-upload host wait. Callers supply depth values in `[0,1]`,
as required by the [Vulkan copy contract](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdCopyBufferToImage.html).
This qualifies 2D depth sampling and uploads, including complete mip chains.
Explicit raster plans can render to AOT-owned attachments as described below.

GFX AOT exports containing `rw_texture` arguments require metadata v4 and
paired ABI90 components. Argument metadata preserves the exact format and mip
level, including kernels that only inspect the image shape. Ordinary exports
keep their earlier metadata version. Old SPIR-V offline-cache entries missing
the new argument field are rejected by strict cache decoding and recompiled;
CPU/CUDA cache formats are unchanged.

`image/sampled` accepts an optional `sampler` object with the fields of
`ti.hardware.sampling.SamplerConfig`: `min_filter`, `mag_filter`, `mip_filter`,
`address_mode_u/v/w`, `lod_bias`, `min_lod`, `max_lod`, `max_anisotropy` and
`compare_op`. Omitted fields use the same defaults; `max_lod=null` leaves the
upper LOD unclamped and `compare_op=null` disables comparison. Do not combine
`sampler` with the legacy `filter` or `address_modes` fields.

Vulkan reconstruction and borrowed images preserve this state using the existing
sampler cache. Finite-number, device-limit and format checks run before use.
Comparison requires 2D `depth32f` and a comparison-sampling kernel or matching RT
shader descriptor; ordinary filtered sampling cannot use a comparison sampler.
CUDA rejects extended state and unequal min/mag filters. Windows tests cover
color filtering/addressing, depth comparison, anisotropy where available,
public/private ordinary and selected Graphs, uploads, borrowing and downstream
gradients. Sampler derivatives are not provided. Depth uploads through
`Texture.from_ndarray()` and `Texture.from_field()` also use the managed
depth-aspect graphics transfer path, preserving layout tracking between Graph
replays. Vulkan Field uploads require matching shape/type and tightly packed
root-dense storage; convenience uploads update only mip zero.

For 2D Vulkan images, `image/sampled` also accepts `mip_levels` (default `1`).
The count cannot exceed the complete chain implied by the dimensions or device
limits. Initialize all allocated levels with `image/upload_mips` version 1:
parameters are `{"mip_levels": N}`, and bindings are `image` plus `pixels_0`
through `pixels_{N-1}`. Each array describes tightly packed texels with extent
`(max(1, X >> level), max(1, Y >> level))` in the layout above.

This operation records the complete chain in the existing command stream and
establishes the whole-image initialization fact. Put it in the plan's
`initialization` sequence for a one-time upload, or its normal operations for
updates. No per-level host wait is added. Single-level `image/upload` and the
construction-time `pixels` binding reject mip chains; they cannot establish
that other levels have valid contents. Conditional uploads alone cannot justify
subsequent unconditional sampling. Region updates use the transfer operations
below; automatic mip generation is a separate capability.

Graph export reconstructs a private mip chain with
`resource_bindings={texture: {"pixels_0": "seed0", "pixels_1": "seed1", ...}}`,
covering every allocated level. Public resources can be imported initialized
from JIT, or initialized explicitly in the consuming plan. Windows tests cover
non-square three-level color/depth chains, fractional LOD filtering, fetching,
ordinary/selected export, borrowing, updates, retirement and downstream gradients.
Logical texel bytes include every level; they are not driver residency estimates.

### Explicit Vulkan image region transfers

The common native registry provides version 1 operations under provider `image`:

| Operation | Capability kind | Bindings | Parameters |
| --- | --- | --- | --- |
| `copy_region` | `image_copy_region` | `source`, `destination` images | `source_region`, `destination_region` |
| `blit` | `image_blit` | `source`, `destination` images | Both regions; optional `filter`: `nearest` (default) or `linear` |
| `buffer_to_image` | `buffer_to_image` | `buffer`, `image` | `region`; optional `byte_offset`, `row_length`, `image_height` |
| `image_to_buffer` | `image_to_buffer` | `image`, `buffer` | Same buffer layout and region fields |

Query `native_operation_capabilities(kind)` for `vulkan_inline`; CPU/CUDA do not
advertise these operations. A region has a required three-component `extent`,
optional `offset` (default zero), `mip_level` (zero), `base_layer` (zero), and
`layer_count` (one). Current managed images have one array layer. Copy requires
equal extents; copy/blit require distinct same-format, same-dimension color
images. Blit also checks actual device format/filter support and uses a graphics
queue. Buffer transfers support color, `depth16` and `depth32f` owners; depth
uses a graphics queue. Packed depth/stencil is not supported.

`depth16` stores normalized unsigned 16-bit texels (`u16` pixels); shader fetches
return floating depth. Both depth formats require four-byte buffer offsets after
composing view and operation offsets. This also applies to construction-time
pixels and `upload`/`upload_mips`. Comparison samplers remain limited to the
existing 2D `depth32f` contract.

Buffer offsets are bytes relative to the supplied finite array/view. Row length
and image height are texel counts; zero means tightly packed. Shape/type need
not mirror the image because these operations copy raw texels. Preparation checks
actual allocation bounds, the finite view, texel/aspect alignment, pitch and
overflow. Unaccessed padding after the last row/slice does not count toward the
required range. Bytes and texels outside the region remain unchanged.

Sources must already be initialized. A complete single-level destination write
establishes initialization; partial regions or one level of a mip chain require
an initialized destination. Use `upload_mips` to initialize a full chain first.
The existing conditional-control initialization and resource retirement contracts
apply. No warm host probes, extra workspaces or staging copies are introduced.
Windows tests cover private/public resources, finite views, padding, mip updates,
depth transfers, ownership retirement and downstream numeric gradients in both
standalone and borrowed consumers. This does not differentiate image copies or
sampling.

Ordinary Graph export also reconstructs `VulkanImageCopyRecording`,
`VulkanImageBlitRecording`, `VulkanBufferToImageRecording` and
`VulkanImageToBufferRecording`. Supply each source Texture in `source_bindings`;
use `resource_bindings` for public images or consumer pixel seeds. A first complete
single-level overwrite needs no seed. Partial destinations still need one.
Fixed packed ndarray views reconstruct their finite range against a named parent
in `source_bindings`; module Field views retain the existing explicit snapshot
contract. Export never snapshots texture contents implicitly. The native
consumer qualifies actual ranges and initialization before recording. These JIT
recordings currently use rerecording, so this does not add a selected immutable
Vulkan whole-graph recipe. Tests include whole integer-image copy and region
transfers followed by real numeric gradients.

### Explicit Vulkan SPD pyramids

The native registry provides version 1 resource `spd/plan` (capability kind
`spd_plan`) and operation `spd/run` (`spd_run`, `vulkan_inline`). CPU/CUDA do not
advertise them. A plan binds same-owner `source` and `output` images and immutable
SPIR-V host blob `shader`. Its `reduction` parameter is `mean`, `min` or `max` and
must match the producer's generated code. Consumers need neither SPD headers nor
a shader compiler. Creation consumes the blob synchronously; image owners remain
retained until recordings retire.

Run binds `plan`, `source` and `output`, checking fixed image identities during
preparation. Optional `expected_plan` is a length-delimited SHA256 contract over
code, formats, dimensions, mip counts and reduction for public plan replacement;
it does not prove the semantics of arbitrary shader code. Images require storage
usage. Source initialization is required; SPD writes every allocated output mip,
establishing the same initialization fact for later sampling.

Existing SPD2.0 formats, half-size output, shorter-axis stopping and odd-edge crop
semantics remain. Each new plan owns a four-byte counter, cleared and retired once
during cold creation. Replay remains one dispatch without per-frame clearing,
CPU readback or extra staging. AOT records inline without an unused secondary
replay. Content-addressed code, GPU image/counter payload and opaque driver storage
remain separate accounting categories.

Small Windows Vulkan tests cover single-group and 129x131 seven-level multi-group
paths, every mip, positive/zero/negative inputs, standalone/borrowed owners,
public/private resources, invalid contracts/aliases and close. Real gradients
exercise downstream numeric kernels, not differentiation of SPD or texture
sampling itself.

`VulkanSpdPlan.record()` also exports ordinary Graphs and existing selected Vulkan
whole-Graph recipes. Plan construction freezes the actual shader and code/image
contract. Export invokes no shader compiler and serializes no local SDK paths or
handles. Repeated recordings share one code blob and plan resource. Supply the
original textures through `source_bindings`, for example:

```python
module.add_execution_plan(
    "step", graph,
    source_bindings={"source": source, "pyramid": pyramid},
    resource_bindings={source: {"pixels": "source_seed"}},
)
```

This assumes `record(source="source", output="pyramid")`. Source seeds use the
existing explicit one-time initialization; fully overwritten outputs need none.
A source already initialized by a provable earlier upload reuses that fact.
For a public plan, use `{spd_plan: "public_plan"}` and also expose both images.
Export records shared dependency aliases; consumers reject different images even
when their shapes match. Tests cover a preceding kernel, two calls to one plan,
every mip, downstream backward execution and reuse after initialization inputs
close. JIT recipe admission is unchanged.

### Explicit Vulkan compute plans

`vulkan_compute/pipeline@1` reconstructs a fixed SPIR-V compute pipeline from
one `shader` host-blob binding. Its `expected_shader` parameter is the lowercase
SHA256 of the code. The pipeline consumes the blob during preparation, so the
blob can then close. `vulkan_compute/dispatch@1` binds this pipeline and arrays
through the same resource registry, owner and prepared-execution contracts used
by other native operations:

```python
operation = {
    "kind": "native", "provider": "vulkan_compute", "operation": "dispatch",
    "version": 1,
    "bindings": {"pipeline": "compute", "array_0": "input", "array_1": "output"},
    "parameters": {
        "expected_shader": shader_sha256, "groups": [group_count, 1, 1],
        "array_count": 2,
        "buffers": [
            {"set": 0, "binding": 0, "array": 0, "storage": True},
            {"set": 0, "binding": 1, "array": 1, "storage": True},
        ],
    },
}
```

The example requires a shader with exactly those two storage-buffer descriptors.
V1 supports `main`, fixed local sizes, and scalar storage/uniform-buffer
descriptors across descriptor sets; images, descriptor arrays, push constants
and specialization overrides are outside this version. Uniform buffers use
`storage=False`. Logical views must have the required physical usage,
alignment and range. Shader reflection verifies descriptor coverage and fixed
block sizes; valid dynamic indices, shader data types and safe aliasing remain
the shader author's responsibility.

Cold preparation checks the code digest, local/group counts against device
limits, and all bindings before recording. Zero group counts perform no work.
Replay uses prepared descriptors and the existing Vulkan command batch; it
does not compile a DSL kernel, resolve names or rebuild pipelines. Buffer
contents may change between runs; different array bindings prepare a new
immutable execution. Pipelines and arrays retire with their owning execution.
The artifact can own the shader/pipeline and scratch arrays or explicitly bind
an external pipeline with the same code contract. Driver pipeline compilation
still occurs at preparation. Reported zero provider workspace excludes driver
pipeline/descriptor residency and caller-owned arrays.

Windows qualification covers independent and borrowed consumption, chained
compute dispatches, array updates, rejected bindings, owner retirement and a
downstream DSL gradient kernel. The external shader is not differentiated.
CPU/CUDA do not advertise these Vulkan capabilities. This is a shared foundation
for native preprocessing, including the RasterPass line expansion below.

### Exporting high-level RasterPass recordings

Vulkan `RasterPass` mesh, mesh_instance, particles and lines recordings export through
ordinary explicit Graphs. Preparation kernels, private geometry scratch and the
native graphics pass share one AOT execution sequence. The artifact contains no
Window, Camera or producer Python objects. Consumers do not import Forge, call
the UI frontend or need a shader compiler; first preparation still creates native
pipelines and descriptors. Discover `vulkan_graphics_pass_direct` version 5 before
choosing this route. CPU/CUDA do not provide this drawing path.

For an existing `raster` whose draws use `centers`/`transforms` ndarrays:

```python
builder = ti.graph.GraphBuilder()
builder.append_native(raster.record(), admission="explicit")
graph = builder.compile()
module = ti.aot.Module()
module.add_execution_plan(
    "draw", graph, snapshot_state=True,
    source_bindings={"centers": centers, "transforms": transforms},
    resource_bindings={raster.color_texture: "color", raster.depth_texture: "depth"},
)
module.archive("renderer.tcm")
```

Name only external ndarrays actually used by the draws. Template Fields retain
the explicit dense-root snapshot contract. Camera state freezes at export;
ambient and point lights retain their recorded values. Change these parameters
by exporting again at a cold boundary. Consumer code can update geometry, indices
and instance matrices with the same layouts, with no new per-frame host checks.
External mesh indices require `index` allocation usage, plus `storage` when preparation
kernels also read them. Line indices require `storage` usage for compute expansion.
Index values must satisfy the source draw's valid vertex
range contract; export does not scan device indices on the host.

Without `resource_bindings`, plans rebuild and clear their attachments without
pixel seeds. Public `image/sampled` attachments require `attachment=True`, the
recorded dimensions, RGBA8 color and depth32f depth. Preparation checks these
layouts; closing public attachments retires dependent executions. A downstream
Texture kernel uses matching `source_bindings` to share the same image. Empty
draws retain the clear but omit unused geometry preparation, uniforms and pipelines.

Lines use the packaged compute shader through `vulkan_compute_pipeline` and
`vulkan_compute_dispatch` v1, followed by the same graphics pass. Discover both
compute capabilities before choosing this route. Each complete line needs 152B
of persistent expanded vertex/index storage in addition to the common source
packing buffers. Repeated recording occurrences share this storage and overwrite
it in sequence; replay performs no geometry allocation. Requested offsets/counts
and per-vertex colors retain JIT semantics, including discarding an incomplete
odd tail. No complete lines means no expansion resources. The shared line uniform
builder checks the shader's actual signed-index arithmetic before narrowing;
dispatch and buffer ranges also obey the active Vulkan device limits.

Small Windows fixtures compare JIT color/depth with standalone and borrowed
consumers for indexed/non-indexed meshes, instance offsets/count clipping,
particle ranges, line offsets and odd tails, updated geometry/indices/colors,
mixed draws, close/reset and downstream numeric gradients.
Rasterization itself is not differentiated. JIT automatic admission and selected
recipe qualification are unchanged.

### Explicit Vulkan raster plans

The common native registry exposes `vulkan_graphics/pipeline` and
`vulkan_graphics/pass_direct`, both version 1. Discover the resource kind
`vulkan_graphics_pipeline` and operation kind `vulkan_graphics_pass_direct`
before allocating; the latter requires `vulkan_inline`. CPU and CUDA do not
advertise these operations. Pipelines and images may be private plan resources
or externally shared native resources, with the existing owner/retirement rules.

Pipeline parameters contain `shader_count`, `shaders` (each with `stage` equal
to `vertex` or `fragment`), `vertices` (`binding`, byte `stride`, optional
`instance`) and `attributes` (`location`, `binding`, format name, byte `offset`).
Bind SPIR-V host blobs as `shader_0`, `shader_1`, etc. Creation consumes their
bytes synchronously; the blobs may then close. Optional raster parameters are
`topology`, `polygon_mode`, `front_face_cull`, `back_face_cull`, `depth_test`,
`depth_write`, `depth_compare`, `depth_bias_constant`, `depth_bias_slope`, plus
per-target blend/write-mask descriptions in `color_targets`. These use the same
states as `VulkanGraphicsPipeline`. Push-constant shaders are rejected because
this draw contract has no push-constant binding.

Pass parameters declare `pipeline_count`, `array_count`, `image_count` and
`color_count`, bound respectively as `pipeline_N`, `array_N`, `image_N` and
`color_N`; `depth` is optional. `colors` contains one `{clear, clear_bits}` per
attachment, where `clear_bits` is four uint32 words containing the float clear
values. `depth_clear`, `clear_depth` and `[x,y,width,height]` `viewport` are
optional. Each entry in `draws` specifies:

- `pipeline`: the pipeline ordinal, and `element_count`;
- `vertices`: `{binding, array}` entries, and optional `index` array ordinal;
- `buffers`: `{set, binding, array, storage}` descriptors (`storage=false` is uniform);
- `images`: `{set, binding, image}` sampled-image descriptors;
- optional `instance_count`, `first_vertex`, `first_index`, `first_instance`
  and `vertex_offset`. Indexed draws require u32 indices and explicit inclusive
  `index_min`/`index_max`, which the caller must maintain when indices change.

Resource ordinals address those numbered groups, not allocation IDs. Both a
pipeline description and a draw may specify `expected_pipeline`, a SHA-256
contract over compiled code and normalized raster/vertex state. Mismatch is
rejected before recording. Descriptor coverage/type/count, physical usage,
logical ranges, alignment and device limits are checked during preparation.

Use `runtime.ndarray(..., usage=("storage", "vertex"))` for arrays shared by
compute and drawing; supported allocation usages are `storage`, `uniform`,
`vertex`, `index` and `indirect`. The C API exposes `TI_MEMORY_USAGE_INDIRECT_BIT`;
explicit indirect-only buffers need no storage usage unless a kernel accesses
them. Existing Vulkan storage allocations retain their implicit indirect usage
for GPU-produced dispatch packets. The default remains `("storage",)`. This declaration
does not add synchronization or inspect index contents. Plan-private ndarray
resources accept the same `usage` list with execution format v18 and paired
ABI88 components. Omitted usage keeps the previous storage allocation and
does not raise the plan version. `ndarray_view` inherits its parent's actual
usages and cannot redeclare them. Archived `initial_data` works with vertex,
index and uniform buffers, allowing fixed drawing data to remain private.
Vulkan kernel/Graph bindings check actual storage-buffer usage for primal,
gradient and struct ndarrays during preparation. A vertex-only allocation is
valid for copies and drawing, but cannot be a kernel ndarray. Views inherit
the physical allocation's restrictions; no replay-time usage check is added.
For images, `image/sampled` accepts `attachment=true` on Vulkan. Attachments
must be single-level 2D images. A first complete clear needs no seed image or
upload, even with a partial draw viewport. LOAD and sampling require an earlier
unconditional initialization; a conditional-only clear does not suffice.
Sampling an attachment alias within the same pass is rejected. Queue choice,
layout transitions, completion and resource retirement use the existing GFX
executor, with no per-draw Python callback or extra host wait.

Windows qualification covers standalone and borrowed execution, direct and
indexed/instanced draws, uniform/sampled descriptors, color/depth clear and
LOAD, changed inputs and bindings, shared/private retirement, and downstream
numeric gradients. This is not differentiation of rasterization. Storage arrays,
direct mesh and classic indirect draws use the forms below.

Ordinary Vulkan Graphs containing `VulkanGraphicsDrawRecording` or
`VulkanGraphicsPassRecording` can export these qualified direct/indexed/instanced
draws through `module.add_execution_plan`. Supply the producer Textures in
`source_bindings`; drawing ndarrays keep their Graph names and require the
appropriate consumer buffer usages. For a privately reconstructed pipeline,
`resource_bindings={pipeline: {"shader_0": vertex_shader, "shader_1": fragment_shader}}`
accepts the original immutable `SpirvShader` assets or consumer host-blob names.
Alternatively bind the pipeline owner to a public consumer resource name.
The frozen creation-state and shader-digest contract checks public pipelines
and consumer-supplied blobs before recording; no driver handles are archived.

Image aliases collect attachment/storage usage before reconstruction. A first
full clear permits an unseeded private attachment; earlier reads or LOAD require
explicit pixel seeds or initialized public images. Native preparation checks
initialization along the complete control flow, including conditional clears.
Windows validation covers both recording classes, integer MRT clears, sampled
and uniform descriptors, aliases, public/private owners, replay, retirement and
actual downstream kernel gradients. The explicitly selected complete graphics
queue recipe can also export qualified pass recordings: both JIT and AOT use
the same fixed GFX graph builder on the compute-capable graphics queue. This
adds no format version or consumer interpreter. Segmented compute/graphics and
independent fork/join recipes remain rejected until their distinct submission
contracts are preserved. The cold export reuses compiled kernel entries.

`vulkan_graphics/pass_direct` version 2 additionally accepts managed shader AS
bindings. Query `native_operation_capabilities("vulkan_graphics_pass_direct",
version=2)` before choosing this form; version 1 remains unchanged. Add
`scene_count`, numbered `scene_N` bindings and each draw's `scenes` list of
`{"set": 0, "binding": 0, "scene": 0}` records. These reference existing
`vulkan_ray/instance_tlas` resources. Pipeline resources stay at version 1.
The shared owner service binds descriptors and merges reflected vertex/fragment
read stages per AS before recording the pass. Initialization, refit visibility,
queue handoff and retirement reuse the existing AS core. Duplicate/missing
descriptors and different-device owners are rejected before execution.

Graph export selects pass version 2 when AS descriptors occur without arrays. Supply each
source `InstanceTLAS` under its Graph name in `source_bindings`; use the same
explicit BLAS/TLAS seeds or public resource mapping as compute ray queries.
Refit and raster consumers share the reconstructed scene. Paired private ABI91
components are required for the shared shader-stage contract. Windows checks
cover simultaneous vertex/fragment queries, device transform refit with
hit/miss/hit output, private/public scenes, actual downstream kernel gradients
and owner retirement. External ray queries and rasterization remain outside AD.

`vulkan_graphics/pass_direct` version 3 adds fixed storage-buffer descriptor
arrays, with paired ABI92 components. Query
`native_operation_capabilities("vulkan_graphics_pass_direct", version=3)` for
schema availability; preparation also checks device descriptor-indexing/count
limits and the reflected shader interface. Pipeline resources remain version 1.
Each `buffers` item has either scalar `array` or an `arrays` list of at least
two ordinals, for example `{"set": 0, "binding": 0, "arrays": [0, 1, 2, 0],
"storage": true}`. These reference the existing `array_N` ports and may repeat
an owner. Version 3 retains version 2's AS fields; supply `scene_count=0` when
unused. Versions 1/2 reject the new array form.

Ordinary and selected complete graphics-queue Graphs automatically choose v3
for `ShaderBufferArrayBinding`. Explicit AOT plans may bind public/private
`ndarray_view` resources, preserving each element's offset and finite byte range
through descriptor caching and recording. No array payload copy is required.
Windows tests cover actual shader array-length observations, aliases, public
and private parents, GPU selector updates, replacement bindings, retirement and
downstream gradients in standalone and borrowed consumers.

Direct mesh draws use `vulkan_graphics/pipeline` v2 and `pass_direct` v4.
Query `native_resource_capabilities("vulkan_graphics_pipeline", version=2)` and
`native_operation_capabilities("vulkan_graphics_pass_direct", version=4)` for
factory availability; preparation also checks the physical mesh/task features
and their separate workgroup limits. Pipeline v2 accepts mesh/fragment with an
optional task stage, or the existing vertex/fragment form. Its code/layout
contract and shader blob ports are unchanged; mesh pipelines have empty
`vertices` and `attributes`.

A mesh draw replaces `element_count` and the other vertex/index draw fields
with `"mesh": [x, y, z]` and has empty `vertices`. It must bind a mesh pipeline;
mixing these fields, binding a vertex/index input, or exceeding device limits is
rejected before recording. Pass v4 retains v3's descriptor arrays and AS fields
(`scene_count=0` when unused). AS reads in mesh/task stages remain unqualified
and are rejected; the existing vertex/fragment AS contract is unchanged.

`VulkanMeshPassRecording` exports through ordinary or selected complete
graphics-queue Graphs using the same prepared pass, descriptors and retirement
path as raster draws. Provide every shader asset, including task when present,
in pipeline stage order, or bind a public pipeline. Windows tests cover mesh-only
and task+mesh, compute-to-mesh data updates, public/private owners, replacement
bindings, close/reset and downstream numeric gradients in standalone and
borrowed consumers. No consumer shader compiler is required. Older raster
exports keep their lowest required versions. Mesh indirect and segmented/fork-join
submission coverage remains unchanged.

`pass_direct` v5 retains v4 and permits empty `draws` for a clear/load-only pass.
An optional `attachment_contract={"width": w, "height": h, "colors": ["rgba8"],
"depth": "depth32f"}` checks exact attachment dimensions, formats and single-mip
2D layouts during preparation, with no per-frame checks. Omit `depth` when there
is no depth attachment. The color format list matches `color_count`; specify
`scene_count=0` without AS bindings. Query
`native_operation_capabilities("vulkan_graphics_pass_direct", version=5)`.
Earlier versions reject this contract and retain their nonempty-draw requirement.

Classic draw indices accept `i32/u32` storage, both interpreted as Vulkan uint32
indices without conversion copies. Signed values must also be nonnegative and
within the declared `index_min/index_max`: these are caller promises about device
contents, not host scan results. Indirect command/count arrays still require `u32`.

Classic indirect draws use `vulkan_graphics/pass_indirect` v1. Query
`native_operation_capabilities("vulkan_graphics_pass_indirect")` before choosing
this operation; preparation separately checks device count/multi-draw features
and limits. Ordinary and selected complete graphics-queue Graphs export
`IndirectDraw` automatically, including mixed classic direct/indirect passes.
Mesh draws cannot be mixed into this form. Direct versions retain their schemas.

The operation keeps the existing pipeline/array/image/color/scene groups
(`scene_count=0` when unused). An indirect draw replaces direct count/offset
fields with an `indirect` object containing `commands` (array ordinal), optional
`count`, `max_draw_count`, `stride`, `vertex_record_limit`, optional
`instance_record_limit=1`, `index_element_limit=0`, `command_offset=0`,
`count_offset=0` and `first_instance_may_be_nonzero=false`. Top-level `index`
selects indexed drawing; direct `index_min/index_max` fields are not used.
Command/count data use u32, offsets are bytes relative to the finite logical
view, and stride is the Vulkan command ABI stride (16/20 bytes minimum).
Declared vertex/instance/index limits are producer promises; preparation does
not read GPU commands back to verify their contents. GPU count zero draws nothing.

JIT/AOT share range, feature, geometry and pass-wide alias checks. Writable
shader storage cannot overlap command/count spans in the same pass; use separate
passes for dependent draws. Known imported-memory aliases are checked, and
unknown identities require proof of separation. No per-frame host command
inspection or extra submission manager is introduced. Windows tests cover
non-indexed fixed-count and indexed GPU-count draws, mixed direct draws,
0/1/0/1 updates, public/private finite views, replacement bindings, close/reset
and downstream gradients in standalone/borrowed consumers. This does not add
graphics-shader differentiation, mesh indirect, or segmented/fork-join export.

Explicit Vulkan plans also accept `image/storage` version 1 resources with
`shape`, `format`, optional `mip_levels` and `attachment` parameters. They use
the same owner, `image/upload` and `image/upload_mips` operations as sampled
images. Optional `pixels` supplies a completed cold upload for a single-level
image. Storage images have no sampler and do not implicitly acquire sampled
usage; use `image/sampled` with `storage=true` for compatible floating images
that need both usages. CPU/CUDA reject the storage resource factory.

Kernel entries bind these resources by name to `ti.types.rw_texture` arguments.
Preparation checks storage usage, exact format, dimensionality and allocated
mip level; `image.shape` reflects the selected mip. Signed/unsigned integer
color attachments retain their exact clear/output bits. A seed, unconditional
upload or complete attachment clear must initialize the image before kernel
use. An arbitrary kernel store does not establish full-image initialization.
This uses existing GFX ordering, barriers and owner retirement, without a warm
host readback or a new submission boundary. Storage format support is checked
against the selected Vulkan device before allocation.

Windows qualification covers 2D floating/unsigned/signed images, integer MRT
clear/draw → kernel load/store → readback, public/private owners, non-square
mip updates, replay, close/reset and downstream numeric gradients. Neither
storage-image operations nor rasterization become differentiable. This does
not qualify legacy CGraph texture arguments; these plans use compiled kernel
entries directly. Qualified ordinary graphics export is described above.

Ordinary and selected Vulkan compute Graphs can also export RWTexture kernel
arguments through `module.add_execution_plan`. Supply producer Textures in
`source_bindings` and consumer pixel seeds or public resource names in
`resource_bindings`, as for sampled images below. Frozen Graph requirements
collect storage usage before resource creation, so aliases that sample first
and write later share one image and one initialization. Floating images retain
sampled semantics with storage usage; integer images become `image/storage`.
All allocated mips need their explicit seeds. No producer pixels are implicitly
copied. Wrong dimensions/formats fail before publishing module kernels.

Windows checks cover ordinary/selected export, aliased sampling and writes,
unsigned image updates, nonzero mip shape/load/store, public/private resources,
initialization-only seed retirement, replay and real downstream kernel gradients.
Private/public variants reuse the same compiled kernel entries. CUDA RWTexture
and imported integer Program Textures remain outside this qualification. The
export adds no submission or runtime layer.

CUDA and Vulkan Graph export also accept sampled Texture kernel arguments,
including qualified immutable compute recipes. CUDA preserves its
`cuda_immutable_argument_frames` resource-sharing strategy; Vulkan preserves the
complete ordered compute sequence per preparation:

```python
module.add_execution_plan(
    "sample", compiled_graph,
    source_bindings={"image": texture},
    resource_bindings={texture: {"pixels": "initial_pixels"}},
)
```

The Graph's `image` port is replaced by consumer array `initial_pixels`. Export
preserves format, dimensions and sampler, without copying producer texels. Public
aliases and declared private bindings to one Texture share one image resource.
New exports place seed uploads in the plan's one-time initialization sequence.
Call `prepared.initialize()` before modifying or closing seed arrays; first
execution also initializes automatically. Uploads are recorded together, and
initialization-only inputs are released after completion. An input also read by
replay, including through a view, remains a dependency. CUDA sibling binding
frames share the initialized resource; use independent preparation for a different
image. Output buffers can change across
CUDA siblings; separate Vulkan preparations own independent images and commands.
No extra pixel copy or Python metadata lookup is added to replay. CUDA still
rejects nonmatching min/mag filters, extended sampler settings and mip chains.
Vulkan mip chains use the complete initialization contract above.

Existing artifacts and explicit `image/sampled` resources with a `pixels` binding
keep their construction-time copy and retained-input contract. To express staged
initialization explicitly, omit that resource binding and put an `image/upload`
operation in `ExecutionPlan(initialization=[...])`.

For these qualified sampled Textures, export can instead name a public resource:

```python
module.add_execution_plan(
    "sample", compiled_graph,
    source_bindings={"image": texture},
    resource_bindings={texture: "shared_image"},
)
# Consumer: image is a same-runtime NativeResource, created or explicitly imported.
prepared = loaded_module.prepare_execution(
    "sample", {"shared_image": image, "output": output},
)
```

This explicit alternative emits no private image or seed upload. Public/private
Graph aliases of that producer Texture refer to the same public resource name;
compiled kernel entries and selected CUDA binding-family strategy are reused.
Contents and sampler come from the consumer object. Its kernel dimension/format
and operation contracts are checked at preparation; export does not copy producer
resource state. Names cannot collide with array/scalar/gradient arguments, private
Graph storage or construction inputs. Distinct owners require distinct names.
The public-owner form opts in sampled Textures and the qualified OptiX resources
described below; other families
continue to require their documented reconstruction inputs. Windows ordinary and
selected CUDA/Vulkan checks cover standalone creation, borrowed JIT Texture
imports, aliases, downstream gradients, CUDA siblings and resource retirement.

Matching ABI83 components also expose runtime-owned native resources, shared
across modules and plans through the existing factory catalog:

```python
image = runtime.native_resource(
    "image", "sampled",
    parameters={"shape": [3, 5], "format": "r32f", "filter": "nearest"},
    bindings={"pixels": initial_pixels},
)
initial_pixels.close()  # This public constructor completes the seed upload.
call = module.prepare_execution("sample", {"image": image, "output": output})
call()
```

Here `sample` is a plan whose public `image` argument feeds a sampled Texture
kernel or a native operation; it does not declare a private resource of that name.
`bindings` maps factory input ports to same-runtime ndarrays or `NativeResource`
objects. The same handle can feed multiple plans, image upload operations and
ordinary sampled kernel `bind()`/`prepare()` calls. CUDA binding siblings retain
the same resources and may omit them when rebinding outputs; changing a resource
requires independent preparation. Graph exports reconstruct private resources from
seeds by default; use the explicit public-resource mapping above to supply owners.

Creation is a cold allocation/preparation boundary. Factories conservatively
retain their inputs unless they explicitly confirm that construction has completed
all input use; public sampled images release their seed ownership immediately.
Other retained inputs must remain open. Closing one retires dependent resources
and prepared calls transitively after in-flight work completes. Closing a module
leaves shared public resources and other modules usable. Closing a resource retires
its users; Python variable collection alone cannot retire a resource still retained
by a prepared call. Initialization-only resource dependencies are released after
successful `initialize()`, including across CUDA binding siblings.

Resources use owner-local opaque identities, checked at cold binding boundaries;
foreign, closed and retired handles fail before recording. Replay adds no owner
scan or provider lookup. Seeded images may be updated and sampled across plans
on the same runtime queue. Unseeded images still require an unconditional upload
within each sampling plan: recording or executing a separate upload does not
publish a global initialization fact. Explicit Vulkan RWTexture plans follow
the storage-image contract above; legacy CGraph texture arguments remain outside
this interface. Existing Program Textures require the explicit borrowed
runtime import described below.

Windows qualification covers CUDA/Vulkan images in standalone and borrowed
runtimes, cross-module sampling/upload, CUDA binding families, initialization-only
retirement, and real downstream numeric gradients. CUDA also covers public OptiX
context → GAS → instance-scene resources and transitive memory/parent retirement.
CPU rejects image factories and retains its numeric forward/backward path. This
does not qualify every provider's resource factory or differentiate image sampling
or ray intersection. External owners are not included in prepared-call private
resource/workspace byte counters; those counters are not GPU residency measures.

Windows CUDA/Vulkan export checks cover ordinary Graph public/private aliases with real
downstream numeric gradients, selected binding frames with public texture aliases
and an explicit adjoint, independent seeds and parent retirement in standalone
and borrowed runtimes. CUDA additionally covers shared-family output rebinding.
This does not expand the JIT selected
recipe's existing qualification for private fixed textures or implicit gradients.

`refit` also accepts an updateable GAS. `instance_refit` requires an updateable
instance scene: supplying `transforms` updates its matrices from device memory;
omitting them refreshes IAS bounds while preserving current matrices, as required
after a referenced GAS changes bounds. Arrange GAS refits and IAS refreshes before
the next query. Preparation rejects non-finite or singular initial matrices;
subsequent device updates must obey the same input contract. Nonuniform scaling
and negative determinants are supported; face filtering uses winding after the
object-to-world transform. Topology, instance mapping and metadata stay frozen
until another preparation. Prepared calls retain GAS, IAS, context and provider
owners through in-flight completion and retirement.

Windows CUDA qualification covers standalone/borrowed consumption, two bindings,
surrounding kernels, shared GAS with different instance rules, mirrored and
nonuniform device transforms, live device-generated/shared face tables,
bounds-only updates, refit hit/miss transitions and
retirement. Real gradients of
following kernels work; intersection differentiation with respect to vertices or
rays is not provided. CPU/Vulkan do not register these factories. Operations use
`cuda_ordered`; CUDA capture, OMM and programmable pipelines remain pending.
Explicit v3 alpha qualification additionally covers non-square r32f/RGBA images,
GPU texel/UV/face updates, shared-GAS instances, first-hit and closest queries,
compact occlusion, hit/miss writeback, device refit and post-query kernel gradients.

Compiled Graphs containing triangle or instance scene typed queries, compact
occlusion, GAS refits and IAS transform/bounds refits can export automatically.
Supply explicit resource reconstruction inputs:

```python
module.add_execution_plan(
    "trace", compiled_graph,
    resource_bindings={scene: {"vertices": "initial_vertices", "indices": "initial_indices"}},
)
```

`resource_bindings` maps producer resource identities to factory input ports and
public consumer array names. It is a common export contract; each resource family
must explicitly support its ports. Missing/unused owners, wrong ports and private
Graph storage names are rejected before publishing kernels. Input names may reuse
public Graph bindings; additional names become preparation inputs. The consumer
must initialize these arrays before preparing the plan: resource creation precedes
both plan initialization and replay. Source layouts/counts stay fixed, but the
consumer can supply new vertices/topology with those counts for each preparation.

This policy rebuilds a fresh AS. It does not snapshot the producer AS, its original
arrays or the last refit inputs, which may no longer match its contents.
`snapshot_state=True` does not replace these inputs. Preparation builds the scene
without executing queries; subsequent refits and live face tables keep their
ordered semantics. Recordings sharing a scene/provider reuse their reconstruction
descriptions. No provider path, device pointer or producer callback is archived.
Consumer configuration still selects the external adapter. For an instance scene,
name each distinct GAS's seed arrays and the IAS's initial transform array:

```python
resource_bindings = {
    gas_a: {"vertices": "vertices_a", "indices": "indices_a"},
    gas_b: {"vertices": "vertices_b", "indices": "indices_b"},
    instance_scene: {"transforms": "initial_transforms"},
}
module.add_execution_plan("step", compiled_graph, resource_bindings=resource_bindings)
```

Instances sharing a GAS reuse one resource description. Instance ordinal,
custom ID, visibility mask and opaque metadata remain distinct; initial matrices
come from the consumer, not potentially stale constructor transforms. A live IAS
can reconstruct geometry whose original GAS handle has closed, using its retained
metadata and explicit seeds. This does not enable direct refits through a closed
GAS handle. Windows CUDA checks cover shared geometry, device transforms,
bounds-only refits, independent bindings and downstream gradients. Legacy untyped
hits, OMM/program recordings and nonzero SBT offsets remain unsupported by
this automatic route and are rejected during export.

These qualified OptiX contexts, triangle scenes, GAS and instance scenes may
instead be public resource inputs. Map a source object to a string name:

```python
module.add_execution_plan(
    "step", compiled_graph,
    resource_bindings={instance_scene: "shared_scene", gas_a: "shared_gas"},
)
# The consumer creates these with runtime.native_resource("optix", ...).
call = module.prepare_execution("step", {**arrays, "shared_scene": scene, "shared_gas": gas})
```

Include the GAS mapping when a recording refits that GAS separately. A query-only
graph only needs the scene mapping. No private AS or context is rebuilt for those
public owners; multiple modules may bind the same runtime-local owners. To rebuild
only an IAS, provide public context/GAS mappings and retain the IAS transform-seed
mapping. Preparation still validates operation layouts and factory contracts.

Export preserves source sharing with v17 `resource_alias` constraints: a public
IAS must reference the separately bound GAS at the corresponding `gas_N` ports,
repeated source GAS instances must share one registered owner, and shared source
contexts must resolve to one owner. These checks run at prepare/rebind, not replay.
A source dependency cannot both remain inside a public owner and be rebuilt
privately elsewhere in the same plan; make the shared dependency public too.
This prevents updating unrelated geometry with identical counts. It does not
serialize AS data or borrow JIT OptiX handles. Consumer scene contents are explicit
inputs, and public resource storage is not included in private workspace counters.

For alpha recordings, supply the live Texture associated with each public Graph
texture name in `source_bindings`, and an explicit initial pixel input through
the same `resource_bindings` contract:

```python
module.add_execution_plan(
    "alpha_trace", compiled_graph,
    source_bindings={"alpha_texture": texture},
    resource_bindings={
        scene: {"vertices": "initial_vertices", "indices": "initial_indices"},
        texture: {"pixels": "initial_alpha"},
    },
)
```

The texture name must match `OptixAlphaMask.texture`. Export freezes the actual
creation format, shape and sampler, without reading producer texels. The consumer
binds `initial_alpha` instead of a texture handle. New exports copy it in the
one-time initialization sequence described above; call `prepared.initialize()`
before modifying or closing the seed. Later writes do not upload the image. Native
`image/upload` operations can explicitly update it in an authored execution plan.
Public names and fixed private Graph bindings referring to the same Texture share
one reconstructed image per preparation. UV and face-table bindings remain live.
Missing/unused sources, retired textures and unsupported texture uses fail before
publishing kernels. Ordinary sampled Texture kernel arguments use the same
reconstruction contract described above.

Alternatively use `resource_bindings={scene: "shared_scene", texture: "shared_image"}`
with the same `source_bindings`. The consumer binds a public OptiX scene and an
image created by `Runtime.native_resource()` or imported with the borrowed
runtime's `import_texture()`. No private AS, image or initialization upload is
generated. Qualified CUDA checks cover both r32f and rgba32f, JIT uploads visible
to AOT alpha queries, AOT image uploads, face-and-alpha filtering, miss writeback,
downstream gradients and Program-reset retirement. This does not add derivatives
of OptiX intersections or texture sampling.

Windows CUDA automatic-export checks cover non-square r32f/RGBA images, shared
public/private image bindings, producer/consumer contents that differ, separate
prepared instances, live UV/face updates, refit hit-to-miss transitions, real
post-query kernel gradients, and independent/borrowed consumer retirement.

When a JIT Program already exists, `ti.aot.borrow_runtime()` explicitly shares
its CPU/CUDA executor or Vulkan GFX runtime, device and allocator:

```python
import taichi_forge as ti

ti.init(arch=ti.cuda)
jit_output = ti.ndarray(ti.i32, 17)
with ti.aot.borrow_runtime() as runtime:
    module = runtime.load_module("simulation.tcm")
    output = runtime.import_ndarray(jit_output)
    call = module.kernel("advance").prepare(output)
    call()
    ti.sync()
```

Use an exported entry whose signature matches the supplied arguments. The
borrower owns its modules, arrays and recordings. Closing it retires those
resources and leaves the JIT Program usable. `ti.reset()` or native Program
finalization retires borrowers before destroying the backend; retained wrappers
then reject execution and can still be closed. Each borrower has a separate
resource namespace. JIT ndarrays and arrays from another consumer are not
automatically accepted as bindings. The ordinary `aot.Runtime(arch)` constructor remains
independent. Borrowing requires a matching Python shim and split native library
built with the C API; it does not add a frontend dependency to standalone AOT
consumption. Windows CPU/CUDA/Vulkan checks cover direct and fixed execution,
explicit gradients, module state isolation, GC and concurrent finalization.

Borrowers can explicitly import compact scalar/vector/matrix ndarrays from the
same Program with `runtime.import_ndarray(jit_array)`. The returned view works
with ordinary and prepared AOT calls and shares the original device storage.
JIT and AOT observe the same data and owner submission order without a device
copy. Import `jit_array.grad` separately for explicit `gradients` bindings;
importing does not clear, seed or derive gradients. Shape and dtype come from
native storage metadata.

The view retains a generation-checked native storage lease and remains valid
after the source Python ndarray is collected. Closing it completes in-flight
work, retires associated AOT recordings and drops the lease; storage retained by
other owners stays live. `ti.reset()` invalidates all old views. StructNdarray,
SOA tensors, another Program and arbitrary external pointers are not accepted.
Standalone Runtime instances cannot import JIT storage. Windows CPU/CUDA/Vulkan
checks cover JIT→AOT→JIT ordering, explicit gradients, tensors, two borrowers,
source-wrapper GC and in-flight reset. This adds shared storage, not provider
operation coverage.

With paired ABI83 components, CUDA/Vulkan borrowers can also call
`image = runtime.import_texture(jit_texture)`. It returns the same public
`NativeResource` type used by execution plans and sampled kernel bindings,
retaining the Program's generation-qualified Texture lease. No image storage,
pixel copy or replacement sampler is created. JIT uploads, AOT `image/upload`
and sampling share the owner's queue and existing layout tracking. The caller
must initialize valid texels before sampling, following the JIT Texture contract;
import does not clear or initialize contents.

Vulkan imports preserve sampled/storage/attachment usage from the actual native
allocation. A qualified color Texture can therefore be the storage-image target
of an AOT ray-program launch and then be sampled by JIT without another image or
pixel copy. An import does not grant usage absent from the original allocation;
the existing format and initialization checks still apply.

The source Python Texture may be collected after import. Closing an imported
handle completes in-flight use and retires its AOT users, while independent
imports and the source Texture remain usable. Closing a borrower leaves the
Program alive; `ti.reset()` retires all borrowed calls and leases before backend
destruction. Imports accept only live managed Textures from that Program, on
CUDA or Vulkan. Current qualification follows the native sampled-image boundary:
floating/normalized sampled formats. Vulkan preserves qualified 2D mip chains,
extended LOD, anisotropy and comparison state; CUDA retains one-mip basic sampling.
Unsupported settings fail at import.
Standalone runtimes cannot import Program resources.

Windows checks cover non-square r32f images, nearest filtering and repeat
addressing, JIT→AOT and AOT→JIT updates, two borrowers and multiple imports,
source-wrapper GC, stale-view rejection, in-flight close/reset, and downstream
numeric gradients. Vulkan additionally requests validation. These checks do not
qualify RWTexture/legacy CGraph bindings or differentiation through sampling.

The runtime wheel includes `taichi_forge_runtime.aot`, a Python consumer using
the standard library and this shared C API. Ordinary kernel/CGraph execution
does not need NumPy, the DSL frontend or the CPython shim. Import alone does not
load native libraries or initialize a device. After exporting the example
module below, run `python examples/aot_sdk/consume.py cuda MODULE` with a matching
runtime wheel; the script initializes arrays and verifies forward and reverse
results without importing the producer.

Create an explicit `aot.Runtime("cpu" | "cuda" | "vulkan")`, then use
`runtime.load_module(path)` and `runtime.ndarray("f32", shape)`.
`element_shape=(2,)` makes a vector ndarray; matrices use two element dimensions.
`write(buffer)` and `read_into(writable_buffer)` transfer contiguous host bytes
synchronously; `read()` returns bytes. `Scalar("f64", value)` selects a scalar
width, while plain Python integers/floats mean i32/f32.

`module.kernel(name)(*args, gradients={index: ndarray})` and
`module.graph(name)(args_by_name, gradients={name: ndarray})` execute exported
entries. Omit `gradients` for entries that do not need them. Gradients are neither
cleared nor seeded automatically. `entry.bind(...)` freezes Python argument
descriptors for repeated `call()` submissions; this is ordinary execution,
not a captured GPU Graph.

`entry.prepare(...)` (or `bound_call.prepare()`) prepares immutable native
execution without running the kernels. CPU uses resolved function entries and
one ordered batch, CUDA uses a CUDA Graph, and Vulkan reuses fixed commands.
Call the resulting object repeatedly, inspect its `info`, and `close()` it when
finished. Array contents may change; changing scalar values, shapes or storage
identities requires a new prepared object. Preparation is explicit because its
cold cost may not amortize for a one-shot call. No silent backend fallback occurs.
Explicit ndarray gradients use the same bindings as ordinary execution.
Add `--prepared` to the Python consumer example to exercise this route.
Closing a dependency completes submitted work and invalidates its prepared
objects; unrelated executions remain usable. Warm replay checks a native handle
without re-marshalling arguments or scanning every array owner.

The corresponding C API is `ti_prepare_kernel_ext` /
`ti_prepare_compute_graph_ext` with `TiNdArrayBinding` /
`TiNamedNdArrayBinding`, followed by `ti_launch_prepared_ext` and
`ti_destroy_prepared_ext`. This slice covers ordinary no-return kernel batches;
textures, ArgPacks and GPU sparse topology tasks require later resource/control
extensions. It does not serialize device Graph handles into the artifact.

Entries retain their module; submissions retain code and arrays until completion.
Closing an array/module waits before retiring it. The runtime context manager
waits and closes its children; later calls on them fail. Prefer explicit lifetime
management to GC timing. Optional `library_path` and `runtime_path` select the
shared binary and LLVM resources. A mixed AOT/JIT process uses one compatible
core/resource set. The Python consumer currently excludes structure arrays,
textures, arbitrary native Graphs, unlisted providers and arbitrary external-memory imports; the C/C++
capabilities below are separate. Windows x64 checks cover all three backends,
forward/reverse work and both AOT/JIT loading orders.

The installed `examples/aot_sdk` directory contains an exporter and standalone
consumer. Use absolute paths for `SDK` and `MODULE`:

```sh
python SDK/examples/aot_sdk/export.py cuda MODULE
cmake -S SDK/examples/aot_sdk -B consumer-build -DCMAKE_PREFIX_PATH=SDK
cmake --build consumer-build --config Release
consumer-build/aot_runner cuda MODULE SDK/runtime
```

Use `cpu` or `vulkan` in both producer and consumer commands for those backends.
The exporter needs the matching Forge Python/native build. The C++ program sets
`TI_LIB_DIR` to the supplied runtime directory before creating its runtime.
Windows multi-configuration generators put the executable under `Release/`;
the example copies the SDK DLL beside it. Other systems need their normal shared
library search path or an appropriate RPATH. Platform C/C++ runtimes and GPU
drivers remain external requirements. A custom build using additional shared
dependencies must distribute those dependencies as well.

The Windows x64 development SDK was tested with CPU/CUDA/Vulkan direct kernels
and ordinary CGraphs, including forward results and explicitly bound ndarray
gradients, in a consumer process with Python paths removed. The tested C API DLL
embeds LLVM and has no Python or external LLVM DLL import. Linux/macOS SDK
deployment has not been validated by this check. Initialization/loading timings
printed by this small example are diagnostics, not application benchmarks.

`ExecutionPlan` provides a version 1 persistent sequence of exported kernels,
device buffer copies, 32-bit pattern fills, stable sort, compaction and the FFT slice below. Export the referenced kernels
before adding the description:

```python
from taichi_forge_runtime.aot import ExecutionPlan

exported.add_execution_plan("step", ExecutionPlan([
    {"kind": "kernel", "entry": "forward", "arguments": ["x", "scratch"]},
    {"kind": "copy", "source": "scratch", "destination": "y"},
    {"kind": "fill32", "destination": "scratch", "value": 0},
]))
exported.save("module")
# Consumer: loaded is the result of runtime.load_module(...).
prepared = loaded.prepare_execution("step", {"x": x, "scratch": scratch, "y": y})
prepared.run()
runtime.wait()
```

Descriptions live under `executions/` in directory and `.tcm` modules, or can be
saved separately with `ExecutionPlan.save/load`. C/C++ passes the same JSON
document to `ti_prepare_execution_plan_ext`, with named arguments and explicit
gradients through `TiNamedNdArrayBinding`. Preparation checks all bindings, ranges
and operations before recording native commands; it executes no mathematical
work or buffer modification. Warm replay submits one prepared handle without
parsing JSON or calling Python for each node.

ExecutionPlan version 10 can explicitly select
`binding_strategy="cuda_immutable_argument_frames"` for a flat, fixed CUDA
replay sequence. Prepare its first binding normally, then prepare siblings:

```python
first = loaded.prepare_execution("step", first_arguments)
second = first.prepare_binding(second_arguments, gradients=second_gradients)
first.run()
second.run()
```

Siblings share one CUDA executable, reconstructed private arrays, native plan
resources and once-only initialization. Each binding retains its own captured
argument image and input/output views. Replay inputs must be supplied again;
initialization-only inputs can be omitted. Shape-only metadata can be inherited
after its original array retires. Private resource layouts and the physical
array bindings used to construct a shared native resource remain fixed; prepare
a new independent family to change them. Array contents may still change in
place. Explicit ndarray gradient bindings work on this path.

Closing one sibling leaves the others usable. Initialization survives closing
the first binding; retiring an input still needed by initialization invalidates
the dependent siblings. After successful initialization, its exclusive inputs
are released. Closing the module/runtime or a persistent shared input retires
the affected family members. `binding_family_info` reports a shared identity,
live binding count, executable count and shared private-array bytes. Per-call
`resource_bytes` includes those shared bytes, so do not sum it across siblings.

This explicit consumer strategy currently accepts flat fixed captures, including
qualified native recording commands. WHILE, parallel regions and bounded or
indirect kernel launches are rejected. It does not change CPU/Vulkan or ordinary
`prepare_execution()` defaults. Exporting a selected JIT CUDA buffer binding-frame
recipe preserves this strategy automatically, including reconstructible private
storage and native recordings with their own AOT export contract. Kernel sequences
with explicit adjoints and selected uniform FFT recordings are verified. Sampled
texture bindings are still rejected before export; the JIT implicit ndarray-gradient
restriction is unchanged. An incompatible CUDA graph update raises an error; it
never silently creates an independent executable. C callers use
`ti_prepare_execution_binding_ext` and `ti_get_prepared_binding_family_info_ext`.

Persistent format version 11 accepts explicit IF/SWITCH descriptions through
one `branch` operation:

```python
branch = dict(
    kind="branch", selector="selected", selection="indexed", has_default=True,
    condition=[dict(kind="kernel", entry="choose", arguments=["input", "selected"])],
    bodies=[case_zero_operations, case_one_operations, default_operations],
    lowering=dict(kind="cuda_conditional", version=1),  # use "cpu" on CPU
)
plan = ExecutionPlan([*prefix_operations, branch, *suffix_operations])
```

The selector is one aligned scalar i32 ndarray. The condition executes once;
`indexed` selects a zero-based case using signed values, and negative/out-of-range
values select the last body only when `has_default=True`. Otherwise no body runs.
`nonzero` expresses IF with one body, or IF/ELSE with two bodies and a default.
Empty conditions and bodies are allowed. There is no software branch-count cap;
CUDA's node ABI requires a uint32 body count.

CPU uses prepared host steps. `cuda_conditional` requires CUDA conditional graph
support and keeps condition, selected body and surrounding operations in one root
launch. It adds 24 control bytes per branch, sharing the control allocation with
nested loops. The declared backend must match; it never silently changes route.
All bodies participate in code, resource, gradient and provider preflight, and
only initialization facts common to every possible path reach the suffix.
Ordinary resource and in-flight retirement rules apply.

This explicit-plan slice is verified on CPU/CUDA with branch/loop nesting,
independent bindings, actual forward/reverse kernels and conditional cuBLAS with
an explicit matrix adjoint. It does not differentiate the control structure.
`Module.add_execution_plan(name, graph, gradients=...)` also exports CPU
IF/SWITCH and CUDA branches containing ordinary kernels or qualified provider
commands, explicitly using `native_required`
with the selected conditional route. It preserves physical branch order,
prefix/suffix operations, private condition storage and fixed scalars; shared
kernels are stored once. CPU nesting, eight-way SWITCH, default/no-op behavior,
actual `kernel.grad`, standalone/borrowed consumption and execution after
producer retirement are verified. Gradients require explicit bindings; JIT
Graph restrictions on implicit ndarray gradients remain unchanged.
Complete cuDSS refactor/solve recordings are verified inside IF/SWITCH with shared
plans, independent bindings, live values, actual `kernel.grad` and explicit SPD
adjoints. JIT and export use the same conditional-body qualification; ordinary
capture capability alone does not qualify arbitrary providers.
Version 12 adds explicit `cuda_host` / `vulkan_host` branch lowerings. Existing
Vulkan IF/SWITCH and CUDA Graphs using `lowering_mode="portable"` can be exported,
including nested host branches. Execution information reports `host_control`:
continuous native blocks are recorded during preparation; after the condition
completes, the host reads one i32 and submits the selected body and suffix.
Vulkan reuses one four-byte staging allocation per prepared instance and
prerecorded copy commands. This route includes synchronized readback and multiple
submission boundaries; it does not provide CUDA conditional's single root launch.
Actual forward kernels, explicit `kernel.grad`, standalone/borrowed consumption,
independent bindings and execution after producer retirement are verified across
the three backends. CUDA host branches also cover cuBLAS with an explicit matrix
adjoint. Each provider still needs its existing recording and reconstruction
contracts. Host branches may contain native blocks; native GPU control containing
a host branch is rejected during preparation.
Defaults are unchanged. CUDA auto export, automatic export of nested native
branches and branch binding families remain unavailable. Unqualified routes fail
explicitly during export. Explicit host lowerings cannot be labeled version 11.

Version 16 adds explicit `cuda_masked` branch lowering (recipe 1), sharing the
gated kernel payload and root graph executor used by v15 loops. The condition
uses ordinary entries; every body uses `cuda_graph_gate_v1` entries. Regions
must contain flat kernels without launch overrides; conditions and bodies may
be empty. Each nonempty branch occurrence owns a four-byte gate and latches its
selection before issuing gated body tasks. Writing the original selector inside
a body does not change that selection. All-empty bodies need no gate. Unselected
tasks may still issue, so their cost differs from conditional-node skipping;
defaults are unchanged. CPU/Vulkan reject this lowering; provider bodies and
nested control are outside this slice. IF/SWITCH, empty cases, default/no-op,
signed selectors, actual `kernel.grad`, standalone/borrowed consumers and in-flight
retirement are verified. Branches share code entries and replay does not parse
descriptions. A v16 consumer and matching ABI83 components are required.
`add_execution_plan(name, graph, gradients=...)` also exports IF/SWITCH selected
for `cuda_masked_bounded_graph` under `native_required`, preserving physical
boundaries, conditions and prefix/suffix operations. It deduplicates code by
source specialization and launch ABI, without unused ordinary body variants.
Actual in-body `kernel.grad`, six entries shared across four branch kinds and
repeated plans, consumption after producer teardown and a changed consumer
masked-control environment are verified. Ordinary `auto` remains outside this
export route; provider bodies are rejected before publishing code.

Persistent format version 3 also accepts explicit `while` operations:

```python
loop = dict(
    kind="while", predicate="active", max_iterations=1000,
    condition=[dict(kind="kernel", entry="check", arguments=["state", "active"])],
    body=[dict(kind="kernel", entry="advance", arguments=["state"])],
    continue_while_nonzero=True,
)
plan = ExecutionPlan([loop])
```

Bind `active` to one aligned scalar i32 ndarray with every extent equal to one.
The initial condition always executes, including with a zero budget; each body
execution is followed by the condition. Budgets are nonnegative int64 values.
The body must be nonempty and is stored once; nesting is allowed. An empty
`condition` uses a precomputed predicate: initialize it in the prefix and update
it in the body. This preserves an already composed iteration segment. Ordinary
prefix and suffix operations retain their order. CPU runs prepared
native steps. With CUDA conditional graph support, the entire region uses one
root graph and one `cuGraphLaunch`, with 32 bytes of control storage per logical
loop and no per-iteration host readback. Prepare again to change frozen arguments;
device predicate and array contents can change between replays.

`add_execution_plan(name, graph)` automatically exports exact CPU loops and
selected CUDA conditional loops, including the nested conditional recipe. It
preserves the compiled body-plus-condition
segment, initial condition, prefix and suffix without regenerating source
definitions or repeating the trailing condition. Control scalar shapes and
non-aliasing contracts are saved too. Small tests cover parent Sequential
embedding, private controls, status and explicit adjoints. Nested CUDA export
uses the selected flat physical dispatches and their condition/body boundaries.
With matching private ABI 70 components, compact nested CUDA execution uses the
same nonnegative int64 budget contract as AOT and 32 bytes of private device
control per loop. It imposes no 64-iteration, eight-inner-region or 4096-static-
dispatch software cap. `structured_control_capabilities()["device_control"]`
reports this route under `nested_cuda_conditional_limits`; the legacy
`nested_native_*` limit fields describe expanded command routes. Small Windows
checks cover 65-iteration budget exhaustion, nine ordered inner regions with a
`2**40` budget and early exit, independent bindings, actual suffix `kernel.grad`
and owner retirement. Large budgets do not duplicate the compact loop body.
For `native_required` nested CUDA loops, construction can select the qualified
compact conditional route when the expanded representation exceeds its action
budget. The runtime must expose explicit nested-route support. Explicit recipe
and forced-masked choices remain binding; ordinary `auto` selection is unchanged.
Automatic Graph export of selected CUDA device-update routes uses the v13 contract
below. Masked payloads and other specialized Vulkan routes still need their own
export contracts; export rejects implicit replacement. Kernel-backed CG/PCG
SolvePlan export uses these selected control paths as described below. CUDA operations requiring host synchronization are
excluded from loop regions; providers with opaque cold facts still require an
explicit branch/repetition merge contract. Initialization facts established only
in a body cannot escape a possibly zero-trip loop. Closing storage, modules or a
borrowed owner retires dependent executions under the existing rules. Small
CPU/CUDA cases cover nesting, zero trips, early exit, budget exhaustion and an
explicitly exported suffix gradient; this does not promise differentiation of
arbitrary dynamic loops. Matching private ABI 46 components are required.

Version 13 adds explicit `cuda_nested_device_update` lowering (recipe version 1):

```python
loop = dict(
    kind="while", predicate="outer_predicate", max_iterations=outer_budget,
    condition=[outer_initial],
    body=[inner_initial, inner_payload_and_tail, outer_tail],
    lowering=dict(kind="cuda_nested_device_update", version=1, inners=[dict(
        predicate="inner_predicate", condition_begin=1, body_begin=2, end=3,
        max_iterations=inner_budget,
    )]),
)
plan = ExecutionPlan([*prefix_operations, loop, *suffix_operations])
```

All four example operations are ordinary exported kernels. Boundaries index the
static `condition + body` list; `end` is exclusive. Tail conditions are already
in the physical body and are not appended again. The nonempty initial condition
runs exactly once, including with a zero budget; inner regions execute in table
order. Budgets use nonnegative int64 without software caps such as 64 iterations
or eight regions. Preparation still checks actual node representations, driver
support and available resources. Each predicate binds one aligned i32 element,
including resources referenced only by the lowering. Providers, bounded/indirect
payload launches and arbitrary control depth are outside this route.

Preparation rebuilds device-update control/node tables from ordinary loaded
kernels, joining the prefix, loops and suffix in one CUDA root graph submission.
It executes no user computation, persists no producer device addresses, and does
not switch to conditional or host loops. Each binding owns separate controls and
retires them after submitted work completes. Host table copies are freed after
upload. The package stores static code/operations once; consumer command and
control storage grows with encoded budgets. Small Windows CUDA tests cover two
ordered inner loops, zero budgets, early exit/exhaustion, an actual suffix
`kernel.grad`, standalone/borrowed consumption, two bindings and predicate,
module and owner retirement. Gradients require explicit binding; this does not
differentiate control flow. A v13 consumer is required; private ABI71 remains.
`add_execution_plan(name, graph)` also exports a materialized nested Graph whose
submission route is frozen to `device_update`, including a frozen baseline recipe.
It saves the selected physical kernel sequence, initial/body/tail boundaries,
budgets, private resources and scalar/disjoint binding constraints. It neither
expands kernels in the package nor reconstructs the loop from logical children.
Small nine-inner-region tests cover actual forward/suffix gradient execution,
zero trips after active replay, early exit/exhaustion, two bindings and retirement
in standalone and borrowed runtimes. An unfrozen `auto` node that may fall back
to masked execution is rejected before publishing entries, even if device update
is a candidate. Defaults are unchanged.

Version 14 also accepts `cuda_nested_masked` with the same recipe-1 boundaries
and predicate bindings. Initial-condition entries use the ordinary launch ABI;
every body entry, including inner conditions and tails, must already contain
`cuda_graph_gate_v1` code declared in artifact metadata schema 8. Preparation
rejects an ABI mismatch, including for zero budgets. It generates neither kernel
code nor device-update nodes, and does not silently change the selected route.
Each prepared instance owns its gate storage until submitted work retires.
Static code is shared across plans; captured command storage grows with budgets.
This requires a v14 consumer and matching private ABI83 components.
`add_execution_plan(name, graph)` exports a materialized nested Graph frozen to
`masked`, preserving its physical dispatch boundaries and generating only the
required body variants. Code is deduplicated by source specialization and launch
ABI across plans; an ordinary entry cannot overwrite a gated entry. Small
nine-inner-region tests cover actual forward/suffix gradients, two bindings,
active-to-zero replay and standalone/borrowed retirement. No consumer compiler
is needed to create gated variants. Unfrozen auto routes, native provider bodies,
masked branches are outside this export route. Flat masked loops use v15 below. Existing
device-update plans remain version 13 and retain their ordinary-kernel contract.

Version 15 adds explicit flat CUDA masked loops with
`lowering=dict(kind="cuda_masked", version=1)`. Unlike the nested recipes, this
description has no `inners` field. It permits an empty initial `condition` and
either predicate polarity through `continue_while_nonzero`; a nonempty body
contains precompiled `cuda_graph_gate_v1` kernels, including its tail condition.
A nonempty initial condition runs once even for a zero budget and uses ordinary
entries. Static code does not grow with the budget, but recorded commands do.
Preparation retains the same binding, gradient and retirement contracts and
rejects non-CUDA runtimes, native provider bodies and launch overrides. This
format requires a v15 consumer and matching ABI83 components.
`add_execution_plan(name, graph)` also exports materialized flat Graphs selected
for `cuda_masked_bounded_graph`. It preserves the selected physical body and tail
condition, generates only their gated variants, and shares code across plans.
Zero budgets export the initial condition and surrounding work without unused
body variants. Public/private controls, scalar and disjoint constraints, and
explicit gradient bindings remain intact. Consumer environment changes do not
reselect the route; defaults are unchanged.

Vulkan exports selected flat native loops with a version 4 `lowering` recipe:

```python
loop["lowering"] = dict(
    kind="vulkan_structured", version=1, counter="counter", status="status",
    chunks=[dict(iterations=500, strategy="compact"),
            dict(iterations=500, strategy="coarse_conditional")],
)
```

This illustrates the format, not a recommendation for chunk sizes: retain the
recipe generated by Graph export when available. Chunks must cover the loop
budget exactly. Each chunk's native descriptor uses uint32 iterations; there is
no format-level eight-chunk or 512-iteration cap. Backend recording eligibility
is checked at prepare. Strategies are explicit `compact`, `chained`,
`conditional` or `coarse_conditional`; environment changes do not reselect them
in the consumer. Conditional strategies require device support. Counter and
optional status have the same scalar i32 layout as predicate and must be disjoint.
This recipe requires nonzero predicate polarity and flat kernel-only regions;
Unqualified Vulkan routes are rejected without host-loop substitution.
Without a selected recipe, a generic while still cannot execute on Vulkan.

Selected depth-2 Vulkan native loops also export through
`add_execution_plan(name, graph)`, using persistent format 5 and a
`vulkan_nested` version 1 lowering. It retains one flat physical dispatch list,
the initial outer condition, ordered inner condition/body/end boundaries and
frozen budgets/chunk sizes. Tail conditions are already in that list and are
not added again. Each inner predicate, counter and optional status belongs to
the resource closure, including controls referenced only by the lowering;
all outer/inner controls must be disjoint scalar i32 arrays. Budgets use uint32
because the native controller encodes them that way, without a fixed 64-round
or eight-inner-loop format limit. Existing JIT route qualification still applies.
Preparation reuses the common nested recorder, and warmed positive-budget
prefix/loop/suffix execution uses one queue submission. Zero outer budgets run
only the initial condition between the prefix and suffix. Small standalone and
borrowed checks cover two ordered inner loops, zero inner budgets, early exits,
budget exhaustion, private and external controls, parent embedding, input updates,
explicit adjoints and owner retirement. Use matching private ABI 53 components.
This covers the selected kernel-only depth-2 route, not arbitrary nesting or
every Vulkan SolvePlan strategy.

Vulkan uses the common JIT recording core and batches prefix, ordered loop
chunks and suffix into one queue submission. Preparation performs no mathematical
work; replay neither rebinds descriptors nor reads predicates back to the host.
The serialized body and shader definitions are shared, while native command and
control storage depend on the encoded chunk budgets. Prepared statistics separate
argument bytes from control workspace. Small standalone and borrowed-runtime
cases cover zero/early/budget exits, first-condition side effects, frozen strategies,
two bindings, private buffers, explicit adjoints and in-flight retirement.

`add_execution_plan(name, graph, gradients=None)` also accepts a compiled Forge
Graph. It exports the referenced kernel specializations and qualified native
recordings into this same execution format, and returns the frozen `ExecutionPlan`:

```python
# Producer: builder contains ordinary dispatches and supported append_native calls.
graph = builder.compile()
plan = exported.add_execution_plan("step", graph)
exported.archive("step.tcm")
# Consumer: use the execution-plan API, with the original public binding names.
prepared = loaded.prepare_execution("step", bindings)
prepared()
```

The current automatic path covers sequential scalar/ndarray kernel dispatches
on CPU/CUDA/Vulkan, CUDA `CublasGemmRecording`, `CutensorCaptureRecording`, and
`CufftRecording` with ordinary `CufftPlan1D`/`CufftPlanND` plans, and refreshing
`CusparseLtCaptureRecording` (including shared-A groups).
Ordinary Sequential prefixes/suffixes preserve order. Provider-owned descriptions
reconstruct the original compute/workspace policy and share one resource for
multiple uses of the same plan; distinct plan objects remain independent.
No vendor handle, function address, DLL path or Python callback is serialized.
Kernel specializations are deduplicated across exports in one module. Backend
compilation still follows the module's AOT target/options; exporting a JIT Graph
does not make those target-dependent costs disappear.

For qualified native operations that retain concrete producer arrays, name those
arrays at export using `source_bindings`. The consumer supplies its own storage:

```python
plan = exported.add_execution_plan(
    "step", graph, source_bindings={"keys": producer_keys, "order": producer_order}
)
```

Use the existing symbolic kernel name when a kernel and the native operation
share an array. Source bindings accept packed scalar, vector and matrix ndarrays.
The mapping freezes the scalar dtype, exact array shape and element shape; it
does not copy contents, serialize addresses or prewarm the sequence. Equal byte
sizes with different element shapes are rejected. Each source
allocation needs one unambiguous name. Unused mappings, incompatible kernel
bindings and private-storage conflicts are rejected before kernel publication.
Explicitly named fixed ndarray bindings can also become public plan inputs.
Supported contiguous subviews reference the named parent array without copying
tensor records; each operation retains its own type and layout qualification.
CPU/CUDA/Vulkan checks cover ordinary kernels with vector/matrix source bindings
and explicit ndarray gradients. CUDA/Vulkan checks also cover offset vector
subviews in prepared compact operations.
For qualified f32 CUDA CSR/BSR recordings, a `SparseMatrix` source names its
compressed numeric values; the topology remains frozen. See the sparse export
contract below.

CPU/CUDA/Vulkan `primitive_sequence()` can export native `sort`, single-key-part
`sort_by_key`, `compact`, `unique` and `unique_by_key`. Sorting accepts compact 1D
scalar keys and optional scalar payload, with stable ascending order and NaNs
last; `sort_by_key` requires lexicographic order. `auto` must resolve to the
existing native backend; explicit `cpu_native`, `cuda_device` or
`vulkan_native_radix_u32` sorting requires its matching backend. The explicit
Vulkan compact/unique method is `vulkan_native`. Other strategies,
descending/bitwise ordering, multiple key parts, strided storage, fused sequences
and unlisted operations still require their own export contracts.

Retained scalar/vector/matrix ndarrays use `source_bindings`. Qualified packed dense Fields
use `add_execution_plan(..., snapshot_state=True)` to export module array views
automatically, sharing the same roots as ordinary Field kernels. Native-only
graphs also snapshot their required Fields. Views are published after structural
preflight and reused across exports in the same module. CUDA/Vulkan recordings
from `prepare_sort`, `prepare_compact`, and `prepare_unique`/`prepare_unique_by_key`
also export; prepared compact checks include offset packed Field vector-record
views. CPU retains the sequence route.

Unique keeps the first item of each consecutive integer-key run without sorting.
Fixed `size` preserves producer semantics; only the count prefix is valid and
output tails remain unchanged. Nonempty input uses one head kernel and one
multi-column compact sharing private flags. The head fully overwrites flags, so
producer scratch is not serialized. A sequence with `size=0` only zeros the
first count element, emitting no helper or keys/output access. Vulkan 64-bit key
helpers require `Module(caps=["spirv_has_int64"])`.

Windows checks cover standalone/borrowed consumption, repeated keys, empty
results, partial ranges, independent bindings, retirement and actual `.grad`
for downstream kernels. The existing CPU sequence and CUDA/Vulkan prepared
routes of GeoPhys's generic candidate pipeline pass source-free consumption for
duplicates, empty results, overflow and stale input. This does not qualify its
dense ordinal or segmented branches, or the complete rigid solver.
Selection/sorting itself gains no automatic derivative.

Selected CUDA `GraphBuilder.segmented_scan()` i32/u32 sum recipes now export automatically:

```python
plan = module.add_execution_plan(
    "scan", selected_graph,
    source_bindings={"input": values, "result": output},
)
```

Export preserves the selected serial, global scan/correction, warp/block chunk,
shuffle/hierarchical or length-bucket hybrid strategy without consumer reselection.
Immutable SegmentedLayout offsets and bucket indices become read-only data; each
preparation reconstructs its own bases and scan scratch without saving producer
scratch contents. The global route uses `cuda_driver/retained_scan` v1 in the common
native registry, with `size`/`dtype` parameters and `values`/`scratch` bindings,
reusing the existing driver kernels. That native operation performs an in-place
inclusive prefix sum; the full segmented recipe retains its surrounding kernels,
inclusive/exclusive semantics and unused tail. Explicit scratch counts toward
`resource_bytes`; there is no additional hidden workspace.

Preparation does not execute user scan. Replay does not query the Program arena,
read offsets back or reallocate scratch. Bindings retain fixed shapes and disjoint
input/output requirements. Topology or storage changes require materialization,
export and preparation again. Retained subgraphs have separate symbolic scopes
and share identical read-only payloads. Windows CUDA checks cover seven strategies,
32-bit modular overflow, empty/all-empty segments, partial ranges and tails, two
subgraphs, standalone/borrowed consumption and in-flight retirement. This extends
the existing CUDA integer scan, without adding its automatic derivative or claiming
CPU/Vulkan Graph-native segmented recipes or the complete GeoPhys segmented branch.

Ordinary map compositions export the materialized physical kernel sequence,
including fused IR and its argument remapping, rather than the original source
dispatches. Repeated export shares those kernel definitions. CPU/CUDA/Vulkan
checks cover explicit adjoints, runtime array aliases, and consumption after
closing the producer. Custom storage/executor selections and binding proofs still
need reconstruction descriptions; export rejects them instead of dropping their
constraints. This preserves the selected map transformation, not a serialized
CUDA Graph handle or a guarantee of identical target machine code.

Mixed CUDA Graphs also preserve fused kernels before, between, and after
qualified native recordings. Export follows the final physical sequence while
matching native command barriers to their retained reconstruction contracts;
it does not restore the unfused source maps. The cuFFT regression covers two
library calls, three fused map regions, repeated export, and standalone/borrowed
consumption after producer retirement. Parallel branches containing native
recordings and custom storage routes still
require their own qualified reconstruction descriptions.

Selected fixed-capacity `dispatch_bounded(..., physical_grid="capacity")`
now exports on CPU/CUDA/Vulkan. Vulkan also exports the selected indirect route
and `dispatch_ordered_segments`, preserving device packet preparation, payload
order, and private packet/state storage. ExecutionPlan version 6 represents a
kernel launch as `"launch": {"kind": "vulkan_indirect", "packet": "name"}`;
ordinary kernels keep their existing format. Use matching current runtime/shim components.
The packet is a contiguous scalar `u32` array with at least three values; public
or private AOT storage supports indirect usage without another allocation or copy.
Extent bindings use scalar `i32[2]` (count, overflow); ordered offsets use
`i32[segment_count + 1]`. Capacity remains part of the selected compiled recipe.
Preparation validates these layouts once, and replay does not read counts back.

Tests cover zero/partial/full work, packet reuse between dependent segments,
two bindings, real downstream `.grad`, explicit payload adjoints, and retirement
in standalone/borrowed consumers. CPU exact and CUDA logical-exact launches also
export with paired private ABI 64 components. ExecutionPlan version 7 records
`"launch": {"kind": "cpu_exact", "extent": "name", "capacity": 65}` or the
`cuda_logical_exact` kind. CPU schedules the valid range; CUDA preserves its
saturation-capped grid-stride launch and executes only valid logical iterations.
This does not resize CUDA's physical grid or skip its launch for zero work.
The shared launch binding adds 16 argument bytes per exact dispatch, retained
with the prepared call; CUDA still uploads one combined argument image. The
extent is retained even when used only by the launch description. Direct kernel
invocation without this description is rejected before execution.

CUDA adaptive physical-grid updates also export, including the selected
`per_node` and `grouped_stateful` policies. ExecutionPlan version 8 uses
`"launch": {"kind": "cuda_adaptive", "extent": "name", "capacity": 65,
"block_dim": 32, "update_policy": "grouped_stateful"}`. Cold preparation checks
the compiled block dimension and driver support, resolves the consumer executor's
grid limit, and reconstructs device-updatable nodes through the same core as JIT.
Replay updates grid/enable state on the device without reading counts on the host.
Consumer environment settings do not replace the serialized update policy.
Consecutive grouped consumers must share a logical extent binding; equal addresses
under independently bindable names do not merge groups. A per-node control uses
32 bytes; a group uses 80 bytes plus one 8-byte device node handle per payload,
reported in `prepared.info["workspace_bytes"]`. Handles are rebuilt locally and
retired with the prepared graph. Repeated calls can share one compiled payload.
Windows CUDA tests cover ordinary/selected export, BC/PTX payloads, independent/
borrowed consumers, different consumer grid limits, zero work and overflow,
explicit adjoints with downstream `.grad`, and retirement.

Selected ordinary CUDA parallel regions export as ExecutionPlan version 9
`{"kind": "cuda_parallel", "branches": [[kernel_operation, ...], ...]}`.
Each nonempty branch preserves kernel order; surrounding operations retain the
prefix and join suffix. Export checks the selected groups against the native
physical schedule and preserves private resources, layouts and disjointness
constraints. Each preparation creates its own branch streams and events through
the shared JIT capture core. Warm replay launches one root CUDA graph without
creating streams, parsing the plan or consulting the producer. Use paired private
ABI 65 components. Standalone and borrowed CUDA checks cover 2/8 branches, selected
two-chain recipes, private arrays, alias rejection, real Field gradients,
independent frames and in-flight retirement. The lightweight consumer needs no
producer Python source. CPU/Vulkan reject this CUDA-specific node.

Branches currently require ordinary kernels; nested parallel/control branches,
parallel regions inside control bodies, native callbacks and bounded/indirect
branch launches require separate qualification. Automatic discovery continues to
select the existing two-chain shape; the explicit schedule has no four-branch cap.

Host-count clamping, producer-owned launch state, and exact/adaptive/indirect
dispatch inside structured control still require separate reconstruction support;
export does not replace them with fixed-capacity work.

Graph-instance-owned ndarray declarations (`private_ndarray` / `GraphOwnedNdarray`)
now export as private resources with one-time zero initialization. Symbols sharing
the same declaration retain one allocation, including kernel/provider and explicit
gradient bindings. Every consumer preparation owns fresh state; this reconstructs
the declaration, not the contents of an already executed producer. Live fixed arrays
still require separate reconstruction contracts. Small
CPU/CUDA/Vulkan tests cover accumulation, explicit reverse kernels, aliases and
re-preparation; a CUDA GEMM sequence covers a private intermediate and its explicit
input adjoint after producer closure.

Fixed built-in Python scalars now use the final Graph binding frame and each
physical dispatch's argument dtype. Equal typed bit payloads share one constant
description; different dtypes remain distinct. Three-backend tests cover integer
counts, coefficients and explicit reverse kernels, with exact wide integers and
negative zero. A CPU JIT Graph comparison also checks merged zero and F32 overflow.
Custom numeric objects and native recordings without a typed scalar reconstruction
contract still fail during export preflight. Structured helper recipes continue
to require their own reconstruction contract.

Kernel dispatch labels are preserved by automatic export and by the optional
`"label"` field on a kernel operation. Labels belong to individual operations:
reusing a kernel under different labels does not duplicate its compiled code.
Preparation forwards them to the common launch context. Fixed replay does not
thereby enable individual kernel-profiler or NVTX events. Labels must be nonempty
strings without NUL or line breaks, with no fixed length limit. Small
CPU/CUDA/Vulkan standalone and borrowed consumer tests cover long UTF-8 labels,
explicit reverse kernels and retirement. This requires matching private ABI 46
components. Reduction/helper capacity, layout and disjointness certificates are
preserved through the cold constraints below; unsupported binding recipes still fail preflight.

Handwritten execution plans can persist cold array-binding requirements:

```python
plan = ExecutionPlan(
    [{"kind": "copy", "source": "x", "destination": "y"}],
    constraints=[
        {"kind": "array_layout", "array": "x", "minimum_shape": [17],
         "element_shape": [], "record_bytes": 4, "alignment": 4},
        {"kind": "disjoint", "left": "x", "right": "y"},
    ],
)
```

The optional `dtype` field freezes the scalar type, rejecting equal-width type
substitutions before recording. It requires paired private ABI 55 components.
Without it, kernel/provider contracts continue to determine permitted types.

`array_layout` requires a packed scalar/AoS ndarray, matching rank and element
shape, index extents at least `minimum_shape`, exact bytes per record, and the
specified base alignment. `element_shape` defaults to `[]`; kernel/provider dtype
checks also continue to apply. Empty index extents are allowed. These
requirements apply only to bindings already used by the selected execution or
its retained resources/gradients; they do not retain unrelated arrays.
`disjoint` compares CPU/CUDA address ranges or Vulkan buffer/memory bindings,
including overlapping imports with different allocation handles. Missing Vulkan
memory identity cannot prove disjointness between different buffers. Imported
storage must describe its actual backing allocation; independent aliases created
by external virtual-memory mappings are not tracked.

Use `exact_shape` in place of `minimum_shape` when every index extent must match
exactly; the two fields are mutually exclusive. A disjoint constraint may set
`allow_same_view=True` for an operation that supports in-place execution. This
also accepts the identical physical byte range with matching scalar dtype,
index shape and element shape; partial overlap or merely equal byte counts do
not qualify. The default remains strict disjointness. Native copy/provider
restrictions still apply independently. These are declared operation contracts,
not fixed array-size limits. Small three-backend checks cover oversized/undersized
arrays, in-place replay, distinct imports of the same storage, and metadata or
partial-overlap rejection; Vulkan additionally checks different buffers bound
to the same device memory.

`disjoint_writes` expresses a complete access group without pairwise expansion:

```python
{"kind": "disjoint_writes", "reads": ["b0", "c0", "b1", "c1", "a"],
 "writes": [{"array": "d0", "inplace_read": 1},
            {"array": "d1", "inplace_read": 3}]}
```

Reads may share storage. Writes must be disjoint from other writes and all read
ports, except the optional zero-based `inplace_read` port with identical physical
range, dtype and shape. A second read of that storage still forbids the write.
Omitting `inplace_read` allows no overlap. The description is linear in port
count, and cold validation uses O(N log N) time and O(N) temporary storage on
CPU/CUDA/Vulkan. Imported Vulkan memory must provide enough identity to prove
the relationships; the same-buffer case also works without a memory handle.
Older readers reject this unknown constraint; replay adds no validation or work.

Execution-plan v17 adds a cold `resource_alias` constraint for public native
owners. For example, this requires the scene's retained GAS to be the same
registered resource supplied separately for refitting:

```python
{"kind": "resource_alias", "left": "scene", "left_path": ["gas_0"],
 "right": "gas", "right_path": []}
```

Paths follow retained resource input ports from the common factory catalog;
`[]` denotes the public owner itself, and `['gas_0', 'context']` follows two
levels. Both roots must already be used public NativeResource arguments. Cold
preparation rejects missing ports, retired owners and identity mismatches before
recording. Equality means the same registered resource handle, not equal contents
or a coincident device address; reuse an owner when aliasing is required. Named
links share the existing retirement graph and are stored in compact sorted
vectors. Warm replay performs no path traversal. Use a v17 reader and paired
private ABI83 components; v1–v16 plans keep their original requirements.

Preparation validates these requirements before recording commands or creating
provider state. Failures execute no user kernels. Fixed replay performs no new
constraint checks; changed bindings require another preparation, retaining the
existing owner/retirement contract. Struct and affine views need their own layout
contract. CPU/CUDA/Vulkan tests cover actual reverse kernels, zero-length arrays,
tensor layout, imported aliases and retirement. Graph export translates qualified
scalar/AoS and packed two-dimensional layout requirements, canonicalizing private
aliases and removing duplicate constraints. Custom binding executors, storage
plans and lifetime validators still need their own reconstruction contract.
Use matching private ABI 46 components.

Selected CUDA typed reductions can now export direct and block-partial/finalize
recipes, preserving the chosen kernels, fixed counts, labels, private partials
and binding requirements. Repeated export reuses kernel definitions. Small i32
and f32 tests cover independent preparations and producer/consumer retirement;
the f32 chain uses an explicit sum adjoint followed by a real `.grad` kernel.
This does not synthesize a reduction derivative, qualify every memory recipe,
or provide full LinearOperator/SolvePlan or structured-control export.

For `LinearOperator.from_kernel` and field-free `from_graph` providers and their sum, scale, shift,
composition, adjoint, parameterized-affine and recordable `block_diagonal` combinations, explicitly pin their
generations while keeping symbolic input/output ports:

```python
builder.append_native(operator.graph_action(x, y, update_policy="pinned"))
builder.append_native(operator.graph_action(y, dx, adjoint=True, update_policy="pinned"))
graph = builder.compile()
module.add_execution_plan("apply_and_adjoint", graph)
```

The action selects its numeric/topology and coefficient generations when compiled.
Later publications do not change it; create a new action to select new generations.
Pinning neither copies provider data nor reads it back to the host. Each Graph or
AOT preparation owns its composition scratch; scratch is reconstructed without
exporting its contents. Export synchronizes and copies
the selected immutable arrays into ordinary artifact data, once per allocation
within that export. Shared data entries and kernel definitions are deduplicated;
each preparation owns its reconstructed arrays. Input/output contents remain live,
and Graph-private intermediate ports remain private. The consumer validates exact
scalar f32 vector extents and disjoint input/output storage during preparation.

Windows CPU/CUDA/Vulkan tests cover rectangular forward/transpose pairs, nested
affine/compose/sum/shift operators, numeric and coefficient publications, independent
scratch, two independent consumers, imported aliases, re-preparation and
producer/consumer retirement. This exports an explicitly supplied adjoint, not an
automatically generated provider derivative. The default `update_policy="rebind"`
continues to follow compatible publications and is not silently frozen by export.
`from_graph` preserves its ordered dispatches, fixed i32 parameters and native
topology/numeric/workspace roles. Providers without Fields or mutable workspace export with the default
options. A provider workspace can hold nonzero, mutable state; explicitly choose
its current contents at the export boundary:

```python
module.add_execution_plan("stateful_apply", graph, snapshot_state=True)
```

This waits for preceding work and copies all selected provider arrays under one
native submission lock. Shared allocations are sampled once across forward,
adjoint and composition leaves. Each consumer preparation initializes independent
arrays; aliases within that preparation keep referring to the same resource.
Later producer execution cannot change the artifact, and preparing it again
starts from the exported state. Pinning a JIT action still shares its provider's
live workspace; it does not take this state snapshot. Only declared composition
scratch uses zero initialization.

Ordinary Graphs and pinned `from_graph(state=...)` operators can also export pure
dense Field initial state with `snapshot_state=True`. The actual compiled AOT
entries select the dependency closure; their roots and selected provider arrays
are read in one native snapshot. Fields belong to the **module**: all entries and
prepared calls of one loaded module share and advance the same state. Preparing
again does not reset Fields. Loading another module instance creates independent
initial state. Provider workspace remains private to each prepared call.

Snapshots contain each referenced complete SNodeTree, including colocated fields
and layout padding, rather than splitting storage by names in the `state` mapping.
Bytes use the artifact's backend layout; export does not translate between backend
layouts. Sparse, dynamic, bitpacked and other non-dense storage remain unsupported.
Repeated export into one producer module can reuse identical initial state;
conflicting state requires a new module. Finalization failures after compilation
prevent saving that producer module, so partial artifacts cannot be published.
Artifacts with initial roots require LLVM schema 7 for BC/PTX/cubin or GFX metadata
3 and matching newer runtime components. Loading validates and initializes the data
once; replay adds no state uploads or copy kernels. Windows CPU/CUDA/Vulkan tests
cover scalar/vector Fields, real kernel.grad, shared Fields with private workspace,
independent modules, repeated exports, bad payloads and close/reset. Complete
GeoPhys Field-backed SolvePlans and arbitrary native providers still need their
own qualification.

CPU/CUDA/Vulkan tests cover multi-dispatch square and explicit-adjoint providers,
nonzero state, old/new numeric publications, compositions, interleaved independent
consumers and retirement. Nested blocks reconstruct packed views over public ports,
Graph-private arrays and composition scratch through the common `ndarray_view`
resource. Parent dependencies stay explicit even when only child views reach
kernels; repeated sequential embedding shares descriptions and storage. The views
add no buffer copies or kernel definitions. CPU/CUDA/Vulkan tests cover rectangular
blocks, explicit adjoints, blocks inside compositions, independent preparations,
numeric publications and retirement in standalone and borrowed consumers. Existing
backend/provider recordability and affine-view restrictions still apply; this does
not qualify arbitrary block leaves or add automatic derivatives. Automatic export
in this section requires matching private ABI 50 components; no provider library
is added to the consumer.

Kernel-backed f32 CG/PCG SolvePlan actions can also retain their complete selected
program, including initialization, operator/preconditioner calls, control, private
workspace and terminal writes:

```python
solve = solver.graph_action(rhs_arg, output_arg, initial_guess=initial_arg,
                            update_policy="pinned")
builder = ti.graph.GraphBuilder()
builder.append_native(solve)
graph = builder.compile()
module = ti.aot.Module()
module.add_execution_plan("solve", graph)
module.archive("solve.tcm")
```

The default `update_policy="rebind"` retains live numeric updates and is rejected
by export. Pinning selects the operator and fixed preconditioner generations when
the Graph action is compiled; later numeric publications require a new Graph and
artifact. Existing topology/runtime invalidation checks still apply. The consumer
loads compiled kernels and resources without constructing a Python solver or
selecting another algorithm. The loop body is stored once, independent of its
iteration budget; Vulkan retains its selected command strategy.

The public bindings include RHS, output, optional initial guess, and the names
`solve.terminal.state.name` and `solve.terminal.metrics.name`. Include these names
in the application interface manifest. Allocate terminal arrays of shape `(4,)`,
with `i32` state and `f32` metrics. State contains status (`0` budget exhausted,
`1` breakdown, `2` converged), completed iterations, breakdown flag and completion
marker. Metrics contain squared initial residual, final residual, RHS norm and
effective tolerance. Each invocation overwrites the terminal packet. Vector ports
require packed scalar f32 arrays of the operator's exact size. RHS must be disjoint
from output and initial guess; initial guess and output may be disjoint or the
same physical typed view, never partially overlapping. Checks run at preparation.

Each prepared execution owns its scratch. Tests cover CPU, CUDA conditional and
flat Vulkan control, independent bindings, numeric generation changes, zero budget,
initial convergence, early exit, budget exhaustion, breakdown, aliases and close/reset.
They include an explicit symmetric adjoint solve and real downstream gradient
kernels; this does not differentiate the dynamic solver automatically. Complete
Field-backed solves still need application qualification; unqualified provider
actions, other solver methods and specialized control routes need separate
reconstruction support. Provider workspace and dense Field state exports require
`snapshot_state=True` where applicable.

For Graphs with explicit reverse kernels, `gradients={"x": "dx", "y": "dy"}`
binds their ndarray gradients. Export the gradient clearing/seeding operations
too. This does not synthesize derivatives of native provider calls.
Plans pass each gradient only to kernel arguments compiled with `needs_grad`;
ordinary kernels may also read or write that primal. A gradient used by no
kernel in the plan is rejected during preparation. Standalone kernel calls
retain their strict per-kernel gradient contract. Windows
CPU/CUDA/Vulkan checks cover forward/reverse execution in standalone and borrowed
consumers. cuBLAS/cuTENSOR checks include complete mixed sequences, explicit
coefficient/input adjoints, shared and independent plans, source-owner retirement
before consumption, and in-flight consumer close/reset.

Unsupported recording types/subclasses, control/launch routes, observations,
and selected composition/private-resource recipes
without a reconstruction contract are rejected during preflight, before adding
kernel definitions. They are not silently replaced by a baseline recipe. If a
backend compilation subsequently fails, the module cannot be saved: create a new
module to avoid publishing a partial artifact. `add_graph` still exports the
ordinary CGraph format; this new path is consumed through `prepare_execution`.
The standalone consumer and native executor are shared with handwritten plans.

CPU/CUDA LLVM modules and Vulkan modules can load directly from `.tcm` files.
The loader retains compressed archive bytes and reads kernel payloads on demand;
it does not extract the whole archive to a temporary directory.

`copy` defaults to the source's remaining logical bytes; optional
`source_offset`, `destination_offset` and `size_bytes` select nonoverlapping
ranges. `fill32.value` is a uint32 bit pattern; optional `offset/size_bytes`
select a region, defaulting to the remaining logical destination. Fill offsets
and sizes require 4-byte alignment; Vulkan copies also require that alignment.
Memory operations accept primitive/tensor ndarrays, including ownership checks
for arrays referenced only by native operations. Contents can change between replays;
new storage identities or shapes require preparation again. Separate prepared
bindings are independent, but application-owned arrays may still alias.

The `compact` operation has `values`, `flags`, `output` and `count` argument
names. It stably copies records whose i32 flag is nonzero, and writes the i32
count on every replay, including empty input. Values/output are matching 1D
primitive or tensor ndarrays with a record width divisible by four bytes;
output capacity must cover all input records. Flags have one entry per record;
count is a scalar ndarray or a nonempty 1D ndarray (only its first item is
written). Writable ranges must not overlap input or each other. Only the
selected prefix is written; the remaining output is unchanged. CPU/CUDA/Vulkan
use independent owned scratch per prepared instance; `prepared.info` reports
`workspace_bytes`, excluding borrowed arrays, code and driver metadata. C/C++
can query the same count with `ti_get_prepared_execution_workspace_bytes_ext`.
Selection has no automatic derivative. A tested composition compacts integer
indices, then runs an exported gather kernel and its explicit backward kernel.

Several columns can share the same selection with
`{"kind": "compact", "flags": "active", "count": "n", "columns":
[{"values": "keys", "output": "selected_keys"},
 {"values": "payload", "output": "selected_payload"}]}`.
Use either `columns` or the single-column `values/output` shorthand. GPU columns
share one predicate prefix and scan workspace; CPU prepares one selection index
array and copies columns in separate passes. Input columns may share read-only
ranges; writable ranges must remain disjoint across all columns. The known
scratch report includes the CPU selection array.

`{"kind": "sort", "keys": "keys", "values": "indices"}` sorts keys and optional
payloads in place, in stable ascending order. Keys are 1D scalar
i32/u32/f32/i64/u64/f64; payloads have the same length and a record width
divisible by four bytes. Key and payload ranges cannot overlap. NaNs sort last;
equal keys, including signed zeros, retain their input order and original bits.
Empty and single-item inputs require no scratch. Native CPU/CUDA/Vulkan execution
owns the scratch for larger inputs; replay reuses it. There is no automatic sort
derivative: an explicit integer permutation can feed an exported gather/backward
pair. `prepared.info["dispatch_count"]` counts exported kernel entries, not the
internal shader dispatches of native algorithms.

Execution plans may declare private `resources`. For example, a temporary with
the first dimension of an external `output` array can be reconstructed without
exposing its allocation to the SDK caller:

```python
plan = aot.ExecutionPlan(
    [{"kind": "fill32", "destination": "scratch", "value": 0},
     {"kind": "copy", "source": "scratch", "destination": "output"}],
    resources=[{"kind": "ndarray", "name": "scratch", "dtype": "f32",
                "shape": [{"array": "output", "axis": 0}]}],
)
```

Dimensions may be literal uint32 values or `{array, axis}` references; tensor
records use a literal positive `element_shape`. A resource of kind `"extent"`
with `name`, `dtype`, `array` and `axis` binds a derived scalar (i32/u32/i64/u64).
The existing C API rank and native indexing constraints still apply. Preparation
resolves the required dependency closure before allocating arrays. Unused
declarations and arrays used only for shape inference allocate no device storage;
shape-only external arrays are not retained after preparation. Invalid cycles,
duplicate names and conflicting external bindings are errors. `plan.argument_names`
reports the external slots, including shape sources and declared gradients.

Immutable kernel parameters can use `aot.Scalar("i32", 17).resource("n")` or
`aot.Scalar("f32", -1.75).resource("gain")` in the same `resources` list. These
parameters are absent from `plan.argument_names` and require no private device
allocation or initialization command. They use the existing typed kernel binding
path; preparing a different value requires a different description/preparation.
The serialized `"scalar"` resource has `name`, `dtype` and hexadecimal `bits`,
preserving 64-bit integers and negative zero without JSON numeric rounding.
Bit widths follow the C API: F16 scalar payloads are float32, unlike F16 array
elements. Use matching Python/native components (current private ABI 46). Small
CPU/CUDA/Vulkan checks cover exact wide integers, fixed-parameter forward/reverse
kernels and retirement; CPU additionally covers narrow integers and F16/F64.
Ordinary Graph fixed scalars use this same resource format during automatic export.

Declare `gradients={"primal_name": "gradient_name"}` on the plan to bind private
or external arrays to exported explicit gradients. This reuses ordinary kernel
binding checks and supplies no automatic derivative for native operations. A
caller-supplied gradient for the same primal is rejected. Arrays default to
uninitialized storage: express required per-replay clears or copies as operations.
An ndarray resource may set `"zero_initialize": true` to start at zero once per
prepared instance, before explicit `initialization` and the first replay. Preparing
records that work without executing it; `prepared.initialize()` or the first launch
performs it. Later launches preserve the array's contents. Native resource factories
must defer data-dependent work until an initialization or replay operation.
Each preparation owns independent arrays; rebinding or resizing prepares another
instance. Private arrays retire after their commands,
including failed preparation, module closure and in-flight buffer retirement.
`prepared.info["resource_bytes"]` reports their requested allocation bytes
(one byte per empty array by default). Zero-initialized private allocations round
that physical capacity up to four bytes, without changing logical shapes; this
supports odd byte counts using existing native fills and adds no user shader.
Shape-only and unused arrays still allocate nothing. These bytes are separate from
algorithm `workspace_bytes` and exclude allocator/driver overhead. Small Windows CPU/CUDA/Vulkan checks cover
primal computation, explicit reverse gradients and these ownership transitions.

`prepared.info["native_resource_bytes"]` separately reports known device
payload owned by private native resources (paired ABI89 components). Managed
CUDA/Vulkan images report packed texel bytes, including all mip levels. This
does not change the array-only `resource_bytes` or provider `workspace_bytes`
counters. Public/borrowed resources, host shader code/descriptors, allocation
padding and driver-private residency are excluded. An opaque provider reporting
zero does not promise zero GPU memory. The common resource interface supplies
this number; the executor does not special-case image types or query a driver.
Totals are cached during preparation and refreshed when initialization-only
resources retire. Shared family payload appears in each sibling's report, so
do not sum those values across siblings. Python reads the three memory counters
through one `ti_get_prepared_execution_memory_info_ext` call under one owner
lock; the legacy array/workspace C getters remain available.

Artifact data can supply the initial contents of a private ndarray:

```python
key = module.add_data(packed_weights)  # producer: any buffer-protocol object
weights = dict(kind="ndarray", name="weights", dtype="f32", shape=[17],
               initial_data=key)
module.add_execution_plan("step", aot.ExecutionPlan(operations, resources=[weights]))
```

`add_data()` snapshots the bytes and returns their lowercase SHA-256 key. The
caller supplies packed data matching the declared dtype and shape; no dtype or
byte-order conversion is performed. Export requires every selected payload to be
registered. Directory and `.tcm` artifacts store identical bytes once under
`data/<key>.bin`; unused registrations and shape-only declarations add no payload.
Use a fresh directory for `save()` when producing a distribution; it does not
clean files left by previous exports. `archive()` uses a fresh staging directory.

Preparation reads and verifies each selected payload once, checks its exact byte
count, and uploads it before recording user operations. It executes no user kernel
or provider computation. Separate preparations own independent **mutable** arrays,
even when they refer to the same data entry. Nonempty private capacities and
uploads round up to four bytes for native transfer alignment, without changing
logical shapes or stored bytes. Empty payloads are valid. Data cannot be combined
with `zero_initialize=True`. Explicit initialization may subsequently overwrite it.
Missing, truncated or corrupted selected data rejects preparation; already prepared
instances and unrelated entries remain usable. There is no archive read, hashing
or upload during replay, and expanded host payloads are not cached by the module.
This is shared resource infrastructure, not automatic serialization of arbitrary
live operator/provider state. Small CPU/CUDA/Vulkan tests cover separate consumers,
explicit gradients, byte tails, empty data and in-flight retirement.

Immutable provider inputs that are consumed on the host can use
`dict(kind="host_blob", name="code", data=key, byte_size=len(code_bytes))`, with
the same `add_data()` key. This resource has no device allocation and cannot be
bound as an ndarray. A qualified native resource factory receives it through a
resource binding. Preparation loads and verifies only reachable blobs, shares
identical payloads within that prepared owner, and retains them through resource
retirement. `plan.host_data_bytes` counts unique declared reachable host bytes;
it is separate from GPU workspace and does not estimate process RSS or copies
made by a provider. Archive payloads also deduplicate across host and ndarray
uses. This establishes the shared loading contract; it does not by itself enable
callback export for a provider that has not implemented reconstruction.

Packed array slices can be reconstructed without allocating or copying device
storage. A view inherits its parent's scalar dtype and element shape:

```python
resources = [
    {"kind": "ndarray_view", "name": "part", "array": "input",
     "offset": 4, "shape": [16]},
]
```

`offset` counts packed elements relative to the parent view, including all
components of a tensor element. `shape` describes the view's logical dimensions
and can use the same extent references as an owned ndarray. Bounds, overflow,
layout and dependency cycles are checked at preparation or earlier. Views cannot
reinterpret dtype, introduce arbitrary strides, or describe Struct storage.
Parents may be external arrays, private arrays with initial data, or other views.
Storage used through a view remains live across initialization and replay;
shape-only references do not allocate or retain device storage for replay.
Closing a public parent invalidates its dependent preparation.

Kernels, copies, fills and compatible native operations use the selected range.
Explicit gradient bindings may name separate views with different parent offsets;
the exported kernel must still declare its gradient ABI. CPU/CUDA/Vulkan tests
cover actual `.grad`, nested views, tensor elements, reshape, empty views, native
sort, alias rejection and retirement. No additional device allocation or replay
copy is introduced by a view. Use matching private ABI 46 components. This common
resource support is separate from automatic export of operator-derived views.

Packed dense Fields can expose the same module-owned storage to kernels and native
execution operations on CPU/CUDA/Vulkan:

```python
module.add_field("keys", keys, as_array=True)
module.add_field("order", order, as_array=True)
module.add_execution_plan("sort_fields", aot.ExecutionPlan(
    [{"kind": "sort", "keys": "keys", "values": "order"}],
    resources=[{"kind": "module_array", "name": name, "view": name}
               for name in ("keys", "order")],
))
```

`as_array` defaults to false. Opting in freezes scalar dtype, shape, tensor element
shape and byte offset in `root_storage.json`; tree IDs are relocated with the
artifact. `module_array` uses those layouts rather than caller-provided pointers
or copies. LLVM requires relocatable plain dense trees. Nonpacked/strided Fields
are rejected, and an exported view name cannot be rebound to another Field.
Ordinary Field export remains unchanged. This does not capture initial Field
values; existing dense-state snapshots or explicit initialization remain necessary.

A module array may parent an `ndarray_view`, supply extent metadata, or participate
in explicit ndarray gradient bindings. Aliases add no device storage to
`resource_bytes`; module roots remain accounted separately. Preparation checks
names, layouts and root bounds. Replay uses the already-bound storage. Closing a
preparation releases its aliases; closing the module retires dependent recordings
before releasing roots. Independent module instances have independent state.
CPU/CUDA/Vulkan tests cover Field→native sort→Field kernels, real `.grad`, tensor
copies, subviews, malformed bounds and standalone/borrowed retirement. This is the
shared storage contract, not automatic export of arbitrary Field-based helpers.
Current components require matching private ABI 64.

Plans can separate one-time initialization from replay:

```python
plan = aot.ExecutionPlan(
    [{"kind": "copy", "source": "snapshot", "destination": "output"}],
    initialization=[{"kind": "copy", "source": "input", "destination": "snapshot"}],
    resources=[{"kind": "ndarray", "name": "snapshot", "dtype": "f32", "shape": [17]}],
)
prepared = plan.prepare(module, {"input": source, "output": output})
prepared.initialize()  # samples source now and waits for completion
source.close()         # no longer a dependency of replay
prepared()             # uses the fixed snapshot
```

`initialization` is a nonempty sequence using the same operation schema and
serializes as execution-plan version 2; plans without it retain version 1.
Preparation executes neither phase. Initialization is idempotent after success;
the first launch also initializes when necessary. It is a cold synchronization
boundary, not a per-replay wait. C/C++ uses `ti_initialize_prepared_ext` or the
automatic first-launch path. Construct a new prepared instance to resample inputs.

After successful initialization, its commands retire before initialization-only
external leases, private arrays and native resources. Replay's transitive resource
dependencies, aliases and explicitly bound gradients remain live. Python releases
the corresponding initialization-only owners too. `resource_bytes`,
`workspace_bytes` and `argument_bytes` report current retained storage; capture
them before and after initialization to compare the two phases. `dispatch_count`
and `launches` describe replay. A partially failed initialization makes the object
unusable; it retains its leases until synchronized retirement and cannot be retried.
Small CPU/CUDA/Vulkan tests cover standalone and borrowed owners, explicit reverse
gradients, independent snapshots, source retirement, aliases and owner reset.

`runtime.device_capabilities()` returns a snapshot of the device's enabled
capability levels, for example `caps.get("spirv_has_ray_query", 0)`. Treat absent
entries as zero. This cold query uses `ti_get_runtime_capabilities` without
allocating device storage or loading providers. Device capabilities, registered
operations and optional dependencies are separate conditions; preparation still
qualifies concrete layouts and hardware limits.

Before allocating operation buffers, query
`runtime.native_operation_capabilities("compact", version=1)` (also `"sort"`).
The returned set contains `"cpu"`, `"cuda_capture"`, `"vulkan_inline"`, or
`"cuda_synchronous"` / `"cuda_ordered"` for the
registered implementation on this runtime's backend; an unknown kind/version
returns an empty set. The query loads no operation code and allocates no scratch.
Concrete dtype, layout and device requirements are still checked at preparation.
The C API equivalent is `ti_get_native_operation_capabilities_ext`. Registration
and selection happen in the shared native preparation layer; replay uses the
already prepared object and performs no registry lookup. These registrations do
not imply that an optional provider is installed.

Resource factories use the same catalog through
`runtime.native_resource_capabilities(kind, version=1)` and
`runtime.probe_native_resource(kind, version=1)`. The first query never loads
dependencies or allocates a resource; the explicit probe may load configured
libraries and returns `available`, `capabilities` and `library_version`, as for
operations. Resource and operation kinds are checked separately: querying a
resource through the operation API returns an empty result. This preserves
existing operation-discovery behavior. The C equivalents are
`ti_get_native_resource_capabilities_ext` and `ti_probe_native_resource_ext`.

| Resource factory | Query kind | Compiled backends |
| --- | --- | --- |
| `image/sampled` v1 | `sampled_image` | CUDA, Vulkan |
| `image/collection` v1 | `sampled_image_collection` | Vulkan |
| `image/storage` v1 | `storage_image` | Vulkan |
| `vulkan_graphics/pipeline` v1 | `vulkan_graphics_pipeline` | Vulkan |
| `vulkan_graphics/pipeline` v2 (mesh/task) | `vulkan_graphics_pipeline` | Vulkan |

These are preparation-time SDK choices, not replay queries. A successful probe
does not admit every format, extent, shader or device feature; creation still
checks the concrete description. The query applies to registered native resource
factories, not built-in ndarray or host-blob allocation. It adds no artifact
fields, allocation or warm execution cost.

`cuda_synchronous` is for a native provider action that cannot be captured or
does not expose its execution stream. The native executor records adjacent
capture-compatible operations as CUDA Graph segments and invokes the action
under the owner submission lock, with context completion on both sides.
Consecutive synchronous actions avoid a duplicate incoming wait. An ordinary
trailing graph remains asynchronous; `runtime.wait()` completes the whole batch.
Such a prepared batch reports `execution_kind="cuda_mixed"`. Pure capture batches
retain the single-graph path. Replay invokes no Python callback or symbolic binder.
If an action fails after submitting work, the executor waits, skips the remaining
sequence and rejects further replay of that instance.

`cuda_ordered` is for native host actions that enqueue onto the supplied runtime
stream, or explicitly join their private streams to it. Adjacent CUDA Graph
segments keep stream order without context fences. The executor records a
completion event for the submitted batch and retires provider transaction tokens
only after completion; shared owners can observe completion across prepared
instances. An action failure skips the suffix, retains partially submitted work
through retirement, and rejects replay of that instance. GPU conditional bodies
cannot contain host-invoked actions. Pure ordered batches report
`execution_kind="cuda_ordered"`; batches also containing synchronous actions
report `cuda_mixed`. This shared contract was introduced in private ABI 61. Ordinary cuDSS
transaction export uses it with the paired ABI 62 components described below.

This composition capability still requires a registered provider-specific schema,
resource reconstruction and numeric qualification. NVIDIA documents the
context-wide wait in
[cuCtxSynchronize](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__CTX.html).

`runtime.probe_native_operation(kind, version=1)` explicitly checks optional
dependencies before allocating operation arrays. It returns `available`,
`capabilities` and a provider-specific `library_version` (`None` for built-ins or
unavailable libraries). This cold call may load a vendor library, but creates no
plan/workspace. Unregistered operations have no capabilities; a registered
provider with a missing library keeps its compiled capabilities and reports
`available=False`. Unrelated operations remain usable. Concrete device/layout
admission still happens during preparation. C/C++ uses
`ti_probe_native_operation_ext` and `TiNativeOperationAvailabilityExt`.

External operations also accept a common versioned envelope. Registered providers
decode their own parameters and binding contracts:

```python
operation = {
    "kind": "native", "provider": "cublas", "operation": "gemm_f32", "version": 1,
    "parameters": {"rows": 17, "columns": 9, "inner": 13, "alpha": 1.25},
    "bindings": {"a": "left", "b": "right", "output": "result"},
}
plan = ti.aot.ExecutionPlan([operation])
```

Binding keys are operation ports; values name execution arguments or private
resources. Parameters are frozen values without process addresses. The common
reader/binder does not interpret vendor mathematics. A native registration owns
parameter/layout/alias validation and uses the existing typed factory; warm replay
retains no JSON interpreter. Examples of registered version 1 schemas include
`cublas/gemm_f32`, `cublas/dot_f32`, the cuBLAS vector operations below, and
`cufft/fft_c2c_f32`. GEMM/FFT parameters
match the legacy forms below after removing `kind` and storage names. Registered
lowering preserves those legacy forms; dot uses the common envelope directly.

Descriptions can be saved/read without a native library. Preparation checks the
concrete schema, backend and dependencies before allocating private buffers. This
does not automatically export arbitrary JIT recordings or invoke arbitrary DLL
symbols; other operations still need registered reconstruction contracts. The
qualified precision, Graph and explicit-adjoint boundaries below remain unchanged.

The original CUDA `fft_c2c_f32` operation uses external cuFFT for compact,
row-major interleaved complex f32 ndarrays (`element_shape=(2,)`):

```python
exported.add_execution_plan("fft", ExecutionPlan([
    {"kind": "fft_c2c_f32", "input": "source", "output": "frequency",
     "dimensions": [16, 32], "batch_count": 2, "direction": "forward"},
]))
```

Bind arrays with exactly `batch_count * product(dimensions)` complex elements.
`batch_count` defaults to 1; `direction` must be `"forward"` or `"inverse"`.
Inverse transforms are unnormalized; scale by the reciprocal transform size
when needed. Exact in-place storage is supported; partial overlap is rejected.
The cuFFT adapter retains its existing PlanMany rank 1–3 and int32 dimension/
batch-distance ABI. This original operation excludes strided layouts and real
transforms; the shared-plan interface below supports them. Double precision and
callbacks are not supported. CPU/Vulkan have no AOT FFT factory in this slice.

Each prepared instance owns an independent vendor plan/workspace. Preparation
does not execute FFTs or change supplied buffers; mixed kernel/FFT sequences
capture once and replay as a CUDA Graph without provider discovery or plan
creation on every call. `workspace_bytes` includes vendor-reported scratch and
may be zero for small transforms; opaque vendor bookkeeping is not included.
Closing a dependency retires its recording and plan after completion.
`info.dispatch_count` counts exported kernel entries, excluding vendor-internal
GPU kernels.

For shared plans, register a native resource `cufft/plan_f32` with `dimensions`,
`transform` (`c2c`, `r2c`, `c2r`), optional `batch_count` (default 1), and optional
`input_layout`/`output_layout`. An explicit layout contains `embed`, `stride` and
`batch_distance`, measured in real or complex elements for that port. Omitted
layouts are compact, including the last-axis half spectrum for real transforms.
Then use `cufft/execute_f32`, binding `plan`, `input`, `output` and declaring
`direction`. R2C requires forward; C2R requires inverse. C2R may overwrite its
half-spectrum input. All inverses remain unnormalized.

This interface accepts f32 scalar storage (interleaved complex components) or
complex `vector(2)` storage with exactly the physical span implied by the plan.
Input/output must be disjoint. Multiple executions of one resource share its
plan/workspace; separate prepared instances remain independent, and scratch is
accounted once. `Module.add_execution_plan(name, graph)` uses this representation
for ordinary cuFFT recordings, preserving frozen creation layouts and plan
identity. Unqualified callback and plan subclasses need their own
export contract and are rejected before kernel export. Tested automatic exports
cover batched 1D C2C, padded/strided 2D C2C, 3D R2C/C2R, explicit spectral-energy
adjoints, nonzero shared workspace and standalone/borrowed owner retirement.

Selected `linalg.record_fft` plans also export through this shared core. They
retain `whole_transform`, `row_batch_column_inplace` or
`row_batch_cross_batch_columns`, direction and any separate output-scale kernel.
The two decompositions require compact 2D C2C storage; real transforms retain the
whole-transform route. Optional `strategy` in `cufft/plan_f32` defaults to
`whole_transform`. Selected exports include `expected_workspace_bytes`, which
must match the reconstructed plan before execution. Consumers do not search for
or substitute another strategy. Existing descriptors remain readable; older
readers reject these additional fields instead of ignoring the selection.
Small Windows CUDA checks cover positive/negative scaling, complex forward and
inverse transforms, real half-spectrum weighting, exact public array shapes,
binding changes, workspace mismatch, and standalone/borrowed retirement. They
combine real downstream `kernel.grad` with explicitly assembled FFT adjoints;
FFT provider differentiation is still explicit.

The selected `whole_transform_store_scale` C2C recipe also exports its compiled
LTO callback. A content-addressed `host_blob` retains the immutable code, shared
by plans with identical payloads. No callback source, producer library path or
Python callable is required by the consumer. Configure the external toolchain
before preparing this selected route:

```python
runtime.configure_library("nvrtc", nvrtc_library_path)
runtime.configure_library("nvjitlink", nvjitlink_library_path)
prepared = module.prepare_execution("step", bindings)
```

The plan has `store_callback={"format": "cufft-store-scale-lto-v1",
"compiler_version": [major, minor]}` and a `store_callback_ir` host-resource
binding. This contract fixes the built-in store-scale symbol and does not enable
arbitrary callback exporters. The loader checks code integrity, compatible NVRTC/
nvJitLink versions and the frozen workspace, then uses the shared cuFFT plan core.
It rejects missing or incompatible dependencies without substituting an unscaled
transform. Plans retain code/library leases through retirement; replay does no
library loading, hashing or compilation. Plain recipes do not load these auxiliary
libraries. Existing descriptors remain valid; older readers reject the new fields.

Precompiled user LTO code still needs NVRTC for cuFFT's internal wrapper and
nvJitLink for linking. NVIDIA requires compatible toolkit families; successful
vendor plan creation remains the final cuFFT compatibility check. See the
[cuFFT LTO dependency contract](https://docs.nvidia.com/cuda/archive/13.2.0/cufft/index.html#lto-load-and-store-callback-routines).
Windows cuFFT `12200`, NVRTC/nvJitLink 13.2 and RTX 5090 checks cover positive and
negative scaling, forward/inverse C2C, downstream `kernel.grad` plus explicit FFT
adjoints, two bindings, damaged/missing code, dependency failure/recovery and
standalone/borrowed retirement. This does not qualify other platforms or change
recipe defaults; preparation still includes vendor compilation/linking costs.

Install cuFFT separately and make it visible to the process loader (for example,
Windows `PATH`, or an explicitly preloaded library). Forge does not bundle the
vendor DLL. Windows cuFFT 12.2 / RTX 5090 checks cover ranks 1–3, batches,
in-place execution, independent workspaces, rollback and borrowed-owner reset.
An exported scale/FFT/energy chain also checks an explicitly assembled adjoint
against an analytic gradient; this does not add automatic FFT differentiation.
See NVIDIA's [cuFFT transform and Graph contracts](https://docs.nvidia.com/cuda/cufft/index.html).

Vulkan registers a `vkfft/plan_f32` resource (binding `data`) and a
`vkfft/execute_f32` operation (binding `plan`). Plan parameters are `dimensions`,
`batch_count`, `direction`, `normalization`, and `batch_tile`, with the compact
row-major, in-place complex-f32 contract of `VulkanFftPlan`. Ordinary recordings
and selected batch partitions export through `Module.add_execution_plan`, keeping
the actual tile. Nonzero buffer subview offsets and unqualified plan subclasses
are rejected. Each prepared binding rebuilds an independent plan; commands retain
its plan and storage until retirement. Configure the adapter with the common
query and inline extension using
`runtime.configure_provider("vkfft", adapter_path=...)`, without a separate vendor
`library_path`. Preparation may compile shaders and initialize LUTs, but never
executes the user FFT or changes input. Warm replay never rebuilds plans. Windows
Vulkan checks cover mixed kernel/FFT 1D and 2D transforms, normalized inverse,
explicit spectral-energy adjoints, tail partitions, two bindings, standalone and
borrowed consumers, and retirement. This does not imply automatic FFT AD or
arbitrary recipe coverage. Complete ordered Vulkan compute binding-frame recipes
also export their selected nodes into one immutable AOT command sequence per
binding. Segmented/graphics variants and executor subclasses need their own
qualification; storage recipes remain separate. Independent AOT uses the explicit
`Runtime("vulkan", device_index=...)` enumeration index (default 0), which can
differ from JIT's automatically chosen GPU. Compare performance on the same device.
The optional SDK component is `TaichiSDKProvider_vkfft`.

`VulkanParallelSortPlan.record()` also exports through `Module.add_execution_plan`.
The frozen `fidelityfx_parallel_sort/plan_u32` resource binds `keys`, optional
`values`, and a `code` host blob; `execute_u32` binds the rebuilt `plan`.
Its parameters are `key_count`, `payload`, `fused_prefix`, and `shader_sizes`
(bytes per SPIR-V module). The existing compact 1D u32 ascending stable-sort
contract and selected prefix strategy are preserved. Physical key/payload
overlap is rejected. Counts and ranges must satisfy the shader's uint32 ABI,
platform byte addressing and the Vulkan device's storage-buffer limits.

DXC is needed only to construct the producer plan. Export retains its compiled
SPIR-V, deduplicates identical host blobs, and serializes no compiler path or
producer address. Consumers need neither DXC, HLSL source nor an optional
provider DLL. Cold preparation creates pipelines, independent workspace and
inline commands without sorting user data or creating an unused secondary
command sequence. Repeated references to one plan reuse its resource within a
prepared binding. Warm replay uses the existing commands; commands retain native
storage and pipeline leases until retirement. `workspace_bytes` counts device
scratch, while host shader bytes and opaque driver allocations are separate.
The producer memory report also identifies its retained host shader code.

Windows Vulkan checks cover keys-only/payload, both prefix strategies, stable
duplicates, two independent bindings, failed-binding recovery, producer closure,
standalone/borrowed consumption and owner retirement on NVIDIA and AMD devices.
Mixed Graph tests use actual downstream `kernel.grad`, including gradients
indexed through the sorted payload; this is not differentiation of integer
sorting. This root ordered path does not qualify Parallel Sort for Vulkan
device-condition loop bodies. CPU/CUDA do not register this native factory;
unqualified plan subclasses require their own reconstruction contract.

`ti.hardware.matrix.CudaMatrixMmaRecording` exports through the same Graph API.
Its native operation is `cuda_driver/matrix_mma_f16_f32`, version 1, with
`parameters={"batch_count": n}` and bindings `a`, `b`, `output`. It reuses Forge's
embedded driver PTX: compact row-major f16 inputs and f32 output, shaped `(16,16)`
for one tile or `(n,16,16)` for multiple tiles. Buffers require 32-byte alignment;
output must not overlap either input, including separately imported aliases.
The positive batch count must fit the existing signed-int launch ABI. CUDA
compute capability 7.0 or newer is required; CPU/Vulkan have no factory.

Cold preparation loads the embedded module before CUDA capture and does not
execute the multiplication. Replay uses the retained Graph and owns no scratch;
CUDA context/module/Graph storage is outside `workspace_bytes`. No cuBLAS,
Toolkit compiler or optional adapter is needed. Windows CUDA checks cover one
and three tiles, standalone/borrowed execution, shared binding families,
physical aliases, failed-binding recovery and owner retirement. A downstream
`kernel.grad` plus explicitly assembled MMA adjoints checks reverse results;
this does not add automatic differentiation of MMA or new JIT recipe selection.

CUDA also registers `gemm_f32` using external cuBLAS:

```python
plan = ExecutionPlan([
    {"kind": "gemm_f32", "a": "a", "b": "b", "output": "c",
     "rows": 17, "columns": 9, "inner": 13, "alpha": 1.25, "beta": 0.25},
])
```

It computes `C = alpha * op(A) * op(B) + beta * C` on compact row-major scalar
f32 matrices. `alpha=1`, `beta=0`, `transpose_a=False` and `transpose_b=False`
are the defaults. Logical dimensions are `(rows, inner)`, `(inner, columns)` and
`(rows, columns)`; a transposed input must have the reversed physical shape.
Dimensions are positive and must fit the cuBLAS SGEMM int32 ABI. Scalars must
be finite f32 values. Output cannot overlap either input; inputs may share
storage. Strided/batched layouts, other dtypes and cuBLASLt algorithm selection
are outside this operation. CPU/Vulkan do not register this factory.

Shapes, transposes and host scalars are frozen during preparation. Each prepared
instance owns an independent handle; preparation captures without executing the
GEMM or modifying application buffers. Replay launches the retained CUDA Graph,
with no per-call cuBLAS lookup or handle setup. Retirement destroys the Graph
before the handle. cuBLAS destruction may itself synchronize the device; it is
a cold retirement cost. `workspace_bytes` reports known Forge-owned scratch,
**excluding unknown cuBLAS handle/capture allocations**; zero does not mean the
provider uses no memory. See NVIDIA's [cuBLAS Graph and resource contracts](https://docs.nvidia.com/cuda/cublas/index.html#cuda-graphs-support).

cuBLAS and its loader dependencies remain application-managed. In the Windows
cuBLAS 13.3 / RTX 5090 check, even SGEMM loaded both `cublas64_13.dll` and
`cublasLt64_13.dll`; include the selected vendor package's actual dependency
closure in deployment sizing. Checks cover four transpose combinations, nonzero
beta, independent frames, rollback, in-flight close and borrowed-owner reset.
An exported energy/reverse kernel plus explicit GEMM adjoints validates gradients
with respect to A and B. This does not provide automatic differentiation of the
external operation or qualify every vendor version/device.

The same cuBLAS registration also reconstructs a device-output f32 dot product:

```python
plan = ExecutionPlan([{
    "kind": "native", "provider": "cublas", "operation": "dot_f32", "version": 1,
    "parameters": {"size": 37},
    "bindings": {"x": "left", "y": "right", "output": "dot"},
}])
```

Inputs are compact scalar f32 vectors of shape `(size,)`; the f32 output has
shape `()` or `(1,)` and cannot overlap either input. Read-only inputs may
alias. `size` is nonnegative. The optional 64-bit entry point is preferred when
available; otherwise `size` must fit the loaded SDOT int32 API. Empty input
writes zero on every replay. `runtime.probe_native_operation("dot_f32")` reports
this operation independently. CPU/Vulkan have no factory for it.

The prepared handle uses device pointer mode, so subsequent Graph kernels can
consume the scalar without a host read or synchronization. Preparation does not
change the output. It shares the handle retirement and workspace accounting above.
Windows cuBLAS 13.3 checks cover repeated device writes, empty input, independent
bindings, invalid layout/alias rollback and in-flight close/reset. An exported
energy/reverse kernel with an explicit dot adjoint checks input gradients;
automatic differentiation of the external call is not provided.

Three additional version 1 cuBLAS operations use the same registered envelope:

| Operation | Required bindings | Behavior |
| --- | --- | --- |
| `axpy_f32` | `x`, `y` | `y = alpha * x + y` |
| `scal_f32` | `x` | `x = alpha * x` |
| `nrm2_f32` | `x`, `output` | Write the Euclidean norm to a device scalar |

All take the nonnegative `size` parameter, compact scalar f32 vectors of shape
`(size,)`, and an optional `context` binding. AXPY/SCAL accept a finite f32 `alpha`
parameter (default 1), or an `alpha` binding to an f32 device scalar of shape `()`
or `(1,)`. Supplying both forms is rejected at preparation. Device coefficients
are read on each replay without a host round trip. NRM2 uses the same scalar
output shapes and overwrites the output with zero for empty input; empty AXPY and
SCAL do not mutate vector storage.

Packed array views are accepted. Writable storage must be disjoint from read-only
inputs, including partial overlaps: AXPY rejects overlapping `x`/`y`, device alpha
must not overlap the mutated vector, and NRM2 output must not overlap `x`.
Read-only inputs may alias each other. These checks run before capture, without
executing the operations. The optional 64-bit vector APIs are preferred, with
int32 fallback when absent; no arbitrary vector-size limit is added. See the
[cuBLAS integer and pointer-mode contracts](https://docs.nvidia.com/cuda/cublas/index.html).

These CUDA operations have no CPU/Vulkan factory. Windows cuBLAS 13.3 checks cover
independent standalone/borrowed frames, changing device coefficients, shared
HOST/DEVICE pointer modes, views, invalid-binding rollback, empty inputs and
retirement. An exported energy/reverse kernel and explicit vector adjoint check
gradients; external operations are not automatically differentiated. The tests
exercise the 64-bit entry points at small sizes, not multi-billion-element
allocations or compatibility with older vendor libraries. Use matching private
ABI 44 components. This supplies reusable operations for persistent plans;
automatic export of every solver or JIT recording remains a separate contract.

Native resources use the same registered envelope in `resources`, with an added
`name`. The CUDA `cublas/context` constructor (version 1, empty parameters and
bindings) creates a reusable handle for one prepared instance:

```python
plan = ExecutionPlan([{
    "kind": "native", "provider": "cublas", "operation": "gemm_f32", "version": 1,
    "parameters": {"rows": 17, "columns": 9, "inner": 13},
    "bindings": {"a": "a", "b": "b", "output": "c", "context": "blas"},
}], resources=[{
    "kind": "native", "name": "blas", "provider": "cublas", "operation": "context",
    "version": 1, "parameters": {}, "bindings": {},
}])
```

Additional GEMM/dot/vector operations can reference `"blas"` through their optional
`context` port. Each preparation reconstructs its own context; omitting that port
preserves independent handles. Sharing is explicit and does not imply faster
preparation. Pointer modes are set while recording, not while replaying. The
Windows CUDA checks include alternating GEMM/device-dot/GEMM capture, independent
frames, explicit gradients and rollback/retirement.

Resource dependencies participate in the same required closure as arrays. Unused
native resources are not probed or created; required constructors are decoded
before private allocation, then created in dependency order. Commands retire
before native resources, which retire before private arrays. Resources are not
dispatchable operations and cannot bind kernel arguments or gradients. This is
an internal registered resource contract, not automatic reconstruction of every
external provider plan. Known resource scratch contributes once to
`workspace_bytes`; opaque vendor memory remains excluded.

CUDA CSR SpMV also uses the common resource/operation registry (version 1):

```python
plan = ExecutionPlan([{
    "kind": "native", "provider": "cusparse", "operation": "spmv_f32", "version": 1,
    "parameters": {"transpose": False, "alpha": 1.0, "beta": 0.0,
                   "algorithm": "csr_alg1", "preprocess": True},
    "bindings": {"matrix": "matrix", "input": "x", "output": "y"},
}], resources=[{
    "kind": "native", "name": "matrix", "provider": "cusparse", "operation": "csr_f32",
    "version": 1, "parameters": {"rows": 3, "columns": 4},
    "bindings": {"row_offsets": "offsets", "column_indices": "indices", "values": "values"},
}])
```

All arrays are compact scalar vectors. Offsets and column indices must both be
i32 or both i64; values, input and output are f32. Offsets have `rows + 1`
elements and start at zero, are monotonic, and end at `nnz`; indices and values
have `nnz` elements. Columns must be in range and unique within each row;
unsorted rows and empty rows are accepted. Dimensions are nonnegative, including
zero-dimensional matrices. `y = alpha * op(A) * x + beta * y` uses the selected
transpose with matching vector lengths. Finite host alpha/beta default to 1/0;
algorithm is `csr_alg1` (default) or `csr_alg2`. Preprocessing defaults on and
uses the common plan's fallback when vendor preprocessing is unavailable.

Topology is immutable until the prepared instance retires. It must already be
initialized before preparation; `initial_data` payloads satisfy this requirement,
but a later plan initialization kernel or zero fill does not. Preparation waits
for prior CUDA work, reads and validates topology once per resource, and completes
optional preprocessing before capture. It does not run SpMV or alter user
outputs. Values may change in place between ordered replays. Changing topology
or storage requires retirement and preparation again. These are explicit caller
contracts; there is no per-replay host read or topology comparison.

Writable output must be disjoint from all CSR arrays and the input. Mutable
values must not overlap topology. Packed views use physical storage ranges for
these checks. Multiple operations may reference one CSR resource; each owns its
own descriptor and workspace so preprocessing state is not overwritten by a
different operation or frame. The executor retains borrowed arrays and retires
captured commands before plans and storage. Known workspace and private array
bytes are reported separately; opaque vendor allocations remain excluded.

Probe `runtime.probe_native_operation("cusparse_spmv_f32")` before allocating the
route. The CUDA factory uses the existing external cuSPARSE loader, with no new
vendor copy in the package; CPU/Vulkan have no factory. Windows cuSPARSE 12.7.9
checks cover standalone/source-free and borrowed consumption, i32/i64 topology,
transpose, coefficient updates, empty matrices, views, rejection and retirement.
An exported energy kernel and its real `.grad` compose with an explicit transpose
SpMV to check input gradients. This does not automatically differentiate an
external call. CSR support started at ABI 44. Triangular solves and BSR use the
separate reconstruction contracts below; components must match the current private ABI.

Root sequential mixed Graphs containing `CusparseSpmvRecording` can be exported
with `module.add_execution_plan(name, graph)`. Export takes one coherent CSR
snapshot per matrix in that transaction, under the submission and numeric-update
gates. Repeated recordings reference one resource. Read-only arrays with identical
bytes, dtype and shape share both archive payload and prepared storage; mutable
workspace is not shared through this rule. The source recording keeps using live
coefficients. An existing export remains frozen; a later export reads the then
current values. Each prepared instance still owns independent execution plans
and workspace, and runs after the producing Graph/matrix has retired.

To keep numeric values live, name the matrix in `source_bindings`:

```python
exported.add_execution_plan("step", graph, source_bindings={"coefficients": matrix})
# The consumer supplies coefficients as a compact one-dimensional f32 ndarray.
```

This opt-in binding denotes the matrix's numeric allocation, not the ndarray
copied by `SparsePattern.matrix()` or `SparseMatrix.update_values()`. CSR values
follow compressed storage order; BSR values follow block-row order with
row-major entries inside each block. The length must equal the stored scalar
count. Export copies only topology, once per matrix; no numeric payload or
initialization copy is added. Consumers must populate valid coefficients before
preparation and may update them in place through ordered writes or preceding
Graph commands. Changing topology, shape or storage requires a new preparation.
Omitting the mapping retains the frozen snapshot behavior above.

The same binding works for CSR/BSR SpMV, CSR SpMM and SpSV/SpSM. Triangular export
sets `mutable_values=True` and inserts `triangular_update_f32` immediately before
each solve, so retained analysis sees changes made earlier in that replay.
Workspace stays private to each prepared instance; warm execution uses the
existing CUDA Graph path without host numeric checks or analysis reconstruction.
Windows/cuSPARSE 12.7.9 checks cover ordinary/selected exports, independent and
borrowed consumers, two bindings, in-Graph updates, source reset, retirement,
and real `energy.grad` followed by an explicit input adjoint. External provider
derivatives and structured-control combinations retain their separate contracts.
See NVIDIA's [cuSPARSE value-update and preprocessing contracts](https://docs.nvidia.com/cuda/cusparse/index.html).

This automatic route covers scalar f32 CUDA CSR, including empty matrices;
it does not convert BSR to scalar CSR. Snapshot reads occur only during export,
not during construction or replay of ordinary JIT recordings. Bounded Windows
checks cover repeated recordings, updated snapshots, shared topology, archive
rollback, source-free/borrowed consumption and real exported gradients with
explicit adjoints. Backend-command nesting in structured `Sequential` and JIT
CGraph ndarray-gradient binding remain separate unsupported paths. The AOT
consumer's explicit gradient binding is covered here.

BSR uses the same registry while retaining blocks without expanding them to CSR:

```python
resource = {
    "kind": "native", "name": "matrix", "provider": "cusparse",
    "operation": "bsr_f32", "version": 1,
    "parameters": {"block_rows": 3, "block_columns": 4, "block_size": 2},
    "bindings": {"row_offsets": "offsets", "column_indices": "indices", "values": "values"},
}
operation = {
    "kind": "native", "provider": "cusparse", "operation": "bsr_spmv_f32", "version": 1,
    "parameters": {"alpha": 1.0, "beta": 0.0},
    "bindings": {"matrix": "matrix", "input": "x", "output": "y"},
}
```

Native resources accept matching i32/i64 block indices, zero-based topology and
packed f32 row-major square blocks. Values have `nnzb * block_size**2` elements;
x/y lengths are block column/row counts times block_size. Columns within a row
may be unsorted but must be unique. Empty rows, zero blocks and zero dimensions
are supported. Preparation validates topology and physical aliases without
executing user SpMV. Explicitly bound values may change in place between ordered
replays with fixed topology; changing topology or storage requires preparation
again. Probe `runtime.probe_native_operation("cusparse_bsr_spmv_f32")` first.
This CUDA factory uses non-transpose BSR ALG1 and the existing compatible external
cuSPARSE library, without packaging another vendor copy.

Ordinary and materialized sequential Graphs containing BSR
`CusparseSpmvRecording` export automatically, taking one snapshot per matrix per
export and retaining the public producer's format contract. Exported read-only
values remain frozen when the source matrix changes; a new export takes current
values. Consumers needing live coefficients can use `source_bindings` as above
or explicit resource bindings.
Windows/cuSPARSE 12.7.9 checks cover standalone consumption without the DSL,
borrowed execution, snapshot isolation, in-place updates, close and in-flight
retirement. Real `energy.grad` composes with SpMV of an explicitly transposed BSR
to check input gradients; the external adjoint is not generated automatically.
Use matching current native/shim components. CPU/Vulkan do not expose this CUDA provider.

SpMM uses the same `cusparse/csr_f32` resource and common registry:

```python
operation = {
    "kind": "native", "provider": "cusparse", "operation": "spmm_f32", "version": 1,
    "parameters": {"rhs_count": 8, "algorithm": "csr_alg3", "preprocess": "required",
                   "alpha": 1.0, "beta": 0.0},
    "bindings": {"matrix": "matrix", "input": "x", "output": "y"},
}
```

The operation is `Y = alpha * A * X + beta * Y`. X/Y must be compact row-major
scalar f32 matrices shaped `(columns, rhs_count)` / `(rows, rhs_count)`. The native
RHS count is positive and includes one; the existing multiple-RHS contract of
`CusparseSpmmRecording` is unchanged. Algorithms are `csr_alg2` (default) and
`csr_alg3`; preprocessing is `none` (default), `optional` or `required`. Required
preprocessing failure rejects preparation without selecting a direct alternative.
CSR resource, aliasing, fixed topology and ordered in-place value-update rules
apply as above. Each prepared operation owns independent descriptors/workspace
and shares the JIT SpMM core. Preparation executes no multiplication; replay
does no analysis, allocation or rebinding. Probe `cusparse_spmm_f32` at the cold
boundary. CPU/Vulkan do not advertise this CUDA provider. A matching native build
containing the factory is required; no additional cuSPARSE copy is packaged.

Root `CusparseSpmmRecording` and selected `SparseSpmmOperation` Graphs export
automatically, preserving RHS, algorithm and preprocessing for `row_streamed`,
`tree_direct` or `pattern_preprocessed`. Prepared plans also persist
`expected_workspace_bytes`; mismatched reconstruction is rejected during
preparation. An unprepared legacy `deterministic` recording keeps its optional
policy without creating a JIT plan during export; a prepared recording freezes
the actual preprocessing outcome. The name does not promise bitwise results
across vendor releases. SpMV/SpMM share one matrix snapshot per export transaction;
unnamed matrix values require a new export when changed, while named values use
the live source-binding contract above. Materialized Graphs retain minimal command
ordering, while frozen search definitions release old execution-plan references.

Windows/cuSPARSE 12.7.9 checks cover all three selected strategies, two independent
binding frames, empty matrices, i32/i64, value updates, shape/alias rejection and
in-flight close/reset. A real energy `.grad` followed by explicit transpose-matrix
SpMM validates input gradients. This does not automatically differentiate an
external SpMM. Triangular solves use the separate contract below.

Triangular solves reuse `cusparse/csr_f32`. A `triangular_plan_f32` resource retains
SpSV/SpSM analysis and binds `matrix/input/output`. Parameters are a required
positive `rhs_count`, `fill_mode="lower"|"upper"`, `unit_diagonal`, `transpose`, and
`mutable_values` (the three booleans default to false). Single-RHS ports have shape
`[n]`; multiple-RHS ports are compact row-major `[n, rhs_count]`. The matrix must
be square; non-unit diagonals require finite nonzero pivots. Solves perform no
host-side numerical validation.

`CusparseSpsvRecording` and `CusparseSpsmRecording` export through
`Module.add_execution_plan`, including ordinary and qualified materialized Graphs.
Export freezes one CSR snapshot per matrix per transaction; forward and transpose
plans share it, and unchanged payloads are deduplicated across exports. Cold
export does not create a JIT triangular analysis plan. This automatic route uses
fixed coefficients (`mutable_values=False`) by default; changing the source matrix
then requires a new export. Named sparse values use automatic refresh as described
above; explicit plans use the mutable resource contract below. Unknown recording subclasses require their own export
qualification.

```python
triangular = {
    "kind": "native", "name": "triangular", "provider": "cusparse",
    "operation": "triangular_plan_f32", "version": 1,
    "parameters": {"rhs_count": 2, "mutable_values": True},
    "bindings": {"matrix": "matrix", "input": "x", "output": "y"},
}
refresh = {
    "kind": "native", "provider": "cusparse", "operation": "triangular_update_f32",
    "version": 1, "parameters": {}, "bindings": {"plan": "triangular"},
}
solve = dict(refresh, operation="triangular_solve_f32")
# Declare the CSR matrix before triangular in resources; execute refresh, then solve.
```

Keep `mutable_values=False` and record only solve for fixed coefficients. Mutable
coefficients require explicit refresh after their producer and before solving;
overwriting values does not update cuSPARSE's retained numeric analysis. A copy
or coefficient producer, refresh, and solve can share one CUDA Graph, without
replay-time analysis, allocation, or added host waits. Each binding frame owns
independent analysis/workspace and may share CSR topology. Completion retirement
protects resource destruction. Missing update symbols reject mutable plans at
preparation.

Cold-probe operations `cusparse_triangular_solve_f32` and
`cusparse_triangular_update_f32`; the plan is a resource, outside operation
discovery. Availability means at least one RHS route is supported; preparation
checks the selected SpSV/SpSM and update capability. CPU/Vulkan do not register
these CUDA factories. Windows CUDA checks cover standalone/borrowed consumption,
i32/i64, both triangles, unit diagonals, transpose explicit adjoints, captured
value refresh, two binding frames, and close/reset. External solves are not
automatically differentiated.

Adapter-backed operations take library locations from the runtime owner:

```python
runtime.configure_provider("cutensor", adapter_path=adapter_file,
                           library_path=vendor_file)
status = runtime.probe_native_operation("cutensor_contract_f32")
# Remove this owner's setting when no longer needed:
runtime.configure_provider("cutensor", adapter_path=None)
```

Configuration copies paths without loading code. Python resolves explicit paths
when configured; omitting `library_path` uses the adapter's vendor-library search.
C/C++ uses `ti_configure_native_provider_ext`; prefer absolute paths there.
Each preparation snapshots the configuration, probes each required adapter
contract once, and shares one provider runtime across its dependent plans. Plans
and workspaces remain local to that prepared instance. Unused providers do not
load. Reconfiguration affects later probes/preparations; existing prepared objects
retain their original library/resource leases. Borrowed runtimes sharing one JIT
device have independent configuration. Paths are not serialized into artifacts.

This configures registered adapter-backed factories, including the cuDSS, cuTENSOR,
cuSPARSELt and AmgX operations below. It does not register arbitrary vendor symbols, change existing
built-in cuBLAS/cuFFT discovery, or automatically export JIT recordings.

Registered built-in factories can also request auxiliary libraries through
`runtime.configure_library(name, path)` (`ti_configure_native_library_ext` in C).
This records a location without loading code; `path=None` removes it. The same
cold preparation session loads only requested names, shares their library leases,
and lets each consuming factory check its required symbols and versions. Existing
prepared owners retain their leases after reconfiguration. Python resolves paths
when configured; prefer absolute paths in C. Paths stay outside artifacts, and no
process search path is changed. This is a factory dependency contract, not a way
to register arbitrary functions. OS loader rules still apply; owner-local settings
do not provide process isolation between incompatible vendor toolchains.

CUDA cuDSS 0.8 supports explicit persistent f32/i32 CSR plans. Configure the
`"cudss"` provider and probe `"cudss_solve_group_f32"`. Supply a Forge adapter with
the common runtime-provider entry point and an external cuDSS library; the
optional installation component is `TaichiSDKProvider_cudss`.

```python
resource = {
    "kind": "native", "name": "solver", "provider": "cudss",
    "operation": "plan_f32", "version": 1,
    "parameters": {"rows": n, "nonzeros": nnz, "matrix_type": "spd",
                   "matrix_view": "full", "phase": "refactor_solve"},
    "bindings": {"row_offsets": "row", "column_indices": "column",
                 "values": "seed_values"},
}
operation = {
    "kind": "native", "provider": "cudss", "operation": "solve_group_f32", "version": 1,
    "parameters": {"rhs_count": 2},
    "bindings": {"plan": "solver", "values": "current_values",
                 "rhs_0": "b0", "output_0": "x0", "rhs_1": "b1", "output_1": "x1"},
}
plan = ExecutionPlan([operation], resources=[resource])
```

`row` is an `(n+1,)` i32 vector, `column` is `(nnz,)` i32, and values are `(nnz,)`
f32. RHS and output vectors are `(n,)` f32. All require compact scalar storage on
the same device. CSR is zero-based; unsorted rows are allowed but duplicate column
entries are rejected. Preparation validates topology once and takes a private
snapshot. Both dimensions must be positive and `nnz` must fit i32 row offsets.
Replay does not read back or validate the matrix on the host.

`matrix_type` accepts `general` (default), `symmetric` or `spd`; `matrix_view`
accepts `full` (default), `lower` or `upper`, with `general` requiring `full`.
`reordering` accepts `default`, `amd`, `nested_dissection` or `natural`; `solve`
accepts `default` or `general`. The default `phase="solve"` retains factors from
the seed matrix and forbids an operation `values` port. `factor_solve` and
`refactor_solve` require current `values`, execute their numerical phase once per
group, then solve every RHS. There is no fixed RHS-count limit; all numbered
`rhs_0…rhs_(count-1)` and matching `output_*` ports must be supplied. Outputs must
not overlap any group input or another output; read-only RHS storage may repeat.

Each prepared instance owns its plan, without duplicating factors or plan storage
per RHS. Close and reconfiguration follow the owner contract above. Cold
preparation includes private seed analysis/factorization/warmup and capture,
without executing user outputs. `workspace_bytes` includes accountable plan
requests and capture parameters; opaque driver/vendor physical residency remains
unknown. Windows CUDA qualification covers standalone/borrowed consumers, all
three phases, independent bindings and kernel.grad with explicit adjoint solves.
It does not provide provider autodiff or cuSOLVER support.

Graph-owned `CudssCaptureRecording` also supports ordinary and selected Graph
export through `module.add_execution_plan(name, graph)`. Export samples the native
plan's frozen CSR seed, matrix type/view and configuration, without resampling the
original matrix or serializing vendor factors or machine-local library paths.
Each plan is sampled once per export; multi-RHS groups retain one numerical
factorization, operations share one reconstructed plan, and identical read-only
payloads are deduplicated within the module. This export requires paired private
ABI 54; rebuilding analysis/factors remains cold preparation. Qualification covers
changes to original seed/display fields, independent source-free consumption,
explicit adjoints and selected recipe order.

Capture export also sets the boolean `solve_group_f32` parameter
`disjoint_bindings=true`, preserving the original graph's requirement that all
array bindings be disjoint, including read-only RHS vectors. Cold validation uses
the existing sorted group check rather than generating quadratically many pair
constraints. Explicit low-level operations default this parameter to `false`
when omitted and continue to allow read-only RHS storage sharing; outputs must
still be disjoint from every input and other output.

Ordinary compilation of `ti.linalg.record_sparse_solve` and strategies selected
by `SparseSolveRecipeProvider` also use this export path. Archives retain the
materialized plan's ordering/solve configuration, `solve`/`factor_solve`/
`refactor_solve` phase and complete RHS group. Consumers do not search again or
duplicate factors per RHS. Fixed seeds remain distinct from values supplied on
each invocation. Small Windows CUDA SPD checks cover standalone/borrowed use
after source closure, changing RHS/values, actual `kernel.grad` and explicit
symmetric adjoints; they do not qualify every matrix class or control composition.

Ordinary `CudssPlan.recording()` (`CudssSolveRecording`) also exports a frozen
`solve` resource with paired private ABI 59. Under the native storage/submission
locks, export verifies that successful factors still match the matrix identity,
pattern and numeric version, then copies its CSR data and native configuration
once per plan per export. Updating matrix values requires a new successful
factorization before re-export; previously exported packages keep their original
values. Export rejects unfactored, stale, in-flight, explicit-value-only or closed
owners. Consumer preparation rebuilds factors, and replay uses the retained
plan without host matrix queries. Windows CUDA checks cover standalone/borrowed
consumption after closing the source plan, separate bindings, and real kernel
gradients with an explicit symmetric solve adjoint.

Ordinary `CudssPlan.record_refactor_solve()` exports through the same API with
paired private ABI 62. Its `cudss/transaction_plan_f32` v1 resource uses the CSR
seed ports and configuration above, without a `phase` parameter. The
`cudss/refactor_solve_f32` v1 operation has empty parameters and `plan`, `values`,
`rhs`, `output` ports. It reports `cuda_ordered`, retaining ordinary host-invoked
transaction semantics and stream ordering with preceding/following kernels.
Export requires completed analysis and a matching pattern, but no valid factors.
Consumer preparation makes a private seed copy and analyzes it; it does not
factor the seed or write user outputs. Each invocation reads explicit values,
performs full factorization if no valid factors exist, otherwise refactorization,
and then solves. Unconfigured source plans preserve vendor default policies.

The source may have explicit-value factors, stale numerical factors, or a retired
failed transaction. In-flight, unanalyzed, closed and incompatible-pattern sources
are rejected. All three operation arrays must be compact f32 vectors and mutually
disjoint. The existing owner permits one in-flight transaction: preparation rejects
repeated use of that owner in one submission; a completed event must retire before
reuse. Independent owners can compose forward and adjoint solves. For multiple RHS
with one factorization, use the captured group above. Mixing frozen solve and
transactional export of the same source owner is rejected, not silently split.

A provider failure retains partial work until completion, skips subsequent work,
and invalidates that prepared execution. Retire it and prepare again to retry;
failed factors cannot be reused. GPU conditional bodies still require a captured
operation. `workspace_bytes` accounts for the private CSR seed; vendor factor
allocations/residency remain opaque. Windows CUDA checks cover analysis-only seeds,
updated values, independent bindings, partial failure and recovery, standalone and
borrowed retirement, real downstream `kernel.grad` and explicit symmetric RHS
adjoints. This does not add provider autodiff or ordinary cuDSS inside GPU loops.

Graph-owned cuDSS groups can also be appended to a CUDA `Sequential` loop body
and exported from selected flat or nested conditional loops. The common capture protocol
requires explicit conditional-body qualification; other provider commands remain
unqualified by default. Each group has ready seed factors and a frozen phase:
`solve` retains them, while an update group performs its numerical phase before
every RHS. Zero iterations leave the ready resource intact. Captured parameter
storage is frozen and retained until the graph retires; changed JIT allocation
bindings require a matching capture instead of patching only kernel arguments.
Windows CUDA checks cover all three phases, multiple RHS, changing values,
zero/early/budget exits, alternating bindings and explicit reverse-order adjoint
solves. Nested JIT construction selects the qualified conditional route for these
groups; expanded kernel-only routes reject provider commands before preparation.
AOT export translates physical dispatch boundaries to portable operation boundaries,
so a recording that emits multiple operations stays in its original loop region.
Small Windows CUDA tests cover nested `solve`/`refactor_solve`, two RHS, changing
limits and bindings, terminal counters, explicit JIT adjoints, and real downstream
`kernel.grad` followed by reverse-order solves in standalone and borrowed AOT
consumers. They also cover close/reset retirement. This does not qualify every
nested provider recipe or fuse the entire JIT provider step into one submission.

CUDA cuTENSOR reconstruction uses the existing provider C ABI and the same
native operation/resource registry. Select the optional cuTENSOR SDK adapter and
configure a compatible external vendor library as above. For example, a packed
f32 contraction `D = alpha * A @ B + beta * C` uses a named plan:

```python
def tensor(shape, modes):
    return {"shape": list(shape), "modes": [ord(mode) for mode in modes]}

plan = ExecutionPlan([{
    "kind": "native", "provider": "cutensor", "operation": "contract_f32", "version": 1,
    "parameters": {"alpha": 1.0, "beta": 0.0},
    "bindings": {"plan": "contraction", "a": "a", "b": "b", "c": "c", "d": "d"},
}], resources=[{
    "kind": "native", "name": "contraction", "provider": "cutensor",
    "operation": "contraction_plan_f32", "version": 1, "bindings": {},
    "parameters": {
        "a": tensor((5, 7), "ik"), "b": tensor((7, 3), "kj"),
        "c": tensor((5, 3), "ij"), "d": tensor((5, 3), "ij"),
        "compute_mode": "f32", "workspace_preference": "default",
    },
}])
# Producer: exported.add_execution_plan("contract", plan).
# Consumer: module.execution_plan("contract").prepare(module, bindings).
```

Version 1 accepts compact scalar f32 arrays with positive extents. Signed int32
mode labels must be unique within each tensor and have consistent extents across
tensors. Modes describe the contraction/permutation; storage remains packed in
row-major order. All four array ports are required, including C when beta is zero.
D must be disjoint from A/B; it may exactly alias C when shapes and modes match.
Partial overlaps are rejected. Shapes and scalar parameters are fixed at
preparation; change contents in place or prepare again for new bindings/settings.

`compute_mode` accepts `f32` (default) or `tf32`; workspace preference accepts
`min`, `default` or `max`. `alignment_bytes` defaults to 256 and must be a power of
two satisfied by every bound pointer. `workspace_limit_bytes=0` uses the adapter's
workspace estimate; a positive value caps the requested plan workspace. The
provider may reject a configuration its algorithms cannot support. The adapter's
existing rank limit also applies; Forge adds no lower arbitrary limit.

Ordinarily compiled and selected `ti.linalg.record_contraction` Graphs export
through `Module.add_execution_plan`. Export preserves physical input packing,
vendor/separate alpha-beta epilogues, ReLU kernels and private intermediates.
Public shape/type and output/input disjointness constraints remain enforced even
when the vendor reads packed arrays; identical C/output bindings remain legal.
Selected recipes include the optional version-1 `expected_workspace_bytes`
parameter, including zero. Preparation rejects a different reconstructed workspace
size before allocating scratch or executing the plan. This preserves the declared
dataflow and vendor plan request, not a portable vendor kernel binary; it does not
repeat Forge strategy search. Older readers reject this unknown parameter.

Preparation creates the vendor plan and scratch, then captures contractions with
the surrounding AOT kernels into one CUDA Graph. It does not execute the
contraction or mutate its outputs. Replays perform no provider lookup or plan
creation. Plans retain their provider runtime, adapter and vendor library until
dependent commands retire. Known scratch is counted once in
`prepared.info["workspace_bytes"]`; opaque vendor allocations are excluded.
CPU/Vulkan do not advertise this CUDA provider operation.

Windows CUDA with cuTENSOR 2.7 checks standalone consumers without the DSL and
borrowed JIT owners, forward contraction, exported energy/reverse kernels with
explicit contraction adjoints, output permutation, exact C/D aliasing,
independent frames, invalid-binding rollback, reconfiguration and in-flight
close/reset. This does not provide automatic differentiation of cuTENSOR calls,
arbitrary control/provider combinations, or qualification of every tensor
topology/device. Selected f32 matrix/batched fixtures additionally cover packed
and separate epilogues, public-binding rejection and workspace-drift rejection;
their vendor workspace is zero, so nonzero-workspace device qualification remains
separate.

CUDA cuSPARSELt uses the same configuration and resource path. Install its optional
SDK adapter and call `runtime.configure_provider("cusparselt", ...)`. A version 1
`cusparselt/matmul_plan_f16` resource owns the plan, compressed A and scratch;
`compress_f16` writes that storage, and subsequent `matmul_f16` operations reuse it:

```python
def lt(operation, bindings, **parameters):
    return {"kind": "native", "provider": "cusparselt", "operation": operation,
            "version": 1, "bindings": bindings, "parameters": parameters}

plan = ExecutionPlan([
    lt("compress_f16", {"plan": "weights", "a": "a"}),
    lt("matmul_f16", {"plan": "weights", "b": "b", "c": "c", "d": "d"},
       alpha=0.75, beta=0.25),
    lt("matmul_f16", {"plan": "weights", "b": "b", "c": "c", "d": "e"},
       alpha=-0.5, beta=0.5),
], resources=[dict(lt("matmul_plan_f16", {}, m=32, n=48, k=64), name="weights")])
```

Inputs are packed row-major f16: A(m,k), B(n,k), and C/D(m,n). The operation is
`D = alpha * A @ B.T + beta * C`, with f32 accumulation and f16 output. A must
already satisfy row-wise 2:4 sparsity; Forge neither prunes it nor reads it back
to validate values. The adapter requires positive m/n/k divisible by 16. Default
`alignment_bytes=16` must be a power of two satisfied by bound pointers. D may
exactly alias C but must not overlap B; partial C/D overlap is rejected.

Plan parameters also accept `algorithm_id` (default -1 for the vendor default),
`split_k`, `split_k_mode`, `split_k_buffers` (omission keeps vendor defaults), and
boolean `relu` (default false). These explicit physical choices use the existing
configured-plan adapter ABI; incompatible choices fail at preparation and are
not silently retuned. ReLU is applied to each matmul output when enabled.

The unconditional sequence must compress a plan before reading its compressed
storage. This is checked before recording, independently for each preparation;
no initialization check is added to replay. Each replay follows the declared
sequence, so one compression can feed multiple multiplies without duplicate
compression. Preparation does not advance the mathematical work. The example
above refreshes on every replay. For fixed weights, move the `compress_f16`
operation into `initialization=[...]` and leave the multiplies in `operations`.
`prepared.initialize()` samples A and waits; A can then be changed or closed if
no replay operation/resource needs it. A new preparation creates a new snapshot.
Conditional initialization will need control-flow validation when structured
AOT plans are supported.

Each prepared instance owns independent compressed data and scratch. Their known
bytes contribute once to `workspace_bytes`, regardless of the number of uses;
opaque vendor storage remains excluded. Compression operations share one scratch
lease per plan. If compression occurs only during initialization, that scratch
retires with the initialization commands; if replay compresses too, it remains
live. Compressed values and matmul workspace persist. An explicit adjoint that
needs dense A must retain it or declare a private array copied during initialization;
the runtime does not invent a gradient-specific weight representation.
Windows CUDA/cuSPARSELt 0.9.1 checks cover
DSL-free and borrowed consumers, changing valid sparse inputs, shared compression,
fixed-pattern explicit f32 adjoints, one-time snapshots, scratch retirement,
live B updates, ReLU, C/D feedback, independent instances,
uninitialized-read/alias rollback and in-flight close/reset. The explicit adjoint
does not differentiate sparsity selection or f16 rounding. CPU/Vulkan do not
advertise these CUDA operations. Arbitrary recording export remains separate.

Refreshing cuSPARSELt recordings can also be exported directly from a sequential
Graph. The description preserves each compression epoch, shared-A groups,
coefficients, alignment and configured algorithm/split-K/ReLU facts. A group
compresses once before its products; separate recordings retain their individual
compression steps even when they use the same resource. Unknown plan subclasses
are rejected.

Each recording also retains its full access-group contract: outputs cannot
overwrite raw A or another product's inputs/outputs. Read-only sharing and each
product's own identical C/D view remain legal. Low-level explicit operation
sequences retain their individually declared contracts.

`linalg.record_sparse_matmul` also exports its ordinary, baseline and selected
physical recipes. A shared-A group retains one compression per invocation, all
ordered products, alpha/beta, the selected algorithm/split-K attributes and
either vendor ReLU or the separate ReLU kernels. Current valid 2:4 A values are
compressed during replay; no host readback, pruning or algorithm search is added.
The native plan's optional `expected_resources` contains the frozen compressed-A,
compression-scratch and matmul-workspace byte counts, in that order. When present,
preparation checks both the exact restored configuration and these sizes before
allocating Forge's plan buffers. A mismatch rejects preparation instead of
silently choosing a different plan. Older readers reject this new parameter.

Windows CUDA/cuSPARSELt 0.9.1 checks cover two/three-product groups, both ReLU
routes, source retirement, independent/borrowed consumers, valid sparsity-pattern
updates, own C/D feedback, read sharing and rejected cross-product aliases. They
also check a real downstream f32 `energy.grad` and explicit fixed-pattern
provider adjoints; this is not automatic differentiation through compression,
sparsity selection or f16 rounding. No additional SDK component is required.

For a portable snapshot recording, explicitly retain raw A at compression time:

```python
plan.compress(weights, retain_aot_data=True)
builder.append_native(plan.record(b="b", c="c", d="d"))
module.add_execution_plan("step", builder.compile())
```

The opt-in performs one cold readback under the same Program submission guard as
compression. `plan.aot_data_bytes` reports retained host payload bytes, separately
from device workspace; the default retains zero bytes and adds no readback.
Recompression replaces that retained epoch, and starting a refreshing recording
or closing the plan releases its reference. Changes to the original A after
compression do not affect the exported snapshot. An exported module keeps its
own reference, so it remains savable after producer retirement.

Graph export stages the immutable bytes through the shared data-resource path,
with no device sampling. It emits one initialization compression per shared plan,
followed by the original replay operations. Multiple exports/content-identical
payloads share one artifact entry. Each consumer preparation owns its raw A and
compressed state; initialization retires raw A and compression scratch when no
replay dependency needs them. B/C remain live external bindings. Export preflight
or backend compilation failure publishes no data payloads. Without the opt-in,
snapshot export still reports the missing raw-A payload; the explicit
initialization plan above remains available. Vendor compressed buffers are
never serialized. Bounded Windows CUDA/cuSPARSELt 0.9.1 checks cover source
mutation/retirement, repeated/shared export, separate instances, explicit dense
addend adjoints, initialization retirement and in-flight close/reset. This does
not introduce automatic differentiation of compression or sparsity selection.

CUDA AmgX uses the same provider configuration, resource registry and external
adapter C ABI. Configure `"amgx"` with its SDK adapter and an application-supplied
compatible vendor library. Its scalar CSR resource and operations are version 1:

| Operation | Bindings | Parameters |
| --- | --- | --- |
| `solver_csr` resource | none | positive int32 `rows`, `nonzeros`; `dtype`: `"f32"`/`"f64"`; inline `config` string; `compute_residual` (default true) |
| `setup_csr` | `solver`, `row_offsets`, `column_indices`, `values` | none |
| `replace_coefficients` | `solver`, `values` | none |
| `solve` | `solver`, `rhs`, `solution`; optional `status`, `iterations`, `residual` | `initial_guess`: `"zero"` (default), `"upload"`, `"retained"`; `retain_initial_guess` (default false) |

For example, the producer can declare:

```python
def amgx(operation, bindings, **parameters):
    return dict(kind="native", provider="amgx", operation=operation,
                version=1, bindings=bindings, parameters=parameters)

solver = dict(amgx("solver_csr", {}, rows=n, nonzeros=nnz, dtype="f32",
                  config=amgx_config_string), name="solver")
setup = amgx("setup_csr", dict(solver="solver", row_offsets="rows",
                              column_indices="columns", values="values"))
solve = amgx("solve", dict(solver="solver", rhs="rhs", solution="solution",
                          status="status"), retain_initial_guess=True)
exported.add_execution_plan("solve", ti.aot.ExecutionPlan(
    [solve], initialization=[setup], resources=[solver]))
```

All arrays are compact scalar vectors. CSR offsets/indices use i32 with extents
`rows + 1`/`nonzeros`; values, RHS and solution use the resource dtype with extents
`nonzeros`, `rows`, `rows`. Optional status/iterations are i32[1], residual is
f64[1]. Diagnostic outputs must not overlap any other bound input/output. RHS and
solution may share identical storage. Status 0 denotes convergence; a successful
API call may report nonconvergence and still publish an iterate. Omitting the
residual pass writes NaN when a residual output is bound.

Preparation creates a deferred resource without sampling matrix contents.
`setup_csr` samples topology and coefficients when executed, stages topology to
host and calls vendor upload/setup. Put it in `initialization` to sample once;
its input leases and topology staging retire after completion when replay does
not need them. The known staging contributes to `workspace_bytes`; opaque
vendor-owned matrix/solver allocations do not. Zero known workspace after
initialization does not mean zero GPU storage. A replay `setup_csr` explicitly
rebuilds the solver and clears its retained guess. `replace_coefficients` keeps
topology fixed and performs the vendor resetup.

`"retained"` requires a preceding solve in the recorded sequence, including
initialization. `retain_initial_guess=True` instead bootstraps with zero/upload
and subsequently reuses the solver-owned iterate; changing the caller's solution
array does not replace that iterate. These facts belong to each prepared instance.
Runtime failures invalidate that execution; nonconvergence remains a solve status.

The three actions advertise `cuda_synchronous` and use `cuda_mixed` execution.
The current AmgX ABI has no capture/stream contract. Adjacent kernels still form
native graph segments; context completion fences protect the synchronous action.
This does not promise a single graph launch or a warm speedup over JIT.
Windows CUDA/AmgX 2.5.0 checks cover small f32/f64 tridiagonal systems, DSL-free and
borrowed consumers, independent instances, retained guesses, coefficient updates,
rebuilds, nonconvergence, setup failure, input retirement and in-flight close/reset.
Explicit SPD adjoint solves and coefficient gradients are checked against a scalar
reference; AmgX calls do not acquire automatic derivatives. CPU/Vulkan do not
advertise these operations. Automatic JIT recording export remains separate.

The producer exports registered recipes through
`module.add_primitive(name, kind, version=1, **options)`. Version 1 supports
`sort`, `compact` and consecutive `unique`. Sort/compact use the symbolic
bindings and columns described above. For example:

```python
exported.add_primitive("runs", "unique", key_dtype="i32",
                       keys="keys", output="first_keys", count="run_count",
                       values="payload", unique_values="first_payload",
                       indices="first_indices")
# Lightweight consumer: bind external arrays once, then replay.
prepared = module.execution_plan("runs").prepare(module, bindings)
prepared()
```

Unique accepts 1D scalar i32/u32/i64/u64 keys. It processes the whole bound key
extent, keeps the first item of each consecutive run and does not sort. Keys,
output and count default to those argument names. Payload input/output must be
specified together and follow compact's scalar/vector/matrix record contract.
Optional `indices` writes i32 first-element indices. Outputs cover the key
extent; count is rewritten every replay, including zero for empty input. Output
tails remain unchanged. A new extent requires another prepared instance. This
recipe does not yet expose a separate fixed-prefix `size` option.

Export expands the recipe into the existing boundary kernel and registered
compact operation. Modules share helpers by dtype/index variant; preparation
reconstructs private i32 flags and optional starting indices. The consumer needs
no Python recipe or new unique backend algorithm. CPU batches, CUDA capture and
Vulkan inline recording use the expanded plan. The native capability query for
`"unique"` remains empty: this is a kernel/compact composition. Vulkan 64-bit
keys require `Module(caps=["spirv_has_int64"])` and a compatible device.
Selection has no automatic derivative. Windows CPU/CUDA/Vulkan checks cover
explicit gather forward/reverse through these indices, content changes, integer
boundaries, payloads, empty input and in-flight retirement.

This format does not automatically export arbitrary JIT native Graphs,
other mathematical primitives/providers, structured control or rendering resources.
Unknown versions and operations report errors. The buffer/module/runtime
in-flight lifetime contract above still applies.

Memory handles returned by C API allocation/import functions belong to that
runtime. Using another runtime's handle or a freed handle reports an argument
error before backend memory access. Copy slices must fit within the declared
allocation size. Applications must still retain buffers until submitted work
finishes and unmap mapped memory before releasing it.

`ti_copy_memory_from_host_ext` and `ti_copy_memory_to_host_ext` provide explicit
synchronous host transfers on CPU/CUDA/Vulkan, including allocations without
host mapping flags. They validate the device slice, wait for prior runtime work
and complete the transfer before returning. The host buffer must contain enough
bytes and must not alias the device slice. Empty slices are validated no-ops.
Device-to-device copies are also available on CPU/CUDA; overlapping slices are
invalid. Separate imported handles that alias external storage remain the
caller's responsibility.

Fixed contiguous AoS `StructNdarray` elements can be exported on CPU, CUDA and
Vulkan, including nested structures and vector/matrix members. Integer/float
types still require the target capabilities; for example, Vulkan modules using
8/16-bit integers must declare `spirv_has_int8`, `spirv_has_int16`,
`spirv_has_int8_storage` and `spirv_has_int16_storage` in `caps`.

Obtain a module-owned `TiNdArrayLayout` once with
`ti_get_kernel_ndarray_layout_ext` or
`ti_get_compute_graph_ndarray_layout_ext`. Query its version, element stride,
alignment and leaf members with `ti_get_ndarray_layout_info_ext` and
`ti_get_ndarray_layout_member_ext`. Members report dotted paths, scalar types,
tensor shapes and byte offsets. Pack data using this target layout; host C/C++
structs are not assumed to have the same padding.

Pass memory and outer shape through the existing `TiArgument` ndarray, setting
`elem_type=TI_DATA_TYPE_UNKNOWN` and an empty `elem_shape`. Supply the queried
layout and version in `TiNdArrayBinding` or `TiNamedNdArrayBinding`, then call
`ti_launch_kernel_with_bindings_ext` or
`ti_launch_compute_graph_with_bindings_ext`. `UNKNOWN` alone is not a structure
binding. Calls reuse immutable layouts without resending or recursively
rebuilding member tables. The same bindings accept primitive/tensor ndarray
gradients through an optional `gradient` pointer. Older gradient entrypoints
remain compatible.

Wrong layout/version/rank, short buffers, misaligned imported storage, foreign
runtime handles and retired handles fail before structure dispatch. Graphs
validate all structure and gradient bindings before the first dispatch. Kernel,
Graph and layout handles, including member path strings, expire with their
module; query new handles after reloading. Descriptors need only survive the
launch call; device memory must survive completion. Destroy modules before
their runtime and serialize host-side module/binding lifetime changes with use.

Structure Graphs use `graphs.v2.json`; primitive/tensor-only Graphs retain the
old representation. New loaders read both; old loaders cannot consume structure
Graph exports. This covers direct calls and ordinary CGraphs, not arbitrary
strides/slices, pointer members, structure gradients or native Graph capture.
Small Windows CPU/CUDA/Vulkan and CUDA PTX consumers verify nested mixed-type
records, rebinding, retirement, and Graphs combining structure forward work
with explicitly bound primitive-array reverse differentiation.

Vulkan modules can export dense fields created after `Module()` and reuse kernels
compiled before unrelated fields were created. The archive includes only layouts
referenced by exported kernels or explicit `add_field()` entries; ndarray-only
modules retain no field roots. Referenced trees must remain active until export
finishes. This preserves module-local storage and explicit gradient execution;
ordinary `add_kernel`/`add_graph` exports do not snapshot producer field contents.
The `add_execution_plan(..., snapshot_state=True)` path described above explicitly
exports qualified dense state.

Without initial state, CPU/CUDA bitcode and CUDA PTX exports containing only dense fields and
serial/range tasks use schema 5 (`resource_binding="dense_v1"`); CUDA cubin uses
schema 6 with the same resource binding. The module compacts its required resource
closure and reserves independent runtime IDs before field allocation. Two
instances of the same artifact can coexist in one runtime, with independent
primal and gradient storage. Bitcode resolves explicit tree symbols before code
generation; PTX/cubin bind per-module constants once before exposing a kernel. The
kernel argument and ordinary Graph layouts are unchanged.

Exporting CPU/CUDA field kernels does not allocate the producer's compile-only
field trees. If the same Python session subsequently uses JIT kernels, prepares
an ordinary Graph or accesses fields on the host, Forge allocates their storage
once using the existing layouts. Previously materialized field contents are
preserved. Small CPU/CUDA/Vulkan tests cover this transition with primal,
reverse and forward-mode differentiation; this is separate from consumer-side
AOT module resource isolation.

Destroying a CPU/CUDA module waits for submitted work, retires its executable
registrations and releases its demoted dense trees and reservations. Legacy
artifacts and modules requiring sparse/list/mesh/quantized operations retain
producer IDs; overlapping IDs require separate runtimes and are rejected before
initialization. Sparse/list-managed storage still lives until runtime teardown.
Vulkan dense AOT resources also use module-local mappings. Always destroy
modules before their runtime.

CUDA exporters can select `ti.aot.Module(artifact_format="ptx")`, or pass
`--artifact-format ptx` to the example exporter. The default remains LLVM
bitcode. PTX exports perform user-kernel LLVM optimization and PTX generation
once on the producer; consumers load PTX directly through the CUDA driver.
The driver still compiles PTX for the device, and runtime initialization still
uses bundled LLVM. No external `ptxas` is required for this path.

Select a deployment target with `caps=["cuda_compute_capability=80"]`, for
example. The default target is compute capability 80 with PTX 7.0 (Ampere and
newer); select the deployment device explicitly for a more specific target.
For example, compute capability 120 uses PTX 8.7. Selecting a
newer target narrows the device range. Export-only `compile_options` are baked
into PTX. Schemas 4/5 record the target, entry points, register limit, numerical
policy and payload checksum, and binds the payloads to their launch metadata.
Missing or damaged PTX never falls back to a stale bitcode file. A failed kernel
lookup remains failed within that module; reload the module after replacing
an artifact. Schema 5 additionally checks the dense resource relocation contract.
Legacy schema 2 text, schema 3 bitcode and schema 4 PTX remain readable.

PTX deployment was checked on Windows CUDA with dense-field and scalar/vector/
matrix ndarray forward/reverse kernels and ordinary CGraphs. This does not
establish a runtime speedup or remove the field-resource and native-recording
boundaries above. Compare producer cost, consumer load time, size and execution
on the intended device before choosing PTX for deployment.

For a known deployment GPU, select cubin to also assemble device code on the
producer. An explicit target and an external NVIDIA `ptxas` supporting that
target and PTX version are required:

```python
module = ti.aot.Module(artifact_format="cubin",
                      caps=["cuda_compute_capability=120"])
```

Set `TI_CUDA_PTXAS_PATH` to the assembler executable, or make `ptxas` available
on `PATH`. The example accepts `--artifact-format cubin --cuda-target 120`.
Explicit cubin export invokes the assembler even if `TI_CUDA_PTXAS_MODE=driver`;
assembly failure fails export instead of saving PTX under a cubin extension.
Consumers need neither `ptxas` nor a CUDA Toolkit installation. They still need
a compatible NVIDIA driver and this SDK's LLVM runtime for initialization.
The tested producer uses LLVM 20.1.7 and ptxas 13.3.73 on Windows, targeting
`sm_120`/PTX 8.7, with an RTX 5090 consumer. This does not require every producer
to use that exact ptxas patch version.

Cubins without initial root state use schema 6, recording payload/ABI/resource checksums and assembler
identity. Payloads load on demand and are released with the module; missing or
damaged files do not fall back to PTX or bitcode. The loader requires the same
compute-capability major and an equal or higher minor. This is narrower than
PTX compatibility; architecture-specific suffix targets are not exposed by this
API. Check NVIDIA's [binary compatibility rules](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html)
and validate each deployment device/driver combination. GPU availability in the
exporting process must not be used to infer the deployment target.

Small Windows cubin consumers cover dense fields, scalar/vector/matrix ndarray
gradients, fixed structure arrays, direct/ordinary Graph execution, independent
module resources, retirement and reload. A real VBD artifact demonstrates lower
kernel load time and payload size; its deployed heavy execution has not been
qualified. External assembly can change execution performance as well as
producer cost. Compare warm execution on the target GPU before choosing cubin.
No JIT provider or AOT artifact default changes with this feature.

PTX 6.0 is needed for Forge's synchronized warp collectives on older targets;
the processor's minimum ISA alone is insufficient. See NVIDIA's
[shfl.sync specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync).

### cuSOLVERDn retained Cholesky recordings

The CUDA registry provides `cusolverdn/cholesky_plan` v1 resources and
`cusolverdn/cholesky` v1 operations. Configure the optional common adapter through
`runtime.configure_provider("cusolverdn", adapter_path=adapter, library_path=vendor)`.
Probe `cusolverdn_cholesky` before preparation; its composition capability is
`cuda_ordered`. CPU and Vulkan have no factory for this operation.

The resource parameters are `rows`, `rhs_count`, `dtype` (`f32` or `f64`) and
`seeded` (default false); ports are `factor` and `status`. Factor storage is a
compact `(rows, rows)` array with meaningful row-major lower triangle; status is
device `i32[2]` for factor and solve results. Preparation retains this storage,
creates vendor handles and workspace, and does not execute a factorization or
write caller arrays. `seeded=True` declares supplied factors; it does not prove
numerical success. A solve needs that seed or a preceding unconditional refactor.

The operation parameter `phase` is `solve` or `factor_and_solve`, with ports
`plan`, `a`, `rhs`, `solution`. A is preserved. RHS and solution shapes are
`(rows,)` for one RHS and `(rhs_count, rows)` otherwise. Only identical RHS/solution
aliasing is allowed. All dimensions, layouts, allocation spans and provider
availability are checked at preparation; there are no per-element host
checks or fixed application-size limits. Numerical status remains on device.

Existing `binding.capture(...).record(...)` recordings export automatically in
ordinary and selected Graphs through `module.add_execution_plan(name, graph)`.
The first `factor_and_solve` recording rebuilds private factor scratch and stores
no unused old factor. A first `solve` recording requires `snapshot_state=True`,
copying the actual private factor and status at one synchronized export boundary.
It never substitutes the current A, which may have changed since factorization.
Operations sharing one producer plan share one reconstructed plan per prepared
instance. Different prepared bindings have independent plan state.

When Graph kernels use `plan.info`, supply its symbolic name explicitly:

```python
module.add_execution_plan(
    "step", graph, snapshot_state=True,
    source_bindings={"status": plan.info},
)
```

Here `status` must be the same symbolic binding used by those kernels. The initial
status snapshot is copied once on first execution, not at prepare; later kernels
and solver operations use the same public device array. Without that mapping,
status remains private to the reconstructed provider plan.

Cold preparation records the retained vendor graph. Replay preserves root order
with surrounding kernels and uses the common completion/retirement owner; it
does not call vendor factor/solve APIs or rebuild captures on the host. This
contract does not include enclosing Graph fusion or GPU conditional bodies.
Windows CUDA tests cover f32/f64, single/multiple RHS, source retirement,
standalone and borrowed runtimes, independent bindings, failure status, and
exported kernel gradients with explicit symmetric adjoint solves. Provider
autodiff and the separate cuSOLVERSp sparse route are not included.

### Compiled application hosts (experimental)

The private producer helper `taichi_forge.tools._compiled_host.lower_module`
can replace DSL definitions with compiled template constructors while retaining
ordinary host code. Its output is intermediate build input: compile it with the
host and distribute the extension, not the generated Python or C files. The
consumer uses the existing Forge compiler for genuinely new specializations;
loading previously exported AOT artifacts still uses the lightweight runtime.
Plain host modules without DSL definitions acquire no dependency on this helper.

For larger hosts, the private `taichi_forge.tools._native_host.NativeConstructors`
producer can supply `lower_module(..., _constructor=constructors.constructor)`.
Its `emit(header_name)` returns a Cython declaration and matching C++/header
build inputs; link the C++ into the same host extension. Shared construction and
cleanup reduce repeated native code without changing the frontend template or
requiring a new consumer library. Build against the matching CPython headers.
Native-constructor tests cover exact AST values/locations, Unicode and wide
integers, Python exception propagation and temporary-object retirement.

The current bridge preserves explicit signatures, lexical captures, class and
static methods, `ti.func`/`ti.pyfunc`/`ti.real_func`, and compiler-generated
gradients in bounded tests. Windows CPython 3.10/Cython 3.3.0 host fixtures pass on
CPU/CUDA/Vulkan; this is not qualification of an entire application or of other
Python/Cython ABIs. Production builds use `binding=True`,
`annotation_typing=False` and `emit_code_comments=False`; see the
[Cython compiler directives](https://docs.cython.org/en/latest/src/userguide/source_files_and_compilation.html#compiler-directives).
The private template protocol must match the producer's Forge frontend. It does
not change JIT or AOT defaults, and still needs application dependency and size
qualification. Omitting source files is not confidential execution: compilation
templates can be inspected in the consuming process.

The private `_host_build.py` producer creates a clean, immutable host layout
from an explicit module inventory. Install the producer-only versions in
`packaging/constraints/host-build.txt` in a separate build environment; the
consumer does not need Cython or setuptools. Initialize its C++ compiler and
use a matching CPython interpreter. For example, from a Forge source checkout:

```sh
python python/taichi_forge/tools/_host_build.py --spec app-host.json \
    --work host-cache --output host-layout --jobs 4
```

The specification names canonical modules, including every required package
initializer, and locks the producer versions:

```json
{
  "version": 1,
  "source_root": "private-source",
  "modules": {
    "simulation": "simulation/__init__.py",
    "simulation.solver": "simulation/solver.py"
  },
  "toolchain": {
    "python": "3.10.11",
    "cython": "3.2.9",
    "setuptools": "82.0.1"
  }
}
```

`source_root` is relative to the specification; module paths stay within that
root. Cache, output and private sources must be separate. Aliases already
registered by application initialization may be recorded as
`"aliases": {"solver_alias": {"target": "simulation.solver", "registered_by": "simulation"}}`.
This records an existing import contract; it does not install an import hook or
compile a second copy. Imports that replace their own module or `__file__` need
explicit inventory entries. An observed import list is not proof of complete
dynamic dependency coverage.

The output contains only declared extensions and `host-manifest.json` with
relative artifact names, hashes, template entries and producer identity. Cache
hits verify the artifact bytes and skip template construction and compilation.
Changed producer/source/toolchain settings invalidate the affected recipes.
Existing output must match its manifest exactly; use a fresh output directory
for a different build. Compilation failure leaves the previous output intact.
Build logs and generated Python/C++ remain in the producer cache. One producer
owns a cache at a time, with parallel per-module workers.

This producer keeps Python object inference for ordinary host code and boxes
literal-tuple conditional branches to avoid accidental fixed C tuple types.
Host modules use Python syntax mode, so Python identifiers such as `include`
and `cdef` remain valid. Native template declarations are separate producer-only
files; installed extensions need no Cython imports.
Unresolved globals retain their runtime `NameError`. Cython 3.2.9 is the current
qualified producer choice for this path; broader host code exposed a Cython
3.3 compilation failure despite the earlier bounded fixtures passing. Package
initializers, aliases, damaged-cache recovery, lazy branches, source exclusion
and CPU/CUDA/Vulkan specialization/gradient fixtures are tested.

The independent `_host_wheel.py` producer packages a sealed layout with public
stubs, resources and standard Python metadata, without recompiling host modules:

```sh
python python/taichi_forge/tools/_host_wheel.py --spec release.json --output wheels
```

```json
{
  "version": 1,
  "host_layout": "host-layout",
  "project": "public/pyproject.toml",
  "files": {
    "simulation/solver.pyi": "public/solver.pyi",
    "simulation/py.typed": "public/py.typed",
    "simulation/settings.json": "public/settings.json"
  }
}
```

Paths are relative to the release specification; public files belong to compiled
host namespaces. Use the matching Python ABI/platform with PyPA `build`,
`packaging`, setuptools and, on Python 3.10, `tomli` installed. The project's
build requirements must already be installed; packaging is offline and does not
resolve application dependencies. `pyproject.toml` uses `setuptools.build_meta`
and static `[project]` metadata, including dependencies, extras and entry points.
Omit source-discovery `[tool.setuptools]` settings. Declared readme/license files
are copied from the project directory.

Setuptools generates native wheel tags, `METADATA`, `WHEEL` and `RECORD`; the
verified host manifest is retained in `.dist-info/forge-host.json`. Generated
build scripts and compiler caches stay outside the wheel. The tool checks every
payload hash and rejects undeclared files, source, destination collisions and
incompatible host ABIs/platforms. The public-file list cannot vendor Forge or
other native runtime libraries.

Output wheels are immutable. Identical inputs reproduce the same wheel; changed
public files require another version or output directory without rebuilding host
extensions. `SOURCE_DATE_EPOCH` is honored, with the earliest ZIP timestamp as the
default. An installed fixture verifies public resources, stubs, entry points,
aliases and two static sizes with forward/Tape gradients on CPU/CUDA/Vulkan.
A frozen 481-module GeoPhys closure passes installed public scene/runtime calls
for consumer-created two-sphere and three-box MJCF scenes on Windows CPU, CUDA
and Vulkan, with private source access denied and source/analytic references.
The installed closure also passes bounded CUDA SAP export and independent AOT
consumption. This does not qualify the complete GeoPhys public API,
all dependency profiles, solver branches or nonlinear solver gradients; those remain
separate application/deployment work.

For external dependencies, use standard resolution with the qualified producer
versions in `packaging/constraints/sdk-release.txt`. Resolve the selected SDK
wheel/extras and its Forge dependencies for the target Python/platform; installed
version constraints alone are not a complete lock. The following assumes
`requirements.in` describes that selected deployment and any local wheels are
made available through pip's normal index or `--find-links` options:

```sh
python -m pip lock --only-binary=:all: -r requirements.in -o pylock.online.toml
python -m pip download --only-binary=:all: -r pylock.online.toml -d downloads
python python/taichi_forge/tools/_wheelhouse.py --lock pylock.online.toml --wheels downloads --output offline
python -m pip install --no-index -r offline/pylock.toml
```

The sealer validates the standard pylock model, package/version filenames,
SHA256 and declared sizes, then copies only the selected wheels and emits
relative `wheels/` paths. It preserves lock metadata and publishes a new directory
only after all checks succeed; the input lock/cache remain unchanged. It does
not resolve versions, install packages, bundle a second runtime or import the
DSL. Source distributions and build-on-install inputs are excluded from this
wheel-only delivery path. Move the complete `offline` directory together.
pip26.2.1's lock/pylock support is experimental; keep the qualified tool version
and revalidate other Python/platform profiles. See the
[pip lock contract](https://pip.pypa.io/en/stable/cli/pip_lock/) and
[standard pylock paths](https://packaging.python.org/en/latest/specifications/pylock-toml/).

Windows CPython3.10 qualification resolves the declared external SDK/Forge
requirements into 44 wheels (166,914,963B), checks an offline installation, and
runs both public scenes on CPU/CUDA/Vulkan using those dependencies with the
staged Forge and installed compiled host. This is an external dependency profile,
not qualification of complete Forge/SDK wheel installation, every optional extra
or arbitrary solver/material input. Build tools remain producer-only; external
libraries remain shared dependencies rather than copies inside the host wheel.
