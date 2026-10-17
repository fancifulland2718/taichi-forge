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
workspace, which does not include image residency. Ordinary AOT kernel texture
arguments remain unsupported; alpha recordings can export through the explicit
image reconstruction contract below.

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
binds `initial_alpha` instead of a texture handle. It is copied once during cold
preparation; later seed-array writes do not implicitly upload the image. Native
`image/upload` operations can explicitly update it in an authored execution plan.
Public names and fixed private Graph bindings referring to the same Texture share
one reconstructed image per preparation. UV and face-table bindings remain live.
Missing/unused sources, retired textures and unsupported texture uses fail before
publishing kernels. Ordinary kernel Texture arguments still require separate
generation-qualified launch support.

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
descriptions. A v16 consumer and matching ABI76 components are required.
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
This requires a v14 consumer and matching private ABI76 components.
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
format requires a v15 consumer and matching ABI76 components.
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
