# Native raster and ray programs

[中文](native_ray_programs.zh.md) · [Hardware API](forge_api_reference.en.md)

These interfaces are available in 0.6.3. Use matching release
documentation and compatible shim/runtime packages. They do not require matching
Git commits. Vulkan examples have been exercised on Windows/NVIDIA; availability
on another device or platform is a capability question, not a performance promise.

## Choose the smallest appropriate interface

| Need | Public entry | Execution |
| --- | --- | --- |
| Triangle batch queries without custom shaders | `ti.hardware.ray.triangle_scene()` | Vulkan AS/ray query or CUDA/OptiX on the active backend |
| Query inside a Taichi Vulkan kernel | `ti.types.acceleration_structure()` and ray-query intrinsics | Inline ray query |
| Rasterize and query the same AS from a graphics shader | `ti.hardware.graphics.record_pass()` with an AS binding | Vulkan graphics pipeline; see the hardware API |
| User ray-generation, miss and triangle hit programs | `VulkanRayTracingPipeline` or `OptixProvider.program()` | Native Vulkan RT pipeline or OptiX pipeline |

The batch factory chooses the native route once, during creation. It does not
switch `ti.init()` backends, migrate arrays, enable experimental retained recipes,
or silently run a software renderer when hardware/dependencies are unavailable.
Existing backend-specific entry points remain available. Hardware use does not
guarantee lower total cost for a small workload.

```python
ti.init(arch=ti.vulkan)  # or ti.cuda with an available OptiX runtime
with ti.hardware.ray.triangle_scene(vertices, indices) as owner:
    owner.trace_typed(rays, hit_values, hit_indices)
    # Consume the outputs on-device, or read back after completion.
```

Geometry is compact `f32 (N, 3)` vertices and `i32 (M, 3)` indices, or the supported
AOS vector-3 equivalents. Ray/typed-hit shapes follow the existing batch API;
consult its reference before allocating outputs. The factory's `native_scene`
is the actual backend object, not a copy. Its batch/refit methods are the same
bound native methods; `refit()` returns that native scene. Close dependent Graphs
before closing the owner. A supplied `provider=` remains caller-owned; an OptiX
provider created by the factory is closed with it.

## Dependencies and availability

| Path | Forge provides | Application/environment provides |
| --- | --- | --- |
| Vulkan | Resource owners, RHI pipeline, SBT, recording and dependency handling | Vulkan driver/device with the required features; precompiled SPIR-V or an explicitly chosen compiler |
| CUDA/OptiX | Thin adapters, managed pipeline/SBT/launch, built-in query PTX | Compatible NVIDIA driver/OptiX vendor runtime; user PTX, or CUDA compiler plus compatible OptiX headers to build it |

No shader compiler is downloaded or loaded by these program constructors. The
portable wheel does not bundle the vendor runtime, CUDA Toolkit or CUDART.
User-produced PTX can require newer driver/PTX support than the built-in batch
programs; selecting an adapter does not translate unsupported PTX. Use
`provider_path=` for a **Forge adapter**, `library_path=` for the **vendor runtime**.
Neither is a CUDA installation directory.

Vulkan AS, inline query and RT-pipeline features are distinct. After Vulkan
initialization, `ray.is_available()` retains its existing batch/query meaning;
`ray.is_program_available()` checks the programmable RT path. For OptiX:

```python
provider = ti.hardware.ray.OptixProvider(required_features=("program", "instances"))
```

This filters adapters before creating a context. An explicitly selected adapter
that lacks a requested feature fails; it is not silently replaced. Omitting
`required_features` preserves ordinary batch selection. Passive
`ti.hardware.report()` describes known capability/provider state without loading
an optional library. Operation IDs `ray.program.vulkan` and `ray.program.optix`
distinguish these paths; an unloaded provider is not proof of unavailable hardware.

## Complete, runnable examples

Both examples perform line-of-sight queries and consume distances in a Taichi
kernel. They are useful without a renderer. Source export creates new files and
does not overwrite existing ones. Run from a shell where your chosen tools work.

Vulkan (PowerShell; `glslc` is supplied by you):

```powershell
python -m taichi_forge.examples.features.native_ray_program_vulkan --write-sources ray-shaders
glslc --target-env=vulkan1.2 -O ray-shaders/query.rgen -o ray-shaders/query.rgen.spv
glslc --target-env=vulkan1.2 -O ray-shaders/query.rmiss -o ray-shaders/query.rmiss.spv
glslc --target-env=vulkan1.2 -O ray-shaders/query.rchit -o ray-shaders/query.rchit.spv
python -m taichi_forge.examples.features.native_ray_program_vulkan --shader-dir ray-shaders --mode direct
python -m taichi_forge.examples.features.native_ray_program_vulkan --shader-dir ray-shaders --mode graph
```

OptiX (use your SDK include path and a compatible CUDA host-compiler environment):

```powershell
python -m taichi_forge.examples.features.native_ray_program_optix --write-source query.cu
nvcc --ptx --std=c++17 --gpu-architecture=compute_75 -I"C:/path/to/OptiX/include" query.cu -o query.ptx
python -m taichi_forge.examples.features.native_ray_program_optix --ptx query.ptx --mode direct
python -m taichi_forge.examples.features.native_ray_program_optix --ptx query.ptx --mode graph
```

Use `--provider` / `--library` only to override adapter/runtime discovery. Skip
source export and compilation when you already have compatible artifacts.
`compute_75` is the example target, not a guarantee for every PTX/compiler/driver
combination. The source files show the exact bindings and parameter ABI used.

## Resource, shader and SBT contracts

Programs are **trusted application code**, not sandboxed code. Forge validates
the declared layout, resources and supported contracts at preparation boundaries;
it cannot prove that arbitrary external shaders stay within those declarations.

- Vulkan `SpirvShader` stages are raygen, miss, closest-hit and any-hit; triangle
  groups use `VulkanHitGroup`. `VulkanRayBinding` declares binding, resource kind,
  access and storage-image mip selection. Match these to the shader; use actual typed
  resources, not raw device addresses.
- Buffer bindings accept supported dense storage/views; their dtype, byte range,
  element layout and shader interpretation must agree. There is no implicit
  packing or conversion. Vulkan images use managed `Texture` objects: sampled
  bindings use the Texture sampler/mip chain, while a storage-image binding
  selects one `mip_level`. Uniform, sampled-image and AS bindings are read-only.
  OptiX texture references use the managed CUDA texture's existing capabilities;
  a programmable pipeline does not add CUDA mip support by itself.
- OptiX `PtxModule`, `OptixShaderEntry`, `OptixHitGroup` and
  `OptixParameterLayout` describe modules, entry points and the launch ABI.
  Field offsets, alignment, scalar widths and total size must match the compiled
  parameter structure. Declare buffer access and bind scene/buffer references
  through the managed layout. A Python dictionary is not CUDA struct reflection.
- SBT record order is significant. Instance `sbt_record_offset`, geometry index,
  ray-type offset/stride, miss index and the shader's trace operands must select
  valid matching records. Defaults preserve the zero-offset single-ray-type
  case. Changing instance record mappings requires a new scene generation;
  position-only refit preserves mappings.
- Any-hit filtering requires compatible geometry/ray flags and hit records;
  opaque/disable-any-hit flags deliberately bypass it. The examples use opaque
  rays. Texture alpha rules, material dispatch and payload interpretation belong
  to the application.
- Current managed program paths cover triangles and the four listed stages.
  Callable/intersection/custom-AABB, motion blur and cross-backend shader
  translation are not supplied by this interface.

## Prepare once, execute, then close

Build resources and a program, call `program.record(...)`, then
`recording.prepare(bindings).initialize()`. Initialization explicitly prepares
device-side launch/SBT state outside repeated execution and Graph capture.
Use `launch.run()` directly, or append `launch.graph_recording()` to a
`GraphBuilder` before freezing it. The examples include a real device consumer
and close Graph users before launch, program, scene and provider owners.

For OptiX, preparation selects a direct native adapter call when the installed
shim supports it; older compatible shims retain the callback path. Graphs submit
this prepared call in runtime order alongside any captured CUDA segments. This
does not make the OptiX launch capturable or remove its submission cost. Execution
diagnostics distinguish the submission bridge from capture support. Closing the
launch or resetting the runtime retires its dependent Graphs and prepared calls.

Reuse the prepared launch with in-place data updates that preserve the declared
layout and resources. Replacing bindings, changing shader/layout/SBT structure,
resizing storage or resetting the runtime requires the corresponding new
preparation/generation. Keep borrowed arrays, textures and scenes alive until
their consumers complete. `run()` is submission, not a CPU-consumption fence;
wait at a real completion boundary. Do not add a global sync after every command.

| Execution form | What is reused | Important boundary |
| --- | --- | --- |
| Direct prepared launch | Program, SBT and fixed bindings | Native command submission still occurs |
| Ordinary Graph | Frozen order and retained owners | A native action may re-record on each execution |
| Vulkan fixed-frame recipe | Immutable command/argument frames where legal | Refit and other segments may still submit separately |
| OptiX program in a Graph | Prepared launch and runtime ordering | This path is **not CUDA Graph capture**; do not infer capture from a Graph label |

## Recipes, reports and diagnosis

Freeze the complete producer/query/consumer Graph before using
`definition.search_recipes(engine="compileiq", ...)`. Providers may compose map
fusion and immutable Vulkan frames when the actual region is legal. No alternative
region means no such candidate; the searcher does not invent another shader.
CompileIQ sees complete recipe identities and application evaluation costs, not
bare RT block/tile/vendor switches. Keep the baseline and measure a complete useful
window, including preparation amortization, packing, refit, consumers and completion.

If an immutable CUDA frame candidate is absent, inspect
`definition.recipe_catalog().discovery_report()["providers"]` and the binding-frame
provider's `provider_explanation`. Its `reasons` distinguish unsupported SNode/AS
bindings, native commands without a capture contract, and non-single-Graph
topology. These are candidate-specific admission reasons, not a statement that
CUDA capture is unavailable everywhere. In particular, the typed OptiX batch
query still uses its ordered native path; eligible surrounding kernel segments
can replay. Do not remove lifetime checks or change the baseline just to obtain
an immutable-frame candidate.

Reports preserve provider-declared code/layout/SBT/build and resource-plan facts
separately from measurements. Physical execution identity is not a live pointer
or a cold/warm allocation snapshot. Resource replacement can retain an equivalent
plan identity; shader/layout/SBT changes invalidate the old contract. Recreate an
equivalent definition and program in a new process, then use the public resolve
and applicability APIs described in [recipe integration](graph_recipe_integration.en.md).
Resolving a recipe is not proof that previous performance evidence still applies.

`launch.memory_report()` describes its stated ownership scope. AS storage,
borrowed outputs, temporary upload storage and opaque driver allocations may be
outside it; requested bytes are not total VRAM residency. Missing measurements
remain unknown. Native execution, capturability and being faster are separate facts.
For diagnosis, distinguish missing device feature, unloaded/missing adapter,
program compilation failure, invalid binding, ordinary/re-recorded execution and
a valid but slower candidate. Profiling is opt-in and is not a replay-time gate.
