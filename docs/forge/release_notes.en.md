# Taichi Forge release notes

[中文](release_notes.zh.md) · [Documentation](index.en.md)

This page summarizes user-visible changes and upgrade considerations.
Use the documentation at your release tag for version-specific API details.

## Quick index

| Version | Main additions |
| --- | --- |
| [0.6.4](#064) | Unreleased; compilation improvements, AOT coverage and backend/autodiff fixes |
| [0.6.3](#063) | Complete Graph recipes, hardware rendering, reusable operations and basic ROCm/HIP |
| [0.6.2](#062) | Execution plans, dynamic work, Graph storage and solver improvements |
| [0.6.1](#061) | Task policies/labels, device worklists, SNode lifecycle and Graph telemetry |
| [0.6.0](#060) | Structured Graph control, operators/solvers, driver-only CUDA primitives and interop |
| [0.5.0](#050) | Dense Field Graph, asynchronous runtime safety and completion tickets |
| [0.4.25](#0425) | GGUI event handling and frame lifecycle |
| [0.4.23](#0423) | Split runtime/shim packaging and device checks |
| [0.4.1](#041) | Graph/native replay, PrimitiveSequence, DisplayFrame and compile helpers |
| [0.4.0](#040) | Native algorithms and StructNdarray paths |
| [0.3.13](#0313) | Experimental Hash SNode |
| [0.3.0](#030)–[0.3.12](#0312) | Vulkan sparse/quantized support and sparse-runtime improvements |
| [0.2.4](#024) | Compile/cache and memory diagnostics |
| [0.1.0](#010)–[0.1.3](#013) | Forge package/import identity and toolchain |

<a id="unreleased"></a>
<a id="064"></a>

## 0.6.4 (unreleased)

This release focuses on compilation improvements and correctness fixes, with broader AOT
deployment coverage and shared JIT/AOT contracts. These notes describe the resulting behavior;
they do not indicate publication to PyPI or qualification of every backend/provider/layout
combination. Consult the linked API contracts and capability queries for exact support.

### Upgrading and compilation settings

- The frontend/native private ABI is **185**, persistent execution-plan version **26**, and
  compiled-cache schema **74**. Update compatible components together; matching Git commits
  are not required for ABI-compatible components. Intermediate development ABIs/plan formats
  have no compatibility bridge: re-export and validate deployment packages with matching tools.
  Old compiled caches are invalidated; distinguish the resulting cold start from warm replay.
- Vulkan domains can explicitly select `queue="graphics"` for instance-private offscreen
  color/depth drawing and compute consumers on one queue. Flat storage kernels and described
  low-level graphics passes can borrow owned Textures exclusively until instance retirement.
  Device-geometry RasterPass mesh/mesh_instance/particles/lines can also prepare fixed domain
  instances: they borrow the selected helper code and allocate private scratch. Preparation
  freezes camera/light data; later camera changes require a new preparation. Passes with AS
  resources still need separate integration.
  Completion GPU timing uses the
  `execution_envelope` scope, covering drawing and computation but excluding parameter uploads
  and external waits; see the [execution-domain contract](cuda_execution_domains.en.md).
- Vulkan AS build/refit barriers use shader-read access for geometry and instance inputs,
  covering transfer uploads and device transform packing. AS storage/scratch dependencies remain intact.
- Flat AOT plans on Vulkan domains can reconstruct private triangle scenes or TLAS resources
  with shared BLAS dependencies. Factories prepare resources; `initialize()` builds them on
  the selected domain, which also executes warm native refit/query commands. The owner closure
  includes every BLAS and seed input. DSL kernels in the same plan can query a private TLAS,
  sharing native initialization and ownership checks. Flat-plan ray programs can also use
  private TLAS resources and images. SBT uploads join selected-domain initialization, with
  retained upload storage counted in instance costs. Public live AS, micromaps and
  AS operations inside control flow remain outside this capability. Graphics domains can
  bind private TLAS resources in vertex/fragment shaders, sharing one submission with
  preceding device refit and subsequent computation. Preparation checks actual queue
  capabilities and the full AS closure. Cold data uploads and bootstrap
  still depend on the ordinary queue.
- Python `ti.init()`, including wheel installations, keeps `compile_tier="fast"` and
  `advanced_optimization=False` as defaults. Explicit `balanced/full` enables advanced IR
  optimization unless separately overridden. Kernel/export overrides use the shared request
  resolver. JIT may target the current device while AOT declares a deployment target;
  they need not use identical settings.
- CUDA requests freeze their effective target before compilation. A fixed AOT target's cache
  identity no longer reads the producer's implicit CUDA device/provider. Actual code-shaping
  target, option, launch-ABI and relocation differences still produce distinct artifacts.
- See [compile/cache](cache_compile.en.md) and [compilation trade-offs](compilation_tradeoffs.en.md).
  This release makes no universal compile-time or runtime speedup claim.

### Compilation cost and artifacts

- Ordinary fixed ranges can absorb proven-safe integer preludes, removing a serial task
  and cross-task temporaries. CUDA/Vulkan also admit dense scalar i32 Field reads when
  the loop cannot modify that source. CPU retains Field snapshots to avoid repeated
  address work. Field snapshots with external aliases or opaque operations remain in
  place. Autodiff does not apply this transform; no workload-size threshold is introduced.

- Type checks after index linearization are coalesced per block edit batch,
  avoiding repeated whole-block scans while preserving type promotion, debug
  checks and forward/adjoint behavior. Static expansion has no new size limit.
- Large static expansions and repeated inlining perform fewer AST copies, source formatting
  operations, per-statement rewrites and whole-block scans. Immutable preparation data is reused;
  each expansion retains independent mutable state. Excessive expansion produces a warning,
  without a new hard size limit.
- CFG analysis filters candidates by address class and definition index, shares numbered bitsets,
  and caches definite alias queries. CSE, load forwarding and reference replacement avoid repeated
  comparisons while respecting opaque calls, atomics and sparse-resource lifetimes.
- The LLVM fast path removes duplicate typed expressions/address calculations within basic blocks.
  SPIR-V redundancy elimination uses dominator scopes instead of copying maps per block.
  These changes do not enable the entire advanced optimization pipeline by default.
- JIT caches and AOT share compiled ABI/task/resource metadata and codecs, with separate containers,
  invalidation and deployment policies. Demand loading and fewer duplicate serializations,
  unused helpers and diagnostic artifacts reduce unnecessary work and storage.
- LLVM linking reuses immutable runtime/SNode bitcode and loads function bodies on demand,
  avoiding complete library clones for each kernel. Retaining these snapshots uses additional memory.
- CUDA prepares internal runtime modules by function group, separating base initialization from
  sparse functionality; using every group can increase cumulative cost. Workspace reuse, smaller
  initial host allocations and incremental native builds also reduce unnecessary allocation/build work.
- `inline_ir_cache=True` remains experimental and disabled by default. Reusing eligible frontend
  function IR does not eliminate backend expansion or driver compilation costs.

### Graph and runtime

- CUDA AOT preserves each entry's selected occupancy waves/work-per-thread launch policy.
  Ordinary execution and Graph recording share cold resolution on the consumer device
  instead of silently using the baseline grid. Different policies on shared code remain
  independent. Current ordinary LLVM artifacts use schema 11; aliases or selected
  device-range entries require schema 12 and are rejected by older readers.
- CUDA AOT can preserve explicitly selected device-range tasks in flat Graph recordings,
  including independent execution domains and explicit gradients. Ordinary and selected
  entries with identical effective code share one LLVM payload and one runtime registration;
  each prepared instance retains its own range controls and workspace policy. Direct kernel
  launch cannot bypass the selected recording policy. This remains an explicit choice;
  it adds a controller per selected task and does not imply a runtime speedup.
- Explicit CUDA device-range tasks can compose with a native WHILE's prefix, body and
  suffix in one root submission. JIT, AOT and independent instances share the range
  recorder. Repeated conditions retain separate controls for each physical capture
  site, without unrolling the iteration budget. Each selected task occurrence adds
  48 bytes and one controller; masked, branch, parallel and binding-frame combinations
  remain outside this qualification.

- Empty ordinary CUDA dynamic-range blocks exit before reduction setup and teardown, avoiding neutral
  global atomics without useful iterations. Partially active blocks still participate fully in
  reduction synchronization, and grid-stride iteration coverage is preserved.
- Shared frozen-Graph lowering preserves selected physical dispatches, launch policies and binding
  constraints. JIT code loans borrow compiled code; AOT adapters construct deployment closures.
  Ordinary call execution/submission paths remain in place.
- CPU/CUDA/Vulkan registration belongs to execution owners while compiled code can be shared.
  In-flight leases, independent instances, reset/close and process exit retire resources in dependency
  order, fixing borrowed-runtime and Vulkan process-exit access violations.
- CUDA domains count complete prepared instances: initialization and replay share one capacity
  reservation. The same owner cleans up partial recordings, so `instance_capacity=1` accepts
  a complete graph with initialization.
- Eligible CUDA prefix/conditional-WHILE/suffix regions are assembled in one root graph, including
  initial condition evaluation and counters. Other regions retain their existing route. Macro Graphs
  also accept host native actions: AmgX supports explicit host control and export/reconstruction,
  without claiming CUDA capture or asynchronous host-control submission.
- Impossible single-stage CUDA fusion recipes no longer retain template owners; valid multistage
  candidates retain required owners. Row-wise multidimensional accesses accept dynamic components
  while cross-row and dynamic leading indices remain subject to dependency checks.
- `ndrange` coordinate decoding no longer requires advanced CFG optimization for Graph recognition.
  Valid 2D stencil tiles remain eligible, and wrapped coordinates cannot enter shared staging.
  Ordinary/grouped coordinates and explicit reverse gradients have three-backend regressions.
- Fixes include host-action submission-lock self-waits, external CUDA calls without the owning
  context, binding errors and failed-transaction cleanup. Vendor errors retain library-specific
  names and numeric status codes.
- See [Graph execution](graph_runtime_optimization.en.md) and [recipe integration](graph_recipe_integration.en.md).

### AOT and native/provider integration

- Compiled artifacts are authoritative for calling ABI and resource dependencies. Native operations
  and resources use shared versioned descriptions, schemas, binders and validation. Ordinary creation,
  recording and export share semantic definitions while preserving workspace, handle, capture and
  ownership policies.
- CPU/CUDA/Vulkan coverage expands for direct/prepared kernel inputs and returns, dynamic composite
  values, named layouts, StructNdarray and explicit ndarray gradient bindings. Prepared return frames
  belong to bindings; `result()` explicitly waits, old tickets do not retain value snapshots, and
  Graph/ExecutionPlan dispatch returns can still be discarded.
- ArgPack supports direct kernels and declared execution-plan binding/reconstruction, including
  nested values, resource members and gradient source mappings. Ordinary JIT `Graph.run()` now
  accepts `ArgKind.ARGPACK`, member rebinding, different kernel argument positions and explicit
  ndarray adjoints. Value/ndarray packs can be frozen into independent CUDA domains and
  automatically exported as CPU/CUDA/Vulkan sequential execution plans through shared lowering,
  reusing native pack storage and recording. Consumers can create public owners by argument name
  with `ExecutionPlan.argpack()` and reuse them across kernel argument positions. Frozen export
  may share a dispatch with public packs, preserving resource and explicit gradient bindings. It
  shares equivalent storage for repeated uses of one source pack. Image members reuse image
  reconstruction, public bindings and alias preflight; compiler texture expressions no longer
  enter device-value uploads. Vulkan AS members reuse TLAS reconstruction/public bindings;
  unsupported backends and template routes diagnose nested member paths. Ordinary CGraphs
  gain v3 logical-member encoding and a named consumer pack factory; run/prepare reuse the
  kernel binder and validate the entire sequence before execution.
- Native sort/compact, scan/reduce, transform, gather/scatter and grouped/segmented recording/AOT
  routes support more layouts. Coverage remains specific to operation, backend, type and strategy.
  Packed Field/vector/matrix routes reuse existing cores, preserving stable order, payload bits and aliases.
- Supported cuBLAS, cuFFT, cuSPARSE, cuSOLVER, OptiX and other descriptions reconstruct provider
  resources while keeping shared plans, instance workspaces and retirement distinct. Vendor libraries
  remain separately installed dependencies, outside the application's `.tcm`; not every vendor method
  is automatically exportable.
- Independent CPU AOT consumers survive unrelated JIT resets; borrowed consumers retire with their
  owner. Repeated export does not mutate the producer, and shared code does not imply shared mutable state.
- Host-module build/cache and SDK packaging tools preserve public APIs, documentation and test entry
  points. Fixes cover source newline normalization and `__future__` import bindings. Excluding source
  files does not provide a guarantee against reverse engineering.
- See [AOT SDK](aot_sdk.en.md), [deployment profiles](aot_deployment_profiles.en.md),
  [native algorithms](native_algorithms.en.md) and [external providers](external_hardware_providers.en.md).

### Autodiff and backend correctness

- Vulkan C API image import translates the supplied `VkImageLayout` explicitly
  instead of casting it to a different internal enum. The first transition now
  preserves the external layout; layouts without an exact internal representation
  are rejected before allocating a view. Import does not transfer queue ownership
  or wait for an external producer.
- Fixes cover reverse-loop values/scopes, local struct initialization and loop ordering, plus forward
  tensor shapes, dynamic component writes and zero-tangent singularities. Zero/one exponents avoid
  unnecessary invalid derivative terms.
- Tape/FwdMode restore state after repeated calls and exceptions. Custom gradients retain keyword
  arguments without nested duplicate recording; numerical gradient checks restore values without
  extra user-callback replay.
- Discrete sort/compaction rejects unsupported automatic differentiation before writing output.
  Explicit gradients of downstream floating-point kernels remain available; this does not imply
  automatic adjoints for the native algorithms themselves.
- Vulkan fixes cover SIMT thread indices, cross-task SSA, boolean instructions/parameter layouts,
  local-array decorations, SharedArray floating CAS storage, unsigned negation and ray-loop SPIR-V
  structure. Optimizer passes run for each task, and aligned Field/view descriptors retain logical offsets.
- Vulkan emits conditional selection for branches that only assign private scalars. This avoids an
  NVIDIA pipeline-compiler crash with mutable Struct cursors and nested branches, including fast
  compilation and disabled optional optimization. Branch-local loads, arithmetic, atomics and
  shared-memory operations are not speculated.
- CUDA `real_func` root bindings/dependencies, CPU worker-ID propagation and deterministic pointer
  SNode child slots are corrected. Advanced optimization preserves pointer-type and branch distinctions.
- OptiX typed-hit/compact-occlusion traversal filters one-/two-sided candidates per instance/primitive,
  with alpha AND, shared GAS and device refit. Opacity-micromap combinations are explicitly rejected.
  Exact interval endpoints retain floating-point caveats: see [AOT SDK](aot_sdk.en.md) and
  [face rules](external_hardware_providers.en.md#optix-face-rules).

### Execution domains, windows and qualification

- Independent CUDA and fixed-compute Vulkan domains accept explicitly declared `u64` parameters frozen per accepted submission.
  Upload and Graph execution use that domain; busy returns before upload. Domains can preallocate
  timing events for sparse `Completion.gpu_timing` samples. The compute interval excludes predecessor
  waits, parameter uploads and external acquire/release, and includes internal Graph/provider gaps.
  Unsampled, pending, unsupported and failed states are explicit, never a fabricated zero duration.
- [Completion sets](completion_wait_sets.en.md) provide cancellable, timed wait-any for ordinary
  CPU/CUDA/Vulkan JIT/AOT tickets and CUDA/Vulkan domain tickets without taking runtime submission locks.
  Polling uses a configurable interval, without per-ticket threads or extra GPU nodes.
- Vulkan JIT/AOT fixed buffer-only recipes now use additional device-created compute queues,
  exact timeline dependencies, per-submit parameters, sparse GPU timing and independent instance
  retirement. Queue/timeline availability is checked on the chosen device. Selected device while,
  instance-private images and read-only sampling of same-Program managed Textures are also supported.
  Shared Textures currently require flat dispatches and one mip/layer; writes and layout changes are
  blocked until the last reader retires. Graphics/parallel recipes retain their ordinary routes.
  Qualification includes source-free AOT,
  actual explicit gradients and controlled independent progress on Windows RTX 5090.
  See [execution-domain contracts](cuda_execution_domains.en.md).
- Vulkan Canvas supports selected output domains with cold-prepared packed copies or local writable
  display slots. Graph/AOT writers and graphics readers share exact completion/retirement contracts.
  Writable slots avoid the extra display copy with caller-configured cold capacity; mixed geometry,
  GUI widgets use cold font preparation and exact reader tracking; runtime offscreen paths
  remain separately restricted. Vulkan submission also retains
  in-flight resources safely when host allocation fails around native enqueue.
- Python/native `Window.get_scroll_delta()` returns horizontal/vertical floating increments and clears
  them on read, without extra event polling or GPU waits. Vulkan resize/close no longer frees signal
  semaphores still referenced by the presentation queue.
- Current closeout checks cover representative Windows CPython 3.10 CPU/CUDA/Vulkan cases. Domain evidence
  also includes two instances with 1000 parameter triples each, controlled independent progress and
  timing-event reuse. Window tests inject OS messages, rather than qualify physical wheel interaction.
  Linux, other Python wheels and the complete release platform matrix require separate qualification.
- GeoPhys module/geometry/small deployment checks do not qualify the whole application, every solver/scene
  or application frame rates. Downstream OptiX integration and joint optimization are outside this closeout.


<a id="063"></a>

## 0.6.3

### Graph search and reusable execution

- Public `freeze → search_recipes → materialize` workflow for complete Graph
  execution plans with the maintained CompileIQ fork.
- External recipe providers, staged search, explicit budgets and repeated
  evaluation, checkpoint continuation and cross-process selection resolution.
- JSON and Markdown reports with measurements, failures, Pareto trade-offs,
  selection reasons and reuse context. Search does not change runtime defaults.
- Stable execution identity separated from live resource instances and memory
  observations; selection reuse does not serialize Python executables or AOT binaries.
- Broader eligible template/dense-Field recipes and clearer candidate-generation
  and execution-path diagnostics.
- Independent mixed-control materialization and transactional close/reset
  lifetimes; zero-task CUDA dispatches can be captured as no-ops.
- Optional compressed nested CUDA conditional recipes for supported control
  shapes, alongside the existing expanded alternatives.
- Public build-time execution selection and binding-admission diagnostics;
  opt-in GPU stage timing for supported cached Graph paths. Timing scope and
  unavailable measurements remain explicit rather than inferred from replay labels.

### Hardware and data integration

- Standard Windows/Linux runtime wheels include the optional basic ROCm/HIP
  backend (`ti.amdgpu`). Users supply HIP, its driver and linker. Default backend
  selection is unchanged; advanced Graph/rendering features are not included.
  See the [ROCm guide](rocm_backend.en.md) for requirements and build inputs.
- Public capability, provider status, execution and memory reports, with explicit
  operation-specific Graph/search support.
- Prepared and recorded matmul, sparse operations, FFT and contraction regions
  backed by user-provided libraries; optional Vulkan FFT/Parallel Sort and
  Toolkit source addons.
- Texture sampling/storage and Graph binding improvements, typed ray-hit outputs,
  dense-storage ray bindings and Vulkan acceleration-structure arguments.
- Managed texture mip/subresource views, raster device outputs/prepared draws and
  explicit Vulkan SPD plans.
- Vulkan gradient/mip sampling, anisotropy and LOD controls; depth comparison
  sampling and depth-only passes for shadow-map consumers.
- Vulkan graphics supports multiple color attachments, per-attachment blend
  equations and depth state, including floating-point and integer targets.
  The convenience `RasterPass` remains RGBA8; use the low-level graphics API for HDR.
- Native Vulkan/OptiX ray programs, compact occlusion queries, alpha-mask
  filtering and imported opacity micromaps within documented device/provider
  limits. These do not supply an application renderer or transparency algorithm.
- Public GPU display-target borrowing and opt-in consumer completion support
  bounded asynchronous source reuse. Existing Graph-to-Canvas device ordering
  remains distinct from permission to overwrite an in-flight source.
- Prepared sort, compact/unique and operator plans for fixed-binding reuse.
- Device-resident cuSOLVERDn Cholesky and AmgX data paths within their documented
  dtype, layout and lifetime contracts.

### Fixes and upgrade notes

- Updated HIP ABI handling, Windows AMDGPU builds/binary linking and separate
  CUDA/AMDGPU allocation ownership. Inclusion in the wheel does not mean every
  AMD GPU/driver combination has been qualified on hardware.
- Corrected Vulkan storage-texture writes and preservation of storage image
  formats, mixed Graph submission boundaries and texture transitions.
- Corrected retired Graph error handling, materialized-executor retirement,
  rejected-observation cleanup and reporting of linked graphics capability.
- Corrected lock ordering for texture creation versus Graph submission and
  concurrent compute/display recording; minimized windows continue pumping
  events. Completion tracking does not require a default global synchronization.
- Reduced repeated binding/preparation work and improved control/solver execution;
  actual benefit remains workload-dependent.
- Deferred CPU argument-upload storage is reported separately from device memory.
- Use a compatible runtime/shim pair; equal source commits are not required.
  Complete-recipe search requires the maintained CompileIQ fork, not the base package.
- Old physical identity schemas or changed provider/evaluation contracts can
  require renewed measurements. Rebuild equivalent definitions and check
  applicability instead of copying live execution handles.
- Optional vendor libraries remain user-managed. OptiX uses Forge adapters/PTX
  with the application's compatible driver/vendor runtime. Installing a library
  does not automatically change an algorithm.
- Unsupported capture, dtype/layout or numerical combinations must not be
  treated as supported because a capability probe or ordinary kernel succeeded.

Usage: [Graph execution](graph_runtime_optimization.en.md),
[recipe integration](graph_recipe_integration.en.md),
[hardware/providers](external_hardware_providers.en.md) and
[API reference](forge_api_reference.en.md). For rendering, see
[native ray programs](native_ray_programs.en.md) and
[display ownership/completion](display_frame.en.md).

## 0.6.2

- Expanded Graph-private storage, bounded/ordered physical dispatch and execution
  plans, active worklists and deterministic reduction choices.
- Improved reusable dense SNode executables, Graph replay/bindings, workspace
  ownership and nested telemetry.
- Expanded LinearOperator/SolvePlan composition, direct Field use and
  device-convergent execution for supported providers.
- Isolated the runtime/shim native link boundary and improved wheel compatibility.
- Added experimental limited MUSA support; this is not general CUDA parity.

Check operation-specific provider tables when upgrading; a backend being
available does not imply every solver, graph or hardware operation is supported.

## 0.6.1

- Added task manifests, explicit task-launch policies and dispatch labels for
  correlation with Graph/kernel diagnostics.
- Expanded device-resident worklists, bounded and nested Graph execution and
  submission telemetry.
- Improved SNode creation/destruction, dense binding reuse and sparse runtime
  lifetime behavior.
- Expanded solver-plan/recordable operator composition, direct Field bindings
  and workspace-lane submission.
- Improved native CUDA primitives and runtime/JIT resource handling.

## 0.6.0

- Added structured Graph while/if/switch, bounded nested control, explicit
  telemetry and Vulkan device-written indirect dispatch.
- Added runtime-bound LinearOperator, experimental SolvePlan/batch plans and
  provider-specific Krylov solvers with fixed-pattern value updates.
- Added managed dense storage views, DLPack/external-allocation interoperability
  and CUDA–Vulkan shared display paths.
- Standard CUDA primitives use driver-only providers; ordinary execution does
  not require CUB/CUDART or a local Toolkit.
- Improved cache coordination, runtime lifetimes, numeric/AD handling and UI layout.

When upgrading, use public `method="auto"` or documented explicit methods,
not development-only `cuda_cub*` references. Query backend capabilities for
indirect/control features and recheck numerical tolerances. Runtime and shim
distribution versions must be compatible; commit equality is not the criterion.

## 0.5.0

- Added dense scalar/vector/matrix Field Graph bindings and definition-time
  `template_args`.
- Hardened asynchronous compute/display submission, backend failure handling,
  graph/resource lifetimes and reset behavior.
- Added public runtime statistics/trace and Graph execution diagnostics,
  completion tickets and strict runtime argument contracts.
- Added native capability descriptors, consecutive RLE/unique and reusable
  segmented reduction/scan layouts.
- Reduced retained runtime memory for small applications.

Native algorithms, the original Graph modernization, PrimitiveSequence,
DisplayFrame and compile profiling were already available in earlier releases;
they were not introduced in 0.5.0.

## 0.4.25

- Added `poll=False` to GGUI event-reading APIs and prevented redundant
  per-frame native-cursor updates, so `window.show()` can remain the only
  event pump in asynchronous render loops.
- Balanced empty ImGui frame lifecycles with `EndFrame()` and skipped
  unnecessary ImGui draw submission.


## 0.4.24

- Packed common CUDA/Vulkan Field and ndarray images to RGBA8 on the device,
  and used a direct host path for contiguous `uint8` RGBA NumPy images.
- Reduced render-only frame overhead and corrected package/version metadata.


## 0.4.23

- Split the platform-native runtime into `taichi-forge-runtime` while keeping
  a small per-CPython `taichi-forge` shim.
- Fixed repeated Vulkan ArgPack updates and dense CPU/CUDA native-field access
  after sparse SNode creation.
- Added device-side numeric checks/metrics and native Graph result nodes.
- Hardened Vulkan ArgPack mapping, small-integer SPIR-V, CUDART linkage,
  version propagation, and release workflows.
- Retired obsolete compile/runtime switches; consult the options guide when migrating old configuration.


## 0.4.2

- Fixed ArgPack allocation lifetime, Vulkan small-integer fields,
  Vector/Matrix ndarray release, and the internal PrefixSum warning.
- Fixed hidden/offscreen GGUI window teardown and early Vulkan sparse-SNode
  inactive-read/full-activation failures.


## 0.4.1

- Added `ti.compile_kernels()`, `ti.parallel_compile()`, expanded
  `ti.compile_profile()`, compile tiers, and offline-cache sharding/locking.
- Modernized Graph execution below the existing GraphBuilder/CGraph API and
  added Forge native replay nodes and `PrimitiveSequence`.
- Added `ti.ui.DisplayFrame`, `Canvas.submit_frame()`, display statistics,
  direct packed-u32 Vulkan rendering, texture upload, and bounded in-flight
  frame handling.
- Optimized native primitive plans, workspace reuse, dense-field routes, and
  GGUI staging.


## 0.4.0

- Added the stable Forge sort dispatcher plus CPU/CUDA/Vulkan sort, scan,
  compact, reduce, histogram, transform, gather, scatter, scatter-add,
  bucket-builder, and grouped-reduce paths.
- Added reusable native plans/workspaces, capability-based `method="auto"`
  fallback, multi-dtype support, and Vulkan shader implementations.
- Added StructNdarray opaque payload and scalar/tensor member-view paths.
- Added Vulkan offscreen support and Linux/GCC wheel-build fixes.


## 0.3.13

- Added experimental fixed-capacity Hash SNodes on CPU, CUDA, and Vulkan.
- Added optional active lists, compact child pools, probe/list-generation
  telemetry, tests, and benchmarks.


## 0.3.12

- Added deterministic CUDA pointer slots, fast reset, sparse-list reuse, and
  safer pool lifetime management.
- Improved Vulkan list-generation reuse, descriptor/resource caches,
  task-adaptive SPIR-V optimization, lazy submit, and runtime statistics.
- Made GGUI windows retire during reset and added pipeline-cache persistence.


## 0.3.11

- Added per-SNode CUDA sparse-pool auto-sizing with `element_list` budget
  tracing and LLVM runtime diagnostics.


## 0.3.9

- Used `vk_max_active` as an explicit capacity hint for Vulkan pointer SNodes
  and CUDA sparse-pool sizing.
- Completed the first broadly usable public Vulkan sparse-SNode line.


## 0.3.7

- Reverted unsafe implicit CUDA sparse-pool auto-sizing and restored the
  conservative behavior while measurements continued.


## 0.3.5

- Added intermediate-list-generation controls, ballot/grid-dimension
  improvements, and explicit CUDA sparse-pool tuning knobs.


## 0.3.4

- Added clear-on-deactivate behavior for bitmasked nodes.
- Fused two-level sparse deactivation and fixed index validation.


## 0.3.2

- Added deterministic-slot pointer activation to remove the full-activation
  CAS/spin device-loss path.
- Kept a documented fallback for layouts that cannot use deterministic slots.


## 0.3.1

- Made inactive Vulkan pointer-cell reads return the dtype zero value through
  an ambient zone.
- Hardened pointer allocation, freelists, nested SNode list generation, and
  allocator metadata.


## 0.3.0

- Introduced experimental Vulkan `pointer`, `bitmasked`, and `dynamic`
  SNode support, including SPIR-V list generation and pointer allocation.
- Introduced the experimental Vulkan quantized-field gate. Unsupported
  quantized operations continued to reject rather than silently miscompile.


## 0.2.4

- Expanded per-kernel optimization levels, compile profiling, materialization
  fast paths, source/backend cache separation, and atomic cache writes.
- Added cached/parallel SPIR-V code generation and optimizer reuse while
  preventing nested compiler-pool oversubscription.
- Added memory-pool statistics, Vulkan buffer pooling, compiler telemetry, and
  updated MSVC/UTF-8/toolchain dependencies.


## 0.1.3

- Established the `taichi-forge` distribution and `taichi_forge` import
  identity on the LLVM 20/scikit-build-core toolchain.
- Added the first compile profiling, cache warmup, compiler-tier, and
  backend-separated cache controls.
- Published the Python 3.10-3.14 Windows/Linux wheel line.


## 0.1.2

- Fixed remaining Python import/rewrite issues.
- Exposed the CUDA compile option in the release build path.


## 0.1.1

- Renamed the Python import tree from `taichi` to `taichi_forge`.
- Fixed scikit-build-core install paths, manifests, package data, examples,
  and internal imports for the new package identity.


## 0.1.0

- Migrated the Python build to scikit-build-core and established the initial
  `taichi-forge` distribution identity.
- Began the Forge-specific build/toolchain and compiler-configuration line
  while retaining the upstream Taichi DSL model.
