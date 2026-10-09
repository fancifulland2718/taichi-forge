# Taichi Forge release notes

[中文](release_notes.zh.md) · [Documentation](index.en.md)

This page summarizes user-visible changes and upgrade considerations.
Use the documentation at your release tag for version-specific API details.

## Quick index

| Version | Main additions |
| --- | --- |
| [0.6.4](#064) | In development; compilation improvements and miscellaneous backend/autodiff fixes |
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

## 0.6.4 (in development)

The release focuses on compilation improvements and miscellaneous fixes, with
changes in both areas already in source. These include lower compilation overhead
for large functions and fixes for backend correctness, autodiff and resource
lifetimes. The entries below describe the changes.

- Different optimization requests for one kernel may retain different Field
  dependencies, fixing an assertion when switching configurations. Definition
  retirement accumulates dependencies across variants; artifacts and Graphs keep
  their exact bindings. A previously field-free artifact no longer lets a changed
  request skip the SNode lifecycle lock.
- LLVM offline JIT caches store bitcode to reduce serialized size and read costs,
  while retaining legacy text payload reading. New writes use a separate cache
  schema. JIT and AOT share the module codec; AOT still exports text IR by default,
  with optimization tiers and numerical rules unchanged.
- LLVM AOT loading transfers ownership instead of making two intermediate module
  copies. Callables retain only their ABI, reducing load peaks and retained IR;
  the backend continues to own the executable modules and resources.
- LLVM and GFX AOT exports apply per-kernel optimization tiers and full-tier
  normalization through the same request resolver as JIT, including kernel
  templates. Equivalent resolved requests can share cached code; exported LLVM
  modules retain the selected backend options. JIT and AOT defaults are unchanged.
- Kernel cache lookups validate the effective configuration, device capabilities,
  ABI and optimization metadata, preventing cross-request reuse between JIT and
  AOT. Precompile key queries apply the same kernel-tier normalization as JIT.
  Context comparison avoids hashing unchanged requests; old compiled caches are
  invalidated. The Python shim and native runtime require matching ABI revision 11.
- Algebraic simplification maintains actual statement users across its rewrite
  iterations, building the index only when a reference replacement needs it.
  This reduces repeated IR scans without changing algebraic rules.
- Dead-instruction elimination propagates unused operand dependencies through
  a worklist and compacts affected blocks once, avoiding repeated whole-IR
  scans while retaining side effects, container operands and offload bounds.
- CFG forwarding and dead-store elimination share a maintained statement-use
  index, rewriting actual users instead of repeatedly traversing the IR.
  Alias, visibility and optimization-tier rules are unchanged.
- CFG load/store forwarding rejects unknown incoming values early and groups
  local tensor definitions by allocation, preserving fact order, alias checks
  and visibility while avoiding scans of unrelated definitions.
- CFG dead-store elimination checks local uses and full overwrites before
  querying successor liveness, avoiding redundant alias scans while preserving
  partial tensor writes and atomic return values.
- CFG dataflow analysis narrows direct local tensor alias candidates to their
  owning allocation, preserving dynamic-index checks and nonlocal fallbacks.
- CFG store/load forwarding indexes scalar local definitions to avoid repeated
  full-block scans in large generated functions, retaining
  control-flow and alias checks.
  Incoming definitions share one directory per CFG instead of rebuilding the
  full index at each node; nodes filter only the addresses they query.
- GPU LLVM options are retained per kernel through cached modules and
  delayed JIT registration, fixing CUDA kernel tiers being overridden by the
  Program tier. Old compiled caches are invalidated.
- Advanced load/address reuse now respects opaque calls and sparse-node lifetime
  changes, including conservative CFG handling of unmodeled effects and escaped
  local storage. Old compiled caches are invalidated.
- Reduce whole-kernel CSE reference-rewrite costs for large generated functions.
  Default optimization tiers and
  static-loop semantics remain unchanged; no size cutoff is introduced.
- SPIR-V redundancy elimination uses dominator scopes instead of per-block map
  copying, retaining the existing elimination rules and optimization levels.
- Fast LLVM compilation now eliminates duplicate typed expressions and lowered
  field-address calculations within basic blocks, including repeated sparse activation
  lookups. This reduces generated-code and driver compilation work without
  rewriting static loops or enabling the full advanced optimization pipeline.
  This pass is not enabled for SPIR-V backends.
  Field-load reuse also respects atomic and call side effects; older compiled
  artifacts are invalidated.
- Add opt-in `ti.init(inline_ir_cache=True)` for kernel-local reuse of eligible
  scalar `ti.func` IR on CPU, CUDA and Vulkan, expanding before autodiff and
  offloading. Static values and captures separate specializations; unsupported
  constructs keep ordinary expansion. It remains experimental and off by
  default; backend IR growth and driver compilation are not reduced by this cache.
- Reuse parsed AST copy layouts for repeated inline `ti.func` calls, reducing
  frontend preparation overhead while keeping each expansion's mutable state
  independent. Closure reads, static callbacks and per-call lowering retain
  their existing behavior; no function-size or call-count limit is introduced.
- Mask zero-tangent lanes before evaluating forward derivatives of `sqrt`,
  `asin`, `acos` and `rsqrt`, avoiding NaNs from inactive singular or overflowing
  derivative terms. Primal values and nonzero-tangent derivative rules stay
  unchanged; older compiled artifacts are invalidated.
- Restore computed Field values even when numerical gradient replays raise or
  the gradient comparison fails. Failed checks no longer leave perturbed inputs
  or outputs behind, and cleanup does not invoke user callbacks.
- Capture numerical gradient-check inputs when entering Tape, after clearing
  the loss. Successful checks restore the actual computed Field values without
  replaying user callbacks, preserving accumulated losses across repeated Tapes.
- Preserve vector/matrix shape and element type in forward derivatives of
  `tanh`, `sqrt`, `asin`, `acos` and `rsqrt`. These operations no longer fail
  compilation with a scalar/tensor type mismatch.
- Preserve keyword arguments for custom-gradient Tape calls, including bound
  methods and required keyword-only parameters. Reverse callbacks and numerical
  gradient checking replay the same arguments as the recorded primal call.
- Run the primal specialization inside custom/no-gradient scopes even when the
  same kernel was previously used in `FwdMode` or a validation Tape. Restore the
  enclosing mode after each suppressed call, including argument failures.
- Record only the outermost `grad_replaced`/`no_grad` call on a Tape. Nested
  decorators restore the enclosing suppression state, including when a body or
  recording step raises, preventing duplicate adjoints and leaked AD state.
- Keep floating-point dual Field dtypes when debug mode allocates integer
  adjoint checkbits, allowing forward AD and validation Tape on the same Fields.
- Avoid forward-AD NaNs from inactive runtime exponent tangents, including
  scalar arguments, derived expressions and Fields without dual storage.
  Runtime exponents zero and one use their boundary-safe base derivatives.
  The backend's primal `pow` domain is unchanged.
- Inherit the caller's CPU worker ID in `real_func` call contexts, including
  nested and recursive calls. This also preserves worker-local random-state
  selection without changing the RuntimeContext ABI.
- Preserve tangent storage aliases for dynamic local vector/matrix component
  writes in forward AD, including repeated updates and constant overwrites.
- Restore kernel modes in reverse call order when leaving `FwdMode` or a
  validation `Tape`. Repeated calls no longer leave ordinary kernels in AD mode;
  forward-mode restoration also runs if seed cleanup fails.
- Fix CUDA `real_func` Field access using the callee's return buffer as a root
  binding. Pass bindings separately through nested/recursive calls and include
  callee-only SNodeTrees in dependency collection for binding and lifetime
  validation. Invalidate old compiled artifacts; the RuntimeContext ABI stays
  unchanged.
- Fix forward-mode matrix construction using local tangent addresses as values,
  which caused zero-dimensional SOA matrix fields to fail an IR assertion.
  Load each component's tangent before constructing the matrix and invalidate
  previously compiled artifacts.
- Fix overlapping CUDA deterministic pointer slots when a pointer SNode has
  multiple children. Device metadata now uses the complete cell size for slot
  addressing and activation clearing; cached kernels with the old size are
  invalidated. Cover static/dynamic vector component access and reactivation.
- Fix forward-mode derivatives of constant powers at zero and negative bases
  under fast compilation. Zero-tangent logarithmic terms are not generated;
  constant exponents zero and one have explicit derivatives. Old compiled
  caches are invalidated so previously generated NaNs do not persist on upgrade.
- Reuse formatted source snippets across static-loop visits and copies of the
  same inlined function template. This reduces Python frontend compilation work
  while preserving source locations, caret formatting and specialization values.
  The cache retains source text rather than expanded AST nodes or runtime values.
- Rebuild blocks in order during AST lowering, and batch constant-fold/atomic
  replacements with indexed operand updates. Large static expansions no longer
  perform a full block scan and vector shift for every replacement in these
  passes. Immutable-local removal also compacts blocks once, and load-reuse
  searches index candidates by pointer identity while retaining store checks.
  Static-loop semantics and compile-tier defaults are unchanged.
- Warn once per kernel materialization about cumulative static source expansion,
  including inlined functions. `unrolling_kernel_warning_limit=1024` counts
  expanded source statements; `0` disables this diagnostic. It does not cap
  expansion or reject compilation. Explicit hard limits remain disabled by default.
- Open the 0.6.4 development cycle from the 0.6.3 baseline. Source version
  metadata and the default runtime dependency now target 0.6.4.
- Default Python `ti.init()` to `compile_tier="fast"` with
  `advanced_optimization=False`, including packaged installs. Explicit
  `balanced`/`full` initialization enables advanced IR optimization unless
  the caller sets that flag separately. Keyword options still override
  environment options. Existing compile-cache keys separate these settings.
- Create a fresh SPIR-V optimizer for each task. SPIRV-Tools consumes its pass
  list after a run; reusing the old optimizer silently skipped optimization for
  later tasks and could expose a Vulkan driver compiler crash with loop-carried
  struct cursors and nested vector branches. Existing pass options are preserved.
- Invalidate older compiled-kernel caches so affected shaders are regenerated.
  Explicitly disabled optimization and the fast compile tier retain their
  existing behavior; this repair does not qualify the unoptimized cursor path.
- Repair SPIR-V SIMT thread indices: `ti.simt.block.global_thread_idx()` now
  selects the correct backend and lowers the registered `vkGlobalThreadIdx`
  intrinsic; local thread IDs are included in the shader entry-point interface.
  Unsupported SPIR-V intrinsics fail during compilation instead of emitting
  invalid ID zero and reaching the driver.
- Keep scalarized local-vector pointer constants within their offloaded task.
  Dynamic vector indexing in consecutive tasks no longer shares shader-local
  SSA values across shaders, fixing Vulkan `query_value` compilation failures.
  Invalidate compiled-kernel caches to regenerate affected artifacts.
- Repair branch ownership when optimization merges adjacent `if` statements
  with complementary empty branches. Balanced compilation now passes IR
  validation for this case instead of retaining the erased statement as parent.
- Lower one-bit predicate AND, OR and XOR to SPIR-V logical instructions.
  Integer bitwise instructions with boolean operands produced invalid shaders
  and could crash SPIRV-Tools during balanced optimization.
- Restrict explicit SPIR-V array strides to buffer layouts. Function-local and
  ordinary Workgroup arrays no longer carry illegal `ArrayStride` decorations;
  buffer strides, including the physical i32 storage of bool arrays, remain intact.
- Preserve pointer return types during common subexpression elimination. Scalar
  and tensor pointers to the same address must stay distinct: merging them could
  reinterpret byte offsets as component indices and crash balanced compilation
  while lowering field accesses. Invalidate affected compiled-kernel caches.
- Compare both branch-presence flags when eliminating equivalent `if`
  statements. An extra nested `else` can no longer be silently erased by CSE.
  CPU, CUDA and Vulkan regressions cover direct calls and Graph execution;
  compiled-kernel cache schema 41 invalidates affected cached artifacts.
- Store reaching-definition and liveness facts in shared-index bitsets and
  cache definite-alias queries during CFG analysis. This reduces compilation
  memory and repeated set work without relaxing alias or multi-destination
  kill rules. The reported GeoPhys FEM balanced Vulkan fixture now completes
  on Windows with physical and deterministic-replay checks passing. This is
  bounded compiler validation, not a general application speedup claim.
- Balance IR verifier scopes and isolate offloaded tasks while preserving
  visibility from a task's prologues to its body and epilogues. Branch-local
  or cross-task SSA references now fail validation before code generation.
- Restore the aggregate C++ test build: update launch-policy arguments and
  include transitive native object libraries when linking split-runtime tests.
- Add OptiX candidate face filtering to typed and compact occlusion recordings:
  per-instance or per-primitive rules, shared GAS, alpha AND acceptance, transformed
  winding and device refit. Negotiate the new face-filter features before resource
  creation. Imported opacity micromaps remain an explicitly rejected combination.
  See [the interface and ownership contract](external_hardware_providers.en.md#optix-face-rules).
- This section does not announce a PyPI release. For installed releases, use
  the documentation at the corresponding release tag.

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
