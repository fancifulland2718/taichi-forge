# Taichi Forge release notes

[中文](release_notes.zh.md) · [Documentation](index.en.md)

This page summarizes user-visible changes and upgrade considerations.
Repository documentation describes its source revision; an unreleased feature
is not a claim about the wheel currently installed. Use the documentation at
your release tag for that version's exact contract. Older distribution files
may no longer be available from a package index.

## Quick index

| Version | Main additions |
| --- | --- |
| [Unreleased / 0.6.3](#unreleased) | Complete Graph recipes, reusable reports, hardware providers, typed resources and prepared operations |
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

## Unreleased

Development line: **0.6.3**. Availability remains specific to the installed
runtime, backend and optional provider.

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

### Hardware and data integration

- Public capability, provider status, execution and memory reports, with explicit
  operation-specific Graph/search support.
- Prepared and recorded matmul, sparse operations, FFT and contraction regions
  backed by user-provided libraries; optional Vulkan FFT/Parallel Sort and
  Toolkit source addons.
- Texture sampling/storage and Graph binding improvements, typed ray-hit outputs,
  dense-storage ray bindings and Vulkan acceleration-structure arguments.
- Managed texture mip/subresource views, raster device outputs/prepared draws and
  explicit Vulkan SPD plans.
- Prepared sort, compact/unique and operator plans for fixed-binding reuse.
- Device-resident cuSOLVERDn Cholesky and AmgX data paths within their documented
  dtype, layout and lifetime contracts.

### Fixes and upgrade notes

- Corrected Vulkan storage-texture writes and preservation of storage image
  formats, mixed Graph submission boundaries and texture transitions.
- Corrected retired Graph error handling, materialized-executor retirement,
  rejected-observation cleanup and reporting of linked graphics capability.
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
[API reference](forge_api_reference.en.md).

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
