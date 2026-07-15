# Taichi Forge Release Notes

This is the canonical version index for Taichi Forge user-visible changes.
The current source release is `0.5.0`; `0.4.25` is the final public
`0.4.x` baseline.

PyPI storage is limited, so some nonessential older distributions have been
removed. Absence from the current PyPI release list does not mean that a
version never existed. The source-boundary column below is the durable history
anchor; packaging-only, CI-only, test-only, and documentation-only commits are
grouped under the behavior they shipped.

## Quick index

| Version | History status | Source boundary | Main scope |
| --- | --- | --- | --- |
| [0.1.0](#010) | historical source release; artifact may be removed | `91ad177685` | scikit-build-core migration and Forge distribution rebrand |
| [0.1.1](#011) | historical source release; artifact may be removed | `c771969781` | `taichi_forge` import rename and install-layout fixes |
| [0.1.2](#012) | historical source release; artifact may be removed | `fe5844390b` | import fixes and CUDA build option |
| [0.1.3](#013) | retained on PyPI | `e87d42433` | LLVM 20 toolchain, Forge package identity, compile/cache controls |
| [0.2.4](#024) | retained on PyPI | `b42aca5d9` | compiler/cache expansion, SPIR-V parallelism, memory diagnostics |
| [0.3.0](#030) | historical; artifact may be removed | `e9056fa7c` | first Vulkan sparse/quantized development release |
| [0.3.1](#031) | historical; artifact may be removed | `166da399b` | inactive-read and allocator fixes |
| [0.3.2](#032) | historical; artifact may be removed | `fac7faad9` | deterministic pointer-slot activation |
| [0.3.4](#034) | historical; artifact may be removed | `769622584` | bitmasked/deactivation fixes |
| [0.3.5](#035) | historical; artifact may be removed | `6df723879` | list-generation and sparse-pool controls |
| [0.3.7](#037) | historical; artifact may be removed | `7d095f5d5` | conservative CUDA sparse-pool rollback |
| [0.3.9](#039) | retained on PyPI | `11b321dce` | Vulkan/CUDA sparse capacity policy |
| [0.3.11](#0311) | retained on PyPI | `da79573cf` | per-SNode CUDA pool sizing and diagnostics |
| [0.3.12](#0312) | retained on PyPI | `653eaf468` | sparse reuse, adaptive SPIR-V, GGUI/pipeline lifetime |
| [0.3.13](#0313) | retained on PyPI | `5e58c34b7` | experimental Hash SNode |
| [0.4.0](#040) | retained on PyPI | `1c788298d` | native algorithms, StructNdarray paths, offscreen Vulkan |
| [0.4.1](#041) | retained on PyPI | `0382b4d6b` | Graph modernization, PrimitiveSequence, DisplayFrame, compile profiling |
| [0.4.2](#042) | historical; artifact may be removed | `a1bac433b` | ArgPack, small integer, ndarray lifetime, hidden-window fixes |
| [0.4.23](#0423) | retained on PyPI | `1f36185c7` | split runtime/shim packaging, device checks, Vulkan fixes |
| [0.4.24](#0424) | retained on PyPI | `f8dfb3e2a` | GGUI device-image staging and render cadence |
| [0.4.25](#0425) | retained on PyPI; final public 0.4.x baseline | `7dad067ca` | GGUI event-pump and ImGui lifecycle fixes |
| [0.5.0](#050) | current source release | current `master` | async runtime safety, Graph replay/lifetime work, Dense Field Graph |

## 0.1.0

- Migrated the Python build to scikit-build-core and established the initial
  `taichi-forge` distribution identity.
- Began the Forge-specific build/toolchain and compiler-configuration line
  while retaining the upstream Taichi DSL model.

## 0.1.1

- Renamed the Python import tree from `taichi` to `taichi_forge`.
- Fixed scikit-build-core install paths, manifests, package data, examples,
  and internal imports for the new package identity.

## 0.1.2

- Fixed remaining Python import/rewrite issues.
- Exposed the CUDA compile option in the release build path.

## 0.1.3

- Established the `taichi-forge` distribution and `taichi_forge` import
  identity on the LLVM 20/scikit-build-core toolchain.
- Added the first compile profiling, cache warmup, compiler-tier, and
  backend-separated cache controls.
- Published the Python 3.10-3.14 Windows/Linux wheel line.

## 0.2.4

- Expanded per-kernel optimization levels, compile profiling, materialization
  fast paths, source/backend cache separation, and atomic cache writes.
- Added cached/parallel SPIR-V code generation and optimizer reuse while
  preventing nested compiler-pool oversubscription.
- Added memory-pool statistics, Vulkan buffer pooling, compiler telemetry, and
  updated MSVC/UTF-8/toolchain dependencies.

## 0.3.0

- Introduced experimental Vulkan `pointer`, `bitmasked`, and `dynamic`
  SNode support, including SPIR-V list generation and pointer allocation.
- Introduced the experimental Vulkan quantized-field gate. Unsupported
  quantized operations continued to reject rather than silently miscompile.

## 0.3.1

- Made inactive Vulkan pointer-cell reads return the dtype zero value through
  an ambient zone.
- Hardened pointer allocation, freelists, nested SNode list generation, and
  allocator metadata.

## 0.3.2

- Added deterministic-slot pointer activation to remove the full-activation
  CAS/spin device-loss path.
- Kept a documented fallback for layouts that cannot use deterministic slots.

## 0.3.4

- Added clear-on-deactivate behavior for bitmasked nodes.
- Fused two-level sparse deactivation and fixed index validation.

## 0.3.5

- Added intermediate-list-generation controls, ballot/grid-dimension
  improvements, and explicit CUDA sparse-pool tuning knobs.

## 0.3.7

- Reverted unsafe implicit CUDA sparse-pool auto-sizing and restored the
  conservative behavior while measurements continued.

## 0.3.9

- Used `vk_max_active` as an explicit capacity hint for Vulkan pointer SNodes
  and CUDA sparse-pool sizing.
- Completed the first broadly usable public Vulkan sparse-SNode line.

## 0.3.11

- Added per-SNode CUDA sparse-pool auto-sizing with `element_list` budget
  tracing and LLVM runtime diagnostics.

## 0.3.12

- Added deterministic CUDA pointer slots, fast reset, sparse-list reuse, and
  safer pool lifetime management.
- Improved Vulkan list-generation reuse, descriptor/resource caches,
  task-adaptive SPIR-V optimization, lazy submit, and runtime statistics.
- Made GGUI windows retire during reset and added pipeline-cache persistence.

## 0.3.13

- Added experimental fixed-capacity Hash SNodes on CPU, CUDA, and Vulkan.
- Added optional active lists, compact child pools, probe/list-generation
  telemetry, tests, and benchmarks.

## 0.4.0

- Added the stable Forge sort dispatcher plus CPU/CUDA/Vulkan sort, scan,
  compact, reduce, histogram, transform, gather, scatter, scatter-add,
  bucket-builder, and grouped-reduce paths.
- Added reusable native plans/workspaces, capability-based `method="auto"`
  fallback, multi-dtype support, and Vulkan shader implementations.
- Added StructNdarray opaque payload and scalar/tensor member-view paths.
- Added Vulkan offscreen support and Linux/GCC wheel-build fixes.

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

## 0.4.2

- Fixed ArgPack allocation lifetime, Vulkan small-integer fields,
  Vector/Matrix ndarray release, and the internal PrefixSum warning.
- Fixed hidden/offscreen GGUI window teardown and early Vulkan sparse-SNode
  inactive-read/full-activation failures.

## 0.4.23

- Split the platform-native runtime into `taichi-forge-runtime` while keeping
  a small per-CPython `taichi-forge` shim.
- Fixed repeated Vulkan ArgPack updates and dense CPU/CUDA native-field access
  after sparse SNode creation.
- Added device-side numeric checks/metrics and native Graph result nodes.
- Hardened Vulkan ArgPack mapping, small-integer SPIR-V, CUDART linkage,
  version propagation, and release workflows.
- Removed the abandoned `use_fused_passes` / `pipeline_dirty` experiment
  and retired the standalone Vulkan buffer-pool/listgen-barrier
  implementations after they showed negligible ROI. The latter fields
  remained accepted no-op compatibility names; the cache schema rejects
  artifacts from the transient fused-pass configuration.

## 0.4.24

- Packed common CUDA/Vulkan Field and ndarray images to RGBA8 on the device,
  and used a direct host path for contiguous `uint8` RGBA NumPy images.
- Reduced render-only frame overhead and corrected package/version metadata.

## 0.4.25

- Added `poll=False` to GGUI event-reading APIs and prevented redundant
  per-frame native-cursor updates, so `window.show()` can remain the only
  event pump in asynchronous render loops.
- Balanced empty ImGui frame lifecycles with `EndFrame()` and skipped
  unnecessary ImGui draw submission.

## 0.5.0

Only work after the `0.4.25` boundary belongs here. Native algorithms,
the original Graph modernization, `PrimitiveSequence`, DisplayFrame, compile
profiling, and GGUI device-image staging were already public by `0.4.25`.

- Externally synchronized Vulkan queue submit/present by actual queue handle,
  replaced queue-wide idle with submission-fence waits, and protected
  per-thread streams, profiler queries, descriptors, pipeline caches, and GFX
  recording state.
- Hardened CPU/CUDA/Vulkan runtime initialization, whole-kernel submission,
  allocation identity/generation/range validation, mapping/reset lifetimes,
  CUDA-Vulkan external-memory fallback, and CPU scheduler/native replay.
- Separated CUDA device capability from LLVM code-generation targets, isolated
  target-specific caches, removed the CUDA-13.2-only iterator dependency,
  hardened the single-runtime-wheel contract, and avoided unused CUDA
  void-kernel result allocations.
- Added safe CUDA Graph argument patching/capture recovery and Vulkan Graph
  identity, in-flight retirement, and fixed eight-slot replay fallback.
- Added stable `Graph.execution_stats()` diagnostics, strict runtime argument
  validation, mixed-segment isolation, Graph/reset/resource lifetime
  contracts, and opt-in `Graph.submit()` / `SubmissionTicket` completion
  tracking without changing the default `Graph.run()` hot path.
- Added Dense Field Graph for statically bound scalar/vector/matrix Fields,
  definition-time `template_args`, generation-qualified SNodeTree
  dependencies, zero-argument CUDA capture, explicit AD guards, and the
  block-level heterogeneous-environment model.
- Added production-shaped CPU/CUDA/Vulkan concurrency, numerical, lifetime,
  memory, and replay regression/benchmark coverage. Remaining Linux release
  evidence is tracked in [Linux revalidation](linux_revalidation.en.md).

Detailed current contracts live in:

- [Graph runtime and optimization](graph_runtime_optimization.en.md)
- [Dense Field Graph](dense_field_graph.en.md)
- [Compilation trade-offs](compilation_tradeoffs.en.md)
- [Building wheels](build_wheels.en.md)
