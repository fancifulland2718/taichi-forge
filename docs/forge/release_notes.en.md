# Taichi Forge Release Notes

This is the canonical version index for Taichi Forge user-visible changes.
The declared package version is `0.5.0`; current `master` also contains the
explicitly separated Unreleased changes below. `0.4.25` is the final public
`0.4.x` baseline.

PyPI storage is limited, so some nonessential older distributions have been
removed. Absence from the current PyPI release list does not mean that a
version never existed. The source-boundary column below is the durable history
anchor; packaging-only, CI-only, test-only, and documentation-only commits are
grouped under the behavior they shipped.

## Quick index

| Version | History status | Source boundary | Main scope |
| --- | --- | --- | --- |
| [Unreleased](#unreleased) | current source after the published 0.5.0 runtime boundary | current `master` | driver-only CUDA primitives, bounded workspaces, and runtime safety |
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
| [0.5.0](#050) | published runtime source boundary | `95626e8036` | async runtime safety, Graph replay/lifetime work, Dense Field Graph |

## Unreleased

These changes are after the published 0.5.0 runtime source boundary and are not
retroactively attributed to the 0.5.0 artifact:

- Replaced automatic CUDA native-primitive dispatch with Forge-owned,
  driver-only providers for diagnostics, scan/reduce/histogram, composite
  primitives, and stable radix sorting. The standard runtime no longer links or
  bundles CUB/CUDART; explicit `cuda_cub*` methods are deprecated and isolated
  in a non-publishing Toolkit-reference workflow.
- Moved CUDA and Vulkan primitive resources into Program-owned arenas with
  bounded leases and explicit clear/statistics paths. Vulkan recycles completed
  descriptor/resource sets without queue-wide waits; CPU retains at most 8 MiB
  of primitive scratch per family/worker and uses bounded transient/fallback
  policies for larger requests.
- Added opt-in schema-v1 `get_primitive_runtime_diagnostics()` and
  `get_primitive_workspace_statistics()` snapshots. Provider dependencies,
  fallbacks, Program provider bytes, and per-Python-thread default caches are
  observable without a device synchronization. `workspace=None` caches default
  to 64 entries per context and 16 process-wide contexts; explicit clearing
  requires quiescent submissions.
- Changed CUDA scan to a 1,024-item tiled hierarchy, fused compact flag
  normalization with local ranks so only tile counts are scanned, and replaced
  one-bit stable sort passes with hierarchical 4-bit LSD radix passes. Windows
  million-item correctness, two-host-submitter stress, and idle-guarded
  reference comparisons are complete. Histogram and compact meet this
  iteration's gates; the remaining scan/reduce/sort gap to CUB is recorded as a
  future structural opportunity instead of adding device-specific branches for
  marginal tuning.
- Changed future standard runtime-wheel validation to the `driver-only`
  dependency class while retaining loader, repair, and validation compatibility
  for already-published 0.5.0 bundled-CUDART wheels. The project still publishes
  one runtime wheel per operating system, not per CUDA version.
- Completed the Windows driver-only/reference build and primitive correctness
  matrices. Linux wheel/import/dependency scans, compute-sanitizer, and execution
  on each claimed older NVIDIA driver remain required before lowering any
  published driver floor.
- Hardened debug execution and indexing contracts. CPU assertion failures now
  cooperatively cancel remaining debug work, publish one coherent first fault,
  and leave the worker pool reusable. Matrix/vector accesses validate and clamp
  each logical axis instead of accepting a linearly aliased component, while
  `assume_in_range` validates supported integer ranges without narrow-integer
  overflow. An explicit `check_out_of_bound=False` now overrides the implicit
  `debug=True` bounds default without disabling other debug behavior. Generated
  assertions remain unavailable on Vulkan; per-axis clamp behavior is supported.
- External PyTorch tensors no longer receive a full `zeros_like` gradient merely
  because a primal kernel sees `requires_grad=True`. Forge allocates the tensor-
  sized gradient lazily for reverse/forward AD, Tape, or an explicit in-kernel
  `.grad` access, and reuses an existing user gradient without replacement. A
  primal-only call therefore avoids one same-sized allocation per affected
  tensor while preserving the established AD paths.
- GFX kernels now track primal and gradient external-array access separately.
  Vulkan stages a host gradient into its own device buffer and reads it back
  only when the grad kernel writes it; device `ti.ndarray` gradients remain
  direct device allocations. Torch grad shape, dtype, contiguity, and device
  mismatches are rejected before launch instead of producing a false or unsafe
  gradient.
- Extended `ti.ad.FwdMode` parameter seeding from scalar fields to dense vector
  and matrix fields on CPU, CUDA, and Vulkan. Shaped seeds follow
  `field_shape + element_shape`; flat seeds use row-major order. The contract is
  layout-independent across AoS/SoA and retains the existing one-parameter-
  group boundary.
- Defined the automatic-differentiation order boundary explicitly. First-order
  Tape, manual reverse, and FwdMode paths are verified across CPU/CUDA/Vulkan;
  nested contexts, manual reverse inside Tape, and forward-on-reverse now fail
  before compilation/submission. Tape no longer runs adjoints after its body
  raises, and dynamic early-return control flow remains an explicit frontend
  rejection rather than an incomplete derivative.
- AOT module creation now enforces its actual same-target contract. Passing an
  `arch` different from the active `ti.init()` architecture raises before the
  backend builder is created instead of warning and silently changing the
  requested artifact target.
- CUDA LLVM AOT now compiles against an explicit, cache-keyed target capability
  (SM 60 / PTX 50 by default) instead of consulting the build GPU inside
  target-sensitive codegen. Artifacts record compute/PTX requirements in a
  sidecar, and the loader rejects an insufficient device before kernel
  registration. Newer exact targets are opt-in and add no Toolkit/CUDART
  runtime dependency. CUDA LLVM AOT artifacts made before this sidecar contract
  must be rebuilt.
- GFX AOT metadata now carries all dense root-buffer sizes, per-field tree ids,
  and per-kernel SNodeTree dependencies. The C API loader allocates every
  artifact root and registers kernels with the recorded count instead of
  hard-coding one tree. Non-contiguous live tree ids fail at build time;
  sparse SNode AOT remains unsupported.

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
- Added a Program-owned first-fault domain for context-fatal CUDA errors and
  Vulkan device loss. Kernel, Graph, ticket, synchronization, and GGUI paths
  now fail fast with the original cause; fault-aware reset/finalize avoids
  unsafe backend waits without claiming in-process device recovery.
- Added `ti.runtime.stats()`, `ti.runtime.capabilities()`, and the bounded
  `ti.runtime.trace()` context. Immutable Program-generation snapshots expose
  submission, synchronization, memory, transfer, Graph, display, fault, and
  trace data; unavailable optional measurements remain `None`, and trace
  export records bounded host events without pretending to be a GPU profiler.
- Extended runtime statistics to schema v2 with exact host allocator capacity,
  cursor consumption, alignment/unreclaimed-release waste, slab/large/exclusive
  chunk classes, and lifetime peaks. Windows reserve+commit is distinguished
  from Linux anonymous-mapping residency instead of fabricating committed
  bytes.
- Replaced the fixed 1 GiB host slab with a 16 MiB geometrically growing
  policy capped at the existing 1 GiB ceiling. Oversized individual requests
  keep request-sized mappings, while a per-chunk address index and newest-slab
  search avoid linear-scan regressions as the mapping count grows. An internal
  environment rollback can restore the legacy policy for release diagnosis.
  In controlled Windows fresh-process A/B, CPU/Vulkan initialization host
  commit fell from 1 GiB to 16 MiB; incremental private bytes fell by about
  97.4% on CPU and 86.8% on Vulkan. Ordinary kernel/Graph median changes stayed
  within 1.5% across CPU/CUDA/Vulkan; Linux measurement remains pending.
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
- Added immutable schema-v1 native-primitive capability descriptors and active
  Program provider resolution. Operand dtype/rank/layout/storage, backend
  methods, determinism/atomic ordering, AD, Graph/AOT, workspace, and fallback
  contracts now share the method/AD registry used by dispatch. FwdMode uses
  verified kernel fallbacks for transform, reduce-sum, gather, scatter, and
  scatter-add on CPU/CUDA/Vulkan; unsupported native, scan/grouped-reduce, and
  discrete automatic-AD paths reject before writing.
- Added device-resident consecutive run-length encode, unique, and
  unique-by-key primitives for integer keys on dense ndarray/field storage.
  Fixed-capacity `size=0` logical-empty input, device-side count, first-payload
  semantics, reusable `RunLengthWorkspace`, PrimitiveSequence Graph replay,
  alias/AD guards, StructNdarray payloads, and independent-workspace
  multithreaded submission are covered on CPU/CUDA/Vulkan. The implementation
  reuses existing compact providers and adds no runtime-wheel ABI dependency.
- Added reusable dense `SegmentedLayout` topology plus device-resident
  segmented sum reduce and inclusive/exclusive sum scan for scalar ndarray and
  root-dense field storage. Empty segments, fixed-capacity padding, stable
  serial floating order, grouped-ndarray reverse AD, Graph replay, independent
  workspace concurrency, scratch/topology accounting, and coarse
  backend-aware integer-scan dispatch are covered on CPU/CUDA/Vulkan. This
  composes existing providers and adds no runtime-wheel ABI dependency.
- Added production-shaped CPU/CUDA/Vulkan concurrency, numerical, lifetime,
  memory, and replay regression/benchmark coverage. Remaining Linux release
  evidence is tracked in [Linux revalidation](linux_revalidation.en.md).

Detailed current contracts live in:

- [Graph runtime and optimization](graph_runtime_optimization.en.md)
- [Dense Field Graph](dense_field_graph.en.md)
- [Native algorithms](native_algorithms.en.md)
- [Compilation trade-offs](compilation_tradeoffs.en.md)
- [Building wheels](build_wheels.en.md)
