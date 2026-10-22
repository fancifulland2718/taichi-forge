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

- Macro Graph sequences accept host native actions, including host-only bodies
  in explicit portable control. Host loops remain exact, without speculative
  calls after convergence. AmgX host branches export and reconstruct with
  path-sensitive retained-guess validation. Windows CPU/CUDA/Vulkan small Graph
  checks and real CUDA AmgX source-free/borrowed branch tests pass. CUDA capture,
  and asynchronous host-control submission are not added. Exact portable CUDA
  WHILE now exports through execution plan version 26, including AmgX host
  operations and explicit reverse kernels. Pair frontend/native ABI 144.

- AmgX device bindings expose an explicit host-controlled Graph action with
  automatic AOT reconstruction from frozen configuration and CSR topology.
  Consumers provide initial coefficients; each instance starts a new solver
  session. Version 2 solve descriptions preserve per-binding initial-guess state
  on shared solvers. Windows CUDA f32/f64, source-free/borrowed consumers, updates,
  independent retirement and explicit-adjoint regression pass. This is not CUDA
  capture; vendor dependencies remain external, with no ABI/module-format change.

- cuSPARSE triangular recording export supports public shared plans with an
  explicit public matrix binding. Version 2 update/solve descriptions check
  plan parameters, matrix identity and dense-port storage during preparation,
  then reuse existing refresh/solve paths. Public plans use `mutable_values=True`
  and require a runtime supporting these v2 descriptions; private ABI and module
  format are unchanged. SpSV/SpSM, invalid bindings and retirement pass on Windows CUDA.

- Retained Graphs from `DeterministicScatterReducePlan` can export to AOT,
  preserving the selected single dispatch, fixed group count and actual
  permutation/offset data. External ndarray bindings and explicit Field state
  snapshots are supported. Scalar, vector, empty-topology, repeated export and
  independent consumption checks pass on Windows CPU/CUDA/Vulkan.

- Normal Python exit now uses the resource cleanup order of explicit reset,
  retiring Graph owners, borrowed AOT runtimes and the Program before backend
  libraries unload. This fixes a Windows Vulkan ray borrower exit access
  violation with validation enabled, without an application-side `ti.reset()`.
  Pending submissions, borrowed resources and repeated cleanup are verified on
  Windows CPU/CUDA/Vulkan.

- Direct prepared AOT kernels on CPU/CUDA/Vulkan can return values. Each binding
  owns an independent result frame; `run/submit` stay asynchronous and `result()`
  explicitly waits for the latest successful submission. Older tickets do not
  snapshot values. Composite decoding reuses compiled layouts; void preparations
  add no result allocation or readback. Prepared CGraph/ExecutionPlan dispatches
  discard returns using shared scratch per sequential result layout, with
  separate storage for concurrent CUDA branches and independent bindings.
  Initialization, conditional/nested control, explicit gradients and in-flight
  retirement are qualified on Windows CPU/CUDA/Vulkan without per-dispatch readback.

- CUDA JIT and AOT consumers prepare the internal runtime by functional group:
  core entries at initialization, sparse allocation and statistics on first
  use. Groups share the existing executor state and module ownership; user
  kernels keep their compilation path. This reduces startup work for consumers
  that do not need every group. Using all groups can increase total preparation
  time and code size, especially with an already populated driver cache.
  CPU and Vulkan retain their existing runtime preparation paths.

- Nested ArgPack structures use pack setters recursively and share the
  compiler's value-type lowering, fixing compound writes and Vulkan nested
  vector/matrix layout mismatches. Ordinary AOT export and template families
  now synthesize fully declared packs or accept exemplars for resource
  specialization. Independent immutable AOT ArgPack owners now support ordinary
  CPU/CUDA/Vulkan calls and no-return kernel preparation through a shared
  C/Python member factory, including explicit ndarray gradients and borrowed
  JIT runtimes. Prepared calls retain root owners and Vulkan uniform-read
  dependencies. Flat execution plans accept external packs through the same
  logical member binding. Persistent plans reconstruct values through the shared
  `memory/argpack` registry, with compiled producer layouts, existing private
  arrays/adjoints and CUDA binding-family ownership. Explicit conditional and
  nested plans support immutable values and lifted arrays; selected Vulkan
  recordings share uniform binding and coalesce read dependencies at preparation.
  Suffix adjoints and retirement are qualified. Automatic symbolic Graph pack
  export and mutable JIT pack Graph bindings remain pending.
  Repeated resource member names across packs retain distinct bindings in
  declaration order, fixing frontend `NoneType` compilation errors for nested
  packs and their explicit adjoints.
  AOT exports now preserve explicit public member paths across template removal
  and resource lifting, share layouts across aliases, and validate these paths
  against loaded kernel parameters. This metadata adds no device code;
  ordinary and prepared kernel consumers reuse that mapping.
  Reusing a pack with kernels that have different numbers of other root
  arguments remaps lifted resources and explicit gradients through the target
  origins. Changed resource slots no longer invalidate unchanged pack storage.

- Vulkan loads boolean vector and nested structure arguments using their
  integer buffer slots before converting them to boolean expressions. This
  fixes invalid SPIR-V access-chain types in JIT/AOT without changing layouts.

- Reverse autodiff distinguishes generated structure-argument initialization
  from user loops, fixing false mixed-loop errors for nested structure/matrix
  parameters while preserving reverse loop order. Treat structure parameters
  as constants across loops; create a local copy inside a loop before modifying
  members to avoid incorrect gradients from cross-loop state. Ordinary forward
  copy semantics are unchanged. Update the runtime and frontend together.

- Public AOT kernel calls accept dynamically sized vector/matrix and nested
  structure values through the shared compiled layout, for ordinary and
  prepared execution on CPU/CUDA/Vulkan. Python consumers use sequences and
  mappings without NumPy or the frontend. The C descriptor copies payloads
  during binding; prepared replay retains its native storage. ArgPack resources,
  ordinary CGraph input types and prepared returns retain their existing scope.

- AOT Graph export reconstructs `VulkanBufferCommandRecording` through the
  common native registry, preserving fill/copy offsets and explicit barriers.
  Independent and borrowed consumers retain the existing resource lifecycle;
  dense Field/view snapshots and downstream kernel gradients are covered.
  Buffer preparation checks physical aliases and address overflow, and removes
  the arbitrary 4096-command limit. CPU/CUDA do not advertise this Vulkan route.

- CUDA profiler synchronization and toolkit management activate Forge's CUDA
  context and restore the caller's context. Clearing or querying profiling
  records no longer requires a Forge context to remain current on the thread.

- Remove an unused internal sparse allocator entry from the LLVM runtime.
  CUDA startup no longer compiles and loads this redundant device kernel;
  sparse allocation continues to use the existing adaptive allocator.

- Legacy cuSOLVER, cuSPARSE, cuBLAS and cuFFT wrappers report their own status
  names and numeric codes instead of interpreting them as CUDA driver errors.
  Unknown statuses preserve the original number; reporting does not retry the
  failed operation or change solver behavior.
- External CUDA calls through the shared submission boundary activate Forge's
  context and restore the caller's context on success or failure. OptiX cold
  initialization, pipeline preparation, queries and retirement no longer rely
  on the caller already having the correct context. Windows CUDA checks cover
  empty/foreign contexts, ordinary/instanced scenes, Graphs, typed hits,
  occlusion, miss writes and retirement; complementary CPU/Vulkan checks pass.
  Update runtime/frontend together to private ABI126; AOT plan25/root2 and the
  provider C ABI are unchanged.

- `Graph.submit()` supports synchronous host operations between device segments
  without waiting on its own submission reader. Vulkan submits and waits for
  the pending batch prefix while preserving the outer batch for later work.
  Windows CPU/CUDA/Vulkan checks cover host sorting, parent Graphs, timing,
  failure recovery and completion/retirement; native batch fence reuse remains.
  The private ABI and AOT format are unchanged.

- Ordinary `sort()` / single-key `sort_by_key()` with scalar Field keys and
  packed vector/matrix Field payloads share prepared native sorting on
  CPU/CUDA/Vulkan, eliminating the previous host sorting fallback. AOT exports
  the same records without extra helper code. Installed cold/warm consumers,
  stable ordering and payload bits, downstream gradients, retirement and Vulkan
  synchronization validation pass; the ABI125 runtime is reused.

- AOT `PrimitiveSequence.grouped_reduce()` exports packed vector/matrix Fields
  on CPU/CUDA/Vulkan by preserving the existing zero-fill and scatter-add stages.
  Replays overwrite empty groups and ignore invalid keys. It reuses ABI125
  runtime components without new helpers or format changes; normal installations,
  downstream gradients and Vulkan synchronization validation pass.

- AOT scatter-add supports packed vector/matrix Fields on CPU/CUDA/Vulkan,
  with scalar i32 Field or ndarray indices. It preserves accumulation into
  existing values, mutable indices and the JIT algorithms without packing
  helpers or an artificial component limit. Paired private ABI125 components
  are required; installed consumers, explicit downstream gradients and Vulkan
  synchronization validation pass.

- AOT native sum/min/max reductions support whole packed vector/matrix Fields
  on CPU/CUDA/Vulkan. They preserve per-component algorithms and share scratch
  across components without new application shaders or an artificial component
  limit. Paired private ABI124 components are required; normal installations,
  explicit downstream gradients and Vulkan synchronization validation pass.

- Incremental native builds retain an unchanged runtime anchor and use Clang
  depfiles for LLVM runtime bitcode. Reconfiguring a wheel build no longer
  recompiles those unchanged inputs; transitive header changes still rebuild.
  Qualified on Windows/Ninja. This does not change application kernel options.

- AOT modules can publish named structure layouts without helper kernels used
  only for layout queries. The catalog deduplicates types, loads on demand and
  uses existing module handles and retirement. Requires matching private ABI116
  components; the plan format is unchanged.

- The lightweight AOT consumer accepts fixed-AoS arrays through module-owned
  layout queries. Ordinary and prepared kernel/Graph calls and explicit plans
  share the existing native binding contract, including downstream float AD;
  no DSL/NumPy import or new native ABI is required.

- AOT native transforms accept packed vector/matrix Fields on CPU/CUDA/Vulkan,
  preserving module storage and the selected CPU policy, single flat CUDA
  launch and packed Vulkan dispatch. No packing kernels or copies are added;
  matching private ABI 115 components are required.

- Host producer scripts retain LF checkout bytes across Git newline settings,
  preventing unchanged source-hidden modules from recompiling solely because
  the checkout uses `core.autocrlf`. Exact toolchain and artifact checks remain.

- Graph submission telemetry accepts single-element scalar i32 control arrays
  at any index rank, matching ordinary structured execution. Enabling telemetry
  no longer rejects valid `(1,)` or `(1, 1)` predicate/counter arrays.

- Host pools size their first reusable slab from the allocation request and
  alignment, reducing idle reservation for small independent runtimes while
  retaining geometric growth and request-sized large allocations.

- CPU AOT runtime memory is isolated from unrelated JIT resets, fixing an access
  violation when an independent consumer continued after `ti.reset()`.
  Borrowed consumers retain their existing owner retirement contract.

- Prepared sorting, prefix sorting and compaction now reject automatic reverse
  and forward AD before writing outputs, closing a path that silently omitted
  gradients. Primal execution and explicitly differentiated downstream kernels remain supported.

- Full-range `prepare_sort()` now supports CPU through the shared AOT stable-sort
  core, retaining lazy workspace across calls. CPU/CUDA/Vulkan ordinary mixed
  scalar Field/ndarray sorting uses the same native route and exports cold or
  warm to AOT. Sorting remains discrete; downstream compiled gradients are supported.

- CPU stable payload sorting computes floating-point ordering keys once per input,
  reducing repeated NaN and signed-zero checks while preserving ordering and payload
  bits. The additional keys are included in workspace accounting. CPU finalization
  also retires allocation records before releasing the owning runtime's host pool.

- Fix invalid pointer types in Vulkan `SharedArray` floating-point CAS fallback.
  Each allocation uses consistent storage for atomics and ordinary loads/stores,
  while other allocations retain native operations; f64 CAS min/max also use the
  correct result type. Shared integer/float atomic features are reported, enabled
  and checked independently of buffer features, including supported shared int64
  atomics. AOT targets declare them through `spirv_has_shared_atomic_*` capabilities.

- Fix access violations when a ray-query kernel runs through JIT before being
  recompiled for an AOT target. Frontend expressions requiring one-time
  materialization now retain results per lowering pass, without reusing IR
  pointers destroyed after an earlier compilation.

- Vulkan closest-ray queries emit `OpLoopMerge` after evaluating the traversal
  condition, immediately before the branch. This fixes SPIR-V structural
  validation for full-hit and occlusion-only results while preserving traversal
  semantics and result-buffer contracts.

- CPU/CUDA/Vulkan AOT transform sequences support packed vector/matrix ndarrays
  through the shared `builtin/transform` descriptor. Tensor shape, prefix ranges
  and exact in-place aliases are checked during preparation. CPU retains its
  tensor partition policy, CUDA retains per-lane strided launches on the owner's
  stream, and Vulkan retains its packed dispatch. No application shader or scratch
  is added; downstream vector-kernel gradients remain usable. This requires the
  matched private ABI 108 runtime/frontend components.

- CPU/CUDA/Vulkan `PrimitiveSequence.segmented_scan` exports its serial or integer
  global-scan strategy, inclusive/exclusive mode, exact in-place bindings and
  strategy-specific padding behavior. Consumers retain immutable host/device
  topology, private bases and dense Field snapshots without the frontend. Shared
  workspaces preserve each call's selection; downstream DSL gradients remain usable
  (the scan itself has no automatic adjoint). Disjoint segment correction now uses
  ordinary stores, avoiding unnecessary atomics and the 64-bit atomic capability.
  Vulkan scan also aligns input-view and hierarchical-scratch descriptors to the
  device's storage-buffer requirement without copying the input.

- Vulkan compiles unsigned integer negation with modular arithmetic, including
  negation introduced by subtraction simplification. This fixes `u32/u64`
  kernels and unsigned segmented-scan correction helpers failing in SPIR-V codegen.

- CPU/CUDA/Vulkan segmented-reduce sequences export published immutable layouts
  generated on the device. Cold export captures only the selected offsets or
  segment IDs at the shared Field/provider completion boundary, once per layout.
  Rejected structures perform no readback; consumers do not read topology back.

- Vulkan scalar transform/gather/scatter fusion exports cold or preserves its
  selected strategy in AOT. Consumers share JIT's single-dispatch core and all five
  bindings, without additional application shaders or scratch. Coverage includes
  standalone/borrowed consumers, offset views, retirement and downstream gradients.

- CPU/CUDA/Vulkan native affine transforms route packed vector/matrix ndarrays
  through the existing per-lane core, fixing rejected payloads or partial writes.
  Ordered PrimitiveSequence and Graph replay preserve all lanes; mismatched
  element shapes are rejected before dispatch.

- Vulkan `PrimitiveSequence` transform/gather/scatter fusion preserves all
  intermediate outputs and zero values from out-of-range gathers. Appending an
  operation invalidates the old plan. Read/write aliases retain ordered execution
  to avoid applying in-place transforms twice; eligible fusion uses one dispatch.

- Vulkan native/AOT sorting accepts tensor payloads at scalar-aligned Field
  offsets. It no longer requires the offset to be a multiple of the complete
  record size; existing descriptor-alignment staging and range checks remain.

- CUDA/Vulkan `prepare_sort()` accepts packed vector/matrix payloads, including
  dense Fields and offset ndarray views. Direct execution, Graph recording and
  cold/warm AOT export share the existing raw-record sort and storage ownership;
  payload bits survive unchanged, without extra AOT helper kernels. The separate
  `count=` prefix-sort route and CPU prepared-sort coverage are unchanged.

- Repeated `SortWorkspace.reserve()` calls no longer retain unused request
  dictionaries. Native multi-key sorting can reuse its workspace without growing
  a host-side reservation history; lazy allocation and caller capacity checks stay
  unchanged.

- Native multi-key sorting now handles packed vector/matrix ndarray payloads
  on CPU/CUDA/Vulkan, including Graph replay. Whole-record gathers preserve
  NaN payloads and signed zero, reuse temporary storage and avoid component-wise
  launches. CPU requires explicit `cpu_native` for this native route.

- Single-key native sort sequences export packed vector/matrix ndarray payloads
  on CPU/CUDA/Vulkan without prewarming or extra helper code. Payload bits,
  including NaNs and signed zero, remain intact. Standalone/borrowed consumers
  and downstream kernel gradients are covered; prepared sort retains its separate
  contract.

- AOT native sorts share scratch for matching extents, key types and payload
  layouts within each ordered replay region. CPU/CUDA/Vulkan retain separate
  storage across recursive regions, initialization and independent binding frames,
  with resource accounting and retirement preserved. No artifact-format change
  or new input-size limit is required.

- Native multi-key `PrimitiveSequence.sort_by_key()` exports packed scalar ndarray
  keys and optional scalar/vector/matrix payload to CPU/CUDA/Vulkan AOT, retaining
  stable lexicographic order and scalar affine semantics. Tensor payloads use
  whole-record gathers and one identity index array initialized at preparation,
  preserving raw bits without additional helper code. CPU requires explicit
  `cpu_native`; cold export avoids producer prewarming and shares reorder arrays
  by dtype and element shape.
  Independent runtime consumption, executor borrowing and downstream kernel
  gradients are covered; sorting itself has no automatic adjoint.

- CPU AOT scalar sorting shares JIT's stable key/value sorting core, preserving
  duplicate-key, NaN and signed-zero ordering. Prepared owners reuse the primary
  array and avoid indirect index sorting and extra copies. The standard library
  stable sort may still allocate a transient merge buffer; this is not a claim
  of allocation-free execution.

- Vulkan cached kernels bind only roots actually referenced by their shaders,
  without requiring unrelated producer Fields. This fixes cached ndarray helpers
  and kernels loaded with fewer unrelated Fields, while preserving missing-resource
  checks, AOT local root mappings and existing retirement contracts.

- Explicit AOT `builtin/bucket_builder` v1 supports CPU/CUDA/Vulkan packed scalar
  and tensor payloads, offset views and empty inputs, reusing backend cores with
  scratch owned by each prepared execution. Standalone runtime consumption and
  executor borrowing are supported. `PrimitiveSequence.bucket_builder()` also
  exports packed scalar/tensor ndarrays and scalar Field snapshots without
  prewarming; downstream kernel adjoints work, but bucket adjoints are not automatic.
- Vulkan bucket builders share preparation and recording, fix dense Field
  descriptor alignment, and retain explicit logical extents and strides.
  Packed offsets use buffer fills; private counting skips unused cursor writes,
  and prefix scans move strategy branches outside the loop. Fixed native
  operations own their scratch storage.
- CUDA bucket builders combine the leading-offset and histogram clears and
  scan only bucket counts, reducing redundant launches and boundary scratch.
  Payload addressing no longer has the obsolete flattened-word count limit;
  actual count and address-range checks remain.
- CPU bucket builders reuse one count/cursor table, halving parallel scratch
  storage while preserving bucket order and payload bits. Workspace clear and
  statistics include the retained CPU table.
- TCM archives use fast deflate to reduce code and frozen data size. Existing
  native readers decompress entries on demand; payload contents, device storage
  and warm execution remain unchanged. Directory exports retain raw files.
- Segmented reduction sequences now export their original grouped/serial
  strategy on CPU/CUDA/Vulkan. Frozen topology, CPU partition order, padded
  inputs, serial Field snapshots and explicit adjoint composition are retained;
  unused topology arrays are omitted from each strategy's closure.
- Selected CUDA Graph segmented reductions now export their actual helper
  kernels and live topology/scratch arrays. Serial, warp, block and partial
  strategies preserve i32/u32 sums and empty-segment writes in standalone AOT
  consumers, without unused per-item IDs or a new automatic AD route.
- CPU jobs wake workers once when becoming active, according to their useful
  parallelism. Queued submissions and worker joins no longer issue redundant
  wakeups. Full-pool jobs and shutdown retain broadcasts; task partitioning,
  completion and exception handling are unchanged.
- Packed scalar grouped sum exports through native sequences or
  `builtin/grouped_reduce` v1 on CPU/CUDA/Vulkan, including empty input, mutable
  keys, snapshots and explicit adjoint composition. CPU shares owner/partial
  management with scatter-add; GPU preserves direct atomic execution without
  substituting for Vulkan bucket strategies.
- Vulkan grouped sum resets contiguous outputs with an exact-range buffer fill,
  eliminating one zeroing pipeline and compute dispatch, plus six unused embedded
  shader variants and their build targets. Strided outputs retain element-wise
  zeroing; empty inputs overwrite outputs, preserving Graph and supported gradients.
- CUDA grouped-sum aliases for segmented/two-level execution no longer allocate
  three unused bucket buffers. They retain their existing zero-plus-atomic
  computation, cached Graph replay and supported gradients.
- Vulkan native grouped reduction now binds aligned descriptor views for dense
  fields and passes explicit scalar offsets and lengths to existing shaders.
  This fixes validation errors while preserving neighboring fields, Graph
  replay and the supported reverse-mode gradient paths.
- Native check/metric sequences and selected result nodes now export through the
  shared registry on CPU/CUDA/Vulkan. Preparation does not execute diagnostics;
  private scalars can be reconstructed and consumed directly by later kernels.
  Count/NaN semantics, offsets and empty-prefix behavior are retained. This route
  covers packed scalars, with f32 metrics on Vulkan and unchanged automatic AD.
- CPU checks and metrics share packed/strided reduction cores and the existing
  scheduler. Predicate selection occurs outside counting loops. Wide internal
  counts avoid signed overflow; an unrepresentable i32 result raises on exact
  host reads while retaining zero/nonzero truth, without an input-size cap.
- Native device checks correctly classify unsigned indices against negative
  upper bounds on CPU/CUDA/Vulkan. CUDA zero/nonzero checks also treat `-0.0`
  as zero for both f32 and f64 while retaining NaN/Inf classification.
- Histogram now exports through the common native AOT registry on CPU/CUDA/Vulkan:
  packed i32/u32 IDs, i32/i64 counters, fixed prefixes and empty-input clearing.
  Prepared owners retain scratch and freeze algorithm preferences; replay neither
  reallocates partials nor probes the environment. Pure-native archives reuse the
  installed runtime's code. This discrete operation adds no automatic derivative.
- Vulkan histogram shares preparation across ndarray and dense fields while
  retaining its direct and private/shared strategies. Aligned descriptor views
  use explicit logical counts and offsets, fixing field alignment errors without
  counting adjacent storage. Private counting also removes an unused output
  descriptor; replay keeps each Graph's parameters and exact partial range.
- Packed scalar scatter-add now exports through the common AOT native registry
  on CPU/CUDA/Vulkan, including duplicate targets, mutable indices and retained
  destination values. CPU replay shares the runtime scheduler and retains private
  partials; GPU paths require no scratch. Vulkan checks actual type-specific
  atomic support. This route supports explicit adjoints, not automatic generation.
- Vulkan scatter-add also shares preparation across scalar, strided and packed
  field layouts. Field-backed indices use aligned views with residual offsets,
  fixing descriptor validation errors while retaining duplicate-target atomic
  accumulation and immutable Graph parameters without extra scratch.
- Vulkan gather/scatter now share preparation across ndarray and field layouts,
  including field-backed indices. Source, index and destination descriptors use
  aligned enclosing views, fixing validation errors for nonaligned field offsets
  without extra copies or scratch storage.
- Fix unaligned root-field descriptors in Vulkan native gather/scatter, including
  real Tape backward execution. Packed fields use the offset-aware shader when
  their starts do not meet the device's descriptor alignment.
- Vulkan strided gather/scatter now pass dispatch parameters as push constants,
  removing a 28-byte parameter buffer, seven fills and its barrier from each
  recording. Graph replay retains its own parameter values.
- Fix Vulkan native transform Tape accumulation for packed gradient fields whose
  starts are not descriptor-aligned. Dense scatter-add uses its existing offset-aware
  shader, avoiding a validation error that numerical checks alone did not detect.
- Fix Vulkan compilation of dynamic range starts with ndarray-shape ends. Dynamic
  starts retain temporary-storage snapshots, preserving device-memory bound semantics.
- Vulkan reads by-value scalar range ends directly from launch arguments, removing
  serial tasks used only to stage those arguments and their temporary-buffer round trips.
  JIT, Graph and newly exported AOT share this path. Complex scalar expressions,
  argpacks and device-memory bounds retain their existing paths; CPU/CUDA do not
  enable this rematerialization.
- CPU/CUDA SNode trees retain their creation-time element-list storage contract.
  Memory accounting and retirement no longer depend on later changes to
  `demote_dense_struct_fors`. Paired components use private ABI 98.
- Fix CUDA pool ownership for multiple non-demoted dense list trees. The
  default automatic pool path assigns independent pools even without a
  NodeManager, preventing a second tree from reusing a sealed global pool.
- Independent automatic CUDA pools for purely dense traversal now budget list
  capacity, metadata and alignment without a repeated 32 MiB runtime baseline.
  An eight-element fixture reduces pool reservation from 33,582,464 B to 48,512 B.
  Explicit floors/shared-pool choices remain; capacity and warm-path work are unchanged.
- CPU/CUDA non-demoted dense AOT declares exact task/tree list requirements
  through `list_v1`, with per-tree allocation, relocation and dense initial
  state. This fixes missing storage when consumer settings differ. CPU bitcode
  and CUDA PTX checks cover ordinary/fixed calls, actual field gradients and lifetimes.
- AOT ordinary CGraphs and execution plans share exports with the same compiled
  kernel identity and launch ABI, reducing duplicate LLVM/PTX/cubin payloads
  and Vulkan aliases. Distinct specializations and explicit public aliases are
  preserved; conflicting overwrites are rejected. Source Graphs and consumer
  formats are unchanged. Paired components use private ABI 97.
- CUDA sparse list reuse now latches its rebuild decision in the serial clear
  phase, preventing one generation block from publishing completion early.
  Ordinary and hash lists keep the same kernel count and execution order.
  CUDA compilation caches with reuse enabled are regenerated; existing AOT
  artifacts retain their original generated behavior.
- Added resource relocation for CPU/CUDA pointer, bitmasked and dynamic AOT
  trees with dense-loop demotion enabled (`sparse_v1` on CPU, `sparse_v2` on CUDA). Independent module
  instances, sparse-field gradients and close/reload are tested with CPU
  bitcode and CUDA bitcode/PTX/cubin. New exports support sequential prepared
  batches/CUDA fixed Graphs with coherent ordinary-call interleaving. Sparse
  parallel branches are rejected; root snapshots and hash/mesh/quantized layouts
  remain separate. Old artifacts keep their contract. Matching components use private ABI 96.
- Fixed tree-ID collisions when JIT fields are created after AOT modules load
  on a borrowed CPU/CUDA/Vulkan runtime. Both owners now share cold resource
  reservations, including compile-only JIT layouts; kernel launch paths are
  unchanged. Matching producer/runtime components require private ABI 95.
- Fixed CPU/CUDA AOT sparse/list tree retirement so a closed module can reload
  in the same runtime. JIT and AOT share list/allocator/root cleanup; live JIT
  trees remain intact. Simultaneous legacy tree-ID collisions remain rejected.
- Added AOT OptiX programs and prepared launch resources using the existing
  programmable adapter extension and shared ownership. Explicit plans rebuild
  address-free parameters/SBT once and replay on the ordered CUDA path; instance
  refits, indirect buffers/textures and downstream DSL gradients are qualified.
  Ordinary Graph export shares program/scene/image ownership and repeated launch
  packets. Explicit immutable PTX assets use the common shader-asset binding
  protocol, without adding JIT PTX residency. Instance-scene v2 preserves nonzero
  SBT offsets through reconstruction/refit, including fixed-query compatibility;
  baked OMM resources and GAS v2 also reconstruct fixed/program queries, alpha
  fallback and refit without new JIT asset residency. Face filtering with OMM
  remains unsupported; qualified contracts are in the AOT SDK guide.
- Added native AOT Vulkan compute pipeline/dispatch resources for fixed SPIR-V
  with scalar buffer descriptors. Cold code/layout/range checks and prepared
  replay share the existing resource owner and graphics descriptor validation;
  independent/borrowed consumers and downstream gradients are qualified.
- Vulkan RasterPass lines now export through shared native compute expansion and
  graphics. Persistent expansion storage is reused across replay; indexed and
  non-indexed offsets/counts, per-vertex color and incomplete tails retain JIT
  behavior, and empty draws omit the expansion resources.
- Vulkan RasterPass mesh/mesh_instance/particles export from explicit ordinary
  Graphs through shared preparation kernels and native graphics passes. Camera
  state requires an explicit snapshot; fixed-layout geometry/transforms can change,
  and attachments can be public or rebuilt. Empty draws only clear, with no geometry
  preparation. Lines use the shared compute reconstruction path described above.
- RasterPass recordings retain their effective default ambient light, so replay
  no longer inherits ambient changes from later recordings of the same pass.
- The shared Vulkan mesh shader skips transform-buffer reads when instance
  transforms are disabled, preventing uninitialized NaNs from contaminating
  vertices and nonzero instance offsets from accessing missing transforms.
- Fixed Vulkan RasterPass line expansion ignoring requested draw counts. Indexed
  and non-indexed intervals/offsets are honored; zero complete lines skip expansion
  allocation and drawing.
- GGUI Vulkan passes reuse the shared graphics barriers for vertex/index reads
  after compute expansion. Managed offscreen color attachments retain their
  attachment final layout, matching Program layout tracking.
- Vulkan AOT registers `spd/plan` and `spd/run` with frozen code, fixed image
  identity checks, complete mip initialization and independent counters. Consumers
  need no SPD SDK/compiler. Small standalone/borrowed tests cover repeated values,
  close and downstream gradients; SPD itself is not claimed differentiable.
  Ordinary Graphs and existing selected whole-Graph recipes export SPD recordings,
  sharing frozen code/plans across repeated calls and preserving public image aliases.
- Fixed the missing Vulkan `depth16` texel mapping for JIT/AOT image allocation
  and `u16` uploads. Shared depth transfer validation checks the final four-byte
  buffer offset, including finite subviews and construction-time AOT pixels.
- Vulkan AOT explicit plans reconstruct region image copy, blit and buffer/image
  transfers through the common native registry. Shared range checks preserve finite
  views and padding; tests cover mip/depth updates, retirement and downstream
  gradients. Ordinary Graph export reconstructs all four recordings, retaining
  finite views and consumer seeds; full single-level overwrites can omit seeds.
  Partial writes require initialized destinations. These rerecorded JIT operations
  do not add a selected immutable whole-graph recipe.
- Vulkan AOT explicit plans bind `texture_collection` kernel parameters through
  `image/collection` v1, sharing immutable tables, layouts and retirement with
  JIT/GFX without copying images. These artifacts require target indexing/capacity
  capabilities and GFX metadata v5. Tests cover 2D dynamic indexing, GPU updates,
  replacement, retirement and downstream gradients. Ordinary/selected Vulkan
  whole-graph binding recipes export private/public/mixed tables, sharing member
  reconstruction, initialization and read/write usage while checking public
  member aliases. Legacy CGraph and CUDA collections remain unsupported.
- Vulkan AOT `pass_indirect` v1 reconstructs classic direct/indirect passes from
  ordinary or selected complete graphics-queue Graphs. Shared preparation checks
  finite command/count views, device limits and shader-write aliases; tests cover
  GPU zero/nonzero counts, replacement bindings, retirement and downstream gradients.

- AOT arrays accept explicit `indirect` usage through the lightweight consumer,
  C API and private-resource reconstruction. Views inherit the parent usage;
  storage defaults and their existing Vulkan dispatch-packet usage are preserved.

- Vulkan 1.2 graphics enables `drawIndirectCount` only when the queried feature
  is supported, fixing validation errors for GPU-count indirect draws. Device
  creation folds selected promoted features into the Vulkan 1.2 structure while
  preserving other enabled features.

- Vulkan graphics removes the fixed 64-buffer table and million-draw pass
  caps. Device capacities and command ranges remain checked; shared reflection
  rejects incompatible cross-stage declarations and excessive merged storage
  descriptor counts.

- Vulkan AOT `pass_direct` v3 and Graph export support fixed storage descriptor
  arrays. Shared RHI range bindings preserve offsets, finite lengths, aliases and
  retirement; paired components require ABI92. Standalone/borrowed tests include
  shader-observed view lengths and downstream numeric gradients.

- Vulkan descriptor-layout caching now stores layout metadata without retaining
  data-resource owners. Cache identity includes descriptor binding flags, so
  ordinary and update-after-bind layouts are distinguished. Recorded descriptor
  sets continue to retain their resources through completion.

- AOT direct mesh/task draws now use graphics pipeline v2 and pass_direct v4,
  with ordinary/selected complete graphics-queue export and public/private
  owners. Preparation shares overflow-safe task/mesh device-limit checks with
  JIT. Windows standalone/borrowed tests include compute-to-mesh updates and
  downstream gradients; mesh/task AS reads and indirect draws remain separate.

- Vulkan indirect-draw preparation now shares device and logical-range checks
  without constructing per-draw capability maps. Prepared offsets use 64 bits
  and preserve view bounds; split native components require paired ABI93.

- JIT/AOT graphics passes now establish memory dependencies with surrounding
  commands at both pass boundaries. This fixes compute-written uniforms being
  read by vertex shaders without a matching dependency; the existing compute-only
  barrier did not cover graphics stages. No per-buffer replay scan is added.

- Vulkan enables supported vertex/fragment storage-write device features and
  rejects writable shader interfaces when the corresponding feature is disabled.
  Mixed readonly/writable storage-block members remain writable for admission.
  Split native components require paired ABI94; AOT graphics formats are unchanged.

- Indirect pass preparation rejects command/count spans overlapping writable
  shader storage, including descriptor-array views and different imported Vulkan
  buffers bound to the same memory. Unknown imported memory must still prove
  separation; distinct buffer handles alone are insufficient.
  Readonly shader bindings are excluded, and adjacent/non-overlapping spans stay
  valid. Cold checks share the native storage identity and range sweep; replay
  adds no alias scans.

- GFX AOT preserves RWTexture formats and mip levels, including shape-only
  kernels. These exports require metadata v4 and paired ABI90 components;
  ordinary artifacts retain their earlier version. Incompatible SPIR-V cache
  entries are recompiled; CPU/CUDA cache formats are unchanged.

- Explicit Vulkan AOT plans reconstruct integer storage images and bind
  RWTexture kernels, including integer MRT and selected mip read/write.
  They reuse the image owner, upload, initialization and GFX recording paths.
  Standalone/borrowed replay and downstream numeric gradients are qualified;
  ordinary/selected Vulkan compute Graph export includes RWTexture aliases,
  complete mip seeds and shared compiled entries.

- Explicit Vulkan AOT plans reconstruct raster pipelines and color/depth
  attachments, then record direct, indexed and instanced draws through the
  shared JIT graphics core. Complete clears need no seed uploads; LOAD and
  sampling retain initialization checks. The lightweight runtime accepts
  explicit vertex/index/uniform array usages. Standalone/borrowed execution,
  resource retirement and downstream gradients are qualified. Ordinary Graphs
  export draw/pass recordings, immutable shader assets or public pipelines,
  attachment aliases and integer MRT clears. Selected complete graphics queue
  recipes reuse the same GFX builder; segmented/fork-join export and mesh/indirect
  coverage remain pending. See the
  [AOT SDK contract](aot_sdk.en.md#explicit-vulkan-raster-plans).

- Vulkan AOT raster pass v2 binds existing managed TLAS resources, including
  shared vertex/fragment query stages and device-refitted scenes. Graph export
  uses the common AS reconstruction and retirement paths; v1 remains available.
  Requires paired ABI91 components. Small standalone/borrowed tests cover
  hit/miss/hit, private/public owners and downstream numeric gradients.

- AOT v18 preserves explicit storage/uniform/vertex/index usages on private
  ndarrays, including archived initial data and inherited view usages. Plans
  without this field retain their existing format/default. Requires paired
  ABI88 components; CPU/CUDA/Vulkan private-array gradients and Vulkan drawing
  from archived private buffers are qualified.

- AOT diagnostics add `native_resource_bytes` for known private native device
  payload, including CUDA/Vulkan image mip chains. Array and workspace meanings
  remain unchanged. A batch memory query reduces Python diagnostic crossings;
  it adds no replay scan. Requires paired ABI89 components; these counts exclude
  public resources and driver-private residency.

- Completion waits release the ticket's status lock and avoid holding the CUDA
  submission lock or Vulkan fence query lock. Queries and paced Graph submission
  can proceed while another thread waits on the same ticket. Runtime reset and
  display teardown drain backend readers before destroying their handles, while
  preserving the first error and exactly-once resource release.

- `Graph.submit(..., wait_mode="blocking")` opts CUDA tickets into low CPU event
  waiting; the default wait is unchanged. The mode is fixed at submission, with
  separate event reuse under a combined cache limit. Vulkan retains fence waits,
  CPU is normally complete, and other backends reject explicit blocking before
  submission. Measure wake latency and complete-step throughput.
- `ti.validate_runtime_contract()["features"]` independently declares
  `display_shared_image_reuse`, `completion_wait_concurrent` and
  `completion_blocking_wait`, also before initialization. Missing keys mean
  unknown; capabilities do not replace ABI/compiler checks or device/platform qualification.
- CUDA Graph argument retirement reuses a successful event query as its
  completion proof, avoiding a redundant event synchronize while recycling
  completed parameter storage. Pending backlog and teardown waits remain.
- Vulkan honors `Window(vsync=False)` by preferring IMMEDIATE when the surface
  supports it, instead of always selecting MAILBOX or FIFO. Disabling vsync can
  tear; supported synchronized modes remain the fallback. Vsync-enabled windows
  retain MAILBOX/latest-ready preference and the adaptive FIFO fallback. This
  changes presentation policy, not CUDA execution streams or the monitor's
  refresh rate.
- `Canvas.repeat_frame()` reuses a renderable holding the current shared image,
  or an empty renderable, and retains other completed images for future writes.
  Completed aliases are detached before reuse. Repeating a CUDA/Vulkan frame no
  longer frees imported image storage and waits for unrelated CUDA work; external
  semaphore ownership and teardown synchronization remain in place.

- `ti.ui.Window.get_scroll_delta()` consumes accumulated horizontal and vertical
  GLFW scroll offsets, preserving fractional values without polling events.
  ImGui callbacks select the event window's context; closing and recreating a
  window cannot replay old offsets. ImGui still receives the raw events; camera
  controls should use `is_render_input_available()` to respect UI capture.
  Windows candidate tests exercised native windows; Ubuntu desktop and released
  wheel application acceptance remain separate checks.

- Inlined `ti.func` calls inside `ti.real_func` now use the active function's AST
  builder, keeping control flow and local variables in the same callable. This
  fixes `MatrixPtrStmt` assertions and access violations in nested tensor helpers,
  as well as incorrect scalar loop and branch results. Real function call
  boundaries and the native ABI are unchanged.

- `ti.real_func` reads returned values at the call site. A subsequent inline loop
  in the same expression can split execution into tasks; scalar and tensor
  results no longer retain references to another task's local return storage.
  Multiple returns retain their value semantics.

- Vulkan shader reflection maintains a parser-local result-ID index instead of
  repeatedly scanning the module. Validation without an artifact capture skips
  unused diagnostic reflection. `ti.compile_profile()` separates SPIR-V emission,
  optimization, validation, reflection/layout preparation and driver pipeline
  creation. Compiler tiers, shader bytes and native ABI remain unchanged.

- AOT v17 adds public resource relationship checks, preventing a refit GAS from
  differing from the GAS retained by a queried scene. Named retained-input paths
  are checked at preparation, with no replay lookup and the existing retirement
  contract. Requires paired ABI83 components; older plans remain supported.

- OptiX Graph export can bind public contexts, triangle scenes, GAS and IAS,
  reusing resources across AOT modules without rebuilding them per plan.
  v17 alias certificates preserve source GAS/context sharing and reject
  mismatched resources before execution. Typed/occlusion queries, refits,
  downstream gradients and transitive retirement are qualified on CUDA.

- Sampled Texture Graph export accepts `resource_bindings={texture: "public_name"}`
  to reuse created or imported NativeResources without private images or seed
  uploads. Ordinary/selected CUDA/Vulkan paths, aliases, downstream gradients
  and retirement are covered; other provider families remain opt-in work.

- Borrowed CUDA/Vulkan AOT runtimes can `import_texture()` from their JIT
  Program, retaining storage and sampler without copying pixels. JIT/AOT
  updates, independent imports, source-wrapper collection and in-flight
  retirement are covered. Requires paired ABI83 components; current sampled
  image format/mip/sampler boundaries still apply.

- AOT adds `Runtime.native_resource()` through the existing factory catalog,
  allowing resources to be shared across modules and plans. CUDA/Vulkan images
  and CUDA OptiX parent/child resources are qualified, including transitive
  retirement; public image constructors release completed seed inputs. Paired
  private components are required. This does not qualify every provider.
  See [AOT SDK](aot_sdk.en.md).

- New AOT Graph exports batch sampled-image seeds into one-time initialization.
  After `prepared.initialize()`, seed arrays used only for initialization can
  close without invalidating replay. CUDA binding families share initialized
  images. CUDA/Vulkan kernel sampling, Vulkan depth uploads and CUDA OptiX alpha
  queries are covered; existing artifacts retain their original seed contract.

- Vulkan color `Texture.from_ndarray()` now uses runtime-tracked transfers for
  every mip count. This fixes stale image layouts when replaying an existing
  Graph after an upload, avoids reading a nonexistent second dimension for 1D
  inputs, and rejects incompatible shapes or texel sizes before copying.
  CUDA/Vulkan sampling and downstream gradients, asymmetric 2D/3D images and
  preservation of higher mip levels are covered.

- Tensor component alias analysis now preserves unknown ndarray origins and
  component indices. This fixes a compiler abort in conditional vector ndarray
  adjoints and avoids treating potentially overlapping accesses as independent.
  CPU, CUDA and Vulkan JIT and standalone AOT forward/reverse checks pass.

- CUDA parallel Graph construction uses a shared fork/join capture core and
  removes the former four-branch construction ceiling. Existing automatic
  two-chain selection is unchanged; explicit eight-branch capture, rebinding,
  dense-Field gradients and retirement are covered. AOT execution plans now
  retain qualified kernel branches, prefixes and join suffixes, private storage
  and selected disjointness constraints. Preparation uses the same core; replay
  uses one CUDA graph launch. Version 9 plans require paired private ABI 65
  components. Parallel regions inside control bodies and non-kernel branches
  still require separate qualification.

- AOT Graph export accepts named live f32 CUDA CSR/BSR values through
  `source_bindings`. It archives topology without copying numeric values and
  automatically refreshes retained SpSV/SpSM state before solving. CSR/BSR SpMV,
  CSR SpMM, independent bindings, ordered updates and explicit input adjoints
  are covered; omitted bindings keep the existing frozen snapshots.

- AOT preserves CUDA adaptive bounded launches and their per-node/grouped update
  policy through the shared JIT construction core. Device controls are rebuilt
  on the consumer, with no host count readback during replay. Version 8 plans
  require paired private ABI 64 components; structured-control combinations
  remain separately qualified.

- cuSOLVERDn Cholesky shares the common provider adapter and supports AOT resource
  reconstruction and ordinary/selected Graph export of retained f32/f64 recordings.
  Solve-only export preserves actual factor/status state; refactor routes omit
  unused old factors. Public device status can be shared with surrounding kernels.
  Standalone/borrowed consumers cover preserved inputs, failure status, retirement
  and explicit adjoints while keeping root-ordered submissions. The optional SDK
  component excludes vendor libraries; GPU conditional bodies, provider autodiff
  and the separate cuSOLVERSp route are outside this scope.

- AOT exports selected CUDA integer segmented-scan strategies, including native
  global scan with retained scratch, chunk, shuffle and length-bucket hybrid.
  Existing driver kernels are reused, immutable topology is retained, each
  preparation rebuilds private workspace, and subgraph binding scopes stay separate.

- CUDA CSR/BSR SpMV share an execution-plan core. BSR keeps its native block
  layout and existing matrix descriptor, replacing duplicated vector-binding
  and workspace management. AOT now supports f32 BSR resources, SpMV and automatic
  recording snapshots, with standalone/borrowed consumption and ordered in-place
  value updates for explicitly bound resources. Exported snapshots remain frozen;
  an explicit transposed BSR can compose with exported kernel gradients.
  Current producer/runtime components require matching private ABI 58.

- Reverse AD preserves integer power exponents, fixing NaN gradients at negative
  bases inside Vulkan conditionals. Base derivatives at zero with exponents 0/1
  are also corrected. Explicit reverse and forward AD regressions pass on
  CPU/CUDA/Vulkan; old kernel caches are invalidated to avoid stale derivatives.

- Native compact/sort/unique sequences now export automatically to AOT, sharing
  packed dense Field storage with kernels. Matching CUDA/Vulkan prepared
  recordings also export. GeoPhys's generic candidate pipeline passes source-free
  consumption on all three backends for duplicates, empty results, overflow and
  stale input; other candidate branches and the complete rigid solver still need
  separate qualification.

- Qualified native sort sequences now export with their surrounding kernels on
  CPU/CUDA/Vulkan. `source_bindings` names retained producer arrays without
  serializing contents or addresses; typed exact-shape certificates preserve
  their binding contract. Standalone and borrowed consumers cover forward and
  explicit adjoint work, ordering edge cases and retirement. Use paired private
  ABI 55 components; other sequence strategies and operations remain separate.

- Graph-owned cuDSS groups now run in qualified CUDA conditional loop bodies and
  export with selected flat loops to AOT. Three numerical phases, multiple RHS,
  alternating bindings, explicit adjoints, zero/early/budget exits and retirement
  are tested; explicit nested AOT plans are covered too. Provider capture uses
  retained parameter storage and exact allocation bindings. Use paired private
  ABI 54 components. Ordinary cuDSS solve snapshots and flat refactor transactions
  export through private ABI 59/62 respectively; cuSOLVERDn coverage is described above.

- Ordinary asynchronous Vulkan nested Graph submissions no longer generate
  unused loop-observation shaders, trace tables, host-readable buffers or
  readback copies. Predicate gates and indirect dispatch retain their semantics.
  Diagnostic paths still generate observations on demand, with distinct cache
  identities. Selected kernel-only depth-2 Vulkan loops now export persistent
  AOT plans through `add_execution_plan`, preserving physical boundaries and
  private control resources. Standalone and borrowed consumers reuse the shared
  recorder, with one warmed prefix/loop/suffix submission and no host predicate
  readback. Zero/early/budget exits, two inner loops, explicit adjoints and
  retirement are tested. Use matching private ABI 53 components; other nested
  routes and complete SolvePlan coverage remain separate work.

- Prepared AOT executions and native resource preparation now share their runtime's
  fault state; borrowed runtimes retain the JIT owner's state without finalizing it
  on close. Fixed CUDA graphs report capture, launch and retirement errors, reject
  replay after a fatal fault, and skip unsafe graph/cuDSS cleanup on a known failed
  device.

- The cuDSS 0.8 adapter adds a common runtime-provider entry point alongside its
  existing ABI1 interfaces. Shared native plans can retain a preparation session's
  runtime through capture and retirement, without another vendor loader per plan.
  AOT can explicitly reconstruct f32/i32 CSR plans and record multi-RHS solve
  groups with fixed factors, factorization or refactorization, sharing one
  numerical factorization within a group. Standalone and borrowed consumers cover
  forward execution and explicit adjoints. The optional `TaichiSDKProvider_cudss`
  component installs the adapter and manifest separately; vendor libraries remain
  application-managed. Graph-owned `CudssCaptureRecording` exports ordinary and
  selected Graphs from the native private seed, sampled once per plan per export;
  paired private ABI 52 is required. Subsequent conditional and ordinary-recording
  export coverage is described above and in the AOT SDK documentation.

- cuDSS Graph memory reports count already allocated plan snapshots and vendor
  payloads as resident requested bytes. Opaque driver storage remains unknown,
  so the report no longer presents a complete zero-byte resident footprint.

- Ordinary Graphs and pinned `from_graph` operators can automatically export pure
  dense Field initial state with `snapshot_state=True`, sampling actual AOT root
  dependencies together with provider arrays. Fields are shared by a loaded
  module; provider workspace remains private to each prepared call. Re-preparation
  does not reset Fields. Three-backend tests cover forward/real kernel.grad,
  module isolation and failure cleanup. Initial-state artifacts require LLVM
  schema 7 / GFX metadata 3 and matching private ABI 50 components; complete
  GeoPhys Field-backed solver qualification remains separate.

- Kernel-backed f32 CG/PCG SolvePlan Graph actions support explicit
  `update_policy="pinned"` and AOT export of selected operator/preconditioner
  generations, private scratch, control and terminal state. CPU/CUDA/Vulkan
  consumers preserve vector layout/alias rules and independent prepared storage.
  Default live rebinding remains unchanged; automatic solver differentiation,
  other solver methods and unqualified providers are not included.

- Zero-budget Graph loops keep native action manifests consistent with their
  reachable condition, fixing parent Sequential compilation failures when the
  unused body contains a native action. Flat Vulkan loops also skip compilation
  and binding retention for that unreachable body; the initial condition still runs.

- Vulkan structured control shaders now omit unused optional status bindings
  from their declarations and entry-point interfaces. This fixes descriptor
  layout failures in nested loops when an outer or inner status is omitted.

- Parent Graph private array declarations now propagate into its created
  Sequential regions, including loop conditions and bodies. Arrays used only
  in those regions remain private, with declaration symbol identity validated.

- Explicit AOT execution descriptions support persistent `while` regions:
  CPU prepares native loops, while CUDA assembles prefix, initial condition,
  loop and suffix in one root graph. Bodies are not unrolled by the budget;
  nesting and 64-bit budgets are supported. Graph export preserves exact CPU
  loops and selected CUDA conditional code, private resources and control binding
  constraints. Vulkan exports selected flat native loops with their chunk and
  strategy recipe, using the shared recording core and one queue submission for
  prefix, chunks and suffix. Complete SolvePlan and other selected routes remain
  separate work. See the [AOT SDK contract](aot_sdk.en.md); matching private ABI 46
  is required.

- AOT can reconstruct fixed CSR resources and f32 cuSPARSE SpMV, including
  transpose, i32/i64 topology and ordered coefficient updates. Each captured
  operation owns its preprocessing state while sharing the JIT execution core.
  Root `CusparseSpmvRecording` sequences automatically export coherent frozen
  data, sharing identical read-only storage. See the [AOT SDK contract](aot_sdk.en.md)
  for the validated scope; other sparse operation families remain separate work.

- AOT adds f32 CSR SpMM using the shared JIT retained plan and existing CSR
  resource. Prepared binding frames own independent descriptors/workspace.
  Ordinary recordings and selected SparseSpmmOperation Graphs export their exact
  algorithm/preprocessing choices; SpMV/SpMM share one matrix snapshot per export.
  Checks cover explicit adjoints, empty matrices, value updates and retirement.
  Materialized mixed Graphs also retain export command order without making frozen
  definitions retain search execution plans. See the [AOT SDK contract](aot_sdk.en.md).

- cuSPARSE generic CSR/COO and dense/sparse vector descriptors now use the
  vendor's signed 64-bit extent ABI, including CSR extent queries. This fixes
  narrowing in the dynamic calls; it does not expand the public `SparseMatrix`
  index/storage contract or claim full cuSPARSE AOT coverage. Use matching
  Python/native components (private ABI 46).

- The runtime wheel now includes a stdlib-only `taichi_forge_runtime.aot`
  consumer sharing JIT's native library. Ordinary AOT kernels and CGraphs can
  explicitly prepare fixed CPU batches, CUDA Graphs and Vulkan commands with
  explicit ndarray gradients. Preparation executes no user kernels; closing
  bound code/storage invalidates its dependent executions. This does not yet
  cover every provider or generic scene specialization; see the
  [AOT SDK contract](aot_sdk.en.md).

- `Module.add_execution_plan(name, graph)` can now export sequential kernel
  Graphs and the qualified CUDA cuBLAS GEMM/cuTENSOR recordings through the shared
  reconstruction registry. It deduplicates kernel specializations and shared
  provider resources without persisting live handles. Small CPU/CUDA/Vulkan
  tests cover explicit reverse kernels; CUDA mixed sequences also cover explicit
  adjoints and independent consumers after producer retirement. Structured
  control and unregistered recording recipes remain separate coverage work.

- Automatic AOT Graph export now retains selected ordinary map compositions.
  It exports physical fused kernels with their argument remapping, deduplicates
  repeated exports, and rejects unreconstructed storage or binding policies.
  Small CPU/CUDA/Vulkan tests cover explicit adjoints and runtime array aliases.

- AOT execution plans now preserve per-kernel dispatch labels without duplicating
  compiled kernel definitions. Preparation uses the common label context; this
  does not enable per-kernel profiling on fixed replay. Long labels no longer
  have an arbitrary byte limit, while NUL and line breaks remain invalid.
  Small CPU/CUDA/Vulkan checks cover long labels, explicit reverse kernels and
  consumer retirement. This introduced private ABI 35; use the matching current revision.

- AOT execution plans can persist packed-array layout requirements and disjoint
  binding pairs, checked once before provider creation or command recording.
  CPU/CUDA address ranges and Vulkan memory/buffer bindings detect overlapping
  imports without adding replay checks. Focused tests cover actual reverse kernels,
  empty arrays, tensor layouts and retirement. Matching Python/native components
  now require private ABI 36.

- Rebuilding a selected CUDA reduction recipe now restores private input/output
  bindings and their aliases. Direct and staged recipes keep the public argument
  set unchanged and allocate only their own selected scratch resources.
  `GraphBuilder.reduce()` also binds declared private endpoints when they are
  used only by helpers, including reduction-to-reduction intermediate storage.

- Mixed AOT execution plans bind ndarray gradients only to dispatches whose
  compiled ABI declares `needs_grad`. Ordinary helper kernels can share the
  same primal arrays; gradients unused by the entire plan remain errors.
  CPU/CUDA/Vulkan checks cover actual reverse kernels before and after ordinary
  copies, without adding checks to replay.

- Automatic Graph export preserves qualified packed-array helper constraints
  and selected CUDA direct/block-partial reduction recipes, including private
  aliases, fixed counts and partial storage. i32/f32 tests cover independent
  preparations and explicit adjoints; custom storage/binding recipes and full
  LinearOperator/SolvePlan export remain outside this qualified slice.

- Private AOT ndarray resources can request `zero_initialize: true`. They clear
  once before initialization/replay and retain state across launches; odd byte
  counts use padded owned storage without another shader. The changed internal
  resource layout introduced private ABI 33; later changes require the matching newer revision.

- Automatic Graph export now reconstructs declared private ndarrays, preserving
  shared allocation identity and gradient aliases across kernel/provider bindings.
  Independent preparations start with fresh state; live array snapshots remain
  outside this automatic reconstruction contract.

- ExecutionPlan resources can now include immutable typed scalars through
  `Scalar.resource(name)`. Exact bit payloads preserve wide integers and negative
  zero, reuse ordinary kernel argument binding and allocate no private device
  storage. This shared resource extension advances the private ABI to 34.

- Automatic Graph export now reconstructs fixed built-in scalar values from the
  final binding frame and physical argument types. Constant descriptions are shared
  by dtype and bits, preserving wide integers and signed zero without extra kernels
  or device allocations. Native scalar bindings still need their own typed contract.

- Ordinary CUDA FFT recordings now export shared native plans, including
  batched C2C/R2C/C2R and explicit physical layouts. Forward/inverse uses of
  one plan retain one workspace. Small standalone/borrowed checks cover explicit
  spectral adjoints and retirement; callback/separable variants remain excluded.

- Refreshing cuSPARSELt recordings use the same automatic export contract,
  preserving shared compression, algorithm/split-K/ReLU configuration and
  explicit fixed-pattern adjoints. Snapshots still use handwritten execution
  plans with explicit raw-weight initialization.

- A versioned `ExecutionPlan` can persist exported kernels interleaved with
  device buffer copies, 32-bit fills, stable sort and compact, then prepare one native CPU/CUDA/Vulkan
  execution with fixed bindings. Descriptions are included in directory and
  `.tcm` modules. LLVM AOT now reads archives through the same lazy virtual
  directory layer as Vulkan, without extracting the archive to disk. This
  format uses registered operation/resource factories for additional provider
  slices. Unregistered primitives, structured control and rendering resources
  still require their own reconstruction contracts.

- AOT compact owns per-execution scratch, preserves record order, and refreshes
  counts even for empty selections. CPU/CUDA/Vulkan tests cover replay with
  changing predicates, independent bindings, retirement and a subsequent
  gather kernel's explicit gradient. Scratch bytes are queryable separately
  from arguments and compiled code; compact itself has no automatic derivative.
  Multiple columns can share one selection and GPU prefix/scan workspace.

- AOT stable sort supports 32/64-bit integer and floating keys, with optional
  scalar/tensor payloads and NaNs last. CUDA uses the existing radix kernels
  with independently owned scratch; Vulkan shares command preparation with JIT.
  The consumer needs no Python compiler. Sort does not add an automatic derivative.

- Repeated LLVM AOT `save()`/`archive()` preserves kernels and dense resource
  mappings, including when more entries are added between exports. Export
  transforms one temporary IR module at a time and keeps the builder's original
  compiled entries intact; deployed code and its runtime cost are unchanged.

- Fixed Vulkan nested ArgPack descriptor layout failures. Compute pipelines
  preserve the module-wide descriptor contract for older shaders; new ArgPack
  code emits the ordinary argument buffer only when it is used. Graphics/ray
  pipelines keep entry-point-specific reflection.

- CUDA AOT can explicitly export `cubin` with a declared compute capability and
  external NVIDIA ptxas. Schema 6 reuses lazy loading, dense resource relocation
  and module retirement; corruption fails without a PTX/bitcode fallback.
  Small Windows deployment tests cover forward/reverse and ordinary Graphs.
  Consumers still use bundled LLVM for runtime initialization. Bitcode remains
  the default; see the [target and toolchain contract](aot_sdk.en.md).

- Fixed AoS `StructNdarray` AOT exports now support CPU/CUDA/Vulkan direct
  execution and ordinary CGraphs, including CUDA PTX. Module-owned target layout
  descriptors compose with existing ndarray gradient bindings; complete binding
  checks precede Graph dispatch. Nested mixed-type records, rebinding and
  retirement are covered by small independent consumers. Structure gradients,
  arbitrary views remain outside this scope. Fixed native kernel/CGraph
  preparation is covered separately by the extension above. See the
  [SDK contract](aot_sdk.en.md).
- C API memory handles now identify live allocations in their owning runtime,
  including imported memory. Foreign and retired handles report an argument
  error before backend access, and device-copy slices are checked against the
  declared allocation capacity without overflowing offset arithmetic. Existing
  `TiMemory`, `TiNdArray` and `TiArgument` layouts are unchanged.
- CUDA runtime retirement preserves another runtime's live allocations instead
  of resetting the process-wide memory pool. Fallback allocation caches are
  scoped to their Device and discarded on retirement; cache reuse no longer
  reads an erased iterator. Small independent-runtime C API tests verify that
  the surviving runtime continues querying and reading data.
- Fix producer-side field storage after AOT export: subsequent JIT calls,
  ordinary Graph preparation and host field access allocate missing storage
  once from the existing layout. CPU/CUDA export-only trees remain unallocated;
  destroying them and reusing their IDs is supported.
- CPU/CUDA dense-field AOT schema 5 binds module-local resources at load time.
  Independent instances can share one runtime without sharing primal/gradient
  storage; close/reload and ordinary Graph replay are verified for bitcode and
  CUDA PTX. Unused PTX relocation constants are removed after optimization.
  The sparse extension above separately covers pointer/bitmasked/dynamic trees;
  `list_v1` covers non-demoted dense lists. Unsupported layouts retain their contract.
- Correct CUDA driver JIT option encoding for register limits and diagnostic
  buffer lengths. Explicit register limits now reach the shared JIT/PTX AOT
  loader as integer values instead of host addresses.
- Restore the Windows C API build with the split runtime's object dependencies
  and stable Graph texture ownership. CUDA stream interop reports the existing
  legacy-default-stream contract and rejects unsupported non-default streams.
- GFX AOT loading rejects incomplete SPIR-V reads and invalid headers. The C API
  now checks serialized capability requirements, accepting devices with equal
  or higher capability levels and rejecting insufficient devices before root
  allocation.
- AOT exports explicit `kernel.grad` targets, including kernel templates and
  bound methods. Dense-field reverse kernels and ordinary AOT Graph replay are
  verified through separate C API processes on CPU, CUDA and Vulkan. LLVM AOT
  includes referenced field layouts automatically and handles nonconsecutive
  tree IDs; Python Tape orchestration and Vulkan autodiff stacks are not added.
- ZIP-backed artifact sources own the compressed bytes and decompress requested
  entries individually, avoiding an expanded copy of every archive entry.
  Entry payload corruption is reported when that entry is read.
- Add explicit C API ndarray gradient bindings for kernels and AOT Graphs,
  preserving existing argument layouts. Separate deployment processes verify
  f32 scalar/vector/matrix forward and reverse execution on CPU, CUDA and Vulkan,
  including reseeding, rebinding and rejection before Graph dispatch. CPU C API
  ndarray allocations can now be released through the existing device allocator.
- LLVM AOT transfers loaded IR to the executable kernel, avoiding a retained
  reader copy and a module clone. Invalid deployments remain untouched on disk.
- GFX AOT modules retain owned artifact sources and load shader bytes only for
  requested kernels or Graph dependencies. Directory/ZIP and in-memory `.tcm`
  deployment are verified on Vulkan, including releasing the input buffer before
  lookup. Failed payload reads are cached and report `TI_ERROR_CORRUPTED_DATA`
  separately from missing export names.
- LLVM O0 removes private scalar spill slots after runtime helper inlining,
  reducing temporary loads/stores before machine-code generation without
  enabling higher-tier arithmetic optimizations.
- Delayed IR edits batch independent insertions and erasures by block, avoiding
  repeated statement searches and array moves during scalarization. Edits that
  depend on newly inserted anchors preserve their sequential order.
- LLVM lowers autodiff stack counters and addresses directly, avoiding repeated
  runtime-helper expansion. Pushes initialize only the adjoint after writing the
  complete primal value; stack layout and generic runtime-helper behavior remain
  unchanged.
- GFX AOT modules and loaded kernels share immutable SPIR-V storage, avoiding
  duplicate host shader buffers and registration copies. Schema 2 records export
  aliases separately: compatible names in one module share shader reads, cached
  failures and runtime registration while retaining their individual callable
  names. Older artifact versions remain readable. Sharing requires matching ABI,
  task metadata and resource dependencies; separate modules retain separate
  execution registrations.
- GFX AOT dense exports retain only layouts referenced by exported kernels,
  Graphs or explicitly exported fields. Producer tree-ID holes are compacted
  into module-local IDs, with shader root bindings validated against the
  declared dependencies. Sparse runtime state is not relocated by this path.
- C API GFX AOT modules bind local roots to independent runtime allocations.
  Closing a module synchronizes its submitted work and retires its registrations
  and roots; freed IDs can be reused without rebinding another live module.
  Small Vulkan tests cover interleaved modules, forward/reverse Graph results
  and close/reload. This is ordinary AOT Graph support, not native Graph export.
- Track the origins of the six shared compilation policy options (tier,
  advanced IR optimization, LLVM/SPIR-V levels, fast math and global simplify
  iterations). Explicit `full_simplify_global_iter_cap=1` is preserved under
  the full tier, including environment and direct Python config assignments.
  Provenance does not split equivalent code-cache keys. Python and native
  components must now be paired at ABI revision 13.
- CPU/CUDA/Vulkan AOT accepts `ti.aot.Module(compile_options={...})` for those
  six options. Explicit export entries override kernel/Program selections;
  omitted entries inherit them. The mapping is copied, frozen before kernel
  export and applied to direct kernels, templates and ordinary Graphs without
  mutating the JIT configuration. Optimization defaults are unchanged.
- Local matrix additions with at most one nonzero component, including matrix
  adjoint updates, avoid expanding zero contributions into redundant loads,
  additions and stores when the local load, addition and store are adjacent.
  This runs at the scalarization boundary under `fast_math`, preserving the
  earlier full-tier optimization path and strict-math behavior.
- Reverse-mode autodiff maps outer-value adjoints to the corresponding reverse
  loop scope. Advanced optimization no longer produces invalid adjoint references
  when loads or expressions are shared across nested loops.
- Reverse-mode autodiff preserves per-iteration local load values, including
  matrix components used by nonlinear expressions. Nested elastic-energy loops
  no longer compute gradients using only the final iteration's matrix values.
- Adaptive autodiff stacks infer capacity from finite nested loop bounds and
  stack lifetimes, avoiding an undersized fallback and repeated per-stack CFG
  analysis for these cases. CFG fallback preserves explicitly sized stacks;
  stack-free integer-power helper loops do not disable the analysis. Dynamic
  or unmodeled early-exit stack paths retain the existing fallback behavior.
- Reverse-mode autodiff constructs empty reverse loop bodies directly instead of
  deep-copying and discarding forward subtrees. Loop execution metadata and
  `stop_grad` scope markers are preserved.
- Reverse-mode autodiff now backs up nested loop bounds across forward/reverse
  scopes, including argument and locally derived bounds. The fast tier no longer
  depends on advanced optimization to make these references valid.
- LLVM autodiff stacks now allocate the complete 64-bit counter header, matching
  the runtime layout and preventing out-of-bounds adjoint accesses and incorrect
  gradients in the fast tier. IR allocation and runtime share the layout;
  older kernel-cache entries are invalidated without changing tier defaults.
- SPIR-V compute shaders without ray queries skip the use-index construction
  needed only for ray-query result analysis. Code generation, GFX registration
  and AOT export avoid intermediate shader copies while preserving query
  results, artifact formats and resource ownership contracts.
- Whole-kernel CSE avoids creating empty candidate tables during ancestor-scope
  lookups and groups global pointer candidates by field. No-op iterations skip
  use-index construction while preserving branch-hoisting invalidation,
  equivalent-index, type, activation and sparse-lifetime checks.
- Constant folding and atomic demotion build batch-replacement use indexes only
  when a rewrite occurs, avoiding whole-IR collection during no-op traversals.
  Analysis results transfer ownership and avoid temporary operand vectors and
  result-container copies. Optimization rules, convergence and numerical
  behavior remain unchanged.
- JIT Graphs retain their build-time execution artifact, callable ABI and launch
  policy. First replay no longer selects different code or recompiles after a
  global configuration change; rebuild the Graph to adopt new compilation
  options. Valid artifacts and their actual resource bindings
  remain usable after frontend definition retirement. AOT Graph serialization is
  unchanged; Python/native pairs require ABI revision 12.
- Different optimization requests for one kernel may retain different Field
  dependencies, fixing an assertion when switching configurations. Definition
  retirement accumulates dependencies across variants; artifacts and Graphs keep
  their exact bindings. A previously field-free artifact no longer lets a changed
  request skip the SNode lifecycle lock.
- LLVM offline JIT caches store bitcode to reduce serialized size and read costs,
  while retaining legacy text payload reading. New writes use a separate cache
  schema. JIT and AOT share the module codec. CPU/CUDA AOT now writes bitcode
  with schema 3 metadata declaring the encoding; schema 2 text artifacts remain
  readable. Invalid declared payloads fail without selecting a stale alternative
  file. Optimization tiers and numerical rules are unchanged.
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
  invalidated. The Python shim and native runtime require matching ABI revision 12.
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
- CFG dead-store elimination updates live-address state once per retained store,
  avoiding a copied map and repeated identical updates while preserving partial
  tensor coverage and the existing alias predicate.
- CFG dataflow analysis groups definite-alias candidates by local allocation,
  global field, and global/thread-local temporary offset, reducing unrelated
  address comparisons. Dynamic-index checks and conservative fallbacks for
  unclassified pointers remain unchanged.
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
