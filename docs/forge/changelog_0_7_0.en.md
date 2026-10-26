# Taichi Forge 0.7.0 — full changelog

[中文](changelog_0_7_0.zh.md) · [Release summary and upgrade requirements](release_notes.en.md#070)

Implementation details and feature-specific behavior are listed below.

## Compilation and launch policies

- CPU compilation removes statically empty trailing range chunks while retaining productive
  partitions, explicit `parallelize` worker requests and the existing scheduler. Dynamic or
  unproven ranges are unchanged. The unused `block_dim_adaptive=False` setting now produces a
  compile-time deprecation notice instead of an ineffective global configuration write.
- LLVM GPU sparse listgen defaults to declared-capacity scheduling matched to its root/nonroot
  slice mapping, removing idle CTAs without freezing later topology to the first active set.
  Unproven layouts, including Hash, retain their previous scheduling. CPU/Vulkan keep their
  own implementations. See [listgen options and validation boundaries](forge_options.en.md#27-sparse-struct-for--listgen-optimisations).
- Ordinary CUDA/Vulkan loops now select default block sizes at compile time.
  For constant or source-bounded ranges without thread-group dependencies, compute-unit
  count and subgroup width can reduce idle lanes and spread underfilled work.
  There are no trial launches, kernel occupancy queries, per-submit decisions
  or extra candidate artifacts. `ti.ndrange`, `ti.grouped` and dense Field loops
  lowered to ranges share final-task analysis. Unknown dynamic ranges, explicit
  block/task policies, reductions and shared-memory tasks retain their geometry.
  Unrelated compile options no longer disable default selection; complete task
  plans preserve explicit workgroup ownership per task. Unknown unit counts
  allow only single-block trimming; portable AOT does not inherit the producer's
  compute-unit count. See the [compile guide](cache_compile.en.md).
- Explicit `TaskLaunchPolicy` supports multiple ordinary range, ndarray and
  demotable dense Field loops, preserving per-loop source ownership and serial
  dependencies. Dense ranges no longer clip explicit blocks to element counts.
  Compile-time advice for proven idle lanes retains the user's configuration;
  performance guesses neither reject nor override it.
- Ordinary CUDA/Vulkan grid-stride kernels can reduce idle blocks from host-known
  ndarray shapes, shape products or `i32` range bounds with constant addition,
  subtraction and multiplication (for example `3 * n - 1`), retaining the
  existing block size and saturation grid envelope. This needs no timing search,
  device-count readback or extra compiled specialization; unqualified tasks retain
  their scheduling. Ordinary Graph updates re-record when the resolved grid changes.
  Prepared instances retain the original grid for bounds that depend on mutable
  submission parameters; unrelated updates do not disable shape-based selection.
  Every intermediate integer operation must fit i32; otherwise the existing grid
  is retained. The common affine path is normalized at compile time. Composite
  bounds also support `min`/`max` and positive compile-time integer division, using
  a checked DAG that preserves signed rounding and original arithmetic order.
  Shared subexpressions are described once; workload and expression depth are not capped.
  Products of checked host expressions also cover `ti.ndrange` clamped dimensions
  and nonzero starts. Every intermediate product must fit i32, even when a later
  clamp or multiplication by zero would hide the overflow.
- Ordinary CUDA/Vulkan range tasks can read immutable scalar/shape end bounds
  directly. When both bounds are supported host expressions, the start is also
  evaluated directly, removing serial tasks used only to transport these values.
  Device-memory and mixed host/device range snapshots, debug arithmetic and
  effectful prefixes retain their existing execution. CPU scheduling is unchanged.
  Expression rewrites and task cloning preserve both direct bound references and
  their ownership within the task body.
- Range metadata separates workload sources from qualification for a specific transformation.
  CUDA can also elide whole empty groups for verified scalar sum/dot/min/max TLS reductions,
  preserving productive blocks, tail participation and reduction grouping. Unknown TLS,
  physical-thread observations, shared memory and unproven calls retain their existing route.
  The same proof admits device-count collectives to explicit Graph controller selections;
  each offload retains its range snapshot and serial initialization/finalization. The extra
  controller cost must be measured. CPU scheduling and default device-count launches are unchanged.
- Explicit complete Graph recipe search can consume CUDA/Vulkan compiled device-range
  qualifications, retaining code sharing, mutable bindings and selected-plan AOT export.
  Direct-kernel trials do not include recording-only alternatives. **CompileIQ autotuning
  remains disabled by default**; ordinary JIT/Graph/AOT execution never starts a search.
  [Search reports](graph_recipe_integration.en.md) add whole-search accounting, measurement
  sources and cold-cost execution equivalents, separately from break-even counts. Failed and
  unselected trials are included; missing historical wall time after resume stays unknown.
- Explicit CUDA/Vulkan offload-plan search retains a shared baseline and immutable edits instead
  of a full task plan for every candidate. Inspection or binding constructs the selected
  plan; prepared execution is unchanged. The compact search manifest is version 2, with
  versioned candidate identities distinct from the unchanged execution/code identities.
- Complete Graph and experimental CUDA/Vulkan offload-plan searches support candidate paging,
  comparisons across pages and resumption after a budget interruption. The 4096-ID limit
  bounds one transport frame, not the full candidate domain. Paging requires the companion
  CompileIQ fork's `logical_stage_pages_v1` capability. Large kernel searches require an
  explicit trial budget; cross-session resume requires a portable evaluation context.
  Catalogs, measurements and compiled caches still grow with search work. Paging does not
  imply constant memory or fewer trials. See the [search contract](graph_recipe_integration.en.md).
- Explicit Graph search generates survivor combinations incrementally within the caller's
  budget and accepts `terminal_repeat_count` for stronger final measurements. The companion
  fork interleaves forward/reverse observation rounds and preserves uncertainty/Pareto layers
  at the halving boundary. These improve search quality and preparation cost; application
  speedups still require independent validation. Autotuning remains off by default.
- Complete Graph search adds lazily compiled kernel configuration candidates: tier and
  legal workgroup choices on CUDA/Vulkan, plus CUDA launch bounds. Repeated specializations
  share candidate code while retaining each binding. Survivor neighborhoods compose and
  restore controls; selected code uses existing AOT export with original derivative dispatches.
  Changed source semantics are rejected during deferred materialization. Automatic tuning
  remains disabled.
- Legal workgroup discovery covers multiple range tasks with serial prefixes/suffixes by
  intersecting each task's resource qualification. Experimental per-task plans also support
  Vulkan workgroups without exposing CUDA-only controls. Early plan admission and final-task
  identity share the same dense-Field-to-range proof, preserving source blocks and existing
  derivative paths.
- Vulkan task manifests, caches and AOT metadata retain explicit source block ownership,
  allowing discovery to exclude conflicting workgroup requests before compiling them.
  Ordinary execution geometry and semantics remain unchanged.

- Vulkan domains support fixed sparse Field execution with complete tree-generation
  proofs. AOT reconstruction and selected JIT code loans share exclusive admission,
  private traversal lists and diagnostics. Independent/chunked pools, device loops
  and explicit gradients are qualified; conflicting same-tree access is rejected
  before submission. Public CUDA JIT/AOT preparation also supports materialized sparse
  closures, including initialization, device loops and explicit gradients. Mixed dense
  Fields share directory lifetime while callers retain responsibility for data dependencies.
  Completed logical errors do not prevent parameter-slot reuse or safe retirement; original
  tickets retain their errors. CUDA loans prevent new trees from replacing the shared
  allocator while borrowed. See the
  [execution-domain contract](cuda_execution_domains.en.md).
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
  retained upload storage counted in instance costs. Same-runtime native public AS resources
  support shared-read/exclusive-write loans over the full dependency closure, also enforced
  on existing recordings. Different TLAS writers can share read-only BLAS dependencies.
  A borrowed runtime's `import_acceleration_structure()` explicitly binds original
  Program triangle scenes/BLAS/TLAS to AOT plans, retaining resource identity,
  ordered instance inputs and independent import retirement. Flat selected JIT Graphs
  also borrow original AS resources for native query/refit, DSL TLAS parameters,
  graphics shaders and initialized ray programs, retaining selected code and source
  lifetimes. Ray programs retain immutable shader assets for shared cold lowering and
  validate fixed buffer identity per Graph port. Native-only recipes need no placeholder
  kernel; fixed recording ndarrays are live loans, including persistent sort on
  CUDA/Vulkan. Opacity micromaps use their separately declared domain capability;
  AS/parallel and AS/control-flow combinations still require separate qualification.
  Graphics domains can
  bind private TLAS resources in vertex/fragment shaders, sharing one submission with
  preceding device refit and subsequent computation. Preparation checks actual queue
  capabilities and the full AS closure. Private cold payloads upload on the selected
  domain. Initialization retains an ordinary-queue prefix when its resource dependencies require one.
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

## Compilation cost and artifacts

- Repeated dispatches in large Graphs share immutable kernel access metadata,
  and compilation transfers dispatch storage to the finished Graph. Frozen
  definitions retain compressed canonical JSON snapshots, decoded only for
  inspection or export. Graph identity, bindings, retry and execution order
  remain unchanged; warm replay does not decompress snapshots.
- Cold Graph IR conversion shares read-only metadata instead of constructing
  a complete diagnostic dictionary for every dispatch. Full diagnostic
  snapshots remain independently mutable. Within one Graph compilation,
  identical kernels, execution handles and frozen launch policies share their
  invocation ABI; per-dispatch bindings, runtime state and resource retirement
  retain their existing ownership. Repeated argument-name sets and built-in
  immutable effect/binding descriptions are deduplicated locally, without a
  permanent global table, dispatch-count cap or extra warm-path lookup.
- Weak execution-handle registrations are pruned on cold paths instead of
  accumulating with historical calls. Temporary Python kernel definitions
  are reclaimed on a cold path after their last consumer releases them;
  Graphs, launch contexts, AOT builders and native operators retain the
  definitions they need. Reusable artifacts remain managed by the compilation
  cache. Cleanup introduces no definition-count cap and does not disable retry.

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

## Graph and runtime

- CUDA AOT preserves each entry's selected occupancy waves/work-per-thread launch policy.
  Ordinary execution and Graph recording share cold resolution on the consumer device
  instead of silently using the baseline grid. Different policies on shared code remain
  independent. Readers that do not support the required metadata version reject the
  artifact; see the version information under upgrade guidance above.
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

## AOT and native/provider integration

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

## Autodiff and backend correctness

- Within each Vulkan pipeline, compatible render passes reuse variants across attachment load/clear
  and final-layout changes when formats match. Format order, attachment count
  and depth format remain distinct. Warm graphics cache hits use one lookup,
  without allocating a normalized key or changing pass commands.
- Command binding propagates lazy Vulkan pipeline preparation errors instead
  of terminating at a `noexcept` boundary; unfinished recording can be discarded
  and rebuilt after the resource condition is resolved.
- Vulkan graphics construction now releases shader modules after partial failure.
  Pipeline layout and graphics pipeline creation errors are reported before an
  invalid variant can enter the cache; later valid preparation can retry. The
  change adds no draw/replay checks.
- `SpirvShader.specialize()` explicitly freezes scalar specialization constants
  into a new immutable shader at preparation time. Reflection, pipeline identity
  and AOT export consume the same effective bytes, including descriptor-array
  lengths. It uses bundled SPIRV-Tools with Vulkan 1.3 validation and adds no
  draw/replay work or automatic variant search.
- Vulkan mesh/task pipeline creation checks known local workgroup sizes and
  declared mesh output counts against device limits, shared by JIT and AOT.
  Capability queries now include per-axis local sizes. The path uses Vulkan
  1.3 or earlier core functionality plus optional `VK_EXT_mesh_shader`.
  See the [API reference](forge_api_reference.en.md) for validation boundaries.
- The generic experimental Vulkan sparse SNode startup warning is removed. Actual capability,
  capacity and independent Hash diagnostics remain.
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

## Execution domains and windows

- Independent CUDA and fixed-compute Vulkan domains accept explicitly declared `i32/u32/i64/u64` parameters frozen per accepted submission, with compiled ABI widths and integer range checks.
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
  blocked until the last reader retires. Qualified offscreen graphics and flat kernel parallel
  recipes also have independent-domain routes, with explicit resource and control-flow restrictions.
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
