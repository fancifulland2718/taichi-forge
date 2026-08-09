# Taichi Forge Release Notes

This is the canonical version index for Taichi Forge user-visible changes.
Version `0.6.1` is the latest published release. Its final Python shim/source
boundary is `b129ad94c`; the paired published native runtime wheel reports
build identity `c268ca5671e8`. Current `master` targets `0.6.2`, and its source,
shim, and runtime package metadata are aligned at `0.6.2`. Version `0.6.0`
remains the previous published runtime source boundary, and `0.4.25` is the
final public `0.4.x` baseline.

PyPI storage is limited, so some nonessential older distributions have been
removed. Absence from the current PyPI release list does not mean that a
version never existed. The source-boundary column below is the durable history
anchor; packaging-only, CI-only, test-only, and documentation-only commits are
grouped under the behavior they shipped.

## Quick index

| Version | History status | Source boundary | Main scope |
| --- | --- | --- | --- |
| [Unreleased](#unreleased) | 0.6.2 development | current `master` | Graph lifecycle/telemetry fixes, numeric-generation rebinding, recordable Graph PCG, and device-convergent batched PCG |
| [0.6.1](#061) | published release | `b129ad94c` | task launch manifests/policies, dynamic LLVM SNode directories, device-resident dynamic worklists, bounded Graph dispatch, and correlated pipeline telemetry |
| [0.6.0](#060) | published release | `106ad65d25` | structured Graph control/telemetry and Vulkan indirect dispatch, sparse runtime/linear algebra, driver-only CUDA primitives, managed interoperability/display, and bounded runtime lifetimes |
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

- Graph cache reset and destruction now avoid constructing the CUDA context
  when a CUDA-enabled runtime has only used CPU or Vulkan Graph state. CUDA
  caches retain submission-lock ordering; the 0.6.1 split-shim compatibility
  override has moved to its final native-runtime owner.
- Nested Graph execution statistics now consume a single flattened backend
  mapping, and asynchronous telemetry instruments structured regions inside
  recordable/native Sequential actions by stable region path. Missing or
  duplicate regions continue to fail closed.
- Recordable f32 compiled-Graph A/M providers can now execute fixed-linear PCG
  with device-resident convergence control on CUDA and Vulkan. Compiled-Graph
  PCG selects this path automatically; compiled-Graph CG keeps its previous
  default and may opt in explicitly. Qualified contiguous Field operands bind
  directly with no SolvePlan pack/unpack submission. Solver statistics expose
  logical, encoded, and masked iteration counts so early stopping remains
  observable on exact and bounded backends.
- Recordable scale/sum composition now normalizes ordered weighted leaves and
  lowers each accumulation to one in-place `axpby`. It preserves provider
  order and generation/lifetime checks, reuses one Graph-owned scratch vector,
  and does not claim cross-provider kernel fusion. In a paired local 262,144-
  element, three-leaf qualification, dispatches fell from 8 to 5, temporary
  storage from 2 MiB to 1 MiB, and warm submit/wait median fell from 270.2 to
  203.5 us on CUDA and from 555.3 to 465.9 us on Vulkan. These are workload-
  specific qualification results, not universal speedup guarantees.
- Equal-extent pure `compose()` chains now flatten in exact forward/adjoint
  leaf order and ping-pong through the destination plus one Graph scratch
  vector. Depth-4/depth-8 chains therefore use one N-vector rather than three/
  seven. In a local 262,144-element source qualification, depth-8 warm medians
  improved by 12.6% on CPU, 15.9% on CUDA, and 11.5% on Vulkan; depth-2 stayed
  in the previous noise band. Rectangular and mixed-extent chains retain the
  conservative nested lowering.
- Recordable f32 CG/PCG `SolvePlan.submit()` on CUDA/Vulkan now wraps the
  complete device-convergent solve in a cached one-action Graph and returns one
  `SolvePlanSubmission`. Its terminal
  packet remains device-resident through `done()`/`wait()` and is materialized
  once by `result()`. Optional workspace lanes provide independent Krylov
  storage, while statistics expose variant/lane memory and terminal readbacks.
  The managed path uses the same persistent bytes as an equivalent hand-built
  Graph. In a paired local 262,144-element probe, wrapper overhead versus that
  Graph was 2.1% on CUDA and within noise (-1.2%) on Vulkan; terminal-result
  materialization remains an explicit additional synchronization/readback.
  CPU uses the existing exact native solve and returns a completed lane-0
  submission, avoiding GPU-style masked replay; it exposes no Graph terminal
  packet or Graph telemetry.
- A nontrivial qualification uses a three-dispatch SPD stencil A and a two-
  dispatch Jacobi M. At 262,144 elements both GPU backends stopped at iteration
  13: CUDA encoded 13 with no masked tail, while Vulkan encoded 64 and masked
  51. In the final 60-sample source qualification, warm managed submit/wait
  medians were 1.856 ms on CUDA and 4.107 ms on Vulkan; the same manual Graph
  was 1.825/4.053 ms, fused-provider upper bounds were 1.789/3.864 ms, and
  kernel host-check-K4 controls were 4.042/4.727 ms. Each managed/manual Graph
  used 5,244,972 persistent bytes.
  These paired local measurements qualify overhead and stop telemetry, not a
  universal solver-speed claim.
- Compatible values-only `LinearOperator` generations now rebind into cached
  Graph actions and SolvePlan Graphs at launch. Validation is two-phase across
  every composition leaf; topology/schema/state-tree/runtime changes still
  fail closed, and each asynchronous ticket pins the exact immutable numeric
  owners that it submits. A local 262,144-element update/run qualification
  measured cached rebind versus Graph rebuild at 0.924/1.583 ms on CPU,
  0.438/1.044 ms on CUDA, and 0.587/1.094 ms on Vulkan. Thirteen retired
  generations were all released after completion, with no active lease left.
- `BatchedSolvePlan` now accepts explicit `device_convergent` CUDA/Vulkan
  execution when A and the optional fixed-linear M are recordable f32 actions.
  One structured Graph contains initialization, A/M, reductions, recurrence,
  per-system status, and the global active predicate; `submit()` returns one
  ticket, and terminal materialization exposes the exact logical stopping
  iteration without an iteration-loop host readback. Vulkan retains bounded
  encoded/masked tail semantics. The existing host-check default is unchanged.
  Vulkan fixed-budget execution uses direct recurrence dispatches because a
  nested replay synchronization inside an active submission batch is not
  qualified.
- Graph provider arguments and generation owners are now prepared as one
  immutable launch snapshot. Composition leaves are validated before any
  binding is published, so a values-only update cannot mix replacement
  arguments from one generation with lifetime owners from another. A setup
  fixed-linear `PreconditionerPlan` is recordable when its approved action is
  recordable; every Graph ticket pins the exact approved target/action pair,
  while stale or unapproved generations continue to fail closed.
- Batched solver statistics use schema v5 and every solve publishes one packed
  terminal packet. Opt-in per-ticket telemetry reports logical/executed/provider
  work, active efficiency, available encoded/masked work, Graph launches,
  physical queue submissions, and non-inferred timing. A lazy
  `workspace_pool()` adds independently fenced, memory-accounted Graph/workspace
  lanes with explicit wait/raise saturation; it does not claim independent GPU
  streams or physical overlap.
- CUDA device-convergent batched CG/PCG can explicitly compact active systems
  for recurrence reductions and vector updates without a host count readback.
  A/M provider applies remain full-batch, and the capability truthfully reports
  a capacity-grid masked prefix rather than exact indirect dispatch. The option
  is off by default and unavailable on CPU/Vulkan. In a local 262,144-scalar,
  64-system interleaved qualification, a heterogeneous batch improved from
  19.855 to 18.044 ms (9.1%); an all-hard control improved from 58.508 to
  56.679 ms (3.1%). Runtime, host-pool, and device-pool memory were stable;
  borderline floating-point termination differed by at most one iteration in
  the heterogeneous case. These measurements qualify this workload, not a
  universal speedup.
- Added `inverse_block_diagonal()` for caller-supplied row-major f32 inverse
  blocks of size 1 through 4. Each size now uses a specialized kernel and a
  constant-size topology word instead of two offsets per scalar row. At
  262,144 scalars this reduces the operator-owned topology snapshot from about
  2 MiB to 4 bytes; paired warm apply medians remained within backend noise.
  It is recordable on CPU/CUDA/Vulkan and uses the same numeric rebind/pinning
  contract as other compiled providers. The caller
  must explicitly assert SPD; Forge does not read back, invert, regularize, or
  infer the blocks. On a local 64-system, 262,144-scalar diagonal workload, an
  exact inverse reduced roughly 72-100 unpreconditioned/sqrt-scaled iterations
  to two. With the harder sqrt-scaled preconditioner, device-convergent PCG was
  3.1% faster than host-check-K4 on CUDA and 17.3% faster on Vulkan. With the
  exact two-iteration inverse, host-check remained faster, preserving the new
  policy as explicit-only and demonstrating that preconditioner quality and
  convergence length determine the crossover.
- Added a headless fixed-topology implicit spring reference using public
  `LinearOperator`, 2x2 `inverse_block_diagonal`, `PreconditionerPlan`, compact
  Vector fields, numeric-generation updates, and one reusable `SolvePlan`.
  A local 2,304-node qualification reduced logical iterations from 54 (CG) to
  6 (PCG). Warm median solve latency changed from 37.061 to 7.281 ms on CPU and
  3.455 to 2.412 ms on CUDA; Vulkan was within noise at 7.377/7.460 ms because
  its bounded path encoded 96 slots for 6 useful iterations (90 masked).
  Rebind-plus-solve remained far below rebuild-plus-solve on CUDA and Vulkan,
  and a 1,000-frame test on all three backends kept one GPU Graph, released all
  1,004 retired generations, left no active lease, and observed stop positions
  from 4 through 6. These figures qualify this example and expose the Vulkan
  tail cost; they are not universal solver-speed guarantees.
- Added `LinearOperator.shifted(shift)`. Recordable f32 GPU lowering executes
  the base provider followed by one in-place `axpby`, with no second identity
  provider dispatch and no identity-sized temporary. Non-square operators,
  non-finite shifts, and unsupported dtype/backend combinations fail
  explicitly. In a local 262,144-element paired Graph qualification, this
  reduced dispatches from three to two and warm submit/wait median from
  0.298 to 0.222 ms on CUDA and from 0.568 to 0.441 ms on Vulkan, with zero
  numerical error.

- Generalized `LinearOperator.apply(alpha=..., beta=..., addend=...)` now uses
  device-native f32 ndarray transform/scaled-add lowering on CUDA and Vulkan.
  It performs no host readback and allocates no N-vector in the non-aliasing
  path; exact addend/output aliasing reuses one persistent scratch. In a local
  paired 262,144-element qualification, native apply versus an equivalent
  two-dispatch Graph measured 0.822/0.955 ms on CPU, 0.240/0.380 ms on CUDA,
  and 0.401/0.422 ms on Vulkan. These host-synchronous measurements are
  workload-specific rather than a universal speedup claim.
- Added atomically updateable `parameterized_affine()` operators. Mandatory
  closed coefficient ranges bound conservative trait derivation, optimistic
  versions reject stale updates, and every submission pins one complete
  alpha/beta generation. f32 coefficients and range endpoints are canonicalized
  before trait derivation, so unrepresentable values fail and a positive bound
  that rounds to zero cannot preserve an invalid SPD claim. Cached Graphs patch
  two scalar bindings from one native immutable snapshot without rebuilding,
  allocating a dummy provider, or uploading a device parameter array. A local
  262,144-element update-plus-run versus rebuild-plus-run qualification measured
  0.996/3.534 ms on CPU, 0.372/3.675 ms on CUDA, and 0.410/4.531 ms on Vulkan.
  These synchronous measurements are workload-specific; correctness tests also
  retain an old snapshot across an overlapping update/submission boundary.
- Fixed-layout `block_diagonal()` standalone apply now supports qualified f32
  CUDA/Vulkan leaves, and Program-bound CPU leaves use the same runtime-storage
  subrange contract. Consecutive domain/range slices are resolved once and
  submitted in leaf order without gather/scatter, whole-vector staging, or an
  N-sized temporary. It deliberately does not fuse leaf kernels: a two-leaf
  262,144-element identity probe cost 0.792 ms CPU, 0.264 ms CUDA, and 0.458 ms
  Vulkan versus 0.388/0.135/0.335 ms for one fused kernel. Permuted,
  overlapping, non-affine, and recordable-container forms remain unsupported.
- Added `SmallBlockInverseBuilder` for fixed f32 row-major blocks of size 1-4.
  Its direct and one-dispatch Graph forms run partial-pivot Gauss-Jordan
  on device with block-scale-relative pivot tolerance, leave per-block
  success/non-finite/singular status resident, and
  zero failed outputs without inferring SPD. For 16,384 4x4 identity blocks,
  local device build versus host readback/NumPy inverse/upload measured
  0.824/5.175 ms on CPU, 0.410/6.195 ms on CUDA, and 0.722/7.667 ms on Vulkan.
  A 1,000-build reuse stress on all three backends retained the same caller-
  owned output/status allocations. The implicit spring reference now keeps
  coefficient assembly, inverse construction, and status on device before
  publishing the preconditioner generation. In a local 2,304-node, 30-step
  end-to-end qualification, device refresh versus the prior host assembly path
  measured 9.282/8.966 ms on CPU, 6.240/6.527 ms on CUDA, and 10.017/13.130 ms
  on Vulkan, with the same six logical PCG iterations and zero failed blocks.
  CPU therefore keeps only a small measured overhead while the GPU paths avoid
  the host round trip; these timings are workload-specific.

## 0.6.1

- Release provenance is intentionally split: `b129ad94c` is the final Python
  shim/source boundary, while the immutable published runtime wheel identifies
  its native build as `c268ca5671e8`. The distribution versions remain matched;
  their source commits need not be equal after a compatible shim-only fix.
- The final split-wheel shim now retires CPU/Vulkan Graph caches without
  initializing the CUDA driver. It acquires the CUDA submission lock only when
  a cache actually owns CUDA Graph state, preserving CUDA lock ordering without
  adding work to the Graph execution hot path. This keeps the already-published
  native runtime wheel unchanged and fixes installed-wheel validation on
  no-driver Windows/Linux hosts.
- LLVM CPU/CUDA SNode metadata is now addressed through a geometrically grown
  per-Program tree directory and exact-sized per-tree runtime-state blocks.
  The former fixed global SNode and tree tables no longer define a runtime
  capacity ceiling; allocation overflow and stale tree generations still fail
  closed. Tree diagnostics expose the runtime-state component without double
  counting it, and internal Program diagnostics expose directory capacity,
  active trees, reserved bytes, and growth events. Qualification covered a
  4,098-node mixed dense/pointer/dynamic/hash tree, global ids above 1,024,
  513 simultaneously live trees, destruction, and generation-safe slot reuse
  on CPU and CUDA. In the current-runtime scaling benchmark, a 4,099-node
  tree's lookup median was 1.011x the 3-node tree on CPU and 0.919x on CUDA;
  the 513-tree phase grew the directory from 16 to 1,024 entries (8 KiB) and
  recovered the active-tree count after destruction. These are scaling and
  ownership results, not a historical pre-refactor speedup. AMDGPU uses the
  same LLVM representation but remains unqualified; Vulkan has an independent
  sparse-runtime implementation. CUDA no-return kernels now receive an
  immutable compact root binding after live-tree validation at kernel
  registration. A one-tree kernel carries the root pointer directly and needs
  no binding allocation; a multi-tree kernel owns eight bytes per dependency,
  and every offload uses the same full-kernel mapping even when its tree set is
  disjoint from other tasks. The generated root load is outside the
  grid-stride loop, while Graph replay performs no directory lookup, host
  readback, allocation, or synchronization. CPU and result-returning LLVM
  kernels retain the general directory accessor. Directory bounds, generation,
  and lifetime checks remain at registration and launch boundaries. A matched
  synthetic HVP probe on the Windows RTX 5090 qualification machine measured
  5,504 versus 5,632 executed SASS instructions and 1.66 versus 2.144 us under
  Nsight Compute for the candidate and public 0.6.0 respectively; Graph replay
  medians were 11.122 versus 11.255 us. This qualifies removal of the hot-path
  regression, not a universal application speedup. Twenty-three targeted
  lifecycle and high-id cases covered two-tree/two-offload mapping,
  destruction, unrelated-tree retirement, tree-id reuse, 4,098-node trees,
  and 513 simultaneously live trees without weakening fail-closed behavior.
- Windows CPU JIT sessions now allocate each LLVM RuntimeDyld COFF object from
  one page-aligned `Code -> read-only -> read-write` mapping. This satisfies
  the image-relative ordering and 32-bit span required by `ADDR32NB`, including
  section alignments larger than the OS page, and replaces the earlier
  insufficient policy of merely retaining all object sections. This prevents
  intermittent
  `IMAGE_REL_AMD64_ADDR32NB` ordered-layout failures after repeated runtime
  reset or mixed CPU/GPU backend initialization. The change is restricted to
  the Windows COFF JIT setup and does not add synchronization to CUDA or Vulkan
  Graph replay. Five consecutive mixed-backend bounded-Graph lifecycle suites
  exited cleanly after the fix; the environment-sensitive CUDA driver setup
  probe also passed when isolated.
- CPU device-known bounded Graph dispatch now selects the exact scheduler by
  default. The scheduler snapshots and clamps the extent once, skips zero work,
  and invokes positive work as adaptive contiguous JIT loops instead of
  per-element callbacks.
  This decouples CPU grain from GPU `block_dim` and restores LLVM loop
  vectorization. The forced `masked_capacity` route remains available for
  fallback and A/B diagnosis. The lowering reuses the existing CPU bounded
  binding and runtime scheduler symbols, so it adds no split-runtime ABI or
  symbol requirement. On the Windows qualification machine, a
  262,144-element payload with 16 nontrivial operations measured exact/fixed
  masked p50 speedups of 6.55x at zero count and 2.78x at 10% count, while the
  full-count ratio was 0.997x; p95 ratios were 6.50x, 2.67x, and 0.999x.
  Correctness covered clamping, overflow, TLS reductions, `continue`, and two
  independent concurrent Graph callers; 1,000 alternating replays retained
  stable runtime, host-pool, and device-pool ownership.
- CUDA device-known bounded dispatch now has an exact logical-range lowering
  on every supported driver. The default route reads and clamps `DeviceExtent`
  on device and keeps the ordinary saturation-capped grid-stride launch, so it
  needs neither host readback nor CUDA 12.4 node updates. On qualified 12.4+
  drivers, the forced `device_update` route remains available as a physical
  optimization: it caps the updated grid at the same saturation grid and can
  skip a zero-count payload, while correctness still comes from the logical
  range. Capabilities now separate logical exactness from physical launch kind
  and report schema v4; `masked_capacity` remains an explicit A/B baseline.
  Paired same-runtime tests covered zero, block boundaries, overflow, ndarray
  rebinding, labeled ordinary fallback, a two-block saturation cap, concurrent
  replay, and 1,000-2,000 replay memory stress. On this Windows CUDA machine,
  the default exact route was within 0.4% of masked at full count and 2.2%
  faster at 10% of a 4,194,304-item capacity; at a 16,777,216-item capacity it
  was 4.9%/4.0%/1.2% faster at zero/1%/full count. The adaptive route has a
  workload-dependent updater crossover and therefore is not the default. A
  final paired-wheel forced-route run at 4,194,304 capacity and 16 payload
  operations measured masked/exact medians of 34.026/34.428 us at zero,
  34.328/33.384 us at 10%, and 34.782/38.144 us at full count. All routes were
  correct, exposed their distinct physical counts, and retained stable runtime,
  host-pool, and device-pool ownership across 1,000 replays. The close and
  workload-dependent crossover is why the older-driver masked route remains a
  qualified fallback rather than being removed.
- The opt-in CUDA 12.4+ adaptive physical route now groups two or more
  consecutive bounded payloads with the same extent/capacity/block contract
  under one stateful updater. Persistent grid/enabled state is reused when it
  is unchanged; singleton payloads retain the per-node route. In a 64-payload,
  16,777,216-capacity, one-operation qualification, grouped/stateful replay
  was about 1.3x/1.9x/1.04x faster than per-node control at zero/1%/full count
  across repeated runs; a stable full-count rerun measured 5056 us versus
  5420 us. The grouped control now includes opt-in device-side replay,
  state-change, cache-hit, and actual node-API-call counters exposed by
  `Graph.execution_stats()` without replay-time host readback. Persistent
  bounded-control storage for 64 payloads is 592 bytes versus 2,048 bytes for
  per-node control, and 1,000 alternating replays retained stable ownership. These
  figures qualify the policy crossover on the tested RTX 5090 rather than
  making the adaptive route the general CUDA default.
- CUDA structured control now has a Forge-owned bounded masked Graph route for
  drivers older than 12.8. Qualified Driver API 12.8+ runtimes continue to use
  native conditional Graph nodes; older runtimes use a device latch and
  task-entry gates when ordinary CUDA Graph capture is available, and otherwise
  retain exact portable control. Capabilities distinguish exact native,
  bounded masked, and portable execution instead of presenting them as the same
  physical launch. The forced masked route on a current driver qualifies the
  Forge-owned fallback semantics and performance without requiring obsolete
  hardware. It does not claim to validate a particular old driver's loader or
  vendor implementation. The paired 0.6.1 wheels passed the forced while,
  `if`, and `switch` contracts. For a 262,144-item, 16-iteration workload over
  1,000 replays, forced masked Graph measured 366.9 us median versus 465.3 us
  for native conditional Graph and 1,410.9 us for portable control; all routes
  stopped at iteration 16. This crossover is workload-specific, not a general
  claim that masked control is faster than native CUDA control.
- Added single-ticket execution for the strictly qualified depth-two
  `while -> ordered while[1..8]` shape. The outer body may place ordinary
  actions between leaf inner loops; inner controls must be disjoint and the
  complete hierarchy remains capped at 4,096 encoded actions. CPU performs
  exact host control and returns a
  completed ticket; Vulkan uses bounded conditional replay; CUDA uses bounded
  static topology without an intermediate host readback. A qualified Driver
  API 12.4+ CUDA runtime first passes a cached setup probe and then uses
  device-updatable kernel-node groups, compiling each business dispatch once.
  Unqualified or older runtimes use Forge's version-independent two-gate
  task-entry masking; `TI_GRAPH_CUDA_FORCE_MASKED_CONTROL=1` qualifies that
  fallback on a current driver. Both loop bounds are at most 64 and the
  complete encoded program is capped at 4096 actions. Capabilities report the
  selected/candidate/fallback route and explicitly keep exact dynamic command
  termination false. An outer suffix can preserve each inner stop position on
  device, and a recordable provider can expose a terminal packet after ticket
  completion.
  In the local Windows qualification with 4,096 items, an 8x16 budget, four
  active outer iterations, stop positions 6/7/8/6, five warmups, and 30 timed
  replays, two fresh-process CUDA node-update medians were 652.55 and
  640.95 us; two forced-fallback medians were 742.45 and 745.15 us. The
  median-of-process-medians warm reduction was 13.0%. The same nested Graph
  measured 1,592.85 us on Vulkan versus 9,392.0 us for an optimistic
  host-known direct-call oracle and 1,929.7 us for a compact root-level Graph.
  CPU measured 12,260.3 us versus 17,207.95 us direct, but remained 9.7%
  slower than its 11,171.55 us compact oracle. Persistent argument storage was
  14,808 bytes for CUDA node update (14,272 bytes of controls), 648 bytes for
  forced CUDA masking, 1,304 bytes for Vulkan, and zero for CPU. CUDA cold
  invocation remained about 104 ms versus 101 ms for the fallback, so the
  higher-memory route is a warm replay optimization, not a cold-start claim.
  These figures qualify this machine and workload rather than a universal
  backend speedup.
  A separate ordered-two-inner qualification used four active outer steps,
  4,096 items, first-inner stops 6/7/8/9, and second-inner stop 2. The single
  ticket measured 796.15 us on CUDA and 2,045.45 us on Vulkan, versus
  4,463.7 us and 6,522.1 us for an optimistic host-known outer loop that still
  waited for each adaptive inner Graph: 5.61x and 3.19x faster respectively.
- Graph submission telemetry schema v5 now separates logical Graph and region
  invocations, backend Graph launches, CUDA stream enqueues, and physical queue
  submissions. CUDA leaves the physical-queue count explicitly unavailable;
  Vulkan reports a device transaction-window delta and marks it non-exact.
  Ticket-owned nested telemetry preserves every inner stop position and uses
  `logical_invocations` to distinguish repeated child calls from the final
  iteration count. The opt-in path adds no host readback to normal submissions.
- Graphs containing exclusive Graph-owned solver storage can now compile with
  one to 64 workspace lanes. Lanes are materialized lazily, selected from
  completed lanes in round-robin order, and may be pinned per submission;
  saturation either waits or fails immediately by policy. This removes the
  completion-fence dependency between queued solves while preserving separate
  terminals and workspaces. It does not create backend streams or promise GPU
  overlap. Memory reporting includes lane capacity, materialized/busy counts,
  waits, saturation failures, and aggregate persistent bytes; each additional
  materialized lane has linear workspace cost. In paired-order local Windows
  diagnostics, two lanes reduced two-solve completion by 11.6-22.5% on CUDA
  at 4,194,304 items and by 6.1-7.9% on Vulkan at 262,144 items. Internal
  storage doubled exactly from 83,918,892 to 167,837,784 bytes on CUDA and
  from 5,244,972 to 10,489,944 bytes on Vulkan. These non-strict-idle figures
  qualify the queueing mechanism, not general solver throughput or GPU
  parallelism.
- Added read-only offloaded-task manifests and JIT dispatch labels. A manifest
  reports stable task identity, `cpu_scheduler`/`grid_stride`/
  `device_bounded_grid_stride`/`one_to_one`/`not_applicable` range mapping,
  requested, selected, and actual grid/block
  geometry, and static shared-memory context without launching a profiler
  probe. CPU leaves GPU geometry absent, while runtime-indirect Vulkan work
  explicitly reports that actual geometry is device owned. Labels preserve the
  same task identity in profiler and optional NVTX names, and manifest queries
  remain no-submit and allocation-stable.
- Added `TaskLaunchPolicy` for constrained direct-JIT block tuning. CUDA and
  Vulkan accept `hint`/`require` policies for a single safe parallel range
  task, expose resolved geometry and compile-time resource constraints through
  immutable reports, and reject unsupported task shapes or block-sensitive
  rewrites before enqueue. CPU hints explicitly retain worker scheduling and
  requirements fail. Policies do not override grid extent, are cache-separated
  specializations, and reuse the normal warm launch path after read-only
  validation. Register/local-memory values stay explicitly unavailable when a
  no-submit query cannot obtain them safely; no autotuning or profiler launch is
  introduced. Prepare a cold GPU policy with `report()` on the Python main
  thread before worker-thread use, and retain `auto` as the performance
  baseline.
- Opt-in Graph submission telemetry now includes an immutable, ticket-owned
  `GraphPipelineReport` for the post-optimization execution root. Each stage
  reports its logical and physical dispatch counts, runtime argument names,
  native-action composition, declared temporary bytes, and any existing
  structured-region GPU timestamp; ordinary CGraph stages additionally expose
  physical `GraphTaskManifest` entries. Pipeline
  schema v2 correlates labeled bounded dispatches with task identity and
  selected/actual launch geometry, count source, capacity, block size,
  selected route, and ticket-owned useful/executed/encoded work. Device-known
  counts add one deduplicated two-word tail snapshot per distinct extent;
  host-known counts add no device buffer. Ordered segments report the reliable
  aggregate extent but leave per-segment useful work unavailable without an
  offsets snapshot. Ordinary CGraph stages mark these mappings `available`;
  structured while/if/switch stages explicitly mark them
  `structured_runtime_dependent` instead of inventing a flattened physical
  task sequence. `NativeActionManifest` freezes the
  provider's symbolic bindings, effects, temporary requirements, and
  recordability/backend contract without exposing storage objects or device
  addresses. Ordinary stages do not invent a per-stage duration when only a
  whole-ticket timestamp is available. The default `telemetry=False` path does
  not materialize the report or its telemetry arena. A Windows RTX 5090/CPU
  qualification retained one 8-byte slot and measured the optional complete
  report cost over `submit().wait()` at about 0.529 ms on CPU, 0.350 ms on
  CUDA, and 0.510 ms on Vulkan for a 4,097-of-65,536 bounded payload. This is a
  sampling/debugging cost; continuous per-step telemetry is not a zero-cost
  mode.
- CUDA driver-only stable radix sort now derives the 16 digit bases inside
  each scatter block and removes the standalone digit-base kernel and
  workspace. In a matched full-pipeline 1,048,576-item random-key A/B on an
  RTX 5090, the qualified candidate reduced median sort time from 508.11 us to
  454.44 us (11.8%) and p95 from 562.41 us to 498.49 us while preserving stable
  duplicate-key ordering and bounded replay memory. These local qualification
  figures characterize that device/workload; they are not a universal speedup
  promise.
- CUDA driver-only stable radix sort now stops its histogram hierarchy when
  the current level fits one 1,024-item scan tile. For a 32-bit sort whose
  first histogram level has 1,024 blocks, this removes eight redundant scan
  launches and eight no-op uniform-add launches per sort (`53 -> 37` total
  device-kernel launches) and removes the unused one-element parent from the
  workspace. A paired public-0.6.0 versus release-candidate-0.6.1 wheel test on
  the same RTX 5090/610.62 system used three fresh processes per wheel, ten
  warmups and 100 end-synchronized calls per process. The median of process
  medians for 1,048,576 `i32` items fell from 0.51245 ms to 0.36455 ms, a
  28.9% latency reduction (1.41x throughput), while reported peak workspace
  fell by 512 bytes to 29,425,664 bytes. Thirteen installed-wheel CUDA
  key/payload dtype and large-hierarchy stability cases passed. This is a
  device/workload qualification, not a new CUB-parity or universal-speedup
  claim.
- Added `DeviceWorklist`, a Graph-independent fixed-capacity front/back
  container with a device-owned `DeviceExtent`, atomic append, stable
  selection, and deterministic integer-key conflict resolution. Atomic append
  uses one slot-reservation atomic per accepted item in the overflow-free path;
  append order is unspecified, one producer owns each transition, and
  `commit_next()`/the recorded finalize node must publish counters and extent
  before consumption. `DeviceWorklistSequence` records reset, finalize,
  selection, or keyed claim as reusable native Graph actions. Graph arguments
  can attach generated/accepted/rejected/conflict/winner/overflow counters to a
  `SubmissionTicket` without steady-state replay host count readback or
  allocation.
  An adjacent Vulkan finalize and bounded consumer automatically publish into
  one Graph-owned exact indirect packet, with no public launch-state object or
  preparation dispatch; matching consecutive consumers reuse the packet. CPU
  uses its exact adaptive scheduler by default, while CUDA uses its exact
  logical range without consuming a Vulkan packet and may optionally trim the
  physical grid on qualified 12.4+ drivers.
  Deterministic keyed claim has
  an intentional workload crossover: at 262,144 active items it measured
  8.63x/9.05x over a full host round trip on CUDA/Vulkan, while a sparse 1,638
  item claim was slower on all three backends. The qualification harness keeps
  those cases separate and observed stable memory across 1,000 CPU and 3,000
  CUDA/Vulkan replays.
- Added `DeviceDispatchState` and `DevicePrefixSequence` for fixed-topology,
  device-count-driven pipelines. Vulkan compact can now publish its bounded
  dispatch packet with the output count and pass it to
  `dispatch_bounded(launch_state=...)`, removing the consumer preparation
  dispatch. CPU uses its exact adaptive scheduler by default; CUDA independently
  uses an exact logical range and may select 12.4+ adaptive physical control. The unified
  `dynamic_work_capabilities()` report separates physical launch semantics,
  structured iteration termination, and completion observation.
- Graph terminal observations are completion-attached by default. Vulkan/CPU
  use host-visible arena slots, while CUDA appends an asynchronous copy from
  device-local snapshot storage to persistent pinned host memory before the
  ticket completion. This avoids a second readback at
  `ticket.observations()` and avoids the page migration of managed CUDA
  storage. The previous deferred path remains available as a diagnostic
  fallback.
- Added `DevicePrefix` and `DevicePrefixWorkspace` to compose compact, scan,
  reduce, sort, consecutive unique/RLE, grouped reduce, and bucket building
  through a shared device-written `DeviceExtent`. The fixed-capacity provider
  and reusable workspace contracts remain visible; the wrapper removes count
  readback between operations without claiming active-count execution for every
  primitive. A compact-to-scan qualification with a 10% active prefix measured
  1.05x/1.32x/1.90x over explicit host observation on CPU/CUDA/Vulkan.
- Added `GraphBuilder.dispatch_bounded()` for host-known exact ranges and
  device-known bounded work, plus `dispatch_ordered_segments()` for globally
  ordered offset ranges using one reusable payload specialization. Vulkan uses
  device-written indirect packets and a compiler-proven one-to-one range
  mapping. CPU uses its exact adaptive scheduler for ordinary bounded dispatch,
  while ordered segmented CPU dispatch retains its globally ordered masked
  route. CUDA reports logical exactness separately from its saturation-capped
  static or 12.4+ adaptive physical launch.
  Consecutive standalone Vulkan consumers with the same extent, capacity, and
  block dimension now share one prepared 12-byte packet; any intervening
  action conservatively invalidates it. In a 64-consumer, 4,194,304-capacity,
  one-operation qualification, packet reuse reduced zero/1%/full medians from
  3.14/3.14/3.17 ms to 1.68/1.70/1.72 ms and reduced packet storage from 768
  to 12 bytes. The bounded/fixed median ratio recovered from about 0.53-0.54x
  to 0.97-0.98x; 1,000 bounded-slot replays retained stable ownership.
  The fixed eight-slot Vulkan replay ring now applies bounded backpressure
  when an indirect Graph has more submissions in flight than available slots.
  Such a Graph cannot preserve its device-written dispatch packet through the
  ordinary-launch fallback, so saturation waits for the oldest slot instead
  of failing or growing replay memory; unsaturated and ordinary Graph paths
  are unchanged.
  Overflow, useful/executed/skipped/encoded work, invalid offsets, workspace,
  and zero-command behavior are available through capability and explicit
  snapshot objects. Provider-qualified recorded producers can now publish a
  Graph-owned Vulkan packet directly; an intervening action restores the
  conservative prepare path. Exact Vulkan work reduction is not presented as
  an unconditional speedup: the preparation dispatch can outweigh it for
  light standalone payloads.
- Added `DeviceExtent`, a stable two-slot device state for bounded counts and
  sticky overflow. Device-side publish clamps without host readback; the same
  allocation can be shared by ordinary kernels, JIT Graph arguments, and
  compatible count-producing primitives. Reset/normalize stay device-side,
  explicit snapshot/check operations synchronize, and stale runtime
  generations fail closed. This state contract does not itself claim exact
  indirect dispatch or alter kernel grids.

## 0.6.0

Version `0.6.0` consolidates the changes after the published `0.5.0` runtime
source boundary. It does not retroactively change the behavior attributed to
the `0.5.0` artifacts:

### Upgrade overview from 0.5.0

| Area | Main change in 0.6.0 relative to 0.5.0 |
| --- | --- |
| Graph and execution | Fixed-schema `while`/`if`/`switch`, structured composition to depth two, CUDA conditional Graphs, Vulkan bounded/compound/nested while execution, Vulkan device-written indirect dispatch, and stop-position, region, and queue telemetry. |
| Linear algebra and sparse runtime | Public runtime-bound `LinearOperator`, experimental `SolvePlan` and batch plans, fixed sparse pattern/value updates, and documented provider matrices for CG/PCG, MINRES, BiCGSTAB, GMRES, and FGMRES on CPU/CUDA/Vulkan. |
| Data, interoperability, and display | A common dense-storage/view contract, managed DLPack and external allocations, CUDA-Vulkan shared display, window edge regions, continuous font scaling, and collapsible auto-height panels. |
| Native primitives and packaging | Forge-owned driver-only CUDA primitive providers in standard wheels, Program-owned workspaces and diagnostics, stable radix/compact/scan improvements, and runtime/shim build-identity gates. |
| Correctness and lifetimes | `SharedArray` block ownership, Tensor/AD/SVD and dense-field alignment fixes, crash-safe offline-cache locks, and bounded allocator, specialization, trace, SNode, and reset lifetimes. |

When upgrading an existing 0.5.0 application, check the following:

- Local or offline installations must use runtime/shim wheels with matching
  distribution versions. Their source commits may differ when the split-wheel
  workflow selects a compatible runtime explicitly; final link, import,
  dependency, and functional validation—not commit equality—decide whether
  that pair is usable.
- CUDA primitive code should select `method="auto"` rather than depend on a
  `cuda_cub*` provider that exists only in the non-publishing reference build.
- Declare `ti.simt.block.SharedArray` inside a parallel range-for block scope.
  CUDA permits at most 48 KiB of static shared storage per block; larger
  requests fail explicitly instead of enabling dynamic shared memory.
- Query capabilities before selecting structured Graph or
  `dispatch_indirect()` paths. Vulkan indirect dispatch currently requires one
  offloaded task; CPU and CUDA do not silently emulate it.
- Rebuild Graphs, storage views, external owners, and solver plans after
  `ti.reset()` instead of reusing an old generation.
- Existing `from_dlpack()` and provider-specific Vulkan-CUDA import spellings
  remain compatible. New code may use the common `from_external()` and
  `import_external_allocation()` entry points.

- Offline-cache metadata locks now use operating-system advisory locks held by
  an open file handle. Process termination releases ownership automatically, so
  a persistent `.lock` file no longer causes repeated load/dump warnings or
  requires deleting compiled cache state.

- Added `ti.experimental.ndarray_view(source, slices=...)` for strict zero-copy
  binding of qualified runtime-owned dense storage to `ti.types.ndarray(...)`
  kernel arguments on CPU, CUDA, and Vulkan. It accepts contiguous Ndarrays,
  qualified dense scalar/vector/matrix fields, and rank-preserving
  positive-stride subviews. View composition combines checked byte offsets and
  per-axis strides without staging or temporary allocation. Negative,
  broadcast, overlapping, permuted, sparse, and externally owned layouts fail
  before enqueue. Stale owners are rejected and GPU submissions retain the
  runtime resource through completion.
- Added `ti.interop.from_dlpack()` and `ExternalDenseView` for strict managed
  zero-copy import. CPU/CUDA-host producers bind on CPU; CUDA/CUDA-managed
  producers bind on CUDA. The runtime owns the capsule deleter, validates the
  byte range and owner generation before every submission, defers retirement
  through in-flight work, and makes `close()` safe after runtime reset. Vulkan,
  cross-device import, unsupported layouts, and copy fallback fail explicitly.
- Managed external submissions now use synchronization-domain access epochs.
  An ordinary launch or Graph submission acquires each distinct domain once
  and releases it in reverse order after enqueue or failure. Historical NumPy,
  PyTorch, and Paddle argument signatures remain compatible; synchronous CPU
  NumPy retains its low-overhead direct ABI and established incompatible-layout
  fallback.
- Added provider-neutral `ti.interop.from_external()` and
  `import_external_allocation()`. The existing `from_dlpack()` spelling remains
  compatible and enters the same managed owner/view protocol. The initial raw
  provider imports dedicated Vulkan-exported memory and paired binary
  semaphores into CUDA, exposes multiple compact typed-offset views, groups
  them into one Graph access epoch, and fails closed on device, handle,
  layout, lifetime, or synchronization mismatch. The provider-specific
  `import_vulkan_cuda_allocation()` spelling remains available.
- The public Vulkan-CUDA provider and GGUI shared-display path now use one
  checked raw-handle import core. Compared with the previous internal importer,
  GPU resource topology and measured per-process GPU-memory peaks are
  unchanged, Windows concurrent display timing shows no regression, and
  invalid/duplicate handles plus partial construction or cleanup failures are
  handled without a global CUDA submission lock.
- GGUI `canvas.set_image()` now automatically packs qualified CUDA field and
  ndarray images into Vulkan-owned exportable storage imported by CUDA.
  External semaphores provide a bounded CUDA-produce/Vulkan-consume cycle, and
  steady state uses the normal render submission without a host round trip or
  same-frame cross-device copy. Capability/device mismatch falls back to the
  established staging path. `Window.get_display_stats()` reports actual
  zero-copy render submissions. On the qualified Windows 2048 x 2048 workload,
  the complete warm frame loop improved by 6.2% with byte-identical output.
- Concurrent CUDA production and Vulkan presentation now rearm superseded
  shared-display frames before reuse. This closes the intermittent
  `Shared display storage is not available for CUDA` failure without adding
  the global CUDA submission lock that reduced the affected engine workload by
  4.5%-8.8%.
- GGUI exposes fixed font scaling and continuous logical-height tracking through
  `Gui.set_font_scale()` and
  `Gui.set_font_scale_from_window_height()`. Vulkan and Metal share the same
  linear policy. It uses the existing logical display size at the frame
  boundary, performs no GPU readback, and does not rebuild the font atlas.
- GGUI also exposes bounded logical-pixel font sizing, auto-height subwindows,
  and independently collapsible sections. Responsive control panels now fit
  their visible text and widgets and grow or shrink as sections are toggled,
  without application-side height calculations or extra GPU submission.
- GGUI Window now provides independently optional top, bottom, left, and right
  root regions around a central render viewport. Region resize/collapse,
  Vulkan/Metal viewport and scissor, scene aspect, viewport-local input, and
  fullscreen images share one logical/framebuffer layout snapshot. This adds
  no intermediate render target or copy. Responsive font policy now composes
  with per-window user zoom and edge-local Ctrl+wheel, Ctrl++/-, and Ctrl+0
  shortcuts without rebuilding the font atlas.
- `ti.simt.block.SharedArray` now has a fail-closed block-ownership contract.
  Declarations inside a parallel Taichi range-for, including declarations in
  an inlined `@ti.func`, retain the existing one-task fast path. Kernel-root
  and serialized-loop declarations are rejected consistently by JIT, AOT, and
  Graph compilation before offload separation can promote their storage to a
  kernel-global temporary. CUDA and Vulkan carry runtime regression coverage;
  other GPU backends are not newly qualified by this change. CUDA limits total
  static `SharedArray` storage to 48 KiB per block. Larger requests report an
  explicit error; Forge does not enable opt-in dynamic shared memory.
- JIT Graph `ArgKind.NDARRAY` runtime arguments now consume the common runtime
  storage protocol for Ndarrays, dense fields, and explicit
  `DenseNdarrayView` objects. Compact Program-owned Ndarray and SNode payload
  bindings are eligible for CUDA capture, exact replay, and compatible
  allocation patching; owner generation and byte ranges are revalidated before
  replay. Positive affine views execute through CUDA ordinary fallback and
  Vulkan command record/replay with the same result contract. Managed external
  owners use ordinary/replay access epochs rather than CUDA capture. AOT
  borrowed storage and ArgPack nesting remain unsupported.
- Added `GraphBuilder.dispatch_indirect()` and
  `Sequential.dispatch_indirect()`. Vulkan Graph replay executes
  `vkCmdDispatchIndirect` directly from a device-written three-element u32
  packet, supports zero-group payload skipping, and safely re-records after a
  packet-allocation change. The target kernel must produce exactly one
  offloaded task, and the packet must currently be an owning Taichi ndarray.
  Field, external-storage, and AOT packets plus CPU/CUDA execution fail
  explicitly instead of pretending to provide fixed-size or exact indirect
  dispatch.
- Added fixed-schema structured Graph control with `GraphBuilder.while_loop()`,
  `if_then_else()`, and `switch()`. Condition kernels can combine tolerance,
  cancellation, activity, and breakdown values without a Python callback.
  Continue predicates and user-defined terminal status are independent;
  `Graph.control_flow_stats()` reports lowering, logical/executed iterations,
  status traces, observation traffic, and fallback reasons. Qualified CUDA
  `while`, `if`, and `switch` regions automatically use native conditional
  Graph nodes; `native_required` regions also support asynchronous
  `Graph.submit()` without a host control readback. Conditional metadata upload
  is asynchronous and retains at most two deferred replay batches. CPU retains
  exact portable control. `Sequential` now exposes the same structured
  builders for one nested level, with a maximum structured depth of two. CPU
  executes both levels exactly. At depth two, the parent uses exact portable
  control; a qualified `auto` leaf may retain its flat native route: CUDA
  `while`/`if`/`switch`, or Vulkan `while`. This is the default
  portable-parent/native-leaf route, not a general native depth-two contract;
  a strictly qualified Vulkan while-to-while `auto` definition may additionally
  upgrade to the single bounded replay described below. Nested
  `native_required` definitions fail closed. Vulkan supports both exact portable control and
  qualified bounded `native_required` `while` regions with a per-region
  `chunk_size` capped at 64, an eight-chunk/512-iteration limit, and compound
  asynchronous submission of multiple ordered regions through one terminal
  ticket. Each region may select compact or coarse-gated first-chunk execution;
  automatic Vulkan lowering combines compact masking in the active chunk with
  coarse conditional-rendering gates for later chunks. Runtime transactions
  coalesce their command buffers into one queue batch while preserving
  semaphore order and bounded replay-slot retirement. Opt-in
  `submit(telemetry=True)` records per-region entry and terminal snapshots and
  reports the actual stop iteration, encoded/masked work, active/skipped
  chunks, enqueue time, and a qualified queue-counter window after ticket
  completion. The default submission path allocates no telemetry buffers or
  snapshot kernels. A qualified Vulkan while-to-while definition can encode
  both levels as one bounded replay when conditional rendering is available,
  both bounds are at most 64, the complete program contains at most 4096
  encoded actions, and the regions use independent one-element i32 controls.
  Other nested shapes use exact portable-parent control; an eligible leaf
  `while` may still retain the flat Vulkan route above.
  `Graph.run(trace=True)` uses portable-parent exact execution and returns every nested invocation;
  `GraphWhileReport` includes nested paths and logical/encoded stop positions.
  Asynchronous submission of nested structured Graphs remains unsupported.
  Vulkan structured replay may fall back only before queue submission; a
  completion or terminal-observation failure after submission raises instead
  of executing the side-effecting body again.
  Vulkan `if`/`switch` and exact dynamic command termination remain unsupported
  and are reported independently by `structured_control_capabilities()`.
- Added `LinearOperator.graph_action()` for recording compiled-kernel f32
  providers directly into Graph roots and structured bodies. Provider-owned
  topology/numeric generations remain zero-copy fixed bindings, input/output
  dense storage uses the common runtime protocol, and stale numeric generations
  require rebuilding the Graph. The generic control and provider contracts are
  qualified with preconditioned CG and nonsymmetric BiCGSTAB programs without
  adding solver-specific Graph APIs. Consecutive CGraph/provider regions fuse
  into one backend region, and providers may bind private per-invocation Graph
  temporaries without exposing them as runtime arguments.
  `LinearOperator.from_graph(..., state=...)` additionally accepts one
  representative live root-dense scalar, Vector, or Matrix Field for each
  distinct dependent pure-dense SNodeTree and retains that tree's existing
  storage without a copy. Matching is tree-granular; keys and Field components
  are not access-level capabilities. Generic compiled-Graph operators
  preserve ordered multi-dispatch forward and explicit-adjoint actions; the
  legacy square form records its forward action but does not infer an adjoint.
  Missing or extra dependency trees, any sparse/dynamic descendant in a
  dependent tree, indirect dispatch, and stale numeric, SNode, or runtime
  generations fail closed. Recording one action alone does not promise
  a speedup; the intended gain is composition with surrounding Graph actions.
- `LinearOperator` scale, sum, compose, and explicit-adjoint trees now lower
  recursively when every leaf exposes a recordable f32 action. Sum and compose
  use typed f32 storage from the Graph-owned bounded temporary arena, retain no
  public scratch argument, and preserve independent lanes for concurrent
  submissions. Standalone f32 scale/sum/compose also execute on CUDA/Vulkan;
  sum and compose retain private persistent workspace. Recordable composed
  CG/PCG providers automatically use the qualified device-convergent path on
  both GPU backends and fail instead of selecting a host-check substitute.
  `linear_operator_composition_bench.py` compares the automatic Graph with an
  equivalent explicit Graph, standalone/no-Graph execution, and direct versus
  staged compact Field bindings while reporting correctness and temporary
  memory.
- Added explicit CUDA `device_convergent` execution for compiled-kernel f32
  CG/PCG through the generic structured Graph and recordable A/M actions. It
  reads one terminal packet per solve and fails closed on unavailable or stale
  providers. Parallel vector updates and persistent two-stage shared-block
  reductions keep recurrence work on the device without per-iteration host
  observation. The plan reports its reduction geometry and fixed scratch
  bytes. This path is correctness-qualified as `explicit_only`; automatic
  compiled-kernel plans retain K=4 `host_check_every_k` so construction and
  first-execution amortization remains an explicit workload decision. Stored
  f32 CSR/BSR CG/PCG retains its automatic conditional-Graph upgrade. The new
  `linear_operator_graph_krylov_bench.py` reports build, first, warm, profiler,
  terminal, and true-residual evidence per policy.
- Qualified CUDA/Vulkan recordable f32 CG/PCG plans now bind canonical compact
  full-field RHS, output, and initial-guess operands as solver-Graph runtime
  arguments. The Graph preamble/epilogue moves boundary values through one
  plan-owned iterative ndarray, removing separate pack/unpack submissions, one
  completion synchronization, and one of the former two boundary staging
  vectors. This is a Graph-fused boundary path, not provider-native zero-copy;
  indexed/non-compact layouts and other solver/provider combinations remain
  staged. New telemetry distinguishes support, enablement, latest full-boundary
  selection, direct bindings, and fallback transfers. In a local Windows
  qualification with ten effective CG iterations, repeated 262,144-scalar
  composition runs reduced warm median latency by 2.3%–4.3% on CUDA and
  3.9%–11.2% on Vulkan; 2,304-scalar runs were about 4.1%–4.2% and
  10.3%–11.1%. These measurements establish the local crossover and observed
  desktop-run spread, not a universal speedup.
  `linear_operator_graph_field_solve_bench.py` provides the paired
  ndarray/forced-staging/direct test.
- `ti.linalg.LinearOperator.apply()` and single-system `SolvePlan.solve()` accept
  supported 1D/2D/3D root-dense scalar, Vector, and Matrix fields. Overwrite
  `LinearOperator.apply()` directly binds canonical compact full fields for
  compiled-kernel and compiled-Graph providers on CPU/CUDA/Vulkan, and for
  fixed native CSR/BSR providers on CPU/CUDA. Generalized apply forms, Vulkan
  native sparse providers, indexed/non-compact views, and SolvePlan combinations
  outside the qualified recordable f32 CG/PCG scope use reusable device staging.
  Warm solves do not allocate staging, and conversion never enters a Krylov
  iteration.
  Stable raw-field bindings reuse qualified implicit views and transfer plans;
  execution telemetry distinguishes direct submissions from native or compiled
  Graph pack/unpack paths.
- Added runtime-bound `VectorView` and `vector_view(field, indices=...)` for
  validated, frozen scalar subsets or permutations, together with versioned
  capability, layout metadata, direct/staging/pack/unpack/indexed-copy
  telemetry, and provider-qualified zero-copy candidate reporting.
  Sparse SNodes, noncanonical layouts, invalid indices, and unsafe aliases fail
  explicitly without a host vector fallback.
- Native algorithms and LinearOperator vector adapters now derive dtype, shape,
  owner generation, byte range, offset, and record stride from the shared dense
  storage descriptor. Provider-specific handles and warm native-plan replay are
  preserved.

- `ti.linalg.LinearOperator` is now the public runtime-bound operator API.
  Operator traits, composition, vector views, and operator qualification are
  exported from `ti.linalg`. The callback-only field wrapper is named
  `ti.linalg.FieldLinearOperator`, removing the former ambiguity between two
  unrelated `LinearOperator` contracts. Solver execution plans remain under
  `ti.linalg.experimental`.
- LinearOperator compiled-kernel and compiled-Graph providers now bind qualified
  Program-owned Ndarrays, dense fields, and explicit `DenseNdarrayView` objects
  through the common runtime-storage argument protocol. Compact operands bind
  directly on CPU, CUDA, and Vulkan. Rank-one scalar positive-stride views bind
  directly to compiled kernels and preserve zero-copy Graph execution; provider
  combinations without affine support fail explicitly.

### Numerical tooling support boundary

The `0.6.0` `LinearOperator` tooling supports fixed-topology, runtime-owned
operators and qualified CPU/CUDA/Vulkan Krylov execution. The documented
provider matrix covers CG/PCG, MINRES, BiCGSTAB, restarted GMRES, and FGMRES
with a finite cyclic variable-linear action table. Solver plans expose true
residual termination, immutable generation ownership, persistent workspace,
structured capability results, and provider-neutral qualification reports.

The current contract does not include nonlinear or callback-driven
preconditioners, automatic restart selection, block/recycling/pipelined GMRES,
MINRES-QLP or singular minimum-norm semantics, GPU `f64` GMRES-family
execution, variable-action CUDA Graph/Vulkan command replay, or asynchronous
single-system submission. Forge also does not construct IC/ILU/AMG,
multigrid, Schur/field-split, domain-decomposition, contact, KKT, or nonlinear
outer-solver policy. Unsupported combinations fail explicitly without silent
host staging or provider replacement.

### Sparse runtime and linear algebra modernization

- Reduced fixed sparse-runtime overhead with on-demand active-list metadata,
  adaptive traversal-list chunks, separate ambient allocation, bounded
  traversal/recycle budgets, and correct non-contiguous SNode slots. CPU list
  generation is parallel and stable sparse topologies reuse generated lists;
  CUDA coalesces duplicate activation; Vulkan bounds resident traversal lists.
- Added validated scalar sparse assembly on CPU, CUDA, and Vulkan. Builder
  insertion is bounded, CUDA/Vulkan assembly publishes a completed CSR
  generation transactionally, unsupported formats fail explicitly, and matrix
  ownership is safe across `ti.reset()`.
- Added immutable `SparsePattern.csr()` and
  `SparsePattern.bsr()` storage. Multiple matrices share canonical indices
  while owning independent numeric buffers; `update_values()` replaces values
  without rebuilding topology. BSR supports block sizes 2, 3, 6, and 12, plus
  rectangular SpMV operators. CPU values support `f32/f64`;
  CUDA/Vulkan fixed storage uses `f32`.
- Extended `SparseCG` with relative tolerance, explicit scalar Jacobi,
  fixed CPU CSR/BSR, and fixed CUDA BSR providers. Fixed providers reuse solve
  workspace and automatically refresh only numeric Jacobi/block-Jacobi state
  after a value update.
- Added CPU `SparseMINRES` for complete symmetric-indefinite
  CSR/BSR systems and CPU `SparseBiCGSTAB` for nonsymmetric CSR/BSR systems.
  Iterative solvers report convergence from the true residual contract
  `||b-Ax|| <= max(atol, rtol*||b||)`. These legacy stored-solver
  constructors remain CPU-only.
- Added `ti.linalg.LinearOperator` and the experimental
  `SolvePlan` API. Fixed stored CSR/BSR, exact compiled-kernel providers, and
  role-qualified compiled Graphs share one runtime/lifetime/capability
  contract. Explicit mathematical traits gate CG/PCG; persistent plans expose
  unified `SolveResult` terminal state and support CPU/CUDA/Vulkan CG, fixed
  stored Jacobi/block-Jacobi PCG, provider-neutral MINRES, and
  provider-neutral BiCGSTAB, restarted GMRES, and variable-linear FGMRES
  within the documented provider matrix. CPU also supports minimal operator
  composition.
- Extended compiled-kernel and Graph `LinearOperator` providers with
  `(range, domain)` rectangular shapes, independent explicit adjoints,
  `A.adjoint().adjoint()`, and shared immutable numeric generations.
  `apply()` adds the CPU generalized `alpha/beta/addend` contract with
  `beta=0` no-read semantics; unsupported GPU coefficient combinations fail
  explicitly. The provider-neutral `qualify_operator()` produces versioned
  JSON evidence with oracle, adjoint, capability, timing, and native-counter
  records.
- Extended `SolvePlan(method="pcg")` with trusted fixed-linear
  `LinearOperator` preconditioners. CPU accepts supported operator provider
  combinations; CUDA and Vulkan accept paired compiled-kernel A/M providers.
  CUDA can keep CG recurrence scalars device-resident and check convergence
  every 4 or 8 iterations. Vulkan supports the same chunk sizes and relative
  tolerance while retaining fixed-budget masked execution as its compatible
  default. Logical, executed, and wasted iterations and host checks are
  reported separately.
- Added native CUDA/Vulkan solver-chunk replay for fixed stored f32 CSR/BSR.
  CUDA captures K=4/8 CG/PCG chunks as CUDA Graphs; Vulkan records the same
  sparse recurrence as reusable command sequences, covering identity, Jacobi,
  and block-Jacobi. Values-only numeric refresh preserves the recorded
  sequence, while external output-binding or structural changes explicitly
  invalidate and rebuild it. Compiled-kernel and Graph A/M providers retain
  direct submission without a host fallback.
- Extended `SolvePlan(method="bicgstab")` with fixed-linear right
  preconditioning and CUDA/Vulkan `f32` execution. Device plans keep
  recurrence state resident, qualify terminal status with the original-system
  true residual, and report structured rho/alpha/omega breakdown reasons.
  Fixed stored identity plans reuse CUDA Graph or Vulkan command-sequence
  chunks; compiled A/M actions retain direct submission. Exact A/M, dot,
  vector, logical/executed/wasted work and persistent workspace are exposed
  through `statistics()`.
- Added restarted `SolvePlan(method="gmres")` with restart 8, 16, or 32.
  CPU supports compatible `f32/f64` providers; CUDA and Vulkan support `f32`
  fixed CSR/BSR and compiled kernel/Graph providers. Every Arnoldi step uses
  two-pass CGS with multi-dot reduction and fused projection, and every
  restart boundary verifies the original-system true residual. Fixed-linear
  right preconditioning is supported. Fixed stored identity cycles use CUDA
  Graph or Vulkan command replay; other qualified providers use direct native
  submission. Basis/workspace bytes, A/M, dot/multi-dot/vector work, restart
  cycles, happy breakdowns, and logical/executed/wasted iterations are
  reported through `statistics()`.
- Added restarted `SolvePlan(method="fgmres")` with a bounded
  variable-linear `PreconditionerPlan` action table. One to 32 compatible
  linear actions are selected cyclically by solve-global scheduled inner
  iteration without resetting at restart boundaries. CPU supports `f32/f64`
  host actions; CUDA/Vulkan support compatible device-native `f32` fixed
  stored and compiled providers. All action generations are pinned at solve
  entry, a persistent `Z` basis stores preconditioned vectors, and
  `statistics()` reports its bytes, action selections, schedule wraps, and
  update outcomes. GPU execution uses direct native submission and reports
  the unavailable replay contract explicitly.
- Added public `PreconditionerPlan` and pinned `PreconditionerSession` types.
  External approximate inverses support explicit setup, rebuild updates, and
  lagged reuse while recording built-from provenance separately from
  accepted-target compatibility. Target updates are stale by default.
  CPU/CUDA/Vulkan PCG consumes approved immutable generations without a Python
  callback in the iteration hot path. Variable-linear tables preflight every
  action before publishing an update and are consumed only by FGMRES. A 10k
  numeric-generation churn contract verifies bounded retirement; nonlinear
  behavior remains explicitly unsupported.
- Added exact CUDA conditional-Graph execution for single-system fixed stored
  f32 CSR/BSR CG/PCG. Eligible drivers and capture-composable identity,
  Jacobi, or block-Jacobi providers use it automatically through the default
  `bounded_convergent` policy. Each solve retains only its initial and terminal
  state observations, avoids per-iteration host scalar synchronization, and
  stops at the exact logical iteration without masked tail work. Unqualified
  runtimes preserve the same numerical contract through the documented Graph
  chunk fallback; explicit `host_each_iteration` remains available.
- Added homogeneous independent batched f32 CG/PCG on CPU, CUDA, and Vulkan.
  Each contiguous system has independent tolerance, status, iteration, and
  residual state; fixed stored and compiled-kernel A/M providers are
  qualified. CUDA/Vulkan plans reuse plan-owned Taichi Graphs for the stable
  iteration recurrence while retaining A/M as pinned provider actions; output
  replacement patches the Graph binding, and workspace clones own independent
  replay plans. CUDA/Vulkan fixed-budget plans also provide `submit()`,
  `SolveSubmission`, and explicit workspace cloning for bounded concurrent
  execution. Execution-policy capabilities and unsupported reasons are
  queryable; batched conditional device-convergent execution remains
  unsupported. Plan
  telemetry reports the logical workspace payload and exclusions for every
  clone; host-asynchronous completion is not a device-parallel guarantee.
- Added provider-neutral solve qualification for `SolvePlan` and
  `BatchedSolvePlan`. Versioned detached reports cover solution and true
  residual checks, terminal state, A/M identity, policy, logical/executed/
  provider work, complete preconditioner action-table provenance, chunk
  counters, transfers, resources, memory-pool deltas, and
  optional pacing. Factory construction, first solve, warm wall time, and
  qualified fixed-budget host submission are separated; unavailable device
  timestamps and driver identity remain explicit rather than inferred.
- Added `ti.graph.SubmissionPacer` for explicit cooperative cadence control
  across `Graph.submit()` and fixed-budget batch solves sharing CUDA or Vulkan.
  Global and per-lane in-flight bounds, a finite admission queue, cross-lane
  round robin, and per-lane FIFO can be configured together. Callers may choose
  blocking backpressure or rejection before backend submission. Public
  telemetry covers queue peaks, admission waits, per-lane completion, and
  backend failure. It also states that admission is measured in invocation
  counts and does not budget workspace or numeric-generation bytes. Runtime
  reset and the first completion failure have explicit invalidation semantics.
- Hardened direct-solver symbolic reuse. `factorize()` may reuse an analyzed
  pattern only when the complete compressed index pattern is identical, and a
  value update after factorization makes the factorization stale until it is
  refreshed. The stable-fluid example pins its pressure gauge, and the
  implicit mass-spring example now reuses symbolic analysis across value-only
  steps.
- The user workflow, feature set, backend/format/dtype matrix, failure
  semantics, and lifecycle rules are documented in
  [Sparse runtime and linear algebra](sparse_runtime_and_linear_algebra.en.md).
  The general operator API is documented in
  [LinearOperator and SolvePlan](linear_operator.en.md).
  See also [layout selection](sparse_layout_selection.en.md) and
  [physics solver selection](physics_sparse_solver_selection.en.md).

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
  reference comparisons are complete. In the relative measurements, histogram
  and compact are closest to the listed CUB reference; scan, reduce, and stable
  sort remain materially slower. Standard wheels select the correct,
  asynchronous, driver-only Forge provider but do not claim CUB performance
  parity. See [Native algorithms](native_algorithms.en.md) for the measurements
  and test conditions.
- Changed the 0.6.0 standard runtime-wheel validation to the `driver-only`
  dependency class while retaining loader, repair, and validation compatibility
  for already-published 0.5.0 bundled-CUDART wheels. The project still publishes
  one runtime wheel per operating system, not per CUDA version.
- The CPU native dense-field path now uses the root-child offset from the
  compiled SNode layout instead of deriving an address from preceding payload
  sizes. Alignment padding between mixed f32/f64 root children can no longer
  make `to_numpy()`, `from_numpy()`, or native field operations alias an
  adjacent field. This adds no branch or copy to the normal kernel hot path.
- Final runtime/shim wheel validation now records and validates the native
  runtime commit identity without requiring it to equal the shim source commit.
  It also exercises f32/f64 fields of shapes `()`, `1`, and `7`, host/kernel
  round-trips, serial/atomic f64 reduction, offline-cache modes, and
  single/default CPU thread configurations.
- Completed the Windows driver-only/reference build and primitive correctness
  matrices. Linux wheel/import/dependency scans, compute-sanitizer, and execution
  on each claimed older NVIDIA driver remain required before lowering any
  published driver floor.

### Bounded host memory and runtime lifetimes

- The host allocator now unmaps a non-exclusive chunk after all valid requests
  in that chunk have been released, while removing its capacity, cursor,
  alignment-waste, and released-byte accounting. Repeated creation and release
  of large/adaptive chunks no longer retains every historical OS mapping;
  chunks with live allocations remain owned as required.
- Process-lifetime internal histories that could previously keep growing now
  have explicit budgets. Blender temporary source files use a 32-entry LRU and
  remove evictions; compile/timeline traces, raw kernel-profiler records, and
  Python kernel specializations are bounded. A Program compiles at most 1,024
  specializations by default. Existing specializations remain usable at the
  limit, while a new cache miss fails clearly instead of consuming more host
  memory.
- Repeated `ti.init()`/`ti.reset()` lifecycles no longer retain launchers,
  accessors, frontend field mappings, or GFX runtime state for destroyed
  SNodeTrees. Python runtime-object registration uses weak references and the
  version-check thread starts at most once per process. Ordinary kernel, Graph,
  and UI runtime paths create no persistent helper subprocess; applications
  remain responsible for joining or terminating multiprocessing workers they
  create.
- These changes close runtime-owned unbounded-history sources; they are not a
  process-wide RSS limit. Live user fields, ndarrays, and Graphs, partially live
  allocator chunks, a finite specialization set, driver/context high-water
  marks, and the on-disk offline cache can still consume workload-proportional
  resources. See [Forge API reference](forge_api_reference.en.md#memory-growth-and-ownership-boundaries)
  and [Forge options](forge_options.en.md) for diagnostics and controls.

### Correctness, capability, and explicit support boundaries

- Version 0.6.0 completes contracts with defined correctness, safety, or
  production value in the shared CPU/CUDA/Vulkan frontend, IR, AD, AOT,
  runtime, and RHI. A full tile/block/warp/subgroup DSL, heterogeneous
  multi-device runtime,
  sparse-specific work, and new capabilities for other backends remain outside
  this scope. Their entry points must report unsupported/fail fast rather than
  pretend success through an empty implementation or silent fallback.
- Completed foundational lifecycle, capability, and observability contracts.
  Field/AD enumeration now returns only active SNodeTrees, so a destroyed
  generation cannot re-enter execution. Vulkan advertises f16/f32/f64
  atomic-add capabilities independently and does not present an unsupported
  feature as native. CUDA profiler updates aggregate only records created since
  the previous query, making repeated queries idempotent. Twelve unimplemented
  subgroup operations now report the operation, architecture, and support state
  at compile time instead of returning `None` from Python `pass` bodies.
- The native Windows build and targeted CPU/CUDA/Vulkan matrices are complete;
  GPU cases ran only while no other Python/GPU compute process was active. The
  matrix also covers fixed-dimension 3x3 tuple/vector/outer-product/matrix
  composition, first-order reverse AD through dynamic local Vector/Matrix
  reads, and 3D SVD primal boundaries including rotation, inversion,
  near-singular, and repeated-singular-value inputs. Linux GCC/Clang, headless
  Vulkan validation, CUDA driver-only
  import/execution, and real Torch AD remain pre-release revalidation items in
  the [Linux checklist](linux_revalidation.en.md). Windows results are not
  presented as cross-platform proof.
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
  sparse SNode AOT remains unsupported. When explicit trees already provide a
  valid layout, first kernel materialization neither before nor after AOT
  module creation appends an unused trailing empty root absent from metadata.
  A field-free first kernel still receives the required default empty root.
- AOT kernel templates now accept bounded ndarray/external-array exemplars on
  CPU, CUDA, and Vulkan. Specializations use a capacity-independent
  element/layout ABI key, reject unsupported or non-contiguous inputs before
  compilation, deduplicate equal contracts, and use filesystem-safe names
  with a SHA-256 fallback for long signatures.
- Vulkan storage images now select f32, i32, or u32 sampled values from their
  declared formats. Signed and unsigned r/rg/rgba 16/32-bit images have matching
  frontend, SPIR-V, and Vulkan format contracts; the prior r16u-family UNORM
  mapping has been corrected to UINT.
- Kernel launch, Graph, and AOT type checks now share one internal structural
  descriptor for scalar, vector, matrix, ndarray, texture, and StructNdarray
  arguments. Graph ndarray metadata preserves tensor element types end to end;
  StructNdarray remains supported by ordinary kernels and is rejected explicitly
  by Graph until its serialized schema can represent structured elements.
- Matrix and vector Graph arguments now use canonical rank-2 and rank-1 tensor
  shapes internally, and Graph injection caches reuse equal structural
  contracts rather than Python object identities. The 0.5.x flat Matrix and
  nested symbolic-list adapters remain accepted; genuinely mismatched rank-2
  shapes are rejected, and the 128-byte runtime limit is checked before copying.
- Unsupported rank>2 ndarray elements and out-of-contract quant widths now fail
  in Python/type validation. Quant float mirrors its native exp/significand and
  f32-compute limits without C++ assertions; arbitrary-stride external arrays
  remain explicit rejections. Graph texture descriptors validate dimension and
  RW format before compilation. LLVM 20 signed constants now preserve their
  fixed-width bit patterns instead of aborting on signed quant host access.
- Mixed-type global reductions now try a stable size-ordered TLS layout, but
  adopt it only when it strictly reduces scratch bytes. For example, f32 then
  f64 reductions in one offload use 12 bytes per TLS instance instead of 16.
  Equal-dtype reduction order remains stable, and non-power-of-two tensor
  layouts keep their original order when the candidate cannot help. This adds
  no runtime branch or device synchronization.

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
