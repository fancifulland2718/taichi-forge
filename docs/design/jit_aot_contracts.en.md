# JIT and AOT internal contracts

This describes the first contract consolidation in Forge 0.6.4. It is a
maintenance guide, not a new public plugin ABI or a claim that every native
operation can be reconstructed. [中文版](jit_aot_contracts.zh.md)

## Authority and ownership

| Fact | Authority | Consumers and boundaries |
| --- | --- | --- |
| Effective kernel compilation options | `KernelCompileRequest` | Existing JIT and AOT compilation entries; target/cache identity work remains separate. |
| Actual LLVM call ABI and resource dependencies | `KernelCallContract` in compiled data | JIT registration and AOT metadata project compiled facts. Source `Kernel` is not a second ABI source after compilation. |
| GFX call ABI and dependencies | Compiled GFX attributes | Preserve backend layout differences; no universal physical ABI. |
| Selected Graph execution structure | `graph/_lowering.py` | Traversal, physical operation boundaries, binding rules and control lowering are shared. |
| Deployment publication and snapshots | AOT `_ExportContext` | Export transaction, structure/Field publication, data capture and post-compilation root closure. |
| Selected JIT code and live storage | JIT `_JitContext` | Borrow frozen physical dispatches and validate live owner generations. No fabricated AOT Module. |
| Native semantic definition | Provider's normalized definition and native schema/binder | Python creation, recording and reconstruction consume the same semantic facts. |
| Workspace, handles, submission and retirement | Existing prepared/execution owners | Shared definitions do not imply shared mutable state. |

The LLVM contract lives in
[`kernel_call_contract.h`](../../taichi/codegen/llvm/kernel_call_contract.h).
LLVM task/module ownership and serializers remain backend-specific. AOT root
relocation still clones producer code before mutation; its launch ABI is
projected from transformed code. Source-analysis Graph metadata and deployed
root-tree identifiers have different meanings and must not be remapped together.

The common [`GraphLoweringContext`](../../python/taichi_forge/graph/_lowering.py)
is producer-side cold construction state. It has no Module publication or
snapshot transaction. `lower_frozen_graph()` preserves `operation_ends` for
submission recipes; traversal alone does not finalize an export. AOT confirms
the actual compiled resource closure before publishing snapshots. JIT references
the already selected code and launch policies. Runtime-only consumers read
`ExecutionPlan` descriptions without importing this DSL layer.

Ordinary kernel JIT and `Graph.run()` keep their execution/submission paths.
The shared JIT plan path remains the qualified fixed CUDA execution-domain
route. `.tic` caches and `.tcm` deployment containers keep their distinct
ownership and failure policies.

## Adding a native operation or external library entry

Use an operation/resource as the extension unit, rather than a base class for an
entire external library. Reuse
[`NativeOperationRegistry`](../../taichi/runtime/native_operation_registry.h),
`NativeOperation`/`NativeResource`, existing recordings and Graph adapters.

1. **Normalize and freeze semantic parameters at creation.** A typed provider
   definition may wrap [`NativeDefinition`](../../python/taichi_forge/graph/_native_definition.py).
   It contains provider/operation/version and pointer-free parameters. Defaults
   and layout decisions belong to the operation, not its exporter. Live handles,
   mutable data and owners belong to bindings/resources.
2. **Describe ports and resource dependencies once.** `NativeOperationBinding`
   connects `NativeArrayBinding` roles and `NativeResourceBinding` owners. Its
   default adapter supplies symbolic names, array effects, resource references
   and the native description to the lowering context. A provider may append
   additional workspace/resource effects required by its existing recording.
3. **Keep concrete admission at the cold boundary.** Verify the actual variant,
   backend and reconstruction policy before `binding.lower(context)`. Retain
   exact-class `_export_aot` admission where a subclass could add callbacks or
   epilogues. Inheritance alone does not certify complete exported behavior.
4. **Register schema, typed binder and backend factories.** Schema decoding
   checks semantic parameters and symbolic ports without allocating workspace.
   The binder applies common type/layout rules when binding facts are available.
   Device/range/alias checks follow actual storage resolution. Probe and factory
   stages retain their provider-specific availability and preparation policies.
5. **Retain the existing direct callable.** A typed JIT adapter can use the
   frozen definition and common normalizer directly. It need not serialize JSON,
   query the registry or recreate a vendor plan on each invocation. Release
   newly retained binding references on close, under the existing owner policy.

For an extension expressible by this contract, edits belong mainly to the
operation definition, schema/binder, factories and tests, plus explicit registry
installation/build entries. Do not add a separate semantic branch to common
Graph lowering or the prepared executor. New execution or ownership semantics
may require a deliberate shared-contract extension; the descriptor does not
automatically make arbitrary Python callbacks deployable.

## Concrete examples and retained policies

[`hardware/_fft.py`](../../python/taichi_forge/hardware/_fft.py) freezes ordinary
cuFFT plan construction in `_CufftPlanDefinition`. Creation and reconstruction
use it; there is no separate `_aot_plan_parameters`. A recording binds the
execute definition to the plan resource and input/output ports through the
default adapter. Vendor handle/cache, capture and workspace ownership remain
with their original JIT/AOT owners. Callback and separable variants retain their
own qualification boundaries.

[`hardware/_cublaslt.py`](../../python/taichi_forge/hardware/_cublaslt.py) and
[`hardware/_cutensor.py`](../../python/taichi_forge/hardware/_cutensor.py) construct
vendor plans from frozen typed definitions. Recordings and AOT connect ports and
plan resources through the same default adapter. cuBLASLt freezes the selected
algorithm, layout and epilogue at its existing materialization boundary.
cuTENSOR direct calls retain per-call alpha/beta; a recording freezes its own
coefficients. Materialized workspace requirements constrain reconstruction
without replacing semantic parameters or rerunning selection during export.

`NativeResourceBinding.arrays` describes a resource's own array inputs. VkFFT
creation, detached sources and reconstruction consume one frozen definition
and the selected batch tile. Child resources use resolved
`NativeResourceReference` symbols; the existing context deduplicates owners
without expanding a shared DAG at every use. cuSPARSE SpMM/triangular parameters
and recording ports also use shared definitions. Matrix snapshots remain an
export transaction; frozen parameters do not freeze live numeric values.

CSR/BSR support `resource_bindings={matrix: "name"}` for a consumer-created
native matrix, without reading back or packaging its contents. The matrix owns
its values and cannot also map to a separate `source_bindings` coefficient port.
Triangular execution sharing a public CSR keeps private analysis and refreshes
it before every solve. Public triangular plan handles remain unqualified for
export. Existing resource/alias checks govern explicit public identities;
different owners with equal configurations are not implicitly merged.

[`_primitive_definitions.py`](../../python/taichi_forge/algorithms/_primitive_definitions.py)
provides cold definitions for Scan, Reduce and Transform. Ordinary native calls
consume their dtype, extent and operation/coefficients; cached plans retain the
definition for AOT after concrete layout and selected-method qualification.
An unexecuted sequence can construct the same definition without executing the
algorithm. Transform encodes legal NaN/Inf/signed-zero coefficients as bits;
integer coefficients retain the existing binding's exact-representation limits.
Layout admission and backend/workspace policies still have separate validation
stages. This migration adds no AOT layouts, and warm replay directly invokes
the cached method and arguments.

Gather/Scatter, ScatterAdd, Histogram, scalar GroupedReduce, BucketBuilder and
check/metric also use this definition layer and the default binding adapter.
Packed GroupedReduce retains its clear-then-scatter composition; CUDA grouped
aliases retain their selected atomic route. Ordinary BucketBuilder still accepts
raw StructNdarray payloads, while AOT retains its existing numeric-layout admission.
Check/metric keep private scalars and existing layout admission without adding
autodiff support. CPU thread counts and Vulkan export preferences remain separate
preparation policies.

Ordinary cuBLAS GEMM, recordings and export share `GemmDefinition`, retaining
existing handles, submission and beta-dependent output access. Dot/axpy/scal/nrm2
already have native descriptions and factories; this does not add ordinary Python
recordings for them. Shared definitions do not imply new backend, layout or
autodiff qualification.

Ordinary CUB source plans and selected segmented capture share frozen operation,
extent and external binary/target/driver facts. Selected CUTLASS recordings and
capture share the same NativeDefinition, including actual strategy and workspace.
Both exporters connect resources through the default adapter without querying
providers again. Manifest projections carry deployment requirements; external
addons remain separately installed. SPIR-V/PTX reuse existing immutable assets
and content identities. Topology bytes and workspace retain separate lifetimes;
large binaries are not ordinary parameters. Concrete shader, pipeline and AS
closures still require their own adapter qualification.

Image copy/blit and buffer-image transfer share region, mip and offset/pitch
definitions between direct prepared commands and the default native adapter.
Array and resource references retain precise ResourceEffects across symbol
relocation. Texture creation, sampler/mip state and ordered TextureCollection
members also define reconstruction. Texels, consumer seeds, first full writes
and public aliases remain the lowering context's responsibility; definitions
neither allocate nor initialize resources.

SPD creation and reconstruction share fixed image layouts, reduction and the
existing SpirvShader asset. The expected-plan certificate, public plan aliases,
counters and retirement retain their existing contracts. Small source-free and
borrowed checks include explicit downstream gradients; this does not add
automatic differentiation to image operations or SPD itself.

[`_prepared_primitive.py`](../../python/taichi_forge/algorithms/_prepared_primitive.py)
and [`_prepared_compact.py`](../../python/taichi_forge/algorithms/_prepared_compact.py)
use that same adapter for Sort/Compact. Creation and recorded effects derive
from frozen roles; reporting dictionaries are not semantic authority.
[`native_primitive_contract.h`](../../taichi/runtime/native_primitive_contract.h)
exposes common cold binders/validators for the direct JIT and native schema
paths. The direct GPU driver paths remain intact.

| Policy | Preserved behavior |
| --- | --- |
| JIT Sort/Compact preparation | Validate storage without sorting/compacting or allocating scratch. First execution may prepare workspace. |
| JIT workspace | GPU workspace remains Program-owned; CPU Sort workspace remains plan-owned. |
| AOT Sort workspace | Independent instances own state; compatible sorts share scratch only in the existing ordered region. The executor retains this policy branch, not duplicate input validation. |
| NaNs | CUDA JIT still accepts `bitwise`; reconstructed Sort accepts `last` and rejects unsupported policy at cold admission. |
| Public backend support | Prepared Compact remains CUDA/Vulkan. The native CPU factory does not automatically add a public CPU prepared API. |
| Legacy plan representation | The reader normalizes `kind="sort"/"compact"` into the same native schemas. Prepared exporters now write native envelopes. |

The registry's typed `bind()` projection lets an owner apply a preparation
policy before `prepare()` performs device validation and construction. It does
not replace preparation or authorize device access. Required binding groups are
bounded by the supplied binding inventory, not an arbitrary column limit.

## Review evidence

Review one real creation/recording/export/consumer slice before expanding an
interface. Existing tests cover frozen cuFFT parameters, ordinary replay without
description conversion, prepared tensor/Field/view Sort with explicit downstream
gradients, and multi-column Compact with private instances and source-free
consumption. Keep checks for:

- no extra serialization, registry lookup or scratch preparation during warm replay;
- selected physical dispatch and launch policy surviving lowering;
- repeated export without changing the producer;
- explicit sharing, independent private state, failure rollback and retirement;
- a pre-change artifact consumed by the new reader, in addition to round trips.

Useful tests are `test_graph_execution_domain.py`,
`test_aot_graph_recording_export.py`, `test_aot_fft_recording_export.py`,
`test_aot_prepared_tensor_sort.py`, `test_aot_prepared_compact.py`,
`test_aot_cublaslt_recording_export.py`, `test_aot_cutensor_recording_export.py`,
`test_aot_scan.py`, `test_native_arithmetic_definitions.py`, and the native
schema/workspace tests under `tests/cpp/aot`. These are representative contracts,
not an exhaustive backend/provider support matrix or a performance claim.
