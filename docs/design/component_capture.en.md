# Component capture compiler: initial implementation

`taichi_forge.aot._component` is an internal producer-side API. It compiles the
original decorated method through the normal AST transformer; it does not parse
the application a second time or create a numerical wrapper.

`Component` declarations describe the owner's attributes. `RuntimeValue` supports
scalar and fixed vector/matrix values, `Array` currently supports scalar Ndarray,
and `StaticValue` accepts immutable scalar/tuple configuration. Entry arguments
support scalars, fixed vectors/matrices and scalar Ndarrays without gradients.
The public SDK API is not frozen.

LLVM launches use the compiler-sized argument buffer through `RuntimeContext`;
they are not limited to the historical 64 argument slots. Python launches and
the C API now follow that storage contract, including loaded Kernel/Graph
binding. Exported ABI counts, types, layouts and resource lifetime checks still
apply. Other backends retain their existing argument-count guard. CPU/CUDA
qualification covers 72 captured Fields plus two scalar arguments, changed
dimensions and two instances under one loaded module, as well as large ArgPacks
and named Graph arguments. This does not imply a fixed per-capture cost or
qualify a full application component.

Component Graphs also accept original free Kernel helpers: ordinary
`template_args` identifies the same declared owner without a method wrapper.
Helpers without an owner parameter still use constrained capture compilation to
check global dependencies. Additional templates must be recordable static values
and enter code identity; owner identity, array addresses and shapes do not.
Explicit arrays reuse argument feature extraction and native binding checks,
without becoming component-private storage.

Internal `compile(entry, ..., owner=model, template_args={...})` compiles these
original helpers; bound methods may omit owner. CPU/CUDA checks in
`test_aot_component_helpers.py` cover reuse across owners/sizes, changed static
templates, wrong resource types and frozen Runtime consumption. These mechanism
checks do not qualify real EFC/TGS control.

`Record` declares a nested plain-data namespace while preserving original
`self.binding.field` and `ti.func` accesses. Optional presence supports only
`ti.static(record is [not] None)` and selects a code variant; scalar, dimension
and Field leaves retain their own contracts. The whole object is never a static
constant. Plain attributes and built-in frozen-dataclass slots are supported;
properties, custom descriptors and record identity comparisons are rejected.
Unused members do not enter code identity. Local dimension references receive
stable namespaces while actual roots still validate SNode groups.

Assigning an already declared child through normal `ResourceScope`, such as
`scope.buffers = buffers`, makes `Component.from_owner` collect that explicit
declaration edge recursively, without a per-field or per-kernel export table.
For a finite specialization used only by numerical entries, declare
`scope.static("key_shift", choices=(shift,), kernel_only=True)` (or the matching
`StaticValue` policy). The final exported entry closure determines whether that
value belongs in root admission; an unused sort layout therefore cannot block
an unrelated dynamics component. Used values still require the consumer's exact
compiled selection. The default remains conservative for Host recording choices,
even when no Kernel reads them. Do not mark Host branches, template-dispatch
counts or independent layout assumptions as kernel-only. Domain overlays must
preserve this policy. Family build reports retain full declaration snapshots in
`declaration_configurations`, while requested/completed coverage and selection
use the admitted root configurations. Families still require a common admitted
domain and reject ambiguous duplicate selections. This is dependency projection,
not automatic tracing of Python Host reads or inference of capacity constraints.

Normal constructors can declare dynamic values with
`scope.value("gravity", ti.types.vector(3, ti.f32))` and finite static domains with
`scope.static("mode", choices=("fast", "accurate"))`. Original attribute values
remain unchanged. Read-only record scopes support the same roles without
rewriting frozen dataclasses. `from_owner` collects roles and source locations;
separate value lists are unnecessary. Explicit `Record` overlays remain available
and may narrow a normal static domain to a component's qualified scope. Widening
the domain, changing roles or repeating value declarations is rejected. Out-of-domain values,
getters and declaration cycles are rejected. CPU/CUDA checks in
`test_aot_component_values.py` cover dynamic value/size reuse, actual static
variants, unused declarations excluded from Kernel ABI, original JIT comparison
and frozen Runtime consumption. Compiler facts expose `record_presence` and
`capture_paths`, so consumers need not decode internal IDs. CPU/CUDA tests in
`test_aot_component_records.py` cover new dimensions, aliases, presence variants,
unused members, rejected getters and Runtime consumption.

Builtin `getattr(owner_or_record, static_name)` resolves through this same
declaration path. It accepts two positional arguments, a compile-time string
name and a declared owner/record; defaults and arbitrary Python getters are not
supported. Selecting a declared Field/Array with `ti.static` creates a resource
alias, not a snapshot of its storage. Shape, contents and declared runtime
scalars remain dynamic and cannot be used as static numerical values. A static
configuration that selects a member remains a guarded code dependency. CPU/CUDA
tests cover original helper calls, nested records, aliases, changed dimensions
and frozen Runtime consumption. A CUDA GeoPhys temporal-contact entry also
passed generic and loaded execution against its original JIT definition;
complete TGS root qualification is still in progress.

Captured `Field.dtype` remains the declared scalar component type, including
vector/matrix Fields and inline helper arguments. It is guarded by the existing
type/ABI contract; changing dtype selects different code, while changing a
legal extent can reuse it. This metadata does not make shape or contents static.

A normal kernel's `ti.template()` Field argument can refer to a Dense Field
already declared by the component owner, including a declared alias. Its first
use joins the existing capture/hidden-argument path; unused Field templates do
not add resource ABI slots or specialize code by the selected resource. Actual
template-to-resource bindings enter the code contract, so exchanging source and
target cannot reuse the wrong binding. Undeclared and foreign-owner Fields are
rejected. CPU/CUDA tests cover inline helpers, aliases, different extents and
frozen Graph consumption with two instances of one loaded module. GeoPhys's
normal CUDA merge-sort stage also passes new-capacity consumption and independent
stable-order checks; this does not qualify its complete TGS root.

`ResourceScope.borrow(name, field, dimensions={...})` declares a normal foreign
Field binding without changing the Field or retaining its Python owner. For
example, `scope.borrow("mass", source.mass, dimensions={"n": scope.dim("nv")})`
maps a declared source dimension to the target's normal count. The map must cover
all dimensions of the complete source group, including layout-only siblings.
Their provenance comes from an explicit normal `FieldsBuilder` declaration;
ordinary undeclared Fields are not automatically qualified. One such builder
uses one declaration namespace. No relationship is inferred from equal integers.

`from_owner` projects the full source group into the existing native
`DenseSNodeSchema`; native SNode layout and consumer instantiation still compute
offsets, strides and allocation sizes. Only numerical captures enter the Kernel
ABI. Layout-only siblings do not become numerical inputs or producer snapshots.
Aliases preserve one root, and nested declared records qualify target dimension
names. Missing/conflicting relations, replaced sources and shape mismatches
reject. CPU/CUDA tests in `test_aot_component_borrowed.py` cover new capacities,
changed sibling offsets, aliasing, tail markers, ordinary JIT and Runtime
consumption. Adding an unused independent borrow does not change code identity.
The first normal binding names a borrowed group in its target namespace; later
aliases preserve that name.

`record = scope.record("binding", value, optional=True)` declares a read-only
record scope at the normal binding boundary. Metadata belongs to the parent,
so the original frozen dataclass stays unchanged. `record.dim("count")` names
an existing count; `record.parent_dim("nv")` explicitly relates its layout to
the parent's normal dimension. `record.borrow(...)` must refer to an existing
Field member and reuses the complete source-group contract. An absent optional
record retains its presence variant without requiring nonexistent leaves.
Assigning members or allocating Fields through this scope is rejected.

This is explicit normal resource binding, not automatic introspection of arbitrary
dataclasses or a per-kernel table. Frozen-record CPU/CUDA tests cover changed
sizes, parent relations, immutable members and absent variants. No new shared
instance-pointer switching is introduced; application control and whole-step
qualification remain separate.

Optional `StaticValue(choices=(...))` declares a finite deployment domain. This
is admission policy; numerical identity still uses only actual static captures.
Narrowed choices are checked on cache hits and direct calls too.

The component environment and each entry's ABI are separate. Attribute accesses,
including those inside inline `ti.func`, append only used captures to the native
callable. `finalize_params()` runs after this single frontend traversal; argument
initialization still ends before the numerical body. First-use order determines
the physical projection and is recorded in `contract["captures"]`. Unused
independent declarations never become parameters or code dependencies.

The candidate cache checks used declaration schemas, static values, and referenced
numerical definitions. Dynamic values and owner identity do not specialize code.
`semantic_key` identifies the frontend definition/ABI projection; the existing
native compile contract additionally keys target capabilities and effective
compiler settings. Code objects retain dependency facts, not the producer owner.
They bind actual values through the existing launch context on each invocation.
Ordinary JIT keeps its original specialization path.

Frontend definitions are published only after capture validation succeeds.
Backend compilation remains lazy and uses the existing native cache. Uncontrolled
Python captures/calls, dynamic values in `ti.static`, and unqualified `real_func`
are rejected. Inline IR reuse and automatic real-function promotion are disabled
only in this deployment path until their capture contracts are qualified.

`tests/python/test_aot_component_capture.py` covers single lowering/backend compile
across owners and sizes, unused declarations, static/helper changes, conditional
capture use, ordinary Field JIT numerical comparison, owner release, failure
recovery and Runtime retirement.

This is not yet a deployed component SDK. Dense Fields use `DenseField`
declarations, not `Array`. `Dimension` distinguishes positive extents from active
counts, which may be zero and may reference an extent as their capacity bound.
Relations come from declarations, never from equal producer integer values.

The native `DenseSNodeSchema` is the next layout building block. It records the
original root/dense/place construction and scalar placement order, replacing only
explicitly identified dense axes with dimension parameters. It stores no live
node IDs, producer offsets, addresses or model values. Serialization uses the
existing native type/JSON contracts. The first layout family excludes nested
dense, sparse, quantized, Morton, index-offset and ambient layouts.

Instantiation calls the original `SNode::dense`/`insert_children` constructors,
then `Program::add_snode_tree` invokes the existing backend StructCompiler and
allocator. On CPU/CUDA, the original LLVM types and target DataLayout determine
all byte sizes and offsets. There is no Python padding calculator. The schema
validates version, placement order, required dimensions and positive i32 index
products before allocating a tree.

`test_dense_snode_schema.py` compares every actual compiled node's type, shape,
cell size and offset to ordinary FieldsBuilder at previously unseen extents.
It includes scalar, vector, matrix and rank-zero placements, and validates that
rejected bindings do not leave active tree instances.

Original `self.field` captures now retain the complete root schema and member
paths. Sibling blocks require declared dimension provenance; logical groups
cannot split or merge physical roots. Attribute aliases of the same member share
one physical parameter. Calls bind existing zero-copy Field views, retaining the
original root, offsets and owner. They neither copy members into independent
Ndarrays nor mutate module root pointers. Qualified compact and interleaved
record members preserve the full root in the schema.

`scope.struct_field(members, shape=...)` declares a normal flat AOS StructField,
including rank-zero state and arrays with declared dynamic extents. Members use
the existing Record/DenseField capture path; aliases share an actual argument.
Dimensions retain their original declaration scope, including through aliases.
Consumers preserve the complete record through the same SNode layout rules and
bind each instance at the actual member offsets and record strides. CPU/CUDA
tests cover scalar/vector/matrix members, ordinary JIT and borrowed storage,
and different-size instances sharing loaded code. Packed Host buffers transfer
only the selected logical elements; neighboring members and inactive tails stay
intact. Raw packed-memory access cannot reinterpret a strided member handle.
Loaded LLVM entries must also carry compiler-proven runtime-stride inputs.
These facts come from the existing IR dependency pass and are preserved in
bitcode and checksummed deployment metadata, including PTX/cubin artifacts.
An old or packed-specialized entry cannot acquire this ability by changing its
binding metadata: rebuild using an affine view exemplar or component capture.
Legacy Graph calls cannot represent these views; use the qualified Kernel or
complete ExecutionPlan path. Unqualified real-function/shared-staged cases are
rejected conservatively. Packed bindings retain their existing behavior.
Nested structs, nested dense trees, component SoA, gradients and whole-struct
element indexing are not qualified by this API; Vulkan needs separate admission.

`test_aot_component_dense.py` compares the same numerical definition with ordinary
JIT on CPU/CUDA. It covers new capacities, range/Field-shape/full-Field iteration,
vector/matrix elements, aliases, empty active ranges and capacity-tail sentinels.
Different sizes share one frontend definition and one native compilation; invalid
dimension/root bindings fail before execution.

The standalone Runtime now provides experimental
`dense_snode_group(schema_json, dimensions)`. It reconstructs actual SNodes with
the same schema, then invokes `StructCompilerLLVM::run_layout`, extracted from
and also used by ordinary JIT layout compilation. Queries project actual compiled
nodes; consumers do not supply offsets/strides. The complete simple dense root
is allocated once and zeroed through the LLVM allocator. Numerical addressing
uses arguments, without registering or switching fixed module root pointers.

`group.member(paths, dtype, element_shape=...)` validates root-relative place
paths, element types and compact or positive uniform-record layout, returning
finite views of that root. Contiguous element lanes may share a larger record;
the existing affine argument ABI carries its authoritative index strides.
Paths will come from compiler-produced contracts, not application per-kernel
field tables. Groups reuse existing memory handles, borrowed leases and prepared
retirement. Closing is currently cold: wait, revoke all aliases, then release
storage. Other groups remain valid; revoked view handles remain closeable.

`test_aot_dense_snode_group.py` compares new-size layouts with ordinary SNodes
and covers scalar/vector/matrix/rank-zero transfers, aliasing, wrong Runtime,
invalid dimensions, and two groups used by loaded Kernel preparations with
interleaved execution and retirement. This qualifies simple dense storage and
its actual standalone calling path, not the full component instance contract,
nonblocking retirement or capture replacement.

The internal code object's `export(module, name)` now passes its existing
frontend definition to the AOT builder and writes a versioned entry capture
projection. It neither recompiles the decorated owner nor snapshots model data.
Residual fixed-Field dependencies reject artifact publication. Consumers query
`module.component_entry_contract(name)`; native preparation still validates the
actual calling ABI.

`test_aot_component_export.py` freezes two original Field entries, copies only
the Runtime package into an independent process and forbids DSL/test/GeoPhys
numerical-source imports. Fresh N=37/129, active=31/0 and gravity values drive
compiler-contract group construction and member binding on CPU/CUDA. Active
tails and full Field iteration are checked; the TCM hash stays fixed, with no
model payloads or fixed-root dependencies. This is an A2 mechanism probe, not a
release wheel or complete real-component acceptance.

Experimental `ti.experimental.ResourceScope(owner, builder=...)` records dimension
provenance during normal construction. Bounded declarations such as
`dim("n").maximum(1)` distinguish logical counts from allocated capacities.
Factories use ordinary Field/FieldsBuilder construction, with no Python source
interpreter or separate byte-layout calculator. Internal `Component.from_owner`
projects these declarations and actual SNode roots; individual entries retain
only their numerical and layout dependencies. Independent unused groups do not
affect ABI or code identity. Source locations are queryable but excluded from
the semantic key. Definition guards hash immutable bytecode contents rather than
marshal reference encoding, which can change during frontend inspection.
Internal Matrix DSL functions participate in the same dependency guards.

Use `scope.derived_dim("rows", facts * 6)` to relate a capacity already computed
by the normal constructor. `facts` comes from `dim` or `derived_dim` in that scope.
This declaration validates the existing integer attribute without assigning it.
Both `self.rows` in original loops and Field layout use the same relation.
Consumers supply only base dimensions and cannot override rows independently;
instantiation reuses bounded expressions and native SNode layout rules. Invalid
values, unknown sources, cycles and conflicting redeclarations are rejected.
Used attributes are checked again on cache hits and owner binding, including
nested Record paths. CPU/CUDA gates cover new capacities, zero counts, consumer
layout and original loop behavior, plus recovery after rejected declarations.

Normal algorithm validity conditions can declare
`scope.require_le("merge_coverage", scope.dim("count"), covered_items, when="scratch")`.
Bounds must reflect the selected code/layout, not arbitrary SDK-wide size caps.
The named trigger must be a declared role; only an exported numerical user brings
this inequality into its component contract. Expressions reuse the bounded
dimension algebra, including derived/record dimensions. This policy does not
add unrelated inputs to a Kernel ABI or change its numerical identity. The
component root includes any dimensions needed only to check the constraint.
Code binding validates current owner values; component instantiation validates
before native allocation, and prepare validates count updates before binding.
Errors name the component, constraint, trigger, values and declaration location.

Constrained roots use contract v5 and require `component_dimension_constraints_v1`;
older consumers reject them. Existing v1–v4 artifacts retain their contracts and
do not acquire new safeguards without rebuilding. Source locations are diagnostic
provenance; inequalities themselves participate in root identity and the existing
ExecutionPlan fingerprint. Native SNode layout remains authoritative. These
declarations do not derive algorithm validity ranges, validate arbitrary topology data,
or impose high-level component policy on raw Kernel calls. CPU/CUDA checks in
`test_aot_component_constraints.py` cover preserved code reuse, absent unrelated
constraints, no allocation on rejection, prepare updates, zero/tail behavior,
and qualified nested sources. GeoPhys uses the same contract for sort word
capacity and the coverage of its normal finite merge sequence.

Normal GeoPhys construction now provides a bounded numerical RNE gate on CPU/CUDA:
3/7/5 bodies with different topology, mass, qpos/qvel and gravity reuse the original
entry and agree with ordinary JIT. A resting chain also checks analytic gravity
moments. Body state, generalized dynamics and bias-force workspaces have explicit
normal groups shared by JIT and deployment, with no copied numerical definitions
or per-kernel resource tables.

`Component.compile_graph(normal_record)` now records original decorated entries
through GraphBuilder, collects actual captures, and exports the closure through
existing Graph lowering and `Module.add_execution_plan`. Applications do not
maintain kernel lists or binding tables. Coverage includes sequential roots with
one normal owner and the while slice below. If/switch, native nodes, captured
Ndarray instance ownership and recipe search need separate qualification. Root identity includes argument
connections and explicit types; dynamic model values do not specialize code.

Sequential subregions inherit the capture context; `Sequential.create_sequential()`
propagates it together with private resource declarations. The CPU/CUDA while
component check covers grouped work, per-step stopping masks, exact tails and
private control arrays. All numerical entries remain normal Kernel/func definitions;
there is no Python per-iteration executor. Version 4 projects control policies
from existing Graph IR and checks semantic identity plus the actual exported Plan
fingerprint. A static zero budget retains the initial condition and suffix, without
requiring unreachable body captures. Layout siblings remain in their original roots.
Component instances own their resource groups; control scratch uses existing Plan
private-resource construction.

`test_aot_component_control.py` checks budgets 0/1/3/4/5/9, empty active work, early
termination, larger consumer dimensions and tail sentinels. Changing the artifact's
loop budget fails before consumption. Finite RootOption admission stays separate
from numerical identity. Task queries describe numerical entries, not dynamic
execution counts or generated control kernels. This slice does not qualify nested
TGS, real EFC convergence or Vulkan; earlier loaded-Plan Nsight traces are not
performance evidence for this new component capture path.

Runtime `module.component(name)` queries the immutable contract;
`instantiate(dimensions)` creates independent roots using authoritative native
SNode rules. `initialize(data, zero=...)` explicitly supplies model data or zeros.
Version 2 projects resource access from the compiled Graph; unknown effects stay
conservative and version 1 remains readable. Every referenced resource still
requires initialization: write-only access does not prove full-capacity coverage
or justify skipping tail initialization. No producer live state is serialized.
`instance.prepare(values=...,
dimensions=...)` binds through the existing ExecutionPlan and PreparedCall, with
no new Python execution interpreter. Fixed vector/matrix values reuse native ABI
encoding; every entry consuming a named value must have compatible value ABI.

Each prepared call owns its resource/value bindings, without changing shared
Module root pointers. Instances retain separate state under one loaded code
owner. Ordinary CPU execution and independent CUDA domains are qualified for
this finite component. Use `instance.prepare(..., execution_domain=domain)`,
then `call.initialize()` before obtaining completion tickets with `call.submit()`.
A new preparation may change non-layout active counts and values; each binding
and capture keeps its own snapshot. Submitted descriptors are never rewritten.
One instance's persistent state permits only one ordered domain (or ordinary
execution lane) while bindings remain live, not arbitrary concurrent mutation.

Close and reset remain blocking cold operations. Close first revokes every
binding's submission rights, then uses native per-call retirement without holding
the Python Runtime submission lock while waiting. Storage cleanup may still
synchronize the device; the entire close is neither nonblocking nor guaranteed
to leave other queues unsynchronized. Retire existing preparations before
reinitializing; changed capacity requires a new instance. `call.release()` drops
only the host handle: domain ownership retains the instance and roots and blocks
reset until domain retirement. Peers remain usable. Closing the code Module
rejects new submissions and finishes accepted work. Failed preparation does not
switch instance state or reuse a partial binding.

GeoPhys `build_bias_force_graph` serves ordinary prestep Graphs and component
building. CPU/CUDA probes freeze a TCM, create two new models with different
topology, mass, qpos/qvel and gravity, copy only Runtime into an isolated process,
and forbid private numerical-source imports. One loaded module serves two
independent instances with interleaved submissions, close with accepted work,
cold reset and code-module close. A ball-joint model with bodies=5, nq=20 and
nv=15 verifies independent dimensions beyond equal-sized hinge examples.
World-state inputs are prepared by normal JIT kinematics outside that process;
this proves isolated finite RNE consumption, not source-free model/FK/whole-step
construction. Initial cost samples separate load, allocation, initialization,
prepare and warm component execution; no speed or scaled-memory claim follows.

`contract["compilation"]` also exposes the existing compiler's final task
manifest: work-range sources, selected block/grid, mapping and launch-bound
facts. This cold query adds no range interpreter. Its
`compiled_not_observed_replay` label separates compiled choices from observed
submission. Unknown ranges or actual grid sizes remain unknown, not zero or
producer constants. Multiple loops, ordinary while/continue and invocation-value
extraction preserve known resource identities. Atomic effects retain their
synchronization/fusion restrictions; unknown side effects remain conservative.

The finite RNE root projects read-only model inputs, bias-force writes and atomic
workspaces on CPU/CUDA while retaining isolated-consumer correctness. A separate
producer-N=7 probe consumes C=257/B=251 and B=0 with one 64-thread CUDA block;
active tails, capacity traversal and six sentinel elements remain correct.

Local Nsight Systems observation of frozen artifacts shows one CUDA Graph replay
per RNE call, containing two numerical kernels. All three tasks in the small-grid
probe actually launch at 1x64 and cover the new range with grid-stride loops.
Each five-replay region waits only at its explicit completion boundary, with no
per-kernel Host callback submission observed. Empty active work still launches
an empty kernel; zero-launch execution is not claimed. These are scoped physical
submission observations, not a speed advantage or general execution-domain and
concurrency qualification. Compiler manifests alone do not establish them.

Finite-choice roots require a normal bound recording method. Configuration is
frozen before Python graph construction and checked again afterward, including
choices which only affect the dispatch sequence. Such choices never enter the
ABI or numerical identity of kernels that do not read them. Unused choices and
widened domains do not recompile numerical code. Root code identity is separate
from its fixed exported configuration; identical code does not prove identical
model admission.

Version 3 adds `configuration`, containing finite choices and this root's
selected values; v1/v2 remain readable. Consumers supply actual normal-model
configuration to `instantiate(dimensions, configuration=...)`. Missing inputs,
out-of-domain values or a different exported selection reject before native
allocation; bool and int remain distinct. The finite RNE root explicitly limits
four unqualified algorithm switches to False, including in its isolated
consumer. Shape compatibility alone cannot admit another algorithm. The normal
model-preparation layer still owns consistency between topology data and these
declarations.

`RootOption(choices=(...))` declares a normal recording argument, such as
`report_power`, in the same environment. Its value comes from normal call
arguments or defaults, not a numerical `self` capture. It uses the same v3
configuration contract; selection never executes Python graph construction in
the consumer. The normal RNE power diagnostic already passes CPU/CUDA isolated
consumption: its new kernel and rank-0 resource enter the closure automatically.

Internal `_component_variants` provides `Variant(env, normal_record, **options)`,
`BuildBudget` and `build_variants`. Requests describe legal normal configurations,
not execution stages or an automatically enumerated Boolean product. Builds
record actual kernel semantic variants and reuse relationships. Each request
records its normal Graph once; existing lowering, ExecutionPlan export and
module kernel deduplication remain authoritative. Query built roots through
`family.select(configuration).contract`.

Build counts and elapsed time are reported without default hard limits. Callers
can opt into `BuildBudget` limits for configuration count, actual kernel variants
or elapsed time; each omitted/`None` limit is disabled. These are build-cost
controls, not algorithm correctness or model-size constraints. The variant limit
is checked after frontend dependency discovery and before backend compilation.
Time limits do not preempt an active native
compiler. `ComponentBuildError.report` distinguishes completed requests, known
missing requests and whether admission exhausted the input. Incomplete families
cannot be returned/exported. Complete frontend definitions can be reused after
failure, but do not count as completed artifacts. Partial native export failures
retain the existing rule forbidding module publication.

`family.export(module, name)` publishes an index and actual entry closure.
Runtime `module.component_variants(name)` checks it against root static choices,
code identity and ExecutionPlans. `select(configuration)` returns an existing
component; `instantiate(..., configuration=...)` binds it under the same loaded
module. Missing inputs, out-of-domain values and legal but unexported combinations
reject without source fallback, repeated loading or an execution interpreter.
Coverage is `exact_requested_configurations`, not all combinations of choices.

CPU/CUDA RNE qualification covers plain and power-reporting roots sharing RNE
code, with two numerical kernel entries in one artifact. New-size models select
from the frozen artifact in Runtime-only consumers. Broader algorithm domains,
full initialization/constraint/provenance projections and complete C lifetimes
remain in progress. Wheels and whole-engine qualification remain in E; this
batch builder is not a cross-platform or distributed compilation service.
