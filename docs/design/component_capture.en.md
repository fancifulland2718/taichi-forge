# Component capture compiler: initial implementation

Producers use the public `ti.aot.Component` entry; `_component` remains its
implementation module. It compiles the original decorated method through the normal AST transformer; it does not parse
the application a second time or create a numerical wrapper.

`Component` declarations describe the owner's attributes. `RuntimeValue` supports
scalar and fixed vector/matrix values, `Array` supports whole managed Ndarrays,
and `StaticValue` accepts immutable scalar/tuple configuration. Entry arguments
support scalars, fixed vectors/matrices and managed scalar/tensor/StructNdarrays
without gradients.
Declare resources, dimensions, and value roles at normal allocation sites with
`ti.experimental.ResourceScope`, then collect them with
`ti.aot.Component.from_owner(owner)`. The declaration protocol remains experimental;
the per-layout, backend, capture, and control qualifications below still apply.
Exposing the producer entry does not expand these qualifications.

```python
component = ti.aot.Component.from_owner(owner)
root = component.compile_graph(owner.record_step, initializer=owner.record_initialize)
module = ti.aot.Module()
root.export(module, "step")
module.archive("component.tcm")
```

The recording methods are the application's normal JIT Graph definitions.
Build finite normal configurations with public `ti.aot.Variant` and
`ti.aot.build_variants`. Optional `ti.aot.BuildBudget` axes default to `None`;
catch `ti.aot.ComponentBuildError` to inspect an incomplete build report.
`ti.aot.StaticValue` can narrow a normal static domain, and `ti.aot.RootOption`
declares Host recording options. Standalone consumers use
`taichi_forge_runtime.aot.Runtime` with `module.component(name)` or
`module.component_variants(name)`; they do not import these producer types.

Module constants can opt into static capture with a `typing.Final`
annotation, for example `MATERIAL_ID: Final[int] = 3`. Only immutable scalars
and tuples are accepted. Original JIT expressions remain unchanged; actual
reads in the entry and nested helpers contribute their typed values to code
identity and the queryable `constants` facts. Candidate reuse and publication
recheck both the declaration and value, so rebinding a constant invalidates the
producer cache rather than silently reusing old numerical code. Unused constants
add no ABI inputs or identity dependencies. Unannotated globals, mutable values,
and same-named closure captures remain rejected. Deferred annotations from
`from __future__ import annotations` support imported `Final` and a `typing`
module alias. The compiler checks only the outer annotation role and its actual
import binding; it does not evaluate type parameters, annotation calls or
arbitrary Python getters. Replacing the import binding also invalidates reuse.

Primitive dtype attributes can use `scope.static_dtype("precision", choices=(ti.f32, ti.f64))`
at their normal selection site. Only existing AOT scalar types are qualified;
builtin Python `float`/`int`, tensor types and arbitrary type objects are excluded.
Actual reads contribute a typed `dtype` fact, separate from string configuration.
An unused type selection adds no numerical dependency. Loaded consumers select it
with `taichi_forge_runtime.aot.StaticDType("f64")`, which requires no DSL import.
Producers keep the actual DSL type; consumer markers cannot replace it in captures.
Changing precision selects different numerical code; dynamic model sizes can
reuse that code. CPU/CUDA gates cover rounding differences, cache guards and frozen
artifact consumption. This does not change backend support for the selected dtype.

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

`scope.struct_field(members, shape=...)` declares a normal flat StructField,
including rank-zero state and arrays with declared dynamic extents. AOS is the
default. With the implicit root, `layout=ti.Layout.SOA` uses the original
`Struct.field` allocation: one dense block per member, with the usual vector
and matrix element layout. Explicit builders/placement currently require AOS;
an unsupported combination is rejected before allocation. Members use
the existing Record/DenseField capture path; aliases share an actual argument.
Dimensions retain their original declaration scope, including through aliases.
Consumers preserve the complete record through the same SNode layout rules and
bind each instance at the actual member offsets and record strides. CPU/CUDA
tests cover AOS/SOA scalar/vector/matrix members, ordinary JIT and borrowed storage,
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
Implicit-root SOA records can also share a fixed SNode root with sparse fields.
CPU/CUDA tests preserve that mixed root, whole-record indexing and member aliases,
then consume new values in two instances without the producer's numerical source.
This fixed schema does not make the sparse tree or its dense capacities dynamic.
Nested structs, nested dense trees, component-wise vector/matrix SoA and gradients
are not qualified by this API. Whole-struct element indexing requires the fixed
SNode path below; Vulkan needs separate admission.

`scope.ndarray(dtype, shape=...)` declares an independent normal managed array.
Scalar, vector/matrix and fixed-AoS StructNdarray elements retain their existing
storage types. Pass these arrays to the original Kernel's ordinary ndarray
parameters; a normal recording can use
`builder.dispatch(self.update, self.records, template_args={"self": self})`.
Both root and Sequential recordings accept concrete managed arrays: ordinary
Graph retains them through its existing fixed bindings, while component export
derives instance resource bindings from their normal declarations. Neither path
introduces a numerical wrapper. Direct implicit `self.ndarray[i]` access is not
a newly supported normal JIT syntax.

Component contracts distinguish independent arrays (`kind="ndarray"`) from
SNode group members. Struct layout names refer to the existing native named AoS
layout registry; consumers query `module.ndarray_layout(name)` for member types,
offsets, alignment and stride. No Python layout calculator is introduced.
Declared dimensions and aliases retain their original scope, including across
records. Resource-triggered capacity constraints survive explicit array calls.
Each instance allocates its own arrays, uses explicit consumer initialization,
and shares the existing prepared-call retirement rules. `instance.nbytes`
includes independent arrays once per allocation as well as complete SNode roots;
it excludes driver overhead. CPU/CUDA gates cover mixed Dense/StructNdarray
state, nested structure members, new extents, alias/type/shape failures and
same-module CUDA Graph instances. Member-view captures and gradients remain
separately qualified features.

Vulkan now has an array-only component gate using scalar Ndarrays and fixed-AoS
StructNdarrays through the existing GFX arguments and primary queue. After
exporting capacity 7, `test_aot_component_vulkan.py` creates capacities 53/263
in a Runtime-only isolated process. It checks the original initializer/step,
aliases, block-boundary coverage, empty active ranges, untouched tails and
independent reset/close. Cold initialization and ordinary prepared-call close
use the existing Runtime completion wait. Closing one instance can drain the
primary queue; this does not qualify independent asynchronous domains or
synchronization-free retirement performance.

Simple Vulkan Dense groups have a separate native-layout and source-free
consumer gate. Every node at new extents is compared with ordinary FieldsBuilder,
including scalar/vector/matrix/rank-zero placement. Packed Host IO for interleaved
records copies only member bytes through one staging buffer and one cold
submission. Strided regions require four-byte aligned offsets, strides and
element widths; unaligned subword members reject explicitly. The original
`self.field` component covers interleaved StructField members, sibling blocks,
aliases and count/shape/Field loops. Two capacities (53/263) share one loaded
module and check tails, initialization, repeated runs and independent reset/close.
Fixed SNode/Sparse instance calls, complex layouts and real applications still
require their own Vulkan qualification.

GFX metadata v17 carries final-IR runtime-affine argument facts. Corresponding
declarations are retained in the SPIR-V and checked on load; changing JSON alone
cannot grant stride support to packed code. This is not an artifact signature
or a security guarantee for arbitrary hostile shaders. The shared launch boundary
also checks the actual reachable relative-byte range against SPIR-V's signed-i32
offset arithmetic. A 64-bit base pointer does not widen that arithmetic; LLVM's
wider addressing is preserved. This is a code-generation ABI bound, not a
configuration or model-count budget.

Fixed-schema Sparse SNode component binding reuses the native runtime foundation.
`clone_snode_tree_runtime` recreates empty
state through the executor's original `SNodeTreeLayout` construction recipe;
values, activation sets and allocator cursors are not copied. Code-slot bindings
require that exact construction provenance and live code/state generations.
Reusing a retired tree ID gives the new instance a fresh generation.

CUDA `SparseRuntimeView` maps those slots in a private directory and retains
private compact-root bindings, temporary storage and diagnostics. Both root
addressing paths select the same instance; neither registered code nor the shared
module directory is rewritten. Existing tree loans protect allocator/listgen
managers until ordered retirement. Allocation is a cold operation and retains
the existing requirement to return outstanding sparse views before changing
allocator storage. CPU/CUDA gates cover the embedded mapping transaction; real
CUDA kernels cover pointer/dense activation, traversal, deactivation, independent
retirement and single/multiple roots under one loaded module.

The standalone Runtime now exposes `module.snode_instance()` and
`module.prepare_execution(plan, bindings, execution_domain=domain,
snode_instance=state)`. The state owns empty clones of the module's fixed trees,
retains the exact loaded code/prototype owner, and reports actual code/state tree
IDs through read-only `state.info`; LLVM also reports tree generations. Vulkan
IDs are diagnostic mappings, while native-resource epochs control lifetime.
Preparation derives an
immutable mapping for the kernels actually used by the plan. Numerical
initialization and control continue through the original ExecutionPlan; mutable
submission parameters remain supported on CUDA. CPU uses the same call without
`execution_domain`: its prepared contexts retain an immutable private runtime
directory and execute synchronously through the existing CPU launcher. The
module's shared pointers and registered code are never switched between states.
Fixed module-root array aliases cannot
be mixed with this instance binding.

Vulkan uses private Kernel resource views sharing the loaded executable
pipelines. Its default-domain preparation and additional-queue domains reuse
the existing ABI binder, sparse admission and completion leases. Queries project
the same validated resource recipes used by JIT/AOT reconstruction, normalizing
only binding identities; allocator sizes, offsets, capacities and policies stay
intact. Component export uses the native artifact's final tree relocation map.

Normal declarations can retain a fixed sparse placement without changing the
numerical kernel or helper signatures:

```python
builder = ti.FieldsBuilder()
scope = ti.experimental.ResourceScope(owner, builder=builder)
leaf = builder.pointer(ti.i, 4).dense(ti.i, 4)
scope.grid = scope.field(ti.i32, shape=16, placement=leaf)
builder.finalize()
component = Component.from_owner(owner)
code = component.compile(owner.evaluate, output, 0)
code.export(module, "evaluate")
```

The explicit placement keeps a fixed tree schema; dynamic `Extent` declarations
are rejected here. Runtime counts remain separate value/dimension declarations.
Captured Fields and their aliases retain ordinary indexing, struct-for and
activation/deactivation lowering. The contract projects the same native
StructCompiler allocation layout used by fixed AOT and runtime cloning, plus
the native structural access fingerprint; no Python byte-layout computation is
introduced. Binding IDs/generations stay outside the semantic code identity.
Producer definitions still guard their exact source tree bindings, while loaded
code serves independent consumer instances through the existing prepared API.
Entry export verifies every compiled tree dependency and never snapshots live
values or activation. Sparse tree schema changes require another code
variant and are not advertised as dynamic capacity.

Placement may also come from the ordinary current `ti.root`. Explicit scopes
validate their own FieldsBuilder; default scopes validate the current pending
root, rejecting a materialized or foreign builder's placement. Fixed
`struct_field` declarations preserve `state[I].member` and grouped struct-for:
the existing AST resolves declarations when indexing or iterating, then uses
ordinary StructField lowering without another field/address interpreter.
Member-first `state.member[I]` access still captures only that member. Whole
element indexing constructs all member references as the ordinary frontend does
and retains the full fixed group. Whole elements of parameterized dense structs
still require separate qualification.

`snode.deactivate_all(builder=builder)` records the same cleanup in an ordinary
Graph or Sequential region. Omitting the builder still calls the original library
kernels immediately. Both paths retain child-before-parent order and the fused
two-level cleanup; no numerical cleanup definition is duplicated. Component
capture resolves SNode templates as actual ancestors of declared fixed Fields,
preserving group and ancestor relationships and rejecting undeclared foreign trees.
Consumer execution uses the existing immutable instance tree bindings. CPU/CUDA
gates cover pointer/bitmasked fusion and pointer/dense recursion, ordinary JIT,
deferred recording, before/after readback, repeated activation and independent
instances. Other sparse tree families retain their separate qualification boundaries.

An ordinary component recording through `component.compile_graph(owner.record)`
collects its actual entries and fixed-tree dependencies. `root.export(module, name)`
exports them without a per-kernel list. The v6 root contract distinguishes
`fixed_snode` groups from parameterized dense groups. Consumer bindings come from
the native module's final dump relocation map, including entries exported after
the component was recorded. Before allocating instances, the consumer compares
the contract with normalized `SNodeTreeLayout` facts from the plan's compiled
dependencies. Recomputing JSON identities cannot bypass that native layout check.
Where sparse Graph lowering lacks per-member effects, the contract reports
conservative group access, never an absence of reads or writes.

`module.component(name).instantiate(dimensions)` creates the required empty fixed
trees and binds them alongside the instance's dense groups through the existing
prepared API. A v7 root also exports coordinate transfers for its actual captured
members, using the same library kernels as ordinary `Field.write_at(coordinates,
values)` and `read_at(coordinates)`. Consumers call
`instance.resource("grid.mass").write_at(coords, packed_bytes)` or `read_at(coords)`;
the latter returns contiguous element bytes without importing the numerical DSL.
The ordinary Field API accepts values of shape `(count, *element_shape)` and
returns NumPy arrays. Writes require distinct coordinates and preserve normal
SNode activation. Reads do not activate cells. Empty batches are valid; invalid
coordinates or payload sizes reject before submission. Public nested member paths
come from compiler declarations rather than decoding internal names.

Instance binding includes the validated transfer entries as well as the original
numerical and initializer closure. This allocation-only projection is never
executed; it neither inserts transfers into the step nor includes unrelated
module trees. Vulkan therefore creates private resource views for cold uploads
and reads without changing the module's shared bindings.

These are synchronous cold transfers: close the instance's prepared calls first.
They use existing instance bindings without changing shared module root pointers;
they are not qualified as efficient per-step or general asynchronous transfers.
Original component entries can still initialize, reset and deactivate state.
Sparse storage is not exposed as an uploadable dense view;
`initialize(data, zero=...)` still applies only to qualified dense/array resources.
The extra library entries appear in build statistics. There is no default count
or time limit; only an explicitly supplied BuildBudget limits construction cost.
Older v6 artifacts remain consumable but cannot acquire transfers they did not export.
`instance.nbytes` explicitly rejects sparse instances because fixed
root bytes do not measure total dynamic allocator storage. Contract loading has
no separate fixed JSON size cap; version, capability, closure and native layout
validation still apply.

`module.snode_instance(plan=plan)` allocates only the fixed-tree dependencies of
that plan's compiled entries, including initialization and nested control
regions. Entry aliases resolve to the same compiler-owned dependency facts,
which remain available after code loading. Trees outside this closure are not
cloned for the instance; the loaded module still owns its prototype resources.
Preparing a different plan that needs an omitted tree fails before execution.
This is cold resource selection over the existing ExecutionPlan, not a new
execution path or a sparse Field-to-Ndarray conversion. No numerical initializer
runs during instance creation. Omitting `plan` preserves whole-module cloning.

Closing a state retires only its dependent calls, including accepted submissions;
another state in the same execution domain remains usable. Closing the public
module prevents new calls and drains its prepared users; remaining state handles
retain the code/prototype lease until their own retirement. One state cannot be
lent to concurrent prepared views. After a view retires, another plan can borrow
the same state without resetting it. CUDA allocations still require returning all
outstanding sparse views at a cold boundary. CPU uses the stable host allocator;
creating a new state does not require retiring another state's prepared view.
CPU preparation requires the compiled sparse-view qualification and rejects
random, external, mesh, print or unknown runtime-state uses. Existing fixed AOT
artifacts without that qualification can still use ordinary calls, but must be
rebuilt for CPU instance binding (offline-cache schema 99). Private assertion
diagnostics are checked through function addresses resolved during preparation.

`test_aot_snode_instances.py` exports ordinary pointer/dense and bitmasked/dense
kernels, freezes the artifact, and consumes it in a physically runtime-only
`-I -S` process. It covers empty activation/value state, distinct initialization,
multi-root traversal/deactivation, same-domain interleaving, native foreign-owner
rejection, failed preparation, per-state close, generation reuse, and module
close with accepted work on CUDA. CPU additionally covers loops, allocation while
a sibling view remains live, failed RNG admission, and assertion isolation. The
original module state and loaded kernel handle remain unchanged.
`test_aot_component_sparse.py` additionally covers normal fixed placement,
implicit Field/helper capture, source-binding guards, stable deployable identity,
and runtime-only entry consumption on CPU/CUDA. A normal mixed sparse/dense Graph
additionally covers automatic export, tree relocation after later exports, native
layout tampering rejection, empty and distinct states under one loaded module,
independent closure, and allocation of only the actual plan dependencies. The
mixed pointer/dense case now also qualifies Vulkan source-isolated component
consumption, equivalent native layout queries, default calls, and two additional
compute queues with interleaved accepted work and independent instance close.
Other Vulkan sparse layouts, real application components and steady-state
performance require separate qualification.

`test_aot_component_dense.py` compares the same numerical definition with ordinary
JIT on CPU/CUDA. It covers new capacities, range/Field-shape/full-Field iteration,
vector/matrix elements, aliases, empty active ranges and capacity-tail sentinels.
Different sizes share one frontend definition and one native compilation; invalid
dimension/root bindings fail before execution.

The standalone Runtime now provides experimental
`dense_snode_group(schema_json, dimensions)`. It reconstructs actual SNodes with
the same schema. CPU/CUDA invoke the normal `StructCompilerLLVM::run_layout`;
Vulkan invokes the same `spirv::compile_snode_structs` used by SNodeTreeManager,
including its own member ordering, alignment and padding. Queries project actual
compiled nodes; consumers do not supply offsets/strides. The complete root is
allocated once and zeroed through its LLVM/Vulkan allocator. Numerical addressing
uses instance arguments, without registering or switching fixed module pointers.

`group.member(paths, dtype, element_shape=...)` validates root-relative place
paths, element types and compact or positive uniform-record layout, returning
finite views of that root. Contiguous element lanes may share a larger record;
the existing affine argument ABI carries its authoritative index strides.
Paths will come from compiler-produced contracts, not application per-kernel
field tables. Groups reuse existing memory handles, borrowed leases and prepared
retirement. Closing is currently cold: wait, revoke all aliases, then release
storage. Other groups remain valid; revoked view handles remain closeable.

Dense deployment does not impose separate 4096-node, child-index or element-lane
caps, or a fixed JSON byte limit on this construction API. Node references use
the ordinary signed SNode index range. Member validation uses the actual tree,
placement, ndarray ABI extents and checked byte products, and allocated root
bounds. CPU/CUDA coverage includes 4098 places in one root and transfers through
a large element and its last-member alias; this does not qualify large local
matrix computation performance.

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

Use `scope.power_of_two_dim("capacity", (scope.dim("count") * 2).maximum(2))`
for a normal hash capacity rounded up to a power of two. The existing integer
attribute is checked; the consumer derives it from `count` through the shared
dimension evaluator and cannot override it independently. A minimum of zero
rounds to one. Existing count/address integer limits still apply; this adds no
model-size budget. An older runtime rejects the unknown `ceil_pow2` relation.
CPU/CUDA frozen-consumer tests cover changed capacities, masks and same-root
member offsets; application hash publication still needs its own qualification.

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
one normal owner and the while slice below. If/switch, native nodes, external
array views and recipe search need separate qualification. Root identity includes argument
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

A normal component Graph can dispatch original kernels belonging to explicitly
declared child resource owners. Assignment through `ResourceScope`, for example
`scope.worker = worker`, supplies the declaration edge. A child entry must have
exactly one path from the root; undeclared or ambiguous owners reject. Nested
`ti.func` calls retain their qualified paths and definition guards. Different
children can use the same local resource names; shared Fields keep their
allocation-site dimension namespace and one actual SNode group.

This uses ordinary Graph lowering and the existing instance call bindings, with
no module pointer switching or extra execution interpreter. CPU/CUDA tests
cover sequential child calls, shared storage, distinct private state, new sizes
and interleaved instances of one loaded module. Independent child dimensions
and borrowed foreign layouts still need explicit compatible relations; this
does not automatically qualify a complete multi-solver application.

When a source allocation and its borrower are both declared under the root,
the allocation site's dimension and SNode layout remain authoritative. Explicit
borrow mappings become equality constraints checked during compilation and
consumer preparation; equal producer integers never establish this relation.
The borrowed member and source therefore refer to one instance allocation,
while private child resources retain separate ownership. CPU/CUDA tests cover
new capacities and reject inconsistent child counts before execution.

An original initializer recording can join the same owner's step recording:
`component.compile_graph(owner.record_step, initializer=owner.record_initialize)`.
Both phases share capture and resource closure, including initializer-only SNode
roots. The normal Graph compiler preserves their boundary; existing lowering's
actual operation boundaries populate ExecutionPlan initialization. The v8 query
and code identity retain this lifecycle without sampling model state.

`component.inputs(initialization=True)` queries initializer values/arguments;
the default queries replay inputs. Upload ordinary model inputs first, then call
`instance.initialize_state(...)`. Preparation is rejected until it succeeds.
Repreparing preserves state; reinitialization requires explicit `reset=True`
with all instance prepared calls closed. Initialization completes synchronously
using the existing instance binding on CPU/CUDA, never a module pointer swap.
Failure does not mark success or promise rollback of accepted numerical writes.
Dense/array explicit initialization prerequisites remain: write effects alone do
not prove that a numerical initializer covers every element.

This slice qualifies instance-owned Dense/fixed-SNode resources. Initializers
combined with Plan-private resources, cross-phase binding constraints or special
submission/storage recipes reject pending shared-lifetime qualification. Existing
components without model initialization retain their behavior. Tests cover an
initializer-only root, two instances, repreparation, reset, close isolation and
Runtime-only consumption; this is not complete GeoPhys model preparation or
qualification of all backends.
