# Component capture compiler: initial implementation

Producers use the public `ti.aot.Component` entry; `_component` remains its
implementation module. It compiles the original decorated method through the normal AST transformer; it does not parse
the application a second time or create a numerical wrapper.

Ordinary kernels that use only explicit arguments can also use an empty
`ti.aot.Component()` environment and compose through a normal Graph recording.
They need no artificial application owner or numerical wrapper. Global/closure
classification and static guards still apply; an environment with resource
declarations requires its actual owner. Stateless roots use the same exported
component query, instantiation, and execution contracts.

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

Dense group membership is indexed from immutable component declarations. A
Kernel capture or Graph completion check reuses each group's native layout
projection within that operation; repeated dispatches validate each distinct
code definition once at Graph completion. Publication still rechecks complete
groups, physical roots, member paths and extents to reject resource replacement
during compilation. These temporary snapshots never cross Host recording
callbacks, owners or execution calls, and do not change code identity, ABI or
artifacts.

```python
component = ti.aot.Component.from_owner(owner)
root = component.compile_graph(owner.record_step, initializer=owner.record_initialize)
module = ti.aot.Module()
root.export(module, "step")
module.archive("component.tcm")
```

Consumers can inspect a frozen root before creating a device or loading native
code with `taichi_forge_runtime.aot.inspect_component(path, "step")`. It accepts
a TCM or exported directory. The returned `ComponentInfo` exposes `target`,
`semantic_key`, a copied `contract`, and `configuration_sources()`;
`check_configuration(normal_model_choices)` uses the same exact static-input
rules as loaded-component instantiation. Derive those choices from the normal
Host model, rather than copying the artifact's selected values. This supports
cold artifact selection without loading every candidate module. It does not
admit execution: load, instantiate and prepare still validate native entries,
backend capabilities, dynamic resources and lifetimes. Missing provenance in an
older artifact requires a rebuild before source-based selection. Inspection
checks headers and static facts, not every layout or execution-plan fact.

For Vulkan, `required_device_capabilities()` reads the module-wide requirements
directly from native GFX `metadata.json`, including requirements of other entries
in the same module. Use `check_device_capabilities(runtime.device_capabilities())`
to filter cold candidates before loading. Missing levels are zero and unknown
required capabilities cannot be ignored. These are distinct from component
protocol `required_capabilities`; native load/prepare still perform admission.
Other targets currently reject this query rather than reporting no requirements.
Queries return copies of the inspection snapshot without loading a native library.

For example, Vulkan `Module(caps=["spirv_has_atomic_float_add"])` explicitly
requires native float atomic addition. Portable artifacts without that capability
retain the existing CAS fallback. Performance comparisons must match physical
devices and exported capabilities, not just the `vulkan` backend name. Applications
may ship qualified target variants and select them during cold preparation;
optimizing one profile must not silently raise every artifact's hardware requirements.

The recording methods are the application's normal JIT Graph definitions.
Build finite normal configurations with public `ti.aot.Variant` and
`ti.aot.build_variants`. Optional `ti.aot.BuildBudget` axes default to `None`;
catch `ti.aot.ComponentBuildError` to inspect an incomplete build report.
`ti.aot.StaticValue` can narrow a normal static domain, and `ti.aot.RootOption`
declares Host recording options. Standalone consumers use
`taichi_forge_runtime.aot.Runtime` with `module.component(name)` or
`module.component_variants(name)`; they do not import these producer types.
After exporting, `built.close()` releases the Producer family's retained Graph
handles without decoding its typed build report or evicting shared code caches.

Before loading, `inspect_component_variants(path, name)` validates the family's
case references and coverage against its actual component contracts. Its
`configuration_sources()` query requires identical source paths in every case;
`select(configuration)` accepts the complete typed configuration and returns an
ordinary `ComponentInfo`. This shares the loaded family's exact selection rule,
but does not replace native load, Plan/kernel closure or resource admission.
No native library, device state or numerical frontend is needed for inspection.

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
use the admitted root configurations. Families require the same admission input
names, but individual roots may declare different finite choices. Family v2 records
their union and selects only explicitly built complete configurations, never a
Cartesian product. Consumers verify the union against the actual roots and retain
each root's guards; additional choices and ambiguous duplicates are rejected.
Equal domains still export v1. Numerical dependencies and ABI, rather than this
union, determine code sharing. This does not trace Python Host reads or infer
capacity constraints.

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

Use `scope.constant("JOINT_KIND")` for an immutable numerical definition such as
an enum code or a fixed algorithm coefficient. It is not a model configuration
input: consumers do not supply it to `instantiate`. Actual reads retain its typed
value in Kernel identity and producer cache guards; changing a used definition
requires rebuilding the affected code. Unused constants do not change numerical
identity. Recording rejects definition changes during Graph construction.
Model-dependent choices still require `scope.static(..., choices=...)`, and
runtime quantities require `scope.value`; declaring them constant would specialize
the artifact rather than make them dynamic. CPU/CUDA tests in
`test_resource_declaration.py` cover these distinctions and frozen consumption.

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

Builtin `isinstance(field, ti.StructField/ti.MatrixField)` branches are qualified
for declared Fields. A complete StructField's `keys` retain member order as a
code dependency. Statically selected members keep declaration references until
actual indexing: inspecting a member's dtype does not add its storage argument.
Model tensor `n/m` remain dynamic element axes, including runtime lowering of
the original static-loop spelling. Application instance hooks and arbitrary Host
reflection are not executed. CPU/CUDA definition-package checks cover new sizes,
scalar/vector members, skipped integer members, NaN/Inf detection and restore,
and consumption without numerical source.

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

Explicit static templates also enter code identity only when read by the original
AST. Unread parameters and values excluded by a static branch do not create code
copies; read booleans, constants and immutable tuples retain their specialization.
Mixed resource tuples record selected static leaves. A wholly static tuple read
as a value remains one complete value, without a claim of element-level pruning.
Invalid mutable objects still reject at template admission. Ordinary JIT template
calls are unchanged. Focused CPU/CUDA checks cover four Graph calls exporting
three actual bitcode files, necessary branch/value differences, new-size independent
instances and adjacent resource-template behavior. This is not a measurement of
whole-engine package savings.

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
existing native type/JSON contracts. The initial family covered a single Dense
level. The CPU/CUDA extension below adds a qualified nested family; sparse,
quantized, Morton, index-offset and ambient layouts remain excluded.

CPU/CUDA now qualify ordered Dense paths that introduce each axis exactly once,
including the trees produced by `order="kij"` and `"lijk"`. The frontend associates
dimension provenance with native local extractors. Field descriptors and consumer
member views share the compiled offset/stride projection, preserving physical
placement. Repeated tiled axes remain rejected. Members must be packed axis
permutations or existing uniform records; arbitrary affine layouts with holes
are not included. Unique mapping, nonnegative strides and capacity remain mandatory.
This nested family has not been qualified on Vulkan.

Cold Host transfers pack/unpack a physically contiguous permutation once and use
one bulk transfer, without element-by-element GPU copies or writes to neighbors.
Execution uses the existing affine Kernel ABI. The ordered Dense component test
compares ordinary JIT, captured execution, and frozen-code consumers at new sizes,
including node layouts, scalar/vector storage, borrowing, two instances of one
loaded module, and cross-level index-product overflow rejection. These mechanism
gates do not qualify the complete staggered TGS solver or its performance.

Known CUDA synchronization and scalar warp intrinsics preserve the compiler's
affine-access proof: they do not interpret array descriptors. Unknown internal
calls, runtime-context calls and pointer consumers remain conservative. This is
a compilation-time classification; it adds no per-step runtime check. Artifacts
exported without this proof must be rebuilt rather than relabeled by the loader.

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

Version 2 layout recipes can also represent adjacent, same-type vector/matrix
scalar components as one place with symbolic element extents.
`DenseSNodeSchema.capture_tensors` validates component order, completeness and
shared dimensions. Consumers expand it into ordinary SNodes and use the same
backend layout rules. Fixed mathematical extents remain constants; version 1
recipes are unchanged. Focused checks cover mixed types and fixed vec3 siblings,
model extents above 64, independent instances and invalid dimensions. This is a
persistent-storage capability; the scalar numerical ABI below builds on it.
Definition packages can select this recipe from proven model element dimensions.
Local vectors and shared workspaces remain separate requirements.

An internal scalar-axis projection appends element axes to the runtime index
axes while preserving native byte strides, offsets and ownership. It reuses
the affine Ndarray argument ABI so one scalar-access implementation can bind
different model widths, without copying the original Field into another array.
Standalone `group.member(..., scalar_axes=True)` uses the same native projection.
Host transfers coalesce contiguous inner axes and retain the existing bulk
transfer paths. Focused CPU/CUDA/Vulkan checks cover interleaved members, two
instances, rejected tensor ABI mismatches and a frontend-blocked consumer with
a new extent of 129. This foundation is not complete dynamic TensorType support.

`capture_tensor_members` also returns producer SNode IDs mapped to persistent
recipe nodes. `group.schema_member(nodes, ..., scalar_axes=True)` resolves these
nodes through the original native instantiation, including expanded tensor
lanes. Fixed siblings keep the same selector when preceding model tensors grow;
no producer IDs or expanded lane paths enter the persistent selector. Legacy
path-based members remain supported. Invalid, repeated and non-place recipe
selectors are rejected before a view is published.

The internal model-tensor declaration now lowers original `field[i][j]` and
matrix component accesses through that scalar ABI. Field traversal uses only
the logical index axes through existing ndrange lowering; fixed mathematical
element axes retain static metadata. Model axes read runtime shapes. All
members of an affected group use runtime strides, including fixed siblings
whose cell stride changes with the model tensor. Aliases with conflicting
element-dimension provenance are rejected.

Component Graphs carry the versioned `component_model_tensor_members_v1`
capability and instantiate these views using native recipe selectors. Focused
CPU/CUDA/Vulkan checks cover one compiled definition across different widths,
unchanged ordinary JIT, independent prepared instances and a frontend-blocked
consumer at a new width of 129. Whole Field-element value copies are still explicitly
rejected. Direct component calls now support model-sized Vector/Matrix.zero locals
and local copies through private invocation workspace. Package Graph uses existing
Plan private resources; CPU/CUDA real_func forwards workspace and work-item indices.
The older explicit Component Graph local workspace path remains unqualified.
CUDA model-sized SharedArray now uses native block storage and launch sizes derived
from argument snapshots; other backends and declarations inside real_func reject. Definition packages
infer model axes from constructor parameters and existing declarations within
the [documented package scope](../forge/package_aot.en.md). Local tensor arithmetic
remains pending; no application decorator is added.

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

Extent expressions support subtraction, for example
`(jobs + 31) // 32 + candidates - 1`. Every intermediate must remain in the
nonnegative i64 range; an enclosing `maximum(1)` cannot hide underflow. A CPU
gate checks frozen new-size loops/layout and rejection before allocation.
Older runtimes reject the unknown `sub` expression.

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
Cold uploads reuse immutable resource-kind qualifications validated at load time,
without decoding the complete public contract for every Field. Public queries
still return independent copies; sparse rejection, alias uniqueness, instance
state and cold-boundary checks remain active on each initialization.
Version 2 projects resource access from the compiled Graph; unknown effects stay
conservative and version 1 remains readable. Every referenced resource still
requires initialization: write-only access does not prove full-capacity coverage
or justify skipping tail initialization. No producer live state is serialized.
`instance.prepare(values=...,
dimensions=...)` binds through the existing ExecutionPlan and PreparedCall, with
no new Python execution interpreter. Fixed vector/matrix values reuse native ABI
encoding; every entry consuming a named value must have compatible value ABI.

Native struct offset reflection stops once a pointer or explicit offset proves
that the declared layout applies, avoiding repeated full sibling scans for wide
captured ABIs. Implicit aligned layouts retain their original rule; this changes
neither serialized ABI nor per-instance binding and lifetime checks.

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

`component.configuration_sources()` queries each admitted input's declaration
path, for example `{"kind": "attribute", "path": ["time_plan", "substeps"]}`.
Normal root options use `kind="recording_argument"`. Consumers need not decode
generated capture IDs or read the producer's selected values as new-model inputs.
The query returns a copy of `compilation.configuration_sources`; it does not
traverse objects, invoke getters or relax variant checks. New Graph exports
include this optional compiler projection without changing numerical ABI or code
identity. Old artifacts remain loadable, but the query requires rebuilding when
the provenance is absent. Present provenance must cover all configuration inputs
and contain valid paths and kinds.

`component.input_sources()` similarly returns `dimensions`, `values` and
`resources` from the actual compiled closure. Dimensions include only base
inputs; derived extents keep their expressions in `contract`. Each resource maps
all its aliases to normal declaration paths, so nested owners do not require
decoding generated IDs. A borrowed alias that directly names its member keeps an
attribute path. An indirect sibling reference has `kind="borrowed_group_member"`,
an anchor `path` and its original `member` identity; the anchor is not necessarily
that member. This is provenance, not a Python evaluator or an automatic binding
table. Normal model preparation supplies the input data and checks model validity.
The loader checks complete role/alias coverage and source structure. Old artifacts
without this optional projection remain loadable but require rebuilding for the
query. CPU/CUDA tests consume frozen nested fields and arrays at new extents and
values, preserve aliases, exclude unused declarations and reject malformed paths.

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

`scope.borrow(name, field, source_scope=True, dimensions=...)` links to the
original allocation scope already declared in the component root. This avoids
making a child restate layout-only sibling dimensions. The optional mapping
expresses target equalities; the complete layout still comes from the source.
When that allocation scope is absent, actual numerical capture rejects with
the declaration location. An unused link does not block a smaller component
or enter its ABI. Existing complete dimension mappings still support standalone
borrowers. CPU/CUDA tests cover both modes, new layout-only capacities and
pre-execution rejection of mismatched target counts.

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

This slice qualifies instance-owned Dense/fixed-SNode resources and replay-only
compact, zero-initialized Plan-private ndarrays. The existing Plan reference
closure proves that initialization cannot access those private buffers; they
retain preparation-local allocation and retirement, including a fresh scratch
state after repreparation. Export and Runtime use the same phase check. Private
resources referenced by initialization, other private resource kinds, cross-phase
binding constraints that cannot be validated wholly in either phase and special
submission/storage recipes still reject pending
shared-lifetime qualification. Existing
components without model initialization retain their behavior. Tests cover an
initializer-only root, two instances, repreparation, reset, close isolation and
Runtime-only consumption; this is not complete GeoPhys model preparation or
qualification of all backends.

Binding constraints are projected using the existing validator's reference
sets. Each requirement stays intact in every phase that retains all its
bindings; layout and disjointness checks are neither weakened nor dropped.

Free kernels may also receive declared child records through `ti.template()`
parameters, including nested declaration paths. They use the same attribute and
helper capture pipeline as bound methods; cached code retains paths, not Python
objects. Unused parameters add no resource dependencies. Undeclared or ambiguous
objects reject. CPU/CUDA tests cover cache reuse, frozen-module new-size
instances, shared resources and interleaved instance execution.

Source-scope borrowing also accepts declared Dense Fields on the default SNode
root. The component root must contain the original Field declaration. The borrow
reuses its existing group and shape contract, including native layout completeness
checks; no storage regrouping or separate layout recipe is introduced. A missing
source remains acceptable only while the link is unused.

Normal `ResourceScope` allocation can explicitly declare `initialization="zero"`
for a scope or an individual Field/ndarray. The default remains
`"consumer_explicit"`; a Field can override a zero scope for required model input.
The compiler projects the actual allocation policy into each captured resource,
including aliases, and requires `component_zero_initial_state_v1`. New instances
zero only these declared resources through their own bindings. Producer contents
are never sampled. Preparing again preserves persistent state; an explicit cold
reset can still use `initialize(zero=[...])` followed by the normal initializer.
Zero is not inferred from write effects and does not establish complete coverage
by a numerical initializer. Fixed SNode placement retains its existing empty
activation contract. CPU/CUDA checks cover new capacities, required model input,
two instances, reprepare and cold reset; no native ABI or layout rule changes.

Normal cold model preparation can also contain device work before Host response
adjustments. `compile_graph(record, initializer=finish, setup={"body": unpack})`
exports those original recordings into the same component. Each recording must
belong to the root owner, be nonempty and precede a nonempty final initializer.
This is not a per-kernel signature adapter or a new execution interpreter.
The existing Graph lowering supplies actual operation boundaries; v9 contracts
include `component_cold_setup_v1`, with boundaries checked against the execution
fingerprint. Numerical definitions and native ABI remain shared.

`component.setup_entries` queries names; `component.inputs(setup="body")` queries
that entry's values/arguments. `instance.run_setup("body", values=...)` completes
the original device work on that instance at a cold boundary, allowing normal
Host reads/uploads before the next entry or `initialize_state()`. Each setup
requires its own referenced resources to be initialized, while final readiness
still requires all declared resources. Setup alone does not enable replay.
After readiness, setup requires explicit `reset=True` and invalidates readiness;
an accepted execution failure leaves replay disabled. No rollback is promised.
The normal Host constructor owns entry ordering and model validity. Reprepare
and reset do not automatically repeat setup or invent missing model data.

Plan constraints remain whole in each applicable phase; requirements spanning
distinct cold entries without an existing qualified lifetime contract reject.
Existing initializer/private-workspace restrictions continue to apply. CPU/CUDA
Dense tests cover two setup entries, Host adjustment, distinct instance sizes,
readiness and cold-boundary rejection, reset, and corrupted boundary rejection.
GeoPhys additionally verifies its original body decoder in a frozen CUDA PGS
full step, with poisoned mass outputs proving that setup executes. This does not
qualify every rigid constructor, resource family or backend combination.


### Named normal operations on one instance

`compile_graph(record, initializer=finish, operations={"refresh": refresh})`
exports additional normal recordings under the same component owner. Names identify
whole normal operations, not individual kernels. The default replay remains `record`;
it does not execute `refresh`. The compiler collects their resource union once and
projects boundaries through the existing Graph lowering. Contract v10 requires
`component_named_operations_v1` and authenticates the lowered boundaries in its
execution fingerprint. This currently requires an explicit final initializer.

`component.operation_entries` lists names; `component.inputs(operation="refresh")`
queries only that operation's value/argument requirements. Call
`instance.prepare(operation="refresh", values=...)` to obtain an ordinary prepared
call. It shares the instance's resources and ordered execution domain with the
other calls, retaining the same native leases and immutable bindings. Preparation
and invocation do not implicitly rerun initialization, alter readiness or replace
module-level pointers. Closing/resetting an instance still retires all its calls.
Loaded Graph composition queries the default replay's ABI, so an input used only
by a named operation or initializer is not required by default dispatch.

Whole binding constraints stay attached to each applicable entry. A requirement
spanning separately prepared entries rejects; Plan-private scratch must belong to
exactly one entry. Persistent shared work belongs in instance resource declarations.
Unknown names and modified lowered boundaries reject. CPU/CUDA gates cover independent
sizes, retained state, setup compatibility, loaded default dispatch and these failures.
Other resource/backend combinations retain their separate qualification requirements.

### Public Host state without numerical dependencies

`compile_graph(record, host_resources=(owner.history, ...))` retains declared
Dense Fields and managed Ndarrays used by normal public state/observation APIs. The existing native
SNode schema supplies group layout, sibling extent dependencies, member paths
and initialization; producer values are never included. These resources do not
add Graph arguments, Kernel ABI slots, dispatches or numerical code dependencies.
Kernel effects remain compiler facts (`none` for unaccessed state).

The additive `component_host_resources_v1` contract extension lists the retained
resource identities explicitly. Consumers validate these names and reject other
unreferenced requirements; older runtimes reject the required extension. Existing
instance resource transfers, initialization and retirement apply. Managed Ndarrays
retain the existing array type/shape/alias contract and require the additive
`component_host_array_resources_v1` capability. Host-only arrays add no numerical
bindings; arrays already used by kernels/native operations retain their original
effects. Undeclared resources and unsupported storage kinds reject. CPU/CUDA tests
cover independent dynamic array lengths, vector elements, group extents, rank-zero
state, unchanged Kernel ABI/code, two instances and malformed retention metadata.

`host_resource_sets={"snapshot": (...), "diagnostics": (...)}` additionally names
normal Host publication closures. Sets may overlap or be empty; their union joins
`host_resources` without duplicating storage or widening numerical Kernel ABIs.
The additive `component_host_resource_sets_v1` capability preserves these sets in
the queryable contract, so a consumer can validate its snapshot closure separately
from extra diagnostic resources. Members must refer to retained resource identities;
unknown, duplicate or unretained references reject. Set names do not define execution
phases, access permissions, initialization policy or a second resource layout.

### Native sorting of Dense Field instances

Component capture of `primitive_sequence().sort(...)` also accepts declared,
packed one-dimensional Dense Fields. It uses the existing SNode group/member
contract and instance views, retaining layout-only siblings and native read/write
effects even when no captured kernel accesses the sorted arrays. It does not
flatten their allocation into independent Ndarrays or retain producer pointers.
The ordinary native sort exporter still validates dtype, packed storage, stable
ordering, strategy and workspace. Required capability
`component_native_dense_sort_v1` distinguishes this resource combination.

CPU/CUDA/Vulkan tests cover scalar u32 keys/i32 values, a shared root, production
at 7 elements and instances at 257/519, stable NumPy results, repeated calls and
independent closure. Native scratch is prepared from each instance's length by
the existing ExecutionPlan. This does not qualify arbitrary sparse/strided
views, other native primitives, every execution domain or whole applications.

### Resource extents for Graph-private scratch

Normal Graph recording can declare scratch using
`region.private_ndarray("previous", ti.f32, (ti.graph.ArrayExtent(owner.velocity), 6))`.
`ArrayExtent(array, axis=0)` refers to a logical index axis of a declared Dense
Field or whole Ndarray, excluding vector/matrix element axes. Ordinary Graphs use
the current shape. Component capture retains the resource reference, and the
existing ExecutionResources allocator obtains the extent from each consumer
instance at prepare time. This does not resize an in-flight allocation.

The required extension `component_private_array_extents_v1` exposes `private_shapes`
and validates them against the lowered Plan. Shape-only sources remain part of the
resource/layout closure without widening Kernel ABIs or claiming numerical access.
Existing scoped private-storage ownership and exclusive-submission rules apply.
Unsupported sources and undeclared resources reject; private extents remain positive
under the existing allocation contract. CPU/CUDA gates cover 7-element production,
11/37-element instances, repeated private regions, interleaved calls and malformed
shape metadata. These gates do not qualify an application's conditional solver.


### Declared Host owner sequences

`scope.owner_sequence("solvers", self.solvers)` exposes a normal plain list or
tuple of already declared resource owners. The compiler follows each owner's
existing declarations and original SNode groups; it does not build a second
application dispatch table. Declare the collection again after changing its
composition, and finish ordinary lazy resource preparation before capture.
Changing membership while recording is rejected, including same-length reorder.

An aggregate that borrows the same owners uses
`scope.owner_sequence_reference("solvers", self.solvers)`. Each member must have
exactly one owning path elsewhere in the component root. The compiler resolves
the sequence members through the existing owner-reference contract, retaining
canonical resource and dimension sources. It does not duplicate state or SNode
groups. Static iteration and compile-time integer selection remain supported;
changing referenced members requires
rebuilding declarations. Missing or ambiguous ownership rejects before export.
CPU/CUDA gates cover new dimensions, same-length reordering, unchanged code for
unrelated borrowed views, and source-free execution of the canonical resources.

This is a Host composition namespace, not a device-indexable container. Each
Kernel retains only its actual dependencies; adding an unrelated owner or changing
another owner's configuration does not widen that Kernel's ABI or code identity.
The Graph separately admits the used collection length as a static configuration.
`configuration_sources()` reports `kind="sequence_length"`; the caller supplies
its actual model composition length to the existing configuration admission.

Inside a Kernel, `for owner in ti.static(self.solvers)` unrolls this explicitly
declared fixed composition. Members retain declaration paths; their Fields,
dimensions and values keep their normal dynamic contracts. No Python owner is
stored in the artifact. The iterating Kernel records the sequence length and
checks it before cached member dependencies, including for an empty sequence.
Adding or removing members recompiles that iterating Kernel; a Kernel called on
one independent member keeps its existing reuse boundary. Device indexing still
rejects. CPU/CUDA tests cover new-size reuse, changed/empty compositions and frozen
artifact consumption at new dimensions. A compile-time integer subscript selects
only that member, including Python negative indices; bounds are checked during
compilation. Device-expression indices and slices reject.

Source paths contain attribute-name strings and explicit nonnegative integer
positions, for example `["simulator", "solvers", 0, "n_bodies"]`. These queries
require `component_owner_sequence_sources_v1`; older runtimes reject the new
contract. No new native resource layout or execution interpreter is introduced.
CPU/CUDA gates cover cross-size reuse, independent instances, compilation guards,
wrong composition rejection and the existing non-indexable-device restriction.

`component.resource_name(["simulator", "solvers", 0, "state"])` resolves a
declared source path to its canonical resource ID. The instance convenience
`instance.resource_from_source(path)` returns that instance's existing resource;
aliases return the same object, while another instance keeps its own storage.
For `borrowed_group_member` provenance, pass the queried `member=` explicitly:
an anchor path alone never means a sibling member. Unknown or ambiguous sources
reject. These queries use a read-only index of existing compiler facts, do not
evaluate Python getters, and do not change bindings or module pointers.


Lifecycle recordings (`setup`, `initializer`, and `operations`) may be bound to
the root or a uniquely declared child owner, including an owner-sequence member.
They record directly into the root's existing capture environment, so resources,
static guards, effects and phase boundaries retain the same authority. An owner
outside that closure or reachable through ambiguous owner paths rejects. No
forwarding method on the root or separate numerical wrapper is required.

Each independently recorded lifecycle phase uses the existing Graph private
scope. Reusing a normal recording in the main step and a named operation does
not merge identically named scratch, predicates or counters. Root
`private_ndarray()` declarations and child Sequential regions inherit that
phase scope; explicit instance resources and arguments retain their sharing.
Structured materialization may clone execution objects, but phase boundaries
still require a one-to-one immutable IR match, including dispatch identities,
control policy and order. Cross-phase fusion without a qualified boundary map
still rejects. CPU/CUDA gates cover repeated while operations and root/nested
private names. Existing restrictions on Plan-private initializer resources
remain in force.

Normal stages can declare a non-owning back-reference with
`scope.owner_reference("motion", motion)`. Collection resolves the target to one
ownership path in the same component root; outside or ambiguously owned targets
reject. References do not recursively declare resources, allocate another root,
or introduce module-level mutable bindings. Original methods and nested helpers
follow the referenced owner's existing declarations. Only references actually
used by a Kernel enter its dependency identity; candidate reuse and recording
completion validate the relationship again. Retargeting a live reference does
not silently redirect compiled code. Deployment uses the resulting ordinary
resource/value ABI and retains no producer Python object. CPU/CUDA tests cover
parent/child links, new-size consumption and invalid retargeting.
`scope.owner_dim("motion", "count")` relates this owner's layout or `derived_dim`
to a declared non-owning owner's normal dimension. The target requires a unique
ownership path in the same component root. Its dimension remains authoritative:
the consumer binds it once, without a second capacity input or a fabricated Field
alias. Named derived dimensions are supported; internal `parent_dim`/`owner_dim`
aliases are not valid targets. Unused source dimensions do not enter Kernel code
identity. CPU/CUDA tests cover new-size loops, layouts and tails, plus rejection
of missing or ambiguous ownership and stale normal relations.

An original SNode root can contain allocations from several declared owners.
Dimension names in a `source_scope` borrow refer to the anchor's allocation
namespace, not every equal leaf name in that physical root. The complete root
still retains all owners' qualified layout dependencies and placement order.
CPU/CUDA checks vary both owners' extents independently and reject inconsistent
borrow relations before execution.

Whole-record indexing of a declared dense `StructField` uses the existing
Struct value and assignment semantics over its captured member views. Plain
and grouped traversal use the runtime extent; each member retains its original
SNode placement and stride. CPU/CUDA gates compare whole-record copies, stores
and helper calls with normal JIT at new sizes for SOA and AOS. Fixed sparse
records retain their existing SNode activation and traversal path.


`parent.child("observations", owner, builder=...)` creates a writable declaration
scope for a normal child resource owner and records its ownership edge. The child
keeps its own Field/dimension namespace and original SNode layout. Its
`parent_dim("count")` can participate in `derived_dim` and Field shapes, so a
parent loop and child storage share one consumer dimension rather than accepting
two inconsistent capacities. `record` continues to describe read-only records.
Conflicting roles, replacing an owned child, and direct self-ownership are
rejected; complete closure checks reject other ownership cycles. This does not
relocate existing Fields, rebind in-flight instances, or support moving a child
with inherited dimensions to an undeclared parent path. CPU/CUDA gates cover
code reuse from 3 to 19 elements and a frozen artifact instantiated at 37.


`scope.field_like(source, initialization="zero", layout=ti.Layout.SOA)` allocates
independent dense backup storage during normal construction, retaining the source
symbolic shape provenance and element types. It neither aliases the data nor
copies producer values or freezes the producer capacity. The source declaration
scope must be visible in the complete component root. Undeclared and fixed-SNode
sources are rejected. This supports scalar/vector/matrix Fields and flat Structs
with implicit dense allocation, not sparse activation or arbitrary tree cloning.
The backup group identity follows its own ownership path while its dimensions
can originate at the source owner. Newly built group names may change; existing
frozen artifacts continue to use their versioned contracts. CPU/CUDA gates cover
independent record copies, code reuse at 3/19 elements and frozen consumption at 37.

Definition package format 43 supports several declared allocation sources for
one class borrowing a Field. Export derives resource contracts from actual
argument tuples at declared constructor calls, without executing constructors or
forming a Cartesian product of individual parameter domains. Host retains one
class; construction selects an instance contract from same-package allocation
provenance, then numerical methods select implementations of the original
definitions. Model sizes do not select types; methods that ignore the inputs
continue sharing code. `field_like` allocates an independent backup from the
selected source contract. Dimension checks, aliases, failure cleanup and prepared
snapshots remain effective. Local CPU/CUDA checks cover scalar and model-vector
Struct Fields, new sizes and source-free consumption. Each declared call must
currently have one proved source; dynamic source selection, heterogeneous owner
collection discovery, optional/type-domain combinations and cold Kernel factories
inside these objects are not qualified by this slice.

Format 44 projects the existing `owner_sequence` declaration into Host
construction. It reuses finite tuple/list/zip and single-comprehension call
proofs; each algorithm instance keeps its own input contract. Host conditions
may select an empty collection or existing call branches. Collection order,
length and presence do not create combined code identities; prepared calls retain
their actual owners and resource snapshots. Unproved source factories, rewritten
collection declarations and numerical Kernel iteration over these collections
still reject. This does not qualify arbitrary application state-inventory inference.

The same constructor input families can also receive distinct declared owner
classes. Actual argument tuples determine instance contracts; source annotations
still constrain them, and input domains are not expanded into Cartesian products.
Format 47 separates instance admission from numerical code: when the compiler
proves an identical code/ABI identity, lightweight `entry_bindings` reference that
code while retaining their own type, provenance and dimension checks. They do not
copy code or complete contracts. Direct calls and Graph preparation enforce the
same checks; prepared calls retain independent resource snapshots. Local CPU/CUDA
checks cover differing owner methods, actual code counts for identical methods,
new sizes, source-free consumption and wrong-type rejection. Unused inputs do not
create code copies. Late optional-owner binding, model-selected owner types and
arbitrary reflective factories remain outside this slice; it does not qualify
complete TAA/EFC definition export.

`ti.aot.written_resources(owner, owner.record)` inspects possible writes of a
normal Graph during cold preparation and returns the original resource objects.
JIT reuses Component compiler effect analysis; definition export retains the same
compiler's argument effects for the format 48 Host projection and
`Graph.written_resources()`. Inspection runs ordinary Host recording, never a
numerical step. The cold consumer imports no frontend and allocates no workspace,
prepares no call and submits no work. Unknown/opaque effects retain possible
writers; unbound explicit writable arrays reject. Private control arrays are not
owner state, and a zero-iteration loop contributes only its condition. Model
tensors return their original members instead of scalar-axis ABI views. Local
CPU/CUDA checks pass, and GeoPhys capacity checkpoint uses this public query.
Its resource inventory and shared-copy plan migration remain incomplete; this
does not qualify complete checkpoint/EFC definition export.

Constructors may sequentially reuse a local ResourceScope name, including an
owner scope after an unconditional resource context. Export distinguishes handles
by assignment position while retaining the original dimension namespace,
initialization policy and independent FieldsBuilder boundaries. It executes no
constructor and adds no numerical copies or format version. Local CPU/CUDA checks
cover new sizes, separate instances, independent group retirement and source-free
consumption. Conditional rebinding, captured locals and ambiguous mixed contexts
still reject; this does not admit arbitrary Python scope behavior.

Normal inventories may form immutable Field-source tuples using finite literal
names, local append/extend operations, fixed loops and `attrgetter`. Definition
export proves possible resource paths without executing model branches or
enumerating condition combinations. Original Host helpers choose the actual
collection. Helpers must resolve to verified scanned definitions; dynamic names,
unknown effects, recursion, mutable returns and rebinding reject. This has proved
possible sources in GeoPhys's original CCD inventory, but complete resource types,
optional presence and consumption still require their own validation.

Format 45 also permits an already allocated conditional Field as a proved
constructor input. Its source existence predicate remains a binding/admission
check, not a numerical type choice; `field_like` creates independent backup
storage whenever that consumer is constructed. Whole Struct inputs retain source
identity checks even when only their members are used. CPU/CUDA source-free
checks cover absent/present storage, new sizes, changed predicates/aliases and
prepared snapshots. This does not add conditional model-tensor allocation or
qualify GeoPhys's complete CCD/EFC resource closure.

Format 46 connects model-tensor Field allocation to declared boolean presence.
Native recipe selectors retain dynamic element axes. Removing absent members
and empty ancestors still delegates all byte layout to the same SNode compiler;
model sizes and combinations of unrelated options do not duplicate code.
Local CPU/CUDA checks cover standalone vectors, AOS/SOA Struct members,
explicit shared placement, new sizes, original JIT layout equality, independent
instances and source-free consumption. Dynamic Sparse and conditional owner
type selection remain outside this qualification.


Declared resource records and non-owning owner references can pass through
unannotated `ti.func` arguments and local aliases. The frontend retains their
lexical declaration path, not a scalar or serialized Python owner. Subsequent
member access still collects resources, method definitions and dependencies.
CPU/CUDA gates cover ordinary JIT, cross-instance cache reuse, new capacities
after freezing, and rejection after owner-link retargeting.


### Named operations with Host-initialized state

A normal component whose initial state is fully supplied by declared Host inputs
or zero-initialized resources may export `operations={...}` without a numerical
`initializer`. Contract version 11 records `initialization="host_data"`; existing
initializer/setup contracts retain their version and behavior. Each required
consumer resource must still be initialized before any operation is prepared.

Operations and default replay remain projections of the same validated
ExecutionPlan. Per-entry private resources and whole constraints keep the
existing lifetime qualification; the default replay never executes an operation
implicitly. No synthetic initializer, extra numerical launch, or module-global
instance pointer is introduced. CPU/CUDA tests cover different-size instances,
missing initial inputs, malformed lifecycle metadata and loaded Graph replay.

`Component.runtime` is a checked, read-only reference to its owning Runtime. Use
identity checks when a Host adapter also accepts a Runtime for completion or
temporary allocations; equal backends/device UUIDs do not imply the same owner.
The property does not transfer Runtime ownership.

### Compact addressing for extent-invariant Dense members

On CPU/CUDA, component capture can use the existing compact ndarray access ABI
for qualified rank-zero members and one-dimensional, single-level Dense cells
containing only scalar place nodes. The compiler reads the actual byte stride
and element layout from the native storage description. It also checks that the
cell's structure cannot depend on an extent; it does not infer a general layout
from the producer's current shape or `compact_contiguous` flag.

An interleaved view with one producer element still retains affine access when
its cell stride exceeds the element width. A cell containing a dimensioned
child also retains affine access. More complex layouts and Vulkan keep the
existing path. This optimization does not expand storage qualification.

The per-entry capture contract records `dense_access` as part of code identity.
The original SNode schema and native StructCompiler still determine instance
allocation, offsets and strides. Existing native argument binding rejects a
strided resource passed to code without runtime-stride support. Group ownership,
aliases, dynamic extents and independent instance bindings are unchanged.


### Declared interleaved Dense placement

`cell = scope.dense(ti.i, scope.dim("count"))` creates a normal Dense node while
retaining its dimension provenance. Pass the same cell through `placement=cell`
to several scoped Field factories, with identical shape expressions. This keeps
their original interleaved placement and permits zero-initialized scratch. Equal
sample sizes from different dimension declarations are not interchangeable.

Native Dense schema capture and StructCompiler still determine allocation,
member offsets and strides. The placement introduces no Python layout evaluator
or new runtime ABI. Raw SNode placement keeps its fixed-tree contract, including
Sparse activation initialization. A placement from another declaration namespace,
changed extent, or conflicting shape expression is rejected before adding a Field.
CPU/CUDA gates cover singleton-to-37 growth, vector/scalar interleaving, a following
Dense block, an untouched replay sentinel, aliases, and stable semantic identity.


`scope.owner_reference("mesh", mesh, optional=True)` permits an absent link
in normal construction. Absence uses the existing optional-record presence
guard; a present target still needs exactly one owning path in the component
root. Recollect/compile the component and prepare bindings after a presence
change. Existing bindings reject retargeting. This does not add an external
object to the closure or switch instances through a mutable module pointer.

`record == None` and `record != None` also express presence when the record
uses default object equality. The existing presence/reference guards remain
active. Custom `__eq__` or `__ne__` is rejected with an `is not None` diagnostic,
including after a cache hit or a binding change; proxy comparison never runs
user equality code.


### LLVM argument snapshots within a call

The LLVM backend loads the used top-level Kernel ABI values and Ndarray descriptor
members at each generated function entry and reuses their SSA values within that
function. The current binding still supplies the dynamic base, shape, affine mode,
and strides. A later invocation reads its own values; no metadata is cached across
instances or prepared generations. Real functions retain their separate argument
context, and ArgPack content access keeps its existing semantics.

Only argument metadata is immutable within the call. Array payloads remain mutable
and may legally alias. This does not apply lifetime-wide `invariant.load` metadata
or unproven `noalias` assertions to arrays. CPU/CUDA checks cover aliased arguments,
changed shapes and scalar values, interleaved Dense layouts, multiple instances,
ArgPack, and nested real functions. The Runtime ABI and default JIT optimization
tier are unchanged. Existing TCM files remain valid; re-export affected code to
obtain this optimization.

Affine-capable LLVM entries also select effective byte strides from the current
descriptor before applying indices, instead of computing both packed and strided
addresses at every access. Top-level arguments reuse that selection at function
entry. Packed mode still derives strides from shape and element type, so existing
callers need not populate extra stride values. Vector/matrix lane offsets remain
separate. CPU/CUDA checks switch one Graph between packed arrays, strided views,
and new shapes, including rank-zero arrays, matrix elements, and two writable
arguments sharing storage. This adds no read-only or disjointness assumptions and
does not change layout rules, the Runtime ABI, or the default compilation tier.

### Explicit Vulkan acceleration structure arguments

A normal component Graph can use `ArgKind.ACCELERATION_STRUCTURE` for an
original Kernel's `ti.types.acceleration_structure()` parameter. Export requires
`caps=["spirv_has_ray_query"]`; the consumer binds an `instance_tlas` native
resource created by the same Runtime. Geometry, dimensions and AS addresses do
not specialize the Kernel or capture producer scene data. Ordinary Graph
ExecutionPlan export also preserves this explicit binding without a producer
source owner.

The component contract retains AS read effects and requires
`component_explicit_acceleration_structure_v1`. The existing Kernel ABI, native
resource and execution domain validate types, Runtime ownership, generations
and in-flight leases. An Ndarray or BLAS cannot substitute for the TLAS.
Qualification covers two new geometries and output sizes sharing loaded code,
ordinary and Vulkan-domain submission, independent AS retirement, and consumption
without the frontend. Implicit AS captures remain outside this qualification; OptiX has the separate contract below.


### Component device-transform refits

`scene.record_refit_transforms(transforms="poses", scene="scene")` explicitly
associates the original recording owner with a typed Graph AS port. Normal JIT
bindings must reference that same owner. Component conversion preserves the
`vulkan_ray.instance_refit` operation and its read/write effects; it captures no
producer geometry, instance count or handle. The consumer supplies its own TLAS
and f32 transforms, validated by the existing native provider. Without an
explicit scene port, component capture rejects the recording; ordinary JIT and
fixed-state AOT retain their existing behavior.

The transform port is an explicit Graph array or declared private Graph array.
The component requires `component_native_ray_v1`. Kernel transform writes,
refit and ray queries retain their original ordered Plan execution; no new Host
frame interpreter is involved. Tests use new 3/7-instance topologies, repeated
transform updates, interleaved independent instances, ordinary and Vulkan-domain
submission, wrong-count rejection and AS retirement during accepted work, in a
consumer without the numerical frontend. This qualifies transform refit with
fixed per-instance topology, not topology mutation or OptiX.


### OptiX scene ports and whole-array queries

`record_typed()`, `record_occlusion()` and `record_refit_transforms()` accept an
optional `scene="name"` port. Ordinary Graph bindings must reference the recording's
original scene owner. A component instead retains typed ports, native operation
semantics and effects; its consumer provides a same-Runtime scene. OptiX scenes
are native-only Graph arguments and do not become CUDA Kernel AS parameters.

Component `optix.query` version 4 derives its positive uint32 ray count from the
prepared ray array's first extent. The existing compact layout check verifies
rays, hits and IDs together. New resolution requires cold array binding/prepare;
replay does not mutate descriptors or reread Host scene state. Uniform two-sided
or front-only policy is independent of producer instance count. Heterogeneous
per-instance tuples, alpha masks and micromaps are rejected by this component capture.
The original `graph_fixed_query` selection is retained, including the legacy
query path when false. Versions 1–3 keep their fixed-count semantics.

The component requires `component_native_optix_v1`. Configure the installed
OptiX adapter through `Runtime.configure_provider()` before creating its context,
GAS and instance scene with the existing native resource factories. No adapter
or vendor-library path is baked into the TCM. Tests freeze a one-instance,
two-ray producer then consume 3/7-instance scenes with 5/11 rays, interleaved
refit/query/occlusion, ordinary and CUDA-domain submission, invalid output counts
and scene kinds, and in-flight retirement without the numerical frontend.
This is a compiler/Runtime qualification, not by itself a full GeoPhys OptiX frame or
a renderer throughput claim.

Passing `face_rules=OptixFaceRuleTable("faces")` directly supplies one table in
instance ordinal / GAS primitive order. Repeated instances of a shared GAS each
have their own slice. Ordinary JIT and components use the same query declaration;
component `optix.query` version 5 retains one array port. Consumer preparation
partitions it using the actual scene's GAS counts and checks total extent,
integer type, compact layout, aliases and ownership. Neither producer instance
count nor table contents become artifact constants. The table is borrowed
without a Host readback or copy. Values must be 0/1 and remain unchanged while
bindings are in use; replace storage and prepare at a cold boundary. Accepted
work retains the old table and scene leases until retirement.

Native-only arrays also enter the component contract. Arrays used by Kernels
retain their full element ABI; native dtype/layout alternatives remain checked
by provider preparation. Tests freeze two instances, then consume 3/7 instances
with different GAS primitive counts, shared GAS and different face policies.
Analytic ray results, typed/occlusion queries, both `graph_fixed_query` choices,
ordinary/domain submission and in-flight table closure are checked. Alpha and
micromap qualification is unchanged; this is not a throughput claim.

### Independent Field inputs and cold state changes

Read-only external Field ports bind an independent consumer to qualified Dense
members owned by a source component instance. `source.borrow(name)` returns a
generation-bound input for `consumer.instantiate(dimensions, inputs={...})`.
Type, shape, native layout and Runtime ownership remain checked. Source storage
stays with the source; consumer output, history and scratch stay independent.

`source.retire_bindings()` closes its prepared writers and external readers while
keeping storage and existing input tokens. Cold mutations implemented by ordinary
operations, such as a selected-world state write, can instead call
`source.retire_bindings(invalidate_borrows=True)`. This also revokes existing input
tokens and prevents old consumers from preparing again. Obtain fresh inputs after
the cold mutation and prepare new calls, including the source step. Normal steps
do not invalidate inputs. Accepted work retires without holding the Runtime
submission lock; caller-owned execution domains remain open. This operation does
not mutate state or authorize concurrent cold mutation and new preparation.

### Shared compiler records in release contracts

`Module.archive(path, shared_component_contracts=True)` can store repeated complete
call facts, compiler facts and control Kernel leaves in `component_contracts/<sha256>.json`.
Each component retains its own document in a versioned storage envelope. Sharing is
selected only when it reduces bytes at the archive's deflate settings, including new
file headers. Small contracts may keep their original representation. `save()` accepts
the same option; it defaults to false for compatibility with existing consumers.

`component_contract_records_v2` also shares exact complete resources, groups,
dimensions, values and compilation.input_sources sections. For example, a local
TAA resolve-filter specialization need not duplicate its common resource/layout
contract. The builder compares v1, section-only sharing and sections plus call
records, choosing the smallest compressed representation including file headers.
Small record overhead cannot hide section-sharing gains. Component names,
semantic identities and configuration guards remain separate; expanded contracts
are unchanged and have independent mutable dictionaries. v2 requires its matching
reader, which also accepts v1. It neither merges instances nor removes necessary
static algorithm differences.

The same option shares exact resource-group schema strings in `component_entries/`
using `component_entry_records_v1`. Group names, member bindings and entry identities
remain local, and schema strings retain their exact spelling. `export_package()`
enables this storage selection by default, only sharing when compression and file
headers together become smaller. `Module.component_entry_contract()` restores entries
during cold preparation without adding per-step work. The matching Runtime entry
reader is required; the frozen-archive tool supports these records as well.

Runtime inspection and loaded components share one reader. It verifies record digests,
reference positions and the expanded contract digest, then runs the existing configuration,
ABI, resource, effects and plan checks. Logical calls receive independent mutable
dictionaries; disk sharing does not share instance state. This storage layer compares
complete JSON records without dropping fields, normalizing paths or inferring code
equivalence. Numerical code and native plans are unchanged.

New readers accept old contracts. Old Runtime readers reject the new envelope, so release
packages must ship with a compatible reader. Build and inspection tools should use
`inspect_component()` instead of treating `components/*.json` as an unwrapped public
contract API. Contract storage savings, numerical code sharing and final wheel size
remain separate acceptance results.

The producer tool `_component_archive.py SOURCE DESTINATION --report FILE` applies
the same storage conversion to an existing frozen compilation. It requires a new
destination, verifies exact contract roundtrips, retains numerical code, plans and
other payloads, and reports hashes, byte totals and non-contract compression changes.
It does not combine modules or infer code equivalence. This reuse of qualified
compilations does not replace migration to normal production libraries.

`shared_execution_plans=True` separately enables `execution_plan_records_v1`
storage for exact repeated Kernel call records under `execution_records/`.
Only operation and initialization leaves are shared; control ordering, labels,
arguments, resources, domain recipes and all validation facts remain intact.
`Module.execution_plan()` restores the original document before the ordinary
ExecutionPlan parser and native preparation. No extra per-step interpreter or
mutable module binding is introduced. Old readers reject this storage envelope.
The frozen-archive tool accepts `--shared-execution-plans`; its unchanged-payload
hash list then excludes plan storage as well as contract storage. Both original
documents are checked for exact restoration, while numerical code stays unchanged.
