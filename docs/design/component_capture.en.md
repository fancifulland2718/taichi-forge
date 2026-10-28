# Component capture compiler: initial implementation

`taichi_forge.aot._component` is an internal producer-side API. It compiles the
original decorated method through the normal AST transformer; it does not parse
the application a second time or create a numerical wrapper.

`Component` declarations describe the owner's attributes. `RuntimeValue` supports
scalar and fixed vector/matrix values, `Array` currently supports scalar Ndarray,
and `StaticValue` accepts immutable scalar/tuple configuration. Entry arguments
currently support scalar and fixed vector/matrix values. The public SDK API is
not frozen.

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
Ndarrays nor mutate module root pointers. This first slice requires qualified
compact members while preserving the full root in the schema.

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
paths, element types and compact layout, returning finite views of that root.
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

Normal GeoPhys construction now provides a bounded numerical RNE gate on CPU/CUDA:
3/7/5 bodies with different topology, mass, qpos/qvel and gravity reuse the original
entry and agree with ordinary JIT. A resting chain also checks analytic gravity
moments. Body state, generalized dynamics and bias-force workspaces have explicit
normal groups shared by JIT and deployment, with no copied numerical definitions
or per-kernel resource tables.

`Component.compile_graph(normal_record)` now records original decorated entries
through GraphBuilder, collects actual captures, and exports the closure through
existing Graph lowering and `Module.add_execution_plan`. Applications do not
maintain kernel lists or binding tables. The first slice covers sequential roots
with one normal owner. Control regions, native nodes, external Ndarray ownership
and recipe search need separate qualification. Root identity includes argument
connections and explicit types; dynamic model values do not specialize code.

Runtime `module.component(name)` queries the immutable contract;
`instantiate(dimensions)` creates independent roots using authoritative native
SNode rules. `initialize(data, zero=...)` explicitly supplies model data or zeros.
Effects are conservatively read/write and every referenced resource requires
initialization; no producer live state is serialized. `instance.prepare(values=...,
dimensions=...)` binds through the existing ExecutionPlan and PreparedCall, with
no new Python execution interpreter. Fixed vector/matrix values reuse native ABI
encoding; every entry consuming a named value must have compatible value ABI.

Each prepared call owns its resource/value bindings, without changing shared
Module root pointers. Instances retain separate state under one loaded code
owner. Close and reset are currently synchronous cold operations: close existing
prepared calls before reinitializing; changed capacity requires a new instance.
A new prepare may change active counts that do not affect layout. Independent
execution domains explicitly reject until lifecycle qualification. This is not
complete C coverage for in-flight retirement, capture updates or concurrency.

GeoPhys `build_bias_force_graph` serves ordinary prestep Graphs and component
building. CPU/CUDA probes freeze a TCM, create two new models with different
topology, mass, qpos/qvel and gravity, copy only Runtime into an isolated process,
and forbid private numerical-source imports. One loaded module serves two
independent instances with interleaved execution and cold-close isolation.
World-state inputs are prepared by normal JIT kinematics outside that process;
this proves isolated finite RNE consumption, not source-free model/FK/whole-step
construction. Initial cost samples separate load, allocation, initialization,
prepare and warm component execution; no speed or scaled-memory claim follows.

Full initialization/effects/constraint projections, finite algorithm-domain
coverage, actual task/launch observation and complete instance lifetimes remain
in progress. Release wheels and whole-engine qualification remain in E.
