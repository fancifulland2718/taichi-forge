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

This is not yet a deployed component SDK. Dense Field groups are explicitly
rejected by `Array`; capture lowering must preserve their group contract.

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

The current schema entry uses Program-owned trees. Connecting it to the
standalone Runtime owner, Dense capture addressing, dimension relations, Graph
export, TCM-only consumption and loaded-module instance lifetimes remains work
in progress. Neither this layout comparison nor per-call Ndarray binding qualifies
same-loaded-module Field instances or asynchronous capture.
