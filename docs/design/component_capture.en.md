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

This is the first A1 slice, not a deployed component SDK. Dense Field groups must
use the shared authoritative SNode layout and consumer instantiation rules; they
are explicitly rejected by `Array`. Symbolic layout, declared dimension relations,
Graph export, TCM-only consumption and loaded-module instance lifetimes remain
separate implementation gates. The current per-call Ndarray binding test does not
qualify same-loaded-module Field instances or asynchronous capture.
