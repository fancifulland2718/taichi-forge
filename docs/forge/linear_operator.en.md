# LinearOperator and Experimental SolvePlan

`ti.linalg.LinearOperator` provides a runtime-bound linear-map abstraction for
stored sparse matrices, compiled Taichi kernels, and compiled Graphs. It is a
general numerical API: applications may use it for discretized PDEs, implicit
systems, graph problems, optimization, or other linear algebra without
introducing physics-domain objects into the Taichi DSL.

`LinearOperator`, `OperatorTraits`, storage-view helpers, composition helpers,
and operator qualification are public `ti.linalg` APIs. Solver execution
objects such as `SolvePlan`, `PreconditionerPlan`, and `BatchedSolvePlan`
remain under `ti.linalg.experimental`; their backend and numerical support
boundaries are documented separately below.

## Core model

A `LinearOperator` represents `y = A x` with:

- a scalar dtype and `(rows, columns)` shape;
- one concrete provider bound to the current Taichi `Program`;
- explicit mathematical traits such as self-adjointness and positive
  definiteness;
- observable capabilities and resource-generation metadata; and
- one reusable native execution plan.

Public vector arguments retain a one-dimensional scalar-flat mathematical ABI.
`LinearOperator.apply()` accepts scalar one-dimensional Taichi ndarrays,
supported dense fields, explicit `DenseNdarrayView` objects, or `VectorView`
objects. Single-system `SolvePlan.solve()` accepts scalar one-dimensional
ndarrays, supported dense fields, or `VectorView` objects; explicit
`DenseNdarrayView` operands are not part of its contract. Vector payloads do
not pass through the host, matrix-free providers are not materialized, and
unsupported operations do not change backend.
All operators, plans, providers, ndarrays, and field views belong to one runtime
generation. They become invalid after `ti.reset()` and cannot be rebound to a
later `ti.init()` session.

### Precision boundary

Compiled-kernel and compiled-Graph matrix-free providers currently use an
exact `ti.f32` vector ABI on CPU, CUDA, and Vulkan. Consequently, GPU
matrix-free solver execution through these providers is qualified for
`ti.f32` only. `ti.f64` support for this ABI is outside the current release
boundary.

The runtime does not silently downcast `ti.f64` vectors, materialize a
matrix-free provider, or substitute a different provider or backend.
Unsupported dtype/provider/backend combinations fail explicitly. Stored
operators and dense fields may still use `ti.f64` where the corresponding
provider, solver, and backend rows in the support tables below allow it.

## Dense fields and VectorView

A supported dense field may be passed directly as `input`, `out`, or `addend`
to `LinearOperator.apply()`, and as `rhs`, `initial_guess`, or `out` to
`SolvePlan.solve()`:

```python
rhs = ti.field(ti.f32, shape=(nx, ny))
solution = ti.field(ti.f32, shape=(nx, ny))

operator.apply(rhs, out=solution)
result = plan.solve(rhs, initial_guess=solution, out=solution)
assert result.solution is solution
```

The supported field contract is:

- `ti.f32` or `ti.f64`, subject to matching dtype support in the selected
  operator, provider, and backend;
- a 1D, 2D, or 3D `root -> dense -> place` scalar field;
- a canonically packed `ti.Vector.field` or `ti.Matrix.field`; and
- fixed shape, the current `Program`, and a live `SNodeTree`.

A scalar field whose records contain fixed sibling padding can still be a
supported staged full view, but it is not a direct compact candidate. Packed
Vector and Matrix fields must retain their canonical component layout.

A field maps to operator space in scalar-flat order: canonical index-shape
order first, followed by Vector lanes or row-major Matrix components. The
scalar extent is therefore `prod(index_shape) * prod(element_shape)` and must
match the operator domain or range exactly. Sparse SNodes such as `pointer`,
`bitmasked`, `dynamic`, and `hash`, quantized storage, arbitrary nested dense
trees, and noncanonical packed-component placement fail before submission.

Use `vector_view()` to declare an explicit scalar subset or permutation of a
dense field:

```python
indices = ti.ndarray(ti.i32, shape=active_size)
indices.from_numpy(active_scalar_indices)

rhs_view = ti.linalg.vector_view(rhs, indices=indices)
solution_view = ti.linalg.vector_view(solution, indices=indices)
result = active_plan.solve(rhs_view, out=solution_view)
```

`indices` may be a one-dimensional `ti.i32` ndarray or root-dense scalar field.
View construction copies, validates, and freezes the index topology. Indices
must be nonempty, in the source scalar-extent range, and unique. Later mutation
of the original indices does not change an existing view. Construction performs
one explicit host validation; vector values remain device resident throughout
apply and solve. Indexed scatter overwrites selected scalar entries only and
leaves all other field entries unchanged.

Dense-field execution is provider-qualified. The overwrite form
`operator.apply(input, out=output, alpha=1, beta=0)` directly binds canonical
compact full fields when the selected provider reports
`dense_storage_operands=True`:

```text
dense field descriptor -> resolved range + submission lease -> provider operands
```

Compiled-kernel providers accept direct field operands on CPU, CUDA, and
Vulkan. Fixed native CSR/BSR providers accept them on CPU and CUDA; the Vulkan
native sparse provider remains staged. Compiled Graph providers do not accept
direct field operands. Direct input and output must be non-aliasing, have the
exact dtype and scalar extent, and both qualify for a compact scalar-flat
mapping.

All other supported cases retain an explicit device-staging path:

```text
dense field/view -> device pack or gather -> scalar ndarray provider ABI
scalar ndarray result -> device unpack or scatter -> dense field/view
```

This includes indexed views, padded or non-compact fields, generalized apply,
`out=None`, and every `SolvePlan.solve()` field boundary. Each operator or plan
owns and reuses compatible staging ndarrays. A warm solve does not allocate
staging, and conversions occur only at apply/solve boundaries, never inside a
Krylov iteration. A field `out` is unpacked or scattered before the synchronous
API returns. RHS/input may not overlap output. An `initial_guess` or `addend`
may be the exact same view as output; nonexact overlap fails. Stable raw-field
bindings are qualified once per operator or plan and then reuse the same
implicit view and transfer plan. Native bulk transfer is used where the backend
supports it; other staged layouts use compiled conversion Graph replay.

Capabilities and actual conversion costs are observable:

```python
capabilities = ti.linalg.vector_io_capabilities()
view_metadata = rhs_view.metadata
stats = plan.statistics()["vector_io"]
```

`VectorView.metadata["zero_copy_candidate"]` reports only whether the physical
full-field layout can be flattened without a copy; the provider capability and
the requested operation still decide execution. Statistics report direct
dense-field submissions, staging builds/reuses/reserved bytes, implicit-view
and transfer-plan builds/reuses/evictions, native/Graph transfer submissions,
pack/unpack and indexed gather/scatter calls, logical bytes, direct ndarray
bindings, completion synchronizations, and coalesced operator
synchronizations. `execution_mode="device_staged"` means the field API is
supported without moving vector values through the host; it does not mean
provider-native zero-copy.

## Stored operator and CG

Use `aslinearoperator()` or `LinearOperator.from_sparse_matrix()` with a fixed
CSR/BSR matrix. The operator strongly retains the matrix and uses its existing
pattern and numeric storage without copying it.

```python
import taichi_forge as ti

ti.init(arch=ti.cuda)

operator = ti.linalg.aslinearoperator(
    A,
    traits=ti.linalg.OperatorTraits.spd(),
)

plan = ti.linalg.experimental.SolvePlan(
    operator,
    method="cg",
    max_iterations=200,
    atol=1e-7,
    rtol=1e-5,
)
result = plan.solve(rhs)

if not result.converged:
    raise RuntimeError(result.termination_reason)
x = result.solution
```

Mutable Eigen CSR/CSC matrices remain supported by the established
`SparseCG`, `SparseMINRES`, `SparseBiCGSTAB`, and `SparseSolver` APIs. The
stored provider accepts fixed CSR/BSR only so that topology,
numeric generations, and provider ownership have one stable contract.

## Compiled kernel provider

`LinearOperator.from_kernel()` accepts an exact f32 ndarray ABI. With separate
topology and numeric data, the signature is:

```python
@ti.kernel
def apply_diagonal(
    active_size: ti.i32,
    topology: ti.types.ndarray(dtype=ti.i32, ndim=1),
    numeric: ti.types.ndarray(dtype=ti.f32, ndim=1),
    x: ti.types.ndarray(dtype=ti.f32, ndim=1),
    y: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for i in range(active_size):
        y[i] = numeric[i] * x[topology[i]]

operator = ti.linalg.LinearOperator.from_kernel(
    apply_diagonal,
    size,
    topology,
    numeric=numeric,
    traits=ti.linalg.OperatorTraits.spd(),
)
```

`size` may also be `(range_extent, domain_extent)`. A rectangular provider
must register an independent adjoint kernel through `adjoint=` before
`operator.adjoint()` is available:

```python
operator = ti.linalg.LinearOperator.from_kernel(
    forward_kernel,
    (rows, columns),
    topology,
    adjoint=adjoint_kernel,
    numeric=values,
)
```

The forward `active_size` is the range extent and the adjoint `active_size` is
the domain extent. Put dimensions required by both actions in an explicit
topology resource. A missing adjoint fails explicitly; the implementation does
not use autodiff, host materialization, or a self-adjoint trait as a fallback.

Without `numeric=`, the exact signature is `(active_size, operator_data, x,
y)`. All data arguments are scalar ndarrays; topology and numeric inputs may
have their own scalar dtypes but vectors are f32. The kernel must overwrite
every output entry and must not depend on an SNode tree. Construction compiles
one specialization and copies topology/numeric inputs into operator-owned
snapshots.

## Compiled Graph provider

`LinearOperator.from_graph()` binds a compiled Graph whose dynamic vector
arguments are named `input` and `output`. Every other Graph argument must be
assigned exactly one role:

```python
operator = ti.linalg.LinearOperator.from_graph(
    graph,
    size,
    fixed_i32={"active_size": size},
    topology={"row_offsets": row_offsets, "columns": columns},
    numeric={"values": values},
    workspace={"temporary": temporary},
    traits=ti.linalg.OperatorTraits.spd(),
)
```

The provider is f32. `size` may be an integer square shorthand or `(range,
domain)`, and `adjoint=adjoint_graph` may register an independent adjoint Graph
with the same resource-role schema. It requires at least one topology ndarray
and rejects SNode-dependent dispatches. Topology, numeric data, and workspace
are copied into operator-owned resources. CPU lowers the Graph to an explicit
sequence; CUDA and Vulkan use the compiled-Graph execution contract. Backend
capture/replay may use an ordinary Graph fallback when the documented Graph
runtime rules require it, without changing the mathematical provider. Vulkan
Graph replay requires at least two dispatches. A one-dispatch Graph still
executes correctly, while `operator.statistics()` reports
`ordinary_graph_fallback`.

## Mathematical traits

`OperatorTraits` uses `None` for unknown and `bool` for an explicit caller
claim:

```python
traits = ti.linalg.OperatorTraits(
    self_adjoint=True,
    positive_definite=True,
    positive_semidefinite=True,
    singular=False,
)
```

`OperatorTraits.spd()` is the equivalent convenience constructor. CG and PCG
require trusted `self_adjoint=True` and `positive_definite=True` claims and
reject an operator declared singular. Shape alone, a positive diagonal sample,
or an empirical product check does not establish these properties.

Traits are contracts for every numeric generation used through the operator.
When `update_numeric()` changes coefficients, the caller must ensure that the
declared properties remain valid. Structurally safe compositions derive the
traits they can prove; unsupported inferences remain unknown.

## Apply and composition

```python
y = operator.apply(x)
operator.apply(x, out=y)
y = operator.apply(x, alpha=2.0, beta=-0.5, addend=z)
y = operator @ x

B = 2.0 * operator
C = operator + B
D = operator.compose(B)       # operator(B(x))
E = operator.adjoint()
F = ti.linalg.block_diagonal((operator, B))
I = ti.linalg.identity(size, dtype=ti.f32)
```

The generalized form is `out = alpha * A(x) + beta * addend`. Input/output
aliasing is always rejected. `addend` may alias `out` for in-place
accumulation. When `beta == 0`, `addend` is neither validated nor read.
Generalized coefficient lowering is currently available on CPU. CUDA and
Vulkan accept overwrite apply (`alpha == 1`, `beta == 0`) and fail explicitly
for other combinations without a host fallback.

`adjoint()` is available only when the
provider exposes explicit adjoint application; no self-adjointness assumption
is used as an implementation fallback.

Scale, sum, composition, block diagonal, and identity execute on CPU. Their
GPU lowering is not part of the current API. Attempting GPU composition fails
at construction instead of running host code or synchronizing through an
implicit staging path.

## SolvePlan and SolveResult

`SolvePlan` retains its operator, solver state, and persistent workspace across
calls. Supported methods are:

- `method="cg"`: identity-preconditioned conjugate gradient for SPD systems;
- `method="pcg"`: PCG with `"jacobi"` on fixed CSR,
  `"block_jacobi"` on fixed BSR, or a trusted SPD `LinearOperator` or
  `PreconditionerPlan`
  that applies a fixed-linear approximate inverse; and
- `method="minres"`: identity- or SPD-preconditioned MINRES for square,
  self-adjoint systems that may be indefinite; and
- `method="bicgstab"`: identity- or fixed-linear right-preconditioned
  BiCGSTAB for general square systems; and
- `method="gmres"`: restarted, identity- or fixed-linear
  right-preconditioned GMRES for general square systems; and
- `method="fgmres"`: restarted flexible GMRES with a finite
  variable-linear right-preconditioner action table.

```python
result = plan.solve(rhs, initial_guess=x0, out=x)
print(result.iterations, result.residual_norm, result.termination_reason)
stats = plan.statistics()
```

For the coefficient-invariant compatibility path, a fixed-linear
preconditioner may be passed as an operator rather than as an application
callback:

```python
plan = ti.linalg.experimental.SolvePlan(
    operator,
    method="pcg",
    preconditioner=inverse_operator,
    max_iterations=200,
    atol=1e-7,
    rtol=1e-5,
)
```

`inverse_operator` maps the range of `operator` back to its domain and must
have the same dtype. It must carry trusted self-adjoint, positive-definite,
and nonsingular traits. CPU accepts any provider combination supported by
the operator execution plan. CUDA and Vulkan require both the system
operator and preconditioner to be compiled-kernel providers. Their topology
and numeric generations are pinned together for each solve; there is no
host callback or backend fallback.

### MINRES

`method="minres"` consumes the same `LinearOperator` and lifecycle contracts
as CG/PCG, but requires a trusted `self_adjoint=True` trait rather than
positive definiteness:

```python
operator = ti.linalg.LinearOperator.from_sparse_matrix(
    A,
    traits=ti.linalg.OperatorTraits(
        self_adjoint=True,
        singular=False,
    ),
)
plan = ti.linalg.experimental.SolvePlan(
    operator,
    method="minres",
    max_iterations=300,
    atol=1e-8,
    rtol=1e-5,
    execution_policy="host_check_every_k",
    check_interval=4,
)
result = plan.solve(rhs)
```

CPU supports identity-preconditioned `f32/f64` MINRES for every compatible
CPU operator provider. CUDA and Vulkan support `f32` identity-preconditioned
MINRES for fixed CSR/BSR and compiled providers. On CUDA and Vulkan, a fixed
CSR may select `"jacobi"`, a fixed BSR with block size 2, 3, 6, or 12 may
select `"block_jacobi"`, and a trusted device-native fixed-linear
`LinearOperator` or `PreconditionerPlan` may be supplied directly. A MINRES
preconditioner must be self-adjoint, positive-definite, and nonsingular;
applications remain responsible for ensuring that a selected scalar Jacobi
inverse satisfies that mathematical contract.

MINRES rejects an operator declared `singular=True`. It does not implement
MINRES-QLP or minimum-length semantics for compatible singular systems. Both
halves of an explicitly stored symmetric matrix must be present and
consistent. The terminal status is qualified with the true residual of the
original system, including when preconditioning is active.

A CUDA/Vulkan MINRES plan owns nine persistent length-`n` `f32` vectors and
144 bytes of persistent scalar state. These figures exclude caller-owned
operator values, preconditioner resources, RHS/output arrays, backend handles,
and native replay objects. Inspect `statistics()` for the exact plan/provider
telemetry of a concrete configuration.

### BiCGSTAB

`method="bicgstab"` is the fixed-memory Krylov route for square nonsymmetric
operators. It accepts identity preconditioning or a fixed-linear
`LinearOperator`/`PreconditionerPlan` applied on the right:

```python
plan = ti.linalg.experimental.SolvePlan(
    operator,
    method="bicgstab",
    preconditioner=inverse_operator,
    max_iterations=300,
    atol=1e-8,
    rtol=1e-5,
    execution_policy="host_check_every_k",
    check_interval=4,
)
result = plan.solve(rhs)
```

The preconditioner maps the operator range back to its domain, has the same
dtype, is fixed-linear, and must not be declared singular. Unlike PCG and
MINRES preconditioners, it is not required to be self-adjoint or positive
definite. Right preconditioning keeps terminal qualification in the original
system: convergence is accepted only after evaluating the true residual
`b - A x`.

CPU supports `f32/f64` host-action providers. CUDA and Vulkan support `f32`
fixed CSR/BSR and compiled kernel/Graph providers. Compiled A/M providers use
direct native submissions. Identity-preconditioned fixed stored A providers
can reuse CUDA Graph or Vulkan command-sequence iteration chunks. No provider
is copied to the host to satisfy another backend path.

A device identity plan owns six persistent length-`n` vectors; right
preconditioning adds two vectors for the preconditioned directions. Both
configurations own 112 bytes of scalar state. `statistics()` reports exact
A/M applications, dot products, vector updates, logical/executed/wasted
iterations, host observations, replay activity, workspace bytes, and
`preconditioning_side`. `SolveResult.breakdown_reason` distinguishes
`nonfinite`, `rho`, `alpha_denominator`, `omega_denominator`, and `omega`
failures from ordinary maximum-iteration termination.

BiCGSTAB can stagnate or break down on a nonsingular problem. It is a
low-storage general-system option, not a stability substitute for a qualified
GMRES-family method.

### Restarted GMRES

`method="gmres"` provides a bounded-memory Krylov route for general square
operators when BiCGSTAB's short recurrence is not sufficiently robust.
`restart` is a plan-build parameter and must be `8`, `16`, or `32`; the
default is `16`.

```python
plan = ti.linalg.experimental.SolvePlan(
    operator,
    method="gmres",
    preconditioner=inverse_operator,
    restart=16,
    max_iterations=160,
    atol=1e-8,
    rtol=1e-5,
    execution_policy="host_check_every_k",
    check_interval=16,
)
result = plan.solve(rhs)
```

The implementation uses two-pass classical Gram-Schmidt (CGS2) on every
Arnoldi step. Orthogonalization uses a multi-dot reduction followed by a
fused projection, rather than issuing one host-observed dot product for every
basis vector. A cycle applies Givens rotations on device and qualifies its
terminal result with the original-system true residual `b - A x`.
`happy_breakdown` is counted separately from
`arnoldi_breakdown`, `orthogonalization_failure`,
`hessenberg_singular`, and `nonfinite` failures.

CPU supports compatible `f32/f64` host-action providers. CUDA and Vulkan
support `f32` fixed CSR/BSR and compiled kernel/Graph providers. An optional
preconditioner must be a nonsingular fixed-linear map from the operator range
back to its domain and is applied on the right. It need not carry the SPD
traits required by PCG or MINRES. `method="gmres"` intentionally accepts only
identity or fixed-linear preconditioning; use FGMRES for the supported
variable-linear schedule.

A plan preallocates a contiguous `(restart + 1) * n` basis. Device identity
plans own `restart + 5` persistent length-`n` vectors; right preconditioning
adds one vector. Hessenberg, Givens, least-squares, multi-dot partial, and
terminal state storage is also persistent, and warm solves allocate no
transient solver workspace. `statistics()` reports
`basis_vector_count`, `basis_reserved_bytes`,
`persistent_vector_reserved_bytes`, `persistent_scalar_reserved_bytes`,
exact A/M, dot, multi-dot, and vector-pass counts, restart cycles,
logical/executed/wasted iterations, and replay activity. Applications should
inspect these values before selecting a larger restart.

For fixed stored identity-preconditioned operators, CUDA Graph and Vulkan
command replay cover a complete restart cycle. Compiled providers and
right-preconditioned plans retain the same numerical implementation but use
direct native submission. CUDA supports `host_check_every_k` with
`check_interval == restart`; Vulkan supports that policy and
`fixed_budget_masked`. The complete cycle is submitted before the host
observes its terminal state, so up to `restart - 1` inactive tail steps can be
executed. Larger restarts may improve a difficult system's convergence but
also increase basis memory, cycle work, and its worst-case inactive tail.

### FGMRES with a variable-linear action table

`method="fgmres"` accepts a `PreconditionerPlan` whose
`behavior="variable_linear"`. The plan contains a finite table of 1 to 32
linear actions; this bounded table makes allocation, generation pinning, and
backend execution explicit without introducing a Python callback into an
Arnoldi step:

```python
schedule = ti.linalg.experimental.PreconditionerPlan(
    operator,
    (inverse0, inverse1, inverse2),
    method="external_multilevel_cycle",
    behavior="variable_linear",
    selection="cyclic",
).setup()

plan = ti.linalg.experimental.SolvePlan(
    operator,
    method="fgmres",
    preconditioner=schedule,
    restart=16,
    max_iterations=160,
    atol=1e-8,
    rtol=1e-5,
    execution_policy="host_check_every_k",
    check_interval=16,
)
result = plan.solve(rhs)
```

The selected action for solve-global scheduled inner slot `k` is
`actions[k % len(actions)]`. A restart does not reset the schedule. CPU
scheduled and logical slots coincide; masked GPU execution may schedule
inactive tail slots after an earlier logical termination, so telemetry reports
selection and executed-iteration counts separately from logical iterations.
All target and action generations are pinned at solve entry. Every action must
belong to the same Program, have the solver dtype, map the operator range back
to its domain, and not be declared singular.

CPU supports compatible `f32/f64` host-action providers. CUDA and Vulkan
support `f32` compatible fixed stored and compiled kernel/Graph providers.
FGMRES uses the same CGS2 Arnoldi, true-residual termination, restart values,
and GPU observation policies as GMRES, but stores every preconditioned basis
vector in a persistent `Z` basis. The additional reservation is
`restart * n * sizeof(dtype)` bytes and is exposed as
`preconditioned_basis_vector_count` and
`preconditioned_basis_reserved_bytes`. For `restart=8`, the qualified CPU and
device plans own 20 and 22 persistent solver vectors respectively, excluding
operator, action, RHS/output, backend, and replay resources.

Variable-action FGMRES currently uses direct native submission. It does not
claim CUDA Graph or Vulkan command replay until every action in the table has
a qualified capture and binding contract; `solver_replay_unavailable_reason`
reports `variable_action_capture_contract_unavailable`. The API does not
support nonlinear preconditioners, Python iteration callbacks, automatic
restart selection, block GMRES, or domain-specific outer-solver policy.
Passing a variable-linear plan to CG, PCG, MINRES, BiCGSTAB, or ordinary GMRES
fails during plan construction.

### Current unsupported boundary

The `0.5.1` numerical-tooling contract intentionally does not provide:

- nonlinear, residual-dependent, adaptive, or Python-callback preconditioners;
- automatic restart selection, block or multi-RHS Krylov methods, recycling,
  deflation, pipelining, or communication-avoiding GMRES variants;
- MINRES-QLP, singular minimum-norm/minimum-length guarantees, or automatic
  nullspace handling;
- GPU `f64` GMRES-family execution, GPU operator composition, or generalized
  GPU `alpha/beta/addend` apply;
- variable-action CUDA Graph/Vulkan command replay, single-system asynchronous
  solve submission, or device-convergent conditional termination;
- dynamic-topology solve plans, ragged batches, or transparent host fallback;
  and
- built-in IC/ILU/AMG, multigrid hierarchy construction, Schur/field splitting,
  domain decomposition, discretization, contact/KKT policy, or nonlinear outer
  iteration.

Independent fixed-size batched CG/PCG remains distinct from block Krylov or
multi-RHS solving. Unsupported backend/provider/policy combinations fail at
construction or capability validation; they are not approximated by changing
the provider, execution policy, or mathematical problem.

## PreconditionerPlan lifecycle

Use `PreconditionerPlan` when coefficients change or when provenance, explicit
reuse, and generation telemetry are required:

```python
preconditioner = ti.linalg.experimental.PreconditionerPlan(
    operator,
    inverse_operator,
    method="external_block_inverse",
).setup()

z = preconditioner.apply(r)
pinned = preconditioner.pin()

# Each provider publishes its next numeric generation first.
operator.update_numeric(next_a, expected_topology_version=1,
                        expected_numeric_version=3)
inverse_operator.update_numeric(next_m, expected_topology_version=1,
                                expected_numeric_version=7)
preconditioner.update()  # next_m was rebuilt from the current operator

pcg = ti.linalg.experimental.SolvePlan(
    operator, method="pcg", preconditioner=preconditioner
)
```

`built_from_operator_stamp` records the operator generation from which the
action was actually built. `accepted_target_stamp` records the generation the
action is currently approved to serve. A target update makes the plan stale by
default. If lagged preconditioning is valid, call
`preconditioner.update(accept_reuse=True)` while the action is unchanged. This
updates only compatibility and preserves provenance. A changed action requires
an ordinary `update()` rebuild attestation.

`pin()` retains the exact target and action generations together. The returned
`PreconditionerSession` can continue applying that old generation after later
generations are published. A fixed-linear session uses `apply(r, out=None)`;
a variable-linear session accepts `iteration=k` and selects the pinned cyclic
action for that scheduled slot; the default `iteration=0` selects the first
action. `metadata` exposes provenance and compatibility stamps.
`PreconditionerPlan.statistics()` reports setup, rebuild, reuse, stale
rejections, schedule update success/failure, and approved-generation
publish/retire/release counters. `SolvePlan.statistics()` additionally reports
action selections and schedule wraps. Setup and update execute at host
boundaries; session apply and solver iterations invoke native `OperatorAction`
objects without Python callbacks.

For a variable-linear table, `update(accept_reuse=...)` accepts either one
boolean for all actions or one boolean per action. The complete next table is
validated before any generation is published: one stale or incompatible
action rejects the whole update. A solve that already pinned the previous
table retains that immutable snapshot. `fixed_linear` is supported by the
documented PCG, MINRES, BiCGSTAB, and GMRES consumers; `variable_linear` is
supported only by FGMRES. `nonlinear` remains descriptive and returns a
structured unsupported reason. Built-in Jacobi and block-Jacobi use the same
native setup/update/pin lifecycle and remain selected with
`preconditioner="jacobi"` or `"block_jacobi"`.

`rhs`, `initial_guess`, and `out` must match the operator dtype and scalar
extent and belong to the current runtime. If `out` is omitted, the plan creates
a result ndarray. If `initial_guess` is omitted, the result is initialized to
zero. RHS/output aliasing is rejected.

`SolveResult` contains the solution and a terminal snapshot: status code,
termination reason, convergence/breakdown/max-iteration flags, iteration
count, initial and final residual norms, both tolerances, relative reference
norm, and effective tolerance. CG, PCG, MINRES, BiCGSTAB, GMRES, and FGMRES use:

The result record is frozen; its `solution` ndarray remains caller-writable.

```text
||b - A x||_2 <= max(atol, rtol * ||b||_2)
```

`statistics()` exposes backend-neutral plan/provider/workspace counters; it is
diagnostic data, not part of the numerical result.

### GPU execution policy

`execution_policy` controls when a GPU solve observes convergence on the
host:

```python
plan = ti.linalg.experimental.SolvePlan(
    operator,
    method="cg",
    max_iterations=200,
    atol=1e-7,
    rtol=1e-5,
    execution_policy="host_check_every_k",
    check_interval=8,
)
```

- CPU defaults to `"host_each_iteration"`. Stored f32 CG/PCG also accepts
  explicit `"bounded_convergent"`, which preserves the same native loop.
- CUDA fixed stored f32 CSR/BSR CG/PCG defaults to
  `"bounded_convergent"`. In `bounded_mode="auto"`, a qualified runtime uses
  an exact device-side CUDA conditional Graph; an unavailable native path
  falls back to reusable Graph chunks with host checks. Explicit
  `"host_each_iteration"` remains available as an opt-out.
- Replay-qualified CUDA fixed stored f32 MINRES/BiCGSTAB defaults to
  `"host_check_every_k"` with `check_interval=4`.
- Qualified square matrix-free Kernel/Graph plans on CUDA also select
  `"host_check_every_k"` automatically: CG, Kernel-PCG, MINRES, and BiCGSTAB
  use K=4, while GMRES/FGMRES use `check_interval == restart`. Chunked
  execution keeps recurrence scalars on the device and reads one terminal
  snapshot per chunk. K=8 remains available for the non-GMRES methods, and
  explicit `"host_each_iteration"` remains available where supported. CUDA
  BiCGSTAB with a compiled Graph provider retains `"host_each_iteration"` by
  default because short converging solves did not show a stable K=4 benefit;
  callers can still opt in.
- CUDA GMRES/FGMRES defaults to `"host_check_every_k"` and requires
  `check_interval == restart`. Stored identity-preconditioned GMRES records a
  reusable restart-cycle Graph; FGMRES and other non-recordable provider
  combinations preserve direct submission.
- Replay-qualified Vulkan fixed stored f32 CG/PCG/MINRES/BiCGSTAB defaults to
  `"host_check_every_k"` with K=4; stored identity-preconditioned GMRES uses
  the same default with `check_interval == restart`. Qualified square
  matrix-free Kernel/Graph plans use the same automatic K=4 or restart-sized
  host checks, including FGMRES. Explicit `"fixed_budget_masked"` remains
  available for workloads that intentionally consume the full iteration
  budget. Both policies support `atol`, `rtol`, and their combined effective
  tolerance.

For fixed stored f32 CSR/BSR, CUDA `host_check_every_k` and Vulkan
`host_check_every_k`/`fixed_budget_masked` record supported CG/PCG/MINRES and
identity-preconditioned BiCGSTAB iteration chunks and
identity-preconditioned GMRES restart cycles as reusable native execution
sequences. The recordable CG/PCG/MINRES combinations include identity, stored
Jacobi, and stored block-Jacobi preconditioners. The first compatible execution
builds a CUDA Graph or Vulkan command sequence; later executions with the same
topology, workspace, and output binding replay it. A values-only matrix update
does not re-record an identity-preconditioned GMRES sequence; supported
preconditioner refreshes retain the existing CG/PCG/MINRES sequence.
Replacing the output ndarray, changing topology or schema, or recreating the
runtime invalidates and safely rebuilds it.

For qualified CUDA stored CG/PCG, the conditional Graph owns the complete
recurrence loop and evaluates convergence on the device after every logical
iteration. A solve retains one initial and one terminal state observation but
does not perform an iteration-by-iteration host scalar reduction or implicit
synchronization. The Graph terminates at the exact logical iteration, so this
path has no masked tail work, and a persistent plan replays the cached
executable while its topology, workspace, and output binding remain stable.

FGMRES action tables use direct native submission on both GPU backends. No
identity-GMRES replay path is silently reused for a variable action schedule.

Compiled-kernel and compiled Graph A/M providers use direct outer solver-chunk
submission. This outer recurrence boundary is independent of provider
execution: each compiled Graph apply uses its provider-owned compiled Graph
plan, while a compiled-kernel apply uses an ordinary compiled kernel launch.
Replay-qualified multi-dispatch Graphs record and replay on CUDA/Vulkan;
ineligible plans preserve ordinary execution and report the backend path. In
particular, single-dispatch Vulkan Graphs intentionally use the ordinary path
because recording them adds no useful consolidation. The solver does not nest
either provider inside another captured Graph. Providers are not staged
through the host or replaced to obtain replay.

`statistics()` exposes the outer boundary through `solver_chunk_builds`,
`solver_chunk_replays`,
`solver_chunk_direct_submissions`, `solver_chunk_rebinds`,
`solver_chunk_invalidations`, `solver_graph_enabled`, and
`solver_replay_unavailable_reason`. It independently reports provider execution
through `operator_execution_kind`, `operator_compiled_graph_submissions`,
`operator_backend_captures`, and `operator_backend_replays`. Build cost belongs
to cold execution, so qualification should report first-solve and warm-solve
timing separately.

A chunk or GMRES-family restart cycle always completes before its terminal state is
inspected. The reported
`SolveResult.iterations` is the logical convergence or breakdown iteration;
`statistics()["operations"]` separately reports `executed_iterations`,
`wasted_iterations`, host synchronization counts, and direct chunk
submissions. A chunked solve can therefore execute up to `K - 1` masked
or otherwise inactive tail iterations. Vulkan fixed-budget execution may
execute the full `max_iterations` while preserving an earlier logical
termination result.

For CG/PCG/MINRES/BiCGSTAB, use `K=4` when earlier termination is more
important than synchronization frequency and `K=8` when amortizing host
checks is more important. GMRES and FGMRES use their selected restart as the
observation interval. The faster choice depends on vector size, operator cost, iteration
count, driver, and backend. Unsupported policies and intervals fail during
plan construction; they do not silently fall back.

`plan.execution_capabilities()` reports the policy matrix and a structured
reason for unavailable conditional execution, together with the selected
`default_execution_policy`.
The `automatic_solver_batching` object reports matrix-free Kernel/Graph
host-check selection, the default interval, the direct-chunk backend primitive,
and whether provider execution uses a compiled Graph plan or compiled-kernel
launch. Batching does not require or claim outer solver replay.
The separate `automatic_solver_replay` object reports whether outer recurrence
replay was selected, whether the operator and preconditioner combination is
qualified, and the backend primitive (`cuda_conditional_graph_or_chunk_replay`,
`cuda_graph_chunk_replay`, or `vulkan_command_replay`). Post-solve statistics
remain authoritative for the actual replay or direct-submission path.
Direct `"device_convergent"` execution is qualified only for single-system
stored f32 CSR/BSR CG/PCG on CUDA when the driver,
conditional-Graph entry points, device setter, provider capture, and cuBLAS
workspace requirements are all satisfied. An explicit direct request fails
without fallback. The default `"bounded_convergent"` policy instead attempts
that native path automatically and uses its documented chunked fallback when
necessary. Compiled-kernel, compiled Graph, batched, CPU, and Vulkan providers
do not claim CUDA conditional-Graph execution.

## Independent batched CG and PCG

`BatchedSolvePlan` solves a homogeneous batch of independent SPD systems with
one persistent plan. It uses one flat direct-sum operator with shape
`(B * N, B * N)` and partitions every vector into `B` contiguous systems of
length `N`:

```python
plan = ti.linalg.experimental.BatchedSolvePlan(
    operator,
    batch_size=B,
    independent_systems=True,
    method="cg",
    max_iterations=100,
    atol=1e-7,
    rtol=1e-5,
    execution_policy="host_check_every_k",
    check_interval=4,
)
result = plan.solve(rhs_flat, out=x_flat)

for env, reason in enumerate(result.termination_reasons):
    if not result.converged[env]:
        raise RuntimeError(f"system {env}: {reason}")
```

`independent_systems=True` is a required caller assertion. The system
operator and, for PCG, the fixed-linear preconditioner must preserve every
partition: output entries for environment `e` may depend only on input entries
from environment `e`. Global SPD traits do not prove this partition property.
Violating it gives an invalid independent-batch problem rather than a coupled
solve.

The first public layout is deliberately homogeneous and flat:

- the operator extent must be divisible by `batch_size`;
- all systems have the same scalar extent and f32 dtype;
- `rhs`, `initial_guess`, and `out` have shape `(B * N,)`;
- `atol` and `rtol` may be scalars or length-`B` sequences; and
- variable offsets, active compaction, and ragged systems are not part of this
  contract.

`method="cg"` uses identity preconditioning. `method="pcg"` requires a
trusted SPD `LinearOperator` that applies a fixed-linear approximate inverse
over the same flat partitions. Fixed stored and compiled-kernel A/M providers
are qualified on CPU, CUDA, and Vulkan. Other provider kinds remain subject to
their ordinary backend capability and qualification boundaries; the batched
plan never stages through the host or changes provider.

Each environment owns its recurrence scalars, effective tolerance, status,
logical iteration count, and residual norms. One environment may converge,
break down, or reach `max_iterations` without changing another environment's
terminal result. `BatchedSolveResult` exposes these values as immutable tuples
and returns the flat solution ndarray separately. A batch of one follows the
same numerical contract as a single-system CG/PCG solve.

CPU uses `"host_each_iteration"`. CUDA and Vulkan default to
`"host_check_every_k"` with `K=4`; they also accept `K=8`, explicit
`"host_each_iteration"`, or `"fixed_budget_masked"`. A host-check chunk may
issue inactive tail iterations before observing that all environments have
terminated. Recurrence and vector-update kernels mask inactive environments,
but the monolithic A/M provider still applies to the full flat batch. Provider
apply compaction is therefore not implied by convergence masking.

CUDA and Vulkan compile the stable iteration recurrence into plan-owned
Taichi Graphs and reuse those graphs across solves. CG submits one recurrence
Graph per iteration; PCG submits one segment after A and another after M. The
operator and preconditioner remain pinned provider actions outside these
graphs, so stored and compiled-kernel generations retain their normal update
and retirement contracts. Replacing `out` patches the Graph binding after the
previous solve completes. Each workspace clone owns an independent replay
plan and therefore does not serialize through another clone's Graph lock.

`statistics()` makes this distinction observable through executed system
iterations, provider system iterations, masked provider system iterations,
active efficiency, host checks, transfers, and persistent resource sizes. A CG
plan owns three length-`B * N` workspace vectors; PCG owns four. It also owns
per-environment recurrence, tolerance, and status state. Caller-owned RHS,
solution, initial guess, and provider resources are excluded from these plan
workspace counts. Batched-plan statistics use schema version 4 and report
`recurrence_replay_builds`, `recurrence_replay_graph_builds`,
`recurrence_replay_submissions`, `recurrence_replay_logical_kernels`, output
`recurrence_replay_rebinds`, and direct recurrence-kernel submissions. The
`recurrence_replay` record explicitly states that A/M provider applies are not
part of the Graph replay scope.

### Asynchronous fixed-budget submission

CUDA and Vulkan batch plans using `execution_policy="fixed_budget_masked"`
can submit the complete masked iteration budget without materializing terminal
state at the call boundary:

```python
plan = ti.linalg.experimental.BatchedSolvePlan(
    operator,
    batch_size=B,
    independent_systems=True,
    max_iterations=8,
    atol=1e-6,
    execution_policy="fixed_budget_masked",
)

submission = plan.submit(rhs_flat, out=x_flat)
# Submit independent application work here.
submission.wait()
result = submission.result()
```

When multiple asynchronous solve producers share a GPU, assign independent
lanes to workspace clones and use one `SubmissionPacer` to bound aggregate
backlog and provide fair admission:

~~~python
secondary = plan.clone_workspace()
pacer = ti.graph.SubmissionPacer(
    2,
    max_in_flight_per_lane=1,
    max_queued=8,
)
primary_ticket = plan.submit(
    rhs_a, out=x_a, pacer=pacer, lane='primary_physics'
)
secondary_ticket = secondary.submit(
    rhs_b, out=x_b, pacer=pacer, lane='secondary_physics'
)
primary_result = primary_ticket.result()
secondary_result = secondary_ticket.result()
~~~

The complete host launch sequence for one solve is submitted in one admission
turn. After launch, both tickets may remain incomplete within the
`max_in_flight` bound. Asynchrony here defines a host-completion boundary; it
does not guarantee concurrent GPU kernels, independent streams or queues, or
device preemption. `max_in_flight_per_lane` prevents a high-rate producer
from occupying every slot, while cross-lane round robin gives an existing
waiter bounded admission delay. A blocked caller polls all in-flight
completions with bounded adaptive backoff, so a later fast solve may release
capacity before an older slow solve completes. Only calls sharing the same
pacer are covered; engine frame deadlines, task dependencies, and cancellation
remain application policies.

`SolveSubmission.done()` observes backend completion without releasing the
workspace slot. `wait()` waits when necessary, materializes the complete
per-system terminal snapshot, and releases the slot. `result()` performs the
same operation when needed and returns the immutable `BatchedSolveResult`.
The submission retains the exact A/M generations, input/output ndarrays,
workspace, and backend completion until this boundary. Backend failures are
re-raised by `wait()` and `result()`; `ti.reset()` waits for retained backend
work and makes an outstanding ticket explicitly stale.

Publishing a new operator or preconditioner numeric generation before an old
submission completes retains the old generation through its completion. A high
update rate can therefore keep multiple complete value buffers resident. Include
update cadence in the memory budget, or complete the relevant tickets before
publishing. A pacer bounds participating invocation counts, not generation
bytes.

One `BatchedSolvePlan` owns one submission slot. Submitting again before the
pending ticket is completed and materialized fails instead of sharing Krylov
vectors. Use `clone = plan.clone_workspace()` when independent submissions
must be in flight concurrently. Each clone owns another complete set of CG or
PCG workspace vectors and state. Chunked host-check policies and CPU plans are
not qualified for `submit()`; a call fails explicitly rather than moving a
synchronous loop to a worker thread.

For batch size `B` and per-system size `N`, each f32 CG plan or clone has a
logical workspace payload of `12 * B * N + 68 * B + 8` bytes. PCG uses
`16 * B * N + 68 * B + 8` bytes. Inspect
`statistics()["resources"]["clone_workspace_payload_bytes"]` before creating
a clone pool. Allocator and driver overhead, caller vectors, and operator or
preconditioner resources are excluded. Use `max_in_flight=1` by default for a
large solve. Increase it to 2 only when profiling demonstrates useful host
overlap with acceptable memory and tail latency. For small systems, increase
batching instead of creating one plan clone per application entity.

This API is independent batching. It is not global-scalar CG over an
implicitly coupled block matrix, multi-RHS CG, block CG, or another block
Krylov method. Coupled systems require an operator and solver whose mathematics
model that coupling explicitly.

## Support matrix

### Providers and apply

| Provider | CPU | CUDA | Vulkan |
| --- | --- | --- | --- |
| Fixed stored CSR/BSR | `f32`, `f64` | `f32` | `f32` |
| Compiled kernel | `f32` | `f32` | `f32` |
| Compiled Graph | `f32` | `f32` | `f32` |
| Identity/composition | `f32`, `f64` | Unsupported | Unsupported |

Kernel and Graph providers support rectangular shapes and explicit adjoints.
Generalized `alpha/beta` apply is available on CPU; GPU currently supports
overwrite apply only.

### Solvers

| Method/provider | CPU | CUDA | Vulkan |
| --- | --- | --- | --- |
| CG, fixed stored | CSR/BSR, `f32/f64` | CSR, `f32` | CSR/BSR, `f32` |
| CG, compiled kernel/Graph | `f32` | `f32` | `f32` |
| CG, CPU composition | `f32/f64` | Unsupported | Unsupported |
| PCG + Jacobi | Fixed CSR, `f32/f64` | Fixed CSR, `f32` | Fixed CSR, `f32` |
| PCG + block-Jacobi | Fixed BSR, `f32/f64` | Fixed BSR, `f32` | Fixed BSR, `f32` |
| PCG + fixed-linear operator/plan | Supported providers, `f32/f64` | Compiled-kernel A/M, `f32` | Compiled-kernel A/M, `f32` |
| MINRES + identity | Supported providers, `f32/f64` | Fixed CSR/BSR or compiled provider, `f32` | Fixed CSR/BSR or compiled provider, `f32` |
| MINRES + Jacobi/block-Jacobi | Unsupported | Fixed CSR/BSR respectively, `f32` | Fixed CSR/BSR respectively, `f32` |
| MINRES + fixed-linear operator/plan | Unsupported | Compatible device-native A/M, `f32` | Compatible device-native A/M, `f32` |
| Independent batched CG/PCG | Fixed stored or compiled-kernel A/M, `f32` | Fixed stored or compiled-kernel A/M, `f32` | Fixed stored or compiled-kernel A/M, `f32` |
| Batched fixed-budget submission | Unsupported | Fixed stored or compiled-kernel A/M, `f32` | Fixed stored or compiled-kernel A/M, `f32` |
| Device-convergent conditional execution | Unsupported | Unsupported | Unsupported |
| BiCGSTAB + identity | Supported host-action providers, `f32/f64` | Fixed CSR/BSR or compiled provider, `f32` | Fixed CSR/BSR or compiled provider, `f32` |
| BiCGSTAB + fixed-linear right preconditioner | Supported host-action providers, `f32/f64` | Compatible device-native A/M, `f32` | Compatible device-native A/M, `f32` |
| GMRES + identity | Supported host-action providers, `f32/f64` | Fixed CSR/BSR or compiled provider, `f32` | Fixed CSR/BSR or compiled provider, `f32` |
| GMRES + fixed-linear right preconditioner | Supported host-action providers, `f32/f64` | Compatible device-native A/M, `f32` | Compatible device-native A/M, `f32` |
| FGMRES + variable-linear action table | Supported host-action providers, `f32/f64` | Compatible device-native A/actions, `f32` | Compatible device-native A/actions, `f32` |

Direct factorization remains a stored-matrix API.

## Numeric updates and ownership

Stored operators use `operator.update_numeric(values)`. Compiled providers use
optimistic version checks:

```python
operator.update_numeric(
    next_values,
    expected_topology_version=1,
    expected_numeric_version=3,
)
```

Graph updates pass a complete mapping of numeric roles. A successful compiled
update publishes the next immutable numeric generation. In-flight work keeps
its pinned generation alive; later apply/solve calls observe the new
generation. A stored Jacobi/block-Jacobi PCG plan refreshes its numeric
preconditioner before the next solve while retaining its pattern and Krylov
workspace. Scalar CSR Jacobi stores diagonal reciprocals. BSR block-Jacobi
stores a lower Cholesky factor for every diagonal block; supported block sizes
are 2, 3, 6, and 12. Every block must be finite, symmetric, and positive
definite. Invalid blocks fail explicitly without symmetrization,
regularization, or fallback, and a failed refresh does not publish a new
preconditioner generation. Topology changes require constructing a new
operator.

CUDA and Vulkan perform value-only Jacobi and block-Jacobi refreshes on the
device. A successful refresh preserves the committed resource address, so a
replayable solve plan may rebind the numeric generation without rebuilding its
recurrence program. Warm refreshes do not copy the complete values or factors
through the host and do not allocate device memory; validation reads back only
a fixed-size status. `statistics()` exposes the refresh contract, operation
counters, transfer bytes, and device-allocation count.

The public API has no borrowed-resource mode. Stored operators strongly retain
their matrix; compiled providers own copied topology/numeric/workspace
resources; compositions and solve plans strongly retain their operands.

## Relationship to the legacy matrix-free API

The callback-based `ti.linalg.FieldLinearOperator`, `MatrixFreeCG`, and
`MatrixFreeBICGSTAB` retain their field-shaped vector ABI. They use a `(x, y)`
kernel callback and are not implicitly adapted because that ABI does not carry
explicit topology, numeric-resource, runtime-generation, or capability
information.

Use `FieldLinearOperator` when an application intentionally needs the legacy
field callback contract. Use `LinearOperator` when provider capabilities,
resource generations, runtime storage views, composition, or solver-plan
integration are required.

## Qualification boundary

`qualify_operator()` generates versioned, JSON-serializable protocol evidence
for any public `LinearOperator`:

```python
report = ti.linalg.qualify_operator(
    operator,
    reference=dense_reference,
    samples=4,
    warmup=2,
    repetitions=10,
    metadata={"case": "poisson_level_3"},
)
report.to_json()
matrix = ti.linalg.summarize_operator_qualifications([report])
```

The report contains backend/build identity, provider, shape, capabilities,
resource stamps, forward/adjoint oracle errors, linearity and dot-product
identity, generalized-apply `beta=0` no-read status, synchronous boundary
timing, and native counters. `summarize_operator_qualifications()` builds a
deterministic support matrix from detached backend/provider reports. An
unsupported GPU generalized coefficient is recorded as `unsupported`; it is
neither reported as a pass nor executed by a host fallback. Timing describes
the current machine and run and is not a cross-machine performance gate.

`qualify_solve_plan()` produces the corresponding execution evidence for a
public `SolvePlan` or `BatchedSolvePlan`:

```python
report = ti.linalg.experimental.qualify_solve_plan(
    lambda: ti.linalg.experimental.SolvePlan(
        operator,
        method="pcg",
        preconditioner=preconditioner,
        execution_policy="host_check_every_k",
        check_interval=4,
    ),
    rhs,
    reference=expected_solution,
    expected_termination="converged",
    warmup=2,
    repetitions=10,
    metadata={"case": "poisson_level_3"},
)
matrix = ti.linalg.experimental.summarize_solve_qualifications([report])
```

The example uses the CUDA/Vulkan chunked policy; a CPU factory uses
`host_each_iteration` instead.

Passing a zero-argument factory records plan construction separately. An
already constructed plan is also accepted, with build timing explicitly
unavailable. `reference` is either a flat expected solution or a callable
receiving a detached NumPy copy of the RHS. Independent batches use one flat
reference and may provide one expected termination reason per system.

The report separates first and warm synchronous wall time. A qualified
fixed-budget GPU batch additionally separates host submission from completion
wait and may record a supplied `SubmissionPacer`. It records A/M providers,
policy and check interval, terminal state, an independent `b - A(x)` residual,
logical/executed/provider iterations, inactive-work efficiency, chunk
direct/replay counters, transfers, plan resources, and process-global device
pool deltas. Device timestamp spans, device identity, and driver versions stay
explicitly unavailable when the runtime has no safe query; wall time is never
relabeled as device time. Nsight or another profiler can be retained as a
sidecar using the report metadata.

Qualification performs one first solve, the requested warmups and repetitions,
and one untimed operator apply for the independent true residual. It mutates
the plan counters and output supplied to it. Use a dedicated plan/workspace for
performance evidence, especially when measuring an asynchronous batch or a
shared pacer. The function returns detached evidence and never writes files.

The API is covered by backend correctness, lifecycle, trait, composition, 10k
approved-generation churn, and solver regression tests. Application production
qualification remains workload-specific: validate operator semantics,
conditioning, tolerances, preconditioner suitability, failure handling, memory
budgets, and backend driver behavior on representative physical and
non-physical systems.

See also [Sparse runtime and linear algebra](sparse_runtime_and_linear_algebra.en.md)
and [Choosing sparse operators and solvers](physics_sparse_solver_selection.en.md).
