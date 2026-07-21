# Experimental LinearOperator and SolvePlan

`ti.linalg.experimental` provides a runtime-bound linear-map abstraction for
stored sparse matrices, compiled Taichi kernels, and compiled Graphs. It is a
general numerical API: applications may use it for discretized PDEs, implicit
systems, graph problems, optimization, or other linear algebra without
introducing physics-domain objects into the Taichi DSL.

The namespace is experimental. Its provider, lifetime, capability, and
failure contracts are defined here; source compatibility may still change
before promotion into the stable `ti.linalg` namespace.

## Core model

A `LinearOperator` represents `y = A x` with:

- a scalar dtype and `(rows, columns)` shape;
- one concrete provider bound to the current Taichi `Program`;
- explicit mathematical traits such as self-adjointness and positive
  definiteness;
- observable capabilities and resource-generation metadata; and
- one reusable native execution plan.

Public vector arguments are one-dimensional scalar Taichi ndarrays. `apply()`
completes before returning. It does not accept NumPy arrays, copy through the
host, materialize a matrix-free provider, or change backend when a requested
operation is unsupported.

All operators, plans, providers, and ndarrays belong to one runtime generation.
They become invalid after `ti.reset()` and cannot be rebound to a later
`ti.init()` session.

## Stored operator and CG

Use `aslinearoperator()` or `LinearOperator.from_sparse_matrix()` with a fixed
CSR/BSR matrix. The operator strongly retains the matrix and uses its existing
pattern and numeric storage without copying it.

```python
import taichi_forge as ti

ti.init(arch=ti.cuda)

operator = ti.linalg.experimental.aslinearoperator(
    A,
    traits=ti.linalg.experimental.OperatorTraits.spd(),
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
experimental stored provider accepts fixed CSR/BSR only so that topology,
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

operator = ti.linalg.experimental.LinearOperator.from_kernel(
    apply_diagonal,
    size,
    topology,
    numeric=numeric,
    traits=ti.linalg.experimental.OperatorTraits.spd(),
)
```

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
operator = ti.linalg.experimental.LinearOperator.from_graph(
    graph,
    size,
    fixed_i32={"active_size": size},
    topology={"row_offsets": row_offsets, "columns": columns},
    numeric={"values": values},
    workspace={"temporary": temporary},
    traits=ti.linalg.experimental.OperatorTraits.spd(),
)
```

The provider is square and f32. It requires at least one topology ndarray and
rejects SNode-dependent dispatches. Topology, numeric data, and workspace are
copied into operator-owned resources. CPU lowers the Graph to an explicit
sequence; CUDA and Vulkan use the compiled-Graph execution contract. Backend
capture/replay may use an ordinary Graph fallback when the documented Graph
runtime rules require it, without changing the mathematical provider.

## Mathematical traits

`OperatorTraits` uses `None` for unknown and `bool` for an explicit caller
claim:

```python
traits = ti.linalg.experimental.OperatorTraits(
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
y = operator @ x

B = 2.0 * operator
C = operator + B
D = operator.compose(B)       # operator(B(x))
E = operator.adjoint()
F = ti.linalg.experimental.block_diagonal((operator, B))
I = ti.linalg.experimental.identity(size, dtype=ti.f32)
```

Input/output aliasing is rejected. `adjoint()` is available only when the
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
- `method="pcg"`: ordinary PCG with a fixed-linear `"jacobi"` or
  `"block_jacobi"` preconditioner on fixed stored matrices; and
- `method="bicgstab"`: identity-preconditioned CPU BiCGSTAB for general square
  systems.

```python
result = plan.solve(rhs, initial_guess=x0, out=x)
print(result.iterations, result.residual_norm, result.termination_reason)
stats = plan.statistics()
```

`rhs`, `initial_guess`, and `out` must match the operator dtype and scalar
extent and belong to the current runtime. If `out` is omitted, the plan creates
a result ndarray. If `initial_guess` is omitted, the result is initialized to
zero. RHS/output aliasing is rejected.

`SolveResult` contains the solution and a terminal snapshot: status code,
termination reason, convergence/breakdown/max-iteration flags, iteration
count, initial and final residual norms, both tolerances, relative reference
norm, and effective tolerance. CG, PCG, and BiCGSTAB use:

The result record is frozen; its `solution` ndarray remains caller-writable.

```text
||b - A x||_2 <= max(atol, rtol * ||b||_2)
```

Vulkan currently uses bounded masked convergence and accepts `rtol=0` only.
`statistics()` exposes backend-neutral plan/provider/workspace counters; it is
diagnostic data, not part of the numerical result.

## Support matrix

### Providers and apply

| Provider | CPU | CUDA | Vulkan |
| --- | --- | --- | --- |
| Fixed stored CSR/BSR | `f32`, `f64` | `f32` | `f32` |
| Compiled kernel | `f32` | `f32` | `f32` |
| Compiled Graph | `f32` | `f32` | `f32` |
| Identity/composition | `f32`, `f64` | Unsupported | Unsupported |

### Solvers

| Method/provider | CPU | CUDA | Vulkan |
| --- | --- | --- | --- |
| CG, fixed stored | CSR/BSR, `f32/f64` | CSR, `f32` | CSR/BSR, `f32`, absolute tolerance |
| CG, compiled kernel/Graph | `f32` | `f32` | `f32`, absolute tolerance |
| CG, CPU composition | `f32/f64` | Unsupported | Unsupported |
| PCG + Jacobi | Fixed CSR, `f32/f64` | Fixed CSR, `f32` | Fixed CSR, `f32`, absolute tolerance |
| PCG + block-Jacobi | Fixed BSR, `f32/f64` | Fixed BSR, `f32` | Fixed BSR, `f32`, absolute tolerance |
| BiCGSTAB | Any supported CPU operator, `f32/f64` | Unsupported | Unsupported |

Compiled-kernel inverse preconditioners remain available through lower-level
provider APIs but are not accepted by this public `SolvePlan` contract.
MINRES and direct factorization remain stored-matrix APIs.

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
generation. A stored Jacobi/block-Jacobi PCG plan refreshes its numeric inverse
before the next solve while retaining its pattern and Krylov workspace.
Topology changes require constructing a new operator.

The public API has no borrowed-resource mode. Stored operators strongly retain
their matrix; compiled providers own copied topology/numeric/workspace
resources; compositions and solve plans strongly retain their operands.

## Relationship to the legacy matrix-free API

The field-based `ti.linalg.LinearOperator`, `MatrixFreeCG`, and
`MatrixFreeBICGSTAB` remain available with their existing behavior. They use
field-shaped vectors and a `(x, y)` kernel callback and are not implicitly
adapted because that ABI does not carry explicit topology, numeric-resource,
runtime-generation, or capability information.

Migration requires an explicit scalar-ndarray kernel or Graph provider, an
explicit vector extent, and mathematical traits. Existing applications may
retain the legacy route until those contracts are available; no removal
schedule is attached to this experimental API.

## Qualification boundary

The API is covered by backend correctness, lifecycle, trait, composition, and
solver regression tests. Application production qualification remains
workload-specific: validate operator semantics, conditioning, tolerances,
preconditioner suitability, failure handling, memory budgets, and backend
driver behavior on representative physical and non-physical systems.

See also [Sparse runtime and linear algebra](sparse_runtime_and_linear_algebra.en.md)
and [Choosing sparse operators and solvers](physics_sparse_solver_selection.en.md).
