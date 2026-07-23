# Sparse Runtime and Linear Algebra

> This guide describes sparse storage, assembly, operators, solvers, backend
> support, and lifecycle rules in Taichi Forge 0.5.x.

[中文版本](sparse_runtime_and_linear_algebra.zh.md)

## Design scope

The sparse work is not limited to MPM. It covers two different layers used by
physics engines:

1. **Spatial sparse storage and assembly**: pointer, bitmasked, dynamic, or
   hash SNodes; active-block traversal; contact and adjacency construction.
2. **Algebraic sparse operators and solves**: CSR/BSR patterns, SpMV,
   direct factorization, CG, MINRES, and BiCGSTAB.

Keep those layers separate. A mutable spatial SNode is useful while coordinates
are activated. Once active coordinates have stable degrees of freedom, repeated
linear-solver iterations should normally consume compact CSR/BSR arrays or a
matrix-free operator. Re-running SNode list generation, pointer chasing, or
hash probing in every Krylov iteration often loses both memory locality and
predictable cost.

## Recommended end-to-end workflow

For a typical implicit simulation, constraint solve, or pressure projection:

1. Assemble spatial state in dense fields, sparse SNodes, or application-owned
   arrays.
2. Count active rows, blocks, or constraints before allocation when possible.
3. Assign compact, stable DOF indices.
4. Publish one validated CSR/BSR pattern or a matrix-free operator generation.
5. Update only numeric values while topology is unchanged.
6. Select a solver from the mathematical operator class, not only its storage
   format.
7. Rebuild the pattern and any symbolic analysis only after a topology change.

This arrangement applies to pressure Poisson systems, mass-spring models,
implicit FEM, implicit MPM, rigid-body constraints, and general sparse linear
systems.

## Construction paths

### Triplet assembly with `SparseMatrixBuilder`

Use a builder when the sparsity pattern is created incrementally in a Taichi
kernel:

```python
import taichi_forge as ti

ti.init(arch=ti.cpu)

n = 4
builder = ti.linalg.SparseMatrixBuilder(
    n, n, max_num_triplets=12, dtype=ti.f32,
    storage_format="row_major",
)

@ti.kernel
def assemble(A: ti.types.sparse_matrix_builder()):
    for i in range(n):
        A[i, i] += 2.0
        if i + 1 < n:
            A[i, i + 1] += -1.0
            A[i + 1, i] += -1.0

assemble(builder)
A = builder.build()
```

Current contract:

- `max_num_triplets` is a hard insertion budget. Exceeding it is an
  error, not a request to grow without bound.
- Builder kernels support `+=` and `-=` insertion.
- `build()` produces scalar compressed storage. Passing a non-CSR
  `_format` is rejected; use `SparsePattern.bsr()` for BSR.
- CPU accepts `f32` and `f64`. CUDA and Vulkan builders accept `f32`.
- CPU, CUDA, and Vulkan validate bounded insertion. Device builders publish a
  completed matrix generation instead of exposing an unvalidated partial
  compressed array.
- Reusing a builder is not the same as sharing a fixed pattern. If topology is
  stable across steps, prefer `SparsePattern`.

### Fixed CSR

Use `SparsePattern.csr()` when row offsets and column indices remain
unchanged:

```python
import numpy as np
import taichi_forge as ti

ti.init(arch=ti.cpu)

row_offsets = ti.ndarray(ti.i32, shape=4)
column_indices = ti.ndarray(ti.i32, shape=7)
row_offsets.from_numpy(np.array([0, 2, 5, 7], np.int32))
column_indices.from_numpy(np.array([0, 1, 0, 1, 2, 1, 2], np.int32))

pattern = ti.linalg.SparsePattern.csr(
    rows=3,
    cols=3,
    row_offsets=row_offsets,
    column_indices=column_indices,
)

values = ti.ndarray(ti.f32, shape=7)
values.from_numpy(np.array([2, -1, -1, 2, -1, -1, 2], np.float32))
A = pattern.matrix(values)

# Preserve the pattern and replace only values in compressed order.
values.from_numpy(np.array([4, -2, -2, 4, -2, -2, 4], np.float32))
A.update_values(values)
```

The index arrays must be one-dimensional scalar `ti.i32` ndarrays
owned by the current runtime. Each row must contain sorted, unique, in-range
column indices. NumPy arrays are not accepted directly; copy them explicitly so
host/device transfers remain visible.

Every matrix created from a pattern shares immutable index storage and owns an
independent numeric value buffer. `update_values()` requires the same
stored scalar count and compressed order. It does not change topology.

### Fixed BSR

Use `SparsePattern.bsr()` for natural small dense blocks such as
2D/3D nodal DOFs or 6-DOF rigid bodies:

```python
pattern = ti.linalg.SparsePattern.bsr(
    block_rows=number_of_nodes,
    block_cols=number_of_nodes,
    block_size=3,
    row_offsets=block_row_offsets,
    column_indices=block_column_indices,
)

# Values are block-row-major, with row-major scalars inside each block.
A = pattern.matrix(block_values)
A.update_values(next_block_values)
```

Supported square block sizes are 2, 3, 6, and 12. Rectangular BSR operators are
valid for SpMV, but stored-matrix solvers require a square operator.
Do not pad mixed KKT fields into one uniform block size merely to reuse BSR.

## Matrix-operation support

`SparseMatrix` rejects an unsupported operation explicitly; it does
not silently convert the matrix to another format or copy a GPU matrix to the
host.

| Matrix/provider | Values | ndarray SpMV | Value update | Element/algebra operations | Stored solver |
| --- | --- | --- | --- | --- | --- |
| CPU mutable Eigen CSR/CSC | `f32`, `f64` | Yes | Provider-dependent mutable path | Full scalar operation set | Direct, CG, MINRES, BiCGSTAB |
| CPU fixed CSR | `f32`, `f64` | Yes | Yes | Read-only narrow operation set | CG, MINRES, BiCGSTAB |
| CPU fixed BSR | `f32`, `f64` | Yes | Yes | No element access or matrix-matrix algebra | CG, MINRES, BiCGSTAB |
| CUDA scalar CSR, including fixed CSR | `f32` | Yes | Yes for fixed pattern | Supported CUDA scalar subset | Direct and CG |
| CUDA fixed BSR | `f32` | Yes | Yes | Narrow BSR operation set | CG |
| Vulkan scalar/fixed CSR | `f32` | Yes | Yes for fixed pattern | Narrow Vulkan scalar subset | None |
| Vulkan fixed BSR | `f32` | Yes | Yes | Narrow BSR operation set | None |

`A @ x` accepts a scalar Taichi ndarray on every fixed provider above.
NumPy and field SpMV are CPU Eigen conveniences and must not be treated as
portable GPU input contracts.

Vector input and result ownership also depends on the provider. CPU stored
iterative solvers accept NumPy arrays or current-runtime scalar Taichi ndarrays.
CUDA `SparseCG` requires scalar Taichi ndarrays and returns a Taichi ndarray.
For direct solves, CPU accepts NumPy, field, or ndarray right-hand sides;
CUDA's documented route requires Taichi ndarrays. Shapes and dtypes must match
the matrix exactly, and no implicit host fallback is performed.

## Solver selection and usage

### Capability summary

| Solver | Required operator class | CPU | CUDA | Vulkan |
| --- | --- | --- | --- | --- |
| `SparseSolver` | Depends on LLT/LDLT/LU | Mutable Eigen CSR/CSC, `f32/f64` | Scalar CSR, `f32`; documented CUDA factorization restrictions apply | Unsupported |
| `SparseCG` | Symmetric positive-definite | Mutable CSR/CSC and fixed CSR/BSR, `f32/f64` | Scalar CSR and fixed BSR, `f32` | Unsupported |
| `SparseMINRES` | Explicit complete symmetric, possibly indefinite | Mutable CSR/CSC and fixed CSR/BSR, `f32/f64` | Unsupported | Unsupported |
| `SparseBiCGSTAB` | Explicit nonsymmetric square matrix | Mutable CSR/CSC and fixed CSR/BSR, `f32/f64` | Unsupported | Unsupported |
| `MatrixFreeCG` | SPD application operator | Field/kernel route | Field/kernel route | Available where the backend and dtype support the operator |
| `MatrixFreeBICGSTAB` | Nonsymmetric application operator | Field/kernel route | Field/kernel route | Available where the backend and dtype support the operator |
| `experimental.SolvePlan(method="cg")` | Trait-qualified SPD stored/kernel/Graph operator | Fixed CSR/BSR and compositions, `f32/f64`; compiled providers `f32` | Fixed CSR and compiled providers, `f32` | Fixed CSR/BSR and compiled providers, `f32` |
| `experimental.SolvePlan(method="pcg")` | Trait-qualified SPD operator and preconditioner | CSR Jacobi, BSR block-Jacobi, or fixed-linear operator, `f32/f64` | CSR/BSR built-in or compiled-kernel A/M, `f32` | CSR/BSR built-in or compiled-kernel A/M, `f32` |
| `experimental.SolvePlan(method="minres")` | Trait-qualified self-adjoint, nonsingular-in-use operator; SPD preconditioner when present | Identity, any compatible provider, `f32/f64` | Fixed CSR/BSR or compiled provider, with identity, built-in, or compatible fixed-linear preconditioning, `f32` | Fixed CSR/BSR or compiled provider, with identity, built-in, or compatible fixed-linear preconditioning, `f32` |
| `experimental.SolvePlan(method="bicgstab")` | General square operator | Any supported experimental CPU provider, `f32/f64` | Unsupported | Unsupported |

Taichi does not infer symmetry, definiteness, nullspaces, or conditioning from
CSR/BSR shape. The caller owns those mathematical contracts.

### Runtime-bound LinearOperator

`ti.linalg.experimental.LinearOperator` unifies fixed stored CSR/BSR,
compiled-kernel, and compiled-Graph apply behind one capability and lifecycle
contract. It uses scalar one-dimensional Taichi ndarrays and retains a reusable
native execution plan. Mathematical properties are attached through
`OperatorTraits`; CG/PCG refuse unknown SPD properties. CPU provides minimal
scale/sum/composition/adjoint/block-diagonal algebra, while unsupported GPU
composition fails without a host fallback.

`experimental.SolvePlan` retains solver workspace across RHS calls and returns
a `SolveResult` containing the solution and complete terminal state. CUDA and
Vulkan support explicit 4- or 8-iteration host-check chunks; Vulkan also keeps
fixed-budget masked execution as its default. Both GPU backends use the same
absolute/relative residual contract. This API does not replace mutable Eigen
sparse matrices or direct factorization. It provides provider-neutral MINRES
for fixed and compiled operators, while the legacy `SparseMINRES` constructor
retains its CPU stored-matrix contract. See
[Experimental LinearOperator and SolvePlan](linear_operator.en.md) for provider
ABIs, ownership, update generations, examples, and the exact
backend matrix.

### CG and preconditioners

```python
cg = ti.linalg.SparseCG(
    A,
    rhs,
    x0=None,
    max_iter=200,
    atol=1e-8,
    rtol=1e-5,
    preconditioner="jacobi",
)
x, converged = cg.solve()
if not converged:
    raise RuntimeError("CG did not meet the residual contract")
```

Convergence means:

```text
||b - A x||_2 <= max(atol, rtol * ||b||_2)
```

Both tolerances must be finite and non-negative, and at least one must be
positive. `rtol=0` preserves the earlier absolute-only behavior.

Preconditioner behavior:

- mutable CPU matrices keep the legacy Eigen diagonal default;
- mutable/fixed CUDA CSR keeps identity CG when `preconditioner=None`
  and supports explicit `"jacobi"` for `f32`;
- fixed CPU CSR uses Jacobi-PCG;
- fixed CPU/CUDA BSR uses block-Jacobi PCG;
- unsupported names and format/backend combinations fail instead of degrading
  to another solver.

After `update_values()`, fixed CSR/BSR CG refreshes its numeric preconditioner
before the next solve while retaining the immutable pattern and solve
workspace. CSR Jacobi stores diagonal reciprocals. BSR block-Jacobi stores
lower Cholesky factors for block sizes 2, 3, 6, and 12 and requires every
diagonal block to be finite, symmetric, and positive definite. CUDA and Vulkan
perform warm value-only refreshes on the device with stable committed resource
addresses, no complete-values or factor transfer through the host, and no
device allocation. Invalid blocks fail without regularization or fallback.

### MINRES for symmetric-indefinite systems

Use `SparseMINRES` for an explicitly stored complete symmetric KKT,
saddle-point, or constraint matrix that may be indefinite:

```python
solver = ti.linalg.SparseMINRES(A, rhs, max_iter=300, atol=1e-8, rtol=1e-5)
x, converged = solver.solve()
```

This legacy `SparseMINRES` route is CPU-only and identity-preconditioned. For a
fixed or compiled `LinearOperator`, use the runtime-bound plan:

```python
operator = ti.linalg.experimental.LinearOperator.from_sparse_matrix(
    A,
    traits=ti.linalg.experimental.OperatorTraits(
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
)
result = plan.solve(rhs)
```

The experimental route supports identity-preconditioned CPU `f32/f64` and
CUDA/Vulkan `f32`. CUDA/Vulkan fixed CSR/BSR may use the documented
Jacobi/block-Jacobi options, and compatible device-native fixed-linear actions
may be supplied as a `LinearOperator` or `PreconditionerPlan`. A preconditioner
must be SPD. Operators declared singular are rejected because this route does
not provide MINRES-QLP or minimum-length semantics. Both symmetric
off-diagonal halves must be stored consistently; square shape or a positive
diagonal alone does not satisfy the operator contract.

### BiCGSTAB for nonsymmetric systems

The legacy `SparseBiCGSTAB` constructor serves explicit CPU systems:

```python
solver = ti.linalg.SparseBiCGSTAB(
    A, rhs, x0=None, max_iter=300, atol=1e-8, rtol=1e-5
)
x, converged = solver.solve()
```

For a runtime-bound fixed or compiled operator, use the provider-neutral plan:

```python
operator = ti.linalg.experimental.LinearOperator.from_sparse_matrix(A)
plan = ti.linalg.experimental.SolvePlan(
    operator,
    method="bicgstab",
    preconditioner="identity",
    max_iterations=300,
    atol=1e-8,
    rtol=1e-5,
)
result = plan.solve(rhs)
```

CPU supports compatible `f32/f64` host-action providers. CUDA and Vulkan
support `f32` fixed CSR/BSR and compiled providers. A fixed-linear
preconditioner is applied on the right and must not be declared singular; it
does not need the SPD traits required by PCG or MINRES. Fixed stored identity
plans can replay native iteration chunks, while compiled A/M actions use direct
submissions. The final true residual is checked before reporting convergence,
and `breakdown_reason` identifies rho/alpha/omega failures. Numerical
breakdown remains possible and is not evidence that the matrix is
symmetric-indefinite; classify such systems before selecting a solver.

### Direct factorization and symbolic reuse

For a matrix supported by the direct solver with fixed topology and changing values,
analyze once and factorize each new numeric state:

```python
solver = ti.linalg.SparseSolver(
    dtype=ti.f32, solver_type="LLT", ordering="AMD"
)

solver.analyze_pattern(A)
solver.factorize(A)
x0 = solver.solve(rhs0)

A.update_values(next_values)
solver.factorize(A)
x1 = solver.solve(rhs1)
```

`compute(A)` is equivalent to analysis followed by factorization.
`factorize(B)` may use another matrix object only if its complete
compressed index pattern matches the analyzed pattern. A topology change
requires another `analyze_pattern()`. Updating values after
factorization makes that factorization stale; `solve()` rejects it
until factorization is refreshed.

On CPU, supported factorization types are LLT, LDLT, and LU with AMD/COLAMD
ordering. The CUDA route is scalar CSR `f32` and retains its documented
factorization restrictions. Vulkan has no stored direct solver.

## Sparse SNode runtime behavior

Sparse SNode traversal and allocation use the following memory and execution
rules:

- active-list index metadata grows on demand;
- traversal-list chunks adapt to bounded workload estimates;
- ambient/inactive storage is separated from active payload allocation;
- traversal lists have explicit budgets and recycled CPU payload is bounded;
- non-contiguous SNode slots are preserved correctly;
- CPU sparse list generation uses deterministic parallel execution for large
  workloads and reuses generated lists for stable topologies;
- CUDA coalesces duplicate sparse activation requests;
- Vulkan bounds resident traversal lists and retires them with their Program.

These are implementation improvements, not removal of capacity planning.
Pointer/dynamic/hash metadata, allocator pools, list generation, old/new
generation overlap, native plans, Graph caches, and driver allocations still
consume memory. Dense storage can win at medium or high occupancy.

For SNode capacity meanings, overflow behavior, and backend-specific layout
selection, use [Choosing a sparse layout](sparse_layout_selection.en.md).

## Lifecycle, failure, and ownership rules

- `SparsePattern`, `SparseMatrix`, builders, ndarrays,
  solvers, and preconditioner state belong to one Taichi Program generation.
  `ti.reset()` invalidates them.
- Fixed patterns own immutable indices; each matrix owns its numeric values.
  Pattern sharing does not share values.
- Value-only updates preserve topology. A count or ordering change requires a
  new pattern.
- Device builder overflow and SNode capacity overflow are explicit errors.
  Mutable SNodes can contain successful mutations made before an overflow;
  rebuild or clear them instead of treating the failed update as transactional.
- Unsupported backend/format combinations do not silently create an Eigen
  shadow matrix or execute a GPU solve on the host.
- Keep runtime Graph arguments generation-neutral. Do not bake native addresses
  from a matrix, ndarray, or SNodeTree into long-lived application state.

Offline-cache metadata now uses a process-owned OS advisory lock. A persistent
`.lock` file is normal and does not mean the cache is busy; process
termination releases ownership automatically. This lock change does not alter
exclusive creation of compiled cache artifacts. See
[Compile and cache guide](cache_compile.en.md#metadata-lock-lifetime).

## Practical migration checklist

- Replace per-iteration sparse SNode traversal with compact DOFs and CSR/BSR
  when topology is already known.
- Replace repeated fixed-topology triplet construction with
  `SparsePattern.csr/bsr` and `update_values()`.
- Select CG, MINRES, or BiCGSTAB from the operator class.
- Add `rtol` for scale-aware convergence; keep a meaningful
  `atol` floor.
- Reuse direct symbolic analysis only while the complete compressed pattern is
  identical.
- Vulkan does not provide the legacy stored sparse solver classes. For
  provider-neutral `f32` MINRES or BiCGSTAB, use `experimental.SolvePlan`
  with a supported fixed or compiled operator.
- Measure payload, metadata, list/workspace, overlapping generations, and driver
  memory separately.
- Recreate all sparse runtime objects after `ti.reset()`.

## Related documentation

- [Choosing a sparse layout](sparse_layout_selection.en.md)
- [Choosing sparse operators and solvers](physics_sparse_solver_selection.en.md)
- [Sparse matrices and fixed patterns](../lang/articles/math/sparse_matrix.md)
- [Linear solvers](../lang/articles/math/linear_solver.md)
- [Vulkan sparse SNodes](sparse_snode_on_vulkan.en.md)
- [Hash SNode](hash_snode.en.md)
- [Compile and cache guide](cache_compile.en.md)
