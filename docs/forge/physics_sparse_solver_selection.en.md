# Choosing Sparse Operators and Solvers for Physics Workloads

> Applies to the Taichi Forge **0.5.x** release line.

[中文版本](physics_sparse_solver_selection.zh.md)

## Short answer

Choose in this order:

1. classify the linearized operator as SPD, symmetric-indefinite, or
   nonsymmetric;
2. decide whether its topology is fixed, value-only changing, or rebuilt;
3. choose a storage/operator and a backend provider that explicitly supports
   that solver category.

| Physics workload | Preferred starting point |
|---|---|
| Regular pressure Poisson | `LinearOperator` + `MatrixFreeCG`, or fixed CSR + `SparseCG` when an explicit matrix is required |
| Mass-spring or SPD implicit FEM | Fixed CSR/BSR + CG with Jacobi/block-Jacobi; use BSR for natural 2/3/6/12-DOF blocks |
| Implicit MPM with a changing active grid | Use SNodes for spatial assembly, then publish compact DOFs and an explicit or matrix-free operator before iteration |
| Per-step particle/contact adjacency | Count-scan-fill or sorted arrays for topology; this is assembly, not a solver choice |
| Bilateral constraints or symmetric KKT | Complete symmetric CSR/BSR + MINRES; stored-matrix support is CPU-only |
| Frictional or other nonsymmetric linearization | BiCGSTAB on a supported stored matrix, or `MatrixFreeBICGSTAB` for an application-owned operator |

There is no safe selector based only on matrix size, CSR/BSR format, or the word
"sparse". Taichi does not infer symmetry or positive definiteness from storage.

## Classify the operator first

### Symmetric positive-definite

Use CG/PCG only when the operator and preconditioner meet the SPD contract.
Typical candidates include a pressure Poisson problem with its nullspace and
boundary conditions handled, a stabilized mass-spring Hessian, or an implicit
elasticity system whose linearization is known to be SPD.

`SparseCG` uses the true residual condition
`||b - A x|| <= max(atol, rtol * ||b||)`. Scalar Jacobi is available on
supported CPU/CUDA paths; fixed BSR can select block-Jacobi where its format
capability reports support. A late CG breakdown is not a substitute for
classifying the operator.

### Symmetric-indefinite

Bilateral constraints, saddle-point systems, mixed formulations, and KKT
matrices can be symmetric but indefinite. Store both symmetric off-diagonal
halves and use `SparseMINRES`; do not route them to CG because they are square
or block sparse.

`SparseMINRES` supports CPU mutable CSR/CSC and caller-owned
fixed CSR/BSR providers. CUDA and Vulkan stored matrices are rejected without
a silent host solve. There is no matrix-free MINRES or general
field-split/Schur provider in this release.

### Nonsymmetric

Frictional contact linearization, advection-like terms, and some coupled
systems are nonsymmetric. `SparseBiCGSTAB` supports the documented CPU
stored-matrix providers. `MatrixFreeBICGSTAB` is available for an
application-supplied `LinearOperator` on supported Taichi backends.

BiCGSTAB can break down and does not prove conditioning. Complementarity,
active-set logic, Newton iteration, and nonlinear contact remain outside the
linear runtime contract.

## Match storage to topology lifetime

| Topology lifetime | Storage/operator choice | Update contract |
|---|---|---|
| Regular and implicit | Dense/compact field stencil or `LinearOperator` | Keep topology in kernel structure; update coefficient fields explicitly |
| Fixed compressed pattern | `SparsePattern.csr()` or `.bsr()` plus values | Reuse analysis/operator/workspace; publish value-only updates in compressed order |
| Rebuilt each step | Count-scan-fill, sort/RLE, then exact CSR/BSR arrays | Build a new generation and publish only after validation |
| Online unknown count | Bounded `dynamic` SNode only when a count pass is unavailable | Overflow is an error; mutable partial state is possible |
| Spatial coordinate grid | Pointer/bitmasked bricks during assembly | Assign stable compact DOFs before repeated Krylov iterations |

Do not traverse pointer/hash SNodes inside every solver iteration after active
coordinates already have stable DOFs. Conversely, do not use CSR as an online
spatial activation directory.

Fixed BSR supports block sizes 2, 3, 6, and 12. It is useful when each mesh
node or rigid body naturally owns a small dense block. Mixed KKT fields such as
`6 + 6 + 1/3` should not be padded into uniform 6-lane BSR merely to reuse a
solver interface.

## Supported provider matrix

| Route | CPU | CUDA | Vulkan |
|---|---|---|---|
| Explicit sparse SpMV | Supported formats | Supported formats | Supported formats |
| `SparseSolver` direct solve | CSR/CSC providers | Documented CSR provider | Unsupported |
| `SparseCG` | Mutable and fixed CSR/BSR capabilities | CSR and fixed BSR capabilities; dtype/format restrictions apply | Unsupported |
| `SparseMINRES` | Mutable and fixed CSR/BSR capabilities | Unsupported | Unsupported |
| `SparseBiCGSTAB` | Mutable and fixed CSR/BSR capabilities | Unsupported | Unsupported |
| `MatrixFreeCG` | Kernel/field route | Kernel/field route | Available where the backend/dtype is supported |
| `MatrixFreeBICGSTAB` | Kernel/field route | Kernel/field route | Available where the backend/dtype is supported |

Always check the format, dtype, shape, and provider-specific error. BSR
availability does not imply that direct solve, MINRES, or every dtype is
available on the same backend.

## Workload notes

### Pressure and regular grids

Prefer a matrix-free stencil when it avoids repeated index/value storage and
the boundary/nullspace contract is explicit. Use fixed CSR when an existing
stored solver, export, or irregular boundary representation justifies it.
Vulkan provides matrix-free CG but no stored `SparseCG`.

### FEM, mass-spring, and implicit MPM

Keep a fixed FEM or spring pattern across value updates and rebuild it only on
remeshing/topology change. Use BSR for natural block DOFs. For MPM, block-sparse
SNodes remain useful in the spatial stage, but the solve stage should consume a
compact operator generation. Whether CG is valid depends on the actual
linearization and contact/material treatment, not on MPM itself.

### Contact and constraints

Separate contact adjacency from the linear solve. If counts are available,
build row offsets and payload exactly instead of appending through `dynamic`.
A symmetric bilateral KKT can use CPU MINRES; frictional or otherwise
nonsymmetric systems need BiCGSTAB/GMRES-class treatment. Current CUDA/Vulkan
stored non-SPD solvers remain explicitly unsupported.

## Failure and lifecycle rules

- Provider absence is an explicit unsupported result, not permission to copy a
  GPU matrix to the host silently.
- Fixed-pattern value updates must preserve stored scalar count and compressed
  order; topology changes require a new pattern/generation.
- A solver/preconditioner binding becomes stale when a dependent numeric or
  topology generation changes. Rebuild or explicitly refresh according to the
  solver provider contract.
- Treat SNode overflow as mutable partial-state failure and exact ndarray
  publication failure as transactional only when that builder says so.
- `ti.reset()` invalidates runtime-owned matrices, plans, ndarrays, and
  generation objects. Do not retain native addresses across Programs.

## Avoid these mistakes

- Do not select CG from square shape, CSR/BSR format, or positive diagonal.
- Do not send symmetric-indefinite KKT systems to CG.
- Do not assume Vulkan sparse storage implies a Vulkan stored sparse solver.
- Do not rebuild fixed row/column indices for value-only updates.
- Do not put SNode list generation or hash probing inside every Krylov step.
- Do not tune block size, damping, tolerance, or crossover before the workload
  contract and overflow/lifecycle tests are fixed.

## Related documentation

- [Sparse runtime and linear algebra](sparse_runtime_and_linear_algebra.en.md)
- [Choosing a sparse layout](sparse_layout_selection.en.md)
- [Linear solvers](../lang/articles/math/linear_solver.md)
- [Sparse matrices and fixed patterns](../lang/articles/math/sparse_matrix.md)
- [Forge API reference](forge_api_reference.en.md)
- [Native algorithms](native_algorithms.en.md)
