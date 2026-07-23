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
| Bilateral constraints or symmetric KKT | Complete symmetric CSR/BSR or a compiled self-adjoint operator + `experimental.SolvePlan(method="minres")`; the legacy `SparseMINRES` class remains CPU-only |
| Frictional or other nonsymmetric linearization | Start with `experimental.SolvePlan(method="bicgstab")` for low storage, or restarted `method="gmres"` when an Arnoldi minimum-residual route is required; both consume supported fixed/compiled operators |

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
matrices can be symmetric but indefinite. Do not route them to CG because they
are square or block sparse. The legacy `SparseMINRES` class supports CPU
mutable CSR/CSC and caller-owned fixed CSR/BSR with identity preconditioning.

`experimental.SolvePlan(method="minres")` accepts a trusted self-adjoint
`LinearOperator`. CPU supports identity-preconditioned `f32/f64`; CUDA and
Vulkan support `f32` fixed CSR/BSR and compiled providers. The GPU route may
use identity, the documented fixed-CSR Jacobi or fixed-BSR SPD block-Jacobi,
or a compatible trusted fixed-linear preconditioner. Explicit matrices must
store both symmetric off-diagonal halves consistently.

This is a reusable linear-runtime primitive, not a complete constraint or
multiphysics solver. The application remains responsible for operator
classification, nullspace removal, scaling, constraint regularization,
nonlinear/active-set sequencing, and domain-specific preconditioner design.
The route rejects operators declared singular and does not provide
MINRES-QLP, minimum-length, field-split, or Schur-complement policy.

### Nonsymmetric

Frictional contact linearization, advection-like terms, and some coupled
systems are nonsymmetric. `experimental.SolvePlan(method="bicgstab")`
supports compatible CPU `f32/f64` host actions and CUDA/Vulkan `f32` fixed
or compiled providers. It accepts identity or a fixed-linear right
preconditioner and qualifies convergence with the original-system true
residual. The legacy `SparseBiCGSTAB` remains a CPU stored-matrix route;
`MatrixFreeBICGSTAB` remains the field-based route.

BiCGSTAB can break down and does not prove conditioning. Complementarity,
active-set logic, Newton iteration, and nonlinear contact remain outside the
linear runtime contract.

`experimental.SolvePlan(method="gmres")` is the restarted Arnoldi alternative.
It supports compatible CPU `f32/f64` host actions and CUDA/Vulkan `f32` fixed
or compiled providers, with identity or fixed-linear right preconditioning.
Restart 8, 16, and 32 are available; the plan preallocates its complete basis
and reports basis/workspace bytes and logical versus executed work. It checks
the original-system true residual at restart boundaries. This support is a
general linear-algebra primitive: it does not provide contact policy, an
active-set/Newton loop, FGMRES, or a domain preconditioner.

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
| `experimental.SolvePlan(method="minres")` | Compatible operator, identity, `f32/f64` | Fixed CSR/BSR or compiled operator, identity/built-in/compatible fixed-linear preconditioner, `f32` | Fixed CSR/BSR or compiled operator, identity/built-in/compatible fixed-linear preconditioner, `f32` |
| `SparseBiCGSTAB` | Mutable and fixed CSR/BSR capabilities | Unsupported | Unsupported |
| `experimental.SolvePlan(method="bicgstab")` | Compatible host action, identity/fixed-linear right preconditioner, `f32/f64` | Fixed CSR/BSR or compiled A/M, `f32` | Fixed CSR/BSR or compiled A/M, `f32` |
| `experimental.SolvePlan(method="gmres")` | Compatible host action, identity/fixed-linear right preconditioner, `f32/f64` | Fixed CSR/BSR or compiled A/M, restart 8/16/32, `f32` | Fixed CSR/BSR or compiled A/M, restart 8/16/32, `f32` |
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
A qualified nonsingular symmetric bilateral KKT can use the experimental
MINRES plan on a supported CPU/CUDA/Vulkan provider. The legacy stored-matrix
class remains CPU-only. Frictional or otherwise nonsymmetric systems can use
the experimental BiCGSTAB or restarted GMRES plan on a supported fixed or
compiled provider. BiCGSTAB remains the lower-storage method with possible
numerical breakdown; GMRES trades a preallocated basis and restart-cycle work
for an Arnoldi minimum-residual route. Neither route provides complementarity
or active-set handling.

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
