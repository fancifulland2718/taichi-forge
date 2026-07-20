---
sidebar_position: 3
---

# Linear Solver

Solving linear equations is a common task in scientific computing. Taichi provides basic direct and iterative linear solvers for
various simulation scenarios. Currently, there are two categories of linear solvers available:
For the consolidated Forge backend/format/dtype matrix and update/reset
contracts, see the bilingual
[Sparse runtime and linear algebra guide](../../../forge/sparse_runtime_and_linear_algebra.en.md).
1. Solvers built for `SparseMatrix`, including:
- Direct solver `SparseSolver`
- Iterative (conjugate-gradient method) solver `SparseCG`
- Iterative (minimum-residual method) solver `SparseMINRES`
- Iterative (biconjugate-gradient stabilized method) solver `SparseBiCGSTAB`
2. Solvers built for `LinearOperator`
- Iterative (matrix-free conjugate-gradient method) solver `MatrixFreeCG`
- Iterative (matrix-free BiCGSTAB method) solver `MatrixFreeBICGSTAB`

It's important to understand that those solvers are built for specific matrix
properties and storage providers. `SparseCG` is for symmetric
positive-definite systems, while `SparseBiCGSTAB` is for nonsymmetric systems.
Below we explain the supported combinations and usage of each solver.

## Sparse linear solver
There are two types of linear solvers available for `SparseMatrix`, direct solver and iterative solver.

### Sparse direct solver
To solve a linear system whose coefficient matrix is a `SparseMatrix` using a direct method, follow the steps below:
1. Create a `solver` using `ti.linalg.SparseSolver(solver_type, ordering)`. Currently, the factorization types supported on CPU backends are `LLT`, `LDLT`, and `LU`, and supported orderings include `AMD` and `COLAMD`. The sparse solver on CUDA supports the `LLT` factorization type only.
2. Analyze and factorize the sparse matrix you want to solve using `solver.analyze_pattern(sparse_matrix)` and `solver.factorize(sparse_matrix)`
3. Call `x = solver.solve(b)`, where `x` is the solution and `b` is the right-hand side of the linear system. On CPU backends, `x` and `b` can be NumPy arrays, Taichi Ndarrays, or Taichi fields. On the CUDA backend, `x` and `b` *must* be Taichi Ndarrays.
4. Call `solver.info()` to check if the solving process succeeds.

Here's a full example.

```python
import taichi_forge as ti

arch = ti.cpu # or ti.cuda
ti.init(arch=arch)

n = 4

K = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100)
b = ti.ndarray(ti.f32, shape=n)

@ti.kernel
def fill(A: ti.types.sparse_matrix_builder(), b: ti.types.ndarray(), interval: ti.i32):
    for i in range(n):
        A[i, i] += 2.0

        if i % interval == 0:
            b[i] += 1.0

fill(K, b, 3)

A = K.build()
print(">>>> Matrix A:")
print(A)
print(">>>> Vector b:")
print(b)
# outputs:
# >>>> Matrix A:
# [2, 0, 0, 0]
# [0, 2, 0, 0]
# [0, 0, 2, 0]
# [0, 0, 0, 2]
# >>>> Vector b:
# [1. 0. 0. 1.]
solver = ti.linalg.SparseSolver(solver_type="LLT")
solver.analyze_pattern(A)
solver.factorize(A)
x = solver.solve(b)
success = solver.info()
print(">>>> Solve sparse linear systems Ax = b with the solution x:")
print(x)
print(f">>>> Computation succeed: {success}")
# outputs:
# >>>> Solve sparse linear systems Ax = b with the solution x:
# [0.5 0.  0.  0.5]
# >>>> Computation was successful?: True
```

Please have a look at our two demos for more information:
+ [Stable fluid](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/stable_fluid.py): A 2D fluid simulation using a sparse Laplacian matrix to solve Poisson's pressure equation.
+ [Implicit mass spring](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/implicit_mass_spring.py): A 2D cloth simulation demo using sparse matrices to solve the linear systems.

### Sparse iterative solver
To solve a linear system whose coefficient matrix is a `SparseMatrix` using an iterative (conjugate-gradient) method, follow the steps below:
1. Create a `solver` using `ti.linalg.SparseCG(A, b, x0, max_iter, atol, preconditioner=None, rtol=0.0)`, where `A` is a `SparseMatrix` that stores the coefficient matrix of the linear system, `b` is the right-hand side of the equations, `x0` is the initial guess, and `atol`/`rtol` control convergence. Pass `preconditioner="jacobi"` to explicitly select Jacobi-preconditioned CG on CPU or CUDA.
2. Call `x, exit_code = solver.solve()` to obtain the solution `x` along with the `exit_code` that indicates the status of the solution. `exit_code` should be `True` if the solving was successful. Here is an example:

For compatibility, `preconditioner=None` preserves the historical backend
default: CPU uses Eigen's diagonal preconditioner, while CUDA uses identity
CG. Explicit `"jacobi"` provides a backend-independent choice across the
currently supported CPU and CUDA `SparseCG` paths (CPU `f32`/`f64`, CUDA
`f32`). On fixed-pattern CUDA matrices, a value-only update refreshes the
Jacobi inverse before the next solve while retaining the CG workspace. Other
preconditioner names are rejected instead of silently falling back.

Convergence uses the unpreconditioned residual contract
`||b - A x||_2 <= max(atol, rtol * ||b||_2)`. Both tolerances must be finite
and non-negative, and at least one must be positive. The default `rtol=0`
preserves the historical absolute-only behavior. The right-hand-side norm and
effective threshold are recomputed for every solve, so one solver can safely
consume differently scaled right-hand sides. CUDA performs one additional
scalar reduction per solve only when `rtol` is positive.

`SparseCG` assumes that the coefficient matrix and the selected
preconditioner satisfy the symmetry and positive-definiteness requirements of
conjugate gradients. Taichi does not infer or numerically verify these
properties from a square CSR/BSR representation. Use a solver intended for
symmetric-indefinite or nonsymmetric systems instead of relying on a late CG
breakdown.

```python
import taichi_forge as ti

ti.init(arch=ti.cpu)

n = 4

K = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100)
b = ti.ndarray(ti.f32, shape=n)

@ti.kernel
def fill(A: ti.types.sparse_matrix_builder(), b: ti.types.ndarray(), interval: ti.i32):
    for i in range(n):
        A[i, i] += 2.0
        if i % interval == 0:
            b[i] += 1.0

fill(K, b, 3)

A = K.build()
print(">>>> Matrix A:")
print(A)
print(">>>> Vector b:")
print(b.to_numpy())
# outputs:
# >>>> Matrix A:
# [2, 0, 0, 0]
# [0, 2, 0, 0]
# [0, 0, 2, 0]
# [0, 0, 0, 2]
# >>>> Vector b:
# [1. 0. 0. 1.]
solver = ti.linalg.SparseCG(A, b, preconditioner="jacobi")
x, exit_code = solver.solve()
print(">>>> Solve sparse linear systems Ax = b with the solution x:")
print(x)
print(f">>>> Computation was successful?: {exit_code}")
# outputs:
# >>>> Solve sparse linear systems Ax = b with the solution x:
# [0.5 0.  0.  0.5]
# >>>> Computation was successful?: True
```
Note that the building process of `SparseMatrix` `A` is exactly the same as in the case of `SparseSolver`, the only difference here is that we created a `solver` whose type is `SparseCG` instead of `SparseSolver`.

### Symmetric-indefinite sparse iterative solver

Use `ti.linalg.SparseMINRES(A, b, x0=None, max_iter=50, atol=1e-6,
rtol=0.0)` for a complete explicitly stored symmetric square system that may
be indefinite, such as a KKT or constraint system. Taichi does not infer or
numerically verify symmetry from the storage format; both triangular halves
must represent the same operator. A matrix that is merely square is not enough
to satisfy the MINRES contract.

The implementation is a Forge-owned identity-preconditioned recurrence for
mutable Eigen CSR/CSC and caller-owned fixed-pattern CSR/BSR matrices with
`f32` or `f64` values on CPU. Fixed operators execute their native raw SpMV;
they do not copy pattern/values into an Eigen shadow. The recurrence checks
initial convergence before normalization, counts completed iterations, and
handles a closed Krylov space by verifying the true residual before reporting
convergence or breakdown. Both the solution and the unpreconditioned residual
must be finite. Convergence uses
`||b - A x||_2 <= max(atol, rtol * ||b||_2)`.

Fixed CSR/BSR patterns must be square, shared immutable patterns created by
the caller. Value-only updates reuse the same MINRES workspace. CUDA/Vulkan
matrices are rejected without running the solve on the host. Jacobi is
deliberately not a default MINRES preconditioner: an indefinite matrix can
have zero or negative diagonal entries, while a MINRES preconditioner must be
symmetric positive definite and compatible with the operator.

```python
import numpy as np
import taichi_forge as ti

ti.init(arch=ti.cpu)

# KKT matrix [H, J^T; J, 0]. Store both symmetric off-diagonal entries.
builder = ti.linalg.SparseMatrixBuilder(
    3, 3, max_num_triplets=6, storage_format="row_major"
)

@ti.kernel
def fill(A: ti.types.sparse_matrix_builder()):
    A[0, 0] += 2.0
    A[1, 1] += 3.0
    A[0, 2] += 1.0
    A[2, 0] += 1.0
    A[1, 2] += -1.0
    A[2, 1] += -1.0

fill(builder)
A = builder.build()
b = np.array([1.0, 2.0, 0.0], dtype=np.float32)
solver = ti.linalg.SparseMINRES(A, b, rtol=1e-5)
x, converged = solver.solve()
```

### Nonsymmetric sparse iterative solver

Use `ti.linalg.SparseBiCGSTAB(A, b, x0=None, max_iter=50, atol=1e-6,
rtol=0.0)` for an explicit nonsymmetric square system. Its convergence
contract is the same unpreconditioned true residual used by `SparseCG`:
`||b - A x||_2 <= max(atol, rtol * ||b||_2)`. The final residual is
recomputed from the returned solution; a non-finite residual or provider
numerical failure is reported as breakdown rather than success.

Mutable Eigen CSR/CSC matrices with `f32` or `f64` values on CPU use the Eigen
Jacobi-preconditioned provider. Caller-owned square fixed-pattern CPU CSR/BSR
matrices use a Forge-owned identity-preconditioned recurrence and execute the
operator's native raw SpMV, without pattern/value copies, an Eigen shadow, or
scalar expansion of BSR blocks. The fixed recurrence explicitly checks the
`rho`, `alpha`, and `omega` breakdown denominators and verifies solution
finiteness and the true residual. CUDA/Vulkan matrices are rejected without
copying the solve to the host or silently falling back to CG.

Reusing one mutable Eigen solver with a new right-hand side or initial guess
retains its provider state. A mutable matrix pattern or numeric-value update
rebuilds that state before the next solve. Fixed CSR/BSR value updates reuse
the same eight-vector recurrence workspace.

```python
import numpy as np
import taichi_forge as ti

ti.init(arch=ti.cpu)

n = 3
builder = ti.linalg.SparseMatrixBuilder(
    n, n, max_num_triplets=7, storage_format="row_major"
)

@ti.kernel
def fill(A: ti.types.sparse_matrix_builder()):
    A[0, 0] += 4.0
    A[0, 1] += -1.0
    A[1, 0] += 2.0
    A[1, 1] += 5.0
    A[1, 2] += 1.0
    A[2, 1] += -2.0
    A[2, 2] += 6.0

fill(builder)
A = builder.build()
b = np.array([2.0, 9.0, 14.0], dtype=np.float32)
solver = ti.linalg.SparseBiCGSTAB(A, b, rtol=1e-5)
x, converged = solver.solve()
```

BiCGSTAB does not prove that a system is well conditioned and can still break
down for difficult inputs. Symmetric-indefinite KKT/contact systems should use
`SparseMINRES`; they should not be routed to CG merely because their storage
is square or block sparse.

## Matrix-free iterative solver
Apart from `SparseMatrix` as an efficient representation of matrices, Taichi also support the `LinearOperator` type, which is a matrix-free representation of matrices.
Keep in mind that matrices can be seen as a linear transformation from an input vector to a output vector, it is possible to encapsulate the information of a matrice as a `LinearOperator`.

To create a `LinearOperator` in Taichi, we first need to define a kernel that represent the linear transformation:
```python
import taichi_forge as ti
from taichi_forge.linalg import LinearOperator

ti.init(arch=ti.cpu)

@ti.kernel
def compute_matrix_vector(v:ti.template(), mv:ti.template()):
    for i in v:
        mv[i] = 2 * v[i]
```
In this case, `compute_matrix_vector` kernel accepts an input vector `v` and calculates the corresponding matrix-vector product `mv`. It is mathematically equal to a matrice whose diagonal elements are all 2. In the case of `n=4`, the equivalent matrice `A` is:
```python cont
# >>>> Matrix A:
# [2, 0, 0, 0]
# [0, 2, 0, 0]
# [0, 0, 2, 0]
# [0, 0, 0, 2]
```
Then we can create the `LinearOperator` as follows:
```python cont
A = LinearOperator(compute_matrix_vector)
```
To solve a system of linear equations represented by this `LinearOperator`, we can use the built-in matrix-free solver `MatrixFreeCG`. Here is a full example:

`MatrixFreeCG` requires a symmetric positive-definite operator.
`MatrixFreeBICGSTAB` is available for nonsymmetric operators. In both
functions, `tol` is a positive absolute threshold for
`||b - A x||_2`, `maxiter` must be a non-negative integer, and
`maxiter=0` only checks the initial residual. The returned boolean reports
whether that residual contract was reached; the operator properties are still
the caller's responsibility.

```python
import taichi_forge as ti
import math
from taichi_forge.linalg import MatrixFreeCG, LinearOperator

ti.init(arch=ti.cpu)

GRID = 4
Ax = ti.field(dtype=ti.f32, shape=(GRID, GRID))
x = ti.field(dtype=ti.f32, shape=(GRID, GRID))
b = ti.field(dtype=ti.f32, shape=(GRID, GRID))

@ti.kernel
def init():
    for i, j in ti.ndrange(GRID, GRID):
        xl = i / (GRID - 1)
        yl = j / (GRID - 1)
        b[i, j] = ti.sin(2 * math.pi * xl) * ti.sin(2 * math.pi * yl)
        x[i, j] = 0.0

@ti.kernel
def compute_Ax(v: ti.template(), mv: ti.template()):
    for i, j in v:
        l = v[i - 1, j] if i - 1 >= 0 else 0.0
        r = v[i + 1, j] if i + 1 <= GRID - 1 else 0.0
        t = v[i, j + 1] if j + 1 <= GRID - 1 else 0.0
        b = v[i, j - 1] if j - 1 >= 0 else 0.0
        # Avoid ill-conditioned matrix A
        mv[i, j] = 20 * v[i, j] - l - r - t - b

A = LinearOperator(compute_Ax)
init()
MatrixFreeCG(A, b, x, maxiter=10 * GRID * GRID, tol=1e-18, quiet=True)
print(x.to_numpy())
```
