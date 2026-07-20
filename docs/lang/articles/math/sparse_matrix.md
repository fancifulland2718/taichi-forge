---
sidebar_position: 2
---

# Sparse Matrix

Sparse matrices are frequently involved in solving linear systems in science and engineering. Taichi provides sparse matrix APIs on the CPU, CUDA, and Vulkan backends, with backend- and format-specific capabilities described below.
For the consolidated Forge backend/format matrix and lifecycle rules, see the bilingual
[Sparse runtime and linear algebra guide](../../../forge/sparse_runtime_and_linear_algebra.en.md).

To assemble a scalar CSR matrix from triplets, follow these three steps:

1. Create a `builder` using `ti.linalg.SparseMatrixBuilder()`.
2. Call `ti.kernel` to fill the `builder` with your matrices' data.
3. Build sparse matrices from the `builder`.

:::caution WARNING
The sparse matrix feature is still under development. There are some limitations:
- The sparse matrix data type on the CPU backend only supports `f32` and `f64`.
- The sparse matrix data type on the CUDA and Vulkan backends only supports `f32`.

:::
Here's an example:
```python
import taichi_forge as ti
arch = ti.cpu # or ti.cuda
ti.init(arch=arch)

n = 4
# step 1: create sparse matrix builder
K = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100)

@ti.kernel
def fill(A: ti.types.sparse_matrix_builder()):
    for i in range(n):
        A[i, i] += 1  # Only +=  and -= operators are supported for now.

# step 2: fill the builder with data.
fill(K)

print(">>>> K.print_triplets()")
K.print_triplets()
# outputs:
# >>>> K.print_triplets()
# n=4, m=4, num_triplets=4 (max=100)(0, 0) val=1.0(1, 1) val=1.0(2, 2) val=1.0(3, 3) val=1.0

# step 3: create a sparse matrix from the builder.
A = K.build()
print(">>>> A = K.build()")
print(A)
# outputs:
# >>>> A = K.build()
# [1, 0, 0, 0]
# [0, 1, 0, 0]
# [0, 0, 1, 0]
# [0, 0, 0, 1]
```

## Fixed CSR patterns

For repeated scalar systems whose sparsity topology stays fixed, separate the
immutable CSR pattern from numeric values. Row offsets and column indices are
shared by every matrix created from the pattern; each matrix owns independent
values in row-major compressed order.

~~~python
import numpy as np
import taichi_forge as ti

ti.init(arch=ti.cpu)

row_offsets = ti.ndarray(dtype=ti.i32, shape=4)
column_indices = ti.ndarray(dtype=ti.i32, shape=7)
row_offsets.from_numpy(np.array([0, 2, 5, 7], dtype=np.int32))
column_indices.from_numpy(
    np.array([0, 1, 0, 1, 2, 1, 2], dtype=np.int32)
)

pattern = ti.linalg.SparsePattern.csr(
    rows=3,
    cols=3,
    row_offsets=row_offsets,
    column_indices=column_indices,
)

values = ti.ndarray(dtype=ti.f32, shape=7)
values.from_numpy(
    np.array([2, -1, -1, 2, -1, -1, 2], dtype=np.float32)
)
A = pattern.matrix(values)

x = ti.ndarray(dtype=ti.f32, shape=3)
x.from_numpy(np.array([1, 2, 3], dtype=np.float32))
y = A @ x

values.from_numpy(
    np.array([4, -2, -2, 4, -2, -2, 4], dtype=np.float32)
)
A.update_values(values)
~~~

The fixed-CSR contract is deliberately explicit:

- indices must be current-runtime, one-dimensional scalar ti.i32 ndarrays;
- the non-empty pattern must have sorted unique columns in every row;
- CPU values may be f32 or f64; CUDA and Vulkan values are currently f32;
- all backends support scalar Taichi ndarray SpMV and same-count value updates;
- fixed CPU CSR is a read-only native operator, not a mutable Eigen matrix;
  public direct solvers remain unavailable, while `SparseCG` uses the native
  f32/f64 Jacobi-PCG path and reuses its four-vector workspace across solves;
- `SparseCG` refreshes only the Jacobi numeric state after `update_values()`;
  the immutable pattern and solve workspace remain shared/reused. `None` and
  `preconditioner="jacobi"` both select this provider without an Eigen shadow
  matrix;
- CUDA fixed CSR retains the existing CUDA direct, CG, and Jacobi capabilities;
- Vulkan fixed CSR does not expose a public solver;
- SparseMatrixBuilder remains the triplet assembly path and does not
  automatically create a shared pattern.

NumPy inputs are not accepted by SparsePattern.csr() or matrix(). Copy them
explicitly into Taichi ndarrays so transfers remain visible.

## Fixed BSR patterns

For block-structured systems with a topology reused across steps or value
snapshots, create an immutable BSR pattern from scalar Taichi ndarrays. This
path shares row offsets and column indices across matrices while each matrix
owns its own values.

~~~python
import numpy as np
import taichi_forge as ti

ti.init(arch=ti.cpu)

row_offsets = ti.ndarray(dtype=ti.i32, shape=3)
column_indices = ti.ndarray(dtype=ti.i32, shape=2)
row_offsets.from_numpy(np.array([0, 1, 2], dtype=np.int32))
column_indices.from_numpy(np.array([0, 1], dtype=np.int32))

pattern = ti.linalg.SparsePattern.bsr(
    block_rows=2,
    block_cols=2,
    block_size=2,
    row_offsets=row_offsets,
    column_indices=column_indices,
)

values = ti.ndarray(dtype=ti.f32, shape=8)
values.from_numpy(np.tile(np.eye(2, dtype=np.float32).reshape(-1), 2))
A = pattern.matrix(values)

x = ti.ndarray(dtype=ti.f32, shape=4)
x.from_numpy(np.array([1, 2, 3, 4], dtype=np.float32))
y = A @ x

values.from_numpy(
    2 * np.tile(np.eye(2, dtype=np.float32).reshape(-1), 2)
)
A.update_values(values)

rhs = ti.ndarray(dtype=ti.f32, shape=4)
rhs.from_numpy(np.array([2, 4, 6, 8], dtype=np.float32))
cg = ti.linalg.SparseCG(
    A, rhs, max_iter=32, atol=1e-6, preconditioner="block_jacobi"
)
solution, converged = cg.solve()
~~~

The fixed-BSR contract remains deliberately explicit:

- row offsets and column indices must be current-runtime scalar ti.i32 ndarrays;
- the pattern must be non-empty, canonical, and have sorted unique columns in each block row;
- supported block sizes are 2, 3, 6, and 12;
- CPU values may be f32 or f64; CUDA and Vulkan values are currently f32;
- fixed BSR supports scalar Taichi ndarray SpMV and same-count value updates;
- square fixed-pattern CPU BSR exposes f32/f64 `SparseCG`, and square
  fixed-pattern CUDA BSR exposes f32 `SparseCG`, through the native
  block-Jacobi PCG provider. `None` and
  `preconditioner="block_jacobi"` select it without creating a scalar CSR or
  Eigen shadow matrix;
- `SparseCG` refreshes only the block inverse after `update_values()` and
  reuses the immutable pattern and four-vector solve workspace;
- CUDA BSR PCG currently uses host scalar reductions to drive adaptive
  convergence. It is not a device-resident Graph loop;
- rectangular BSR and Vulkan BSR do not expose a public solver yet;
- element access, matrix-matrix algebra, and public direct solvers remain
  unavailable for BSR;
- SparseMatrixBuilder remains a scalar CSR triplet builder and does not accept BSR.

NumPy inputs are not accepted by SparsePattern.bsr() or matrix() in this
release. Copy explicitly into a Taichi ndarray so host/device transfers remain
visible and controllable.

The basic operations like `+`, `-`, `*`, `@` and transpose are available for the scalar CSR/CSC formats on their supported backends. Fixed BSR uses the narrower operation set above.

```python cont
print(">>>> Summation: C = A + A")
C = A + A
print(C)
# outputs:
# >>>> Summation: C = A + A
# [2, 0, 0, 0]
# [0, 2, 0, 0]
# [0, 0, 2, 0]
# [0, 0, 0, 2]

print(">>>> Subtraction: D = A - A")
D = A - A
print(D)
# outputs:
# >>>> Subtraction: D = A - A
# [0, 0, 0, 0]
# [0, 0, 0, 0]
# [0, 0, 0, 0]
# [0, 0, 0, 0]

print(">>>> Multiplication with a scalar on the right: E = A * 3.0")
E = A * 3.0
print(E)
# outputs:
# >>>> Multiplication with a scalar on the right: E = A * 3.0
# [3, 0, 0, 0]
# [0, 3, 0, 0]
# [0, 0, 3, 0]
# [0, 0, 0, 3]

print(">>>> Multiplication with a scalar on the left: E = 3.0 * A")
E = 3.0 * A
print(E)
# outputs:
# >>>> Multiplication with a scalar on the left: E = 3.0 * A
# [3, 0, 0, 0]
# [0, 3, 0, 0]
# [0, 0, 3, 0]
# [0, 0, 0, 3]

print(">>>> Transpose: F = A.transpose()")
F = A.transpose()
print(F)
# outputs:
# >>>> Transpose: F = A.transpose()
# [1, 0, 0, 0]
# [0, 1, 0, 0]
# [0, 0, 1, 0]
# [0, 0, 0, 1]

print(">>>> Matrix multiplication: G = E @ A")
G = E @ A
print(G)
# outputs:
# >>>> Matrix multiplication: G = E @ A
# [3, 0, 0, 0]
# [0, 3, 0, 0]
# [0, 0, 3, 0]
# [0, 0, 0, 3]

print(">>>> Element-wise multiplication: H = E * A")
H = E * A
print(H)
# outputs:
# >>>> Element-wise multiplication: H = E * A
# [3, 0, 0, 0]
# [0, 3, 0, 0]
# [0, 0, 3, 0]
# [0, 0, 0, 3]

print(f">>>> Element Access: A[0,0] = {A[0,0]}")
# outputs:
# >>>> Element Access: A[0,0] = 1.0
```
