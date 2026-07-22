# 稀疏 Runtime 与线性代数

> 本文介绍 Taichi Forge 0.5.x 的稀疏存储、装配、算子、求解器、后端支持与生命周期
> 规则。

[English version](sparse_runtime_and_linear_algebra.en.md)

## 设计范围

稀疏工作并不局限于 MPM，它覆盖物理引擎常用的两个不同层次：

1. **空间稀疏存储与装配**：pointer、bitmasked、dynamic、hash SNode，active-block
   遍历，以及 contact/adjacency 构造。
2. **代数稀疏算子与求解**：CSR/BSR pattern、SpMV、直接分解、CG、MINRES 和
   BiCGSTAB。

应当把两层分开。坐标仍在动态激活时，可变空间 SNode 很有用；活动坐标取得稳定
DOF 后，反复执行的线性求解迭代通常应消费紧凑 CSR/BSR 数组或 matrix-free
operator。每轮 Krylov 迭代重新做 SNode listgen、pointer chasing 或 hash probe，
往往同时损失内存局部性和可预测开销。

## 推荐的端到端流程

典型 implicit simulation、constraint solve 或 pressure projection 可按以下流程组织：

1. 在 dense field、sparse SNode 或应用自有数组中装配空间状态。
2. 条件允许时，先统计 active row、block 或 constraint 数量再分配。
3. 分配紧凑、稳定的 DOF 编号。
4. 发布一个通过验证的 CSR/BSR pattern 或 matrix-free operator generation。
5. 拓扑未变时只更新数值。
6. 根据算子的数学类别选择 solver，而不是只看 storage format。
7. 只有拓扑变化时才重建 pattern 和 symbolic analysis。

该流程同样适用于 pressure Poisson、mass-spring、implicit FEM、implicit MPM、
rigid-body constraint 和普通稀疏线性方程组。

## 构造路径

### 使用 `SparseMatrixBuilder` 做 triplet 装配

在 Taichi kernel 中逐项创建稀疏拓扑时使用 builder：

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

当前合同：

- `max_num_triplets` 是插入硬预算；超出时会报错，不会请求无界扩容。
- builder kernel 当前支持 `+=` 和 `-=` 插入。
- `build()` 生成 scalar compressed storage；传入非 CSR 的
  `_format` 会被拒绝。BSR 应使用 `SparsePattern.bsr()`。
- CPU 支持 `f32`、`f64`；CUDA 和 Vulkan builder 支持 `f32`。
- CPU、CUDA、Vulkan 都执行 bounded insertion 验证；device builder 发布完整 matrix
  generation，不把未验证的 partial compressed array 暴露给用户。
- 复用 builder 不等于共享 fixed pattern。拓扑跨 step 稳定时应优先使用
  `SparsePattern`。

### Fixed CSR

row offset 和 column index 不变时使用 `SparsePattern.csr()`：

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

# 保留 pattern，只按 compressed order 替换数值。
values.from_numpy(np.array([4, -2, -2, 4, -2, -2, 4], np.float32))
A.update_values(values)
```

index 数组必须是当前 runtime 拥有的一维 scalar `ti.i32` ndarray；每行
column index 必须有序、唯一且在范围内。不直接接受 NumPy 数组；请显式复制，使
host/device transfer 保持可见。

从同一 pattern 创建的 matrix 共享 immutable index storage，但各自拥有独立 numeric
value buffer。`update_values()` 要求 stored scalar 数量与 compressed
order 不变，它不会改变拓扑。

### Fixed BSR

2D/3D nodal DOF 或 6-DOF rigid body 等天然小型 dense block 应使用
`SparsePattern.bsr()`：

```python
pattern = ti.linalg.SparsePattern.bsr(
    block_rows=number_of_nodes,
    block_cols=number_of_nodes,
    block_size=3,
    row_offsets=block_row_offsets,
    column_indices=block_column_indices,
)

# values 按 block-row-major 排列，block 内 scalar 为 row-major。
A = pattern.matrix(block_values)
A.update_values(next_block_values)
```

当前支持 2、3、6、12 的方形 block size。rectangular BSR operator 可执行 SpMV，
但 stored-matrix solver 要求 square operator。不要只为复用 BSR 而把 mixed KKT
field padding 成统一 block size。

## Matrix operation 支持面

`SparseMatrix` 会明确拒绝不支持的操作，不会静默转换 format，也不会把 GPU
matrix 复制到 host。

| Matrix/provider | values | ndarray SpMV | value update | element/algebra operation | stored solver |
| --- | --- | --- | --- | --- | --- |
| CPU mutable Eigen CSR/CSC | `f32`、`f64` | 支持 | provider-dependent mutable path | 完整 scalar operation set | direct、CG、MINRES、BiCGSTAB |
| CPU fixed CSR | `f32`、`f64` | 支持 | 支持 | read-only narrow operation set | CG、MINRES、BiCGSTAB |
| CPU fixed BSR | `f32`、`f64` | 支持 | 支持 | 不支持 element access 和 matrix-matrix algebra | CG、MINRES、BiCGSTAB |
| CUDA scalar CSR，含 fixed CSR | `f32` | 支持 | fixed pattern 支持 | 受支持 CUDA scalar subset | direct、CG |
| CUDA fixed BSR | `f32` | 支持 | 支持 | narrow BSR operation set | CG |
| Vulkan scalar/fixed CSR | `f32` | 支持 | fixed pattern 支持 | narrow Vulkan scalar subset | 无 |
| Vulkan fixed BSR | `f32` | 支持 | 支持 | narrow BSR operation set | 无 |

上述 fixed provider 都允许 `A @ x` 接受 scalar Taichi ndarray。NumPy 和
field SpMV 是 CPU Eigen 的便利路径，不能当作可移植 GPU input 合同。

vector 输入和返回值所有权同样取决于 provider。CPU stored iterative solver 接受
NumPy array 或当前 runtime 的 scalar Taichi ndarray；CUDA `SparseCG` 要求 scalar
Taichi ndarray，并返回 Taichi ndarray。direct solve 在 CPU 上接受 NumPy、field 或
ndarray RHS；CUDA 文档路径要求 Taichi ndarray。shape 与 dtype 必须与 matrix 精确
匹配，不执行隐式 host fallback。

## Solver 选择与用法

### 能力总表

| Solver | 要求的 operator 类别 | CPU | CUDA | Vulkan |
| --- | --- | --- | --- | --- |
| `SparseSolver` | 取决于 LLT/LDLT/LU | mutable Eigen CSR/CSC，`f32/f64` | scalar CSR，`f32`；受文档列出的 CUDA factorization 限制 | 不支持 |
| `SparseCG` | 对称正定 | mutable CSR/CSC 与 fixed CSR/BSR，`f32/f64` | scalar CSR 与 fixed BSR，`f32` | 不支持 |
| `SparseMINRES` | 显式完整对称，允许不定 | mutable CSR/CSC 与 fixed CSR/BSR，`f32/f64` | 不支持 | 不支持 |
| `SparseBiCGSTAB` | 显式非对称 square matrix | mutable CSR/CSC 与 fixed CSR/BSR，`f32/f64` | 不支持 | 不支持 |
| `MatrixFreeCG` | SPD 应用 operator | field/kernel 路径 | field/kernel 路径 | backend/dtype 支持该 operator 时可用 |
| `MatrixFreeBICGSTAB` | 非对称应用 operator | field/kernel 路径 | field/kernel 路径 | backend/dtype 支持该 operator 时可用 |
| `experimental.SolvePlan(method="cg")` | trait-qualified SPD stored/kernel/Graph operator | fixed CSR/BSR 与 composition，`f32/f64`；compiled provider 为 `f32` | fixed CSR 与 compiled provider，`f32` | fixed CSR/BSR 与 compiled provider，`f32` |
| `experimental.SolvePlan(method="pcg")` | trait-qualified SPD operator 与 preconditioner | CSR Jacobi、BSR block-Jacobi 或 fixed-linear operator，`f32/f64` | CSR/BSR 内置项或 compiled-kernel A/M，`f32` | CSR/BSR 内置项或 compiled-kernel A/M，`f32` |
| `experimental.SolvePlan(method="minres")` | trait-qualified self-adjoint、使用时 nonsingular 的 operator；若有 preconditioner 则必须 SPD | identity、任意兼容 provider，`f32/f64` | fixed CSR/BSR 或 compiled provider，支持 identity、内置项或兼容 fixed-linear preconditioning，`f32` | fixed CSR/BSR 或 compiled provider，支持 identity、内置项或兼容 fixed-linear preconditioning，`f32` |
| `experimental.SolvePlan(method="bicgstab")` | 一般 square operator | 任意受支持 experimental CPU provider，`f32/f64` | 不支持 | 不支持 |

Taichi 不会从 CSR/BSR shape 推断 symmetry、definiteness、nullspace 或
conditioning，这些数学合同由调用方负责。

### 绑定 runtime 的 LinearOperator

`ti.linalg.experimental.LinearOperator` 使用统一 capability 与 lifecycle 合同覆盖 fixed
stored CSR/BSR、compiled-kernel 和 compiled-Graph apply。vector 使用一维 scalar Taichi
ndarray，operator 保留可复用 native execution plan。数学性质通过 `OperatorTraits`
附加；SPD 性质未知时，CG/PCG 会拒绝构造。CPU 提供最小的
scale/sum/composition/adjoint/block-diagonal 代数；不受支持的 GPU composition 明确失败，
不执行 host fallback。

`experimental.SolvePlan` 跨 RHS 调用保留 solver workspace，并返回同时包含 solution 与
完整 terminal state 的 `SolveResult`。CUDA/Vulkan 支持显式的 4 或 8 iteration
host-check chunk；Vulkan 还保留 fixed-budget masked execution 作为默认策略。两个 GPU
backend 使用相同的 absolute/relative residual 合同。该 API 不替代 mutable Eigen sparse
matrix 或 direct factorization；它为 fixed 与 compiled operator 提供 provider-neutral
MINRES，而旧 `SparseMINRES` 构造器保留 CPU stored-matrix 合同。provider ABI、所有权、
update generation、示例和精确 backend 矩阵见
[实验性 LinearOperator 与 SolvePlan](linear_operator.zh.md)。

### CG 与 preconditioner

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
    raise RuntimeError("CG 未满足残差合同")
```

收敛条件为：

```text
||b - A x||_2 <= max(atol, rtol * ||b||_2)
```

两个 tolerance 都必须有限且非负，并且至少一个大于零。`rtol=0` 保留此前
只使用绝对阈值的行为。

preconditioner 行为：

- mutable CPU matrix 保留 legacy Eigen diagonal 默认值；
- mutable/fixed CUDA CSR 在 `preconditioner=None` 时保留 identity CG，
  并为 `f32` 支持显式 `"jacobi"`；
- fixed CPU CSR 使用 Jacobi-PCG；
- fixed CPU/CUDA BSR 使用 block-Jacobi PCG；
- 不支持的名称或 format/backend 组合会明确失败，不会退化成其它 solver。

`update_values()` 之后，fixed CSR/BSR CG 会在下次 solve 前刷新 numeric
preconditioner，同时保留 immutable pattern 和 solve workspace。CSR Jacobi 保存
diagonal reciprocal；BSR block-Jacobi 为 block size 2、3、6、12 保存 lower Cholesky
factor，并要求每个 diagonal block 都有限、对称且正定。CUDA 与 Vulkan 在 device 上完成
warm value-only refresh，保持 committed resource address 稳定，不经 host 传输完整 values
或 factor，也不分配 device memory。不合法的 block 会明确失败，不执行正则化或 fallback。

### 用于对称不定系统的 MINRES

显式存储的完整对称 KKT、saddle-point 或可能不定的 constraint matrix 应使用
`SparseMINRES`：

```python
solver = ti.linalg.SparseMINRES(A, rhs, max_iter=300, atol=1e-8, rtol=1e-5)
x, converged = solver.solve()
```

旧 `SparseMINRES` 路径仅支持 CPU 与 identity preconditioner。fixed 或 compiled
`LinearOperator` 应使用绑定 runtime 的 plan：

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

实验性路径支持 identity-preconditioned CPU `f32/f64` 与 CUDA/Vulkan `f32`。
CUDA/Vulkan fixed CSR/BSR 可选择文档列出的 Jacobi/block-Jacobi，也可用
`LinearOperator` 或 `PreconditionerPlan` 提供兼容的 device-native fixed-linear action；
preconditioner 必须 SPD。该路径不提供 MINRES-QLP 或 minimum-length 语义，因此拒绝声明为
singular 的 operator。两个对称 off-diagonal half 必须一致地存储。matrix 为 square 或
diagonal 为正，都不足以满足
MINRES 合同。

### 用于非对称系统的 BiCGSTAB

显式非对称系统应使用 `SparseBiCGSTAB`：

```python
solver = ti.linalg.SparseBiCGSTAB(
    A, rhs, x0=None, max_iter=300, atol=1e-8, rtol=1e-5
)
x, converged = solver.solve()
```

stored-matrix provider 仅支持 CPU。报告收敛前会检查最终真实残差。数值
breakdown 仍可能发生，也不能据此认定 matrix 属于对称不定；应在选择 solver 前完成
分类。

### 直接分解与 symbolic reuse

对于 direct solver 支持、拓扑固定且数值变化的 matrix，pattern 只分析一次，
每个新 numeric state 重新 factorize：

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

`compute(A)` 等价于 analyze 后 factorize。只有完整 compressed index pattern
相同，`factorize(B)` 才能接受另一个 matrix object；拓扑改变后必须再次
`analyze_pattern()`。factorization 后更新 values 会使 factorization stale，
在重新 factorize 前 `solve()` 会拒绝执行。

CPU 支持 LLT、LDLT、LU 和 AMD/COLAMD ordering。CUDA 路径是 scalar CSR
`f32`，并保留文档列出的 factorization 限制。Vulkan 没有 stored direct solver。

## Sparse SNode runtime 行为

Sparse SNode 遍历与分配遵循以下内存和执行规则：

- active-list index metadata 按需增长；
- traversal-list chunk 根据有界 workload estimate 自适应；
- ambient/inactive storage 与 active payload allocation 分离；
- traversal list 使用显式预算，CPU recycled payload 有上界；
- 正确保留 non-contiguous SNode slot；
- CPU sparse listgen 对大型 workload 使用确定性并行执行，并为稳定拓扑复用已生成 list；
- CUDA 合并重复 sparse activation 请求；
- Vulkan 对常驻 traversal list 设置上界，并随 Program 一起回收。

这些是实现优化，不代表 capacity planning 已经消失。pointer/dynamic/hash metadata、
allocator pool、listgen、旧/新 generation 并存、native plan、Graph cache 和 driver
allocation 仍然占用内存。中高 occupancy 下 dense 仍可能更合适。

SNode capacity 含义、overflow 行为和后端布局选择见
[稀疏布局选择指南](sparse_layout_selection.zh.md)。

## 生命周期、失败与所有权

- `SparsePattern`、`SparseMatrix`、builder、ndarray、solver 和
  preconditioner state 都属于一个 Taichi Program generation；`ti.reset()`
  会使它们失效。
- fixed pattern 持有 immutable indices，每个 matrix 单独持有 numeric values；共享
  pattern 不等于共享 values。
- value-only update 保留拓扑；count 或 order 改变时必须创建新 pattern。
- device builder overflow 与 SNode capacity overflow 都是明确错误。mutable SNode
  可能保留 overflow 前已经成功的 mutation；应 rebuild/clear，不能把失败更新当作
  transactional。
- unsupported backend/format 不会静默建立 Eigen shadow matrix，也不会把 GPU solve
  放到 host 执行。
- runtime Graph argument 应保持 generation-neutral；不要把 matrix、ndarray 或
  SNodeTree 的 native address 固化到长期应用状态。

offline-cache metadata 现在由进程持有的 OS advisory lock 保护。持久
`.lock` 文件是正常状态，不表示 cache 正忙；进程终止时操作系统自动释放
所有权。该修改不改变 compiled cache artifact 的独占创建语义。详见
[编译与缓存说明](cache_compile.zh.md#metadata-lock-生命周期)。

## 实际迁移清单

- 拓扑已知时，把每轮 sparse SNode 遍历改为 compact DOF 与 CSR/BSR。
- fixed topology 不再重复 triplet 构造，改用
  `SparsePattern.csr/bsr` 和 `update_values()`。
- 根据 operator 类别选择 CG、MINRES 或 BiCGSTAB。
- 使用 `rtol` 获得与 scale 相关的收敛条件，同时保留有意义的
  `atol` 下限。
- 只有完整 compressed pattern 相同时才复用 direct symbolic analysis。
- Vulkan 不提供旧 stored sparse solver class。provider-neutral `f32` MINRES 应使用
  `experimental.SolvePlan` 与受支持的 fixed 或 compiled operator。
- 分别测量 payload、metadata、list/workspace、重叠 generation 和 driver memory。
- `ti.reset()` 后重建所有 sparse runtime object。

## 相关文档

- [稀疏布局选择指南](sparse_layout_selection.zh.md)
- [物理稀疏算子与求解器选择指南](physics_sparse_solver_selection.zh.md)
- [稀疏矩阵与 fixed pattern](../lang/articles/math/sparse_matrix.md)
- [线性求解器](../lang/articles/math/linear_solver.md)
- [Vulkan sparse SNode](sparse_snode_on_vulkan.zh.md)
- [Hash SNode](hash_snode.zh.md)
- [编译与缓存说明](cache_compile.zh.md)
