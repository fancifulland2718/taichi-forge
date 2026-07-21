# 实验性 LinearOperator 与 SolvePlan

`ti.linalg.experimental` 提供绑定 runtime 的线性映射抽象，可使用 fixed sparse
matrix、已编译 Taichi kernel 或已编译 Graph 作为 provider。这是通用数值 API：应用可以
用它处理离散 PDE、隐式系统、图问题、优化和其它线性代数任务，无需把物理领域对象引入
Taichi DSL。

该命名空间目前属于实验性 API。本文定义 provider、生命周期、能力和失败合同；在提升到
稳定的 `ti.linalg` 命名空间之前，source compatibility 仍可能调整。

## 核心模型

一个 `LinearOperator` 表示 `y = A x`，包含：

- scalar dtype 和 `(rows, columns)` shape；
- 绑定当前 Taichi `Program` 的一个具体 provider；
- self-adjoint、positive definite 等显式数学 trait；
- 可观察的 capability 与 resource-generation metadata；
- 一个可复用的 native execution plan。

公开 vector 参数是一维 scalar Taichi ndarray。`apply()` 返回前会完成本次应用。它不接受
NumPy array，不经 host copy，不 materialize matrix-free provider；请求不受支持的操作时
也不会切换 backend。

operator、plan、provider 和 ndarray 都属于同一 runtime generation。执行 `ti.reset()`
后它们全部失效，也不能重新绑定到后续 `ti.init()` session。

## Stored operator 与 CG

fixed CSR/BSR matrix 可通过 `aslinearoperator()` 或
`LinearOperator.from_sparse_matrix()` 转换。operator 会强引用 matrix，并直接复用现有
pattern 与 numeric storage，不复制矩阵。

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

mutable Eigen CSR/CSC matrix 继续由已有的 `SparseCG`、`SparseMINRES`、
`SparseBiCGSTAB` 与 `SparseSolver` API 支持。实验性 stored provider 只接受 fixed
CSR/BSR，从而让 topology、numeric generation 和 provider ownership 具有统一合同。

## 已编译 kernel provider

`LinearOperator.from_kernel()` 接受精确的 f32 ndarray ABI。topology 与 numeric 分离时，
签名为：

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

未传入 `numeric=` 时，精确签名为 `(active_size, operator_data, x, y)`。所有 data
参数都是 scalar ndarray；topology 与 numeric 可以有各自的 scalar dtype，vector 必须是
f32。kernel 必须覆盖写入每个 output entry，且不能依赖 SNode tree。构造时编译一个
specialization，并把 topology/numeric 输入复制到 operator-owned snapshot。

## 已编译 Graph provider

`LinearOperator.from_graph()` 绑定已编译 Graph，其中动态 vector 参数必须命名为
`input` 与 `output`。其它每个 Graph 参数必须且只能分配一个角色：

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

该 provider 是 square f32 operator，至少需要一个 topology ndarray，并拒绝依赖 SNode
的 dispatch。topology、numeric data 与 workspace 会复制到 operator-owned resource。
CPU 把 Graph lowering 成 explicit sequence；CUDA/Vulkan 使用 compiled-Graph execution
合同。当已记录的 Graph runtime 规则要求时，backend capture/replay 可以使用 ordinary
Graph fallback，但不会改变数学 provider。

## 数学 trait

`OperatorTraits` 用 `None` 表示未知，用 `bool` 表示调用方的显式声明：

```python
traits = ti.linalg.experimental.OperatorTraits(
    self_adjoint=True,
    positive_definite=True,
    positive_semidefinite=True,
    singular=False,
)
```

`OperatorTraits.spd()` 是等价的便捷构造。CG/PCG 要求可信的
`self_adjoint=True` 与 `positive_definite=True`，并拒绝声明为 singular 的
operator。仅凭 shape、一次正 diagonal sample 或经验性乘积检查，不能建立这些性质。

trait 是 operator 后续使用的每个 numeric generation 都必须满足的合同。通过
`update_numeric()` 改变系数后，调用方必须确保已声明性质仍成立。结构上安全的组合会推导
能够证明的 trait；不能证明的性质保持 unknown。

## Apply 与组合

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

input/output alias 会被拒绝。只有 provider 提供显式 adjoint apply 时才能调用
`adjoint()`；实现不会把 self-adjoint trait 当作 fallback。

scale、sum、composition、block diagonal 和 identity 当前在 CPU 上执行，GPU lowering
不属于现有 API。GPU 组合会在构造时明确失败，不会执行 host code，也不会通过隐式 staging
路径进行同步。

## SolvePlan 与 SolveResult

`SolvePlan` 会跨多次调用保留 operator、solver state 和 persistent workspace。支持：

- `method="cg"`：面向 SPD 系统、使用 identity preconditioner 的 conjugate gradient；
- `method="pcg"`：fixed CSR 上使用 `"jacobi"`、fixed BSR 上使用
  `"block_jacobi"`，或使用一个可信 SPD `LinearOperator` 应用 fixed-linear
  近似逆的 PCG；
- `method="bicgstab"`：面向一般 square 系统、使用 identity preconditioner 的 CPU
  BiCGSTAB。

```python
result = plan.solve(rhs, initial_guess=x0, out=x)
print(result.iterations, result.residual_norm, result.termination_reason)
stats = plan.statistics()
```

fixed-linear preconditioner 作为 operator 传入，而不是应用回调：

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

`inverse_operator` 必须把 `operator` 的 range 映射回 domain，且 dtype 相同；它必须
携带可信的 self-adjoint、positive-definite 与 nonsingular trait。CPU 接受 operator
execution plan 支持的 provider 组合；CUDA/Vulkan 要求系统 operator 与 preconditioner
都是 compiled-kernel provider。每次 solve 会成对 pin 它们的 topology 与 numeric
generation，不调用 host callback，也不执行 backend fallback。

`rhs`、`initial_guess` 和 `out` 必须与 operator 的 dtype、scalar extent 一致，并属于当前
runtime。未提供 `out` 时会创建结果 ndarray；未提供 `initial_guess` 时结果初始化为零。
RHS/output alias 会被拒绝。

`SolveResult` 同时包含 solution 与 terminal snapshot：status code、termination reason、
convergence/breakdown/max-iteration flag、iteration count、初始与最终 residual norm、两个
tolerance、relative reference norm 和 effective tolerance。CG、PCG、BiCGSTAB 使用：

result record 本身是 frozen 的；其中的 `solution` ndarray 仍可由调用方写入。

```text
||b - A x||_2 <= max(atol, rtol * ||b||_2)
```

`statistics()` 返回 backend-neutral plan/provider/workspace counter；它是诊断信息，
不属于数值结果。

### GPU 执行策略

`execution_policy` 控制 GPU solve 在何时由 host 观察收敛状态：

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

- CPU 只支持 `"host_each_iteration"`。
- CUDA 为兼容性默认使用 `"host_each_iteration"`，还支持
  `"host_check_every_k"`，其中 `check_interval` 可为 4 或 8。分块策略把 recurrence
  scalar 保留在 device 上，每个 chunk 只读取一次 terminal snapshot。
- Vulkan 默认使用 `"fixed_budget_masked"`，还支持
  `"host_check_every_k"`，其中 `check_interval` 可为 4 或 8。两种策略均支持
  `atol`、`rtol` 及其组合后的 effective tolerance。

host 检查状态前，一个 chunk 总会完整执行。`SolveResult.iterations` 表示逻辑上的
convergence 或 breakdown iteration；`statistics()["operations"]` 另行报告
`executed_iterations`、`wasted_iterations`、host synchronization 次数和 direct
chunk submission。因此 chunked solve 最多可能多执行 `K - 1` 轮 masked 或其它 inactive
tail。Vulkan fixed-budget execution 可以执行完整的 `max_iterations`，同时保留更早发生
的逻辑终止结果。

当更早终止比同步频率更重要时可选 `K=4`；当摊薄 host check 更重要时可选 `K=8`。
较快选择取决于 vector size、operator 成本、iteration count、driver 与 backend。不受支持
的 policy 或 interval 会在 plan 构造时失败，不会静默 fallback。

## 支持矩阵

### Provider 与 apply

| Provider | CPU | CUDA | Vulkan |
| --- | --- | --- | --- |
| Fixed stored CSR/BSR | `f32`、`f64` | `f32` | `f32` |
| Compiled kernel | `f32` | `f32` | `f32` |
| Compiled Graph | `f32` | `f32` | `f32` |
| Identity/composition | `f32`、`f64` | 不支持 | 不支持 |

### Solver

| Method/provider | CPU | CUDA | Vulkan |
| --- | --- | --- | --- |
| CG，fixed stored | CSR/BSR，`f32/f64` | CSR，`f32` | CSR/BSR，`f32` |
| CG，compiled kernel/Graph | `f32` | `f32` | `f32` |
| CG，CPU composition | `f32/f64` | 不支持 | 不支持 |
| PCG + Jacobi | fixed CSR，`f32/f64` | fixed CSR，`f32` | fixed CSR，`f32` |
| PCG + block-Jacobi | fixed BSR，`f32/f64` | fixed BSR，`f32` | fixed BSR，`f32` |
| PCG + fixed-linear operator | 受支持 provider，`f32/f64` | compiled-kernel A/M，`f32` | compiled-kernel A/M，`f32` |
| BiCGSTAB | 任意受支持 CPU operator，`f32/f64` | 不支持 | 不支持 |

MINRES 和 direct factorization 继续使用 stored-matrix API。

## Numeric update 与所有权

stored operator 调用 `operator.update_numeric(values)`。compiled provider 使用乐观版本
检查：

```python
operator.update_numeric(
    next_values,
    expected_topology_version=1,
    expected_numeric_version=3,
)
```

Graph update 传入完整 numeric role mapping。compiled update 成功后会发布下一个 immutable
numeric generation。in-flight work 会继续保留其 pinned generation；后续 apply/solve
观察新 generation。stored Jacobi/block-Jacobi PCG plan 会在下一次 solve 前刷新 numeric
inverse，同时保留 pattern 与 Krylov workspace。topology 改变后必须构造新 operator。

公开 API 不提供 borrowed-resource 模式。stored operator 强引用 matrix；compiled
provider 拥有复制后的 topology/numeric/workspace resource；composition 与 solve plan
强引用其 operand。

## 与 legacy matrix-free API 的关系

field-based `ti.linalg.LinearOperator`、`MatrixFreeCG` 和 `MatrixFreeBICGSTAB` 继续保留原有
行为。它们使用 field-shaped vector 与 `(x, y)` kernel callback；由于该 ABI 不携带显式
topology、numeric resource、runtime generation 和 capability 信息，不会执行隐式转换。

迁移需要提供显式 scalar-ndarray kernel 或 Graph provider、显式 vector extent 和数学
trait。现有应用可以在具备这些合同前继续保留 legacy 路径；此实验性 API 不附带旧路径的
移除计划。

## 资格验证边界

API 已覆盖 backend correctness、lifecycle、trait、composition 和 solver regression。
应用的生产资格验证仍由 workload 决定：应在具有代表性的物理与非物理系统上验证
operator 语义、conditioning、tolerance、preconditioner 适用性、失败处理、内存预算和
backend driver 行为。

另见[稀疏 runtime 与线性代数](sparse_runtime_and_linear_algebra.zh.md)和
[物理稀疏算子与求解器选择](physics_sparse_solver_selection.zh.md)。
