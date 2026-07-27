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

公开 vector 参数保持一维 scalar-flat 数学 ABI，但 `apply()` 与单系统
`SolvePlan.solve()` 的边界可以接收一维 scalar Taichi ndarray、受支持的 dense field，
或显式 `VectorView`。vector payload 不经 host copy，不 materialize matrix-free
provider；请求不受支持的操作时也不会切换 backend。

operator、plan、provider、ndarray、field view 都属于同一 runtime generation。执行
`ti.reset()` 后它们全部失效，也不能重新绑定到后续 `ti.init()` session。

## Dense field 与 VectorView

受支持的 dense field 可以直接作为 `LinearOperator.apply()` 的 `input`、`out` 或
`addend`，也可以作为 `SolvePlan.solve()` 的 `rhs`、`initial_guess` 或 `out`：

```python
rhs = ti.field(ti.f32, shape=(nx, ny))
solution = ti.field(ti.f32, shape=(nx, ny))

operator.apply(rhs, out=solution)
result = plan.solve(rhs, initial_guess=solution, out=solution)
assert result.solution is solution
```

支持的 field 合同为：

- `ti.f32` 或 `ti.f64`；具体 operator/provider/backend 仍须支持相同 dtype；
- 1D、2D 或 3D `root -> dense -> place` scalar field；
- canonical packed `ti.Vector.field` 或 `ti.Matrix.field`；
- fixed shape、当前 `Program` 和仍然存活的 `SNodeTree`。

field 按 scalar-flat 顺序映射到 operator space：先按 index shape 的 canonical 顺序遍历，
再按 Vector lane 或 Matrix row-major component 顺序展开。因此 scalar extent 为
`prod(index_shape) * prod(element_shape)`，并且必须与 operator 的 domain/range extent
精确匹配。`pointer`、`bitmasked`、`dynamic`、`hash` 等 sparse SNode、quantized storage、
arbitrary nested dense tree 和 noncanonical component placement 会在提交前明确失败。

`vector_view()` 可声明 dense field 的显式 scalar 子集或排列：

```python
indices = ti.ndarray(ti.i32, shape=active_size)
indices.from_numpy(active_scalar_indices)

rhs_view = ti.linalg.experimental.vector_view(rhs, indices=indices)
solution_view = ti.linalg.experimental.vector_view(solution, indices=indices)
result = active_plan.solve(rhs_view, out=solution_view)
```

`indices` 可以是一维 `ti.i32` ndarray 或 root-dense scalar field。构造 view 时会复制、
检查并冻结 index topology；index 必须非空、在 source scalar extent 范围内且唯一。
后续修改原始 indices 不会改变既有 view。该构造执行一次显式 host validation；vector
数值的 apply/solve 路径始终留在设备上。indexed scatter 只覆盖选中的 scalar entry，
其它 field entry 保持不变。

dense field 互操作采用设备 staging，而不是宣称 zero-copy：

```text
dense field/view -> device pack or gather -> scalar ndarray provider ABI
scalar ndarray solver result -> device unpack or scatter -> dense field/view
```

每个 operator/plan 持有并复用兼容的 staging ndarray。warm solve 不重新分配 staging，
转换只发生在 apply/solve 边界，不进入 Krylov iteration。field `out` 会在同步 API 返回前
完成一次 unpack/scatter；`out=None` 继续返回一维 scalar ndarray。RHS/input 不能与 output
重叠；`initial_guess` 或 `addend` 可以与 output 是精确相同的 view，非精确重叠会失败。
稳定的 raw field binding 会在每个 operator/plan 内只完成一次资格解析，并复用同一个 implicit
view。canonical contiguous full-field 转换在 CPU/Vulkan 上使用原生 bulk copy；CUDA 以及 indexed
或带步幅的 field view 使用已编译 Graph replay。两条路径都避免重复 kernel specialization 与参数准备。
field output 的 overwrite `apply()` 在同一 completion boundary 内提交 provider 与 output
conversion；generalized coefficient 路径保持既有同步合同。

可以查询支持面和实际转换开销：

```python
capabilities = ti.linalg.experimental.vector_io_capabilities()
view_metadata = rhs_view.metadata
stats = plan.statistics()["vector_io"]
```

`stats` 报告 staging build/reuse/reserved bytes、implicit view 与 transfer plan 的
build/reuse/eviction、native bulk/Graph submission、pack/unpack、indexed gather/scatter、logical bytes、
direct ndarray binding、completion sync 和合并的 operator sync 次数。
`execution_mode="device_staged"` 表示支持 field API 且数值不经过 host；它不等同于
provider-native zero-copy。

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

`size` 也可以写成 `(range_extent, domain_extent)`。矩形 provider 必须通过
`adjoint=` 登记独立的伴随 kernel，才能使用 `operator.adjoint()`：

```python
operator = ti.linalg.experimental.LinearOperator.from_kernel(
    forward_kernel,
    (rows, columns),
    topology,
    adjoint=adjoint_kernel,
    numeric=values,
)
```

forward 的 `active_size` 是 range extent，adjoint 的 `active_size` 是 domain
extent；两侧共同需要的尺寸必须存放在显式 topology resource 中。没有登记 adjoint
时会明确失败，不使用 autodiff、host materialization 或 self-adjoint trait 猜测。

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

该 provider 是 f32 operator；`size` 可为整数方阵简写或 `(range, domain)`，并可通过
`adjoint=adjoint_graph` 登记具有相同 resource role schema 的独立伴随 Graph。它至少需要
一个 topology ndarray，并拒绝依赖 SNode 的 dispatch。topology、numeric data 与
workspace 会复制到 operator-owned resource。
CPU 把 Graph lowering 成 explicit sequence；CUDA/Vulkan 使用 compiled-Graph execution
合同。当已记录的 Graph runtime 规则要求时，backend capture/replay 可以使用 ordinary
Graph fallback，但不会改变数学 provider。Vulkan Graph replay 要求至少两个 dispatch；
单 dispatch Graph 仍正确执行，但 `operator.statistics()` 会将路径报告为
`ordinary_graph_fallback`。

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
y = operator.apply(x, alpha=2.0, beta=-0.5, addend=z)
y = operator @ x

B = 2.0 * operator
C = operator + B
D = operator.compose(B)       # operator(B(x))
E = operator.adjoint()
F = ti.linalg.experimental.block_diagonal((operator, B))
I = ti.linalg.experimental.identity(size, dtype=ti.f32)
```

通用形式为 `out = alpha * A(x) + beta * addend`。input/output alias 始终被拒绝；
`addend` 可以与 `out` 相同，以表达原地累加。当 `beta == 0` 时，`addend` 不会被验证或
读取。通用系数 lowering 当前支持 CPU；CUDA/Vulkan 只接受 `alpha == 1`、`beta == 0`
的 overwrite apply，其它组合明确失败且不经 host fallback。

只有 provider 提供显式 adjoint apply 时才能调用
`adjoint()`；实现不会把 self-adjoint trait 当作 fallback。

scale、sum、composition、block diagonal 和 identity 当前在 CPU 上执行，GPU lowering
不属于现有 API。GPU 组合会在构造时明确失败，不会执行 host code，也不会通过隐式 staging
路径进行同步。

## SolvePlan 与 SolveResult

`SolvePlan` 会跨多次调用保留 operator、solver state 和 persistent workspace。支持：

- `method="cg"`：面向 SPD 系统、使用 identity preconditioner 的 conjugate gradient；
- `method="pcg"`：fixed CSR 上使用 `"jacobi"`、fixed BSR 上使用
  `"block_jacobi"`，或使用一个可信 SPD `LinearOperator`/`PreconditionerPlan` 应用 fixed-linear
  近似逆的 PCG；
- `method="minres"`：面向允许不定的 square self-adjoint 系统，使用 identity 或 SPD
  preconditioner 的 MINRES；
- `method="bicgstab"`：面向一般 square 系统、使用 identity 或 fixed-linear 右
  preconditioner 的 BiCGSTAB；
- `method="gmres"`：面向一般 square 系统、使用 identity 或 fixed-linear 右
  preconditioner 的 restarted GMRES；
- `method="fgmres"`：使用有限 variable-linear 右预条件 action table 的 restarted
  flexible GMRES。

```python
result = plan.solve(rhs, initial_guess=x0, out=x)
print(result.iterations, result.residual_norm, result.termination_reason)
stats = plan.statistics()
```

对于系数不变的兼容路径，fixed-linear preconditioner 可以直接作为 operator 传入，而不是
应用回调：

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

### MINRES

`method="minres"` 使用与 CG/PCG 相同的 `LinearOperator` 与 lifecycle 合同，但要求可信的
`self_adjoint=True` trait，而不要求 operator 为正定：

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
    execution_policy="host_check_every_k",
    check_interval=4,
)
result = plan.solve(rhs)
```

CPU 为所有兼容的 CPU operator provider 提供 identity-preconditioned `f32/f64` MINRES。
CUDA 与 Vulkan 为 fixed CSR/BSR 和 compiled provider 提供 `f32`
identity-preconditioned MINRES。在 CUDA/Vulkan 上，fixed CSR 可选择 `"jacobi"`，block
size 为 2、3、6 或 12 的 fixed BSR 可选择 `"block_jacobi"`，也可直接提供可信的
device-native fixed-linear `LinearOperator` 或 `PreconditionerPlan`。MINRES preconditioner
必须 self-adjoint、positive-definite 且 nonsingular；应用仍需保证所选 scalar Jacobi
inverse 满足该数学合同。

MINRES 拒绝声明为 `singular=True` 的 operator，不提供 MINRES-QLP，也不提供兼容 singular
系统的 minimum-length 语义。显式存储的对称 matrix 必须完整且一致地保存两个 half。
即使启用 preconditioner，terminal status 也使用原系统的真实 residual 进行资格判断。

一个 CUDA/Vulkan MINRES plan 持有九个长度为 `n` 的持久 `f32` vector 与 144 bytes
持久 scalar state。该数字不包含调用方持有的 operator values、preconditioner resource、
RHS/output array、backend handle 与原生 replay object；具体配置应通过 `statistics()`
检查完整 plan/provider telemetry。

### BiCGSTAB

`method="bicgstab"` 为 square 非对称 operator 提供固定内存 Krylov 路径。它可使用
identity preconditioner，也可在右侧应用 fixed-linear
`LinearOperator`/`PreconditionerPlan`：

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

preconditioner 必须把 operator 的 range 映射回 domain、dtype 相同、为 fixed-linear，
并且不能声明为 singular。它与 PCG/MINRES preconditioner 不同，不要求 self-adjoint
或 positive-definite。右预条件使 terminal qualification 保持在原系统上：只有计算
真实 residual `b - A x` 后，求解才会报告收敛。

CPU 支持 `f32/f64` host-action provider。CUDA/Vulkan 支持 `f32` fixed CSR/BSR
以及 compiled kernel/Graph provider。compiled A/M provider 采用直接原生提交；
identity-preconditioned fixed stored A 可复用 CUDA Graph 或 Vulkan command-sequence
iteration chunk。系统不会为了适配其它 backend 路径把 provider 复制到 host。

device identity plan 持有六个长度为 `n` 的持久 vector；右预条件额外持有两个
preconditioned direction vector。两种配置均持有 112 bytes scalar state。
`statistics()` 精确报告 A/M apply、dot product、vector update、
logical/executed/wasted iteration、host observation、replay、workspace bytes 与
`preconditioning_side`。`SolveResult.breakdown_reason` 将 `nonfinite`、`rho`、
`alpha_denominator`、`omega_denominator`、`omega` 与普通 max-iteration
终止区分开。

BiCGSTAB 在 nonsingular 问题上仍可能停滞或 breakdown。它是低存储一般系统选项，
不能替代经过资格验证的 GMRES-family 方法在稳定性上的作用。

### Restarted GMRES

当 BiCGSTAB 的短 recurrence 不够稳健时，`method="gmres"` 为一般 square operator
提供有界内存的 Krylov 路径。`restart` 是 plan 构造参数，只能取 `8`、`16` 或
`32`，默认值为 `16`。

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

每个 Arnoldi step 都执行两遍 classical Gram-Schmidt（CGS2）。正交化先通过
multi-dot reduction 一次生成一组内积，再执行 fused projection，而不是为每个 basis
vector 发起一次由 host 观察的 dot product。每个 cycle 在 device 上更新 Givens rotation，
并使用原系统真实 residual `b - A x` 判断最终状态。`happy_breakdown` 单独计数，
不会与 `arnoldi_breakdown`、`orthogonalization_failure`、
`hessenberg_singular` 或 `nonfinite` 混为一类。

CPU 支持兼容的 `f32/f64` host-action provider；CUDA/Vulkan 支持 `f32` fixed
CSR/BSR 与 compiled kernel/Graph provider。可选 preconditioner 必须是 nonsingular
fixed-linear map，把 operator 的 range 映射回 domain，并应用在右侧；它不需要 PCG
或 MINRES 所要求的 SPD trait。`method="gmres"` 有意只接受 identity 或 fixed-linear
preconditioning；受支持的 variable-linear schedule 应使用 FGMRES。

plan 会预分配连续的 `(restart + 1) * n` basis。device identity plan 持有
`restart + 5` 个长度为 `n` 的持久 vector；右预条件再增加一个 vector。
Hessenberg、Givens、least-squares、multi-dot partial 与 terminal state 也都使用
持久内存，warm solve 不分配 transient solver workspace。`statistics()` 报告
`basis_vector_count`、`basis_reserved_bytes`、
`persistent_vector_reserved_bytes`、`persistent_scalar_reserved_bytes`、
精确的 A/M、dot、multi-dot、vector-pass 计数、restart cycle、
logical/executed/wasted iteration 与 replay 活动。应用在选择更大的 restart 前应检查
这些数据。

对于使用 identity preconditioner 的 fixed stored operator，CUDA Graph 与 Vulkan
command replay 覆盖完整 restart cycle。compiled provider 与右预条件 plan 保持相同
数值实现，但采用 direct native submission。CUDA 支持
`host_check_every_k` 且要求 `check_interval == restart`；Vulkan 支持该策略与
`fixed_budget_masked`。host 观察 terminal state 前会提交完整 cycle，因此最多可能执行
`restart - 1` 个 inactive tail step。更大的 restart 可能改善困难系统的收敛，但也会
同时扩大 basis 内存、cycle 工作量和最坏情况下的 inactive tail。

### 使用 variable-linear action table 的 FGMRES

`method="fgmres"` 接受 `behavior="variable_linear"` 的 `PreconditionerPlan`。plan
包含 1 到 32 个 linear action 的有限表；这一有界结构使 allocation、generation pinning
和 backend execution 均可显式审计，同时不会把 Python callback 引入 Arnoldi step：

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

solve-global scheduled inner slot `k` 选择 `actions[k % len(actions)]`，restart 不会重置
调度。CPU 的 scheduled slot 与 logical slot 一致；masked GPU execution 在较早发生逻辑
终止后仍可能调度 inactive tail slot，因此 telemetry 会把 action selection 与 executed
iteration 同 logical iteration 分开报告。每次 solve 进入时会同时 pin target 和所有
action generation。每个 action 都必须属于同一 Program、使用 solver dtype、把 operator
range 映射回 domain，并且不能声明为 singular。

CPU 支持兼容的 `f32/f64` host-action provider；CUDA/Vulkan 支持 `f32` 兼容 fixed
stored 与 compiled kernel/Graph provider。FGMRES 复用 GMRES 的 CGS2 Arnoldi、真实
residual 终止、restart 取值和 GPU observation policy，但会把每个预条件 basis vector
保存在持久 `Z` basis 中。额外预留量为 `restart * n * sizeof(dtype)` bytes，并通过
`preconditioned_basis_vector_count` 与 `preconditioned_basis_reserved_bytes` 暴露。
当 `restart=8` 时，经过资格验证的 CPU/device plan 分别持有 20/22 个 solver 持久
vector；该数字不包含 operator、action、RHS/output、backend 与 replay resource。

variable-action FGMRES 当前采用 direct native submission。只有 action table 中每个
action 都具备经过资格验证的 capture/binding 合同后，系统才会声明 CUDA Graph 或 Vulkan
command replay；当前 `solver_replay_unavailable_reason` 返回
`variable_action_capture_contract_unavailable`。API 不支持 nonlinear preconditioner、
Python iteration callback、自动 restart 选择、block GMRES 或领域 outer-solver policy。
把 variable-linear plan 传给 CG、PCG、MINRES、BiCGSTAB 或普通 GMRES 会在 plan 构造
阶段失败。

### 当前未支持边界

`0.5.1` 数值工具合同有意不提供：

- nonlinear、依赖 residual、adaptive 或 Python callback 驱动的 preconditioner；
- 自动 restart 选择、block/multi-RHS Krylov、recycling、deflation、pipelining 或
  communication-avoiding GMRES 变体；
- MINRES-QLP、singular minimum-norm/minimum-length 保证或自动 nullspace 处理；
- GPU `f64` GMRES-family 执行、GPU operator composition 或通用 GPU
  `alpha/beta/addend` apply；
- variable-action CUDA Graph/Vulkan command replay、single-system 异步 solve
  submission 或 device-convergent 条件终止；
- dynamic-topology solve plan、ragged batch 或透明 host fallback；
- 内建 IC/ILU/AMG、multigrid hierarchy 构建、Schur/field split、domain
  decomposition、离散、contact/KKT policy 或 nonlinear outer iteration。

独立 fixed-size batched CG/PCG 与 block Krylov/multi-RHS solve 是不同合同。不受支持的
backend/provider/policy 组合会在构造或 capability 校验阶段失败；系统不会通过更换
provider、execution policy 或数学问题来近似执行。

## PreconditionerPlan 生命周期

需要更新系数、显式复用或审计来源时，应使用 `PreconditionerPlan`：

```python
preconditioner = ti.linalg.experimental.PreconditionerPlan(
    operator,
    inverse_operator,
    method="external_block_inverse",
).setup()

z = preconditioner.apply(r)
pinned = preconditioner.pin()

# 应用先通过各自 provider 发布新的 numeric generation。
operator.update_numeric(next_a, expected_topology_version=1,
                        expected_numeric_version=3)
inverse_operator.update_numeric(next_m, expected_topology_version=1,
                                expected_numeric_version=7)
preconditioner.update()  # 声明 next_m 由当前 operator generation 重建

pcg = ti.linalg.experimental.SolvePlan(
    operator, method="pcg", preconditioner=preconditioner
)
```

`built_from_operator_stamp` 记录 action 实际由哪个 operator generation 构建；
`accepted_target_stamp` 记录它当前被批准服务于哪个 generation。target 更新后 plan 默认
stale。若算法允许 lagged preconditioning，可在 action 未改变时显式调用
`preconditioner.update(accept_reuse=True)`；此操作只更新 accepted stamp，不改写来源。
action 已改变时必须使用普通 `update()`。

`pin()` 同时保留精确 target 与 action generation，返回的 `PreconditionerSession` 可在后续
generation 发布后继续安全应用旧 generation。fixed-linear session 使用
`apply(r, out=None)`；variable-linear session 可通过 `iteration=k` 选择该 scheduled
slot 对应的 pinned cyclic action，默认 `iteration=0` 选择第一个 action。`metadata`
暴露 provenance/compatibility stamp；`PreconditionerPlan.statistics()` 暴露 setup、
rebuild、reuse、stale rejection、schedule update success/failure 以及 approved
generation 的 publish/retire/release 计数，`SolvePlan.statistics()` 另行报告 action
selection 与 schedule wrap。setup/update 在 host 边界执行；session apply 与 solver
iteration 只调用 native `OperatorAction`，不执行 Python callback。

对于 variable-linear table，`update(accept_reuse=...)` 可以接收一个应用于所有 action 的
boolean，也可以为每个 action 分别提供 boolean。任何 generation 发布前都会验证完整的
next table；一个 stale 或不兼容 action 会拒绝整个 update。已经 pin 旧 table 的 solve
继续持有该 immutable snapshot。`fixed_linear` 由文档列出的 PCG、MINRES、BiCGSTAB 与
GMRES consumer 支持；`variable_linear` 仅由 FGMRES 支持。`nonlinear` 仍只有描述能力，
并返回结构化 unsupported reason。内置 Jacobi/block-Jacobi 使用同一 native
setup/update/pin 生命周期；其 provider 构造仍通过 `preconditioner="jacobi"` 或
`"block_jacobi"` 选择。

`rhs`、`initial_guess` 和 `out` 必须与 operator 的 dtype、scalar extent 一致，并属于当前
runtime。未提供 `out` 时会创建结果 ndarray；未提供 `initial_guess` 时结果初始化为零。
RHS/output alias 会被拒绝。

`SolveResult` 同时包含 solution 与 terminal snapshot：status code、termination reason、
convergence/breakdown/max-iteration flag、iteration count、初始与最终 residual norm、两个
tolerance、relative reference norm 和 effective tolerance。CG、PCG、MINRES、
BiCGSTAB、GMRES 与 FGMRES 使用：

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
- CUDA 对 CG/PCG/MINRES/BiCGSTAB 默认使用 `"host_each_iteration"`，还支持
  `"host_check_every_k"`，其中 `check_interval` 可为 4 或 8。分块策略把 recurrence
  scalar 保留在 device 上，每个 chunk 只读取一次 terminal snapshot。CUDA GMRES/FGMRES
  默认使用 `"host_check_every_k"`，并要求 `check_interval == restart`。
- Vulkan 默认使用 `"fixed_budget_masked"`，还支持
  `"host_check_every_k"`。CG/PCG/MINRES/BiCGSTAB 接受 interval 4 或 8；
  GMRES/FGMRES 要求 `check_interval == restart`。两种策略均支持 `atol`、`rtol`
  及其组合后的 effective tolerance。

对于 fixed stored f32 CSR/BSR，CUDA 的 `host_check_every_k` 以及 Vulkan 的
`host_check_every_k`/`fixed_budget_masked` 会把受支持的 CG/PCG/MINRES 与
identity-preconditioned BiCGSTAB iteration chunk 与 identity-preconditioned GMRES
restart cycle 录制为可复用的原生执行序列。
可录制的 CG/PCG/MINRES 组合包括 identity、stored Jacobi 和 stored block-Jacobi
preconditioner。首次兼容执行建立 CUDA Graph 或 Vulkan command sequence；相同
topology、workspace 与 output binding 的后续执行直接 replay。仅更新 matrix values
不会重录 identity-preconditioned GMRES 序列；受支持的 preconditioner refresh 会保留
现有 CG/PCG/MINRES 序列。更换 output ndarray、改变 topology/schema 或重建 runtime
会使旧序列失效并安全重建。

FGMRES action table 在两个 GPU backend 上都使用 direct native submission；系统不会为
variable action schedule 静默复用 identity-GMRES replay 路径。

compiled-kernel 与 compiled Graph A/M provider 仍按 direct chunk submission 执行；它们不会
为取得 replay 而进行 host staging 或更换 provider。runtime 无法安全录制时也会保持相同数值
路径并报告不可用原因。`statistics()` 通过 `solver_chunk_builds`、
`solver_chunk_replays`、`solver_chunk_direct_submissions`、`solver_chunk_rebinds`、
`solver_chunk_invalidations`、`solver_graph_enabled` 和
`solver_replay_unavailable_reason` 暴露这一边界。构建开销属于 cold execution，性能资格应分别
记录 first solve 与 warm solve。

host 检查状态前，一个 chunk 或 GMRES-family restart cycle 总会完整执行。
`SolveResult.iterations` 表示逻辑上的
convergence 或 breakdown iteration；`statistics()["operations"]` 另行报告
`executed_iterations`、`wasted_iterations`、host synchronization 次数和 direct
chunk submission。因此 chunked solve 最多可能多执行 `K - 1` 轮 masked 或其它 inactive
tail。Vulkan fixed-budget execution 可以执行完整的 `max_iterations`，同时保留更早发生
的逻辑终止结果。

对于 CG/PCG/MINRES/BiCGSTAB，当更早终止比同步频率更重要时可选 `K=4`；
当摊薄 host check 更重要时可选 `K=8`。GMRES 与 FGMRES 使用选定的 restart 作为观察 interval。
较快选择取决于 vector size、operator 成本、iteration count、driver 与 backend。
不受支持的 policy 或 interval 会在 plan 构造时失败，不会静默 fallback。

`plan.execution_capabilities()` 返回执行策略矩阵，以及条件执行不可用时的结构化原因。
当前 CPU、CUDA 和 Vulkan solver 路径均不支持 `"device_convergent"`。显式请求会直接
失败，不会切换为 host check 或 fixed-budget execution；backend 能力也不会自动改变 plan
的默认策略。

## 独立批量 CG 与 PCG

`BatchedSolvePlan` 使用一个持久 plan 求解一组同构、相互独立的 SPD 系统。它接受 shape
为 `(B * N, B * N)` 的扁平 direct-sum operator，并把每个 vector 划分为 `B` 个连续、
长度均为 `N` 的系统：

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

`independent_systems=True` 是调用方必须给出的显式断言。系统 operator 以及 PCG 的
fixed-linear preconditioner 都必须保持分区：环境 `e` 的 output 只能依赖环境 `e` 的
input。全局 SPD trait 不能证明这一分区性质。违反该合同得到的是无效的独立批量问题，
而不是一个耦合系统求解。

首个公开布局有意保持同构和扁平：

- operator extent 必须能被 `batch_size` 整除；
- 所有系统使用相同 scalar extent 与 f32 dtype；
- `rhs`、`initial_guess` 和 `out` 的 shape 为 `(B * N,)`；
- `atol` 与 `rtol` 可以是 scalar，也可以是长度为 `B` 的 sequence；
- variable offset、active compaction 与 ragged system 不属于当前合同。

`method="cg"` 使用 identity preconditioner。`method="pcg"` 要求一个可信 SPD
`LinearOperator`，在相同扁平分区上应用 fixed-linear 近似逆。CPU、CUDA 与 Vulkan
均已验证 fixed stored 和 compiled-kernel A/M provider。其它 provider kind 仍受其普通
backend capability 与资格边界约束；batch plan 不会经 host staging，也不会切换 provider。

每个环境拥有独立的 recurrence scalar、effective tolerance、status、逻辑 iteration count
和 residual norm。一个环境可以收敛、breakdown 或达到 `max_iterations`，而不改变其它
环境的 terminal result。`BatchedSolveResult` 用 immutable tuple 暴露这些值，并单独返回
扁平 solution ndarray。batch size 为 1 时使用同一套单系统 CG/PCG 数值合同。

CPU 使用 `"host_each_iteration"`。CUDA 与 Vulkan 默认使用 K=4 的
`"host_check_every_k"`；也可选择 K=8、显式 `"host_each_iteration"` 或
`"fixed_budget_masked"`。host-check chunk 在观察到所有环境已终止前，可能发出 inactive
tail iteration。recurrence 与 vector-update kernel 会屏蔽 inactive 环境，但整体 A/M
provider 仍作用于完整扁平 batch；因此 convergence masking 不代表 provider apply 已做
compaction。

CUDA 与 Vulkan 会把稳定的 iteration recurrence 编译为 plan-owned Taichi Graph，并在
后续 solve 中复用。CG 每轮提交一个 recurrence Graph；PCG 在 A 之后和 M 之后分别提交一个
segment。operator 与 preconditioner 仍是 Graph 外部的 pinned provider action，因此 stored
和 compiled-kernel generation 继续遵循原有 update/retirement 合同。上一轮 solve 完成后
更换 `out` 只需 patch Graph binding。每个 workspace clone 拥有独立 replay plan，不会因
共享另一 clone 的 Graph lock 而串行化。

`statistics()` 会分别报告 executed system iteration、provider system iteration、masked
provider system iteration、active efficiency、host check、transfer 和 persistent resource
大小，使上述区别可观察。CG plan 拥有三个长度为 `B * N` 的 workspace vector，PCG 拥有
四个；此外还保留逐环境 recurrence、tolerance 与 status state。调用方拥有的 RHS、
solution、initial guess 和 provider resource 不计入这些 plan workspace 数字。batch plan
统计使用 schema version 4，并报告 `recurrence_replay_builds`、
`recurrence_replay_graph_builds`、`recurrence_replay_submissions`、
`recurrence_replay_logical_kernels`、output `recurrence_replay_rebinds` 与 direct recurrence
kernel submission；`recurrence_replay` record 会明确说明 A/M provider apply 不属于 Graph
replay 范围。

### 异步 fixed-budget submission

使用 `execution_policy="fixed_budget_masked"` 的 CUDA/Vulkan batch plan 可以提交完整的
masked iteration budget，而不在调用返回前读取 terminal state：

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
# 在这里提交应用中无依赖的工作。
submission.wait()
result = submission.result()
```

多个异步 solve producer 共享 GPU 时，可为 workspace clone 指定独立 lane，并用同一个
`SubmissionPacer` 控制整体 backlog 与公平准入：

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

完整 solve 的 host launch 序列以一次准入 turn 提交；提交结束后，两张票据可在
`max_in_flight` 边界内同时保持未完成。这里的异步只承诺宿主完成边界，不承诺两次 solve 的
GPU kernel 并发执行、独立 stream/queue 或设备抢占。`max_in_flight_per_lane` 防止高频 producer
独占全部 slot，lane 间 round-robin 则保证已有等待者获得有限的准入延迟。阻塞调用会以有界
自适应退避轮询全部 in-flight completion，使较晚完成的快 solve 可以先于更早提交的慢 solve
释放容量。该控制只覆盖共享同一 pacer 的调用，不能代替 engine 自身的 frame deadline、
任务依赖或取消策略。

`SolveSubmission.done()` 只观察 backend completion，不释放 workspace slot。`wait()` 在
需要时等待、生成完整逐系统 terminal snapshot，并释放 slot；`result()` 在需要时执行
相同操作并返回 immutable `BatchedSolveResult`。submission 会保留精确的 A/M
generation、input/output ndarray、workspace 和 backend completion，直到这一完成边界。
backend error 会由 `wait()` / `result()` 重新抛出；`ti.reset()` 会先等待已保留的 backend
work，再把尚未完成取值的 ticket 明确标记为 stale。

若在旧 submission 完成前发布新的 operator/preconditioner numeric generation，旧 generation
仍会保留到对应 completion。因此频繁数值更新可能使多个完整 values buffer 同时驻留；应用应
把 update cadence 纳入显存预算，或在更新前完成相关 ticket。Pacer 只限制共享它的 invocation
数量，不按 generation 字节数准入。

一个 `BatchedSolvePlan` 只拥有一个 submission slot。在 pending ticket 完成并生成结果前
再次提交会失败，不会共享 Krylov vector。需要多个独立 in-flight submission 时，使用
`clone = plan.clone_workspace()`；每个 clone 都拥有另一套完整 CG/PCG workspace vector
与 state。chunked host-check policy 和 CPU plan 没有通过 `submit()` 资格；调用会明确
失败，而不是把同步循环移动到 worker thread。

对 batch size `B`、每个系统大小 `N`，f32 CG 每个 plan/clone 的逻辑 workspace payload 为
`12 * B * N + 68 * B + 8` 字节，PCG 为 `16 * B * N + 68 * B + 8` 字节。使用
`statistics()["resources"]["clone_workspace_payload_bytes"]` 在创建 clone pool 前计算总量。
该数字不包含 allocator/driver 开销、调用方 vector 或 operator/preconditioner 资源。大型 solve
默认使用 `max_in_flight=1`；只有 profile 证明宿主重叠带来有效收益且资源与尾延迟仍满足预算时，
才增加到 2。小系统应优先扩大 batch，而不是按物理实体创建 plan clone。

该 API 表示 independent batching，不是对隐式耦合 block matrix 使用 global-scalar CG，
也不是 multi-RHS CG、block CG 或其它 block Krylov 方法。耦合系统必须使用在数学上显式
表达该耦合关系的 operator 与 solver。

## 支持矩阵

### Provider 与 apply

| Provider | CPU | CUDA | Vulkan |
| --- | --- | --- | --- |
| Fixed stored CSR/BSR | `f32`、`f64` | `f32` | `f32` |
| Compiled kernel | `f32` | `f32` | `f32` |
| Compiled Graph | `f32` | `f32` | `f32` |
| Identity/composition | `f32`、`f64` | 不支持 | 不支持 |

kernel/Graph provider 支持矩形 shape 和显式 adjoint。通用 `alpha/beta` apply 支持 CPU；
GPU 当前只支持 overwrite apply。

### Solver

| Method/provider | CPU | CUDA | Vulkan |
| --- | --- | --- | --- |
| CG，fixed stored | CSR/BSR，`f32/f64` | CSR，`f32` | CSR/BSR，`f32` |
| CG，compiled kernel/Graph | `f32` | `f32` | `f32` |
| CG，CPU composition | `f32/f64` | 不支持 | 不支持 |
| PCG + Jacobi | fixed CSR，`f32/f64` | fixed CSR，`f32` | fixed CSR，`f32` |
| PCG + block-Jacobi | fixed BSR，`f32/f64` | fixed BSR，`f32` | fixed BSR，`f32` |
| PCG + fixed-linear operator/plan | 受支持 provider，`f32/f64` | compiled-kernel A/M，`f32` | compiled-kernel A/M，`f32` |
| MINRES + identity | 受支持 provider，`f32/f64` | fixed CSR/BSR 或 compiled provider，`f32` | fixed CSR/BSR 或 compiled provider，`f32` |
| MINRES + Jacobi/block-Jacobi | 不支持 | 分别为 fixed CSR/BSR，`f32` | 分别为 fixed CSR/BSR，`f32` |
| MINRES + fixed-linear operator/plan | 不支持 | 兼容的 device-native A/M，`f32` | 兼容的 device-native A/M，`f32` |
| 独立批量 CG/PCG | fixed stored 或 compiled-kernel A/M，`f32` | fixed stored 或 compiled-kernel A/M，`f32` | fixed stored 或 compiled-kernel A/M，`f32` |
| 批量 fixed-budget submission | 不支持 | fixed stored 或 compiled-kernel A/M，`f32` | fixed stored 或 compiled-kernel A/M，`f32` |
| device-convergent 条件执行 | 不支持 | 不支持 | 不支持 |
| BiCGSTAB + identity | 受支持 host-action provider，`f32/f64` | fixed CSR/BSR 或 compiled provider，`f32` | fixed CSR/BSR 或 compiled provider，`f32` |
| BiCGSTAB + fixed-linear 右预条件 | 受支持 host-action provider，`f32/f64` | 兼容的 device-native A/M，`f32` | 兼容的 device-native A/M，`f32` |
| GMRES + identity | 受支持 host-action provider，`f32/f64` | fixed CSR/BSR 或 compiled provider，`f32` | fixed CSR/BSR 或 compiled provider，`f32` |
| GMRES + fixed-linear 右预条件 | 受支持 host-action provider，`f32/f64` | 兼容的 device-native A/M，`f32` | 兼容的 device-native A/M，`f32` |
| FGMRES + variable-linear action table | 受支持 host-action provider，`f32/f64` | 兼容的 device-native A/actions，`f32` | 兼容的 device-native A/actions，`f32` |

direct factorization 继续使用 stored-matrix API。

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
preconditioner，同时保留 pattern 与 Krylov workspace。scalar CSR Jacobi 保存 diagonal
reciprocal；BSR block-Jacobi 为每个 diagonal block 保存 lower Cholesky factor，支持的
block size 为 2、3、6 和 12。每个 block 都必须有限、对称且正定；不合法的 block 会明确
失败，不会执行对称化、正则化或 fallback，失败的 refresh 也不会发布新的 preconditioner
generation。topology 改变后必须构造新 operator。

CUDA 与 Vulkan 在 device 上完成 value-only Jacobi 和 block-Jacobi refresh。成功刷新后
committed resource address 保持稳定，因此可重放 solve plan 可以只 rebind numeric
generation，而不必重建 recurrence program。warm refresh 不经 host 复制完整 values 或
factor，也不分配 device memory；validation 只回读固定大小的 status。`statistics()` 提供
refresh contract、operation counter、transfer bytes 和 device-allocation count。

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

`qualify_operator()` 可对任意公开 `LinearOperator` 生成版本化、JSON 可序列化的协议证据：

```python
report = ti.linalg.experimental.qualify_operator(
    operator,
    reference=dense_reference,
    samples=4,
    warmup=2,
    repetitions=10,
    metadata={"case": "poisson_level_3"},
)
report.to_json()
matrix = ti.linalg.experimental.summarize_operator_qualifications([report])
```

报告包含 backend/build、provider、shape、capability、resource stamp、forward/adjoint oracle
误差、线性与 dot-product identity、通用 apply 的 `beta=0` no-read 状态、同步边界计时和
native counter。`summarize_operator_qualifications()` 可从多组 detached
backend/provider report 自动生成确定性的支持矩阵。GPU 不支持的通用系数会记录为
`unsupported`，不会伪装为通过或触发 host fallback。计时只描述本机本次运行；它不是
跨机器性能门。

`qualify_solve_plan()` 为公开 `SolvePlan` 或 `BatchedSolvePlan` 生成对应的执行证据：

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

该示例使用 CUDA/Vulkan chunk policy；CPU factory 应改用 `host_each_iteration`。

传入零参数 factory 时会单独记录 plan 构建时间；也可以传入已构造的 plan，此时 build timing
明确记为 unavailable。`reference` 可以是扁平 expected solution，也可以是接收 RHS NumPy
副本的 callable。独立 batch 使用一份 flat reference，并可逐系统指定 expected termination。

报告区分首次与稳态同步 wall time。合格的 GPU fixed-budget batch 还会拆分 host submit 和
completion wait，并可记录调用方提供的 `SubmissionPacer`。记录内容包括 A/M provider、policy/K、
terminal state、独立计算的 `b - A(x)` 真实残差、logical/executed/provider iteration、inactive-work
efficiency、chunk direct/replay counter、transfer、plan resource 和进程全局 device pool 增量。runtime
没有安全查询时，device timestamp span、device identity 和 driver version 会明确保持 unavailable；
不会把 wall time 改名为 device time。Nsight 等 profiler 结果可通过 metadata 作为 sidecar 保存。

一次资格运行会执行一次 first solve、指定数量的 warmup/repetition，并额外执行一次不计时的 operator
apply 来计算独立真实残差；因此会改变 plan counter 和传入的 output。性能证据应使用专用 plan/workspace，
尤其不要让异步 batch 或共享 pacer 的资格运行污染生产调度。函数只返回 detached evidence，不写文件。

API 已覆盖 backend correctness、lifecycle、trait、composition、10k approved-generation
churn 和 solver regression。
应用的生产资格验证仍由 workload 决定：应在具有代表性的物理与非物理系统上验证
operator 语义、conditioning、tolerance、preconditioner 适用性、失败处理、内存预算和
backend driver 行为。

另见[稀疏 runtime 与线性代数](sparse_runtime_and_linear_algebra.zh.md)和
[物理稀疏算子与求解器选择](physics_sparse_solver_selection.zh.md)。
