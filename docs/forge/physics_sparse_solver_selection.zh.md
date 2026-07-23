# Taichi Forge 物理稀疏算子与求解器选择指南

> 适用于 Taichi Forge **0.5.x** 发布线。

[English version](physics_sparse_solver_selection.en.md)

## 快速选择

按以下顺序选择：

1. 先把线性化算子分类为SPD、对称不定或非对称；
2. 再判断拓扑是固定、只改values，还是需要重建；
3. 最后选择明确支持该solver类别的storage/operator和backend provider。

| 物理workload | 当前优先起点 |
|---|---|
| 规则pressure Poisson | `LinearOperator` + `MatrixFreeCG`；只有需要显式矩阵时才用fixed CSR + `SparseCG` |
| mass-spring或SPD implicit FEM | fixed CSR/BSR + CG和Jacobi/block-Jacobi；天然2/3/6/12-DOF block优先BSR |
| active grid变化的implicit MPM | 空间装配使用SNode；迭代前发布compact DOF和显式或matrix-free operator |
| 每步particle/contact adjacency | 用count-scan-fill或sorted arrays建立拓扑；这属于装配，不是solver选择 |
| bilateral constraint或对称KKT | 完整对称 CSR/BSR 或 compiled self-adjoint operator + `experimental.SolvePlan(method="minres")`；旧 `SparseMINRES` class 仍仅支持 CPU |
| friction或其它非对称线性化 | 低存储起点可用 `experimental.SolvePlan(method="bicgstab")`；需要 Arnoldi minimum-residual 路径时使用 restarted `method="gmres"`；两者都消费受支持的 fixed/compiled operator |

不存在只看矩阵size、CSR/BSR格式或“稀疏”标签就安全的selector。Taichi不会从storage
自动推断symmetry或positive definiteness。

## 先判断算子类别

### 对称正定

只有operator和preconditioner都满足SPD合同时才能使用CG/PCG。常见候选包括已正确
处理nullspace和boundary condition的pressure Poisson、稳定化mass-spring Hessian，
以及已知线性化为SPD的implicit elasticity系统。

`SparseCG`使用真实残差条件
`||b - A x|| <= max(atol, rtol * ||b||)`。受支持CPU/CUDA路径可选scalar Jacobi；
fixed BSR在format capability允许时可选block-Jacobi。CG运行到breakdown不能替代事前
判断operator类别。

### 对称不定

bilateral constraint、saddle-point、mixed formulation和KKT matrix可能对称但不定。
不能因为它们是 square 或 block sparse 就送进 CG。旧 `SparseMINRES` class 支持 CPU
mutable CSR/CSC 与 caller-owned fixed CSR/BSR，并使用 identity preconditioner。

`experimental.SolvePlan(method="minres")` 接受携带可信 self-adjoint trait 的
`LinearOperator`。CPU 支持 identity-preconditioned `f32/f64`；CUDA/Vulkan 支持 `f32`
fixed CSR/BSR 与 compiled provider。GPU 路径可使用 identity、文档列出的 fixed-CSR
Jacobi、fixed-BSR SPD block-Jacobi，或兼容的可信 fixed-linear preconditioner。显式 matrix
必须完整且一致地存储两个对称 off-diagonal half。

这是可复用的线性 runtime primitive，不是完整的 constraint 或 multiphysics solver。
operator 分类、nullspace 消除、scaling、constraint regularization、nonlinear/active-set
执行顺序与领域 preconditioner 设计仍由应用负责。该路径拒绝声明为 singular 的 operator，
也不提供 MINRES-QLP、minimum-length、field-split 或 Schur-complement policy。

### 非对称

frictional contact线性化、advection-like项和一些coupled系统是非对称的。
`experimental.SolvePlan(method="bicgstab")` 支持兼容的 CPU `f32/f64`
host action，以及 CUDA/Vulkan `f32` fixed 或 compiled provider。它可使用 identity
或 fixed-linear 右 preconditioner，并以原系统真实 residual 认定收敛。旧
`SparseBiCGSTAB` 保留 CPU stored-matrix 路径；`MatrixFreeBICGSTAB` 保留
field-based 路径。

BiCGSTAB可能breakdown，也不能证明condition良好。complementarity、active-set、
Newton iteration和nonlinear contact不属于线性runtime合同。

`experimental.SolvePlan(method="gmres")` 是 restarted Arnoldi 备选。它支持兼容的
CPU `f32/f64` host action，以及 CUDA/Vulkan `f32` fixed 或 compiled provider；
可使用 identity 或 fixed-linear 右 preconditioner。restart 可为 8、16 或 32，plan
会预分配完整 basis，并报告 basis/workspace bytes 与 logical/executed work。
每个 restart boundary 都检查原系统真实 residual。这是通用线性代数 primitive，
不提供 contact policy、active-set/Newton outer loop 或领域 preconditioner。

当应用已经持有一个有界 linear 近似逆 action 序列时，
`experimental.SolvePlan(method="fgmres")` 可接收包含 1 到 32 个 action 的
variable-linear `PreconditionerPlan`。cyclic selection 跨 restart boundary 连续，solve
期间会 pin 全部 generation，并预留持久 preconditioned `Z` basis。这只是为外部设计
action 提供 execution/lifecycle primitive；它不会构造 multilevel cycle、field split、
Schur complement、contact preconditioner 或 nonlinear outer policy。

## 根据拓扑生命周期选择存储

| 拓扑生命周期 | storage/operator选择 | update合同 |
|---|---|---|
| 规则且隐式 | dense/compact field stencil或`LinearOperator` | topology留在kernel结构中；显式更新coefficient fields |
| fixed compressed pattern | `SparsePattern.csr()`或`.bsr()`加values | 复用analysis/operator/workspace；按compressed order发布value-only update |
| 每步重建 | count-scan-fill、sort/RLE，再构造exact CSR/BSR arrays | 完整验证后发布新generation |
| online且count未知 | 只有无法先count时才用bounded `dynamic` SNode | overflow为error；可能留下mutable partial state |
| 空间coordinate grid | 装配阶段使用pointer/bitmasked bricks | 反复Krylov迭代前先分配稳定compact DOF |

active coordinate已经获得稳定DOF后，不要在每轮solver iteration里继续遍历pointer/hash
SNode；反过来也不要把CSR当成online spatial activation目录。

fixed BSR支持2、3、6、12 block size，适合每个mesh node或rigid body天然拥有小型
dense block的场景。`6 + 6 + 1/3`这类mixed KKT field不应只为复用solver接口就
padding成统一6-lane BSR。

## 支持的provider矩阵

| 路径 | CPU | CUDA | Vulkan |
|---|---|---|---|
| 显式sparse SpMV | 受支持formats | 受支持formats | 受支持formats |
| `SparseSolver` direct solve | CSR/CSC providers | 文档列出的CSR provider | 不支持 |
| `SparseCG` | mutable和fixed CSR/BSR capabilities | CSR和fixed BSR capabilities；受dtype/format限制 | 不支持 |
| `SparseMINRES` | mutable和fixed CSR/BSR capabilities | 不支持 | 不支持 |
| `experimental.SolvePlan(method="minres")` | 兼容 operator、identity，`f32/f64` | fixed CSR/BSR 或 compiled operator，支持 identity/内置项/兼容 fixed-linear preconditioner，`f32` | fixed CSR/BSR 或 compiled operator，支持 identity/内置项/兼容 fixed-linear preconditioner，`f32` |
| `SparseBiCGSTAB` | mutable和fixed CSR/BSR capabilities | 不支持 | 不支持 |
| `experimental.SolvePlan(method="bicgstab")` | 兼容 host action、identity/fixed-linear 右 preconditioner，`f32/f64` | fixed CSR/BSR 或 compiled A/M，`f32` | fixed CSR/BSR 或 compiled A/M，`f32` |
| `experimental.SolvePlan(method="gmres")` | 兼容 host action、identity/fixed-linear 右 preconditioner，`f32/f64` | fixed CSR/BSR 或 compiled A/M、restart 8/16/32，`f32` | fixed CSR/BSR 或 compiled A/M、restart 8/16/32，`f32` |
| `experimental.SolvePlan(method="fgmres")` | 兼容 host A/actions、variable-linear table，`f32/f64` | 兼容 device-native A/actions、restart 8/16/32，`f32` | 兼容 device-native A/actions、restart 8/16/32，`f32` |
| `MatrixFreeCG` | kernel/field路径 | kernel/field路径 | backend/dtype受支持时可用 |
| `MatrixFreeBICGSTAB` | kernel/field路径 | kernel/field路径 | backend/dtype受支持时可用 |

始终检查format、dtype、shape和provider-specific error。某个backend支持BSR并不表示同一
backend也支持direct solve、MINRES或所有dtype。

## workload说明

### pressure与规则grid

matrix-free stencil能避免重复index/value storage，且boundary/nullspace合同明确时应
优先使用。只有既有stored solver、导出需求或不规则boundary表示确有价值时才用fixed
CSR。Vulkan提供matrix-free CG，但没有stored `SparseCG`。

### FEM、mass-spring与implicit MPM

fixed FEM或spring pattern应跨value update保留，只在remesh/topology change时重建；
天然block DOF使用BSR。MPM空间阶段仍适合block-sparse SNode，但solve阶段应消费
compact operator generation。CG是否适用取决于实际linearization、contact和material
treatment，而不是取决于“MPM”这个名称。

### contact与constraint

contact adjacency和linear solve必须分层。能先得到count时，应精确构造row offsets和
payload，不应通过`dynamic`逐项append。经过资格确认的 nonsingular 对称 bilateral KKT
可在受支持的 CPU/CUDA/Vulkan provider 上使用实验性 MINRES plan；旧 stored-matrix class
仍仅支持 CPU。friction 或其它非对称系统可在受支持 fixed 或 compiled provider 上使用
实验性 BiCGSTAB 或 restarted GMRES plan。BiCGSTAB 仍是可能发生数值 breakdown
的低存储方法；应用提供 variable-linear table 时也可以使用 FGMRES。GMRES-family
以预分配 V/Z basis 与 restart-cycle 工作量换取 Arnoldi minimum-residual 路径。
这些路径都不提供 complementarity 或 active-set 处理。

## 失败与生命周期规则

- provider缺失必须显式unsupported，不能据此把GPU matrix无声复制到host求解。
- fixed-pattern value update必须保持stored scalar count和compressed order；topology
  改变时必须新建pattern/generation。
- 依赖的numeric或topology generation改变后，solver/preconditioner binding会stale；
  应按solver provider合同重建或显式refresh。
- 只有具体builder声明时，exact ndarray publication失败才是transactional；SNode
  overflow属于可能已有partial mutation的失败。
- `ti.reset()`会失效runtime-owned matrix、plan、ndarray和generation object；
  不能跨Program保留native address。

## 避免这些错误

- 不要根据square shape、CSR/BSR格式或positive diagonal选择CG。
- 不要把对称不定KKT送进CG。
- 不要认为Vulkan支持sparse storage就等于支持Vulkan stored sparse solver。
- value-only update不要重建固定row/column indices。
- 每轮Krylov step不要做SNode listgen或hash probe。
- workload合同和overflow/lifecycle测试未冻结前，不要调block size、damping、tolerance
  或crossover。

## 相关文档

- [稀疏 runtime 与线性代数](sparse_runtime_and_linear_algebra.zh.md)
- [稀疏布局选择指南](sparse_layout_selection.zh.md)
- [线性求解器](../lang/articles/math/linear_solver.md)
- [稀疏矩阵与固定 pattern](../lang/articles/math/sparse_matrix.md)
- [Forge API 参考](forge_api_reference.zh.md)
- [Native algorithms](native_algorithms.zh.md)
