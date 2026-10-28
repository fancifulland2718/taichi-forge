# 局部稠密矩阵求解

[English](local_matrix_solve.en.md) · [文档入口](index.zh.md)

本页描述待发布的 FG-MATH-01 增强。已发布 0.7.0 的 `ti.solve` 仅支持 2/3 阶，
`Matrix.inverse()` 仅支持 1–4 阶。

定向验证覆盖 CPU/CUDA 的 1、5、9、16、32 阶 f32/f64，Vulkan 的这些尺寸 f32，
以及 CPU 64 阶 f32。对照 NumPy 解和逆矩阵，并检查归一化方程残差，包含零首主元的换行。
CUDA 64 阶探针在 90 秒限时内未完成编译（此前长运行已明显消耗编译时间和主机内存）；
该尺寸的 GPU 路径不列入本轮资格化范围，也不设人为拒绝上限。

当前源码中，`ti.solve(A, b, dt=...)` 支持正整数固定尺寸的方阵和同长度向量，
`A.inverse()` 支持固定尺寸方阵，**没有 16 阶或其他固定上限**。
2/3 阶 solve 与 1–4 阶 inverse 保留原有实现；新增尺寸使用带部分选主元的高斯消元。

```python
@ti.kernel
def solve_block(a: ti.types.matrix(9, 9, ti.f64),
                b: ti.types.vector(9, ti.f64)) -> ti.types.vector(9, ti.f64):
    return ti.solve(a, b, dt=ti.f64)
```

- 新 solve 路径使用 f32/f64，`dt` 未指定时沿用 `default_fp`，不会自动根据 A 推断。
  inverse 保留浮点输入精度；整数输入先转换为浮点。各后端须支持请求的精度。
- 选主元、消元和回代采用运行期循环，不按三重静态循环展开；线程内保持串行，
  因此也可在外层并行循环中逐项求解。求逆复用同一次消元处理所有单位矩阵列。
- 输入应有限且非奇异；不自动正则化，不用固定 epsilon 将小主元当作零。
  检测到零主元时，debug 模式断言失败。新增路径在非 debug 且 `fast_math=False` 时
  返回 NaN；开启 fast-math 时不保证奇异输入的返回值。Python scope 的新 inverse 路径会断言。
- 本轮资格化是前向数值求解，不包含新增尺寸的自动微分、任意病态系统的精度或吞吐保证。

固定尺寸支持不等于任意大矩阵都适合放在每个线程的局部 `Matrix` 中。输入展开和局部
存储随尺寸平方增长，消元工作随尺寸立方增长；较大矩阵可能增加编译成本和设备局部内存访问。
运行期可变尺寸或大型系统应使用 ndarray，并按问题结构选择
[LinearOperator / SolvePlan](linear_operator.zh.md)；CUDA 稠密 SPD 场景另有显式
[cuSOLVERDn provider](external_hardware_providers.zh.md)。这些接口的后端和结构限制仍适用，
不能把本次局部矩阵增强解释成通用动态尺寸稠密 LU provider。
