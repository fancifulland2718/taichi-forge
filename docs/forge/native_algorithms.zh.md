# Native 算法

本文说明 Forge 公开算法入口中可能走 CPU、CUDA 或 Vulkan native 实现的部分。对于
`method="auto"`，只有当 dtype、shape、layout 和后端能力都满足已知合同时才选择 native
路径；不支持的组合必须回退到正确的通用路径。显式指定 native method 时，不支持应清晰拒绝。

按模块整理的 Forge-only API 符号清单见 [Forge API 参考](forge_api_reference.zh.md)。

核心 native 算法族首次发布于 Forge 0.4.0；Graph native replay 与 device-side 检查分别
在 0.4.1 和 0.4.23 进入公开版本。本文说明当前 0.5.x 的可移植性与安全合同；只有
[发行说明](release_notes.zh.md#050)明确列出的变化才属于 0.5.0 新增。

## 公开入口

| 入口 | 用途 |
| --- | --- |
| `ti.algorithms.sort(keys, values=None, ...)` | Forge 稳定排序调度器。 |
| `ti.algorithms.sort_by_key(keys, values, ...)` | 排序 keys 并同步移动 payload values。 |
| `ti.algorithms.parallel_sort(keys, values=None)` | vanilla 兼容的 legacy sorter。 |
| `ti.algorithms.PrefixSumExecutor(n).run(values)` | Prefix sum / scan。 |
| `ti.algorithms.experimental_compact(values, flags, output, count, ...)` | 按 flags 过滤并写入紧凑输出。 |
| `ti.algorithms.experimental_reduce(values, output, op="sum", ...)` | 将 values reduce 到 `output[0]`。 |
| `ti.algorithms.experimental_histogram(values, bins, ...)` | 将整数 values 统计到 bins。 |
| `ti.algorithms.experimental_transform(src, dst, scale=..., bias=..., ...)` | 元素级 affine transform 和 copy。 |
| `ti.algorithms.experimental_gather(src, indices, dst, ...)` | Indexed read。 |
| `ti.algorithms.experimental_scatter(src, indices, dst, ...)` | Indexed write。 |
| `ti.algorithms.experimental_scatter_add(src, indices, dst, ...)` | Indexed add；支持时走后端 atomics 或 staged reduction。 |
| `ti.algorithms.experimental_bucket_builder(keys, values, offsets, output, ...)` | 构建 grouped/bucketed value output。 |
| `ti.algorithms.experimental_grouped_reduce(keys, values, output, ...)` | 按整数 group key reduce values。 |
| `ti.algorithms.count_if(flags, ...)` / `any_if()` / `all_if()` | 在 kernel 外部发起 device-side 数值谓词检查。 |
| `ti.algorithms.nan_count(values, ...)` / `inf_count()` / `all_finite()` | 在 device 上统计 NaN/Inf/非有限值。 |
| `ti.algorithms.index_bounds_check(indices, lower=..., upper=...)` | 在 device 上统计越界 index。 |
| `ti.algorithms.max_abs(values, ...)` / `max_abs_delta(values, reference, ...)` | 在 device 上计算收敛/误差类最大绝对值指标。 |

名称里的 `experimental_` 表示这是 Forge 公开入口，但演进节奏会比长期 vanilla API 更保守。

## 后端选择

大多数 API 支持 `method="auto"`。auto 只在输入合同明确支持时选择 native 后端，否则保持正确
性并走通用或 host fallback。

常见显式 method 家族包括：

- `cpu_native`
- 可用时的 CUDA native/CUB method
- 可用时的 Vulkan native method
- sort 类操作的 `host_stable` 或 legacy fallback method

显式 native method 适合测试或受控部署，不应被当成跨所有后端的可移植承诺。

## 机器可读 capability 合同

Forge 0.5.0 为当前每个 primitive family 公开不可变的 schema-v1 描述：

```python
contract = ti.algorithms.primitive_capability("experimental_reduce")
for operand in contract.operands:
    print(operand.name, operand.dtypes, operand.ranks, operand.layouts)

ti.init(arch=ti.vulkan)
active = ti.algorithms.resolve_primitive_capability("reduce")
for method in active.methods:
    print(method.method, method.program_available)
```

`primitive_capability(name)` 与 `primitive_capabilities()` 是静态查询，可在
`ti.init()` 前调用。它们按 operand role 报告 dtype、rank、layout、storage，
并描述 backend method、稳定性、确定性、atomic 顺序依赖、primal/forward/reverse/
explicit-adjoint、Graph、AOT、workspace 与 fallback 合同。也可以用
`experimental_reduce` 等公开入口名作为 family alias。

`resolve_primitive_capability(name)` 要求已有 active Program。它将 method 过滤到
当前 CPU/CUDA/Vulkan 后端，并执行调度层使用的同一组无副作用 provider probe。
`program_available=True` 只表示当前 Program 含有该 provider，不表示任意 dtype/
layout 请求一定有效；method 仍标记 `input_dependent=True`，具体公开操作会在写入前
完成最终请求校验。

catalog 同时是公开 `method=` 校验与 native AD policy 的单一来源，避免文档中的
method/adjoint 能力与真实调度逐渐漂移。

### 自动微分

- 在 `ti.ad.Tape()` 中，只有具体输入同时具备完整注册 backward 时，
  `method="auto"` 才选择 native primal；否则走声明的 kernel fallback。没有
  backward 的显式 native method 会在写入前拒绝。
- 在 `ti.ad.FwdMode()` 中，transform、reduce-sum、gather、scatter 和 scatter-add
  走可微 helper-kernel fallback；其 JVP 已在 CPU、CUDA、Vulkan 上做回归。
  由于还没有 native forward launcher，显式 native method 会拒绝。
- scan 与 grouped-reduce 当前会在写入前拒绝 `FwdMode`；它们的现有 fallback
  不能提供可移植的实数 forward 合同。
- sort、compact、histogram、bucket-builder、device check 与 device metric 被明确
  标为不可微，在写入前拒绝 automatic AD context。应在 Tape/FwdMode 外完成这类
  预处理或诊断。

这些规则只说明 automatic AD。native Graph node 仍是 primal-only，native-node AOT
serialization 仍不支持。

### CUDA runtime 可移植性

CUDA native/CUB provider 随平台级 `taichi-forge-runtime` wheel 发行。用户不需要安装本机
CUDA Toolkit，也不需要选择 CUDA 版本化包；`method="auto"` 仍以运行时 capability 为准，
不支持时走既有正确 fallback。显式 CUDA native method 在 provider 或 driver 不兼容时会清晰
拒绝。构建 Toolkit、包内 CUDART 与最低 driver 是不同边界；当前默认构建基线和降低门槛前的
验证要求见 [构建 Wheel](build_wheels.zh.md)，尚待 Linux 实测的项目见
[Linux 复测状态](linux_revalidation.zh.md)。

## 数据合同

- Dense 1D `ti.ndarray` 是主要 native 算法 ABI。
- Dense field/SNode 只有在能证明兼容 dense layout，或能提供安全 staging 路径时才走支持路径。
- `StructNdarray` 可作为 order/copy 类 primitive 的 opaque payload。部分数值 primitive 支持 scalar 或 packed tensor member view；完整语义见 [StructNdarray primitive 语义](struct_ndarray_api.zh.md)。
- 稀疏、非连续、复杂 SNode 拓扑不能默认假设走 native。
- 普通 `experimental_scatter()` 要求所有有效 destination index 唯一。
  CPU native scatter 会在写入前验证，并拒绝 duplicate；需要 duplicate target 时应使用
  `experimental_scatter_add()`。
- duplicate target 的 floating scatter-add 可能受后端 atomic 顺序影响；只有数值合同允许时才应使用。

## Device-side 数值检查

这些 API 是 Forge 新增公开 API，不是 vanilla Taichi 1.7.4/1.8.0 API。它们必须在
Python scope 调用，不能在 `@ti.kernel` 或 `@ti.func` 内部调用。调用本身只提交后端
native primitive，并把结果写入 device-side scalar；只有调用 `to_int()`、`to_bool()`、
`ok()` 或 `to_float()` 时才把标量读回 Python。

`check_count` 类 API 支持 `i32/u32/i64/u64/f32/f64` 的 scalar 1D `ti.ndarray`、
`StructNdarray` scalar member view，以及 root-dense-place dense field。`metric_reduce`
类 API 支持 `f32/f64` 的同类输入；Vulkan 当前只开放 `f32` metric fast path。
`max_abs_delta` 可以直接比较 shape/dtype 相同的 dense field 与 plain ndarray 或
`StructNdarray` scalar member view；这会走后端 native mixed-storage 路径，不经过
host staging。

```python
workspace = ti.algorithms.CheckWorkspace(max_items=n)
bad = ti.algorithms.index_bounds_check(indices, lower=0, upper=n, workspace=workspace)

# Python 分支会同步读取 1 个标量。
if not bad.ok():
    raise RuntimeError("indices out of bounds")
```

对热循环，显式复用 `CheckWorkspace` / `MetricWorkspace` 可以复用 result scalar、scratch
buffer 和后端 replay plan。

## Workspace

多数 native 算法接受 `workspace=` 或返回可复用 workspace。复用 workspace 可以让后端 scratch
buffer 和 native plan 跨帧或跨重复调用存活，是热循环推荐写法。

```python
workspace = None
for _ in range(num_steps):
    workspace = ti.algorithms.experimental_transform(
        src, dst, scale=2.0, bias=1.0, method="auto", workspace=workspace
    )
```

## 与 graph 的关系

Forge 可以把由算法层产出的 DSL-defined native primitive sequence 放进 graph replay。
`DeviceCheckResult`、`DeviceMetricResult` 和 `PrimitiveSequence` 都可以通过
`GraphBuilder.append_native(...)` 作为 native graph node 追加。graph replay 只更新
device-side scalar，不会自动把结果读回 Python，也不会把检查结果转换成 graph 内部 host
控制流。

```python
workspace = ti.algorithms.MetricWorkspace(max_items=n)
err = ti.algorithms.max_abs_delta(values, reference, workspace=workspace)

builder = ti.graph.GraphBuilder()
builder.append_native(err)
graph = builder.compile()

graph.run({})
print(err.to_float())  # 显式读取 1 个 device scalar
```

这不等于向用户暴露任意 native callback；普通 Python 算法调用也不要求 graph 参与。
native graph node 是 JIT replay node；`ti.aot.Module.add_graph()` 当前只导出普通
kernel CGraph，会拒绝包含 Forge native node 的 graph。

## 与 vanilla Taichi 的关系

vanilla 兼容的 `parallel_sort()` 仍然保留。更广的 `sort()` dispatcher 和 `experimental_*`
primitive API 是 Forge 增量能力。
