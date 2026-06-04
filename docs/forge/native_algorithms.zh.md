# Native 算法

本文说明 Forge 公开算法入口中可能走 CPU、CUDA 或 Vulkan native 实现的部分。对于
`method="auto"`，只有当 dtype、shape、layout 和后端能力都满足已知合同时才选择 native
路径；不支持的组合必须回退到正确的通用路径。显式指定 native method 时，不支持应清晰拒绝。

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

## 数据合同

- Dense 1D `ti.ndarray` 是主要 native 算法 ABI。
- Dense field/SNode 只有在能证明兼容 dense layout，或能提供安全 staging 路径时才走支持路径。
- `StructNdarray` 可作为 order/copy 类 primitive 的 opaque payload。部分数值 primitive 支持 scalar 或 packed tensor member view。
- 稀疏、非连续、复杂 SNode 拓扑不能默认假设走 native。
- duplicate target 的 floating scatter-add 可能受后端 atomic 顺序影响；只有数值合同允许时才应使用。

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
这不等于向用户暴露任意 native callback；普通 Python 算法调用也不要求 graph 参与。

## 与 vanilla Taichi 的关系

vanilla 兼容的 `parallel_sort()` 仍然保留。更广的 `sort()` dispatcher 和 `experimental_*`
primitive API 是 Forge 增量能力。
