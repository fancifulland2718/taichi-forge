# StructNdarray primitive 语义

本文说明 `StructNdarray` 与 Forge native primitive 的公开使用规则。目标是让结构化 AOS
数据可以作为低开销 payload 参与排序、压缩、分桶、拷贝等操作，同时让成员 view 在受支持的
数值 primitive 中作为 strided 数值数组使用。

该能力首次发布于 Forge 0.4.0；本文维护已发布的 0.6.2 发行合同，并作为
[Native 算法](native_algorithms.zh.md)的中文深入说明，不单独充当版本更新记录。

## 支持模式

### 整个 StructNdarray 作为 opaque payload

当 `StructNdarray` 作为 `values`、`src`、`dst`、`output` 等 payload 参与 order/copy 类
primitive 时，native 后端只按固定字节宽度移动完整结构元素，不解释字段语义。

适用入口：

- `ti.algorithms.sort(keys, values)`
- `ti.algorithms.sort_by_key(keys, values)`
- `ti.algorithms.experimental_compact(values, flags, output, count)`
- `ti.algorithms.experimental_gather(src, indices, dst)`
- `ti.algorithms.experimental_scatter(src, indices, dst)`
- `ti.algorithms.experimental_bucket_builder(keys, values, offsets, output)`

示例：

```python
payload = ti.types.struct(depth=ti.f32, color=ti.types.vector(3, ti.f32), idx=ti.i32)
keys = ti.ndarray(ti.i32, shape=n)
values = ti.ndarray(payload, shape=n)

ti.algorithms.sort(keys, values, method="auto")
```

排序只比较 `keys`，`values` 中每个结构化元素随对应 key 一起移动。

### 成员 view 作为数值数组

`StructNdarray.field(path, component=None)` 返回共享父 `StructNdarray` allocation 的
strided member view。它记录成员 byte offset 和 AOS stride，可在受支持 primitive 中作为数
值数组使用。

适用入口：

- `PrefixSumExecutor.run()`
- `ti.algorithms.experimental_transform()`
- `ti.algorithms.experimental_reduce()`
- `ti.algorithms.experimental_gather()`
- `ti.algorithms.experimental_scatter()`
- `ti.algorithms.experimental_scatter_add()`
- `ti.algorithms.experimental_grouped_reduce()`
- `ti.algorithms.experimental_histogram()` 的 scalar integer member path

示例：

```python
values = ti.ndarray(payload, shape=n)
red = values.field("color", component=0)

ti.algorithms.experimental_transform(red, red, scale=0.5, bias=0.0)
```

## Tensor member view

对 vector/matrix 成员，Forge 支持两类公开用法：

- 指定 `component=` 后，把单个 scalar lane 当作 strided scalar member view。
- 在支持 whole tensor member 的 primitive 中，把整个 vector/matrix member view 作为多 lane
  payload 或数值输入处理。

支持 whole tensor member 的路径包括 transform、reduce、gather、scatter、scatter-add、
grouped-reduce、sort values、compact values 和 bucket-builder values/output。具体是否走
packed native fast path取决于成员 lane 在 AOS 布局中是否连续以及后端能力。

## 后端与 fallback

- CPU、CUDA、Vulkan 都是目标后端。
- `method="auto"` 只在 layout 和 dtype 满足已知合同时选择 native 路径。
- 非连续、复杂或无法证明安全的 layout 会回退到 scalar-lane 或通用路径。
- 显式 native method 不支持时应清晰报错。

## 边界

- `StructNdarray` 的成员 view 不是普通 ndarray ABI；它是带 base、offset、stride 的 view。
- `field/SNode` 与 `StructNdarray` 是不同抽象。本文不声明任意 SNode 都可当作
  `StructNdarray` native payload。
- duplicate target 的 floating scatter-add 可能受后端 atomic 顺序影响。需要确定性顺序时，
  应使用先 reduce 再 merge 的方法或显式排序/分组方案。
- whole vector/matrix histogram 语义尚未定义；当前 histogram 只覆盖 scalar integer member。

## 推荐写法

热循环中应复用 workspace：

```python
workspace = None
for _ in range(num_steps):
    workspace = ti.algorithms.experimental_grouped_reduce(
        keys, values.field("mass"), output, method="auto", workspace=workspace
    )
```

这样可以复用 scratch buffer 和 native plan，避免每帧重复准备后端资源。
