# StructNdarray 原生 primitive 语义

本文记录 `StructNdarray` 当前面向原生算法 primitive 的稳定语义。它的目标是把结构化
AOS 数据作为低开销 payload 参与排序、拷贝、压缩和分桶，同时允许标量成员作为数值数组
参与 scan/reduce/scatter/grouped-reduce 等计算。

## 两种支持模式

### 1. 整个 StructNdarray 作为 opaque raw payload

`StructNdarray` 的整元素内存布局是 AOS。原生后端只把它当作固定字节宽度 payload 移动，
不会解释成员语义。

适用 primitive：

- `ti.algorithms.sort(keys, values)`：`values` 可为 `StructNdarray`。
- `ti.algorithms.experimental_compact(values, flags, output, count)`：`values/output` 可为
  同类型 `StructNdarray`。
- `ti.algorithms.experimental_gather(src, indices, dst)` 和
  `ti.algorithms.experimental_scatter(src, indices, dst)`：`src/dst` 可为同类型
  `StructNdarray`。
- `ti.algorithms.experimental_bucket_builder(keys, values, offsets, output)`：`values/output`
  可为同类型 `StructNdarray`。

示例：

```python
payload = ti.types.struct(depth=ti.f32, color=ti.types.vector(3, ti.f32), idx=ti.i32)
keys = ti.ndarray(ti.i32, shape=n)
values = ti.ndarray(payload, shape=n)

ti.algorithms.sort(keys, values, method="auto")
```

这里排序只按 `keys` 比较，`values` 的每个结构化元素随 key 一起移动。

### 2. 标量成员 view 作为数值数组

`StructNdarray.field(path, component=None)` 返回一个 strided scalar member view。它共享父
`StructNdarray` 的 base allocation，内部记录成员 byte offset 和 AOS stride。

支持形式：

- 顶层标量：`arr.field("depth")`
- 嵌套标量：`arr.field("payload.mass")` 或 `arr.field(("payload", "mass"))`
- vector/matrix 的显式标量 component：`arr.field("color", component=1)` 或
  `arr.field("jacobian", component=(0, 2))`

适用 primitive 包括当前已显式接入 member view 的 scan/reduce/transform/scatter-add/
grouped-reduce 等数值算法。示例：

```python
particles = ti.ndarray(
    ti.types.struct(group=ti.i32, mass=ti.f32, color=ti.types.vector(3, ti.f32)),
    shape=n,
)
group_mass = ti.ndarray(ti.f32, shape=num_groups)

ti.algorithms.experimental_grouped_reduce(
    particles.field("group"),
    particles.field("mass"),
    group_mass,
    method="auto",
)

red_channel = particles.field("color", component=0)
ti.algorithms.experimental_transform(red_channel, red_channel, scale=0.5, bias=0.0)
```

## Host convenience API

以下 API 是 Python 侧调试和初始化便利接口，不是 hot path：

- `arr.to_numpy_fields("a", "payload.b")`：一次 readback 后返回字段名到 NumPy 数组的
  dict。
- `arr.from_numpy_fields({"a": a_np, ("payload", "b"): b_np})`：一次结构化数组 roundtrip
  后写回指定字段。
- `arr.debug_getitem(i)` / `arr.debug_setitem(i, value)`：用于单元素调试访问。计算密集
  路径应使用 primitive 或 kernel。

## 不支持或暂缓支持的内容

- member view 不能作为 `ti.kernel` 参数传入。要支持这一点需要在 kernel 参数 ABI 和
  external tensor lowering 中表达 `base + offset + i * stride`。
- 整个 `StructNdarray` 的 kernel-side 字段访问暂不支持，例如 `arr[i].x`。
- `arr.field("color")` 这种整个 vector/matrix member view 暂不作为数值数组；需要显式
  `component=...` 选择标量 lane。
- segmented grouped-reduce 的 member view 路径暂不默认启用。当前 atomic/native grouped
  reduce 已覆盖 member key/value/output；segmented 路径需要额外证明 strided key/value
  接入不会引入全尺寸 staging 或语义偏差。
- Field/SNode structured fast path 暂不属于 `StructNdarray` 的稳定语义。Field/SNode 仍使用
  Forge kernel 或现有 SNode 访问模型。

## 性能边界

- Opaque payload primitive 不解释成员，因此 payload 越宽，每次移动的字节数越多。
- Scalar member view 是 AOS strided 访问，通常比连续 scalar ndarray 更吃内存带宽；它的价值
  是避免拆字段 staging 和额外显存。
- Python convenience API 会 readback/writeback 整个结构化数组，不应放在每帧 hot path。
