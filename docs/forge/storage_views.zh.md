# 实验性 Dense Storage 零拷贝视图

Taichi Forge 可以把经过资格验证、由 runtime 持有的 dense storage 直接接入现有
`ti.types.ndarray(...)` kernel ABI，不分配第二份 buffer。该入口是显式的实验性 API：

```python
import taichi_forge as ti

ti.init(arch=ti.cuda)

positions = ti.Vector.field(3, dtype=ti.f32, shape=4096)
positions_view = ti.experimental.ndarray_view(positions)


@ti.kernel
def integrate(values: ti.types.ndarray(dtype=ti.types.vector(3, ti.f32), ndim=1)):
    for i in values:
        values[i].y -= 0.01


integrate(positions_view)
```

`ndarray_view()` 创建 non-owning metadata view。kernel 直接读写 source allocation；
view 创建和提交不会执行 pack、copy，也不会分配临时 storage。kernel 的正常顺序与后端
同步规则仍然有效。

## API

```python
ti.experimental.ndarray_view(
    source,
    *,
    slices=None,
    access="readwrite",
)
```

`source` 可以是 Forge `Ndarray`、另一个 `DenseNdarrayView`，也可以是经过资格验证的
root-dense scalar、Vector 或 Matrix field。返回对象可传给使用
`ti.types.ndarray(...)` 标注的 kernel 参数。

`slices` 可选地选择保持 rank 不变的正步长 subview。每个 logical index axis 对应一个
Python `slice`；rank-1 source 也可直接传入单个 `slice`。边界遵循 Python 的 slice
规范化规则。组合 view 时只合并 offset 与 stride，不分配也不复制：

```python
particles = ti.Vector.field(3, dtype=ti.f32, shape=8192)
even_particles = ti.experimental.ndarray_view(
    particles,
    slices=slice(0, None, 2),
)
```

integer indexing、axis insertion/permutation、负或零 step、broadcast stride 与可写 overlap
会被拒绝。切片不改变 source rank，也不改变 vector/matrix element shape。

当前可执行合同只接受 `access="readwrite"`。dtype、index rank 与 element shape 必须和
kernel annotation 匹配；该 view 不公开 gradient storage。

调用只会创建经过资格验证的 direct view 或抛出 `ValueError`；资格验证失败时绝不静默
退回 staging。

## 当前支持面

| Source 或用途 | CPU | CUDA | Vulkan | 行为 |
| --- | --- | --- | --- | --- |
| Contiguous Forge `Ndarray` | 支持 | 支持 | 支持 | 直接绑定已有 allocation |
| 合格的 root-dense scalar、Vector 或 Matrix field | 支持 | 支持 | 支持 | 直接绑定 SNode root allocation 与 byte offset |
| 保持 rank 的正步长 `slices` | 支持 | 支持 | 支持 | direct affine addressing，不 pack、copy 或分配临时 storage |
| Padded dense field | 资格验证 | 资格验证 | 资格验证 | element storage contiguous 且可证明 writable address 唯一时接受 |
| Bitmasked、pointer、dynamic、hash、bit-packed SNode | 不支持 | 不支持 | 不支持 | 不属于 dense affine view |
| 负 stride、broadcast、overlap、axis permutation 或 integer indexing | 不支持 | 不支持 | 不支持 | 需要不同的 read/scatter 合同 |
| StructNdarray member stride | 不支持 | 不支持 | 不支持 | 由共享 storage 模型描述，仅由理解 record stride 的 consumer 接受 |
| 使用 compact internal storage 的 Graph | Cached dispatch | CUDA Graph capture/replay | Command record/replay | submission 前重新验证 runtime owner 与 generation |
| 使用 positive affine view 的 Graph | Ordinary dispatch | Ordinary fallback | Command record/replay | 结果合同相同；CUDA capture 仍仅接受 compact mapping |
| ArgPack 嵌套 | 不支持 | 不支持 | 不支持 | backend submission 前拒绝 |
| 通过 view gradient 自动微分 | 不支持 | 不支持 | 不支持 | 不绑定 gradient owner |

“资格验证”表示 Forge 已证明 reachable byte range、native alignment、正 index stride、
contiguous element storage 与唯一 writable addressing。compact view 还必须满足 canonical
Ndarray ABI。仅凭 source class 不能保证一定接受。

## Consumer 特定的执行能力

多个 runtime consumer 共享同一 storage descriptor，但每个 consumer 都独立发布执行
capability。一个 descriptor 可被原生 strided algorithm 接受，不代表它必然可作为
Ndarray kernel 参数或 LinearOperator direct operand。

对于 `ti.linalg.experimental.LinearOperator.apply()`，canonical compact full field
只有同时满足以下条件时才绕过 scalar-vector staging：

- 显式传入 `out`，并使用 `alpha=1, beta=0` 的 overwrite 形式；
- input/output 是不互相 alias、dtype 与 scalar extent 精确匹配的 full field；
- 当前 provider 报告 `dense_storage_operands=True`。

当前 provider 矩阵如下：

| LinearOperator provider | CPU | CUDA | Vulkan |
| --- | --- | --- | --- |
| 已编译 Taichi kernel | Direct | Direct | Direct |
| Fixed native CSR/BSR | Direct | Direct | Device staged |
| 已编译 Graph | Device staged | Device staged | Device staged |
| `SolvePlan.solve()` vector 边界 | Device staged | Device staged | Device staged |

indexed view、padded/non-compact field、generalized apply coefficient 和 `out=None`
沿用 device-staging 路径，vector 数值不会经过 host。可以通过
`operator.capabilities.dense_storage_operands`、`vector_io_capabilities()`、
`VectorView.metadata` 和 `operator.statistics()["vector_io"]` 区分“可作为 direct
候选”与实际执行路径。

原生算法也使用同一 descriptor 获取 dtype、shape、owner、offset 与 record stride，
同时保留 provider 特定的 handle。对相同对象进行 warm plan replay 时，会直接复用 native
plan，不重新构造 descriptor。

## 生命周期与失败行为

view 会保持 Python source 存活；runtime 仍会在每次 submission 时重新验证带 generation
的 owner。SNode tree 已销毁、Ndarray 已 retire、Program generation 不同、layout
fingerprint 改变或 byte span 越界时，kernel enqueue 之前即失败。

GPU submission 会把底层 runtime resource 保留到已提交工作完成。CPU submission 受
native Ndarray 参数使用的同一个 Program transaction 保护。逻辑 view 不长期保存 raw
pointer。

使用时应保留以下边界：

- view 不是 owning tensor，也不会改变 source 原有的 indexing API；
- `copy=False` 语义是严格的：不支持的 storage 会失败，而不是复制；
- zero-copy 不等于不需要同步，consumer 仍须遵守 Taichi kernel 与 stream 的正常顺序；
- 外部框架 ownership、DLPack，以及负 stride、broadcast、overlap 或任意 element stride
  等一般 affine mapping 需要额外生命周期与同步合同，不能从本 API 推导得到。

## 何时使用

当可复用 kernel 接受 Ndarray argument model，而同一份 dense 数据由 Forge field 持有，
或保持 rank 的正步长 subset 需要继续 zero-copy 时，可以使用 `ndarray_view()`。数据本来
就是 `Ndarray` 且不需要 abstraction boundary 或 subview 时，继续直接传入即可。sparse、
permuted、overlapping 或其它不支持的 layout 应使用显式 pack/gather/scatter 工具；这些
操作具有不同 storage 语义，应保持可见。
