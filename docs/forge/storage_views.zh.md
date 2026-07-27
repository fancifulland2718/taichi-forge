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
ti.experimental.ndarray_view(source, *, access="readwrite")
```

`source` 可以是 Forge `Ndarray`，也可以是经过资格验证的 root-dense scalar、Vector 或
Matrix field。返回对象可传给使用 `ti.types.ndarray(...)` 标注的 kernel 参数。

当前可执行合同只接受 `access="readwrite"`。dtype、index rank，以及 vector/matrix
element shape 必须与 kernel annotation 匹配。该 view 不公开 gradient storage。

调用只会创建 direct view 或抛出 `ValueError`；资格验证失败时绝不静默退回 staging。

## 当前支持面

| Source 或用途 | CPU | CUDA | Vulkan | 行为 |
| --- | --- | --- | --- | --- |
| Contiguous Forge `Ndarray` | 支持 | 支持 | 支持 | 直接绑定已有 allocation |
| Canonical root-dense scalar field | 支持 | 支持 | 支持 | 直接绑定 SNode root allocation 与 byte offset |
| Canonically packed root-dense Vector/Matrix field | 资格验证 | 资格验证 | 资格验证 | 仅在布局兼容 Ndarray ABI 时 direct |
| Padded 或 non-canonical dense field | 不支持 | 不支持 | 不支持 | 明确拒绝，不 materialize |
| Bitmasked、pointer、dynamic、hash、bit-packed SNode | 不支持 | 不支持 | 不支持 | 不属于 dense affine view |
| Indexed subset 或 permutation | 不支持 | 不支持 | 不支持 | 需要显式 indexed consumer |
| StructNdarray member stride | 不支持 | 不支持 | 不支持 | 留给 record-stride 执行阶段 |
| Graph capture/replay 或 ArgPack 嵌套 | 不支持 | 不支持 | 不支持 | backend submission 前拒绝 |
| 通过 view gradient 自动微分 | 不支持 | 不支持 | 不支持 | 不绑定 gradient owner |

“资格验证”表示 Forge 已证明 byte range compact、满足 native alignment、布局兼容
Ndarray ABI，并且 writable address mapping 唯一。仅凭 source class 不能保证一定接受。

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
- Graph capture、外部框架 ownership、DLPack 和一般 affine stride 需要额外生命周期与
  同步合同，不能从本 API 推导得到。

## 何时使用

当可复用 kernel 已接受 Ndarray ABI，而同一份 dense 数据由 Forge field 持有时，可使用
`ndarray_view()` 消除边界 staging。若数据本来就是 `Ndarray`，且不存在需要统一抽象的
consumer 边界，继续直接传入即可。indexed、sparse 或 non-canonical layout 应使用显式
pack/gather/scatter 工具；这些操作具有不同 storage 语义，应保持可见。
