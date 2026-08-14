# Hash SNode 使用指南

> `hash` SNode 首次发布于 **Taichi Forge 0.3.13**。本文说明已发布的 0.6.2 发行合同：它是实验性的固定容量稀疏 SNode，可在 CPU、CUDA、Vulkan 后端使用；默认开启，并会在第一次使用 `SNode.hash()` 时给出实验功能警告。

---

## 1. 功能变化

vanilla Taichi 1.7.4 将 `hash` SNode 保留在前端 gate 之后，并没有作为可用功能发布。Taichi Forge 将它恢复为一个受控的稀疏结构：

- 默认开启。第一次调用 `SNode.hash()` 会发出警告，因为该功能仍然是实验性的。
- 如果需要隔离回归，或复现 vanilla Taichi 中 hash 被禁用的行为，可以传入 `ti.init(hash_snode_experimental=False)` 显式关闭。
- 容量在 JIT 前固定。没有 device-side grow，也没有自动 rehash。
- overflow 会被诊断出来，而不是静默丢写。
- 当前支持 CPU、CUDA、Vulkan；其它后端会拒绝。
- 适合有界稀疏域中 pointer tree 过度浪费内存，或者显式 hash 结构更清晰的场景。

这不是旧 vanilla hash 合同。依赖 `ti.root.hash(axis, n)` 且不提供容量信息的代码需要迁移。

---

## 2. 基本 API

```python
import taichi_forge as ti

ti.init(arch=ti.cuda)

x = ti.field(ti.f32)

# 逻辑坐标域是 4096 * 4096。
# expected_active 表示预计活跃 hash entry 数。
ti.root.hash(ti.ij, (4096, 4096), expected_active=8192).place(x)
```

`SNode.hash()` 必须且只能传入下列参数之一：

| 参数 | 含义 |
|---|---|
| `expected_active=N` | 推荐写法。Table 大小由 `ceil(N / hash_load_factor)` 推出，再向上取到 2 的幂。 |
| `max_active=N` | `expected_active` 的兼容别名，会推导出相同 table slots，并不是活跃 entry 的硬上限。新代码建议使用 `expected_active`。 |
| `capacity=N` | 显式 table slot 数。会向上取到 2 的幂。 |

可选参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `hash_load_factor` | `ti.cfg.hash_snode_default_load_factor` (`0.5`) | 只在使用 `expected_active` / `max_active` 时生效。必须在 `(0, 1]` 内。 |

规则：

- 必须在 `expected_active`、`max_active`、`capacity` 中恰好选择一个。
- `expected_active` 及其 `max_active` 别名都是调尺输入；真正的物理硬边界是
  推导出的 2 的幂 table slot 数。
- 逻辑坐标域乘积必须落在 32-bit signed 范围内。
- 显式 `capacity` 可能被向上取整到 2 的幂。
- 如果 `expected_active` 大于逻辑域大小，SNode 仍合法，但会浪费 table slot。

---

## 3. 支持的拓扑

当前公开合同在 CPU、CUDA、Vulkan 上覆盖这些形态：

| 拓扑 | 示例 |
|---|---|
| root hash | `ti.root.hash(...).place(x)` |
| hash -> dense | `ti.root.hash(...).dense(...).place(x)` |
| hash -> bitmasked | `ti.root.hash(...).bitmasked(...).place(x)` |
| hash -> dynamic | `ti.root.hash(...).dynamic(...).place(x)`，支持 `ti.append` / `ti.length` |
| hash -> pointer | `ti.root.hash(...).pointer(...).place(x)` |
| nested hash | `outer = ti.root.hash(...); inner = outer.hash(...); inner.place(x)` |
| pointer -> hash | `ti.root.pointer(...).hash(...).place(x)` |
| dynamic -> hash | `ti.root.dynamic(...).hash(...).place(x)` |

不支持：

- 在 `quant_array` / `bit_struct` 下创建 hash。
- bit-level payload layout。
- 无界坐标域或 device-side rehash。

---

## 4. 示例

### 4.1 稀疏 2D field

```python
import taichi_forge as ti

ti.init(arch=ti.vulkan)

x = ti.field(ti.i32)
ti.root.hash(ti.ij, (8192, 8192), expected_active=20000).place(x)

@ti.kernel
def write():
    for n in range(20000):
        i = (n * 17) & 8191
        j = (n * 131) & 8191
        x[i, j] = n
```

### 4.2 Nested hash

```python
import taichi_forge as ti

ti.init(arch=ti.cuda)

value = ti.field(ti.i32)

outer = ti.root.hash(ti.i, 4096, expected_active=512)
inner = outer.hash(ti.j, 1024, expected_active=8)
inner.place(value)
```

当 parent block 和 child entry 都很稀疏时，这种结构比较合适。如果 outer active set 远小于 outer capacity，可以用 `hash_snode_compact_child_pool=True` 为 `hash -> hash` 降低 child container 预留内存。

### 4.3 Hash under pointer

```python
import taichi_forge as ti

ti.init(arch=ti.cuda)

x = ti.field(ti.f32)

blk = ti.root.pointer(ti.i, 1024)
h = blk.hash(ti.j, 4096, expected_active=16)
h.place(x)
```

当 parent 层天然是 chunked 结构，并且每个活跃 parent 下只有少量稀疏 entry 时，可以使用这种形态。

---

## 5. 调参开关

这些都是 `ti.init(...)` 关键字参数，也可按 `TI_<大写参数名>` 的环境变量形式设置，例如 `TI_HASH_SNODE_EXPERIMENTAL=0`。

| 参数 | 允许值 / 默认值 | 用途 | 风险与使用建议 |
|---|---|---|---|
| `hash_snode_experimental` | `True`，可设为 `False` | 启用 CPU / CUDA / Vulkan 上的公开 `SNode.hash()` 路径。第一次使用会提示实验功能警告。 | 正常 Forge 使用保持 `True`。只有在隔离回归、保留 vanilla 兼容禁用行为、或禁止生产代码误用时才设为 `False`。 |
| `hash_snode_default_load_factor` | `(0, 1]` 内的浮点数，默认 `0.5` | 使用 `expected_active` / `max_active` 且没有传 per-node `hash_load_factor` 时的默认负载因子。 | 较低值更占内存但通常降低 probe；较高值节省内存但增加碰撞、probe 和 overflow 风险。接近 `1.0` 前应先看 probe telemetry。 |
| `hash_snode_active_list` | bool，默认 `False` | 实验性 active bucket list，用于 hash 遍历优化。 | 会改变生成布局/代码，对 churn-heavy workload 可能退步。只建议在 focused benchmark 证明收益后启用。 |
| `hash_snode_diagnostics` | bool，默认 `False` | 启用额外运行时计数器，用于调试 probe / tombstone 行为。 | 诊断模式，不是生产性能默认项；会增加计数器流量和少量内存/运行时开销。 |
| `hash_snode_compact_child_pool` | bool，默认 `False` | `hash -> hash` / nested hash 的实验性内存模式。将 child hash container 放入紧凑 active-parent pool。 | 可降低 sparse nested hash 的预留内存，但会增加 parent bucket 到 child slot 的解析。仅在 nested-hash 内存占用主导且 benchmark 确认延迟可接受时启用。 |

推荐使用方式：

- 优先使用 `expected_active` 和默认 load factor。
- 实验性优化 flag 默认保持关闭，直到有针对当前 workload 的测试证据。
- 调容量或排查 overflow 时再打开 `hash_snode_diagnostics=True`。
- 只有 nested hash 且 outer active count 明显小于 outer capacity 时，再考虑 `hash_snode_compact_child_pool=True`。

---

## 6. 常见坑

### 6.1 固定容量

Hash SNode 不会在运行时增长。如果激活的 distinct key 超过 table 可承载范围，后端会报告 hash overflow。GPU 后端通常在 `ti.sync()` 或下一个同步边界观察到错误。

```python
ti.init(arch=ti.cuda)

x = ti.field(ti.i32)
ti.root.hash(ti.i, 1024, capacity=2).place(x)
```

上面的例子故意过小。实际使用时请改用 `expected_active` 或更大的 `capacity`。

### 6.2 Load factor 同时影响正确性风险和性能

较低 load factor 会使用更多内存，但 probe 链更短。较高 load factor 会省内存，但增加碰撞和 overflow 风险。`1.0` 只适合 key 集合非常明确，并且已经测过 collision/probe 行为的场景。

### 6.3 遍历顺序不稳定

对 hash SNode 使用 `for I in field:` 会访问所有活跃 cell，但访问顺序不是公开合同，可能随后端或 capacity 改变。浮点 reduce 不保证 byte-identical。需要稳定顺序时，请使用原子加或后处理排序。

### 6.4 Inactive read

读取 inactive cell 返回 dtype 的零值，与 LLVM 后端稀疏 SNode 行为一致。

### 6.5 Compact child pool 是内存优先

`hash_snode_compact_child_pool=True` 用于减少 nested hash 的预留内存。它会增加 parent bucket 到 child slot 的一次解析。目前 profile 不支持把它作为默认性能优化启用。

---

## 7. 从 vanilla Taichi 迁移

历史 vanilla hash 路径不是稳定公开功能。Forge 有意移除了在 GPU 上不安全的隐式行为：

| 旧行为 / 旧假设 | Forge 行为 |
|---|---|
| `ti.root.hash(axis, n)` 不带容量信息 | 拒绝。必须在 `expected_active`、`max_active`、`capacity` 中恰好传一个。 |
| 无界增长 | 不支持。容量在 JIT 前固定。 |
| overflow 静默发生 | 不支持。overflow 会诊断出来。 |
| 各后端 hash 行为分叉 | 避免。CPU、CUDA、Vulkan 共享同一容量与 overflow 合同。 |

Forge 从 0.3.13 起默认允许 `hash` API；需要 vanilla 兼容的拒绝行为时，显式传入 `ti.init(hash_snode_experimental=False)`。

---

## 8. 什么时候不该用 hash SNode

下列场景优先使用 `pointer` / `bitmasked` / `dynamic`：

- 坐标域天然有界且块状结构明显。
- 需要稳定高吞吐遍历。
- 实现 MPM/SPH voxel grid、OpenVDB-like brick、固定表 hash-grid encoding。

下列场景优先使用用户级 hash 到 dense table：

- Collision handling 是算法本身的一部分，例如 instant-NGP 风格 hash-grid encoding。
- 需要完全控制 bucket layout 和替换策略。

Hash SNode 应被视为实验性的稀疏存储工具，而不是通用 GPU hash map。

---

## 9. 参见

- 稀疏布局选择指南：[sparse_layout_selection.zh.md](sparse_layout_selection.zh.md)
- Forge 选项总览：[forge_options.zh.md](forge_options.zh.md)
- Vulkan 稀疏 SNode 指南：[sparse_snode_on_vulkan.zh.md](sparse_snode_on_vulkan.zh.md)
