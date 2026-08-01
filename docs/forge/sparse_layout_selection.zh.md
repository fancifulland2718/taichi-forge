# Taichi Forge 稀疏布局选择指南

> 适用于 Taichi Forge **0.6.0**。

[English version](sparse_layout_selection.en.md)

## 快速选择

先看拓扑生命周期和访问模式，不要只看“稀疏”两个字：

| 需求 | 优先路径 |
|---|---|
| 逻辑域大部分活跃 | dense field 或 dense ndarray |
| 可变coordinate grid、inactive read必须为0、需要struct-for | `pointer -> dense/bitmasked brick` |
| 插入必须在线完成，事前无法知道count | 有显式逻辑上界的`dynamic` SNode |
| 持久one-state-per-key随机lookup/update | 实验性bounded `hash` SNode |
| 每步按重复item keys重建 | sort/RLE/scan primitives；只有不关心桶内顺序时才用`experimental_bucket_builder` |
| fill前可得到contact/matrix row counts | device ndarray count-scan-fill |
| unique blocks构建一次、反复采样 | 应用自有sorted arrays；需要mutable container语义时选择SNode |
| 线性求解器反复apply固定拓扑 | CSR/BSR或matrix-free operator，不在每轮重复SNode/hash遍历 |

不存在一个万能稀疏容器默认值。CPU、CUDA、Vulkan可以共享逻辑语义，但allocator和
active-list/listgen物理实现并不相同。

## 先回答这些问题

### 这已经是线性算子了吗？

活动坐标一旦获得稳定DOF编号，迭代求解器就应消费紧凑CSR/BSR数组或matrix-free
stencil/operator。每轮Krylov迭代都重新pointer chasing、hash probe或sparse
struct-for listgen，会把拓扑成本留在错误的层次。

这同样适用于普通Poisson/pressure、隐式FEM、block elasticity、约束和其它物理引擎
线性系统，并非MPM专用建议。每个图节点天然带小型固定块时优先考虑BSR。mixed或
对称不定KKT必须选择支持该operator类别的solver，不能为了复用接口强塞进SPD CG。

### coordinate拓扑是否在线变化？

kernel必须按坐标直接写入、inactive read必须返回0、且算法需要struct-for时，使用
块化SNode：

```python
x = ti.field(ti.f32)
blocks = ti.root.pointer(ti.ij, (64, 64))
blocks.dense(ti.ij, (8, 8)).place(x)
```

优先选择浅层、每个active block内部有足够工作的brick。每个pointer cell只放一个
标量时，allocator/list metadata很容易超过payload。

只有插入确实必须在线完成、无法先count时才用`dynamic`。`dimension`是逻辑硬上界；
`chunk_size`是allocator geometry，不是expected-active合同。

### fill之前能否先count？

接触邻接、约束行、particle-cell list和稀疏矩阵装配通常可以：

1. 按row/key计数；
2. 对count做exclusive或inclusive scan得到offset；
3. 写入互不重叠的精确区间；
4. 发布versioned ndarray/CSR/BSR generation。

Forge提供scan、sort、RLE、compact和segmented consumer primitives，可用于这些流程。

重复key需要确定顺序时，先stable sort `(key, source ordinal)`再做RLE。
`experimental_bucket_builder`使用atomic bucket cursor，不保证桶内顺序；非法bin
按该API既有ignore合同处理。

### key是否持久并原位更新？

每个key持有长期状态且随机lookup/update重要时，可以使用实验性`hash` SNode：

```python
state = ti.field(ti.f32)
ti.root.hash(
    ti.i,
    dimensions=1 << 20,
    expected_active=20_000,
).place(state)
```

`expected_active`及其兼容别名`max_active`与load factor共同推导power-of-two
physical table capacity。`max_active`不是活跃entry硬上限。table运行时不会grow/
rehash；deactivate产生tombstone；overflow显式诊断；struct-for顺序未定义。

Hash SNode不是用户可定制collision/replacement policy的通用GPU hash table。算法特有
策略应放在应用自有arrays中。

### 拓扑是否冻结且以读取为主？

sorted unique block keys加连续brick payload可比低load-factor mutable hash table更
紧凑，适合SDF、冻结碰撞场和read-mostly coefficient bricks。代价是构建时排序，以及
A个active blocks下每次uncached lookup为`O(log A)`；同一brick内多次读取应缓存
block ordinal。

该表示应使用应用自有ndarray/kernel实现；需要mutable container语义时可使用
pointer/hash SNode。

## 容量与失败语义

| 名称 | 实际含义 |
|---|---|
| pointer `vk_max_active` | backend-specific兼容参数：Vulkan固定pointer容量、CUDA pool/list调尺输入、CPU traversal-list调尺输入 |
| dynamic `dimension` | 每个parent的逻辑最大length |
| dynamic `chunk_size` | 物理分配/寻址geometry |
| hash `expected_active` / `max_active` | 与load factor一起使用的调尺estimate |
| hash `capacity` | physical open-address table slots，并向上取2的幂 |
| ndarray capacity | 应用或generation builder拥有的显式item/byte上界 |

Vulkan pointer/dynamic超容地址会被安全钳制，并在下一个同步边界抛出诊断；hash
overflow同样显式。这些mutable SNode不是transactional snapshot：错误前已完成的合法
mutation可能仍存在。应clear/deactivate/rebuild，不能假设旧generation仍完整。

exact ndarray generation可以在publish前验证并在失败时保留旧代。provider缺失必须
显式unsupported；不得把拓扑payload经host复制作为silent fallback。

## 显存/内存记账

不要只用active payload估算显存。至少分开记录：

- logical domain与active cells/items；
- field或brick payload；
- state/key/offset/list metadata；
- allocator或fixed-pool reservation；
- Program共享sort/scan/listgen workspace；
- rebuild时并存的旧代和candidate；
- Graph/runtime cache已知下界与opaque driver state。

`dtype * shape` logical bytes有用，但allocator alignment、native plan对象或driver
cache未知时，它不是total owned memory。中高occupancy下dense可能更合适。

## Graph生命周期

ndarray generation应作为runtime Graph arguments传入，使同一compiled Graph可绑定
replacement generation，而不是把device地址固化进kernel定义。SNode Graph会记录
SNodeTree id/generation dependency；tree销毁后继续运行会被拒绝。

CUDA/Vulkan sparse struct-for在native Graph capture不可用时使用ordinary execution。
该路径保证正确性，但不提供native replay性能。

## 功能状态

| 路径 | Forge 0.6.0 状态 |
|---|---|
| dense、pointer、bitmasked、dynamic SNode | 可用 |
| Hash SNode | 实验性；默认开启并在首次使用时warning |
| sort/scan/RLE/compact/bucket primitives | 按各自stable或experimental名称提供 |
| CSR/BSR与matrix-free solver paths | 可用性取决于backend、format、dtype和solver capability |

## 避免这些错误

- 不要假设sparse一定比dense省内存。
- 不要依赖sparse struct-for或hash遍历顺序。
- 不要把`vk_max_active`称为backend-neutral硬上限。
- 不要把hash `max_active`称为活跃key硬上限。
- 自然能count-scan-fill时不要使用`dynamic`。
- 不要在每轮线性求解迭代中保留空间SNode遍历。
- 没有workload证据与overflow覆盖时，不要调整load factor、brick size或pool fraction。

## 相关文档

- [稀疏 runtime 与线性代数](sparse_runtime_and_linear_algebra.zh.md)
- [物理稀疏算子与求解器选择指南](physics_sparse_solver_selection.zh.md)
- [Hash SNode](hash_snode.zh.md)
- [Vulkan 稀疏 SNode](sparse_snode_on_vulkan.zh.md)
- [Forge API 参考](forge_api_reference.zh.md)
- [Forge options](forge_options.zh.md)
- [Native algorithms](native_algorithms.zh.md)
