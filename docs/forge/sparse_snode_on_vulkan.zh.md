# Sparse SNode on Vulkan — 使用指南

> 适用范围：当前源码。参见[版本说明](index.zh.md#版本与安装)。

---

## 1. 速览

| 数据结构 | vanilla 1.7.4 Vulkan | 当前 Taichi Forge Vulkan | LLVM (cpu/cuda) |
|---|---|---|---|
| `dense` | ✅ | ✅ | ✅ |
| `bitmasked` | ❌ | ✅ | ✅ |
| `pointer` | ❌ | ✅ | ✅ |
| `dynamic` | ❌ | ✅ | ✅ |
| `hash` | ❌ | ⚠️ 实验功能，默认开启，首次使用警告（详见 §6） | ⚠️ 实验功能，默认开启，首次使用警告 |
| `quant_array` / `bit_struct` | ❌ | ⚠️ 实验性（详见 §7） | ✅ |

支持的 op：`activate` / `deactivate` / `is_active` / `length` / `append` / `ti.deactivate` / struct-for (`for I in field:`) / `ti.ndrange` 上的稀疏 listgen。

---

## 2. 启用方式

`pointer`、`bitmasked`、`dynamic`、`hash` 不需要额外开关——只要 `ti.init(arch=ti.vulkan)` 即可使用。`hash` 仍是实验路径，第一次使用会发出警告；如需禁用，传入 `hash_snode_experimental=False`。`quant_array` / `bit_struct` 仍是独立实验路径，需要显式开启。

```python
import taichi_forge as ti
ti.init(arch=ti.vulkan)

x = ti.field(ti.f32)
ti.root.pointer(ti.ij, 32).dense(ti.ij, 8).place(x)

@ti.kernel
def fill():
    for i, j in ti.ndrange(256, 256):
        if (i + j) % 17 == 0:
            x[i, j] = i * 0.1 + j * 0.01

fill()
```

可选构建期开关（默认全 ON，仅在排查回归时使用）：

| CMake 选项 | 默认 | 用途 |
|---|---|---|
| `TI_VULKAN_POINTER` | ON | 总开关：关闭后 pointer / bitmasked 在 Vulkan 上回退到 vanilla 行为（`TI_NOT_IMPLEMENTED`）。 |
| `TI_VULKAN_DYNAMIC` | ON | dynamic SNode 总开关。OFF 时 `dynamic` 在 Vulkan 上 `TI_NOT_IMPLEMENTED`。 |
| `TI_VULKAN_POINTER_POOL_FRACTION` | ON | 启用 `TI_VULKAN_POOL_FRACTION` 环境变量（见 §3.2）。OFF 时该 env var 完全被忽略，capacity 按最坏情况预留。 |

### 2.1 运行期 env var

| 环境变量 | 取值 | 默认 | 作用 |
|---|---|---|---|
| `TI_VULKAN_POOL_FRACTION` | `(0.0, 1.0]` | `1.0` | 缩减每个 pointer SNode 的物理 cell pool 容量 = `max(num_cells_per_container, round(total_cells × fraction))`。详见 §3.2。非法 / `≤0` / `>1` 自动回退 `1.0`。 |

---

## 3. 关键限制（**用前必读**）

### 3.1 Pointer / Dynamic 的 capacity 是**编译期静态**的

**与 LLVM 后端最大的语义差**：vanilla LLVM 把每个 cell 当成 `node_allocators` 上的动态 chunk，运行时按需扩容；Vulkan 后端**没有 device-side 动态分配器**，capacity 会在静态定长的 backing buffer 中预留。

后果：

- **超出静态 capacity 的 activate/append 会被拒绝，不会访问 pool 越界地址。**
- **下一个同步边界会抛出运行时错误**，标明 pointer/dynamic 类型、root/SNode id 与配置容量。pointer 应增大 `vk_max_active`（或取消池缩容），dynamic 应增大声明容量。

例：

```python
# pointer(N=32) 后跟 dense(M=8) → 物理 capacity = 32 cells。
# 超过 32 个独立 i 被 activate 时，会在同步边界报告 overflow。
ptr = ti.root.pointer(ti.i, 32)
blk = ptr.dense(ti.j, 8)
blk.place(x)
```

### 3.2 `TI_VULKAN_POOL_FRACTION` 缩减 capacity

如果你**确知** SNode 的稳态工作集远小于最坏情况（例如稀疏率 < 25%），可以缩减实际 pool 大小以减少 root buffer 与 GPU 内存占用：

```bash
# Linux / macOS
export TI_VULKAN_POOL_FRACTION=0.25
python your_app.py

# Windows PowerShell
$env:TI_VULKAN_POOL_FRACTION = '0.25'
python your_app.py
```

**何时使用**：

- 预设 N 远超实际激活数（例如 `pointer(ti.i, 4096)` 但每帧只激活 ~200 个）；
- 配合 deactivate-freelist（始终启用，`ti.deactivate` 自动归还 cell）；
- 性能/内存敏感的部署场景。

**何时不要用**：

- 你不确定稳态激活数；
- 测试 / 调试阶段；
- 单帧内有"先激活全集再 deactivate" 的工作流（峰值激活数 = 全集）。

**安全失败**：超过缩减后 capacity 的 activate 会被地址钳制，并在下一个同步边界报错（与 §3.1 同机制）。

### 3.3 Dynamic SNode 协议差异

Vulkan 上的 `dynamic` 使用 **flat-array + length 后缀** 协议：

- 容器布局：`[data: cell_stride × N][length: u32]`；
- `ti.append(field, [i], val)` = 有容量边界的原子 length 增长 + 写入 cell；
- `ti.length(field, [i])` = `OpAtomicLoad(length)`；
- `ti.deactivate(dynamic_node, [i])` = `OpAtomicStore(length, 0)`；
- 不支持 chunk 链；总容量 = 编译期静态 N。
- 超过 N 的 append 会被地址钳制，并在下一个同步边界报错。

数值结果与 LLVM 完全等价（已通过完整回归集验证）。

### 3.4 SPIR-V warp lockstep 限制

任何"基于 spin 等 winner 写完 slot"的协议（pointer / dynamic 的 race-to-activate）都会受 SPIR-V `OpLoopMerge` + GPU warp lockstep 影响。本 fork 的实现已在 NVIDIA / AMD / Intel iGPU 上验证稳定。如在新硬件上撞到 hang，请打开 issue 附 GPU 型号与 driver 版本。

### 3.5 不支持

- `quant_array` / `bit_struct` 之下创建 `hash`：详见 §6。
- `quant_array` / `bit_struct`：详见 §7。
- 跨多个 SNode tree 的 cross-tree pointer：与 LLVM 同步限制。
- `ti.deactivate` 在 ambient（root 直挂的 dense）上：与 vanilla 同——dense 不支持 deactivate。

### 3.6 Listgen 顺序非合同

LLVM 后端 struct-for（`for I in field:`）按 SNode 树拓扑序遍历活跃 cell；Vulkan 后端只保证**覆盖所有当前活跃 cell**，不保证遍历顺序。如果你的 reduce kernel 依赖确定遍历顺序（例如 `for i in field: result[0] += i.id`），不同后端的中间累加路径**可能给出不同浮点字节**（最终值在 1e-5 容差内等价，但不字节等价）。

规避：用 `ti.atomic_add` 或在 reduce 前先排序。

### 3.7 显式声明 Vulkan pointer 容量：`vk_max_active`

用于嵌套 pointer 或大 N 但稀疏度低的工作流。给定 `pointer(ti.ij, N)` SNode，默认按最坏情况预留 `N²` cell 到 root buffer；若你确知该层级的稳态最大活跃数远小于此，可显式提示：

```python
# Vulkan 后端：把该 pointer 的物理 pool 容量缩到 1024（而非默认 N²）
blk = ti.root.pointer(ti.ij, 1024, vk_max_active=1024)
blk.dense(ti.ij, 8).place(x)
```

规则：

- 这个kwarg有意保留backend-specific语义：它决定Vulkan固定pointer容量；在CUDA
  只指导sparse-pool sizing，并不是精确的per-SNode语义硬上限（推导出的有限CUDA
  pool仍可能显式耗尽）；CPU sparse payload仍按需分配，只用它选择traversal-list
  chunk geometry。
- 4 级 fallback 优先级：`vk_max_active` (kwarg) > `vulkan_pointer_pool_fraction`（`ti.init` kwarg）> `TI_VULKAN_POOL_FRACTION` (env var) > 最坏情况预留。
- 容量下限锁死 = 该 pointer 容器单 cell 数（`num_cells_per_container`），低于此值会被强制提升以保证 deactivate-freelist 至少能容纳一个 cell。
- 高于Vulkan推导worst-case的值会被保留而不是向下钳制；若对应buffer无法分配，
  runtime会显式报告allocation OOM。
- 超容时的行为：地址会安全钳制，并在下一个同步边界抛出诊断错误（与 §3.1 一致）。
- 嵌套 pointer 上每层独立标注：`outer = ti.root.pointer(ti.ij, OUTER, vk_max_active=N1); inner = outer.pointer(ti.ij, MID, vk_max_active=N2)`。

---

## 4. 验证实际工作负载

使用代表性输入检查激活/去激活、非活跃位置读取、峰值容量和遍历结果。
小示例成功不能证明完整应用的容量足够。排查错误结果或 device lost 时保留诊断信息。

## 5. 排错

| 现象 | 可能原因 | 处置 |
|---|---|---|
| `ti.sync()` 报告 pointer/dynamic capacity overflow | activate/append 超出配置的静态 capacity（§3.1 / §3.3）；越界地址已安全钳制 | 增大 `vk_max_active` / N，或减少并发 activate。不能把失败后的原位mutation当作已发布snapshot继续使用。 |
| 设置 `TI_VULKAN_POOL_FRACTION=0.5` 后 `ti.sync()` 报告 pointer pool overflow | 缩减后 capacity 不够（§3.2） | 调高 fraction，或恢复默认 1.0。 |
| `Hash SNode is experimental` 警告 | 第一次使用默认开启的实验性 `hash` API | 这是预期提示。阅读 §6 / [hash_snode.zh.md](hash_snode.zh.md)；如需禁用，传入 `ti.init(hash_snode_experimental=False)`。 |
| `ti.sync()` 附近报告 hash overflow | distinct hash key 超出固定 table 容量 | 增大 `expected_active` / `capacity`，或降低 `hash_load_factor`。 |
| Vulkan 上首次启动很慢，第二次秒开 | offline cache 首次编译 | 正常行为；第二次起命中 cache。 |
| `~/.cache/taichi/` 中 `ticache.tcb` 损坏 → 启动 | fallback 重编译路径已内建 | Forge 0.2.4+ 自动处理 `kVersionNotMatched` / `kCorrupted`，不抛异常。 |
| `hang` / `device lost` 在 race 测试 | warp lockstep（§3.4） | 提供 GPU 型号 + driver 版本到 issue。 |

---

## 6. 关于 `hash` SNode

`hash` SNode 可在 Vulkan 上作为**实验性固定容量稀疏 SNode**使用。它默认开启，第一次使用会提示警告：

```python
ti.init(arch=ti.vulkan)

x = ti.field(ti.f32)
ti.root.hash(ti.ij, (4096, 4096), expected_active=8192).place(x)
```

它和 `pointer` / `dynamic` 的关键差异：

- `SNode.hash()` 必须在 `expected_active`、`max_active`、`capacity` 中恰好传一个。
- table 容量在 JIT 前固定；没有 device-side grow 或 rehash。
- overflow 会报错，而不是静默丢写。
- struct-for 会访问所有活跃元素，但遍历顺序不是公开合同。

完整 API、支持拓扑、调参开关和迁移说明见：[hash_snode.zh.md](hash_snode.zh.md)。

---

## 7. 关于 `quant_array` / `bit_struct`

`quant_array`（位打包整数 / 定点字段）与 `bit_struct`（位打包复合结构）在 vanilla taichi 中**仅 LLVM 后端可用**。Forge 0.3.0 在 Vulkan 后端上提供**实验性 codegen**：

- 前端 extension 闸门需主动启用：`ti.init(arch=ti.vulkan, vulkan_quant_experimental=True)` 或 env var `TI_VULKAN_QUANT=1`（详见 [forge_options.zh.md](forge_options.zh.md) §3）。默认 OFF，行为与 vanilla 1.7.4 完全一致（quant 路径在 codegen 入口直接 `TI_ERROR`）。
- 闸门 ON 后已支持的能力：
  - **`quant_array`**：`QuantInt` / `QuantFixed` 子字段的**读 + 写（含多线程并发 `ti.atomic_add` 经 SPIR-V `OpAtomicCompareExchange` 自旋 RMW）**，与 cpu / cuda 后端**字节等价**；
  - **`bit_struct` / `BitpackedFields(max_num_bits=32 或 64)`**：多字段同字 RMW 写（IR pass `optimize_bit_struct_stores` 在 `quant_opt_atomic_demotion=ON` 默认下合并为单条 `BitStructStoreStmt`；`is_atomic == true` 残留路径用 CAS-loop 真原子写），与 cpu / cuda 后端**字节等价**。MPM 风格 11/11/10 quant_fixed 粒子位置打包基线 [tests/p4/g9_quant_baseline.py](../../tests/p4/g9_quant_baseline.py) 三后端 max_err 全为 9.77e-4 ≤ bound 1.95e-3；并发原子加 race 基线 [tests/p4/g9_quant_atomic_race.py](../../tests/p4/g9_quant_atomic_race.py) N=1024、K=64 路并发同字争用，三后端 max_err 全为 3.94e-3 ≤ bound 1.57e-2。
  - **原子加** `ti.atomic_add(quant_field, delta)`：仅 `AtomicOpType::add`，physical_type 须为 i32 或 i64（与 LLVM `quant_type_atomic` 限制对齐）。返回值为输入 `delta`（非旧值）——quant 字段上的 `atomic_add` 用户代码极少消费返回值，省去 dequant 旁路代价。
- **尚未支持**：
  - **`QuantFloat` 共享指数**（`ti.types.quant.float(...)` + `BitpackedFields(shared_exponent=True)`）：visitor 入口处 `TI_NOT_IMPLEMENTED`。**显式暂缓**：本 fork 本身未需求 shared-exponent（原始诉求是 quant_fixed），且该路径 float 位操作跨驱动微妙差异风险高。生产负载如需使用，请继续走 LLVM cpu / cuda 后端。
  - **非 add 的原子操作**（`atomic_min` / `max` / `bit_and` / `bit_or` / `bit_xor`）到量化字段：与 LLVM 后端一致地不支持（vanilla 设计决定，不是 Vulkan 后端的回退）。
- 未实现点会抛出带详细位置的 `TI_NOT_IMPLEMENTED` / `TI_ERROR`，不会静默误编译。

变通方案（闸门 OFF 或命中未实现 codegen 点时）：

- 用 `ti.f16` 半精度作为简易量化；
- 在 `ti.u32` 字段上手工位运算打包。

回归基线脚本：[tests/p4/g9_quant_baseline.py](../../tests/p4/g9_quant_baseline.py)（`bit_struct` MPM 风格 11/11/10 打包）、[tests/p4/g9_quant_array_baseline.py](../../tests/p4/g9_quant_array_baseline.py)（`quant_array` 8-bit 单字段）、[tests/p4/g9_quant_atomic_race.py](../../tests/p4/g9_quant_atomic_race.py)（atomic_add 多线程同字争用）。

---

## 8. CPU/CUDA 生命周期与内存

CPU/CUDA 与 Vulkan 的稀疏存储实现不同，不应跨后端照搬容量假设。
销毁树会使相关 Graph 和 kernel 资源绑定失效，替换树或 reset 后应重新建立这些对象。
诊断字段 `runtime_state_reserved_bytes` 已包含在 `root_reserved_bytes` 中，不能重复相加。

## 9. 兼容性与版本

- **API 兼容**：所有公开 Python API（`ti.root.pointer/.dense/.bitmasked/.dynamic/.place`、`ti.activate/.deactivate/.is_active/.length/.append`、`ti.root.deactivate_all` 等）行为与 vanilla 1.7.4 在 LLVM 后端上**严格一致**。Vulkan 上多出的 SNode 类型只新增可用性，不破坏现有用法。
- **Offline cache**：cache key 已纳入 SNode tree 结构 hash，pool fraction / dynamic 协议变更**自动**触发缓存失效。
- **C-API**：sparse SNode 的 root buffer 布局通过既有 `c_api/include/` 接口暴露，AOT 产物格式不变。
- **Wheel 二进制**：默认 build 已 ON 全部稀疏 SNode 后端能力（pointer / bitmasked / dynamic / quant 实验闸门）；用户无需额外编译参数。

---

## 10. 参考

- 稀疏布局选择指南：[sparse_layout_selection.zh.md](sparse_layout_selection.zh.md)
- 本 fork 新增编译/运行时/架构/现代化选项一览：[forge_options.zh.md](forge_options.zh.md)
- 测试代码：本仓库 `tests/p4/vulkan_*.py` 与 `tests/p4/g*.py`
