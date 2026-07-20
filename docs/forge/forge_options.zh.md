# Taichi Forge — 编译 / 运行时 / 架构 / 现代化选项一览

> 适用于 **Taichi Forge 0.5.x** 发布线。除非特别说明，本文列出的选项均为**可选启用**；未显式启用的功能尽量保留 upstream Taichi 1.7.4 行为。
> 各选项首次公开版本见[版本更新说明](release_notes.zh.md)；本文描述当前合同，
> 不会把历史选项重新归类为 `0.5.0` 更新。

本文档是本 fork 公开新增配置项与工具链变更的**唯一权威清单**。应用只应暴露“受支持”
章节中的选项。第 2.9 节专门记录已删除、仅兼容保留或仅供验证的旧名称，帮助旧配置明确
失败或迁移；列入该节不代表建议使用。按模块整理的 Forge-only API 符号清单见
[Forge API 参考](forge_api_reference.zh.md)。

---

## 1. 新增 Python API（fork 专属）

| 符号 | 用途 |
|---|---|
| `ti.compile_kernels(kernels)` | 在热循环**之前**用线程池预编译一组 kernel。支持装饰器 kernel 或 `(kernel, args_tuple)` 对。返回提交数量。所有后端可用。 |
| `ti.compile_profile()` | Context manager。退出时输出每个 pass 的耗时报告，可选导出 CSV / Chrome trace JSON。开发期定位编译热点。 |
| `@ti.kernel(opt_level=...)` | 每个 kernel 单独覆盖 LLVM 优化级别。接受 `"fast"` / `"balanced"` / `"full"` 或 `0`–`3`。每个覆盖独立 cache key，混合 tier 批次互不污染。 |

### CLI

| 命令 | 用途 |
|---|---|
| `ti cache warmup script.py [-- script-args]` | 强制开 offline cache 跑一次 `script.py`，为后续冷启动写入 kernel 产物。需与最终运行同 arch / driver。 |

---

## 2. `ti.init(...)` / `CompileConfig` 关键字参数

未注明的默认值均与 vanilla 1.7.4 一致。

### 2.1 编译期 tier 选择

| 参数 | 默认 | 用途 |
|---|---|---|
| `compile_tier` | `"balanced"` | `"fast"` 强制 LLVM 使用 `-O0`（NVPTX / AMDGCN 下兜底 `-O1`），并以 level 0 跳过 `spirv-opt`；`"balanced"` 与 `"full"` 保留已配置的后端等级。应用层推荐只使用该选项。 |
| `llvm_opt_level` | `3` | 非 `fast` tier 下显式指定 LLVM `-O` 等级（`0`–`3`）；`compile_tier="fast"` 仍会强制后端安全的 O0/O1。除非代表性 benchmark 证明有必要，否则优先使用 `compile_tier`。 |

### 2.2 编译管线 / 线程

| 参数 | 默认 | 用途 |
|---|---|---|
| `num_compile_threads` | 逻辑核数 | `ti.compile_kernels` 使用的线程池大小。 |
| `compile_dag_scheduler` | `True` | `ti.compile_kernels` 批次的 DAG 反饱和调度器；平衡内 LLVM 线程池与外 kernel 池。`False` 回退两层调度。 |
| `spirv_parallel_codegen` | `False` | 启用每 kernel 的任务级并行 SPIR-V codegen。 |

### 2.3 Pass / IR 控制

| 参数 | 默认 | 用途 |
|---|---|---|
| `tiered_full_simplify` | `True` | 把 `full_simplify` 拆为局部 fixed-point + 每外圈一次 global pass，并有独立 cache key。建议保持 `True`；只有隔离编译器回归、对照旧 cadence 时才设 `False`。 |
| `unrolling_hard_limit` | `0`（关） | 每个 `ti.static(for ...)` 的 unroll 迭代上限；超出抛 `TaichiCompilationError`，避免静默吃编译时间。 |
| `unrolling_kernel_hard_limit` | `0`（关） | 单 kernel 内 unroll 总迭代上限。 |
| `func_inline_depth_limit` | 上游默认 | `@ti.func` 内联递归深度硬上限。 |
| `kernel_specialization_limit` | `1024` | 当前 Program generation 允许编译的 `@ti.kernel` specialization 总预算。达到正整数上限后，已有 specialization 继续运行，新的 cache miss 明确失败；只在应用确实需要且能够约束模板参数集合时调高。`ti.reset()` 会创建新 generation。 |
| `check_out_of_bound` | `False`；未指定且 `debug=True` 时隐式为 `True` | 在支持 assertion 的后端生成越界断言。显式传入 `check_out_of_bound=False`（或 `TI_CHECK_OUT_OF_BOUND=0`）现在会覆盖 debug 默认值，但不会关闭其它 debug 行为。 |

越界检查会改变生成代码，其最终有效布尔值进入 offline-cache key。只有在隔离检查成本，或
应用已经提供独立验证过的 bounds 合同时，才应显式关闭；此后非法索引将恢复为后端未定义
行为。CPU 与 CUDA 当前支持生成 assertion；Vulkan 尚未声明 assertion extension，因此会
警告并关闭该选项。

### 2.4 Real-function 与内联

| 参数 | 默认 | 用途 |
|---|---|---|
| `auto_real_function` | `False` | 实验性地将昂贵 `@ti.func` 单向提升为 `is_real_function=True`（LLVM-only、非 autodiff、有独立 cache key）。不要全引擎启用，也不要用于 AD/跨后端路径；仅在具体 workload 的编译与运行测试都通过后使用。 |
| `auto_real_function_threshold_us` | `1000` | 提升阈值（微秒，估算编译耗时）。 |

### 2.5 Vulkan quantization

| 参数 | 默认 | 用途 |
|---|---|---|
| `vulkan_quant_experimental` | `False` | **0.3.0 新增**。启用后 Vulkan 后端接受 `quant_array` / `bit_struct` 字段（即 `Extension::quant` / `Extension::quant_basic` 在 Vulkan 上可用）。已支持 `QuantInt` / `QuantFixed` 的读、写与多线程并发 `ti.atomic_add`（`OpAtomicCompareExchange` 自旋 RMW，`quant_array` 与 `BitpackedFields` / `bit_struct` 多字段同字均 OK），三后端字节等价。明确不支持：`QuantFloat` 共享指数、非 add 的原子操作（`atomic_min/max/and/or/xor`，与 LLVM 后端一致）。未实现路径会抛 `TI_NOT_IMPLEMENTED` / `TI_ERROR` 而非静默误编译。等价 env var：`TI_VULKAN_QUANT=1`。 |

### 2.6 CUDA sparse 内存池

Forge 默认从已物化的 SNode 树推导 CUDA sparse SNode pool，并在同一块 owning allocation 内为每个可分配 SNode 切出独立数据区。显式设置 `device_memory_fraction` 或 `cuda_sparse_pool_size_GB` 时仍切换到对应的固定预算路径。

| 参数 | 默认 | 用途 |
|---|---|---|
| `cuda_sparse_pool_size_GB` | `0.0`（无显式 override） | 显式 pool 大小（GiB）。设为正值则跳过其它调尺路径——需要固定 sparse-pool 预算时使用。 |
| `cuda_sparse_pool_auto_size` | `True` | 当 `device_memory_fraction == 0` 且 `cuda_sparse_pool_size_GB == 0` 时，按每个可分配 SNode 的全局 cell 上界、实际 `NodeManager` chunk 几何、list 元数据和有界 GC/重复激活余量推导 pool。`device_memory_GB` 只作 warn-only 合理性阈值，不会把推导结果静默截断到不足容量。 |
| `cuda_sparse_per_snode_pool` | `True` | 在自动调尺路径中，为每个可分配 SNode 切出独立数据区，同时保留共享的全局元数据/list 区；隔离嵌套 allocator 压力，但不会为每个 SNode 增加一次 CUDA allocation。 |
| `cuda_sparse_pool_size_floor_MiB` | `0` | 推导 pool 的可选用户下限（MiB）。全局元数据/list baseline 与 per-SNode chunk 预算始终计入，因此默认不再追加防御性 floor；绕过 auto-size 时无效。 |

`device_memory_fraction > 0` 与 `cuda_sparse_pool_size_GB > 0` 都会完全绕过自动调尺。`vk_max_active` 可降低单个 SNode 的 expected-active 上界；未给 hint 时使用该 SNode 从 root 展开的全局 cell 数，而不是单个父容器的大小。在 CPU/其他 LLVM 后端，该 hint 不限制按需增长的 sparse payload 容量，只用于选择下游 traversal element-list 的 chunk 几何。

### 2.7 Sparse struct-for / listgen 优化

两个 flag 均默认关闭，关闭时产出与 legacy 路径字节一致。启用后会改变 kernel 代码（CUDA grid_dim 或 SPIR-V 原子操作），变化已纳入 offline cache hash。

| 参数 | 默认 | 用途 |
|---|---|---|
| `spirv_listgen_subgroup_ballot` | `False` | 仅 Vulkan/SPIR-V。在 listgen kernel 内将逐线程 `OpAtomicIAdd` 聚合为每活跃 subgroup 一次 ballot 原子操作，降低 dense-active sparse struct-for 的原子争用。需设备支持 subgroup ballot（标准 SPIR-V 能力，Vulkan adapter 上报），否则该 flag 无效。 |
| `listgen_static_grid_dim` | `False` | 仅 CUDA / AMDGPU。sparse-listgen kernel 使用静态上限推出的 `grid_dim`（= 被列 SNode 严格祖先的 `num_cells_per_container` 乘积，不含 root），并以硬件饱和值封顶。消除浅稀疏树上的空闲 block。Vulkan 已通过 task attribs 计算等价量，该 flag 在 SPIR-V 后端为空操作。正确性由 `element_listgen_nonroot` 现有 grid-stride 循环保证。 |

### 2.8 Hash SNode

`hash` SNode 是实验功能，从 Taichi Forge 0.3.13 起默认开启。它可在 CPU、CUDA、Vulkan 上使用，第一次调用 `SNode.hash()` 会提示实验功能警告；如果需要复现 vanilla 兼容的拒绝行为，或隔离回归，可以传入 `hash_snode_experimental=False` 关闭。API 与迁移说明见 [hash_snode.zh.md](hash_snode.zh.md)。

| 参数 | 允许值 / 默认值 | 用途 | 风险与建议 |
|---|---|---|---|
| `hash_snode_experimental` | bool，默认 `True`；可设为 `False` | 启用 CPU / CUDA / Vulkan 上的 `SNode.hash()`。第一次使用会提示实验功能警告。 | 正常 Forge 使用保持 `True`。只在隔离回归、复现 vanilla 兼容拒绝行为、或避免生产代码误用时设为 `False`。环境变量别名：`TI_HASH_SNODE_EXPERIMENTAL=0/1`。 |
| `hash_snode_default_load_factor` | `(0, 1]` 内浮点数，默认 `0.5` | 使用 `SNode.hash(..., expected_active=N)` 或 `max_active=N` 且没有传 per-node `hash_load_factor` 时的默认负载因子。 | 较低值会保留更多内存并通常缩短 probe；较高值节省内存但增加碰撞、probe 成本和 overflow 风险。环境变量别名：`TI_HASH_SNODE_DEFAULT_LOAD_FACTOR`。 |
| `hash_snode_active_list` | bool，默认 `False` | 实验性 active bucket list，用于 hash 遍历优化。 | 会改变生成布局/代码，对 churn-heavy workload 可能退步。只建议在 focused benchmark 证明收益后启用。环境变量别名：`TI_HASH_SNODE_ACTIVE_LIST=0/1`。 |
| `hash_snode_diagnostics` | bool，默认 `False` | 启用额外运行时计数器，用于调试 probe / tombstone 行为。 | 诊断模式，不是生产性能默认项；会增加计数器流量和少量内存/运行时开销。环境变量别名：`TI_HASH_SNODE_DIAGNOSTICS=0/1`。 |
| `hash_snode_compact_child_pool` | bool，默认 `False` | `hash -> hash` / nested hash 的实验性内存模式。当 parent active count 远小于 parent capacity 时，可减少 child-container 预留内存。 | 会增加 parent bucket 到 child slot 的解析，可能以延迟换内存。只在 nested hash 内存占用主导且 benchmark 支持时启用。环境变量别名：`TI_HASH_SNODE_COMPACT_CHILD_POOL=0/1`。 |

### 2.9 已删除、仅兼容保留与仅供验证的设置

以下名称只用于说明旧配置和历史实验，不应由应用或引擎暴露。

| 名称 | 当前行为 | 必须采取的操作 |
|---|---|---|
| `use_fused_passes`、`fused_pass_verify` | 已在公开 0.4.23 基线前随低 ROI 的 `pipeline_dirty` 实验一起物理删除；当前 wheel 会把它们作为未知 `ti.init` 参数拒绝。 | 从配置中删除；稳定管线会始终执行必要的 simplify 路径。 |
| `spv_opt_level` | 不是当前 Python/CompileConfig 字段，wheel 会直接拒绝；实现中的低层字段名是 `external_optimization_level`。 | 使用 `compile_tier`，不要在应用代码中把旧名称机械改成低层字段。 |
| `skip-loop-unroll`、`skip_loop_unroll` | 不是可接受的 `ti.init` 名称；当前原始实验字段是 `spirv_skip_loop_unroll`。 | 删除，不要转换成引擎配置。 |
| `vulkan_listgen_lite_barrier` | 仅作为 deprecated no-op 兼容字段被接受；当前窄 barrier 路径属于 `vulkan_dispatch_cache`。 | 删除；调整该值没有受支持的效果。 |
| `vulkan_launch_buffer_pool`、`vulkan_launch_buffer_pool_capacity` | 被接受但已是 deprecated no-op。旧独立 pool 因 ROI 很低而移除，并由 fence-safe GFX context 处理替代。 | 删除，不要调整 capacity。 |

以下字段仍为编译器实验保留，但公开命名、cache 合同或跨 driver 证据不足以支持生产配置：

| 名称 | 当前实现合同 | 生产建议 |
|---|---|---|
| `external_optimization_level` | 原始 SPIR-V optimizer 等级，默认 `3`，进入 offline-cache key；`compile_tier="fast"` 会覆盖为 level `0`。 | 应用保持使用 `compile_tier`；GeoPhys 或其它引擎不要暴露该字段。 |
| `spirv_disabled_passes` | 默认 `[]`；会改变 SPIR-V，并使用排序后的独立 cache key。当前内部 pass ID 区分大小写，例如 `LoopUnroll`，但该词表不是稳定公开 API。 | 在命名与跨 driver 验证完成前保持空列表。 |
| `spirv_skip_loop_unroll` | 默认 `False`；会改变 optimizer chain 与 SPIR-V，但当前没有进入 offline-cache key。 | 保持 `False`；不要暴露，也不要用于生产/offline-cache workload。 |
| `spirv_adaptive_opt`、`spirv_adaptive_opt_threshold` | 默认 `False` / `64`，有独立 cache key，但会按 task 形态改变 optimizer chain。 | 仅供验证和 benchmark，等待 driver matrix 收敛。 |
| `cache_loop_invariant_global_vars` | 默认 `False`；只对窄 workload 改变 IR，当前没有进入 offline-cache key。历史测量显示 cold-compile 成本明显，而物理运行收益有限。 | 保持 `False`，不要作为通用性能开关暴露。 |

---

## 3. 环境变量

| 变量 | 取值 | 默认 | 用途 |
|---|---|---|---|
| `TI_VULKAN_POOL_FRACTION` | `(0.0, 1.0]` | `1.0` | 缩减每个 `pointer` SNode 的物理 cell pool 到 `max(num_cells_per_container, round(total × fraction))`。越界地址会被安全钳制，并在下一个同步边界抛出诊断。非法 / `≤ 0` / `> 1` 回退 `1.0`。详细语义见 [sparse_snode_on_vulkan.zh.md](sparse_snode_on_vulkan.zh.md)。 |
| `TI_VULKAN_QUANT` | `0` / `1` | `0` | **0.3.0 新增**。等价于 `ti.init(arch=ti.vulkan, vulkan_quant_experimental=True)`。开启后 `quant_array` 与 `BitpackedFields` / `bit_struct` 的读、写、`ti.atomic_add` 均可用。`QuantFloat` 共享指数、非 add 原子明确不支持。OFF 时行为与 vanilla 1.7.4 相同。 |
| `TI_KERNEL_PROFILER_MAX_RECORDS` | `1`–`1048576` | `131072` | kernel profiler raw record 的进程内预算；达到上限时明确报错，不继续增长。长期 profiling 应定期调用 `ti.profiler.clear_kernel_profiler_info()`；只在确认 host-memory 预算后调高。 |

> 上游 taichi 已有的环境变量（`TI_ARCH` / `TI_DEVICE_MEMORY_GB` 等）保持原行为，不在此重列。

---

## 4. CMake 构建选项（开发者向）

> 仅在从源码构建 Forge 时暴露。安装发布 wheel 的最终用户无需关心；所有默认 ON 路径已编入。

| 选项 | 默认 | 用途 |
|---|---|---|
| `TI_VULKAN_POINTER` | ON | Vulkan 上 `pointer` / `bitmasked` SNode 总开关。OFF 回退 vanilla 的 `TI_NOT_IMPLEMENTED`。 |
| `TI_VULKAN_DYNAMIC` | ON | Vulkan 上 `dynamic` SNode 总开关。OFF 回退 `TI_NOT_IMPLEMENTED`。 |
| `TI_VULKAN_POINTER_POOL_FRACTION` | ON | 启用 `TI_VULKAN_POOL_FRACTION`。OFF 时该 env var 完全失效，按最坏情况预留 capacity。 |

发行 wheel 构建会启用以上三项。

---

## 5. SNode 覆盖度扩展

| SNode 类型 | vanilla 1.7.4 Vulkan | Taichi Forge 0.5.x Vulkan |
|---|---|---|
| `dense` | ✅ | ✅ |
| `bitmasked` | ❌ | ✅ |
| `pointer` | ❌ | ✅ |
| `dynamic` | ❌ | ✅ |
| `hash` | ❌ | ⚠️ 实验功能，默认开启，首次使用警告 |

Vulkan 稀疏 SNode 用法与语义见 [sparse_snode_on_vulkan.zh.md](sparse_snode_on_vulkan.zh.md)。Hash SNode API 见 [hash_snode.zh.md](hash_snode.zh.md)。

---

## 6. 工具链与依赖升级

Forge 同步至现代工具链；下表对比 vanilla 1.7.4。

| 组件 | vanilla 1.7.4 | Forge 0.5.x |
|---|---|---|
| LLVM | 15 | **20.1.7** |
| Python | 3.7 – 3.12 | **3.10 – 3.14** |
| Windows MSVC | VS 2019 / 2022 | **VS 2026（MSVC 14.50+）** |
| `spdlog` | 1.14.1 | **1.15.3** |
| `Vulkan-Headers` / `volk` / `SPIRV-Headers` / `SPIRV-Tools` | 较旧 | 对齐至 **Vulkan SDK 1.4.341** |
| `googletest` | 1.10.0 | **1.17.0** |
| `glm` | 0.9.9.8 + 187 | **1.0.3** |
| `imgui` | v1.84（WIP） | **v1.91.9b**（non-docking 分支） |

Vulkan ImGui 后端已迁移到新的 `ImGui_ImplVulkan_InitInfo` 布局（`RenderPass` + `ApiVersion` 字段、自管理 font texture、`LoadFunctions(api_version, loader)` 签名）。GGUI 视觉回归 90 / 90 通过（Vulkan + CUDA）。

---

## 7. 架构 / 健壮性改进

以下改进默认启用，**不可调**；列出仅供调试参考。

- **Offline cache 跨版本兼容**：损坏或版本不匹配的 `ticache.tcb` 自动 fallback 重编译，不崩溃。
- **`rhi_cache.bin` 原子写入**：write-then-rename 避免崩溃中断时留下半写文件。
- **LLVM CPU 路径单 offload 旁路**：移除前期 batch-compile 工作引入的 0.89× CPU 回归。
- **类型上下文防御性 assert**：`linking_context_data->llvm_context` 上的 forbidden-zone 注释 + assert，提前捕捉跨上下文类型查询。

---

## 8. 兼容性声明

- 已支持的上游 Taichi 1.7.4 Python API 是兼容性参考；文档明确列出的 Forge 改动和实验路径
  属于显式例外。
- 第 1–2.8 节中的受支持 fork 选项均为 additive；除非条目明确说明，默认值保留上游
  行为。第 2.9 节是迁移/诊断登记表，不是受支持选项清单。
- PyPI 包使用 `taichi_forge` 导入，不会替换上游 `taichi` 包。PyPI shim wheel 不包含
  C API package tree；需要 C API 时必须单独构建。

---

## 9. 另见

- Sparse SNode on Vulkan 使用指南：[sparse_snode_on_vulkan.zh.md](sparse_snode_on_vulkan.zh.md)
- 稀疏布局选择指南：[sparse_layout_selection.zh.md](sparse_layout_selection.zh.md)
- Hash SNode 使用指南：[hash_snode.zh.md](hash_snode.zh.md)
