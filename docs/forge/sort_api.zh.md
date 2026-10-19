# 并行排序 API

Forge native sort dispatcher 首次发布于 0.4.0。本文说明当前源码API 与后端选择
合同；兼容入口 `parallel_sort()` 早于 Forge 即已存在。

Taichi Forge 保留与标准 Taichi 兼容的入口：

```python
import taichi_forge as ti

ti.algorithms.parallel_sort(keys, values=None)
```

该入口仍使用 legacy odd-even merge sorter，用于兼容已有标准 Taichi 代码。

Forge 额外提供新的稳定排序 dispatcher：

```python
ti.algorithms.sort(
    keys,
    values=None,
    stable=True,
    method="auto",
    precision="exact",
    workspace=None,
    nan_policy="last",
)
```

这个 API 是 Forge 扩展，不是标准 Taichi 1.7.4 API。需要同时兼容标准 Taichi 的代码应继续使用 `ti.algorithms.parallel_sort()`。

## 默认后端选择

当 `method="auto"` 时：

- CUDA：默认使用 Forge 自有的 stable driver-only radix provider。该路径只依赖动态加载的
  CUDA Driver API，不要求 CUB、CUDART 或本机 CUDA Toolkit；
  `cuda_cub_split32` 永远不会被自动选择。
- Vulkan：对受支持的一维 dense key/payload 使用当前 native radix8 sorter。
- 其它组合：回退到 host stable sort。

显式方法：

- `method="cuda_device"`：强制标准 driver-only CUDA radix provider。
- `method="cuda_cub_native"` / `method="cuda_cub_split32"`：仅供开发期参考的弃用
  method。只有以 `TI_WITH_CUDA_TOOLKIT_PRIMITIVE_REFERENCE=ON` 构建时才可用，调用会产生
  `DeprecationWarning`，标准 runtime wheel 不包含这些 provider。
- `method="vulkan_native_radix_u32"`：强制当前 Vulkan radix8 路径，仅支持 32-bit ndarray key。
- `method="host_stable"`：强制 host stable fallback。
- `method="legacy"`：使用与标准兼容的 odd-even merge 实现。

## 快路径支持范围

CUDA driver-only fast path：

- Keys：`ti.u32`、`ti.i32`、`ti.f32`、`ti.u64`、`ti.i64`、`ti.f64`
- Values：可选标量数值、vector/tensor ndarray 或 4-byte 对齐的 StructNdarray raw payload
- 容器：一维 `ti.ndarray`，以及 capability 明确接受的 root-dense scalar field
- 实现：每 pass 4 bit 的 stable LSD radix；每 block 处理 1024 项，并分层扫描 16 路
  block histogram。嵌入 PTX 以 sm_50/PTX 4.0 为兼容目标，不包含 CUDA Toolkit header
  或 runtime call。

Vulkan native fast path：

- Keys：`ti.u32`、`ti.i32`、`ti.f32`、`ti.u64`、`ti.i64`、`ti.f64`
- Values：可选的受支持数值或 raw payload
- 容器：一维 `ti.ndarray`，以及 capability 明确接受的 root-dense field

未覆盖的组合会通过 host stable fallback 保持正确性；如果用户显式指定某个后端专用 method，则不满足条件时会报错。

## 预绑定排序

`ti.algorithms.prepare_sort(keys, values=None)` 固定存储绑定，可在 CUDA/Vulkan 上
重复直接执行，或通过 `plan.record()` 放入 Graph。key 为一维标量；等长载荷支持
标量、向量和矩阵，lane 类型为 i32/u32/f32/i64/u64/f64。紧凑 ndarray、dense Field
和紧凑 ndarray view 共用存储生命周期合同。整记录搬运保留 NaN payload、正负零等
全部位模式。准备阶段不排序、不提交设备工作；首次执行由既有后端 arena 提供 scratch。
`plan.report()` 的物理计划标识包含 `value_element_shape`。使用完毕关闭计划；Field
退休后相应绑定失效。

这些 recording 可在执行前或执行后导出 AOT，要求 `nan_policy="last"`。ndarray 使用
具名 source bindings，Field 使用 `snapshot_state=True`。既有 AOT sort 操作重建
相同记录，无需新增辅助 kernel。CPU 继续使用普通／sequence 或显式 AOT 排序。
此次扩展不改变独立的 `count=` 前缀排序合同，也不接纳 Struct／strided 载荷。
排序自身无自动伴随；后续编译的梯度可以使用排序结果。见 [AOT SDK](aot_sdk.zh.md)。

## 语义说明

- 默认语义是 stable ascending sort。`descending=True` 在 CPU native/host 路径可用；
  GPU `auto` 会回退到 host stable，显式 GPU native method 会在写入前拒绝。
- `nan_policy="last"` 是默认策略；`nan_policy="bitwise"` 需要后端 sortable-key 路径支持。
- `SortWorkspace` 可复用 backend scratch allocation，适合重复排序场景。
- CUDA driver-only sort 已通过所有公开 key dtype、NaN bitwise、重复 key/payload stability、
  dense field、两层 histogram 和多 host submitter 回归。它优先保证单一 wheel、旧 PTX
  兼容面和异步执行，并不声称达到 CUB 吞吐；当前统一性能证据见
  [Native 算法](native_algorithms.zh.md)。
