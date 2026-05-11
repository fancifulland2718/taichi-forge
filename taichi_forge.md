# Taichi Forge 功能变动摘要

本文只记录对外可见的功能变化。具体 API、限制、flag 风险和示例请阅读 `docs/forge/` 下的使用指南。

## 稀疏 SNode

### Vulkan sparse SNode

Taichi Forge 在 Vulkan/SPIR-V 后端补齐了 vanilla Taichi 1.7.4 缺失的稀疏 SNode 路径：

- `pointer`
- `bitmasked`
- `dynamic`

这些类型在 Vulkan 上可直接使用：

```python
ti.init(arch=ti.vulkan)
ti.root.pointer(ti.ij, 32).dense(ti.ij, 8).place(x)
```

主要差异：

- Vulkan 没有 device-side dynamic allocator，`pointer` / `dynamic` 使用静态容量模型。
- `dynamic` 在 Vulkan 上使用 flat-array + length 后缀协议。
- `vk_max_active` 可用于显式收缩 Vulkan pointer pool 容量。

使用指南：`docs/forge/sparse_snode_on_vulkan.zh.md`

### Hash SNode

Taichi Forge 新增实验性的固定容量 `hash` SNode，支持 CPU、CUDA、Vulkan。该 API 默认开启，第一次使用会发出实验功能警告：

```python
ti.init(arch=ti.cuda)
ti.root.hash(ti.ij, (4096, 4096), expected_active=8192).place(x)
```

重要变化：

- 默认开启；需要禁用时传 `ti.init(hash_snode_experimental=False)`。
- `SNode.hash()` 必须传 `expected_active`、`max_active`、`capacity` 之一。
- 不支持运行时 grow / rehash。
- overflow 会被诊断，不会作为默认行为静默丢写。
- 支持 root hash、hash child sparse、nested hash、pointer/dynamic under hash，以及 pointer/dynamic parent 下的 hash。

使用指南：`docs/forge/hash_snode.zh.md`

## Vulkan quant 实验路径

Vulkan 后端提供实验性 `quant_array` / `bit_struct` codegen：

- 默认关闭。
- 使用 `ti.init(arch=ti.vulkan, vulkan_quant_experimental=True)` 或 `TI_VULKAN_QUANT=1` 开启。
- 支持 `QuantInt` / `QuantFixed` 读写和 `ti.atomic_add`。
- 不支持 `QuantFloat` shared exponent，也不支持非 add 原子操作。

## 编译与运行时选项

Forge 增加了若干编译、缓存和运行时调优选项：

- `compile_tier`
- `ti.compile_kernels(...)`
- `ti.compile_profile()`
- `@ti.kernel(opt_level=...)`
- `spirv_disabled_passes`
- `spirv_listgen_subgroup_ballot`
- `listgen_static_grid_dim`
- CUDA sparse pool sizing 相关参数
- hash SNode 相关参数

完整列表：`docs/forge/forge_options.zh.md`

## 兼容性说明

- 默认导入方式仍是 `import taichi as ti`。
- `hash` SNode 是新增实验路径中的例外：Forge 0.3.13 默认允许 API 使用，但保留 `hash_snode_experimental=False` 作为显式禁用开关。
- Forge 的 `hash` 合同是固定容量、JIT 前确定容量、overflow 诊断，不复刻历史 vanilla 中未完成的动态 hash 语义。
- 对外文档只描述 API、限制、迁移和使用方式；实现细节与优化实验记录保留在内部规划文档中。
