---
sidebar_position: 5
---

# SIMT Intrinsics

For the CUDA backend, Taichi supports warp-level and block-level intrinsics, which
are needed for writing high-performance SIMT kernels. You can use them in Taichi
similar to the [usage in CUDA kernels](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/). Currently, the following functions are supported:


| Operation                  | Mapped CUDA intrinsic         |
| -------------------------- | ----------------------------- |
|`ti.simt.warp.all_nonzero`  | `__all_sync`      |
|`ti.simt.warp.any_nonzero`  | `__any_sync`      |
|`ti.simt.warp.unique`       | `__uni_sync`      |
|`ti.simt.warp.ballot`       | `__ballot_sync`   |
|`ti.simt.warp.shfl_sync_i32`| `__shfl_sync`     |
|`ti.simt.warp.shfl_sync_f32`| `__shfl_sync`     |
|`ti.simt.warp.shfl_up_i32`  | `__shfl_up_sync`  |
|`ti.simt.warp.shfl_up_f32`  | `__shfl_up_sync`  |
|`ti.simt.warp.shfl_down_i32`| `__shfl_down_sync`|
|`ti.simt.warp.shfl_down_f32`| `__shfl_down_sync`|
|`ti.simt.warp.shfl_xor_i32` | `__shfl_xor_sync` |
|`ti.simt.warp.match_any`    | `__match_any_sync`|
|`ti.simt.warp.match_all`    | `__match_all_sync`|
|`ti.simt.warp.active_mask`  | `__activemask`    |
|`ti.simt.warp.sync`         | `__syncwarp`      |

See [Taichi's API reference](https://docs.taichi-lang.org/api/taichi/lang/simt/warp/#module-taichi.lang.simt.warp)
for more information on each function.

Here is an example of performing data exchange within a warp in Taichi:


```python
a = ti.field(dtype=ti.i32, shape=32)

@ti.kernel
def foo():
    ti.loop_config(block_dim=32)
    for i in range(32):
        a[i] = ti.simt.warp.shfl_up_i32(ti.u32(0xFFFFFFFF), a[i], 1)

for i in range(32):
    a[i] = i * i

foo()

for i in range(1, 32):
    assert a[i] == (i - 1) * (i - 1)
```

## Vulkan subgroup exchange

After `ti.init(arch=ti.vulkan)`, query each operation with
`ti.simt.subgroup.is_shuffle_supported("shuffle_down")`. The query checks the
selected device's compute-stage subgroup capabilities. `shuffle` and
`shuffle_xor` require `VK_SUBGROUP_FEATURE_SHUFFLE_BIT`; `shuffle_up` and
`shuffle_down` require `VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT`. Unsupported
devices and backends fail during compilation, before pipeline submission.
For AOT export, also declare the corresponding target capabilities in
`ti.aot.Module(..., caps=["spirv_has_subgroup_shuffle",
"spirv_has_subgroup_shuffle_relative"])`; the current device does not implicitly
enable capabilities on the export target.

All four operations accept scalar `i32`, `u32`, or `f32` values. The lane index,
mask, or offset accepts a Python integer in `[0, 2**32 - 1]` or a scalar
`i32`/`u32` expression; signed expressions must be nonnegative at execution and
are converted to `u32`. Floating-point, tensor and wider integer expressions
are rejected rather than truncated. They preserve the value's type. The source lane is
respectively `index`, `invocation_id() ^ mask`, `invocation_id() - offset`, or
`invocation_id() + offset`. Use `group_size()` to obtain the actual subgroup
width; Vulkan does not promise CUDA's 32-lane width.

The source must be in range and active in the same dynamic instruction.
Results from inactive or out-of-range sources are undefined, including at the
edges of a partial subgroup. These operations do not clamp, wrap, or return the
calling lane's value at the boundary. Call the exchange before filtering out
boundary results so that lanes supplying valid values still participate.
These are the [SPIR-V group shuffle semantics](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpGroupNonUniformShuffle).

Vulkan 子组交换支持标量 `i32/u32/f32`，可通过 `is_shuffle_supported` 按操作查询当前设备。
索引、掩码和偏移接受 `[0, 2**32 - 1]` 内的 Python 整数或标量 `i32/u32` 表达式；
有符号表达式在执行时须非负，包装层统一转换为 `u32`，不截断浮点数、张量或宽整数。
子组宽度由 `group_size()` 给出；越界或未参与同一次指令的源 lane 结果未定义，不能按 CUDA
固定 32 lane 或边界返回自身值的假设使用。共享内存和 barrier 的块尺寸约束仍由调用方保证。
