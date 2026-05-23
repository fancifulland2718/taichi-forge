# S4 Dense Field Native 路径性能对比

日期：2026-05-22；更新：2026-05-23

本文记录 `taichi-forge 0.4.0` 在 dense 1D scalar field 上的 native
`scan`、`reduce(sum)`、`transform_affine(dst = src * 3 + 7)` 实测结果，并与
本机安装的 vanilla Taichi 1.8.0 对比。当前包名为 `taichi_forge`，源码目录结构
与 Taichi 基本一致。

## 测试口径

- Forge：`taichi-forge 0.4.0`，LLVM 20.1.7，通过
  `PYTHONPATH=D:\taichi\python` 加载当前工作区构建产物。
- Vanilla：本机 `taichi 1.8.0`。
- Python：CPython 3.10.11。
- 数据：`ti.field(ti.i32, shape=N)`。
- 规模：`N = 1024` 用于观察固定开销，`N = 1048576` 用于观察规模开销。
- 每个组合使用独立子进程冷启动，`offline_cache=False`。
- `first_call_ms` 记录第一次 primitive 调用到同步完成的耗时，包含 native
  pipeline/module 创建和首次 dispatch，不是纯 compiler phase。
- `runtime_median_ms` 是 warmup 后重复调用的 median，调用后显式 `ti.sync()`。
- `workspace_peak_bytes` 是 Forge workspace 对象记录的显式临时空间峰值；vanilla
  custom kernel 没有同类统计，因此为 0。
- `gpu_peak_delta_mb` 来自 Windows GPU Process Memory 的 Dedicated Usage，只作为
  粗粒度显存占用参考。

关键结果文件：

- 全后端最终矩阵：
  `benchmarks/results/s4_dense_field_native_replay_final_20260522/summary.csv`
- CPU repeat30 稳定复测：
  `benchmarks/results/s4_dense_field_native_cpu_runtime_repeat30_20260522/summary.csv`
- CUDA repeat30 稳定复测：
  `benchmarks/results/s4_dense_field_native_cuda_runtime_repeat30_20260522/summary.csv`
- Vulkan repeat30 稳定复测：
  `benchmarks/results/s4_dense_field_native_vulkan_runtime_repeat30_20260522/summary.csv`
- NativePrimitivePlan 清理与 StructNdarray tensor member 复测：
  `benchmarks/results/s4_native_plan_replay_clean_struct_tensor_20260523/summary.csv`
- dense field 清理后对 vanilla 1.8.0 复测：
  `benchmarks/results/s4_dense_field_native_clean_plan_20260523/summary.csv`

## 本轮 S4 更新

本轮继续保持 field/SNode API 不变，优化发生在 descriptor、workspace replay 和
backend native path 内部：

- `ReduceWorkspace` 的 replay 从 dense field 专用逻辑收敛到
  `_NativePrimitivePlan`。重复调用同一组 field/ndarray/StructNdarray scalar
  member 时，直接 replay 到已经确认可用的 native C++ entrypoint。
- `TransformWorkspace` 同样使用 `_NativePrimitivePlan`；StructNdarray whole
  vector/matrix member transform replay 到 packed strided native call。
- `PrefixSumExecutor` 使用 executor-local native scan plan。重复扫描同一 native
  dense 对象时跳过 Python view/proof 分发。
- CPU dense field native path 增加 contiguous fast path：当 field cell stride
  等于元素大小时，scan/reduce/transform 直接使用 contiguous typed loop。
- CPU reduce 的 `op=sum` 增加 typed range-sum helper，避免每个元素走通用
  `cpu_reduce_combine()` 的 signed wrapping `memcpy` 组合路径。
- Vulkan 继续保留前一轮优化：scan 默认关闭高冷启动成本的 subgroup path；
  i32 reduce 使用一阶段 atomic shader；i32/u32 transform 使用 push constants；
  reduce/transform resource set 和输出 barrier 范围已收窄。

## CPU 结果

CPU 使用 repeat30 复测。vanilla Taichi 1.8.0 的 `PrefixSumExecutor` 不支持 CPU
scan，因此 CPU scan 只记录 Forge 结果。

| op | N | Forge first ms | Vanilla first ms | Forge warm ms | Vanilla warm ms | Forge workspace bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scan | 1024 | 0.398 | N/A | 0.015 | N/A | 0 |
| scan | 1048576 | 1.148 | N/A | 0.727 | N/A | 0 |
| reduce | 1024 | 0.459 | 42.424 | 0.014 | 0.063 | 0 |
| reduce | 1048576 | 1.141 | 42.965 | 0.108 | 0.133 | 128 |
| transform | 1024 | 0.995 | 32.077 | 0.015 | 0.062 | 0 |
| transform | 1048576 | 1.306 | 39.724 | 0.141 | 0.186 | 0 |

结论：

- CPU reduce/transform 的 first-call 从 vanilla 的 32-43 ms 降到 0.5-1.3 ms。
- CPU reduce/transform 的 warm runtime 在 repeat30 下也优于 vanilla。
- CPU scan 当前是 Forge-only native 能力，first-call 和 warm runtime 都不生成
  Python helper IR。

## CUDA 结果

CUDA 使用 repeat30 复测。

| op | N | Forge first ms | Vanilla first ms | Forge warm ms | Vanilla warm ms | Forge workspace bytes | Forge/Vanilla GPU delta MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| scan | 1024 | 10.102 | 396.766 | 0.028 | 0.340 | 0 | 793.660 / 791.660 |
| scan | 1048576 | 9.745 | 362.089 | 0.041 | 0.625 | 0 | 793.660 / 791.660 |
| reduce | 1024 | 10.404 | 81.793 | 0.018 | 0.049 | 1 | 793.660 / 791.660 |
| reduce | 1048576 | 9.228 | 84.787 | 0.026 | 0.063 | 17407 | 793.660 / 791.660 |
| transform | 1024 | 9.851 | 50.134 | 0.035 | 0.041 | 0 | 791.660 / 791.660 |
| transform | 1048576 | 8.963 | 51.300 | 0.019 | 0.044 | 0 | 791.660 / 791.660 |

结论：

- CUDA 三类 primitive 的 first-call 均明显优于 vanilla，主要收益来自绕过
  Python helper kernel / Taichi IR 编译。
- CUDA 三类 primitive 的 warm runtime 在 repeat30 下也全部优于 vanilla。
- CUDA reduce 仍有 CUB 临时 workspace；这是运行时 workspace，不会膨胀 Taichi IR。

## Vulkan 结果

Vulkan 使用 repeat30 复测。

| op | N | Forge first ms | Vanilla first ms | Forge warm ms | Vanilla warm ms | Forge workspace bytes | Forge/Vanilla GPU delta MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| scan | 1024 | 20.137 | 122.600 | 0.182 | 0.420 | 0 | 89.145 / 121.012 |
| scan | 1048576 | 21.694 | 123.870 | 0.201 | 0.733 | 0 | 89.145 / 121.012 |
| reduce | 1024 | 15.389 | 22.814 | 0.156 | 0.131 | 0 | 89.145 / 121.012 |
| reduce | 1048576 | 14.922 | 23.467 | 0.160 | 0.153 | 0 | 89.145 / 121.012 |
| transform | 1024 | 10.398 | 16.030 | 0.157 | 0.126 | 0 | 89.145 / 121.012 |
| transform | 1048576 | 10.163 | 18.308 | 0.164 | 0.141 | 0 | 89.145 / 121.012 |

结论：

- Vulkan scan 已同时解决 first-call 和 warm runtime。
- Vulkan reduce/transform 的 first-call 和 GPU delta 均优于 vanilla，但 warm runtime
  仍略慢，差距集中在 0.02-0.03 ms 级固定提交成本。
- 这不是 IR 问题，而是 native Vulkan 调用的 descriptor/command/barrier/sync 固定
  成本还没有被完全摊薄。

## 固定开销与规模开销

- 1024 规模主要暴露固定成本。CPU/CUDA 通过 replay 和 native entrypoint 已明显
  压低固定成本；Vulkan scan 也已压低。Vulkan reduce/transform 仍需要继续做更低层
  command replay 或提交摊销。
- 1048576 规模主要暴露吞吐。CPU 的 contiguous typed loop 和 sum fast path 已把
  reduce/transform 拉回到 vanilla 以上；CUDA device/CUB 路径也已优于 vanilla。
- 当前 S4 不要求应用层把 field 改写成 ndarray，避免引入额外数据拷贝。只在
  dense field proof 成功时走 native path；proof 失败仍保留 legacy field/SNode
  fallback 或清晰报错。

## NativePrimitivePlan 替换后验证

本轮将原 dense-field 专用 replay plan 泛化为 `_NativePrimitivePlan`，并让
`field`、普通 `ndarray`、`StructNdarray` scalar member 共同使用该层；
`StructNdarray` whole vector/matrix member transform 也已接入同一层，replay
到 packed strided native call。验证结果：

- 新增可复跑脚本：
  `benchmarks/s4_native_plan_replay_bench.py`。
- 结果目录：
  `benchmarks/results/s4_native_plan_replay_refactor_20260522/summary.csv`。
- 覆盖矩阵：CPU/CUDA/Vulkan × field/ndarray/StructNdarray member ×
  scan/reduce/transform × 1024/1048576。
- 54 个组合无失败；所有非 skip 组合均 `ok=True` 且 `plan_reused=True`。
- 该层仍为 Python-only，不生成 Taichi helper IR，不改变 offline cache key、
  C++ ABI 或 runtime bitcode。

代表性结果：

| backend/storage/op | first 1024 ms | warm 1024 ms | first 1M ms | warm 1M ms | workspace |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU field reduce | 0.452 | 0.0145 | 1.360 | 0.149 | 0 / 128 B |
| CPU ndarray reduce | 0.126 | 0.0159 | 0.804 | 0.126 | 0 / 128 B |
| CPU StructMember reduce | 0.131 | 0.0151 | 0.927 | 0.146 | 0 / 128 B |
| CUDA field transform | 8.471 | 0.0185 | 8.808 | 0.0208 | 0 B |
| CUDA ndarray transform | 9.675 | 0.0362 | 10.123 | 0.0382 | 0 B |
| CUDA StructMember transform | 10.309 | 0.0400 | 9.415 | 0.0234 | 0 B |
| Vulkan field transform | 10.429 | 0.1456 | 9.931 | 0.1625 | 0 B |
| Vulkan ndarray transform | 18.520 | 0.1655 | 11.908 | 0.1669 | 0 B |
| Vulkan StructMember transform | 19.984 | 0.1686 | 20.373 | 0.1541 | 40 B |

存储占用观察：

- CPU transform/scan 为 0 workspace；reduce 大规模为 128 B。
- CUDA transform 为 0 workspace；CUB scan/reduce 仍按 CUB temp storage 变化，
  本轮最大约 20 KB。
- Vulkan field/ndarray reduce/transform 为 0 workspace；StructNdarray member
  strided 参数路径需要 12-40 B；scan workspace 随规模增长，1M 约 16 KB。
- Windows GPU dedicated memory counter 主要反映进程级 runtime/context 常驻成本：
  CUDA 约 792-794 MB，Vulkan 约 57-89 MB；该指标不应直接解释为单个 primitive 的
  额外显存。

2026-05-23 清理复测：

- 已移除 `_NativeDenseFieldPlan`、`_dense_*_plan`、`_vulkan_dense_*_plan`
  兼容层，dense field 分支直接记录 `_NativePrimitivePlan`。
- `StructNdarray` whole vector/matrix member transform 已接入同一 plan 层，
  replay 到 packed strided native call。
- 结果目录：
  `benchmarks/results/s4_native_plan_replay_clean_struct_tensor_20260523/summary.csv`。
- 72 个组合中，whole tensor member 的 scan/reduce 因当前没有单次 packed
  primitive 明确 skip；其余所有非 skip 组合均 `ok=True` 且 `plan_reused=True`。
- 当前代码中可由 `_NativePrimitivePlan` 替换的旧 replay 包装均已替换：
  reduce/transform workspace 和 scan executor 只保留 `_native_*_plan` 记录源；
  旧 dense-field 兼容字段不再作为可读写状态存在。
- S4.5 追加替换 `IndexedCopyWorkspace`：`experimental_gather()` /
  `experimental_scatter()` 的 plain ndarray、`StructNdarray` scalar member 和
  whole vector/matrix member direct native indexed-copy 调用已记录为
  `_native_indexed_copy_plan`。StructNdarray member 不再回退到旧的 component
  loop 或 field/kernel fallback；缺少 packed/strided native backend 时直接报错。

新增代表性结果：

| backend/storage/op | first 1024 ms | warm 1024 ms | first 1M ms | warm 1M ms | workspace |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU StructTensor transform | 0.267 | 0.0244 | 1.820 | 0.400 | 0 B |
| CUDA StructTensor transform | 9.206 | 0.0549 | 9.849 | 0.0344 | 0 B |
| Vulkan StructTensor transform | 19.670 | 0.1961 | 22.159 | 0.1807 | 40 B |

field 与 vanilla 1.8.0 复测结果：

- 结果目录：
  `benchmarks/results/s4_dense_field_native_after_plan_refactor_20260522/summary.csv`。
- CPU reduce/transform first-call 仍比 vanilla 快 28-102 倍；warm runtime 在
  两个规模均不慢于 vanilla。
- CUDA scan/reduce/transform first-call 仍比 vanilla 快 5-36 倍；warm runtime
  仍为 vanilla 的约 2-20 倍速度。
- Vulkan scan first-call 和 warm runtime 仍显著优于 vanilla。
- Vulkan reduce/transform first-call 仍优于 vanilla；warm runtime 在小数组或
  部分 1M reduce 场景仍受 native submission 固定成本影响，尚未保证所有规模都
  优于 vanilla。

清理后 dense field 复测结果：

- 结果目录：
  `benchmarks/results/s4_dense_field_native_clean_plan_20260523/summary.csv`。
- CPU reduce/transform first-call 仍比 vanilla 快 28-104 倍；warm runtime
  仍为 1.04-5.10 倍速度。
- CUDA scan/reduce/transform first-call 仍比 vanilla 快 4.7-36 倍；warm runtime
  仍为 1.14-23.19 倍速度。
- Vulkan scan first-call 与 warm runtime 仍优于 vanilla；Vulkan reduce/transform
  first-call 仍优于 vanilla；warm runtime 仍是后续 command/submission
  摊销优化目标。

## 剩余问题

当前尚未完全满足“Vulkan reduce/transform 所有规模 warm runtime 都优于 vanilla
1.8.0”。下一步应独立推进：

- 更底层的 Vulkan command descriptor replay，避免每次重建同构 command 序列。
- 评估 secondary command buffer 或内部 graph replay，但必须保证 field/SNode
  allocation lifetime、reset safety 和 descriptor 更新语义。
- 建立 Vulkan 小规模 cost model：不要为了 0.02 ms 级小数组 runtime 差距重新引入
  Python helper IR 或 offline cache 膨胀。

## 编码和验证记录

- 已使用 `cmd /c _run_build.cmd` 构建，并同步
  `build_llvm20_test/taichi_python.cp310-win_amd64.pyd` 到
  `python/taichi_forge/_lib/core/` 和 `python/taichi/_lib/core/`。
- 已通过 focused correctness：
  - replay 回归：CPU/CUDA/Vulkan scan/reduce/transform 共 9 passed。
  - dtype/native 回归：CPU/CUDA/Vulkan dense field scan/reduce/transform 共 9 passed。
  - CPU sum fast path focused 回归：4 passed。
- `_NativePrimitivePlan` 统一替换并移除旧 dense plan 兼容层后，focused replay
  回归：10 passed。
- StructNdarray indexed gather/scatter plan 化后，`tests/python/test_indexed_copy.py`
  文件级回归：20 passed。
- CPU native 子集回归：reduce 7 passed、transform 9 passed、scan 8 passed。
- CUDA/Vulkan 代表性回归：20 passed。
- 本地 wheel 已重新生成：
  `dist/taichi_forge-0.4.0-cp310-cp310-win_amd64.whl`。
- 3.10 wheel smoke 通过：安装到
  `build_llvm20_test/wheel_smoke_s4_clean_struct_plan_20260523` 后
  `import taichi_forge as ti; ti.init(arch=ti.cpu)` 成功。当前机器用户给定的
  miniforge 3.10 路径不可执行，本轮使用
  `C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe`
  完成等价版本检查和 smoke。
- S4.5 indexed-copy plan 更新后已重新生成同一 wheel，并安装到
  `build_llvm20_test/wheel_smoke_s4_indexed_plan_20260523` 做
  `import taichi_forge as ti; ti.init(arch=ti.cpu)` smoke，通过。offline cache
  lock warning 不影响导入和 CPU 初始化。
- 当前修改未触达 runtime bitcode，因此无需同步 `runtime_cuda.bc` 或 `runtime_x64.bc`。

## S5 dense field indexed-copy

本轮将 `experimental_gather()` / `experimental_scatter()` 的 1D dense field
路径接入 native primitive：

- Python routing 使用 `_PrimitiveView` 识别 dense field，并将 CPU/CUDA/Vulkan
  direct call 记录到 `IndexedCopyWorkspace._native_indexed_copy_plan`。
- 新增 C++/pybind entrypoint：
  - `cpu_gather_dense_field` / `cpu_scatter_dense_field`
  - `cuda_device_gather_dense_field` / `cuda_device_scatter_dense_field`
  - `vulkan_gather_dense_field` / `vulkan_scatter_dense_field`
- Vulkan indexed-copy cache 改为按实际 primitive 懒创建 pipeline，避免首次
  gather/scatter 同时创建 scatter-add 等无关 pipeline。
- Vulkan 32-bit scatter 增加专用 shader：
  `indexed_copy_u32_by_i32.comp` -> `scatter_dense_u32_by_i32.comp.spv.h`。
  gather 复测显示专用 shader 对 1M 吞吐不稳定，因此保留通用懒创建 pipeline。
- CPU indexed-copy 内层从逐元素 `std::memcpy` 改为 32-bit word copy，覆盖
  ndarray、StructNdarray member 和 dense field 的共享 native path。

最终代表性结果：

- 主结果目录：
  `benchmarks/results/s5_dense_field_indexed_wordcopy_20260523/summary.csv`。
- Vulkan gather 1M 高 repeats 复测：
  `benchmarks/results/s5_dense_field_indexed_vulkan_gather_repeat_20260523/summary.csv`。

| backend/op/n | forge first ms | vanilla first ms | forge warm ms | vanilla warm ms | workspace | GPU delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU gather 4K | 0.505 | 32.587 | 0.0258 | 0.0660 | 0 B | N/A |
| CPU gather 1M | 1.764 | 45.560 | 0.3727 | 0.1822 | 0 B | N/A |
| CPU scatter 4K | 0.456 | 32.226 | 0.0253 | 0.0673 | 0 B | N/A |
| CPU scatter 1M | 1.683 | 34.048 | 0.3456 | 0.1557 | 0 B | N/A |
| CUDA gather 4K | 9.274 | 49.836 | 0.0233 | 0.0446 | 0 B | 791.66 MB |
| CUDA gather 1M | 8.646 | 48.902 | 0.0576 | 0.0509 | 0 B | 791.66 MB |
| CUDA scatter 4K | 9.554 | 51.704 | 0.0224 | 0.0457 | 0 B | 791.66 MB |
| CUDA scatter 1M | 9.714 | 46.941 | 0.0237 | 0.0863 | 0 B | 791.66 MB |
| Vulkan gather 4K | 12.342 | 16.056 | 0.2225 | 0.2992 | 0 B | 89.21 MB |
| Vulkan gather 1M repeat20 | 12.842 | 17.140 | 0.2130 | 0.3580 | 0 B | 89.15 MB |
| Vulkan scatter 4K | 11.087 | 15.360 | 0.2059 | 0.3026 | 0 B | 89.21 MB |
| Vulkan scatter 1M | 10.935 | 23.908 | 0.2011 | 0.4903 | 0 B | 89.15 MB |

结论：

- 编译/first-call：dense field gather/scatter 在 CPU/CUDA/Vulkan 上均显著少于
  vanilla field kernel；Vulkan pipeline 懒创建将 first-call 从此前约 75-85 ms
  降到约 11-13 ms。
- 运行时：CUDA scatter、Vulkan gather/scatter 和小规模 CPU 均优于 vanilla；
  CPU 1M warm runtime 仍慢于 vanilla field kernel，后续需要独立优化 CPU
  大规模路径或建立 auto cost model。
- 存储：workspace 均为 0 B；GPU dedicated delta 上 forge Vulkan 约 89 MB，
  低于 vanilla Vulkan 约 121 MB。

## S5 dense field scatter-add / histogram

本轮继续完成 S5 中与 dense field 和 StructNdarray 对齐的两类聚合 primitive：

- `experimental_scatter_add()`：
  - dense field + ndarray i32 indices 接入 CPU/CUDA/Vulkan native entrypoint；
  - StructNdarray scalar member 与 tensor member component 路径使用
    `_NativePrimitivePlan`，重复调用复用 plan；
  - 未加入自动选择策略，显式 native method 直接走对应后端。
- `experimental_histogram()`：
  - contiguous dense field values/bins 接入 CPU/CUDA/Vulkan native entrypoint；
  - StructNdarray scalar member histogram 暂不走旧 fallback，显式提示需要
    native strided histogram；
  - Vulkan histogram pipeline 改为按实际路径懒创建，将 first-call 从旧实现的
    约 186-191 ms 降到约 19-21 ms。

代表性结果：

- 结果目录：
  `benchmarks/results/s5_dense_field_scatter_add_histogram_lazy_vkhist_20260523/summary.csv`。
- 基准形态：`n = 4096 / 1048576`，输出 bin 数为 `min(n, 4096)`，
  repeat 10，warmup 3，forge 0.4.0 对比本地 vanilla Taichi 1.8.0。

| backend/op/n | forge first ms | vanilla first ms | forge warm ms | vanilla warm ms | workspace | GPU delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU scatter_add 4K | 0.596 | 33.315 | 0.304 | 0.112 | 0 B | N/A |
| CPU scatter_add 1M | 1.838 | 38.182 | 0.709 | 3.774 | 256 KiB | N/A |
| CPU histogram 4K | 0.439 | 50.497 | 0.226 | 0.131 | 0 B | N/A |
| CPU histogram 1M | 1.689 | 52.549 | 0.616 | 2.068 | 512 KiB | N/A |
| CUDA scatter_add 4K | 9.902 | 51.988 | 0.365 | 0.081 | 0 B | 791.66 MB |
| CUDA scatter_add 1M | 9.393 | 51.209 | 0.370 | 0.094 | 0 B | 791.66 MB |
| CUDA histogram 4K | 9.481 | 82.858 | 0.276 | 0.053 | 16.5 KiB | 793.66 MB |
| CUDA histogram 1M | 9.615 | 85.497 | 0.298 | 0.066 | 2.67 MiB | 795.66 MB |
| Vulkan scatter_add 4K | 11.725 | 19.500 | 0.788 | 0.303 | 0 B | 89.21 MB |
| Vulkan scatter_add 1M | 11.616 | 19.579 | 0.800 | 0.178 | 0 B | 89.15 MB |
| Vulkan histogram 4K | 20.864 | 26.563 | 0.705 | 0.189 | 0 B | 89.15 MB |
| Vulkan histogram 1M | 19.072 | 26.119 | 0.781 | 0.331 | 0 B | 89.15 MB |

结论：

- 编译/first-call：scatter_add 和 histogram 在 CPU/CUDA/Vulkan 上均优于
  vanilla 1.8.0；Vulkan histogram 的 pipeline 懒创建已经消除最明显的
  first-call 膨胀。
- 运行时：CPU 大规模路径明显优于 vanilla；CPU 小规模、CUDA、Vulkan 的 warm
  runtime 仍受 native 调用固定成本影响。这里没有加入自动选择策略，因此该结果
  应作为后续 GPU command replay、轻量 CUDA kernel、Vulkan descriptor/command
  复用优化的基线。
- 存储：CPU/CUDA 聚合类 primitive 会使用小型 partial workspace；Vulkan 当前
  dense field scatter_add/histogram workspace 为 0 B，GPU dedicated delta 仍低于
  vanilla Vulkan 的约 121 MB。

S5 correctness：

- `tests/python/test_indexed_copy.py`：23 passed。
- `tests/python/test_scatter_add.py tests/python/test_histogram.py`：22 passed
  （关闭 offline cache；pytest cache 权限 warning 不影响结果）。
