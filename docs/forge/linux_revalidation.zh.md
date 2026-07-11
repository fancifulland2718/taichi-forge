# Linux 复测状态

本文是 R8 runtime 加固后仍需在 Linux 上完成的 release 验证矩阵。它是测试计划，**不**表示这些
路径已经在 Linux 通过。请在干净的 x86_64 Linux runner 上，以目标发布依赖运行，并记录 GPU、
driver、Vulkan loader、窗口系统以及（构建 native method 时）CUDA Toolkit。

## Release 阻断项

### Runtime package 与随附 libdevice

构建并安装 release-equivalent runtime wheel，随后确认：

- 已安装 runtime 恰好包含一个 `slim_libdevice.<major>.bc` 文件；
- `taichi_forge._lib.core.cuda_version()` 返回带点的兼容版本字符串，且其主版本与该 asset
  文件名一致；
- CUDA-enabled build 中，生成的 header 不会与 NVIDIA 表示数值 toolkit 版本的
  `CUDA_VERSION` 宏冲突；该查询值不得被描述为已安装 CUDA Toolkit 或 driver 版本；
- shim wheel 仍能解析对应 runtime wheel，并在发布矩阵的所有受支持 CPython 版本导入。

这验证了包更新随附 libdevice asset 时无需再依赖源码中写死的新版本号。

### 单 runtime wheel 与 CUDA 驱动兼容边界

以最终候选 `CUDA_TOOLKIT_VERSION` 构建 runtime workflow，并对 auditwheel 处理后的上传候选
执行以下检查：

- Linux 只产出一个项目名为 `taichi-forge-runtime` 的 manylinux wheel；distribution、版本、
  extra 和 wheel tag 都不带 `cu11` / `cu12` / `cu13` 后缀；
- wheel 中恰好包含一个 `libtaichi_runtime.so`、一个 `cuda_runtime_major.txt`，以及一个与清单
  major 一致的 `libcudart.so.<major>*` 或 auditwheel hash 名；检查 `DT_NEEDED`、RPATH 和实际
  loader 路径均指向包内库；
- 在无 CUDA Toolkit 的干净环境安装 wheel，运行 CUDA native scan/reduce/sort、device-check、
  native AD、reset 和 workspace 稳定性矩阵；
- 若候选基线低于当前默认 CUDA 13.2，必须在目标旧 driver 上运行相同 wheel。只在新 driver
  编译或执行通过，不能证明最低 driver 已降低。

这个矩阵可以用 11.8/12.x 候选做内部验证，但最终只发布一套通过门槛的 Linux wheel，不建立
按 CUDA 版本分叉的包系列。

### CUDA 执行、graph 与 allocator 路径

在发布支持的 Linux NVIDIA driver 和真实 GPU 上运行 C++ backend safety target 与 CUDA
Python regressions。至少覆盖一个被随附 LLVM 原生支持的 target；如有条件，也覆盖一个走兼容
target fallback 的较新设备。验证数值结果、offline-cache target 隔离、capture/recapture/reset
和 1/2/4 submitter telemetry。

- 运行 `tests/python/cuda_driver_telemetry_stress.py`，保留其采样的 lock 与 allocation-route
  输出；诊断不得改变结果或引入默认同步点；
- 在 fresh process 中运行 `tests/python/cuda_graph_runtime_bench.py`。它用于检查 p50/p95 与
  reset 稳定性，不可作为跨机器性能对比；
- 以 `TI_WITH_CUDA_TOOLKIT=ON` 和 dynamic CUDART 构建，并实际覆盖 native CUB reduce。
  该可选路径不可用的 runner 必须报告既有 fallback，不能被计为 CUB 覆盖；
- 对受影响 CUDA regression 运行 `compute-sanitizer --tool memcheck`。只有已知当前 CUDA
  版本支持的 device-side atomic/duplicate-sensitive 用例才追加 `racecheck`。

### Vulkan、GGUI 与 Vulkan-CUDA interop

以 validation layer 运行 Vulkan RHI safety target；当 loader 提供时也启用 synchronization
validation。分别覆盖 offscreen 与 headed GGUI：headed 路径必须覆盖 release runner 使用的 Linux
窗口系统（X11 和/或 Wayland）、resize/out-of-date、关闭，以及 worker 持续提交 kernel 的同时
以 `set_image()`/`show()` 在 fresh process 中运行至少 30–60 秒。

对于同时暴露 Vulkan external-memory FD 和 CUDA external-memory import 的 runner，运行
Vulkan-CUDA external-memory copy 与 allocation-teardown regression。确认 Linux
`VK_KHR_external_memory_fd` / `CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD` 路径只会在 CUDA
import 成功后移交 FD 所有权。若设备缺少任一平台扩展，应验证同步 host-staging fallback；不能因
只支持基础 external-memory 就视为支持 GPU-direct interop。

使用 `tests/python/ggui_vulkan_queue_concurrency_stress.py` 收集 queue stress 的 frame/producer
p50 与 p95。只比较同一 runner 的重复样本；Windows 数据不是 Linux 性能基线。

### CPU scheduler 与生命周期安全

在 Linux 运行 CPU allocation、native primitive 和 graph concurrency regression。正式 gate 是
对 scheduler 和 allocation registry 生命周期路径执行 ThreadSanitizer；也应使用
AddressSanitizer/UBSan 覆盖析构、reset 与 range-validation。integer copy/gather/unique-scatter
仍要求精确结果；浮点 reduction 使用公开约定的 tolerance。

## 验收记录

每项都记录 command/configuration、通过或失败、硬件与 driver 版本，以及 validation layer 或
sanitizer diagnostics。缺少可选 capability 只有在 fallback 被明确执行且结果正确时才可接受。任何
device loss、sanitizer finding、synchronization-validation error、stale-cache result 或数值不一致
均阻断发布，直到完成诊断。
