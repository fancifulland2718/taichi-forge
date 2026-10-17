# Vulkan JIT and pipeline diagnostics / Vulkan JIT 与 pipeline 诊断

```python
import taichi_forge as ti

ti.init(arch=ti.vulkan, offline_cache=False)
# my_kernel and its specialization arguments are application-owned.
manifest = ti.tools.export_vulkan_kernel(
    my_kernel, *args, directory="new-jit-artifacts", validate=True
)
runtime = ti.tools.vulkan_runtime_diagnostics()
```

The export compiles the **resident JIT specialization**, including per-kernel
compile-tier overrides. It writes the exact SPIR-V bytes, SHA-256 hashes, task
and resource-binding metadata, resolved configuration, kernel identity, and
device/driver/cache provenance. It does not submit the kernel or create its
pipeline. Bound `@ti.data_oriented` methods include their owner automatically.
The destination must be new; an existing artifact directory is never overwritten.
`validate=True` uses `spirv-val` from PATH and preserves failed artifacts and
validator output before raising. The SDK tool is optional; pipeline-time
validation below uses the validator bundled into Forge.

Set these variables **before `ti.init()`** for reproducible pipeline diagnosis:

| Variable | Effect |
| --- | --- |
| `TI_VULKAN_VALIDATE_SPIRV=1` | Validate compute SPIR-V before reflection, shader-module creation and pipeline creation; reject invalid modules before calling the driver. Default off. |
| `TI_VULKAN_DIAGNOSTICS_DIR=<directory>` | Capture each compute pipeline's actual shaders, hashes, descriptor bindings, entry point, local size, specialization constants, device identity and flushed construction events. Default off. |
| `TI_VULKAN_PIPELINE_CACHE=0` | Disable both in-memory pipeline-cache attachment and disk load/save for the runtime. Default enabled. |
| `TI_VULKAN_PIPELINE_CACHE_PATH=<file>` | Use an independent pipeline-cache file. Default remains the repository cache directory's `rhi_cache.bin`. |

The pipeline cache is independent of `offline_cache=False`. Its path and enabled
state are captured at runtime initialization; later environment changes cannot
redirect teardown writes. `vulkan_runtime_diagnostics()` reports the load state
and hash of bytes supplied to the backend. `supplied_to_backend` does **not**
prove a cache hit: Vulkan may silently ignore incompatible data. Cache writes
use distinct temporary files and replacement without first deleting the old file.

Each capture has `pipeline.json`, `shader-*.spv`, and `events.jsonl`. Match a JIT
export to a pipeline by shader hash, not by a changing generated task name.
`pipeline_call_started` followed by no return event records an interrupted
construction attempt, **not** a Vulkan return code or proof of a driver defect.
Returned compute-pipeline failures preserve the numeric `VkResult` and operation
name. Driver register allocation and occupancy are not inferred from SPIR-V.
Diagnostic capture is opt-in and incurs file I/O; validation and capture are
not performance-measurement settings. These captures cover compute pipelines,
not graphics or native ray-tracing pipeline construction.

上述导出针对**实际 JIT specialization**，包含 kernel 自身的编译档位覆盖，不以
AOT 产物代替 JIT。输出 shader 原始字节、SHA-256、task/resource binding 元数据、
解析后的配置、kernel 身份及设备、驱动、缓存来源；导出过程不提交 kernel，也不创建
其 pipeline。绑定的 data-oriented 方法会自动带入对象。目录必须尚不存在。
`validate=True` 使用 PATH 中可选的 `spirv-val`；失败时保留产物及验证日志。

表中的环境变量应在 `ti.init()` 前设置。pipeline 前验证使用 Forge 已链接的验证器，
默认关闭；不新增外部运行库依赖。诊断目录记录实际传给驱动的 compute shader、布局和
已刷新到文件的阶段事件。没有返回事件时只说明构建尝试被中断，不能虚构 `VkResult`，
也不能仅凭这一点归责驱动。驱动返回失败时保留数值返回码及操作名。

pipeline cache 与 kernel offline cache 独立：`offline_cache=False` 不会关闭前者。
关闭 pipeline cache 后不加载、不挂接、不写回；自定义路径与启用状态固定在 runtime
初始化时。报告的 `supplied_to_backend` 只表示向 Vulkan 提供过这些字节，不能充当命中率。
独立临时文件及替换写入避免不同进程共用未写完的文件，也不在替换失败前删除旧缓存。
诊断模式有验证和文件 I/O 开销，不用于性能计分；不据 shader 反推寄存器或 occupancy，
也不将 compute pipeline 的覆盖扩展为 graphics/ray-tracing pipeline 覆盖。
