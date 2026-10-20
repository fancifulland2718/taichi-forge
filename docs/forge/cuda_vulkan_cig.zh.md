# Vulkan CUDA CiG 初始化

Forge 可在第一次 CUDA 使用之前，将进程的 CUDA context 关联到调用方拥有的 Vulkan
external compute queue。此接口不依赖 Flora 或 NVIGI，不替代执行域、完成票据及外部
存储的 ready/consumed 协议。设备能力位、context 创建成功、应用性能合格是三个不同结论。

**当前为显式选择的实验路线，普通 CUDA 仍为默认值。** 本机驱动在 CiG 的一条 stream
等待外部信号时，会阻塞另一条 stream 上独立的 kernel；因此不能直接替代普通 CUDA
的绘图／物理隔离调度。没有 GeoPhys/Flora 并发帧率的合格或加速声明。
外部输入现在可绑定原生就绪门禁，避免把未就绪的等待排入 CiG；这不改变驱动中已排入
等待的行为，应用仍需使用下述准入方式。

## 初始化和所有权

Vulkan 所有者在创建设备时预留 external compute queue，以真实渲染 queue 作为
`preferredQueue` 创建它，并按 `externalDataSize` 获取 NVIDIA opaque data。Forge
复制 bytes，不解释其内容。`device_uuid` 必须精确匹配 CUDA UUID，不能使用 API ordinal
代替。`vkGetExternalComputeQueueDataNV` 返回 void；大小来自扩展属性。

C 消费者在建立任何 CUDA runtime 之前调用 `ti_configure_cuda_context_ext()`，参数为
`taichi_cuda.h` 中的 `TiCudaContextConfig`，`schema_version=1`、`graphics_api=1`。
`owner` 和原生 retain/release 回调须共同保留 external queue **以及 Vulkan device**。
回调不得抛异常或依赖 Python 解释器。不能只保留 opaque bytes 或窗口对象。

Python 接受由该原生所有者产生的 capsule，名称为
`taichi.cuda_vulkan_context_binding.v1`，内容指向上述描述。capsule 自身必须保留描述及
其 bytes，直到 Forge 完成复制和取得原生 lease。

```python
from taichi_forge_runtime.aot import (
    CudaVulkanContextBinding, configure_cuda_context, Runtime,
)

def start_forge(native_capsule):
    binding = CudaVulkanContextBinding(native_capsule)
    info = configure_cuda_context(binding, mode="required")
    assert info["context_mode"] == "vulkan_cig"
    return Runtime("cuda")
```

JIT 使用同一个入口：先配置，再 `ti.init(arch=ti.cuda)`。配置不得晚于 CUDA capability
探测、Program、runtime、module、array、Graph、stream 或导入资源的首次建立。

| 模式 | 行为 |
| --- | --- |
| `required` | 必须提供完整绑定；CiG 创建或实际分配资格失败即报错 |
| `auto` | 初始化前可使用普通 CUDA；仅在完整回滚后允许回退，诊断报告原因 |
| `disabled` | 不接受绑定，选择普通 CUDA |

无效描述、错误 UUID、晚配置不静默回退。失败退休会保留精确的原生 owner，并拒绝在
同一进程重试。第一版保持 Forge 原有进程常驻 context：`Runtime.close()` / `ti.reset()`
退休该 runtime 的资源，但 context 和 Vulkan lease 保留到进程结束。可以重建 Forge
runtime；**不能销毁并换绑 Vulkan device，也不能把窗口关闭当成 device 已释放**。
需要新 Vulkan device 时使用新进程。

## 分配与稳态成本

本机 Windows / RTX 5090 / 610.88 / Driver API 13030 的原生检查发现：普通设备内存和
managed 内存可执行 kernel 和 Graph；async allocation 返回成功后，直接 kernel
访问多次触发 `CUDA_ERROR_ILLEGAL_ADDRESS`，但也观察到直接访问和 Graph 访问成功。
这组结果不足以证明 async/pool 路线可靠，也不能称其必定失败；CiG 因此使用已验证的
普通 device allocation 保留池。不能仅用分配返回码或单次成功认定可用，也不能将
本机结果外推到所有 GPU 和未来驱动。

CiG 的等价后端使用普通 device allocation，覆盖 runtime array、参数包、固定 binding
frame 和 Graph 自有存储。冷初始化或容量增长时分配，释放时记录该 stream 前缀的事件，
仅复用已完成的块。稳态复用不调用 stream/context synchronize 或原生 free。
调用方仍须限制在途工作及尺寸集合；保留量随容量高水位增长，不提供任意动态负载的固定
VRAM 上限。`trim()` 是显式冷操作：其 `cuMemFree` 可能等待无关设备工作，不能逐帧调用。

`cuda_context_info()` / `ti_get_cuda_context_info_ext()` 不创建 context、不提交 GPU 工作。
返回实际模式、UUID、generation、驱动入口、设备能力、分配检查、分配路线及回退原因。
`retained_pool` 统计进程参数/暂存池的分配、释放、复用、退休次数和 used/reserved 高水位。
Graph 自有池独立报告 `retained_device_pool=1` 及同类计数；不会伪报 native pool 的
release threshold。普通 CUDA 原有的 native async pool 路径继续使用。

CiG 下 cuDSS 的原生 stream-ordered allocator 未获资格，Graph cuDSS 路线继续明确拒绝。
其他外部 provider 的原生内存行为需要各自验证，不能根据普通 kernel 的结果扩展支持声明。

## AOT 和调度

CiG 绑定是消费进程的初始化参数，不写入 AOT artifact。现有 schema 8、runtime context
ABI、BC/PTX/cubin 导出合同保持不变；独立消费者无须导入 DSL 或 NumPy。module、Graph、
执行域、event、外部导入全部从已选 context 创建；编译字节可缓存，native handle 不跨
runtime generation 复用。调用线程进入 native 时绑定 Forge context，退出恢复原 context。

沿用[执行域](cuda_execution_domains.zh.md)：simulation → snapshot、snapshot → render、
render completion → slot reuse 由精确设备依赖表达。物理下一步只等待 live state 拷贝
完成，不等待屏幕呈现；输入/输出各用有界执行域，普通画面请求可保留最新一个。CiG 不提供
对 Vulkan 的严格抢占保证，也不消除 GPU 带宽或 SM 竞争。

本机不依赖 Forge runtime 的原生对照进一步发现：Vulkan producer 被尚未满足的 host
timeline 信号挡住时，CiG 中另一条 nonblocking stream 的直接 kernel 和 Graph 在
500 ms 观察期内都未完成；普通 CUDA 的独立 Graph 完成，而被挡住的 stream 仍为
NOT_READY。解除信号后数值正确。该实验测量依赖隔离，不是应用延迟或吞吐 benchmark。
设置更多 CUDA connections 未消除 Forge AOT 对照中的相同现象。

在这一驱动上试用 CiG 时，由 Vulkan 所有者非阻塞查询输入就绪状态，**就绪后才准入**
需要外部输入的工作；仍执行原有 ready/consumed 信号协议，不允许靠主机观察绕过
所有权。未就绪时跳过普通渲染请求或保留最新请求，物理继续提交；不要先向 CiG 排入
一个可能长期挂起的外部 wait。所需就绪信息属于原生 producer，Forge 不猜测其私有
句柄。若应用必须预先排入未就绪的外部等待，继续使用普通 CUDA。此建议仍需 G3 对照。

### 原生就绪准入

`Runtime.import_cuda_external_memory(..., ready_gate=gate)` 接受
`CudaExternalReadyGate`，capsule 名为 `taichi.cuda_external_ready_gate.v1`，指向
`TiCudaExternalReadyGate`。C 入口是
`ti_import_cuda_external_memory_with_ready_gate_ext()`；原有 import 接口保持原语义。
Forge 复制描述并 retain 原生 owner，直到导入分配安全退休。owner 必须保留查询状态和
Vulkan device；原有共享分配及 semaphore 的存活义务仍然适用。

producer 的原生 `query(owner, epoch, ready)` 必须是无等待、无提交、不可重入 Forge 的
查询，例如已提交 producer release 的 fence 状态或 timeline counter。`epoch` 从 1
开始，每个实际消费的 ready 信号递增一次。只在对应轮次的 ready 信号已经完成时返回
true，并保持该状态直到消费；旧轮次的已完成 fence 不能放行下一轮。callback 不能依赖
Python 解释器或抛异常。查询错误会拒绝本次提交，不消费信号。

提交先校验依赖和容量，再检查冷绑定时已去重的外部输入，最后才排入任何设备等待或
执行 provider 准入。任一输入未就绪时返回 `None`，不分配提交序号、不消费信号、不
产生部分 GPU 工作。检查次数取决于显式外部输入数量，不扫描 kernel 或 Graph 指令。
就绪后仍执行完整的 ready/consumed 协议；不会以主机观察代替设备所有权交接。

`PreparedCall.submit(wait_for=..., ready_only=True)` 还会在 native 中一次性校验所有前驱，
任一票据未完成也返回 `None`。已完成票据保持原有 runtime/generation 与所有权校验。
JIT 的 prepared Graph execution 同样接受 `ready_only`；C 入口为
`ti_try_submit_prepared_ready_ext()`。默认值保留原有设备等待语义。

```python
from taichi_forge_runtime.aot import CudaExternalReadyGate

gate = CudaExternalReadyGate(producer_ready_capsule)  # 原生 producer 提供
image = runtime.import_cuda_external_memory(**external_import, ready_gate=gate)
render = module.prepare_execution("render", {"image": image}, execution_domain=render_domain)
render.initialize()  # 冷阶段；若初始化访问外部输入，该输入须已就绪

ticket = render.submit(wait_for=(snapshot_ticket,), ready_only=True)
if ticket is not None:
    enqueue_external_consumer_wait()  # 继续原有 consumed 协议
# None 时保留最新普通渲染请求，继续独立仿真；无需在此轮询等待。
```

这些是消费端运行时状态，AOT schema 8、runtime context ABI 2 和 BC/PTX/cubin 的编译
合同均不改变。内部同步域虚表增加就绪查询，因此 Python/native 配对 ABI 一次更新至
112；不是给 artifact 增加版本或元数据字段。

原生隔离实验还检查了只挂起 Vulkan 队列、只挂起 CUDA 外部 timeline 等待，以及
Graph wait/独立普通 CUDA context 的 event bridge。仅 Vulkan 队列挂起时独立 CiG
计算能完成；另外两种设备等待改写仍未解除阻塞，故未作为运行时后端采用。
`cuda_external_ready_probe.py` 用真实 Vulkan producer fence 和独立 AOT 数值工作验证
新门禁，并保留直接等待的负对照。人工延迟只用于隔离判据，不是正常帧延迟或 FPS。

就绪准入应按需启用：它消除输入未就绪带来的调度阻塞，但对本来已就绪的短交接仍有
查询开销。建议有界地保留最新渲染请求，在 producer 进展或下一次 UI 更新时重试，
不要持续轮询到提交成功；必须提前排入未满足设备等待的工作仍适合普通 CUDA。

## 本机验证范围与复现

2026-10-20 的真实渲染器接入进一步暴露了本机 RTX 5090 / 610.88 路线的边界：
预留 external compute queue 的 Vulkan device 在较大传输中可能发生 device loss，
即使尚未建立 CUDA context。关闭预留的普通 Vulkan 对照通过；关闭隐式工具层仍可
复现失败。独立 Vulkan 控制中，4 KiB fill / copy 分块可以通过图像大小的数值校验，
但渲染器采用传输分块后仍在 BLAS 构建失败。因此，不能将小缓冲交换资格扩大为
真实 RT 渲染资格，也不能只替换传输路径就宣布问题解决。验证层下的成功运行未作为
生产修复或性能证据。

普通 CUDA 的有头对照已经走通原生就绪查询、导出后加载的 AOT 输出 kernel 和 Canvas
执行域。就绪准入没有获得稳定的应用帧率提升；它仍适用于避免 CiG 未满足等待的
阻塞，不应在普通 CUDA 上默认启用。验收须同时测量新画面吞吐、物理吞吐和完整图像
及物理状态，不能以后台等待函数的耗时或调用次数代替。关闭窗口会退休依赖 Canvas
存储的执行域，消费端须先消费已接受的外部帧，再销毁窗口；这属于冷退出路径。

2026-10-20 的固定容量 AOT 三槽测试在关联的同一个 Vulkan device 上执行了 10,000
次交换，并进行场景资源重建和带在途工作关闭；每次有实际 Vulkan fill、AOT kernel、
device-local 共享显存及其所有权转移、双向 binary semaphore 和数值 oracle；仅在
排空后的数值检查中复制到单独的 host-visible staging。进程参数／暂存保留池在预热后保持 4 次原生
分配、0 次原生释放、20,480 bytes，第二个场景复用相同存储。该计数只覆盖该池，
不能冒充全部显存或全部 CUDA API；测试中的显式票据／fence 等待用于槽回收资格，
不是零等待性能证据。Graph 自有池另有复用与 pending-prefix 测试。

当前 Nsight Systems 2026.1.2 捕获报告 CUPTI 初始化失败，CUDA API 事件数为零，
因此此捕获**不能**证明热路径没有 CUDA API／同步，也不作为完整 G2 性能验收。
原生依赖探针、非空数值工作、池计数和生命周期结果独立保留，不以 profiler 阻塞接口验证。
G1 独立 kernel 进度仍有上述限制；完整 G2 和 GeoPhys G3 尚未验收。

从匹配的当前 native、shim、bitcode 和 Python SDK 运行：

```powershell
$env:TI_TEST_CIG_FIXTURE = '<build>/taichi_vulkan_cig_fixture.dll'
$env:TI_TEST_CPP_EXECUTABLE = '<build>/taichi_cuda_execution_domain_tests.exe'
python tests/python/cuda_cig_test_owner.py tests/python/test_cuda_cig_context.py tests/python/test_cuda_cig_aot.py -v
python -S tests/python/cuda_cig_lifetime_probe.py <AOT module> result.json 10000
```

`test_cuda_cig_aot.py` 检查 BC、隐式／显式目标 PTX、cubin × fast/balanced/full，
包括无 DSL 独立消费者及共享槽交接。普通 CUDA 的 `test_aot_execution_context.py`
仍保留更强的独立进度门禁，没有为 CiG 删除断言。独立原生源码为
`tests/cpp/rhi/common/native_vulkan_cig_probe.cpp`，仅链接 Vulkan loader、动态加载
CUDA driver；参数为分配类型 0/device、1/managed、2/async，随后可加 `graph` 和
`ordinary`。async 探针必须另起进程，因为可能破坏该 context。

验证分为原生能力、Forge 数值/线程合同、固定容量寿命/稳态、GeoPhys fresh-process ABBA
四级。`tests/python/cuda_cig_test_owner.py` 与原生 `taichi_vulkan_cig_fixture` 是独立
Vulkan 所有者夹具，不是 Flora API。将 `TI_TEST_CIG_FIXTURE` 指向该 DLL 后可运行
`test_cuda_cig_context.py`；JIT/AOT 资格由夹具入口先配置再运行相应 pytest。
本接口本身不构成 GeoPhys 的图像、物理或帧率验收。

参考：[CUDA context API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__CTX.html)、
[Vulkan external compute queue](https://docs.vulkan.org/features/latest/features/proposals/VK_NV_external_compute_queue.html)。
