# CUDA 与 Vulkan 执行域（开发中）

[English](cuda_execution_domains.en.md)

独立 AOT consumer 可以将受支持的已保存执行计划绑定到有界 CUDA 或 Vulkan 执行域。
CUDA 域拥有 nonblocking stream；Vulkan 域租用设备创建时准备的额外队列，默认选择 compute。
每个准备好的实例拥有自己的 recording、参数存储和编译临时区。
该接口保留已经选择的完整计划。stream、优先级和完成票据只属于运行时，不写入 AOT 产物。

JIT Graph 也可以冷准备为独立实例，借用已选定的编译代码和当前 Program 的 dense Field。
这是 F3 的一部分；Canvas 显式选域已有 Windows CUDA/Vulkan 资格，跨平台真实窗口
和应用 FPS 仍未完成资格验证。
优先级是设备调度提示，不能保证 kernel 并行或抢占。

两种后端共用准备、提交、参数、计时和退休接口。CUDA 控制流/并行、外部存储和 Canvas
保留原有支持面；Vulkan 接纳固定 buffer compute recipe 和下述有明确所有权的图像路线。

## 能力查询

在 `ti.init()` 前可读取 `ti.runtime_contract_manifest()["features"]`。独立 consumer
可用 `taichi_forge_runtime.runtime_contract_manifest()["features"]`，不导入 DSL 或
创建设备。以下是原生实现声明：

| 能力 | 声明的路线 |
| --- | --- |
| `cuda_execution_domains` | 有界持久队列、显式优先级、精确票据依赖 |
| `cuda_aot_execution_domains` | 已保存完整执行计划的准备和重放 |
| `cuda_jit_graph_execution_domains` | 完整 live JIT Graph 准备 |
| `cuda_external_storage_execution_domains` | 托管外部导入及选域 acquire/release epoch |
| `cuda_canvas_execution_domains` | Canvas copy/writer/empty/repeat 选域及固定预备池 |
| `cuda_submission_parameters_u64` | JIT/AOT 独立实例逐提交更新已声明的 u64 scalar |
| `cuda_completion_gpu_timing` | 预分配、逐提交采样及 completion 的非阻塞 GPU 计时读取 |
| `cuda_execution_instance_retirement` | 显式退休单个实例、复用驻留预算，并保留已有票据 |
| `vulkan_execution_domains` | 额外 compute queue 与精确 timeline 依赖 |
| `vulkan_aot_execution_domains` | 已保存固定 buffer-only 计划的准备与重放 |
| `vulkan_jit_graph_execution_domains` | 已选 JIT buffer-only dispatch/code 借用 |
| `vulkan_submission_parameters_u64` | 在所选 Vulkan 队列更新已声明的 replay-only u64 |
| `vulkan_completion_gpu_timing` | 预分配 timestamp query 与稀疏完整计算区间采样 |
| `vulkan_execution_instance_retirement` | 单实例退休及驻留预算复用 |
| `vulkan_execution_domain_shared_sampled_images` | 同 Program 托管 color Texture 的跨域只读采样 |
| `vulkan_execution_domain_private_graphics` | 实例私有 color/depth attachment、离屏绘制及后续 compute |
| `vulkan_execution_domain_live_image_writes` | 平直 storage kernel 或描述完整的 graphics 计划排他借用 owned Texture |

对旧包使用 `features.get(name, False)`；不能从版本号或较新的提交推定能力。具体设备/
驱动、同 GPU 共享及 recipe 的编译/provider 条件在冷创建和准备时检查。声明为真不代表
Linux 窗口或所有 provider 已通过验收。版本见 [SDK 配对合同](aot_sdk.zh.md#component-pairing)；不匹配的 shim/
native 在进入 C++ 绑定前拒绝。

## Vulkan 队列与配方边界

Vulkan 域要求已启用 timeline semaphore，并且 runtime 所选 queue family 中有未占用的
额外 compute queue。队列零保留给普通 runtime；每个活域租用另一条队列，因此两个域
至少需要三条已创建队列。耗尽时在冷创建阶段拒绝，不悄悄绕回队列零。导入设备也须提供
相应的队列与 timeline 信息。不同队列未必对应独立 GPU 执行引擎，不能据此保证吞吐提升。

现有接口可用于 Vulkan：

```python
# JIT 已经通过 ti.init(arch=ti.vulkan) 选择设备。
domain = ti.graph.execution_domain(capacity=2, instance_capacity=2, gpu_timing=True)
step = graph.prepare(arguments, execution_domain=domain, mutable_parameters=names)
step.initialize()
ticket = step.submit(values=metadata, sample_gpu=True)
```

无源码消费端使用 `Runtime("vulkan", device_index=...)`，再调用相同的
`runtime.execution_domain()` / `module.prepare_execution()`。应明确选择设备：runtime-only
的设备零与 JIT 自动选择不一定是同一 GPU。导出含 u64 的 AOT kernel 时声明
`caps=["spirv_has_int64"]`。Vulkan 优先级在设备创建时固定，此处使用 `priority=0`。

该路线复用已选编译 dispatch 和私有固定工作区。公开验收覆盖 scalar/tensor buffer
kernel、显式 adjoint、初始化/重放、双独立实例、高位 u64 更新、稀疏计时、完成集合、
实例及 reset 退休。`vulkan_execution_domain_while` 能力位表示可重建已选设备 while，
包括分块和嵌套 recipe；具体策略仍须满足所选设备能力。前段、首次条件、循环分块及收尾
共用一次队列提交，逐提交参数更新覆盖所有物理 dispatch 副本，计时覆盖完整计算序列。
native buffer 操作仍须具备适用的固定录制 factory。

`vulkan_execution_domain_private_images` 支持执行计划为每个实例重建的私有 color
image：普通 compute 段可使用 sampled/storage image 和显式 color upload。各阶段共用
固定边界布局，首次转换、初始化和重放都由所选域执行；资源创建时提供初始 pixels
仍可能在冷准备阶段完成上传。图像保留到实例退休，无需每次提交查询布局或重建录制。
`vulkan_execution_domain_image_control` 进一步支持已选设备 while 的条件和循环体中
使用这些私有图像，包括分块和嵌套 recipe。每段录制都恢复相同边界布局，覆盖零轮、
早停和预算耗尽；连续 storage 使用同样记录内存依赖，同一 task 内 sampled/storage
别名在 prepare 阶段拒绝。私有资源可在 AOT plan 的 `resources` 中声明；外部借用
Texture/native image 的共享边界见下段。Windows 验证覆盖
双实例、下游显式 adjoint、准备失败清理和在途退休。

`vulkan_execution_domain_shared_sampled_images` 支持同 Program、同设备及 compute
queue family 的托管 color 2D Texture，当前限单 mip、单 layer 和普通 kernel dispatch
序列。多个域可借用同一图像直接采样，不复制像素。冷准备保留原有资源 lease，将普通
队列的写入和 shader-read 转换封成可复用 timeline 前缀；各域在首次使用前等待该前缀。
借用期间允许普通 sampled 读取，但拒绝上传、storage 写入、布局变更，以及需要
transfer-src 布局的 Canvas 复制；提前录制的写者同样被拒绝，即使它最终恢复只读布局。
读许可维持到实例退休，单次 completion 完成并不恢复写权限。`step.close()` 等待该
实例自己的已接纳工作并释放许可，不等待其他域；最后一个读实例退休后恢复写入。
仅释放底层 prepared handle 不等于退休，在途工作和域仍持有许可。

已有 raw VkImage 别名的图像不能进入该路线，借用期间也不能导入同一 VkImage 的新别名。
外部内存别名、外部 queue family ownership 转移、共享图像控制流及多 mip/layer 不在此能力范围。
Windows 验证覆盖双队列在途读取、关闭其中一个实例、准备失败清理、源视图删除及
Program 退出；这不是完整渲染场景的吞吐结论。

`runtime.execution_domain(queue="graphics")` 显式选择支持 graphics+compute 的队列族。
`info["queue_family_index"]` 与 `info["queue_index"]` 一起标识队列；同族 compute/graphics
共用租约池，不会重复租用同一条队列。默认 compute 选择不变；若其实际队列族具备
graphics 能力，也可以执行已准入的离屏计划。CUDA 不接纳 `queue="graphics"`。

固定平坦的私有离屏计划沿现有 native `vulkan_graphics` 描述重建 pipeline、color/depth attachment，
并在同一域内连接绘制、采样计算和下游显式梯度。每段录制恢复实例的图像边界布局；
附件、pipeline 与 workspace 保留到实例退休。暖态不逐次重建，不回默认队列提交。
图像的初始数据上传仍属于冷准备。

平直的 typed storage kernel 和 `VulkanGraphicsPipeline.record_pass()` 计划也可借用
已有的 owned 2D color/depth Texture（单 mip/layer）。目标不复制，退休后仍可从原
分配观察内容。包含写入的实例在整个存活期排他借用图像：普通读写和其他域借用均
被拒绝，直至该实例退休；纯采样源保留多 reader 共享。冷准备接收普通队列的准确
准备前缀，暖态在指定域恢复边界 layout；仅调用 `release()` 不会退休租约。

设备几何的 `RasterPass` mesh、mesh_instance、particles 和 lines recording 可通过
Graph 显式准入及 `queue="graphics"` 使用该路线。record 创建时保留选定的普通辅助
dispatch；prepare 借用其代码，为每实例重建私有 VBO/normal/index scratch。
camera/light 和 draw uniform 在 `Graph.prepare()` 时冻结；之后修改相机须关闭并
重新准备实例，普通 recording 的相机行为保持原样。提交前调用一次 `initialize()`。
不可变数据沿 AOT 相同的完整性检查，在冷准备阶段经默认 RHI compute 队列同步上传；
`initialize()` 仍等待普通队列的准备前缀。暖态绘制和计算在所选域执行，不重新编译
前端或逐次执行 host 几何上传。Windows 已验证四种 draw 及采样后计算的梯度；光栅化本身
并不可微。共享可变几何仍须与写入者建立显式依赖。

含 AS 的 pass、显示互操作和 graphics fork/join 仍需分别验证。原始图像别名、
packed depth/stencil 和借用图像控制流仍不接纳。

平直 AOT `ExecutionPlan` 可在 compute 域重建实例私有的 native `vulkan_ray`
`triangle_scene`，或 `triangle_blas` 与 `instance_tlas`。同一实例内多个 TLAS instance
可引用同一 BLAS；不同 prepared 实例分别拥有可变 AS 和 scratch。工厂只分配并准备
构建命令，显式 `initialize()` 在所选域按 BLAS→TLAS 顺序执行；此后 native `refit`、
`blas_build`、`instance_refit`、`instance_update` 和 `query` 可在同域重放。
逐提交参数可供前置 kernel 更新几何，查询结果可传给后续计算及其显式梯度；AS 遍历
本身并不可微。对应构建能力为 `vulkan_execution_domain_private_ray_resources`。

所有 seed 输入和 AS 依赖保持有效直至实例退休；提前关闭输入会使该实例失效。
已有 public AS 不能通过包装成私有 TLAS 绕过所有权检查。此能力暂不包含 live AS
借用、micromap、kernel/shader AS 参数或控制流内的 AS 操作。普通同步资源创建合同
保持不变。冷 `initial_data` 上传仍走默认队列，`initialize()` 仍等待 bootstrap；
这不表示冷准备能完全脱离默认队列，暖态 native 查询与更新无需回到该队列。

独立 Vulkan 路线还不接纳 host selection、公共并行 recipe，以及共享
sparse/RNG/runtime-error 状态。
普通 Graph 保持既有能力；CUDA Canvas 和外部内存协议不会因此自动适用于 Vulkan 域。

初始化导入普通队列已提交的准备前缀，并在所选队列执行实例初始化。暖态提交把依赖、
参数上传和固定命令放进一次 `vkQueueSubmit`；busy 不产生部分设备效果。实例退休只等待
该实例已接纳的前缀，不等待无关默认队列工作。

Windows 实机资格使用具有 16 条已创建 compute queue 的 RTX 5090，并含双向受控挂起
推进检查。缺少足够队列的设备不能仅凭构建能力位视为可用。Linux、任意 provider 组合
和应用吞吐尚未在此验证。

## 当前 JIT Graph

```python
domain = ti.graph.execution_domain(capacity=2, instance_capacity=64, priority=0)
step = graph.prepare(arguments, execution_domain=domain)
step.initialize()
ticket = step.submit(wait_for=previous_tickets)
if ticket is not None:
    # 消费结果或退休资源时再等待，不在每次仿真与绘图交接时等待。
    ticket.wait()
step.close()    # 冷阶段，退休该实例并回收驻留预算
domain.close()  # 冷阶段，排空并退休资源
```

`arguments` 接受字典或该 Graph 的 `GraphBindingSet`，在准备时冻结；后续更新绑定集
不会修改已有实例。准备消费完整的物理 dispatch、条件/循环和并行分支描述，保留每个
调用点的启动策略，不导出文件、重新编译前端或复制 Field 根。不同准备拥有独立的参数、
控制缓冲区和声明的私有工作区；共享应用 ndarray 和 Field 仍需精确依赖。

CUDA 当前覆盖 scalar/tensor JIT ndarray、live dense Field/ndarray 的紧凑 AOS
`ti.experimental.ndarray_view()` 范围、隐式 dense Field 访问、普通与 masked 控制及
选定的 CUDA 并行区域。ndarray 的已有 `.grad` 同时借用，支持显式反向内核；不参与
自动 Tape/FwdMode，初始化/提交会排斥该上下文及其并发进入过程。源 Graph 的 `close()`
不会关闭已准备实例，Field 树销毁或 Program reset 则先退休相关实例再释放代码/存储。

CUDA 固定顺序 Graph 也支持嵌套 ArgPack：prepare 冻结设备上的纯值，各实例拥有私有包存储，
ndarray 成员沿已有合同借用。sampled Texture 可作为普通参数或包成员借用，同一源图像
只导入一次；不会复制像素。源图像应先初始化，更新像素与采样之间仍需显式依赖。
已验证源 Graph/包释放及双域独立关闭；其他包含包的控制 recipe 需分别取得资格。

显式参数及 Graph 保留的 dense view 均保留切片偏移和元素形状，不复制字段根，也不要求
应用将 Field 改成新 ndarray。借用 AOT runtime 的 `runtime.import_dense_view(view)`
在 CPU、CUDA、Vulkan 使用同一存储合同。字段树销毁先排空相关执行，再撤销借用及其
全部派生别名，旧句柄仍可安全关闭；ndarray 借用则在源 Python 包装回收后保留分配。
导入属于冷操作，不能嵌套在 borrowed submission scope 内。

普通主机/设备传输和 CPU/CUDA 指针导出保留有限视图的偏移及大小；底层可映射内存的 map
同样应用偏移。旧 CGraph 参数不能表达非零偏移，应使用 kernel 或完整 execution plan，
不能通过旧 `module.graph()` 调用忽略切片。旧 Vulkan memory export 的 offset 描述
VkDeviceMemory 绑定位置，因此拒绝非零 buffer-relative 偏移。两者均在设备提交或
暴露父分配前拒绝。

带同步域的 CUDA 外部视图也可借给 JIT 和已加载的 AOT recipe。有限切片及其别名保留原
同步所有者；完整初始化或重放各自对每个外部所有者消费一次 acquire/release epoch，
不因多个内核或多个别名重复消费。仅操作私有存储的初始化不消费重放阶段的信号；若两个
阶段都访问外部存储，生产者须在阶段之间重新交还所有权。busy 或拒绝准入不消费信号。
普通 AOT 调用、内存复制/映射/导出及默认 prepared 路径会拒绝这些托管导入；冷准备时
可能读取外部输入的 native resource factory 尚未资格。关闭外部所有者属于冷操作，先
排空相关执行域再撤销全部借用和别名。即使相关前缀已完成，CUDA/vendor 的实际显存销毁
仍可能同步无关工作。其他域运行时需要暂停，应停止准入并等待该域的最后完成票据，保留
有界实例和图像池；实际销毁留到冷关闭边界。Canvas 选域及源完成票据仍是另一项接口边界。

绑定帧、多 workspace lane、动态选择 recipe、尚未具备借用描述的 provider data、
非紧凑/只读或无同步域的外部存储视图，以及未资格的 native route 会在冷准备时拒绝，不回退到默认 stream。
默认 `Graph.run()`/`submit()` 合同不变。`ti.graph.execution_domain()` 的借用 runtime
由 Program 的 Python owner 持有，避免丢弃最后一个域包装时意外触发破坏性关闭。

独立 AOT runtime 另提供 [Vulkan 外部 buffer 导入](vulkan_external_buffers.zh.md)，
以原生 ready gate 与 whole-buffer ownership 在所选域消费 Win32 共享缓冲。其
初始化和公开入口边界见专页；不能把 CUDA 的已完成信号规则套用到 Vulkan gate。

## 逐提交参数

在冷准备时声明可更新参数，其他绑定继续冻结：

```python
names = ("epoch", "simulation_step", "timestamp_ns")
step = graph.prepare(arguments, execution_domain=domain, mutable_parameters=names)
step.initialize()
ticket = step.submit(values={"epoch": epoch, "simulation_step": step_id,
                             "timestamp_ns": timestamp}, wait_for=previous_tickets)
```

`module.prepare_execution(..., mutable_parameters=names)` 提供相同的 AOT 合同。
首版支持仅用于重放内核的 `u64` scalar；初始化阶段使用的 scalar、数组或其他类型在
准备时拒绝。名称和数量由调用方声明，每次提交必须完整提供，值必须在 `[0, 2**64)`。
接受提交后，修改原字典不影响该次执行。预分配的 staging 参数和设备参数区属于实例，
上传与完整 Graph 都进入所选域，completion 精确覆盖二者。
CUDA 上传使用读取 pinned 存储的小型 GPU kernel，避免 DMA 拷贝引擎的跨流队头阻塞。
冷准备要求 CUDA unified addressing；重放不分配 pinned 内存。
Vulkan 使用预备 staging 与 buffer copy，在租用队列上显式安排 transfer/compute 依赖，
上传不会绕到普通队列。

CUDA 实例只有一份可变参数存储，在途时再次提交返回 `None`。Vulkan 按实例及队列槽位
预分配 staging，同一实例已接纳的提交在该队列依次执行，不覆盖此前快照。需要独立可变
工作区时在冷阶段准备多个实例；域容量由调用方配置。busy 或 ready-only 未就绪不上传、不消费外部信号、
不分配提交序号。重放不重新 prepare、不创建设备参数存储、不进行全局同步。
CUDA 和 Vulkan 固定 buffer-only 路线共用上述合同。

## Completion GPU 计时

计时默认关闭。创建域时指定 `gpu_timing=True`，为每个队列槽位预分配一对事件；
提交时使用 `sample_gpu=True` 选择需要观测的提交：

```python
domain = ti.graph.execution_domain(capacity=2, gpu_timing=True)
step = graph.prepare(arguments, execution_domain=domain, mutable_parameters=names)
step.initialize()
ticket = step.submit(values=metadata, sample_gpu=sample_this_step)
if ticket is not None:
    measurement = ticket.gpu_timing
```

`measurement` 包含 `status`、`scope`、`owner`、`sequence` 和 `duration_ns`。
状态为 `ready` 时才有时长；`not_sampled`、`pending`、`unsupported` 或 `failed`
的时长均为 `None`。读取仅进行非阻塞查询，不等待 GPU、不持有提交锁。
结果与具体票据关联，事件复用或域关闭后仍保持不变。

范围 `execution_envelope` 是完整执行区间，排除前驱依赖等待、外部 acquire、参数上传和
外部 release；包括 Graph 内部间隙以及 provider/Graph 内部声明的等待，因此不是逐
kernel 耗时之和。不会拆开原 Graph，也不切换到默认域。AOT 使用相同接口。
CUDA 与 Vulkan 独立域均提供上述计时。Vulkan 要求所选 queue family 支持 timestamp，
不支持时采样返回 `unsupported`，没有时长。普通 Vulkan Graph 已有的计时路线保持原合同。

## Canvas 显式显示域

用 `canvas.set_execution_domain(output)` 绑定当前 Program 的 CUDA 域。仿真开始前调用
`views = canvas.prepare_frame_pool(width, height, capacity=3)`，为各物理 view 准备完整
Graph/AOT 写入实例并初始化；容量上限为 renderer 的在途数加最后显示图像。返回 view
只用于提前绑定，只有 `acquire_frame()` 当前借出的同一图像才可写，不应逐帧 prepare。
预备池不在准入时扩容；可用图像均有在途读取者时返回 `None`。重复显示复用空闲别名，
保留最后显示图像。resize 时先关闭旧输出域、绑定新域，再调用 `prepare_frame_pool()`
并准备新写入实例；这个冷重配过程会退休旧 Vulkan 读取者。

```python
canvas.set_execution_domain(output)
views = canvas.prepare_frame_pool(width, height, capacity=3)
writers = {
    id(view.descriptor): graph.prepare({"pixels": view}, execution_domain=output)
    for view in views
}
for writer in writers.values():
    writer.initialize()

frame = canvas.acquire_frame(width, height)
if frame is not None:
    ticket = writers[id(frame.pixels.descriptor)].submit(wait_for=dependencies)
    if ticket is None:
        frame.cancel()
    else:
        canvas.submit_frame(frame, source_completion=ticket, track_source=True)
        window.show()
```

传入票据须覆盖这一帧的完整写入。原生层检查活 runtime、CUDA stream 及图像释放者；
source tracking 保留该精确票据，不记录默认队列尾部，也不纳入所选域后来提交的工作。
`frame.source_completion` 覆盖源读取，`frame.completion` 覆盖 Vulkan 消费，不代表扫描呈现。
空帧只在所选队列完成 semaphore epoch；repeat 使用同一队列，不增加像素 kernel。
满容量直接返回 busy，不消费 semaphore 或封存 writable lease；取消本身不提交 CUDA
工作，再用图像时按需执行 Vulkan rearm。

所选域的 `DisplayFrame.from_packed_u32_ndarray()` 复制同样接受 `source_completion`
作为设备前驱，返回的源票据覆盖这次复制。稳态前预热有限输入集；每张物理输出图像最多
保留 64 个源 ndarray 租约至冷退休，避免丢帧/最小化或完成查询提前释放在途复制源。
busy 不增加源租约。通用 `set_image()` 打包、texture/host 输入不是选域路线，应使用
prepared writer 或 packed buffer。

共享存储 resize 属于冷边界：关闭旧输出域，建立并绑定新域，再准备新 view 的写入实例。
直接替换仍绑定活域的存储会在退休前拒绝。reset 使旧租约失效；窗口关闭先排空依赖的
CUDA/外部访问，再释放 Vulkan 设备。无法完成退休证明时保留有界图形资源闭包，不提前释放。

### Vulkan retained packed copy

Vulkan Program 也可用 `canvas.set_execution_domain(output)` 选择独立输出域。
这条路线接收同 Program 的二维 `u32` ndarray；在冷阶段用
`canvas.prepare_frame_sources(frames, capacity=3)` 冻结源集合并分配显示槽。
容量由调用方选择，至少两个槽以保留最后显示图像并接收下一帧；不设置额外的源数量或
槽数量上限，实际分配、地址和设备存储缓冲范围仍须合法。所有源必须具有相同形状。

```python
frame = ti.ui.DisplayFrame.from_packed_u32_ndarray(pixels)
canvas.set_execution_domain(output)
canvas.prepare_frame_sources([frame], capacity=3)  # 冷阶段

producer = writer.submit()  # 同 Program 的 output 或 peer 域
if producer is not None:
    display = canvas.submit_frame(
        frame, source_completion=producer, track_source=True, track_completion=True
    )
    if display is not None:
        copied = frame.source_completion
        window.show()
```

每次接受执行一次设备内复制；复制在选中的输出域运行，图形队列等待该次复制的精确
timeline 点。`copied.done()/wait()` 覆盖源读取完成，调用方须在复用或改写源数组之前
遵守这一依赖；该原生票据目前不是 Graph `wait_for` 的公共输入。
`display` 覆盖图形消费，不代表屏幕呈现。repeat 使用已有显示槽，不再次复制源。
忙碌时返回 `None`，不排队复制、不替换 `frame.source_completion`；取消待显示帧也不
取消已接受的复制，其槽和源租约保持到复制完成。

当前选定 Vulkan 域支持 prepared packed copy 和下述 prepared writable 路线。
混合几何、texture/host 输入和 runtime offscreen attachments 尚未接通，
在对应入口拒绝。窗口销毁退休自己的复制实例，不关闭整个输出域。始终只使用该选定
Vulkan 图像路线的窗口，等待自己的复制与 graphics readers，不排空无关的默认计算或
输出域尾部。曾使用普通显示路线的窗口仍保留保守设备排空。可见窗口仍需完成
swapchain/present 退休，物理队列共享可能带来额外等待；读回、重配和 Program reset
保留原有同步。缺少退休证明时保留窗口资源闭包，不释放可能仍在途的存储。
源集合或形状变化时关闭旧域、绑定新域并重新准备。这里没有额外像素处理 kernel，
但相对直接写显示槽多一次设备复制；不能据此推断应用帧率提升。

### Vulkan prepared writable 显示槽

选定 Vulkan 输出域后，可用 `canvas.prepare_frame_pool(width, height, capacity=3)`
冷准备本地显示槽，并为返回的每个物理 view 准备 Graph/AOT writer；使用上面的 CUDA
池示例，但 Vulkan 的已写帧必须传入该 writer 的精确 `source_completion`，无论是否
开启票据跟踪。writer 必须属于同一个输出域，普通 kernel/default launch 不能直接写。
每次 `acquire_frame()` 允许一个完整 prepared recipe；其中可含多次 dispatch 和该槽的
多个别名。错误、较晚或旧租约的 completion 会在封存前拒绝，当前租约仍可重试。

这条路线直接写显示存储，不执行 packed-copy 的额外复制。graphics 等待精确 writer，
槽只有在 writer 和全部 graphics readers 完成后才能复用。取消不取消已接受的 writer；
repeat 不重新提交 writer 或空 marker。未写入的帧可不传 producer：首次内容为冷阶段
清零值，后续保留该槽已有内容；空帧提交仍服从输出域容量，Busy 不消费租约。
容量由调用方决定，至少两个槽用于保留最后图像；只检查实际分配、地址、设备范围和
shader 索引约束。没有额外槽数上限，也不在热路径扩容。

窗口关闭只撤销使用这些显示槽的实例，不关闭同域的其他实例。resize 先关闭旧域，
绑定新域并重新准备。reset 后遗留 view/ticket 只保留已退休的安全对象，不能继续写。
packed-copy 和 writable 是两种冷准备模式；不能在活域中切换模式或替换存储。此能力
不等于完整 下游应用 GUI 路由或应用帧率资格。

### 图像与 GUI widgets

选定 Vulkan 显示域可叠加当前窗口的 GUI widgets。首次 `window.get_gui()` 在冷阶段
准备字体及兼容的绘制资源，可能等待 graphics queue；应在稳态循环和受控队列等待前
调用。后续相同资源的 `get_gui()` 不清除本帧已添加的 widgets。resize 或 renderpass
变化会重新进入冷准备。`repeat_frame()` 复用图像，widgets 仍由应用逐帧描述。
可见窗口先调用 `submit_frame()` 或 `repeat_frame()` 接纳图像，再添加本帧 widgets，
最后调用 `show()`。

每份 GUI vertex/index 存储都保留该次 graphics 提交的精确 reader。槽未完成时不覆盖
或扩容；非阻塞 `show()` 没有可用槽会返回 `False` 并丢弃本次待显示帧。阻塞读回只
等待需要的 reader。普通窗口与既有普通 offscreen GUI 共用这一规则；这不改变选定域
对 geometry/runtime offscreen 的限制，也不自动开放 下游应用 完整应用路线。

## 准备、初始化与提交

无前端的独立 runtime 还提供 `runtime.import_cuda_external_memory()`，导入同一 GPU
外部分配中的有限线性缓冲区。参数包括 OS memory handle、分配大小和设备 UUID、二进制
`ready_for_cuda`/`ready_for_external` semaphore 对、dtype/shape 和可选字节偏移。
成功和失败都保留调用者原始 OS handle；runtime 只导入自己的副本。外部分配、设备和
semaphore 对象须保持到 imported memory 关闭，生产者还要自行排空其已提交工作。
每个物理 semaphore 对只导入一次，别名用 execution-plan view 表达；普通 host transfer
和默认 launch 不能绕过外部访问协议。

每次接受的完整配方消费一个 ready 信号并发布一个 consumed 信号。生产者必须先把
ready 信号**提交到 Vulkan 队列**，再调用 CUDA submit；信号可以仍在设备上等待。
CUDA submit 返回后才提交外部 consumer wait。只把请求放入另一个主机线程的队列，
还不能证明 Vulkan 提交已经发生。该协议不要求主机等待 GPU 完成，见
[CUDA 外部 semaphore 顺序](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html)。
当前实机资格限 Windows CUDA/Vulkan；存在 FD 导入代码不等于已验证 Linux 窗口。

无源码 consumer 直接使用 runtime 包；需要共享现有 JIT owner 时，显式通过
`ti.aot.borrow_runtime()` 获得同一套接口。假设模块导出了 `step`，并已准备参数字典：

```python
from taichi_forge_runtime.aot import Runtime

with Runtime("cuda") as runtime:
    module = runtime.load_module(module_path)
    domain = runtime.execution_domain(capacity=2, instance_capacity=64, priority=0)
    step = module.prepare_execution("step", arguments, execution_domain=domain)
    step.initialize()  # 冷阶段：发布初始化后才允许 submit

    ticket = step.submit()
    if ticket is not None:
        # 此处可以继续其他主机工作；等待是调用方的显式选择。
        ticket.wait()
        ticket.close()
    step.close()      # 冷阶段，退休该实例并回收驻留预算
    domain.close()    # 冷阶段破坏性退休
```

即使计划没有显式初始化操作，也必须调用 `initialize()`。它发布包含模块和存储准备的
精确默认队列前缀，再在所选域执行初始化，不调用 runtime 的全域等待。初始化事件和存储
保留到有界实例的冷退休阶段，完成时不立即销毁驱动对象。应用启动流水前应准备并初始化
所有稳态绑定。

`submit(wait_for=[ticket, ...])` 校验精确前驱、执行有界准入并入队完整计划。接受时返回
`Completion`，满容量时返回 `None`；满容量不入队依赖等待或部分工作。`submit()` 和选域
后的 `run()` 都不隐式初始化；`run()` 丢弃返回票据，容量满时抛错。

票据归属于一个 runtime generation，提交依赖时会拒绝外来或 reset 后的 owner。`done()` 查询和
`wait()` 等待不获取 runtime 提交锁，等待线程不会因此占住同一票据的查询或其他域的提交。
`domain.latest_completion()` 只快照已经接受的前缀，后续提交不会延长这个票据。关闭票据
仅释放主机句柄，不取消 GPU 工作。

## 所有权和内部状态

冷准备消费持久化的编译事实。已经资格验证的 dense field、`real_func`、向量参数和
跨 task 临时量沿用现有编译器及 AOT loader，BC/PTX/cubin 均保存这些事实。缺少事实的旧
产物，以及使用尚未隔离的共享 RNG、runtime error、sparse list/allocator、打印、mesh
或未资格外部调用的 kernel，会被独立域拒绝。旧默认队列 API 维持原合同。绑定帧、storage
ring 和 host selection 不会被悄悄改写为其他物理路线；native operation 也必须明确允许
独立域路线。

应用数据 hazard 由调用方负责。域不会自动推断数组视图或 dense field 是否 alias；所有
重叠读写和写写都需要显式依赖，包括 snapshot 覆写和槽复用。

runtime 在冷阶段按需增长域注册存储，并复用已关闭域的位置，没有固定的域数量上限。
后端或主机资源不足时，创建 stream 及其存储仍可能失败。

`capacity` 限制在途提交，`instance_capacity` 限制驻留的逻辑 prepared 实例。一份计划的
初始化和 replay 共同消耗一个实例名额，仍各自保留所需 recording；部分构造失败沿同一
owner 退休已建立的部分。实例预算拒绝，或未接纳实例的部分构造已安全清理时，只撤销
失败实例，不关闭域，也不使已有实例失效；已接纳后的准备失败或无法证明清理安全时，
仍采用域级冷退休。

`step.close()` 只停止该实例，等待它最后一次已接纳提交，退休 recording 并移除存储及
依赖边，随后可复用驻留实例名额。已有 completion 票据仍有效；等待不持有 runtime
提交锁。驱动/provider 析构仍可能产生同步，因此 close 属于冷边界，外部生产者必须
仍能发出已经提交的等待所需的信号。

CUDA 逐提交参数的 staging 在最后一次设备读取完成后归还域所属的锁页池，供后续准备复用。
空闲映射不足时在冷准备阶段替换，不为每一种历史大小持续积累分配。池保留并发使用的
高水位存储到域关闭，避免每次实例退休解除锁页时同步其他队列；参数上传和重放路径不变。

`step.release()` 和垃圾回收只释放主机句柄，代码、数组及反向依赖边保留到域退休，
也不回收驻留实例名额。显式销毁内存时，先退休借用该分配的实例再释放存储，域及无关
实例仍可使用。显式销毁模块或 native resource 时，先关闭受影响的域；runtime
close/reset 退休所属域。无法证明安全退休时保留有界资源闭包并报告
失败，不能提前释放在途对象。

`domain.release()` 只释放主机句柄，runtime 仍保留有界资源。`domain.stop()` 永久停止
准入。需要恢复的暂停应仅停止主机提交、保留域，并等待应用实际需要的精确前缀。
`domain.close()` 是破坏性冷操作，驱动或 provider 清理可能产生同步；已提交外部等待的
生产者必须仍能发出信号。

## 流水调度

独立仿真队列不能消除真实数据依赖。用设备票据表达
`simulation(n) → snapshot(n) → render(n)`；还必须在 live state 覆写前建立
`snapshot(n) → simulation(n+1)`，并在 render ticket 完成后才复用 snapshot 槽。
主机队列保持有界。显示忙时保留最近完成的画面并丢弃冗余呈现请求，避免阻塞仿真准入。

先使用同优先级，分别限制绘图与物理的在途数。显示 acquire 忙时跳过此次呈现，继续满足
依赖的物理工作；只保留最近可用的 snapshot，同时遵守租约及精确 render completion 的
复用条件。分别测量物理和输出进度后再调整优先级。本机 RTX 5090 的合成 ABBA 矩阵中，
提高物理优先级改善了长计算的物理进度，但降低输出进度并增加两项工作全部结束的耗时；
短 kernel 主要受主机提交限制。优先级是显式调度取舍，不能自动等同于吞吐优化。

CPU `wait()` 和字节缓冲区主机传输是显式同步边界，适用于初始化、最终读回与关闭。
若放在每次仿真与渲染提交之间，就会把流水重新串行化。CUDA 完成与 Vulkan 呈现、屏幕
扫描是不同边界。Canvas 选域已通过 Windows CUDA/Vulkan 真实窗口和显式挂起消费者验证。
三槽 1000 帧有头捕获中，热 API 没有 CUDA 同步/分配释放、Vulkan 显式主机等待或共享
图像增长；repeat 不启动像素内核。Vulkan 仍逐提交建立 fence/semaphore/framebuffer
元数据，驱动提交和呈现也有主机耗时，不能宣称呈现零成本或应用吞吐已通过。
独立无头长跑检查每个版本，有头长跑检查实际呈现提交及最终像素版本。

后端创建/准备接口分别位于 `taichi_cuda.h` 和 `taichi_vulkan.h`，包括
`ti_prepare_execution_plan_on_domain_ext` 与 `ti_prepare_execution_plan_on_vulkan_domain_ext`。
共同初始化、提交、completion 与退休接口位于 `taichi_core.h`。
成功提交返回调用方拥有的 completion；返回 false 且无 C API 错误表示 busy，false
伴随错误表示拒绝或失败。公开票据的主机登记在产生设备效果之前预留。
