# CUDA 执行域（开发中）

[English](cuda_execution_domains.en.md)

独立 AOT consumer 可以将保存的完整执行计划绑定到有界 CUDA 执行域。每个域拥有
nonblocking stream，每个准备好的实例拥有自己的 executable、不可变参数和编译临时区。
该接口保留已经选择的完整计划。stream、优先级和完成票据只属于运行时，不写入 AOT 产物。

JIT Graph 也可以冷准备为独立实例，借用已选定的编译代码和当前 Program 的 dense Field。
这是 F3 的一部分；Canvas 显示链、跨平台真实窗口和应用 FPS 仍未完成资格验证。
优先级是设备调度提示，不能保证 kernel 并行或抢占。

## 当前 JIT Graph

```python
domain = ti.graph.execution_domain(capacity=2, instance_capacity=64, priority=0)
step = graph.prepare(arguments, execution_domain=domain)
step.initialize()
ticket = step.submit(wait_for=previous_tickets)
if ticket is not None:
    # 消费结果或退休资源时再等待，不在每次仿真与绘图交接时等待。
    ticket.wait()
step.close()    # 仅释放主机实例句柄
domain.close()  # 冷阶段，排空并退休资源
```

`arguments` 接受字典或该 Graph 的 `GraphBindingSet`，在准备时冻结；后续更新绑定集
不会修改已有实例。准备消费完整的物理 dispatch、条件/循环和并行分支描述，保留每个
调用点的启动策略，不导出文件、重新编译前端或复制 Field 根。不同准备拥有独立的参数、
控制缓冲区和声明的私有工作区；共享应用 ndarray 和 Field 仍需精确依赖。

当前覆盖 scalar/tensor JIT ndarray、live dense Field/ndarray 的紧凑 AOS
`ti.experimental.ndarray_view()` 范围、隐式 dense Field 访问、普通与 masked 控制及
选定的 CUDA 并行区域。ndarray 的已有 `.grad` 同时借用，支持显式反向内核；不参与
自动 Tape/FwdMode，初始化/提交会排斥该上下文及其并发进入过程。源 Graph 的 `close()`
不会关闭已准备实例，Field 树销毁或 Program reset 则先退休相关实例再释放代码/存储。

显式参数及 Graph 保留的 dense view 均保留切片偏移和元素形状，不复制字段根，也不要求
应用将 Field 改成新 ndarray。借用 AOT runtime 的 `runtime.import_dense_view(view)`
在 CPU、CUDA、Vulkan 使用同一存储合同。字段树销毁先排空相关执行，再撤销借用及其
全部派生别名，旧句柄仍可安全关闭；ndarray 借用则在源 Python 包装回收后保留分配。
导入属于冷操作，不能嵌套在 borrowed submission scope 内。

带同步域的 CUDA 外部视图也可借给 JIT 和已加载的 AOT recipe。有限切片及其别名保留原
同步所有者；完整初始化或重放各自对每个外部所有者消费一次 acquire/release epoch，
不因多个内核或多个别名重复消费。仅操作私有存储的初始化不消费重放阶段的信号；若两个
阶段都访问外部存储，生产者须在阶段之间重新交还所有权。busy 或拒绝准入不消费信号。
普通 AOT 调用、内存复制/映射/导出及默认 prepared 路径会拒绝这些托管导入；冷准备时
可能读取外部输入的 native resource factory 尚未资格。关闭外部所有者属于冷操作，先
排空相关执行域再撤销全部借用和别名。Canvas 选域及源完成票据仍是另一项接口边界。

绑定帧、多 workspace lane、动态选择 recipe、尚未具备借用描述的 provider data、
非紧凑/只读或无同步域的外部存储视图，以及未资格的 native route 会在冷准备时拒绝，不回退到默认 stream。
默认 `Graph.run()`/`submit()` 合同不变。`ti.graph.execution_domain()` 的借用 runtime
由 Program 的 Python owner 持有，避免丢弃最后一个域包装时意外触发破坏性关闭。

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
    step.close()      # 仅释放主机句柄，不等待
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

`capacity` 限制在途提交，`instance_capacity` 限制驻留 recording。带显式初始化的计划
可能同时保留初始化和 replay 两个 recording。释放 prepared 句柄后，代码、数组及反向
依赖边仍保留到域退休。显式销毁模块、内存或 native resource 时，先关闭受影响的域，再
释放存储。runtime close/reset 退休所属域；无法证明安全退休时保留有界资源闭包并报告
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

CPU `wait()` 和字节缓冲区主机传输是显式同步边界，适用于初始化、最终读回与关闭。
若放在每次仿真与渲染提交之间，就会把流水重新串行化。CUDA 完成与 Vulkan 呈现、屏幕
扫描是不同边界；external-display/Canvas 链仍需单独接入并验证，才能替换有头路径。

对应 C 接口在 `taichi_cuda.h`：`ti_prepare_execution_plan_on_domain_ext`、
`ti_initialize_prepared_ext`、`ti_try_submit_prepared_ext` 和域/票据句柄接口。
成功提交返回调用方拥有的 completion；返回 false 且无 C API 错误表示 busy，false
伴随错误表示拒绝或失败。公开票据的主机登记在产生设备效果之前预留。
