# 完成集合等待

[English](completion_wait_sets.en.md)

`CompletionSet` 等待一组冻结的精确提交前缀中任意一项完成。普通 JIT Graph 票据、
普通 AOT `SubmissionTicket` 和独立域 `Completion` 共用这一实现；CPU、CUDA、Vulkan
沿用各自已有的完成原语。独立执行域目前仍限 CUDA，本接口不增加 Vulkan 队列。

使用前检查 `runtime_contract_manifest()["features"].get("completion_wait_any", False)`。
runtime-only 消费端无需导入 DSL：

```python
from taichi_forge_runtime.aot import CompletionSet

with CompletionSet([input_done, render_done], poll_interval=0.001) as progress:
    result = progress.wait(timeout=0.02)
    if result.status == "ready":
        consume_completed_prefix(result.index)
    elif result.status == "timeout":
        do_other_host_work()
```

前端的 `ti.graph.CompletionSet` 是同一个类。`result.index` 是首个已完成成员在原输入
中的索引；`timeout`、`cancelled` 时为 `None`。后端完成查询失败会抛出原有错误，
不会伪装为完成、超时或 GPU 时长零。

## 生命周期与取消

- 构造时快照原生 completion，随后可以关闭原票据的主机句柄；后续提交不延长这些前缀。
- `wait(timeout=None)` 持续等待，零表示只做一次非阻塞扫描。可反复查询同一集合，
  报告完成不会消费成员。
- `cancel()` 永久取消此集合的等待并唤醒所有等待者，不取消 GPU 工作，也不归还应用的
  输出或 workspace 租约。
- `close()` 取消并释放集合；已进入原生等待的调用保留自己的引用。新等待与关闭竞争时
  可能收到句柄已关闭错误。
- 等待和取消均不获取 runtime 提交锁，不初始化资源、不重新准备 Graph、不提交设备工作。
- 报告 JIT 成员完成时保留原票据的 pacer 和 completion observation 处理。集合只持有
  完成状态，不复制应用输出。

成员须来自同一个已加载 native runtime 库，可以跨 runtime 实例和已支持后端。
owner 关闭仍按原有退休合同；保留的集合之后仍可观察已完成状态或失败。

## 观察策略

本接口采用轮询型 wait-any。只有调用线程扫描成员并在扫描之间睡眠，不建立逐票据线程、
后台服务、CUDA host callback 或新 Graph 节点。成员数量没有人为上限；公共 C ABI 的
计数字段为 uint64，实际受主机可寻址存储约束。

`poll_interval` 以秒配置，默认 1 ms。较小间隔增加查询次数与 CPU 唤醒，缩短观察延迟。
间隔不是实时保证，线程调度和后端查询还会增加延迟。Windows 使用高分辨率 waitable
timer 与取消事件，不修改全局计时分辨率；旧系统缺少该能力时沿用普通 timer 粒度。
其他平台使用条件变量等待；本轮暂不做 Linux 资格验证。

同一组前缀可以复用集合；新提交需要新集合，仅分配主机登记，不重新准备 GPU 实例。
Windows 等待者首次需要睡眠时创建一个 timer，与票据数量无关。立即完成和零超时扫描
不创建等待 timer。
