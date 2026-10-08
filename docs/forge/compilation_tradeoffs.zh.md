# 编译与高级优化权衡

[English](compilation_tradeoffs.en.md)

> 适用范围：当前源码文档。请按安装版本核对[版本与安装说明](index.zh.md#版本与安装)。

本文说明如何缩短 Taichi Forge 冷编译，同时避免悄悄牺牲生产吞吐、数值可信度或
自动微分覆盖。缓存与复用机制见[编译与缓存说明](cache_compile.zh.md)，完整 Forge
配置清单见 [Forge options](forge_options.zh.md)。

## 推荐决策顺序

生产环境建议按以下顺序决策：

1. 先保证正确性、内存安全和后端结果一致性。
2. 再保证工作负载真正关心的稳态吞吐和延迟。
3. 优先通过 cache、预编译和局部 compile tier 降低冷启动。
4. 只有在有测量依据时，才把关闭大类优化作为诊断或明确的部署 profile。

不要只比较首次启动时间。一次节省 30 秒、但让长时间仿真慢 10% 的设置通常得不偿
失；同一设置对短命令行工具或高频 edit-run 开发循环却可能合理。

## 各配置并不等价

| 配置 | 作用范围 | 主要收益 | 主要代价或风险 |
| --- | --- | --- | --- |
| `offline_cache=True` | 相同后端与编译配置 | 后续进程可免去未改变产物的重复编译 | 首次运行仍需编译；源码、shape、layout、后端或进入 key 的配置变化都会 miss |
| `ti.compile_kernels(...)` | 指定 specialization | 把编译移到热循环前 | 不会减少编译工作量；参数必须有代表性 |
| `compile_tier='fast'` | Python `ti.init()` 默认，或 `@ti.kernel(opt_level='fast')` 指定的单 kernel | CPU 使用 LLVM O0，CUDA/AMDGPU 使用保证正确 lowering 的 O1 下限，SPIR-V 跳过 optimizer | 可能降低 kernel 吞吐并改变浮点舍入；必须测稳态工作负载 |
| `compile_tier='balanced'` | 显式选择的 Program 或单 kernel | 面向生产的折中；LLVM/SPIR-V 保持配置的优化级别 | 冷编译工作多于 `fast` |
| `compile_tier='full'` | Program 或指定 kernel | 默认 global IR cap 未显式改动时，允许全局简化迭代到 fixed point | 编译代价最高；只用于已证明有运行期收益的热点 |
| `advanced_optimization=False` | 大范围 Taichi IR pipeline | 可显著缩短病态 IR 简化，也可隔离 optimizer 故障 | 会成组关闭 LICM、whole-kernel CSE、CFG optimization、store/load forwarding 等；不是细粒度生产调参开关 |
| `debug=True` 及越界/AD validation | Program | 更强诊断与安全检查 | 改变生成代码和运行成本；debug 与 release 必须分开测量 |
| `kernel_profiler=True` | 运行期测量 | 把设备时间归因到 kernel | profiler 可能增加同步或 instrumentation；不能不加说明地把 profiler-on 数字当发布延迟 |

`compile_tier`、`advanced_optimization`、debug 状态、后端 optimizer level 等会改变
代码的配置已经进入 Forge offline-cache identity。切换它们应生成或加载独立产物，而
不是复用不兼容 cache。

在当前 Taichi Forge 源码中，`debug=True` 只会在未显式指定
`check_out_of_bound` 时启用越界检查。传入 `check_out_of_bound=False`，或设置
`TI_CHECK_OUT_OF_BOUND=0`，可以单独隔离 bounds-check 成本，同时保留其它 debug
行为。这是面向诊断或已经验证过的应用 bounds contract 的定向控制，不是通用的生产调优
默认值：关闭检查后，非法索引将恢复为后端未定义行为。

## 嵌套静态循环

嵌套 `ti.static` 循环在 fast 档位仍会展开。`8 x 8` 正常支持，编译成本还取决于
循环体大小及更深的嵌套。AST 降低、常量折叠和原子操作降级通过批量替换 IR 避免
重复扫描块；展开本身、后端代码生成以及其它 pass 仍会随生成的程序规模增加耗时。

前端还会在展开重复访问同一源码位置时复用源码片段排版，同一内联函数模板的副本也可共享。
缓存归属于源码模板，仅保存文本，不持有 AST 节点或特化值；错误位置和下划线保留原格式。
这减少了诊断信息的重复格式化工作，不改变生成的程序。

`unrolling_kernel_warning_limit` 是软性源码展开提示，默认 1024 条展开语句，
同一次编译的内联函数共享计数。只统计实际访问的语句，不把静态 break 或跳过分支
后的假想迭代计入。提示包含函数名、源码位置，继续编译并保留循环语义。
此计数用于估算展开量，并非原生 IR 规模，也不直接判定写法低效。
将它与 `unrolling_limit` 同时设为 `0` 可关闭展开告警；两个显式硬上限仍默认 `0`。

## `real_func` 的定位

普通辅助函数继续使用 `@ti.func`，便于共享 CPU/CUDA/Vulkan 代码和自动微分。
`@ti.real_func` 适合显式用于需要运行时递归，或反复展开较大公共函数体的 LLVM 工作负载。
[最初的提案](https://github.com/taichi-dev/taichi/issues/602) 要解决的是 IR 重复和递归。
加速 AST 访问和 IR 改写可以降低展开成本，但没有消除这两个需求。

Forge 在同一 Program 内按特化与编译档位缓存 real function 的前端和 Taichi IR，
`ti.reset()` 后重新构建。LLVM 仍在调用它的各个模块中生成函数，因此不能保证跨所有
kernel 的机器码只编译一次。函数内部的静态循环仍会展开，不同的 `ti.template()`
实参仍会产生特化。Graph replay 减少重复提交 kernel 的成本，与 kernel 内的函数边界互补。

后端边界是 LLVM（CPU/CUDA），没有 Vulkan 实现；梯度 kernel 会拒绝 real function。
函数体在每个调用线程内部串行执行，设备调用、参数缓冲和递归栈可能增加运行成本。
纯标量或简单递归通过，不代表 Field、ndarray 参数路径已经验证。
当前 0.6.4 开发源码已修复 CUDA 模板 Field 的根绑定问题；回归覆盖嵌套和递归调用、
返回值、独立 Field 树、Graph 重放及依赖退休。采用这条路线前仍须验证实际使用的
参数形式和生命周期模式。

保持 `auto_real_function=False`。它根据累计前端展开时间进行单向提升，并不衡量运行期
收益，应保留为实验能力。对显式 `real_func` 改动，分别测冷预编译、首次启动和暖态完成
耗时；小辅助函数在没有收益证据时继续使用 `@ti.func`。

## 何时使用 `advanced_optimization=False`

Taichi 官方 global settings 文档说明，关闭 advanced optimization 可以节省编译时间并
减少部分潜在错误；官方 debugging 文档也建议用它判断编译失败是否由 optimizer 引起。
这是一项诊断能力，不代表运行性能不变：

- 适合隔离 compiler crash、invalid IR 或极端冷编译离群点。
- 对冷路径、串行、launch-bound 或 I/O 主导 kernel，经测量后可以成为部署 profile。
- 未做稳态 CPU/CUDA/Vulkan benchmark 前，不应把它设为 solver、renderer、sparse
  traversal 或 reduction 的全局默认值。
- 必须重跑数值和梯度检查。关闭优化应保持语言语义，但 instruction selection 与浮点
  reassociation 机会变化可能改变舍入和所需 tolerance。

## 优先局部 tier，而不是全局关闭

从 Python 默认的 `fast` 开始，只为实测运行收益足以抵消额外编译成本的 kernel
选择更高档位：

```python
import taichi_forge as ti

ti.init(arch=ti.cuda, compile_tier='fast', offline_cache=True)

@ti.kernel(opt_level='fast')
def import_once(dst: ti.types.ndarray()):
    for i in dst:
        dst[i] = 0

@ti.kernel(opt_level='full')
def long_running_solver_step():
    # 只有代表性 benchmark 证明运行期收益后才使用 full。
    pass
```

单 kernel tier 有独立 cache identity。当少量超大 specialization 主导启动时间、而主
timestep 仍受益于优化代码时，这比全局关闭更合适。

## 其他编译配置

- `num_compile_threads` 控制外层预编译 worker 预算。LLVM/SPIR-V worker 过量订阅会
  增加 wall time 和峰值内存；可从物理核心数附近开始测量。
- `compile_dag_scheduler=True` 防止批量编译时嵌套 thread pool 相乘；除非诊断 scheduler
  本身，否则建议保持开启。
- `spirv_parallel_codegen=True` 改变调度而非预期结果；除了 wall time，也要测 host 峰值
  内存。
- 应用层 Vulkan 优化只使用 `compile_tier`。`spv_opt_level` 会被拒绝；
  `external_optimization_level` 是实现层原始字段，不应由引擎暴露。
- 编译器实现字段保持默认值，不作为应用调优 API。旧配置处理参见
  [配置迁移说明](forge_options.zh.md#29-已删除仅兼容保留与仅供验证的设置)。
- `use_fused_passes`、`vulkan_listgen_lite_barrier`、
  `vulkan_launch_buffer_pool` 等已删除/no-op 设置应直接从应用配置移除，不要维护
  按版本分支。
- `fast_math=True` 可能采用更快的浮点变换。若严格 IEEE 行为、异常值或紧密跨后端
  一致性比吞吐更重要，应关闭并重新测量。
- 诊断展开导致的编译增长时优先使用警告。unroll/inline hard limit 仅供显式选择，默认
  关闭；主动设置的上限触发时应明确失败，不能静默换算法。

## Graph replay

Graph replay 包含后端特定的容量、生命周期、失败恢复、诊断与显存策略。例如，Vulkan
使用有界的在途 replay 存储；CUDA 则区分结构性
capture 拒绝、暂态失败与 context-fatal 错误。

这些策略及公开 `Graph.execution_stats()` schema 统一维护在
[Graph Runtime 与优化](graph_runtime_optimization.zh.md)。集中维护可以避免这份通用编译
指南变成第二份、以后可能漂移的 graph 规范。
Dense Field 专属编译扩展、prewarm 与静态 binding 权衡见
[Dense Field Graph](dense_field_graph.zh.md)。

## 数值与自动微分验证

每个部署配置应根据实际使用的后端与功能选择相关检查：

- 实际部署后端的 primal output 对可信 reference 的绝对/相对误差；
- 长时间 drift、守恒量、NaN/Inf 行为和确定性 seed；
- 应用实际使用的 reverse/forward AD，以及非光滑点附近的 finite difference；
- 明确的 primal-only Graph 边界：active Tape/FwdMode 必须清晰失败，手工 dispatch 的
  grad-kernel Graph 在 AD context 外运行；
- sparse activate/deactivate、atomic、reduction 和 graph replay；
- release 配置与 `debug=True` / profiler-on 配置分开验证。

optimizer 配置不是应用级同步机制。异步仿真/渲染仍需 snapshot、slot、fence 或其他明确
的 producer-consumer ownership 协议。

## 测量协议

冷编译使用 fresh process；稳态运行使用独立 warm process 或 warm iterations。记录源码
revision、wheel revision、后端、CPU/GPU、driver、编译配置、cache 状态、尺寸和
specialization 数量。报告 median/p95，不只报告最好的一次，并在接受 speedup 前验证
结果。

Taichi 社区案例也说明源码结构的重要性：dynamic indexing 曾把一个静态展开 FEM 示例的
编译从 70 秒降到 2.5 秒；运行性能讨论则显示 scheduling 和 block shape 足以主导后端
比较。重构病态 static unrolling 或 specialization，往往比全局削弱优化更好。

参考：

- [Taichi global settings](https://docs.taichi-lang.org/docs/global_settings)
- [Taichi debugging guide](https://docs.taichi-lang.org/docs/debugging)
- [Taichi v0.9.0 讨论：dynamic indexing 与编译时间](https://github.com/taichi-dev/taichi/discussions/4362)
- [Taichi issue 8526：运行性能测量与 scheduling 讨论](https://github.com/taichi-dev/taichi/issues/8526)
