# 编译与高级优化权衡

[English](compilation_tradeoffs.en.md)

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
| `compile_tier='fast'` | Program，或 `@ti.kernel(opt_level='fast')` 指定的单 kernel | CPU 使用 LLVM O0，CUDA/AMDGPU 使用保证正确 lowering 的 O1 下限，SPIR-V 跳过 optimizer | 可能降低 kernel 吞吐并改变浮点舍入；必须测稳态工作负载 |
| `compile_tier='balanced'` | Program 默认 | 面向生产的折中；LLVM/SPIR-V 保持配置的优化级别 | 冷编译工作多于 `fast` |
| `compile_tier='full'` | Program 或指定 kernel | 默认 global IR cap 未显式改动时，允许全局简化迭代到 fixed point | 编译代价最高；只用于已证明有运行期收益的热点 |
| `advanced_optimization=False` | 大范围 Taichi IR pipeline | 可显著缩短病态 IR 简化，也可隔离 optimizer 故障 | 会成组关闭 LICM、whole-kernel CSE、CFG optimization、store/load forwarding 等；不是细粒度生产调参开关 |
| `debug=True` 及越界/AD validation | Program | 更强诊断与安全检查 | 改变生成代码和运行成本；debug 与 release 必须分开测量 |
| `kernel_profiler=True` | 运行期测量 | 把设备时间归因到 kernel | profiler 可能增加同步或 instrumentation；不能不加说明地把 profiler-on 数字当发布延迟 |

`compile_tier`、`advanced_optimization`、debug 状态、后端 optimizer level 等会改变
代码的配置已经进入 Forge offline-cache identity。切换它们应生成或加载独立产物，而
不是复用不兼容 cache。

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

在本机一次 GeoPhys `stack_cube` CPU 冷启动中，关闭 advanced optimization 将端到端
时间从约 77 秒降到 19 秒，最大的 kernel 从约 43.5 秒降到约 3.0 秒。这只是特定机器和
源码 revision 的诊断数据，不是跨平台性能承诺；生产决策仍需 warm runtime、结果和 AD
测量。

## 优先局部 tier，而不是全局关闭

Program 保持 `balanced`，只标记冷路径或低占空比 kernel：

```python
import taichi_forge as ti

ti.init(arch=ti.cuda, compile_tier='balanced', offline_cache=True)

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
- `spirv_disabled_passes`、`spirv_skip_loop_unroll` 和 adaptive SPIR-V optimization 会
  改变产物；需要在每类目标 Vulkan driver 上验证结果与运行性能。
- `fast_math=True` 可能采用更快的浮点变换。若严格 IEEE 行为、异常值或紧密跨后端
  一致性比吞吐更重要，应关闭并重新测量。
- unroll/inline hard limit 是防止意外编译爆炸的安全栏。触发时应明确失败，不能静默换
  算法。

## 数值与自动微分验证

每个生产 profile 至少应覆盖：

- CPU、CUDA、Vulkan primal output 对可信 reference 的绝对/相对误差；
- 长时间 drift、守恒量、NaN/Inf 行为和确定性 seed；
- 应用实际使用的 reverse/forward AD，以及非光滑点附近的 finite difference；
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
