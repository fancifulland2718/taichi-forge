# 编译与缓存说明

Forge 将可安全复用的前端信息与各后端编译产物分离。目标是在不改变运行语义、不让某个后端覆
盖另一个后端 cache 的前提下，降低重复编译成本。

包含编译辅助和 CLI 入口的模块化 API 参考见 [Forge API 参考](forge_api_reference.zh.md)。

## 公开 API

| API | 用途 |
| --- | --- |
| `ti.compile_kernels(kernels)` | 在热循环前 materialize 并预编译 kernel。任务可为 kernel 或 `(kernel, args)` 对。 |
| `ti.parallel_compile(kernels)` | `compile_kernels(...)` 的别名。 |
| `ti.compile_profile()` | Python 和后端编译耗时 profiling 的 context manager。 |
| `ti cache warmup script.py [-- script_args]` | 强制开启 offline cache 跑一次脚本，写入磁盘 cache。 |
| `@ti.kernel(opt_level="fast"|"balanced"|"full")` | 单个 kernel 的 compile-tier 覆盖。 |
| `ti.init(compile_tier=...)` | Program 级 compile-tier 选择。 |

## 缓存复用规则

Forge 只复用在当前 program、arch、dtype、shape、layout 和 compile configuration 下安全的
数据。

- 同一 Python function source 在同一 program 生命周期内可复用 source-template parse 结果。
- 后端编译产物按后端和 compile configuration 区分 cache key。
- 切换后端不会复用另一个后端的 binary artifact。
- `ti.reset()` 会使 program-lifetime 前端状态失效。
- runtime 值不会通过 cache 复用，除非对应 API 明确把它视为稳定 metadata。

诊断前端行为时，可用 `TI_SOURCE_TEMPLATE_CACHE=0` 关闭 source template cache。

## 推荐用法

重复仿真或渲染循环中，推荐在热循环前显式预编译：

```python
ti.init(arch=ti.cuda, compile_tier="balanced")

ti.compile_kernels([
    (step_kernel, (state,)),
    (render_kernel, (image,)),
])
```

开发迭代优先时可使用 `compile_tier="fast"`；需要最保守 legacy 优化管线时使用
`compile_tier="full"`。

## 边界

- 缓存复用不是任意源码小改的增量编译器。如果代码改动改变 IR、specialization、dtype、shape、layout 或后端配置，受影响编译产物必须重建。
- 后端 native library 和 shader artifact 属于后端 cache 层，不属于前端 parse 层。
- 安全复用不能引入运行时性能亏损或旧语义。
