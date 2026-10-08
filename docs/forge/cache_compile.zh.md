# 编译与缓存说明

> 适用范围：当前源码文档。请按安装版本核对[版本与安装说明](index.zh.md#版本与安装)。

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
| `@ti.kernel(opt_level="fast"\|"balanced"\|"full")` | 单个 kernel 的 compile-tier 覆盖。 |
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

重复内联的 `ti.func` 调用会复用已解析 Python AST 的复制结构。每次展开仍分配独立的
AST 节点和子列表，递归展开与编译失败后的重试也相互隔离。全局变量、闭包 cell 每次
重新读取，静态 Python 回调仍在每次展开时执行。复制结构只保存语法，不保存 Field、
表达式或原生 IR 句柄。这项优化减少 AST 准备开销，尚未合并降级后的函数体，也不引入
设备函数调用；不增加函数大小或调用次数限制。`python.frontend.<name>.ast_parse`
计时项包含首次解析、复制结构构建，以及后续 AST 实例化。

### 实验性内联函数 IR 复用

`ti.init(inline_ir_cache=True)`（或 `TI_INLINE_IR_CACHE=1`）可在一次 kernel materialize
内复用符合条件的标量 `ti.func` 函数体。默认 `False`：只调用一次时，构建可复用函数体
会增加准备开销。上文的 AST 复制结构优化仍独立默认启用。
诊断时关闭 `TI_SOURCE_TEMPLATE_CACHE` 也会绕过 IR 复用。

第一版覆盖标量值参数与返回值、基本类型的静态参数与捕获值、局部计算和分支、静态循环，
以及部分数学 intrinsic。特化键区分参数类型、静态值与实时读取的捕获值。Python 回调、
资源访问、矩阵、运行时循环、递归和其他用户函数调用继续走普通展开路径；这些普通展开
内部调用的合格 helper 仍可受益。Python 常量返回值保留编译期语义。显式启用的静态展开
硬限制和 `auto_real_function` 保留现有路径。

合格函数体只降级一次，在 kernel AST 降级时复制独立局部变量并重绑定实参，随后进入
现有求导和 offload 流程，不引入设备函数调用。模板归原生 kernel 所有，随定义退休，
不进入 Python 源码缓存。即使关闭磁盘缓存，函数体内容也参与 kernel 缓存标识。
依赖函数按调用编号序列化，避免内存地址变化影响缓存标识或混淆不同调用顺序。
这减少重复前端工作，展开后的后端 IR 仍可能随调用次数增长；不新增大小或调用次数限制。

可通过 `ti.compile_profile()` 查看 `python.func.inline_ir_build:<name>` 和
`python.func.inline_ir_call:<name>`；前者包括模板准备和首次原生降级。应用开启前应在
目标后端对比冷编译与暖态运行表现。

## 推荐用法

重复仿真或渲染循环中，推荐在热循环前显式预编译：

```python
ti.init(arch=ti.cuda, compile_tier="balanced")

ti.compile_kernels([
    (step_kernel, (state,)),
    (render_kernel, (image,)),
])
```

Python `ti.init()`（包括打包安装）默认使用 `compile_tier="fast"` 与
`advanced_optimization=False`。显式初始化为 `balanced/full` 时，若没有单独覆盖
该开关，则开启高级 IR 优化。需要最保守 legacy 优化管线且有实测收益时，可选择 `full`。

## Metadata lock 生命周期

Offline-cache metadata 使用操作系统 advisory lock。对应的空 `.lock` 文件是持久文件，
正常退出后仍可能保留在 cache 目录中；文件存在不表示仍有进程持锁。owner 只在加载或
写回 metadata 时保持 OS 文件句柄，正常 unlock 与进程异常终止都会由操作系统释放所有权。
因此，后续进程可以直接复用异常进程留下的 lock 文件，不需要删除已编译 cache。

live process 持有 advisory lock 时，另一个进程会跳过本次 metadata load/dump 并报告
lock busy，不会把文件存在本身当作所有权。owner 正常退出或被强制终止后，下一个进程可以
直接取得同一个持久文件上的 lock。

该修改只影响 metadata coordination。compiled cache artifact 继续使用原有 exclusive
create 发布协议，两个 writer 不能无声覆盖同一 artifact。

Forge 进程仍在运行时不要手动删除 lock 文件。`ti cache clean -p <path>` 仍是要求 cache
空闲的显式维护命令，不再是恢复孤儿锁的必要步骤。

## 边界

- 缓存复用不是任意源码小改的增量编译器。如果代码改动改变 IR、specialization、dtype、shape、layout 或后端配置，受影响编译产物必须重建。
- 后端 native library 和 shader artifact 属于后端 cache 层，不属于前端 parse 层。
- 安全复用不能引入运行时性能亏损或旧语义。
