﻿﻿# Field/SNode PrimitiveView 更新规划

日期：2026-05-22

## 预备规则

### 编码规则

- 仓库文本文件统一使用 UTF-8；中文 Markdown 优先使用 UTF-8 with BOM。
- PowerShell 读写中文文件必须显式指定 UTF-8，或使用 .NET
  `System.Text.UTF8Encoding` 严格读写；不得使用默认 ANSI code page 写文件。
- 读取中文文档做检查时，先设置输出链路：

```powershell
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = $utf8NoBom
[Console]::OutputEncoding = $utf8NoBom
```

- 每个中文文档改动完成后，至少做一次 strict UTF-8 byte check。中文内容若在
  shell 输出中显示异常，先判断是终端解码问题还是文件字节损坏，不直接按乱码
  内容继续编辑。
- 对新增的中文规划文档，默认写成 UTF-8 with BOM，降低 Windows 编辑器误判
  编码的概率。

### Windows 构建、部署与 wheel 规则

本工作区的 VS Code CMake Tools 配置长期存在 `LLVM_DIR`、Python interpreter、
`pybind11Config.cmake` 不完整的问题。涉及 C++、runtime bitcode、pybind、
SNode、CompileConfig、RHI、后端 codegen 的改动，默认使用仓库已验证入口：

```powershell
cmd /c d:\taichi\_run_build.cmd
```

如 `build_llvm20_test` 被 VS Code 或手工配置污染为 Debug，先恢复 Release：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command ". d:\taichi\build_env.ps1; cmake -S d:\taichi -B d:\taichi\build_llvm20_test -DCMAKE_BUILD_TYPE=Release"
```

凡是新增或移动 `CompileConfig` 字段、调整 pybind 暴露顺序、改变跨 TU ABI、
改变 runtime module 字段布局、改变 serialized metadata / offline cache key，
必须执行 clean-first Release 构建，避免旧 object 按旧布局复制配置：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command ". d:\taichi\build_env.ps1; Set-Location d:\taichi\build_llvm20_test; cmake --build . --target taichi_python --config Release --clean-first"
```

构建后检查并部署产物。若改动触达 `taichi/runtime/llvm/runtime_module/`，除了
`taichi_python.cp310-win_amd64.pyd`，还必须同步 `runtime_cuda.bc` 和
`runtime_x64.bc` 到 `python/taichi_forge/_lib/runtime/`；只复制 `.pyd` 会导致
JIT 继续加载旧 runtime bitcode。

大型更新节点或重要 bugfix 完成后，必须把本地构建结果打包为 `dist/` 下的
Python wheel，供其它仓库或环境直接安装验证：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File d:\taichi\build_wheel_local.ps1
```

打包前必须校验 wheel 版本与 `version.txt` 一致：

```powershell
$want = (Get-Content d:\taichi\version.txt -Encoding UTF8).Trim().TrimStart("v")
$got = & C:\Users\Administrator\AppData\Local\ti-build-cache\miniforge\envs\3.10\python.exe -c "import taichi_forge as ti; print('.'.join(map(str, ti.__version__[:3])))"
if ($want -ne $got.Trim()) { throw "version mismatch: version.txt=$want, core=$got" }
```

wheel 打包完成后，至少在当前 3.10 环境验证：

```powershell
C:\Users\Administrator\AppData\Local\ti-build-cache\miniforge\envs\3.10\python.exe -c "import taichi_forge as ti; print(ti.__version__); ti.init(arch=ti.cpu)"
```

## 工作流

根据规划按顺序进行优化，每次优化处理一个问题，在优化过程中，需要通过调研将这个问题的优化拆分为若干个阶段，在每个阶段完成后进行一次测试，确保编译通过性、编译时长、正确性、稳定性、运行时性能和显存占用，确认无误再启用下一步更新，否则定位错误所在并修复然后重新检查直至完全修复，然后将更新内容和结果反馈更新到文档中。中间除非遇到需要重大决策的地方，不要停止，在单个问题完成后停止，并向我汇总更新内容和结果，等待下一步指示。

此外，编写native vulkan时，需要使用cmake组织，来降低构建所需的代码量和文件数。

## 目标

当前通用算法 primitive 已经在 `ndarray`、`StructNdarray` 和部分
`field/SNode` 场景上取得运行时收益，但 field/SNode 路径仍大量依赖
Python `@kernel` helper 和 `template()` field 参数。对大量使用 field/SNode
的物理引擎，这会把运行时收益换成冷编译、IR 体积、offline cache 和
`Program::compile_kernel` 次数的急剧增长。

本规划的目标是：

- 保持现有 field/SNode 用户 API 尽量稳定。
- 将可证明安全的 dense field 路径迁移到 native primitive，不再生成额外
  Taichi helper IR。
- 将 sparse SNode 的优化从 raw pointer 假设改为 active-list/listgen
  descriptor 模型。
- 以 `StructNdarray` 已经验证的 `base + offset + stride` 设计为参考，
  建立统一的内部 `PrimitiveView` 描述层。
- 同时提高编译性能和运行时性能，后续所有优化必须同时报告 runtime、
  compile、IR/cache、workspace/VRAM 四类指标。

## 更新核心要义

这次更新不是单纯追求某个 benchmark 的最快数字，而是要形成一条长期可维护的
Taichi Forge field/SNode 算法路径：

- **编译性能和运行时性能同时提升**：禁止用大量 Python `@kernel` helper、
  更碎的 template specialization 或更大的 offline cache 换取局部 runtime
  speedup。
- **保持跨平台、跨设备兼容**：CUDA device API、CUDA toolkit/CUB 和 native
  Vulkan shader 会比原始 Taichi 的纯 DSL 路径更依赖现代设备能力，但目标是让
  相对现代的 CUDA/Vulkan/CPU 设备保持良好覆盖；不满足能力条件时必须清晰
  fallback。
- **维持架构清晰**：Python algorithm API、internal descriptor、backend
  native primitive、legacy fallback 各自边界清楚，不把后端细节泄漏成新的用户
  API 负担。
- **保留 Taichi field/SNode 特色**：field/SNode 仍然是 DSL 状态、SNode tree、
  sparse topology、activation 和 allocator 语义的载体；优化实现可以下沉到
  native backend，但不能把用户强迫迁移到 `StructNdarray`。
- **允许局部更激进**：对 dense field、strided component、order/apply、native
  Vulkan/CUDA device API 这些语义已清楚的子步，可以比早期规划更积极；对
  sparse topology mutation、dynamic append、deactivation 和 allocator 语义仍然
  必须保守。

## 核心判断

`StructNdarray` 不能直接替代 field/SNode。

它适合 dense AOS payload 和 strided member view，已经证明了
`base allocation + byte offset + byte stride` 可以连接 native CUDA/Vulkan/CPU
后端并避免 helper IR 膨胀。但 field/SNode 还包含：

- SNode tree layout；
- offset 和 physical placement；
- pointer/bitmasked/dynamic/hash 等 sparse topology；
- activation/deactivation；
- allocator 和 lifetime；
- struct-for/listgen traversal；
- 未来可能涉及的 grad/dual/checkbit 语义。

因此，本规划复用的是 `StructNdarray` 的 descriptor 模式，而不是把所有
field/SNode 强行改写为 `StructNdarray`。

## 总体架构

### 1. PrimitiveView

新增内部描述层，位于 Python algorithm routing 与 C++/backend primitive
之间。它不作为第一阶段公开 API。

建议视图类型：

| View | 适用对象 | 第一阶段目标 |
| --- | --- | --- |
| `NdarrayView` | contiguous `ti.ndarray` | 已有 native path 的整理和归一化 |
| `StructMemberView` | `StructNdarray.field(...)` | 作为设计和 ABI 参考 |
| `DenseFieldView` | dense/contiguous `ti.field` | 首个 field/SNode native fast path |
| `StridedFieldView` | zero-copy 可证明 strided field/component | 第二阶段扩展 |
| `SparseSNodeView` | pointer/bitmasked/dynamic/hash SNode | 不 raw pointer 化，只输出拓扑 descriptor |
| `SNodeActiveListView` | sparse active traversal 结果 | sparse native primitive 的核心输入 |

`PrimitiveView` 至少需要描述：

- storage kind；
- backend arch；
- dtype 和 logical shape；
- element size；
- base allocation 或 SNode tree handle；
- byte offset；
- byte stride；
- layout class；
- access mode：contiguous、strided、sparse active、sparse topology mutation；
- mutability：read、write、read-write、atomic；
- alias key：同一 base 下不同 offset/stride 必须不同；
- cache key component：影响 native code 或 parameter ABI 的字段必须进入 key。

### 2. PrimitivePlan

在 `_algorithms.py` 中不要让每个 primitive 独立判断
`Ndarray / StructNdarray / field`，而是先生成 `PrimitiveView`，再由
`PrimitivePlan` 选择后端：

1. contiguous native path；
2. strided native path；
3. dense field native path；
4. sparse active-list native path；
5. legacy helper fallback。

`PrimitivePlan` 的 cache 粒度应按：

```text
backend + op + dtype + layout_class + value_type + semantic_options
```

不应按每个 field 实例或每个 Python helper kernel 生成新的 IR。

### 3. Backend 连接策略

CUDA：

- contiguous/strided dense view 优先走 CUB 或已有 CUDA device API。
- CUB/runtime 依赖统一归入 `TI_WITH_CUDA_TOOLKIT`，不新增 primitive 专用
  build flag。
- 自定义 CUDA kernels 应编译进扩展或 runtime module，不通过 Taichi DSL
  动态生成。
- sparse path 先使用 active-list + compacted dense workspace，再评估
  直接 SNode accessor kernel。

Vulkan：

- 使用固定参数布局的 native compute shader 或已有 generated-header
  workflow。
- 参数变化通过 descriptor/parameter buffer 传入，不动态拼 shader。
- 保持 SPIR-V ABI、descriptor lifetime、reset safety。
- sparse path 需要先有 active-list/listgen contract，再接 reduce/scan/compact
  等 native shader。

CPU：

- 先走简单 native loop 和 parallel threshold。
- 对 dense/strided field 可更积极，CPU 侧实现成本低，适合作为语义先导。
- 不引入复杂 CPU runtime machinery，除非 benchmark 显示真实收益。

## 执行分步方案

分步原则：

- 先做能同时降低 IR 和保持 runtime 的底座，再做单个 primitive 提速。
- 先 dense，再 strided/component，再 sparse active-list，最后 sparse mutation。
- 先读写语义简单的 primitive，再进入 order、bucket、sort、topology mutation。
- 每一步都必须有 fallback，且 fallback 原因能被测试和日志定位。

| 步骤 | 内容 | 修改难度 | 风险 | 兼容性策略 | 进入下一步条件 |
| --- | --- | --- | --- | --- | --- |
| S0 | 编码、构建、wheel、benchmark 规则落地 | 低 | 低 | 不改运行逻辑 | strict UTF-8 检查和文档规则稳定 |
| S1 | compile/cache/IR 指标基线 | 低 | 低 | 只加观测，不改策略 | field/SNode primitive 有 runtime + compile 基线 |
| S2 | `PrimitiveView` 覆盖 ndarray/StructNdarray | 中 | 中低 | 保持现有 API 和 native path | 现有 StructNdarray/ndarray 测试不回归 |
| S3 | `DenseFieldView` 只读/读写基础 descriptor | 中 | 中 | contiguity proof 失败即 fallback | dense field 能被安全识别且不走 raw sparse path |
| S4 | dense field native scan/reduce/transform | 中 | 中 | CPU/CUDA/Vulkan 分 backend 推进 | runtime 不低于 helper 95%，compile calls 明显下降 |
| S5 | dense field gather/scatter/scatter_add/histogram | 中高 | 中 | dtype/backend 能力 gating | correctness 覆盖 offset、alias、small/stress |
| S6 | order/apply primitive 支撑 compact/bucket/sort | 高 | 中高 | in-place 保守，out-of-place 更积极 | order 构造一次，workspace 和 compile 不膨胀 |
| S7 | vector/matrix field component view | 中高 | 中 | packed common lane + scalar fallback | 不破坏相邻 component/field，alias key 正确 |
| S8 | SparseSNodeView 和 active-list contract | 高 | 高 | 初期 opt-in，legacy struct-for fallback | active/inactive/churn correctness 过关 |
| S9 | sparse reduce/histogram/compact/bucket native path | 高 | 高 | active-list + native backend，逐 primitive 启用 | 真实 sparse workload compile/runtime 双达标 |
| S10 | legacy helper 收敛和默认策略切换 | 高 | 高 | 保留可关闭 fallback | 物理引擎级验证通过 |

### S0. 准备阶段：编码、构建、wheel、benchmark 规则

状态：已执行准备性更新，作为后续功能更新的基本规则。

目标：

- 先修正中文文档和 Windows 工具链的编码约束，避免规划和报告继续出现乱码。
- 固化构建、部署、wheel 打包和版本验证入口，后续 C++/runtime/backend 改动不再
  临时选择构建方式。
- 明确 benchmark 不只看 runtime speedup，还必须同时记录 compile/cache/workspace
  指标。
- 按工作流要求，每个阶段完成后将更新内容、验证结果和未执行原因写回文档。

已落地规则：

1. 新增仓库根目录 `.editorconfig`：
   - 默认 `charset = utf-8`；
   - `*.zh.md` 使用 `charset = utf-8-bom`。
2. 中文规划文档使用 UTF-8 with BOM。
3. PowerShell 读写中文文档必须显式使用 UTF-8 或 .NET strict UTF-8 API。
4. 涉及 C++、runtime bitcode、pybind、SNode、CompileConfig、RHI、后端 codegen
   的改动默认使用：

```powershell
cmd /c d:\taichi\_run_build.cmd
```

5. 涉及跨 TU ABI、pybind 暴露顺序、runtime module 字段布局、serialized
   metadata 或 offline cache key 的改动，必须执行 clean-first Release 构建。
6. 触达 `taichi/runtime/llvm/runtime_module/` 时，除了部署 `.pyd`，还必须同步
   `runtime_cuda.bc` 和 `runtime_x64.bc`。
7. 大型更新节点或重要 bugfix 完成后必须打包本地 wheel，并在当前 3.10 环境
   验证 import 和 `ti.init(arch=ti.cpu)`。
8. native Vulkan 更新必须使用 CMake 组织 shader/source 生成流程，避免散落
   生成文件、手工头文件和过多构建文件。

S0 验证记录：

- `.editorconfig` strict UTF-8 检查通过；无 BOM。
- `docs/forge/field_snode_primitive_view_update_plan.zh.md` strict UTF-8 检查通过；
  UTF-8 BOM 存在。
- `git diff --check -- .editorconfig docs/forge/field_snode_primitive_view_update_plan.zh.md`
  通过。
- 本阶段只改规则和文档，不触发 C++、runtime bitcode、pybind、SNode、
  CompileConfig、RHI、backend codegen，因此不运行 build、clean-first build 或
  wheel 打包。

S0 进入 S1 的条件：

- 后续所有文档更新必须保持 strict UTF-8。
- 后续所有中文规划文档默认保持 UTF-8 with BOM。
- 每个功能阶段都必须记录是否触发 `_run_build.cmd`、clean-first、runtime
  bitcode 同步和 wheel 打包条件。
- benchmark/report 模板必须能记录 compile/cache/workspace 指标，S1 将补齐
  具体采集字段和基线矩阵。

### 第一阶段：低风险准备和可观测性

目标：先知道哪里真的膨胀，避免继续凭 microbenchmark 决策。

任务：

1. 固化 UTF-8 / UTF-8 with BOM 文档规则和 Windows 构建规则。
2. benchmark harness 增加 compile/cache/workspace 指标。
3. 对 scan、reduce、histogram、compact、bucket-builder、sort 建立当前
   field/SNode 基线。
4. 增加一份物理引擎级 cold compile + steady-step probe 脚本或报告模板。

风险：低。该阶段只增加规则和观测。

完成标准：

- 能回答每个 primitive 的 runtime median、compile elapsed、
  `Program::compile_kernel` calls、`.tic` count/bytes、workspace peak。
- 展示文档不再只报告 speedup。

当前推进状态：

- 已新增 `benchmarks/primitive_baseline_schema.py`，把既有
  `summary*.json` 归一为 S1 baseline 行。
- schema v1 固定记录 primitive/case、arch、dtype、n、storage、method、
  correctness、median/p95、API return median、workspace peak、compile elapsed、
  `Program::compile_kernel` calls/total seconds、cache files/bytes/path、
  compile CSV 和 chrome trace 路径。
- 已新增 `tests/python/test_primitive_baseline_schema.py`，覆盖 compile top
  去重、cache/workspace 字段保留和 CSV 列顺序。
- 本阶段只建立观测模板，没有重新采集大规模 baseline；后续 S3/P0
  需要按此 schema 补齐 scan/reduce/histogram/compact/bucket/sort 矩阵。

### 第二阶段：PrimitiveView 底座

目标：先把已成功的 `StructNdarray` member view 和 `ndarray` native path
归一到 descriptor 层。

任务：

1. 新增内部 `PrimitiveView` helper，不公开为用户 API。
2. 让 `Ndarray` 和 `StructNdarray` scalar/tensor member view 经由同一
   descriptor 进入 backend planner。
3. 保持现有 C++ `Program` native entrypoint 不变，先做 Python routing 收敛。
4. 把 fallback 原因标准化，例如 unsupported storage、unsupported dtype、
   missing backend capability。

风险：中低。主要风险是 routing 回归，不涉及 SNode 语义。

完成标准：

- 现有 StructNdarray primitive 测试全过。
- 不新增 helper kernel。
- API 行为和错误信息更清楚。

当前推进状态：

- 已在 `python/taichi_forge/algorithms/_algorithms.py` 新增内部
  `_PrimitiveView`，覆盖 `Ndarray`、`StructNdarray`、scalar member view、
  tensor member view 的 dtype、shape、element_shape、payload、offset、stride
  描述。
- 已先把 `PrefixSumExecutor` 的 CPU/CUDA/Vulkan native dense 分支改为通过
  `_PrimitiveView` 路由；field/SNode fallback workspace 路径保持原样。
- `StructNdarrayTensorMemberView` 仍拆成 scalar component 逐分量调用，不改变
  public API 和既有 native entrypoint。
- 本阶段未新增 C++、runtime bitcode、pybind、CompileConfig 或 offline cache
  字段，因此不需要 clean-first Release 构建或 wheel 打包。

### S3. DenseFieldView 只读/读写基础 descriptor

当前推进状态：

- 已将 `_PrimitiveView` 扩展到严格 dense field 子集：
  `root.dense(...).place(scalar_field)` 的 1D scalar field，以及 reduce 输出所需
  的 scalar field。
- descriptor 记录 dtype、shape、SNode 指针、leaf offset、cell stride；proof
  失败时返回 `None`，继续走原有 legacy field/SNode fallback。
- 已增加 proof 回归，确认普通 dense field 可识别，pointer/dense sparse path
  不会被误判成 DenseFieldView。
- 当前仅支持 scalar field；vector/matrix component 和 sparse active-list 仍留在
  S7/S8。

完成情况：

- S3 的 descriptor/proof 层已完成 CPU native 子集所需能力。
- contiguity proof 仍保持窄口径：只接受 root 下的一层 dense place。

### S4. dense field native scan/reduce/transform

当前推进状态：

- 已新增 CPU native dense field C++ entrypoint：
  `cpu_inclusive_scan_dense_field`、`cpu_reduce_dense_field`、
  `cpu_transform_affine_dense_field`。
- 已新增 CUDA/Vulkan dense field C++ entrypoint：
  `cuda_cub_inclusive_scan_dense_field`、`cuda_cub_reduce_dense_field`、
  `cuda_device_transform_affine_dense_field`、
  `vulkan_inclusive_scan_dense_field`、`vulkan_reduce_dense_field`、
  `vulkan_transform_affine_dense_field`。
- Python routing 已让 dense field 在 CPU/CUDA/Vulkan 上优先走 native
  scan/reduce/transform；CUDA 使用 CUB/device API，Vulkan 复用 native shader
  path。capability 或 toolkit 不满足时保留原 fallback/显式错误边界。
- CPU native path 直接映射 root dense field storage，使用 leaf offset 和 cell
  stride，避免新增 Python helper kernel。
- CUDA/Vulkan native path 复用同一 dense field descriptor，通过 root allocation
  + leaf offset + cell stride 绑定原生设备入口，不引入新的 helper kernel IR。
- `ReduceWorkspace` 和 `TransformWorkspace` 的 native replay 已从 dense field
  专用逻辑继续泛化到 `ndarray`、`StructNdarray` scalar member 和 dense
  field；同一 workspace、同一组对象、同一 op/scale/bias 后续调用直接 replay 到
  已确认可用的 native C++ entrypoint。
- `PrefixSumExecutor` 增加 executor-local dense native scan plan；重复扫描同一
  native dense 对象时跳过 `_PrimitiveView` proof 和 backend capability 探测。
- 已将上述 replay 逻辑收敛到内部 `_NativePrimitivePlan`。该对象只保存
  backend、method name、Python 对象 identity、semantic key 和 C++ native call
  arguments；它不生成 Taichi IR、不改变 offline cache key、不新增 C++ ABI，也不
  暴露为 public API。当前作用是让 scan/reduce/transform 以及 direct indexed
  gather/scatter 共享同一套 replay 规则，便于后续 S5 primitive 复用。
- 旧的 `_dense_*_plan`、`_vulkan_dense_*_plan` 和 `_NativeDenseFieldPlan`
  兼容别名已移除；实际记录源统一为 `_native_reduce_plan`、
  `_native_transform_plan` 和 `_native_scan_plan`。
- CPU dense field native path 增加 contiguous fast path。stride 等于元素大小时，
  scan/reduce/transform 直接使用 contiguous typed loop；其他 dense/strided 布局
  继续保留原 strided 语义。
- CPU reduce `op=sum` 增加 typed range-sum helper，避免 hot loop 每个元素都走
  通用 signed wrapping combine。
- 已部署重新构建的 `.pyd` 到 `python/taichi_forge/_lib/core/` 和
  `python/taichi/_lib/core/`；未触达 runtime bitcode。

完成情况：

- CPU/CUDA/Vulkan dense 1D scalar field scan/reduce/transform 子集已接入。
- dense proof 仍保持窄口径；sparse active-list、vector/matrix field component
  仍留到后续阶段。

验证记录：

- `cmd /c _run_build.cmd` 通过，并重新链接 `taichi_python.cp310-win_amd64.pyd`。
- 已同步 `.pyd` 到 `python/taichi_forge/_lib/core/` 和 `python/taichi/_lib/core/`。
- `tests/python/test_scan.py::test_dense_field_view_probe_accepts_only_root_dense_place`
  等 scan 目标回归通过：4 passed。
- `tests/python/test_reduce.py::test_experimental_reduce_cpu_native_dense_field_i32_f32`
  等 reduce 目标回归通过：3 passed。
- `tests/python/test_transform.py::test_experimental_transform_cpu_native_dense_field_i32_f32`
  等 transform 目标回归通过：3 passed。
- S4 CUDA/Vulkan dense native 本轮回归：
  `tests/python/test_scan.py`、`tests/python/test_reduce.py`、
  `tests/python/test_transform.py`、`tests/python/test_primitive_baseline_schema.py`
  文件级回归通过：103 passed。
- 已确认 dense field offset 分 backend 处理：CPU/CUDA 使用 LLVM/SNode cell
  layout offset，Vulkan 使用 `get_field_in_tree_offset()`，避免 native shader
  读写到错误 root allocation 子区。
- S4 Vulkan 深入优化已补充：
  - contiguous dense field 不再误走 strided/member Vulkan shader；
  - scan/reduce/transform pipeline 按实际 dtype/op/path lazy 创建；
  - Vulkan scan 默认关闭当前机器上高冷启动成本且无 warm 收益的 subgroup scan；
  - dense contiguous i32 reduce 增加一阶段 atomic shader，workspace 峰值降为 0；
  - dense contiguous i32/u32 transform 增加 push-constant affine shader，不再需要
    params buffer；
  - transform/reduce 输出 barrier 收窄到实际写入范围；
  - 修复 64-bit dense transform 的非 strided word stride 参数错误；
  - `ReduceWorkspace` / `TransformWorkspace` 增加 dense-field Vulkan replay cache，
    同一 workspace 和同一组 field 后续调用直接 replay 到 C++ native entrypoint。
- Vulkan 复测结果写入
  `docs/forge/s4_dense_field_native_benchmark.zh.md`；scan first-call 从约 190 ms
  降到约 20-22 ms，reduce first-call 降到约 14-15 ms，transform first-call
  降到约 10-13 ms。scan 已同时优于 vanilla 1.8.0 的 first-call 和 warm runtime；
  reduce/transform 在中大规模接近或优于 vanilla，小规模 warm runtime 仍受 native
  command submission 固定成本限制。
- 新增 dense-field workspace/executor replay 回归测试，覆盖 CPU/CUDA/Vulkan 的
  scan/reduce/transform 连续调用时输入数据变化后的正确性。
- `_NativePrimitivePlan` 统一替换 dense field、`ndarray` 与 `StructNdarray`
  scalar member 的重复 replay 逻辑，并补充 `StructNdarray` whole
  vector/matrix member transform replay 后，CPU focused replay 回归通过：
  10 passed。该抽象为 Python-only，未触达 C++、runtime bitcode、pybind、
  CompileConfig 或 offline cache metadata；因此不会引入新的编译成本。
- CPU native 子集回归通过：reduce 7 passed、transform 9 passed、scan 8 passed。
- CUDA/Vulkan 代表性回归通过：18 passed；同时将 Vulkan i32 ndarray transform
  workspace 断言更新为 0 B，匹配 push-constant 路径。
- 新增 `benchmarks/s4_native_plan_replay_bench.py`，结果写入
  `benchmarks/results/s4_native_plan_replay_refactor_20260522/summary.csv`。
  54 个 CPU/CUDA/Vulkan × field/ndarray/StructNdarray member ×
  scan/reduce/transform × 1024/1048576 组合无失败，所有非 skip 组合均
  `ok=True` 且 `plan_reused=True`。
- dense field 对 vanilla 1.8.0 复测写入
  `benchmarks/results/s4_dense_field_native_after_plan_refactor_20260522/summary.csv`。
  CPU/CUDA first-call 和 warm runtime 仍整体优于 vanilla；Vulkan scan 仍优于
  vanilla；Vulkan reduce/transform first-call 优于 vanilla，但部分 warm runtime
  仍受 native submission 固定成本限制。
- 旧 dense plan 兼容层移除并将 StructNdarray whole vector/matrix member transform
  接入 `_NativePrimitivePlan` 后，新增复测写入
  `benchmarks/results/s4_native_plan_replay_clean_struct_tensor_20260523/summary.csv`。
  72 个组合中，whole tensor member 的 scan/reduce 因当前没有单次 packed
  primitive 明确 skip；其余所有非 skip 组合均 `ok=True` 且 `plan_reused=True`。
- 当前可被 `_NativePrimitivePlan` 替换的 replay 实现均已替换；剩余未接入项是
  真实缺少对应单次 native primitive 的 whole tensor member scan/reduce，而不是
  旧兼容包装残留。
- S4.5 继续将 StructNdarray indexed copy/gather/scatter 接入
  `_NativePrimitivePlan`：`IndexedCopyWorkspace` 新增 `_native_indexed_copy_plan`，
  plain ndarray、StructNdarray scalar member 和 StructNdarray whole tensor member
  的 direct native indexed-copy 调用均可 replay。旧的 whole tensor member
  component fallback 和 scalar member field/kernel fallback 已移除；native
  packed/strided backend 不可用时直接报错，避免静默回到多次 helper 调用。
- indexed copy 文件级回归通过：`tests/python/test_indexed_copy.py` 20 passed。
- dense field 对 vanilla 1.8.0 清理后复测写入
  `benchmarks/results/s4_dense_field_native_clean_plan_20260523/summary.csv`。
  CPU/CUDA first-call 与 warm runtime 仍优于 vanilla；Vulkan scan 仍优于 vanilla；
  Vulkan reduce/transform first-call 仍优于 vanilla，warm runtime 仍保留此前的
  submission 固定成本问题。
- 本轮为 Python 调度层更新，未触达 C++、pybind、runtime bitcode、CompileConfig
  或 ABI，因此未重新执行 `_run_build.cmd`。
- 已重新打包本地 wheel：
  `dist/taichi_forge-0.4.0-cp310-cp310-win_amd64.whl`。
- 3.10 wheel smoke 通过：安装到
  `build_llvm20_test/wheel_smoke_s4_clean_struct_plan_20260523` 后
  `import taichi_forge as ti; ti.init(arch=ti.cpu)` 成功。
- S4.5 indexed-copy plan 更新后重新打包同一 wheel，并安装到
  `build_llvm20_test/wheel_smoke_s4_indexed_plan_20260523` 后完成
  `import taichi_forge as ti; ti.init(arch=ti.cpu)` smoke；offline cache lock
  warning 不影响导入和 CPU 初始化。
- S5 已开始推进 dense field indexed-copy：
  - `experimental_gather()` / `experimental_scatter()` 对 1D dense field +
    `ti.ndarray` i32 indices 新增 CPU/CUDA/Vulkan native entrypoint。
  - Python routing 通过 `_PrimitiveView` 证明 dense field，并复用
    `_NativePrimitivePlan` 写入 `IndexedCopyWorkspace._native_indexed_copy_plan`。
  - Vulkan indexed-copy cache 改为按实际 primitive 懒创建 pipeline，避免
    首次调用同时创建 scatter-add/strided 等无关 pipeline。
  - Vulkan 32-bit scatter 增加专用 native shader；gather 复测后保留通用
    lazy pipeline，避免 1M 吞吐回退。
  - CPU indexed-copy 共享内层改为 32-bit word copy，覆盖 ndarray、
    StructNdarray member 与 dense field native path。
  - 文件级回归 `tests/python/test_indexed_copy.py` 更新为 23 passed。
  - 主结果写入
    `benchmarks/results/s5_dense_field_indexed_wordcopy_20260523/summary.csv`；
    Vulkan gather 1M 复测写入
    `benchmarks/results/s5_dense_field_indexed_vulkan_gather_repeat_20260523/summary.csv`。
  - first-call/compile 窗口 CPU/CUDA/Vulkan 均优于 vanilla 1.8.0；workspace
    均为 0 B；Vulkan GPU dedicated delta 约 89 MB，低于 vanilla 约 121 MB。
    CPU 1M warm runtime 仍慢于 vanilla field kernel，后续需要继续优化 CPU
    大规模路径或增加 auto cost model。
- S5 已补齐 dense field `scatter_add` / `histogram`，并同步
  StructNdarray 相关路径：
  - `experimental_scatter_add()` 对 1D dense field + `ti.ndarray` i32
    indices 新增 CPU/CUDA/Vulkan native entrypoint，并通过
    `ScatterAddWorkspace._native_scatter_add_plans` 记录 `_NativePrimitivePlan`。
  - StructNdarray scalar member / tensor member 的 scatter-add component
    调用改为稳定 request signature，重复构造 member view 时复用同一 plan；
    tensor member 按 component 记录 plan，不再回到旧 helper fallback。
  - `experimental_histogram()` 对 contiguous 1D dense field values/bins 新增
    CPU/CUDA/Vulkan native entrypoint；Vulkan histogram cache 改为按实际路径
    懒创建 pipeline，避免 i32/u32/i64 direct/private/shared pipeline 一次性全部创建。
  - S5 当时 StructNdarray scalar member histogram 暂时显式拒绝并提示需要 native
    strided histogram；这比旧路径静默回到 field/helper 更清晰，也避免重新引入
    多 helper IR 膨胀。该限制已在后续 S10 通过 cached native staging 放宽为
    支持 scalar member values/bins，whole vector/matrix histogram 仍关闭。
  - ROI 复核后，将 StructNdarray scalar member native strided histogram
    暂缓到后续有明确 workload 需求时再做。该项需要 CPU/CUDA/Vulkan 都新增
    `base + offset + stride` histogram entrypoint、workspace 统计和 capability
    gating，覆盖面却只补齐较少见的 member histogram；当前建议让用户显式拷贝到
    numeric ndarray 或改用 dense field/ndarray histogram。
  - 本轮未加入自动选择策略：调用者指定 `cpu_native`、`cuda_cub`、
    `cuda_device` 或 `vulkan_native` 时直接走对应 native path；后续 cost model
    另行处理。
  - 代表性结果写入
    `benchmarks/results/s5_dense_field_scatter_add_histogram_lazy_vkhist_20260523/summary.csv`。
    first-call/compile 窗口 CPU/CUDA/Vulkan 均优于 vanilla 1.8.0；CPU 1M
    warm runtime 优于 vanilla；CUDA/Vulkan warm runtime 仍受 native API 固定
    dispatch/sync 成本影响，后续需要专门优化 GPU 小 kernel replay 或更轻量的
    device kernel。
  - ROI 复核后，CUDA/Vulkan scatter-add/histogram warm-runtime 深挖不阻塞
    S6。当前主目标是去 helper IR 和降低 first-call/compile，已经达成；warm
    runtime 继续优化需要后端级 atomic/private histogram、dispatch 融合或
    command replay 调整，风险和验证成本高，且对下一阶段 order/apply 的主线收益
    边际较低。暂列为 GPU backend perf backlog，在 S6/S7 后结合真实 workload
    再决定是否投入。
  - correctness 回归：
    `tests/python/test_scatter_add.py tests/python/test_histogram.py` 22 passed
    （关闭 offline cache；pytest cache 目录权限 warning 不影响结果）。
- S6.1 进入 order/apply 收敛的第一步：
  - 现状：`SortWorkspace`、`CompactWorkspace`、`BucketBuilderWorkspace` 各自维护
    order buffer / order pair / scalar temp buffer；StructNdarray tensor member
    sort、compact、bucket 也各自手写 identity order、order output 清零和 apply
    order 调用。这会让后续 dense field order apply 和 compact/bucket/sort 的
    native 化重复扩散。
  - 本子步目标：先抽出内部 `_OrderApplyWorkspaceMixin`、
    `_prepare_identity_order()`、`_prepare_order_apply_pair()` 和
    `_apply_order_to_values()`，保留现有 public API 和后端 ABI 不变；sort、
    compact、bucket 的 StructNdarray tensor member 路径统一使用同一套 order
    apply helper。
  - 风险边界：本子步只做 Python routing/workspace 收敛，不新增 native
    CUDA/Vulkan shader，不改变 field compact/bucket 的 fallback 结果；in-place
    order apply 仍只对 StructNdarray tensor member 开启，避免普通 ndarray/field
    自覆盖语义变化。
  - 验收：`py_compile`、compact/bucket/sort 相关 pytest 子集通过；workspace
    peak 不因同一 workspace 重复调用而额外线性增长；代表性 CPU/CUDA/Vulkan
    StructNdarray compact/bucket/sort benchmark 不出现明显回归。
  - 实际验证：
    - `python -m py_compile python/taichi_forge/algorithms/_algorithms.py`
      通过。
    - `tests/python/test_compact.py tests/python/test_bucket_builder.py
      tests/python/test_sort.py` 通过：249 passed；仅有既有 Vulkan range cast
      warning 和 pytest cache 权限 warning。
    - benchmark 结果写入
      `benchmarks/results/s6_order_apply_workspace_20260523/summary.csv`：
      CPU/CUDA/Vulkan × sort/compact/bucket × 2048/65536 全部 `ok=True`。
      代表性 65536 warm median：CPU sort 4.046 ms、compact 0.564 ms、
      bucket 0.633 ms；CUDA sort 0.531 ms、compact 0.437 ms、bucket
      0.367 ms；Vulkan sort 0.746 ms、compact 0.556 ms、bucket 0.562 ms。
      workspace peak 保持 order/apply 预期范围：compact 512 KiB，bucket
      约 516-545 KiB，sort 约 768 KiB-1.31 MiB，未出现额外 full-size staging。
- S6.2 推进 dense field compact 的稳定路径收敛：
  - 初始设想是把 field compact 拆成 `flags -> prefix -> order -> native gather`。
    实测后放弃把该路径作为标量 dense field 的默认实现：当前 field compact 只
    支持 i32 标量 payload，order + gather 会比原 fused scatter 多一次 order
    kernel/一次 apply dispatch，workspace 也会额外增加一个 full-size i32 order
    buffer。这里属于抽象一致性收益小、运行时和显存成本明确的低 ROI 路径。
  - 最终实现：
    - StructNdarray tensor/member-view compact 继续使用 S6.1 的 shared
      order/apply helper，保持多 lane/component 共享 permutation 的收益。
    - dense field 标量 compact 继续使用 fused scatter，不引入 order buffer。
    - CPU field compact 使用单 kernel 串行稳定写回，避免 CPU PrefixSumExecutor
      缺失/relocation 问题，workspace 为 0。
    - CUDA/Vulkan field compact 保持 `flags -> native prefix scan -> fused
      scatter`，优先降低 first-call/compile 和大规模 CUDA runtime。
  - 踩坑记录：
    - CPU 直接走 field -> ndarray prefix helper 时，在完整 compact 子集的一次
      顺序中触发过 LLVM `IMAGE_REL_AMD64_ADDR32NB relocation requires an
      ordered section layout`；改为 CPU 单 kernel 稳定路径后未复现。
    - vanilla 1.8.0 的 CPU `PrefixSumExecutor` 不支持 x64 stable scan，因此
      benchmark 中 vanilla CPU compact 使用串行稳定 fallback，只作为可运行的
      语义等价基线。
  - 实际验证：
    - `python -m py_compile python/taichi_forge/algorithms/_algorithms.py
      python/taichi_forge/_kernels.py tests/python/test_compact.py
      benchmarks/s4_dense_field_native_bench.py` 通过。
    - `tests/python/test_compact.py` 通过：21 passed；仅有 pytest cache 权限
      warning。
    - `tests/python/test_compact.py tests/python/test_bucket_builder.py
      tests/python/test_sort.py` 最终回归通过：252 passed；仅有 pytest cache
      权限 warning。
    - stable field compact benchmark 写入
      `benchmarks/results/s6_dense_field_compact_stable3_20260523/summary.csv`。
      与本地 vanilla 1.8.0 稳定 compact 基线相比：
      CPU 4096/65536 first-call 为 35.8/33.5 ms，vanilla 为 26.5/28.9 ms；
      warm median 为 0.116/0.131 ms，vanilla 为 0.025/0.040 ms。CPU 后续需要
      单 kernel 编译缓存或更轻量的 serial path 调优。
      CUDA 4096/65536 first-call 为 168.7/168.8 ms，vanilla 为 529.9/478.2 ms；
      warm median 为 0.564/0.338 ms，vanilla 为 0.416/1.338 ms。小规模 CUDA
      受 CUB/native scan 固定成本影响，大规模收益明确。
      Vulkan 4096/65536 first-call 为 66.8/65.3 ms，vanilla 为 156.1/151.5 ms；
      warm median 为 0.986/0.926 ms，vanilla 为 0.708/0.803 ms。Vulkan
      compile/first-call 达标，warm runtime 仍需后端 command replay 或 fused
      native compact 继续优化。
      workspace：CPU 0；CUDA 为 prefix + CUB temp（4096: 17407 B，
      65536: 263167 B）；Vulkan 为 prefix（4096: 16384 B，65536: 262144 B）。
- S6.3 继续降低 order/apply 的 first-call 和 warm 准备开销：
  - 目标：当前 S6 的主瓶颈不是单个 native gather/transform 的吞吐，而是
    StructNdarray member view 在应用层频繁重建时，workspace 只按 Python object
    identity 复用计划，导致计划 replay 不稳定；同时 compact/bucket 每次都重新
    初始化 identity order 和清零 order output，额外触发小 kernel/fill。
  - 实现：
    - `_NativePrimitivePlan` 增加稳定 object key，按 payload/SNode identity、
      dtype、shape、element shape、offset、stride 识别等价 view；显式传入的
      `TransformWorkspace` / `IndexedCopyWorkspace` 可缓存多个 native plan。
    - 临时 workspace 默认关闭多计划缓存，避免一次性调用为了生成 key 付出额外
      Python 开销。
    - `_OrderApplyWorkspaceMixin` 统一持有内部 `IndexedCopyWorkspace`，compact
      和 bucket 的 out-of-place apply 可以复用 native indexed-copy plan。
    - order pair 按精确 `n` 缓存；identity order 只在该尺寸第一次分配时执行。
      后续同尺寸调用不再重复 fill，小尺寸调用不会复用过大 buffer，避免
      `values.shape != flags.shape` 的隐性风险。
    - S10 后续补齐 sort 的 in-place tensor-member apply：首次调用记录内部
      gather+transform replay group，warm call 直接复用 native 调用序列；compact
      和 bucket 的 order output 不再做无语义清零。
  - 风险边界：
    - stable key 只用于内部 native plan replay，不改变 public API。
    - key 中保留 offset/stride/element shape，防止不同 StructNdarray member
      或 dense field 视图误复用。
    - field compact 标量路径仍保持 fused scatter，不为了抽象一致性引入 order
      buffer。
  - 实际验证：
    - 最新 `py_compile` 覆盖 `_algorithms.py`、`test_transform.py`、
      `test_indexed_copy.py`、`test_compact.py`、`test_bucket_builder.py` 和
      `benchmarks/struct_ndarray_primitives.py`，通过。
    - CPU 定向用例通过：9 passed。覆盖 StructNdarray member wrapper 每次重建
      仍复用 native plan、Transform/IndexedCopy 多 plan cache 回切、compact
      order-pair 精确尺寸缓存、compact/bucket 内部 `IndexedCopyWorkspace` 复用。
    - CUDA/Vulkan tensor-member 定向用例分别通过：4 passed / 4 passed。覆盖
      transform、gather/scatter、compact、bucket 的共享 order/apply 路径。
    - 最新 benchmark 写入
      `benchmarks/results/s6_struct_ndarray_plan_cache_20260523/`，覆盖
      CPU/CUDA/Vulkan × 2048/65536 × transform/scan/reduce/gather/scatter/
      scatter_add/grouped_reduce/sort/compact/bucket；所有记录 `ok=true`，并
      写出 `first_call_ms`、`median_ms`、`workspace_peak`。
    - 代表性 warm median：CPU 65536 transform/gather/compact/bucket 为
      0.2809/0.1681/0.2901/0.3896 ms；CUDA 为
      0.0511/0.0488/0.1325/0.2263 ms；Vulkan 为
      0.2693/0.1884/0.3163/0.3568 ms。
    - workspace peak 未新增额外 full-size staging：65536 compact CPU/CUDA/Vulkan
      均为 512 KiB 量级，bucket 为约 516/514/545 KiB，sort 为约
      768 KiB/1.26 MiB/1.28 MiB。
    - benchmark 期间仍出现既有 `C:/taichi_cache/ticache/ticache.lock` stale
      warning；结果文件完整写出，但后续做正式横向报告时应先清理该 cache
      状态或隔离 cache 目录。
- S6.4 聚焦 field compact 的 native/cache-reuse 路径：
  - 目标：本子步只处理 dense scalar field compact，不继续改 StructNdarray。
    CUDA/Vulkan 优先改善 workspace/cache 命中后的 warm runtime，尤其避免每次
    warm call 仍经过两个 Taichi helper kernel；CPU 继续评估 serial path，但
    不接受 warm runtime 回退。
  - 现状引用：
    - Python field compact 入口在
      `python/taichi_forge/algorithms/_algorithms.py::_compact_field_scan()`。
    - CPU 当前走 `_kernels.py::compact_stable_serial_field`，仍要 JIT 一个 Taichi
      helper kernel。
    - CUDA/Vulkan 当前走 `_kernels.py::compact_flags_to_prefix_ndarray_from_field`
      + native scan + `_kernels.py::compact_scatter_field_from_prefix_ndarray`，
      warm 阶段仍有两个 helper kernel dispatch。
    - C++ 已有 dense field raw allocation 工具：
      `Program::get_dense_field_device_ptr()`、`Program::get_dense_field_stride()`，
      scan/reduce/histogram 已经使用该路径。
  - 做法：
    - CPU 曾试验 `Program::cpu_compact_dense_field()` 和并行 two-pass compact；
      实测 first-call 可降到约 1 ms，但 65536 warm median 从约 0.13 ms 回退到
      约 0.38 ms，因此按工作流回退，不作为默认路径保留。S6 closeout 又复测
      了更小的 C++ serial candidate，first-call 约 0.5 ms，但 warm median
      仍退到 0.26-0.29 ms，同样移除。
    - CPU field compact 最终保留低风险 Python helper 改动：
      `compact_stable_serial_field_static_n()` 按 `n` 缓存静态 serial helper。
      该路径不新增 C++ ABI，不增加 workspace；相比原动态 `N` helper，65536
      first-call/warm 有小幅改善，但仍慢于 vanilla CPU serial 基线。闭包绑定
      具体 field 的 Taichi kernel 也已测试，warm 退到 1.8-1.9 ms，已放弃。
    - 新增 `Program::cuda_cub_select_dense_field()`：对 contiguous i32 dense field
      直接调用 CUB DeviceSelect，替代 `flags -> prefix -> scatter`。
    - 新增 `Program::vulkan_compact_dense_field()`：复用现有 Vulkan compact cache、
      prefix workspace 和 fused recording，小规模走单个记录闭包，大规模走
      cached compact pipeline + scan pipeline + scatter pipeline。
    - Python 侧仅在 CUDA/Vulkan 上先尝试 native dense field compact；若 field
      stride/layout 不满足 contiguous native 条件，回退到 S6.2 的 helper 路径，
      保持 field/SNode 兼容。CPU 继续使用 serial helper。
  - 验收：
    - `tests/python/test_compact.py` 中 field compact correctness 通过，并按
      CPU/CUDA/Vulkan 分进程验证，规避当前混合后端 LLVM relocation 状态污染。
    - `benchmarks/s4_dense_field_native_bench.py --op compact` 重新输出
      CPU/CUDA/Vulkan × 4096/65536 的 forge/vanilla 对比；CUDA/Vulkan warm
      median 应优于 S6.2，CPU 不得因试验路径产生默认回退。
  - 风险：
    - native compact 只对 contiguous i32 dense field 默认开启，非 contiguous
      packed field 或复杂 SNode layout 保持 fallback。
    - 涉及 C++/pybind，完成后必须走本仓验证构建入口，并确认 Python 加载的是
      新构建产物。
  - 实际验证：
    - 构建：`cmd /c _run_build.cmd` 通过，并已同步
      `taichi_python.cp310-win_amd64.pyd` 到 `python/taichi_forge/_lib/core/`
      和 `python/taichi/_lib/core/`；3.10 环境确认新 pybind 中
      `cuda_cub_select_dense_field=True`、`vulkan_compact_dense_field=True`，
      `cpu_compact_dense_field=False`。
    - correctness：
      `tests/python/test_compact.py::test_experimental_compact_field_scan`
      等 4 个 field compact 定向用例通过：10 passed；仅有 pytest cache 权限
      warning。
    - benchmark 写入
      `benchmarks/results/s6_dense_field_compact_native_final_20260523/summary.csv`。
      相对 S6.2：CUDA 4096/65536 first-call 下降 93.9%/94.6%，warm median 下降
      47.6%/13.6%；Vulkan 4096/65536 warm median 下降 45.4%/37.0%，但
      first-call 从约 66 ms 上升到 115/126 ms，这是 native compact pipeline
      首次准备成本，需要后续 command replay/pipeline warmup 继续优化。
      CPU 默认路径未切换，warm median 维持约 0.11/0.13 ms；本轮 CPU native
      试验因 warm runtime 回退已移除。
    - 相对本地 vanilla 1.8.0：CUDA 4096/65536 warm median 分别快 28.8%/49.4%，
      Vulkan 65536 快 19.6%，Vulkan 4096 慢约 4.2%；CPU 仍明显慢于 vanilla，
      继续列为 S6 CPU serial/codegen backlog。
    - S6 closeout 复测写入
      `benchmarks/results/s6_dense_field_compact_closeout_20260523/summary.csv`。
      最新 CPU 4096/65536 first-call 为 38.0/33.9 ms，warm median 为
      0.108/0.124 ms，较原 S6.4 65536 的 38.3/0.132 ms 有小幅改善。
      CUDA 4096/65536 first-call 为 12.7/11.0 ms，warm median 为
      0.291/0.384 ms；Vulkan first-call 为 109.9/118.0 ms，warm median 为
      0.699/0.492 ms。CPU 仍慢于 vanilla，因此不再把 CPU compact 伪装成
      S6 已完全解决项；后续需要从 CPU codegen/serial field lowering 层处理。
- 当前机器用户给定的 miniforge 3.10 路径不可执行，因此本轮使用
  `C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe`
  完成等价版本检查和 smoke。
- repeat30 稳定复测结果写入
  `docs/forge/s4_dense_field_native_benchmark.zh.md`：
  - CPU reduce/transform first-call 从 vanilla 的 32-43 ms 降到 0.5-1.3 ms，
    warm runtime 在 1024 和 1048576 两个规模均优于 vanilla。
  - CUDA scan/reduce/transform first-call 和 warm runtime 在 1024 和 1048576
    两个规模均优于 vanilla。
  - Vulkan scan first-call 和 warm runtime 均优于 vanilla；Vulkan reduce/transform
    first-call 和 GPU delta 优于 vanilla，但 warm runtime 仍有约 0.02-0.03 ms
    固定提交成本差距，需要后续 command replay / submission amortization。
- `tests/python/test_primitive_baseline_schema.py` 通过：2 passed。
- 本地 wheel 已生成：
  `dist/taichi_forge-0.4.0-cp310-cp310-win_amd64.whl`。
- wheel 后 3.10 环境验证 `import taichi_forge as ti; ti.init(arch=ti.cpu)`
  通过。

### 第三阶段：DenseFieldView native 化

目标：优先吃掉物理引擎中最常见、语义最安全的 dense field。

任务：

1. 实现 field contiguity proof：dense、1D、known offset、contiguous allocation。
2. 将通过 proof 的 field 描述成 `DenseFieldView`。
3. CPU 先导实现 scan/reduce/transform。
4. CUDA 连接 CUB 或已有 device API。
5. Vulkan 连接已有 native shader 或新增固定参数 shader。
6. proof 失败的 field 继续走 legacy helper。

风险：中。核心风险是误判 contiguous 或 offset/stride。

兼容性策略：

- 默认只启用严格安全子集。
- CUDA toolkit/CUB 不可用时回退。
- Vulkan shader capability 不满足时回退。

完成标准：

- 同一 dense field primitive 不再生成对应 Python helper IR。
- 多个同 layout field 不导致 `compile_kernel` 线性增长。
- 运行时不低于旧 helper 95%，目标是同时提升。

### 第四阶段：index/order 类 primitive 收敛

目标：compact、bucket-builder、sort 不再各自维护一套 field helper。

任务：

1. 把 order construction 和 apply order 抽为内部 primitive。
2. 对 dense field 支持 direct strided output。
3. compact/bucket-builder 先做 out-of-place。
4. sort in-place 保守处理，避免 self-gather overwrite。
5. 只有 profiling 证明 fixed cost 仍显著时，再做 fused order + writeback。

风险：中高。排序和重排容易破坏稳定性、in-place 语义和 workspace。

完成标准：

- order 只构造一次。
- 多 lane/component 共享 permutation。
- workspace 不出现 full-size staging 回归。

### 第五阶段：vector/matrix component field

目标：把 `StructNdarray` whole tensor/member-view 的经验用于 field component。

任务：

1. 定义 vector/matrix field component descriptor。
2. common packed lane 走 native packed path。
3. uncommon layout 或不安全 alias 走 scalar fallback。
4. 覆盖 transform、gather/scatter、scatter_add、reduce 的 component 子集。

风险：中。主要风险是 component offset/stride、AOS/SOA 和 alias key。

完成标准：

- 不破坏相邻 lane 或其他 field。
- CSE/cache 不把不同 component 合并。
- small/stress 都有 correctness 和 workspace 数据。

### 第六阶段：SparseSNodeView 与 active-list

目标：保留 Taichi sparse SNode 特色，同时让 sparse primitive 避免 helper IR
爆炸。

任务：

1. 定义 `SparseSNodeView`：描述 topology，不承诺 raw contiguous pointer。
2. 定义 `SNodeActiveListView`：active logical index list、active count、
   generation/version。
3. pointer/bitmasked/dynamic/hash 分别建立最小 active-list contract。
4. 先支持只读 traversal，再进入写回和 topology mutation。

风险：高。这里触及 SNode 核心语义。

兼容性策略：

- 初期 opt-in。
- topology mutation、dynamic append、deactivation 保持 legacy fallback。
- hash SNode 可单独试点，不阻塞 pointer/bitmasked。

完成标准：

- active/inactive holes、deactivation、churn、高 active ratio、低 active ratio
  都有测试。
- 真实 sparse workload 的 compile/cache 不再随 helper 线性增长。

### 第七阶段：跨平台默认策略和物理引擎验收

目标：决定哪些 native path 可以默认开启。

任务：

1. 用代表性物理引擎 workload 验证 cold compile、warm-cache compile、
   first-frame latency、1000-step steady runtime。
2. 分 backend 给出默认策略：CPU、CUDA modern device、CUDA toolkit/CUB、
   Vulkan modern device、fallback。
3. 对不满足兼容性或 compile 预算的路径保持 opt-in。
4. 打包 wheel 并在当前 3.10 环境做 import + `ti.init(arch=ti.cpu)` 验证。

风险：中高。风险来自真实 workload 与 microbenchmark 差异。

完成标准：

- 物理引擎级 compile 和 runtime 双达标。
- 不破坏 field/SNode 用户 API。
- native path 不可用时有清晰 fallback。

## 分阶段计划

### P0. 指标和验收系统先行

目标：以后任何 field/SNode primitive 优化都不能只报告 runtime speedup。

改动范围：

- benchmark harness 增加 compile 指标采集和汇总。
- 算法展示文档和后续报告增加 compile/cache/workspace 表。
- 建立 field/SNode primitive support matrix。

必须记录：

- `compile_elapsed_ms`；
- `Program::compile_kernel` calls；
- compiled function count；
- `.tic` file count / bytes；
- IR offload count 或可获得的 IR stmt count；
- public API median/p95；
- workspace peak bytes；
- VRAM/allocator high-water mark；
- correctness status。

验收：

- 已有代表性 field/SNode primitive 至少覆盖 scan、reduce、histogram、
  compact、bucket-builder。
- 每个结果同时有 runtime 和 compile/cache 指标。
- 新优化若 runtime 提升但 compile/cache 明显恶化，默认不进入下一阶段。

### P1. PrimitiveView 基础层

目标：统一 algorithm routing，先不改变 public API。

改动范围：

- `python/taichi_forge/algorithms/_algorithms.py`
- `python/taichi_forge/lang/_ndarray.py`
- 新增内部 view/descriptor helper。

内容：

- 将 `Ndarray`、`StructNdarrayScalarMemberView`、
  `StructNdarrayTensorMemberView` 归一为 `PrimitiveView`。
- 保留现有 native path 行为，但通过 view 层路由。
- 为 field/SNode 先返回 `UnsupportedFieldView` 或 legacy fallback marker。
- 记录 shape/dtype/offset/stride/cache-key 的统一逻辑。

激进点：

- 不再为每个 primitive 写重复的 member-view 检测分支。
- 对 `StructNdarray` 已有能力直接搬到 view planner，不等所有 field/SNode
  设计完成。

验收：

- 现有 StructNdarray primitive 测试不回归。
- 不增加 `Program::compile_kernel` 调用。
- Python routing 复杂度下降，重复 backend dispatch 分支减少。

### P2. DenseFieldView native fast path

目标：解决最常见的 dense field 物理引擎数据，不再通过 helper kernel。

适用范围：

- dense 1D scalar field；
- zero/known offset；
- contiguous allocation；
- primitive 不需要 sparse traversal；
- dtype 属于已有 native backend 支持集合。

改动范围：

- field descriptor extraction；
- C++ `Program` native entrypoints；
- CUDA/Vulkan/CPU primitive routing；
- targeted tests。

内容：

- 建立 field contiguity/layout proof 工具。
- 从 field 获取 base device allocation、byte offset、shape、stride。
- 将 dense field 视作 `DenseFieldView` 传入 native scan/reduce/transform/
  gather/scatter/scatter_add 等路径。
- 对不能证明 contiguous 的 field 保持 legacy fallback。

激进点：

- 该阶段可以默认启用安全子集，不必长期 opt-in。
- 条件是判定必须严格：证明失败就 fallback，不猜测。
- CPU/CUDA/Vulkan 可以分开合入，不要求一次三端完成。

验收：

- dense field native path 冷编译 `Program::compile_kernel` calls 不随 primitive
  helper 增加。
- 对同一 primitive，新建多个同 layout dense field 不应产生线性 helper IR
  增长。
- runtime 不低于现有 helper path 的 95%，目标是同时提升。
- correctness 覆盖 offset、size 0/1/small/stress、dtype、alias。

### P3. Dense vector/matrix field 和 strided/component view

目标：把 `StructNdarray` member-view 的经验推广到 dense field component。

适用范围：

- `ti.Vector.field` / `ti.Matrix.field` 中的 scalar component；
- AOS/SOA 布局可证明 stride/offset；
- 不涉及 sparse topology。

内容：

- 定义 field component descriptor。
- 让 transform/reduce/scatter_add/grouped_reduce 等 numeric primitive 可使用
  component view。
- packed lane copy/transform 可参考 StructNdarray whole tensor member 的
  packed path，但不改变 field public API。

激进点：

- common vector2/3/4 可以直接做 packed native path。
- uncommon layout 保持 scalar-lane fallback。

验收：

- 不引入 full-size staging。
- 不破坏其他 component 或相邻 field。
- alias analysis 区分不同 component offset/stride。

当前推进状态：

- `ti.Vector.field` / `ti.Matrix.field` 的 whole field 输入已先接入
  component routing：对每个 `get_scalar_field(...)` component 复用 S3/S4
  已证明的 DenseFieldView native path。
- 覆盖 primitive：
  - `PrefixSumExecutor.run()`：whole vector/matrix field 按 lane 原地 scan；
  - `experimental_transform()`：whole vector/matrix field source/destination
    按 lane native transform；
  - `experimental_reduce()`：1D whole vector/matrix field reduce 到
    `shape=()` 的 whole vector/matrix field output；
  - `experimental_gather()` / `experimental_scatter()`：按 lane 复用
    dense-field indexed-copy native path；
  - `experimental_scatter_add()`：按 lane 复用 dense-field scatter-add native
    path。
- 该子步是 Python routing-only：未新增 C++ ABI、runtime bitcode、pybind、
  `CompileConfig` 字段或 offline cache key；不会增加新的 Taichi helper IR。
- `ReduceWorkspace` 和 `PrefixSumExecutor` 增加多 native plan cache，避免
  vector/matrix 多 component 场景下显式 workspace 只能缓存最后一个 lane。
- `TransformWorkspace`、`ReduceWorkspace`、`IndexedCopyWorkspace`、
  `ScatterAddWorkspace` 和 `PrefixSumExecutor` 增加 whole vector/matrix field
  的 native component plan group：首次调用仍按 scalar component 建立 native
  plan；同一 workspace/executor 后续调用直接 replay 这组 plan，跳过
  `get_scalar_field()` 重建、递归校验和逐 lane plan 查找。该缓存仍是
  Python-side routing metadata，不生成 Taichi IR，不改变 C++ ABI 或 offline
  cache key。
- component plan group 的 replay 实现进一步收敛为直接调用已记录的 native
  method name 和 call arguments，不再在 group 内重复做逐 plan program-id
  校验；group 自身仍检查当前 `Program`，不会跨 `ti.reset()` 误复用。该改动
  不增加持久引用和额外存储，只减少热路径 Python 判定层级。复测目录：
  `benchmarks/results/s7_dense_matrix_field_group_direct_20260523/`。
- workspace 边界：CPU/CUDA component indexed/scatter-add 仍为 0 额外 workspace；
  Vulkan indexed/scatter-add 可能报告 24-28B 固定 native 状态，未引入
  full-size staging。
- S7 matrix/vector field benchmark 已加入
  `benchmarks/s4_native_plan_replay_bench.py --storages matrix_field`，覆盖
  `first_call_ms`、warm runtime median、workspace peak 和 GPU dedicated memory
  采样。结果目录：
  `benchmarks/results/s7_dense_matrix_field_components_20260523/` 为首次
  component routing 基线，
  `benchmarks/results/s7_dense_matrix_field_plan_group_20260523/` 为 plan group
  后结果。
- plan group 后 warm runtime 相比首次 component routing 的改善范围：

| backend | n=1024/65536 覆盖 primitive | warm runtime 改善 | matrix field 相对 scalar field | workspace peak |
| --- | --- | ---: | ---: | ---: |
| CPU | scan/reduce/transform/gather/scatter/scatter_add | 2.55x-12.57x | 0.24x-5.26x | 0B；reduce 64K 为 8B |
| CUDA | scan/reduce/transform/gather/scatter/scatter_add | 3.90x-10.19x | 0.29x-5.58x | scan 1023B；reduce 1B/17407B；其余 0B |
| Vulkan | scan/reduce/transform/gather/scatter/scatter_add | 2.25x-4.86x | 0.51x-2.14x | scan 36B/1044B；reduce 12B/140B；transform 40B；indexed 28B；scatter_add 24B |

- GPU dedicated memory 采样是进程级峰值 delta，用于发现显著存储回退；本轮
  CUDA 约 792-794MB、Vulkan 约 89MB，主要反映后端 runtime/device 初始化和
  固定资源，不代表每个 primitive 的 full-size staging。workspace peak 未出现
  与 `n` 成比例的新增 staging buffer。
- 首次调用仍保留 backend native 固定成本：CUDA 多数约 10-14ms，Vulkan
  scan 的 two-lane pipeline warmup 约 45ms；这不是 Taichi helper IR 编译，
  后续若要继续压低，需要 packed lane native entrypoint 或 Vulkan command /
  pipeline warmup 级优化。
- 复核已有 packed-strided native 入口后，当前不把 DenseFieldView 强行接到
  StructNdarray 的 packed ndarray entrypoint：这些入口接受 `Ndarray *`，
  field 侧需要保留 SNode/root allocation/leaf offset/cell stride 语义。若后续
  要继续降低 multi-lane dispatch 成本，应新增明确的 packed dense-field native
  entrypoint，而不是绕过 descriptor 边界。
- StructNdarray 与 S7 相关分析：
  - whole tensor member `transform` 和 `gather/scatter` 已有 packed/strided native
    path，本轮不再拆 lane，以免退化到多 dispatch；
  - `scan`、`reduce` 和 `scatter_add` 仍是 component-split path，已复用同一套
    component plan group，使 wrapper 重建或同 workspace/executor 后续调用可
    replay native plan；
  - component plan group 的 try/record 逻辑已抽到统一 helper，matrix field 与
    StructNdarray tensor member 共用 `_NativePrimitivePlanGroup` 的 cache key、
    Program guard 与 temp workspace 统计语义；
  - benchmark 已放开 `struct_tensor_member` 的 scan/reduce 覆盖；helper 抽取后复测
    结果目录：`benchmarks/results/s7_struct_tensor_component_group_refactor_20260523/`。

| storage | backend | primitive 覆盖 | warm median 范围 | first-call 范围 | workspace peak |
| --- | --- | --- | ---: | ---: | ---: |
| matrix_field | CPU | scan/reduce/transform/gather/scatter/scatter_add | 0.029-0.383 ms | 0.95-3.14 ms | 0B；reduce 64K 为 8B |
| matrix_field | CUDA | scan/reduce/transform/gather/scatter/scatter_add | 0.064-0.139 ms | 9.98-12.94 ms | scan 1023B；reduce 1B/17407B；其余 0B |
| matrix_field | Vulkan | scan/reduce/transform/gather/scatter/scatter_add | 0.248-0.370 ms | 12.97-39.51 ms | scan 36B/1044B；reduce 12B/140B；transform 40B；indexed 28B；scatter_add 24B |
| struct_tensor_member | CPU | scan/reduce/transform/gather/scatter/scatter_add | 0.019-0.351 ms | 0.26-1.19 ms | 0B；reduce 64K 为 8B |
| struct_tensor_member | CUDA | scan/reduce/transform/gather/scatter/scatter_add | 0.033-0.166 ms | 9.69-11.90 ms | scan 1023B；reduce 1B/17407B；其余 0B |
| struct_tensor_member | Vulkan | scan/reduce/transform/gather/scatter/scatter_add | 0.202-0.444 ms | 12.86-48.19 ms | scan 36B/1044B；reduce 12B/140B；transform 40B；indexed 28B；scatter_add 24B |

- 暂不处理 `experimental_grouped_reduce()` 的 whole field component native 化：
  当前还没有 dense-field grouped-reduce native backend，强行拆分只会回到多个
  field helper kernel，编译收益不足。compact/bucket/sort 的 whole field
  order/apply 仍留给 P4。
- 定向验证：
  - `test_scan_native_dense_matrix_field_components`：CPU/CUDA/Vulkan 3 passed；
  - `test_experimental_transform_native_dense_matrix_field_components`：
    CPU/CUDA/Vulkan 3 passed；
  - `test_experimental_reduce_native_dense_matrix_field_components`：
    CPU/CUDA/Vulkan 3 passed；
  - `test_experimental_gather_scatter_native_dense_matrix_field_components`：
    CPU/CUDA/Vulkan 3 passed；
  - `test_experimental_scatter_add_native_dense_matrix_field_components`：
    CPU/CUDA/Vulkan 3 passed。
  - 上述测试已扩展第二次调用 replay 断言，确认 plan group 建立后可复用。
  - StructNdarray tensor member 相关回归：
    `test_scan_*_struct_tensor_member_view`、`test_experimental_reduce_*_struct_tensor_member_view`
    和 `test_experimental_scatter_add_*` 的 CPU/CUDA/Vulkan 目标组合通过。
  - 组合跨文件运行仍可能触发当前仓库已知的混合后端状态污染；本轮按文件拆分
    验证，只有 pytest cache 权限 warning。

### P4. Order/selection primitive 统一

目标：sort、compact、bucket-builder、gather/scatter 使用统一 order/apply
机制，避免每个 primitive 自己维护一套 field helper。

S10 当前状态：

- StructNdarray tensor-member 路径已经具备共享 order/apply 语义层：
  compact/bucket out-of-place apply 复用内部 `IndexedCopyWorkspace`，sort
  in-place apply 复用内部 gather+transform replay group。
- order pair 为精确尺寸缓存；只有 identity input order 在分配时初始化，output
  order buffer 不再清零。
- dense field 标量 compact 仍保持 native/fused scatter 路径，不为了抽象一致性
  引入 order buffer。

内容：

- 将已有 StructNdarray order-once 模式提升为内部 primitive。
- 对 DenseFieldView 支持 order apply。
- 对 compact/bucket-builder 支持 direct strided output。
- 对 in-place 路径保持保守，避免 self-gather overwrite。

激进点：

- compact/bucket-builder 可以尝试 fused order construction + strided writeback，
  如果 profiling 显示单独 gather/writeback fixed cost 仍显著。
- CUDA 可使用 CUB select/scan 结果直接驱动 strided scatter。
- Vulkan 可在 native compact/bucket shader 参数中加入 offset/stride。

验收：

- sort/compact/bucket 对 dense field 不再使用 legacy field helper 作为首选。
- workspace 不超过当前 native ndarray 同类路径的合理范围。
- order 只构造一次，多 lane/component 共享同一 permutation。

### P5. SparseSNodeView 和 active-list contract

目标：进入真正 sparse SNode，不把 sparse field raw pointer 化。

适用范围：

- pointer；
- bitmasked；
- dynamic；
- hash；
- nested sparse tree。

内容：

- 定义 `SparseSNodeView`：只描述 topology 和 target leaf，不承诺 contiguous。
- 定义 `SNodeActiveListView`：active logical index list + optional value accessor。
- listgen 结果可缓存，cache key 包含 SNode tree generation/version。
- primitive 消费 active list，而不是重新生成 Python helper IR。

激进点：

- 对只读 traversal primitive 可较早默认启用 active-list path。
- 对 topology mutation、deactivation、dynamic append 保持 opt-in 或 fallback。
- hash SNode 可以单独使用 hash active-list/diagnostics 数据，不等待全部 sparse
  类型完成。

验收：

- sparse field correctness 与 legacy struct-for/helper path 对齐。
- active count、inactive holes、deactivation 后状态都覆盖。
- compile/cache 增长由 active-list native kernels 控制，不随 SNode 实例线性膨胀。
- sparse runtime 至少在代表性物理场景中不低于 legacy 95%，目标是提升。

### P6. Sparse native primitives

目标：基于 active-list/listgen 实现 sparse reduce、histogram、compact、
bucket-builder 和后续 solver primitive。

优先级：

1. sparse reduce / histogram；
2. sparse gather/scatter；
3. sparse compact；
4. sparse bucket-builder；
5. sparse sort/order；
6. solver-facing broadphase/contact helper primitive。

后端策略：

- CUDA：active list + compacted dense workspace + CUB/native kernels。
- Vulkan：active list + parameterized compute shader。
- CPU：active list loop。

激进点：

- 可以接受少量 workspace 增加，前提是 compile 下降和 runtime 有实测收益。
- 可以为物理引擎常见 sparse pattern 做 layout-class specialization，但不能按
  单个 field 生成新 IR。

验收：

- benchmark 包含 synthetic sparse 与真实物理引擎子步骤。
- 记录 active ratio、churn ratio、workspace、compile/cache。
- topology mutation 场景不出现 stale active list。

### P7. Legacy helper 收敛和去膨胀

目标：保留 fallback，但不再让它成为优化主路径。

内容：

- 将 `_kernels.py` 中 field helper 按 primitive 分类。
- 已有 native/PrimitiveView 覆盖的 helper 标记为 legacy fallback。
- 对必须保留的 helper 做 fusion 或 graph batching，减少 Python kernel calls。
- 对不再使用的 benchmark-only helper 做删除或文档降级。

激进点：

- 对重复 helper 可以主动收敛，不必等所有 sparse path 完成。
- 但不能删除仍被 public API fallback 依赖的路径。

验收：

- helper 数量、compile_kernel calls、offline cache 文件数下降。
- fallback 行为和错误信息仍清晰。

### S10-DSL. native-first 策略门控

目标：在暂时跳过 sparse SNode 和物理引擎级验证的前提下，先让 DSL 层可以
显式暴露并关闭 `method="auto"` 下的 legacy helper fallback。这样后续物理
引擎测试可以直接发现哪些 dense/StructNdarray 路径仍在生成 helper IR，而不
需要依赖离线 cache 反查。

现状：

- dense field、StructNdarray 和 vector/matrix component 已经优先尝试
  `_NativePrimitivePlan` / `_NativePrimitivePlanGroup`；
- native backend 不可用时，部分 primitive 的 `auto` 仍会静默落到
  `_kernels.py` helper，例如 field compact/reduce/histogram/transform/
  indexed-copy/scatter-add/bucket/grouped-reduce 以及 PrefixSum field fallback；
- 这些 fallback 必须保留给兼容性和显式 `field_kernel` / `field_scan`
  调试，但不应在 S10 之后继续难以观测。

本子步做法：

- 新增 DSL 层 legacy helper fallback policy，默认保持兼容：允许 `auto`
  fallback；
- 提供 Python 侧开关和环境变量，使物理引擎测试可设置为 native-only
  `auto`，一旦触发 legacy helper 即报错；
- 记录每个 primitive 触发 legacy helper 的次数，便于定位剩余 helper IR；
- 显式 legacy method（如 `field_kernel`、`kernel`、`field_scan`、
  `field_atomic`）不受该 policy 限制。
- 已实现：
  - `set_legacy_helper_auto_fallback_enabled(False)` 可在当前进程关闭
    `method="auto"` 的 legacy helper fallback；
  - `TAICHI_FORGE_LEGACY_HELPER_AUTO_FALLBACK=0` 可在进程启动时关闭；
  - `get_legacy_helper_fallback_counts()` /
    `clear_legacy_helper_fallback_counts()` 用于收集和清零 helper 触发次数；
  - fallback 计数默认关闭，只有显式启用
    `set_legacy_helper_fallback_counting_enabled(True)` 时才写入 dict；
  - 当前计数覆盖 compact、reduce、histogram、transform、gather/scatter、
    scatter_add、bucket_builder、grouped_reduce 和 PrefixSum field fallback。
  - policy helper 是纯 Python 分支，不创建 Taichi kernel，不会产生额外
    compile_kernel；native path 不进入该 helper。
  - 对 CPU dense field compact 下沉到 native C++：
    `Program::cpu_compact_dense_field()` 直接映射 dense field storage，
    使用 typed assignment 快路径，1M 级别启用稳定 two-pass 并行 compact；
  - `CompactWorkspace` 增加 CPU dense native plan replay：首次验证后记录
    轻量 `(id(values), id(flags), id(output), id(count), n)` key 与 SNode
    指针，后续同 workspace 直接调用 C++ primitive，避免重复
    routing/native probe；
  - `ScalarField.from_numpy()` 对 CPU dense scalar field 增加
    `Program::copy_dense_field_from_host()` setup 快路径，绕开
    `ext_arr_to_tensor` helper kernel；unsupported layout/dtype 自动回退到
    原 helper。
  - `ScalarField.to_numpy()` 对 CPU dense scalar field 增加
    `Program::copy_dense_field_to_host()` readback 快路径，绕开
    `tensor_to_ext_arr` helper kernel；dtype 转换、非 CPU 后端或非 dense layout
    仍自动回退到原 helper。

验收：

- 默认策略下现有 dense field / StructNdarray 行为不变；
- policy 关闭时，dense field compact 的 `auto`/`field_scan` 走 native C++，
  不再触发 helper；仍无 native 路径的 field bucket_builder 会给出明确错误；
- 显式 `field_scan` 仍可运行并优先走 native dense field compact；
- fallback 计数可清零、查询，并能记录显式 helper 调用。
- 定向验证：
  - `py_compile` 覆盖 `_algorithms.py` 与 `test_compact.py`；
  - `test_experimental_compact_cpu_dense_field_native_avoids_helper_policy`
    与 `test_experimental_compact_field_scan` 通过；
  - `test_experimental_bucket_builder_legacy_helper_auto_fallback_policy`
    覆盖仍保留 helper fallback 的 CPU field bucket_builder；
  - `tests/python/test_indexed_copy.py::test_experimental_gather_scatter_field_kernel_i32_f32`
    跨 CPU/CUDA/Vulkan 通过，覆盖此前 `field.from_numpy()` setup IMA
    复现路径；
  - reduce/histogram/transform/indexed-copy/scatter-add 显式 field fallback
    组合通过：15 passed；
  - bucket_builder/grouped_reduce 显式 field fallback 组合通过：6 passed。
  - `tests/python/test_compact.py` 全量通过：23 passed。
  - 最新 S10 dense/native 验证集通过：105 passed，覆盖 CPU dense
    from_numpy/to_numpy、compact、IMA 复现、CPU policy、CUDA/Vulkan
    transform/reduce/indexed-copy/scatter_add/bucket/grouped_reduce/histogram
    native 用例。
- benchmark：
  - `benchmarks/results/s10_dsl_legacy_fallback_policy_20260523/` 覆盖 CPU
    field compact 1024/65536；
  - Forge `field_scan` first-call 38.97/35.78 ms，warm median 0.107/0.118 ms；
  - vanilla 1.8.0 对应单 kernel compact first-call 27.03/31.39 ms，
    warm median 0.026/0.039 ms；
  - 结论：本子步没有解决 CPU compact fallback 的结构性 runtime 差距，
    但提供 native-only auto policy 后，物理引擎验证可直接定位这类仍依赖
    legacy helper IR 的路径。
  - workspace replay 优化后复测写入
    `benchmarks/results/s10_cpu_field_compact_light_replay_nocount_20260523/`：
    Forge CPU field compact warm median 降至 0.038 ms / 0.052 ms；
    vanilla 1.8.0 为 0.024 ms / 0.036 ms。
  - `method="auto"` replay 小测：4096 items median 0.0376 ms，plan 数为 1，
    默认 fallback counts 为空，说明默认计数未引入 dict 写入负载。
  - 其他显式 fallback microbench 写入
    `benchmarks/results/s10_dsl_helper_overhead_20260523/`。4096 items CPU
    结果显示默认计数关闭时 counts 均为空；计数开启后每个 primitive 记录
    warmup+repeat 的 14 次调用。默认关闭 / 计数开启 median：
    reduce 0.1788 / 0.1886 ms，histogram 0.2989 / 0.3003 ms，
    transform 0.2617 / 0.2662 ms，gather 0.2564 / 0.2574 ms，
    scatter 0.2602 / 0.2625 ms，scatter_add 0.3112 / 0.3141 ms，
    bucket 0.5763 / 0.5580 ms，grouped_reduce 0.2488 / 0.2690 ms。
    结论：默认关闭计数后，policy helper 不产生额外 dict 运行时负载；
    计数开启应作为诊断模式，不作为性能跑分默认设置。
  - CPU native C++ compact 过程数据：
    `benchmarks/results/s10_cpu_native_cpp_compact_20260523/` 表明仅下沉
    到 C++ 后 first-call 已降至 0.43-1.67 ms，但 warm runtime 受 Python
    routing 和逐元素 memcpy 影响，仍不可接受；
    `benchmarks/results/s10_cpu_dense_native_plan_compact_20260523/`
    增加 native plan replay 与 typed assignment 后，1024/4096/65536/1M
    first-call 为 0.398/0.412/0.680/1.055 ms，warm median 为
    0.016/0.016/0.047/0.549 ms；
    `benchmarks/results/s10_cpu_dense_native_parallel_compact_20260523/`
    加入 two-pass 并行后，first-call 为 0.453/0.395/0.436/1.285 ms，
    warm median 为 0.014/0.016/0.046/0.314 ms；
    `benchmarks/results/s10_cpu_dense_native_parallel_chunk131k_compact_20260523/`
    调整并行粒度后，first-call 为 0.640/0.392/0.443/1.377 ms，
    warm median 为 0.014/0.015/0.046/0.260 ms。该方案小规模和 first-call
    明显优于 vanilla 1.8.0；1M warm runtime 仍慢于 vanilla 约 30%，但避免了
    约 28-32 ms 的 helper 编译窗口。262k chunk 复测写入
    `benchmarks/results/s10_cpu_dense_native_parallel_chunk262k_compact_20260523/`，
    1M median 0.268 ms，差于 131k，因此保留 131k。
  - `field.from_numpy()` setup IMA：已通过 CPU dense scalar native host-copy
    避开 `ext_arr_to_tensor` 编译；此前失败的 indexed-copy 参数化用例现在
    CPU/CUDA/Vulkan 同进程通过。后续仍保留按 primitive/backend 拆分验证，
    用于减少 teardown 状态污染对诊断的干扰。
  - `field.to_numpy()` CPU dense scalar readback benchmark 写入
    `benchmarks/results/s10_cpu_dense_to_numpy_native_20260523/`。1024/4096/
    65536/1M i32 first-call：Forge 为 0.362/0.165/0.206/1.241 ms，
    vanilla 1.8.0 为 47.551/33.889/33.685/35.085 ms；warm median：
    Forge 为 0.052/0.053/0.082/0.723 ms，vanilla 为
    0.116/0.119/0.135/1.106 ms。结论：native readback 同时移除
    `tensor_to_ext_arr` helper 编译窗口，并降低 warm host readback 成本。
  - Vulkan transform dense-field 测试发现 workspace peak 对全局 native
    cache 顺序敏感；已在测试开头清理 cache，并将 i32/u32 dense path 的
    固定 8 B cache 上界显式化，避免把 cache 复用口径误判为内存回归。

- S10-DSL 继续处理优化 3/4/5：
  - dense field `bucket_builder` / `grouped_reduce` 已新增 CPU/CUDA/Vulkan
    native entrypoint。CPU 直接映射 dense field storage 调用 native C++；CUDA
    通过 device/CUB-style bucket 与 atomic grouped-reduce；Vulkan 通过 native
    bucket/grouped-reduce shader，并保留显式 `field_kernel` 作为语义回退。
  - Python `experimental_bucket_builder()` / `experimental_grouped_reduce()` 的
    `auto`/显式 backend 路由已识别 contiguous dense scalar field，不再把 whole
    dense field 固定落到 legacy helper；policy 关闭时，CPU dense field
    bucket_builder 不再触发 fallback。
  - Vulkan bucket/grouped-reduce cache 改成 lazy pipeline 创建：不再在第一次
    grouped-reduce 时顺带创建 bucket、segmented grouped-reduce、宽 dtype 和
    strided path 的全部 pipeline。代表性 65536 grouped-reduce first-call 从
    约 284 ms 降至约 20 ms；bucket first-call 从约 305 ms 降至约 81 ms。
  - CPU dense field compact 保持稳定顺序语义，1M 级别仍使用 two-pass 并行
    compact；该路径的主要收益是避免 helper kernel 编译窗口，warm runtime
    仍可能慢于 vanilla 单 kernel serial compact，后续若要继续压 warm runtime，
    需要保持稳定顺序的单 pass/native prefix 方案，而不是改成非稳定 atomic
    写回。
  - CPU dense scalar/vector/matrix field host I/O 扩展：`ScalarField.to_torch()`、
    `ScalarField.to_paddle()`、`MatrixField.to_numpy()`、`MatrixField.from_numpy()`、
    `MatrixField.to_torch()`、`MatrixField.to_paddle()` 在 contiguous CPU dense
    场景下复用 `copy_dense_field_to_host/from_host`，避免额外编译
    `tensor_to_ext_arr` / `matrix_to_ext_arr` helper。
  - benchmark 写入
    `benchmarks/results/s10_field_native_lazy_20260523/summary.json` 和
    `summary.csv`，覆盖 CPU/CUDA/Vulkan × compact/bucket_builder/grouped_reduce
    × 4096/65536/1048576，对比本地 vanilla 1.8.0。
  - 代表性 65536 结果：CPU bucket first 1.22 ms vs vanilla 150.57 ms、warm
    0.454 ms vs 0.901 ms；CPU grouped first 0.502 ms vs 52.789 ms、warm
    0.351 ms vs 0.348 ms；CUDA bucket first 10.40 ms vs 253.98 ms、warm
    0.395 ms vs 0.291 ms；CUDA grouped first 10.07 ms vs 99.25 ms、warm
    0.241 ms vs 0.064 ms；Vulkan bucket first 81.36 ms vs 75.22 ms、warm
    0.745 ms vs 0.468 ms；Vulkan grouped first 20.20 ms vs 27.64 ms、warm
    0.655 ms vs 0.213 ms。
  - 解释：CPU/CUDA 的主要收益是 compile/first-call 大幅下降；Vulkan lazy
    pipeline 已解决 native cache 的过度首编译，但 bucket/grouped-reduce warm
    runtime 仍受多 dispatch/barrier 固定成本影响。该点应作为后续物理引擎验证
    的默认策略输入：若真实 workload 更看重 warm runtime，可让 Vulkan
    grouped/bucket 在默认策略中保持 opt-in native 或继续做 command replay /
    fused clear+reduce 设计。
  - 验证：`cmd /c _run_build.cmd` 通过；`tests/python/test_bucket_builder.py
    tests/python/test_grouped_reduce.py` 通过：36 passed；`tests/python/test_compact.py`
    加 field/matrix host I/O 定向测试通过：139 passed。仅有既有 pytest cache
    权限警告和 `C:/taichi_cache/ticache.lock` 警告。

- S10 hot-path 调用链收缩（2026-05-24）：
  - CUDA `scatter_add` / grouped-reduce atomic sum 增加 same-key warp
    aggregation kernel，并沿用 block-private shared aggregation。参考
    NVIDIA warp-aggregated atomics 思路，但本机默认 dense-field
    `scatter_add` 分布下 direct atomic 反而更慢，因此默认仍保留当前
    block-private/warp 路由；新增
    `TI_CUDA_SCATTER_ADD_BLOCK_PRIVATE` 和
    `TI_CUDA_SCATTER_ADD_WARP_AGGREGATE` 作为诊断/回归开关，不作为
    benchmark 默认策略。
  - Vulkan native path 新增 `VulkanRwBufferReplay<N>`，按
    `(DeviceAllocation, generation, offset, bytes)` 缓存稳定 storage
    buffer binding；`generation` 在 buffer allocation/import 时递增，避免
    allocation 槽位复用后跳过 rebind 造成 stale descriptor/IMA。设备切换、
    workspace 释放或 cache clear 时重置。当前覆盖
    transform、add-merge、gather/scatter、scatter_add 和 grouped-reduce
    atomic/strided binding。该层只减少重复 descriptor 更新，不改变 shader
    ABI、push constant/params buffer 布局或 SNode allocation 语义。
  - dense-field 对比数据写入
    `benchmarks/results/replay_warp_chain_20260524/summary.csv`，覆盖
    CUDA/Vulkan × transform/gather/scatter/scatter_add/grouped_reduce ×
    4096/65536，并与本机 vanilla 1.8.0 对比。代表性结论：
    CUDA first-call 约 9.7-11.5 ms，vanilla 约 51-92 ms；Vulkan first-call
    约 10.6-19.2 ms，vanilla 约 15.7-28.9 ms。
  - warm runtime 仍保留的最差项：CUDA `scatter_add` 4096/65536 为
    0.126/0.126 ms，vanilla 为 0.0437/0.0428 ms；Vulkan
    `scatter_add` 4096/65536 为 0.285/0.301 ms，vanilla 为
    0.120/0.131 ms；Vulkan `grouped_reduce` 65536 为 0.193 ms，vanilla
    为 0.133 ms。该差距主要来自 atomic primitive 的 device kernel/command
    固定成本和当前 shader 算法，而非 Python helper IR。
  - dense-field workspace peak：CUDA 相关项为 0B；Vulkan transform 为
    8B 固定 params，其余本轮项为 0B。GPU dedicated memory 仍是进程级
    runtime/device 初始化采样，CUDA 约 792MB、Vulkan 约 89MB，不表示新增
    full-size staging。
  - StructNdarray 当前值写入
    `benchmarks/results/replay_warp_chain_20260524/struct_cuda.json` 和
    `struct_vulkan.json`。最慢 warm 项仍是 sort/compact/bucket/scan：
    CUDA 65536 sort median 0.731 ms、workspace 1.32MB；Vulkan 4096 sort
    first-call 578.7 ms、median 0.814 ms，65536 sort workspace 1.34MB。
    这些应作为后续是否做算法级深优化的候选，而不是继续扩大通用 replay
    抽象。
  - 验证：`cmd /c _run_build.cmd` 通过；`tests/python/test_scatter_add.py`
    18 passed；`tests/python/test_grouped_reduce.py` 17 passed；
    `tests/python/test_vulkan_native_lifecycle.py` 1 passed。仅有既有
    pytest cache 权限警告。

- Native vs helper kernel 对照与下一步 ROI 判断（2026-05-24）：
  - 对照 artifact：
    `benchmarks/results/native_vs_helper_field_kernel_20260524/summary.csv`
    和上一轮 native 数据
    `benchmarks/results/replay_warp_chain_20260524/summary.csv`。
  - 公平对照窗口的边界：当前 dense-field native indexed-copy /
    scatter_add 要求 `indices` 为 `ti.ndarray`，而 legacy field helper
    kernel 要求 field/template indices。因此 gather/scatter/scatter_add
    的“native vs helper”不是同一数据形态；这正是物理引擎 field/SNode
    场景的关键缺口。`field -> ndarray` 迁移能进入 native fast path，但
    引入应用层数据搬运和生命周期管理；继续依赖 field helper 又会回到
    JIT helper kernel 和 IR/cache 压力。
  - 已能同形态对照的 transform/grouped-reduce 说明 native 路线本身是
    正确方向：CUDA native warm 为 helper 的 6%-30%，first-call 约为
    helper 的 9%-11%；Vulkan native warm 为 helper 的 25%-42%，first-call
    约为 helper 的 44%-56%。helper fallback 在 `_algorithms.py` 多个
    入口末尾显式 `sync()`，会把本应由外层 `ti.sync()` 或应用调度控制的
    同步提前到每个 primitive 内部；这解释了 Forge helper 明显慢于
    vanilla 单 kernel 的一部分固定开销。
  - 与 vanilla 1.8.0 helper/JIT kernel 对比后，当前 runtime 弱项不再是
    “是否 native”，而是 native 的调用链和算法选择是否足够接近 JIT kernel：
    CUDA transform/gather/scatter native 已经优于 vanilla warm；CUDA
    `scatter_add` 仍为 vanilla 的约 2-3 倍；Vulkan 多个单 kernel 等价
    primitive 仍比 vanilla warm 慢约 1.1-2.4 倍。
  - ROI 排序：
    1. **补齐 dense-field indices native support**：为 indexed-copy、
       scatter_add、bucket/grouped 类输入增加 dense field index/key
       descriptor native overload，避免物理引擎为了进入 native path 复制
       field indices 到 ndarray。这是覆盖面和架构收益最高的更新。
    2. **移除 helper fallback 的过度同步**：保留外层/用户同步语义，helper
       kernel 内部只 enqueue，不主动 `sync()`。这不会让 native 更快，但能
       降低 fallback 固定成本，并更准确地对齐 Taichi JIT kernel 语义。
       需要逐个验证返回后立即读回、workspace 复用和异常路径。
    3. **Vulkan native enqueue 链条继续靠近 GfxRuntime JIT kernel**：
       descriptor replay 已完成第一步；下一步重点是减少每次 lambda 中的
       params buffer fill、barrier 和 resource-set touch，使单 shader
       primitive 接近 vanilla kernel launch host path。暂不优先做具体
       scatter_add shader 深挖。
    4. **CUDA scatter_add 增加 low-contention single-kernel route**：
       当前 block-private/warp 聚合适合热点冲突，但默认分布下 vanilla
       单 atomic JIT kernel 更快。下一步应增加一个“有效 index、低冲突、
       单 launch”的 native route，并保留 two-level/block-private 给高冲突
       场景，而不是继续堆叠默认聚合逻辑。
  - 结论：追齐两端 runtime 的主线不是回退 helper kernel，而是让 native
    覆盖 field-index/key 形态，并把 native launch path 压缩到和 JIT
    kernel 一样短；helper 只作为兼容 fallback，且应去掉内部过度同步。

- Two-level aggregation primitive（2026-05-24 第一阶段）：
  - 边界：先在 DSL/internal plan 层统一 two-level 语义，不新增 C++ ABI、
    shader、runtime bitcode 或公开稳定 API。field/SNode sparse 结构继续跳过；
    dense field 只复用已有 native grouped-reduce 能力，不改变 SNode 语义。
  - `experimental_grouped_reduce()` 新增实验方法别名 `two_level`、
    `cuda_two_level`、`vulkan_two_level`、`cpu_two_level`。CUDA 映射到已有
    segmented/CUB-style grouped reduce；Vulkan 映射到已有 native segmented
    grouped reduce；CPU 映射到 native partial/merge grouped reduce。
  - `GroupedReduceWorkspace` 增加 native scalar plan cache 与 whole
    StructNdarray tensor-member plan group replay。缓存 key 使用
    PrimitivePlan object key，member view wrapper 重建后仍可复用同一 payload
    和 offset/stride 对应的 native plan。
  - workspace 内部 segmented buffers 扩容或 dtype 改变时会清空相关 plan，
    避免 replay 旧 offsets/scratch/cursor 对象；这保持存储占用可解释，也避免
    stale buffer 被计划闭包延长生命周期。
  - 当前第一阶段主要降低 Python 层校验、wrapper 重建和 component-wise
    dispatch 的固定开销；真正降低高冲突 atomic 成本的 per-block/workgroup
    partial aggregation kernel 仍属于后续 C++/CUDA/Vulkan 深入阶段。

- Two-level aggregation primitive（2026-05-24 覆盖扩展）：
  - `experimental_histogram()` 和 `experimental_bucket_builder()` 也接受
    backend-neutral `two_level` 以及 `cuda_two_level`、`vulkan_two_level`、
    `cpu_two_level` 实验方法别名。
  - histogram two-level 映射到已有 CUDA CUB、Vulkan native histogram、
    CPU native histogram；field fallback 的 `two_level` 显式使用
    `field_private`，避免小规模 direct path 与 two-level 策略混淆。Scalar
    StructNdarray member values/bins 现在通过 cached native transform staging
    接入这些后端，whole tensor member histogram 仍因语义不清保持关闭。
  - bucket two-level 映射到已有 native count/prefix/scatter bucket builder；
    StructNdarray tensor-member payload 继续复用 order/apply workspace。
  - `HistogramWorkspace` 和 `BucketBuilderWorkspace` 均补充 native plan cache。
    bucket cache 会在 cursor buffer 扩容时清空，避免 replay 旧 cursor。
  - `histogram`、`bucket_builder`、`grouped_reduce` 的 public entrypoint 现在
    共用同一个 aggregation backend family router。它只在 Python 层把
    `auto`、backend-specific method 和 backend-neutral `two_level` 映射到
    当前 arch 的 native/segmented family，不创建 Taichi kernel、不分配设备
    workspace，也不改变 PrimitivePlan cache key。这样后续新增 aggregation
    primitive 时先复用语义分发，再决定是否需要真正的 CPU/CUDA/Vulkan
    backend kernel。
  - 并行语义保持为通用 two-level aggregation：第一级在 block/workgroup/thread
    私有或分段空间内降低冲突，第二级按 group/bin 做全局合并。当前
    histogram/bucket/grouped-reduce 只把已有正确后端挂到这个策略名下；
    若 measured runtime 显示 atomic/native 更适合某分布，`auto` 仍保持原默认。
  - 验证使用 3.10 工作区环境 `PYTHONPATH=D:\taichi\python`，确认加载
    `D:\taichi\python\taichi_forge` 的 `taichi_forge 0.4.0`；默认
    `python` 当前会加载 Python 3.14 site-packages 的旧 0.1.0，不用于本轮
    结论。
  - 收口验证：`py_compile` 覆盖 `_algorithms.py`、相关三件套 tests 和
    benchmark；`test_histogram.py`、`test_bucket_builder.py`、
    `test_grouped_reduce.py` 合计 `47 passed`。
  - router smoke benchmark 记录在
    `benchmarks/results/s10_aggregation_router_20260524/summary.md`。2048 规模下
    bucket `auto/two_level` 在 CPU/CUDA/Vulkan 上 workspace 相同；CUDA
    grouped-reduce `two_level` 小规模仍慢于 atomic/default 且需要 workspace，
    因此保持显式 opt-in。
  - `scatter_add` 的 two-level alias 已在后续收口阶段补齐：它不是用
    grouped-reduce 覆盖 `dst`，而是 reduce-to-scratch 后调用 native
    add-merge，保持 `dst[indices[i]] += src[i]` 语义。

- Scatter-add plan cache hygiene（2026-05-24）：
  - `ScatterAddWorkspace` 接入同一 aggregation backend family router，但
    现在额外接受 `two_level`、`cuda_two_level`、`vulkan_two_level`、
    `cpu_two_level`。该路径先 grouped-reduce 到 workspace scratch，再用
    native add-merge 写回原 destination，避免破坏已有 `dst` 的累加语义。
  - 新增 CPU/CUDA/Vulkan add-merge native primitive。CPU 是 typed native
    loop；CUDA 走 device/toolkit kernel；Vulkan 走 CMake 管理的 compute
    shader 与 cached resource/pipeline。三者均支持 plain ndarray、
    StructNdarray scalar member stride，以及 plain scratch 到 dense field 的
    add-merge。
  - Vulkan two-level scatter-add 的 StructNdarray member source 会先通过
    native transform staged 到 plain ndarray，再调用已有 native segmented
    grouped-reduce，最后 add-merge 到 plain/member/dense field destination。
    这样避免为 strided segmented reduce 扩张 shader ABI，也避免 Taichi helper
    kernel/IR 膨胀。
  - `ScatterAddWorkspace` 记录 reduce scratch、member staging scratch、嵌套
    transform/grouped-reduce workspace、add-merge plan 和 two-level replay
    group；scratch 扩容或 dtype 改变时清空相关 plan，防止 stale buffer replay。
  - 验证更新：`test_scatter_add.py` 18 passed；`test_histogram.py` 10 passed；
    `test_bucket_builder.py` 20 passed；`test_grouped_reduce.py` 17 passed；
    `test_sort_api.py` 49 passed。
  - 性能/存储记录：
    `benchmarks/results/s10_two_level_dtype_layout_20260524/summary.md`。默认
    atomic/native 对本轮低冲突 synthetic scatter-add/grouped-reduce 仍更快；
    two-level 保持显式 opt-in，作为高碰撞分布和物理引擎验证的可缓存策略。

- Two-level aggregation primitive（2026-05-24 落地收口）：
  - `experimental_grouped_reduce()` 的 Vulkan two-level member-view 缺口已补齐：
    scalar member keys/values/output 通过 cached native transform staging 接入
    plain native segmented reduce；whole tensor member views 继续按 scalar lane
    复用同一路径。CPU/CUDA 仍使用已有 strided/member native reduce。
  - `experimental_scatter_add()` 的 two-level 路径覆盖 ndarray、StructNdarray
    scalar member view、whole tensor member lane decomposition，以及 ndarray
    scratch 到 dense field destination 的 add-merge。`auto` 仍保持 atomic/native
    默认，不因 microbenchmark 的特定冲突分布强行切换。
  - 完整 dtype/layout 语义：本轮 two-level scatter-add 和 grouped-reduce 覆盖
    `i32/f32/u32/u64/i64/f64`；histogram 覆盖 `i32/u32` values 与 `i32/i64`
    bins；Vulkan 宽 dtype 受设备 shader capability gating 约束。StructNdarray
    raw payload 仍只用于 payload/order 类 primitive，不进入 numeric aggregation。
  - 收口验证环境为 Python 3.10.11 + `PYTHONPATH=D:\taichi\python` +
    repo `taichi_forge 0.4.0` extension。`test_scatter_add.py` 18 passed；
    `test_grouped_reduce.py` 17 passed；`test_histogram.py` 10 passed；
    `test_bucket_builder.py` 20 passed；`test_sort_api.py` 49 passed。

- S10 通用运行时弱项优化（2026-05-24 计划）：
  - 现状引用：
    `benchmarks/results/s10_current_all_dense_field_postfix_20260524/summary.json`
    显示 CPU first-call/compile 对比 vanilla 1.8.0 的 27 个可比项均未回退；
    当前 CPU 问题集中在 warm runtime，尤其是 4096/65536 规模的
    scatter_add、histogram、grouped_reduce、gather/scatter、bucket_builder
    和 compact。
  - 做法：
    1. 不新增 Taichi helper kernel，不改变 `PrimitivePlan` cache key，不增加
       workspace；
    2. CPU indexed-copy 的 payload copy/zero 增加常见 4B/8B/16B fast path，
       减少小 scalar field/ndarray 的循环化字节拷贝成本；
    3. CPU histogram/grouped-reduce/scatter-add 继续保留大规模 two-level
       partial/merge 路径，但提高并行阈值，避免 64K 左右低收益规模过早分配
       partial workspace 和触发两次 thread-pool dispatch；
    4. CPU bucket-builder 只收紧 native 并行阈值，避免 64K 左右低收益规模
       过早分配 partial/thread-offset workspace；稳定 order/apply 输出语义不变。
    5. histogram、bucket_builder、grouped_reduce 的 workspace replay 先尝试
       最近一次 `_NativePrimitivePlan`，未命中再计算 dict cache key，减少热路径
       Python metadata 固定开销。
  - 验收标准：
    - focused tests 通过：`test_scatter_add.py`、`test_histogram.py`、
      `test_grouped_reduce.py`、`test_indexed_copy.py` 的 CPU 相关路径；
    - CPU first-call 不退步，workspace peak 不增加；
    - CPU warm runtime 弱项至少在 4096/65536 规模收敛，1M 大规模不能明显
      退化；
    - 若阈值调整让大规模聚合回退，则按单 primitive 回退，不影响其他后端。
  - 风险：
    - 过高阈值可能削弱 1M 以上聚合吞吐；
    - scalar payload fast path 若处理对齐假设过宽，可能影响 strided/member
      view，因此只在既有 4-byte aligned contract 下使用固定宽度读写。
  - 回退策略：
    - 本轮均为 `program.cpp` 内部实现调整，无公开 API/ABI/cache key 变化；
      失败时逐个恢复阈值或 fast path helper 即可。
  - 执行记录（2026-05-24）：
    - CPU 侧已完成：indexed-copy 4B/8B/16B payload fast path、reduce /
      transform / histogram / bucket / grouped-reduce 并行阈值收紧，以及
      histogram / bucket_builder / grouped_reduce 最近 plan replay fast path。
      构建入口 `cmd /c _run_build.cmd` 通过并已部署 `.pyd`。
    - CPU 基准写入
      `benchmarks/results/s10_cpu_native_runtime_plan_replay_final_20260524/`。
      相对本轮前数据，4096/65536 规模 reduce、transform、histogram、
      bucket_builder、grouped_reduce 明显收敛；CPU first-call 仍显著优于
      vanilla 1.8.0。剩余 warm 弱项主要是 gather/scatter 65K/1M、
      scatter_add 4K/65K、compact 65K/1M，以及 histogram/grouped_reduce
      小规模的固定成本。
    - CUDA 侧完成 native runtime 异步提交：除 sort 的就地回写链路外，
      CUB/native scan、reduce、select、transform、indexed-copy、
      scatter_add、histogram、bucket_builder、grouped_reduce 不再在
      primitive 内部强制 `cudaStreamSynchronize` / `cudaDeviceSynchronize`，
      由 `ti.sync()`、readback 或同 stream 后续 work 保证完成语义。
      CUDA 聚合基准写入
      `benchmarks/results/s10_cuda_async_native_submit_20260524/` 和
      `benchmarks/results/s10_cuda_direct_hist_async_submit_20260524/`。
      gather/scatter/transform/reduce warm 固定成本显著下降；histogram i32
      bin direct path 将 workspace 从 CUB temp 降为 0，但对 warm runtime
      只小幅变化。
    - Vulkan 侧本轮只做归因 probe，结果写入
      `benchmarks/results/s10_vulkan_weak_runtime_probe_20260524/`。scatter_add
      从 4096 到 1048576 仅从约 0.81 ms 增至约 1.02 ms，bucket/compact 也
      呈接近固定底座，说明当前弱项主要是 descriptor/resource binding、
      barrier 和 command submission 常数，而不是 two-level 聚合可直接解决的
      规模项。Vulkan command/resource replay 仍列为后续高 ROI 后端专项。
    - focused tests 拆分通过：`test_histogram.py` 10 passed、
      `test_scatter_add.py` 18 passed、`test_grouped_reduce.py` 17 passed、
      `test_bucket_builder.py` 20 passed、`test_indexed_copy.py` 27 passed。
      五个文件混跑仍可能在 teardown/report 阶段触发既有 Windows 后端混跑
      access violation；本轮拆分验证未复现功能失败。
  - 执行记录（2026-05-24 IMA/atomic follow-up）：
    - 修复 CPU dense scalar/matrix field `fill()` 的 helper-kernel IMA 触发点。
      Python scope 下可证明的 CPU dense scalar field 和 matrix/vector field
      component fill 直接调用 `Program::fill_dense_field()`，不再为常见
      `dst.fill(0)` 生成 JIT helper kernel；非 CPU、非 dense 或不支持 dtype
      仍回落到原 helper。
    - CUDA scatter_add / grouped_reduce 的 atomic path 增加小桶数 block-private
      shared aggregation 分支：当 `num_items >= 2048`、bucket/group 数不超过
      1024 且 block flush 工作量受控时，先在 block 内聚合，再写全局
      atomic。默认大桶数路径保持直接 global atomic，避免固定开销扩大。
    - 正确性/稳定性验证通过：
      `test_field.py::test_scalar_field_cpu_dense_native_fill` 6 passed，
      `test_indexed_copy.py::test_experimental_gather_scatter_native_dense_matrix_field_components`
      3 passed，`test_vulkan_native_lifecycle.py` 1 passed，五文件组合
      `test_histogram.py test_scatter_add.py test_grouped_reduce.py
      test_bucket_builder.py test_indexed_copy.py` 92 passed。
    - 全矩阵基准写入
      `benchmarks/results/atomic_ima_full_20260524/`，高碰撞 atomic 窄矩阵写入
      `benchmarks/results/atomic_hot64_20260524/`。结果显示 first-call/compile
      proxy 仍全面优于 vanilla 1.8.0；CUDA/Vulkan scatter_add、histogram、
      grouped_reduce 的 warm runtime 仍主要受 native field 调用固定成本和
      command/sync 窗口限制，block-private 分支不是决定性优化，后续应优先
      做 atomic/order primitive 的 command/resource replay 或 fused submission。
    - PrimitivePlan warm-call replay 更新后，公开 API 路径更接近 JIT kernel
      的“首次选择、后续短路径 launch”范式：同一 workspace、同一对象、同一语义
      先尝试 exact-object hot plan，命中后跳过完整 request check / backend
      routing / dict cache key 生成，miss 时仍回到稳定 object-key cache 和原有
      fallback。本轮不改变 `method="auto"` 的 native 默认策略。CUDA/Vulkan
      复测写入 `benchmarks/results/plan_fast_replay_cuda_vulkan_20260524/`。
      相对 `atomic_ima_full_20260524` 的 Forge warm median，中位改善：
      CUDA scatter_add 7.6x、grouped_reduce 3.9x、bucket_builder 3.0x、
      histogram 1.7x；Vulkan scatter_add 4.3x、grouped_reduce 3.5x、
      bucket_builder 2.1x、histogram 2.0x。相对 vanilla 1.8.0，CUDA
      scatter_add / grouped_reduce / bucket_builder 已接近或优于 vanilla，
      CUDA histogram 和 Vulkan histogram 仍慢，说明剩余瓶颈主要在后端
      histogram/command/resource 路径，而不是 Python routing。
    - hot replay follow-up 后，histogram 顶层入口新增 exact-object 直达
      `_native_histogram_plan` / staged plan group 的短路径，命中时不再先做
      dtype payload 解析和完整 request check。CUDA histogram 同时去掉 direct
      path 的旧 CUB cache/mutex 壳，并为 `num_bins <= 4096`、
      `num_items <= 8192` 增加 single-block shared aggregation，一次 launch
      完成 clear/count/writeback；中大规模仍保留 direct path，避免单 block
      串行化。CUDA/Vulkan histogram 复测写入
      `benchmarks/results/hist_cuda_vulkan_short_final_20260524/`，全操作矩阵写入
      `benchmarks/results/all_ops_hot_replay_followup_20260524/`。该矩阵中
      CUDA histogram warm median 为 0.055/0.052/0.045 ms，对比 vanilla
      0.061/0.117/0.072 ms；Vulkan histogram first-call 仍优于 vanilla，
      warm median 为 0.183/0.211/0.334 ms，对比 vanilla
      0.153/0.162/0.163 ms。Vulkan 4096-bin shared shader 上限已扩到
      4096，以覆盖小规模 single/shared；private 默认上限仍保留 512。
      试验表明把 private 默认上限提高到 4096 虽可改善 warm runtime，但会让
      65536/1M first-call 变差并引入 0.5-8 MiB partial workspace，因此暂不
      作为默认策略。
    - 已重新构建并生成本地 wheel：
      `dist/taichi_forge-0.4.0-cp310-cp310-win_amd64.whl`。文档中旧 miniforge
      Python 路径当前不存在，本轮使用实际验证环境
      `C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe`
      校验 staged wheel 内容可 `import taichi_forge` 并 `ti.init(arch=ti.cpu)`。
    - 1/2/5 后续推进（2026-05-24）：
      - 已将单 plan replay 逻辑抽成 `_try_native_plan_from_cache`、
        `_record_native_primitive_plan` 和统一 cache lookup/store helper。
        reduce、histogram、transform、indexed-copy、scatter-add/add-merge、
        bucket-builder、grouped-reduce、scan 均复用同一条
        exact-object hot plan -> stable object-key cache -> program-id check ->
        native invoke 路径，避免每个 primitive 各维护一套重复逻辑。
      - 已将 component plan group replay 抽成
        `_try_native_plan_group_from_cache`，并为 reduce、transform、
        indexed-copy、scatter-add、grouped-reduce、scan 增加当前 group
        热缓存指针；同一 workspace/executor 上的 matrix field 或
        StructNdarray tensor member 第二次调用可先走 exact-object group
        replay，miss 后仍使用稳定 object-key cache。
      - `SortWorkspace` / `CompactWorkspace` / `BucketBuilderWorkspace`
        共享的 order/apply in-place group replay 也切到统一 group helper；
        out-of-place apply 继续复用内部 `IndexedCopyWorkspace`，tensor member
        in-place apply 继续复用内部 `IndexedCopyWorkspace` +
        `TransformWorkspace`，未新增 helper kernel 或公开 API。
      - 本轮没有推进 3/4（block-private/two-level 扩展和 Vulkan 4096-private
        默认化），因为近期结果显示其 ROI 低于 plan replay 与 order/apply
        完整化；相关后端代码保持既有策略。
      - 正确性验证：
        `py_compile` 通过；plan/cache/order-apply 定点测试 9 passed；
        CPU 聚合子集 8 passed；CUDA 聚合子集 4 passed；Vulkan 聚合子集
        4 passed。pytest 仅报告 `tests/.pytest_cache` 写入权限 warning。
      - dense field 小矩阵基准写入
        `benchmarks/results/primitive_plan_unified_20260524/summary.csv`。
        4096/65536 两个规模下，Forge first-call 相比 vanilla 全部更快：
        CPU 多数为 1.5%-4.2% 的 vanilla first-call，CUDA 为 1.7%-22.0%，
        Vulkan 为 16.2%-87.1%。warm runtime 方面，CPU/CUDA 大多数算法
        优于 vanilla；CPU 64K gather/scatter/compact 和 CUDA scatter_add
        仍偏慢，Vulkan reduce/transform/gather/scatter_add/histogram/
        grouped_reduce 仍主要受 backend command/resource 固定开销限制。
        workspace peak 未出现新的大额分配：CPU native 多数为 0，
        CUDA bucket/reduce 保持既有 runtime workspace，Vulkan transform
        仍为 8B params buffer，compact/bucket 保持既有 order/scan workspace。
      - StructNdarray/order-apply 4K 基准写入同一目录：
        `struct_cpu_4096.json`、`struct_cuda_4096.json`、
        `struct_vulkan_4096.json`。CPU/CUDA/Vulkan 所有条目 `ok=true`；
        sort/compact/bucket tensor-member 路径继续复用共享 order/apply
        workspace，workspace peak 保持 order pair + component apply 的预期范围。
      - 已重新打包本地 wheel：
        `dist/taichi_forge-0.4.0-cp310-cp310-win_amd64.whl`。wheel 解压到
        `build_llvm20_test/wheel_smoke_primitive_plan_unified_20260524_c`
        后在 Python 3.10.11 中完成 `import taichi_forge` 和
        `ti.init(arch=ti.cpu)` 验证。

### S10 暂缓项与架构审视（2026-05-24）

当前决定暂缓继续推进运行时专项优化，先保留问题和后续判断依据，避免在缺少
物理引擎 workload 的情况下为了 microbenchmark 继续扩大后端复杂度。

暂缓项：

1. **Vulkan command/resource replay**：预期对 warm runtime 固定开销有效，但会触达
   descriptor lifetime、command buffer 复用、barrier 放置和 reset/IMA 安全。近期
   Vulkan IMA 更容易暴露在 native 调用链，因此该项应在独立 lifecycle 压测和
   资源所有权审计后再做，不在当前轮继续推进。
2. **Vulkan reduce/histogram/grouped-reduce/scatter-add 深入 shader 调优**：
   当前弱项主要是固定 submission/resource 成本，局部 shader 调优只能改善一部分
   大规模吞吐，不能解决 4K/64K warm 调用底座。先等 command/resource replay 或
   真实 workload 证明瓶颈后再选具体算法深挖。
3. **CPU gather/scatter/compact 继续追齐 vanilla warm runtime**：当前 CPU first-call
   和编译指标已优于 vanilla，warm 弱项集中在 indexed copy 和 compact 的小/中规模
   循环、稳定写回与 Python public-entry 开销。可后续做更低层 C++ loop/threshold
   调整，但不应再引入 Taichi helper kernel。
4. **CUDA grouped-reduce 1M 和高碰撞 atomic 专项**：已有 block-private 分支，但默认
   workload 不总是受益。后续若物理引擎确有高碰撞 group/reduce 热点，再按 key 分布
   和 group 数量选择 warp/block aggregation 策略。
5. **workspace/storage 压缩**：目前 runtime 优先级高于存储；bucket、histogram、
   grouped-reduce 的 partial/order workspace 可以继续压缩或共享，但应在运行时主链路
   稳定后处理，避免反复改 cache key 和 reset 语义。
6. **SparseSNodeView / active-list native path**：本轮继续跳过 sparse 结构。dense
   field 的 `base + offset + stride` 成功经验不能直接推广到 pointer/bitmasked/
   dynamic/hash SNode；sparse 需要 active-list/listgen contract。

当前架构判断：

- **核心架构足够成立**：算法入口仍保持原 public API；内部已经收敛为
  `_PrimitiveView` 证明存储语义，`_NativePrimitivePlan` /
  `_NativePrimitivePlanGroup` 记录 native replay，`Program` 暴露 CPU/CUDA/Vulkan
  entrypoint。该路径不会生成新的 Taichi IR，也不会进入 offline cache key。
- **Python routing 文件仍偏大**：`_algorithms.py` 同时承担 public validation、
  PrimitiveView、workspace、plan replay、fallback 和 backend dispatch。虽然 hot plan
  replay 已经压短 warm 路径，但从维护性看，后续可以把 capability/method routing
  和 workspace support matrix 拆成内部表驱动模块；该拆分不应改变 public API，也不应
  增加 per-call 动态分派成本。
- **重复 capability 检查是兼容性边界，不是主要性能瓶颈**：大量
  `hasattr(prog, "...")` 和 `*_available()` 是为了兼容不同 pybind/build 后端组合。
  在 hot plan 命中后，大多数完整检查已经被绕过。后续若继续清理，应先做内部
  capability cache，而不是直接删除检查。
- **legacy fallback 已经被观测化，但仍是兼容路径**：
  `set_legacy_helper_auto_fallback_enabled()` 和 fallback count API 让 helper IR
  触发可见；默认仍允许 `method="auto"` fallback，保证旧代码可运行。该 API 是新增的
  调试/诊断入口，不改变现有算法函数签名和默认语义。
- **Dense field proof 仍故意保守**：当前只对 1D scalar dense field、可证明
  contiguous 的 field indices/key，以及 vector/matrix component 拆分路径开启 native。
  proof 失败必须 fallback，不能为了覆盖率扩大 raw pointer 假设。

遗留简化实现：

- `sort(method="legacy")` 仍保留 odd-even merge 兼容实现，且多处 `sync()` 属于旧
  helper/host fallback 语义；native radix/CPU sort 才是性能主路径。
- `sort_by_key()` 多 key 仍主要依赖 host stable path，尚未接入完整 device multi-key
  stable sort。
- `experimental_compact()` 的 whole tensor member 通过 order/apply 间接实现，
  strided compact backend 尚未直接支持。
- `experimental_bucket_builder()` 的 whole tensor member 仍通过 order buffer +
  apply values 实现；这保持语义清楚，但不是最少 dispatch 的实现。
- `experimental_histogram()` 对 StructNdarray scalar member 依赖 staging/native
  transform 组合；尚无真正 strided histogram backend。
- `experimental_grouped_reduce()` 对 StructNdarray tensor member 通过 component
  plan group 拆分；对 dense vector/matrix field grouped-reduce 仍未作为重点默认路径。
- `PrefixSumExecutor` 的 field fallback 仍会使用 legacy i32 field workspace；native
  dense/ndarray/member path 不受影响。

不必要或可收敛的防御性编程候选：

- hot-cache 尝试阶段的 `try/except (AttributeError, TypeError, ValueError)` 可在后续
  稳定后收窄，否则可能把真正的内部 bug 伪装成 fallback miss。
- 每个 primitive 自己写 public validation、backend method error message 和
  capability probing，维护成本偏高；适合后续抽出只读 support matrix/route table。
- 一些 `hasattr + available + hasattr(method)` 三段式检查在已统一 pybind 后可以由
  capability cache 合并，但不能先于 wheel/多后端 ABI 稳定性验证删除。

同步和检查策略现状：

- CUDA native primitive 已尽量去掉内部强制同步，由 `ti.sync()`、readback 或同 stream
  后续 work 保证完成；sort 和 host fallback 仍保留同步边界。
- Vulkan native primitive 当前仍有较多 `synchronize()`、`buffer_barrier()` 和
  descriptor/resource binding 保护。barrier 大多是跨 dispatch 数据依赖所需，但
  command submission 和资源准备固定成本仍是 warm runtime 弱项。该项暂缓为
  command/resource replay 专项，不在普通算法路由中继续堆补丁。
- legacy helper fallback 中的 `sync()` 主要用于保持旧 helper/host 路径可观测完成；
  native-first 路径不应继承这些同步。

文档与现状需要修正的点：

- 本轮已把下方支持矩阵从早期 P2/P3/P4 规划标记更新为当前状态；后续新增 primitive
  或默认策略切换时，应继续使用“当前支持 / fallback / 暂缓”三态描述。
- “建议启动顺序”仍是原始规划，可保留为历史，但不应被当成当前执行顺序；当前应进入
  P8 物理引擎级验证或独立 Vulkan command/resource replay 设计。
- benchmark 说明需要继续标注 exact artifact path、first-call/warm/storage 三类指标，
  不再只写单个 runtime 结论。

对外 API 影响：

- 现有 `experimental_*` 函数签名和默认 `method="auto"` 语义保持兼容。
- 新增 workspace/diagnostic API：
  `set_legacy_helper_auto_fallback_enabled()`、
  `reset_legacy_helper_auto_fallback_policy()`、
  `set_legacy_helper_fallback_counting_enabled()`、
  `get_legacy_helper_fallback_counts()` 等，用于检测 helper fallback。
- 新增/扩展的显式 method 名称（如 `cpu_native`、`cuda_device`、`vulkan_native`、
  `two_level`、`*_two_level`、`segmented`）仍属于 experimental algorithm API；
  真实默认策略切换必须等待 P8 workload 验证。

### P8. 物理引擎级验证

目标：用真实 engine workload 决策默认策略，而不是只看 microbenchmark。

推荐验证集合：

- dense particle/rigid body 属性数组；
- sparse grid / contact grid；
- broadphase bucket/sort；
- contact manifold reduce/grouped reduce；
- solver iteration step；
- render/simulation mixed workload。

指标：

- cold compile elapsed；
- warm-cache compile elapsed；
- `.tic` count / bytes；
- first-frame latency；
- steady 1000-step average；
- p95/p99 step time；
- workspace/VRAM high-water mark；
- correctness diff；
- backend availability and fallback reason。

验收：

- 对代表性物理引擎，默认配置不能因算法 primitive 优化导致 cold compile
  不可接受增长。
- 若某 native path 只改善 microbenchmark、恶化 engine compile 或 first-frame
  latency，保持 opt-in。

## 支持矩阵

当前每个 primitive 必须维护以下矩阵。这里记录的是 2026-05-24 代码状态，
不是早期 P2/P3/P4 规划标记。

| Storage | scan | reduce | histogram | transform | gather/scatter | scatter_add | compact | bucket | grouped_reduce | sort |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ndarray | native first | native first | native first | native first | native first | native first | native first | native first | native first | CUDA/Vulkan native first，CPU auto host stable / explicit native |
| StructNdarray raw payload | no numeric | no numeric | no numeric | no numeric | opaque copy/order only | no numeric | opaque payload/order | opaque payload/order | no numeric | payload values supported by native/host paths where method supports payload |
| StructNdarray scalar member | native strided | native strided | staged native scalar member | native strided | native strided | native strided / two-level opt-in | selected native/order path | selected native/order path | native strided / staged two-level | scalar values/keys selected |
| StructNdarray tensor member | component group | component group | unsupported | packed/component group | packed/component group | component group / two-level opt-in | order/apply lanes | order/apply lanes | component group | order once + lane apply |
| DenseFieldView scalar | native first | native first | native first | native first | native first，支持 ndarray 或 contiguous dense-field indices | native first，支持 ndarray 或 contiguous dense-field indices | native/fused compact | native first | native first | field sort 仍为 host/legacy 或显式旧 device helper，未 native-first |
| Dense vector/matrix field component | component group | component group | scalar component only | component group | component group | component group | scalar compact only | scalar bucket only | not default | not default |
| SparseSNodeView | skipped | skipped | skipped | limited fallback | skipped | limited fallback | skipped | skipped | skipped | later |

## 默认策略

- `ndarray` 和 `StructNdarray` 已有 native path 继续保持。
- dense field native path 在严格 contiguity proof 通过后可以默认开启。
- sparse SNode active-list native path 初期使用 opt-in flag，真实物理引擎验证后
  再考虑默认开启。
- topology mutation、dynamic append、deactivation 相关优化默认保守。
- 不将 `StructNdarray` 宣称为 field/SNode 替代方案。

## 风险和回退

风险：

- field contiguity proof 过宽导致错误 raw pointer path。
- offset/stride alias key 不完整导致 CSE 或 cache 误复用。
- Vulkan descriptor/parameter lifetime 导致 reset 或 IMA。
- CUDA toolkit/CUB 依赖影响 wheel 可用性。
- sparse active list stale 导致遗漏或重复处理 active element。

回退：

- 每个新 native path 必须保留 legacy fallback。
- backend-specific path 可单独关闭。
- sparse native 初期使用 explicit flag。
- cache key 变更必须版本化或触发安全重编译。
- 如果真实引擎 compile/cache 指标恶化，默认回退，即使 microbenchmark runtime
  更快。

## 建议启动顺序

第一轮不要同时修改 sparse topology 和全部 backend。建议按以下顺序启动：

1. S0/S1：编码、构建、wheel、benchmark 规则和 compile/cache/IR 基线。
2. S2：`PrimitiveView` 内部抽象，只覆盖 ndarray 和 StructNdarray，不碰
   field/SNode 语义。
3. S3/S4 子集：dense 1D scalar field native path，先做
   scan/reduce/transform。
4. S4 backend 顺序：CPU 先导，CUDA 连接 CUB/device API，Vulkan 连接 fixed
   parameter native shader。
5. S5/S6 子集：gather/scatter/scatter_add 和 order/apply，覆盖
   compact/bucket/sort 的 out-of-place dense field。
6. S7：vector/matrix component dense field。
7. S8：`SparseSNodeView` 和 active-list contract，只做 opt-in proof。
8. S9：sparse reduce/histogram/compact/bucket native path。
9. S10：代表性物理引擎验证，决定哪些路径可以默认开启。

第一轮最小可交付目标：

- 指标基线完整；
- `PrimitiveView` 不改变现有 API；
- dense field scan/reduce/transform 在至少 CPU + CUDA 上证明 compile calls
  下降，runtime 不低于旧 helper 95%；
- Vulkan 至少完成 transform 或 reduce 的 native dense field proof；
- 有一组真实物理引擎 cold compile + steady-step 对照。

这能最快验证本规划的核心假设：`PrimitiveView + native backend` 能否在保持
field/SNode 用户模型的前提下，同时改善 compile、runtime、workspace 和
cache 指标。
