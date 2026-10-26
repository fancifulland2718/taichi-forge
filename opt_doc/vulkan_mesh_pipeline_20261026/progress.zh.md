# S1 验证记录

2026-10-26，Windows。版本 0.7.0，私有 ABI 224；未发布 wheel。

## 交付

- `VulkanCapabilities` 与公开字典新增 mesh/task 各轴每组线程上限；非 Vulkan
  返回同形状的零值字典。既有 draw group count 合同保持。
- 共用 RHI 创建阶段在已验证入口/stage 后检查可确定的 LocalSize、LocalSizeId 默认值、
  WorkgroupSize 常量及 mesh 输出顶点/图元数。WorkgroupSize 遵守覆盖优先级。
- 未知常量表达式不猜值，不误用被覆盖的尺寸。只采用设备硬约束，不增加人为上限。
- 保留现有 shader 字节、pipeline 身份、JIT/AOT 描述、缓存与执行路径。

## 检查及结果

| 检查 | 实际结果 | 范围 |
| --- | --- | --- |
| 原生边界测试 | 3 passed | 每轴、线程乘积、零值、溢出、task/mesh 分离、输出数、所选入口、默认常量、覆盖优先级、截断指令 |
| Python 实机小回归 | 10 passed，11.79 s | mesh/task 直接绘制、Graph、两种越界失败后恢复、能力字典、CPU/CUDA 隔离、AOT 独立/借用加载 |
| AOT 数值验证 | 上述 10 项中的 2 个导出用例 | ordinary mesh 与 selected task+mesh；公开/私有资源；下游数值 kernel 显式梯度，不代表 raster AD |
| SPIR-V 目标环境 | 3 个现有夹具通过 `spirv-val --target-env vulkan1.3` | mesh、task、AOT color mesh；不使用 Vulkan 1.4 core 功能 |
| Python AST / whitespace | 通过 | 两个变动 Python 文件、`git diff --check` |

测试日志位于本地忽略目录 `temp_outputs/vulkan-mesh-20261026/`。
使用现有隔离 native staging 和 CPython 3.13 shim，安装至隔离验证目录；未替换用户正式包。
native DLL SHA256：`3087afe6c33eeef02132fc870fb6fab1f6b8f97f65fa992cb2fdcc4ff12e7a3b`。
shim SHA256：`8f8b4e64ae514613768713b6705e7f4768eba61835689be885228d4b346e655a`。
staging 的旧 git 版本标签不是此次源码身份；以基线、补丁及二进制哈希说明验证对象。

## 收益、成本和剩余边界

收益是将确定的非法配置在管线准备时拒绝，错误指明 stage、尺寸或输出限制；重复失败
不留下注册对象，随后合法创建可继续。新增工作是一遍冷阶段声明扫描，存储量按实际声明
增长，不按不可信的 SPIR-V ID bound 分配。draw/replay/submit 未加扫描或查询。
本次未做性能 A/B，不声明帧率提升或精确冷阶段开销。

共享/payload/输出内存总量、一般常量表达式和 shader 动态访问仍由验证层/驱动及调用者
负责。这不是完整 SPIR-V 安全验证器。现有精度转换提示和 `tools/image.py` 的转义警告
不是本次新增；测试未跳过本切片的 GPU 路径。Linux 及其他 GPU 本轮未验证。

下一步 S2 独立处理标准 specialization 值、有效值校验、pipeline identity 和 AOT
重建；S3 再检查缓存/退休，不重新建立一套渲染调度器。

## S2 显式冷阶段特化

基线 `327e7d52b`，版本 0.7.0，私有 ABI 225。使用 `SpirvShader.specialize()`
显式冻结标量 SpecId。输入、结果按 Vulkan 1.3 验证；应用仍选择合法且语义适合的变体。
未传常量取声明默认值，原 shader 保持不变。产物只包含有效代码，不引入第二套 AOT
override 格式，也没有默认搜索或逐帧转换。

- 20 项针对性 Python 用例覆盖标量类型/值拒绝、源对象不变、不同值着色、相同代码身份、
  descriptor 数组表达式、mesh 有效尺寸、CPU/CUDA 能力隔离，以及 ordinary/selected/
  specialized 三种 AOT 独立和借用加载。另单独确认关闭同内容实例后另一份 Graph 仍可用。
- 首轮 19 passed / 1 failed：测试误要求删除全部 OpSpecConstantOp。固定版本上游 pass
  会保留已固定输入的浮点 Select；实际图像正确且无 SpecId。改为检查无可覆写标量/ID，
  相关 12 项重跑通过。没有为满足错误断言增加优化 pass。
- 新增 fragment、数组 fragment、mesh 夹具均保留源码，输出目标 Vulkan 1.3。
- cache schema 91、ExecutionPlan 27、GFX 格式 16 不变。JIT/AOT 使用同一份有效字节，
  AOT 消费端不运行特化器。仅 producer shim 新增原生入口，要求 ABI225 配套安装。

Windows / RTX 5090 / driver 610.88，小型 native 图形探针：

| 项目 | 实测 | 解释 |
| --- | --- | --- |
| 808 B fragment 显式特化 | 首次 0.864 ms；后续 31 次中位 0.198 ms | 冷准备成本；不推广为大型 shader 成本 |
| 有效 SPIR-V | 728 B | 删除可覆写值并折叠可处理表达式；未运行完整优化流水线 |
| 640×480 同图像 Graph 批量重放 | 原始 0.0464–0.0488 ms；冻结 0.0432–0.0469 ms | ABBA、每进程段 31×32 帧，host 提交加最终同步；不是 GPU 时间戳或应用收益 |

该探针仅证明固定值路径可执行且没有显著重放负担；不据此宣布普遍渲染加速。
更大的材质/meshlet 变体需要应用证明图像和资源语义后才值得选优。

S2 native DLL SHA256：`a1d12e2711e2efbb3d9b85be3fe216539c634ada360ca30195e05db1fe31bde8`。
日志与原始采样保留于本地忽略目录 `temp_outputs/vulkan-specialization-20261026/`。
本轮未替换正式安装、未发布 wheel，未测试 Linux/其他 GPU。

## S3 构造失败清理与缓存发布

生产修复包含两部分，均在冷创建阶段：

1. graphics 构造函数抛出时，显式释放已创建的 shader module；成功时继续保留模块，
   供后续兼容 render-pass 变体使用。正常提交与在途资源退休顺序不变。
2. `vkCreatePipelineLayout` 和两种 `vkCreateGraphicsPipelines` 包装在失败时抛出
   `BackendRuntimeError`。缓存只接收成功结果，不再依赖 assert 处理驱动失败。
   不替调用者吞掉资源不足错误；资源条件恢复后仍可正常重试，不添加人为限制。

原生测试对本地独立设备的 dispatch table 注入失败并透传其余真实 driver 调用。
每次第 2 个 shader 创建失败，核对成功创建数等于销毁数；继续覆盖 layout 失败、
模板校验失败、恢复创建、ordinary/dynamic 两种 variant 失败重试与成功命中缓存。
测试结束恢复原 table，生产代码不新增 hook 或计数器。

旧 RHI 对照实际显示连续 3 个 module 创建而销毁数为 0，随后 layout 失败触发 assert
中止。新 RHI 的 6 项原生用例通过，包括以上 2 项新测试、2 项既有 cache snapshot
测试和 2 项普通图形准备/绑定测试；20 项 Python 图形/AOT 小回归重新通过，覆盖
CPU/CUDA 能力隔离。未以 RSS 波动代替句柄泄漏证明。

注意原生测试直接链接 RHI 对象，单独切换 DLL 搜索路径不是旧实现对照；已分别重建
基线与修复对象确认上述结果。日志 `lifecycle-baseline.log`、`lifecycle-final.log`、
`tests-lifecycle-final.log` 在前述本地忽略目录。

最终 DLL SHA256：`582cdcdfce456e7680a3e7f540c05dd2645ded10efff60b4d4bbbaa817b9dafe`。
ABI225、cache/plan/provider 格式保持。没有新增 draw/replay 工作、测试框架运行时依赖
或跨设备共享 cache。本次不声称已审计所有 Vulkan 对象创建失败路径。
