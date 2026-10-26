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
