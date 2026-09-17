# 原生光栅与射线程序

[English](native_ray_programs.en.md) · [硬件 API](forge_api_reference.zh.md)

本文介绍 0.6.3 的原生光栅与射线接口。使用对应发布文档和兼容的 shim/runtime，不要求 Git commit 相同。
Vulkan 示例已在 Windows/NVIDIA 上执行；其他设备或平台需要核对 capability，不能据此承诺性能。

## 选择适合的入口

| 用途 | 公共入口 | 执行方式 |
| --- | --- | --- |
| 无自定义 shader 的三角形批量查询 | `ti.hardware.ray.triangle_scene()` | 当前后端的 Vulkan AS/ray query 或 CUDA/OptiX |
| 在 Taichi Vulkan kernel 内查询 | `ti.types.acceleration_structure()` 与 ray-query intrinsic | Inline ray query |
| 光栅 shader 查询同一个 AS | 带 AS binding 的 `ti.hardware.graphics.record_pass()` | Vulkan graphics pipeline，参见硬件 API |
| 自定义 ray-generation、miss、triangle hit 程序 | `VulkanRayTracingPipeline` 或 `OptixProvider.program()` | Vulkan RT pipeline 或 OptiX pipeline |

批量工厂在创建时选择原生路线，不切换 `ti.init()` 后端、不迁移数组、不自动开启实验性 retained recipe。
缺硬件或依赖时明确失败，不静默运行软件渲染器。既有后端专属入口继续可用；使用硬件不保证小规模总成本更低。

```python
ti.init(arch=ti.vulkan)  # 也可使用 ti.cuda，但须有可用的 OptiX runtime
with ti.hardware.ray.triangle_scene(vertices, indices) as owner:
    owner.trace_typed(rays, hit_values, hit_indices)
    # 在设备端消费，或在完成后读回。
```

几何输入为紧凑 `f32 (N, 3)` 顶点与 `i32 (M, 3)` 索引，或支持的 AOS vector-3 存储。
ray/typed-hit 输出 shape 沿用批量 API，分配前请查 API 参考。`owner.native_scene` 是实际后端对象，
不是复制；batch/refit 方法直接绑定既有原生方法，`refit()` 返回原生 scene。
先关闭使用它的 Graph，再关闭 owner。调用者传入的 `provider=` 仍由调用者管理；工厂创建的 provider 随 owner 关闭。

## 依赖与可用性

| 路线 | Forge 提供 | 应用／环境提供 |
| --- | --- | --- |
| Vulkan | 资源 owner、RHI pipeline、SBT、recording 和依赖管理 | 支持所需特性的 Vulkan 驱动／设备；预编译 SPIR-V 或显式选择的编译器 |
| CUDA/OptiX | 薄 adapter、受管 pipeline/SBT/launch、内建查询 PTX | 兼容的 NVIDIA 驱动／OptiX vendor runtime；用户 PTX，或用于构建它的 CUDA 编译器和兼容 OptiX 头文件 |

程序构造器不下载或加载 shader 编译器。portable wheel 不携带 vendor runtime、CUDA Toolkit 或 CUDART。
用户 PTX 可能要求比内建批量程序更新的驱动／PTX 支持；选择 adapter 不会翻译不受支持的 PTX。
`provider_path=` 指 **Forge adapter**，`library_path=` 指 **vendor runtime**，均不是 CUDA 安装目录。

Vulkan AS、inline query、RT pipeline 特性彼此独立。初始化 Vulkan 后，`ray.is_available()` 保持既有
batch/query 含义；`ray.is_program_available()` 检查可编程 RT 路线。OptiX 可按所需功能选择：

```python
provider = ti.hardware.ray.OptixProvider(required_features=("program", "instances"))
```

筛选发生在创建 context 前。显式指定但缺功能的 adapter 会失败，不会被偷偷替换。
省略 `required_features` 保留普通 batch 选择行为。被动 `ti.hardware.report()` 不加载可选库；
`ray.program.vulkan` 与 `ray.program.optix` 分别描述两条路线。provider 未加载不等于设备不支持。

## 完整可运行示例

两个示例都执行视线查询，并由 Taichi kernel 消费距离，不依赖渲染器。源文件导出仅创建新文件，不覆盖旧文件。
在已经配置好所选编译工具的 shell 中运行。

Vulkan（PowerShell，`glslc` 由用户提供）：

```powershell
python -m taichi_forge.examples.features.native_ray_program_vulkan --write-sources ray-shaders
glslc --target-env=vulkan1.2 -O ray-shaders/query.rgen -o ray-shaders/query.rgen.spv
glslc --target-env=vulkan1.2 -O ray-shaders/query.rmiss -o ray-shaders/query.rmiss.spv
glslc --target-env=vulkan1.2 -O ray-shaders/query.rchit -o ray-shaders/query.rchit.spv
python -m taichi_forge.examples.features.native_ray_program_vulkan --shader-dir ray-shaders --mode direct
python -m taichi_forge.examples.features.native_ray_program_vulkan --shader-dir ray-shaders --mode graph
```

OptiX（替换 SDK include 路径，并配置兼容的 CUDA host compiler）：

```powershell
python -m taichi_forge.examples.features.native_ray_program_optix --write-source query.cu
nvcc --ptx --std=c++17 --gpu-architecture=compute_75 -I"C:/path/to/OptiX/include" query.cu -o query.ptx
python -m taichi_forge.examples.features.native_ray_program_optix --ptx query.ptx --mode direct
python -m taichi_forge.examples.features.native_ray_program_optix --ptx query.ptx --mode graph
```

仅在需要覆盖发现结果时使用 `--provider`／`--library`。已有兼容产物时直接跳过导出和编译。
`compute_75` 是示例目标，不代表任意 PTX／编译器／驱动组合均兼容。示例源码完整列出 binding 与参数 ABI。

## 资源、shader 与 SBT 合同

程序是**可信应用代码**，不是沙箱代码。Forge 在准备边界验证声明的布局、资源及可检查合同，
不能证明任意外部 shader 都遵守声明。

- Vulkan `SpirvShader` 支持 raygen、miss、closest-hit、any-hit，三角形组使用 `VulkanHitGroup`。
  `VulkanRayBinding` 声明 binding、资源类型、访问方式与 storage-image mip，必须和 shader 相符。
  传入受管类型化资源，不传裸设备地址。
- Buffer 接受支持的 dense storage/view；dtype、字节范围、元素布局和 shader 解释必须一致，不隐式 packing
  或转换。Vulkan 图像使用受管 `Texture`：sampled binding 使用其 sampler/mip chain，storage-image 选择
  一个 `mip_level`。Uniform、sampled-image、AS binding 只读。OptiX texture 引用沿用受管 CUDA texture
  的既有能力，可编程 pipeline 本身不会增加 CUDA mip 支持。
- OptiX 使用 `PtxModule`、`OptixShaderEntry`、`OptixHitGroup` 和 `OptixParameterLayout`。
  参数偏移、对齐、标量宽度和总大小须匹配编译后的结构体；声明 buffer 访问并通过布局绑定 scene/buffer。
  Python 字典不是 CUDA struct reflection。
- SBT record 顺序有语义。实例 `sbt_record_offset`、geometry index、ray-type offset/stride、miss index
  和 trace 指令的操作数必须共同定位到有效记录。默认值保持零偏移、单 ray type。
  改实例 record 映射需要新 scene generation；仅位置 refit 保留映射。
- Any-hit 过滤要求匹配的 geometry/ray flag 与 hit record。opaque／disable-any-hit 明确跳过过滤；
  示例使用不透明射线。alpha 规则、材质选择和 payload 解释由应用负责。
- 当前受管程序覆盖三角形与上述四类 stage，不提供 callable、intersection/custom-AABB、motion blur
  或跨后端 shader 自动翻译。

## 准备一次、执行与关闭

创建资源和程序后调用 `program.record(...)`，再调用 `recording.prepare(bindings).initialize()`。
初始化在重复执行和 Graph capture 之前显式准备设备端 launch/SBT 状态。
可直接 `launch.run()`，也可在 freeze 前将 `launch.graph_recording()` 加入 `GraphBuilder`。
示例包含真正的设备消费者，按 Graph、launch、program、scene、provider 的依赖顺序关闭。

OptiX 在准备阶段选择当前 shim 支持的直接原生 adapter 调用；较旧的兼容 shim 保留回调路径。
Graph 按 runtime 顺序提交这个 prepared call，并与可 capture 的 CUDA 分段组合。
这不代表 OptiX launch 已可 capture，也不消除它的提交成本；执行诊断分别说明提交桥接与 capture 支持。
关闭 launch 或 reset runtime 会使依赖它的 Graph 与 prepared call 失效。

在布局与资源不变的前提下，允许的原位数据更新可以复用 prepared launch。
替换绑定、修改 shader／布局／SBT 结构、改变容量或 reset runtime，需要相应的新准备／generation。
借用的数组、纹理、scene 必须存活到消费者完成。`run()` 是提交，不是 CPU 消费 fence；
在真正的完成边界等待，不要为每条命令增加全局同步。

| 执行形式 | 复用内容 | 需要注意 |
| --- | --- | --- |
| Direct prepared launch | Program、SBT、固定绑定 | 仍发生原生命令提交 |
| Ordinary Graph | 冻结顺序、保留的 owner | native action 仍可能每次重录 |
| Vulkan fixed-frame recipe | 合法区域的不可变命令／参数帧 | refit 等分段仍可能分别提交 |
| Graph 中的 OptiX program | Prepared launch 与 runtime 顺序 | **不是 CUDA Graph capture**，不能从 Graph 标签推断 capture |

## Recipe、报告与诊断

先 freeze 完整 producer/query/consumer Graph，再调用
`definition.search_recipes(engine="compileiq", ...)`。实际区域合法时，provider 可以组合 map fusion
和 Vulkan immutable frame。没有替代区域就没有相应候选；搜索器不会生成另一套 shader。
CompileIQ 只接收完整 recipe identity 与应用评价成本，不接收 RT 裸 block/tile/vendor 开关。
保留 baseline，评价完整有效窗口，包括准备摊销、packing、refit、消费与完成。

如果没有 immutable CUDA frame 候选，可读取
`definition.recipe_catalog().discovery_report()["providers"]` 中 binding-frame provider 的
`provider_explanation`。其中 `reasons` 区分尚不支持的 SNode／AS 绑定、缺少 capture 合同的
native command，以及不符合单一编译 Graph 要求的拓扑。这些仅是该候选的准入原因，
不表示整个 CUDA 后端没有 capture。特别地，typed OptiX batch query 仍走有序 native 路径，
其周围符合条件的 kernel 分段可以 replay。不要为了生成候选而删除生命周期检查或改变 baseline。

报告将 provider 声明的代码／布局／SBT／构建及资源计划事实与实测数据分开。
物理执行身份不是实时指针或冷／热分配快照。资源替换可以保持等价计划身份，
但 shader／布局／SBT 变化会使旧合同不适用。新进程重新建立等价 definition 与 program，
再使用[recipe 接入](graph_recipe_integration.zh.md)中的公开 resolve 与 applicability API。
能解析 recipe 不代表旧性能证据仍适用。

`launch.memory_report()` 只报告声明的所有权范围。AS、借用输出、临时上传和不透明驱动分配可能不在其中；
requested bytes 不等于总 VRAM residency，缺测项仍是 unknown。
原生执行、可 capture 和更快是三个不同事实。诊断时区分缺设备特性、未加载／缺 adapter、程序编译失败、
绑定不合法、ordinary／重录路径，以及正确但更慢的候选。Profiling 显式启用，不成为 replay 的固定门槛。
