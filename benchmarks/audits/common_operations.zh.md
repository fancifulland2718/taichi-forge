# Forge 常用底层操作与高频调用排查清单

2026-11-26 复核 57 项常用底层操作。逐进程样本和构建指纹保存在本机测试目录。

## 逐项处置

| 编号 | 操作 | 处置及剩余边界 |
|---|---|---|
| 01 | 64 位填充 | 已有修复 `f074569f2`。1 MiB、8 MiB 的 CUDA/Vulkan 配对数据见下文。 |
| 02 | 连续设备拷贝 | 已有修复 `0e95555f3`。普通 Vulkan 队列可延迟到消费端等待，借用运行时仍需跨队列同步。 |
| 03 | 结构数组多成员读写 | `646183a81`、`9e07fe501`。所有字节将被覆盖时直接上传。对齐且只选取少量数据时按连续字节段打包，小数组写严格子集时跳过打包规划。重叠、带内部填充的布局保留可见字节及赋值顺序。 |
| 04 | 结构数组单成员 | `646183a81`。大数组按 u32 字复制，修复浮点算术搬运改变负零、NaN 位型的问题。小数组和非字对齐成员沿用整记录读写。 |
| 05 | 矩阵 Field 读回 | `dd5129073`。SOA 分量共用一个主机缓冲和一次生产者等待。3×3 仍需取九个分量。 |
| 06 | 结构 Field 填充／复制 | 按成员发起的操作可以具有不同的稀疏激活、嵌套及梯度语义。尚缺同布局、同语义的逐帧重复派发证据。 |
| 07 | Field 导出 Torch／Paddle | 源码可见同卡经主机的路径。配对环境未安装这两个框架，所选应用也未调用此接口。同卡导出与所有权需要真实框架消费者验证。 |
| 08 | Fortran 连续 NumPy 参数 | `c633642f6`。直接把结果拷回原 F 顺序视图，省去一个整数组临时副本。传入 kernel 的 C 连续暂存以及调用方可见的回写仍保留。 |
| 09 | 跨设备 Torch 参数 | 跨设备兼容分支需交付原设备上的结果和梯度。DLPack 有独立的设备、流及所有权约束，未测跨框架路径。 |
| 10 | 主机逐元素读写 | 每次索引均可被 Python 立即观察。现有批量接口可用于整段消费，本轮未识别可透明合并的主机调用。 |
| 11 | Vulkan 暂存传输 | 短上传内联原型已撤下。四轮配对中 512 B 约 201 → 201 µs，32 KiB 约 233 → 234 µs。批量读回的驱动分配次数仍待实际调用链测量。 |
| 12 | 设备 offsets 重建拓扑 | `1e12d05ac`。GPU 上克隆 i32 offsets 后校验同一份快照，向量化展开分段编号。保留一次读回，用于不可变拓扑、校验和导出。 |
| 13 | 降序稳定排序回退 | `fcc23266d`。用单次稳定索引排序取代 Python 逐组翻转，保留同键次序与 NaN-last。命中主机回退的键和值仍需跨设备搬运。 |
| 14 | 兼容排序逐级等待 | `fcc23266d`。同队列按序提交各级，非空多元素排序仅在末尾等待一次。65,535 个键原有 136 次 Program 同步。 |
| 15 | 前缀和与 Field 回退 | `7950a6192`、`ecae19f65` 修复分段归约及扫描对非零 Field 原点的处理，直接运行与 AOT 都覆盖。普通多级前缀和的工作区和跨块传播仍承担实际数据依赖。 |
| 16 | 集合原语工作区 | 相同对象可命中默认缓存，显式工作区可复用。轮换对象与容量会改变分配需求，缺少可归因的生产 miss 率。 |
| 17 | scatter_add | 成员输入暂存和同地址冲突是两种成本。缺少实际散射分布和载荷布局，未合并不同地址上的更新。 |
| 18 | 标量结果观察 | `to_int`、`to_float` 返回主机值，需要完成读取。设备消费者已有 `device_scalar`。 |
| 19 | Vulkan f32 标量原子 min/max | 已有修复 `c50aeb835`。保留旧值用途、原生能力和目标地址门槛。本轮用 Vulkan Euler 的 5 步与 120 步状态作回归。 |
| 20 | 其他浮点原子 | 地址相同、旧值使用与特殊值处理决定能否合并。当前证据限于 19 的形状。 |
| 21 | 稀疏 SNode | 固定拓扑已有活跃表复用，拓扑变化时需要重建。尚无热负载中重复清理同一未变子树的计数。 |
| 22 | 中间标量返回主机 | `600c3ffac` 在已知消费者中消除系数回写及再取值。用户请求的 Python 返回值和逐轮停止条件仍是主机可见点。 |
| 23 | 普通 kernel 实参映射 | 热调用有静态参数及普通派发快路。新特化需要解析布局和模板参数，本轮未复现同一热特化重复编译。 |
| 24 | 资源租约和锁 | 注册表检查保护并发提交、别名与退休，同批资源已去重。缺少多提交线程的竞争测量。 |
| 25 | CUDA 参数与回传 | 外部数组回传及用户请求返回值会触发同步。08 去除了 NumPy 层的额外副本，设备层需按实际 external/owned 绑定计数。 |
| 26 | Vulkan 描述符 | 已有缓存及回放专用分支。资源身份或布局变化时仍要绑定，缺少相同资源下高 miss 率的证据。 |
| 27 | Vulkan 屏障 | 通用路径跨 transfer 和 compute 保证可见性。未在相同访问依赖下证明可安全删减。 |
| 28 | 在途提交背压 | 到达上限后等待最早 fence 以安全回收命令及资源。需同时测吞吐、峰值占用和尾延迟。 |
| 29 | Graph 字典实参 | 可变字典要在提交时快照。身份稳定的资源已有可复用的 GraphBindingSet。 |
| 30 | Graph 谓词观察 | Vulkan 自动 lowering 在本探针中无谓词 D2H，强制 portable 的 12 次迭代产生 156 B 读取。缓存主机缓冲的原型未取得稳定中大规模收益，已撤下。 |
| 31 | CPU 小循环线程池 | 调度器已有线程数选择。分块成本随每元素工作量变化，缺少足以更改默认值的跨规模证据。 |
| 32 | C API／AOT 绑定 | 普通调用支持变更资源，固定参数已有 prepared execution。本轮用实际归档验证绑定与执行。 |
| 33 | 编译缓存并发 | 同键等待可避免重复编译，磁盘读取和实际编译已在锁外。缺少热命中仍阻塞的轨迹。 |
| 34 | 静态展开与 IR 简化 | 展开、编译开销和稳态收益取决于特化。默认 fast 配置维持原取舍。 |
| 35 | 小整数幂 | 既有 Euler 改写的 120 步末态超差且整步更慢。保留原语义。 |
| 36 | 整除与周期索引 | 既有 RTX 微测及 LBM 整步未显示实用收益。负数与尾块需保留原索引语义。 |
| 37 | 倒数平方根与归一化 | 既有实验无跨后端稳定优势。不同接口对 eps 的定义也不同。 |
| 38 | SVD／极分解 | 默认迭代与精度有关。缺少部分输出即可满足精度及退化输入的真实调用。 |
| 39 | RNG | 每次抽样要推进状态。未定位到在同一抽样中重复推进的指令。 |
| 40 | 小矩阵乘法 | 静态尺寸决定展开、寄存器与梯度。缺少同尺寸同精度的溢出或整步证据。 |
| 41 | 稠密／稀疏寻址 | 动态分量和需激活的稀疏节点有不同依赖。需看最终设备指令是否重复计算同一地址。 |
| 42 | 位打包原子 | 相邻量化成员共享存储字，读改写要保留邻位和旧值。未证明有无观察者的独占更新。 |
| 43 | 私有临时数组清零 | `1e12d05ac`、`646183a81` 仅对首次消费前必被完整覆盖的内部存储省去清零。公共数组继续按既有契约初始化。 |
| 44 | GPU block 选择 | 底层操作与应用主测沿用自动几何，另列的 Euler 历史对照使用两边相同的局部 block 32。 |
| 45 | CUDA 小 GEMM | cuBLAS 句柄的共享提交受锁保护，另有 prepared 路径。未定位到当前负载中的锁瓶颈。 |
| 46 | 纹理访问 | LOD、梯度和过滤需要不同指令。没有相同采样语义下的冗余取样证据。 |
| 47 | 逐元素算术 | 常见操作已有直接 lowering。所选负载未发现可保持精度、又能删去的重复操作。 |
| 48 | 连续 transform／reduce | PrimitiveSequence 可按布局和输出可见性融合。尚无符合条件却反复写回中间数组的生产轨迹。 |
| 49 | stencil 块级缓存 | 缓存能减少全局读，同时增加共享内存和同步。门槛依赖邻域与边界比例。 |
| 50 | 矩阵自由求解器 | `600c3ffac`。系数作为带类型的 kernel 参数传递，合并向量初始化与必要的快照，保留每轮收敛判定和工作场的释放。 |
| 51 | 稀疏矩阵 RHS 转换 | 载体及返回位置决定上传和下载。已有设备 ndarray 分支，尚缺相同消费者上的重复转换证据。 |
| 52 | Tape 清梯度 | 梯度初值影响反向结果，Vulkan 合格稠密字段已有批量路径。未识别可证明失活的梯度集。 |
| 53 | FFT 计划 | 描述符决定计划缓存键。不同形状的首次建计划有必要，尚缺相同形状的异常 miss。 |
| 54 | Graph 物化 | 同定义可对应不同物理配方。当前证据未发现同一物理产物在热回放中重复物化。 |
| 55 | GGUI 索引与变换 | Scene 的排队绘制及 SceneV2 的即时提交有不同的资源持有期。当前构建关闭 GGUI，逐帧字节、跨设备队列归属和图形资源生命周期仍待实际场景验证。 |
| 56 | 帧缓冲与诊断读回 | 属于显式请求主机结果的入口。应用计时只含生产步骤，检查点在计时区间外。 |
| 57 | OptiX GAS 构建 | 初建完成前 scratch 和几何数据必须存活。未取得实际帧循环反复重建同一 GAS 的证据。 |

## 首轮优先项与定位依据

| 证据 | 编号与操作 | 当前最值得验证的事 |
|---|---|---|
| 已修复并复测 | 01 大数组 64 位填充、02 设备内连续拷贝 | 跟踪小数组的逐次同步耗时、借用运行时的队列顺序及临时数组释放 |
| 本轮计数 | 03 结构数组部分更新、12 从设备偏移重建分段布局、22 中间标量返回主机 | 在实际帧循环中测全量搬运和每轮等待对墙钟的贡献 |
| 源码线索 | 05 矩阵 Field 读回、07 Field 导出 GPU 张量、11 Vulkan 暂存传输、13 GPU 排序转主机 | 先记录生产输入实际命中的分支和传输字节 |
| 源码线索 | 30 Graph 控制流读谓词、50 矩阵自由迭代求解、55 GGUI 索引与变换 | 分别记录观察次数／迭代、同步次数／迭代、传输字节／帧 |

## 覆盖与标注口径

- 扫描 Git 管理的 1,716 个实现文件，含 Python 前端和标准库 478 个、C++ 核心／设备实现及 shader 等 1,182 个、C API 源码和头文件 56 个。按同步、主机 I/O、分配、复制、提交、锁、编译查找、集合原语、原子、算术寻址、诊断 11 类文本特征建立定位索引，661 个文件有匹配。随后沿代表入口人工核对触发条件。扫描排除了 external 等第三方目录、构建产物和未跟踪文件。
- 另对 86 个随库／仓库 Python 示例、114 个 Python 基准文件提取 AST 调用点。下表仅用示例统计，逐名称匹配，未对接收对象做类型推断。“循环内”指词法上位于 for/while 中，间接调用和实际迭代次数另算。
- 优先级按潜在影响及验证价值排列。**优先**适合首批复测，**次轮**等待目标场景，**观察**保留作对照，**跟踪**用于回归。**计数**证明路径实际发生，其墙钟收益尚待测量。**源码**只说明分支和触发条件。普通调用用生产状态。诊断、显式捕获、回退、初始化和稳态各自记账。
- GPU 性能主测使用 RTX 5090，在同一张卡上分别测 CUDA 和 Vulkan。Radeon 核显用于跨厂商兼容性抽查或复现该设备特有的问题。CPU 使用 Ryzen 9 9950X，默认编译档位为 fast。

### 示例中的调用密度

| 名称 | 出现位置 | 宿主循环内 | DSL 循环内 | 解读 |
|---|---|---|---|---|
| `to_numpy` | 75 | 43 | 0 | 逐帧输出和求解中间读回值得分开看 |
| `from_numpy` | 43 | 2 | 0 | 较多出现在初始化或循环外 |
| `fill` | 52 | 18 | 0 | 兼有主机批量操作和 DSL 内的局部对象操作 |
| `copy_from` | 5 | 2 | 0 | 出现位置少，一条固定链也可能反复调用 |
| `norm` | 70 | 8 | 43 | 粒子、邻域及几何计算中的每元素工作 |
| `dot` | 54 | 0 | 27 | 归约和小向量乘法的共同入口名 |
| `random` | 81 | 4 | 57 | 按粒子或样本反复推进 RNG 状态 |
| `svd` | 5 | 0 | 4 | 调用处少，一个循环可以覆盖所有粒子 |
| `atomic_add` | 12 | 0 | 12 | 只计显式方法名，未计可能降为原子的 += |

### 可核对的运行时证据

排查基线上的四轮独立进程显示，对 1,048,576 个 f64，公共填充／实验目标端填充并逐次同步，Vulkan 为 6.962／0.162 ms、CUDA 为 4.950／0.029 ms、CPU 为 1.396／0.213 ms。Vulkan 大数组首次调用时，实验 JIT 路径约 55.9 ms，原接口约 8.8 ms。Vulkan 三次 4 MiB 的公开 i32 拷贝／预绑定 Graph 为 490／54 µs。同一 f32 原生路径仅推迟等待的诊断对照为 480／64 µs。

本轮另在 RTX 5090 上用两个独立 Vulkan 进程运行相同探针。关闭 profiler 和离线缓存，输入、编译预热和最终结果校验均在计数区间之外。两轮计数一致。H2D／D2H 是 Program 记录的逻辑传输量，Program 同步计数只覆盖这一层。

| 一次观察 | H2D | D2H | Program 同步 | 输入及校验 |
|---|---|---|---|---|
| 结构数组更新两个成员 | 128 KiB | 128 KiB | 1 | 8,192 条记录，每条四个 i32，只有两列变化。核对全部四列 |
| 结构数组读取两个成员 | 0 | 128 KiB | 1 | 返回两列共 64 KiB，逐列核对 |
| 从设备 offsets 构造分段布局 | 65,540 B | 32,772 B | 3 | 8,193 个偏移，核对 8,192 个元素和分段 |
| 八次 kernel 返回 Python 标量 | — | — | 8 | 核对八个返回值。设备端写入对照在同一区间为 0 次同步，核对末值 |

### 01、02 修复后的公开接口复测

本地 Forge 基线为 `c50aeb835`。设备拷贝修复为 `0e95555f3`，64 位填充修复为 `f074569f2`。两版均走公开 Python 接口和默认 `fast`。最初的 64 个进程覆盖两个数组大小和四种后端与设备组合。随后在 RTX 5090 上用 16 个新进程补测 1 MiB，每个进程均核对实际设备名及构建哈希。两轮实验均做四组正反次序配对，关闭离线缓存与 profiler。Vulkan 每个进程使用独立管线缓存路径，CUDA 禁用驱动磁盘缓存。CPU 使用 16 个工作线程。

RTX 5090 上的 1 MiB 填充使用 131,072 个 f64。下表单位为 µs，首次与稳态填充都逐次同步。Vulkan 拷贝链使用 262,144 个 f32，每次拷贝也是 1 MiB。

| RTX 5090 公开操作 | 修复前 | 修复后 | 修复后更快的配对 |
|---|---:|---:|---:|
| CUDA 首次填充 | 903.75 | 116.95 | 4/4 |
| CUDA 稳态填充 | 796.70 | 20.44 | 4/4 |
| Vulkan 首次填充 | 3,830.05 | 969.75 | 4/4 |
| Vulkan 稳态填充 | 1,162.54 | 175.43 | 4/4 |
| Vulkan 连续三次设备拷贝，含链末同步 | 516.75 | 175.24 | 4/4 |

下表测 1,048,576 个 f64，即 8 MiB，单位为 ms。首次从已创建并清零的数组开始，计入第一次非零填充及同步。稳态逐次填充后同步，取每个进程五组的中位数，再取四个进程的中位数。最后一列另从调用 `ti.init` 开始计时，包含数组创建、清零和首次非零填充。

| 后端与设备 | 首次填充，旧 → 新 | 稳态填充，旧 → 新 | 初始化到首次填完，旧 → 新 |
|---|---:|---:|---:|
| CUDA，RTX 5090 | 5.842 → 0.109 | 5.627 → 0.023 | 379.34 → 375.75 |
| Vulkan，RTX 5090 | 8.821 → 1.005 | 7.030 → 0.200 | 639.77 → 634.42 |
| CPU，Ryzen 9 9950X | 1.669 → 0.377 | 1.444 → 0.097 | 138.09 → 139.04 |
| Vulkan，Radeon 核显 | 5.695 → 1.110 | 3.292 → 0.681 | 205.49 → 201.86 |

8 MiB 的 f64、i64、u64 稳态填充在四种组合中均为四轮全胜。每次填充的 Program 逻辑主机上传从 8 MiB 降为 0。Vulkan 对两个 32 位字相同的模式直接执行缓冲区填充，其余模式最多在命令中内联 1 KiB 种子，然后在目标缓冲区内复制。CUDA 使用运行时位码中的填充入口，保留调用返回时的完成语义。回归检查逐位比较负零、NaN payload 和高位 u64，并覆盖向量、矩阵及多个复制轮次的尾部。

4,096 个 f64 的 Radeon 稳态耗时为 187.55 → 186.44 µs，小数组的主要收益出现在首次填充，2.188 → 0.490 ms。同尺寸 i64 稳态为 197.18 → 171.99 µs。

Vulkan 连续三次拷贝，每次 4 MiB，对整条链做一次末尾同步。RTX 5090 的墙钟为 598.92 → 202.49 µs，Radeon 为 1,137.49 → 882.56 µs。两版每条链的逻辑 D2D 都是 12 MiB，共有末尾同步之外的 Program 同步为 3 → 0。临时数组、拷贝后立即写单元素、AD 以及独立队列的生产者和消费者另有回归测试。普通主机读回仍会等待数据就绪。

最终构建上，ndarray、运行时完成通知及传输计数的相关测试在 CPU／CUDA／RTX Vulkan 分别通过 19／19／25 项。另在 Radeon 上通过 6 项针对性测试，3 个依赖 `data64` 扩展的普通内核读测试由设备能力门槛跳过。独立 CUDA／Vulkan 队列填充和 Vulkan 完成通知另有 3 项探针通过。

## 操作清单

每行给出调用何时会变密、当前指向的开销和最小验证方式。已修复项目的链接指向修复后的实现，其余条目使用排查基线行号。

### 内存、数组与跨设备传输

| 编号 | 优先级／证据 | 操作与触发条件 | 可疑开销或已有结果 | 验证方式 | 源码 |
|---|---|---|---|---|---|
| 01 | 已修复／耗时 | **64 位 ndarray 的标量填充**<br>CPU／CUDA／Vulkan，标量及向量、矩阵的统一填值 | 基线每次在主机构造整份数组。新路径仅转换一个标量的位型。CPU 直接写入，CUDA 使用运行时内核，Vulkan 使用传输命令。 | 回归覆盖负零、NaN payload、高位整数、复制边界和设备读回。持续跟踪小数组与首次调用。 | [_ndarray.py](../../python/taichi_forge/lang/_ndarray.py#L436) · [Vulkan 64 位填充](../../taichi/rhi/vulkan/vulkan_device.cpp#L1896) |
| 02 | 已修复／耗时 | **连续 ndarray 设备拷贝**<br>Vulkan，普通数组在默认队列上连续 copy_from | 三次同队列拷贝的内部 Program 同步由 3 次降至 0 次，拷贝的逻辑字节数相同。运行时把两端数组保留到完成边界，单元素主机写入也按提交次序执行。 | 回归覆盖临时数组销毁、紧接的主机写入、AD 和独立队列生产／消费。借用运行时保留同步。 | [_ndarray.py](../../python/taichi_forge/lang/_ndarray.py#L582) · [队列与生命周期测试](../../tests/python/test_ndarray.py#L743) |
| 03 | 优先／计数 | **结构数组的部分成员批量读写**<br>StructNdarray，to_numpy_fields / from_numpy_fields 的多成员路径 | 8,192 条 16 B 记录只改两个 i32，有效更新 64 KiB，实测先读 128 KiB 再写 128 KiB。仅读两列也读回 128 KiB。 | 固定有效字节，改变记录宽度、成员数和布局。核对其他成员逐字节保留。 | [_ndarray.py:1089](../../python/taichi_forge/lang/_ndarray.py#L1089) · [_ndarray.py:1109](../../python/taichi_forge/lang/_ndarray.py#L1109) |
| 04 | 次轮／源码 | **结构数组的单成员视图**<br>单列读写，原生路径不适用或退回时 | 标量成员有原生 transform 和可复用临时区。回退时从整个 base 取列，写入再传回整个 base。 | 分开报告原生命中与回退。测阈值两侧、向量分量和非连续输入。 | [_ndarray.py:802](../../python/taichi_forge/lang/_ndarray.py#L802) · [_ndarray.py:833](../../python/taichi_forge/lang/_ndarray.py#L833) |
| 05 | 优先／源码 | **向量／矩阵 Field 批量读回**<br>MatrixField.to_numpy，packed 整体路径未命中 | 逐分量分配 NumPy 临时数组，再分别尝试标量读取。后续分量失败还会进入通用 kernel。 | 同一 3×3 字段比较 packed 与分量布局，记录 D2H 次数、同步、分配和逐元素结果。 | [matrix.py:1493](../../python/taichi_forge/lang/matrix.py#L1493) · [matrix.py:1515](../../python/taichi_forge/lang/matrix.py#L1515) |
| 06 | 次轮／源码 | **结构 Field 的 fill / copy_from**<br>每步清理或复制含多个成员的 StructField | Python 按成员逐个调用，实际派发与同步取决于各成员路径。 | 改变成员数，分开计数原生和回退。核对 AD、稀疏激活和嵌套成员。 | [struct.py:468](../../python/taichi_forge/lang/struct.py#L468) |
| 07 | 优先／源码 | **稠密 Field 导出 Torch／Paddle GPU 张量**<br>请求设备张量且原生 dense-to-host 成功 | 标量 Field 先分配 NumPy 缓冲、读回，再用指定 device/place 构造目标张量。同一设备上的消费者也可能经过主机。 | CPU 目标、同 GPU 目标、跨设备目标分别测。记录双向字节并检查输出所有权。 | [field.py:524](../../python/taichi_forge/lang/field.py#L524) · [field.py:544](../../python/taichi_forge/lang/field.py#L544) |
| 08 | 次轮／源码 | **Fortran 连续 NumPy 作为 kernel 参数**<br>外部数组 ABI 中，F 连续但非 C 连续 | 构造 C 连续副本。回调再转为 Fortran 布局并拷回。只读调用也值得确认是否触发回写。 | 相同数据、形状和 kernel 比较 C/F 两种布局，分别覆盖只读和原地写。 | [kernel_impl.py:2044](../../python/taichi_forge/lang/kernel_impl.py#L2044) |
| 09 | 次轮／源码 | **跨设备 Torch kernel 参数**<br>外部视图未接管，Torch CUDA 张量进入非 CUDA 的兼容分支 | 显式复制到 CPU，随后回写原张量。带 grad 时另处理梯度。显式 DLPack 导入有独立的零拷贝能力约束。 | 区分实际命中的 adapter、主机兼容路径、跨设备搬运。计数不能混合。 | [kernel_impl.py:2128](../../python/taichi_forge/lang/kernel_impl.py#L2128) · [_dlpack.py:134](../../python/taichi_forge/interop/_dlpack.py#L134) |
| 10 | 次轮／源码 | **主机循环逐个读写 ndarray 元素**<br>Python 的 a[i]，标量及向量分量访问 | 访问器有缓存，实际读写依然按元素调用原生 read/write。 | 相同可见结果比较逐元素、批量读回、设备端消费。记录 CPU 循环成本与 GPU 等待。 | [_ndarray.py:750](../../python/taichi_forge/lang/_ndarray.py#L750) · [_ndarray.py:1181](../../python/taichi_forge/lang/_ndarray.py#L1181) |
| 11 | 优先／源码 | **Vulkan 普通主机上传／读回**<br>RHI upload_data / readback_data，每批多个缓冲或频繁小传输 | 调用中为每个对象申请 staging，map、memcpy 并同步提交。一批可以共用提交。上层某些 dense 入口还先做 Program 同步。 | 区分请求分配与驱动实际分配。分测同批多对象和多次单对象，验证主机可见时刻。 | [vulkan_device.cpp:3411](../../taichi/rhi/vulkan/vulkan_device.cpp#L3411) · [vulkan_device.cpp:3457](../../taichi/rhi/vulkan/vulkan_device.cpp#L3457) · [field.py:607](../../python/taichi_forge/lang/field.py#L607) |

### 集合原语、归约与稀疏操作

| 编号 | 优先级／证据 | 操作与触发条件 | 可疑开销或已有结果 | 验证方式 | 源码 |
|---|---|---|---|---|---|
| 12 | 优先／计数 | **用设备 offsets 重建 SegmentedLayout**<br>每帧或每次迭代调用 from_offsets，而 offsets 已在 GPU | 8,193 个 i32 偏移实测 D2H 32,772 B、H2D 65,540 B、3 次 Program 同步。主机校验并展开 segment IDs。 | 拓扑不变时复用布局，对照每步重建。变化拓扑另测，保留空段与非法偏移检查。 | [_algorithms.py:4860](../../python/taichi_forge/algorithms/_algorithms.py#L4860) · [_algorithms.py:5130](../../python/taichi_forge/algorithms/_algorithms.py#L5130) |
| 13 | 优先／源码 | **GPU 上的 auto 稳定排序回退到主机**<br>原生排序未接管的键／载荷组合。例如特定 tensor 成员载荷的降序分支 | 回退会取回键及可选载荷，执行主机稳定排序再写回。相同 API 的原生路径已经存在。 | 先记录路由，再按升降序、键宽、载荷宽度和 NaN 策略对比，验证稳定性。 | [_algorithms.py:9436](../../python/taichi_forge/algorithms/_algorithms.py#L9436) · [_algorithms.py:8377](../../python/taichi_forge/algorithms/_algorithms.py#L8377) |
| 14 | 次轮／源码 | **兼容排序逐级等待**<br>明确选择 legacy odd-even merge 时 | 两层阶段循环中每次 sort_stage 后调用 sync，阶段数随规模增加。 | 分别记录阶段数、kernel 数和同步数。与同语义支持的 native 路径比较。 | [_algorithms.py:8263](../../python/taichi_forge/algorithms/_algorithms.py#L8263) |
| 15 | 次轮／源码 | **前缀和的 Field 回退与矩阵分量路径**<br>PrefixSumExecutor 未命中 native plan 时 | i32 Field 回退会拷入工作区、多级 scan、uniform-add、再拷出。某些矩阵路径递归扫描各分量。 | 固定 N 和 dtype 对比 ndarray、标量 Field、矩阵 Field，核对首尾值、派发和字节。 | [_algorithms.py:17061](../../python/taichi_forge/algorithms/_algorithms.py#L17061) · [_algorithms.py:17110](../../python/taichi_forge/algorithms/_algorithms.py#L17110) · [_algorithms.py:17213](../../python/taichi_forge/algorithms/_algorithms.py#L17213) |
| 16 | 次轮／源码 | **compact、reduce、histogram、分桶及 gather 的工作区**<br>重复短原语，输入对象、容量或语义键经常变化 | 默认工作区按键缓存，命中更新 LRU。未命中可能分配。固定计划与变动输入的成本应分别统计。 | 比较同一对象、同形新对象、显式复用 workspace。记录命中、临时字节和每次调用时间。 | [_algorithms.py:1643](../../python/taichi_forge/algorithms/_algorithms.py#L1643) · [_algorithms.py:10148](../../python/taichi_forge/algorithms/_algorithms.py#L10148) · [_algorithms.py:12302](../../python/taichi_forge/algorithms/_algorithms.py#L12302) · [_algorithms.py:14243](../../python/taichi_forge/algorithms/_algorithms.py#L14243) |
| 17 | 次轮／源码 | **scatter_add 的成员输入与热点地址**<br>Vulkan 两级 scatter 使用结构标量成员。粒子重复写少数桶 | 成员输入可能先经 transform 装入 scratch。热点冲突与额外全数组遍历是两种成本。 | 同分布分别换成连续数组，再固定布局改变键分布。检查累加误差和越界语义。 | [_algorithms.py:14899](../../python/taichi_forge/algorithms/_algorithms.py#L14899) · [_algorithms.py:15007](../../python/taichi_forge/algorithms/_algorithms.py#L15007) |
| 18 | 次轮／源码 | **标量检查／度量立即转 Python 数值**<br>每步调用结果对象的 to_int / to_float | 设备 ndarray 分支会读回标量。device_scalar 可以继续供设备计算使用。 | 只比较返回值确实不被主机使用的迭代。记录读回数及最终可见结果。 | [_algorithms.py:5668](../../python/taichi_forge/algorithms/_algorithms.py#L5668) · [_algorithms.py:5823](../../python/taichi_forge/algorithms/_algorithms.py#L5823) |
| 19 | 跟踪／已覆盖 | **Vulkan 单标量 f32 原子 min/max**<br>大规模、旧值未使用、可证明写同一标量 | 原生能力优先。合格的软件原子归约已默认合并。RTX 5090 上 Euler 默认 block 完整步进 1.033 → 0.723 ms。 | 用小范围、动态边界、读取旧值、混合特殊值及支持原生指令的设备监控回归。 | [spirv_codegen.cpp:5229](../../taichi/codegen/spirv/spirv_codegen.cpp#L5229) |
| 20 | 次轮／源码 | **其他浮点原子与离散地址归约**<br>GPU 粒子散射、有热点桶、未命中原生或上项规则 | 软件浮点原子进入逐地址 CAS 路径。运算种类、目标地址与旧值用途会改变可用优化。 | 同时测单桶、多热点、均匀和地址唯一。分别统计重试、有效吞吐及返回旧值。 | [spirv_ir_builder.cpp:2249](../../taichi/codegen/spirv/spirv_ir_builder.cpp#L2249) · [spirv_ir_builder.cpp:2306](../../taichi/codegen/spirv/spirv_ir_builder.cpp#L2306) |
| 21 | 次轮／源码 | **稀疏 SNode 活跃表生成和递归清理**<br>每步激活变化或调用 deactivate_all | LLVM listgen 有拓扑未变时的复用判断。部分两层清理有合并路径，其他树形仍递归派发。 | 固定／变化拓扑、不同活跃率和树深分别测。计数扫描、产出与清理 dispatch。 | [runtime.cpp:2637](../../taichi/runtime/llvm/runtime_module/runtime.cpp#L2637) · [snode.py:483](../../python/taichi_forge/lang/snode.py#L483) · [spirv_codegen.cpp:6875](../../taichi/codegen/spirv/spirv_codegen.cpp#L6875) |

### 调用、调度、Graph 与设备资源

| 编号 | 优先级／证据 | 操作与触发条件 | 可疑开销或已有结果 | 验证方式 | 源码 |
|---|---|---|---|---|---|
| 22 | 优先／计数 | **小 kernel 把中间标量返回 Python**<br>调用声明返回标量，下一步本可在设备继续消费 | 8 次预热后调用实测 8 次 Program 同步。等量设备端写为 0 次，最终读回一致。真正的主机决策仍需完成等待。 | 用实际迭代链比较逐次取值和最终取值。核对提前退出、异常与迭代数。 | [kernel_impl.py:2417](../../python/taichi_forge/lang/kernel_impl.py#L2417) |
| 23 | 次轮／源码 | **Python kernel 实参映射与普通派发**<br>小 kernel 高频、动态数组或 template 参数较多时 | 静态参数有快速返回。动态特征需提取并查映射，弱引用清理只在 miss。另有 ordinary launch plan。 | 按参数数、纯标量／动态数组、命中／新特化拆测 CPU 提交时间。 | [kernel_impl.py:1183](../../python/taichi_forge/lang/kernel_impl.py#L1183) · [kernel_impl.py:2620](../../python/taichi_forge/lang/kernel_impl.py#L2620) |
| 24 | 次轮／源码 | **批量设备资源的租约与提交锁**<br>多个线程连续提交含多个 ndarray 或别名的调用 | acquire_ndarray_leases 按视图查注册表，去重并复用在途租约。检查保护退休与并发使用。 | 单／多提交者、相同／不同资源测锁等待与尾延迟，保留销毁、reset 和别名测试。 | [program.cpp:5422](../../taichi/program/program.cpp#L5422) |
| 25 | 次轮／源码 | **CUDA 普通 kernel 的参数与回传**<br>未保留普通 launch 缓冲，或参数走需要回传的 transfers 分支 | 参数／结果缓冲可能逐次释放。外部数组回传分支会等待 stream 并逐项复制。只请求返回结果时才取回结果缓冲。 | 同一 kernel 比较自有 ndarray 和外部数组，分别记录 alloc/free、H2D/D2H 和提交时间。 | [kernel_launcher.cpp:1969](../../taichi/runtime/cuda/kernel_launcher.cpp#L1969) · [kernel_launcher.cpp:1982](../../taichi/runtime/cuda/kernel_launcher.cpp#L1982) |
| 26 | 次轮／源码 | **Vulkan 重复描述符绑定**<br>资源集变动或描述符缓存未命中 | 普通与 replay 专用资源集走不同分支。缓存可用时会尝试复用。 | 比较固定／轮换资源与变化布局，分开计数新建、更新、命中和主机时间。 | [vulkan_device.cpp:1294](../../taichi/rhi/vulkan/vulkan_device.cpp#L1294) · [vulkan_device.cpp:1470](../../taichi/rhi/vulkan/vulkan_device.cpp#L1470) |
| 27 | 次轮／源码 | **Vulkan buffer barrier 范围**<br>小算子间反复切换传输／计算 | 通用屏障的源和目标同时覆盖 transfer 与 compute 的读写。 | 用具有相同依赖的命令链测 barrier 数和 GPU 时间，验证跨阶段可见性。 | [vulkan_device.cpp:1722](../../taichi/rhi/vulkan/vulkan_device.cpp#L1722) |
| 28 | 观察／源码 | **Vulkan 高在途提交背压**<br>生产者持续比 GPU 快，在途命令缓冲达到上限 | 等待最老的 fence 以回收命令缓冲及所持资源。 | 扫队列深度，同时记录主机等待、设备利用率和内存高水位。 | [vulkan_device.cpp:4027](../../taichi/rhi/vulkan/vulkan_device.cpp#L4027) |
| 29 | 次轮／源码 | **Graph 重复传 dict 与复用绑定**<br>每帧运行相同图，资源身份稳定 | 字典入口逐项快照以隔离调用方修改。GraphBindingSet 可读取已有版本，变更另行验证。 | 相同命令、相同参数比较 dict、绑定复用、局部更新。分离提交时间和设备时间。 | [_graph.py:19708](../../python/taichi_forge/graph/_graph.py#L19708) |
| 30 | 优先／源码 | **Graph 控制流中的主机谓词观察**<br>进入 portable／观察分支，循环短且频繁检查停止条件 | 控制标量会分配小 host 数组并批量读取。合格的嵌套原生路径可只在末端等待。 | 用实际 lowering 区分两条生产路径。记录每轮读回、迭代次数和终端等待，另测 trace。 | [_graph.py:5963](../../python/taichi_forge/graph/_graph.py#L5963) · [_graph.py:8031](../../python/taichi_forge/graph/_graph.py#L8031) · [_graph.py:20128](../../python/taichi_forge/graph/_graph.py#L20128) |
| 31 | 次轮／源码 | **CPU 小范围并行和线程池唤醒**<br>短 range_for 或多个小任务，线程数相对工作量大 | range_for 分块送入线程池。池已有线程数选择与嵌套路径，调度占比要结合具体工作量。 | 固定输入扫 1/4/8/16 线程、范围和每元素工作量。记录每步、提交耗时和尾延迟。 | [runtime.cpp:2867](../../taichi/runtime/llvm/runtime_module/runtime.cpp#L2867) · [threading.cpp:83](../../taichi/system/threading.cpp#L83) |
| 32 | 观察／源码 | **C API／AOT 重复绑定与执行**<br>原生宿主以固定参数布局高频调用 kernel | 普通 C API 每次进入参数绑定。另有 prepared execution，可复用已准备的执行结构。 | 同一 AOT 包比较普通和 prepared，拆开准备、换资源、热调用成本及错误契约。 | [taichi_core_impl.cpp:1458](../../c_api/src/taichi_core_impl.cpp#L1458) · [prepared_execution.cpp:3264](../../c_api/src/prepared_execution.cpp#L3264) |

### 编译、设备算术与每元素操作

| 编号 | 优先级／证据 | 操作与触发条件 | 可疑开销或已有结果 | 验证方式 | 源码 |
|---|---|---|---|---|---|
| 33 | 次轮／源码 | **离线缓存查找与并发编译**<br>新特化、冷启动或多个 worker 遇到相同键 | 同键编译会等待。磁盘产物读取与实际编译已移到锁外。 | 分别测内存命中、磁盘命中、同键并发 miss、不同键 miss。记录首次可用时间。 | [kernel_compilation_manager.cpp:167](../../taichi/compilation_manager/kernel_compilation_manager.cpp#L167) |
| 34 | 次轮／源码 | **静态循环展开以及通用 IR 简化**<br>大静态循环、矩阵展开、频繁生成新特化 | 前端展开会增加 IR。简化有按档位控制的迭代，两者可能互相放大或抵消编译成本。 | 记录展开前后语句数、各 pass 时间、首次执行与稳态，分别测 fast 和 full。 | [ast_transformer.py:1450](../../python/taichi_forge/lang/ast/ast_transformer.py#L1450) · [simplify.cpp:725](../../taichi/transforms/simplify.cpp#L725) |
| 35 | 观察／已有反证 | **平方、立方和小整数幂**<br>Vulkan 的常量幂可能生成额外循环。CUDA N-body 内层立方 | Euler 乘法改写使 calc_dt 的循环数从 6 降到 2，但 0.7251 → 0.7316 ms/步，且 120 步 Q 超差。CUDA N-body 改写亦未显示实用收益。 | 若出现新热点，按 kernel 做消融并核对完整轨迹、符号零、非有限数。 | [demote_operations.cpp:34](../../taichi/transforms/demote_operations.cpp#L34) |
| 36 | 观察／已有反证 | **整除、取模和周期索引**<br>有符号坐标。除数或网格尺寸为 2 的幂 | RTX 上模 32／掩码和整除／移位微测相近。LBM 64³ 边界判断改写后 0.3536 → 0.3559 ms/步。 | 保留正负坐标、非 2 次幂和部分尾块。有整步收益再提升优先级。 | [demote_operations.cpp:145](../../taichi/transforms/demote_operations.cpp#L145) |
| 37 | 观察／已有反证 | **平方根倒数、范数、点积与归一化**<br>粒子邻域与渲染内层，每元素多次调用 | 已测的六次依赖倒数平方根写法没有跨后端稳定优势。normalized 的 eps 在开方外，norm_inv 的 eps 在开方内。 | 若测重复范数或点积，固定精度与 eps 语义，比较寄存器和应用误差。 | [matrix_ops.py:259](../../python/taichi_forge/lang/matrix_ops.py#L259) · [matrix_ops.py:277](../../python/taichi_forge/lang/matrix_ops.py#L277) · [matrix_ops.py:368](../../python/taichi_forge/lang/matrix_ops.py#L368) |
| 38 | 次轮／源码 | **小矩阵 SVD 与极分解**<br>弹塑性粒子／网格点内反复分解 | 3×3 SVD 的默认迭代数为 f32 五次、f64 八次，且产生多项分解结果。调用频率和实际使用的结果决定成本。 | 测仅用部分结果与完整结果。比较寄存器、重构误差、退化矩阵和 AD。 | [_funcs.py:176](../../python/taichi_forge/_funcs.py#L176) · [_funcs.py:527](../../python/taichi_forge/_funcs.py#L527) |
| 39 | 观察／源码 | **随机数状态更新**<br>着色器内每个样本多次 random | SPIR-V 每次抽样推进多字状态，初始化辅助函数按需生成。最终寄存器或访存开销待查。 | 固定每像素样本数，看最终设备指令、状态存放与帧时。校验分布及重现性。 | [spirv_ir_builder.cpp:2378](../../taichi/codegen/spirv/spirv_ir_builder.cpp#L2378) |

### 设备内寻址、算力与临时存储

| 编号 | 优先级／证据 | 操作与触发条件 | 可疑开销或已有结果 | 验证方式 | 源码 |
|---|---|---|---|---|---|
| 40 | 次轮／源码 | **小矩阵乘、矩阵向量乘**<br>每粒子、每单元都执行固定维数的 matmul | DSL 用静态嵌套循环展开。大一档的小矩阵可能同时增加编译 IR、寄存器需求和设备指令。 | 2×2／3×3／6×6 同时测冷 JIT、寄存器溢出和整步时间。保留运算次序与梯度。 | [matrix_ops.py:333](../../python/taichi_forge/lang/matrix_ops.py#L333) |
| 41 | 次轮／源码 | **稠密／稀疏寻址与动态矩阵下标**<br>每个 stencil 邻居或每个向量分量重复定位 | SPIR-V 有独立的 MatrixPtr 和 SNodeLookup lowering。稠密与需激活的稀疏节点生成不同工作。 | 固定算术量比较连续／间接索引、常量／动态分量。看最终地址计算、访存合并和激活次数。 | [spirv_codegen.cpp:795](../../taichi/codegen/spirv/spirv_codegen.cpp#L795) · [spirv_codegen.cpp:3283](../../taichi/codegen/spirv/spirv_codegen.cpp#L3283) |
| 42 | 次轮／源码 | **位打包成员的原子更新**<br>相邻量化值共享一个存储字 | 位指针走专门的原子更新路径。不同逻辑地址也可能争同一物理字。 | 同字不同位、不同字、相同位分别施压。核对相邻位保留、溢出和旧值。 | [spirv_codegen.cpp:5282](../../taichi/codegen/spirv/spirv_codegen.cpp#L5282) · [spirv_codegen.cpp:7285](../../taichi/codegen/spirv/spirv_codegen.cpp#L7285) |
| 43 | 次轮／源码 | **CUDA 临时数组的申请与清零**<br>每步创建／释放许多短命数组，或改变容量 | 运行时分配有异步池与缓存分支，请求 zero_fill 时会清零。反复创建还会产生注册记录。 | 同形复用、同形重建、交替大小分别测真实驱动分配、清零字节与显存高水位。 | [_ndarray.py:681](../../python/taichi_forge/lang/_ndarray.py#L681) · [cuda_device.cpp:232](../../taichi/rhi/cuda/cuda_device.cpp#L232) |
| 44 | 观察／源码 | **GPU range 的 block 选择与尾部利用率**<br>短循环、动态边界或寄存器压力大的 kernel | 编译入口区分默认和用户显式 block。最终几何还受后端策略影响。 | 读实际 task manifest，固定同一 kernel 对比自动与局部 block，观察占用和整步，保留显式覆盖。 | [offload.cpp:217](../../taichi/transforms/offload.cpp#L217) |
| 45 | 观察／源码 | **CUDA 批量小 GEMM**<br>共享 Program 的主机线程重复提交小矩阵计算 | 普通 cuBLAS 路径有 provider 锁和句柄复用，也有预准备的原生操作。 | 固定尺寸对照普通／准备后执行，单／多提交线程分开。记录锁等待和设备耗时。 | [cuda_blas.cpp:451](../../taichi/program/cuda_blas.cpp#L451) · [cuda_blas.cpp:583](../../taichi/program/cuda_blas.cpp#L583) |
| 46 | 观察／源码 | **纹理采样、texel fetch 与图像 load/store**<br>渲染／体积采样中每光线多次取样 | 代码生成针对 LOD、梯度、比较和读写发出不同图像指令。局部性和过滤语义比单纯的调用处数量更关键。 | 同格式、相同过滤和访问图案测缓存命中与帧时。主机截图单列。 | [spirv_codegen.cpp:4716](../../taichi/codegen/spirv/spirv_codegen.cpp#L4716) |
| 47 | 观察／源码 | **逐元素加减乘除、比较、cast 和 select**<br>几乎所有热 kernel，尤其是连续短算子串 | SPIR-V 对常见二元操作有直接 lowering。大量出现本身不足以说明瓶颈。临时数组、隐式转换和重复派发值得对照。 | 以相同读写字节的 copy 为基线，分别测单次和长依赖链，核对最终指令及数值语义。 | [spirv_codegen.cpp:4459](../../taichi/codegen/spirv/spirv_codegen.cpp#L4459) · [spirv_codegen.cpp:4619](../../taichi/codegen/spirv/spirv_codegen.cpp#L4619) |
| 48 | 次轮／源码 | **连续 transform／reduce 的中间结果**<br>一帧内组合多条短原语、反复遍历同一数组 | PrimitiveSequence 有适用条件和 fused plan。能否消除中间写入取决于实际命中路径。 | 固定数据和布局，比较逐次调用与序列，记录实际派发、临时字节及尾部结果。 | [_algorithms.py:2401](../../python/taichi_forge/algorithms/_algorithms.py#L2401) · [_algorithms.py:2427](../../python/taichi_forge/algorithms/_algorithms.py#L2427) |
| 49 | 观察／源码 | **邻域 stencil 的块级缓存及同步**<br>声明 block_local 的场，邻域有重用但工作组资源有限 | 块级缓存变换根据访问标记处理相应场。额外暂存和屏障可能抵消少读的全局数据。 | 按邻域半径、工作组和边界比例测全局读、共享内存、占用率和整步。 | [make_block_local.cpp:24](../../taichi/transforms/make_block_local.cpp#L24) · [spirv_codegen.cpp:5099](../../taichi/codegen/spirv/spirv_codegen.cpp#L5099) |

### 求解器、自动微分与其他高频消费者

| 编号 | 优先级／证据 | 操作与触发条件 | 可疑开销或已有结果 | 验证方式 | 源码 |
|---|---|---|---|---|---|
| 50 | 优先／源码 | **MatrixFreeCG／BiCGSTAB 的迭代标量**<br>GPU 多轮求解、每轮向量工作较小 | CG 每轮至少有两个返回主机的点积，主机再写 alpha 等系数并判断收敛。quiet 模式仍有数据依赖。 | 用相同矩阵与停止准则比较设备控制或分批检查。记录同步/轮、迭代数和残差轨迹。 | [matrixfree_cg.py:119](../../python/taichi_forge/linalg/matrixfree_cg.py#L119) · [matrixfree_cg.py:153](../../python/taichi_forge/linalg/matrixfree_cg.py#L153) · [matrixfree_cg.py:313](../../python/taichi_forge/linalg/matrixfree_cg.py#L313) |
| 51 | 次轮／源码 | **稀疏矩阵乘向量和解方程的载体转换**<br>Field／NumPy RHS 与 ndarray 入口 | 部分 Field 分支调用 to_numpy。特定求解路径再上传 RHS、下载解。可保持设备 ndarray 的分支另计。 | 同一矩阵分开统计分析、分解、重复 solve。只比较相同的返回位置与精度。 | [sparse_matrix.py:495](../../python/taichi_forge/linalg/sparse_matrix.py#L495) · [sparse_solver.py:615](../../python/taichi_forge/linalg/sparse_solver.py#L615) |
| 52 | 次轮／源码 | **Tape 入口清梯度**<br>每步重复创建 Tape、已分配梯度字段很多 | 满足 clear_gradients 条件时清理梯度，Vulkan 已有兼容稠密字段的批量清理。 | 分别改变活跃梯度与全部已分配字段数量，测清理字节／派发及总步进，校验反向结果。 | [_ad.py:242](../../python/taichi_forge/ad/_ad.py#L242) · [_ad.py:350](../../python/taichi_forge/ad/_ad.py#L350) · [_ad.py:466](../../python/taichi_forge/ad/_ad.py#L466) |
| 53 | 观察／源码 | **FFT 计划缓存**<br>尺寸、布局或执行描述反复变化 | CUDA 侧按 descriptor 建键并区分 cuFFT 计划缓存命中与未命中。 | 固定形状和形状轮换分开测，记录 miss、创建耗时及第 N 次执行时间。 | [cuda_fft.cpp:854](../../taichi/program/cuda_fft.cpp#L854) |
| 54 | 观察／源码 | **Graph 配方物化**<br>运行期间反复请求物理执行方案 | 物化、发布和执行已有不同入口，同定义可有不同配方。 | 分别记录首次物化、重复获取和实际回放时间，核对物理产物身份。 | [materialize.py:723](../../python/taichi_forge/graph/_recipes/materialize.py#L723) |
| 55 | 优先／源码 | **GGUI 网格索引与实例变换**<br>Scene 的兼容绘制入口每帧接收设备 Field | 准备索引和变换时可调用 to_numpy。部分 v2 路径还新建主机 VBO。静态几何重复绘制尤其值得检查。 | 每帧记录顶点、索引和变换各自搬运字节。固定／变动拓扑与实际渲染耗时分开。 | [staging_buffer.py:50](../../python/taichi_forge/ui/staging_buffer.py#L50) · [staging_buffer.py:69](../../python/taichi_forge/ui/staging_buffer.py#L69) · [scene.py:185](../../python/taichi_forge/ui/scene.py#L185) · [scene.py:364](../../python/taichi_forge/ui/scene.py#L364) |
| 56 | 观察／源码 | **帧缓冲导出与诊断读取**<br>显式请求深度 NumPy、调试读取结构数组 | 获取主机深度图会复制并重排。debug_getitem 可读回完整结构数组。只在真正调用时计量。 | 区分正常渲染、用户所需捕获及调试。检查每帧实际调用数。 | [window.py:303](../../python/taichi_forge/ui/window.py#L303) · [_ndarray.py:1147](../../python/taichi_forge/lang/_ndarray.py#L1147) |
| 57 | 观察／源码 | **OptiX 加速结构初建**<br>新建 shared GAS。动态图形应用反复重建时 | 该构建成功路径会等待对应 CUDA stream。可更新结构对 scratch 的持有方式不同。 | 只建一次与每帧重建／更新分开，记录 scratch 峰值、build/update 及等待。 | [provider.cpp:1069](../../taichi/optix/provider/provider.cpp#L1069) |

## 复测入口

- 数组与集合原语可从 [ndarray 基准](../ndarray_primitives.py)、[结构数组基准](../struct_ndarray_primitives.py)、[稠密 Field 传输基准](../dense_field_host_copy_fill_cache_bench.py) 开始。
- 图与求解可从 [Graph 参数绑定](../graph_arg_binding_bench.py)、[稠密 Field 回放](../graph_dense_field_replay_bench.py)、[Krylov 求解](../linear_operator_graph_krylov_bench.py)、[稀疏 SNode 生命周期](../sparse_snode_lifecycle_bench.py) 开始。
- 普通派发与工作组有 [CPU launch 基准](../ordinary_cpu_launch_bench.py) 和 [任务 launch 策略基准](../task_launch_policy_bench.py)。优先记录首次、批次末同步稳态、实际路由、同步次数与逻辑字节。数值改写还需比较完整应用检查点。

本机原始证据位于 `.tmp/common-op-inventory-20261126/`（扫描索引与两轮计数）、`.tmp/common-op-audit-20261126/`（修复前的 87 个诊断进程）、`.tmp/default-vk-atomics/`（Euler 原子基线）和 `.tmp/common-op-fixes-final-20261126/`（公开接口的配对驱动、逐进程结果和构建哈希）。RTX 5090 的 1 MiB 补测汇总在最后一个目录的 `rtx5090-midpoint/summary.json`。

本轮本机复测驱动、各次原始样本、AOT 日志及已撤下的原型补丁位于 `.tmp/op-goal-20261126/`。其中 `bench.py` 运行操作矩阵，`apps.py` 运行应用快照，`scores.py` 汇总进程中位数。`local_raw` 给出每次计分所用的原始目录名。
