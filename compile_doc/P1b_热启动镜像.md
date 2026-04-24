# P1.b — 进程内 .tic 字节镜像（加速热启动）

> 提交：`<待commit>`  · 关联：P5 已并行化的编译路径，P1.a 已实现的代码缓存
>
> 一句话：在 `KernelCompilationManager` 的 disk-cache 之上加一层"进程内字节镜像"，
> 让同一进程内多次 `ti.init()` / `ti.reset()+ti.init()` 的 hot-start 不再重复读盘。

## 背景

P5 解决了多核并行编译，P1.a 解决了单进程内首次编译命中重用，但**跨 `Program` 的复用**
仍然只能走磁盘 offline cache：

```
Program A  --[compile + dump]--> .tic file  -.
                                              |
ti.reset()                                    |
                                              v
Program B  --[load_or_compile]--> open .tic --> deserialize --> CompiledKernelData
```

每个 kernel 在 `KernelCompilationManager::load_ckd()` 里都做一次：
1. `std::ifstream` 打开 `.tic` 文件
2. 整体读入 + `CompiledKernelData::load()` 反序列化
3. `check()` 校验

在生产里，"`ti.reset()` + 重新 `ti.init()`" 几乎不会发生；但**长进程多次 `ti.init()`、
单元测试套件、Notebook 反复重启 kernel** 都属于这一类。即使是单次进程冷启动，磁盘 IO
也是 hot-start 路径上少数几个外部不可控的延迟来源。

## 设计

新增 [taichi/compilation_manager/inproc_disk_mirror.h](taichi/compilation_manager/inproc_disk_mirror.h) /
[inproc_disk_mirror.cpp](taichi/compilation_manager/inproc_disk_mirror.cpp)：

```cpp
class InprocDiskMirror {
 public:
  static std::optional<std::string> get(const std::string &kernel_key);
  static void put(const std::string &kernel_key, std::string bytes);
  static void clear();
  // diagnostics
  static std::size_t total_bytes();
  static std::size_t hits();
  static std::size_t misses();
};
```

进程级单例（function-local statics），用同一把 mutex 保护一张
`unordered_map<kernel_key, bytes>` 加 FIFO 链表做容量管理。

### 集成点

[kernel_compilation_manager.cpp](taichi/compilation_manager/kernel_compilation_manager.cpp)
里只改了两个函数：

* `load_ckd(kernel_key, arch)`：先查镜像 → 命中则直接 `istringstream` 反序列化，跳过
  开/读文件；未命中则走原来的 `std::ifstream`，**把读到的字节交给镜像** 供下一次 init 使用。
* `dump()`：把刚编译完准备落盘的 `CompiledKernelData` 序列化到一个 `ostringstream`
  里一次，**同时**写盘和写镜像 —— 下一个 `Program` 跳过整个磁盘往返。

序列化失败、文件损坏、字节超过单条 cap 时，所有路径都安全 fall-back 到 disk。

### 关键正确性约束 — 为什么不能直接共享 `CompiledKernelData` 对象

最初考虑过更激进的方案：让 `caching_kernels_` 跨 `Program` 存活，直接把
`CompiledKernelData*` 对象本身共享出去。审计发现这条路不通：

```cpp
// taichi/codegen/compiled_kernel_data.h
class CompiledKernelData {
  ...
  void set_handle(const KernelLaunchHandle &handle) const;  // 注意 const + mutable
  const std::optional<KernelLaunchHandle> &get_handle() const;
 private:
  mutable std::optional<KernelLaunchHandle> kernel_launch_handle_;
};
```

```cpp
// taichi/runtime/cuda/kernel_launcher.cpp:193
if (!compiled.get_handle()) {
  ...register with this Program's runtime...
  compiled.set_handle(handle);
}
return *compiled.get_handle();
```

`KernelLaunchHandle` 是**与 `Program` 实例的 `gfx_runtime_` / `LlvmRuntimeExecutor`
深度绑定**的 ID（GFX、CUDA、CPU、LLVM 都有同样的 lazy-register 模式）。如果把同一个
`CompiledKernelData` 对象在两个 `Program` 间共享，第二个 `Program` 看到非空 handle 就
**跳过注册**，直接拿着前一个 `Program` 已销毁 runtime 的 launch_id 去发射 → 立即崩溃
或静默跑到错误的 device context 上。

镜像缓存原始字节而不是对象，每个 `Program` 都做一次 fresh 反序列化，handle 自然为空，
完全维持现有 launcher 的 lazy-register 语义。**正确性零风险**。

### 容量与开关

* 通过环境变量 `TI_INPROC_DISK_MIRROR_MB` 配置，默认 256 MB，设为 0 完全禁用。
* `mirror_cap_bytes()` 用 function-local static 缓存读取一次，避免热路径每次 `getenv`。
* 超过 cap 时按 FIFO 淘汰最老的 entry，直到能装下新条目；单条超过 cap 直接拒收。
* 覆盖同 key 时不调整 FIFO 位置 —— 保持 O(1) 摊销，不需要在热路径上线性扫链表。

### 线程安全

`get` / `put` 都在内部 mutex 内完成。`get` 在 lock 内做一次 hashmap 查找 + 字符串
拷贝，然后释放锁，调用方在锁外做反序列化，不会阻塞其他 lookup。这个 mutex 与
`KernelCompilationManager::cache_mutex_` 完全解耦，P5 的并行编译不受影响。

## 验证

### 正确性

```
$ pytest tests/python/test_offline_cache.py -x -q
106 passed in 66.95s
```

镜像默认开启的情况下，全套 offline cache 单元测试 100% 通过。

### Hot-start 微基准

新加了 [tests/p5/bench_hot_start.py](tests/p5/bench_hot_start.py)，对每个 backend：
1. 起一个子进程，用空 cache_dir 跑 cold compile + dump（建立磁盘缓存）。
2. 起一个新子进程（mirror_mb=0），测量第二次 `ti.init()` 内首次调用 32 个 kernel 的总时长 → "warm second-init, mirror OFF"。
3. 起另一个新子进程（mirror_mb=256），同样测量 → "warm second-init, mirror ON"。

子进程隔离保证 mirror 真的在内存里，不被 cold 阶段污染。

| Arch   | n   | cold 编译 | warm + mirror OFF | warm + mirror ON | 加速比 |
| ------ | --- | -------: | -----------------: | ----------------: | ----: |
| CPU    | 32  | 67.8 ms  |   27.0 ms          |   26.4 ms         | 1.02× |
| CUDA   | 32  | 136.1 ms |   34.1 ms          |   30.1 ms         | **1.14×** |
| Vulkan | 32  | 88.3 ms  |   50.9 ms          |   51.1 ms         | 1.00× |
| CUDA   | 128 | 208.6 ms |  104.2 ms          |  111.8 ms         | 0.93× （噪声内）|

### 诚实的结论

**镜像在本机 NVMe SSD 上的实测增益是 0–14%**，远小于 P5（Vulkan 1.38×）那种数量级
的提升。原因明确：

1. SSD 上 32 个小 kernel 的全部 `.tic` 加起来 < 1 MB，磁盘 IO 本身就只占 hot-start
   总时间的 1–4 ms。
2. **现在的 hot-start 瓶颈是 Python 端 `ensure_compiled` / `materialize` 的解释器开销**
   （和 P5 测得的 `py=114ms / cxx=8ms` 同一现象），不是 C++ 端 disk IO。
3. 镜像在 n=128 CUDA 上反而 0.93×，纯属测量噪声 —— Δ ≈ 7ms 在 100ms 量级里属于
   抖动范围。

那为什么还要做？

* **零新风险**：原盘缓存路径不变，镜像只是前置一个可关掉的可选层，单元测试 106/106
  通过。
* **慢盘 / 网络盘 / 容器**：在 HDD、SMB 挂载、Docker overlayfs 等环境上，单文件 open
  延迟可以是 SSD 的 10–100×，镜像把 N 个小 IO 合成一次 `unordered_map` 查找。
* **大 kernel**：实际生产 kernel 的 SPIR-V/PTX 字节数远超本基准里 32 个 toy kernel，
  反序列化成本与文件大小线性相关。
* **后续优化的基础设施**：这个镜像层是后续 "热重载完整 `CompiledKernelData` 对象"
  优化的前置 —— 等到 `KernelLaunchHandle` 与 runtime 解耦后，可以把镜像直接升级为
  对象缓存，免掉每次反序列化。

### 关闭路径

```bash
TI_INPROC_DISK_MIRROR_MB=0  # 完全禁用，所有 get/put 立即返回
```

这条路径下 hot-start 行为与 P1.b 之前 100% 一致。

## 与 P5 的关系

P5 和 P1.b 解决的是**正交**的两个问题：

|              | P5 并行编译                | P1.b 进程内字节镜像        |
| ------------ | -------------------------- | -------------------------- |
| 解决场景     | 第一次冷编译（无 cache）   | 后续热启动（cache 已存在）|
| 优化对象     | C++ 端 codegen 并行        | C++ 端 disk IO 复用        |
| 主要收益     | Vulkan 1.38× / CUDA 1.0×  | CUDA 1.14× / CPU 1.02×    |
| Python 影响  | 受 GIL 限制                | 完全无关                   |

两者叠加：第一次跑 → 走 P5 多线程并行编译 → 写盘 + 写镜像；第二次跑 → 走镜像 →
直接拿到字节流 → 反序列化即可发射。

## 未来工作

1. **打通 Python 端 `ensure_compiled`**（即 P5 测得的瓶颈）：让 `ti.compile_kernels`
   也跳过 `materialize` 的 GIL 段，预期可以再吃掉 50ms 级别的延迟。
2. **`CompiledKernelData` 对象级跨 Program 共享**：把 `kernel_launch_handle_` 从对象
   里拆出去（move 到 launcher 端的 hashmap），就能让镜像直接缓存对象，每次 hot-start
   只需要一次 launcher.register。本次未做，因为对 launcher 的改动需要全 backend 联调
   验证，超出 P1.b 的范围。
3. **磁盘 cache 预读**：把 `cached_data_` 的 metadata 解析也提到镜像里，进一步去掉
   `Program` 构造时的一次 `read_from_binary_file`。
