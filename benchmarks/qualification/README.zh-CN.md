# 本机单操作资格测试

[English](README.md) | 简体中文

本目录包含经过复核的本机 Taichi 单操作 A/B microbench。入口每次只接受一个
操作、一个 backend 和一个规模，不会同时启动不同 backend 的性能进程；每次
Forge/vanilla 比较均由相邻且不重叠的 fresh-process 对组成。

中英文工作规划有意保存在 Git 已忽略的本地区域：
`temp_outputs/qualification/planning/PLAN.zh-CN.md` 与 `PLAN.en.md`。它们不是发布
源码，不得加入 Git。发布门槛写死在规划中，并由 `single_kernel_microbench.py` 的
`QUALIFICATION_MINIMUMS` 及相关资格常量直接编码。

## 范围

`single_kernel_microbench.py` 提供五个共用 ndarray 控制 kernel，以及一个完全同
公开 API 的 PrefixSum 案例：

| 操作 | 逻辑访存模型 |
|---|---|
| `fill` | 每元素一次 f32 写入 |
| `copy` | 每元素一次 f32 读取和一次 f32 写入 |
| `saxpy` | 每元素两次 f32 读取和一次 f32 写入 |
| `stencil2d` | 每格点五次 f32 读取和一次 f32 写入 |
| `reduce_chunks` | 每元素一次 i32 读取、每 chunk 一次 i32 写入 |
| `prefix_sum` | `ti.algorithms.PrefixSumExecutor(n).run(field)` 的 i32 inclusive scan；逻辑输入/输出各一次 |

这些是控制/回归 microbench，用于测量普通 kernel 路径，能够发现运行时额外成本
或真实的基础路径提升；但它们不覆盖 Graph、native primitive、bounded dispatch、
worklist、LinearOperator 或其他 Forge-only API，结论不得外推到这些能力。

`prefix_sum` 是 `DIRECT-001`：两边运行同一份 workload、dense i32 field、确定性
输入、exact oracle 和同步边界。Forge 必须命中 native dense-field scan plan，
vanilla 必须命中其 legacy field workspace；route 不符合时 child 失败。为了保持
单项开发入口，优先使用 `prefix_sum_microbench.py`，它固定操作且不能变成聚合器。

`mpm_graph` 是 `DIRECT-003`：它直接复用 `benchmarks/graph_mpm_replay_bench.py`
中的二维 MLS-MPM kernel 和 ndarray。small preset 为 4,096 粒子、64² 网格、2 个
substep 和 10 个 dispatch/frame。两边只允许 Graph runtime 内部不同；每个 child
用另一组 ndarray 执行相同 direct kernel 序列，对 x/v/C/J/grid/image 全状态按固定
门槛验证，并保存跨 runtime endpoint fingerprint。`mpm_direct` 是同 workload 的
独立控制项，不能与 Graph 在同一进程混测。对应入口分别为
`graph_mpm_microbench.py` 和 `mpm_direct_control.py`。

## runner 已实现的公平性合同

- Forge 与 vanilla 使用两个依赖完备的独立 venv。子进程删除 `PYTHONPATH` /
  `PYTHONHOME`，禁止 user site，证明 package/core/dependency 均来自所选 venv，
  并要求两边 Python 与中性依赖版本一致。
- 两边各运行一次非计分 pilot，随后冻结两者建议值中较大的共同 batch；所有计分
  进程执行相同 launch 数，计分 batch 还必须达到所要求的计时窗口。
- 进程顺序按固定种子交替 AB/BA。主观测量是 pair-level 的
  `vanilla / Forge` 速度比，绝不池化不同进程的样本。
- 系统级命名 mutex 保证同一时刻只有一个资格 driver，因此独立启动的
  CPU/CUDA/Vulkan benchmark 也不能意外重叠。
- 每个子进程使用相同 CPU 线程数与 affinity，关闭 Taichi 离线缓存，分开记录
  import/init/first-call/warm，使用相同同步边界，在计时前后验证正确性，并在退出前
  显式 sync/reset。
- GPU child 固定 device 0。Forge CUDA runtime UUID 必须与 `nvidia-smi` UUID
  匹配；缺少 runtime UUID 的 runtime 仅在本机只有一个 GPU 且显式 device-zero
  绑定时通过，多 GPU 不明确时 fail closed。
- Forge stability 前后读取 runtime live memory 和 host/device memory pool，
  current/live/raw/cached 状态必须 plateau；vanilla 不提供的 Forge 专有计数器明确
  记为 unavailable。RSS、进程 GPU memory 和 reset 证据继续分别保存。
- pilot 前、每对之前以及每个子进程之后，父进程都会检查其他 Python 进程、CPU
  占用、GPU 竞争进程、GPU 利用率与温度，以及必需监控是否可用。准入失败即停止，
  不把污染数据纳入平均，也不静默自动重试。
- 只有编码的方法学、稳定性、波动、配对效果和双语 artifact 门槛全部通过，资格
  结果才可发布。`diagnostic` 运行无论数字如何都不能形成性能宣称。

## 开发 smoke：一次只测一项

先建立或选择两个完整隔离环境，然后运行一个最小 CPU 诊断：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\single_kernel_microbench.py `
  --operation fill --backend cpu --preset small `
  --intent diagnostic --pairs 1 --samples 5 --warmups 2 `
  --target-sample-ms 20 --stability-replays 0
```

该 smoke 只验证执行与证据生成，不能支持速度宣称。开发单项测试时，不得改成聚合
入口或同时启动多 backend。

CUDA PrefixSum 的首个单项探针使用独立入口：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\prefix_sum_microbench.py `
  --backend cuda --preset small --intent diagnostic `
  --pairs 1 --samples 5 --warmups 2 `
  --target-sample-ms 20 --stability-replays 0
```

Graph MLS-MPM 也必须单独启动；direct 控制使用另一个 run ID 和
`mpm_direct_control.py`：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\graph_mpm_microbench.py `
  --backend cuda --preset small --intent diagnostic `
  --pairs 1 --samples 5 --warmups 2 `
  --target-sample-ms 20 --stability-replays 0
```

某个单项验证稳定后，资格模式会强制执行固定最低门槛：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\single_kernel_microbench.py `
  --operation fill --backend cpu --preset small `
  --intent qualification --pairs 10 --samples 30 --warmups 5 `
  --target-sample-ms 100 --stability-replays 1000
```

artifact 写入 `temp_outputs/qualification/single_kernel/<run-id>/`，包括 manifest、
每个子进程的 JSON 与 stdout/stderr、pair-level JSONL/CSV、原始 batch 样本、环境与
wheel hash、噪声观测、`summary.json`，以及配对的中英文报告和方法学验证。

可使用独立审计器从逐子进程 artifact 重新计算证据：

```powershell
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe `
  benchmarks\qualification\audit_single_kernel_run.py `
  temp_outputs\qualification\single_kernel\<run-id>
```

审计器也会根据 `failure.json` 和成对的中英文失败文件审计准入失败的运行；此时
artifact 完整性可以通过，但性能宣称资格始终为 false。

## 解释边界

逻辑 GB/s 是源码层访存估算，不是显存控制器计数器。First call 包含编译与一次
launch；steady-state 计时包含 Python submission，并在冻结的 batch 外围同步一次。
稳定性内存阈值属于资格 guardrail，不是引擎上限；CUDA context 驻留不能与
Taichi live allocation 混为一谈。

早先全矩阵探索 harness 及其源码快照只保留在
`temp_outputs/qualification/legacy_common_kernel_exploration/`，不属于本资格实现。
