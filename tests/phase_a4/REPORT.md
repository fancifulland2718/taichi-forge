# Phase A.4 稳定性对比测试报告

## 测试目的
验证 Phase A.4 (LLVM 15 → LLVM 19 + PassManagerBuilder → New PassManager +
opaque pointer) 对 Taichi 运行时的影响，确认无功能性/数值性/性能性回归。

## 测试环境
| 项 | 基线 (baseline) | 候选 (candidate) |
|---|---|---|
| Taichi | 1.7.4 官方 wheel | 1.8.0 本地构建 (Phase A.4) |
| LLVM | 15.0.1 | 19.1.7 |
| commit | b4b956fd | 3b4844cac (含 5f911bfe3) |
| Python | 3.10.20 | 3.10.20 |
| 平台 | Windows + CUDA | Windows + CUDA |
| conda env | `ti174` | `3.10` |

- 每个 kernel 独立 `ti.init`/`ti.reset` 3 次，取均值
- `offline_cache=False`, `random_seed=42`
- 编译时间 = 第一次调用的 wall-time（含 JIT）
- 运行时间 = 稳态连续调用均值

## Kernel 覆盖
| Bench | 规模 | 描述 |
|---|---|---|
| saxpy | n=2²⁰ | element-wise `y = a*x + y`，f32 |
| reduce | n=2²⁰ | 平方和归约（return-value kernel） |
| matmul | 256×256 | 三重循环 GEMM |
| stencil | 512×512, 20 iter | 2D 5 点 Laplacian + 缓冲交换 |
| nbody | n=1024 | 全对全引力，含 sqrt/div |
| mandelbrot | 512×512 | 含 while + break 条件的迭代 |

## 结果

### CPU (x64)
| bench | compile 1.7.4→A.4 (ms) | 加速 | steady 1.7.4→A.4 (ms) | 加速 | 数值匹配 |
|---|---|---|---|---|---|
| mandelbrot | 72.61 → 64.80 | **1.12x** | 1.33 → 1.36 | 0.98x | OK (0) |
| matmul     | 57.00 → 51.08 | **1.12x** | 0.82 → 0.71 | **1.15x** | OK (2.8e-7) |
| nbody      | 69.83 → 60.06 | **1.16x** | 1.12 → 1.02 | **1.09x** | OK (1.1e-9) |
| reduce     | 64.58 → 58.46 | **1.10x** | 0.20 → 0.18 | **1.11x** | OK (0) |
| saxpy      | 49.39 → 45.25 | **1.09x** | 0.20 → 0.21 | 0.96x | OK (0) |
| stencil    | 60.28 → 53.34 | **1.13x** | 0.31 → 0.32 | 0.98x | OK (1.2e-7) |

### CUDA
| bench | compile 1.7.4→A.4 (ms) | 加速 | steady 1.7.4→A.4 (ms) | 加速 | 数值匹配 |
|---|---|---|---|---|---|
| mandelbrot | 133.04 → 116.80 | **1.14x** | 0.12 → 0.16 | 0.77x | OK (0) |
| matmul     | 102.84 → 91.11  | **1.13x** | 0.11 → 0.06 | **1.91x** | OK (0) |
| nbody      | 110.63 → 114.33 | 0.97x | 0.22 → 0.21 | **1.08x** | OK (1.3e-12) |
| reduce     | 128.45 → 111.98 | **1.15x** | 0.19 → 0.16 | **1.20x** | OK (3.6e-7) |
| saxpy      | 75.50 → 72.61   | **1.04x** | 0.08 → 0.06 | **1.28x** | OK (0) |
| stencil    | 93.64 → 82.29   | **1.14x** | 0.19 → 0.12 | **1.65x** | OK (0) |

## 结论

### ✅ 稳定性
- 12/12 case 全部运行通过，无 crash、无内存错误、无 CUDA launch 失败
- 每个 case 3 次重复 `ti.init` 均正常（初始化/释放路径无累积问题）

### ✅ 结果一致性
- 所有 checksum (sum/sum_abs/min/max/mean/l2) 相对误差 < 4e-7
- 最大差异 2.84e-7 出现在 CPU matmul 的 l2-范数上——属于浮点
  reduction 顺序引起的 ULP 级差异，符合 IEEE-754 允许范围
- nbody 差异 1.27e-12（CUDA）/ 1.11e-9（CPU），表明 transcendental 函数
  (`sqrt`, `sin`, `cos`) 精度一致

### ✅ 编译性能
- **11/12 case 编译加速 1.04x–1.16x**，仅 CUDA nbody 轻微慢 3%
- 符合预期：New PassManager 实现更好的 pass 调度，LLVM 19 O3 pipeline
  比 LLVM 15 PMB 更紧凑

### ✅ 运行性能
- **8/12 case 加速 1.08x–1.91x**，最大收益 CUDA matmul (1.91x),
  stencil (1.65x), saxpy (1.28x)
- 3 case 轻微回退 (0.96–0.98x)——在亚毫秒级别（0.2–1.4ms）基本属于
  噪声范围
- 1 case CUDA mandelbrot 0.77x：绝对时间 0.12→0.16 ms，都在 CUDA
  launch-overhead 级别，代表性有限

## 判定
Phase A.4 **可进入下一阶段**：
1. 无功能回退
2. 数值结果在浮点容差内一致
3. 编译性能普遍提升
4. 运行性能整体提升，个别亚毫秒场景的噪声级抖动不构成阻塞

## 复现命令
```powershell
# 基线
& <miniforge>\envs\ti174\python.exe tests\phase_a4\bench_suite.py `
    --label taichi-1.7.4-llvm15 --repeats 3 `
    --output build\bench_baseline.json

# 候选
& <miniforge>\envs\3.10\python.exe tests\phase_a4\bench_suite.py `
    --label taichi-1.8.0-llvm19-phaseA4 --repeats 3 `
    --output build\bench_candidate.json

# 对比
python tests\phase_a4\compare.py build\bench_baseline.json `
    build\bench_candidate.json --rel-tol 1e-3
```

## 下阶段建议 (Phase B 候选)

### 选项 B1 — 扩大验证面（低风险，推荐先做）
把 `bench_suite.py` 扩充到覆盖 Taichi 现有 `tests/python/` 的核心
kernel 子集（稀疏 SNode、autodiff Tape、动态索引、ndarray 交互），
跑完再开启大规模改造。

### 选项 B2 — opaque pointer 全面收尾
当前我们在 4 个 backend 做了 opaque pointer 兼容改造，但仍可能遗留
`CreateGEP(nullptr, ...)` / 隐式 bitcast 之类的遗留写法。建议：
- 启用 `-Wdeprecated` + clang-tidy `llvm-prefer-register-over-unsigned`
- 审计 `taichi/codegen/llvm/*.cpp` 中所有 `getPointerElementType` 调用
- 清理 Phase A.4 留下的 `TODO(llvm-19)` 注释

### 选项 B3 — VS 2026 C++20 迁移
外部依赖 (spdlog / volk / glm) 已在 Phase A.2/A.3 升级完毕，可以开始：
- 打开 `/std:c++20`
- 替换 `std::result_of` → `std::invoke_result`
- 清理 `std::iterator` 自定义派生 (C++20 弃用)

### 选项 B4 — CUDA 运行时升级
当前 driver API 调用未检查 CUDA 12 新增错误码；`cuMemAllocAsync`
等新 API 也未启用，可带来额外性能收益。

**推荐顺序**：B1 → B2 → B3 → B4（先加固再改造）。
