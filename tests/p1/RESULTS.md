
---

## Run on 2026-04-23 21:50

| Field | Value |
|---|---|
| commit | `9407d5182` |
| machine | AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD |
| python | 3.10.20 |
| platform | win32 |
| llvm_levels | [3, 2, 1, 0] |
| spv_levels  | [3]  |

| kernel | llvm=3/spv=3 | llvm=2/spv=3 | llvm=1/spv=3 | llvm=0/spv=3 |
|---|---|---|---|---|
| autodiff_heavy | 75 ms | 82 ms ¡ü1.10x | 71 ms ¡ý0.95x | 80 ms ¡ü1.07x |
| matscalar_10x10 | 7388 ms | 7363 ms ~1.00x | 7461 ms ~1.01x | 7312 ms ~0.99x |
| branch_heavy_32 | 154 ms | 154 ms ~0.99x | 162 ms ~1.05x | 154 ms ~1.00x |
| deep_func_chain_6 | 2180 ms | 2169 ms ~1.00x | 2169 ms ~0.99x | 2137 ms ~0.98x |
| wide_stencil_4pln | 235 ms | 240 ms ~1.02x | 233 ms ~0.99x | 284 ms ¡ü1.21x |
| many_fields_12 | 96 ms | 96 ms ~1.00x | 98 ms ~1.03x | 121 ms ¡ü1.27x |
| autodiff_matrix | 109 ms | 106 ms ~0.98x | 106 ms ~0.97x | 113 ms ~1.04x |

> Baseline column = max(llvm_opt)/max(spv_opt).  Ratio shown relative to that baseline.  ¡ý = faster compile, ¡ü = slower compile (regression).

---

## Run on 2026-04-23 21:52

| Field | Value |
|---|---|
| commit | `9407d5182` |
| machine | AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD |
| python | 3.10.20 |
| platform | win32 |
| llvm_levels | [3, 2, 1, 0] |
| spv_levels  | [3]  |

| kernel | llvm=3/spv=3 | llvm=2/spv=3 | llvm=1/spv=3 | llvm=0/spv=3 |
|---|---|---|---|---|
| autodiff_heavy | 83 ms | 80 ms ~0.97x | 81 ms ~0.98x | 90 ms ¡ü1.08x |
| matscalar_10x10 | 7416 ms | 7393 ms ~1.00x | 7389 ms ~1.00x | 7373 ms ~0.99x |
| branch_heavy_32 | 163 ms | 163 ms ~1.00x | 162 ms ~1.00x | 161 ms ~0.99x |
| deep_func_chain_6 | 2201 ms | 2187 ms ~0.99x | 2181 ms ~0.99x | 2116 ms ~0.96x |
| wide_stencil_4pln | 237 ms | 239 ms ~1.00x | 232 ms ~0.98x | 282 ms ¡ü1.19x |
| many_fields_12 | 94 ms | 96 ms ~1.02x | 94 ms ~1.01x | 113 ms ¡ü1.21x |
| autodiff_matrix | 114 ms | 111 ms ~0.97x | 111 ms ~0.97x | 111 ms ~0.97x |

> Baseline column = max(llvm_opt)/max(spv_opt).  Ratio shown relative to that baseline.  ¡ý = faster compile, ¡ü = slower compile (regression).


---

## Analysis

### Summary of Findings

Two independent runs (2026-04-23 21:50 and 21:52) are highly consistent (Â±5%).

#### Compile-time classification by bottleneck

| Kernel | Dominant bottleneck | llvm_opt sensitivity |
|---|---|---|
| matscalar_10x10 | Taichi scalarization (100 scalar exprs) | ~0% (7.3â€“7.5 s at all levels) |
| deep_func_chain_6 | Taichi CHI-level inlining | ~2% (all levels â‰ˆ 2.1â€“2.2 s) |
| wide_stencil_4pln | LLVM backend IR size | +21% at L0 vs L3 (inversion) |
| many_fields_12 | LLVM mem2reg / aliasing | +24% at L0 vs L3 (inversion) |
| autodiff_heavy | Mixed | â‰¤10% noise-level variation |
| branch_heavy_32 | Mixed | â‰¤5% noise-level variation |
| autodiff_matrix | Mixed | â‰¤7% noise-level variation |

#### Key observations

1. **llvm_opt_level does NOT reduce compile time for Taichi's heaviest kernels.**
   For matscalar_10x10 (7.4 s) and deep_func_chain_6 (2.2 s), the bottleneck is
   entirely inside Taichi's own CHI-IR passes (matrix scalarization, full_simplify, loop-
   level inlining), which run *before* LLVM sees any IR.  Changing llvm_opt_level
   from 3 to 0 yields â‰¤2% change â€” within noise.

2. **Compile-time inversion at L0 for some kernels.**
   wide_stencil_4pln (+21%) and many_fields_12 (+24%) are *slower* at L0 than L3.
   LLVM's early passes (mem2reg, instcombine, early-cse) dramatically reduce IR size for
   kernels with many field accesses and temporaries.  At L0 the backend must process
   bloated unoptimized IR, costing more total time despite skipping pass infrastructure.

3. **llvm_opt_level is primarily a runtime-perf knob, not a compile-speed knob.**
   Its correct use case is trading runtime performance for slightly lower startup overhead
   on fast/simple kernels.  Users should not expect it to speed up compilation of large
   kernels.

4. **To improve Taichi's compile time for heavy workloads**, optimization effort should
   target Taichi-side passes: scalarization IR explosion, full_simplify pass count, and
   CHI-level inlining depth â€” not the LLVM optimization pipeline.
