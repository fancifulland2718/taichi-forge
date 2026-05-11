# Taichi Forge — Compile, Runtime, Architecture & Modernization Options

> Applies to **Taichi Forge 0.3.13**. Every option listed here is **opt-in** unless explicitly noted; defaults preserve upstream Taichi 1.7.4 behaviour wherever a feature is not intentionally enabled by Forge.
>
> 中文版：[forge_options.zh.md](forge_options.zh.md)

This document is the single canonical reference for Forge-specific knobs and toolchain changes. Older fork-only settings are kept here once they are part of the public API; experimental or internal-only flags are intentionally excluded.

---

## 1. New Python APIs (fork-only)

| Symbol | Purpose |
|---|---|
| `ti.compile_kernels(kernels)` | Pre-compile a list of kernels on a background thread pool **before** the hot loop. Accepts either decorated kernels or `(kernel, args_tuple)` pairs. Returns the number of kernels submitted. Available on every backend. |
| `ti.compile_profile()` | Context manager. On exit, prints a per-pass timing report and (optionally) writes a CSV / Chrome-trace JSON. Use during development to find compile-time hot spots. |
| `@ti.kernel(opt_level=...)` | Per-kernel LLVM optimisation override. Accepts `"fast"` / `"balanced"` / `"full"` or `0`–`3`. Cache key is isolated per override, so mixed-tier batches do not poison each other. |

### CLI

| Command | Purpose |
|---|---|
| `ti cache warmup script.py [-- script-args]` | Run `script.py` once with the offline cache forced on, populating kernel artifacts for subsequent cold starts. Same arch / driver as the eventual run. |

---

## 2. `ti.init(...)` / `CompileConfig` keyword arguments

All defaults match upstream 1.7.4 unless noted.

### 2.1 Compile-time tier selection

| Kwarg | Default | Purpose |
|---|---|---|
| `compile_tier` | `"balanced"` | `"fast"` lowers LLVM to `-O0` (floor `-O1` on NVPTX / AMDGCN) and SPIR-V to optimization level 1. `"full"` preserves the pre-fork pipeline. |
| `llvm_opt_level` | `-1` (use tier) | Explicit LLVM `-O` override (`0`–`3`). |
| `spv_opt_level` | `-1` (use tier) | Explicit `spirv-opt` optimisation level override. |

### 2.2 Compile-pipeline batch & threading

| Kwarg | Default | Purpose |
|---|---|---|
| `num_compile_threads` | logical-core count | Thread-pool size used by `ti.compile_kernels`. |
| `compile_dag_scheduler` | `True` | DAG-aware anti-saturation scheduler for `ti.compile_kernels` batches; balances inner LLVM thread pool vs. outer kernel pool. Set `False` to fall back to the legacy two-tier model. |
| `spirv_parallel_codegen` | `False` | Opt-in per-kernel task-level parallel SPIR-V codegen. |
| `spirv_disabled_passes` | `[]` | Per-call disable list for individual `spirv-opt` passes (e.g. `["loop-unroll"]`). Cache-key isolated. Disabling `loop-unroll` alone yields ~54 % SPIR-V codegen wall-time reduction on the validated Vulkan suite; the three heaviest passes together yield ~61 %, with byte-identical kernel results. |

### 2.3 Pass / IR controls

| Kwarg | Default | Purpose |
|---|---|---|
| `use_fused_passes` | `False` | Enable `pipeline_dirty` short-circuit around `full_simplify`; numerically bit-identical to off. |
| `tiered_full_simplify` | `True` | Splits `full_simplify` into a local fixed-point phase plus a single global round per outer iteration. Set `False` for the legacy cadence. |
| `unrolling_hard_limit` | `0` (off) | Per-`ti.static(for ...)` unroll iteration cap. Aborts with `TaichiCompilationError` instead of silently consuming compile time. |
| `unrolling_kernel_hard_limit` | `0` (off) | Total unroll iteration cap across a single kernel. |
| `func_inline_depth_limit` | upstream default | Hard cap on `@ti.func` inline recursion depth. |
| `cache_loop_invariant_global_vars` | `False` | Opt in to SNode loop-invariant caching in hot loops. |

### 2.4 Real-function & inlining

| Kwarg | Default | Purpose |
|---|---|---|
| `auto_real_function` | `False` | Auto-promote expensive `@ti.func` instances to `is_real_function=True` (LLVM-only, non-autodiff). |
| `auto_real_function_threshold_us` | `1000` | Promotion threshold in microseconds of estimated compile cost. |

### 2.5 Compatibility-only

| Kwarg | Default | Purpose |
|---|---|---|
| `offline_cache_l_sem` | (off) | Internal / testing flag, not for production use. |
| `vulkan_quant_experimental` | `False` | **New in 0.3.0.** When ON, the Vulkan backend accepts `quant_array` / `bit_struct` fields (i.e. `Extension::quant` / `Extension::quant_basic` are reported supported on Vulkan). Supported: `QuantInt` / `QuantFixed` read, write, and concurrent multi-thread `ti.atomic_add` (via SPIR-V `OpAtomicCompareExchange` spin RMW) on `quant_array` and on multi-field `BitpackedFields` / `bit_struct`, byte-equivalent to cpu / cuda. **Explicitly not supported**: `QuantFloat` shared-exponent and the non-add atomic ops (`atomic_min/max/and/or/xor`, identical restriction to the LLVM backend). Unsupported sites raise `TI_NOT_IMPLEMENTED` / `TI_ERROR` rather than silently miscompile. Equivalent env var: `TI_VULKAN_QUANT=1`. |

### 2.6 CUDA sparse memory pool

By default the CUDA sparse SNode dynamic-allocation pool inherits the vanilla 1.7.4 sizing rule: it equals `device_memory_GB` (or `device_memory_fraction × total_VRAM` when set). Two opt-in paths customise this:

| Kwarg | Default | Purpose |
|---|---|---|
| `cuda_sparse_pool_size_GB` | `0.0` (use `device_memory_GB`) | Explicit pool size in GiB. Positive values bypass every other sizing path — use when you need the sparse pool decoupled from `device_memory_GB`. |
| `cuda_sparse_pool_auto_size` | `False` | Opt-in heuristic auto-sizing. When `True` **and** `device_memory_fraction == 0` **and** `cuda_sparse_pool_size_GB == 0`, the pool is derived from the SNode tree (heuristic per gc-able snode × 1024 chunks), capped by `device_memory_GB`, floored at `cuda_sparse_pool_size_floor_MiB`. Default `False` preserves vanilla 1.7.4 semantics. Verify the heuristic covers your workload's `NodeAllocator` activation peak before turning it on; MPM-shaped sparse trees may need higher `cuda_sparse_pool_size_floor_MiB` or explicit `cuda_sparse_pool_size_GB`. |
| `cuda_sparse_pool_size_floor_MiB` | `128` | Floor (MiB) for the auto-sized pool. Each `NodeAllocator` chunk is ~16 MiB, so 128 MiB ≈ 8 chunks. No-op when `cuda_sparse_pool_auto_size=False`. |

`device_memory_fraction > 0` and `cuda_sparse_pool_size_GB > 0` both still bypass auto-sizing entirely.

### 2.7 Sparse struct-for / listgen optimisations

Both flags default OFF and are bit-identical to the legacy path when off. Enabling them changes generated kernel code (CUDA grid_dim or SPIR-V atomics), and the change is keyed into the offline cache hash.

| Kwarg | Default | Purpose |
|---|---|---|
| `spirv_listgen_subgroup_ballot` | `False` | Vulkan/SPIR-V only. Aggregates per-thread `OpAtomicIAdd` into one subgroup-ballot atomic per active subgroup inside the listgen kernel. Reduces atomic contention on dense-active sparse struct-for. Requires the device to support subgroup ballot (the Vulkan adapter advertises this in standard SPIR-V capabilities); otherwise the flag has no effect. |
| `listgen_static_grid_dim` | `False` | CUDA / AMDGPU only. Launches sparse-listgen kernels with a `grid_dim` derived from the static upper bound on parent-element count (= product of `num_cells_per_container` of strict ancestors of the listed SNode, root excluded), capped by the hardware-saturating value. Eliminates idle blocks on shallow sparse trees. The Vulkan backend already computes the equivalent quantity via task attribs, so this flag is a no-op there. Correctness is preserved by the existing grid-stride loop in `element_listgen_nonroot`. |

### 2.8 Hash SNode

`hash` SNode is experimental and default ON in Taichi Forge 0.3.13. It is available on CPU, CUDA, and Vulkan, emits an experimental-feature warning on first `SNode.hash()` use, and can be disabled with `hash_snode_experimental=False`. See [hash_snode.en.md](hash_snode.en.md) for the API and migration notes.

| Kwarg | Values / default | Purpose | Risk / guidance |
|---|---|---|---|
| `hash_snode_experimental` | Bool, default `True`; set `False` to disable. | Enables `SNode.hash()` on CPU / CUDA / Vulkan. First use warns because the API is still experimental. | Keep `True` for normal Forge usage. Set `False` only to isolate regressions, reproduce vanilla-compatible rejection, or prevent accidental production use. Env alias: `TI_HASH_SNODE_EXPERIMENTAL=0/1`. |
| `hash_snode_default_load_factor` | Float in `(0, 1]`, default `0.5`. | Default load factor used when `SNode.hash(..., expected_active=N)` or `max_active=N` is supplied without a per-node `hash_load_factor`. | Lower values reserve more memory and usually shorten probes; higher values save memory but increase collision cost and overflow risk. Env alias: `TI_HASH_SNODE_DEFAULT_LOAD_FACTOR`. |
| `hash_snode_active_list` | Bool, default `False`. | Experimental active-bucket list for hash traversal. | Changes generated layout/code and may regress churn-heavy workloads. Enable only after focused benchmarks show a win. Env alias: `TI_HASH_SNODE_ACTIVE_LIST=0/1`. |
| `hash_snode_diagnostics` | Bool, default `False`. | Extra runtime counters for probe/tombstone debugging. | Diagnostic-only; can add memory/counter traffic and should stay off for production performance. Env alias: `TI_HASH_SNODE_DIAGNOSTICS=0/1`. |
| `hash_snode_compact_child_pool` | Bool, default `False`. | Experimental memory mode for `hash -> hash` / nested hash. Reduces reserved child-container memory when parent active count is much smaller than parent capacity. | Adds a parent-bucket to child-slot lookup, so it can trade latency for memory. Enable only when nested hash memory dominates and benchmark data supports it. Env alias: `TI_HASH_SNODE_COMPACT_CHILD_POOL=0/1`. |

---

## 3. Environment variables

| Variable | Range | Default | Purpose |
|---|---|---|---|
| `TI_VULKAN_POOL_FRACTION` | `(0.0, 1.0]` | `1.0` | Shrinks each `pointer` SNode's physical cell pool to `max(num_cells_per_container, round(total × fraction))`. Out-of-capacity activates fall through the existing `cap_v` silent-inactive guard. Invalid / `≤ 0` / `> 1` falls back to `1.0`. Detailed semantics: see [sparse_snode_on_vulkan.en.md](sparse_snode_on_vulkan.en.md). |
| `TI_VULKAN_QUANT` | `0` / `1` | `0` | **New in 0.3.0.** Equivalent to `ti.init(arch=ti.vulkan, vulkan_quant_experimental=True)`. When ON, `quant_array` and `BitpackedFields` / `bit_struct` read, write, and `ti.atomic_add` are all available on Vulkan. `QuantFloat` shared-exponent and non-add atomics are explicitly not supported. OFF preserves vanilla 1.7.4 behaviour. |

> Other environment variables documented in upstream Taichi remain unchanged (`TI_ARCH`, `TI_DEVICE_MEMORY_GB`, etc.). They are not re-listed here.

---

## 4. CMake build options (developer-side)

> These are surfaced **only** when building Forge from source. End users installing the published wheel get every default-ON path; no flags need to be set.

| Option | Default | Purpose |
|---|---|---|
| `TI_VULKAN_POINTER` | ON | Master switch for `pointer` / `bitmasked` SNode on Vulkan. OFF → vanilla `TI_NOT_IMPLEMENTED`. |
| `TI_VULKAN_DYNAMIC` | ON | Master switch for `dynamic` SNode on Vulkan. OFF → vanilla `TI_NOT_IMPLEMENTED`. |
| `TI_VULKAN_POINTER_POOL_FRACTION` | ON | Activates `TI_VULKAN_POOL_FRACTION`. OFF makes the env var a no-op; capacity is reserved for the worst case. |

The wheel published to PyPI builds with all three flags ON.

---

## 5. SNode coverage extensions

| SNode type | vanilla 1.7.4 Vulkan | Taichi Forge 0.3.13 Vulkan |
|---|---|---|
| `dense` | ✅ | ✅ |
| `bitmasked` | ❌ | ✅ |
| `pointer` | ❌ | ✅ |
| `dynamic` | ❌ | ✅ |
| `hash` | ❌ | ⚠️ experimental, default ON with first-use warning |

Full Vulkan sparse usage and semantics: [sparse_snode_on_vulkan.en.md](sparse_snode_on_vulkan.en.md). Hash SNode API: [hash_snode.en.md](hash_snode.en.md).

---

## 6. Toolchain & dependency upgrades

Forge ships against modern toolchains; the table below summarises the versions vs. vanilla 1.7.4.

| Component | vanilla 1.7.4 | Forge 0.3.13 |
|---|---|---|
| LLVM | 15 | **20.1.7** |
| Python | 3.7 – 3.12 | **3.10 – 3.14** |
| Windows MSVC | VS 2019 / 2022 | **VS 2026 (MSVC 14.50+)** |
| `spdlog` | 1.14.1 | **1.15.3** |
| `Vulkan-Headers` / `volk` / `SPIRV-Headers` / `SPIRV-Tools` | older | aligned to **Vulkan SDK 1.4.341** |
| `googletest` | 1.10.0 | **1.17.0** |
| `glm` | 0.9.9.8 + 187 | **1.0.3** |
| `imgui` | v1.84 (WIP) | **v1.91.9b** (non-docking branch) |

The Vulkan ImGui backend was migrated to the new `ImGui_ImplVulkan_InitInfo` layout (`RenderPass` + `ApiVersion` fields, self-managed font texture, `LoadFunctions(api_version, loader)` signature). The GGUI visual-regression suite passes 90 / 90 on the Vulkan + CUDA backends.

---

## 7. Architecture / robustness improvements

These are not user-tunable; they ship enabled by default. Listed for visibility / debugging.

- **Offline cache cross-version safety** — corrupt or version-mismatched `ticache.tcb` triggers an automatic fallback recompile rather than crashing.
- **`rhi_cache.bin` atomic write** — write-then-rename eliminates half-written cache files after abrupt termination.
- **`pipeline_dirty` precise tracking** — explicit OR-combined dirty marks across the five mutating passes that affect `full_simplify`. Removes spurious dirty marks at no-op call sites; verified across CPU / CUDA / Vulkan smoke matrices with no regression.
- **Single-offload bypass on the LLVM CPU path** — removes the prior 0.89× CPU regression introduced by earlier batch-compile work.
- **Defensive type-context guards** on `linking_context_data->llvm_context` to catch accidental cross-context type queries.

---

## 8. Compatibility statement

- All public Python and C-API surfaces from upstream Taichi 1.7.4 remain unchanged.
- Every fork-only knob in this document is additive and defaults to upstream behaviour.
- The published wheel is drop-in for any code that imports `taichi`.

---

## 9. See also

- Sparse SNode on Vulkan user guide: [sparse_snode_on_vulkan.en.md](sparse_snode_on_vulkan.en.md)
- Hash SNode user guide: [hash_snode.en.md](hash_snode.en.md)
