# Taichi Forge — Compile, Runtime, Architecture & Modernization Options

> Applies to the **Taichi Forge 0.5.x** release line. Every option listed here is **opt-in** unless explicitly noted; defaults preserve upstream Taichi 1.7.4 behaviour wherever a feature is not intentionally enabled by Forge.
> Option introduction versions are indexed in [release notes](release_notes.en.md);
> this current-contract page does not reclassify older options as `0.5.0` work.

This document is the single canonical reference for Forge-specific knobs and toolchain changes. Only options in the supported sections should be surfaced by applications. Section 2.9 records retired, compatibility-only, and validation-only names so old configurations fail or migrate clearly; listing a name there is not a support recommendation. For a module-oriented list of Forge-only API symbols, see [Forge API reference](forge_api_reference.en.md).

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
| `compile_tier` | `"balanced"` | `"fast"` forces LLVM to `-O0` (floor `-O1` on NVPTX / AMDGCN) and skips `spirv-opt` at level 0. `"balanced"` and `"full"` keep the configured backend level. This is the recommended application-facing control. |
| `llvm_opt_level` | `3` | Explicit LLVM `-O` level (`0`–`3`) for non-`fast` tiers. `compile_tier="fast"` still forces the backend-safe O0/O1 level. Prefer `compile_tier` unless a representative benchmark justifies a backend-specific override. |

### 2.2 Compile-pipeline batch & threading

| Kwarg | Default | Purpose |
|---|---|---|
| `num_compile_threads` | logical-core count | Thread-pool size used by `ti.compile_kernels`. |
| `compile_dag_scheduler` | `True` | DAG-aware anti-saturation scheduler for `ti.compile_kernels` batches; balances inner LLVM thread pool vs. outer kernel pool. Set `False` to fall back to the legacy two-tier model. |
| `spirv_parallel_codegen` | `False` | Opt-in per-kernel task-level parallel SPIR-V codegen. |

### 2.3 Pass / IR controls

| Kwarg | Default | Purpose |
|---|---|---|
| `tiered_full_simplify` | `True` | Splits `full_simplify` into a local fixed-point phase plus a single global round per outer iteration. It is cache-key isolated. Keep `True`; use `False` only to isolate a compiler regression against the legacy cadence. |
| `unrolling_hard_limit` | `0` (off) | Per-`ti.static(for ...)` unroll iteration cap. Aborts with `TaichiCompilationError` instead of silently consuming compile time. |
| `unrolling_kernel_hard_limit` | `0` (off) | Total unroll iteration cap across a single kernel. |
| `func_inline_depth_limit` | upstream default | Hard cap on `@ti.func` inline recursion depth. |
| `kernel_specialization_limit` | `1024` | Total compiled `@ti.kernel` specialization budget for the current Program generation. Existing specializations keep running at the positive limit, while new cache misses fail clearly. Raise it only for an application with a deliberately finite template-argument set. `ti.reset()` creates a new generation. |
| `check_out_of_bound` | `False`; implicitly `True` when `debug=True` and unspecified | Enables generated bounds assertions on backends with assertion support. An explicit `check_out_of_bound=False` (or `TI_CHECK_OUT_OF_BOUND=0`) now overrides the debug default without disabling other debug behavior. |

Bounds checks change generated code and are represented by their effective
boolean value in the offline-cache key. Explicitly disabling them can be useful
when isolating their cost or when an application supplies an independently
validated bounds contract, but unsafe indexing then has normal backend-undefined
behavior. CPU and CUDA currently support generated assertions; Vulkan warns and
disables this option because that backend does not yet advertise the assertion
extension.

### 2.4 Real-function & inlining

| Kwarg | Default | Purpose |
|---|---|---|
| `auto_real_function` | `False` | Experimental one-way promotion of expensive `@ti.func` instances to `is_real_function=True` (LLVM-only, non-autodiff, cache-key isolated). Do not enable engine-wide or on AD/cross-backend paths; use only after workload-specific compile and runtime validation. |
| `auto_real_function_threshold_us` | `1000` | Promotion threshold in microseconds of estimated compile cost. |

### 2.5 Vulkan quantization

| Kwarg | Default | Purpose |
|---|---|---|
| `vulkan_quant_experimental` | `False` | **New in 0.3.0.** When ON, the Vulkan backend accepts `quant_array` / `bit_struct` fields (i.e. `Extension::quant` / `Extension::quant_basic` are reported supported on Vulkan). Supported: `QuantInt` / `QuantFixed` read, write, and concurrent multi-thread `ti.atomic_add` (via SPIR-V `OpAtomicCompareExchange` spin RMW) on `quant_array` and on multi-field `BitpackedFields` / `bit_struct`, byte-equivalent to cpu / cuda. **Explicitly not supported**: `QuantFloat` shared-exponent and the non-add atomic ops (`atomic_min/max/and/or/xor`, identical restriction to the LLVM backend). Unsupported sites raise `TI_NOT_IMPLEMENTED` / `TI_ERROR` rather than silently miscompile. Equivalent env var: `TI_VULKAN_QUANT=1`. |

### 2.6 CUDA sparse memory pool

By default Forge derives the CUDA sparse SNode pool from the materialized SNode tree and carves a dedicated data region for each allocatable SNode inside one owning allocation. Explicit `device_memory_fraction` or `cuda_sparse_pool_size_GB` still selects the corresponding fixed-budget path.

| Kwarg | Default | Purpose |
|---|---|---|
| `cuda_sparse_pool_size_GB` | `0.0` (no explicit override) | Explicit pool size in GiB. Positive values bypass every other sizing path — use when you need a fixed sparse-pool budget. |
| `cuda_sparse_pool_auto_size` | `True` | When `device_memory_fraction == 0` and `cuda_sparse_pool_size_GB == 0`, derive the pool from each allocatable SNode's global cell bound, actual `NodeManager` chunk geometry, list metadata, and bounded GC/re-activation headroom. `device_memory_GB` is a warn-only sanity threshold; the derived size is not silently clamped below the required capacity. |
| `cuda_sparse_per_snode_pool` | `True` | With auto-sizing, carve one dedicated data region per allocatable SNode while retaining a shared global metadata/list region. This isolates nested allocator demand without adding one CUDA allocation per SNode. |
| `cuda_sparse_pool_size_floor_MiB` | `0` | Optional user floor (MiB) for the derived pool. The global metadata/list baseline and per-SNode chunk budgets are always included, so no additional defensive floor is applied by default. No-op when auto-sizing is bypassed. |

`device_memory_fraction > 0` and `cuda_sparse_pool_size_GB > 0` both bypass auto-sizing entirely. `vk_max_active` can lower a per-SNode expected-active bound, while the no-hint path uses the global number of cells represented by that SNode (not just one parent's container). On CPU and other LLVM backends, the hint does not cap the on-demand sparse payload; it only selects downstream traversal element-list chunk geometry.

### 2.7 Sparse struct-for / listgen optimisations

Both flags default OFF and are bit-identical to the legacy path when off. Enabling them changes generated kernel code (CUDA grid_dim or SPIR-V atomics), and the change is keyed into the offline cache hash.

| Kwarg | Default | Purpose |
|---|---|---|
| `spirv_listgen_subgroup_ballot` | `False` | Vulkan/SPIR-V only. Aggregates per-thread `OpAtomicIAdd` into one subgroup-ballot atomic per active subgroup inside the listgen kernel. Reduces atomic contention on dense-active sparse struct-for. Requires the device to support subgroup ballot (the Vulkan adapter advertises this in standard SPIR-V capabilities); otherwise the flag has no effect. |
| `listgen_static_grid_dim` | `False` | CUDA / AMDGPU only. Launches sparse-listgen kernels with a `grid_dim` derived from the static upper bound on parent-element count (= product of `num_cells_per_container` of strict ancestors of the listed SNode, root excluded), capped by the hardware-saturating value. Eliminates idle blocks on shallow sparse trees. The Vulkan backend already computes the equivalent quantity via task attribs, so this flag is a no-op there. Correctness is preserved by the existing grid-stride loop in `element_listgen_nonroot`. |

### 2.8 Hash SNode

`hash` SNode is experimental and has been default ON since Taichi Forge 0.3.13. It is available on CPU, CUDA, and Vulkan, emits an experimental-feature warning on first `SNode.hash()` use, and can be disabled with `hash_snode_experimental=False`. See [hash_snode.en.md](hash_snode.en.md) for the API and migration notes.

| Kwarg | Values / default | Purpose | Risk / guidance |
|---|---|---|---|
| `hash_snode_experimental` | Bool, default `True`; set `False` to disable. | Enables `SNode.hash()` on CPU / CUDA / Vulkan. First use warns because the API is still experimental. | Keep `True` for normal Forge usage. Set `False` only to isolate regressions, reproduce vanilla-compatible rejection, or prevent accidental production use. Env alias: `TI_HASH_SNODE_EXPERIMENTAL=0/1`. |
| `hash_snode_default_load_factor` | Float in `(0, 1]`, default `0.5`. | Default load factor used when `SNode.hash(..., expected_active=N)` or `max_active=N` is supplied without a per-node `hash_load_factor`. | Lower values reserve more memory and usually shorten probes; higher values save memory but increase collision cost and overflow risk. Env alias: `TI_HASH_SNODE_DEFAULT_LOAD_FACTOR`. |
| `hash_snode_active_list` | Bool, default `False`. | Experimental active-bucket list for hash traversal. | Changes generated layout/code and may regress churn-heavy workloads. Enable only after focused benchmarks show a win. Env alias: `TI_HASH_SNODE_ACTIVE_LIST=0/1`. |
| `hash_snode_diagnostics` | Bool, default `False`. | Extra runtime counters for probe/tombstone debugging. | Diagnostic-only; can add memory/counter traffic and should stay off for production performance. Env alias: `TI_HASH_SNODE_DIAGNOSTICS=0/1`. |
| `hash_snode_compact_child_pool` | Bool, default `False`. | Experimental memory mode for `hash -> hash` / nested hash. Reduces reserved child-container memory when parent active count is much smaller than parent capacity. | Adds a parent-bucket to child-slot lookup, so it can trade latency for memory. Enable only when nested hash memory dominates and benchmark data supports it. Env alias: `TI_HASH_SNODE_COMPACT_CHILD_POOL=0/1`. |

### 2.9 Retired, compatibility-only, and validation-only settings

These names are documented only to make old configuration files and research
results unambiguous. Applications and engines should not expose them.

| Name | Current behavior | Required action |
|---|---|---|
| `use_fused_passes`, `fused_pass_verify` | Removed before the 0.4.23 public baseline together with the low-ROI `pipeline_dirty` experiment. Current wheels reject them as unknown `ti.init` arguments. | Delete them. The stable pipeline always runs the required simplification path. |
| `spv_opt_level` | Not a current Python/CompileConfig field; current wheels reject it. The implementation's raw field is `external_optimization_level`. | Use `compile_tier`. Do not rename the old setting to the raw field in application code. |
| `skip-loop-unroll`, `skip_loop_unroll` | Not accepted `ti.init` names. The current raw experiment is `spirv_skip_loop_unroll`. | Delete them; do not translate them into engine configuration. |
| `vulkan_listgen_lite_barrier` | Accepted only as a deprecated no-op compatibility field. The active narrow-barrier path belongs to `vulkan_dispatch_cache`. | Delete it; changing the value has no supported effect. |
| `vulkan_launch_buffer_pool`, `vulkan_launch_buffer_pool_capacity` | Accepted deprecated no-op fields. The old standalone pool was removed for negligible ROI and superseded by fence-safe GFX context handling. | Delete them; do not tune the capacity. |

The following fields still exist for compiler experiments, but their public
naming, cache contract, or cross-driver evidence is not strong enough for
production configuration:

| Name | Current implementation contract | Production guidance |
|---|---|---|
| `external_optimization_level` | Raw SPIR-V optimizer level, default `3`; serialized in the offline-cache key. `compile_tier="fast"` overrides it to level `0`. | Keep application code on `compile_tier`; do not expose this field through GeoPhys or another engine. |
| `spirv_disabled_passes` | Default `[]`; changes emitted SPIR-V and uses a sorted, cache-isolated list. Current internal pass IDs are case-sensitive (for example `LoopUnroll`), but that vocabulary is not a stable public API. | Keep empty until naming and cross-driver validation are finalized. |
| `spirv_skip_loop_unroll` | Default `False`; changes the optimizer chain and emitted SPIR-V, but is currently not represented in the offline-cache key. | Keep `False`; do not expose or use it with production/offline-cache workloads. |
| `spirv_adaptive_opt`, `spirv_adaptive_opt_threshold` | Default `False` / `64`; cache-key isolated, but changes the optimizer chain by task shape. | Validation and benchmarking only until the driver matrix converges. |
| `cache_loop_invariant_global_vars` | Default `False`; changes IR for a narrow workload and is currently not represented in the offline-cache key. Prior measurements showed cold-compile cost with limited physics-runtime benefit. | Keep `False`; do not expose it as a general performance knob. |

---

## 3. Environment variables

| Variable | Range | Default | Purpose |
|---|---|---|---|
| `TI_VULKAN_POOL_FRACTION` | `(0.0, 1.0]` | `1.0` | Shrinks each `pointer` SNode's physical cell pool to `max(num_cells_per_container, round(total × fraction))`. Out-of-capacity addresses are safely clamped and the next synchronization boundary raises a diagnostic. Invalid / `≤ 0` / `> 1` falls back to `1.0`. Detailed semantics: see [sparse_snode_on_vulkan.en.md](sparse_snode_on_vulkan.en.md). |
| `TI_VULKAN_QUANT` | `0` / `1` | `0` | **New in 0.3.0.** Equivalent to `ti.init(arch=ti.vulkan, vulkan_quant_experimental=True)`. When ON, `quant_array` and `BitpackedFields` / `bit_struct` read, write, and `ti.atomic_add` are all available on Vulkan. `QuantFloat` shared-exponent and non-add atomics are explicitly not supported. OFF preserves vanilla 1.7.4 behaviour. |
| `TI_KERNEL_PROFILER_MAX_RECORDS` | `1`–`1048576` | `131072` | In-process raw-record budget for the kernel profiler. Reaching it reports a clear error instead of growing further. Long sessions should call `ti.profiler.clear_kernel_profiler_info()` periodically; raise the limit only after budgeting host memory. |

> Other environment variables documented in upstream Taichi remain unchanged (`TI_ARCH`, `TI_DEVICE_MEMORY_GB`, etc.). They are not re-listed here.

---

## 4. CMake build options (developer-side)

> These are surfaced **only** when building Forge from source. End users installing the published wheel get every default-ON path; no flags need to be set.

| Option | Default | Purpose |
|---|---|---|
| `TI_VULKAN_POINTER` | ON | Master switch for `pointer` / `bitmasked` SNode on Vulkan. OFF → vanilla `TI_NOT_IMPLEMENTED`. |
| `TI_VULKAN_DYNAMIC` | ON | Master switch for `dynamic` SNode on Vulkan. OFF → vanilla `TI_NOT_IMPLEMENTED`. |
| `TI_VULKAN_POINTER_POOL_FRACTION` | ON | Activates `TI_VULKAN_POOL_FRACTION`. OFF makes the env var a no-op; capacity is reserved for the worst case. |

Release wheel builds enable all three flags.

---

## 5. SNode coverage extensions

| SNode type | vanilla 1.7.4 Vulkan | Taichi Forge 0.5.x Vulkan |
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

| Component | vanilla 1.7.4 | Forge 0.5.x |
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
- **Single-offload bypass on the LLVM CPU path** — removes the prior 0.89× CPU regression introduced by earlier batch-compile work.
- **Defensive type-context guards** on `linking_context_data->llvm_context` to catch accidental cross-context type queries.

---

## 8. Compatibility statement

- Supported upstream Taichi 1.7.4 Python APIs are the compatibility reference;
  documented Forge changes and experimental paths remain explicit exceptions.
- Every supported fork-only knob in sections 1–2.8 is additive and defaults to
  upstream behaviour unless its entry explicitly says otherwise. Section 2.9
  is a migration/diagnostic registry, not a supported-option list.
- The PyPI package imports as `taichi_forge`; it does not replace the upstream
  `taichi` package. The C API package tree is not included in PyPI shim wheels
  and must be built separately when required.

---

## 9. See also

- Sparse SNode on Vulkan user guide: [sparse_snode_on_vulkan.en.md](sparse_snode_on_vulkan.en.md)
- Sparse layout selection guide: [sparse_layout_selection.en.md](sparse_layout_selection.en.md)
- Hash SNode user guide: [hash_snode.en.md](hash_snode.en.md)
