// Shared projection of code-shaping configuration values. Keep serialization
// and fast in-memory context comparison on this single field list. The caller
// defines CACHE_FIELD, CACHE_TYPE, CACHE_SORTED, CACHE_VALUE, CACHE_CUDA_TARGET.
// No include guard: intentionally instantiated for both operations.
  CACHE_FIELD(arch);
  CACHE_FIELD(debug);
  CACHE_FIELD(cfg_optimization);
  CACHE_FIELD(check_out_of_bound);
  CACHE_FIELD(opt_level);
  CACHE_FIELD(external_optimization_level);
  CACHE_FIELD(llvm_opt_level);
  CACHE_FIELD(compile_tier);
  CACHE_FIELD(tiered_full_simplify);
  CACHE_FIELD(full_simplify_global_iter_cap);
  CACHE_FIELD(move_loop_invariant_outside_if);
  CACHE_FIELD(demote_dense_struct_fors);
  CACHE_FIELD(spirv_skip_intermediate_listgen);
  CACHE_FIELD(spirv_listgen_subgroup_ballot);
  CACHE_FIELD(listgen_static_grid_dim);
  CACHE_FIELD(advanced_optimization);
  CACHE_FIELD(constant_folding);
  CACHE_FIELD(kernel_profiler);
  CACHE_FIELD(fast_math);
  CACHE_FIELD(flatten_if);
  CACHE_FIELD(cache_loop_invariant_global_vars);
  CACHE_FIELD(make_thread_local);
  CACHE_FIELD(make_block_local);
  CACHE_FIELD(detect_read_only);
  CACHE_FIELD(quant_opt_store_fusion);
  CACHE_FIELD(quant_opt_atomic_demotion);
  CACHE_TYPE(default_fp);
  CACHE_TYPE(default_ip);
  if (arch_is_cpu(config.arch)) {
    CACHE_FIELD(default_cpu_block_dim);
    CACHE_FIELD(cpu_max_num_threads);
    CACHE_FIELD(make_cpu_multithreading_loop);
  } else if (arch_is_gpu(config.arch)) {
    CACHE_FIELD(default_gpu_block_dim);
    CACHE_FIELD(gpu_max_reg);
    CACHE_FIELD(saturating_grid_dim);
    CACHE_FIELD(max_block_dim);
    CACHE_FIELD(cpu_max_num_threads);
  }
  CACHE_FIELD(ad_stack_size);
  CACHE_FIELD(default_ad_stack_size);
  // NOTE: config.random_seed is intentionally NOT part of the offline cache
  // key.  It only affects the runtime PRNG seed (see
  // LlvmRuntimeExecutor::materialize_runtime); the generated IR / LLVM module /
  // SPIR-V are identical regardless of its value.  Including it here caused
  // spurious cache misses whenever the user changes ti.init(random_seed=...)
  // between sessions.  [P1.a cache-key trim]
  if (config.arch == Arch::opengl || config.arch == Arch::gles) {
    CACHE_FIELD(allow_nv_shader_extension);
  }
  CACHE_FIELD(make_mesh_block_local);
  CACHE_FIELD(optimize_mesh_reordered_mapping);
  CACHE_FIELD(mesh_localize_to_end_mapping);
  CACHE_FIELD(mesh_localize_from_end_mapping);
  CACHE_FIELD(mesh_localize_all_attr_mappings);
  CACHE_FIELD(demote_no_access_mesh_fors);
  CACHE_FIELD(experimental_auto_mesh_local);
  CACHE_FIELD(auto_mesh_local_default_occupacy);
  CACHE_FIELD(real_matrix_scalarize);
  CACHE_FIELD(hash_snode_active_list);
  CACHE_FIELD(hash_snode_diagnostics);
  CACHE_FIELD(hash_snode_compact_child_pool);
  // P9.A (F2/F3): auto_real_function gating + inline budget influence
  // FuncCallStmt presence and inliner behavior; both must invalidate cache.
  CACHE_FIELD(auto_real_function);
  CACHE_FIELD(auto_real_function_threshold_us);
  CACHE_FIELD(auto_real_function_inline_budget);
  CACHE_FIELD(force_scalarize_matrix);
  CACHE_FIELD(half2_vectorization);
  // B2 (2026-04-26): SPIR-V disabled-pass list affects emitted SPIR-V on
  // SPIR-V backends. Sort first so user-supplied list ordering doesn't
  // produce spurious cache misses. Empty list (default) hashes to a
  // stable empty entry, so legacy users see no cache invalidation.
  CACHE_SORTED(spirv_disabled_passes);
  // G-6 (2026-05): task-level SPIR-V adaptive optimizer changes emitted
  // SPIR-V per task, so ON/OFF and threshold changes must not share cache.
  CACHE_FIELD(spirv_adaptive_opt);
  CACHE_FIELD(spirv_adaptive_opt_threshold);
  if (arch_uses_spirv(config.arch)) {
    // Old SPIR-V may address mixed-width SNodes with unaligned offsets/pools.
    // This is a compiler layout revision, independent of wheel/commit identity.
    CACHE_VALUE(std::string("spirv-snode-scalar-alignment-v1"));
    CACHE_FIELD(spirv_skip_loop_unroll);
  }
  // B-2.b (2026-05): the 4 vulkan_pointer_* runtime fields drive both
  // root-buffer layout and pointer-SNode SPIR-V codegen. They MUST be
  // part of the cache key, otherwise toggling vulkan_pointer_ambient_zone
  // / _freelist / _cas_marker / _pool_fraction silently reuses kernels
  // compiled under the previous flag value. Default values (True/True/
  // True/1.0) hash deterministically so legacy users see no invalidation.
  if (config.arch == Arch::vulkan) {
    CACHE_FIELD(vulkan_pointer_freelist);
    CACHE_FIELD(vulkan_pointer_ambient_zone);
    CACHE_FIELD(vulkan_pointer_cas_marker);
    CACHE_FIELD(vulkan_pointer_pool_fraction);
    // B-3.b (2026-05): independent_pool 影响 SpirvAllocatorContract.
    // pool_buffer_binding_id 与 SNodeTree allocator 申请独立 DeviceAllocation。
    // 即使 codegen 在 B-3.b 不读 binding_id，提前纳入 cache key 避免 B-3.c
    // 切换 codegen 后命中旧缓存。默认 false 哈希稳定。
    CACHE_FIELD(vulkan_pointer_independent_pool);
    // C-2.1 (2026-05): allocator_kind 选不同 allocator 实现 → SPIR-V kernel
    // 寻址（C-2.3 起）和 SNodeTree allocator 构造均不同；必须进入 cache key。
    // 默认 "bump" 字符串哈希稳定，旧用户无 invalidation。
    CACHE_FIELD(vulkan_pointer_allocator_kind);
    CACHE_FIELD(vulkan_pointer_max_chunks);
    // C-9 (2026-05): deterministic_slot 改变 alloc 协议（idx_u32+1 直写
    // vs CasMarker 抢占），必须进入 cache key。详见规划 §14。
    CACHE_FIELD(vulkan_pointer_deterministic_slot);
    // G11-A (2026-05): bitmasked deactivate 是否清 data slot。改变
    // codegen（LLVM 改调函数名、SPIR-V 多发射条件 memset），必须进
    // cache key。默认 false 保持哈希稳定。
    CACHE_FIELD(bitmasked_clear_data_on_deactivate);
    // VS-3 (2026-05): toggles SPIR-V TaskAttributes sparse-list metadata
    // consumed by GfxRuntime host-side listgen skipping.
    CACHE_FIELD(vulkan_listgen_reuse);
  }
  if (config.arch == Arch::cuda) {
    // CS-1/2/3 (2026-05): these CUDA sparse flags alter emitted LLVM IR or
    // runtime metadata decisions. Include them so ON/OFF runs do not reuse
    // stale offline-cache entries.
    CACHE_FIELD(cuda_pointer_deterministic_slot);
    CACHE_FIELD(cuda_pointer_deterministic_pool_enabled());
    CACHE_FIELD(cuda_pointer_fast_reset);
    CACHE_FIELD(cuda_listgen_reuse);
    CACHE_FIELD(bitmasked_clear_data_on_deactivate);
    CACHE_CUDA_TARGET();
  }
