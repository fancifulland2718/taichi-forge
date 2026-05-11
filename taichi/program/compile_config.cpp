#include "compile_config.h"

#include <thread>
#include "taichi/rhi/arch.h"
#include "taichi/util/offline_cache.h"

namespace taichi::lang {

CompileConfig::CompileConfig() {
  arch = host_arch();
  simd_width = default_simd_width(arch);
  opt_level = 1;
  external_optimization_level = 3;
  llvm_opt_level = 3;
  print_ir = false;
  print_preprocessed_ir = false;
  print_accessor_ir = false;
  use_llvm = true;
  demote_dense_struct_fors = true;
  // P-Sparse-Listgen-1 (forge 2026-05): default ON. fit() will additionally
  // force-on for spirv archs; LLVM backends are unaffected because the
  // arch_uses_spirv() gate in offload.cpp short-circuits this flag.
  spirv_skip_intermediate_listgen = true;
  // §16.12 (S2): default OFF; opt-in via ti.init(spirv_listgen_subgroup_ballot=True).
  spirv_listgen_subgroup_ballot = false;
  // §16.13 (S3): default OFF; opt-in via ti.init(listgen_static_grid_dim=True).
  listgen_static_grid_dim = false;
  // VS-2: default OFF; opt-in via ti.init(vulkan_dispatch_cache=True).
  vulkan_dispatch_cache = false;
  // VS-3: default ON after the 3-run VS-1/VS-2/VS-3 Vulkan matrix.
  vulkan_listgen_reuse = true;
  // G-4: adaptive downgrade is opt-in until the cross-backend matrix proves
  // stable positive results. Keep defaults behavior-compatible with current
  // CS-3 / VS-3.
  cuda_listgen_reuse_adaptive = false;
  vulkan_listgen_reuse_adaptive = false;
  // G-6: SPIR-V adaptive optimizer is opt-in; default keeps the existing
  // global pass chain byte-identical.
  spirv_adaptive_opt = false;
  spirv_adaptive_opt_threshold = 64;
  // VS-4: default OFF; opt-in diagnostics via ti.init(vulkan_spv_stats=True).
  vulkan_spv_stats = false;
  vulkan_spv_stats_filter = "sparse";
  vulkan_spv_stats_capacity = 4096;
  vulkan_spv_stats_to_stderr = false;
  advanced_optimization = true;
  constant_folding = true;
  max_vector_width = 8;
  debug = false;
  cfg_optimization = true;
  check_out_of_bound = false;
  serial_schedule = false;
  simplify_before_lower_access = true;
  lower_access = true;
  simplify_after_lower_access = true;
  move_loop_invariant_outside_if = false;
  default_fp = PrimitiveType::f32;
  default_ip = PrimitiveType::i32;
  default_up = PrimitiveType::u32;
  verbose_kernel_launches = false;
  kernel_profiler = false;
  default_cpu_block_dim = 32;
  cpu_block_dim_adaptive = true;
  default_gpu_block_dim = 128;
  gpu_max_reg = 0;  // 0 means using the default value from the CUDA driver.
  verbose = true;
  fast_math = true;
  flatten_if = false;
  make_thread_local = true;
  make_block_local = true;
  detect_read_only = true;
  real_matrix_scalarize = true;
  force_scalarize_matrix = false;
  half2_vectorization = false;
  make_cpu_multithreading_loop = true;

  saturating_grid_dim = 0;
  max_block_dim = 0;
  cpu_max_num_threads = std::thread::hardware_concurrency();
  random_seed = 0;

  // LLVM backend options:
  print_struct_llvm_ir = false;
  print_kernel_llvm_ir = false;
  print_kernel_asm = false;
  print_kernel_amdgcn = false;
  print_kernel_llvm_ir_optimized = false;

  // CUDA/AMDGPU backend options:
  device_memory_GB = 1;  // by default, preallocate 1 GB GPU memory
  device_memory_fraction = 0.0;
  // P-Sparse-Mem-1: 0 = fall back to device_memory_GB (legacy behavior).
  cuda_sparse_pool_size_GB = 0.0;
  // P-Sparse-Mem-3 (2026-05-06): floor removed (set to 0). Phase 1-D's
  // dynamic chunk_elements + auto-hint from num_cells_per_container
  // provides exact worst-case sizing, making a defensive floor unnecessary.
  // Users who want a safety net can still set this explicitly.
  cuda_sparse_pool_size_floor_MiB = 0;
  hash_snode_experimental = true;
  hash_snode_default_load_factor = 0.5;
  hash_snode_active_list = false;
  hash_snode_diagnostics = false;
  hash_snode_compact_child_pool = false;
}

void CompileConfig::fit() {
  if (debug) {
    // TODO: allow users to run in debug mode without out-of-bound checks
    check_out_of_bound = true;
  }
  if (arch_uses_spirv(arch)) {
    demote_dense_struct_fors = true;
    // P-Sparse-Listgen-1: spirv backend's intermediate listgen tasks are
    // dispatch overhead with no functional effect; force-enable the skip.
    spirv_skip_intermediate_listgen = true;
  }
  offline_cache::disable_offline_cache_if_needed(this);
}

}  // namespace taichi::lang
