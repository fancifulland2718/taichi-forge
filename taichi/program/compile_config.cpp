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
  // P-Sparse-Mem-3 (2026-05-05): default 128 MiB floor for auto-sized
  // sparse pool. Lowered from a previous hardcoded 256 MiB after empirical
  // validation on the mpm-shaped 64^3 reference workload.
  cuda_sparse_pool_size_floor_MiB = 128;
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
