#include "taichi/runtime/llvm/llvm_runtime_executor.h"

#include "taichi/rhi/common/host_memory_pool.h"
#include "taichi/runtime/llvm/list_manager_constants.h"
#include "taichi/runtime/llvm/llvm_offline_cache.h"
#include "taichi/runtime/llvm/sparse_tree_statistics.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <unordered_map>
#include "taichi/rhi/cpu/cpu_device.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/platform/cuda/detect_cuda.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/llvm/device_memory_pool.h"

#include <array>
#if defined(TI_WITH_CUDA)
#include "taichi/rhi/cuda/cuda_context.h"
#endif

#include "taichi/platform/amdgpu/detect_amdgpu.h"
#include "taichi/rhi/amdgpu/amdgpu_driver.h"
#include "taichi/rhi/amdgpu/amdgpu_device.h"
#if defined(TI_WITH_AMDGPU)
#include "taichi/rhi/amdgpu/amdgpu_context.h"
#endif

namespace taichi::lang {
namespace {
void assert_failed_host(const char *msg) {
  TI_ERROR("Assertion failure: {}", msg);
}

void *host_allocate_aligned(HostMemoryPool *memory_pool,
                            std::size_t size,
                            std::size_t alignment) {
  return memory_pool->allocate(size, alignment);
}

void host_release(HostMemoryPool *memory_pool,
                  std::size_t size,
                  void *ptr) {
  memory_pool->release(size, ptr);
}

std::size_t direct_ambient_size(SNodeType type,
                                std::size_t cell_size_bytes,
                                std::size_t chunk_size) {
  if (type == SNodeType::pointer) {
    return std::max(cell_size_bytes, sizeof(int32));
  }
  if (type == SNodeType::dynamic) {
    return sizeof(void *) + cell_size_bytes * chunk_size;
  }
  if (type == SNodeType::hash) {
    return cell_size_bytes;
  }
  return 0;
}

}  // namespace

LlvmRuntimeExecutor::LlvmRuntimeExecutor(CompileConfig &config,
                                         KernelProfilerBase *profiler)
    : config_(config) {
  if (config.arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    if (!is_cuda_api_available()) {
      TI_WARN("No CUDA driver API detected.");
      config.arch = host_arch();
    } else if (!CUDAContext::get_instance().detected()) {
      TI_WARN("No CUDA device detected.");
      config.arch = host_arch();
    } else {
      // CUDA runtime created successfully
      use_device_memory_pool_ = CUDAContext::get_instance().supports_mem_pool();
      if (!use_device_memory_pool_ &&
          config.cuda_pointer_deterministic_slot) {
        TI_WARN(
            "cuda_pointer_deterministic_slot requires CUDA device-memory-pool "
            "support; falling back to NodeManager allocation.");
        config.cuda_pointer_deterministic_slot = false;
      }
    }
#else
    TI_WARN("Taichi is not compiled with CUDA.");
    config.arch = host_arch();
#endif

    if (config.arch != Arch::cuda) {
      TI_WARN("Falling back to {}.", arch_name(host_arch()));
    }
  } else if (config.arch == Arch::amdgpu) {
#if defined(TI_WITH_AMDGPU)
    if (!is_rocm_api_available()) {
      TI_WARN("No AMDGPU ROCm API detected.");
      config.arch = host_arch();
    } else if (!AMDGPUContext::get_instance().detected()) {
      TI_WARN("No AMDGPU device detected.");
      config.arch = host_arch();
    } else {
      // AMDGPU runtime created successfully
    }
#else
    TI_WARN("Taichi is not compiled with AMDGPU.");
    config.arch = host_arch();
#endif
  }

  if (config.kernel_profiler) {
    profiler_ = profiler;
  }

  snode_tree_buffer_manager_ = std::make_unique<SNodeTreeBufferManager>(this);

  llvm_runtime_ = nullptr;

  if (arch_is_cpu(config.arch)) {
    config.max_block_dim = 1024;
    device_ = std::make_shared<cpu::CpuDevice>();

  }
#if defined(TI_WITH_CUDA)
  else if (config.arch == Arch::cuda) {
    auto context_guard = CUDAContext::get_instance().get_guard();
    int num_SMs{1};
    CUDADriver::get_instance().device_get_attribute(
        &num_SMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, nullptr);
    int query_max_block_dim{1024};
    CUDADriver::get_instance().device_get_attribute(
        &query_max_block_dim, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, nullptr);
    int version{0};
    CUDADriver::get_instance().driver_get_version(&version);
    int query_max_block_per_sm{16};
    if (version >= 11000) {
      // query this attribute only when CUDA version is above 11.0
      CUDADriver::get_instance().device_get_attribute(
          &query_max_block_per_sm,
          CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, nullptr);
    }

    if (config.max_block_dim == 0) {
      config.max_block_dim = query_max_block_dim;
    }

    if (config.saturating_grid_dim == 0) {
      if (version >= 11000) {
        TI_TRACE("CUDA max blocks per SM = {}", query_max_block_per_sm);
      }
      config.saturating_grid_dim = num_SMs * query_max_block_per_sm * 2;
    }
    if (config.kernel_profiler) {
      CUDAContext::get_instance().set_profiler(profiler);
    } else {
      CUDAContext::get_instance().set_profiler(nullptr);
    }
    CUDAContext::get_instance().set_debug(config.debug);
    if (config.cuda_stack_limit != 0) {
      CUDADriver::get_instance().context_set_limit(CU_LIMIT_STACK_SIZE,
                                                   config.cuda_stack_limit);
    }
    device_ = std::make_shared<cuda::CudaDevice>();
  }
#endif
#if defined(TI_WITH_AMDGPU)
  else if (config.arch == Arch::amdgpu) {
    int num_workgroups{1};
    AMDGPUDriver::get_instance().device_get_attribute(
        &num_workgroups, HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0);
    int query_max_block_dim{1024};
    AMDGPUDriver::get_instance().device_get_attribute(
        &query_max_block_dim, HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, 0);
    // magic number 32
    // I didn't find the relevant parameter to limit the max block num per CU
    // So ....
    int query_max_block_per_cu{32};
    if (config.max_block_dim == 0) {
      config.max_block_dim = query_max_block_dim;
    }
    if (config.saturating_grid_dim == 0) {
      config.saturating_grid_dim = num_workgroups * query_max_block_per_cu * 2;
    }
    if (config.kernel_profiler) {
      AMDGPUContext::get_instance().set_profiler(profiler);
    } else {
      AMDGPUContext::get_instance().set_profiler(nullptr);
    }
    AMDGPUContext::get_instance().set_debug(config.debug);
    device_ = std::make_shared<amdgpu::AmdgpuDevice>();
  }
#endif
#ifdef TI_WITH_DX12
  else if (config.arch == Arch::dx12) {
    // FIXME: add dx12 device.
    // FIXME: set value based on DX12.
    config.max_block_dim = 1024;
    device_ = std::make_shared<cpu::CpuDevice>();
  }
#endif
  else {
    TI_NOT_IMPLEMENTED
  }
  llvm_context_ = std::make_unique<TaichiLLVMContext>(
      config_, arch_is_cpu(config.arch) ? host_arch() : config.arch);
  jit_session_ = JITSession::create(llvm_context_.get(), config, config.arch);
  init_runtime_jit_module(llvm_context_->clone_runtime_module());
}

TaichiLLVMContext *LlvmRuntimeExecutor::get_llvm_context() {
  return llvm_context_.get();
}

JITModule *LlvmRuntimeExecutor::create_jit_module(
    std::unique_ptr<llvm::Module> module) {
  return jit_session_->add_module(std::move(module));
}

bool LlvmRuntimeExecutor::remove_jit_module(JITModule *module) {
  TI_ASSERT(module != nullptr);
  TI_ASSERT(module != runtime_jit_module_);
  return jit_session_->remove_module(module);
}

JITModule *LlvmRuntimeExecutor::get_runtime_jit_module() {
  return runtime_jit_module_;
}

void LlvmRuntimeExecutor::print_list_manager_info(void *list_manager,
                                                  uint64 *result_buffer) {
  auto list_manager_len = runtime_query<int32>("ListManager_get_num_elements",
                                               result_buffer, list_manager);

  auto element_size = runtime_query<int32>("ListManager_get_element_size",
                                           result_buffer, list_manager);

  auto elements_per_chunk =
      runtime_query<int32>("ListManager_get_max_num_elements_per_chunk",
                           result_buffer, list_manager);

  auto num_active_chunks = runtime_query<int32>(
      "ListManager_get_num_active_chunks", result_buffer, list_manager);

  auto size_MB = 1e-6f * num_active_chunks * elements_per_chunk * element_size;

  fmt::print(
      " length={:n}     {:n} chunks x [{:n} x {:n} B]  total={:.4f} MB\n",
      list_manager_len, num_active_chunks, elements_per_chunk, element_size,
      size_MB);
}

void LlvmRuntimeExecutor::synchronize() {
  if (config_.arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    auto context_guard = CUDAContext::get_instance().get_guard();
    CUDADriver::get_instance().stream_synchronize(nullptr);
#else
    TI_ERROR("No CUDA support");
#endif
  } else if (config_.arch == Arch::amdgpu) {
#if defined(TI_WITH_AMDGPU)
    AMDGPUDriver::get_instance().stream_synchronize(nullptr);
    // A better way
    // use `hipFreeAsync` to free the device kernel arg mem
    // notice: rocm version
    AMDGPUContext::get_instance().free_kernel_arg_pointer();
#else
    TI_ERROR("No AMDGPU support");
#endif
  }
  fflush(stdout);
}

uint64 LlvmRuntimeExecutor::fetch_result_uint64(int i, uint64 *result_buffer) {
  // Runtime JIT calls share this device result buffer and the legacy default
  // stream. Waiting that stream is therefore part of the buffer-reuse contract;
  // a freshly recorded event would serialize the same work with extra event
  // overhead. Ordinary kernel return values use their per-launch result buffer
  // and asynchronous copy path instead.
  synchronize();
  uint64 ret;
  if (config_.arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    auto context_guard = CUDAContext::get_instance().get_guard();
    CUDADriver::get_instance().memcpy_device_to_host(&ret, result_buffer + i,
                                                     sizeof(uint64));
#else
    TI_NOT_IMPLEMENTED;
#endif
  } else if (config_.arch == Arch::amdgpu) {
#if defined(TI_WITH_AMDGPU)
    AMDGPUDriver::get_instance().memcpy_device_to_host(&ret, result_buffer + i,
                                                       sizeof(uint64));
#else
    TI_NOT_IMPLEMENTED;
#endif
  } else {
    ret = result_buffer[i];
  }
  return ret;
}

std::size_t LlvmRuntimeExecutor::get_snode_num_dynamically_allocated(
    SNode *snode,
    uint64 *result_buffer) {
  TI_ASSERT(arch_uses_llvm(config_.arch));

  auto node_allocator =
      runtime_query<void *>("LLVMRuntime_get_node_allocators", result_buffer,
                            llvm_runtime_, snode->id);
  auto deterministic_capacity = runtime_query<int32>(
      "NodeManager_get_deterministic_capacity", result_buffer,
      node_allocator);
  if (deterministic_capacity > 0) {
    return (std::size_t)runtime_query<int32>(
        "NodeManager_get_deterministic_peak", result_buffer, node_allocator);
  }
  auto data_list = runtime_query<void *>("NodeManager_get_data_list",
                                         result_buffer, node_allocator);

  return (std::size_t)runtime_query<int32>("ListManager_get_num_elements",
                                           result_buffer, data_list);
}

void LlvmRuntimeExecutor::begin_cpu_sparse_listgen_work() {
  TI_ASSERT(arch_is_cpu(config_.arch));
  get_runtime_jit_module()->call<void *>("runtime_sparse_listgen_work_begin",
                                        llvm_runtime_);
}

LlvmRuntimeExecutor::CpuSparseListgenWork
LlvmRuntimeExecutor::read_cpu_sparse_listgen_work() {
  TI_ASSERT(arch_is_cpu(config_.arch));
  std::uint64_t result[5] = {0, 0, 0, 0, 0};
  get_runtime_jit_module()->call<void *, std::uint64_t *>(
      "runtime_sparse_listgen_work_read", llvm_runtime_, result);
  return {
      result[0] != 0,
      result[1],
      result[2],
      result[3] == 2,
      result[4] != 0,
  };
}

SparseSNodeTreeMemoryStatistics
LlvmRuntimeExecutor::get_snode_tree_memory_statistics(
    SNodeTree *snode_tree,
    uint64 *result_buffer) {
  TI_ASSERT(snode_tree != nullptr);
  TI_ASSERT(result_buffer != nullptr);

  std::vector<SNode *> snodes;
  bool all_dense = config_.demote_dense_struct_fors;
  std::uint64_t direct_ambient_bytes = 0;
  std::function<void(SNode *)> collect = [&](SNode *snode) {
    snodes.push_back(snode);
    if (snode->type != SNodeType::dense &&
        snode->type != SNodeType::place &&
        snode->type != SNodeType::root) {
      all_dense = false;
    }
    direct_ambient_bytes += direct_ambient_size(
        snode->type, snode->cell_size_bytes, snode->chunk_size);
    for (const auto &child : snode->ch) {
      collect(child.get());
    }
  };
  collect(snode_tree->root());

  auto *runtime_jit = get_runtime_jit_module();
  runtime_jit->call<uint64 *>("runtime_sparse_tree_statistics_reset",
                              result_buffer);
  for (SNode *snode : snodes) {
    runtime_jit->call<void *, int, int, uint64 *>(
        "runtime_sparse_snode_statistics_collect", llvm_runtime_, snode->id,
        all_dense ? 0 : 1, result_buffer);
  }
  synchronize();

  std::array<uint64, kLlvmSparseTreeStatisticCount> raw{};
  if (config_.arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    auto context_guard = CUDAContext::get_instance().get_guard();
    CUDADriver::get_instance().memcpy_device_to_host(
        raw.data(), result_buffer, sizeof(raw));
#else
    TI_NOT_IMPLEMENTED;
#endif
  } else if (config_.arch == Arch::amdgpu) {
#if defined(TI_WITH_AMDGPU)
    AMDGPUDriver::get_instance().memcpy_device_to_host(
        raw.data(), result_buffer, sizeof(raw));
#else
    TI_NOT_IMPLEMENTED;
#endif
  } else {
    std::memcpy(raw.data(), result_buffer, sizeof(raw));
  }

  std::uint64_t shared_listgen_workspace_bytes = 0;
  if (arch_is_cpu(config_.arch)) {
    runtime_jit->call<void *, std::uint64_t *>(
        "runtime_cpu_parallel_listgen_workspace_statistics", llvm_runtime_,
        &shared_listgen_workspace_bytes);
  }

  const auto exact = [](std::uint64_t value) {
    return RuntimeOptionalCounter{value, true};
  };
  SparseSNodeTreeMemoryStatistics result;
  const std::uint64_t root_bytes =
      snode_tree_buffer_manager_->get_size(snode_tree->id());
  const auto pool_it = sparse_tree_pool_sizes_.find(snode_tree->id());
  const std::uint64_t sparse_pool_bytes =
      pool_it == sparse_tree_pool_sizes_.end() ? 0 : pool_it->second;
  result.root_reserved_bytes = exact(root_bytes);
  result.sparse_pool_reserved_bytes = exact(sparse_pool_bytes);
  result.tree_owned_reserved_bytes = exact(root_bytes + sparse_pool_bytes);
  result.runtime_metadata_requested_bytes =
      exact(raw[kLlvmSparseRuntimeMetadataRequestedBytes]);
  result.direct_ambient_requested_bytes = exact(direct_ambient_bytes);
  result.allocator_payload_reserved_bytes =
      exact(raw[kLlvmSparseAllocatorPayloadReservedBytes]);
  result.allocator_payload_used_bytes =
      exact(raw[kLlvmSparseAllocatorPayloadUsedBytes]);
  result.allocator_bookkeeping_reserved_bytes =
      exact(raw[kLlvmSparseAllocatorBookkeepingReservedBytes]);
  result.active_list_reserved_bytes =
      exact(raw[kLlvmSparseActiveListReservedBytes]);
  result.active_list_used_bytes =
      exact(raw[kLlvmSparseActiveListUsedBytes]);
  result.allocator_in_use_elements =
      exact(raw[kLlvmSparseAllocatorInUseElements]);
  result.allocator_free_elements =
      exact(raw[kLlvmSparseAllocatorFreeElements]);
  result.allocator_recycled_elements =
      exact(raw[kLlvmSparseAllocatorRecycledElements]);
  result.shared_listgen_workspace_reserved_bytes =
      exact(shared_listgen_workspace_bytes);
  result.tree_owned_scope =
      config_.arch == Arch::cuda
          ? "exclusive_root_and_auto_sized_sparse_pool"
          : "exclusive_root_only";
  result.runtime_resource_scope =
      config_.arch == Arch::cuda
          ? "tree_logical_resources_in_cuda_runtime_pool"
          : "tree_logical_resources_in_program_reuse_pool";
  result.shared_listgen_workspace_scope =
      arch_is_cpu(config_.arch) ? "program_shared_capacity_not_tree_owned"
                                : "not_used";
  return result;
}

void LlvmRuntimeExecutor::reset_hash_snode_probe_stats(uint64 *result_buffer) {
  TI_ASSERT(arch_uses_llvm(config_.arch));
  auto *runtime_jit_module = get_runtime_jit_module();
  runtime_jit_module->call<void *>("runtime_hash_probe_stats_reset",
                                   llvm_runtime_);
  synchronize();
}

std::vector<int64> LlvmRuntimeExecutor::get_hash_snode_probe_stats(
    uint64 *result_buffer) {
  TI_ASSERT(arch_uses_llvm(config_.arch));
  constexpr int kNumHashProbeStats = 6;
  std::vector<int64> stats;
  stats.reserve(kNumHashProbeStats);
  auto *runtime_jit_module = get_runtime_jit_module();
  for (int i = 0; i < kNumHashProbeStats; i++) {
    runtime_jit_module->call<void *, int>("runtime_hash_probe_stats_get",
                                          llvm_runtime_, i);
    stats.push_back((int64)fetch_result<int32>(
        taichi_result_buffer_runtime_query_id, result_buffer));
  }
  return stats;
}

void LlvmRuntimeExecutor::check_runtime_error(uint64 *result_buffer) {
  synchronize();
  auto *runtime_jit_module = get_runtime_jit_module();
  runtime_jit_module->call<void *>("runtime_retrieve_and_reset_error_code",
                                   llvm_runtime_);
  auto error_code =
      fetch_result<int64>(taichi_result_buffer_error_id, result_buffer);

  if (error_code) {
    std::string error_message_template;

    // Here we fetch the error_message_template char by char.
    // This is not efficient, but fortunately we only need to do this when an
    // assertion fails. Note that we may not have unified memory here, so using
    // "fetch_result" that works across device/host memory is necessary.
    for (int i = 0;; i++) {
      runtime_jit_module->call<void *>("runtime_retrieve_error_message",
                                       llvm_runtime_, i);
      auto c = fetch_result<char>(taichi_result_buffer_error_id, result_buffer);
      error_message_template += c;
      if (c == '\0') {
        break;
      }
    }

    if (error_code == 1) {
      const auto error_message_formatted = format_error_message(
          error_message_template,
          [runtime_jit_module, result_buffer, this](int argument_id) {
            runtime_jit_module->call<void *>(
                "runtime_retrieve_error_message_argument", llvm_runtime_,
                argument_id);
            return fetch_result<uint64>(taichi_result_buffer_error_id,
                                        result_buffer);
          });
      throw TaichiAssertionError(error_message_formatted);
    } else {
      TI_NOT_IMPLEMENTED
    }
  }
}

void LlvmRuntimeExecutor::print_memory_profiler_info(
    std::vector<std::unique_ptr<SNodeTree>> &snode_trees_,
    uint64 *result_buffer) {
  TI_ASSERT(arch_uses_llvm(config_.arch));

  fmt::print("\n[Memory Profiler]\n");

  std::locale::global(std::locale("en_US.UTF-8"));
  // So that thousand separators are added to "{:n}" slots in fmtlib.
  // E.g., 10000 is printed as "10,000".
  // TODO: is there a way to set locale only locally in this function?

  std::function<void(SNode *, int)> visit = [&](SNode *snode, int depth) {
    auto element_list =
        runtime_query<void *>("LLVMRuntime_get_element_lists", result_buffer,
                              llvm_runtime_, snode->id);

    if (snode->type != SNodeType::place) {
      fmt::print("SNode {:10}\n", snode->get_node_type_name_hinted());

      if (element_list) {
        fmt::print("  active element list:");
        print_list_manager_info(element_list, result_buffer);

        auto node_allocator =
            runtime_query<void *>("LLVMRuntime_get_node_allocators",
                                  result_buffer, llvm_runtime_, snode->id);

        if (node_allocator) {
          auto free_list = runtime_query<void *>("NodeManager_get_free_list",
                                                 result_buffer, node_allocator);
          auto recycled_list = runtime_query<void *>(
              "NodeManager_get_recycled_list", result_buffer, node_allocator);

          auto free_list_len = runtime_query<int32>(
              "ListManager_get_num_elements", result_buffer, free_list);

          auto recycled_list_len = runtime_query<int32>(
              "ListManager_get_num_elements", result_buffer, recycled_list);

          auto free_list_used = runtime_query<int32>(
              "NodeManager_get_free_list_used", result_buffer, node_allocator);

          auto data_list = runtime_query<void *>("NodeManager_get_data_list",
                                                 result_buffer, node_allocator);
          fmt::print("  data list:          ");
          print_list_manager_info(data_list, result_buffer);

          fmt::print(
              "  Allocated elements={:n}; free list length={:n}; recycled list "
              "length={:n}\n",
              free_list_used, free_list_len, recycled_list_len);
        }
      }
    }
    for (const auto &ch : snode->ch) {
      visit(ch.get(), depth + 1);
    }
  };

  for (auto &a : snode_trees_) {
    visit(a->root(), /*depth=*/0);
  }

  auto total_requested_memory = runtime_query<std::size_t>(
      "LLVMRuntime_get_total_requested_memory", result_buffer, llvm_runtime_);

  fmt::print(
      "Total requested dynamic memory (excluding alignment padding): {:n} B\n",
      total_requested_memory);
}

DevicePtr LlvmRuntimeExecutor::get_snode_tree_device_ptr(int tree_id) {
  DeviceAllocation tree_alloc = snode_tree_allocs_[tree_id];
  return tree_alloc.get_ptr();
}

void LlvmRuntimeExecutor::initialize_llvm_runtime_snodes(
    const LlvmOfflineCache::FieldCacheData &field_cache_data,
    uint64 *result_buffer) {
  auto *const runtime_jit = get_runtime_jit_module();
  // By the time this creator is called, "this" is already destroyed.
  // Therefore it is necessary to capture members by values.
  size_t root_size = field_cache_data.root_size;
  const auto snode_metas = field_cache_data.snode_metas;
  const int tree_id = field_cache_data.tree_id;
  const int root_id = field_cache_data.root_id;

  bool all_dense = config_.demote_dense_struct_fors;
  DeviceAllocationUnique *sparse_tree_pool_alloc = nullptr;
  for (size_t i = 0; i < snode_metas.size(); i++) {
    if (snode_metas[i].type != SNodeType::dense &&
        snode_metas[i].type != SNodeType::place &&
        snode_metas[i].type != SNodeType::root) {
      all_dense = false;
      break;
    }
  }
  struct NodePoolGeometry {
    std::size_t node_size;
    std::size_t chunk_elements;
    std::size_t data_chunks;
  };
  std::unordered_map<int, NodePoolGeometry> node_pool_geometries;
  const bool use_cuda_auto_pool_geometry =
      config_.arch == Arch::cuda && use_device_memory_pool() &&
      config_.cuda_sparse_pool_auto_size &&
      config_.device_memory_fraction == 0 &&
      config_.cuda_sparse_pool_size_GB == 0 && !all_dense;
  if (use_cuda_auto_pool_geometry) {
    constexpr std::size_t kNodeMgrChunkElementsDefault = 16UL * 1024;
    constexpr std::size_t kNodeMgrMaxChunkBytes = 128UL << 20;
    constexpr int kHintHeadroomChunks = 1;
    for (const auto &meta : snode_metas) {
      if (!is_gc_able(meta.type)) {
        continue;
      }
      std::size_t element_size = meta.cell_size_bytes;
      if (meta.type == SNodeType::pointer) {
        element_size = std::max(element_size, sizeof(int32));
      }
      if (element_size == 0) {
        continue;
      }
      const std::size_t node_size =
          meta.type == SNodeType::pointer
              ? element_size
              : sizeof(void *) + element_size * meta.chunk_size;
      std::size_t chunk_elements = kNodeMgrChunkElementsDefault;
      while (chunk_elements > 1 &&
             chunk_elements * node_size > kNodeMgrMaxChunkBytes) {
        chunk_elements /= 2;
      }
      if (meta.num_cells_per_container > 0) {
        const std::size_t desired =
            std::size_t(meta.num_cells_per_container) * 2;
        std::size_t tight = 1024;
        while (tight < desired) {
          tight *= 2;
        }
        if (tight < chunk_elements) {
          chunk_elements = tight;
        }
      }
      int64_t effective = meta.vk_max_active_hint;
      if (effective <= 0) {
        effective = meta.total_num_cells_from_root;
      }
      std::size_t data_chunks = 3;
      if (effective > 0) {
        const int64_t lower_bound =
            meta.num_cells_per_container > 0
                ? meta.num_cells_per_container
                : 1;
        const std::size_t needed_chunks =
            (std::size_t(std::max<int64_t>(effective, lower_bound)) +
             chunk_elements - 1) /
            chunk_elements;
        data_chunks =
            std::max<std::size_t>(needed_chunks, 1) +
            std::size_t(kHintHeadroomChunks);
      }
      node_pool_geometries.emplace(
          meta.id,
          NodePoolGeometry{node_size, chunk_elements, data_chunks});
    }
  }
  const std::size_t element_list_snode_count =
      std::count_if(snode_metas.begin(), snode_metas.end(), [](const auto &meta) {
        return meta.type != SNodeType::place;
      });
  // A non-hash Element covers at most 1024 cells in one child container, so
  // its list-entry upper bound is:
  //   expected active parent cells * ceil(cells_per_container / 1024).
  // Propagate explicit pointer/hash hints through dense descendants. Expected
  // activity selects chunk granularity; separate capacity/budget propagation
  // below keeps CUDA list storage aligned with its allocator pool geometry.
  std::unordered_map<int, std::uint64_t> expected_active_cells;
  std::unordered_map<int, std::uint64_t> capacity_active_cells;
  std::unordered_map<int, std::uint64_t> budget_active_cells;
  std::unordered_map<int, std::size_t> element_list_chunk_elements;
  std::unordered_map<int, std::uint64_t> element_list_pool_entries;
  for (const auto &meta : snode_metas) {
    const std::uint64_t total_cells =
        meta.total_num_cells_from_root > 0
            ? std::uint64_t(meta.total_num_cells_from_root)
            : 1;
    const std::uint64_t cells_per_container =
        meta.num_cells_per_container > 0
            ? std::uint64_t(meta.num_cells_per_container)
            : 1;
    std::uint64_t parent_active_cells = 1;
    std::uint64_t parent_capacity_cells = 1;
    std::uint64_t parent_budget_cells = 1;
    if (meta.parent_id >= 0) {
      auto parent = expected_active_cells.find(meta.parent_id);
      parent_active_cells = parent == expected_active_cells.end()
                                ? total_cells
                                : parent->second;
      auto capacity_parent = capacity_active_cells.find(meta.parent_id);
      parent_capacity_cells =
          capacity_parent == capacity_active_cells.end()
              ? total_cells
              : capacity_parent->second;
      auto budget_parent = budget_active_cells.find(meta.parent_id);
      parent_budget_cells =
          budget_parent == budget_active_cells.end()
              ? total_cells
              : budget_parent->second;
    }

    std::uint64_t active_cells = total_cells;
    std::uint64_t capacity_cells = total_cells;
    std::uint64_t budget_cells = total_cells;
    if (meta.type == SNodeType::root) {
      active_cells = 1;
      capacity_cells = 1;
      budget_cells = 1;
    } else if (parent_active_cells <=
               total_cells / cells_per_container) {
      active_cells = parent_active_cells * cells_per_container;
    }
    if (meta.type != SNodeType::root &&
        parent_capacity_cells <= total_cells / cells_per_container) {
      capacity_cells = parent_capacity_cells * cells_per_container;
    }
    if (meta.type != SNodeType::root &&
        parent_budget_cells <= total_cells / cells_per_container) {
      budget_cells = parent_budget_cells * cells_per_container;
    }
    if (meta.vk_max_active_hint > 0) {
      active_cells = std::min(
          active_cells, std::uint64_t(meta.vk_max_active_hint));
    }
    if (meta.type == SNodeType::hash && meta.vk_max_active_hint > 0) {
      // Hash stores its derived table capacity in vk_max_active_hint. Pointer
      // uses the same field only as a CUDA sizing hint, not a hard limit.
      capacity_cells = std::min(
          capacity_cells, std::uint64_t(meta.vk_max_active_hint));
      budget_cells = capacity_cells;
    }
    if (meta.type == SNodeType::hash &&
        meta.hash_expected_active_hint > 0) {
      active_cells = std::min(
          active_cells, std::uint64_t(meta.hash_expected_active_hint));
    }
    auto geometry = node_pool_geometries.find(meta.id);
    if (geometry != node_pool_geometries.end()) {
      const std::uint64_t physical_cells =
          std::uint64_t(geometry->second.data_chunks) *
          std::uint64_t(geometry->second.chunk_elements);
      budget_cells =
          std::min(capacity_cells, std::max(active_cells, physical_cells));
    }
    expected_active_cells[meta.id] = active_cells;
    capacity_active_cells[meta.id] = capacity_cells;
    budget_active_cells[meta.id] = budget_cells;

    std::uint64_t expected_entries = 1;
    std::uint64_t capacity_entries = 1;
    std::uint64_t budget_entries = 1;
    if (meta.type == SNodeType::hash) {
      expected_entries = active_cells;
      // Hash expected_active selects a useful chunk granularity, while the
      // derived table capacity remains the hard upper bound for live slots.
      capacity_entries = capacity_cells;
      budget_entries = budget_cells;
    } else if (meta.type != SNodeType::root) {
      const std::uint64_t slices_per_container =
          (cells_per_container - 1) / taichi_listgen_max_element_size + 1;
      if (parent_active_cells >
          std::numeric_limits<std::uint64_t>::max() /
              slices_per_container) {
        expected_entries = std::numeric_limits<std::uint64_t>::max();
      } else {
        expected_entries = parent_active_cells * slices_per_container;
      }
      if (parent_capacity_cells >
          std::numeric_limits<std::uint64_t>::max() /
              slices_per_container) {
        capacity_entries = std::numeric_limits<std::uint64_t>::max();
      } else {
        capacity_entries = parent_capacity_cells * slices_per_container;
      }
      if (parent_budget_cells >
          std::numeric_limits<std::uint64_t>::max() /
              slices_per_container) {
        budget_entries = std::numeric_limits<std::uint64_t>::max();
      } else {
        budget_entries = parent_budget_cells * slices_per_container;
      }
    }
    const std::uint64_t list_entries =
        std::min<std::uint64_t>(expected_entries,
                                kLlvmElementListMaxChunkElements);
    const std::size_t target = std::size_t(std::max<std::uint64_t>(
        kLlvmElementListMinChunkElements,
        std::min<std::uint64_t>(list_entries,
                                kLlvmElementListMaxChunkElements)));
    std::size_t chunk_elements = kLlvmElementListMinChunkElements;
    while (chunk_elements < target) {
      chunk_elements *= 2;
    }
    element_list_chunk_elements[meta.id] = chunk_elements;
    element_list_pool_entries[meta.id] =
        std::max<std::uint64_t>(
            std::min(capacity_entries, budget_entries), 1);
  }

  // CUDA's auto-sized pool is fixed for the lifetime of one SNodeTree. Budget
  // every traversal list from the same expected-plus-one-allocator-chunk
  // physical bound as NodeManager instead of hiding all future list payload
  // behind a fixed global headroom. Hash lists use the derived table capacity,
  // which is already their hard live-slot bound. This keeps pointer hints as
  // sizing hints and prevents traversal storage from becoming the tighter
  // implicit limit.
  std::size_t element_list_pool_budget_bytes = 0;
  if (use_cuda_auto_pool_geometry) {
    constexpr std::size_t kElementBytes =
        sizeof(std::uint64_t) +
        (2 + taichi_max_num_indices) * sizeof(std::int32_t);
    for (const auto &meta : snode_metas) {
      if (meta.type == SNodeType::place) {
        continue;
      }
      const std::size_t chunk_elements =
          element_list_chunk_elements.at(meta.id);
      const std::uint64_t pool_entries =
          element_list_pool_entries.at(meta.id);
      const std::uint64_t list_manager_capacity =
          std::min<std::uint64_t>(
              std::numeric_limits<std::int32_t>::max(),
              std::uint64_t(kLlvmListManagerMaxNumChunks) *
                  std::uint64_t(chunk_elements));
      const std::uint64_t representable_capacity =
          std::min(pool_entries, list_manager_capacity);
      if (pool_entries > representable_capacity) {
        TI_WARN(
            "SNode {} traversal-list capacity {} exceeds ListManager's "
            "representable capacity {}; budget is clamped to the runtime "
            "limit. Reduce the logical domain or provide a tighter active "
            "capacity hint.",
            meta.id, pool_entries, list_manager_capacity);
      }
      const std::uint64_t budget_entries =
          std::min(pool_entries, representable_capacity);
      const std::size_t chunks =
          std::size_t((budget_entries + chunk_elements - 1) / chunk_elements);
      const std::size_t payload_bytes =
          chunks * chunk_elements * kElementBytes;
      const std::size_t directory_bytes =
          llvm_list_manager_directory_pages_for_chunks(chunks) *
          kLlvmListManagerDirectoryAllocationBudgetBytes;
      element_list_pool_budget_bytes += payload_bytes + directory_bytes;
    }
  }

  if (config_.arch == Arch::cuda && use_device_memory_pool() && !all_dense) {
    // P-Sparse-Mem-2-A v2 (2026-05-05): when opted-in via
    // `cuda_sparse_pool_auto_size`, mirror the device-side `NodeManager`
    // geometry exactly (runtime.cpp:1026-1031 and NodeManager ctor) to
    // estimate first-activation footprint accurately. The previous
    // `cell_size_bytes * 1024` heuristic ignored chunk halving and
    // underestimated by 10x+ for SNode shapes with large container size,
    // causing silent OOM in `allocate_from_reserved_memory` even with
    // `device_memory_GB` raised (the cap was the actual bug surface).
    //
    // Explicit fixed-pool settings preserve the legacy
    // `pool_size = device_memory_GB * 1GiB` path. The default auto path uses
    // the SNode and traversal geometries computed here.
    std::size_t override_size = 0;
    // Phase 1: per-SNode pool entries, populated during auto-size loop
    // and consumed in the NodeAllocator init loop below.
    struct SnodePoolEntry {
      int snode_id;
      std::size_t metadata_bytes;
      std::size_t data_bytes;
      std::size_t chunk_bytes;
    };
    std::vector<SnodePoolEntry> snode_entries;
    bool do_per_snode_pool =
        config_.cuda_sparse_per_snode_pool &&
        config_.cuda_sparse_pool_auto_size &&
        config_.device_memory_fraction == 0 &&
        config_.cuda_sparse_pool_size_GB == 0;
    if (use_cuda_auto_pool_geometry) {
      // Mirror runtime.cpp constants:
      //   * runtime_NodeAllocator_initialize: chunk_num_elements = 16 * 1024
      //   * NodeManager ctor: while (chunk_elements > 1 &&
      //         chunk_elements * element_size > 128 MiB) chunk_elements /= 2
      // ListManager keeps the first 16 chunk pointers inline and grows the
      // remaining index through 1024-pointer directory pages. Each
      // NodeManager creates 3 instances (free / recycled / data). At first
      // activation, only the data_list's first chunk is touched.
      constexpr std::size_t kListManagerBytes =
          kLlvmListManagerFixedAllocationBudgetBytes;
      constexpr int kListManagersPerNodeManager = 3;
      // Baseline for misc allocations (LLVMRuntime fields, NodeManager
      // structs themselves, ambient_elements, rand_states, etc).
      constexpr std::size_t kBaselineBytes = 32UL << 20;

      std::size_t auto_size = kBaselineBytes;
      // Only structural SNodes participate in listgen/struct-for traversal.
      // A place SNode is field storage, never a list parent or child, so do not
      // reserve its ~1 MiB ListManager metadata.
      auto_size += element_list_snode_count * kListManagerBytes;
      auto_size += element_list_pool_budget_bytes;
      // Phase 1-E (2026-05): emit a budget breakdown so users can see
      // how much of the global pool is consumed by per-SNode element_list
      // metadata vs. explicit capacity payload. Useful for diagnosing
      // implicit allocation pressure on large SNode trees.
      TI_TRACE(
          "Phase-1-E element_list budget: {} SNode(s) × {:.2f} MiB = "
          "{:.2f} MiB metadata + {:.2f} MiB capacity payload",
          element_list_snode_count, kListManagerBytes / 1048576.0,
          (element_list_snode_count * kListManagerBytes) / 1048576.0,
          element_list_pool_budget_bytes / 1048576.0);
      for (size_t i = 0; i < snode_metas.size(); i++) {
        if (!is_gc_able(snode_metas[i].type))
          continue;
        auto geometry = node_pool_geometries.find(snode_metas[i].id);
        if (geometry == node_pool_geometries.end()) {
          continue;
        }
        const std::size_t node_size = geometry->second.node_size;
        const std::size_t chunk_elements =
            geometry->second.chunk_elements;
        const std::size_t data_chunks = geometry->second.data_chunks;
        std::size_t chunk_bytes = chunk_elements * node_size;
        auto_size += std::size_t(kListManagersPerNodeManager) * kListManagerBytes;
        // The dedicated pool backs all three ListManagers. data_list chunks
        // dominate, but legacy deactivate/GC can also touch recycled_list and
        // free_list index chunks. Budget those tiny index chunks explicitly so
        // all-OFF behavior remains safe when per-SNode pools are enabled.
        std::size_t index_bytes =
            std::size_t(2) * data_chunks * chunk_elements * sizeof(int32);
        const std::size_t directory_bytes =
            std::size_t(kListManagersPerNodeManager) *
            llvm_list_manager_directory_pages_for_chunks(data_chunks) *
            kLlvmListManagerDirectoryAllocationBudgetBytes;
        auto_size +=
            data_chunks * chunk_bytes + index_bytes + directory_bytes;
        // Phase 1: collect per-SNode sizing for buffer carving
        if (do_per_snode_pool) {
          snode_entries.push_back(
              {snode_metas[i].id,
               std::size_t(kListManagersPerNodeManager) * kListManagerBytes,
               data_chunks * chunk_bytes + index_bytes + directory_bytes,
               chunk_bytes});
        }
      }
      // User-tunable lower bound (defensive floor for tiny SNode trees).
      const std::size_t floor_bytes =
          std::size_t(std::max(0, config_.cuda_sparse_pool_size_floor_MiB))
          << 20;
      auto_size = std::max(auto_size, floor_bytes);

      // device_memory_GB is not a silent cap for the auto-sizer. If the
      // SNode-derived geometry asks for more, warn but keep the derived size;
      // clamping here under-sizes the preallocated pool and later manifests as
      // device-side OOM/illegal-address inside allocate_from_reserved_memory.
      std::size_t cap =
          std::size_t(config_.device_memory_GB * (1UL << 30));
      if (auto_size > cap) {
        TI_WARN(
            "cuda_sparse_pool_auto_size: SNode-derived sparse pool "
            "{:.2f} MiB exceeds device_memory_GB {:.2f} MiB; using the "
            "derived size to avoid runtime OOM in "
            "allocate_from_reserved_memory. Set cuda_sparse_pool_size_GB "
            "explicitly if you need a hard sparse-pool budget.",
            auto_size / 1048576.0, cap / 1048576.0);
      }
      override_size = auto_size;

      // Phase 1 (2026-05): carve per-SNode data regions from a single
      // buffer instead of using a monolithic global pool. The global region
      // contains the baseline, runtime/list metadata and the explicit
      // traversal-list capacity budget; each SNode's data region is a
      // sub-range of the same allocation.
      if (do_per_snode_pool && !snode_entries.empty()) {
        // Include only traversal-participating SNode element-list metadata.
        std::size_t global_region = kBaselineBytes +
                                    element_list_pool_budget_bytes +
                                    element_list_snode_count *
                                        kListManagerBytes;
        std::size_t total_buffer = global_region;
        for (const auto &e : snode_entries) {
          global_region += e.metadata_bytes;
        }
        total_buffer = global_region;
        for (const auto &e : snode_entries) {
          total_buffer += e.data_bytes;
        }
        if (total_buffer > cap) {
          TI_WARN(
              "cuda_sparse_pool_auto_size: per-SNode sparse pool {:.2f} MiB "
              "exceeds device_memory_GB {:.2f} MiB; using the derived size. "
              "Set cuda_sparse_pool_size_GB explicitly if this is too large.",
              total_buffer / 1048576.0, cap / 1048576.0);
        }
        TI_TRACE(
            "Phase-1 per-SNode pools: global={:.2f} MiB total={:.2f} MiB "
            "({} SNode data regions)",
            global_region / 1048576.0, total_buffer / 1048576.0,
            snode_entries.size());

        // Allocate one contiguous device buffer owned by this sparse
        // SNodeTree. Multiple active trees coexist through independent map
        // entries; destroy_snode_tree() erases only the synchronized tree.
        auto [pool_it, inserted] =
            sparse_tree_pool_allocs_.try_emplace(tree_id);
        TI_ASSERT(inserted);
        const bool size_inserted =
            sparse_tree_pool_sizes_.try_emplace(tree_id, total_buffer).second;
        TI_ASSERT(size_inserted);
        void *buf = preallocate_memory(total_buffer, pool_it->second);
        sparse_tree_pool_alloc = &pool_it->second;
        // Initialize runtime_memory_chunk to cover only the global region.
        auto *const runtime_jit2 = get_runtime_jit_module();
        runtime_jit2->call<void *, std::size_t, void *>(
            "runtime_initialize_memory", llvm_runtime_, global_region, buf);
        // Per-SNode dedicated pools will be set up after NodeAllocator init
        // (see per-snode-pool loop below). Store the buffer base + offset
        // for later use.
        TI_TRACE(
            "P-Sparse-Mem-2-A v2: auto-sized sparse pool = {:.2f} MiB "
            "(NodeManager-mirrored {:.2f} MiB, ceiling device_memory_GB={:.2f})",
            total_buffer / 1048576.0, auto_size / 1048576.0,
            config_.device_memory_GB);
      } else {
        TI_TRACE(
            "P-Sparse-Mem-2-A v2: auto-sized sparse pool = {:.2f} MiB "
            "(NodeManager-mirrored {:.2f} MiB, ceiling device_memory_GB={:.2f})",
            override_size / 1048576.0, auto_size / 1048576.0,
            config_.device_memory_GB);
      }
    }
    if (!do_per_snode_pool || snode_entries.empty()) {
      preallocate_runtime_memory(override_size);
    }
  }

  TI_TRACE("Allocating data structure of size {} bytes", root_size);
  std::size_t rounded_size = taichi::iroundup(root_size, taichi_page_size);

  Ptr root_buffer = snode_tree_buffer_manager_->allocate(rounded_size, tree_id,
                                                         result_buffer);
  if (config_.arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    auto context_guard = CUDAContext::get_instance().get_guard();
    CUDADriver::get_instance().memset(root_buffer, 0, rounded_size);
#else
    TI_NOT_IMPLEMENTED
#endif
  } else if (config_.arch == Arch::amdgpu) {
#if defined(TI_WITH_AMDGPU)
    AMDGPUDriver::get_instance().memset(root_buffer, 0, rounded_size);
#else
    TI_NOT_IMPLEMENTED;
#endif
  } else {
    std::memset(root_buffer, 0, rounded_size);
  }

  DeviceAllocation alloc =
      llvm_device()->import_memory(root_buffer, rounded_size);

  snode_tree_allocs_[tree_id] = alloc;

  for (const auto &meta : snode_metas) {
    runtime_jit->call<void *, int>("runtime_reset_snode_slot", llvm_runtime_,
                                   meta.id);
  }
  runtime_jit->call<void *, std::size_t, int, int, int, std::size_t, Ptr>(
      "runtime_initialize_snodes", llvm_runtime_, root_size, root_id,
      (int)snode_metas.size(), tree_id, rounded_size, root_buffer,
      element_list_chunk_elements.at(root_id), all_dense);

  if (!all_dense) {
    for (const auto &meta : snode_metas) {
      if (meta.id == root_id || meta.type == SNodeType::place) {
        continue;
      }
      runtime_jit->call<void *, int, std::size_t>(
          "runtime_initialize_snode_element_list", llvm_runtime_, meta.id,
          element_list_chunk_elements.at(meta.id));
    }
  }

  for (size_t i = 0; i < snode_metas.size(); i++) {
    if (is_gc_able(snode_metas[i].type)) {
      const auto snode_id = snode_metas[i].id;
      auto element_size = snode_metas[i].cell_size_bytes;
      if (snode_metas[i].type == SNodeType::pointer) {
        element_size = std::max(element_size, (std::size_t)sizeof(int32));
      }
      std::size_t node_size;
      if (snode_metas[i].type == SNodeType::pointer) {
        node_size = element_size;
      } else {
        node_size = sizeof(void *) + element_size * snode_metas[i].chunk_size;
      }
      // Phase 1-D: look up optimal chunk_num_elements computed during
      // auto-size; if not found, fall back to the legacy default.
      int chunk_elems = 1024 * 16;  // legacy default
      auto geometry = node_pool_geometries.find(snode_id);
      if (geometry != node_pool_geometries.end()) {
        chunk_elems = int(geometry->second.chunk_elements);
      }
      TI_TRACE("Initializing allocator for snode {} (node size {}, chunk_elems {})",
               snode_id, node_size, chunk_elems);
      runtime_jit->call<void *, int, std::size_t, int>(
          "runtime_NodeAllocator_initialize_ex", llvm_runtime_, snode_id,
          node_size, chunk_elems);
      if (config_.cuda_pointer_deterministic_pool_enabled() &&
          snode_metas[i].type == SNodeType::pointer &&
          snode_metas[i].num_cells_per_container ==
              snode_metas[i].total_num_cells_from_root) {
        TI_ASSERT(snode_metas[i].num_cells_per_container <=
                  std::numeric_limits<int>::max());
        runtime_jit->call<void *, int, int>(
            "runtime_NodeAllocator_set_deterministic_capacity", llvm_runtime_,
            snode_id, (int)snode_metas[i].num_cells_per_container);
      }
    }
  }

  // Phase 1 (2026-05): after all NodeAllocators are initialized, assign
  // each gc-able SNode its dedicated data region carved from the global
  // pool buffer using the same precomputed geometry as auto-size.
  if (config_.arch == Arch::cuda && use_device_memory_pool() &&
      config_.cuda_sparse_per_snode_pool &&
      config_.cuda_sparse_pool_auto_size &&
      config_.device_memory_fraction == 0 &&
      config_.cuda_sparse_pool_size_GB == 0 &&
      sparse_tree_pool_alloc != nullptr && *sparse_tree_pool_alloc != nullptr) {
    constexpr std::size_t kBaseline = 32UL << 20;
    constexpr std::size_t kMgrBytes =
        kLlvmListManagerFixedAllocationBudgetBytes;
    constexpr int kMgrsPerNode = 3;

    // Account only for structural SNodes which own an element list.
    std::size_t global_region =
        kBaseline + element_list_pool_budget_bytes +
        element_list_snode_count * kMgrBytes;
    std::vector<std::pair<int, std::size_t>> snode_pools;  // (id, data_bytes)
    for (size_t i = 0; i < snode_metas.size(); i++) {
      if (!is_gc_able(snode_metas[i].type))
        continue;
      auto geometry = node_pool_geometries.find(snode_metas[i].id);
      if (geometry == node_pool_geometries.end()) {
        continue;
      }
      const std::size_t node_size = geometry->second.node_size;
      const std::size_t chunk_elems =
          geometry->second.chunk_elements;
      const std::size_t data_chunks = geometry->second.data_chunks;
      std::size_t chunk_bytes = chunk_elems * node_size;
      global_region += std::size_t(kMgrsPerNode) * kMgrBytes;
      std::size_t index_bytes =
          std::size_t(2) * data_chunks * chunk_elems * sizeof(int32);
      const std::size_t directory_bytes =
          std::size_t(kMgrsPerNode) *
          llvm_list_manager_directory_pages_for_chunks(data_chunks) *
          kLlvmListManagerDirectoryAllocationBudgetBytes;
      std::size_t data_bytes =
          data_chunks * chunk_bytes + index_bytes + directory_bytes;
      if (data_bytes > 0)
        snode_pools.push_back({snode_metas[i].id, data_bytes});
    }
    if (!snode_pools.empty()) {
      void *buf_base =
          llvm_device()->get_memory_addr(**sparse_tree_pool_alloc);
      std::size_t offset = global_region;
      for (const auto &p : snode_pools) {
        void *region_ptr = static_cast<char *>(buf_base) + offset;
        TI_TRACE("Phase-1: snode {} pool {:.2f} MiB at +{:.2f} MiB",
                 p.first, p.second / 1048576.0, offset / 1048576.0);
        runtime_jit->call<void *, int, void *, std::size_t>(
            "runtime_NodeAllocator_set_dedicated_pool", llvm_runtime_,
            p.first, region_ptr, p.second);
        offset += p.second;
      }
    }
  }

  // Ambient cells serve inactive reads and are not dynamically active payload.
  // Allocate them directly so allocator high-water, GC and free/recycled lists
  // describe user activation only.
  for (size_t i = 0; i < snode_metas.size(); i++) {
    const std::size_t node_size = direct_ambient_size(
        snode_metas[i].type, snode_metas[i].cell_size_bytes,
        snode_metas[i].chunk_size);
    if (node_size > 0) {
      TI_TRACE("Allocating direct ambient for snode {} (node size {})",
               snode_metas[i].id, node_size);
      runtime_jit->call<void *, int, std::size_t>(
          "runtime_allocate_ambient_direct", llvm_runtime_,
          snode_metas[i].id, node_size);
    }
  }

  if (config_.arch == Arch::cuda && use_device_memory_pool() &&
      config_.cuda_sparse_per_snode_pool &&
      config_.cuda_sparse_pool_auto_size &&
      config_.device_memory_fraction == 0 &&
       config_.cuda_sparse_pool_size_GB == 0 && !all_dense) {
    runtime_jit->call<void *>("runtime_element_lists_prepare_backing_pool",
                              llvm_runtime_);
    for (const auto &meta : snode_metas) {
      runtime_jit->call<void *, int>("runtime_element_list_set_backing_pool",
                                     llvm_runtime_, meta.id);
    }
    runtime_jit->call<void *>("runtime_element_lists_finalize_backing_pool",
                              llvm_runtime_);
  }
}

LlvmDevice *LlvmRuntimeExecutor::llvm_device() {
  TI_ASSERT(dynamic_cast<LlvmDevice *>(device_.get()));
  return static_cast<LlvmDevice *>(device_.get());
}

ThreadPool *LlvmRuntimeExecutor::get_cpu_thread_pool() {
  if (!arch_use_host_memory(config_.arch)) {
    return nullptr;
  }
  // This lock protects only first construction. It does not schedule work or
  // serialize an already-created Program scheduler, and keeping it out of the
  // executor object preserves the stable runtime layout used by incremental
  // native builds.
  static std::mutex thread_pool_creation_mutex;
  std::lock_guard<std::mutex> lock(thread_pool_creation_mutex);
  if (!thread_pool_) {
    thread_pool_ =
        std::make_unique<ThreadPool>(std::max(1, config_.cpu_max_num_threads));
  }
  return thread_pool_.get();
}

DeviceAllocation LlvmRuntimeExecutor::allocate_memory_on_device(
    std::size_t alloc_size,
    uint64 *result_buffer) {
  auto devalloc = llvm_device()->allocate_memory_runtime(
      {{alloc_size, /*host_write=*/false, /*host_read=*/false,
        /*export_sharing=*/false, AllocUsage::Storage},
       get_runtime_jit_module(),
       get_llvm_runtime(),
       result_buffer,
       use_device_memory_pool()});

  TI_ASSERT(allocated_runtime_memory_allocs_.find(devalloc.alloc_id) ==
            allocated_runtime_memory_allocs_.end());
  allocated_runtime_memory_allocs_[devalloc.alloc_id] = devalloc;
  return devalloc;
}

void LlvmRuntimeExecutor::deallocate_memory_on_device(DeviceAllocation handle) {
  TI_ASSERT(allocated_runtime_memory_allocs_.find(handle.alloc_id) !=
            allocated_runtime_memory_allocs_.end());
  llvm_device()->dealloc_memory(handle);
  allocated_runtime_memory_allocs_.erase(handle.alloc_id);
}

void LlvmRuntimeExecutor::fill_ndarray(const DeviceAllocation &alloc,
                                       std::size_t size,
                                       uint32_t data) {
  auto ptr = get_device_alloc_info_ptr(alloc);
  if (config_.arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    auto context_guard = CUDAContext::get_instance().get_guard();
    CUDADriver::get_instance().memsetd32((void *)ptr, data, size);
#else
    TI_NOT_IMPLEMENTED
#endif
  } else if (config_.arch == Arch::amdgpu) {
#if defined(TI_WITH_AMDGPU)
    AMDGPUDriver::get_instance().memset((void *)ptr, data, size);
#else
    TI_NOT_IMPLEMENTED;
#endif
  } else {
    std::fill((uint32_t *)ptr, (uint32_t *)ptr + size, data);
  }
}

uint64_t *LlvmRuntimeExecutor::get_device_alloc_info_ptr(
    const DeviceAllocation &alloc) {
  if (config_.arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    return (uint64_t *)llvm_device()
        ->as<cuda::CudaDevice>()
        ->get_alloc_info(alloc)
        .ptr;
#else
    TI_NOT_IMPLEMENTED
#endif
  } else if (config_.arch == Arch::amdgpu) {
#if defined(TI_WITH_AMDGPU)
    return (uint64_t *)llvm_device()
        ->as<amdgpu::AmdgpuDevice>()
        ->get_alloc_info(alloc)
        .ptr;
#else
    TI_NOT_IMPLEMENTED;
#endif
  }

  return (uint64_t *)llvm_device()
      ->as<cpu::CpuDevice>()
      ->get_alloc_info(alloc)
      .ptr;
}

void LlvmRuntimeExecutor::finalize() {
  profiler_ = nullptr;
  if (config_.arch == Arch::cuda || config_.arch == Arch::amdgpu) {
    const bool backend_calls_safe =
        llvm_device() == nullptr || llvm_device()->backend_calls_safe();
    preallocated_runtime_objects_allocs_.reset();
    preallocated_runtime_memory_allocs_.reset();
    sparse_tree_pool_allocs_.clear();
    sparse_tree_pool_sizes_.clear();

    // Reset runtime memory
    if (backend_calls_safe) {
      auto allocated_runtime_memory_allocs_copy =
          allocated_runtime_memory_allocs_;
      for (auto &iter : allocated_runtime_memory_allocs_copy) {
        // The runtime allocation may have already been freed upon explicit
        // Ndarray/Field destruction. Check if the allocation is still alive.
        void *ptr = llvm_device()->get_memory_addr(iter.second);
        if (ptr == nullptr)
          continue;

        deallocate_memory_on_device(iter.second);
      }
    }
    allocated_runtime_memory_allocs_.clear();

    // Reset device
    llvm_device()->clear();

    // Reset memory pool
    if (backend_calls_safe) {
      DeviceMemoryPool::get_instance().reset();

      // Release unused memory from cuda memory pool.
      synchronize();
    }
  }
  finalized_ = true;
}

LlvmRuntimeExecutor::~LlvmRuntimeExecutor() {
  if (!finalized_) {
    finalize();
  }
}

void *LlvmRuntimeExecutor::preallocate_memory(
    std::size_t prealloc_size,
    DeviceAllocationUnique &devalloc) {
  DeviceAllocation preallocated_device_buffer_alloc;

  Device::AllocParams preallocated_device_buffer_alloc_params;
  preallocated_device_buffer_alloc_params.size = prealloc_size;
  RhiResult res =
      llvm_device()->allocate_memory(preallocated_device_buffer_alloc_params,
                                     &preallocated_device_buffer_alloc);
  TI_ERROR_IF(res != RhiResult::success,
              "Failed to pre-allocate device memory (err: {})", int(res));

  void *preallocated_device_buffer =
      llvm_device()->get_memory_addr(preallocated_device_buffer_alloc);
  devalloc = std::make_unique<DeviceAllocationGuard>(
      std::move(preallocated_device_buffer_alloc));
  return preallocated_device_buffer;
}

void LlvmRuntimeExecutor::preallocate_runtime_memory(
    std::size_t override_size) {
  if (preallocated_runtime_memory_allocs_ != nullptr)
    return;

  std::size_t total_prealloc_size = 0;
  const auto total_mem = llvm_device()->get_total_memory();
  if (override_size > 0) {
    // P-Sparse-Mem-2-A: caller-supplied size derived from snode_metas. Skip
    // the device_memory_GB / fraction / cuda_sparse_pool_size_GB logic; the
    // caller is responsible for sizing.
    total_prealloc_size = override_size;
  } else if (config_.device_memory_fraction == 0) {
    // P-Sparse-Mem-1: cuda_sparse_pool_size_GB > 0 overrides device_memory_GB
    // for the sparse-trigger lazy pool only. This path is only entered on
    // cuda+sparse via initialize_llvm_runtime_snodes(), so capping it here
    // does not affect dense-only programs (which never call this).
    float64 effective_GB = config_.device_memory_GB;
    if (config_.arch == Arch::cuda && use_device_memory_pool() &&
        config_.cuda_sparse_pool_size_GB > 0) {
      effective_GB = config_.cuda_sparse_pool_size_GB;
    }
    TI_ASSERT(effective_GB > 0);
    total_prealloc_size = std::size_t(effective_GB * (1UL << 30));
  } else {
    total_prealloc_size =
        std::size_t(config_.device_memory_fraction * total_mem);
  }
  TI_ASSERT(total_prealloc_size <= total_mem);

  void *runtime_memory_prealloc_buffer = preallocate_memory(
      total_prealloc_size, preallocated_runtime_memory_allocs_);

  TI_TRACE("Allocating device memory {:.2f} MB",
           1.0 * total_prealloc_size / (1UL << 20));

  auto *const runtime_jit = get_runtime_jit_module();
  runtime_jit->call<void *, std::size_t, void *>(
      "runtime_initialize_memory", llvm_runtime_, total_prealloc_size,
      runtime_memory_prealloc_buffer);
}

void LlvmRuntimeExecutor::materialize_runtime(KernelProfilerBase *profiler,
                                              uint64 **result_buffer_ptr) {
  // Starting random state for the program calculated using the random seed.
  // The seed is multiplied by 1048391 so that two programs with different seeds
  // will not have overlapping random states in any thread.
  int starting_rand_state = config_.random_seed * 1048391;

  // Number of random states. One per CPU/CUDA thread.
  int num_rand_states = 0;

  if (config_.arch == Arch::cuda || config_.arch == Arch::amdgpu) {
#if defined(TI_WITH_CUDA) || defined(TI_WITH_AMDGPU)
    // It is important to make sure that every CUDA thread has its own random
    // state so that we do not need expensive per-state locks.
    num_rand_states = config_.saturating_grid_dim * config_.max_block_dim;
#else
    TI_NOT_IMPLEMENTED
#endif
  } else {
    num_rand_states = config_.cpu_max_num_threads;
  }

  // The result buffer allocated here is only used for the launches of
  // runtime JIT functions. To avoid memory leak, we use the head of
  // the preallocated device buffer as the result buffer in
  // CUDA and AMDGPU backends.
  // | ==================preallocated device buffer ========================== |
  // |<- reserved for return ->|<---- usable for allocators on the device ---->|
  auto *const runtime_jit = get_runtime_jit_module();

  size_t runtime_objects_prealloc_size = 0;
  void *runtime_objects_prealloc_buffer = nullptr;
  if (config_.arch == Arch::cuda || config_.arch == Arch::amdgpu) {
#if defined(TI_WITH_CUDA) || defined(TI_WITH_AMDGPU)
    auto [temp_result_alloc, res] =
        llvm_device()->allocate_memory_unique({sizeof(uint64_t)});
    TI_ERROR_IF(
        res != RhiResult::success,
        "Failed to allocate memory for `runtime_get_memory_requirements`");
    void *temp_result_ptr = llvm_device()->get_memory_addr(*temp_result_alloc);

    runtime_jit->call<void *, int32_t, int32_t>(
        "runtime_get_memory_requirements", temp_result_ptr, num_rand_states,
        /*use_preallocated_buffer=*/1);
    runtime_objects_prealloc_size =
        size_t(fetch_result<uint64_t>(0, (uint64_t *)temp_result_ptr));
    temp_result_alloc.reset();
    size_t result_buffer_size = sizeof(uint64) * taichi_result_buffer_entries;

    TI_TRACE("Allocating device memory {:.2f} MB",
             1.0 * (runtime_objects_prealloc_size + result_buffer_size) /
                 (1UL << 20));

    runtime_objects_prealloc_buffer = preallocate_memory(
        iroundup(runtime_objects_prealloc_size + result_buffer_size,
                 taichi_page_size),
        preallocated_runtime_objects_allocs_);

    *result_buffer_ptr =
        (uint64_t *)((uint8_t *)runtime_objects_prealloc_buffer +
                     runtime_objects_prealloc_size);
#else
    TI_NOT_IMPLEMENTED
#endif
  } else {
    *result_buffer_ptr = (uint64 *)HostMemoryPool::get_instance().allocate(
        sizeof(uint64) * taichi_result_buffer_entries, 8);
  }

  TI_TRACE("Launching runtime_initialize");

  auto *host_memory_pool = &HostMemoryPool::get_instance();
  runtime_jit
      ->call<void *, void *, std::size_t, void *, int, void *, void *, void *>(
          "runtime_initialize", *result_buffer_ptr, host_memory_pool,
          runtime_objects_prealloc_size, runtime_objects_prealloc_buffer,
          num_rand_states, (void *)&host_allocate_aligned, (void *)std::printf,
          (void *)std::vsnprintf);

  TI_TRACE("LLVMRuntime initialized (excluding `root`)");
  llvm_runtime_ = fetch_result<void *>(taichi_result_buffer_ret_value_id,
                                       *result_buffer_ptr);
  TI_TRACE("LLVMRuntime pointer fetched");

  // Preallocate for runtime memory and update to LLVMRuntime
  if (config_.arch == Arch::cuda || config_.arch == Arch::amdgpu) {
    if (!use_device_memory_pool()) {
      preallocate_runtime_memory();
    }
  }

  if (config_.arch == Arch::cuda) {
    TI_TRACE("Initializing {} random states using CUDA", num_rand_states);
    runtime_jit->launch<void *, int>(
        "runtime_initialize_rand_states_cuda", config_.saturating_grid_dim,
        config_.max_block_dim, 0, llvm_runtime_, starting_rand_state);
  } else {
    TI_TRACE("Initializing {} random states (serially)", num_rand_states);
    runtime_jit->call<void *, int>("runtime_initialize_rand_states_serial",
                                   llvm_runtime_, starting_rand_state);
  }

  if (arch_use_host_memory(config_.arch)) {
    auto *const thread_pool = get_cpu_thread_pool();
    runtime_jit->call<void *, void *>(
        "LLVMRuntime_set_host_releaser", llvm_runtime_,
        (void *)&host_release);
    runtime_jit->call<void *, void *, void *>(
        "LLVMRuntime_initialize_thread_pool", llvm_runtime_, thread_pool,
        (void *)ThreadPool::static_run);

    runtime_jit->call<void *, void *>("LLVMRuntime_set_assert_failed",
                                      llvm_runtime_,
                                      (void *)assert_failed_host);
  }
  if (arch_is_cpu(config_.arch) && (profiler != nullptr)) {
    // Profiler functions can only be called on CPU kernels
    runtime_jit->call<void *, void *>("LLVMRuntime_set_profiler", llvm_runtime_,
                                      profiler);
    runtime_jit->call<void *, void *>(
        "LLVMRuntime_set_profiler_start", llvm_runtime_,
        (void *)&KernelProfilerBase::profiler_start);
    runtime_jit->call<void *, void *>(
        "LLVMRuntime_set_profiler_stop", llvm_runtime_,
        (void *)&KernelProfilerBase::profiler_stop);
  }
}

void LlvmRuntimeExecutor::destroy_snode_tree(SNodeTree *snode_tree) {
  if (arch_is_cpu(config_.arch)) {
    std::vector<SNode *> snodes;
    bool all_dense = config_.demote_dense_struct_fors;
    std::function<void(SNode *)> collect = [&](SNode *snode) {
      snodes.push_back(snode);
      if (snode->type != SNodeType::dense &&
          snode->type != SNodeType::place &&
          snode->type != SNodeType::root) {
        all_dense = false;
      }
      for (const auto &child : snode->ch) {
        collect(child.get());
      }
    };
    collect(snode_tree->root());

    auto *runtime_jit = get_runtime_jit_module();
    for (std::size_t i = 0; i < snodes.size(); ++i) {
      SNode *snode = snodes[i];
      const std::size_t ambient_size = direct_ambient_size(
          snode->type, snode->cell_size_bytes, snode->chunk_size);
      runtime_jit->call<void *, int, std::size_t, int, int>(
          "runtime_prepare_snode_tree_destroy", llvm_runtime_, snode->id,
          ambient_size, i == 0 ? 1 : 0,
          i + 1 == snodes.size() ? 1 : 0);
    }
    for (SNode *snode : snodes) {
      const std::size_t ambient_size = direct_ambient_size(
          snode->type, snode->cell_size_bytes, snode->chunk_size);
      runtime_jit
          ->call<void *, int, int, int, int, std::size_t>(
              "runtime_destroy_snode_resources", llvm_runtime_, snode->id,
              !all_dense && snode->type != SNodeType::place ? 1 : 0,
              is_gc_able(snode->type) ? 1 : 0,
              ambient_size > 0 ? 1 : 0, ambient_size);
    }
  }
  get_llvm_context()->delete_snode_tree(snode_tree->id());
  snode_tree_buffer_manager_->destroy(snode_tree);
  sparse_tree_pool_allocs_.erase(snode_tree->id());
  sparse_tree_pool_sizes_.erase(snode_tree->id());
}

Device *LlvmRuntimeExecutor::get_compute_device() {
  return device_.get();
}

LLVMRuntime *LlvmRuntimeExecutor::get_llvm_runtime() {
  return static_cast<LLVMRuntime *>(llvm_runtime_);
}

void LlvmRuntimeExecutor::init_runtime_jit_module(
    std::unique_ptr<llvm::Module> module) {
  llvm_context_->init_runtime_module(module.get());
  runtime_jit_module_ = create_jit_module(std::move(module));
}

}  // namespace taichi::lang
