#include "taichi/runtime/llvm/llvm_runtime_executor.h"

#include "taichi/rhi/common/host_memory_pool.h"
#include "taichi/runtime/llvm/llvm_offline_cache.h"
#include "taichi/rhi/cpu/cpu_device.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/platform/cuda/detect_cuda.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/llvm/device_memory_pool.h"

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
  thread_pool_ = std::make_unique<ThreadPool>(config.cpu_max_num_threads);

  llvm_runtime_ = nullptr;

  if (arch_is_cpu(config.arch)) {
    config.max_block_dim = 1024;
    device_ = std::make_shared<cpu::CpuDevice>();

  }
#if defined(TI_WITH_CUDA)
  else if (config.arch == Arch::cuda) {
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
  // TODO: We are likely doing more synchronization than necessary. Simplify the
  // sync logic when we fetch the result.
  synchronize();
  uint64 ret;
  if (config_.arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
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
  auto data_list = runtime_query<void *>("NodeManager_get_data_list",
                                         result_buffer, node_allocator);

  return (std::size_t)runtime_query<int32>("ListManager_get_num_elements",
                                           result_buffer, data_list);
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
  // Phase 1-D (2026-05): optimal chunk_num_elements per gc-able SNode.
  // Declared at function scope because it's populated during auto-size
  // (inside the CUDA device-memory if-block) and consumed during
  // NodeAllocator init (outside that block).
  std::vector<std::pair<int, int>> snode_chunk_elems;
  for (size_t i = 0; i < snode_metas.size(); i++) {
    if (snode_metas[i].type != SNodeType::dense &&
        snode_metas[i].type != SNodeType::place &&
        snode_metas[i].type != SNodeType::root) {
      all_dense = false;
      break;
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
    // Default OFF preserves vanilla taichi 1.7.4 semantics
    // (`pool_size = device_memory_GB * 1GiB`). Opt-in is now safe for the
    // SNode shapes covered by the per-NodeManager headroom below.
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
    if (config_.cuda_sparse_pool_auto_size &&
        config_.device_memory_fraction == 0 &&
        config_.cuda_sparse_pool_size_GB == 0) {
      // Mirror runtime.cpp constants:
      //   * runtime_NodeAllocator_initialize: chunk_num_elements = 16 * 1024
      //   * NodeManager ctor: while (chunk_elements > 1 &&
      //         chunk_elements * element_size > 128 MiB) chunk_elements /= 2
      // ListManager has `Ptr chunks[128 * 1024]` (= 1 MiB on 64-bit) plus
      // small POD fields; allocate_aligned uses 4 KiB pages. Each NodeManager
      // creates 3 ListManager instances (free / recycled / data). At first
      // activation, only the data_list's first chunk is touched.
      constexpr std::size_t kNodeMgrChunkElementsDefault = 16UL * 1024;
      constexpr std::size_t kNodeMgrMaxChunkBytes = 128UL << 20;
      constexpr std::size_t kListManagerBytes =
          (128UL << 10) * sizeof(void *) + 4096;
      constexpr int kListManagersPerNodeManager = 3;
      // Headroom: extra data_list chunks beyond the first. 2 covers typical
      // gc / re-activation cycles; raise via cuda_sparse_pool_size_floor_MiB
      // if a workload activates more chunks per NodeManager.
      constexpr int kHeadroomChunks = 2;
      // Baseline for misc allocations (LLVMRuntime fields, NodeManager
      // structs themselves, ambient_elements, rand_states, etc).
      constexpr std::size_t kBaselineBytes = 32UL << 20;

      std::size_t auto_size = kBaselineBytes;
      // Phase 1-E (2026-05): runtime_initialize_snodes creates ONE
      // element_list (ListManager, ~1 MiB struct) per SNode in the tree,
      // not just per gc_able SNode. With workloads that place many
      // fields on a single bitmasked node (e.g. 9 fields on one
      // pointer.bitmasked → ~25 snodes), the unaccounted element_list
      // metadata easily exceeds the 24 MiB headroom and triggers
      // `Out of CUDA pre-allocated memory` during snode init. Add the
      // per-snode element_list budget here so the global region scales
      // with snode_metas.size().
      auto_size += snode_metas.size() * kListManagerBytes;
      for (size_t i = 0; i < snode_metas.size(); i++) {
        if (!is_gc_able(snode_metas[i].type))
          continue;
        std::size_t element_size = snode_metas[i].cell_size_bytes;
        // Phase 1 (2026-05): mirror runtime_NodeAllocator_initialize's
        // pointer handling: cell_size_bytes is 0 for pointer SNode (the
        // slot size is inferred at runtime), but the data chunk geometry
        // still uses sizeof(int32) per slot. Without this, pointer
        // SNode types are silently excluded from per-snode-pool sizing
        // and their ListManager data allocations fall through to the
        // (now undersized) global pool, causing OOM.
        if (snode_metas[i].type == SNodeType::pointer && element_size == 0) {
          element_size = sizeof(int32);
        }
        if (element_size == 0)
          continue;
        // Compute node_size exactly as the NodeAllocator init code does:
        //   pointer → node_size = element_size  (single-element allocator)
        //   dynamic → node_size = sizeof(void*) + element_size * chunk_size
        std::size_t node_size;
        if (snode_metas[i].type == SNodeType::pointer) {
          node_size = element_size;
        } else {
          node_size =
              sizeof(void *) + element_size * snode_metas[i].chunk_size;
        }
        // Phase 1-D (2026-05): compute optimal chunk_elements. Start from
        // the default and halve for the 128 MiB ceiling, then further
        // tighten if num_cells_per_container is much smaller.
        std::size_t chunk_elements = kNodeMgrChunkElementsDefault;
        while (chunk_elements > 1 &&
               chunk_elements * node_size > kNodeMgrMaxChunkBytes) {
          chunk_elements /= 2;
        }
        // Tighten: if the physical cell count is far below chunk capacity,
        // shrink chunk_elements to ∼2× num_cells (power-of-2, ≥1024).
        // This dramatically cuts per-chunk VRAM for sparse workloads while
        // leaving room for GC/recycle transient overcommit.
        int64_t n_cells = snode_metas[i].num_cells_per_container;
        if (n_cells > 0) {
          std::size_t desired = (std::size_t)n_cells * 2;
          // round up to next power of 2, floor 1024
          std::size_t tight = 1024;
          while (tight < desired) tight *= 2;
          if (tight < chunk_elements) chunk_elements = tight;
        }
        std::size_t chunk_bytes = chunk_elements * node_size;
        auto_size += std::size_t(kListManagersPerNodeManager) * kListManagerBytes;
        // Phase 0.5 (2026-05): when the user provided
        // `vk_max_active=N` on the SNode, size the per-NodeManager data
        // region to fit ceil(N / chunk_elements) chunks plus a small
        // safety margin (kHintHeadroomChunks) instead of the legacy
        // worst-case (1 + kHeadroomChunks) chunks. Lower-bound is one
        // `num_cells_per_container` (a single container must always fit).
        // Default (-1) keeps the legacy worst-case path bit-for-bit.
        //
        // The hint headroom (1 chunk) covers gc/re-activation transient
        // overcommit and miscellaneous runtime allocations that draw from
        // the same preallocated pool; setting it to 0 produced silent
        // `allocate_from_reserved_memory: Out of CUDA pre-allocated
        // memory` errors on workloads where the dominant chunk is large
        // (>50 MiB) and only a single chunk gets reserved. With +1 chunk,
        // savings vs legacy = 1 chunk per gc-able SNode (typically the
        // largest one) which still meaningfully reduces footprint when
        // chunks are big but activations are sparse.
        constexpr int kHintHeadroomChunks = 1;
        std::size_t data_chunks = std::size_t(1 + kHeadroomChunks);
        // Phase 0.5 / Phase 1-B (2026-05): when the user provided
        // `vk_max_active=N` use it directly; otherwise auto-hint from
        // `num_cells_per_container` – the physical upper bound of
        // simultaneously active cells that the SNode tree can hold.
        // This eliminates over-provisioning for sparse workloads
        // (e.g. MPM with 495 pointer cells getting 8192-slot chunks).
        // A kHintHeadroomChunks margin is reserved for GC/recycle
        // transient overcommit and as an extension point for future
        // runtime cooperative grow (Phase 2).
        int64_t effective = snode_metas[i].vk_max_active_hint;
        if (effective <= 0) {
          effective = snode_metas[i].num_cells_per_container;
        }
        if (effective > 0) {
          int64_t lower_bound = snode_metas[i].num_cells_per_container > 0
                                    ? snode_metas[i].num_cells_per_container
                                    : 1;
          int64_t eff = std::max<int64_t>(effective, lower_bound);
          std::size_t needed_chunks =
              (std::size_t(eff) + chunk_elements - 1) / chunk_elements;
          data_chunks = std::max<std::size_t>(needed_chunks, 1) +
                        std::size_t(kHintHeadroomChunks);
        }
        auto_size += data_chunks * chunk_bytes;
        // Phase 1-D: record optimal chunk_elements for NodeAllocator init
        snode_chunk_elems.push_back(
            {snode_metas[i].id, (int)chunk_elements});
        // Phase 1: collect per-SNode sizing for buffer carving
        if (do_per_snode_pool) {
          snode_entries.push_back(
              {snode_metas[i].id,
               std::size_t(kListManagersPerNodeManager) * kListManagerBytes,
               data_chunks * chunk_bytes, chunk_bytes});
        }
      }
      // User-tunable lower bound (defensive floor for tiny SNode trees).
      const std::size_t floor_bytes =
          std::size_t(std::max(0, config_.cuda_sparse_pool_size_floor_MiB))
          << 20;
      auto_size = std::max(auto_size, floor_bytes);

      // device_memory_GB acts as a sanity ceiling, NOT a silent cap. If
      // the heuristic asks for more, warn and clamp; the user can either
      // raise device_memory_GB or set cuda_sparse_pool_size_GB explicitly.
      std::size_t cap =
          std::size_t(config_.device_memory_GB * (1UL << 30));
      if (auto_size > cap) {
        TI_WARN(
            "cuda_sparse_pool_auto_size: SNode-derived sparse pool "
            "{:.2f} MiB exceeds device_memory_GB cap {:.2f} MiB; clamping. "
            "Raise device_memory_GB or set cuda_sparse_pool_size_GB to "
            "avoid runtime OOM in allocate_from_reserved_memory.",
            auto_size / 1048576.0, cap / 1048576.0);
        override_size = cap;
      } else {
        override_size = auto_size;
      }

      // Phase 1 (2026-05): carve per-SNode data regions from a single
      // buffer instead of using a monolithic global pool. The global
      // region (runtime_memory_chunk) shrinks to baseline + metadata +
      // headroom; each SNode's data region is a sub-range of the same
      // allocation, addressed via its own PreallocatedMemoryChunk.
      // Total VRAM = global_region + Σ(data_regions) ≈ auto_size + 16 MiB.
      if (do_per_snode_pool && !snode_entries.empty()) {
        constexpr std::size_t kPerSnodePoolGlobalHeadroom = 24UL << 20;
        // Phase 1-E (2026-05): include per-SNode element_list metadata
        // budget (one ListManager per SNode in the tree). This mirrors
        // the addition in `auto_size` above and prevents snode-init OOM
        // on trees with many leaf places.
        std::size_t global_region = kBaselineBytes +
                                    kPerSnodePoolGlobalHeadroom +
                                    snode_metas.size() * kListManagerBytes;
        std::size_t total_buffer = global_region;
        for (const auto &e : snode_entries) {
          global_region += e.metadata_bytes;
        }
        total_buffer = global_region;
        for (const auto &e : snode_entries) {
          total_buffer += e.data_bytes;
        }
        // Clamp to device_memory_GB ceiling (if auto_size exceeded cap,
        // scale down proportionally; if not, use computed values).
        if (total_buffer > cap) {
          double scale = (double)cap / (double)total_buffer;
          global_region = std::max(kBaselineBytes,
                                   (std::size_t)((double)global_region * scale));
          total_buffer = cap;
        }
        TI_TRACE(
            "Phase-1 per-SNode pools: global={:.2f} MiB total={:.2f} MiB "
            "({} SNode data regions)",
            global_region / 1048576.0, total_buffer / 1048576.0,
            snode_entries.size());

        // Allocate one contiguous device buffer.
        void *buf = preallocate_memory(
            total_buffer, preallocated_runtime_memory_allocs_);
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

  runtime_jit->call<void *, std::size_t, int, int, int, std::size_t, Ptr>(
      "runtime_initialize_snodes", llvm_runtime_, root_size, root_id,
      (int)snode_metas.size(), tree_id, rounded_size, root_buffer, all_dense);

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
      for (size_t j = 0; j < snode_chunk_elems.size(); j++) {
        if (snode_chunk_elems[j].first == snode_id) {
          chunk_elems = snode_chunk_elems[j].second; break;
        }
      }
      TI_TRACE("Initializing allocator for snode {} (node size {}, chunk_elems {})",
               snode_id, node_size, chunk_elems);
      runtime_jit->call<void *, int, std::size_t, int>(
          "runtime_NodeAllocator_initialize_ex", llvm_runtime_, snode_id,
          node_size, chunk_elems);
      TI_TRACE("Allocating ambient element for snode {} (node size {})",
               snode_id, node_size);
      runtime_jit->call<void *, int>("runtime_allocate_ambient", llvm_runtime_,
                                     snode_id, node_size);
    }
  }

  // Phase 1 (2026-05): after all NodeAllocators are initialized, assign
  // each gc-able SNode its dedicated data region carved from the global
  // pool buffer. Re-computes per-SNode sizes from snode_metas using the
  // same formula as the auto-size loop above.
  if (config_.arch == Arch::cuda && use_device_memory_pool() &&
      config_.cuda_sparse_per_snode_pool &&
      config_.cuda_sparse_pool_auto_size &&
      config_.device_memory_fraction == 0 &&
      config_.cuda_sparse_pool_size_GB == 0 &&
      preallocated_runtime_memory_allocs_ != nullptr) {
    // Mirror runtime.cpp constants (same as auto-size block above).
    constexpr std::size_t kBaseline = 32UL << 20;
    constexpr std::size_t kMgrBytes = (128UL << 10) * sizeof(void *) + 4096;
    constexpr int kMgrsPerNode = 3;
    constexpr std::size_t kHeadroom = 24UL << 20;
    constexpr std::size_t kChunkElems = 16UL * 1024;
    constexpr std::size_t kMaxChunk = 128UL << 20;
    constexpr int kHeadroomChunks = 2;
    constexpr int kHintHeadroomChunks = 1;

    // Phase 1-E (2026-05): account for per-SNode element_list metadata
    // (one ListManager per SNode in the tree, ~1 MiB each), which lives
    // in the global region. Without this, the carved global_region can
    // be too small when a sparse tree has many leaf places (e.g. 9
    // fields on one bitmasked node), causing snode-init OOM.
    std::size_t global_region =
        kBaseline + kHeadroom + snode_metas.size() * kMgrBytes;
    std::vector<std::pair<int, std::size_t>> snode_pools;  // (id, data_bytes)
    for (size_t i = 0; i < snode_metas.size(); i++) {
      if (!is_gc_able(snode_metas[i].type))
        continue;
      std::size_t elem_sz = snode_metas[i].cell_size_bytes;
      // Phase 1 (2026-05): mirror NodeAllocator init's pointer handling.
      if (snode_metas[i].type == SNodeType::pointer && elem_sz == 0) {
        elem_sz = sizeof(int32);
      }
      if (elem_sz == 0)
        continue;
      // Compute node_size exactly as the NodeAllocator init does:
      // pointer → elem_sz; dynamic → sizeof(void*) + elem_sz * chunk_size
      std::size_t node_size;
      if (snode_metas[i].type == SNodeType::pointer) {
        node_size = elem_sz;
      } else {
        node_size =
            sizeof(void *) + elem_sz * snode_metas[i].chunk_size;
      }
      // Phase 1-D: tighten chunk_elems as in the auto-size block above.
      std::size_t chunk_elems = kChunkElems;
      while (chunk_elems > 1 && chunk_elems * node_size > kMaxChunk)
        chunk_elems /= 2;
      int64_t n_cells = snode_metas[i].num_cells_per_container;
      if (n_cells > 0) {
        std::size_t desired = (std::size_t)n_cells * 2;
        std::size_t tight = 1024;
        while (tight < desired) tight *= 2;
        if (tight < chunk_elems) chunk_elems = tight;
      }
      std::size_t chunk_bytes = chunk_elems * node_size;
      global_region += std::size_t(kMgrsPerNode) * kMgrBytes;
      std::size_t data_chunks = std::size_t(1 + kHeadroomChunks);
      // Phase 1-B (2026-05): auto-hint from num_cells_per_container
      // when no explicit vk_max_active_hint is set.
      int64_t effective = snode_metas[i].vk_max_active_hint;
      if (effective <= 0) {
        effective = snode_metas[i].num_cells_per_container;
      }
      if (effective > 0) {
        int64_t lb = snode_metas[i].num_cells_per_container > 0
                         ? snode_metas[i].num_cells_per_container
                         : 1;
        int64_t eff = std::max<int64_t>(effective, lb);
        std::size_t need =
            (std::size_t(eff) + chunk_elems - 1) / chunk_elems;
        data_chunks =
            std::max<std::size_t>(need, 1) + std::size_t(kHintHeadroomChunks);
      }
      std::size_t data_bytes = data_chunks * chunk_bytes;
      if (data_bytes > 0)
        snode_pools.push_back({snode_metas[i].id, data_bytes});
    }
    if (!snode_pools.empty()) {
      void *buf_base = llvm_device()->get_memory_addr(
          *preallocated_runtime_memory_allocs_);
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
}

LlvmDevice *LlvmRuntimeExecutor::llvm_device() {
  TI_ASSERT(dynamic_cast<LlvmDevice *>(device_.get()));
  return static_cast<LlvmDevice *>(device_.get());
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
    preallocated_runtime_objects_allocs_.reset();
    preallocated_runtime_memory_allocs_.reset();
    // Phase 1: per-SNode pool allocs share the same underlying buffer
    // as preallocated_runtime_memory_allocs_; just clear the vector.
    per_snode_pool_allocs_.clear();

    // Reset runtime memory
    auto allocated_runtime_memory_allocs_copy =
        allocated_runtime_memory_allocs_;
    for (auto &iter : allocated_runtime_memory_allocs_copy) {
      // The runtime allocation may have already been freed upon explicit
      // Ndarray/Field destruction Check if the allocation still alive
      void *ptr = llvm_device()->get_memory_addr(iter.second);
      if (ptr == nullptr)
        continue;

      deallocate_memory_on_device(iter.second);
    }
    allocated_runtime_memory_allocs_.clear();

    // Reset device
    llvm_device()->clear();

    // Reset memory pool
    DeviceMemoryPool::get_instance().reset();

    // Release unused memory from cuda memory pool
    synchronize();
  }
  finalized_ = true;
}

std::vector<std::pair<int, int>>
LlvmRuntimeExecutor::query_snode_pool_watermarks() {
  std::vector<std::pair<int, int>> results;
  if (!llvm_runtime_ || !use_device_memory_pool())
    return results;

  auto *runtime_jit = get_runtime_jit_module();
  // Allocate a small host-visible buffer for the result (2 × uint64 per SNode)
  constexpr int kMaxSnodes = 32;
  constexpr std::size_t kBufSize = kMaxSnodes * 2 * sizeof(uint64_t);
  LlvmDevice::LlvmRuntimeAllocParams buf_params;
  buf_params.size = kBufSize;
  buf_params.host_write = false;
  buf_params.host_read = true;
  buf_params.export_sharing = false;
  buf_params.usage = AllocUsage::Storage;
  buf_params.runtime_jit = runtime_jit;
  buf_params.runtime = (LLVMRuntime *)llvm_runtime_;
  buf_params.result_buffer = nullptr;
  buf_params.use_memory_pool = false;
  auto buf_alloc = llvm_device()->allocate_memory_runtime(buf_params);

  void *buf = llvm_device()->get_memory_addr(buf_alloc);
  if (!buf) return results;
  std::memset(buf, 0, kBufSize);

  int idx = 0;
  // Iterate over all SNode types and call the runtime watermark function.
  // The snode_metas from field_cache_data would be ideal, but since we're
  // after initialization, we iterate the known gc-able types.
  // Instead, use a simpler approach: call the runtime function for each
  // possible snode_id in node_allocators range.
  for (int snode_id = 0; snode_id < kMaxSnodes; snode_id++) {
    Ptr permille_ptr = (Ptr)((char *)buf + (idx * 2) * sizeof(uint64_t));
    Ptr used_ptr = (Ptr)((char *)buf + (idx * 2 + 1) * sizeof(uint64_t));
    runtime_jit->call<void *, int, Ptr, Ptr>(
        "runtime_NodeAllocator_get_watermark", llvm_runtime_, snode_id,
        permille_ptr, used_ptr);
    idx++;
    if (idx >= kMaxSnodes) break;
  }

  synchronize();

  uint64_t *data = (uint64_t *)buf;
  for (int i = 0; i < idx; i++) {
    uint64_t permille = data[i * 2];
    if (permille > 0) {
      results.push_back({i, (int)permille});
    }
  }

  llvm_device()->dealloc_memory(buf_alloc);
  return results;
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
    runtime_jit->call<void *, void *, void *>(
        "LLVMRuntime_initialize_thread_pool", llvm_runtime_, thread_pool_.get(),
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
  get_llvm_context()->delete_snode_tree(snode_tree->id());
  snode_tree_buffer_manager_->destroy(snode_tree);
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
