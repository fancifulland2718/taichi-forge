#include "taichi/runtime/gfx/runtime.h"
#include "taichi/program/program.h"
#include "taichi/common/filesystem.hpp"

// FIXME: (penguinliong) Special offer for `run_codegen`. Find a new home for it
// in the future.
#include "taichi/codegen/spirv/spirv_codegen.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "fp16.h"

#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

namespace taichi::lang {
namespace gfx {

namespace {

std::atomic<uint64_t> graph_replay_slot_saturation_fallbacks{0};

class HostDeviceContextBlitter {
 public:
  HostDeviceContextBlitter(const KernelContextAttributes *ctx_attribs,
                           const std::vector<CompiledTaichiKernel::RuntimeArrayArg>
                               *runtime_array_args,
                           LaunchContextBuilder &host_ctx,
                           Device *device,
                           DeviceAllocation *device_args_buffer,
                           DeviceAllocation *device_ret_buffer)
      : ctx_attribs_(ctx_attribs),
        runtime_array_args_(runtime_array_args),
        host_ctx_(host_ctx),
        device_args_buffer_(device_args_buffer),
        device_ret_buffer_(device_ret_buffer),
        device_(device) {
  }

  void host_to_device(
      const std::unordered_map<std::vector<int>,
                               DeviceAllocation,
                               hashing::Hasher<std::vector<int>>> &ext_arrays,
      const std::unordered_map<std::vector<int>,
                               size_t,
                               hashing::Hasher<std::vector<int>>> &ext_arr_size,
      const std::unordered_map<std::vector<int>,
                               const ArgPack *,
                               hashing::Hasher<std::vector<int>>> &argpacks) {
    if (!ctx_attribs_->has_args()) {
      return;
    }

    void *device_base{nullptr};
    TI_ASSERT(device_->map(*device_args_buffer_, &device_base) ==
              RhiResult::success);

    for (const auto &array_arg : *runtime_array_args_) {
      const auto &indices = array_arg.indices;
      const auto alloc_type = host_ctx_.device_allocation_type[indices];
      if (alloc_type == LaunchContextBuilder::DevAllocType::kNone &&
          ext_arr_size.at(indices)) {
          // Only need to blit ext arrs (host array)
          // Bug B/C fix (forge 2026-05): always blit host→device for ext
          // arrs regardless of ExternalPtrAccess flag. The previous READ-only
          // optimization left WRITE-only device buffers uninitialized with
          // recycled GPU memory; for kernels whose struct-for over a sparse
          // SNode only writes a subset of cells (e.g. tensor_to_ext_arr from
          // to_numpy on bitmasked/pointer fields after deactivate), the
          // unwritten cells leaked stale data back to the user. The host
          // ndarray (e.g. np.zeros from to_numpy) is the user's contract for
          // the initial state of the buffer; device buffer must match it.
          (void)array_arg.access;
          {
            DeviceAllocation buffer = ext_arrays.at(indices);
            void *device_arr_ptr{nullptr};
            TI_ASSERT(device_->map(buffer, &device_arr_ptr) ==
                      RhiResult::success);
            const void *host_ptr =
                host_ctx_.array_ptrs[array_arg.data_ptr_indices];
            std::memcpy(device_arr_ptr, host_ptr, ext_arr_size.at(indices));
            device_->unmap(buffer);
          }
        }
        // Substitute in the device address.

        if ((alloc_type == LaunchContextBuilder::DevAllocType::kNone ||
             alloc_type == LaunchContextBuilder::DevAllocType::kNdarray) &&
            device_->get_caps().get(
                DeviceCapability::spirv_has_physical_storage_buffer)) {
          uint64_t addr =
              device_->get_memory_physical_pointer(ext_arrays.at(indices));
          uint64_t grad_addr =
              (uint64)host_ctx_.array_ptrs[array_arg.grad_ptr_indices];
          if (alloc_type == LaunchContextBuilder::DevAllocType::kNdarray &&
              host_ctx_.array_ptrs[array_arg.grad_ptr_indices] != nullptr) {
            auto grad_alloc =
                *(DeviceAllocation *)(
                    host_ctx_.array_ptrs[array_arg.grad_ptr_indices]);
            grad_addr = device_->get_memory_physical_pointer(grad_alloc);
          }
          host_ctx_.set_ndarray_ptrs(indices, addr, grad_addr);
        }
    }

    std::memcpy(device_base, host_ctx_.get_context().arg_buffer,
                ctx_attribs_->args_bytes());

    device_->unmap(*device_args_buffer_);
  }

  bool device_to_host(
      CommandList *cmdlist,
      const std::unordered_map<std::vector<int>,
                               DeviceAllocation,
                               hashing::Hasher<std::vector<int>>> &ext_arrays,
      const std::unordered_map<std::vector<int>,
                               size_t,
                               hashing::Hasher<std::vector<int>>>
          &ext_arr_size) {
    if (ctx_attribs_->empty()) {
      return false;
    }

    bool require_sync = ctx_attribs_->rets().size() > 0;
    std::vector<DevicePtr> readback_dev_ptrs;
    std::vector<void *> readback_host_ptrs;
    std::vector<size_t> readback_sizes;

    for (const auto &array_arg : *runtime_array_args_) {
      const auto &indices = array_arg.indices;
      if (host_ctx_.device_allocation_type[indices] ==
              LaunchContextBuilder::DevAllocType::kNone &&
          ext_arr_size.at(indices)) {
        if (array_arg.access & uint32_t(irpass::ExternalPtrAccess::WRITE)) {
          // Only need to blit ext arrs (host array)
          readback_dev_ptrs.push_back(ext_arrays.at(indices).get_ptr(0));
          readback_host_ptrs.push_back(
              host_ctx_.array_ptrs[array_arg.data_ptr_indices]);
          // TODO: readback grad_ptrs as well once ndarray ad is supported
          readback_sizes.push_back(ext_arr_size.at(indices));
          require_sync = true;
        }
      }
    }

    if (require_sync) {
      if (readback_sizes.size()) {
        StreamSemaphore command_complete_sema =
            device_->get_compute_stream()->submit(cmdlist);

        device_->wait_idle();

        // In this case `readback_data` syncs
        TI_ASSERT(device_->readback_data(
                      readback_dev_ptrs.data(), readback_host_ptrs.data(),
                      readback_sizes.data(), int(readback_sizes.size()),
                      {command_complete_sema}) == RhiResult::success);
      } else {
        device_->get_compute_stream()->submit_synced(cmdlist);
      }

      if (!ctx_attribs_->has_rets()) {
        return true;
      }
    } else {
      return false;
    }

    void *device_base{nullptr};
    TI_ASSERT(device_->map(*device_ret_buffer_, &device_base) ==
              RhiResult::success);

    void *ctx_result_buffer = host_ctx_.get_context().result_buffer;
    std::memcpy(ctx_result_buffer, device_base, ctx_attribs_->rets_bytes());

    device_->unmap(*device_ret_buffer_);

    return true;
  }

  static std::unique_ptr<HostDeviceContextBlitter> maybe_make(
      const KernelContextAttributes *ctx_attribs,
      const std::vector<CompiledTaichiKernel::RuntimeArrayArg>
          *runtime_array_args,
      LaunchContextBuilder &host_ctx,
      Device *device,
      DeviceAllocation *device_args_buffer,
      DeviceAllocation *device_ret_buffer) {
    if (ctx_attribs->empty()) {
      return nullptr;
    }
    return std::make_unique<HostDeviceContextBlitter>(
        ctx_attribs, runtime_array_args, host_ctx, device, device_args_buffer,
        device_ret_buffer);
  }

 private:
  const KernelContextAttributes *const ctx_attribs_;
  const std::vector<CompiledTaichiKernel::RuntimeArrayArg>
      *const runtime_array_args_;
  LaunchContextBuilder &host_ctx_;
  DeviceAllocation *const device_args_buffer_;
  DeviceAllocation *const device_ret_buffer_;
  Device *const device_;
};

uint64_t graph_allocation_generation(DeviceAllocation alloc) {
  return 0;
}

void push_graph_allocation_key(std::vector<uint64_t> &key,
                               DeviceAllocation alloc,
                               uint64_t offset = 0,
                               uint64_t bytes = 0) {
  key.push_back(reinterpret_cast<uint64_t>(alloc.device));
  key.push_back(alloc.alloc_id);
  key.push_back(graph_allocation_generation(alloc));
  key.push_back(offset);
  key.push_back(bytes);
}

}  // namespace

uint64_t get_graph_replay_slot_saturation_fallbacks() {
  return graph_replay_slot_saturation_fallbacks.load(
      std::memory_order_relaxed);
}

class GraphReplayRegistry {
 public:
  explicit GraphReplayRegistry(GfxRuntime *runtime) : runtime_(runtime) {
  }

  void retire(uint64_t token) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (runtime_ != nullptr) {
      runtime_->retire_graph_replay(token);
    }
  }

  void close() {
    std::lock_guard<std::mutex> lock(mutex_);
    runtime_ = nullptr;
  }

  GraphReplayStats debug_stats(uint64_t replay_key) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (runtime_ == nullptr) {
      return {};
    }
    return runtime_->debug_graph_replay_stats(replay_key);
  }

 private:
  std::mutex mutex_;
  GfxRuntime *runtime_{nullptr};
};

GraphReplayRegistration::GraphReplayRegistration(
    std::shared_ptr<GraphReplayRegistry> registry,
    uint64_t replay_key)
    : registry_(std::move(registry)), replay_key_(replay_key) {
  TI_ASSERT(registry_ != nullptr);
  TI_ASSERT(replay_key_ != 0);
}

GraphReplayRegistration::~GraphReplayRegistration() {
  if (registry_ != nullptr) {
    registry_->retire(replay_key_);
  }
}

GraphReplayStats GraphReplayRegistration::debug_stats() const {
  if (registry_ == nullptr) {
    return {};
  }
  return registry_->debug_stats(replay_key_);
}

constexpr size_t kGtmpBufferSize = 1024 * 1024;
constexpr size_t kHashOverflowBufferSize = 5 * sizeof(uint32_t);
constexpr size_t kListGenBufferSize = 32 << 20;
constexpr size_t kListGenMinBufferSize = sizeof(uint32_t);
constexpr size_t kListGenAutoSlackEntries = 1024;
constexpr size_t kListGenBufferAlignment = 4096;

size_t align_up_to(size_t value, size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

bool snode_can_use_spirv_listgen(const SNode *snode) {
  if (snode == nullptr || snode->type == SNodeType::root ||
      snode->is_place() || snode->is_path_all_dense) {
    return false;
  }
  return snode->type == SNodeType::dense ||
         snode->type == SNodeType::bitmasked ||
         snode->type == SNodeType::pointer ||
         snode->type == SNodeType::hash ||
         snode->type == SNodeType::dynamic;
}

size_t estimate_listgen_entries(const CompiledSNodeStructs &compiled_structs) {
  size_t result = 0;
  for (const auto &[sid, desc] : compiled_structs.snode_descriptors) {
    (void)sid;
    if (snode_can_use_spirv_listgen(desc.snode)) {
      size_t entries = desc.total_num_cells_from_root;
      size_t suffix = 1;
      for (auto *sn = desc.snode; sn != nullptr && sn->type != SNodeType::root;
           sn = sn->parent) {
        const auto it = compiled_structs.snode_descriptors.find(sn->id);
        if (it != compiled_structs.snode_descriptors.end() &&
            sn->type == SNodeType::hash && it->second.hash_table_capacity > 0) {
          entries = it->second.hash_table_capacity * suffix;
          break;
        }
        suffix *= static_cast<size_t>(sn->num_cells_per_container);
      }
      result = std::max(result, entries);
    }
  }
  return result;
}

// Info for launching a compiled Taichi kernel, which consists of a series of
// Unified Device API pipelines.

CompiledTaichiKernel::CompiledTaichiKernel(const Params &ti_params)
    : ti_kernel_attribs_(*ti_params.ti_kernel_attribs),
      device_(ti_params.device) {
  input_buffers_[BufferType::GlobalTmps] = ti_params.global_tmps_buffer;
  input_buffers_[BufferType::HashOverflow] = ti_params.hash_overflow_buffer;
  input_buffers_[BufferType::ListGen] = ti_params.listgen_buffer;

  // Compiled_structs can be empty if loading a kernel from an AOT module as
  // the SNode are not re-compiled/structured. In this case, we assume a
  // single root buffer size configured from the AOT module.
  for (int root = 0; root < ti_params.num_snode_trees; ++root) {
    BufferInfo buffer = {BufferType::Root, root};
    input_buffers_[buffer] = ti_params.root_buffers[root];
  }
#if defined(TI_WITH_VULKAN_POINTER)
  // B-3.b (2026-05): 注册独立 NodeAllocatorPool buffer。key 中 root_id[0] = sid。
  // OFF 默认该 vector 为空，与历史 input_buffers_ 内容字节等价。
  for (const auto &[sid, alloc] : ti_params.node_allocator_pool_buffers) {
    BufferInfo buffer = {BufferType::NodeAllocatorPool, sid};
    input_buffers_[buffer] = alloc;
  }
  // C-2.5 (2026-05): 注册 chunked allocator 的全部 chunk DeviceAllocation
  // 列表。dispatch 时按 BufferBind.chunk_count > 0 用 rw_buffer_array(N)
  // 把 N 个 buffer 一并 bind 到单个 descriptor array binding。
  for (const auto &[sid, allocs] : ti_params.node_allocator_chunk_arrays) {
    BufferInfo buffer = {BufferType::NodeAllocatorPool, sid};
    chunk_arrays_[buffer] = allocs;
  }
#endif

  const auto arg_sz = ti_kernel_attribs_.ctx_attribs.args_bytes();
  const auto ret_sz = ti_kernel_attribs_.ctx_attribs.rets_bytes();

  args_buffer_size_ = arg_sz;
  ret_buffer_size_ = ret_sz;

  const auto &ctx_attribs = ti_kernel_attribs_.ctx_attribs;
  for (const auto &kv : ctx_attribs.args()) {
    const auto &indices = kv.first;
    const auto &arg = kv.second;
    if (!arg.is_array) {
      continue;
    }
    RuntimeArrayArg array_arg;
    array_arg.indices = indices;
    array_arg.data_ptr_indices = indices;
    array_arg.data_ptr_indices.push_back(TypeFactory::DATA_PTR_POS_IN_NDARRAY);
    array_arg.grad_ptr_indices = indices;
    array_arg.grad_ptr_indices.push_back(TypeFactory::GRAD_PTR_POS_IN_NDARRAY);
    for (const auto &access : ctx_attribs.arr_access) {
      if (access.first == indices) {
        array_arg.access = uint32_t(access.second);
        break;
      }
    }
    runtime_array_args_.push_back(std::move(array_arg));
  }

  for (const auto &kv : ctx_attribs.argpack_types()) {
    runtime_argpack_args_.push_back(kv.first);
  }

  const auto &task_attribs = ti_kernel_attribs_.tasks_attribs;
  const auto &spirv_bins = ti_params.spirv_bins;
  TI_ASSERT(task_attribs.size() == spirv_bins.size());
  cached_resource_sets_.resize(task_attribs.size());
  buffer_binding_plans_.resize(task_attribs.size());

  for (int i = 0; i < task_attribs.size(); ++i) {
    auto &binding_plans = buffer_binding_plans_[i];
    binding_plans.reserve(task_attribs[i].buffer_binds.size());
    for (const auto &bind : task_attribs[i].buffer_binds) {
      BufferBindingPlan plan;
      plan.buffer = bind.buffer;
      plan.binding = bind.binding;
      plan.chunk_count = bind.chunk_count;
      if (bind.binding < 0) {
        plan.kind = BufferBindingKind::Skip;
      } else if (bind.buffer.type == BufferType::ExtArr) {
        plan.kind = BufferBindingKind::ExtArrRw;
      } else if (bind.buffer.type == BufferType::Args) {
        plan.kind = BufferBindingKind::Args;
      } else if (bind.buffer.type == BufferType::ArgPack) {
        plan.kind = BufferBindingKind::ArgPack;
      } else if (bind.buffer.type == BufferType::Rets) {
        plan.kind = BufferBindingKind::RetsRw;
      } else if (bind.buffer.type == BufferType::ListGen) {
        plan.kind = BufferBindingKind::StaticLookupRw;
      } else {
        if (bind.buffer.type == BufferType::NodeAllocatorPool &&
            bind.chunk_count > 0u) {
          auto chunk_it = chunk_arrays_.find(bind.buffer);
          if (chunk_it != chunk_arrays_.end()) {
            plan.kind = BufferBindingKind::ChunkedRwArray;
            plan.chunk_array = &chunk_it->second;
            binding_plans.push_back(plan);
            continue;
          }
        }
        plan.kind = BufferBindingKind::StaticRw;
        auto alloc_it = input_buffers_.find(bind.buffer);
        if (alloc_it != input_buffers_.end()) {
          plan.static_alloc = alloc_it->second;
        }
      }
      binding_plans.push_back(plan);
    }

    PipelineSourceDesc source_desc{PipelineSourceType::spirv_binary,
                                   (void *)spirv_bins[i].data(),
                                   spirv_bins[i].size() * sizeof(uint32_t)};
    auto [vp, res] = ti_params.device->create_pipeline_unique(
        source_desc, task_attribs[i].name, ti_params.backend_cache);
    pipelines_.push_back(std::move(vp));
  }
}

const TaichiKernelAttributes &CompiledTaichiKernel::ti_kernel_attribs() const {
  return ti_kernel_attribs_;
}

size_t CompiledTaichiKernel::num_pipelines() const {
  return pipelines_.size();
}

size_t CompiledTaichiKernel::get_args_buffer_size() const {
  return args_buffer_size_;
}

size_t CompiledTaichiKernel::get_ret_buffer_size() const {
  return ret_buffer_size_;
}

Pipeline *CompiledTaichiKernel::get_pipeline(int i) {
  return pipelines_[i].get();
}

ShaderResourceSet *CompiledTaichiKernel::get_cached_resource_set(int i) {
  if (!cached_resource_sets_[i]) {
    cached_resource_sets_[i] = device_->create_resource_set_unique();
  }
  return cached_resource_sets_[i].get();
}

GfxRuntime::GfxRuntime(const Params &params)
    : device_(params.device),
      profiler_(params.profiler),
      listgen_dynamic_size_(params.listgen_dynamic_size),
      listgen_explicit_size_(params.listgen_buffer_MB > 0),
      dispatch_cache_(params.dispatch_cache),
      listgen_reuse_(params.listgen_reuse),
      listgen_reuse_adaptive_(params.listgen_reuse_adaptive),
      ctx_buffer_ring_enabled_(params.ctx_buffer_ring),
      ctx_buffer_ring_size_(
          params.ctx_buffer_ring_size > 0
              ? static_cast<size_t>(params.ctx_buffer_ring_size)
              : size_t{8}),
      cmdlist_lazy_submit_enabled_(params.cmdlist_lazy_submit),
      cmdlist_lazy_submit_min_dispatches_(
          params.cmdlist_max_dispatches > 0
              ? static_cast<size_t>(params.cmdlist_max_dispatches)
              : size_t{0}),
      debug_mode_(params.debug) {
  graph_replay_registry_ = std::make_shared<GraphReplayRegistry>(this);
  TI_ERROR_IF(params.listgen_buffer_MB < 0,
              "vulkan_listgen_buffer_MB must be >= 0, got {}",
              params.listgen_buffer_MB);
  if (listgen_explicit_size_) {
    listgen_initial_buffer_size_ =
        static_cast<size_t>(params.listgen_buffer_MB) * 1024u * 1024u;
  } else if (listgen_dynamic_size_) {
    listgen_initial_buffer_size_ = 0;
  } else {
    listgen_initial_buffer_size_ = kListGenBufferSize;
  }
  current_cmdlist_pending_since_ = high_res_clock::now();
  init_nonroot_buffers();

  // Read pipeline cache from disk if available.
  std::filesystem::path cache_path(get_repo_dir());
  cache_path /= "rhi_cache.bin";
  std::vector<char> cache_data;
  constexpr uintmax_t kMaxPipelineCacheBytes = 256u * 1024u * 1024u;
  std::error_code cache_ec;
  const bool cache_exists = std::filesystem::exists(cache_path, cache_ec);
  if (cache_exists && !cache_ec) {
    const uintmax_t cache_size =
        std::filesystem::file_size(cache_path, cache_ec);
    if (cache_ec) {
      TI_WARN("Ignoring unreadable Vulkan pipeline cache at {}: {}",
              cache_path.generic_string(), cache_ec.message());
    } else if (cache_size > kMaxPipelineCacheBytes) {
      TI_WARN("Ignoring oversized Vulkan pipeline cache at {} ({} bytes)",
              cache_path.generic_string(), cache_size);
    } else if (cache_size != 0) {
      TI_TRACE("Loading pipeline cache from {}", cache_path.generic_string());
      std::ifstream cache_file(cache_path, std::ios::binary);
      if (cache_file) {
        cache_data.resize(static_cast<size_t>(cache_size));
        cache_file.read(cache_data.data(),
                        static_cast<std::streamsize>(cache_data.size()));
        if (!cache_file ||
            cache_file.gcount() !=
                static_cast<std::streamsize>(cache_data.size())) {
          TI_WARN("Ignoring truncated Vulkan pipeline cache at {}",
                  cache_path.generic_string());
          cache_data.clear();
        }
      }
    }
  } else if (!cache_ec) {
    TI_TRACE("Pipeline cache not found at {}", cache_path.generic_string());
  } else {
    TI_WARN("Failed to inspect Vulkan pipeline cache at {}: {}",
            cache_path.generic_string(), cache_ec.message());
  }
  auto cache_result = device_->create_pipeline_cache_unique(
      cache_data.size(), cache_data.empty() ? nullptr : cache_data.data());
  if (cache_result.second != RhiResult::success && !cache_data.empty()) {
    TI_WARN("Discarding incompatible Vulkan pipeline cache at {}",
            cache_path.generic_string());
    cache_data.clear();
    cache_result = device_->create_pipeline_cache_unique();
  }
  if (cache_result.second == RhiResult::success) {
    backend_cache_ = std::move(cache_result.first);
  }
}

GfxRuntime::~GfxRuntime() {
  // Close the registry before touching device state. A late graph-cache
  // destructor may retain the registry object, but it can no longer call back
  // into this runtime after close() returns.
  graph_replay_registry_->close();
  if (device_->backend_calls_safe()) {
    try {
      synchronize_impl(/*check_hash_overflow=*/false);
    } catch (const BackendRuntimeError &error) {
      // Destructors are noexcept. The reporter preserves the first fatal
      // backend error; teardown below only destroys host-owned wrappers and
      // Vulkan handles, without issuing another wait or submission.
      device_->report_backend_error(error);
    } catch (const std::exception &error) {
      TI_WARN("GfxRuntime teardown synchronization failed: {}", error.what());
    } catch (...) {
      TI_WARN("GfxRuntime teardown synchronization failed");
    }
  }
  graph_replay_states_.clear();

  // Write pipeline cache back to disk.
  if (backend_cache_) {
    uint8_t *cache_data = nullptr;
    size_t cache_size = 0;
    if (device_->backend_calls_safe()) {
      cache_data = (uint8_t *)backend_cache_->data();
      cache_size = backend_cache_->size();
    }
    if (cache_data) {
      // C4 (2026-04-26): atomic write — write to <path>.tmp then rename.
      // Avoids leaving a half-written rhi_cache.bin on the disk if the
      // process is killed mid-write, which would cause the next startup
      // to fail in create_pipeline_cache_unique() with a corrupted blob.
      // (Periodic in-flight flush and cross-process file locks were
      // considered and dropped as low-ROI; see compile_doc/优化总规划.md
      // §7.2 row 4.)
      std::filesystem::path cache_path =
          std::filesystem::path(get_repo_dir()) / "rhi_cache.bin";
      std::filesystem::path tmp_path = cache_path;
      tmp_path += ".tmp";
      bool wrote_ok = false;
      {
        std::ofstream cache_file(tmp_path,
                                 std::ios::binary | std::ios::trunc);
        if (cache_file) {
          cache_file.write(reinterpret_cast<const char *>(cache_data),
                           static_cast<std::streamsize>(cache_size));
          cache_file.flush();
          wrote_ok = static_cast<bool>(cache_file);
        }
      }
      if (wrote_ok) {
        std::error_code ec;
        std::filesystem::rename(tmp_path, cache_path, ec);
        if (ec) {
          // Fallback: remove dest then rename. Some filesystems on Windows
          // refuse cross-existing rename without explicit replace.
          std::filesystem::remove(cache_path, ec);
          std::filesystem::rename(tmp_path, cache_path, ec);
          if (ec) {
            TI_TRACE("Failed to atomically install rhi_cache.bin: {}",
                     ec.message());
            std::filesystem::remove(tmp_path, ec);
          }
        }
      } else {
        std::error_code ec;
        std::filesystem::remove(tmp_path, ec);
      }
    }
    backend_cache_.reset();
  }

  {
    decltype(ti_kernels_) tmp;
    tmp.swap(ti_kernels_);
  }
  global_tmps_buffer_.reset();
  listgen_buffer_.reset();
}

int64 GfxRuntime::get_sparse_list_version(int snode_id) const {
  auto it = sparse_list_states_.find(snode_id);
  if (it == sparse_list_states_.end()) {
    return 0;
  }
  return it->second.version;
}

bool GfxRuntime::sparse_list_task_is_current(
    const TaskAttributes &attribs) {
  if (!listgen_reuse_ ||
      attribs.sparse_list_op != TaskAttributes::kSparseListOpListgen ||
      attribs.sparse_list_snode_id < 0 ||
      attribs.sparse_list_parent_snode_id < 0 ||
      resident_sparse_list_snode_id_ != attribs.sparse_list_snode_id) {
    return false;
  }

  auto it = sparse_list_states_.find(attribs.sparse_list_snode_id);
  if (it == sparse_list_states_.end()) {
    if (listgen_reuse_adaptive_) {
      auto &state = sparse_list_states_[attribs.sparse_list_snode_id];
      record_sparse_list_reuse_sample(state, /*would_skip=*/false);
    }
    return false;
  }
  auto &state = it->second;
  const bool would_skip = state.clean_epoch == state.dirty_epoch &&
                 state.global_dirty_seen ==
                   sparse_list_global_dirty_epoch_ &&
                 state.clean_parent_version ==
                   get_sparse_list_version(
                     attribs.sparse_list_parent_snode_id);
  record_sparse_list_reuse_sample(state, would_skip);
  return would_skip && !state.adaptive_disabled;
}

void GfxRuntime::record_sparse_list_reuse_sample(SparseListState &state,
                                                 bool would_skip) const {
  if (!listgen_reuse_adaptive_) {
    return;
  }
  constexpr int kWindow = 64;
  constexpr int kDisablePercent = 10;
  constexpr int kEnablePercent = 15;
  const int hit = would_skip ? 1 : 0;
  if (state.adaptive_window_size < kWindow) {
    state.adaptive_window_size++;
  } else {
    state.adaptive_hit_count -=
        int((state.adaptive_window_bits >> (kWindow - 1)) & 1u);
  }
  state.adaptive_window_bits = (state.adaptive_window_bits << 1) |
                               static_cast<std::uint64_t>(hit);
  state.adaptive_hit_count += hit;

  if (state.adaptive_window_size < kWindow) {
    return;
  }
  if (!state.adaptive_disabled &&
      state.adaptive_hit_count * 100 < kDisablePercent * kWindow) {
    state.adaptive_disabled = true;
  } else if (state.adaptive_disabled &&
             state.adaptive_hit_count * 100 >= kEnablePercent * kWindow) {
    state.adaptive_disabled = false;
  }
}

void GfxRuntime::mark_sparse_list_task_launched(
    const TaskAttributes &attribs) {
  if (!listgen_reuse_ || attribs.task_type != OffloadedTaskType::listgen) {
    return;
  }
  if (attribs.sparse_list_op != TaskAttributes::kSparseListOpListgen ||
      attribs.sparse_list_snode_id < 0 ||
      attribs.sparse_list_parent_snode_id < 0) {
    resident_sparse_list_snode_id_ = -1;
    return;
  }

  auto &state = sparse_list_states_[attribs.sparse_list_snode_id];
  if (state.parent_snode_id != attribs.sparse_list_parent_snode_id) {
    if (state.parent_snode_id >= 0) {
      auto old_children = child_lists_by_parent_.find(state.parent_snode_id);
      if (old_children != child_lists_by_parent_.end()) {
        old_children->second.erase(attribs.sparse_list_snode_id);
        if (old_children->second.empty()) {
          child_lists_by_parent_.erase(old_children);
        }
      }
    }
    child_lists_by_parent_[attribs.sparse_list_parent_snode_id].insert(
        attribs.sparse_list_snode_id);
    state.parent_snode_id = attribs.sparse_list_parent_snode_id;
  }
  state.clean_epoch = state.dirty_epoch;
  state.clean_parent_version =
      get_sparse_list_version(attribs.sparse_list_parent_snode_id);
  state.global_dirty_seen = sparse_list_global_dirty_epoch_;
  state.version++;
  resident_sparse_list_snode_id_ = attribs.sparse_list_snode_id;
}

void GfxRuntime::invalidate_sparse_list_cache(int sparse_mutation_snode_id) {
  if (!listgen_reuse_) {
    return;
  }
  if (sparse_mutation_snode_id < 0) {
    sparse_list_global_dirty_epoch_++;
    return;
  }

  std::unordered_set<int> affected;
  std::vector<int> stack;
  stack.push_back(sparse_mutation_snode_id);
  while (!stack.empty()) {
    const int snode_id = stack.back();
    stack.pop_back();
    if (!affected.insert(snode_id).second) {
      continue;
    }
    auto children = child_lists_by_parent_.find(snode_id);
    if (children != child_lists_by_parent_.end()) {
      for (int child_snode_id : children->second) {
        stack.push_back(child_snode_id);
      }
    }
  }
  for (int snode_id : affected) {
    auto it = sparse_list_states_.find(snode_id);
    if (it != sparse_list_states_.end()) {
      it->second.dirty_epoch++;
    }
  }
}

void GfxRuntime::clear_sparse_list_cache_resident() {
  sparse_list_states_.clear();
  child_lists_by_parent_.clear();
  sparse_list_global_dirty_epoch_ = 0;
  resident_sparse_list_snode_id_ = -1;
}

void GfxRuntime::register_hash_overflow_checks(
    int root_id,
    const CompiledSNodeStructs &compiled_structs) {
  for (const auto &[sid, desc] : compiled_structs.snode_descriptors) {
    if (desc.snode == nullptr || desc.snode->type != SNodeType::hash ||
        desc.hash_table_capacity == 0 ||
        desc.snode->parent == nullptr ||
        desc.snode->parent->type != SNodeType::root) {
      continue;
    }
    size_t base_offset = 0;
    for (const SNode *sn = desc.snode;
         sn != nullptr && sn->type != SNodeType::root; sn = sn->parent) {
      const auto it = compiled_structs.snode_descriptors.find(sn->id);
      TI_ASSERT_INFO(it != compiled_structs.snode_descriptors.end(),
                     "Hash SNode {} is missing from compiled descriptors",
                     sn->id);
      base_offset += it->second.mem_offset_in_parent_cell;
    }
    const size_t no_offset = static_cast<size_t>(-1);
    hash_overflow_watches_.push_back(
        {root_id, sid, base_offset + desc.hash_overflow_count_offset,
         base_offset + desc.hash_active_count_offset,
         desc.hash_tombstone_count_offset == no_offset
             ? no_offset
             : base_offset + desc.hash_tombstone_count_offset});
  }
}

void GfxRuntime::check_hash_overflow_counters() {
  if (hash_overflow_error_reported_) {
    return;
  }

  std::vector<DevicePtr> ptrs;
  std::vector<uint32_t> values;
  std::vector<void *> host_ptrs;
  std::vector<size_t> sizes;
  std::vector<HashOverflowWatch> live_watches;
  ptrs.reserve(hash_overflow_watches_.size());
  values.resize(hash_overflow_watches_.size());
  host_ptrs.reserve(hash_overflow_watches_.size());
  sizes.reserve(hash_overflow_watches_.size());
  live_watches.reserve(hash_overflow_watches_.size());

  for (const auto &watch : hash_overflow_watches_) {
    if (watch.root_id < 0 ||
        watch.root_id >= static_cast<int>(root_buffers_.size()) ||
        !root_buffers_[watch.root_id]) {
      continue;
    }
    ptrs.push_back(
        root_buffers_[watch.root_id]->get_ptr(watch.overflow_byte_offset));
    host_ptrs.push_back(&values[ptrs.size() - 1]);
    sizes.push_back(sizeof(uint32_t));
    live_watches.push_back(watch);
  }
  int first_overflow = -1;
  if (!ptrs.empty()) {
    auto status = device_->readback_data(ptrs.data(), host_ptrs.data(),
                                         sizes.data(), int(ptrs.size()));
    TI_ERROR_IF(status != RhiResult::success,
                "Failed to read Hash SNode overflow counters.");

    for (int i = 0; i < static_cast<int>(live_watches.size()); ++i) {
      if (values[i] != 0) {
        first_overflow = i;
        break;
      }
    }
  }

  if (first_overflow >= 0) {
    const auto &watch = live_watches[first_overflow];
    uint32_t active_count = 0;
    uint32_t tombstone_count = 0;
    if (watch.root_id >= 0 &&
        watch.root_id < static_cast<int>(root_buffers_.size()) &&
        root_buffers_[watch.root_id]) {
      std::vector<DevicePtr> diag_ptrs;
      std::vector<void *> diag_host_ptrs;
      std::vector<size_t> diag_sizes;
      diag_ptrs.push_back(
          root_buffers_[watch.root_id]->get_ptr(watch.active_byte_offset));
      diag_host_ptrs.push_back(&active_count);
      diag_sizes.push_back(sizeof(uint32_t));
      if (watch.tombstone_byte_offset != static_cast<size_t>(-1)) {
        diag_ptrs.push_back(
            root_buffers_[watch.root_id]->get_ptr(watch.tombstone_byte_offset));
        diag_host_ptrs.push_back(&tombstone_count);
        diag_sizes.push_back(sizeof(uint32_t));
      }
      auto diag_status = device_->readback_data(
          diag_ptrs.data(), diag_host_ptrs.data(), diag_sizes.data(),
          int(diag_ptrs.size()));
      if (diag_status != RhiResult::success) {
        active_count = 0;
        tombstone_count = 0;
      }
    }
    hash_overflow_error_reported_ = true;
    TI_ERROR(
        "Hash SNode table overflow on root {} SNode {}. Increase "
        "capacity=... or expected_active=... before materialization, or "
        "rebuild the SNode tree after high churn. active_count={}, "
        "tombstone_count={}.",
        watch.root_id, watch.snode_id, active_count, tombstone_count);
  }

  if (!hash_overflow_buffer_) {
    return;
  }
  std::array<uint32_t, 5> aggregate{};
  std::array<void *, 1> aggregate_host_ptrs{aggregate.data()};
  std::array<DevicePtr, 1> aggregate_ptrs{
      hash_overflow_buffer_->get_ptr(0)};
  std::array<size_t, 1> aggregate_sizes{aggregate.size() * sizeof(uint32_t)};
  auto aggregate_status = device_->readback_data(
      aggregate_ptrs.data(), aggregate_host_ptrs.data(),
      aggregate_sizes.data(), int(aggregate_ptrs.size()));
  TI_ERROR_IF(aggregate_status != RhiResult::success,
              "Failed to read non-root Hash SNode overflow diagnostics.");
  if (aggregate[0] == 0) {
    return;
  }
  uint32_t active_count = 0;
  uint32_t tombstone_count = 0;
  active_count = aggregate[3];
  tombstone_count = aggregate[4];
  hash_overflow_error_reported_ = true;
  TI_ERROR(
      "Hash SNode table overflow on root {} SNode {}. Increase "
      "capacity=... or expected_active=... before materialization, or rebuild "
      "the SNode tree after high churn. active_count={}, tombstone_count={}.",
      aggregate[1], aggregate[2], active_count, tombstone_count);
}

GfxRuntime::KernelHandle GfxRuntime::register_taichi_kernel(
    GfxRuntime::RegisterParams reg_params) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  CompiledTaichiKernel::Params params;
  params.ti_kernel_attribs = &(reg_params.kernel_attribs);
  params.num_snode_trees = reg_params.num_snode_trees;
  params.device = device_;
  params.root_buffers = {};
  for (int root = 0; root < root_buffers_.size(); ++root) {
    params.root_buffers.push_back(root_buffers_[root].get());
  }
  params.global_tmps_buffer = global_tmps_buffer_.get();
  params.hash_overflow_buffer = hash_overflow_buffer_.get();
  params.listgen_buffer = listgen_buffer_.get();
#if defined(TI_WITH_VULKAN_POINTER)
  // B-3.b (2026-05): 枚举所有独立 pool buffer。allocator->independent_pool_alloc()
  // 返回 nullptr 表示该 SNode 仍走 root_buffer 子区间（在 vector 中跳过）。
  // C-2.2 (2026-05): independent_pool_alloc() 已升级为 DeviceNodeAllocator 基类
  // 虚方法，BumpOnly 与 Chunked skeleton 都能透出 chunk[0] DeviceAllocation。
  for (const auto &[rid, sid_to_alloc] : node_allocators_) {
    (void)rid;
    for (const auto &[sid, allocator_ptr] : sid_to_alloc) {
      if (allocator_ptr == nullptr) {
        continue;
      }
      DeviceAllocation *indep = allocator_ptr->independent_pool_alloc();
      if (indep != nullptr) {
        params.node_allocator_pool_buffers.emplace_back(sid, indep);
      }
      // C-2.5 (2026-05): chunked allocator 在 rw_buffer_array 路径下需要
      // 全部 chunk 的 DeviceAllocation 列表。
      // §13.4 (2026-05-02): 任意 chunk 数（含 1）都走 chunk_arrays 路径，
      // 与 spirv_codegen.cpp::lookup_chunked_pool_contract 同步——后者已
      // 移除 max_chunks>1 限制；codegen 与 runtime 必须对 chunked 是否
      // 走 descriptor array 给出相同答案，避免 binding 不匹配 (§13.4
      // ptr_to_chunk_idx_ vs ptr_to_buffers_ 不变量)。
      auto chunk_list = allocator_ptr->chunks();
      if (!chunk_list.empty() && allocator_ptr->is_chunked()) {
        params.node_allocator_chunk_arrays.emplace_back(
            sid, std::move(chunk_list));
      }
    }
  }
#endif
  params.backend_cache = backend_cache_.get();

  for (int i = 0; i < reg_params.task_spirv_source_codes.size(); ++i) {
    const auto &spirv_src = reg_params.task_spirv_source_codes[i];

    // If we can reach here, we have succeeded. Otherwise
    // std::optional::value() would have killed us.
    params.spirv_bins.push_back(std::move(spirv_src));
  }
  KernelHandle res;
  res.set_launch_id(ti_kernels_.size());
  ti_kernel_snode_tree_ids_.push_back(std::move(reg_params.snode_tree_ids));
  ti_kernels_.push_back(std::make_unique<CompiledTaichiKernel>(params));
  return res;
}

void GfxRuntime::retire_snode_tree_kernels(int tree_id) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  // Graph replay executables cache raw CompiledTaichiKernel pointers. The
  // Program destroy transaction synchronized the device before entering here,
  // so dropping every replay recording is safe. Registrations remain alive;
  // unrelated graphs simply record again on their next run.
  graph_replay_states_.clear();
  TI_ASSERT(ti_kernel_snode_tree_ids_.size() == ti_kernels_.size());
  for (std::size_t i = 0; i < ti_kernels_.size(); ++i) {
    const auto &tree_ids = ti_kernel_snode_tree_ids_[i];
    if (ti_kernels_[i] != nullptr &&
        std::binary_search(tree_ids.begin(), tree_ids.end(), tree_id)) {
      ti_kernels_[i].reset();
      ti_kernel_snode_tree_ids_[i].clear();
    }
  }
}

void GfxRuntime::launch_kernel(KernelHandle handle,
                               LaunchContextBuilder &host_ctx) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  TI_ASSERT(handle.get_launch_id() >= 0 &&
            static_cast<std::size_t>(handle.get_launch_id()) <
                ti_kernels_.size() &&
            ti_kernels_[handle.get_launch_id()] != nullptr);
  auto *ti_kernel = ti_kernels_[handle.get_launch_id()].get();

#if defined(__APPLE__)
  if (profiler_) {
    const int apple_max_query_pool_count = 32;
    int task_count = ti_kernel->ti_kernel_attribs().tasks_attribs.size();
    if (task_count > apple_max_query_pool_count) {
      TI_WARN(
          "Cannot concurrently profile more than 32 tasks in a single Taichi "
          "kernel. Profiling aborted.");
      profiler_ = nullptr;
    } else if (device_->profiler_get_sampler_count() + task_count >
               apple_max_query_pool_count) {
      flush();
      device_->profiler_sync();
    }
  }
#endif

  ensure_listgen_capacity_for_kernel(*ti_kernel);

  std::unique_ptr<DeviceAllocationGuard> args_buffer{nullptr},
      ret_buffer{nullptr};

  if (ti_kernel->get_args_buffer_size()) {
    if (auto pooled = acquire_ctx_buffer(ti_kernel->get_args_buffer_size(),
                                         AllocUsage::Uniform)) {
      args_buffer = std::move(pooled);
    } else {
      auto [buf, res] = device_->allocate_memory_unique(
          {ti_kernel->get_args_buffer_size(),
           /*host_write=*/true, /*host_read=*/false,
           /*export_sharing=*/false, AllocUsage::Uniform});
      TI_ASSERT_INFO(res == RhiResult::success,
                     "Failed to allocate args buffer");
      args_buffer = std::move(buf);
    }
  }

  if (ti_kernel->get_ret_buffer_size()) {
    if (auto pooled = acquire_ctx_buffer(ti_kernel->get_ret_buffer_size(),
                                         AllocUsage::Storage)) {
      ret_buffer = std::move(pooled);
    } else {
      auto [buf, res] = device_->allocate_memory_unique(
          {ti_kernel->get_ret_buffer_size(),
           /*host_write=*/false, /*host_read=*/true,
           /*export_sharing=*/false, AllocUsage::Storage});
      TI_ASSERT_INFO(res == RhiResult::success, "Failed to allocate ret buffer");
      ret_buffer = std::move(buf);
    }
  }

  // Create context blitter
  auto ctx_blitter = HostDeviceContextBlitter::maybe_make(
      &ti_kernel->ti_kernel_attribs().ctx_attribs,
      &ti_kernel->runtime_array_args(), host_ctx, device_, args_buffer.get(),
      ret_buffer.get());

  // `any_arrays` contain both external arrays and NDArrays
  std::unordered_map<std::vector<int>, DeviceAllocation,
                     hashing::Hasher<std::vector<int>>>
      any_arrays;
  // `ext_array_size` only holds the size of external arrays (host arrays)
  // As buffer size information is only needed when it needs to be allocated
  // and transferred by the host
  std::unordered_map<std::vector<int>, size_t,
                     hashing::Hasher<std::vector<int>>>
      ext_array_size;
  std::unordered_map<std::vector<int>, DeviceAllocation,
                     hashing::Hasher<std::vector<int>>>
      textures;
  // `argpacks` holds argpacks that passed to this kernel.
  std::unordered_map<std::vector<int>, const ArgPack *,
                     hashing::Hasher<std::vector<int>>>
      argpacks;
  std::vector<DeviceAllocation> argpack_allocations;

  // Prepare context buffers & arrays
  if (ctx_blitter) {
    TI_ASSERT(ti_kernel->get_args_buffer_size() ||
              ti_kernel->get_ret_buffer_size());

    for (const auto &array_arg : ti_kernel->runtime_array_args()) {
      const auto &indices = array_arg.indices;
      const auto alloc_type = host_ctx.device_allocation_type[indices];
      if (alloc_type != LaunchContextBuilder::DevAllocType::kNone) {
          DeviceAllocation devalloc = kDeviceNullAllocation;
          // NDArray
          if (host_ctx.array_ptrs.count(array_arg.data_ptr_indices)) {
            devalloc =
                *(DeviceAllocation *)(
                    host_ctx.array_ptrs[array_arg.data_ptr_indices]);
          }
          // Texture
          if (host_ctx.array_ptrs.count(indices)) {
            devalloc = *(DeviceAllocation *)(host_ctx.array_ptrs[indices]);
          }

          if (alloc_type == LaunchContextBuilder::DevAllocType::kNdarray) {
            any_arrays[indices] = devalloc;
            any_arrays[array_arg.data_ptr_indices] = devalloc;
            ndarrays_in_use_.insert(devalloc.alloc_id);
            auto grad_it = host_ctx.array_ptrs.find(array_arg.grad_ptr_indices);
            if (grad_it != host_ctx.array_ptrs.end() &&
                grad_it->second != nullptr) {
              auto grad_alloc = *(DeviceAllocation *)(grad_it->second);
              any_arrays[array_arg.grad_ptr_indices] = grad_alloc;
              ndarrays_in_use_.insert(grad_alloc.alloc_id);
            }
          } else if (alloc_type == LaunchContextBuilder::DevAllocType::kTexture) {
            textures[indices] = devalloc;
          } else if (alloc_type ==
                     LaunchContextBuilder::DevAllocType::kRWTexture) {
            textures[indices] = devalloc;
          } else {
            TI_NOT_IMPLEMENTED;
          }
        } else {
          ext_array_size[indices] = host_ctx.array_runtime_sizes[indices];
          uint32_t access = array_arg.access;
          // Alloc ext arr
          size_t alloc_size = std::max(size_t(32), ext_array_size.at(indices));
          // Bug B/C fix (forge 2026-05): host_write must be true regardless
          // of the kernel's READ/WRITE access pattern, because we now always
          // blit host→device for ext arrs (see host_to_device above) and the
          // buffer therefore must be host-visible. The previous code keyed
          // host_write off the READ flag, but mis-named it host_write — it's
          // really "host-can-map".
          (void)access;
          bool host_write = true;
          auto [allocated, res] = device_->allocate_memory_unique(
              {alloc_size, host_write, false, /*export_sharing=*/false,
               AllocUsage::Storage});
          TI_ASSERT_INFO(res == RhiResult::success,
                         "Failed to allocate ext arr buffer");
          any_arrays[indices] = *allocated.get();
          any_arrays[array_arg.data_ptr_indices] = *allocated.get();
          ctx_buffers_.push_back(std::move(allocated));
        }
    }

    for (const auto &indices : ti_kernel->runtime_argpack_args()) {
      TI_ASSERT(host_ctx.device_allocation_type[indices] ==
                LaunchContextBuilder::DevAllocType::kArgPack);
      TI_ASSERT(host_ctx.argpack_ptrs.count(indices));
      const ArgPack *argpack = host_ctx.argpack_ptrs[indices];
      DeviceAllocation devalloc = argpack->get_device_allocation();
      argpacks_in_use_.insert(devalloc.alloc_id);
      argpacks[indices] = argpack;
      argpack_allocations.push_back(devalloc);
    }

    ctx_blitter->host_to_device(any_arrays, ext_array_size, argpacks);
  }

  ensure_current_cmdlist();

  // Record commands
  const auto &task_attribs = ti_kernel->ti_kernel_attribs().tasks_attribs;
  bool argpack_barriers_inserted = false;

  auto mark_storage_buffer_write = [&](const BufferBind &bind) {
    if (!bind.may_write()) {
      return;
    }
    switch (bind.buffer.type) {
      case BufferType::Args:
      case BufferType::ArgPack:
        return;
      case BufferType::ExtArr:
        add_pending_dispatch_barrier(any_arrays.at(bind.buffer.root_id));
        return;
      case BufferType::Rets:
        if (ret_buffer) {
          add_pending_dispatch_barrier(*ret_buffer);
        }
        return;
      case BufferType::NodeAllocatorPool:
        if (bind.chunk_count > 0u) {
          if (bind.chunk_count > 1u) {
            pending_dispatch_global_barrier_ = true;
            return;
          }
          if (auto *chunks = ti_kernel->get_chunk_array(bind.buffer)) {
            for (const auto &chunk : *chunks) {
              add_pending_dispatch_barrier(chunk);
            }
            return;
          }
        }
        break;
      default:
        break;
    }
    DeviceAllocation *alloc = ti_kernel->get_buffer_bind(bind.buffer);
    if (alloc != nullptr) {
      add_pending_dispatch_barrier(*alloc);
    }
  };
  auto mark_task_writes = [&](const TaskAttributes &attribs) {
    if (!dispatch_cache_) {
      return;
    }
    for (const auto &bind : attribs.texture_binds) {
      if (bind.is_storage) {
        pending_dispatch_global_barrier_ = true;
        return;
      }
    }
    for (const auto &bind : attribs.buffer_binds) {
      mark_storage_buffer_write(bind);
    }
  };

  for (int i = 0; i < task_attribs.size(); ++i) {
    const auto &attribs = task_attribs[i];
    if (sparse_list_task_is_current(attribs)) {
      TI_TRACE("Skipping current Vulkan sparse list kernel {}", attribs.name);
      continue;
    }
    if (task_uses_listgen_buffer(attribs)) {
      // VS-1.3 hardening: any dispatched task that binds the shared listgen
      // buffer (writer listgen or reader struct_for) makes a later grow/replace
      // require synchronization. This is deliberately more conservative than
      // tracking only OffloadedTaskType::listgen and protects future AOT / sparse
      // reader paths from replacing an in-flight listgen allocation.
      listgen_buffer_used_ = true;
    }
    insert_pending_dispatch_barriers();
    if (!argpack_barriers_inserted) {
      for (DeviceAllocation alloc : argpack_allocations) {
        current_cmdlist_->buffer_barrier(alloc);
      }
      argpack_barriers_inserted = true;
    }
    auto vp = ti_kernel->get_pipeline(i);
    const int group_x = (attribs.advisory_total_num_threads +
                         attribs.advisory_num_threads_per_group - 1) /
                        attribs.advisory_num_threads_per_group;
    std::unique_ptr<ShaderResourceSet> one_shot_bindings;
    ShaderResourceSet *bindings = nullptr;
    if (dispatch_cache_) {
      bindings = ti_kernel->get_cached_resource_set(i);
    } else {
      one_shot_bindings = device_->create_resource_set_unique();
      bindings = one_shot_bindings.get();
    }
    for (const auto &bind : ti_kernel->buffer_binding_plan(i)) {
      if (bind.binding < 0) {
        continue;
      }
      if (bind.kind == CompiledTaichiKernel::BufferBindingKind::StaticRw) {
        bindings->rw_buffer(bind.binding,
                            bind.static_alloc ? *bind.static_alloc
                                              : kDeviceNullAllocation);
        continue;
      }
      if (bind.kind ==
          CompiledTaichiKernel::BufferBindingKind::ChunkedRwArray) {
        TI_ASSERT_INFO(
            bind.chunk_array->size() == bind.chunk_count,
            "C-2.5: chunk_count mismatch sid={}, attribs={} runtime={}",
            bind.buffer.root_id, bind.chunk_count, bind.chunk_array->size());
        bindings->rw_buffer_array(bind.binding, *bind.chunk_array);
        continue;
      }
      if (bind.kind ==
          CompiledTaichiKernel::BufferBindingKind::StaticLookupRw) {
        DeviceAllocation *alloc = ti_kernel->get_buffer_bind(bind.buffer);
        bindings->rw_buffer(bind.binding,
                            alloc ? *alloc : kDeviceNullAllocation);
        continue;
      }
      if (bind.kind == CompiledTaichiKernel::BufferBindingKind::ExtArrRw) {
        bindings->rw_buffer(bind.binding, any_arrays.at(bind.buffer.root_id));
        continue;
      }
      if (bind.kind == CompiledTaichiKernel::BufferBindingKind::Args) {
        bindings->buffer(bind.binding,
                         args_buffer ? *args_buffer : kDeviceNullAllocation);
        continue;
      }
      if (bind.kind == CompiledTaichiKernel::BufferBindingKind::ArgPack) {
        DeviceAllocation alloc =
            argpacks.at(bind.buffer.root_id)->get_device_allocation();
        bindings->buffer(bind.binding, alloc);
        continue;
      }
      if (bind.kind == CompiledTaichiKernel::BufferBindingKind::RetsRw) {
        bindings->rw_buffer(bind.binding,
                            ret_buffer ? *ret_buffer : kDeviceNullAllocation);
        continue;
      }
      TI_NOT_IMPLEMENTED;
    }

    for (auto &bind : attribs.texture_binds) {
      DeviceAllocation texture = textures.at(bind.arg_id);
      if (bind.is_storage) {
        transition_image(texture, ImageLayout::shader_read_write);
        bindings->rw_image(bind.binding, texture, 0);
      } else {
        transition_image(texture, ImageLayout::shader_read);
        bindings->image(bind.binding, texture, {});
      }
    }

    if (attribs.task_type == OffloadedTaskType::listgen) {
      for (auto &bind : attribs.buffer_binds) {
        if (bind.buffer.type == BufferType::ListGen) {
          // Bug D fix (forge 2026-05): only zero the count slot (4 bytes
          // at offset 0), not the entire 32MB listgen buffer. The listgen
          // kernel writes count via atomic_add starting from 0 and writes
          // entries to listgen[1 + slot] for each unique slot it claims;
          // the consuming struct_for only reads listgen[1..count], so any
          // stale data past `count` is unreachable. Filling the full 32MB
          // every dispatch was costing ~0.15ms per listgen task on
          // discrete GPUs (sparse warm 0.785 -> 0.539 ms expected).
          // FIXME: properly support multiple lists at distinct offsets.
          current_cmdlist_->buffer_fill(
              ti_kernel->get_buffer_bind(bind.buffer)->get_ptr(0),
              /*size=*/sizeof(uint32_t),
              /*data=*/0);
          if (dispatch_cache_) {
            current_cmdlist_->buffer_barrier(
                ti_kernel->get_buffer_bind(bind.buffer)->get_ptr(0),
                sizeof(uint32_t));
          } else {
            current_cmdlist_->buffer_barrier(
                *ti_kernel->get_buffer_bind(bind.buffer));
          }
        }
      }
    }

    current_cmdlist_->bind_pipeline(vp);
    RhiResult status = current_cmdlist_->bind_shader_resources(bindings);
    TI_ERROR_IF(status != RhiResult::success,
                "Resource binding error : RhiResult({})", status);

    if (profiler_) {
      current_cmdlist_->begin_profiler_scope(attribs.name);
    }

    status = current_cmdlist_->dispatch(group_x);

    if (profiler_) {
      current_cmdlist_->end_profiler_scope();
    }

    TI_ERROR_IF(status != RhiResult::success, "Dispatch error : RhiResult({})",
                status);
    ++current_cmdlist_dispatch_count_;
    mark_sparse_list_task_launched(attribs);
    if (attribs.may_mutate_sparse_topology) {
      invalidate_sparse_list_cache(attribs.sparse_mutation_snode_id);
    }
    if (dispatch_cache_) {
      mark_task_writes(attribs);
    } else {
      current_cmdlist_->memory_barrier();
    }
  }

  // Keep context buffers used in this dispatch
  if (ti_kernel->get_args_buffer_size()) {
    if (ctx_buffer_pool_enabled()) {
      pending_pool_.push_back({std::move(args_buffer),
                               ti_kernel->get_args_buffer_size(),
                               AllocUsage::Uniform});
    } else {
      ctx_buffers_.push_back(std::move(args_buffer));
    }
  }
  if (ti_kernel->get_ret_buffer_size()) {
    if (ctx_buffer_pool_enabled()) {
      pending_pool_.push_back({std::move(ret_buffer),
                               ti_kernel->get_ret_buffer_size(),
                               AllocUsage::Storage});
    } else {
      ctx_buffers_.push_back(std::move(ret_buffer));
    }
  }

  // If we need to host sync, sync and remove in-flight references
  if (ctx_blitter) {
    insert_pending_dispatch_barriers();
    if (ctx_blitter->device_to_host(current_cmdlist_.get(), any_arrays,
                                    ext_array_size)) {
      current_cmdlist_ = nullptr;
      current_cmdlist_dispatch_count_ = 0;
      ctx_buffers_.clear();
      // R2.a: device_to_host internally syncs (wait_idle / submit_synced),
      // so all pool buffers are safe to recycle here.
      if (ctx_buffer_pool_enabled()) {
        recycle_pools_to_free();
      }
    }
  }

  submit_current_cmdlist_if_timeout();
}

void GfxRuntime::GraphReplayExecutable::reset() {
  cached_prepared.clear();
  cached_prepare_key.clear();
  slots.clear();
  next_slot = 0;
  device = nullptr;
}

void GfxRuntime::GraphReplayExecutable::bind_device(Device *new_device) {
  if (device != new_device) {
    reset();
    device = new_device;
  }
}

bool GfxRuntime::GraphReplayExecutable::refresh_prepared_cache(
    const std::vector<uint64_t> &key,
    std::vector<PreparedDispatch> &prepared) {
  const bool cache_hit =
      cached_prepare_key == key && cached_prepared.size() == prepared.size();
  if (!cache_hit) {
    cached_prepared.clear();
    cached_prepared.resize(prepared.size());
    for (size_t i = 0; i < prepared.size(); ++i) {
      auto &cache = cached_prepared[i];
      auto &pd = prepared[i];
      cache.any_arrays.clear();
      for (const auto &array_arg : pd.kernel->runtime_array_args()) {
        auto data_it =
            pd.host_ctx->array_ptrs.find(array_arg.data_ptr_indices);
        TI_ASSERT(data_it != pd.host_ctx->array_ptrs.end() &&
                  data_it->second != nullptr);
        DeviceAllocation devalloc =
            *static_cast<DeviceAllocation *>(data_it->second);
        cache.any_arrays[array_arg.indices] = devalloc;
        cache.any_arrays[array_arg.data_ptr_indices] = devalloc;

        auto grad_it =
            pd.host_ctx->array_ptrs.find(array_arg.grad_ptr_indices);
        if (grad_it != pd.host_ctx->array_ptrs.end() &&
            grad_it->second != nullptr) {
          DeviceAllocation grad_alloc =
              *static_cast<DeviceAllocation *>(grad_it->second);
          cache.any_arrays[array_arg.grad_ptr_indices] = grad_alloc;
        }
      }
    }
    cached_prepare_key = key;
  }
  for (size_t i = 0; i < prepared.size(); ++i) {
    prepared[i].any_arrays = &cached_prepared[i].any_arrays;
  }
  return cache_hit;
}

GfxRuntime::GraphReplayExecutable::Slot *
GfxRuntime::GraphReplayExecutable::acquire_ready_slot() {
  if (slots.empty()) {
    slots.resize(kReplaySlots);
  }

  Slot *slot = nullptr;
  size_t slot_index = next_slot;
  for (size_t i = 0; i < slots.size(); ++i) {
    const size_t candidate = (next_slot + i) % slots.size();
    auto &candidate_slot = slots[candidate];
    if (!candidate_slot.completion || candidate_slot.completion->is_ready()) {
      slot = &candidate_slot;
      slot_index = candidate;
      break;
    }
  }
  if (slot != nullptr) {
    next_slot = (slot_index + 1) % slots.size();
    return slot;
  }

  graph_replay_slot_saturation_fallbacks.fetch_add(
      1, std::memory_order_relaxed);
  return nullptr;
}

bool GfxRuntime::GraphReplayExecutable::ready_for_retirement() const {
  for (const auto &slot : slots) {
    if (slot.completion && !slot.completion->is_ready()) {
      return false;
    }
  }
  return true;
}

uint64_t
GfxRuntime::GraphReplayExecutable::known_persistent_argument_bytes() const {
  uint64_t bytes = 0;
  for (const auto &slot : slots) {
    for (size_t size : slot.args_buffer_sizes) {
      bytes += size;
    }
  }
  return bytes;
}

void GfxRuntime::GraphReplayState::reset() {
  executable.reset();
  attempts = 0;
  recorded = 0;
  replayed = 0;
  fallbacks = 0;
  structural_fallbacks = 0;
  runtime_mode_fallbacks = 0;
  slot_saturation_fallbacks = 0;
  last_path = GraphReplayLastPath::none;
  last_fallback_reason = GraphReplayFallbackReason::none;
  diagnostics_enabled = false;
  retirement_requested = false;
}

std::unique_ptr<GraphReplayRegistration> GfxRuntime::register_graph_replay(
    uint64_t replay_token) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  TI_ASSERT(replay_token != 0);
  const uint64_t replay_key = next_graph_replay_registration_id_++;
  TI_ASSERT(replay_key != 0);
  return std::unique_ptr<GraphReplayRegistration>(
      new GraphReplayRegistration(graph_replay_registry_, replay_key));
}

bool GfxRuntime::owns_graph_replay_registration(
    const GraphReplayRegistration &registration) const {
  return registration.registry_.get() == graph_replay_registry_.get();
}

void GfxRuntime::collect_ready_graph_replays() {
  for (auto it = graph_replay_states_.begin();
       it != graph_replay_states_.end();) {
    if (it->second.retirement_requested &&
        it->second.executable.ready_for_retirement()) {
      it = graph_replay_states_.erase(it);
    } else {
      ++it;
    }
  }
}

void GfxRuntime::retire_graph_replay(uint64_t replay_token) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  auto active = graph_replay_states_.find(replay_token);
  if (active == graph_replay_states_.end()) {
    return;
  }
  active->second.retirement_requested = true;
  collect_ready_graph_replays();
}

GraphReplayStats GfxRuntime::debug_graph_replay_stats(
    uint64_t replay_token) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  // A report can opt in before the first launch. Materialize only the tiny
  // host state here (not replay slots or GPU resources) so the next launch
  // observes diagnostics_enabled from its first attempt.
  auto [it, inserted] = graph_replay_states_.try_emplace(replay_token);
  (void)inserted;
  GraphReplayState &state = it->second;
  GraphReplayStats result{
      state.attempts,
      state.recorded,
      state.replayed,
      state.fallbacks,
      state.structural_fallbacks,
      state.runtime_mode_fallbacks,
      state.slot_saturation_fallbacks,
      state.executable.known_persistent_argument_bytes(),
      state.last_path,
      state.last_fallback_reason,
  };
  state.diagnostics_enabled = true;
  return result;
}

bool GfxRuntime::try_launch_graph(
    const std::vector<GraphDispatch> &dispatches,
    uint64_t replay_key,
    RuntimeStatistics *statistics) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  collect_ready_graph_replays();
  TI_ASSERT(replay_key != 0);
  GraphReplayState &state = graph_replay_states_[replay_key];
  TI_ASSERT(!state.retirement_requested);
  GraphReplayExecutable &executable = state.executable;
  if (state.diagnostics_enabled) {
    ++state.attempts;
  }
  state.last_path = GraphReplayLastPath::none;
  state.last_fallback_reason = GraphReplayFallbackReason::none;
  if (dispatches.empty() || profiler_ || dispatch_cache_) {
    if (state.diagnostics_enabled) {
      ++state.fallbacks;
      ++state.runtime_mode_fallbacks;
    }
    state.last_path = GraphReplayLastPath::fallback;
    state.last_fallback_reason = GraphReplayFallbackReason::runtime_mode;
    return false;
  }

  using AllocationMap = GraphReplayExecutable::AllocationMap;
  using PreparedDispatch = GraphReplayExecutable::PreparedDispatch;
  using SizeMap = std::unordered_map<std::vector<int>,
                                     size_t,
                                     hashing::Hasher<std::vector<int>>>;
  using ArgPackMap =
      std::unordered_map<std::vector<int>,
                         const ArgPack *,
                         hashing::Hasher<std::vector<int>>>;

  std::vector<uint64_t> key;
  key.reserve(dispatches.size() * 16);
  std::vector<PreparedDispatch> prepared;
  prepared.reserve(dispatches.size());
  size_t total_tasks = 0;

  auto reject = [&](GraphReplayFallbackReason reason =
                        GraphReplayFallbackReason::structural_unsupported) {
    if (state.diagnostics_enabled) {
      ++state.fallbacks;
      if (reason == GraphReplayFallbackReason::structural_unsupported) {
        ++state.structural_fallbacks;
      }
    }
    state.last_path = GraphReplayLastPath::fallback;
    state.last_fallback_reason = reason;
    return false;
  };

  executable.bind_device(device_);

  for (const GraphDispatch &dispatch : dispatches) {
    if (dispatch.host_ctx == nullptr ||
        dispatch.handle.get_launch_id() >= ti_kernels_.size()) {
      return reject();
    }
    auto *ti_kernel = ti_kernels_[dispatch.handle.get_launch_id()].get();
    if (ti_kernel->get_ret_buffer_size() != 0 ||
        !ti_kernel->runtime_argpack_args().empty()) {
      return reject();
    }

    PreparedDispatch prepared_dispatch;
    prepared_dispatch.kernel = ti_kernel;
    prepared_dispatch.host_ctx = dispatch.host_ctx;

    key.push_back(dispatch.handle.get_launch_id());
    key.push_back(ti_kernel->get_args_buffer_size());
    key.push_back(ti_kernel->num_pipelines());

    for (const auto &array_arg : ti_kernel->runtime_array_args()) {
      const auto alloc_type =
          dispatch.host_ctx->device_allocation_type[array_arg.indices];
      if (alloc_type != LaunchContextBuilder::DevAllocType::kNdarray) {
        return reject();
      }
      auto data_it =
          dispatch.host_ctx->array_ptrs.find(array_arg.data_ptr_indices);
      if (data_it == dispatch.host_ctx->array_ptrs.end() ||
          data_it->second == nullptr) {
        return reject();
      }
      DeviceAllocation devalloc =
          *static_cast<DeviceAllocation *>(data_it->second);
      ndarrays_in_use_.insert(devalloc.alloc_id);
      key.push_back(0xA001u);
      push_graph_allocation_key(key, devalloc);

      auto grad_it =
          dispatch.host_ctx->array_ptrs.find(array_arg.grad_ptr_indices);
      if (grad_it != dispatch.host_ctx->array_ptrs.end() &&
          grad_it->second != nullptr) {
        DeviceAllocation grad_alloc =
            *static_cast<DeviceAllocation *>(grad_it->second);
        ndarrays_in_use_.insert(grad_alloc.alloc_id);
        key.push_back(0xA002u);
        push_graph_allocation_key(key, grad_alloc);
      } else {
        key.push_back(0xA003u);
      }
    }

    const auto &task_attribs = ti_kernel->ti_kernel_attribs().tasks_attribs;
    for (int task_index = 0; task_index < task_attribs.size(); ++task_index) {
      const auto &attribs = task_attribs[task_index];
      if (!attribs.texture_binds.empty() ||
          attribs.task_type == OffloadedTaskType::listgen ||
          attribs.may_mutate_sparse_topology ||
          task_uses_listgen_buffer(attribs)) {
        return reject();
      }
      key.push_back(0xB000u);
      key.push_back(static_cast<uint64_t>(task_index));
      for (const auto &bind : ti_kernel->buffer_binding_plan(task_index)) {
        if (bind.binding < 0) {
          continue;
        }
        switch (bind.kind) {
          case CompiledTaichiKernel::BufferBindingKind::Skip:
          case CompiledTaichiKernel::BufferBindingKind::Args:
          case CompiledTaichiKernel::BufferBindingKind::ExtArrRw:
            break;
          case CompiledTaichiKernel::BufferBindingKind::StaticRw:
            key.push_back(0xB101u);
            key.push_back(static_cast<uint64_t>(bind.binding));
            push_graph_allocation_key(key, bind.static_alloc
                                               ? *bind.static_alloc
                                               : kDeviceNullAllocation);
            break;
          case CompiledTaichiKernel::BufferBindingKind::StaticLookupRw: {
            if (bind.buffer.type == BufferType::ListGen) {
              return reject();
            }
            DeviceAllocation *alloc = ti_kernel->get_buffer_bind(bind.buffer);
            key.push_back(0xB102u);
            key.push_back(static_cast<uint64_t>(bind.binding));
            push_graph_allocation_key(key,
                                      alloc ? *alloc : kDeviceNullAllocation);
            break;
          }
          case CompiledTaichiKernel::BufferBindingKind::ArgPack:
          case CompiledTaichiKernel::BufferBindingKind::RetsRw:
          case CompiledTaichiKernel::BufferBindingKind::ChunkedRwArray:
            return reject();
        }
      }
    }
    total_tasks += task_attribs.size();
    prepared.push_back(std::move(prepared_dispatch));
  }
  if (total_tasks <= 1) {
    return reject(GraphReplayFallbackReason::insufficient_tasks);
  }

  executable.refresh_prepared_cache(key, prepared);

  GfxRuntime::GraphReplayExecutable::Slot *slot =
      executable.acquire_ready_slot();
  if (slot == nullptr) {
    if (state.diagnostics_enabled) {
      ++state.slot_saturation_fallbacks;
    }
    if (statistics != nullptr) {
      statistics->record_graph_slot_saturation_fallback();
    }
    return reject(GraphReplayFallbackReason::slot_saturated);
  }

  const bool is_recapture = slot->recorded || slot->cmdlist != nullptr;
  slot->args_buffers.resize(prepared.size());
  slot->args_buffer_sizes.resize(prepared.size(), 0);
  SizeMap empty_ext_sizes;
  ArgPackMap empty_argpacks;
  for (size_t i = 0; i < prepared.size(); ++i) {
    auto &pd = prepared[i];
    const size_t args_size = pd.kernel->get_args_buffer_size();
    if (args_size > 0 &&
        (!slot->args_buffers[i] || slot->args_buffer_sizes[i] != args_size)) {
      auto [buf, res] = device_->allocate_memory_unique(
          {args_size,
           /*host_write=*/true, /*host_read=*/false,
           /*export_sharing=*/false, AllocUsage::Uniform});
      TI_ASSERT_INFO(res == RhiResult::success,
                     "Failed to allocate Vulkan graph args buffer");
      slot->args_buffers[i] = std::move(buf);
      slot->args_buffer_sizes[i] = args_size;
      slot->recorded = false;
      slot->cmdlist = nullptr;
    }
    pd.args_buffer = slot->args_buffers[i].get();
    auto ctx_blitter = HostDeviceContextBlitter::maybe_make(
        &pd.kernel->ti_kernel_attribs().ctx_attribs,
        &pd.kernel->runtime_array_args(), *pd.host_ctx, device_,
        pd.args_buffer, nullptr);
    if (ctx_blitter) {
      ctx_blitter->host_to_device(*pd.any_arrays, empty_ext_sizes,
                                  empty_argpacks);
    }
  }

  flush_if_pending();

  if (slot->recorded && slot->cmdlist && slot->key == key) {
    slot->completion =
        device_->get_compute_stream()->submit(slot->cmdlist.get());
    if (state.diagnostics_enabled) {
      ++state.replayed;
    }
    if (statistics != nullptr) {
      statistics->record_graph_replay();
    }
    state.last_path = GraphReplayLastPath::replay;
    return true;
  }

  auto [recorded_cmdlist, cmd_res] =
      device_->get_compute_stream()->new_command_list_unique();
  TI_ASSERT_INFO(cmd_res == RhiResult::success,
                 "Failed to allocate Vulkan graph command list");
  auto *cmdlist = recorded_cmdlist.get();
  if (slot->resource_sets.size() < total_tasks) {
    const size_t old_size = slot->resource_sets.size();
    slot->resource_sets.resize(total_tasks);
    for (size_t i = old_size; i < total_tasks; ++i) {
      slot->resource_sets[i] = device_->create_resource_set_unique();
    }
  }

  bool pending_global_barrier = false;
  std::vector<DeviceAllocation> pending_barrier_buffers;
  std::unordered_set<DeviceAllocationId> pending_barrier_ids;
  auto add_barrier = [&](DeviceAllocation alloc) {
    if (alloc == kDeviceNullAllocation || pending_global_barrier) {
      return;
    }
    if (pending_barrier_ids.insert(alloc.alloc_id).second) {
      pending_barrier_buffers.push_back(alloc);
    }
  };
  auto insert_barriers = [&]() {
    if (pending_global_barrier) {
      cmdlist->memory_barrier();
      pending_global_barrier = false;
      pending_barrier_buffers.clear();
      pending_barrier_ids.clear();
      return;
    }
    for (DeviceAllocation alloc : pending_barrier_buffers) {
      cmdlist->buffer_barrier(alloc);
    }
    pending_barrier_buffers.clear();
    pending_barrier_ids.clear();
  };

  size_t resource_set_index = 0;
  for (auto &pd : prepared) {
    const auto &task_attribs = pd.kernel->ti_kernel_attribs().tasks_attribs;
    for (int task_index = 0; task_index < task_attribs.size(); ++task_index) {
      const auto &attribs = task_attribs[task_index];
      insert_barriers();

      ShaderResourceSet *bindings =
          slot->resource_sets[resource_set_index++].get();
      for (const auto &bind : pd.kernel->buffer_binding_plan(task_index)) {
        if (bind.binding < 0) {
          continue;
        }
        switch (bind.kind) {
          case CompiledTaichiKernel::BufferBindingKind::Skip:
            break;
          case CompiledTaichiKernel::BufferBindingKind::StaticRw:
            bindings->rw_buffer(bind.binding,
                                bind.static_alloc ? *bind.static_alloc
                                                  : kDeviceNullAllocation);
            break;
          case CompiledTaichiKernel::BufferBindingKind::StaticLookupRw: {
            DeviceAllocation *alloc = pd.kernel->get_buffer_bind(bind.buffer);
            bindings->rw_buffer(bind.binding,
                                alloc ? *alloc : kDeviceNullAllocation);
            break;
          }
          case CompiledTaichiKernel::BufferBindingKind::ExtArrRw:
            bindings->rw_buffer(bind.binding,
                                pd.any_arrays->at(bind.buffer.root_id));
            break;
          case CompiledTaichiKernel::BufferBindingKind::Args:
            bindings->buffer(bind.binding,
                             pd.args_buffer ? *pd.args_buffer
                                            : kDeviceNullAllocation);
            break;
          case CompiledTaichiKernel::BufferBindingKind::ArgPack:
          case CompiledTaichiKernel::BufferBindingKind::RetsRw:
          case CompiledTaichiKernel::BufferBindingKind::ChunkedRwArray:
            TI_NOT_IMPLEMENTED;
        }
      }

      cmdlist->bind_pipeline(pd.kernel->get_pipeline(task_index));
      RhiResult status = cmdlist->bind_shader_resources(bindings);
      TI_ERROR_IF(status != RhiResult::success,
                  "Vulkan graph resource binding error: RhiResult({})",
                  status);

      const int group_x = (attribs.advisory_total_num_threads +
                           attribs.advisory_num_threads_per_group - 1) /
                          attribs.advisory_num_threads_per_group;
      status = cmdlist->dispatch(group_x);
      TI_ERROR_IF(status != RhiResult::success,
                  "Vulkan graph dispatch error: RhiResult({})", status);

      for (const auto &bind : attribs.buffer_binds) {
        if (!bind.may_write()) {
          continue;
        }
        switch (bind.buffer.type) {
          case BufferType::Args:
          case BufferType::ArgPack:
            break;
          case BufferType::ExtArr:
            add_barrier(pd.any_arrays->at(bind.buffer.root_id));
            break;
          case BufferType::Rets:
          case BufferType::NodeAllocatorPool:
            pending_global_barrier = true;
            break;
          default: {
            DeviceAllocation *alloc = pd.kernel->get_buffer_bind(bind.buffer);
            if (alloc != nullptr) {
              add_barrier(*alloc);
            }
            break;
          }
        }
      }
    }
  }
  insert_barriers();

  slot->key = std::move(key);
  slot->cmdlist = std::move(recorded_cmdlist);
  slot->recorded = true;
  slot->completion = device_->get_compute_stream()->submit(slot->cmdlist.get());
  if (state.diagnostics_enabled) {
    ++state.recorded;
  }
  if (statistics != nullptr) {
    statistics->record_graph_capture();
    if (is_recapture) {
      statistics->record_graph_recapture();
    }
  }
  state.last_path = GraphReplayLastPath::record;
  return true;
}

void GfxRuntime::buffer_copy(DevicePtr dst, DevicePtr src, size_t size) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  ensure_current_cmdlist();
  insert_pending_dispatch_barriers();
  current_cmdlist_->buffer_barrier(src);
  current_cmdlist_->buffer_copy(dst, src, size);
  current_cmdlist_->buffer_barrier(dst);
}

void GfxRuntime::copy_image(DeviceAllocation dst,
                            DeviceAllocation src,
                            const ImageCopyParams &params) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  ensure_current_cmdlist();
  insert_pending_dispatch_barriers();
  transition_image(dst, ImageLayout::transfer_dst);
  transition_image(src, ImageLayout::transfer_src);
  current_cmdlist_->copy_image(dst, src, ImageLayout::transfer_dst,
                               ImageLayout::transfer_src, params);
  transition_image(dst, ImageLayout::transfer_src);
}

DeviceAllocation GfxRuntime::create_image(const ImageParams &params) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  GraphicsDevice *gfx_device = dynamic_cast<GraphicsDevice *>(device_);
  TI_ERROR_IF(gfx_device == nullptr,
              "Image can only be created on a graphics device");
  DeviceAllocation image = gfx_device->create_image(params);
  track_image(image, ImageLayout::undefined);
  last_image_layouts_.at(image.alloc_id) = params.initial_layout;
  return image;
}

void GfxRuntime::track_image(DeviceAllocation image, ImageLayout layout) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  last_image_layouts_[image.alloc_id] = layout;
}
void GfxRuntime::untrack_image(DeviceAllocation image) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  last_image_layouts_.erase(image.alloc_id);
}
void GfxRuntime::transition_image(DeviceAllocation image, ImageLayout layout) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  ImageLayout &last_layout = last_image_layouts_.at(image.alloc_id);
  ensure_current_cmdlist();
  insert_pending_dispatch_barriers();
  current_cmdlist_->image_transition(image, last_layout, layout);
  last_layout = layout;
}

void GfxRuntime::synchronize() {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  synchronize_impl(/*check_hash_overflow=*/true);
}

void GfxRuntime::synchronize_impl(bool check_hash_overflow) {
  flush_if_pending();
  device_->get_compute_stream()->command_sync();
  // The stream is idle, so every retirement-requested state can now release
  // its graph-owned command/resource objects.
  collect_ready_graph_replays();
  // Profiler support
  if (profiler_) {
    device_->profiler_sync();
    auto sampled_records = device_->profiler_flush_sampled_time();
    for (auto &record : sampled_records) {
      profiler_->insert_record(record.first, record.second);
    }
  }
  ctx_buffers_.clear();
  // R2.a: after wait_idle, all submitted/pending pool buffers are GPU-safe.
  if (ctx_buffer_pool_enabled()) {
    recycle_pools_to_free();
  }
  ndarrays_in_use_.clear();
  argpacks_in_use_.clear();
  // Hash SNodes use a static table. Since there is no device-side grow path,
  // overflow must become a host-visible error at the sync boundary instead of
  // silently returning ambient-zero cells.
  if (check_hash_overflow) {
    check_hash_overflow_counters();
  }
  fflush(stdout);
}

StreamSemaphore GfxRuntime::flush() {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  StreamSemaphore sema;
  if (current_cmdlist_) {
    insert_pending_dispatch_barriers();
    sema = device_->get_compute_stream()->submit(current_cmdlist_.get());
    current_cmdlist_ = nullptr;
    current_cmdlist_dispatch_count_ = 0;
    ctx_buffers_.clear();
    if (ctx_buffer_pool_enabled()) {
      flush_pending_pool_to_submitted(sema);
    }
  } else {
    auto [cmdlist, res] =
        device_->get_compute_stream()->new_command_list_unique();
    TI_ASSERT(res == RhiResult::success);
    cmdlist->memory_barrier();
    sema = device_->get_compute_stream()->submit(cmdlist.get());
  }
  current_cmdlist_dispatch_count_ = 0;
  return sema;
}

StreamSemaphore GfxRuntime::flush_if_pending() {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  if (!current_cmdlist_) {
    return nullptr;
  }
  return flush();
}

Device *GfxRuntime::get_ti_device() const {
  return device_;
}

PipelineCache *GfxRuntime::get_backend_cache() const {
  return backend_cache_.get();
}

// G-1: Try to take a buffer from free_pool_ matching (size, usage).
// Returns nullptr if pool disabled or no match. Performs O(N) scan; pool
// capacity is bounded (default 64) so this is acceptable.
std::unique_ptr<DeviceAllocationGuard> GfxRuntime::try_take_pooled_buffer(
    size_t size,
    AllocUsage usage) {
  if (!ctx_buffer_pool_enabled()) {
    return nullptr;
  }
  for (auto it = free_pool_.begin(); it != free_pool_.end(); ++it) {
    if (it->size == size && it->usage == usage) {
      auto guard = std::move(it->guard);
      free_pool_.erase(it);
      return guard;
    }
  }
  return nullptr;
}

bool GfxRuntime::ctx_buffer_pool_enabled() const {
  return ctx_buffer_ring_enabled_;
}

size_t GfxRuntime::count_pooled_buffers(size_t size, AllocUsage usage) const {
  auto count_in = [&](const std::vector<PooledBuffer> &pool) {
    size_t count = 0;
    for (const auto &entry : pool) {
      if (entry.size == size && entry.usage == usage) {
        ++count;
      }
    }
    return count;
  };
  return count_in(pending_pool_) + count_in(submitted_pool_) +
         count_in(free_pool_);
}

std::unique_ptr<DeviceAllocationGuard> GfxRuntime::acquire_ctx_buffer(
    size_t size,
    AllocUsage usage) {
  if (!ctx_buffer_pool_enabled()) {
    return nullptr;
  }
  recycle_completed_pools_to_free();
  if (auto pooled = try_take_pooled_buffer(size, usage)) {
    return pooled;
  }
  if (ctx_buffer_ring_enabled_ &&
      count_pooled_buffers(size, usage) >= ctx_buffer_ring_size_) {
    if (current_cmdlist_) {
      flush();
    }
    recycle_completed_pools_to_free();
    if (auto pooled = try_take_pooled_buffer(size, usage)) {
      return pooled;
    }
    if (wait_for_oldest_submitted_buffer(size, usage)) {
      recycle_completed_pools_to_free();
    } else {
      // Backend does not expose a submission completion token. Fall back to
      // syncing only the compute stream; this is still narrower than the old
      // device_->wait_idle(), which also synced graphics streams.
      device_->get_compute_stream()->command_sync();
      recycle_pools_to_free();
    }
    if (auto pooled = try_take_pooled_buffer(size, usage)) {
      return pooled;
    }
  }
  return nullptr;
}

void GfxRuntime::flush_pending_pool_to_submitted(StreamSemaphore completion) {
  for (auto &entry : pending_pool_) {
    entry.completion = completion;
    submitted_pool_.push_back(std::move(entry));
  }
  pending_pool_.clear();
}

size_t GfxRuntime::recycle_completed_pools_to_free() {
  if (submitted_pool_.empty()) {
    return 0;
  }
  size_t recycled = 0;
  std::vector<PooledBuffer> still_submitted;
  still_submitted.reserve(submitted_pool_.size());
  auto move_to_free = [this, &recycled](PooledBuffer &&entry) {
    entry.completion = nullptr;
    if (free_pool_.size() >= buffer_pool_capacity_) {
      // Drop oldest free entry to bound memory; releases its DeviceAlloc.
      free_pool_.erase(free_pool_.begin());
    }
    free_pool_.push_back(std::move(entry));
    ++recycled;
  };
  for (auto &entry : submitted_pool_) {
    if (entry.completion && entry.completion->is_ready()) {
      move_to_free(std::move(entry));
    } else {
      still_submitted.push_back(std::move(entry));
    }
  }
  submitted_pool_ = std::move(still_submitted);
  return recycled;
}

bool GfxRuntime::wait_for_oldest_submitted_buffer(size_t size,
                                                  AllocUsage usage) {
  for (const auto &entry : submitted_pool_) {
    if (entry.size == size && entry.usage == usage && entry.completion) {
      return entry.completion->wait();
    }
  }
  return false;
}

void GfxRuntime::recycle_pools_to_free() {
  // Caller must guarantee GPU has finished (wait_idle / submit_synced).
  auto move_to_free = [this](std::vector<PooledBuffer> &src) {
    for (auto &entry : src) {
      entry.completion = nullptr;
      if (free_pool_.size() >= buffer_pool_capacity_) {
        // Drop oldest free entry to bound memory; releases its DeviceAlloc.
        free_pool_.erase(free_pool_.begin());
      }
      free_pool_.push_back(std::move(entry));
    }
    src.clear();
  };
  move_to_free(submitted_pool_);
  move_to_free(pending_pool_);
}

void GfxRuntime::ensure_current_cmdlist() {
  // Create new command list if current one is nullptr
  if (!current_cmdlist_) {
    current_cmdlist_pending_since_ = high_res_clock::now();
    current_cmdlist_dispatch_count_ = 0;
    auto [cmdlist, res] =
        device_->get_compute_stream()->new_command_list_unique();
    TI_ASSERT(res == RhiResult::success);
    current_cmdlist_ = std::move(cmdlist);
  }
}

void GfxRuntime::clear_pending_dispatch_barriers() {
  pending_dispatch_global_barrier_ = false;
  pending_dispatch_barrier_buffers_.clear();
  pending_dispatch_barrier_buffer_ids_.clear();
}

bool GfxRuntime::task_uses_listgen_buffer(
    const TaskAttributes &attribs) const {
  for (const auto &bind : attribs.buffer_binds) {
    if (bind.buffer.type == BufferType::ListGen) {
      return true;
    }
  }
  return false;
}

void GfxRuntime::add_pending_dispatch_barrier(DeviceAllocation alloc) {
  if (!dispatch_cache_ || alloc == kDeviceNullAllocation) {
    return;
  }
  if (pending_dispatch_global_barrier_) {
    return;
  }
  if (pending_dispatch_barrier_buffer_ids_.insert(alloc.alloc_id).second) {
    pending_dispatch_barrier_buffers_.push_back(alloc);
  }
}

void GfxRuntime::insert_pending_dispatch_barriers() {
  if (!dispatch_cache_ || !current_cmdlist_) {
    return;
  }
  if (!pending_dispatch_global_barrier_ &&
      pending_dispatch_barrier_buffers_.empty()) {
    return;
  }
  if (pending_dispatch_global_barrier_) {
    current_cmdlist_->memory_barrier();
    clear_pending_dispatch_barriers();
    return;
  }
  for (DeviceAllocation alloc : pending_dispatch_barrier_buffers_) {
    current_cmdlist_->buffer_barrier(alloc);
  }
  clear_pending_dispatch_barriers();
}

void GfxRuntime::submit_current_cmdlist_if_timeout() {
  // If we have accumulated some work but does not require sync
  // and if the accumulated cmdlist has been pending for some time
  // launch the cmdlist to start processing.
  if (current_cmdlist_) {
    if (cmdlist_lazy_submit_enabled_ && debug_mode_) {
      flush();
      return;
    }
    if (cmdlist_lazy_submit_enabled_ &&
        cmdlist_lazy_submit_min_dispatches_ > 0 &&
        current_cmdlist_dispatch_count_ <
            cmdlist_lazy_submit_min_dispatches_) {
      return;
    }
    constexpr uint64_t max_pending_time = 2000;  // 2000us = 2ms
    auto duration = high_res_clock::now() - current_cmdlist_pending_since_;
    if (std::chrono::duration_cast<std::chrono::microseconds>(duration)
            .count() > max_pending_time) {
      flush();
    }
  }
}

void GfxRuntime::ensure_listgen_buffer_bytes(size_t requested_bytes,
                                             const char *reason) {
  requested_bytes = std::max(requested_bytes, kListGenMinBufferSize);
  requested_bytes = align_up_to(requested_bytes, sizeof(uint32_t));
  if (requested_bytes > kListGenMinBufferSize) {
    requested_bytes = align_up_to(requested_bytes, kListGenBufferAlignment);
  }

  if (listgen_buffer_ && requested_bytes <= listgen_buffer_size_) {
    return;
  }
  if (listgen_buffer_ && listgen_explicit_size_) {
    TI_ERROR(
        "Vulkan listgen buffer capacity is fixed at {:.3f} MiB by "
        "vulkan_listgen_buffer_MB, but {} requires {:.3f} MiB. Increase "
        "vulkan_listgen_buffer_MB or unset it and enable "
        "vulkan_listgen_dynamic_size.",
        listgen_buffer_size_ / 1048576.0, reason,
        requested_bytes / 1048576.0);
  }

  const bool replacing_used_buffer = listgen_buffer_ && listgen_buffer_used_;
  if (replacing_used_buffer) {
    synchronize();
  }

  auto [buf, res] = device_->allocate_memory_unique(
      {requested_bytes,
       /*host_write=*/false, /*host_read=*/false,
       /*export_sharing=*/false, AllocUsage::Storage});
  TI_ASSERT_INFO(res == RhiResult::success, "listgen allocation failed");
  listgen_buffer_ = std::move(buf);
  listgen_buffer_size_ = requested_bytes;
  listgen_capacity_entries_ =
      requested_bytes / sizeof(uint32_t) > 0
          ? requested_bytes / sizeof(uint32_t) - 1
          : 0;

  for (auto &kernel : ti_kernels_) {
    kernel->set_listgen_buffer(listgen_buffer_.get());
  }
  if (replacing_used_buffer) {
    clear_sparse_list_cache_resident();
  }
  listgen_buffer_used_ = false;

  Stream *stream = device_->get_compute_stream();
  auto [cmdlist, cmd_res] =
      device_->get_compute_stream()->new_command_list_unique();
  TI_ASSERT(cmd_res == RhiResult::success);
  cmdlist->buffer_fill(listgen_buffer_->get_ptr(0), kBufferSizeEntireSize,
                       /*data=*/0);
  stream->submit_synced(cmdlist.get());

  if (listgen_dynamic_size_ && !listgen_explicit_size_ &&
      requested_bytes > kListGenMinBufferSize) {
    TI_WARN(
        "Vulkan listgen buffer auto-sized to {:.3f} MiB for {} "
        "(capacity {} entries). Set ti.init(vulkan_listgen_buffer_MB=32) "
        "to force the legacy 32 MiB capacity.",
        requested_bytes / 1048576.0, reason, listgen_capacity_entries_);
  }
}

void GfxRuntime::ensure_listgen_capacity_entries(size_t requested_entries,
                                                 const char *reason) {
  if (requested_entries <= listgen_capacity_entries_) {
    return;
  }
  if (!listgen_dynamic_size_ || listgen_explicit_size_) {
    TI_ERROR(
        "Vulkan listgen buffer capacity {} entries ({:.3f} MiB) is smaller "
        "than {} required entries for {}. Set ti.init("
        "vulkan_listgen_buffer_MB=...) high enough or enable "
        "vulkan_listgen_dynamic_size=True for auto sizing.",
        listgen_capacity_entries_, listgen_buffer_size_ / 1048576.0,
        requested_entries, reason);
  }
  const size_t required_words =
      requested_entries + 1 + kListGenAutoSlackEntries;
  ensure_listgen_buffer_bytes(required_words * sizeof(uint32_t), reason);
}

void GfxRuntime::ensure_listgen_capacity_for_kernel(
    const CompiledTaichiKernel &kernel) {
  size_t requested_entries = 0;
  for (const auto &attribs : kernel.ti_kernel_attribs().tasks_attribs) {
    if (attribs.task_type == OffloadedTaskType::listgen) {
      requested_entries = std::max(
          requested_entries,
          static_cast<size_t>(std::max(attribs.advisory_total_num_threads, 0)));
    }
  }
  if (requested_entries > 0) {
    ensure_listgen_capacity_entries(requested_entries, "kernel launch");
  }
}

void GfxRuntime::update_listgen_buffer_for_snode_tree(
    const CompiledSNodeStructs &compiled_structs) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  const size_t requested_entries = estimate_listgen_entries(compiled_structs);
  if (requested_entries > 0) {
    ensure_listgen_capacity_entries(requested_entries,
                                    "SNodeTree materialization");
  }
}

void GfxRuntime::init_nonroot_buffers() {
  {
    auto [buf, res] = device_->allocate_memory_unique(
        {kGtmpBufferSize,
         /*host_write=*/false, /*host_read=*/false,
         /*export_sharing=*/false, AllocUsage::Storage});
    TI_ASSERT_INFO(res == RhiResult::success, "gtmp allocation failed");
    global_tmps_buffer_ = std::move(buf);
  }
  {
    auto [buf, res] = device_->allocate_memory_unique(
        {kHashOverflowBufferSize,
         /*host_write=*/false, /*host_read=*/true,
         /*export_sharing=*/false, AllocUsage::Storage});
    TI_ASSERT_INFO(res == RhiResult::success,
                   "hash overflow diagnostics allocation failed");
    hash_overflow_buffer_ = std::move(buf);
  }

  if (listgen_initial_buffer_size_ > 0) {
    ensure_listgen_buffer_bytes(listgen_initial_buffer_size_, "initialization");
  }

  // Need to zero fill the buffers, otherwise there could be NaN.
  Stream *stream = device_->get_compute_stream();
  auto [cmdlist, res] =
      device_->get_compute_stream()->new_command_list_unique();
  TI_ASSERT(res == RhiResult::success);

  cmdlist->buffer_fill(global_tmps_buffer_->get_ptr(0), kBufferSizeEntireSize,
                       /*data=*/0);
  cmdlist->buffer_fill(hash_overflow_buffer_->get_ptr(0),
                       kBufferSizeEntireSize, /*data=*/0);
  stream->submit_synced(cmdlist.get());
}

void GfxRuntime::add_root_buffer(size_t root_buffer_size) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  add_root_buffer(static_cast<int>(root_buffers_.size()), root_buffer_size);
}

void GfxRuntime::add_root_buffer(int root_id, size_t root_buffer_size) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  TI_ERROR_IF(root_id < 0 ||
                  static_cast<std::size_t>(root_id) > root_buffers_.size(),
              "Cannot install root buffer at invalid id {}.", root_id);
  TI_ERROR_IF(static_cast<std::size_t>(root_id) < root_buffers_.size() &&
                  root_buffers_[root_id],
              "Cannot replace live root buffer id {}.", root_id);
  if (root_buffer_size == 0) {
    root_buffer_size = 4;  // there might be empty roots
  }
  auto [new_buffer, res_buffer] = device_->allocate_memory_unique(
      {root_buffer_size,
       /*host_write=*/false, /*host_read=*/false,
       /*export_sharing=*/false, AllocUsage::Storage});
  TI_ASSERT_INFO(res_buffer == RhiResult::success,
                 "Failed to allocate root buffer");
  Stream *stream = device_->get_compute_stream();
  auto [cmdlist, res_cmdlist] =
      device_->get_compute_stream()->new_command_list_unique();
  TI_ASSERT(res_cmdlist == RhiResult::success);
  cmdlist->buffer_fill(new_buffer->get_ptr(0), kBufferSizeEntireSize,
                       /*data=*/0);
  stream->submit_synced(cmdlist.get());
  if (static_cast<std::size_t>(root_id) == root_buffers_.size()) {
    root_buffers_.push_back(std::move(new_buffer));
  } else {
    root_buffers_[root_id] = std::move(new_buffer);
  }
  // cache the root buffer size
  root_buffers_size_map_[root_buffers_[root_id].get()] = root_buffer_size;
}

void GfxRuntime::remove_root_buffer(int root_id) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  TI_ERROR_IF(root_id < 0 ||
                  static_cast<std::size_t>(root_id) >= root_buffers_.size() ||
                  !root_buffers_[root_id],
              "Cannot remove missing root buffer id {}.", root_id);
  root_buffers_size_map_.erase(root_buffers_[root_id].get());
  root_buffers_[root_id].reset();
  hash_overflow_watches_.erase(
      std::remove_if(hash_overflow_watches_.begin(),
                     hash_overflow_watches_.end(),
                     [root_id](const HashOverflowWatch &watch) {
                       return watch.root_id == root_id;
                     }),
      hash_overflow_watches_.end());
}

DeviceAllocation *GfxRuntime::get_root_buffer(int id) const {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  if (id < 0 || static_cast<size_t>(id) >= root_buffers_.size() ||
      !root_buffers_[id]) {
    TI_ERROR("root buffer id {} not found", id);
  }
  return root_buffers_[id].get();
}

size_t GfxRuntime::get_root_buffer_size(int id) const {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  if (id < 0 || static_cast<size_t>(id) >= root_buffers_.size() ||
      !root_buffers_[id]) {
    TI_ERROR("root buffer id {} not found", id);
  }
  auto it = root_buffers_size_map_.find(root_buffers_[id].get());
  if (it == root_buffers_size_map_.end()) {
    TI_ERROR("root buffer id {} not found", id);
  }
  return it->second;
}

void GfxRuntime::enqueue_compute_op_lambda(
    std::function<void(Device *device, CommandList *cmdlist)> op,
    const std::vector<ComputeOpImageRef> &image_refs) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  for (const auto &ref : image_refs) {
    TI_ASSERT(last_image_layouts_.find(ref.image.alloc_id) !=
              last_image_layouts_.end());
    transition_image(ref.image, ref.initial_layout);
  }

  ensure_current_cmdlist();
  insert_pending_dispatch_barriers();
  op(device_, current_cmdlist_.get());

  for (const auto &ref : image_refs) {
    last_image_layouts_[ref.image.alloc_id] = ref.final_layout;
  }
}

GfxRuntime::RegisterParams run_codegen(
    Kernel *kernel,
    Arch arch,
    const DeviceCapabilityConfig &caps,
    const std::vector<CompiledSNodeStructs> &compiled_structs,
    const CompileConfig &compile_config) {
  const auto id = Program::get_kernel_id();
  const auto taichi_kernel_name(fmt::format("{}_k{:04d}_vk", kernel->name, id));
  TI_TRACE("VK codegen for Taichi kernel={}", taichi_kernel_name);
  spirv::KernelCodegen::Params params;
  params.ti_kernel_name = taichi_kernel_name;
  params.kernel = kernel;
  params.ir_root = kernel->ir.get();
  params.compiled_structs = compiled_structs;
  params.arch = arch;
  params.caps = caps;
  params.enable_spv_opt = compile_config.external_optimization_level > 0;
  params.vulkan_listgen_reuse = compile_config.vulkan_listgen_reuse;
  spirv::KernelCodegen codegen(params);
  GfxRuntime::RegisterParams res;
  codegen.run(res.kernel_attribs, res.task_spirv_source_codes);
  res.num_snode_trees = compiled_structs.size();
  return res;
}

std::pair<const lang::StructType *, size_t>
GfxRuntime::get_struct_type_with_data_layout(const lang::StructType *old_ty,
                                             const std::string &layout) {
  auto [new_ty, size, align] =
      get_struct_type_with_data_layout_impl(old_ty, layout);
  return {new_ty, size};
}

std::tuple<const lang::StructType *, size_t, size_t>
GfxRuntime::get_struct_type_with_data_layout_impl(
    const lang::StructType *old_ty,
    const std::string &layout) {
  TI_TRACE("get_struct_type_with_data_layout: {}", layout);
  TI_ASSERT(layout.size() == 2);
  auto is_430 = layout[0] == '4';
  auto has_buffer_ptr = layout[1] == 'b';
  auto members = old_ty->elements();
  size_t bytes = 0;
  size_t align = 0;
  for (int i = 0; i < members.size(); i++) {
    auto &member = members[i];
    size_t member_align;
    size_t member_size;
    if (auto struct_type = member.type->cast<lang::StructType>()) {
      auto [new_ty, size, member_align_] =
          get_struct_type_with_data_layout_impl(struct_type, layout);
      members[i].type = new_ty;
      member_align = member_align_;
      member_size = size;
    } else if (auto tensor_type = member.type->cast<lang::TensorType>()) {
      size_t element_size = data_type_size_gfx(tensor_type->get_element_type());
      size_t num_elements = tensor_type->get_num_elements();
      if (!is_430) {
        if (num_elements == 2) {
          member_align = element_size * 2;
        } else {
          member_align = element_size * 4;
        }
        member_size = member_align;
      } else {
        member_align = element_size;
        member_size = tensor_type->get_num_elements() * element_size;
      }
    } else if (auto pointer_type = member.type->cast<PointerType>()) {
      if (has_buffer_ptr) {
        member_size = sizeof(uint64_t);
        member_align = member_size;
      } else {
        // Use u32 as placeholder
        member_size = sizeof(uint32_t);
        member_align = member_size;
      }
    } else {
      TI_ASSERT(member.type->is<PrimitiveType>());
      member_size = data_type_size_gfx(member.type);
      member_align = member_size;
    }
    bytes = align_up(bytes, member_align);
    members[i].offset = bytes;
    bytes += member_size;
    align = std::max(align, member_align);
  }

  if (!is_430) {
    align = align_up(align, sizeof(float) * 4);
    bytes = align_up(bytes, 4 * sizeof(float));
  }
  TI_TRACE("  total_bytes={}", bytes);
  return {TypeFactory::get_instance()
              .get_struct_type(members, layout)
              ->as<lang::StructType>(),
          bytes, align};
}

}  // namespace gfx
}  // namespace taichi::lang
