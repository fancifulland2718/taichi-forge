#include "taichi/runtime/gfx/runtime.h"
#include "taichi/program/program.h"
#include "taichi/common/filesystem.hpp"

// FIXME: (penguinliong) Special offer for `run_codegen`. Find a new home for it
// in the future.
#include "taichi/codegen/spirv/spirv_codegen.h"

#include <algorithm>
#include <chrono>
#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "fp16.h"

#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

namespace taichi::lang {
namespace gfx {

namespace {

class HostDeviceContextBlitter {
 public:
  HostDeviceContextBlitter(const KernelContextAttributes *ctx_attribs,
                           LaunchContextBuilder &host_ctx,
                           Device *device,
                           DeviceAllocation *device_args_buffer,
                           DeviceAllocation *device_ret_buffer)
      : ctx_attribs_(ctx_attribs),
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

    for (int i = 0; i < ctx_attribs_->args().size(); ++i) {
      const auto &arg_kv = ctx_attribs_->args()[i];
      const auto &indices = arg_kv.first;
      const auto &arg = arg_kv.second;
      if (arg.is_array) {
        if (host_ctx_.device_allocation_type[indices] ==
                LaunchContextBuilder::DevAllocType::kNone &&
            ext_arr_size.at(indices)) {
          // Only need to blit ext arrs (host array)
          auto access_it = std::find_if(ctx_attribs_->arr_access.begin(),
                                        ctx_attribs_->arr_access.end(),
                                        [indices](const auto &pair) -> bool {
                                          return pair.first == indices;
                                        });
          TI_ASSERT(access_it != ctx_attribs_->arr_access.end());
          uint32_t access = uint32_t(access_it->second);
          // Bug B/C fix (forge 2026-05): always blit host→device for ext
          // arrs regardless of ExternalPtrAccess flag. The previous READ-only
          // optimization left WRITE-only device buffers uninitialized with
          // recycled GPU memory; for kernels whose struct-for over a sparse
          // SNode only writes a subset of cells (e.g. tensor_to_ext_arr from
          // to_numpy on bitmasked/pointer fields after deactivate), the
          // unwritten cells leaked stale data back to the user. The host
          // ndarray (e.g. np.zeros from to_numpy) is the user's contract for
          // the initial state of the buffer; device buffer must match it.
          (void)access;
          {
            DeviceAllocation buffer = ext_arrays.at(indices);
            void *device_arr_ptr{nullptr};
            TI_ASSERT(device_->map(buffer, &device_arr_ptr) ==
                      RhiResult::success);
            auto data_ptr_idx = indices;
            data_ptr_idx.push_back(TypeFactory::DATA_PTR_POS_IN_NDARRAY);
            const void *host_ptr = host_ctx_.array_ptrs[data_ptr_idx];
            std::memcpy(device_arr_ptr, host_ptr, ext_arr_size.at(indices));
            device_->unmap(buffer);
          }
        }
        // Substitute in the device address.

        if ((host_ctx_.device_allocation_type[indices] ==
                 LaunchContextBuilder::DevAllocType::kNone ||
             host_ctx_.device_allocation_type[indices] ==
                 LaunchContextBuilder::DevAllocType::kNdarray) &&
            device_->get_caps().get(
                DeviceCapability::spirv_has_physical_storage_buffer)) {
          uint64_t addr =
              device_->get_memory_physical_pointer(ext_arrays.at(indices));
          auto grad_ptr_idx = indices;
          grad_ptr_idx.push_back(TypeFactory::GRAD_PTR_POS_IN_NDARRAY);
          host_ctx_.set_ndarray_ptrs(
              indices, addr, (uint64)host_ctx_.array_ptrs[grad_ptr_idx]);
        }
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

    for (int i = 0; i < ctx_attribs_->args().size(); ++i) {
      const auto &kv = ctx_attribs_->args()[i];
      const auto &indices = kv.first;
      const auto &arg = kv.second;
      if (arg.is_array &&
          host_ctx_.device_allocation_type[indices] ==
              LaunchContextBuilder::DevAllocType::kNone &&
          ext_arr_size.at(indices)) {
        auto access_it = std::find_if(ctx_attribs_->arr_access.begin(),
                                      ctx_attribs_->arr_access.end(),
                                      [indices](const auto &pair) -> bool {
                                        return pair.first == indices;
                                      });
        TI_ASSERT(access_it != ctx_attribs_->arr_access.end());
        uint32_t access = uint32_t(access_it->second);
        if (access & uint32_t(irpass::ExternalPtrAccess::WRITE)) {
          // Only need to blit ext arrs (host array)
          readback_dev_ptrs.push_back(ext_arrays.at(indices).get_ptr(0));
          auto data_ptr_idx = indices;
          data_ptr_idx.push_back(TypeFactory::DATA_PTR_POS_IN_NDARRAY);
          readback_host_ptrs.push_back(host_ctx_.array_ptrs[data_ptr_idx]);
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
      LaunchContextBuilder &host_ctx,
      Device *device,
      DeviceAllocation *device_args_buffer,
      DeviceAllocation *device_ret_buffer) {
    if (ctx_attribs->empty()) {
      return nullptr;
    }
    return std::make_unique<HostDeviceContextBlitter>(
        ctx_attribs, host_ctx, device, device_args_buffer, device_ret_buffer);
  }

 private:
  const KernelContextAttributes *const ctx_attribs_;
  LaunchContextBuilder &host_ctx_;
  DeviceAllocation *const device_args_buffer_;
  DeviceAllocation *const device_ret_buffer_;
  Device *const device_;
};

}  // namespace

constexpr size_t kGtmpBufferSize = 1024 * 1024;
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
         snode->type == SNodeType::dynamic;
}

size_t estimate_listgen_entries(const CompiledSNodeStructs &compiled_structs) {
  size_t result = 0;
  for (const auto &[sid, desc] : compiled_structs.snode_descriptors) {
    (void)sid;
    if (snode_can_use_spirv_listgen(desc.snode)) {
      result = std::max(result, desc.total_num_cells_from_root);
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

  const auto &task_attribs = ti_kernel_attribs_.tasks_attribs;
  const auto &spirv_bins = ti_params.spirv_bins;
  TI_ASSERT(task_attribs.size() == spirv_bins.size());
  cached_resource_sets_.resize(task_attribs.size());

  for (int i = 0; i < task_attribs.size(); ++i) {
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
      buffer_pool_enabled_(params.enable_buffer_pool),
      buffer_pool_capacity_(
          params.buffer_pool_capacity > 0
              ? static_cast<size_t>(params.buffer_pool_capacity)
              : size_t{64}),
      listgen_dynamic_size_(params.listgen_dynamic_size),
      listgen_explicit_size_(params.listgen_buffer_MB > 0),
      dispatch_cache_(params.dispatch_cache),
      listgen_lite_barrier_(params.listgen_lite_barrier),
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
  if (std::filesystem::exists(cache_path)) {
    TI_TRACE("Loading pipeline cache from {}", cache_path.generic_string());
    std::ifstream cache_file(cache_path, std::ios::binary);
    cache_data.assign(std::istreambuf_iterator<char>(cache_file),
                      std::istreambuf_iterator<char>());
  } else {
    TI_TRACE("Pipeline cache not found at {}", cache_path.generic_string());
  }
  auto [cache, res] = device_->create_pipeline_cache_unique(cache_data.size(),
                                                            cache_data.data());
  if (res == RhiResult::success) {
    backend_cache_ = std::move(cache);
  }
}

GfxRuntime::~GfxRuntime() {
  synchronize();

  // Write pipeline cache back to disk.
  if (backend_cache_) {
    uint8_t *cache_data = (uint8_t *)backend_cache_->data();
    size_t cache_size = backend_cache_->size();
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

GfxRuntime::KernelHandle GfxRuntime::register_taichi_kernel(
    GfxRuntime::RegisterParams reg_params) {
  CompiledTaichiKernel::Params params;
  params.ti_kernel_attribs = &(reg_params.kernel_attribs);
  params.num_snode_trees = reg_params.num_snode_trees;
  params.device = device_;
  params.root_buffers = {};
  for (int root = 0; root < root_buffers_.size(); ++root) {
    params.root_buffers.push_back(root_buffers_[root].get());
  }
  params.global_tmps_buffer = global_tmps_buffer_.get();
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
  ti_kernels_.push_back(std::make_unique<CompiledTaichiKernel>(params));
  return res;
}

void GfxRuntime::launch_kernel(KernelHandle handle,
                               LaunchContextBuilder &host_ctx) {
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
      &ti_kernel->ti_kernel_attribs().ctx_attribs, host_ctx, device_,
      args_buffer.get(), ret_buffer.get());

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

  // Prepare context buffers & arrays
  if (ctx_blitter) {
    TI_ASSERT(ti_kernel->get_args_buffer_size() ||
              ti_kernel->get_ret_buffer_size());

    const auto &args = ti_kernel->ti_kernel_attribs().ctx_attribs.args();
    for (auto &kv : args) {
      const auto &indices = kv.first;
      const auto &arg = kv.second;
      if (arg.is_array) {
        if (host_ctx.device_allocation_type[indices] !=
            LaunchContextBuilder::DevAllocType::kNone) {
          DeviceAllocation devalloc = kDeviceNullAllocation;
          auto data_ptr_indices = indices;
          data_ptr_indices.push_back(TypeFactory::DATA_PTR_POS_IN_NDARRAY);
          // NDArray
          if (host_ctx.array_ptrs.count(data_ptr_indices)) {
            devalloc =
                *(DeviceAllocation *)(host_ctx.array_ptrs[data_ptr_indices]);
          }
          // Texture
          if (host_ctx.array_ptrs.count(indices)) {
            devalloc = *(DeviceAllocation *)(host_ctx.array_ptrs[indices]);
          }

          if (host_ctx.device_allocation_type[indices] ==
              LaunchContextBuilder::DevAllocType::kNdarray) {
            any_arrays[indices] = devalloc;
            ndarrays_in_use_.insert(devalloc.alloc_id);
          } else if (host_ctx.device_allocation_type[indices] ==
                     LaunchContextBuilder::DevAllocType::kTexture) {
            textures[indices] = devalloc;
          } else if (host_ctx.device_allocation_type[indices] ==
                     LaunchContextBuilder::DevAllocType::kRWTexture) {
            textures[indices] = devalloc;
          } else {
            TI_NOT_IMPLEMENTED;
          }
        } else {
          ext_array_size[indices] = host_ctx.array_runtime_sizes[indices];
          auto arr_access =
              ti_kernel->ti_kernel_attribs().ctx_attribs.arr_access;
          auto access_it = std::find_if(arr_access.begin(), arr_access.end(),
                                        [indices](const auto &pair) -> bool {
                                          return pair.first == indices;
                                        });
          TI_ASSERT(access_it != arr_access.end());
          uint32_t access = uint32_t(access_it->second);
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
          ctx_buffers_.push_back(std::move(allocated));
        }
      }
    }

    auto argpack_types =
        ti_kernel->ti_kernel_attribs().ctx_attribs.argpack_types();
    for (const auto &kv : argpack_types) {
      const auto &indices = kv.first;
      TI_ASSERT(host_ctx.device_allocation_type[indices] ==
                LaunchContextBuilder::DevAllocType::kArgPack);
      TI_ASSERT(host_ctx.argpack_ptrs.count(indices));
      const ArgPack *argpack = host_ctx.argpack_ptrs[indices];
      DeviceAllocation devalloc = argpack->get_device_allocation();
      argpacks_in_use_.insert(devalloc.alloc_id);
      argpacks[indices] = argpack;
    }

    ctx_blitter->host_to_device(any_arrays, ext_array_size, argpacks);
  }

  ensure_current_cmdlist();

  // Record commands
  const auto &task_attribs = ti_kernel->ti_kernel_attribs().tasks_attribs;

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
    for (auto &bind : attribs.buffer_binds) {
      // We might have to bind a invalid buffer (this is fine as long as
      // shader don't do anything with it)
      if (bind.buffer.type == BufferType::ExtArr) {
        bindings->rw_buffer(bind.binding, any_arrays.at(bind.buffer.root_id));
      } else if (bind.buffer.type == BufferType::Args) {
        bindings->buffer(bind.binding,
                         args_buffer ? *args_buffer : kDeviceNullAllocation);
      } else if (bind.buffer.type == BufferType::ArgPack) {
        DeviceAllocation alloc =
            argpacks.at(bind.buffer.root_id)->get_device_allocation();
        bindings->buffer(bind.binding, alloc);
      } else if (bind.buffer.type == BufferType::Rets) {
        bindings->rw_buffer(bind.binding,
                            ret_buffer ? *ret_buffer : kDeviceNullAllocation);
      } else {
        // C-2.5 (2026-05): chunked NodeAllocatorPool 走 rw_buffer_array 路径。
        // attribs 端 BufferBind.chunk_count > 0 标记此 binding 是 descriptor
        // array of N storage buffers；runtime 用 GfxRuntime allocator 端的
        // chunks() 列表（已在 ctor 时拷入 chunk_arrays_）一次性 bind 全部
        // N 个 DeviceAllocation。chunk_count == 0 走原 rw_buffer 单 buffer 路径。
        if (bind.chunk_count > 0u) {
          if (auto *chunks = ti_kernel->get_chunk_array(bind.buffer)) {
            TI_ASSERT_INFO(
                chunks->size() == bind.chunk_count,
                "C-2.5: chunk_count mismatch sid={}, attribs={} runtime={}",
                bind.buffer.root_id, bind.chunk_count, chunks->size());
            bindings->rw_buffer_array(bind.binding, *chunks);
            continue;
          }
          // Fallback (no chunked allocator registered): bind first chunk
          // only via rw_buffer; SPIR-V will read chunk[0] safely (single
          // chunk path was byte-equivalent under C-2.4.a Commit A).
        }
        DeviceAllocation *alloc = ti_kernel->get_buffer_bind(bind.buffer);
        bindings->rw_buffer(bind.binding,
                            alloc ? *alloc : kDeviceNullAllocation);
      }
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
          if (dispatch_cache_ || listgen_lite_barrier_) {
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

void GfxRuntime::buffer_copy(DevicePtr dst, DevicePtr src, size_t size) {
  ensure_current_cmdlist();
  insert_pending_dispatch_barriers();
  current_cmdlist_->buffer_barrier(src);
  current_cmdlist_->buffer_copy(dst, src, size);
  current_cmdlist_->buffer_barrier(dst);
}

void GfxRuntime::copy_image(DeviceAllocation dst,
                            DeviceAllocation src,
                            const ImageCopyParams &params) {
  ensure_current_cmdlist();
  insert_pending_dispatch_barriers();
  transition_image(dst, ImageLayout::transfer_dst);
  transition_image(src, ImageLayout::transfer_src);
  current_cmdlist_->copy_image(dst, src, ImageLayout::transfer_dst,
                               ImageLayout::transfer_src, params);
  transition_image(dst, ImageLayout::transfer_src);
}

DeviceAllocation GfxRuntime::create_image(const ImageParams &params) {
  GraphicsDevice *gfx_device = dynamic_cast<GraphicsDevice *>(device_);
  TI_ERROR_IF(gfx_device == nullptr,
              "Image can only be created on a graphics device");
  DeviceAllocation image = gfx_device->create_image(params);
  track_image(image, ImageLayout::undefined);
  last_image_layouts_.at(image.alloc_id) = params.initial_layout;
  return image;
}

void GfxRuntime::track_image(DeviceAllocation image, ImageLayout layout) {
  last_image_layouts_[image.alloc_id] = layout;
}
void GfxRuntime::untrack_image(DeviceAllocation image) {
  last_image_layouts_.erase(image.alloc_id);
}
void GfxRuntime::transition_image(DeviceAllocation image, ImageLayout layout) {
  ImageLayout &last_layout = last_image_layouts_.at(image.alloc_id);
  ensure_current_cmdlist();
  insert_pending_dispatch_barriers();
  current_cmdlist_->image_transition(image, last_layout, layout);
  last_layout = layout;
}

void GfxRuntime::synchronize() {
  flush();
  device_->wait_idle();
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
  fflush(stdout);
}

StreamSemaphore GfxRuntime::flush() {
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

Device *GfxRuntime::get_ti_device() const {
  return device_;
}

// R2.a: Try to take a buffer from free_pool_ matching (size, usage).
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
      ++buffer_pool_hits_;
      return guard;
    }
  }
  ++buffer_pool_misses_;
  return nullptr;
}

bool GfxRuntime::ctx_buffer_pool_enabled() const {
  return buffer_pool_enabled_ || ctx_buffer_ring_enabled_;
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
  stream->submit_synced(cmdlist.get());
}

void GfxRuntime::add_root_buffer(size_t root_buffer_size) {
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
  root_buffers_.push_back(std::move(new_buffer));
  // cache the root buffer size
  root_buffers_size_map_[root_buffers_.back().get()] = root_buffer_size;
}

DeviceAllocation *GfxRuntime::get_root_buffer(int id) const {
  if (id >= root_buffers_.size()) {
    TI_ERROR("root buffer id {} not found", id);
  }
  return root_buffers_[id].get();
}

size_t GfxRuntime::get_root_buffer_size(int id) const {
  auto it = root_buffers_size_map_.find(root_buffers_[id].get());
  if (id >= root_buffers_.size() || it == root_buffers_size_map_.end()) {
    TI_ERROR("root buffer id {} not found", id);
  }
  return it->second;
}

void GfxRuntime::enqueue_compute_op_lambda(
    std::function<void(Device *device, CommandList *cmdlist)> op,
    const std::vector<ComputeOpImageRef> &image_refs) {
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
