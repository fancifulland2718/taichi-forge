#include "taichi/runtime/cuda/kernel_launcher.h"
#include "taichi/rhi/cuda/cuda_context.h"

namespace taichi::lang {
namespace cuda {

bool KernelLauncher::on_cuda_device(void *ptr) {
  unsigned int attr_val = 0;
  uint32_t ret_code = CUDADriver::get_instance().mem_get_attribute.call(
      &attr_val, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (void *)ptr);

  return ret_code == CUDA_SUCCESS && attr_val == CU_MEMORYTYPE_DEVICE;
}

int64 KernelLauncher::get_sparse_list_version(int snode_id) const {
  auto it = sparse_list_states_.find(snode_id);
  if (it == sparse_list_states_.end()) {
    return 0;
  }
  return it->second.version;
}

bool KernelLauncher::sparse_list_task_is_current(const OffloadedTask &task) {
  if (task.sparse_list_op == OffloadedTask::kSparseListOpNone ||
      task.sparse_list_snode_id < 0 ||
      task.sparse_list_parent_snode_id < 0) {
    return false;
  }

  auto it = sparse_list_states_.find(task.sparse_list_snode_id);
  if (it == sparse_list_states_.end()) {
    if (listgen_reuse_adaptive_) {
      auto &state = sparse_list_states_[task.sparse_list_snode_id];
      record_sparse_list_reuse_sample(state, /*would_skip=*/false);
    }
    return false;
  }
  auto &state = it->second;
  const bool would_skip = state.clean_epoch == state.dirty_epoch &&
                 state.clean_parent_version ==
                   get_sparse_list_version(
                     task.sparse_list_parent_snode_id);
  record_sparse_list_reuse_sample(state, would_skip);
  return would_skip && !state.adaptive_disabled;
}

void KernelLauncher::record_sparse_list_reuse_sample(SparseListState &state,
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

void KernelLauncher::mark_sparse_list_task_launched(
    const OffloadedTask &task) {
  if (task.sparse_list_op != OffloadedTask::kSparseListOpListgen ||
      task.sparse_list_snode_id < 0 ||
      task.sparse_list_parent_snode_id < 0) {
    return;
  }

  auto &state = sparse_list_states_[task.sparse_list_snode_id];
  state.clean_epoch = state.dirty_epoch;
  state.clean_parent_version =
      get_sparse_list_version(task.sparse_list_parent_snode_id);
  state.version++;
}

void KernelLauncher::invalidate_sparse_list_cache(
    int sparse_mutation_snode_id) {
  if (sparse_mutation_snode_id >= 0) {
    sparse_list_states_[sparse_mutation_snode_id].dirty_epoch++;
    return;
  }
  for (auto &kv : sparse_list_states_) {
    kv.second.dirty_epoch++;
  }
}

void KernelLauncher::launch_llvm_kernel(Handle handle,
                                        LaunchContextBuilder &ctx) {
  TI_ASSERT(handle.get_launch_id() < contexts_.size());
  auto launcher_ctx = contexts_[handle.get_launch_id()];
  auto *executor = get_runtime_executor();
  listgen_reuse_adaptive_ =
      executor->get_config().cuda_listgen_reuse_adaptive;
  auto *cuda_module = launcher_ctx.jit_module;
  const auto &parameters = launcher_ctx.parameters;
  const auto &offloaded_tasks = launcher_ctx.offloaded_tasks;

  CUDAContext::get_instance().make_current();

  // |transfers| is only used for external arrays whose data is originally on
  // host. They are first transferred onto device and that device pointer is
  // stored in |device_ptrs| below. |transfers| saves its original pointer so
  // that we can copy the data back once kernel finishes. as well as the
  // temporary device allocations, which can be freed after kernel finishes. Key
  // is [arg_id, ptr_pos], where ptr_pos is TypeFactory::DATA_PTR_POS_IN_NDARRAY
  // for data_ptr and TypeFactory::GRAD_PTR_POS_IN_NDARRAY for grad_ptr. Value
  // is [host_ptr, temporary_device_alloc]. Invariant: temp_devallocs.size() !=
  // 0 <==> transfer happened.
  std::unordered_map<std::vector<int>, std::pair<void *, DeviceAllocation>,
                     hashing::Hasher<std::vector<int>>>
      transfers;

  // |device_ptrs| stores pointers on device for all arrays args, including
  // external arrays and ndarrays, no matter whether the data is originally on
  // device or host.
  // This is the source of truth for us to look for device pointers used in CUDA
  // kernels.
  std::unordered_map<std::vector<int>, void *,
                     hashing::Hasher<std::vector<int>>>
      device_ptrs;

  char *device_result_buffer{nullptr};
  CUDADriver::get_instance().malloc_async(
      (void **)&device_result_buffer,
      std::max(ctx.result_buffer_size, sizeof(uint64)), nullptr);
  ctx.get_context().runtime = executor->get_llvm_runtime();

  for (int i = 0; i < (int)parameters.size(); i++) {
    const auto &kv = parameters[i];
    const auto &key = kv.first;
    const auto &parameter = kv.second;
    if (parameter.is_array) {
      const auto arr_sz = ctx.array_runtime_sizes[key];
      // Note: both numpy and PyTorch support arrays/tensors with zeros
      // in shapes, e.g., shape=(0) or shape=(100, 0, 200). This makes
      // `arr_sz` zero.
      if (arr_sz == 0) {
        continue;
      }

      std::vector<int> data_ptr_idx = key;
      data_ptr_idx.push_back(TypeFactory::DATA_PTR_POS_IN_NDARRAY);
      auto data_ptr = ctx.array_ptrs[data_ptr_idx];
      std::vector<int> grad_ptr_idx = key;
      grad_ptr_idx.push_back(TypeFactory::GRAD_PTR_POS_IN_NDARRAY);

      auto grad_ptr = ctx.array_ptrs[grad_ptr_idx];
      if (ctx.device_allocation_type[key] ==
          LaunchContextBuilder::DevAllocType::kNone) {
        // External array
        // Note: assuming both data & grad are on the same device
        if (on_cuda_device(data_ptr)) {
          // data_ptr is a raw ptr on CUDA device
          device_ptrs[data_ptr_idx] = data_ptr;
          device_ptrs[grad_ptr_idx] = grad_ptr;
        } else {
          DeviceAllocation devalloc = executor->allocate_memory_on_device(
              arr_sz, (uint64 *)device_result_buffer);
          device_ptrs[data_ptr_idx] =
              executor->get_device_alloc_info_ptr(devalloc);
          transfers[data_ptr_idx] = {data_ptr, devalloc};

          CUDADriver::get_instance().memcpy_host_to_device(
              (void *)device_ptrs[data_ptr_idx], data_ptr, arr_sz);
          if (grad_ptr != nullptr) {
            DeviceAllocation grad_devalloc =
                executor->allocate_memory_on_device(
                    arr_sz, (uint64 *)device_result_buffer);
            device_ptrs[grad_ptr_idx] =
                executor->get_device_alloc_info_ptr(grad_devalloc);
            transfers[grad_ptr_idx] = {grad_ptr, grad_devalloc};

            CUDADriver::get_instance().memcpy_host_to_device(
                (void *)device_ptrs[grad_ptr_idx], grad_ptr, arr_sz);
          } else {
            device_ptrs[grad_ptr_idx] = nullptr;
          }
        }

        ctx.set_ndarray_ptrs(key, (uint64)device_ptrs[data_ptr_idx],
                             (uint64)device_ptrs[grad_ptr_idx]);
      } else if (arr_sz > 0) {
        // Ndarray
        DeviceAllocation *ptr = static_cast<DeviceAllocation *>(data_ptr);
        // Unwrapped raw ptr on device
        device_ptrs[data_ptr_idx] = executor->get_device_alloc_info_ptr(*ptr);

        if (grad_ptr != nullptr) {
          ptr = static_cast<DeviceAllocation *>(grad_ptr);
          device_ptrs[grad_ptr_idx] = executor->get_device_alloc_info_ptr(*ptr);
        } else {
          device_ptrs[grad_ptr_idx] = nullptr;
        }

        ctx.set_ndarray_ptrs(key, (uint64)device_ptrs[data_ptr_idx],
                             (uint64)device_ptrs[grad_ptr_idx]);
      }
    } else if (parameter.is_argpack) {
      std::vector<int> data_ptr_idx = key;
      data_ptr_idx.push_back(TypeFactory::DATA_PTR_POS_IN_ARGPACK);
      auto *argpack = ctx.argpack_ptrs[key];
      auto argpack_ptr = argpack->get_device_allocation();
      device_ptrs[data_ptr_idx] =
          executor->get_device_alloc_info_ptr(argpack_ptr);
      if (key.size() == 1) {
        ctx.set_argpack_ptr(key, (uint64)device_ptrs[data_ptr_idx]);
      } else {
        auto key_parent = key;
        key_parent.pop_back();
        auto *argpack_parent = ctx.argpack_ptrs[key_parent];
        argpack_parent->set_arg_nested_argpack_ptr(
            key.back(), (uint64)device_ptrs[data_ptr_idx]);
      }
    }
  }
  if (transfers.size() > 0) {
    CUDADriver::get_instance().stream_synchronize(nullptr);
  }
  char *host_result_buffer = (char *)ctx.get_context().result_buffer;
  if (ctx.result_buffer_size > 0) {
    ctx.get_context().result_buffer = (uint64 *)device_result_buffer;
  }
  char *device_arg_buffer = nullptr;
  if (ctx.arg_buffer_size > 0) {
    CUDADriver::get_instance().malloc_async((void **)&device_arg_buffer,
                                            ctx.arg_buffer_size, nullptr);
    CUDADriver::get_instance().memcpy_host_to_device_async(
        device_arg_buffer, ctx.get_context().arg_buffer, ctx.arg_buffer_size,
        nullptr);
    ctx.get_context().arg_buffer = device_arg_buffer;
  }

  for (auto task : offloaded_tasks) {
    if (sparse_list_task_is_current(task)) {
      TI_TRACE("Skipping current sparse list kernel {}", task.name);
      continue;
    }
    TI_TRACE("Launching kernel {}<<<{}, {}>>>", task.name, task.grid_dim,
             task.block_dim);
    cuda_module->launch(task.name, task.grid_dim, task.block_dim, 0,
                        {&ctx.get_context()}, {});
    mark_sparse_list_task_launched(task);
    if (task.may_mutate_sparse_topology) {
      invalidate_sparse_list_cache(task.sparse_mutation_snode_id);
    }
  }
  if (ctx.arg_buffer_size > 0) {
    CUDADriver::get_instance().mem_free_async(device_arg_buffer, nullptr);
  }
  if (ctx.result_buffer_size > 0) {
    CUDADriver::get_instance().memcpy_device_to_host_async(
        host_result_buffer, device_result_buffer, ctx.result_buffer_size,
        nullptr);
  }
  CUDADriver::get_instance().mem_free_async(device_result_buffer, nullptr);
  // copy data back to host
  if (transfers.size() > 0) {
    CUDADriver::get_instance().stream_synchronize(nullptr);
    for (auto itr = transfers.begin(); itr != transfers.end(); itr++) {
      auto &idx = itr->first;
      auto arg_id = idx;
      arg_id.pop_back();
      CUDADriver::get_instance().memcpy_device_to_host(
          itr->second.first, (void *)device_ptrs[idx],
          ctx.array_runtime_sizes[arg_id]);
      executor->deallocate_memory_on_device(itr->second.second);
    }
  }
}

KernelLauncher::Handle KernelLauncher::register_llvm_kernel(
    const LLVM::CompiledKernelData &compiled) {
  TI_ASSERT(compiled.arch() == Arch::cuda);

  if (!compiled.get_handle()) {
    auto handle = make_handle();
    auto index = handle.get_launch_id();
    contexts_.resize(index + 1);

    auto &ctx = contexts_[index];
    auto *executor = get_runtime_executor();

    auto data = compiled.get_internal_data().compiled_data.clone();
    auto parameters = compiled.get_internal_data().args;
    auto *jit_module = executor->create_jit_module(std::move(data.module));

    // Populate ctx
    ctx.jit_module = jit_module;
    ctx.parameters = std::move(parameters);
    ctx.offloaded_tasks = std::move(data.tasks);

    compiled.set_handle(handle);
  }
  return *compiled.get_handle();
}

}  // namespace cuda
}  // namespace taichi::lang
