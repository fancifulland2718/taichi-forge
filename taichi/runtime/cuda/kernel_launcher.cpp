#include "taichi/runtime/cuda/kernel_launcher.h"
#include "taichi/runtime/cuda/jit_cuda.h"
#include "taichi/rhi/cuda/cuda_context.h"

#include <cstring>
#include <cstdint>
#include <unordered_map>

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"

namespace taichi::lang {
namespace cuda {

namespace {

void add_cuda_graph_execution_gate(LLVMCompiledKernel &compiled) {
  TI_ASSERT(compiled.module != nullptr);
  auto &module = *compiled.module;
  auto &llvm_context = module.getContext();
  auto *pointer_type = llvm::PointerType::get(llvm_context, 0);
  auto *i8_type = llvm::Type::getInt8Ty(llvm_context);
  auto *i32_type = llvm::Type::getInt32Ty(llvm_context);
  auto *i64_type = llvm::Type::getInt64Ty(llvm_context);

  for (const auto &task : compiled.tasks) {
    auto *function = module.getFunction(task.name);
    TI_ASSERT(function != nullptr && !function->empty());
    TI_ASSERT(function->getReturnType()->isVoidTy());
    TI_ASSERT(function->arg_size() == 1);
    auto *original_entry = &function->getEntryBlock();
    auto *gate_entry = llvm::BasicBlock::Create(
        llvm_context, "graph_execution_gate", function, original_entry);
    auto *masked_exit = llvm::BasicBlock::Create(
        llvm_context, "graph_execution_masked_exit", function, original_entry);
    llvm::IRBuilder<> builder(gate_entry);
    // RuntimeContext::arg_buffer is its first member. The private gated launch
    // allocation stores GraphExecutionGateBinding immediately before that
    // pointer, so the variant can inline the test without adding a runtime
    // bitcode symbol or changing RuntimeContext's split-wheel ABI.
    auto *arg_buffer = builder.CreateAlignedLoad(
        pointer_type, function->getArg(0), llvm::Align(8), "graph_args");
    auto *binding = builder.CreateInBoundsGEP(
        i8_type, arg_buffer,
        llvm::ConstantInt::getSigned(
            i64_type,
            -static_cast<std::int64_t>(sizeof(GraphExecutionGateBinding))),
        "graph_gate_binding");
    auto *gate_bits = builder.CreateAlignedLoad(
        i64_type, binding, llvm::Align(8), "graph_gate_address");
    auto *expected_address = builder.CreateInBoundsGEP(
        i8_type, binding,
        llvm::ConstantInt::get(i64_type,
                               offsetof(GraphExecutionGateBinding, expected)));
    auto *expected = builder.CreateAlignedLoad(
        i32_type, expected_address, llvm::Align(4), "graph_gate_expected");
    auto *gate = builder.CreateIntToPtr(gate_bits, pointer_type);
    auto *value = builder.CreateAlignedLoad(i32_type, gate, llvm::Align(4),
                                            "graph_gate_value");
    auto *execute = builder.CreateICmpEQ(value, expected);
    builder.CreateCondBr(execute, original_entry, masked_exit);
    builder.SetInsertPoint(masked_exit);
    builder.CreateRetVoid();
  }
  TI_ASSERT(!llvm::verifyModule(module, &llvm::errs()));
}

}  // namespace

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
  SparseListgenNodeStatistics *telemetry = nullptr;
  if (sparse_listgen_telemetry_enabled_ &&
      task.sparse_list_op == OffloadedTask::kSparseListOpListgen &&
      task.sparse_list_snode_id >= 0) {
    telemetry = &sparse_listgen_telemetry_[task.sparse_list_snode_id];
    telemetry->snode_id = task.sparse_list_snode_id;
    telemetry->parent_snode_id = task.sparse_list_parent_snode_id;
    ++telemetry->requests;
  }
  if (task.sparse_list_op == OffloadedTask::kSparseListOpNone ||
      task.sparse_list_snode_id < 0 ||
      task.sparse_list_parent_snode_id < 0) {
    return false;
  }

  auto it = sparse_list_states_.find(task.sparse_list_snode_id);
  if (it == sparse_list_states_.end()) {
    if (telemetry != nullptr) {
      telemetry->last_rebuild_reason = "cold";
    }
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
  if (telemetry != nullptr) {
    if (would_skip && !state.adaptive_disabled) {
      ++telemetry->reuse_hits;
    } else if (state.clean_epoch != state.dirty_epoch) {
      telemetry->last_rebuild_reason = "topology_dirty";
    } else if (state.clean_parent_version !=
               get_sparse_list_version(task.sparse_list_parent_snode_id)) {
      telemetry->last_rebuild_reason = "parent_version_changed";
    } else if (state.adaptive_disabled) {
      telemetry->last_rebuild_reason = "adaptive_reuse_disabled";
    } else {
      telemetry->last_rebuild_reason = "not_current";
    }
  }
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

  if (sparse_listgen_telemetry_enabled_) {
    auto &telemetry = sparse_listgen_telemetry_[task.sparse_list_snode_id];
    telemetry.snode_id = task.sparse_list_snode_id;
    telemetry.parent_snode_id = task.sparse_list_parent_snode_id;
    ++telemetry.rebuilds;
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
    if (sparse_listgen_telemetry_enabled_) {
      auto &telemetry =
          sparse_listgen_telemetry_[sparse_mutation_snode_id];
      telemetry.snode_id = sparse_mutation_snode_id;
      ++telemetry.invalidations;
    }
    return;
  }
  for (auto &kv : sparse_list_states_) {
    kv.second.dirty_epoch++;
    if (sparse_listgen_telemetry_enabled_) {
      auto &telemetry = sparse_listgen_telemetry_[kv.first];
      telemetry.snode_id = kv.first;
      ++telemetry.invalidations;
    }
  }
}

bool KernelLauncher::prepare_cuda_graph_context(Handle handle,
                                                LaunchContextBuilder &ctx,
                                                RuntimeContext &context) {
  std::shared_lock<std::shared_mutex> launch_lock(registration_mutex());
  std::shared_ptr<const Context> launcher_ctx;
  auto iter = contexts_.find(handle.get_launch_id());
  TI_ASSERT(iter != contexts_.end());
  launcher_ctx = iter->second;
  const auto &parameters = launcher_ctx->parameters;
  const auto &offloaded_tasks = launcher_ctx->offloaded_tasks;
  for (const auto &task : offloaded_tasks) {
    if (task.sparse_list_op != OffloadedTask::kSparseListOpNone ||
        task.may_mutate_sparse_topology) {
      return false;
    }
  }
  if (ctx.result_buffer_size > 0) {
    return false;
  }

  auto *executor = get_runtime_executor();
  ctx.get_context().runtime = executor->get_llvm_runtime();
  for (int i = 0; i < (int)parameters.size(); i++) {
    const auto &kv = parameters[i];
    const auto &key = kv.first;
    const auto &parameter = kv.second;
    if (parameter.is_array) {
      const auto arr_sz = ctx.array_runtime_sizes[key];
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
        if (!on_cuda_device(data_ptr)) {
          return false;
        }
        ctx.set_ndarray_ptrs(key, reinterpret_cast<uint64>(data_ptr),
                             reinterpret_cast<uint64>(grad_ptr));
      } else if (ctx.device_allocation_type[key] ==
                 LaunchContextBuilder::DevAllocType::kNdarray) {
        DeviceAllocation *ptr = static_cast<DeviceAllocation *>(data_ptr);
        auto *data_device_ptr = executor->get_device_alloc_info_ptr(*ptr);
        void *grad_device_ptr = nullptr;
        if (grad_ptr != nullptr) {
          ptr = static_cast<DeviceAllocation *>(grad_ptr);
          grad_device_ptr = executor->get_device_alloc_info_ptr(*ptr);
        }
        ctx.set_ndarray_ptrs(key, reinterpret_cast<uint64>(data_device_ptr),
                             reinterpret_cast<uint64>(grad_device_ptr));
      } else if (ctx.device_allocation_type[key] ==
                 LaunchContextBuilder::DevAllocType::kDenseStorage) {
        const auto &binding = ctx.get_resolved_dense_storage(key);
        auto *base = reinterpret_cast<char *>(
            executor->get_device_alloc_info_ptr(binding.allocation));
        ctx.set_ndarray_ptrs(
            key, reinterpret_cast<uint64>(base + binding.byte_offset), 0);
      } else {
        return false;
      }
    } else if (parameter.is_argpack) {
      std::vector<int> data_ptr_idx = key;
      data_ptr_idx.push_back(TypeFactory::DATA_PTR_POS_IN_ARGPACK);
      auto *argpack = ctx.argpack_ptrs[key];
      auto argpack_ptr = argpack->get_device_allocation();
      auto *device_ptr = executor->get_device_alloc_info_ptr(argpack_ptr);
      if (key.size() == 1) {
        ctx.set_argpack_ptr(key, reinterpret_cast<uint64>(device_ptr));
      } else {
        auto key_parent = key;
        key_parent.pop_back();
        auto *argpack_parent = ctx.argpack_ptrs[key_parent];
        argpack_parent->set_arg_nested_argpack_ptr(
            key.back(), reinterpret_cast<uint64>(device_ptr));
      }
    }
  }

  context = ctx.get_context();
  // A zero-size result buffer is never dereferenced by the compiled tasks.
  // Canonicalize it so capture packets do not retain a transient host address
  // and compatible argument updates do not spuriously require recapture.
  context.result_buffer = nullptr;
  return true;
}

bool KernelLauncher::prepare_cuda_graph_launch(Handle handle,
                                               LaunchContextBuilder &ctx,
                                               GraphLaunchPacket &packet,
                                               void *stream) {
  RuntimeContext context;
  if (!prepare_cuda_graph_context(handle, ctx, context)) {
    return false;
  }
  packet.handle = handle;
  packet.arg_buffer_size = ctx.arg_buffer_size;
  packet.arg_buffer_prefix_size = 0;
  packet.device_arg_buffer_size = packet.arg_buffer_size;
  packet.context = context;
  if (packet.arg_buffer_size == 0) {
    // A Field-only graph has no runtime argument storage. The RuntimeContext
    // still carries the runtime/root state used by statically bound fields,
    // but capture needs neither a device allocation nor an H2D upload.
    packet.device_arg_buffer = nullptr;
    packet.context.arg_buffer = nullptr;
    return true;
  }
  CUDADriver::get_instance().malloc_async(&packet.device_arg_buffer,
                                          packet.arg_buffer_size, stream);
  CUDADriver::get_instance().memcpy_host_to_device_async(
      packet.device_arg_buffer, packet.context.arg_buffer,
      packet.arg_buffer_size, stream);
  packet.context.arg_buffer = static_cast<char *>(packet.device_arg_buffer);
  return true;
}

bool KernelLauncher::prepare_cuda_graph_gated_launch(Handle handle,
                                                     LaunchContextBuilder &ctx,
                                                     GraphLaunchPacket &packet,
                                                     void *gate,
                                                     std::uint32_t expected,
                                                     void *stream) {
  if (gate == nullptr || expected == 0) {
    return false;
  }
  RuntimeContext context;
  if (!prepare_cuda_graph_context(handle, ctx, context)) {
    return false;
  }

  GraphExecutionGateBinding binding;
  binding.gate = reinterpret_cast<std::uintptr_t>(gate);
  binding.expected = expected;
  packet.handle = handle;
  packet.arg_buffer_size = ctx.arg_buffer_size;
  packet.arg_buffer_prefix_size = sizeof(binding);
  packet.device_arg_buffer_size =
      packet.arg_buffer_prefix_size + packet.arg_buffer_size;
  packet.context = context;
  CUDADriver::get_instance().malloc_async(
      &packet.device_arg_buffer, packet.device_arg_buffer_size, stream);
  CUDADriver::get_instance().memcpy_host_to_device_async(
      packet.device_arg_buffer, &binding, sizeof(binding), stream);
  auto *device_args = static_cast<char *>(packet.device_arg_buffer) +
                      packet.arg_buffer_prefix_size;
  if (packet.arg_buffer_size > 0) {
    CUDADriver::get_instance().memcpy_host_to_device_async(
        device_args, packet.context.arg_buffer, packet.arg_buffer_size, stream);
  }
  packet.context.arg_buffer = device_args;
  return true;
}

bool KernelLauncher::update_cuda_graph_launch(
    const GraphLaunchPacket &packet,
    LaunchContextBuilder &ctx,
    std::vector<uint8_t> &host_arg_buffer,
    void *stream) {
  if (packet.arg_buffer_size != ctx.arg_buffer_size) {
    return false;
  }
  RuntimeContext context;
  if (!prepare_cuda_graph_context(packet.handle, ctx, context)) {
    return false;
  }
  if (context.runtime != packet.context.runtime ||
      context.result_buffer != packet.context.result_buffer) {
    return false;
  }

  if (packet.arg_buffer_size == 0) {
    if (packet.arg_buffer_prefix_size == 0) {
      if (packet.device_arg_buffer != nullptr ||
          packet.context.arg_buffer != nullptr) {
        return false;
      }
    } else if (packet.device_arg_buffer == nullptr ||
               packet.context.arg_buffer == nullptr ||
               packet.device_arg_buffer_size != packet.arg_buffer_prefix_size) {
      return false;
    }
    host_arg_buffer.clear();
    return true;
  }
  if (packet.device_arg_buffer == nullptr || context.arg_buffer == nullptr) {
    return false;
  }

  host_arg_buffer.resize(packet.arg_buffer_size);
  std::memcpy(host_arg_buffer.data(), context.arg_buffer,
              packet.arg_buffer_size);
  auto *device_args = static_cast<char *>(packet.device_arg_buffer) +
                      packet.arg_buffer_prefix_size;
  CUDADriver::get_instance().memcpy_host_to_device_async(
      device_args, host_arg_buffer.data(), packet.arg_buffer_size, stream);
  return true;
}

void KernelLauncher::capture_cuda_graph_launch(
    const GraphLaunchPacket &packet,
    void *stream) {
  std::shared_lock<std::shared_mutex> launch_lock(registration_mutex());
  std::shared_ptr<const Context> launcher_ctx;
  auto iter = contexts_.find(packet.handle.get_launch_id());
  TI_ASSERT(iter != contexts_.end());
  launcher_ctx = iter->second;
  auto *cuda_module = launcher_ctx->jit_module;
  auto *cuda_jit_module = dynamic_cast<JITModuleCUDA *>(cuda_module);
  TI_ASSERT(cuda_jit_module != nullptr);
  const auto &offloaded_tasks = launcher_ctx->offloaded_tasks;
  for (auto task : offloaded_tasks) {
    TI_TRACE("Capturing kernel {}<<<{}, {}>>>", task.name, task.grid_dim,
             task.block_dim);
    cuda_jit_module->launch_with_stream(
        task.name, task.grid_dim, task.block_dim,
        task.dynamic_shared_array_bytes,
        {const_cast<RuntimeContext *>(&packet.context)}, {}, stream);
  }
}

void KernelLauncher::launch_llvm_kernel(Handle handle,
                                        LaunchContextBuilder &ctx) {
  // Keep argument-buffer preparation, every offloaded task, and stream-ordered
  // cleanup contiguous with respect to CUDA graph replay and other host
  // launchers. The default ndarray/field path only enqueues work while holding
  // this lock; it does not add a device synchronization.
  auto submission_lock =
      CUDAContext::get_instance().get_submission_lock_guard();
  std::shared_lock<std::shared_mutex> launch_lock(registration_mutex());
  std::shared_ptr<const Context> launcher_ctx;
  auto iter = contexts_.find(handle.get_launch_id());
  TI_ASSERT(iter != contexts_.end());
  launcher_ctx = iter->second;
  auto *executor = get_runtime_executor();
  const bool listgen_reuse_adaptive =
      executor->get_config().cuda_listgen_reuse_adaptive;
  auto *cuda_module = launcher_ctx->jit_module;
  const auto &parameters = launcher_ctx->parameters;
  const auto &offloaded_tasks = launcher_ctx->offloaded_tasks;

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
  // Most void kernels with only Taichi fields do not need a device result
  // buffer. Allocate lazily for an actual return value or a host-array
  // transfer, both of which use it as the runtime allocation result channel.
  // This preserves the default-stream ordering of the existing allocation and
  // free while avoiding one async allocation/free pair per ordinary launch.
  auto ensure_device_result_buffer = [&] {
    if (device_result_buffer == nullptr) {
      CUDADriver::get_instance().malloc_async(
          (void **)&device_result_buffer,
          std::max(ctx.result_buffer_size, sizeof(uint64)), nullptr);
    }
    return device_result_buffer;
  };
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
          DeviceAllocation devalloc =
              executor->allocate_memory_on_device(
                  arr_sz, (uint64 *)ensure_device_result_buffer());
          device_ptrs[data_ptr_idx] =
              executor->get_device_alloc_info_ptr(devalloc);
          transfers[data_ptr_idx] = {data_ptr, devalloc};

          CUDADriver::get_instance().memcpy_host_to_device(
              (void *)device_ptrs[data_ptr_idx], data_ptr, arr_sz);
          if (grad_ptr != nullptr) {
            DeviceAllocation grad_devalloc =
                executor->allocate_memory_on_device(
                    arr_sz, (uint64 *)ensure_device_result_buffer());
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
      } else if (ctx.device_allocation_type[key] ==
                 LaunchContextBuilder::DevAllocType::kDenseStorage) {
        const auto &binding = ctx.get_resolved_dense_storage(key);
        auto *base = reinterpret_cast<char *>(
            executor->get_device_alloc_info_ptr(binding.allocation));
        device_ptrs[data_ptr_idx] = base + binding.byte_offset;
        device_ptrs[grad_ptr_idx] = nullptr;
        ctx.set_ndarray_ptrs(key, (uint64)device_ptrs[data_ptr_idx], 0);
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
  char *host_arg_buffer = ctx.get_context().arg_buffer;
  char *host_result_buffer = (char *)ctx.get_context().result_buffer;
  if (ctx.result_buffer_size > 0) {
    ctx.get_context().result_buffer =
        (uint64 *)ensure_device_result_buffer();
  }
  char *device_arg_buffer = nullptr;
  if (ctx.arg_buffer_size > 0) {
    CUDADriver::get_instance().malloc_async((void **)&device_arg_buffer,
                                            ctx.arg_buffer_size, nullptr);
    CUDADriver::get_instance().memcpy_host_to_device_async(
        device_arg_buffer, host_arg_buffer, ctx.arg_buffer_size,
        nullptr);
    ctx.get_context().arg_buffer = device_arg_buffer;
  }

  for (auto task : offloaded_tasks) {
    const bool uses_sparse_state =
        task.sparse_list_op != OffloadedTask::kSparseListOpNone ||
        task.may_mutate_sparse_topology;
    std::unique_lock<std::mutex> sparse_lock(sparse_list_mutex_,
                                             std::defer_lock);
    if (uses_sparse_state) {
      sparse_lock.lock();
      listgen_reuse_adaptive_ = listgen_reuse_adaptive;
    }
    if (sparse_list_task_is_current(task)) {
      TI_TRACE("Skipping current sparse list kernel {}", task.name);
      continue;
    }
    TI_TRACE("Launching kernel {}<<<{}, {}>>>", task.name, task.grid_dim,
             task.block_dim);
    cuda_module->launch(task.name, task.grid_dim, task.block_dim,
                        task.dynamic_shared_array_bytes,
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
  if (device_result_buffer != nullptr) {
    CUDADriver::get_instance().mem_free_async(device_result_buffer, nullptr);
  }
  ctx.get_context().arg_buffer = host_arg_buffer;
  ctx.get_context().result_buffer = (uint64 *)host_result_buffer;
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
  std::unique_lock<std::shared_mutex> lock(registration_mutex());

  if (!compiled.get_handle()) {
    auto handle = make_handle();
    auto index = handle.get_launch_id();

    auto ctx = std::make_shared<Context>();
    auto *executor = get_runtime_executor();

    auto data = compiled.get_internal_data().compiled_data.clone();
    auto parameters = compiled.get_internal_data().args;
    auto *jit_module = executor->create_jit_module(std::move(data.module));

    // Populate ctx
    ctx->jit_module = jit_module;
    ctx->snode_tree_ids = compiled.snode_tree_ids();
    ctx->parameters = std::move(parameters);
    ctx->offloaded_tasks = std::move(data.tasks);
    const bool was_inserted = contexts_.emplace(index, std::move(ctx)).second;
    TI_ASSERT(was_inserted);

    compiled.set_handle(handle);
  }
  return *compiled.get_handle();
}

KernelLauncher::Handle KernelLauncher::register_llvm_kernel_graph_gated(
    const LLVM::CompiledKernelData &compiled) {
  TI_ASSERT(compiled.arch() == Arch::cuda);
  std::unique_lock<std::shared_mutex> lock(registration_mutex());
  if (compiled.get_graph_masked_handle()) {
    return *compiled.get_graph_masked_handle();
  }

  auto handle = make_handle();
  auto ctx = std::make_shared<Context>();
  auto *executor = get_runtime_executor();
  auto data = compiled.get_internal_data().compiled_data.clone();
  add_cuda_graph_execution_gate(data);
  auto parameters = compiled.get_internal_data().args;
  auto *jit_module = executor->create_jit_module(std::move(data.module));

  ctx->jit_module = jit_module;
  ctx->snode_tree_ids = compiled.snode_tree_ids();
  ctx->parameters = std::move(parameters);
  ctx->offloaded_tasks = std::move(data.tasks);
  const bool inserted =
      contexts_.emplace(handle.get_launch_id(), std::move(ctx)).second;
  TI_ASSERT(inserted);
  compiled.set_graph_masked_handle(handle);
  return handle;
}

void KernelLauncher::retire_snode_tree(int tree_id) {
  auto submission_lock =
      CUDAContext::get_instance().get_submission_lock_guard();
  std::unique_lock<std::shared_mutex> registration_lock(registration_mutex());
  auto *executor = get_runtime_executor();
  bool retired_any = false;
  std::vector<int> retired_sparse_snode_ids;
  for (auto iter = contexts_.begin(); iter != contexts_.end();) {
    const auto &context = iter->second;
    if (!std::binary_search(context->snode_tree_ids.begin(),
                            context->snode_tree_ids.end(), tree_id)) {
      ++iter;
      continue;
    }
    for (const auto &task : context->offloaded_tasks) {
      if (task.sparse_list_snode_id >= 0) {
        retired_sparse_snode_ids.push_back(task.sparse_list_snode_id);
      }
    }
    auto module = context->jit_module;
    iter = contexts_.erase(iter);
    executor->remove_jit_module(module);
    retired_any = true;
  }
  if (retired_any) {
    std::lock_guard<std::mutex> sparse_lock(sparse_list_mutex_);
    sparse_list_states_.clear();
    for (int snode_id : retired_sparse_snode_ids) {
      sparse_listgen_telemetry_.erase(snode_id);
    }
  }
}

std::size_t KernelLauncher::debug_registered_kernel_count() {
  std::shared_lock<std::shared_mutex> lock(registration_mutex());
  return contexts_.size();
}

void KernelLauncher::debug_reset_sparse_listgen_statistics() {
  std::lock_guard<std::mutex> lock(sparse_list_mutex_);
  sparse_listgen_telemetry_.clear();
  sparse_listgen_telemetry_enabled_ = true;
}

SparseSNodeTreeListgenStatistics
KernelLauncher::debug_sparse_listgen_statistics(
    const std::vector<int> &snode_ids) {
  std::lock_guard<std::mutex> lock(sparse_list_mutex_);
  SparseSNodeTreeListgenStatistics result;
  result.available = sparse_listgen_telemetry_enabled_;
  if (!result.available) {
    return result;
  }
  for (const auto &[snode_id, telemetry] : sparse_listgen_telemetry_) {
    if (std::binary_search(snode_ids.begin(), snode_ids.end(), snode_id)) {
      result.nodes.push_back(telemetry);
    }
  }
  std::sort(result.nodes.begin(), result.nodes.end(),
            [](const auto &lhs, const auto &rhs) {
              return lhs.snode_id < rhs.snode_id;
            });
  return result;
}

}  // namespace cuda
}  // namespace taichi::lang
