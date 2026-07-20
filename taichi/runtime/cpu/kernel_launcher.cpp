#include "taichi/runtime/cpu/kernel_launcher.h"
#include "taichi/rhi/arch.h"
#include "taichi/system/profiler.h"

namespace taichi::lang {
namespace cpu {

void KernelLauncher::launch_llvm_kernel(Handle handle,
                                        LaunchContextBuilder &ctx) {
  std::unique_lock<std::mutex> execution_lock(execution_mutex_);
  std::shared_lock<std::shared_mutex> launch_lock(registration_mutex());
  std::shared_ptr<const Context> launcher_ctx;
  auto iter = contexts_.find(handle.get_launch_id());
  TI_ASSERT(iter != contexts_.end());
  launcher_ctx = iter->second;
  auto *executor = get_runtime_executor();

  ctx.get_context().runtime = executor->get_llvm_runtime();
  // For taichi ndarrays, context.array_ptrs saves pointer to its
  // |DeviceAllocation|, CPU backend actually want to use the raw ptr here.
  const auto &parameters = launcher_ctx->parameters;
  {
    TI_COMPILE_PROFILER("cpu_launch_bind_args");
    for (int i = 0; i < (int)parameters.size(); i++) {
      const auto &kv = parameters[i];
      const auto &key = kv.first;
      const auto &parameter = kv.second;
      std::vector<int> data_ptr_idx = key;
      data_ptr_idx.push_back(TypeFactory::DATA_PTR_POS_IN_NDARRAY);
      std::vector<int> grad_ptr_idx = key;
      grad_ptr_idx.push_back(TypeFactory::GRAD_PTR_POS_IN_NDARRAY);

      if (parameter.is_array && ctx.device_allocation_type[key] ==
                                    LaunchContextBuilder::DevAllocType::kNone) {
        ctx.set_ndarray_ptrs(key, (uint64)ctx.array_ptrs[data_ptr_idx],
                             (uint64)ctx.array_ptrs[grad_ptr_idx]);
      }
      if (parameter.is_array &&
          ctx.device_allocation_type[key] !=
              LaunchContextBuilder::DevAllocType::kNone &&
          ctx.array_runtime_sizes[key] > 0) {
        DeviceAllocation *ptr =
            static_cast<DeviceAllocation *>(ctx.array_ptrs[data_ptr_idx]);
        uint64 host_ptr = (uint64)executor->get_device_alloc_info_ptr(*ptr);
        ctx.set_array_device_allocation_type(
            key, LaunchContextBuilder::DevAllocType::kNone);

        auto grad_ptr = ctx.array_ptrs[grad_ptr_idx];
        uint64 host_ptr_grad =
            grad_ptr == nullptr
                ? 0
                : (uint64)executor->get_device_alloc_info_ptr(
                      *static_cast<DeviceAllocation *>(grad_ptr));
        ctx.set_ndarray_ptrs(key, host_ptr, host_ptr_grad);
      }
      if (parameter.is_argpack) {
        data_ptr_idx = key;
        data_ptr_idx.push_back(TypeFactory::DATA_PTR_POS_IN_ARGPACK);
        auto *argpack = ctx.argpack_ptrs[key];
        auto argpack_ptr = argpack->get_device_allocation();
        uint64 host_ptr =
            (uint64)executor->get_device_alloc_info_ptr(argpack_ptr);
        if (key.size() == 1) {
          ctx.set_argpack_ptr(key, host_ptr);
        } else {
          auto key_parent = key;
          key_parent.pop_back();
          auto *argpack_parent = ctx.argpack_ptrs[key_parent];
          argpack_parent->set_arg_nested_argpack_ptr(key.back(), host_ptr);
        }
      }
    }
  }
  {
    TI_COMPILE_PROFILER("cpu_launch_tasks");
    if (!sparse_listgen_telemetry_enabled_) {
      for (auto task : launcher_ctx->task_funcs) {
        task(&ctx.get_context());
      }
    } else {
      TI_ASSERT(launcher_ctx->task_funcs.size() ==
                launcher_ctx->sparse_task_metadata.size());
      for (std::size_t i = 0; i < launcher_ctx->task_funcs.size(); ++i) {
        const auto &metadata = launcher_ctx->sparse_task_metadata[i];
        const bool is_listgen =
            metadata.sparse_list_op == OffloadedTask::kSparseListOpListgen &&
            metadata.snode_id >= 0;
        if (is_listgen) {
          auto &telemetry = sparse_listgen_telemetry_[metadata.snode_id];
          telemetry.snode_id = metadata.snode_id;
          telemetry.parent_snode_id = metadata.parent_snode_id;
          ++telemetry.requests;
          executor->begin_cpu_sparse_listgen_work();
        }
        launcher_ctx->task_funcs[i](&ctx.get_context());
        if (is_listgen) {
          const auto work = executor->read_cpu_sparse_listgen_work();
          if (work.available) {
            auto &telemetry = sparse_listgen_telemetry_[metadata.snode_id];
            if (work.reused) {
              ++telemetry.reuse_hits;
              continue;
            }
            const bool cold = telemetry.rebuilds == 0;
            ++telemetry.rebuilds;
            telemetry.last_rebuild_reason =
                cold ? "cold" : "runtime_topology_or_parent_changed";
            if (!telemetry.scanned_elements.available) {
              telemetry.scanned_elements = {0, true};
              telemetry.emitted_elements = {0, true};
              telemetry.serial_rebuilds = {0, true};
              telemetry.parallel_rebuilds = {0, true};
            }
            telemetry.scanned_elements.value += work.scanned_elements;
            telemetry.emitted_elements.value += work.emitted_elements;
            if (work.parallel) {
              ++telemetry.parallel_rebuilds.value;
            } else {
              ++telemetry.serial_rebuilds.value;
            }
          }
        }
      }
    }
  }
}

KernelLauncher::Handle KernelLauncher::register_llvm_kernel(
    const LLVM::CompiledKernelData &compiled) {
  TI_ASSERT(arch_is_cpu(compiled.arch()));
  std::unique_lock<std::shared_mutex> lock(registration_mutex());

  if (!compiled.get_handle()) {
    auto handle = make_handle();
    auto index = handle.get_launch_id();

    auto ctx = std::make_shared<Context>();
    auto *executor = get_runtime_executor();

    const auto &internal_data = compiled.get_internal_data();
    auto data = internal_data.compiled_data.clone();
    auto parameters = internal_data.args;
    auto *jit_module = executor->create_jit_module(std::move(data.module));

    // Construct task_funcs
    using TaskFunc = int32 (*)(void *);
    std::vector<TaskFunc> task_funcs;
    std::vector<Context::SparseTaskMetadata> sparse_task_metadata;
    task_funcs.reserve(data.tasks.size());
    sparse_task_metadata.reserve(data.tasks.size());
    for (auto &task : data.tasks) {
      auto *func_ptr = jit_module->lookup_function(task.name);
      TI_ASSERT_INFO(func_ptr, "Offloaded datum function {} not found",
                     task.name);
      task_funcs.push_back((TaskFunc)(func_ptr));
      sparse_task_metadata.push_back({task.sparse_list_op,
                                      task.sparse_list_snode_id,
                                      task.sparse_list_parent_snode_id});
    }

    // Populate ctx
    ctx->jit_module = jit_module;
    ctx->snode_tree_ids = compiled.snode_tree_ids();
    ctx->parameters = std::move(parameters);
    ctx->task_funcs = std::move(task_funcs);
    ctx->sparse_task_metadata = std::move(sparse_task_metadata);
    const bool was_inserted = contexts_.emplace(index, std::move(ctx)).second;
    TI_ASSERT(was_inserted);

    compiled.set_handle(handle);
  }
  return *compiled.get_handle();
}

void KernelLauncher::retire_snode_tree(int tree_id) {
  std::unique_lock<std::mutex> execution_lock(execution_mutex_);
  std::unique_lock<std::shared_mutex> registration_lock(registration_mutex());
  auto *executor = get_runtime_executor();
  std::vector<int> retired_sparse_snode_ids;
  for (auto iter = contexts_.begin(); iter != contexts_.end();) {
    const auto &context = iter->second;
    if (!std::binary_search(context->snode_tree_ids.begin(),
                            context->snode_tree_ids.end(), tree_id)) {
      ++iter;
      continue;
    }
    for (const auto &metadata : context->sparse_task_metadata) {
      if (metadata.snode_id >= 0) {
        retired_sparse_snode_ids.push_back(metadata.snode_id);
      }
    }
    auto module = context->jit_module;
    iter = contexts_.erase(iter);
    executor->remove_jit_module(module);
  }
  for (int snode_id : retired_sparse_snode_ids) {
    sparse_listgen_telemetry_.erase(snode_id);
  }
}

std::size_t KernelLauncher::debug_registered_kernel_count() {
  std::shared_lock<std::shared_mutex> lock(registration_mutex());
  return contexts_.size();
}

void KernelLauncher::debug_reset_sparse_listgen_statistics() {
  std::unique_lock<std::mutex> lock(execution_mutex_);
  sparse_listgen_telemetry_.clear();
  sparse_listgen_telemetry_enabled_ = true;
}

SparseSNodeTreeListgenStatistics
KernelLauncher::debug_sparse_listgen_statistics(
    const std::vector<int> &snode_ids) {
  std::unique_lock<std::mutex> lock(execution_mutex_);
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

}  // namespace cpu
}  // namespace taichi::lang
