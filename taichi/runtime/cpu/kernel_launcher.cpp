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
    for (auto task : launcher_ctx->task_funcs) {
      task(&ctx.get_context());
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
    task_funcs.reserve(data.tasks.size());
    for (auto &task : data.tasks) {
      auto *func_ptr = jit_module->lookup_function(task.name);
      TI_ASSERT_INFO(func_ptr, "Offloaded datum function {} not found",
                     task.name);
      task_funcs.push_back((TaskFunc)(func_ptr));
    }

    // Populate ctx
    ctx->jit_module = jit_module;
    ctx->snode_tree_ids = compiled.snode_tree_ids();
    ctx->parameters = std::move(parameters);
    ctx->task_funcs = std::move(task_funcs);
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
  for (auto iter = contexts_.begin(); iter != contexts_.end();) {
    const auto &context = iter->second;
    if (!std::binary_search(context->snode_tree_ids.begin(),
                            context->snode_tree_ids.end(), tree_id)) {
      ++iter;
      continue;
    }
    auto module = context->jit_module;
    iter = contexts_.erase(iter);
    executor->remove_jit_module(module);
  }
}

std::size_t KernelLauncher::debug_registered_kernel_count() {
  std::shared_lock<std::shared_mutex> lock(registration_mutex());
  return contexts_.size();
}

}  // namespace cpu
}  // namespace taichi::lang
