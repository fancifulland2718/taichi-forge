#include "argument_binding.h"

#include <algorithm>
#include <unordered_set>
#ifdef TI_WITH_LLVM
#include "taichi/runtime/llvm/aot_graph_data.h"
#include "taichi/runtime/cpu/kernel_launcher.h"
#endif
#ifdef TI_WITH_CUDA
#include "taichi/runtime/cuda/fixed_graph.h"
#include "taichi/runtime/cuda/kernel_launcher.h"
#endif
#ifdef TI_WITH_VULKAN
#include "taichi/runtime/gfx/aot_graph_data.h"
#endif

namespace capi {
using namespace taichi::lang;

struct PreparedExecution {
  AotModule *owner{nullptr};
  std::vector<TiMemory> memories;
  TiPreparedExecutionInfo info{};
  bool live{false};
  // Contexts point into these descriptor owners. Backend retirement always
  // precedes their destruction, including a failed cold preparation.
  std::unique_ptr<BoundKernelArguments> kernel_arguments;
  std::unique_ptr<BoundGraphArguments> graph_arguments;
  std::vector<std::unique_ptr<LaunchContextBuilder>> contexts;
#ifdef TI_WITH_LLVM
  cpu::KernelLauncher *cpu_launcher{nullptr};
  std::vector<cpu::KernelLauncher::PreparedLaunch> cpu_launches;
#endif

  void release_source_bindings() {
    contexts.clear();
    graph_arguments.reset();
    kernel_arguments.reset();
  }
#ifdef TI_WITH_CUDA
  std::unique_ptr<cuda::FixedGraph> cuda_graph;
#endif
#ifdef TI_WITH_VULKAN
  std::unique_ptr<gfx::GraphReplayRegistration> vulkan_graph;
#endif

  void prepare(
      taichi::Arch arch,
      const std::vector<std::pair<aot::Kernel *, LaunchContextBuilder *>>
          &dispatches) {
    if (dispatches.empty())
      throw MemoryBindingError("Prepared execution requires a nonempty batch");
    // Current C API image handles have no generation-qualified owner. Admit
    // them only when the resource-binding slice supplies that missing lease.
    for (const auto &[kernel, context] : dispatches) {
      if (kernel->ret_size || !context->argpack_ptrs.empty())
        throw MemoryBindingError(
            "Prepared execution does not support returns or ArgPacks");
      for (const auto &[index, param] : kernel->nested_parameters) {
        if (param.ptype == ParameterType::kTexture ||
            param.ptype == ParameterType::kRWTexture)
          throw MemoryBindingError(
              "Prepared texture bindings require the resource-owner extension");
      }
    }
    info.dispatch_count = static_cast<uint32_t>(dispatches.size());
#ifdef TI_WITH_LLVM
    if (taichi::arch_is_cpu(arch)) {
      for (const auto &[kernel, context] : dispatches) {
        auto *loaded = dynamic_cast<llvm_aot::KernelImpl *>(kernel);
        auto *launcher =
            loaded ? dynamic_cast<cpu::KernelLauncher *>(&loaded->launcher())
                   : nullptr;
        if (!launcher || (cpu_launcher && cpu_launcher != launcher))
          throw MemoryBindingError(
              "Prepared CPU batch requires one loaded launcher");
        cpu_launcher = launcher;
        cpu_launches.push_back(
            launcher->prepare_launch(loaded->handle(), *context));
        info.argument_bytes += context->arg_buffer_size;
      }
      info.execution_kind = 0;
      release_source_bindings();
      live = true;
      return;
    }
#endif
#ifdef TI_WITH_CUDA
    if (arch == taichi::Arch::cuda) {
      std::vector<cuda::FixedGraph::Dispatch> native;
      for (const auto &[kernel, context] : dispatches) {
        auto *loaded = dynamic_cast<llvm_aot::KernelImpl *>(kernel);
        auto *launcher =
            loaded ? dynamic_cast<cuda::KernelLauncher *>(&loaded->launcher())
                   : nullptr;
        if (!launcher)
          throw MemoryBindingError("Prepared CUDA entry is not registered");
        native.push_back({launcher, loaded->handle(), context});
      }
      cuda_graph = std::make_unique<cuda::FixedGraph>(native);
      info.execution_kind = 1;
      info.argument_bytes = cuda_graph->argument_bytes();
      release_source_bindings();
      live = true;
      return;
    }
#endif
#ifdef TI_WITH_VULKAN
    if (arch == taichi::Arch::vulkan) {
      gfx::GfxRuntime *runtime = nullptr;
      std::vector<gfx::GfxRuntime::GraphRecordingOperation> operations;
      for (const auto &[kernel, context] : dispatches) {
        auto *loaded = dynamic_cast<gfx::KernelImpl *>(kernel);
        if (!loaded || (runtime && runtime != &loaded->runtime()))
          throw MemoryBindingError(
              "Prepared Vulkan entries must share a runtime");
        runtime = &loaded->runtime();
        operations.push_back({{loaded->handle(), context}, {}});
      }
      vulkan_graph = runtime->prepare_fixed_graph(operations, {});
      info.execution_kind = 2;
      info.argument_bytes =
          vulkan_graph->snapshot_stats().known_persistent_argument_bytes;
      release_source_bindings();
      live = true;
      return;
    }
#endif
    throw MemoryBindingError("Prepared execution requires CPU, CUDA or Vulkan");
  }

  void run() {
    if (!live)
      throw MemoryBindingError("Prepared execution has a retired dependency");
#ifdef TI_WITH_LLVM
    if (cpu_launcher)
      cpu_launcher->launch_prepared(cpu_launches);
#endif
#ifdef TI_WITH_CUDA
    if (cuda_graph)
      cuda_graph->run();
#endif
#ifdef TI_WITH_VULKAN
    if (vulkan_graph)
      vulkan_graph->launch_prepared();
#endif
    ++info.launches;
  }

  void invalidate() {
    live = false;
#ifdef TI_WITH_CUDA
    cuda_graph.reset();
#endif
#ifdef TI_WITH_VULKAN
    vulkan_graph.reset();
#endif
#ifdef TI_WITH_LLVM
    cpu_launches.clear();
    cpu_launcher = nullptr;
#endif
    release_source_bindings();
  }
};

class PreparedExecutionRegistry {
 public:
  explicit PreparedExecutionRegistry(Runtime &runtime) : runtime_(runtime) {
  }
  TiPreparedExecution publish(std::unique_ptr<PreparedExecution> execution) {
    const auto handle =
        reinterpret_cast<TiPreparedExecution>(Runtime::new_handle_token());
    auto &memories = execution->memories;
    std::sort(memories.begin(), memories.end(), std::less<TiMemory>());
    memories.erase(std::unique(memories.begin(), memories.end()),
                   memories.end());
    auto *entry = execution.get();
    entries_.emplace(handle, std::move(execution));
    try {
      modules_[entry->owner].insert(handle);
      for (auto memory : memories)
        memories_[memory].insert(handle);
    } catch (...) {
      remove_dependencies(handle, *entry);
      entries_.erase(handle);
      throw;
    }
    return handle;
  }
  PreparedExecution &get(TiPreparedExecution handle) {
    const auto found = entries_.find(handle);
    if (found == entries_.end())
      throw MemoryBindingError("Prepared handle is not owned by this runtime");
    return *found->second;
  }
  void destroy(TiPreparedExecution handle) {
    auto &entry = get(handle);
    if (entry.live)
      runtime_.wait();
    remove_dependencies(handle, entry);
    entries_.erase(handle);
  }
  void retire_memory(TiMemory memory) {
    retire(memories_, memory);
  }
  void retire_module(AotModule *module) {
    retire(modules_, module);
  }
  void clear() {
    if (!entries_.empty())
      runtime_.wait();
    entries_.clear();
    modules_.clear();
    memories_.clear();
  }

 private:
  template <typename Map, typename Key>
  void retire(Map &map, Key key) {
    auto found = map.find(key);
    if (found == map.end())
      return;
    // Retirement is a cold boundary. Reverse indexes visit only dependent
    // executions and remove all their edges; warm replay never scans owners.
    const std::vector<TiPreparedExecution> handles(found->second.begin(),
                                                   found->second.end());
    runtime_.wait();
    for (auto handle : handles) {
      auto &entry = get(handle);
      remove_dependencies(handle, entry);
      entry.invalidate();
    }
  }
  template <typename Map, typename Key>
  void remove_edge(Map &map, Key key, TiPreparedExecution handle) {
    auto found = map.find(key);
    if (found != map.end()) {
      found->second.erase(handle);
      if (found->second.empty())
        map.erase(found);
    }
  }
  void remove_dependencies(TiPreparedExecution handle,
                           PreparedExecution &entry) {
    remove_edge(modules_, entry.owner, handle);
    for (auto memory : entry.memories)
      remove_edge(memories_, memory, handle);
    entry.owner = nullptr;
    entry.memories.clear();
  }
  Runtime &runtime_;
  std::unordered_map<TiPreparedExecution, std::unique_ptr<PreparedExecution>>
      entries_;
  std::unordered_map<AotModule *, std::unordered_set<TiPreparedExecution>>
      modules_;
  std::unordered_map<TiMemory, std::unordered_set<TiPreparedExecution>>
      memories_;
};
}  // namespace capi

capi::PreparedExecutionRegistry &Runtime::prepared_registry() {
  if (!prepared_)
    prepared_ = std::make_shared<capi::PreparedExecutionRegistry>(*this);
  return *prepared_;
}
void Runtime::retire_prepared_memory(TiMemory memory) {
  if (prepared_)
    prepared_->retire_memory(memory);
}
void Runtime::retire_prepared_module(AotModule *owner) {
  if (prepared_)
    prepared_->retire_module(owner);
}
void Runtime::clear_prepared() {
  std::lock_guard<std::recursive_mutex> lock(submission_mutex());
  if (prepared_)
    prepared_->clear();
}

TiPreparedExecution ti_prepare_kernel_ext(TiRuntime runtime,
                                          TiKernel kernel,
                                          uint32_t arg_count,
                                          const TiArgument *args,
                                          uint32_t binding_count,
                                          const TiNdArrayBinding *bindings) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  auto arguments = capi::bind_kernel_arguments(rt, kernel, arg_count, args,
                                               binding_count, bindings);
  if (!arguments)
    return TI_NULL_HANDLE;
  auto prepared = std::make_unique<capi::PreparedExecution>();
  prepared->owner = arguments->owner;
  prepared->memories = arguments->memories;
  prepared->kernel_arguments = std::move(arguments);
  prepared->prepare(rt.arch, {{prepared->kernel_arguments->kernel,
                               prepared->kernel_arguments->context.get()}});
  return rt.prepared_registry().publish(std::move(prepared));
  TI_CAPI_TRY_CATCH_END();
  return TI_NULL_HANDLE;
}

TiPreparedExecution ti_prepare_compute_graph_ext(
    TiRuntime runtime,
    TiComputeGraph graph,
    uint32_t arg_count,
    const TiNamedArgument *args,
    uint32_t binding_count,
    const TiNamedNdArrayBinding *bindings) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  auto arguments = capi::bind_graph_arguments(rt, graph, arg_count, args,
                                              binding_count, bindings);
  if (!arguments)
    return TI_NULL_HANDLE;
  auto prepared = std::make_unique<capi::PreparedExecution>();
  prepared->owner = arguments->owner;
  prepared->memories = arguments->memories;
  prepared->graph_arguments = std::move(arguments);
  const auto &bound = *prepared->graph_arguments;
  using namespace taichi::lang;
  std::vector<std::pair<aot::Kernel *, LaunchContextBuilder *>> dispatches;
  for (const auto &dispatch : bound.graph->dispatches) {
    if (!dispatch.compiled_kernel || dispatch.cuda_bounded_dispatch ||
        dispatch.indirect_dispatch_arg)
      throw MemoryBindingError(
          "Prepared AOT graph requires ordinary loaded dispatches");
    auto context =
        std::make_unique<LaunchContextBuilder>(dispatch.compiled_kernel);
    aot::CompiledGraph::init_runtime_context(dispatch.symbolic_args,
                                             bound.arguments, *context);
    for (int i = 0; i < dispatch.symbolic_args.size(); ++i) {
      const auto &name = dispatch.symbolic_args[i].name;
      const auto grad = bound.gradients.find(name);
      if (grad != bound.gradients.end()) {
        auto *array =
            reinterpret_cast<const Ndarray *>(bound.arguments.at(name).val);
        context->set_arg_ndarray_with_grad({i}, *array, *grad->second);
      }
    }
    dispatches.emplace_back(dispatch.compiled_kernel, context.get());
    prepared->contexts.push_back(std::move(context));
  }
  prepared->prepare(rt.arch, dispatches);
  return rt.prepared_registry().publish(std::move(prepared));
  TI_CAPI_TRY_CATCH_END();
  return TI_NULL_HANDLE;
}

void ti_launch_prepared_ext(TiRuntime runtime, TiPreparedExecution execution) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(execution);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  rt.prepared_registry().get(execution).run();
  TI_CAPI_TRY_CATCH_END();
}
void ti_get_prepared_execution_info_ext(TiRuntime runtime,
                                        TiPreparedExecution execution,
                                        TiPreparedExecutionInfo *info) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(info);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  *info = rt.prepared_registry().get(execution).info;
  TI_CAPI_TRY_CATCH_END();
}
void ti_destroy_prepared_ext(TiRuntime runtime, TiPreparedExecution execution) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(execution);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  rt.prepared_registry().destroy(execution);
  TI_CAPI_TRY_CATCH_END();
}
