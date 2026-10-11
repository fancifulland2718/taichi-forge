#include "argument_binding.h"

#include <algorithm>
#include <unordered_set>
#include <limits>
#include "taichi/aot/execution_plan.h"
#include "taichi/runtime/fixed_execution.h"
#ifdef TI_WITH_VULKAN
#include "taichi_gfx_impl.h"
#endif

namespace capi {
using namespace taichi::lang;

struct PreparedExecution {
  AotModule *owner{nullptr};
  std::vector<TiMemory> memories;
  TiPreparedExecutionInfo info{};
  bool live{false};
  // The backend consumes these cold descriptors before they are released.
  std::unique_ptr<BoundKernelArguments> kernel_arguments;
  std::unique_ptr<BoundGraphArguments> graph_arguments;
  std::vector<std::unique_ptr<LaunchContextBuilder>> contexts;
  std::vector<std::unique_ptr<BoundKernelArguments>> plan_arguments;
  std::unique_ptr<FixedExecution> native;

  void release_source_bindings() {
    contexts.clear();
    graph_arguments.reset();
    kernel_arguments.reset();
    plan_arguments.clear();
  }

  void prepare_operations(
      Runtime &runtime,
      const std::vector<FixedExecution::Operation> &operations) {
    gfx::GfxRuntime *gfx_runtime = nullptr;
#ifdef TI_WITH_VULKAN
    if (auto *gfx = dynamic_cast<::GfxRuntime *>(&runtime))
      gfx_runtime = &gfx->get_gfx_runtime();
#endif
    try {
      native = std::make_unique<FixedExecution>(runtime.arch, runtime.get(),
                                                gfx_runtime, operations);
    } catch (const std::invalid_argument &error) {
      throw MemoryBindingError(error.what());
    }
    info.dispatch_count = static_cast<uint32_t>(std::count_if(
        operations.begin(), operations.end(), [](const auto &operation) {
          return std::holds_alternative<FixedExecution::Dispatch>(operation);
        }));
    info.execution_kind = native->execution_kind();
    info.argument_bytes = native->argument_bytes();
    release_source_bindings();
    live = true;
  }

  void prepare(
      Runtime &runtime,
      const std::vector<std::pair<aot::Kernel *, LaunchContextBuilder *>>
          &dispatches) {
    std::vector<FixedExecution::Operation> operations;
    for (const auto &[kernel, context] : dispatches)
      operations.emplace_back(FixedExecution::Dispatch{kernel, context});
    prepare_operations(runtime, operations);
  }

  void run() {
    if (!live)
      throw MemoryBindingError("Prepared execution has a retired dependency");
    native->run();
    ++info.launches;
  }

  void invalidate() {
    live = false;
    native.reset();
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
  prepared->prepare(rt, {{prepared->kernel_arguments->kernel,
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
  prepared->prepare(rt, dispatches);
  return rt.prepared_registry().publish(std::move(prepared));
  TI_CAPI_TRY_CATCH_END();
  return TI_NULL_HANDLE;
}

TiPreparedExecution ti_prepare_execution_plan_ext(
    TiRuntime runtime,
    TiAotModule module,
    uint64_t description_size,
    const char *description,
    uint32_t arg_count,
    const TiNamedArgument *args,
    uint32_t binding_count,
    const TiNamedNdArrayBinding *bindings) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(module);
  TI_CAPI_ARGUMENT_NULL_RV(description);
  if ((arg_count && !args) || (binding_count && !bindings) ||
      description_size > std::numeric_limits<std::size_t>::max())
    throw MemoryBindingError("Invalid execution plan input descriptors");
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  auto &mod = *reinterpret_cast<AotModule *>(module);
  if (&mod.runtime() != &rt)
    throw MemoryBindingError(
        "Execution plan module belongs to another runtime");
  using namespace taichi::lang;
  aot::ExecutionPlan plan;
  try {
    plan =
        aot::ExecutionPlan::parse(std::string(description, description_size));
  } catch (const std::invalid_argument &error) {
    throw MemoryBindingError(error.what());
  }
  std::unordered_map<std::string, const TiArgument *> arguments;
  for (uint32_t i = 0; i < arg_count; ++i) {
    if (!args[i].name || !*args[i].name ||
        !arguments.emplace(args[i].name, &args[i].argument).second)
      throw MemoryBindingError(
          "Execution plan argument names must be unique and nonempty");
  }
  if (arguments.size() != plan.arguments.size())
    throw MemoryBindingError(
        "Execution plan argument set does not match its slots");
  for (const auto &name : plan.arguments) {
    if (!arguments.count(name))
      throw MemoryBindingError("Missing execution plan argument: " + name);
  }
  std::unordered_map<std::string, const TiNamedNdArrayBinding *> layouts;
  std::unordered_set<std::string> used_bindings;
  for (uint32_t i = 0; i < binding_count; ++i) {
    if (!bindings[i].name || !arguments.count(bindings[i].name) ||
        !layouts.emplace(bindings[i].name, &bindings[i]).second)
      throw MemoryBindingError(
          "Invalid or duplicate execution plan ndarray binding");
  }

  auto prepared = std::make_unique<capi::PreparedExecution>();
  prepared->owner = &mod;
  struct Storage {
    DevicePtr pointer;
    uint64_t size;
  };
  auto storage = [&](const std::string &name, uint64_t offset) {
    const auto &arg = *arguments.at(name);
    if (arg.type != TI_ARGUMENT_TYPE_NDARRAY)
      throw MemoryBindingError("Execution memory operations require ndarrays");
    const auto &array = arg.value.ndarray;
    const auto bytes = capi::validated_ndarray_bytes(rt, array);
    if (offset > bytes)
      throw MemoryBindingError(
          "Execution memory offset exceeds ndarray extent");
    const auto allocation = rt.get_memory(array.memory);
    prepared->memories.push_back(array.memory);
    return Storage{allocation.allocation.get_ptr(offset), bytes - offset};
  };
  std::vector<FixedExecution::Operation> operations;
  for (const auto &operation : plan.operations) {
    if (const auto *kernel =
            std::get_if<aot::ExecutionPlan::Kernel>(&operation)) {
      auto handle = mod.get_kernel_handle(kernel->entry);
      if (!handle)
        throw MemoryBindingError("Execution plan kernel was not exported: " +
                                 kernel->entry);
      std::vector<TiArgument> kernel_args;
      std::vector<TiNdArrayBinding> kernel_bindings;
      for (std::size_t i = 0; i < kernel->arguments.size(); ++i) {
        const auto &name = kernel->arguments[i];
        kernel_args.push_back(*arguments.at(name));
        if (auto binding = layouts.find(name); binding != layouts.end()) {
          used_bindings.insert(name);
          const auto &value = *binding->second;
          kernel_bindings.push_back({static_cast<uint32_t>(i),
                                     value.layout_version, value.layout,
                                     value.gradient});
        }
      }
      auto bound = capi::bind_kernel_arguments(
          rt, handle, static_cast<uint32_t>(kernel_args.size()),
          kernel_args.data(), static_cast<uint32_t>(kernel_bindings.size()),
          kernel_bindings.data());
      if (!bound)
        return TI_NULL_HANDLE;
      prepared->memories.insert(prepared->memories.end(),
                                bound->memories.begin(), bound->memories.end());
      operations.emplace_back(
          FixedExecution::Dispatch{bound->kernel, bound->context.get()});
      prepared->plan_arguments.push_back(std::move(bound));
    } else if (const auto *copy =
                   std::get_if<aot::ExecutionPlan::Copy>(&operation)) {
      const auto src = storage(copy->source, copy->source_offset);
      const auto dst = storage(copy->destination, copy->destination_offset);
      const auto bytes = copy->bytes.value_or(src.size);
      if (bytes > src.size || bytes > dst.size ||
          bytes > std::numeric_limits<std::size_t>::max())
        throw MemoryBindingError("Execution copy exceeds ndarray extent");
      if (bytes && src.pointer.alloc_id == dst.pointer.alloc_id &&
          src.pointer.offset < dst.pointer.offset + bytes &&
          dst.pointer.offset < src.pointer.offset + bytes)
        throw MemoryBindingError("Execution copy slices must not overlap");
      operations.emplace_back(FixedExecution::BufferCopy{
          dst.pointer, src.pointer, static_cast<std::size_t>(bytes)});
    } else {
      const auto &fill = std::get<aot::ExecutionPlan::Fill>(operation);
      const auto dst = storage(fill.destination, fill.offset);
      const auto bytes = fill.bytes.value_or(dst.size);
      const auto &array = arguments.at(fill.destination)->value.ndarray;
      if (bytes > dst.size || bytes > std::numeric_limits<std::size_t>::max() ||
          rt.get_memory(array.memory).base_offset % 4)
        throw MemoryBindingError(
            "Execution fill exceeds ndarray extent or alignment");
      operations.emplace_back(FixedExecution::BufferFill{
          dst.pointer, static_cast<std::size_t>(bytes), fill.value});
    }
  }
  if (used_bindings.size() != layouts.size())
    throw MemoryBindingError(
        "Ndarray layout/gradient binding is unused by any kernel");
  prepared->prepare_operations(rt, operations);
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
