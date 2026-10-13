#include "argument_binding.h"
#include "execution_resources.h"

#include <algorithm>
#include <unordered_set>
#include <set>
#include <limits>
#include "taichi/aot/execution_plan.h"
#include "taichi/runtime/fixed_execution.h"
#include "taichi/runtime/native_operation.h"
#include "taichi/runtime/native_operation_registry.h"
#ifdef TI_WITH_VULKAN
#include "taichi_gfx_impl.h"
#endif

namespace capi {
using namespace taichi::lang;

struct PreparedExecution {
  AotModule *owner{nullptr};
  std::vector<TiMemory> memories;
  std::unordered_set<TiMemory> replay_memories;
  std::unordered_set<const NativeResource *> replay_resources;
  TiPreparedExecutionInfo info{};
  bool live{false};
  // Declared before backend recordings so owned buffers are destroyed last.
  std::unique_ptr<ExecutionResourceStorage> resources;
  std::vector<std::shared_ptr<NativeResource>> native_resources;
  uint64_t native_resource_workspace_bytes{0};
  // The backend consumes these cold descriptors before they are released.
  std::unique_ptr<BoundKernelArguments> kernel_arguments;
  std::unique_ptr<BoundGraphArguments> graph_arguments;
  std::vector<std::unique_ptr<LaunchContextBuilder>> contexts;
  std::vector<std::unique_ptr<BoundKernelArguments>> plan_arguments;
  std::unique_ptr<FixedExecution> native;
  std::unique_ptr<FixedExecution> initialization;

  void release_native_resources() {
    while (!native_resources.empty())
      native_resources.pop_back();
    native_resource_workspace_bytes = 0;
  }
  void refresh_resource_workspace_bytes() {
    uint64_t total = 0;
    for (const auto &resource : native_resources) {
      const auto bytes = resource->workspace_bytes();
      if (bytes > std::numeric_limits<uint64_t>::max() - total)
        throw MemoryBindingError("Native resource workspace size overflow");
      total += bytes;
    }
    native_resource_workspace_bytes = total;
  }
  ~PreparedExecution() {
    initialization.reset();
    native.reset();
    release_native_resources();
  }

  void release_source_bindings() {
    contexts.clear();
    graph_arguments.reset();
    kernel_arguments.reset();
    plan_arguments.clear();
  }

  void prepare_operations(
      Runtime &runtime,
      const std::vector<FixedExecution::Operation> &operations,
      const std::vector<FixedExecution::Operation> &initializers = {}) {
    gfx::GfxRuntime *gfx_runtime = nullptr;
#ifdef TI_WITH_VULKAN
    if (auto *gfx = dynamic_cast<::GfxRuntime *>(&runtime))
      gfx_runtime = &gfx->get_gfx_runtime();
#endif
    try {
      NativeRecordingState sequence;
      if (!initializers.empty())
        initialization = std::make_unique<FixedExecution>(
            runtime.arch, runtime.get(), gfx_runtime, initializers, &sequence);
      native = std::make_unique<FixedExecution>(
          runtime.arch, runtime.get(), gfx_runtime, operations, &sequence);
    } catch (const std::invalid_argument &error) {
      throw MemoryBindingError(error.what());
    }
    info.dispatch_count = static_cast<uint32_t>(std::count_if(
        operations.begin(), operations.end(), [](const auto &operation) {
          return std::holds_alternative<FixedExecution::Dispatch>(operation);
        }));
    info.execution_kind = native->execution_kind();
    info.argument_bytes =
        native->argument_bytes() +
        (initialization ? initialization->argument_bytes() : 0);
    refresh_resource_workspace_bytes();
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
    initialization.reset();
    native.reset();
    release_native_resources();
    release_source_bindings();
    resources.reset();
  }

  void initialize(Runtime &runtime) {
    if (!live)
      throw MemoryBindingError("Prepared execution has a retired dependency");
    if (!initialization)
      return;
    try {
      initialization->run();
      runtime.wait();
      // Commands must retire before any of their input or scratch storage.
      initialization.reset();
      for (std::size_t i = native_resources.size(); i-- > 0;)
        if (!replay_resources.count(native_resources[i].get()))
          native_resources.erase(native_resources.begin() + i);
      refresh_resource_workspace_bytes();
      if (resources)
        resources->retain(replay_memories);
      info.argument_bytes = native->argument_bytes();
    } catch (...) {
      // Initialization may have partially written data. Retain its leases
      // until a synchronized close/owner retirement; never retry implicitly.
      live = false;
      throw;
    }
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
    if (entry.native || entry.initialization)
      runtime_.wait();
    remove_dependencies(handle, entry);
    entries_.erase(handle);
  }
  void initialize(TiPreparedExecution handle) {
    auto &entry = get(handle);
    const bool needed = bool(entry.initialization);
    entry.initialize(runtime_);
    if (!needed)
      return;
    for (auto memory : entry.memories)
      if (!entry.replay_memories.count(memory))
        remove_edge(memories_, memory, handle);
    entry.memories.erase(
        std::remove_if(entry.memories.begin(), entry.memories.end(),
                       [&](TiMemory memory) {
                         return !entry.replay_memories.count(memory);
                       }),
        entry.memories.end());
    entry.replay_memories.clear();
    entry.replay_resources.clear();
  }
  void run(TiPreparedExecution handle) {
    auto &entry = get(handle);
    if (entry.initialization)
      initialize(handle);
    entry.run();
  }
  void retire_memory(TiMemory memory) {
    retire(memories_, memory);
  }
  void retire_module(AotModule *module) {
    retire(modules_, module);
  }
  void clear(bool synchronize) {
    if (synchronize && !entries_.empty())
      runtime_.wait();
    modules_.clear();
    memories_.clear();
    entries_.clear();
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
void Runtime::clear_prepared(bool synchronize) {
  std::lock_guard<std::recursive_mutex> lock(submission_mutex());
  if (prepared_)
    prepared_->clear(synchronize);
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
  // Optional providers are admitted before allocating private plan resources.
  // There is no dependency probing on replay, nor on unrelated operations.
  NativePreparationContext preparation(rt.arch, rt.get(),
                                       rt.native_provider_locations());
  std::set<std::pair<std::string, std::uint32_t>> probed;
  std::map<const aot::ExecutionPlan::Native *,
           NativeOperationRegistry::DecodedOperation>
      decoded;
  auto preflight = [&](const aot::ExecutionPlan::Native *native,
                       NativeObjectKind kind) {
    try {
      auto request = native_operation_registry().decode(*native, kind);
      if (probed.emplace(request.kind(), request.version()).second &&
          !native_operation_registry()
               .probe(request.kind(), request.version(), rt.arch,
                      preparation.providers, kind)
               .available)
        throw MemoryBindingError(native->provider + "/" + native->operation +
                                 " provider unavailable on this runtime");
      decoded.emplace(native, std::move(request));
    } catch (const std::invalid_argument &error) {
      throw MemoryBindingError(error.what());
    }
  };
  std::vector<const aot::ExecutionPlan::Operation *> steps;
  for (const auto &operation : plan.initialization)
    steps.push_back(&operation);
  for (const auto &operation : plan.operations)
    steps.push_back(&operation);
  for (const auto index : plan.resources.order)
    if (const auto *native = std::get_if<NativeOperationDescription>(
            &plan.resources.declarations[index].value))
      preflight(native, NativeObjectKind::resource);
  for (const auto *step : steps) {
    const auto &operation = *step;
    if (const auto *native =
            std::get_if<aot::ExecutionPlan::Native>(&operation))
      preflight(native, NativeObjectKind::operation);
  }
  auto resource_bindings = capi::bind_execution_resources(
      rt, mod.get(), plan.resources, arguments, layouts);
  prepared->resources = std::move(resource_bindings.storage);
  struct Storage {
    DevicePtr pointer;
    uint64_t size;
    uint64_t element_bytes;
  };
  auto storage = [&](const std::string &name, uint64_t offset) {
    const auto found = arguments.find(name);
    if (found == arguments.end())
      throw MemoryBindingError(
          "Execution storage requires an ndarray binding: " + name);
    const auto &arg = *found->second;
    if (arg.type != TI_ARGUMENT_TYPE_NDARRAY)
      throw MemoryBindingError("Execution memory operations require ndarrays");
    const auto &array = arg.value.ndarray;
    uint64_t element_bytes = 0;
    const auto bytes = capi::validated_ndarray_bytes(rt, array, &element_bytes);
    if (offset > bytes)
      throw MemoryBindingError(
          "Execution memory offset exceeds ndarray extent");
    const auto allocation = rt.get_memory(array.memory);
    prepared->memories.push_back(array.memory);
    return Storage{allocation.allocation.get_ptr(offset), bytes - offset,
                   element_bytes};
  };
  std::map<std::string, NativeArrayBinding> native_arrays;
  std::map<std::string, std::shared_ptr<NativeResource>> native_objects;
  auto array_binding =
      [&](const std::string &slot) -> const NativeArrayBinding & {
    auto found = native_arrays.find(slot);
    if (found == native_arrays.end()) {
      const auto view = storage(slot, 0);
      const auto &array = arguments.at(slot)->value.ndarray;
      found =
          native_arrays
              .emplace(slot,
                       NativeArrayBinding{
                           view.pointer,
                           view.size,
                           capi::ndarray_scalar_type(array.elem_type),
                           {array.shape.dims,
                            array.shape.dims + array.shape.dim_count},
                           {array.elem_shape.dims,
                            array.elem_shape.dims + array.elem_shape.dim_count},
                           rt.get_memory(array.memory).base_offset})
              .first;
    }
    return found->second;
  };
  std::map<std::string, NativeStorageRange> storage_ranges;
  auto storage_range =
      [&](const std::string &slot) -> const NativeStorageRange & {
    auto found = storage_ranges.find(slot);
    if (found == storage_ranges.end()) {
      const auto &array = array_binding(slot);
      found = storage_ranges
                  .emplace(slot, resolve_native_storage_range(
                                     rt.arch, array.pointer, array.bytes))
                  .first;
    }
    return found->second;
  };
  // Resolve certificates before creating provider state or recording commands.
  // These maps disappear after prepare; replay carries only frozen bindings.
  for (const auto &constraint : plan.constraints) {
    try {
      if (const auto *layout =
              std::get_if<aot::ExecutionPlan::ArrayLayout>(&constraint)) {
        validate_native_array_layout(array_binding(layout->array),
                                     layout->required);
      } else {
        const auto &pair = std::get<aot::ExecutionPlan::Disjoint>(constraint);
        const auto &left = storage_range(pair.left);
        const auto &right = storage_range(pair.right);
        bool accepted = left.proven_disjoint(right);
        if (!accepted && pair.allow_same_view &&
            left.proven_same_range(right)) {
          const auto &a = array_binding(pair.left);
          const auto &b = array_binding(pair.right);
          accepted = a.scalar_type == b.scalar_type && a.shape == b.shape &&
                     a.element_shape == b.element_shape;
        }
        if (!accepted)
          throw std::invalid_argument(
              std::string("Cannot prove array bindings disjoint") +
              (pair.allow_same_view ? " or the same view: " : ": ") +
              pair.left + " and " + pair.right);
      }
    } catch (const std::invalid_argument &error) {
      throw MemoryBindingError(error.what());
    }
  }
  auto native_inputs = [&](const NativeOperationDescription &description) {
    NativeOperationBindings inputs;
    for (const auto &[port, slot] : description.bindings) {
      if (const auto object = native_objects.find(slot);
          object != native_objects.end()) {
        inputs.emplace(port, object->second);
        continue;
      }
      inputs.emplace(port, array_binding(slot));
    }
    return inputs;
  };
  for (const auto index : plan.resources.order) {
    const auto &resource = plan.resources.declarations[index];
    if (const auto *native =
            std::get_if<NativeOperationDescription>(&resource.value)) {
      try {
        auto object = native_operation_registry().prepare_resource(
            preparation, decoded.at(native), native_inputs(*native));
        prepared->native_resources.push_back(object);
        native_objects.emplace(resource.name, std::move(object));
      } catch (const std::invalid_argument &error) {
        throw MemoryBindingError(error.what());
      }
    }
  }
  std::vector<FixedExecution::Operation> operations;
  for (const auto &zero : resource_bindings.zero_initializations) {
    const auto allocation = rt.get_memory(zero.memory);
    operations.emplace_back(
        FixedExecution::BufferFill{allocation.allocation.get_ptr(0),
                                   static_cast<std::size_t>(zero.bytes), 0});
  }
  const auto initialization_count =
      operations.size() + plan.initialization.size();
  for (const auto *step : steps) {
    const auto &operation = *step;
    if (const auto *kernel =
            std::get_if<aot::ExecutionPlan::Kernel>(&operation)) {
      auto handle = mod.get_kernel_handle(kernel->entry);
      if (!handle)
        throw MemoryBindingError("Execution plan kernel was not exported: " +
                                 kernel->entry);
      std::vector<TiArgument> kernel_args;
      std::vector<TiNdArrayBinding> kernel_bindings;
      const auto &parameters = rt.get_kernel(handle).kernel->nested_parameters;
      for (std::size_t i = 0; i < kernel->arguments.size(); ++i) {
        const auto &name = kernel->arguments[i];
        if (native_objects.count(name))
          throw MemoryBindingError(
              "Kernel arguments cannot bind native resources: " + name);
        kernel_args.push_back(*arguments.at(name));
        if (auto binding = layouts.find(name); binding != layouts.end()) {
          const auto &value = *binding->second;
          const auto parameter = parameters.find({static_cast<int>(i)});
          // A plan-wide primal may also be read by ordinary helper kernels.
          // Supply its gradient only where the compiled dispatch ABI uses it.
          // Keep malformed/structure bindings for the common validator, and
          // reject gradients unused by every dispatch at the end of prepare.
          if (value.gradient && !value.layout && value.layout_version == 0 &&
              (parameter == parameters.end() || !parameter->second.needs_grad))
            continue;
          used_bindings.insert(name);
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
      bound->context->append_dispatch_label(kernel->label);
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
    } else if (const auto *native =
                   std::get_if<aot::ExecutionPlan::Native>(&operation)) {
      try {
        operations.emplace_back(native_operation_registry().prepare(
            preparation, decoded.at(native), native_inputs(*native)));
      } catch (const std::invalid_argument &error) {
        throw MemoryBindingError(error.what());
      }
    } else if (const auto *sort =
                   std::get_if<aot::ExecutionPlan::Sort>(&operation)) {
      const auto keys = storage(sort->keys, 0);
      const auto &array = arguments.at(sort->keys)->value.ndarray;
      if (array.shape.dim_count != 1 || array.elem_shape.dim_count)
        throw MemoryBindingError("Sort requires 1D scalar keys");
      NativeSortKeyType key_type;
      switch (array.elem_type) {
        case TI_DATA_TYPE_U32:
          key_type = NativeSortKeyType::u32;
          break;
        case TI_DATA_TYPE_I32:
          key_type = NativeSortKeyType::i32;
          break;
        case TI_DATA_TYPE_F32:
          key_type = NativeSortKeyType::f32;
          break;
        case TI_DATA_TYPE_U64:
          key_type = NativeSortKeyType::u64;
          break;
        case TI_DATA_TYPE_I64:
          key_type = NativeSortKeyType::i64;
          break;
        case TI_DATA_TYPE_F64:
          key_type = NativeSortKeyType::f64;
          break;
        default:
          throw MemoryBindingError("Sort keys require i32/u32/f32/i64/u64/f64");
      }
      if (rt.get_memory(array.memory).base_offset % keys.element_bytes)
        throw MemoryBindingError("Sort key storage is misaligned");
      SortOperation descriptor{keys.pointer, {}, array.shape.dims[0], key_type};
      if (sort->values) {
        const auto values = storage(*sort->values, 0);
        const auto &payload = arguments.at(*sort->values)->value.ndarray;
        if (payload.shape.dim_count != 1 ||
            payload.shape.dims[0] != array.shape.dims[0] ||
            !values.element_bytes || values.element_bytes % 4 ||
            rt.get_memory(payload.memory).base_offset % 4)
          throw MemoryBindingError(
              "Sort payload requires matching 1D raw-word records");
        if (keys.size && values.size &&
            keys.pointer.alloc_id == values.pointer.alloc_id &&
            (keys.pointer.offset <= values.pointer.offset
                 ? values.pointer.offset - keys.pointer.offset < keys.size
                 : keys.pointer.offset - values.pointer.offset < values.size))
          throw MemoryBindingError(
              "Sort key and payload ranges must not overlap");
        descriptor.values = values.pointer;
        descriptor.value_bytes = static_cast<size_t>(values.element_bytes);
        descriptor.value_scalar_bytes = descriptor.value_bytes;
        for (uint32_t i = 0; i < payload.elem_shape.dim_count; ++i)
          descriptor.value_scalar_bytes /= payload.elem_shape.dims[i];
      }
      try {
        operations.emplace_back(
            prepare_sort_operation(rt.arch, rt.get(), descriptor));
      } catch (const std::invalid_argument &error) {
        throw MemoryBindingError(error.what());
      }
    } else if (const auto *compact =
                   std::get_if<aot::ExecutionPlan::Compact>(&operation)) {
      auto compact_storage = [&](const std::string &name) {
        auto result = storage(name, 0);
        if (rt.get_memory(arguments.at(name)->value.ndarray.memory)
                .base_offset %
            4)
          throw MemoryBindingError("Compact storage must be 4-byte aligned");
        return result;
      };
      const auto flag_storage = compact_storage(compact->flags);
      const auto count_storage = compact_storage(compact->count);
      const auto &flags = arguments.at(compact->flags)->value.ndarray;
      const auto &count = arguments.at(compact->count)->value.ndarray;
      if (flags.shape.dim_count != 1 || flags.elem_type != TI_DATA_TYPE_I32 ||
          flags.elem_shape.dim_count || count.elem_type != TI_DATA_TYPE_I32 ||
          count.elem_shape.dim_count || count.shape.dim_count > 1 ||
          (count.shape.dim_count == 1 && count.shape.dims[0] < 1))
        throw MemoryBindingError(
            "Compact requires 1D i32 flags and a writable i32 count");
      CompactOperation descriptor{
          {}, flag_storage.pointer, count_storage.pointer, flags.shape.dims[0]};
      for (const auto &column : compact->columns) {
        const auto source = compact_storage(column.values);
        const auto destination = compact_storage(column.output);
        const auto &values = arguments.at(column.values)->value.ndarray;
        const auto &output = arguments.at(column.output)->value.ndarray;
        if (values.shape.dim_count != 1 || output.shape.dim_count != 1 ||
            values.shape.dims[0] != flags.shape.dims[0] ||
            output.shape.dims[0] < values.shape.dims[0] ||
            values.elem_type != output.elem_type ||
            values.elem_shape.dim_count != output.elem_shape.dim_count)
          throw MemoryBindingError(
              "Compact requires matching 1D records and sufficient output "
              "capacity");
        for (uint32_t i = 0; i < values.elem_shape.dim_count; ++i)
          if (values.elem_shape.dims[i] != output.elem_shape.dims[i])
            throw MemoryBindingError(
                "Compact input and output element shapes differ");
        descriptor.columns.push_back(
            {source.pointer, destination.pointer,
             static_cast<size_t>(source.element_bytes)});
      }
      try {
        operations.emplace_back(
            prepare_compact_operation(rt.arch, rt.get(), descriptor));
      } catch (const std::invalid_argument &error) {
        throw MemoryBindingError(error.what());
      }
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
  if (initialization_count) {
    for (const auto &name : plan.replay_bindings) {
      if (const auto object = native_objects.find(name);
          object != native_objects.end()) {
        prepared->replay_resources.insert(object->second.get());
        continue;
      }
      const auto &argument = *arguments.at(name);
      if (argument.type == TI_ARGUMENT_TYPE_NDARRAY)
        prepared->replay_memories.insert(argument.value.ndarray.memory);
      if (const auto binding = layouts.find(name);
          binding != layouts.end() && binding->second->gradient)
        prepared->replay_memories.insert(binding->second->gradient->memory);
    }
  }
  if (prepared->resources)
    prepared->resources->remove_owned_dependencies(prepared->memories);
  const auto split = operations.begin() + initialization_count;
  const std::vector<FixedExecution::Operation> initializers(operations.begin(),
                                                            split);
  operations.erase(operations.begin(), split);
  prepared->prepare_operations(rt, operations, initializers);
  return rt.prepared_registry().publish(std::move(prepared));
  TI_CAPI_TRY_CATCH_END();
  return TI_NULL_HANDLE;
}

void Runtime::configure_native_provider(
    std::string provider,
    std::optional<taichi::lang::NativeProviderLocation> location) {
  if (provider.empty() || provider.find('\0') != std::string::npos)
    throw MemoryBindingError("Provider name must be nonempty without NUL");
  if (location) {
    try {
      taichi::lang::validate_native_provider_location(provider, *location);
    } catch (const std::invalid_argument &error) {
      throw MemoryBindingError(error.what());
    }
    native_provider_locations_.insert_or_assign(std::move(provider),
                                                std::move(*location));
  } else {
    native_provider_locations_.erase(provider);
  }
}

void ti_configure_native_provider_ext(TiRuntime runtime,
                                      const char *provider,
                                      const char *adapter_path,
                                      const char *library_path) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(provider);
  TI_CAPI_INVALID_ARGUMENT(!adapter_path && library_path && *library_path);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  std::optional<taichi::lang::NativeProviderLocation> location;
  if (adapter_path)
    location = {adapter_path, library_path ? library_path : ""};
  rt.configure_native_provider(provider, std::move(location));
  TI_CAPI_TRY_CATCH_END();
}

TiFlags ti_get_native_operation_capabilities_ext(TiRuntime runtime,
                                                 const char *kind,
                                                 uint32_t version) {
  TI_CAPI_TRY_CATCH_BEGIN();
  if (!runtime || !kind) {
    ti_set_last_error(TI_ERROR_ARGUMENT_NULL, "runtime or operation kind");
    return 0;
  }
  if (!*kind || !version) {
    ti_set_last_error(TI_ERROR_INVALID_ARGUMENT, "operation kind or version");
    return 0;
  }
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  const auto *capability =
      taichi::lang::native_operation_registry().find(kind, version, rt.arch);
  static_assert(static_cast<TiFlags>(taichi::lang::NativeOperationMode::cpu) ==
                TI_NATIVE_OPERATION_CAPABILITY_CPU_BIT_EXT);
  static_assert(
      static_cast<TiFlags>(taichi::lang::NativeOperationMode::cuda_capture) ==
      TI_NATIVE_OPERATION_CAPABILITY_CUDA_CAPTURE_BIT_EXT);
  static_assert(
      static_cast<TiFlags>(taichi::lang::NativeOperationMode::vulkan_inline) ==
      TI_NATIVE_OPERATION_CAPABILITY_VULKAN_INLINE_BIT_EXT);
  static_assert(static_cast<TiFlags>(
                    taichi::lang::NativeOperationMode::cuda_synchronous) ==
                TI_NATIVE_OPERATION_CAPABILITY_CUDA_SYNCHRONOUS_BIT_EXT);
  return capability ? static_cast<TiFlags>(capability->mode) : 0;
  TI_CAPI_TRY_CATCH_END();
  return 0;
}

void ti_probe_native_operation_ext(
    TiRuntime runtime,
    const char *kind,
    uint32_t version,
    TiNativeOperationAvailabilityExt *availability) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(availability);
  *availability = {};
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(kind);
  if (!*kind || !version) {
    ti_set_last_error(TI_ERROR_INVALID_ARGUMENT, "operation kind or version");
    return;
  }
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  const auto &registry = taichi::lang::native_operation_registry();
  const auto *capability = registry.find(kind, version, rt.arch);
  if (capability) {
    taichi::lang::NativeProviderSession providers(
        rt.native_provider_locations());
    const auto status = registry.probe(kind, version, rt.arch, providers);
    availability->capabilities = static_cast<TiFlags>(capability->mode);
    availability->available = status.available ? TI_TRUE : TI_FALSE;
    availability->library_version = status.library_version;
  }
  TI_CAPI_TRY_CATCH_END();
}

void ti_initialize_prepared_ext(TiRuntime runtime,
                                TiPreparedExecution execution) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(execution);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  rt.prepared_registry().initialize(execution);
  TI_CAPI_TRY_CATCH_END();
}

void ti_launch_prepared_ext(TiRuntime runtime, TiPreparedExecution execution) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(execution);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  rt.prepared_registry().run(execution);
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
uint64_t ti_get_prepared_execution_workspace_bytes_ext(
    TiRuntime runtime,
    TiPreparedExecution execution) {
  TI_CAPI_TRY_CATCH_BEGIN();
  if (!runtime || !execution) {
    ti_set_last_error(TI_ERROR_ARGUMENT_NULL, "runtime or execution");
    return 0;
  }
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  const auto &prepared = rt.prepared_registry().get(execution);
  const auto operations =
      (prepared.native ? prepared.native->workspace_bytes() : 0) +
      (prepared.initialization ? prepared.initialization->workspace_bytes()
                               : 0);
  if (operations > std::numeric_limits<uint64_t>::max() -
                       prepared.native_resource_workspace_bytes)
    throw MemoryBindingError("Prepared workspace size overflow");
  return operations + prepared.native_resource_workspace_bytes;
  TI_CAPI_TRY_CATCH_END();
  return 0;
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

uint64_t ti_get_prepared_execution_resource_bytes_ext(
    TiRuntime runtime,
    TiPreparedExecution execution) {
  TI_CAPI_TRY_CATCH_BEGIN();
  if (!runtime || !execution) {
    ti_set_last_error(TI_ERROR_ARGUMENT_NULL, "runtime or execution");
    return 0;
  }
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  const auto &prepared = rt.prepared_registry().get(execution);
  return prepared.resources ? prepared.resources->bytes() : 0;
  TI_CAPI_TRY_CATCH_END();
  return 0;
}
