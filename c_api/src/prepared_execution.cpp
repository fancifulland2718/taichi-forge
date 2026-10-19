#include "argument_binding.h"
#include "execution_resources.h"
#include "execution_binding_family.h"
#include "native_resource_registry.h"

#include <algorithm>
#include <functional>
#include <unordered_set>
#include <set>
#include <limits>
#include "taichi/aot/execution_plan.h"
#include "taichi/runtime/fixed_execution.h"
#include "taichi/runtime/native_operation.h"
#include "taichi/runtime/native_sort.h"
#include "taichi/runtime/native_operation_registry.h"
#include "taichi/runtime/native_host_blob.h"
#include "taichi/runtime/native_image.h"
#include "taichi/program/runtime_fault.h"
#ifdef TI_WITH_CUDA
#include "taichi/runtime/cuda/binding_frame_replay.h"
#endif
#ifdef TI_WITH_VULKAN
#include "taichi_gfx_impl.h"
#endif

namespace capi {
using namespace taichi::lang;

struct PreparedExecution {
  AotModule *owner{nullptr};
  std::shared_ptr<ExecutionBindingFamily> family;
  std::vector<TiMemory> memories;
  std::unordered_set<TiMemory> replay_memories;
  std::vector<TiNativeResource> external_resources;
  std::unordered_set<TiNativeResource> replay_external_resources;
  std::unordered_set<const NativeResource *> replay_resources;
  TiPreparedExecutionInfo info{};
  bool live{false};
  bool binding_retired{false};
  // Declared before backend recordings so owned buffers are destroyed last.
  std::shared_ptr<ExecutionResourceStorage> resources;
  std::vector<std::shared_ptr<NativeResource>> native_resources;
  uint64_t native_resource_workspace_bytes{0};
  uint64_t native_resource_payload_bytes{0};
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
    native_resource_payload_bytes = 0;
  }
  void refresh_resource_bytes() {
    uint64_t workspace = 0, payload = 0;
    std::unordered_set<const NativeResource *> counted;
    counted.reserve(native_resources.size());
    for (const auto &resource : native_resources) {
      if (!counted.insert(resource.get()).second)
        continue;
      const auto bytes = resource->workspace_bytes();
      if (bytes > std::numeric_limits<uint64_t>::max() - workspace)
        throw MemoryBindingError("Native resource workspace size overflow");
      workspace += bytes;
      const auto contents = resource->device_payload_bytes();
      if (contents > std::numeric_limits<uint64_t>::max() - payload)
        throw MemoryBindingError("Native resource payload size overflow");
      payload += contents;
    }
    native_resource_workspace_bytes = workspace;
    native_resource_payload_bytes = payload;
  }
  uint64_t workspace_bytes() const {
    uint64_t total = native_resource_workspace_bytes;
    for (const auto bytes :
         {native ? native->workspace_bytes() : 0,
          initialization ? initialization->workspace_bytes() : 0}) {
      if (bytes > std::numeric_limits<uint64_t>::max() - total)
        throw MemoryBindingError("Prepared workspace size overflow");
      total += bytes;
    }
    return total;
  }
  uint64_t array_bytes() const {
    const auto local = resources ? resources->bytes() : 0;
    const auto shared = family && family->arrays->storage
                            ? family->arrays->storage->bytes()
                            : 0;
    if (shared > std::numeric_limits<uint64_t>::max() - local)
      throw MemoryBindingError("Prepared resource size overflow");
    return local + shared;
  }
  ~PreparedExecution() {
    initialization.reset();
    retire_binding();
    native.reset();
    release_native_resources();
  }

  void retire_binding(bool already_waited = false) noexcept {
    if (!family || !native || binding_retired)
      return;
    bool safe = family->runtime->runtime_fault_domain()->backend_calls_safe();
    if (safe && !already_waited) {
      try {
        family->runtime->wait();
      } catch (...) {
        safe = false;
      }
    }
    native->retire_completed_binding(safe);
    binding_retired = true;
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
      const std::vector<FixedExecution::Operation> &initializers = {},
      const VulkanSubmissionRecipe *submission_recipe = nullptr) {
    gfx::GfxRuntime *gfx_runtime = nullptr;
#ifdef TI_WITH_VULKAN
    if (auto *gfx = dynamic_cast<::GfxRuntime *>(&runtime))
      gfx_runtime = &gfx->get_gfx_runtime();
#endif
    try {
      NativeRecordingState sequence =
          family ? family->initial_facts : NativeRecordingState{};
      struct FrameOperands {
        std::shared_ptr<ExecutionResourceStorage> views;
        std::shared_ptr<SharedExecutionResources> shared;
        std::vector<std::shared_ptr<NativeResource>> objects;
      };
      std::shared_ptr<FrameOperands> operands;
      if (family) {
        operands = std::make_shared<FrameOperands>();
        operands->views = resources;
        operands->shared = family->arrays;
        for (const auto &object : native_resources)
          if (replay_resources.count(object.get()))
            operands->objects.push_back(object);
      }
      if (family && !family->prepared) {
        if (!initializers.empty()) {
          auto initial_operands = std::make_shared<FrameOperands>(*operands);
          initial_operands->objects = native_resources;
          family->initialization_operands = std::move(initial_operands);
          family->initialization = std::make_unique<FixedExecution>(
              runtime.arch, runtime.get(), gfx_runtime, initializers, &sequence,
              runtime.runtime_fault_domain());
        } else {
          family->initialized = true;
        }
        family->initial_facts = sequence;
      } else if (!family && !initializers.empty()) {
        initialization = std::make_unique<FixedExecution>(
            runtime.arch, runtime.get(), gfx_runtime, initializers, &sequence,
            runtime.runtime_fault_domain());
      }
      native = std::make_unique<FixedExecution>(
          runtime.arch, runtime.get(), gfx_runtime, operations, &sequence,
          runtime.runtime_fault_domain(), family ? family->replay : nullptr,
          family ? operands : nullptr, submission_recipe);
      if (family) {
        family->prepared = true;
        family->arrays->sealed = true;
      }
    } catch (const std::invalid_argument &error) {
      throw MemoryBindingError(error.what());
    }
    std::uint64_t dispatch_count = 0;
    std::function<void(const std::vector<FixedExecution::Operation> &)>
        count_dispatches;
    count_dispatches = [&](const std::vector<FixedExecution::Operation>
                               &steps) {
      for (const auto &step : steps) {
        if (std::holds_alternative<FixedExecution::Dispatch>(step))
          ++dispatch_count;
        else if (const auto *branch =
                     std::get_if<std::shared_ptr<const FixedExecution::Branch>>(
                         &step)) {
          count_dispatches((*branch)->condition);
          for (const auto &body : (*branch)->bodies)
            count_dispatches(body);
        } else if (const auto *parallel = std::get_if<
                       std::shared_ptr<const FixedExecution::Parallel>>(
                       &step)) {
          for (const auto &branch : (*parallel)->branches)
            count_dispatches(branch);
        } else if (const auto *loop = std::get_if<
                       std::shared_ptr<const FixedExecution::While>>(&step)) {
          count_dispatches((*loop)->condition);
          count_dispatches((*loop)->body);
        }
      }
    };
    count_dispatches(operations);
    if (dispatch_count > std::numeric_limits<std::uint32_t>::max())
      throw MemoryBindingError(
          "Static dispatch count exceeds its C API representation");
    info.dispatch_count = static_cast<std::uint32_t>(dispatch_count);
    info.execution_kind = native->execution_kind();
    info.argument_bytes = native->argument_bytes() +
                          (initialization ? initialization->argument_bytes()
                           : family && family->initialization
                               ? family->initialization->argument_bytes()
                               : 0);
    refresh_resource_bytes();
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
    if (family && family->failed)
      throw MemoryBindingError(
          "Binding family initialization failed; prepare a new family");
    native->run();
    ++info.launches;
  }

  void invalidate() {
    live = false;
    initialization.reset();
    retire_binding(true);
    native.reset();
    release_native_resources();
    release_source_bindings();
    resources.reset();
    family.reset();
  }

  bool needs_initialization() const {
    return family ? !family->initialized : bool(initialization);
  }

  void finish_initialization() {
    initialization.reset();
    for (std::size_t i = native_resources.size(); i-- > 0;)
      if (!replay_resources.count(native_resources[i].get()))
        native_resources.erase(native_resources.begin() + i);
    refresh_resource_bytes();
    if (resources)
      resources->retain(replay_memories);
    info.argument_bytes = native->argument_bytes();
  }

  void initialize(Runtime &runtime) {
    if (!live)
      throw MemoryBindingError("Prepared execution has a retired dependency");
    if (family) {
      family->initialize();
      return;
    }
    if (!initialization)
      return;
    try {
      initialization->run();
      runtime.wait();
      // Commands must retire before any of their input or scratch storage.
      finish_initialization();
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
      if (entry->family)
        families_[entry->family.get()].insert(handle);
      for (auto memory : memories)
        memories_[memory].insert(handle);
      for (auto resource : entry->external_resources)
        resources_[resource].insert(handle);
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
  uint64_t family_size(const ExecutionBindingFamily *family) const {
    const auto found =
        families_.find(const_cast<ExecutionBindingFamily *>(family));
    return found == families_.end() ? 0 : found->second.size();
  }
  void destroy(TiPreparedExecution handle) {
    auto &entry = get(handle);
    if ((entry.native || entry.initialization) &&
        runtime_.runtime_fault_domain()->backend_calls_safe())
      runtime_.wait();
    entry.retire_binding(true);
    remove_dependencies(handle, entry);
    entries_.erase(handle);
  }
  void initialize(TiPreparedExecution handle) {
    auto &entry = get(handle);
    const bool needed = entry.needs_initialization();
    entry.initialize(runtime_);
    if (!needed)
      return;
    const auto finish = [&](TiPreparedExecution member) {
      auto &call = get(member);
      if (!call.live)
        return;
      if (call.family)
        call.finish_initialization();
      for (auto memory : call.memories)
        if (!call.replay_memories.count(memory))
          remove_edge(memories_, memory, member);
      call.memories.erase(
          std::remove_if(call.memories.begin(), call.memories.end(),
                         [&](TiMemory memory) {
                           return !call.replay_memories.count(memory);
                         }),
          call.memories.end());
      call.replay_memories.clear();
      call.replay_resources.clear();
      for (auto resource : call.external_resources)
        if (!call.replay_external_resources.count(resource))
          remove_edge(resources_, resource, member);
      call.external_resources.erase(
          std::remove_if(call.external_resources.begin(),
                         call.external_resources.end(), [&](auto resource) {
                           return !call.replay_external_resources.count(resource);
                         }),
          call.external_resources.end());
      call.replay_external_resources.clear();
    };
    if (entry.family) {
      auto &family = *entry.family;
      const auto members = families_.at(&family);
      for (auto member : members)
        finish(member);
      for (auto object = family.objects.begin();
           object != family.objects.end();) {
        if (!family.plan.replay_bindings.count(object->first))
          object = family.objects.erase(object);
        else
          ++object;
      }
      std::unordered_set<TiMemory> retained;
      for (auto array = family.arrays->arrays.begin();
           array != family.arrays->arrays.end();) {
        if (!family.plan.replay_bindings.count(array->first))
          array = family.arrays->arrays.erase(array);
        else {
          retained.insert(array->second.memory);
          ++array;
        }
      }
      if (family.arrays->storage)
        family.arrays->storage->retain(retained);
      family.initialization_memories.clear();
      family.initialization_external_resources.clear();
    } else {
      finish(handle);
    }
  }
  void run(TiPreparedExecution handle) {
    auto &entry = get(handle);
    if (entry.needs_initialization())
      initialize(handle);
    entry.run();
  }
  void retire_memory(TiMemory memory) {
    retire(memories_, memory);
  }
  void retire_resource(TiNativeResource resource, bool synchronize) {
    retire(resources_, resource, synchronize);
  }
  void retire_module(AotModule *module) {
    retire(modules_, module);
  }
  void clear(bool synchronize) {
    if (synchronize && !entries_.empty() &&
        runtime_.runtime_fault_domain()->backend_calls_safe())
      runtime_.wait();
    modules_.clear();
    memories_.clear();
    resources_.clear();
    families_.clear();
    for (auto &entry : entries_)
      entry.second->retire_binding(true);
    entries_.clear();
  }

 private:
  template <typename Map, typename Key>
  void retire(Map &map, Key key, bool synchronize = true) {
    auto found = map.find(key);
    if (found == map.end())
      return;
    // Retirement is a cold boundary. Reverse indexes visit only dependent
    // executions and remove all their edges; warm replay never scans owners.
    const std::vector<TiPreparedExecution> handles(found->second.begin(),
                                                   found->second.end());
    if (synchronize && runtime_.runtime_fault_domain()->backend_calls_safe())
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
    if (entry.family)
      remove_edge(families_, entry.family.get(), handle);
    remove_edge(modules_, entry.owner, handle);
    for (auto memory : entry.memories)
      remove_edge(memories_, memory, handle);
    for (auto resource : entry.external_resources)
      remove_edge(resources_, resource, handle);
    entry.owner = nullptr;
    entry.memories.clear();
    entry.external_resources.clear();
  }
  Runtime &runtime_;
  std::unordered_map<TiPreparedExecution, std::unique_ptr<PreparedExecution>>
      entries_;
  std::unordered_map<AotModule *, std::unordered_set<TiPreparedExecution>>
      modules_;
  std::unordered_map<TiMemory, std::unordered_set<TiPreparedExecution>>
      memories_;
  std::unordered_map<TiNativeResource, std::unordered_set<TiPreparedExecution>>
      resources_;
  std::unordered_map<ExecutionBindingFamily *,
                     std::unordered_set<TiPreparedExecution>>
      families_;
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
void Runtime::retire_prepared_native_resource(TiNativeResource resource,
                                             bool synchronize) {
  if (prepared_)
    prepared_->retire_resource(resource, synchronize);
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
  prepared->external_resources = arguments->native_resources;
  prepared->kernel_arguments = std::move(arguments);
  taichi::lang::FixedExecution::Dispatch dispatch{
      prepared->kernel_arguments->kernel,
      prepared->kernel_arguments->context.get()};
  dispatch.resources = prepared->kernel_arguments->kernel_resources;
  prepared->prepare_operations(rt, {std::move(dispatch)});
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

static TiPreparedExecution prepare_execution_plan_impl(
    TiRuntime runtime,
    TiAotModule module,
    uint64_t description_size,
    const char *description,
    uint32_t arg_count,
    const TiNamedArgument *args,
    uint32_t binding_count,
    const TiNamedNdArrayBinding *bindings,
    std::shared_ptr<capi::ExecutionBindingFamily> family) {
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
  const bool rebinding = bool(family);
  aot::ExecutionPlan parsed;
  if (family) {
    if (family->runtime != &rt || family->module != &mod || family->failed)
      throw MemoryBindingError("Binding family owner is invalid or failed");
  } else {
    try {
      parsed =
          aot::ExecutionPlan::parse(std::string(description, description_size));
      if (parsed.cuda_binding_frames)
        family = std::make_shared<capi::ExecutionBindingFamily>(
            rt, mod, std::move(parsed));
    } catch (const std::invalid_argument &error) {
      throw MemoryBindingError(error.what());
    }
  }
  const auto &plan = family ? family->plan : parsed;
  // The compact format is validated independently before replay is enabled.
  // Never interpret its expanded frames as one ordinary execution sequence.
  if (plan.storage_ring)
    throw MemoryBindingError("Ordered storage ring replay is not available in this runtime");
  if (plan.vulkan_submission && rt.arch != taichi::Arch::vulkan)
    throw MemoryBindingError("Selected Vulkan submission requires Vulkan");
  const auto &resource_description =
      rebinding ? family->frame_resources : plan.resources;
  std::unordered_map<std::string, const TiArgument *> arguments;
  for (uint32_t i = 0; i < arg_count; ++i) {
    if (!args[i].name || !*args[i].name ||
        !arguments.emplace(args[i].name, &args[i].argument).second)
      throw MemoryBindingError(
          "Execution plan argument names must be unique and nonempty");
  }
  std::unordered_set<std::string> frozen_metadata;
  if (rebinding) {
    for (const auto &[name, value] : arguments)
      if (!family->frame_arguments.count(name))
        throw MemoryBindingError(
            "Binding family accepts only replay argument names: " + name);
    for (const auto &name : family->frame_arguments) {
      if (arguments.count(name))
        continue;
      if (plan.replay_bindings.count(name) &&
          !family->persistent_public_arguments.count(name))
        throw MemoryBindingError("Missing binding family replay argument: " +
                                 name);
      arguments.emplace(name, &family->original_arguments.at(name));
      if (!plan.replay_bindings.count(name))
        frozen_metadata.insert(name);
    }
  } else {
    if (arguments.size() != plan.arguments.size())
      throw MemoryBindingError(
          "Execution plan argument set does not match its slots");
    for (const auto &name : plan.arguments)
      if (!arguments.count(name))
        throw MemoryBindingError("Missing execution plan argument: " + name);
    if (family)
      for (const auto &[name, value] : arguments)
        family->original_arguments.emplace(name, *value);
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
  prepared->family = family;
  // Optional providers are admitted before allocating private plan resources.
  // There is no dependency probing on replay, nor on unrelated operations.
  auto local_preparation =
      family ? nullptr : capi::make_native_preparation_context(rt);
  auto &preparation = family ? *family->preparation : *local_preparation;
  preparation.fault_domain->throw_if_submission_disallowed("prepare AOT plan");
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
  std::function<void(const std::vector<aot::ExecutionPlan::Operation> &)>
      collect_steps;
  collect_steps = [&](const std::vector<aot::ExecutionPlan::Operation>
                          &sequence) {
    for (const auto &operation : sequence) {
      steps.push_back(&operation);
      if (const auto *value =
              std::get_if<std::shared_ptr<const aot::ExecutionPlan::Branch>>(
                  &operation)) {
        const auto &branch = **value;
        using Lowering = aot::ExecutionPlan::Branch::Lowering;
        const bool matches = branch.lowering == Lowering::cpu
                                 ? arch_is_cpu(rt.arch)
                             : branch.lowering == Lowering::vulkan_host
                                 ? rt.arch == taichi::Arch::vulkan
                                 : rt.arch == taichi::Arch::cuda;
        if (!matches)
          throw MemoryBindingError(
              "AOT branch requires its declared backend lowering");
        collect_steps(branch.condition);
        for (const auto &body : branch.bodies)
          collect_steps(body);
      }
      if (const auto *parallel =
              std::get_if<std::shared_ptr<const aot::ExecutionPlan::Parallel>>(
                  &operation)) {
        if (rt.arch != taichi::Arch::cuda)
          throw MemoryBindingError("AOT CUDA parallel requires a CUDA runtime");
        for (const auto &branch : (*parallel)->branches)
          collect_steps(branch);
      }
      if (const auto *loop =
              std::get_if<std::shared_ptr<const aot::ExecutionPlan::While>>(
                  &operation)) {
        if ((*loop)->cuda_control && rt.arch != taichi::Arch::cuda)
          throw MemoryBindingError(
              "AOT CUDA control requires a CUDA runtime");
        if ((rt.arch == taichi::Arch::vulkan) != bool((*loop)->vulkan))
          throw MemoryBindingError(
              "AOT while requires a matching selected Vulkan recipe on "
              "Vulkan only");
        collect_steps((*loop)->condition);
        collect_steps((*loop)->body);
      }
    }
  };
  if (!rebinding)
    collect_steps(plan.initialization);
  collect_steps(plan.operations);
  for (const auto index : resource_description.order)
    if (const auto *native = std::get_if<NativeOperationDescription>(
            &resource_description.declarations[index].value))
      preflight(native, NativeObjectKind::resource);
  for (const auto *step : steps) {
    const auto &operation = *step;
    if (const auto *native =
            std::get_if<aot::ExecutionPlan::Native>(&operation))
      preflight(native, NativeObjectKind::operation);
  }
  if (rebinding) {
    for (const auto &name : family->persistent_public_arguments) {
      if (!family->frame_arguments.count(name))
        continue;
      const auto &old = family->original_arguments.at(name);
      const auto &next = *arguments.at(name);
      if (old.type == TI_ARGUMENT_TYPE_NATIVE_RESOURCE_EXT &&
          next.type == TI_ARGUMENT_TYPE_NATIVE_RESOURCE_EXT) {
        if (old.value.native_resource != next.value.native_resource)
          throw MemoryBindingError(
              "Shared native resource changed; prepare a new binding family: " +
              name);
        rt.native_resource_registry().get(next.value.native_resource);
        continue;
      }
      if (old.type != TI_ARGUMENT_TYPE_NDARRAY ||
          next.type != TI_ARGUMENT_TYPE_NDARRAY)
        throw MemoryBindingError(
            "Shared provider resources require unchanged array bindings: " +
            name);
      const auto &a = old.value.ndarray;
      const auto &b = next.value.ndarray;
      const auto size_a = capi::validated_ndarray_bytes(rt, a);
      const auto size_b = capi::validated_ndarray_bytes(rt, b);
      const auto memory_a = rt.get_memory(a.memory);
      const auto memory_b = rt.get_memory(b.memory);
      const auto range_a = resolve_native_storage_range(
          rt.arch, memory_a.allocation.get_ptr(memory_a.byte_offset), size_a);
      const auto range_b = resolve_native_storage_range(
          rt.arch, memory_b.allocation.get_ptr(memory_b.byte_offset), size_b);
      if (a.elem_type != b.elem_type ||
          a.shape.dim_count != b.shape.dim_count ||
          a.elem_shape.dim_count != b.elem_shape.dim_count ||
          !std::equal(a.shape.dims, a.shape.dims + a.shape.dim_count,
                      b.shape.dims) ||
          !std::equal(a.elem_shape.dims,
                      a.elem_shape.dims + a.elem_shape.dim_count,
                      b.elem_shape.dims) ||
          !range_a.proven_same_range(range_b))
        throw MemoryBindingError(
            "Shared provider resource input changed; prepare a new binding "
            "family: " +
            name);
    }
  }
  std::map<std::string, std::shared_ptr<NativeResource>> native_objects;
  for (const auto &[name, arg] : arguments) {
    if (arg->type != TI_ARGUMENT_TYPE_NATIVE_RESOURCE_EXT)
      continue;
    auto object = rt.native_resource_registry().get(arg->value.native_resource);
    native_objects.emplace(name, std::move(object));
    prepared->external_resources.push_back(arg->value.native_resource);
    if (family && !rebinding && plan.replay_bindings.count(name))
      family->persistent_public_arguments.insert(name);
  }
  std::map<std::string, const NativeOperationDescription *> allocator_descriptions;
  for (const auto index : resource_description.order) {
    const auto &resource = resource_description.declarations[index];
    if (const auto *native =
            std::get_if<NativeOperationDescription>(&resource.value))
      allocator_descriptions.emplace(resource.name, native);
  }
  auto resolve_allocator = [&](const std::string &name) {
    auto found = native_objects.find(name);
    if (found == native_objects.end()) {
      std::shared_ptr<NativeResource> object;
      if (rebinding) {
        object = family->objects.at(name);
      } else {
        const auto description = allocator_descriptions.find(name);
        if (description == allocator_descriptions.end() ||
            !description->second->bindings.empty())
          throw MemoryBindingError(
              "Array allocator requires an input-free native resource: " + name);
        try {
          object = native_operation_registry().prepare_resource(
              preparation, decoded.at(description->second), {});
        } catch (const std::invalid_argument &error) {
          throw MemoryBindingError(error.what());
        }
      }
      prepared->native_resources.push_back(object);
      found = native_objects.emplace(name, std::move(object)).first;
    }
    auto allocator =
        std::dynamic_pointer_cast<NativeMemoryAllocator>(found->second);
    if (!allocator)
      throw MemoryBindingError("Native resource is not an array allocator: " + name);
    return allocator;
  };
  auto resource_bindings = capi::bind_execution_resources(
      rt, mod.get(), resource_description, arguments, layouts,
      family ? family->arrays.get() : nullptr, frozen_metadata, resolve_allocator);
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
    return Storage{
        allocation.allocation.get_ptr(allocation.byte_offset + offset),
        bytes - offset, element_bytes};
  };
  std::map<std::string, NativeArrayBinding> native_arrays;
  auto array_binding =
      [&](const std::string &slot) -> const NativeArrayBinding & {
    auto found = native_arrays.find(slot);
    if (found == native_arrays.end()) {
      storage(slot, 0);
      const auto &array = arguments.at(slot)->value.ndarray;
      found = native_arrays.emplace(slot, capi::bind_native_array(rt, array)).first;
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
    if (rebinding) {
      bool relevant = true;
      const auto used = [&](const std::string &name) {
        return plan.replay_bindings.count(name) != 0;
      };
      if (const auto *layout =
              std::get_if<aot::ExecutionPlan::ArrayLayout>(&constraint))
        relevant = used(layout->array);
      else if (const auto *pair =
                   std::get_if<aot::ExecutionPlan::Disjoint>(&constraint))
        relevant = used(pair->left) && used(pair->right);
      else if (const auto *pair =
                   std::get_if<aot::ExecutionPlan::ResourceAlias>(&constraint))
        relevant = used(pair->left) && used(pair->right);
      if (!relevant)
        continue;
    }
    try {
      if (const auto *layout =
              std::get_if<aot::ExecutionPlan::ArrayLayout>(&constraint)) {
        validate_native_array_layout(array_binding(layout->array),
                                     layout->required);
      } else if (std::holds_alternative<aot::ExecutionPlan::Disjoint>(
                     constraint)) {
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
      } else if (const auto *pair =
                     std::get_if<aot::ExecutionPlan::ResourceAlias>(&constraint)) {
        const auto resolve = [&](const std::string &name,
                                 const std::vector<std::string> &path) {
          const auto found = arguments.find(name);
          if (found == arguments.end() ||
              found->second->type != TI_ARGUMENT_TYPE_NATIVE_RESOURCE_EXT)
            throw MemoryBindingError("Resource alias requires public native resources: " + name);
          return rt.native_resource_registry().resolve(
              found->second->value.native_resource, path);
        };
        if (resolve(pair->left, pair->left_path) !=
            resolve(pair->right, pair->right_path))
          throw MemoryBindingError("Native resource alias mismatch: " + pair->left +
                                   " and " + pair->right);
      } else {
        const auto &group =
            std::get<aot::ExecutionPlan::DisjointWrites>(constraint);
        std::vector<NativeStorageRange> reads;
        std::vector<NativeStorageWrite> writes;
        std::vector<std::optional<std::size_t>> read_indices(
            group.reads.size());
        for (std::size_t i = 0; i < group.reads.size(); ++i) {
          if (rebinding && !plan.replay_bindings.count(group.reads[i]))
            continue;
          read_indices[i] = reads.size();
          reads.push_back(storage_range(group.reads[i]));
        }
        for (const auto &write : group.writes) {
          if (rebinding && !plan.replay_bindings.count(write.array))
            continue;
          auto inplace = write.inplace_read;
          if (inplace && !read_indices[*inplace])
            inplace.reset();
          if (inplace) {
            const auto &a = array_binding(write.array);
            const auto &b = array_binding(group.reads[*inplace]);
            if (a.scalar_type != b.scalar_type || a.shape != b.shape ||
                a.element_shape != b.element_shape)
              inplace.reset();
            else
              inplace = read_indices[*inplace];
          }
          writes.push_back({storage_range(write.array), inplace});
        }
        validate_native_disjoint_writes(reads, writes);
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
  std::map<std::string, std::shared_ptr<const NativeHostBlobResource::Bytes>>
      host_blobs;
  for (const auto index : resource_description.order) {
    const auto &resource = resource_description.declarations[index];
    // Allocator factories were resolved after metadata validation, before
    // arrays. Reuse their owner instead of creating a second private pool.
    if (native_objects.count(resource.name))
      continue;
    if (rebinding &&
        (std::holds_alternative<NativeOperationDescription>(resource.value) ||
         std::holds_alternative<aot::ExecutionResources::HostBlob>(
             resource.value))) {
      auto object = family->objects.at(resource.name);
      prepared->native_resources.push_back(object);
      native_objects.emplace(resource.name, std::move(object));
      continue;
    }
    if (const auto *blob =
            std::get_if<aot::ExecutionResources::HostBlob>(&resource.value)) {
      auto found = host_blobs.find(blob->data);
      if (found == host_blobs.end()) {
        auto bytes = std::make_shared<const NativeHostBlobResource::Bytes>(
            mod.get().load_data(blob->data, blob->byte_size));
        found = host_blobs.emplace(blob->data, std::move(bytes)).first;
      } else if (found->second->size() != blob->byte_size) {
        throw MemoryBindingError(
            "Execution host blob size differs between declarations");
      }
      auto object = std::make_shared<NativeHostBlobResource>(rt.arch, rt.get(),
                                                             found->second);
      prepared->native_resources.push_back(object);
      native_objects.emplace(resource.name, std::move(object));
      continue;
    }
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
      operations.size() + (rebinding ? 0 : plan.initialization.size());
  const auto control_pointer = [&](const std::string &name) {
    const auto &array = array_binding(name);
    if (array.scalar_type != PrimitiveType::i32 || array.bytes != 4 ||
        !array.element_shape.empty() ||
        !std::all_of(array.shape.begin(), array.shape.end(),
                     [](std::uint64_t extent) { return extent == 1; }) ||
        array.base_offset % 4)
      throw MemoryBindingError(
          "Control requires one aligned scalar i32 element: " + name);
    return array.pointer;
  };
  std::function<bool(const std::vector<aot::ExecutionPlan::Operation> &,
                     std::vector<FixedExecution::Operation> &)>
      bind_steps;
  bool preparing_initialization = true;
  bind_steps = [&](const std::vector<aot::ExecutionPlan::Operation> &sequence,
                   std::vector<FixedExecution::Operation> &operations) {
    // One map per ordered region and prepared frame, never in the potentially
    // shared binding-family context. Recursive control/parallel regions have
    // independent storage. Initializers retain their original private owners.
    std::map<SortWorkspaceDescription::Signature,
             std::shared_ptr<NativeSortWorkspace>> sort_workspaces;
    for (const auto &operation : sequence) {
      if (const auto *value =
              std::get_if<std::shared_ptr<const aot::ExecutionPlan::Branch>>(
                  &operation)) {
        const auto &branch = **value;
        auto bound = std::make_shared<FixedExecution::Branch>();
        bound->selector = control_pointer(branch.selector);
        bound->selection = branch.selection;
        using Lowering = aot::ExecutionPlan::Branch::Lowering;
        if (branch.lowering == Lowering::cuda_masked)
          bound->mode = BranchExecutionMode::cuda_masked;
        else if (branch.lowering == Lowering::cuda_host ||
                 branch.lowering == Lowering::vulkan_host)
          bound->mode = BranchExecutionMode::host;
        if (!bind_steps(branch.condition, bound->condition))
          return false;
        for (const auto &body : branch.bodies) {
          bound->bodies.emplace_back();
          if (!bind_steps(body, bound->bodies.back()))
            return false;
        }
        operations.emplace_back(std::move(bound));
        continue;
      } else if (const auto *parallel = std::get_if<
                     std::shared_ptr<const aot::ExecutionPlan::Parallel>>(
                     &operation)) {
        auto bound = std::make_shared<FixedExecution::Parallel>();
        for (const auto &branch : (*parallel)->branches) {
          bound->branches.emplace_back();
          if (!bind_steps(branch, bound->branches.back()))
            return false;
        }
        operations.emplace_back(std::move(bound));
        continue;
      } else if (const auto *value = std::get_if<
                     std::shared_ptr<const aot::ExecutionPlan::While>>(
                     &operation)) {
        const auto &loop = **value;
        auto bound = std::make_shared<FixedExecution::While>();
        bound->predicate = control_pointer(loop.predicate);
        bound->max_iterations = loop.max_iterations;
        bound->continue_while_nonzero = loop.continue_while_nonzero;
        if (loop.cuda_control) {
          bound->cuda_control.emplace();
          bound->cuda_control->encoding = loop.cuda_control->encoding;
          for (const auto &source : loop.cuda_control->inners)
            bound->cuda_control->inners.push_back(
                {control_pointer(source.predicate), source.condition_begin,
                 source.body_begin, source.end, source.max_iterations});
        }
        if (loop.vulkan) {
          bound->vulkan.emplace();
          bound->vulkan->recipe = loop.vulkan->recipe;
          bound->vulkan->counter = control_pointer(loop.vulkan->counter);
          std::vector<std::string> controls{loop.predicate,
                                            loop.vulkan->counter};
          if (loop.vulkan->status) {
            controls.push_back(*loop.vulkan->status);
            bound->vulkan->status = control_pointer(*loop.vulkan->status);
          }
          if (loop.vulkan->nested) {
            bound->vulkan->nested.emplace();
            for (const auto &source : loop.vulkan->nested->inners) {
              VulkanNestedControlRecipe<DevicePtr>::Inner inner;
              inner.predicate = control_pointer(source.predicate);
              inner.counter = control_pointer(source.counter);
              controls.push_back(source.predicate);
              controls.push_back(source.counter);
              if (source.status) {
                inner.status = control_pointer(*source.status);
                controls.push_back(*source.status);
              }
              inner.condition_begin = source.condition_begin;
              inner.body_begin = source.body_begin;
              inner.end = source.end;
              inner.max_iterations = source.max_iterations;
              inner.chunk_size = source.chunk_size;
              bound->vulkan->nested->inners.push_back(std::move(inner));
            }
          }
          std::vector<const NativeStorageRange *> ranges;
          ranges.reserve(controls.size());
          for (const auto &name : controls)
            ranges.push_back(&storage_range(name));
          for (std::size_t i = 0; i < controls.size(); ++i)
            for (std::size_t j = 0; j < i; ++j)
              if (!ranges[i]->proven_disjoint(*ranges[j]))
                throw MemoryBindingError(
                    "Vulkan loop controls must be disjoint");
        }
        if (!bind_steps(loop.condition, bound->condition) ||
            !bind_steps(loop.body, bound->body))
          return false;
        operations.emplace_back(std::move(bound));
        continue;
      }
      if (const auto *kernel =
              std::get_if<aot::ExecutionPlan::Kernel>(&operation)) {
        auto handle = mod.get_kernel_handle(kernel->entry);
        if (!handle)
          throw MemoryBindingError("Execution plan kernel was not exported: " +
                                   kernel->entry);
        std::vector<TiArgument> kernel_args;
        std::vector<TiNdArrayBinding> kernel_bindings;
        std::map<uint32_t, std::shared_ptr<NativeResource>> kernel_resources;
        const auto &parameters =
            rt.get_kernel(handle).kernel->nested_parameters;
        for (std::size_t i = 0; i < kernel->arguments.size(); ++i) {
          const auto &name = kernel->arguments[i];
          if (const auto object = native_objects.find(name);
              object != native_objects.end()) {
            kernel_resources.emplace(static_cast<uint32_t>(i), object->second);
            // The private owner map supplies this slot. Never synthesize or
            // reinterpret a public native-resource handle.
            kernel_args.emplace_back();
            continue;
          }
          kernel_args.push_back(*arguments.at(name));
          if (auto binding = layouts.find(name); binding != layouts.end()) {
            const auto &value = *binding->second;
            const auto parameter = parameters.find({static_cast<int>(i)});
            // A plan-wide primal may also be read by ordinary helper kernels.
            // Supply its gradient only where the compiled dispatch ABI uses it.
            // Keep malformed/structure bindings for the common validator, and
            // reject gradients unused by every dispatch at the end of prepare.
            if (value.gradient && !value.layout && value.layout_version == 0 &&
                (parameter == parameters.end() ||
                 !parameter->second.needs_grad))
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
            kernel_bindings.data(),
            kernel->bounded_range && !kernel->bounded_range->cuda,
            kernel_resources);
        if (!bound)
          return false;
        bound->context->append_dispatch_label(kernel->label);
        DevicePtr indirect_packet{kDeviceNullPtr};
        DevicePtr bounded_extent{kDeviceNullPtr};
        std::uint32_t bounded_capacity = 0;
        std::uint32_t adaptive_block_dim = 0;
        bool grouped_update = false;
        const void *bounded_binding_identity = nullptr;
        if (kernel->bounded_range) {
          const auto &range = *kernel->bounded_range;
          if ((range.cuda && rt.arch != taichi::Arch::cuda) ||
              (!range.cuda && !arch_is_cpu(rt.arch)))
            throw MemoryBindingError("Exact bounded launch backend mismatch");
          const auto &extent = array_binding(range.extent);
          validate_native_array_layout(extent, {{2}, {}, 4, 4, true, "i32"});
          if (extent.scalar_type != PrimitiveType::i32 || extent.bytes != 8)
            throw MemoryBindingError(
                "Exact bounded extent requires a packed i32[2]");
          bounded_extent = extent.pointer;
          bounded_capacity = range.capacity;
          adaptive_block_dim = range.block_dim;
          grouped_update = range.grouped_update;
          // std::map references are stable through cold preparation. Equal
          // storage under two symbolic names remains independently bindable.
          bounded_binding_identity = &extent;
          const auto storage = resolve_native_storage_range(
              rt.arch, extent.pointer, extent.bytes);
          if (storage.memory_offset % 4)
            throw MemoryBindingError("Exact bounded extent must be i32 aligned");
          if (!range.cuda) {
            bound->context->set_cpu_bounded_range(
                reinterpret_cast<void *>(
                    static_cast<std::uintptr_t>(storage.memory_offset)),
                range.capacity);
          }
        }
        if (kernel->indirect_packet) {
          if (rt.arch != taichi::Arch::vulkan)
            throw MemoryBindingError("Vulkan indirect launch requires Vulkan");
          const auto &packet = array_binding(*kernel->indirect_packet);
          if (packet.scalar_type != PrimitiveType::u32 || !packet.element_shape.empty() ||
              packet.shape.size() != 1 || packet.shape[0] < 3 ||
              packet.bytes < 3 * sizeof(std::uint32_t) || packet.pointer.offset % 4)
            throw MemoryBindingError(
                "Indirect launch packet requires at least three packed u32 values");
          indirect_packet = packet.pointer;
        }
        prepared->memories.insert(prepared->memories.end(),
                                  bound->memories.begin(),
                                  bound->memories.end());
        operations.emplace_back(
            FixedExecution::Dispatch{bound->kernel, bound->context.get(),
                                     indirect_packet, bounded_extent,
                                     bounded_capacity, adaptive_block_dim,
                                     grouped_update, bounded_binding_identity,
                                     bound->kernel_resources});
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
            throw MemoryBindingError(
                "Sort keys require i32/u32/f32/i64/u64/f64");
        }
        if (rt.get_memory(array.memory).base_offset % keys.element_bytes)
          throw MemoryBindingError("Sort key storage is misaligned");
        SortOperation descriptor{
            keys.pointer, {}, array.shape.dims[0], key_type};
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
          if (!preparing_initialization) {
            const auto description = SortWorkspaceDescription::from(descriptor);
            auto &workspace = sort_workspaces[description.signature()];
            if (!workspace) {
              workspace = prepare_sort_workspace(rt.arch, rt.get(), description);
              prepared->native_resources.push_back(workspace);
              prepared->replay_resources.insert(workspace.get());
            }
            descriptor.workspace = workspace;
          }
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
        CompactOperation descriptor{{},
                                    flag_storage.pointer,
                                    count_storage.pointer,
                                    flags.shape.dims[0]};
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
        if (bytes > dst.size ||
            bytes > std::numeric_limits<std::size_t>::max() ||
            rt.get_memory(array.memory).base_offset % 4)
          throw MemoryBindingError(
              "Execution fill exceeds ndarray extent or alignment");
        operations.emplace_back(FixedExecution::BufferFill{
            dst.pointer, static_cast<std::size_t>(bytes), fill.value});
      }
    }
    return true;
  };
  if (!rebinding && !bind_steps(plan.initialization, operations))
    return TI_NULL_HANDLE;
  if (family && !rebinding) {
    family->objects = native_objects;
    std::vector<TiMemory> dependencies;
    for (const auto &name : plan.initialization_bindings) {
      const auto found = arguments.find(name);
      if (found != arguments.end() &&
          found->second->type == TI_ARGUMENT_TYPE_NATIVE_RESOURCE_EXT)
        family->initialization_external_resources.insert(
            found->second->value.native_resource);
      if (found != arguments.end() &&
          found->second->type == TI_ARGUMENT_TYPE_NDARRAY)
        dependencies.push_back(found->second->value.ndarray.memory);
      if (const auto binding = layouts.find(name);
          binding != layouts.end() && binding->second->gradient)
        dependencies.push_back(binding->second->gradient->memory);
    }
    if (prepared->resources)
      prepared->resources->remove_owned_dependencies(dependencies);
    if (family->arrays->storage)
      family->arrays->storage->remove_owned_dependencies(dependencies);
    family->initialization_memories.insert(dependencies.begin(),
                                           dependencies.end());
    for (const auto &name : family->persistent_public_arguments) {
      if (!plan.replay_bindings.count(name))
        continue;
      const auto &argument = *arguments.at(name);
      if (argument.type == TI_ARGUMENT_TYPE_NDARRAY)
        family->persistent_memories.insert(argument.value.ndarray.memory);
      else if (argument.type == TI_ARGUMENT_TYPE_NATIVE_RESOURCE_EXT)
        family->persistent_external_resources.insert(
            argument.value.native_resource);
    }
  }
  preparing_initialization = false;
  if (!bind_steps(plan.operations, operations))
    return TI_NULL_HANDLE;
  if (used_bindings.size() != layouts.size())
    throw MemoryBindingError(
        "Ndarray layout/gradient binding is unused by any kernel");
  {
    for (const auto &name : plan.replay_bindings) {
      if (const auto arg = arguments.find(name); arg != arguments.end() &&
          arg->second->type == TI_ARGUMENT_TYPE_NATIVE_RESOURCE_EXT)
        prepared->replay_external_resources.insert(
            arg->second->value.native_resource);
      if (const auto object = native_objects.find(name);
          object != native_objects.end()) {
        prepared->replay_resources.insert(object->second.get());
        continue;
      }
      const auto &argument = *arguments.at(name);
      if (argument.type == TI_ARGUMENT_TYPE_NDARRAY) {
        prepared->replay_memories.insert(argument.value.ndarray.memory);
        // A view's public storage ancestor may not appear as a direct kernel
        // argument. Retiring that ancestor must still invalidate this plan.
        prepared->memories.push_back(argument.value.ndarray.memory);
      }
      if (const auto binding = layouts.find(name);
          binding != layouts.end() && binding->second->gradient)
        prepared->replay_memories.insert(binding->second->gradient->memory);
    }
  }
  if (family) {
    if (!family->initialized)
      prepared->external_resources.insert(
          prepared->external_resources.end(),
          family->initialization_external_resources.begin(),
          family->initialization_external_resources.end());
    prepared->external_resources.insert(
        prepared->external_resources.end(),
        family->persistent_external_resources.begin(),
        family->persistent_external_resources.end());
    prepared->replay_external_resources.insert(
        family->persistent_external_resources.begin(),
        family->persistent_external_resources.end());
    if (!family->initialized)
      prepared->memories.insert(prepared->memories.end(),
                                family->initialization_memories.begin(),
                                family->initialization_memories.end());
    prepared->memories.insert(prepared->memories.end(),
                              family->persistent_memories.begin(),
                              family->persistent_memories.end());
    prepared->replay_memories.insert(family->persistent_memories.begin(),
                                     family->persistent_memories.end());
    if (family->arrays->storage)
      family->arrays->storage->remove_owned_dependencies(prepared->memories);
  }
  if (prepared->resources)
    prepared->resources->remove_owned_dependencies(prepared->memories);
  const auto split = operations.begin() + initialization_count;
  const std::vector<FixedExecution::Operation> initializers(operations.begin(),
                                                            split);
  operations.erase(operations.begin(), split);
  if (plan.vulkan_submission && operations.size() != plan.operations.size())
    throw MemoryBindingError("Selected Vulkan submission changed replay boundaries");
  prepared->prepare_operations(rt, operations, initializers,
                              plan.vulkan_submission ? &*plan.vulkan_submission : nullptr);
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
  return prepare_execution_plan_impl(runtime, module, description_size,
                                     description, arg_count, args,
                                     binding_count, bindings, {});
}

TiPreparedExecution ti_prepare_execution_binding_ext(
    TiRuntime runtime,
    TiPreparedExecution source,
    uint32_t arg_count,
    const TiNamedArgument *args,
    uint32_t binding_count,
    const TiNamedNdArrayBinding *bindings) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(source);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  const auto &entry = rt.prepared_registry().get(source);
  if (!entry.live || !entry.family)
    throw MemoryBindingError(
        "Preparing a binding requires a live CUDA binding family");
  return prepare_execution_plan_impl(
      runtime, reinterpret_cast<TiAotModule>(entry.owner), 0, "", arg_count,
      args, binding_count, bindings, entry.family);
  TI_CAPI_TRY_CATCH_END();
  return TI_NULL_HANDLE;
}

void ti_get_prepared_binding_family_info_ext(
    TiRuntime runtime,
    TiPreparedExecution execution,
    TiPreparedBindingFamilyInfo *info) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(info);
  *info = {};
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(execution);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  const auto &entry = rt.prepared_registry().get(execution);
  if (!entry.live)
    throw MemoryBindingError("Prepared execution has a retired dependency");
  if (entry.family) {
    info->identity = entry.family->identity;
    info->bindings = rt.prepared_registry().family_size(entry.family.get());
    info->resource_bytes = entry.family->arrays->storage
                               ? entry.family->arrays->storage->bytes()
                               : 0;
#ifdef TI_WITH_CUDA
    info->executables = entry.family->replay->snapshot().executables;
#endif
  }
  TI_CAPI_TRY_CATCH_END();
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

void Runtime::configure_native_library(std::string name, const char *path) {
  if (name.empty() || name.find('\0') != std::string::npos)
    throw MemoryBindingError(
        "Native library name must be nonempty without NUL");
  try {
    if (path)
      taichi::lang::validate_native_library_location(name, path);
  } catch (const std::invalid_argument &error) {
    throw MemoryBindingError(error.what());
  }
  if (path)
    native_library_locations_.insert_or_assign(std::move(name), path);
  else
    native_library_locations_.erase(name);
}

void ti_configure_native_library_ext(TiRuntime runtime,
                                     const char *name,
                                     const char *path) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(name);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  rt.configure_native_library(name, path);
  TI_CAPI_TRY_CATCH_END();
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

namespace {
TiFlags native_object_capabilities(TiRuntime runtime,
                                   const char *kind,
                                   uint32_t version,
                                   taichi::lang::NativeObjectKind object_kind) {
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
  const auto *capability = taichi::lang::native_operation_registry().find(
      kind, version, rt.arch, object_kind);
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
  static_assert(
      static_cast<TiFlags>(taichi::lang::NativeOperationMode::cuda_ordered) ==
      TI_NATIVE_OPERATION_CAPABILITY_CUDA_ORDERED_BIT_EXT);
  return capability ? static_cast<TiFlags>(capability->mode) : 0;
  TI_CAPI_TRY_CATCH_END();
  return 0;
}

void probe_native_object(TiRuntime runtime,
                         const char *kind,
                         uint32_t version,
                         TiNativeOperationAvailabilityExt *availability,
                         taichi::lang::NativeObjectKind object_kind) {
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
  const auto *capability = registry.find(kind, version, rt.arch, object_kind);
  if (capability) {
    taichi::lang::NativeProviderSession providers(
        rt.native_provider_locations(), rt.native_library_locations());
    const auto status =
        registry.probe(kind, version, rt.arch, providers, object_kind);
    availability->capabilities = static_cast<TiFlags>(capability->mode);
    availability->available = status.available ? TI_TRUE : TI_FALSE;
    availability->library_version = status.library_version;
  }
  TI_CAPI_TRY_CATCH_END();
}
}  // namespace

TiFlags ti_get_native_operation_capabilities_ext(TiRuntime runtime,
                                                 const char *kind,
                                                 uint32_t version) {
  return native_object_capabilities(runtime, kind, version,
                                    taichi::lang::NativeObjectKind::operation);
}

TiFlags ti_get_native_resource_capabilities_ext(TiRuntime runtime,
                                                const char *kind,
                                                uint32_t version) {
  return native_object_capabilities(runtime, kind, version,
                                    taichi::lang::NativeObjectKind::resource);
}

void ti_probe_native_operation_ext(
    TiRuntime runtime,
    const char *kind,
    uint32_t version,
    TiNativeOperationAvailabilityExt *availability) {
  probe_native_object(runtime, kind, version, availability,
                      taichi::lang::NativeObjectKind::operation);
}

void ti_probe_native_resource_ext(
    TiRuntime runtime,
    const char *kind,
    uint32_t version,
    TiNativeOperationAvailabilityExt *availability) {
  probe_native_object(runtime, kind, version, availability,
                      taichi::lang::NativeObjectKind::resource);
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
  return rt.prepared_registry().get(execution).workspace_bytes();
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
  return rt.prepared_registry().get(execution).array_bytes();
  TI_CAPI_TRY_CATCH_END();
  return 0;
}

void ti_get_prepared_execution_memory_info_ext(
    TiRuntime runtime,
    TiPreparedExecution execution,
    TiPreparedExecutionMemoryInfo *info) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(execution);
  TI_CAPI_ARGUMENT_NULL(info);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  const auto &prepared = rt.prepared_registry().get(execution);
  *info = {prepared.array_bytes(), prepared.native_resource_payload_bytes,
           prepared.workspace_bytes()};
  TI_CAPI_TRY_CATCH_END();
}
