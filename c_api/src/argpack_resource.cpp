#include "argpack_resource.h"
#include "argument_binding.h"
#include "native_resource_registry.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/runtime/native_kernel_resource.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace capi {
ExpandedArgPackArguments expand_argpack_arguments(
    Runtime &runtime, const Runtime::KernelBinding &kernel,
    uint32_t argument_count, const TiArgument *arguments,
    uint32_t binding_count, const TiNdArrayBinding *bindings,
    const std::map<uint32_t, std::shared_ptr<taichi::lang::NativeResource>>
        &resources) {
  ExpandedArgPackArguments result;
  if (!kernel.kernel->argument_origins)
    return result;
  if (binding_count && !bindings)
    throw MemoryBindingError("Missing ndarray binding descriptors");
  std::map<uint32_t, uint32_t> roots;
  std::map<std::vector<int>, uint32_t> resource_slots;
  std::map<uint32_t, std::shared_ptr<ArgPackResource>> packs;
  for (const auto &origin : *kernel.kernel->argument_origins) {
    if (origin.logical.size() != 1) {
      if (origin.kind != "argpack" && origin.kind != "value")
        resource_slots.emplace(origin.logical, uint32_t(origin.physical.at(0)));
      continue;
    }
    const auto index = uint32_t(origin.logical.front());
    roots.emplace(index, uint32_t(origin.physical.front()));
    if (origin.kind != "argpack" || index >= argument_count)
      continue;
    std::shared_ptr<taichi::lang::NativeResource> resource;
    if (const auto found = resources.find(index); found != resources.end())
      resource = found->second;
    else if (arguments[index].type == TI_ARGUMENT_TYPE_NATIVE_RESOURCE_EXT)
      resource = runtime.native_resource_registry().get(
          arguments[index].value.native_resource);
    auto pack = std::dynamic_pointer_cast<ArgPackResource>(resource);
    if (pack)
      packs.emplace(index, std::move(pack));
  }
  if (packs.empty())
    return result;  // Existing physical callers keep their explicit ABI.
  if (argument_count != roots.size())
    throw MemoryBindingError("ArgPack calls require logical root arguments");
  result.arguments.resize(kernel.kernel->args_type->elements().size());
  std::vector<bool> assigned(result.arguments.size(), false);
  const auto assign = [&](uint32_t index, const TiArgument &value) {
    if (index >= assigned.size() || assigned[index])
      throw MemoryBindingError("ArgPack resource mapping is not unique");
    result.arguments[index] = value;
    assigned[index] = true;
  };
  for (uint32_t index = 0; index < argument_count; ++index) {
    const auto found = roots.find(index);
    if (found == roots.end())
      throw MemoryBindingError("Missing logical root argument origin");
    assign(found->second, arguments[index]);
    if (const auto resource = resources.find(index); resource != resources.end())
      result.resources.emplace(found->second, resource->second);
  }
  for (const auto &[index, pack] : packs) {
    if (!pack->storage->layout().matches(*kernel.kernel,
                                         {int(roots.at(index))}))
      throw MemoryBindingError("ArgPack does not match this kernel argument");
    std::map<uint32_t, uint32_t> relocated;
    for (const auto &origin : pack->storage->layout().resources()) {
      auto logical = origin.logical;
      logical.front() = int(index);
      relocated.emplace(uint32_t(origin.physical.at(0)),
                         resource_slots.at(logical));
    }
    for (const auto &member : pack->members) {
      const auto physical = relocated.at(member.physical);
      assign(physical, member.argument);
      TiNdArrayBinding binding{};
      binding.argument_index = physical;
      if (member.structure_type) {
        const auto layout = kernel.layouts.find(physical);
        if (layout == kernel.layouts.end() ||
            runtime.get_layout(layout->second).type != *member.structure_type)
          throw MemoryBindingError("ArgPack structure member layout changed");
        // Handles belong to the consumer module, not the factory's module.
        binding.layout = layout->second;
        binding.layout_version = taichi::lang::aot::StructLayout::kVersion;
      }
      if (member.gradient)
        binding.gradient = &*member.gradient;
      if (binding.layout || binding.gradient)
        result.bindings.push_back(binding);
    }
  }
  if (std::find(assigned.begin(), assigned.end(), false) != assigned.end())
    throw MemoryBindingError("ArgPack call is missing a resource member");
  for (uint32_t i = 0; i < binding_count; ++i) {
    auto binding = bindings[i];
    const auto root = roots.find(binding.argument_index);
    if (root == roots.end())
      throw MemoryBindingError("Ndarray binding is not a logical root argument");
    binding.argument_index = root->second;
    result.bindings.push_back(binding);
  }
  result.expanded = true;
  return result;
}
}  // namespace capi

TiNativeResource ti_create_kernel_argpack_ext(
    TiRuntime runtime, TiKernel kernel, uint32_t logical_argument,
    uint32_t member_count, const TiArgPackMember *members) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  if (member_count && !members)
    throw MemoryBindingError("Missing ArgPack member descriptors");
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  rt.runtime_fault_domain()->throw_if_submission_disallowed("create AOT ArgPack");
  using namespace taichi::lang;
  const auto layout = rt.get_kernel_argpack_layout(kernel, logical_argument);
  const auto &compiled = rt.get_kernel(kernel);
  std::map<std::vector<int>, const TiArgPackMember *> inputs;
  for (uint32_t i = 0; i < member_count; ++i) {
    const auto &member = members[i];
    if (!member.path_count || !member.path)
      throw MemoryBindingError("ArgPack members require nonempty relative paths");
    std::vector<int> path{int(logical_argument)};
    for (uint32_t j = 0; j < member.path_count; ++j) {
      if (member.path[j] > std::numeric_limits<int>::max())
        throw MemoryBindingError("ArgPack member index exceeds the compiled ABI");
      path.push_back(int(member.path[j]));
    }
    if (!inputs.emplace(std::move(path), &member).second)
      throw MemoryBindingError("Duplicate ArgPack member path");
  }
  const auto take = [&](const std::vector<int> &path) -> const TiArgPackMember & {
    const auto found = inputs.find(path);
    if (found == inputs.end())
      throw MemoryBindingError("Missing ArgPack value or resource member");
    const auto *result = found->second;
    inputs.erase(found);
    return *result;
  };
  NativeArgPackStorage::Values values;
  for (const auto &pack : layout->storages()) {
    for (const auto &value : pack.values) {
      const auto &input = take(value.origin.logical);
      const auto &payload = input.argument.value.value;
      if (input.argument.type != TI_ARGUMENT_TYPE_VALUE_EXT || input.layout ||
          input.layout_version || input.gradient ||
          payload.size != value.layout.buffer_size ||
          (payload.size && !payload.data))
        throw MemoryBindingError("ArgPack value must match its compiled payload");
      auto &bytes = values[value.origin.logical];
      bytes.resize(value.layout.buffer_size);
      if (!bytes.empty())
        std::memcpy(bytes.data(), payload.data, bytes.size());
    }
  }
  std::vector<capi::ArgPackResource::Member> resources;
  std::vector<TiMemory> memories;
  std::vector<TiNativeResource> parents;
  std::unique_ptr<LaunchContextBuilder> qualification;
  std::vector<std::shared_ptr<NativeKernelResourceBinding>> native_bindings;
  for (const auto &origin : layout->resources()) {
    const auto &input = take(origin.logical);
    const auto index = uint32_t(origin.physical.at(0));
    capi::ArgPackResource::Member member{index, input.argument, {}, {}};
    if (input.argument.type == TI_ARGUMENT_TYPE_NDARRAY) {
      TiNdArrayBinding binding{index, input.layout_version, input.layout,
                               input.gradient};
      capi::validate_kernel_ndarray_binding(
          rt, compiled, index, input.argument,
          (input.layout || input.layout_version || input.gradient) ? &binding
                                                                 : nullptr);
      memories.push_back(input.argument.value.ndarray.memory);
      if (input.layout)
        member.structure_type = rt.get_layout(input.layout).type;
      if (input.gradient) {
        member.gradient = *input.gradient;
        memories.push_back(input.gradient->memory);
      }
    } else if (input.argument.type == TI_ARGUMENT_TYPE_NATIVE_RESOURCE_EXT) {
      if (input.layout || input.layout_version || input.gradient)
        throw MemoryBindingError("Native ArgPack members do not accept array bindings");
      auto resource = rt.native_resource_registry().get(
          input.argument.value.native_resource);
      if (!qualification)
        qualification = std::make_unique<LaunchContextBuilder>(compiled.kernel);
      try {
        auto binding = prepare_native_kernel_resource(
            std::move(resource), {int(index)}, rt.arch, rt.get(),
            *compiled.kernel, *qualification);
        binding->validate(rt.arch, rt.get(), *compiled.kernel, *qualification, {});
        native_bindings.push_back(std::move(binding));
      } catch (const std::invalid_argument &error) {
        throw MemoryBindingError(error.what());
      }
      parents.push_back(input.argument.value.native_resource);
    } else {
      throw MemoryBindingError(
          "ArgPack resources require ndarrays or managed native resources");
    }
    resources.push_back(std::move(member));
  }
  if (!inputs.empty())
    throw MemoryBindingError("Unknown ArgPack member or explicit nested-pack node");
  // All logical members and dependencies are qualified before device allocation.
  auto storage = std::make_shared<NativeArgPackStorage>(rt.arch, rt.get(),
                                                       layout, values);
  return rt.native_resource_registry().publish(
      std::make_shared<capi::ArgPackResource>(std::move(storage),
                                             std::move(resources)),
      std::move(memories), std::move(parents));
  TI_CAPI_TRY_CATCH_END();
  return TI_NULL_HANDLE;
}
