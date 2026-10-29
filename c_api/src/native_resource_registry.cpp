#include "native_resource_registry.h"
#include "argument_binding.h"

#include <algorithm>
#include <limits>
#include <cstring>
#include "taichi/program/runtime_fault.h"
#include "taichi/runtime/native_host_blob.h"
#ifdef TI_WITH_LLVM
#include "taichi_llvm_impl.h"
#endif
#ifdef TI_WITH_VULKAN
#include "taichi_gfx_impl.h"
#endif

namespace capi {
using namespace taichi::lang;

std::unique_ptr<NativePreparationContext> make_native_preparation_context(
    Runtime &runtime) {
  auto context = std::make_unique<NativePreparationContext>(
      runtime.arch, runtime.get(), runtime.native_provider_locations(),
      runtime.runtime_fault_domain(), runtime.native_library_locations());
#ifdef TI_WITH_LLVM
  if (auto *llvm = dynamic_cast<LlvmRuntime *>(&runtime))
    llvm->prepare_cpu_scheduler(*context);
#endif
#ifdef TI_WITH_VULKAN
  if (auto *gfx = dynamic_cast<::GfxRuntime *>(&runtime))
    context->gfx_runtime = &gfx->get_gfx_runtime();
#endif
  return context;
}

TiNativeResource NativeResourceRegistry::publish(
    std::shared_ptr<NativeResource> object,
    std::vector<TiMemory> memories,
    std::vector<TiNativeResource> parents,
    ParentLinks parent_links,
    bool retire_instances_only) {
  if (!object || object->arch() != runtime_.arch ||
      &object->device() != &runtime_.get())
    throw MemoryBindingError("Native resource belongs to another device");
  const auto deduplicate = [](auto &handles) {
    std::sort(handles.begin(), handles.end(), std::less<>{});
    handles.erase(std::unique(handles.begin(), handles.end()), handles.end());
  };
  deduplicate(memories);
  const auto by_port = [](const auto &a, const auto &b) {
    return a.first < b.first;
  };
  if (!std::is_sorted(parent_links.begin(), parent_links.end(), by_port))
    std::sort(parent_links.begin(), parent_links.end(), by_port);
  for (const auto &[port, parent] : parent_links)
    parents.push_back(parent);
  deduplicate(parents);
  for (auto memory : memories)
    runtime_.get_memory(memory);
  for (auto parent : parents)
    get(parent);
  const auto handle =
      reinterpret_cast<TiNativeResource>(Runtime::new_handle_token());
  const auto it =
      entries_
          .emplace(handle, Entry{std::move(object), std::move(memories),
                                 std::move(parents), std::move(parent_links),
                                 {}, retire_instances_only})
          .first;
  try {
    for (auto memory : it->second.memories)
      memories_[memory].insert(handle);
    for (auto parent : it->second.parents)
      children_[parent].insert(handle);
  } catch (...) {
    remove_edges(handle, it->second);
    entries_.erase(it);
    throw;
  }
  return handle;
}

std::shared_ptr<NativeResource> NativeResourceRegistry::get(
    TiNativeResource handle) const {
  const auto found = entries_.find(handle);
  if (found == entries_.end() || !found->second.object)
    throw MemoryBindingError(
        "Native resource is foreign, closed or has a retired dependency");
  return found->second.object;
}

TiNativeResource NativeResourceRegistry::publish_borrowed(
    std::shared_ptr<NativeResource> object,
    const std::vector<std::pair<std::string,
        std::shared_ptr<NativeResource>>> &inputs) {
  std::unordered_set<std::string> ports;
  for (const auto &[port, input] : inputs) {
    if (port.empty() || port.find('\0') != std::string::npos ||
        !ports.insert(port).second || !input || input == object)
      throw MemoryBindingError("Invalid borrowed native resource input");
  }
  std::vector<TiNativeResource> created;
  // Reserve before publishing anything, including room for the root. No GPU
  // work can use these handles until the complete transaction returns.
  created.reserve(inputs.size() + 1);
  try {
    std::unordered_map<NativeResource *, TiNativeResource> aliases;
    ParentLinks links;
    for (const auto &[port, input] : inputs) {
      auto found = aliases.find(input.get());
      TiNativeResource handle;
      if (found == aliases.end()) {
        handle = publish(input, {}, {});
        created.push_back(handle);
        aliases.emplace(input.get(), handle);
      } else {
        handle = found->second;
      }
      links.emplace_back(port, handle);
    }
    const auto root = publish(std::move(object), {}, {}, std::move(links));
    created.push_back(root);
    entries_.at(root).owned_parents.assign(created.begin(), created.end() - 1);
    return root;
  } catch (...) {
    // Unpublished cold handles have no recorded consumers. Roll back without
    // synchronizing or retiring unrelated work on the borrowed runtime.
    for (auto it = created.rbegin(); it != created.rend(); ++it) {
      const auto found = entries_.find(*it);
      remove_edges(*it, found->second);
      entries_.erase(found);
    }
    throw;
  }
}

TiNativeResource NativeResourceRegistry::resolve(
    TiNativeResource handle, const std::vector<std::string> &path) const {
  get(handle);
  for (const auto &port : path) {
    const auto &links = entries_.at(handle).parent_links;
    const auto found = std::lower_bound(
        links.begin(), links.end(), port,
        [](const auto &entry, const auto &name) { return entry.first < name; });
    if (found == links.end() || found->first != port)
      throw MemoryBindingError("Native resource has no retained input port: " + port);
    handle = found->second;
    get(handle);
  }
  return handle;
}

void NativeResourceRegistry::remove_edges(TiNativeResource handle,
                                          const Entry &entry) {
  const auto remove = [handle](auto &index, auto key) {
    const auto found = index.find(key);
    if (found == index.end())
      return;
    found->second.erase(handle);
    if (found->second.empty())
      index.erase(found);
  };
  for (auto memory : entry.memories)
    remove(memories_, memory);
  for (auto parent : entry.parents)
    remove(children_, parent);
}

void NativeResourceRegistry::retire(std::vector<TiNativeResource> roots) {
  std::unordered_set<TiNativeResource> seen(roots.begin(), roots.end());
  std::unordered_set<TiNativeResource> owned;
  for (std::size_t i = 0; i < roots.size(); ++i) {
    const auto children = children_.find(roots[i]);
    if (children != children_.end())
      for (auto child : children->second)
        if (seen.insert(child).second)
          roots.push_back(child);
    for (auto parent : entries_.at(roots[i]).owned_parents) {
      owned.insert(parent);
      if (seen.insert(parent).second)
        roots.push_back(parent);
    }
  }
  if (roots.empty())
    return;
  if (runtime_.runtime_fault_domain()->backend_calls_safe())
    runtime_.wait();
  // A child is published after all its parents. Retire each dependent recording
  // before any factory-owned state; destroy children before their parents.
  std::sort(roots.begin(), roots.end(), [](auto a, auto b) {
    return reinterpret_cast<std::uintptr_t>(a) >
           reinterpret_cast<std::uintptr_t>(b);
  });
  for (auto handle : roots)
    runtime_.retire_prepared_native_resource(
        handle, false, entries_.at(handle).retire_instances_only);
  for (auto handle : roots) {
    auto &entry = entries_.at(handle);
    remove_edges(handle, entry);
    entry.object.reset();
    entry.memories.clear();
    entry.parents.clear();
    entry.parent_links.clear();
    entry.owned_parents.clear();
  }
  // These handles never escaped the import transaction. Public roots keep
  // their usual tombstones, while private input handles do not accumulate.
  for (auto handle : owned)
    entries_.erase(handle);
}

void NativeResourceRegistry::retire_memory(TiMemory memory) {
  const auto found = memories_.find(memory);
  if (found != memories_.end())
    retire({found->second.begin(), found->second.end()});
}

void NativeResourceRegistry::destroy(TiNativeResource handle) {
  const auto found = entries_.find(handle);
  if (found == entries_.end())
    throw MemoryBindingError(
        "Native resource handle is not owned by this runtime");
  if (found->second.object)
    retire({handle});
  entries_.erase(handle);
}

void NativeResourceRegistry::clear(bool synchronize) {
  if (synchronize && !entries_.empty() &&
      runtime_.runtime_fault_domain()->backend_calls_safe())
    runtime_.wait();
  // The enclosing runtime has already retired every prepared execution.
  std::vector<TiNativeResource> handles;
  for (const auto &[handle, entry] : entries_)
    handles.push_back(handle);
  std::sort(handles.begin(), handles.end(), [](auto a, auto b) {
    return reinterpret_cast<std::uintptr_t>(a) >
           reinterpret_cast<std::uintptr_t>(b);
  });
  memories_.clear();
  children_.clear();
  for (auto handle : handles)
    entries_.erase(handle);
}
}  // namespace capi

capi::NativeResourceRegistry &Runtime::native_resource_registry() {
  if (!native_resources_)
    native_resources_ = std::make_shared<capi::NativeResourceRegistry>(*this);
  return *native_resources_;
}
void Runtime::retire_native_memory(TiMemory memory) {
  if (native_resources_)
    native_resources_->retire_memory(memory);
}
void Runtime::clear_native_resources(bool synchronize) {
  std::lock_guard<std::recursive_mutex> lock(submission_mutex());
  if (native_resources_)
    native_resources_->clear(synchronize);
}

TiNativeResource ti_create_native_resource_ext(TiRuntime runtime,
                                               uint64_t description_size,
                                               const char *description,
                                               uint32_t arg_count,
                                               const TiNamedArgument *args,
                                               uint32_t *retained_args) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(description);
  if ((arg_count && !args) ||
      description_size > std::numeric_limits<std::size_t>::max())
    throw MemoryBindingError("Invalid native resource input descriptors");
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  rt.runtime_fault_domain()->throw_if_submission_disallowed(
      "create AOT resource");
  using namespace taichi::lang;
  try {
    auto request =
        NativeOperationDescription::parse(liong::json::parse_complete(
            std::string(description, description_size)));
    const auto &registry = native_operation_registry();
    const auto decoded = registry.decode(request, NativeObjectKind::resource);
    auto context = capi::make_native_preparation_context(rt);
    if (!registry
             .probe(decoded.kind(), decoded.version(), rt.arch,
                    context->providers, NativeObjectKind::resource)
             .available)
      throw MemoryBindingError(
          "Native resource provider unavailable on this runtime");
    std::map<std::string, uint32_t> indices;
    for (uint32_t i = 0; i < arg_count; ++i)
      if (!args[i].name || !*args[i].name ||
          !indices.emplace(args[i].name, i).second)
        throw MemoryBindingError(
            "Native resource arguments require unique nonempty names");
    NativeOperationBindings inputs;
    std::unordered_set<uint32_t> used;
    for (const auto &[port, slot] : request.bindings) {
      const auto found = indices.find(slot);
      if (found == indices.end())
        throw MemoryBindingError("Missing native resource input: " + slot);
      used.insert(found->second);
      const auto &arg = args[found->second].argument;
      if (arg.type == TI_ARGUMENT_TYPE_NDARRAY)
        inputs.emplace(port, capi::bind_native_array(rt, arg.value.ndarray));
      else if (arg.type == TI_ARGUMENT_TYPE_NATIVE_RESOURCE_EXT)
        inputs.emplace(
            port, rt.native_resource_registry().get(arg.value.native_resource));
      else
        throw MemoryBindingError(
            "Native resource input requires an ndarray or resource");
    }
    if (used.size() != arg_count)
      throw MemoryBindingError("Unused native resource argument");
    auto object = registry.prepare_resource(*context, decoded, inputs);
    std::vector<uint32_t> retained(arg_count, 0);
    std::vector<TiMemory> memories;
    std::vector<TiNativeResource> parents;
    capi::NativeResourceRegistry::ParentLinks parent_links;
    for (const auto &[port, slot] : request.bindings) {
      if (!object->retains_input(port))
        continue;
      const auto index = indices.at(slot);
      retained[index] = 1;
      const auto &arg = args[index].argument;
      if (arg.type == TI_ARGUMENT_TYPE_NDARRAY)
        memories.push_back(arg.value.ndarray.memory);
      else
        parent_links.emplace_back(port, arg.value.native_resource);
    }
    auto handle = rt.native_resource_registry().publish(
        std::move(object), std::move(memories), std::move(parents),
        std::move(parent_links));
    if (retained_args)
      std::copy(retained.begin(), retained.end(), retained_args);
    return handle;
  } catch (const std::invalid_argument &error) {
    throw MemoryBindingError(error.what());
  }
  TI_CAPI_TRY_CATCH_END();
  return TI_NULL_HANDLE;
}

TiNativeResource ti_create_host_blob_ext(TiRuntime runtime,
                                         uint64_t size,
                                         const void *data) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  if ((size && !data) || size > std::numeric_limits<std::size_t>::max())
    throw MemoryBindingError("Invalid immutable host blob input");
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  rt.runtime_fault_domain()->throw_if_submission_disallowed(
      "create AOT host blob");
  using taichi::lang::NativeHostBlobResource;
  auto bytes = std::make_shared<NativeHostBlobResource::Bytes>(size);
  if (size)
    std::memcpy(bytes->data(), data, size);
  return rt.native_resource_registry().publish(
      std::make_shared<NativeHostBlobResource>(rt.arch, rt.get(),
                                               std::move(bytes)),
      {}, {});
  TI_CAPI_TRY_CATCH_END();
  return TI_NULL_HANDLE;
}

void ti_destroy_native_resource_ext(TiRuntime runtime,
                                    TiNativeResource resource) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(resource);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  rt.native_resource_registry().destroy(resource);
  TI_CAPI_TRY_CATCH_END();
}
