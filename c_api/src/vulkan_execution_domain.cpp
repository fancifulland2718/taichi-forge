#include "vulkan_execution_domain.h"

#ifdef TI_WITH_VULKAN
#include <mutex>
#include <unordered_map>
#include "taichi/runtime/vulkan_execution_domain.h"

namespace capi {
namespace {
using taichi::lang::vulkan::ExecutionDomain;
struct VulkanInstanceHandle {
  std::shared_ptr<ExecutionDomain> domain;
  std::shared_ptr<taichi::lang::vulkan::ExecutionInstance> instance;
};
struct VulkanHandles {
  std::mutex mutex;
  std::unordered_map<TiVulkanExecutionDomain, std::shared_ptr<ExecutionDomain>> domains;
  std::unordered_map<TiVulkanExecutionInstance, VulkanInstanceHandle> instances;
};
VulkanHandles &vulkan_handles() {
  // Explicit release, never backend teardown from process-global destructors.
  static auto *registry = new VulkanHandles;
  return *registry;
}
}

std::shared_ptr<ExecutionDomain> get_execution_domain(TiVulkanExecutionDomain handle) {
  auto &registry = vulkan_handles();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto found = registry.domains.find(handle);
  if (found == registry.domains.end() || !found->second)
    throw MemoryBindingError("Vulkan execution-domain handle is closed or unknown");
  return found->second;
}

TiVulkanExecutionInstance publish_execution_instance(
    std::shared_ptr<ExecutionDomain> domain,
    std::shared_ptr<taichi::lang::vulkan::ExecutionInstance> instance) {
  if (!domain || !instance)
    throw MemoryBindingError("Prepared execution has no Vulkan instance");
  const auto handle = reinterpret_cast<TiVulkanExecutionInstance>(Runtime::new_handle_token());
  auto &registry = vulkan_handles();
  std::lock_guard<std::mutex> lock(registry.mutex);
  registry.instances.emplace(handle, VulkanInstanceHandle{std::move(domain), std::move(instance)});
  return handle;
}
}

TiVulkanExecutionDomain ti_create_vulkan_execution_domain_ext(
    TiRuntime runtime, const TiVulkanExecutionDomainCreateInfo *info) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(info);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  TI_CAPI_NOT_SUPPORTED_IF_RV(rt.arch != taichi::Arch::vulkan);
  if (!info->capacity || !info->instance_capacity || info->gpu_timing > TI_TRUE)
    throw MemoryBindingError("Invalid Vulkan execution-domain options");
  const auto handle = reinterpret_cast<TiVulkanExecutionDomain>(Runtime::new_handle_token());
  auto &registry = capi::vulkan_handles();
  {
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.domains.emplace(handle, nullptr);
  }
  try {
    auto domain = rt.vulkan_execution_domains()->create(
        {info->capacity, info->instance_capacity, info->gpu_timing != TI_FALSE});
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.domains.at(handle) = std::move(domain);
  } catch (...) {
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.domains.erase(handle);
    throw;
  }
  return handle;
  TI_CAPI_TRY_CATCH_END();
  return TI_NULL_HANDLE;
}

void ti_get_vulkan_execution_domain_info_ext(
    TiVulkanExecutionDomain domain, TiVulkanExecutionDomainInfo *info) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(info);
  *info = {};
  const auto value = capi::get_execution_domain(domain)->snapshot();
  *info = {value.identity, value.owner, value.queue_index, value.capacity,
           value.pending, static_cast<TiBool>(value.accepting), static_cast<TiBool>(value.closed)};
  TI_CAPI_TRY_CATCH_END();
}

void ti_stop_vulkan_execution_domain_ext(TiVulkanExecutionDomain domain) {
  TI_CAPI_TRY_CATCH_BEGIN();
  capi::get_execution_domain(domain)->stop_accepting();
  TI_CAPI_TRY_CATCH_END();
}
void ti_close_vulkan_execution_domain_ext(TiVulkanExecutionDomain domain) {
  TI_CAPI_TRY_CATCH_BEGIN();
  capi::get_execution_domain(domain)->close();
  TI_CAPI_TRY_CATCH_END();
}
void ti_release_vulkan_execution_domain_ext(TiVulkanExecutionDomain domain) {
  TI_CAPI_TRY_CATCH_BEGIN();
  std::shared_ptr<taichi::lang::vulkan::ExecutionDomain> released;
  auto &registry = capi::vulkan_handles();
  {
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto found = registry.domains.find(domain);
    if (found == registry.domains.end())
      throw MemoryBindingError("Vulkan execution-domain handle is closed or unknown");
    released = std::move(found->second);
    registry.domains.erase(found);
  }
  TI_CAPI_TRY_CATCH_END();
}
uint32_t ti_collect_vulkan_execution_domain_ext(TiVulkanExecutionDomain domain, uint32_t budget) {
  TI_CAPI_TRY_CATCH_BEGIN();
  return static_cast<uint32_t>(capi::get_execution_domain(domain)->collect_ready(budget));
  TI_CAPI_TRY_CATCH_END();
  return 0;
}
TiCompletion ti_get_vulkan_execution_domain_completion_ext(TiVulkanExecutionDomain domain) {
  TI_CAPI_TRY_CATCH_BEGIN();
  return capi::CompletionPublication{}.commit(capi::get_execution_domain(domain)->latest_completion());
  TI_CAPI_TRY_CATCH_END();
  return TI_NULL_HANDLE;
}
void ti_retire_vulkan_execution_instance_ext(TiVulkanExecutionInstance instance) {
  TI_CAPI_TRY_CATCH_BEGIN();
  capi::VulkanInstanceHandle retained;
  auto &registry = capi::vulkan_handles();
  {
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto found = registry.instances.find(instance);
    if (found == registry.instances.end())
      throw MemoryBindingError("Vulkan instance handle is closed or unknown");
    retained = found->second;
  }
  retained.domain->retire_instance(retained.instance);
  TI_CAPI_TRY_CATCH_END();
}
void ti_release_vulkan_execution_instance_ext(TiVulkanExecutionInstance instance) {
  TI_CAPI_TRY_CATCH_BEGIN();
  capi::VulkanInstanceHandle released;
  auto &registry = capi::vulkan_handles();
  {
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto found = registry.instances.find(instance);
    if (found == registry.instances.end())
      throw MemoryBindingError("Vulkan instance handle is closed or unknown");
    released = std::move(found->second);
    registry.instances.erase(found);
  }
  TI_CAPI_TRY_CATCH_END();
}
#endif
