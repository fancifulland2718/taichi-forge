#include "vulkan_execution_domain.h"

#ifdef TI_WITH_VULKAN
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <mutex>
#include <unordered_map>
#include "taichi/program/runtime_fault.h"
#include "taichi/rhi/interop/native_ready_gate.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_external_access.h"
#include "taichi/runtime/vulkan_execution_domain.h"

#ifdef _WIN64
static_assert(sizeof(TiVulkanExternalBufferImportInfo) == 128);
static_assert(offsetof(TiVulkanExternalBufferImportInfo, device_uuid) == 80);
static_assert(offsetof(TiVulkanExternalBufferImportInfo, byte_offset) == 112);
static_assert(sizeof(TiVulkanExternalReadyGate) == 40);
#endif

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

TiMemory ti_import_vulkan_external_buffer_ext(
    TiRuntime runtime,
    const TiVulkanExternalBufferImportInfo *import_info,
    const TiVulkanExternalReadyGate *ready_gate) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(import_info);
  TI_CAPI_ARGUMENT_NULL_RV(ready_gate);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  TI_CAPI_NOT_SUPPORTED_IF_RV(rt.arch != taichi::Arch::vulkan);
  rt.runtime_fault_domain()->throw_if_new_submission_disallowed(
      "import external Vulkan buffer");
  try {
    using namespace taichi::lang;
    const auto info = vulkan::decode_vulkan_external_buffer_import(*import_info);
    validate_native_ready_gate(*ready_gate);
#ifdef _WIN64
    auto &device = static_cast<vulkan::VulkanDevice &>(rt.get());
    auto owner = vulkan::import_vulkan_external_ndarray(device, info, *ready_gate);
    const auto allocation = owner->ndarray_allocation();
    auto storage = std::make_shared<Runtime::BorrowedMemory>();
    storage->owner = owner;
    storage->external_access = owner;
    return rt.register_memory(allocation, info.byte_size, false,
                              info.byte_offset, {}, info.byte_offset, false,
                              std::move(storage));
#else
    ti_set_last_error(TI_ERROR_NOT_SUPPORTED,
                      "Vulkan OPAQUE_WIN32 buffer import requires Windows x64");
    return TI_NULL_HANDLE;
#endif
  } catch (const std::invalid_argument &error) {
    throw MemoryBindingError(error.what());
  }
  TI_CAPI_TRY_CATCH_END();
  return TI_NULL_HANDLE;
}

TiVulkanExecutionDomain ti_create_vulkan_execution_domain_ext(
    TiRuntime runtime, const TiVulkanExecutionDomainCreateInfo *info) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(info);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  TI_CAPI_NOT_SUPPORTED_IF_RV(rt.arch != taichi::Arch::vulkan);
  if (!info->capacity || !info->instance_capacity || info->gpu_timing > TI_TRUE ||
      (info->queue_kind != TI_VULKAN_EXECUTION_QUEUE_COMPUTE &&
       info->queue_kind != TI_VULKAN_EXECUTION_QUEUE_GRAPHICS))
    throw MemoryBindingError("Invalid Vulkan execution-domain options");
  const auto handle = reinterpret_cast<TiVulkanExecutionDomain>(Runtime::new_handle_token());
  auto &registry = capi::vulkan_handles();
  {
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.domains.emplace(handle, nullptr);
  }
  try {
    auto domain = rt.vulkan_execution_domains()->create(
        {info->capacity, info->instance_capacity, info->gpu_timing != TI_FALSE,
         info->queue_kind == TI_VULKAN_EXECUTION_QUEUE_GRAPHICS
             ? taichi::lang::vulkan::VulkanExecutionQueueKind::graphics
             : taichi::lang::vulkan::VulkanExecutionQueueKind::compute});
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
           value.pending, static_cast<TiBool>(value.accepting), static_cast<TiBool>(value.closed),
           value.queue_family_index};
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
#else
// Keep this symbol available when the source is built without Vulkan. The
// opaque descriptors require no Vulkan headers for the unsupported stub.
extern "C" {
struct TiVulkanExternalBufferImportInfo;
struct TiVulkanExternalReadyGate;
TI_DLL_EXPORT TiMemory TI_API_CALL ti_import_vulkan_external_buffer_ext(
    TiRuntime,
    const TiVulkanExternalBufferImportInfo *,
    const TiVulkanExternalReadyGate *) {
  ti_set_last_error(TI_ERROR_NOT_SUPPORTED, "Vulkan support is not compiled in");
  return TI_NULL_HANDLE;
}
}
#endif
