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
    if (import_info->struct_size != sizeof(TiVulkanExternalBufferImportInfo) ||
        import_info->schema_version != 1)
      throw std::invalid_argument(
          "Invalid external Vulkan import descriptor size or version");
    const auto info = *import_info;
    constexpr std::uint32_t supported_usage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    constexpr auto max_handle = std::numeric_limits<std::uintptr_t>::max();
    if (!info.buffer_size || info.allocation_size < info.buffer_size ||
        info.allocation_size > std::numeric_limits<std::size_t>::max() ||
        info.bind_offset || info.dedicated != 1 || info.buffer_flags ||
        info.sharing_mode != VK_SHARING_MODE_EXCLUSIVE ||
        !(info.buffer_usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ||
        (info.buffer_usage & ~supported_usage) ||
        (info.allocation_flags & ~VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT) ||
        info.memory_type_index >= VK_MAX_MEMORY_TYPES || !info.byte_size ||
        info.byte_offset > info.buffer_size ||
        info.byte_size > info.buffer_size - info.byte_offset ||
        !info.memory_handle || !info.ready_for_consumer ||
        !info.ready_for_external || info.memory_handle >= max_handle ||
        info.ready_for_consumer >= max_handle ||
        info.ready_for_external >= max_handle ||
        info.memory_handle == info.ready_for_consumer ||
        info.memory_handle == info.ready_for_external ||
        info.ready_for_consumer == info.ready_for_external ||
        std::all_of(std::begin(info.device_uuid), std::end(info.device_uuid),
                    [](auto value) { return value == 0; }) ||
        std::all_of(std::begin(info.driver_uuid), std::end(info.driver_uuid),
                    [](auto value) { return value == 0; }))
      throw std::invalid_argument(
          "Invalid external Vulkan buffer import descriptor");
    validate_native_ready_gate(*ready_gate);
#ifdef _WIN64
    auto &device = static_cast<vulkan::VulkanDevice &>(rt.get());
    const bool address =
        info.buffer_usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    const bool allocation_address =
        info.allocation_flags & VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    if ((address && !allocation_address) ||
        ((address || allocation_address) &&
         !device.vk_caps().buffer_device_address) ||
        (device.get_caps().get(
             DeviceCapability::spirv_has_physical_storage_buffer) &&
         (!address || !allocation_address)))
      throw std::invalid_argument(
          "External Vulkan buffer lacks required device-address flags");
    vulkan::VulkanExternalBufferDescription description;
    description.size = info.buffer_size;
    description.allocation_size = info.allocation_size;
    description.bind_offset = info.bind_offset;
    description.memory_type_index = info.memory_type_index;
    description.flags = info.buffer_flags;
    description.usage = info.buffer_usage;
    description.allocation_flags = info.allocation_flags;
    description.sharing_mode = static_cast<VkSharingMode>(info.sharing_mode);
    description.dedicated = true;
    std::copy(std::begin(info.device_uuid), std::end(info.device_uuid),
              description.device_uuid.begin());
    std::copy(std::begin(info.driver_uuid), std::end(info.driver_uuid),
              description.driver_uuid.begin());
    const vulkan::VulkanExternalBufferHandles handles{
        static_cast<std::uintptr_t>(info.memory_handle),
        static_cast<std::uintptr_t>(info.ready_for_consumer),
        static_cast<std::uintptr_t>(info.ready_for_external)};
    // Device/driver identity, native import support and buffer requirements
    // are checked by the RHI importer while the producer lease is retained.
    auto owner = vulkan::VulkanExternalBufferOwner::import_borrowed(
        device, description, handles, retain_native_ready_gate(*ready_gate));
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
