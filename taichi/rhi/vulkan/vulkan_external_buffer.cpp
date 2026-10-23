#include "taichi/rhi/vulkan/vulkan_external_buffer.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <system_error>

#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::lang::vulkan {
namespace {

#ifdef _WIN64
void check(VkResult result, const char *operation) {
  if (result != VK_SUCCESS)
    throw std::runtime_error(std::string(operation) +
                             " failed: " + std::to_string(result));
}

class BorrowedHandleCopy {
 public:
  explicit BorrowedHandleCopy(uintptr_t original) {
    if (!DuplicateHandle(
            GetCurrentProcess(), reinterpret_cast<HANDLE>(original),
            GetCurrentProcess(), &value_, 0, FALSE, DUPLICATE_SAME_ACCESS))
      throw std::system_error(GetLastError(), std::system_category(),
                              "DuplicateHandle for Vulkan import");
  }
  ~BorrowedHandleCopy() {
    CloseHandle(value_);
  }
  BorrowedHandleCopy(const BorrowedHandleCopy &) = delete;
  BorrowedHandleCopy &operator=(const BorrowedHandleCopy &) = delete;
  HANDLE get() const {
    return value_;
  }

 private:
  HANDLE value_{nullptr};
};

void validate(VulkanDevice &device,
              const VulkanExternalBufferDescription &d,
              const VulkanExternalBufferHandles &h) {
  constexpr VkBufferUsageFlags supported_usage =
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  if (!d.size || d.allocation_size < d.size || !d.dedicated || d.bind_offset ||
      d.flags || d.sharing_mode != VK_SHARING_MODE_EXCLUSIVE || !d.usage ||
      (d.usage & ~supported_usage) ||
      (d.allocation_flags & ~VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT))
    throw std::invalid_argument(
        "Unsupported external Vulkan buffer description");
  if (!h.memory || !h.ready || !h.consumed || h.memory == h.ready ||
      h.memory == h.consumed || h.ready == h.consumed)
    throw std::invalid_argument(
        "External Vulkan import requires three distinct NT handles");
  const bool address = d.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  if (address &&
      (!(d.allocation_flags & VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT) ||
       !device.vk_caps().buffer_device_address))
    throw std::invalid_argument(
        "External Vulkan buffer device address is unavailable");
  if ((d.allocation_flags & VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT) &&
      !device.vk_caps().buffer_device_address)
    throw std::invalid_argument(
        "External Vulkan allocation requires device address support");
  const auto native = device.vk_device();
  const auto &api = native.functions();
  const auto &instance = native.instance_functions();
  if (!device.vk_caps().external_memory ||
      !device.vk_caps().external_semaphore ||
      !api.vkImportSemaphoreWin32HandleKHR ||
      !instance.vkGetPhysicalDeviceProperties2 ||
      !instance.vkGetPhysicalDeviceExternalBufferProperties ||
      !instance.vkGetPhysicalDeviceExternalSemaphoreProperties)
    throw std::invalid_argument("Vulkan external Win32 import is unavailable");
  VkPhysicalDeviceIDProperties ids{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
  VkPhysicalDeviceProperties2 properties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  properties.pNext = &ids;
  instance.vkGetPhysicalDeviceProperties2(device.vk_physical_device(),
                                          &properties);
  if (std::all_of(d.device_uuid.begin(), d.device_uuid.end(),
                  [](auto b) { return !b; }) ||
      std::all_of(d.driver_uuid.begin(), d.driver_uuid.end(),
                  [](auto b) { return !b; }) ||
      std::memcmp(ids.deviceUUID, d.device_uuid.data(), VK_UUID_SIZE) ||
      std::memcmp(ids.driverUUID, d.driver_uuid.data(), VK_UUID_SIZE))
    throw std::invalid_argument(
        "External Vulkan device or driver UUID does not match");
  VkPhysicalDeviceMemoryProperties memory{};
  instance.vkGetPhysicalDeviceMemoryProperties(device.vk_physical_device(),
                                               &memory);
  if (d.memory_type_index >= memory.memoryTypeCount)
    throw std::invalid_argument(
        "External Vulkan producer memory type is missing or invalid");
  VkPhysicalDeviceExternalBufferInfo info{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO};
  info.flags = d.flags;
  info.usage = d.usage;
  info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
  VkExternalBufferProperties external{
      VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
  instance.vkGetPhysicalDeviceExternalBufferProperties(
      device.vk_physical_device(), &info, &external);
  if (!(external.externalMemoryProperties.externalMemoryFeatures &
        VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT))
    throw std::invalid_argument(
        "External Vulkan buffer usage cannot import OPAQUE_WIN32");
  VkPhysicalDeviceExternalSemaphoreInfo semaphore_info{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO};
  semaphore_info.handleType =
      VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
  VkExternalSemaphoreProperties semaphore_properties{
      VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES};
  instance.vkGetPhysicalDeviceExternalSemaphoreProperties(
      device.vk_physical_device(), &semaphore_info, &semaphore_properties);
  if (!(semaphore_properties.externalSemaphoreFeatures &
        VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT))
    throw std::invalid_argument(
        "External Vulkan binary semaphore import is unavailable");
}

vkapi::IVkSemaphore import_semaphore(const vkapi::DeviceHandle &device,
                                     HANDLE handle) {
  auto result = vkapi::create_semaphore(device, 0);
  if (!result)
    throw std::runtime_error("Cannot create external Vulkan semaphore");
  VkImportSemaphoreWin32HandleInfoKHR info{
      VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR};
  info.semaphore = result->semaphore;
  info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
  info.handle = handle;
  check(device.functions().vkImportSemaphoreWin32HandleKHR(device, &info),
        "vkImportSemaphoreWin32HandleKHR");
  return result;
}
#endif
}  // namespace

std::shared_ptr<VulkanExternalBuffer> VulkanExternalBuffer::import_borrowed(
    VulkanDevice &device,
    const VulkanExternalBufferDescription &description,
    const VulkanExternalBufferHandles &handles) {
#ifdef _WIN64
  validate(device, description, handles);
  // Duplicate all handles before allocating any GPU objects. Vulkan does not
  // consume NT handle ownership; every copy closes on success or rollback.
  BorrowedHandleCopy memory_handle(handles.memory), ready_handle(handles.ready),
      consumed_handle(handles.consumed);
  using CompareHandles = BOOL(WINAPI *)(HANDLE, HANDLE);
  const auto compare = reinterpret_cast<CompareHandles>(GetProcAddress(
      GetModuleHandleW(L"kernelbase.dll"), "CompareObjectHandles"));
  if (!compare)
    throw std::invalid_argument(
        "Vulkan NT handle identity requires Windows 10");
  if (compare(ready_handle.get(), consumed_handle.get()) ||
      compare(memory_handle.get(), ready_handle.get()) ||
      compare(memory_handle.get(), consumed_handle.get()))
    throw std::invalid_argument(
        "External Vulkan import handles alias one payload");
  auto result = std::shared_ptr<VulkanExternalBuffer>(
      new VulkanExternalBuffer(description));
  result->device = device.vk_device();
  result->usage = description.usage;
  const auto &api = result->device.functions();
  VkExternalMemoryBufferCreateInfo sharing{
      VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
  sharing.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
  VkBufferCreateInfo create{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  create.pNext = &sharing;
  create.size = description.size;
  create.flags = description.flags;
  create.usage = description.usage;
  create.sharingMode = description.sharing_mode;
  check(api.vkCreateBuffer(result->device, &create, nullptr, &result->buffer),
        "vkCreateBuffer");
  VkMemoryRequirements requirements{};
  api.vkGetBufferMemoryRequirements(result->device, result->buffer,
                                    &requirements);
  if (requirements.size > description.allocation_size ||
      !(requirements.memoryTypeBits & (1u << description.memory_type_index)))
    throw std::invalid_argument(
        "External Vulkan allocation does not match buffer requirements");
  VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  flags.flags = description.allocation_flags;
  VkMemoryDedicatedAllocateInfo dedicated{
      VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
  dedicated.buffer = result->buffer;
  dedicated.pNext = &flags;
  VkImportMemoryWin32HandleInfoKHR imported{
      VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR};
  imported.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
  imported.handle = memory_handle.get();
  imported.pNext = &dedicated;
  VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocation.pNext = &imported;
  allocation.allocationSize = description.allocation_size;
  allocation.memoryTypeIndex = description.memory_type_index;
  check(api.vkAllocateMemory(result->device, &allocation, nullptr,
                             &result->memory_),
        "vkAllocateMemory");
  check(api.vkBindBufferMemory(result->device, result->buffer, result->memory_,
                               0),
        "vkBindBufferMemory");
  result->ready_ = import_semaphore(result->device, ready_handle.get());
  result->consumed_ = import_semaphore(result->device, consumed_handle.get());
  return result;
#else
  throw std::invalid_argument(
      "Vulkan OPAQUE_WIN32 buffer import requires Windows x64");
#endif
}

VulkanExternalBuffer::~VulkanExternalBuffer() {
  // Command references, not this destructor, provide pending-work protection.
  // No device-wide wait, no destruction of the producer's Vulkan objects.
  consumed_.reset();
  ready_.reset();
  if (buffer)
    device.functions().vkDestroyBuffer(device, buffer, nullptr);
  if (memory_)
    device.functions().vkFreeMemory(device, memory_, nullptr);
}

void VulkanExternalBuffer::record_transfer(VulkanCommandList &commands,
                                           bool acquire) {
  auto command = commands.vk_command_buffer();
  if (!command || command->device != device || !command->ref_pool)
    throw std::invalid_argument(
        "External Vulkan buffer command belongs to another device");
  command->refs.push_back(shared_from_this());
  const auto family = command->ref_pool->queue_family_index;
  VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  barrier.srcQueueFamilyIndex = acquire ? VK_QUEUE_FAMILY_EXTERNAL : family;
  barrier.dstQueueFamilyIndex = acquire ? family : VK_QUEUE_FAMILY_EXTERNAL;
  barrier.srcAccessMask =
      acquire ? 0 : VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  barrier.dstAccessMask =
      acquire ? VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT : 0;
  barrier.buffer = buffer;
  barrier.size = description_.size;
  device.functions().vkCmdPipelineBarrier(command->buffer,
                                          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                                          0, nullptr, 1, &barrier, 0, nullptr);
}

void VulkanExternalBuffer::record_acquire(VulkanCommandList &commands) {
  record_transfer(commands, true);
}
void VulkanExternalBuffer::record_release(VulkanCommandList &commands) {
  record_transfer(commands, false);
}

}  // namespace taichi::lang::vulkan
