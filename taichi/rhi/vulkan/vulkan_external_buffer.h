#pragma once

#include <array>
#include <cstdint>
#include <memory>

#include "taichi/rhi/vulkan/vulkan_api.h"

namespace taichi::lang::vulkan {

class VulkanDevice;
class VulkanCommandList;

// Actual producer facts, not consumer guesses. The first route accepts a
// dedicated, exclusive, whole buffer with bind offset zero. No live OS handles
// enter this description. Import does not acquire access to the payload.
struct VulkanExternalBufferDescription {
  VkDeviceSize size{0};
  VkDeviceSize allocation_size{0};
  uint32_t memory_type_index{UINT32_MAX};
  VkBufferCreateFlags flags{0};
  VkBufferUsageFlags usage{0};
  VkMemoryAllocateFlags allocation_flags{0};
  VkSharingMode sharing_mode{VK_SHARING_MODE_EXCLUSIVE};
  VkDeviceSize bind_offset{0};
  bool dedicated{true};
  std::array<uint8_t, VK_UUID_SIZE> device_uuid{};
  std::array<uint8_t, VK_UUID_SIZE> driver_uuid{};
};

struct VulkanExternalBufferHandles {
  // Borrowed OPAQUE_WIN32 NT handles, all distinct payloads. Originals remain
  // owned by the caller on success and every failure path.
  uintptr_t memory{0};
  uintptr_t ready{0};
  uintptr_t consumed{0};
};

// Internal cold-stage RHI building block. It owns the local buffer, memory and
// semaphore imports. Command references keep these objects alive while pending;
// the VulkanDevice must outlive those references. Epoch admission, completion
// retirement and queue selection belong to the existing execution owner.
// Deliberately not an ExternalSynchronizationDomain until that owner is wired.
class VulkanExternalBuffer final
    : public vkapi::DeviceObjVkBuffer,
      public std::enable_shared_from_this<VulkanExternalBuffer> {
 public:
  // Read-only cold qualification shared by native frontends. The actual
  // importer repeats it before touching handles; no rule is copied by callers.
  static void validate_import(VulkanDevice &device,
                             const VulkanExternalBufferDescription &description,
                             const VulkanExternalBufferHandles &handles);
  static std::shared_ptr<VulkanExternalBuffer> import_borrowed(
      VulkanDevice &device,
      const VulkanExternalBufferDescription &description,
      const VulkanExternalBufferHandles &handles,
      std::shared_ptr<void> producer_retention = {});
  ~VulkanExternalBuffer() override;

  VulkanExternalBuffer(const VulkanExternalBuffer &) = delete;
  VulkanExternalBuffer &operator=(const VulkanExternalBuffer &) = delete;

  const VulkanExternalBufferDescription &description() const {
    return description_;
  }
  VkDeviceMemory memory() const {
    return memory_;
  }
  const vkapi::IVkSemaphore &ready() const {
    return ready_;
  }
  const vkapi::IVkSemaphore &consumed() const {
    return consumed_;
  }

  // Record the complete-buffer ownership transfer on the command's family.
  // Caller must wait ready before acquire and signal consumed after release,
  // in the same queue submission. The producer's binary signal must already
  // be submitted (it need not be complete). These methods never submit or
  // wait on host. They are recording operations, not an epoch admission API.
  void record_acquire(VulkanCommandList &commands);
  void record_release(VulkanCommandList &commands);

 private:
  explicit VulkanExternalBuffer(VulkanExternalBufferDescription description,
                                std::shared_ptr<void> producer_retention)
      : description_(std::move(description)),
        producer_retention_(std::move(producer_retention)) {
  }
  void record_transfer(VulkanCommandList &commands, bool acquire);
  const VulkanExternalBufferDescription description_;
  // Declared before native members; survives destruction of all imports even
  // when command references outlive the public allocation owner.
  const std::shared_ptr<void> producer_retention_;
  VkDeviceMemory memory_{VK_NULL_HANDLE};
  vkapi::IVkSemaphore ready_;
  vkapi::IVkSemaphore consumed_;
};

}  // namespace taichi::lang::vulkan
