#pragma once

#if defined(TI_WITH_VULKAN)
#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::lang {

// Cold baked-data import. The BLAS owns allocations; recorded commands retain
// native objects independently, exactly as for existing AS and buffer objects.
class VulkanOpacityMicromap {
 public:
  explicit VulkanOpacityMicromap(vulkan::VulkanDevice *device);
  void build(const std::string &data,
             const std::string &descriptors,
             const std::string &indices,
             bool indexed,
             std::size_t triangle_count);
  void retain(const vkapi::IVkCommandBuffer &commands) const;

  VkAccelerationStructureTrianglesOpacityMicromapEXT attachment{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_TRIANGLES_OPACITY_MICROMAP_EXT};
  std::array<std::size_t, 3>
      memory{};  // array, indices, released import scratch

 private:
  struct Handle : vkapi::DeviceObj {
    VkMicromapEXT micromap{VK_NULL_HANDLE};
    PFN_vkDestroyMicromapEXT destroy{nullptr};
    vkapi::IVkBuffer storage;
    ~Handle() override;
  };
  DeviceAllocationUnique allocate(std::size_t bytes,
                                  AllocUsage usage,
                                  bool host_write = false);
  vulkan::VulkanDevice *device_;
  DeviceAllocationUnique storage_, indices_;
  std::shared_ptr<Handle> handle_;
  std::vector<VkMicromapUsageEXT> usage_;
};

}  // namespace taichi::lang
#endif
