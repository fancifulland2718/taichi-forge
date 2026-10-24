#pragma once

#if defined(TI_WITH_VULKAN)
#include <string_view>
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/runtime/vulkan_ray_types.h"

namespace taichi::lang {

// Cold baked-data import. The BLAS owns allocations; recorded commands retain
// native objects independently, exactly as for existing AS and buffer objects.
class VulkanOpacityMicromap {
 public:
  explicit VulkanOpacityMicromap(vulkan::VulkanDevice *device);
  void build(std::string_view data,
             std::string_view descriptors,
             std::string_view indices,
             bool indexed,
             std::size_t triangle_count);
  // Cold import into the caller's command stream. The caller must complete
  // this one-shot construction before publishing the micromap to AS users.
  // Recorded commands retain every native input and scratch allocation.
  void record_build(CommandList *commands,
                    std::string_view data,
                    std::string_view descriptors,
                    std::string_view indices,
                    bool indexed,
                    std::size_t triangle_count);
  void release_build_inputs();  // Only after the construction completes.
  void retain(const vkapi::IVkCommandBuffer &commands) const;
  VulkanTriangleRaySceneMemoryStatistics memory_statistics() const;

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
  // Kept on the owner until the cold submission completes. A faulted AOT owner
  // can abandon this complete core without freeing buffers still in flight.
  std::vector<DeviceAllocationUnique> temporary_;
};

}  // namespace taichi::lang
#endif
