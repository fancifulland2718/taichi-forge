#pragma once

#include "taichi/runtime/vulkan_ray_types.h"

#if defined(TI_WITH_VULKAN)
#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::lang {
class VulkanOpacityMicromap;

// Device-local geometry and build/refit commands. Program/AOT wrappers own
// submission, prepared leases and retirement; this core owns no Program.
class VulkanTriangleBlas {
 public:
  VulkanTriangleBlas(vulkan::VulkanDevice *device,
                     std::size_t vertex_count,
                     std::size_t triangle_count,
                     std::shared_ptr<VulkanOpacityMicromap> micromap = {},
                     bool opaque = true);
  ~VulkanTriangleBlas();
  VulkanTriangleBlas(const VulkanTriangleBlas &) = delete;
  VulkanTriangleBlas &operator=(const VulkanTriangleBlas &) = delete;

  VulkanTriangleRaySceneMemoryStatistics memory_statistics() const;
  vulkan::VulkanDevice *device() const;
  std::size_t vertex_count() const;
  std::size_t triangle_count() const;
  VkDeviceAddress device_address() const;
  void record_build(CommandList *commands,
                    DevicePtr vertices,
                    DevicePtr indices,
                    bool update);
  void retain(const vkapi::IVkCommandBuffer &commands) const;

 private:
  struct State;
  std::unique_ptr<State> state_;
};
}  // namespace taichi::lang
#endif
