#pragma once

#include <cstddef>
#include <memory>
#include "taichi/runtime/vulkan_ray_query_packet.h"
#include "taichi/runtime/vulkan_ray_types.h"

namespace taichi::lang {

#if defined(TI_WITH_VULKAN)
namespace vulkan {
class VulkanDevice;
}

// Device-local AS storage and commands shared by JIT and standalone consumers.
// The caller retains this object and device until recorded work has retired.
class VulkanTriangleRayScene {
 public:
  VulkanTriangleRayScene(vulkan::VulkanDevice *device,
                         std::size_t vertex_count,
                         std::size_t triangle_count);
  ~VulkanTriangleRayScene();
  VulkanTriangleRayScene(const VulkanTriangleRayScene &) = delete;
  VulkanTriangleRayScene &operator=(const VulkanTriangleRayScene &) = delete;

  VulkanTriangleRaySceneMemoryStatistics memory_statistics() const;
  std::size_t vertex_count() const;
  std::size_t triangle_count() const;
  void record_build(CommandList *commands,
                    DevicePtr vertices,
                    DevicePtr indices);
  void record_refit(CommandList *commands, DevicePtr vertices);
  void prepare_query_variant(unsigned variant);
  void record_query(CommandList *commands, const VulkanRayQueryPacket &packet);

 private:
  struct State;
  std::unique_ptr<State> state_;
};
#endif

}  // namespace taichi::lang
