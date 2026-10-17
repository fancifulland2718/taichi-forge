#pragma once

#include "taichi/runtime/vulkan_ray_query_packet.h"

#if defined(TI_WITH_VULKAN)
#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::lang {
VulkanRayQueryPacket prepare_vulkan_ray_query_packet(
    vulkan::VulkanDevice *device,
    std::size_t ray_count,
    bool typed,
    const std::array<VulkanRayQueryPacket::Binding, 3> &bindings);

// Scene and TLAS share shaders/recording. Each resource owns its pipelines and
// bindings and serializes prepare/record/clear with its existing lock.
class VulkanRayQueryPipelines {
 public:
  void clear();
  void prepare(vulkan::VulkanDevice *device, unsigned variant);
  void record(CommandList *commands,
              const vkapi::IVkAccelerationStructureKHR &tlas,
              const VulkanRayQueryPacket &packet);

 private:
  std::array<std::unique_ptr<Pipeline>, 4> pipelines_;
  std::array<std::unique_ptr<ShaderResourceSet>, 4> bindings_;
};
}  // namespace taichi::lang
#endif
