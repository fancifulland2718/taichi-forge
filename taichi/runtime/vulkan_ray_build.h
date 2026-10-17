#pragma once

#if defined(TI_WITH_VULKAN)
#include <limits>
#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::lang::vulkan_ray_detail {
// Kernel/transfer producers, previous AS builds and queries may precede this
// command on the same runtime queue. buffer_copy itself inserts no dependency.
// Also order reuse of provider-owned geometry, AS storage and build scratch.
inline void geometry_reuse_barrier(const vkapi::IVkCommandBuffer &commands) {
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT |
                          VK_ACCESS_TRANSFER_WRITE_BIT |
                          VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                          VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                          VK_ACCESS_TRANSFER_WRITE_BIT |
                          VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                          VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  commands->device.functions().vkCmdPipelineBarrier(
      commands->buffer,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
          VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_PIPELINE_STAGE_TRANSFER_BIT |
          VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      0, 1, &barrier, 0, nullptr, 0, nullptr);
}

template <typename Function>
Function load_vulkan_device_function(const vkapi::DeviceHandle &device,
                                     const char *name) {
  auto function = reinterpret_cast<Function>(
      device.instance_functions().vkGetDeviceProcAddr(device, name));
  TI_ERROR_IF(function == nullptr,
              "Vulkan ray provider could not load required function '{}'.",
              name);
  return function;
}

inline std::size_t checked_mul(std::size_t lhs,
                               std::size_t rhs,
                               const char *description) {
  TI_ERROR_IF(lhs != 0 && rhs > (std::numeric_limits<std::size_t>::max)() / lhs,
              "Vulkan ray {} size overflow.", description);
  return lhs * rhs;
}

inline VkDeviceAddress aligned_address(VkDeviceAddress address,
                                       VkDeviceSize alignment) {
  TI_ASSERT(alignment != 0 && (alignment & (alignment - 1)) == 0);
  return (address + alignment - 1) & ~(alignment - 1);
}

}  // namespace taichi::lang::vulkan_ray_detail
#endif
