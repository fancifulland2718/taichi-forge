#pragma once

#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::lang::vulkan {

// Native Vulkan programs share the existing pipeline/layout/descriptor owners.
// This is intentionally not a new cross-backend RHI or a raw Python handle API.
struct VulkanRayTracingGroup {
  enum class Kind { raygen, miss, triangles };
  Kind kind{Kind::raygen};
  std::uint32_t general{VK_SHADER_UNUSED_KHR};
  std::uint32_t closest_hit{VK_SHADER_UNUSED_KHR};
  std::uint32_t any_hit{VK_SHADER_UNUSED_KHR};
};

struct VulkanRayTracingPipelineParams {
  std::vector<VulkanRayTracingGroup> groups;
  std::uint32_t max_recursion_depth{1};
  bool opacity_micromap{false};
};

struct VulkanShaderBindingRecord {
  std::uint32_t group{0};
  // Shader-record bytes follow the implementation's group handle. User data
  // obeys the explicit shader ABI; it is not a resource-ownership mechanism.
  std::vector<std::uint8_t> data;
};

// Cold preparation allocates and uploads once. Command buffers retain the Vk
// allocation independently, so releasing a prepared owner cannot free in-flight
// SBT memory. Runtime reset still retires program owners before the device.
class VulkanShaderBindingTable {
 public:
  VulkanShaderBindingTable(VulkanDevice &device,
                           const VulkanPipeline &pipeline,
                           const VulkanShaderBindingRecord &raygen,
                           const std::vector<VulkanShaderBindingRecord> &miss,
                           const std::vector<VulkanShaderBindingRecord> &hit);

  std::size_t allocated_bytes() const {
    return allocated_bytes_;
  }
  const VkStridedDeviceAddressRegionKHR &raygen() const {
    return regions_[0];
  }
  const VkStridedDeviceAddressRegionKHR &miss() const {
    return regions_[1];
  }
  const VkStridedDeviceAddressRegionKHR &hit() const {
    return regions_[2];
  }
  const vkapi::IVkBuffer &buffer() const {
    return buffer_;
  }

 private:
  DeviceAllocationUnique allocation_;
  vkapi::IVkBuffer buffer_;
  std::array<VkStridedDeviceAddressRegionKHR, 3> regions_{};
  std::size_t allocated_bytes_{0};
};

// Called when preparing fixed dimensions, never from ordinary dispatch/replay.
void validate_ray_dispatch(const VulkanDevice &device,
                           std::uint32_t width,
                           std::uint32_t height,
                           std::uint32_t depth);

}  // namespace taichi::lang::vulkan
