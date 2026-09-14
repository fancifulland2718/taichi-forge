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

// Cold construction packs staging data but submits no GPU work. Initialization
// is recorded explicitly on the runtime's ordered compute queue. Command
// buffers retain both allocations, including after the prepared owner is
// released.
class VulkanShaderBindingTable {
 public:
  VulkanShaderBindingTable(VulkanDevice &device,
                           const VulkanPipeline &pipeline,
                           const VulkanShaderBindingRecord &raygen,
                           const std::vector<VulkanShaderBindingRecord> &miss,
                           const std::vector<VulkanShaderBindingRecord> &hit);

  // One-shot initialization, outside steady trace/replay. The owner publishes
  // readiness only after accepting this command into its ordered execution
  // flow.
  void record_initialization(VulkanCommandList &commands);
  bool initialization_recorded() const {
    return !staging_;
  }

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
  DeviceAllocationUnique staging_;
  vkapi::IVkBuffer buffer_;
  std::array<VkStridedDeviceAddressRegionKHR, 3> regions_{};
  std::size_t allocated_bytes_{0};
};

// Called when preparing fixed dimensions, never from ordinary dispatch/replay.
void validate_ray_dispatch(const VulkanDevice &device,
                           std::uint32_t width,
                           std::uint32_t height,
                           std::uint32_t depth);

// Runtime-ordered compute/transfer <-> programmable RT dependencies. Recorded
// around an external program, not inserted into ordinary dispatch or replay.
// AS-build -> RT reads are supplied by the bound AS owner's recording callback.
void record_ray_program_begin(VulkanCommandList &commands);
void record_ray_program_end(VulkanCommandList &commands);

}  // namespace taichi::lang::vulkan
