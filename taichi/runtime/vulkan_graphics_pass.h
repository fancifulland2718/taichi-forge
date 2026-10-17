#pragma once

#include "taichi/runtime/vulkan_graphics_types.h"
#include "taichi/rhi/device.h"

#ifdef TI_WITH_VULKAN
#include <functional>
#include <map>
#include <optional>

namespace taichi::lang {
struct VulkanGraphicsShaderBufferView {
  std::uint32_t set_index{0};
  std::uint32_t binding{0};
  DeviceAllocation allocation{kDeviceNullAllocation};
  std::vector<DeviceAllocation> array_elements;
  bool storage{false};
  size_t offset{0};
  size_t bytes{kBufferSizeEntireSize};
};

struct VulkanGraphicsShaderImageView {
  std::uint32_t set_index{0};
  std::uint32_t binding{0};
  DeviceAllocation allocation{kDeviceNullAllocation};
  ImageSamplerConfig sampler{};
};

struct VulkanGraphicsIndirectPacket {
  DeviceAllocation command_buffer{kDeviceNullAllocation};
  DeviceAllocation count_buffer{kDeviceNullAllocation};
  std::uint32_t command_offset{0};
  std::uint32_t count_offset{0};
  std::uint32_t max_draw_count{1};
  std::uint32_t stride{0};
  std::uint32_t vertex_record_limit{0};
  std::uint32_t instance_record_limit{0};
  std::uint32_t index_element_limit{0};
  bool first_instance_may_be_nonzero{false};
};

struct VulkanGraphicsDrawResources {
  std::vector<std::pair<std::uint32_t, std::unique_ptr<ShaderResourceSet>>>
      shader_resource_sets;
  std::unique_ptr<RasterResources> raster;
};

using VulkanGraphicsResourceSets =
    std::map<uint32_t, std::unique_ptr<ShaderResourceSet>>;
// Caller validates owners, reflection and ranges, and retains all resources
// through completion. The optional binder is synchronous, for owner-specific
// descriptors such as managed acceleration structures. It is never retained.
VulkanGraphicsDrawResources prepare_vulkan_graphics_draw_resources(
    GraphicsDevice &device,
    const std::vector<VulkanGraphicsShaderBufferView> &shader_buffers,
    const std::vector<VulkanGraphicsShaderImageView> &shader_images,
    const std::vector<std::pair<uint32_t, DevicePtr>> &vertex_buffers,
    DevicePtr index_buffer,
    bool indexed,
    bool mesh,
    const std::function<void(VulkanGraphicsResourceSets &)> &extra_bindings =
        {});

void record_vulkan_graphics_indirect_barriers(
    CommandList *commands,
    bool indexed,
    const VulkanGraphicsIndirectPacket &indirect);
void record_vulkan_graphics_draw(
    CommandList *commands,
    Pipeline *pipeline,
    bool task_shader,
    const VulkanGraphicsDrawInfo &draw,
    const std::optional<VulkanGraphicsIndirectPacket> &indirect,
    const std::optional<VulkanGraphicsMeshDrawInfo> &mesh,
    const VulkanGraphicsDrawResources &draw_resources);
}  // namespace taichi::lang
#endif
