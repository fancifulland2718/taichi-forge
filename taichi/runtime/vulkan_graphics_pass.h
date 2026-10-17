#pragma once

#include "taichi/runtime/vulkan_graphics_types.h"
#include "taichi/rhi/device.h"

#ifdef TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_common.h"
#include <functional>
#include <map>
#include <optional>

namespace taichi::lang {
namespace vulkan {
class VulkanDevice;
class VulkanPipeline;
}  // namespace vulkan

struct VulkanGraphicsColorTarget {
  DeviceAllocation allocation{kDeviceNullAllocation};
  bool clear{true};
  std::array<uint32_t, 4> clear_bits{};
};

// Physical attachment certificate prepared once, without a Program. The caller
// retains image owners and supplies initialization, layout and queue ordering.
// This packet owns no images and does not establish initialized contents.
struct VulkanGraphicsPassTargets {
  std::vector<DeviceAllocation> colors;
  std::vector<VkClearColorValue> clear_colors;
  std::shared_ptr<bool[]> color_clears;
  std::vector<int> color_types;
  std::vector<bool> color_blendable;
  DeviceAllocation depth{kDeviceNullAllocation};
  bool depth_clear{true};
  float clear_depth{0};
  uint32_t width{0}, height{0};
  std::array<uint32_t, 4> viewport{};
};

VulkanGraphicsPassTargets prepare_vulkan_graphics_pass_targets(
    vulkan::VulkanDevice &device,
    const std::vector<VulkanGraphicsColorTarget> &colors,
    DeviceAllocation depth,
    bool depth_clear,
    float clear_depth,
    std::array<uint32_t, 4> viewport = {});
void validate_vulkan_graphics_pipeline_targets(
    vulkan::VulkanPipeline &pipeline,
    const VulkanGraphicsPassTargets &targets);
void begin_vulkan_graphics_pass(CommandList *commands,
                                const VulkanGraphicsPassTargets &targets);

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
