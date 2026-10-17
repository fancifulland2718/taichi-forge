#include "taichi/runtime/vulkan_graphics_pass.h"

#ifdef TI_WITH_VULKAN
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_set>
#include "taichi/program/texture.h"
#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::lang {
VulkanGraphicsPassTargets prepare_vulkan_graphics_pass_targets(
    vulkan::VulkanDevice &device,
    const std::vector<VulkanGraphicsColorTarget> &colors,
    DeviceAllocation depth,
    bool depth_clear,
    float clear_depth,
    std::array<uint32_t, 4> viewport) {
  TI_ERROR_IF(colors.empty() && depth == kDeviceNullAllocation,
              "Vulkan graphics pass requires a color or depth attachment.");
  TI_ERROR_IF(colors.size() > device.vk_caps().max_color_attachments,
              "Vulkan color attachment count exceeds the device limit");
  TI_ERROR_IF(!std::isfinite(clear_depth) || clear_depth < 0 || clear_depth > 1,
              "Vulkan graphics clear_depth must be finite and in [0, 1].");
  VulkanGraphicsPassTargets result;
  result.depth = depth;
  result.depth_clear = depth_clear;
  result.clear_depth = clear_depth;
  result.color_clears = std::shared_ptr<bool[]>(new bool[colors.size()]);
  std::unordered_set<VkImage> images;
  std::unordered_map<VkFormat, VkFormatFeatureFlags> formats;
  const auto &limits = device.get_vk_physical_device_props().limits;
  auto validate = [&](DeviceAllocation allocation, bool is_depth) {
    TI_ERROR_IF(
        allocation == kDeviceNullAllocation || allocation.device != &device,
        "Vulkan graphics attachment belongs to another device or is null.");
    const auto [image, view, format] = device.get_vk_image(allocation);
    TI_ERROR_IF(image->type != VK_IMAGE_TYPE_2D ||
                    view->type != VK_IMAGE_VIEW_TYPE_2D || image->width == 0 ||
                    image->height == 0 || image->depth != 1,
                "Vulkan graphics attachments must be nonempty 2D images.");
    const auto &range = view->subresource_range;
    TI_ERROR_IF(image->mip_levels != 1 || range.baseMipLevel != 0 ||
                    range.levelCount != 1 || image->array_layers != 1 ||
                    range.baseArrayLayer != 0 || range.layerCount != 1,
                "Graphics attachments currently require single-level, "
                "single-layer Texture views");
    TI_ERROR_IF(!images.insert(image->image).second,
                "Graphics attachments must not alias");
    if (!result.width) {
      result.width = image->width;
      result.height = image->height;
    }
    TI_ERROR_IF(image->width != result.width || image->height != result.height,
                "Vulkan graphics attachments must have the same 2D shape.");
    TI_ERROR_IF(
        image->width > limits.maxFramebufferWidth ||
            image->height > limits.maxFramebufferHeight ||
            image->width > uint32_t((std::numeric_limits<int>::max)()) ||
            image->height > uint32_t((std::numeric_limits<int>::max)()),
        "Vulkan graphics attachment exceeds the device framebuffer or command "
        "coordinate limit.");
    const bool depth_format = format == VK_FORMAT_D16_UNORM ||
                              format == VK_FORMAT_D24_UNORM_S8_UINT ||
                              format == VK_FORMAT_D32_SFLOAT;
    TI_ERROR_IF(is_depth && format != VK_FORMAT_D32_SFLOAT,
                "Vulkan graphics depth attachments require depth32f.");
    TI_ERROR_IF(!is_depth && depth_format,
                "Color attachments cannot use depth formats");
    const auto usage = is_depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                                : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    TI_ERROR_IF(!(image->usage & usage),
                "Vulkan graphics image lacks the required attachment usage.");
    auto [it, inserted] = formats.emplace(format, 0);
    if (inserted) {
      VkFormatProperties properties{};
      device.vk_device()
          .instance_functions()
          .vkGetPhysicalDeviceFormatProperties(device.vk_physical_device(),
                                               format, &properties);
      it->second = properties.optimalTilingFeatures;
    }
    const auto feature = is_depth
                             ? VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
                             : VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
    TI_ERROR_IF(!(it->second & feature),
                "Texture format cannot be an attachment on this device");
    return std::make_pair(format, it->second);
  };
  for (size_t i = 0; i < colors.size(); ++i) {
    const auto &color = colors[i];
    const auto [format, features] = validate(color.allocation, false);
    const auto [status, buffer_format] = vulkan::buffer_format_vk_to_ti(format);
    TI_ERROR_IF(status != RhiResult::success,
                "Unsupported Vulkan graphics color attachment format");
    const auto [type, channels] = buffer_format2type_channels(buffer_format);
    const int numeric_type = is_float_sampled_texture_format(buffer_format)
                                 ? 1
                                 : (is_signed(type) ? 2 : 3);
    result.color_types.push_back(numeric_type);
    result.color_blendable.push_back(
        numeric_type == 1 &&
        (features & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT));
    result.colors.push_back(color.allocation);
    result.color_clears[i] = color.clear;
    VkClearColorValue value{};
    std::memcpy(&value, color.clear_bits.data(), sizeof(value));
    result.clear_colors.push_back(value);
  }
  if (depth != kDeviceNullAllocation)
    validate(depth, true);
  if (viewport[2] == 0 && viewport[3] == 0)
    viewport = {0, 0, result.width, result.height};
  const uint64_t x_end = uint64_t(viewport[0]) + viewport[2];
  const uint64_t y_end = uint64_t(viewport[1]) + viewport[3];
  TI_ERROR_IF(viewport[2] == 0 || viewport[3] == 0 || x_end > result.width ||
                  y_end > result.height,
              "Vulkan graphics viewport must be a nonempty rectangle inside "
              "the color attachment.");
  TI_ERROR_IF(viewport[2] > limits.maxViewportDimensions[0] ||
                  viewport[3] > limits.maxViewportDimensions[1] ||
                  double(viewport[0]) < limits.viewportBoundsRange[0] ||
                  double(viewport[1]) < limits.viewportBoundsRange[0] ||
                  double(x_end) > limits.viewportBoundsRange[1] ||
                  double(y_end) > limits.viewportBoundsRange[1],
              "Vulkan graphics viewport exceeds the device viewport limits.");
  result.viewport = viewport;
  return result;
}

void validate_vulkan_graphics_pipeline_targets(
    vulkan::VulkanPipeline &pipeline,
    const VulkanGraphicsPassTargets &targets) {
  pipeline.validate_color_attachment_types(targets.color_types);
  for (size_t i = 0; i < targets.colors.size(); ++i) {
    TI_ERROR_IF(
        pipeline.color_attachment_blends(i) && !targets.color_blendable[i],
        "Blending requires a blendable floating-point or normalized color "
        "attachment");
  }
}

void begin_vulkan_graphics_pass(CommandList *commands,
                                const VulkanGraphicsPassTargets &targets) {
  auto *vk = static_cast<vulkan::VulkanCommandList *>(commands);
  vk->set_next_renderpass_color_final_layout(ImageLayout::color_attachment);
  vk->set_next_renderpass_depth_clear_value(targets.clear_depth);
  DeviceAllocation depth = targets.depth;
  auto *depth_ptr = depth == kDeviceNullAllocation ? nullptr : &depth;
  vk->begin_renderpass_typed(
      0, 0, int(targets.width), int(targets.height), targets.colors.size(),
      targets.colors.data(), targets.color_clears.get(),
      targets.clear_colors.data(), depth_ptr, depth_ptr && targets.depth_clear);
  const auto &viewport = targets.viewport;
  commands->set_raster_viewport_and_scissor(int(viewport[0]), int(viewport[1]),
                                            int(viewport[0] + viewport[2]),
                                            int(viewport[1] + viewport[3]));
}

VulkanGraphicsDrawResources prepare_vulkan_graphics_draw_resources(
    GraphicsDevice &device,
    const std::vector<VulkanGraphicsShaderBufferView> &shader_buffers,
    const std::vector<VulkanGraphicsShaderImageView> &shader_images,
    const std::vector<std::pair<uint32_t, DevicePtr>> &vertex_buffers,
    DevicePtr index_buffer,
    bool indexed,
    bool mesh,
    const std::function<void(VulkanGraphicsResourceSets &)> &extra_bindings) {
  VulkanGraphicsDrawResources draw_resources;
  VulkanGraphicsResourceSets resource_sets;
  for (const auto &shader : shader_buffers) {
    auto &resource_set = resource_sets[shader.set_index];
    if (!resource_set) {
      resource_set = device.create_resource_set_unique();
      TI_ERROR_IF(!resource_set,
                  "Vulkan graphics shader resource set creation failed.");
    }
    if (!shader.array_elements.empty()) {
      resource_set->rw_buffer_array(shader.binding, shader.array_elements);
    } else if (shader.bytes != kBufferSizeEntireSize || shader.offset != 0) {
      if (shader.storage)
        resource_set->rw_buffer(shader.binding,
                                shader.allocation.get_ptr(shader.offset),
                                shader.bytes);
      else
        resource_set->buffer(shader.binding,
                             shader.allocation.get_ptr(shader.offset),
                             shader.bytes);
    } else if (shader.storage) {
      resource_set->rw_buffer(shader.binding, shader.allocation);
    } else {
      resource_set->buffer(shader.binding, shader.allocation);
    }
  }
  for (const auto &image : shader_images) {
    auto &resource_set = resource_sets[image.set_index];
    if (!resource_set) {
      resource_set = device.create_resource_set_unique();
      TI_ERROR_IF(!resource_set,
                  "Vulkan graphics shader resource set creation failed.");
    }
    resource_set->image(image.binding, image.allocation, image.sampler);
  }
  if (extra_bindings)
    extra_bindings(resource_sets);
  draw_resources.shader_resource_sets.reserve(resource_sets.size());
  for (auto &[set_index, resource_set] : resource_sets) {
    const RhiResult prepare_result =
        resource_set->prepare_for_replay(/*patch_existing=*/false);
    TI_ERROR_IF(prepare_result != RhiResult::success,
                "Vulkan graphics shader resource set {} preparation failed: "
                "RhiResult({}).",
                set_index, prepare_result);
    draw_resources.shader_resource_sets.emplace_back(set_index,
                                                     std::move(resource_set));
  }
  if (!mesh) {
    draw_resources.raster = device.create_raster_resources_unique();
    TI_ERROR_IF(!draw_resources.raster,
                "Vulkan graphics raster resource creation failed.");
    for (const auto &[binding, allocation] : vertex_buffers) {
      draw_resources.raster->vertex_buffer(allocation, binding);
    }
    if (indexed) {
      draw_resources.raster->index_buffer(index_buffer, 32);
    }
  }
  return draw_resources;
}

void record_vulkan_graphics_indirect_barriers(
    CommandList *commands,
    bool indexed,
    const VulkanGraphicsIndirectPacket &indirect) {
  const BufferTransition indirect_transition{
      BufferBarrierStage::Transfer | BufferBarrierStage::Compute,
      BufferBarrierAccess::TransferWrite | BufferBarrierAccess::ShaderWrite,
      BufferBarrierStage::IndirectCommand,
      BufferBarrierAccess::IndirectCommandRead};
  const std::size_t command_size = indexed
                                       ? sizeof(VkDrawIndexedIndirectCommand)
                                       : sizeof(VkDrawIndirectCommand);
  const std::size_t command_bytes =
      (static_cast<std::size_t>(indirect.max_draw_count) - 1) *
          indirect.stride +
      command_size;
  commands->buffer_transition(
      indirect.command_buffer.get_ptr(indirect.command_offset), command_bytes,
      indirect_transition);
  if (indirect.count_buffer != kDeviceNullAllocation) {
    commands->buffer_transition(
        indirect.count_buffer.get_ptr(indirect.count_offset),
        sizeof(std::uint32_t), indirect_transition);
  }
}

void record_vulkan_graphics_draw(
    CommandList *commands,
    Pipeline *pipeline,
    bool task_shader,
    const VulkanGraphicsDrawInfo &draw,
    const std::optional<VulkanGraphicsIndirectPacket> &indirect,
    const std::optional<VulkanGraphicsMeshDrawInfo> &mesh,
    const VulkanGraphicsDrawResources &draw_resources) {
  commands->bind_pipeline(pipeline);
  if (!mesh.has_value()) {
    const RhiResult raster_result =
        commands->bind_raster_resources(draw_resources.raster.get());
    TI_ERROR_IF(raster_result != RhiResult::success,
                "Vulkan graphics raster resource binding failed: "
                "RhiResult({}).",
                raster_result);
  }
  for (const auto &[set_index, resource_set] :
       draw_resources.shader_resource_sets) {
    const RhiResult shader_result =
        commands->bind_shader_resources(resource_set.get(), set_index);
    TI_ERROR_IF(shader_result != RhiResult::success,
                "Vulkan graphics shader resource set {} binding failed: "
                "RhiResult({}).",
                set_index, shader_result);
  }

  if (mesh.has_value()) {
    const auto &mesh_draw = *mesh;
    const RhiResult mesh_result = commands->draw_mesh_tasks(
        mesh_draw.group_count_x, mesh_draw.group_count_y,
        mesh_draw.group_count_z, task_shader);
    TI_ERROR_IF(mesh_result != RhiResult::success,
                "Vulkan mesh draw failed: RhiResult({}).", mesh_result);
  } else if (indirect.has_value()) {
    const auto &indirect_draw = *indirect;
    const auto command_ptr =
        indirect_draw.command_buffer.get_ptr(indirect_draw.command_offset);
    RhiResult indirect_result = RhiResult::not_supported;
    if (indirect_draw.count_buffer != kDeviceNullAllocation) {
      const auto count_ptr =
          indirect_draw.count_buffer.get_ptr(indirect_draw.count_offset);
      indirect_result =
          draw.indexed
              ? commands->draw_indexed_indirect_count(
                    command_ptr, count_ptr, indirect_draw.max_draw_count,
                    indirect_draw.stride)
              : commands->draw_indirect_count(command_ptr, count_ptr,
                                              indirect_draw.max_draw_count,
                                              indirect_draw.stride);
    } else {
      indirect_result =
          draw.indexed ? commands->draw_indexed_indirect(
                             command_ptr, indirect_draw.max_draw_count,
                             indirect_draw.stride)
                       : commands->draw_indirect(command_ptr,
                                                 indirect_draw.max_draw_count,
                                                 indirect_draw.stride);
    }
    TI_ERROR_IF(indirect_result != RhiResult::success,
                "Vulkan graphics indirect draw failed: RhiResult({}).",
                indirect_result);
  } else if (draw.indexed &&
             (draw.instance_count > 1 || draw.first_instance != 0)) {
    commands->draw_indexed_instance(draw.element_count, draw.instance_count,
                                    draw.vertex_offset, draw.first_index,
                                    draw.first_instance);
  } else if (draw.indexed) {
    commands->draw_indexed(draw.element_count, draw.vertex_offset,
                           draw.first_index);
  } else if (draw.instance_count > 1 || draw.first_instance != 0) {
    commands->draw_instance(draw.element_count, draw.instance_count,
                            draw.first_vertex, draw.first_instance);
  } else {
    commands->draw(draw.element_count, draw.first_vertex);
  }
}
}  // namespace taichi::lang
#endif
