#include "taichi/runtime/vulkan_graphics_pass.h"

#ifdef TI_WITH_VULKAN
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_set>
#include "taichi/program/texture.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/runtime/native_array_contract.h"

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
  vk->graphics_memory_barrier(true);
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

void end_vulkan_graphics_pass(CommandList *commands) {
  commands->end_renderpass();
  static_cast<vulkan::VulkanCommandList *>(commands)->graphics_memory_barrier(
      false);
}

namespace {
size_t validate_buffer_range(vulkan::VulkanDevice &device,
                             DevicePtr pointer,
                             size_t bytes,
                             AllocUsage usage) {
  TI_ERROR_IF(pointer == kDeviceNullPtr || pointer.device != &device,
              "Vulkan graphics buffer belongs to another device or is null.");
  const size_t capacity = device.get_vkbuffer_size(pointer);
  TI_ERROR_IF(pointer.offset >= capacity,
              "Vulkan graphics buffer offset is outside the buffer.");
  if (bytes == kBufferSizeEntireSize)
    bytes = capacity - pointer.offset;
  TI_ERROR_IF(bytes == 0 || bytes > capacity - pointer.offset,
              "Vulkan graphics buffer range is empty or exceeds the buffer.");
  TI_ERROR_IF(!int(device.allocation_usage(pointer) & usage),
              "Vulkan graphics buffer lacks the required usage.");
  return bytes;
}
}  // namespace

void VulkanGraphicsIndirectAccesses::add_indirect(
    bool indexed,
    const VulkanGraphicsIndirectPacket &packet) {
  const size_t command_size = indexed ? sizeof(VkDrawIndexedIndirectCommand)
                                      : sizeof(VkDrawIndirectCommand);
  const size_t bytes =
      size_t(packet.max_draw_count - 1) * packet.stride + command_size;
  reads.push_back(
      {packet.command_buffer.get_ptr(packet.command_offset), bytes});
  if (packet.count_buffer != kDeviceNullAllocation)
    reads.push_back(
        {packet.count_buffer.get_ptr(packet.count_offset), sizeof(uint32_t)});
}

void VulkanGraphicsIndirectAccesses::add_shader_buffers(
    const vulkan::VulkanPipeline &pipeline,
    const std::vector<VulkanGraphicsShaderBufferView> &buffers) {
  const auto &writable = pipeline.graphics_writable_buffers();
  if (writable.empty())
    return;
  for (const auto &buffer : buffers) {
    if (!buffer.storage ||
        !std::binary_search(
            writable.begin(), writable.end(),
            (uint64_t(buffer.set_index) << 32) | buffer.binding))
      continue;
    if (buffer.array_elements.empty())
      writes.push_back(
          {buffer.allocation.get_ptr(buffer.offset), buffer.bytes});
    else
      writes.insert(writes.end(), buffer.array_elements.begin(),
                    buffer.array_elements.end());
  }
}

void VulkanGraphicsIndirectAccesses::validate(
    vulkan::VulkanDevice &device) const {
  if (reads.empty() || writes.empty())
    return;
  auto resolve = [&](const auto &ranges, AllocUsage usage) {
    std::vector<NativeStorageRange> result;
    result.reserve(ranges.size());
    for (const auto &range : ranges) {
      const auto bytes =
          validate_buffer_range(device, range.pointer, range.bytes, usage);
      result.push_back(
          resolve_native_storage_range(Arch::vulkan, range.pointer, bytes));
    }
    return result;
  };
  const auto command_reads = resolve(reads, AllocUsage::Indirect);
  const auto shader_writes = resolve(writes, AllocUsage::Storage);
  try {
    validate_native_read_write_disjoint(command_reads, shader_writes);
  } catch (const std::invalid_argument &error) {
    TI_ERROR(
        "Vulkan indirect command/count ranges may overlap writable shader "
        "storage in the same pass; use separate passes for dependent draws. "
        "{}",
        error.what());
  }
}

VulkanGraphicsIndirectPacket prepare_vulkan_graphics_indirect(
    vulkan::VulkanDevice &device,
    bool indexed,
    const VulkanGraphicsIndirectParameters &p,
    const DeviceBufferRange &commands,
    const std::optional<DeviceBufferRange> &count) {
  const auto &caps = device.vk_caps();
  TI_ERROR_IF(
      !p.max_draw_count || p.max_draw_count > caps.max_draw_indirect_count,
      "Vulkan indirect graphics draw count exceeds the active device limit.");
  TI_ERROR_IF(
      p.max_draw_count > 1 && !caps.multi_draw_indirect,
      "Vulkan multi-draw indirect is unavailable on the active device.");
  TI_ERROR_IF(
      p.first_instance_may_be_nonzero && !caps.draw_indirect_first_instance,
      "Vulkan indirect firstInstance is unavailable on the active device.");
  TI_ERROR_IF(
      count && !caps.draw_indirect_count,
      "Vulkan indirect-count commands are unavailable on the active device.");
  const size_t command_size = indexed ? sizeof(VkDrawIndexedIndirectCommand)
                                      : sizeof(VkDrawIndirectCommand);
  TI_ERROR_IF(p.stride < command_size || p.stride % sizeof(uint32_t) ||
                  p.command_offset % sizeof(uint32_t) ||
                  p.count_offset % sizeof(uint32_t),
              "Vulkan indirect command offsets and stride violate the "
              "four-byte command ABI.");
  TI_ERROR_IF(!count && p.count_offset,
              "Vulkan indirect count offset requires a count buffer.");
  TI_ERROR_IF(!p.vertex_record_limit || !p.instance_record_limit ||
                  (indexed && !p.index_element_limit) ||
                  (!indexed && p.index_element_limit),
              "Vulkan indirect graphics requires positive declared "
              "vertex/instance limits "
              "and an indexed-only index limit.");
  TI_ERROR_IF(
      size_t(p.max_draw_count - 1) >
          ((std::numeric_limits<size_t>::max)() - command_size) / p.stride,
      "Vulkan indirect command byte range overflows size_t.");
  const size_t required =
      size_t(p.max_draw_count - 1) * p.stride + command_size;
  auto resolve = [&](const DeviceBufferRange &view, uint64_t offset,
                     size_t bytes, const char *role) {
    const auto available = validate_buffer_range(
        device, view.pointer, view.bytes, AllocUsage::Indirect);
    TI_ERROR_IF(offset > available || bytes > available - offset,
                "Vulkan graphics {} buffer is too small for the declared "
                "command range.",
                role);
    DevicePtr pointer = view.pointer;
    // Validated view + relative range implies this sum is within the
    // allocation.
    pointer.offset += offset;
    TI_ERROR_IF(
        pointer.offset % sizeof(uint32_t),
        "Vulkan graphics {} buffer offset violates the four-byte command ABI.",
        role);
    return pointer;
  };
  const auto command_ptr =
      resolve(commands, p.command_offset, required, "indirect command");
  VulkanGraphicsIndirectPacket result;
  static_cast<VulkanGraphicsIndirectParameters &>(result) = p;
  result.command_buffer = command_ptr;
  result.command_offset = command_ptr.offset;
  if (count) {
    const auto count_ptr =
        resolve(*count, p.count_offset, sizeof(uint32_t), "indirect count");
    result.count_buffer = count_ptr;
    result.count_offset = count_ptr.offset;
  }
  return result;
}

void validate_vulkan_graphics_vertex_ranges(
    vulkan::VulkanDevice &device,
    const std::vector<VertexInputBinding> &bindings,
    const std::vector<VulkanGraphicsVertexRange> &vertices,
    const std::optional<VulkanGraphicsBufferRange> &indices,
    const VulkanGraphicsDrawInfo &draw,
    const std::optional<VulkanGraphicsIndirectPacket> &indirect) {
  TI_ERROR_IF(
      draw.indexed != indices.has_value(),
      "Vulkan graphics indexed draw and index-buffer binding must agree.");
  TI_ERROR_IF(!indirect && (!draw.element_count || !draw.instance_count),
              "Vulkan graphics direct draw counts must be positive.");
  std::unordered_map<uint32_t, size_t> supplied;
  for (const auto &vertex : vertices) {
    const auto bytes = validate_buffer_range(
        device, vertex.range.pointer, vertex.range.bytes, AllocUsage::Vertex);
    TI_ERROR_IF(!supplied.emplace(vertex.binding, bytes).second,
                "Vulkan graphics vertex binding {} is duplicated.",
                vertex.binding);
  }
  TI_ERROR_IF(supplied.size() != bindings.size(),
              "Vulkan graphics draw must bind every declared vertex buffer.");
  for (const auto &binding : bindings) {
    const auto found = supplied.find(binding.binding);
    TI_ERROR_IF(found == supplied.end(),
                "Vulkan graphics draw is missing vertex binding {}.",
                binding.binding);
    uint64_t records;
    if (indirect) {
      records = binding.instance ? indirect->instance_record_limit
                                 : indirect->vertex_record_limit;
    } else if (binding.instance) {
      records = uint64_t(draw.first_instance) + draw.instance_count;
    } else if (draw.indexed) {
      const int64_t first = int64_t(draw.index_min) + draw.vertex_offset;
      const int64_t last = int64_t(draw.index_max) + draw.vertex_offset;
      TI_ERROR_IF(first < 0 || last < first,
                  "Vulkan graphics indexed vertex binding {} has an invalid "
                  "declared index range after applying vertex_offset.",
                  binding.binding);
      records = uint64_t(last) + 1;
    } else {
      records = uint64_t(draw.first_vertex) + draw.element_count;
    }
    TI_ERROR_IF(
        !binding.stride || !records || records > found->second / binding.stride,
        "Vulkan graphics vertex binding {} is too small for the declared draw "
        "range.",
        binding.binding);
  }
  if (indices) {
    const auto bytes = validate_buffer_range(device, indices->pointer,
                                             indices->bytes, AllocUsage::Index);
    const uint64_t count =
        indirect ? indirect->index_element_limit
                 : uint64_t(draw.first_index) + draw.element_count;
    TI_ERROR_IF(indices->pointer.offset % sizeof(uint32_t) || !count ||
                    count > bytes / sizeof(uint32_t),
                "Vulkan graphics index buffer is too small or misaligned for "
                "the declared draw range.");
  }
}

VulkanGraphicsDrawResources prepare_vulkan_graphics_draw_resources(
    vulkan::VulkanDevice &device,
    vulkan::VulkanPipeline &pipeline,
    const std::vector<VulkanGraphicsShaderBufferView> &shader_buffers,
    const std::vector<VulkanGraphicsShaderImageView> &shader_images,
    const std::vector<std::pair<uint32_t, DevicePtr>> &vertex_buffers,
    DevicePtr index_buffer,
    bool indexed,
    bool mesh,
    const std::function<void(VulkanGraphicsResourceSets &)> &extra_bindings) {
  VulkanGraphicsDrawResources draw_resources;
  VulkanGraphicsResourceSets resource_sets;
  std::unordered_set<uint64_t> claimed;
  auto claim = [&](uint32_t set, uint32_t binding) {
    TI_ERROR_IF(set > uint32_t((std::numeric_limits<int>::max)()) ||
                    binding > uint32_t((std::numeric_limits<int>::max)()) ||
                    !claimed.insert((uint64_t(set) << 32) | binding).second,
                "Vulkan graphics descriptor binding is duplicated or exceeds "
                "the command ABI.");
  };
  const auto &limits = device.get_vk_physical_device_props().limits;
  auto validate_descriptor_buffer = [&](DeviceAllocation allocation,
                                        size_t offset, size_t bytes,
                                        bool storage) {
    bytes = validate_buffer_range(
        device, allocation.get_ptr(offset), bytes,
        storage ? AllocUsage::Storage : AllocUsage::Uniform);
    const auto alignment = storage ? limits.minStorageBufferOffsetAlignment
                                   : limits.minUniformBufferOffsetAlignment;
    const auto maximum =
        storage ? limits.maxStorageBufferRange : limits.maxUniformBufferRange;
    TI_ERROR_IF(offset % alignment || bytes > maximum,
                "Vulkan graphics descriptor buffer range or alignment exceeds "
                "the device limits.");
  };
  for (const auto &shader : shader_buffers) {
    claim(shader.set_index, shader.binding);
    if (!shader.array_elements.empty()) {
      TI_ERROR_IF(!shader.storage || shader.offset != 0 ||
                      shader.bytes != kBufferSizeEntireSize,
                  "Vulkan graphics buffer arrays require per-element storage "
                  "ranges, without an outer buffer range.");
      for (const auto &range : shader.array_elements) {
        TI_ERROR_IF(range.pointer.offset > (std::numeric_limits<size_t>::max)(),
                    "Vulkan graphics descriptor offset exceeds size_t.");
        validate_descriptor_buffer(range.pointer, size_t(range.pointer.offset),
                                   range.bytes, true);
      }
    } else {
      validate_descriptor_buffer(shader.allocation, shader.offset, shader.bytes,
                                 shader.storage);
    }
    auto &resource_set = resource_sets[shader.set_index];
    if (!resource_set) {
      resource_set = device.create_resource_set_unique();
      TI_ERROR_IF(!resource_set,
                  "Vulkan graphics shader resource set creation failed.");
    }
    if (!shader.array_elements.empty()) {
      resource_set->rw_buffer_array_ranges(shader.binding, shader.array_elements);
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
    claim(image.set_index, image.binding);
    TI_ERROR_IF(
        image.allocation == kDeviceNullAllocation ||
            image.allocation.device != &device,
        "Vulkan graphics sampled image belongs to another device or is null.");
    TI_ERROR_IF(!(std::get<0>(device.get_vk_image(image.allocation))->usage &
                  VK_IMAGE_USAGE_SAMPLED_BIT),
                "Vulkan graphics image lacks sampled usage.");
    pipeline.validate_sampled_texture(image.set_index, image.binding,
                                      image.sampler.compare_op >= 0);
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
  const auto &expected = pipeline.get_resource_set_templates();
  TI_ERROR_IF(
      resource_sets.size() != expected.size(),
      "Vulkan graphics descriptor sets do not match shader reflection.");
  for (const auto &[index, expected_set] : expected) {
    const auto found = resource_sets.find(index);
    TI_ERROR_IF(
        found == resource_sets.end() || !found->second ||
            !vulkan::VulkanResourceSet::SetLayoutCmp{}(
                expected_set,
                *static_cast<vulkan::VulkanResourceSet *>(found->second.get())),
        "Vulkan graphics descriptor type/count/coverage mismatch in set {}.",
        index);
  }
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
