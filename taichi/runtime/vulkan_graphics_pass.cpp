#include "taichi/runtime/vulkan_graphics_pass.h"

#ifdef TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::lang {
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
