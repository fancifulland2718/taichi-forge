#include "taichi/runtime/vulkan_spd.h"

#ifdef TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::lang {
namespace {
struct SpdResources final : vkapi::DeviceObj {
  std::shared_ptr<void> images;
  DeviceAllocationUnique counter;
  std::unique_ptr<Pipeline> pipeline;
  std::unique_ptr<ShaderResourceSet> bindings;
};
}  // namespace

VulkanSpdCommands prepare_vulkan_spd(vulkan::VulkanDevice &device,
                                     const ImageParams &src_params,
                                     DeviceAllocation src,
                                     const ImageParams &dst_params,
                                     DeviceAllocation dst,
                                     const std::vector<uint32_t> &shader,
                                     std::shared_ptr<void> image_owners,
                                     bool secondary_replay) {
  const auto width = src_params.x, height = src_params.y;
  const auto levels = dst_params.mip_levels;
  TI_ERROR_IF(!image_owners || src.device != &device || dst.device != &device ||
                  src == dst || src_params.dimension != ImageDimension::d2D ||
                  dst_params.dimension != ImageDimension::d2D ||
                  src_params.z != 1 || dst_params.z != 1,
              "SPD requires distinct local 2D images and retained owners.");
  // This is the standalone SPD 2.0 algorithm contract, not an allocation cap.
  TI_ERROR_IF(width < 2 || height < 2 || width > 4096 || height > 4096 ||
                  dst_params.x != width / 2 || dst_params.y != height / 2 ||
                  levels < 1 || levels > 12 || (width >> levels) < 1 ||
                  (height >> levels) < 1,
              "SPD requires half-size output and 2D mip levels within a "
              "2..4096 source.");
  TI_ERROR_IF(!((src_params.format == BufferFormat::r32f &&
                 dst_params.format == BufferFormat::r32f) ||
                ((src_params.format == BufferFormat::rgba8 ||
                  src_params.format == BufferFormat::rgba32f) &&
                 dst_params.format == BufferFormat::rgba32f)),
              "SPD requires R32F to R32F or RGBA8/RGBA32F to RGBA32F.");
  TI_ERROR_IF(shader.size() < 5 || shader.front() != 0x07230203u ||
                  shader.size() > SIZE_MAX / sizeof(uint32_t),
              "SPD requires a complete SPIR-V binary fitting address space.");
  const auto &props = device.get_vk_physical_device_props();
  TI_ERROR_IF(
      props.limits.maxComputeWorkGroupInvocations < 256 ||
          props.limits.maxComputeWorkGroupSize[0] < 256 ||
          props.limits.maxComputeSharedMemorySize < 4100 ||
          props.limits.maxPerStageDescriptorStorageImages < levels + 1 ||
          props.limits.maxDescriptorSetStorageImages < levels + 1,
      "SPD compute or storage-image limits are unavailable.");
  const auto source_image = std::get<0>(device.get_vk_image(src));
  const auto output_image = std::get<0>(device.get_vk_image(dst));
  TI_ERROR_IF(!source_image || !output_image ||
                  source_image->image == output_image->image ||
                  !(source_image->usage & VK_IMAGE_USAGE_STORAGE_BIT) ||
                  !(output_image->usage & VK_IMAGE_USAGE_STORAGE_BIT),
              "SPD requires distinct storage-capable images.");
  auto resources = std::make_shared<SpdResources>();
  resources->device = device.vk_device();
  resources->images = std::move(image_owners);
  auto [counter, alloc_result] = device.allocate_memory_unique(
      {4, false, false, false, AllocUsage::Storage});
  TI_ERROR_IF(alloc_result != RhiResult::success,
              "SPD counter allocation failed.");
  resources->counter = std::move(counter);
  PipelineSourceDesc desc{PipelineSourceType::spirv_binary, shader.data(),
                          shader.size() * sizeof(uint32_t),
                          PipelineStageType::compute};
  auto [pipeline, result] =
      device.create_pipeline_unique(desc, "fidelityfx_spd");
  TI_ERROR_IF(result != RhiResult::success, "SPD pipeline creation failed.");
  resources->pipeline = std::move(pipeline);
  resources->bindings.reset(device.create_resource_set());
  resources->bindings->rw_image(0, src, 0);
  resources->bindings->rw_buffer(1, *resources->counter);
  for (uint32_t mip = 0; mip < levels; ++mip)
    resources->bindings->rw_image(mip + 2, dst, mip);
  const uint32_t groups_x = (width + 63) / 64, groups_y = (height + 63) / 64;
  VulkanSpdCommands prepared;
  prepared.record_inline = [resources, groups_x, groups_y,
                            levels](CommandList *commands) {
    auto *list = static_cast<vulkan::VulkanCommandList *>(commands);
    list->begin_external_compute(resources);
    list->bind_pipeline(resources->pipeline.get());
    TI_ERROR_IF(list->bind_shader_resources(resources->bindings.get()) !=
                    RhiResult::success,
                "SPD inline binding failed.");
    const uint32_t constants[]{levels, groups_x * groups_y};
    list->push_constants(constants, sizeof(constants));
    TI_ERROR_IF(list->dispatch(groups_x, groups_y) != RhiResult::success,
                "SPD inline dispatch failed.");
  };
  if (secondary_replay) {
    auto commands = device.get_compute_stream()->new_secondary_command_list();
    TI_ERROR_IF(!commands, "SPD requires secondary compute recording.");
    auto *list = static_cast<vulkan::VulkanCommandList *>(commands.get());
    list->memory_barrier();
    prepared.record_inline(list);
    list->memory_barrier();
    prepared.replay = list->finalize_secondary(resources);
    TI_ERROR_IF(!prepared.replay, "SPD secondary command finalization failed.");
  }
  prepared.statistics = {{"width", width},
                         {"height", height},
                         {"levels", levels},
                         {"dispatch_count", 1},
                         {"barrier_count", 2},
                         {"workspace_bytes", 4},
                         {"inline_recording_available", 1},
                         {"initial_counter_clear_count", 1},
                         {"per_run_counter_clear_count", 0},
                         {"threadgroups", groups_x * groups_y},
                         {"device_copy_count", 0},
                         {"device_vendor_id", props.vendorID},
                         {"device_id", props.deviceID},
                         {"vulkan_driver_version", props.driverVersion}};
  auto initialize = device.get_compute_stream()->new_secondary_command_list();
  TI_ERROR_IF(!initialize, "SPD counter initialization recording failed.");
  initialize->buffer_fill(resources->counter->get_ptr(0), 4, 0);
  initialize->memory_barrier();
  prepared.initialize =
      static_cast<vulkan::VulkanCommandList *>(initialize.get())
          ->finalize_secondary(resources);
  TI_ERROR_IF(!prepared.initialize,
              "SPD counter initialization finalization failed.");
  return prepared;
}
}  // namespace taichi::lang
#endif
