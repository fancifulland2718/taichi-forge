#include "gtest/gtest.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/rhi/vulkan/vulkan_ray_pipeline.h"

#include <array>
#include <cstring>
#include <stdexcept>

using namespace taichi::lang;
using namespace taichi::lang::vulkan;

namespace {

void publish_shader_writes(VulkanCommandList *commands) {
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask =
      VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                          VK_ACCESS_SHADER_READ_BIT |
                          VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(commands->vk_command_buffer()->buffer,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0,
                       nullptr, 0, nullptr);
}

const std::uint32_t kRecordRaygen[] =
#include "tests/cpp/aot/vulkan/shaders/sbt_record.rgen.spv.h"
    ;

VulkanPipeline::Params record_pipeline_params(VulkanDevice *device) {
  SpirvCodeView code;
  code.data = kRecordRaygen;
  code.size = sizeof(kRecordRaygen);
  code.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  return {device, {code}, "sbt_record_test", nullptr};
}

VulkanRayTracingPipelineParams record_groups() {
  VulkanRayTracingPipelineParams params;
  params.groups.push_back({VulkanRayTracingGroup::Kind::raygen, 0});
  params.groups.push_back({VulkanRayTracingGroup::Kind::raygen, 0});
  return params;
}

}  // namespace

TEST(VulkanRayPipeline, ExecutesPreparedSbtAndRetainsRecordedResources) {
  if (!is_vulkan_api_available()) {
    GTEST_SKIP() << "Vulkan unavailable";
  }
  VulkanDeviceCreator creator({});
  auto *device = creator.device();
  if (!device->vk_caps().ray_tracing_pipeline) {
    GTEST_SKIP() << "Vulkan RT pipeline unavailable";
  }
  auto pipeline = std::make_unique<VulkanPipeline>(
      record_pipeline_params(device), record_groups());
  VulkanShaderBindingRecord record;
  record.group = 1;  // Nonzero group handle and nonempty shader-record payload.
  std::uint32_t bias = 17;
  record.data.resize(sizeof(bias));
  std::memcpy(record.data.data(), &bias, sizeof(bias));
  auto sbt = std::make_unique<VulkanShaderBindingTable>(
      *device, *pipeline, record, std::vector<VulkanShaderBindingRecord>{},
      std::vector<VulkanShaderBindingRecord>{});
  const auto &props = device->vk_caps().ray_tracing_properties;
  EXPECT_EQ(sbt->raygen().deviceAddress % props.shaderGroupBaseAlignment, 0);
  EXPECT_EQ(sbt->raygen().stride % props.shaderGroupHandleAlignment, 0);
  EXPECT_EQ(sbt->raygen().stride, sbt->raygen().size);
  EXPECT_EQ(sbt->miss().deviceAddress, 0);
  EXPECT_EQ(sbt->hit().deviceAddress, 0);

  std::array<std::uint32_t, 257> values;
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = i;
  auto [storage, alloc_result] = device->allocate_memory_unique(
      {sizeof(values), false, false, false, AllocUsage::Storage});
  ASSERT_EQ(alloc_result, RhiResult::success);
  auto pointer = storage->get_ptr(0);
  const void *upload = values.data();
  std::size_t bytes = sizeof(values);
  ASSERT_EQ(device->upload_data(&pointer, &upload, &bytes, 1),
            RhiResult::success);
  VulkanResourceSet bindings(device);
  bindings.rw_buffer(0, pointer, bytes);
  auto *stream = device->get_compute_stream();
  auto [commands, command_result] = stream->new_command_list_unique();
  ASSERT_EQ(command_result, RhiResult::success);
  auto *vk_commands = static_cast<VulkanCommandList *>(commands.get());
  validate_ray_dispatch(*device, values.size(), 1, 1);
  // Explicit producer/consumer dependency, not part of trace_rays hot logic.
  publish_shader_writes(vk_commands);
  vk_commands->bind_pipeline(pipeline.get());
  ASSERT_EQ(vk_commands->bind_shader_resources(&bindings), RhiResult::success);
  vk_commands->trace_rays(*sbt, values.size(), 1, 1);
  publish_shader_writes(vk_commands);
  vk_commands->trace_rays(*sbt, values.size(), 1, 1);
  publish_shader_writes(vk_commands);
  // Recorded Vk owners keep the allocation, layout and pipeline alive.
  sbt.reset();
  pipeline.reset();
  stream->submit_synced(commands.get());
  void *readback = values.data();
  ASSERT_EQ(device->readback_data(&pointer, &readback, &bytes, 1),
            RhiResult::success);
  for (std::size_t i = 0; i < values.size(); ++i)
    EXPECT_EQ(values[i], 9 * i + 68);
}

TEST(VulkanRayPipeline, RejectsInvalidProgramsAndSbtAtPreparation) {
  if (!is_vulkan_api_available())
    GTEST_SKIP() << "Vulkan unavailable";
  VulkanDeviceCreator creator({});
  auto *device = creator.device();
  if (!device->vk_caps().ray_tracing_pipeline)
    GTEST_SKIP() << "Vulkan RT unavailable";
  auto params = record_pipeline_params(device);
  auto groups = record_groups();
  groups.groups[0].general = 99;
  EXPECT_THROW(VulkanPipeline(params, groups), std::invalid_argument);
  groups = record_groups();
  groups.groups[0].kind = VulkanRayTracingGroup::Kind::miss;
  EXPECT_THROW(VulkanPipeline(params, groups), std::invalid_argument);
  groups = record_groups();
  groups.max_recursion_depth = 0;
  EXPECT_THROW(VulkanPipeline(params, groups), std::invalid_argument);
  params.code[0].entry_point = "not_an_entry";
  EXPECT_THROW(VulkanPipeline(params, record_groups()), std::invalid_argument);
  params = record_pipeline_params(device);
  VulkanPipeline pipeline(params, record_groups());
  EXPECT_THROW((VulkanShaderBindingTable(*device, pipeline, {99, {}}, {}, {})),
               std::invalid_argument);
  EXPECT_THROW(
      (VulkanShaderBindingTable(*device, pipeline, {0, {}}, {{1, {}}}, {})),
      std::invalid_argument);
  VulkanShaderBindingRecord oversized{
      0,
      std::vector<std::uint8_t>(
          device->vk_caps().ray_tracing_properties.maxShaderGroupStride + 1)};
  EXPECT_THROW((VulkanShaderBindingTable(*device, pipeline, oversized, {}, {})),
               std::invalid_argument);
  EXPECT_THROW(validate_ray_dispatch(*device, 0, 1, 1), std::invalid_argument);
  EXPECT_THROW(validate_ray_dispatch(*device, UINT32_MAX, UINT32_MAX, 2),
               std::invalid_argument);
  // Failed preparation must leave the same device able to create a valid SBT.
  EXPECT_NO_THROW(
      (VulkanShaderBindingTable(*device, pipeline, {0, {}}, {}, {})));
}
