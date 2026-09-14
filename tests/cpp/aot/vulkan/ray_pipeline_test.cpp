#include "gtest/gtest.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/rhi/vulkan/vulkan_ray_pipeline.h"
#include "taichi/program/program.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/storage_view.h"

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

// Rebuild checked-in fixtures with an explicitly supplied glslc:
// glslc --target-env=vulkan1.2 -O -mfmt=c -o <source>.spv.h <source>
// These are test artifacts only; no compiler or fixture enters the wheel.
const std::uint32_t kMappingRaygen[] =
#include "tests/cpp/aot/vulkan/shaders/instance_mapping.rgen.spv.h"
    ;
const std::uint32_t kMappingMiss[] =
#include "tests/cpp/aot/vulkan/shaders/instance_mapping.rmiss.spv.h"
    ;
const std::uint32_t kMappingClosestHit[] =
#include "tests/cpp/aot/vulkan/shaders/instance_mapping.rchit.spv.h"
    ;
const std::uint32_t kMappingAnyHit[] =
#include "tests/cpp/aot/vulkan/shaders/instance_mapping.rahit.spv.h"
    ;
const std::uint32_t kMappingConsumer[] =
#include "tests/cpp/aot/vulkan/shaders/instance_mapping_consumer.comp.spv.h"
    ;

class MappingLease final : public PreparedResourceLease {
 public:
  void clear() noexcept override {
    closed = true;
  }
  bool closed{false};
};

VulkanShaderBindingRecord marker_record(std::uint32_t group,
                                        std::uint32_t marker) {
  VulkanShaderBindingRecord record{group, std::vector<std::uint8_t>(4)};
  std::memcpy(record.data.data(), &marker, 4);
  return record;
}

SpirvCodeView shader_view(const std::uint32_t *words,
                          std::size_t bytes,
                          VkShaderStageFlagBits stage) {
  SpirvCodeView result;
  result.data = words;
  result.size = bytes;
  result.stage = stage;
  return result;
}

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
  EXPECT_FALSE(sbt->initialization_recorded());
  sbt->record_initialization(*vk_commands);
  EXPECT_TRUE(sbt->initialization_recorded());
  EXPECT_THROW(sbt->record_initialization(*vk_commands), std::logic_error);
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

TEST(VulkanRayPipeline, TraversesSharedInstancesAndOrdersRefitAndConsumer) {
  if (!is_vulkan_api_available())
    GTEST_SKIP() << "Vulkan unavailable";
  Program program(taichi::Arch::vulkan);
  program.materialize_runtime();
  auto *device = static_cast<VulkanDevice *>(program.get_compute_device());
  if (!device->vk_caps().ray_tracing_pipeline)
    GTEST_SKIP() << "Vulkan RT pipeline unavailable";

  // A nonopaque front triangle is rejected by any-hit; traversal must continue
  // to the back triangle in both shared-BLAS instances and both ray types.
  auto *vertices = program.create_ndarray(PrimitiveType::f32, {6, 3});
  auto *indices = program.create_ndarray(PrimitiveType::i32, {2, 3});
  auto *first = program.create_ndarray(PrimitiveType::u32, {5, 8});
  auto *second = program.create_ndarray(PrimitiveType::u32, {5, 8});
  const std::array<float, 18> points{0, 0, 0, 1, 0, 0, 0, 1, 0,
                                     0, 0, 1, 1, 0, 1, 0, 1, 1};
  const std::array<std::int32_t, 6> triangles{0, 1, 2, 3, 4, 5};
  std::array<DevicePtr, 2> destinations{
      vertices->get_device_allocation().get_ptr(),
      indices->get_device_allocation().get_ptr()};
  std::array<const void *, 2> sources{points.data(), triangles.data()};
  std::array<std::size_t, 2> sizes{sizeof(points), sizeof(triangles)};
  ASSERT_EQ(
      device->upload_data(destinations.data(), sources.data(), sizes.data(), 2),
      RhiResult::success);
  const auto blas =
      program.create_vulkan_triangle_blas_resource_with_opacity(6, 2, false);
  const auto vertex_storage = storage::describe_ndarray_storage(*vertices);
  const auto index_storage = storage::describe_ndarray_storage(*indices);
  ASSERT_TRUE(vertex_storage);
  ASSERT_TRUE(index_storage);
  const auto geometry = program.prepare_vulkan_ray_geometry(
      blas, true, *vertex_storage.descriptor, &*index_storage.descriptor, 6, 2);
  program.execute_vulkan_ray_geometry(geometry);
  const auto tlas = program.create_vulkan_instance_tlas_resource_with_sbt(
      {blas, blas}, {1, 4});
  std::vector<VulkanRayInstanceInfo> instances(2);
  instances[0].custom_index = 7;
  instances[1].custom_index = 11;
  instances[1].transform[3] = 2;
  program.vulkan_instance_tlas_build(tlas, instances, false);

  VulkanPipeline::Params params{device};
  params.code = {shader_view(kMappingRaygen, sizeof(kMappingRaygen),
                             VK_SHADER_STAGE_RAYGEN_BIT_KHR),
                 shader_view(kMappingMiss, sizeof(kMappingMiss),
                             VK_SHADER_STAGE_MISS_BIT_KHR),
                 shader_view(kMappingClosestHit, sizeof(kMappingClosestHit),
                             VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR),
                 shader_view(kMappingAnyHit, sizeof(kMappingAnyHit),
                             VK_SHADER_STAGE_ANY_HIT_BIT_KHR)};
  VulkanRayTracingPipelineParams ray_params;
  ray_params.groups = {
      {VulkanRayTracingGroup::Kind::raygen, 0},
      {VulkanRayTracingGroup::Kind::miss, 1},
      {VulkanRayTracingGroup::Kind::triangles, VK_SHADER_UNUSED_KHR, 2, 3}};
  VulkanPipeline pipeline(params, ray_params);
  std::vector<VulkanShaderBindingRecord> hits;
  for (unsigned i = 0; i < 6; ++i)
    hits.push_back(marker_record(2, 10 * i + 3));
  VulkanShaderBindingTable sbt(*device, pipeline, {0, {}},
                               {marker_record(1, 200), marker_record(1, 201)},
                               hits);
  VulkanPipeline::Params consumer_params{device};
  consumer_params.code = {shader_view(
      kMappingConsumer, sizeof(kMappingConsumer), VK_SHADER_STAGE_COMPUTE_BIT)};
  VulkanPipeline consumer(consumer_params);
  auto lease = std::make_shared<MappingLease>();
  VulkanResourceSet first_bindings(device), second_bindings(device);
  const auto first_read = program.prepare_vulkan_ray_shader_binding(
      tlas, &first_bindings, 0, lease,
      VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, false);
  const auto second_read = program.prepare_vulkan_ray_shader_binding(
      tlas, &second_bindings, 0, lease,
      VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, false);
  first_bindings.rw_buffer(1, first->get_device_allocation());
  second_bindings.rw_buffer(1, second->get_device_allocation());
  VulkanResourceSet first_consumer(device), second_consumer(device);
  first_consumer.rw_buffer(1, first->get_device_allocation());
  second_consumer.rw_buffer(1, second->get_device_allocation());
  program.enqueue_compute_op_lambda(
      [&](Device *, CommandList *commands) {
        sbt.record_initialization(*static_cast<VulkanCommandList *>(commands));
      },
      {});
  const auto enqueue_trace = [&](VulkanResourceSet &bindings,
                                 VulkanResourceSet &consumer_bindings,
                                 std::function<void(CommandList *)> read) {
    program.enqueue_compute_op_lambda(
        [&, read](Device *, CommandList *commands) {
          auto *vk_commands = static_cast<VulkanCommandList *>(commands);
          record_ray_program_begin(*vk_commands);
          read(commands);
          vk_commands->bind_pipeline(&pipeline);
          ASSERT_EQ(vk_commands->bind_shader_resources(&bindings),
                    RhiResult::success);
          vk_commands->trace_rays(sbt, 5, 1, 1);
          record_ray_program_end(*vk_commands);
          vk_commands->bind_pipeline(&consumer);
          ASSERT_EQ(vk_commands->bind_shader_resources(&consumer_bindings),
                    RhiResult::success);
          ASSERT_EQ(vk_commands->dispatch(1), RhiResult::success);
          vk_commands->memory_barrier();
        },
        {});
  };
  enqueue_trace(first_bindings, first_consumer, first_read);
  instances[0].transform[11] = 1;
  instances[1].transform[11] = 1;
  program.vulkan_instance_tlas_build(tlas, instances, true);
  enqueue_trace(second_bindings, second_consumer, second_read);
  program.synchronize();

  struct Hit {
    std::array<std::uint32_t, 4> ids;
    std::array<float, 4> geometry;
  };
  static_assert(sizeof(Hit) == 32);
  std::array<Hit, 5> first_hits{}, second_hits{};
  std::array<DevicePtr, 2> readback_sources{
      first->get_device_allocation().get_ptr(),
      second->get_device_allocation().get_ptr()};
  std::array<void *, 2> outputs{first_hits.data(), second_hits.data()};
  std::array<std::size_t, 2> output_sizes{sizeof(first_hits),
                                          sizeof(second_hits)};
  ASSERT_EQ(device->readback_data(readback_sources.data(), outputs.data(),
                                  output_sizes.data(), 2),
            RhiResult::success);
  const std::array<std::uint32_t, 4> record_markers{14, 24, 44, 54};
  for (unsigned i = 0; i < 4; ++i) {
    const std::array<std::uint32_t, 4> expected{1, i / 2, i < 2 ? 7u : 11u,
                                                record_markers[i]};
    EXPECT_EQ(first_hits[i].ids, expected);
    EXPECT_EQ(second_hits[i].ids, expected);
    EXPECT_FLOAT_EQ(first_hits[i].geometry[0], 2.0f);
    EXPECT_FLOAT_EQ(second_hits[i].geometry[0], 3.0f);
    EXPECT_FLOAT_EQ(first_hits[i].geometry[1], 0.25f);
    EXPECT_FLOAT_EQ(first_hits[i].geometry[2], 0.25f);
  }
  const std::array<std::uint32_t, 4> miss{UINT32_MAX, UINT32_MAX, UINT32_MAX,
                                          201};
  EXPECT_EQ(first_hits[4].ids, miss);
  EXPECT_EQ(second_hits[4].ids, miss);
  EXPECT_FALSE(lease->closed);
  program.destroy_vulkan_ray_resource(tlas);
  EXPECT_TRUE(lease->closed);
  program.destroy_vulkan_ray_resource(blas);
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
