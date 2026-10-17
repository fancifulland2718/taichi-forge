#include "gtest/gtest.h"
#include "taichi/runtime/vulkan_graphics_pipeline.h"
#include "taichi/runtime/vulkan_graphics_pass.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include <algorithm>
#include <array>

using namespace taichi::lang;
using namespace taichi::lang::vulkan;

namespace {
const uint32_t vertex[] =
#include "cpp_examples/rhi_examples/shaders/2_triangle.vert.spv.h"
    ;
const uint32_t fragment[] =
#include "cpp_examples/rhi_examples/shaders/2_triangle.frag.spv.h"
    ;
const uint32_t sampled_vertex[] =
#include "cpp_examples/rhi_examples/shaders/3_triangle.vert.spv.h"
    ;
const uint32_t sampled_fragment[] =
#include "cpp_examples/rhi_examples/shaders/3_triangle.frag.spv.h"
    ;
}  // namespace

TEST(VulkanGraphicsPipeline, SharedConstructionUsesPhysicalInputLimits) {
  if (!is_vulkan_api_available())
    GTEST_SKIP() << "Vulkan unavailable";
  VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  const auto &limits = device.get_vk_physical_device_props().limits;
  const std::vector<PipelineSourceDesc> sources{
      {PipelineSourceType::spirv_binary, vertex, sizeof(vertex),
       PipelineStageType::vertex},
      {PipelineSourceType::spirv_binary, fragment, sizeof(fragment),
       PipelineStageType::fragment}};
  const std::vector<VertexInputAttribute> attributes{
      {0, 0, BufferFormat::rg32f, 0}, {1, 0, BufferFormat::rgb32f, 8}};
  std::vector<VertexInputBinding> bindings{{0, 20, false}};
  auto create = [&]() {
    return prepare_vulkan_graphics_pipeline(device, sources, RasterParams{},
                                            bindings, attributes,
                                            "shared_graphics");
  };
  EXPECT_NE(create(), nullptr);
  // A device with >16 bindings is no longer truncated to the old software cap.
  if (limits.maxVertexInputBindings > 16) {
    for (uint32_t i = 1; i <= 16; ++i)
      bindings.push_back({i, 20, false});
    EXPECT_NE(create(), nullptr);
  }
  bindings = {{limits.maxVertexInputBindings, 20, false}};
  EXPECT_ANY_THROW(create());
  bindings = {{0, size_t(limits.maxVertexInputBindingStride) + 1, false}};
  EXPECT_ANY_THROW(create());
  bindings = {{0, 20, false}};
  auto bad_attributes = attributes;
  bad_attributes[0].location = limits.maxVertexInputAttributes;
  EXPECT_ANY_THROW(prepare_vulkan_graphics_pipeline(
      device, sources, RasterParams{}, bindings, bad_attributes,
      "invalid_location"));
  auto bad_sources = sources;
  bad_sources[0].size = sizeof(uint32_t);
  EXPECT_ANY_THROW(prepare_vulkan_graphics_pipeline(device, bad_sources,
                                                    RasterParams{}, bindings,
                                                    attributes, "short_spirv"));
  bad_sources = sources;
  bad_sources.push_back(sources[0]);
  EXPECT_ANY_THROW(prepare_vulkan_graphics_pipeline(
      device, bad_sources, RasterParams{}, bindings, attributes,
      "duplicate_stage"));
  EXPECT_NE(create(), nullptr);
}

TEST(VulkanGraphicsPipeline, SharedDrawRecordsBufferRangesAndSampledImages) {
  if (!is_vulkan_api_available())
    GTEST_SKIP() << "Vulkan unavailable";
  VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  RasterParams raster;
  raster.depth_test = raster.depth_write = false;
  raster.front_face_cull = raster.back_face_cull = false;
  auto pipeline = prepare_vulkan_graphics_pipeline(
      device,
      {{PipelineSourceType::spirv_binary, sampled_vertex,
        sizeof(sampled_vertex), PipelineStageType::vertex},
       {PipelineSourceType::spirv_binary, sampled_fragment,
        sizeof(sampled_fragment), PipelineStageType::fragment}},
      raster, {{0, 28, false}},
      {{0, 0, BufferFormat::rg32f, 0},
       {1, 0, BufferFormat::rgb32f, 8},
       {2, 0, BufferFormat::rg32f, 20}},
      "shared_sampled_draw");
  const float vertices[] = {-1, -1, 1, 0,  0, 0, 0, 3, -1, 1, 0,
                            0,  0,  0, -1, 3, 1, 0, 0, 0,  0};
  auto [vertex_buffer, vr] = device.allocate_memory_unique(
      {sizeof(vertices) + 16, false, false, false, AllocUsage::Vertex});
  ASSERT_EQ(vr, RhiResult::success);
  const size_t uniform_offset =
      std::max<uint64_t>(16, device.get_vk_physical_device_props()
                                 .limits.minUniformBufferOffsetAlignment);
  auto [uniform_buffer, ur] = device.allocate_memory_unique(
      {uniform_offset + 16, false, false, false, AllocUsage::Uniform});
  ASSERT_EQ(ur, RhiResult::success);
  auto [pixels, pr] = device.allocate_memory_unique(
      {4, false, false, false, AllocUsage::Storage});
  ASSERT_EQ(pr, RhiResult::success);
  const std::array<float, 4> uniform{1, 0, 0, 0};
  const float pixel = 0.5f;
  DevicePtr pointers[] = {vertex_buffer->get_ptr(16),
                          uniform_buffer->get_ptr(uniform_offset),
                          pixels->get_ptr()};
  const void *sources[] = {vertices, uniform.data(), &pixel};
  size_t sizes[] = {sizeof(vertices), sizeof(uniform), sizeof(pixel)};
  ASSERT_EQ(device.upload_data(pointers, sources, sizes, 3),
            RhiResult::success);
  ImageParams image_params;
  image_params.dimension = ImageDimension::d2D;
  image_params.format = BufferFormat::r32f;
  image_params.x = image_params.y = image_params.z = 1;
  image_params.usage = ImageAllocUsage::Sampled;
  auto sampled = device.create_image_unique(image_params);
  image_params.format = BufferFormat::rgba8;
  image_params.x = image_params.y = 8;
  image_params.usage = ImageAllocUsage::Attachment;
  auto target = device.create_image_unique(image_params);
  auto [output, output_result] = device.allocate_memory_unique(
      {8 * 8 * 4, false, false, false, AllocUsage::Storage});
  ASSERT_EQ(output_result, RhiResult::success);
  VulkanGraphicsShaderBufferView uniform_view{0, 3, *uniform_buffer};
  uniform_view.offset = uniform_offset;
  uniform_view.bytes = 16;
  auto resources = prepare_vulkan_graphics_draw_resources(
      device, {uniform_view}, {{0, 5, *sampled, {}}},
      {{0, vertex_buffer->get_ptr(16)}}, {}, false, false);
  auto *stream = device.get_graphics_stream();
  auto [commands, command_result] = stream->new_command_list_unique();
  ASSERT_EQ(command_result, RhiResult::success);
  BufferImageCopyParams copy;
  copy.buffer_row_length = copy.buffer_image_height = 1;
  copy.image_extent = {1, 1, 1};
  commands->buffer_barrier(pixels->get_ptr());
  commands->image_transition(*sampled, ImageLayout::undefined,
                             ImageLayout::transfer_dst);
  commands->buffer_to_image(*sampled, pixels->get_ptr(),
                            ImageLayout::transfer_dst, copy);
  commands->image_transition(*sampled, ImageLayout::transfer_dst,
                             ImageLayout::shader_read);
  commands->image_transition(*target, ImageLayout::undefined,
                             ImageLayout::color_attachment);
  auto *vk = static_cast<VulkanCommandList *>(commands.get());
  vk->set_next_renderpass_color_final_layout(ImageLayout::color_attachment);
  const bool clear = true;
  const VkClearColorValue clear_value{};
  vk->begin_renderpass_typed(0, 0, 8, 8, 1, target.get(), &clear, &clear_value,
                             nullptr, false);
  commands->set_raster_viewport_and_scissor(0, 0, 8, 8);
  VulkanGraphicsDrawInfo draw;
  draw.element_count = 3;
  record_vulkan_graphics_draw(commands.get(), pipeline.get(), false, draw, {},
                              {}, resources);
  commands->end_renderpass();
  commands->image_transition(*target, ImageLayout::color_attachment,
                             ImageLayout::transfer_src);
  copy.buffer_row_length = copy.buffer_image_height = 8;
  copy.image_extent = {8, 8, 1};
  commands->image_to_buffer(output->get_ptr(), *target,
                            ImageLayout::transfer_src, copy);
  commands->buffer_barrier(output->get_ptr());
  stream->submit_synced(commands.get());
  std::array<uint8_t, 8 * 8 * 4> result{};
  auto source = output->get_ptr();
  void *destination = result.data();
  size_t bytes = result.size();
  ASSERT_EQ(device.readback_data(&source, &destination, &bytes),
            RhiResult::success);
  for (size_t i = 0; i < result.size(); i += 4) {
    EXPECT_NEAR(result[i], 128, 1);
    EXPECT_EQ(result[i + 1], 0);
    EXPECT_EQ(result[i + 2], 0);
    EXPECT_EQ(result[i + 3], 255);
  }
}
