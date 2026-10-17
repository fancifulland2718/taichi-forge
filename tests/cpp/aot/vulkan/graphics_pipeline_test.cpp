#include "gtest/gtest.h"
#include "taichi/runtime/vulkan_graphics_pipeline.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"

using namespace taichi::lang;
using namespace taichi::lang::vulkan;

namespace {
const uint32_t vertex[] =
#include "cpp_examples/rhi_examples/shaders/2_triangle.vert.spv.h"
    ;
const uint32_t fragment[] =
#include "cpp_examples/rhi_examples/shaders/2_triangle.frag.spv.h"
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
