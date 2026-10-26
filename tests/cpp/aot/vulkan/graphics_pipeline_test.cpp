#include "gtest/gtest.h"
#include "taichi/runtime/vulkan_graphics_pipeline.h"
#include "taichi/runtime/vulkan_graphics_pass.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/rhi/vulkan/vulkan_mesh_shader.h"
#include "taichi/rhi/backend_error.h"
#include "external/SPIRV-Headers/include/spirv/unified1/spirv.hpp"
#include "taichi/program/program.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/texture.h"
#include "taichi/runtime/gfx/runtime.h"
#include <algorithm>
#include <array>
#include <limits>
#include <thread>

using namespace taichi::lang;
using namespace taichi::lang::vulkan;

TEST(VulkanGraphicsPipeline, OrdinaryImageTransferCrossesAttachmentLayouts) {
  if (!is_vulkan_api_available())
    GTEST_SKIP();
  Program program(taichi::Arch::vulkan);
  program.materialize_runtime();
  auto &device = *program.get_compute_device();
  auto *runtime = program.get_gfx_runtime_for_borrowing();
  auto *source = program.create_ndarray(PrimitiveType::f32, {2, 2, 4},
                                       ExternalArrayLayout::kNull, false);
  auto *target = program.create_ndarray(PrimitiveType::f32, {2, 2, 4},
                                       ExternalArrayLayout::kNull, false);
  auto *image = program.create_texture(BufferFormat::rgba32f, {2, 2});
  auto *copy_image = program.create_texture(BufferFormat::rgba32f, {2, 2});
  auto *depth = program.create_texture(BufferFormat::depth32f, {2, 2});
  const std::array<float, 16> expected{1, 2, 3, 4, 5, 6, 7, 8,
                                      9, 10, 11, 12, 13, 14, 15, 16};
  auto pointer = source->get_device_allocation().get_ptr();
  const void *data = expected.data();
  size_t bytes = sizeof(expected);
  ASSERT_EQ(device.upload_data(&pointer, &data, &bytes), RhiResult::success);
  // No intermediate host waits: the ordinary upload, attachment transitions,
  // image copy and readback must carry their device dependencies themselves.
  for (int repeat = 0; repeat < 2; ++repeat) {
    program.vulkan_copy_ndarray_to_texture(image, source, 0, 0, 0,
                                           {0, 0, 0}, {2, 2, 1}, 0, 0, 1);
    runtime->transition_image(image->get_device_allocation(), ImageLayout::color_attachment);
    runtime->transition_image(copy_image->get_device_allocation(), ImageLayout::color_attachment);
    runtime->transition_image(depth->get_device_allocation(), ImageLayout::depth_attachment);
    runtime->transition_image(depth->get_device_allocation(), ImageLayout::depth_attachment_read);
    runtime->transition_image(depth->get_device_allocation(), ImageLayout::shader_read);
    ImageCopyParams copy{};
    copy.width = copy.height = 2;
    runtime->copy_image(copy_image->get_device_allocation(), image->get_device_allocation(), copy);
    program.vulkan_copy_texture_to_ndarray(target, copy_image, 0, 0, 0,
                                           {0, 0, 0}, {2, 2, 1}, 0, 0, 1);
    program.synchronize();
    std::array<float, 16> actual{};
    pointer = target->get_device_allocation().get_ptr();
    void *output = actual.data();
    ASSERT_EQ(device.readback_data(&pointer, &output, &bytes), RhiResult::success);
    EXPECT_EQ(actual, expected);
  }
}

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
// glslc --target-env=vulkan1.2 -O -mfmt=c -o <output> <source>
// Shared: -DCOLOR_BINDING=0; mismatch additionally: -DCOLOR_COUNT=3.
const uint32_t storage_vertex[] =
#include "tests/cpp/aot/vulkan/shaders/graphics_storage.vert.spv.h"
    ;
const uint32_t storage_fragment[] =
#include "tests/cpp/aot/vulkan/shaders/graphics_storage.frag.spv.h"
    ;
const uint32_t shared_storage_fragment[] =
#include "tests/cpp/aot/vulkan/shaders/graphics_storage_shared.frag.spv.h"
    ;
const uint32_t mismatched_storage_fragment[] =
#include "tests/cpp/aot/vulkan/shaders/graphics_storage_mismatch.frag.spv.h"
    ;
const uint32_t write_vertex[] =
#include "tests/cpp/aot/vulkan/shaders/graphics_storage_write.vert.spv.h"
    ;
const uint32_t mixed_fragment[] =
#include "tests/cpp/aot/vulkan/shaders/graphics_storage_mixed.frag.spv.h"
    ;
}  // namespace

namespace {
// Instrument only this test device's dispatch table. Production tables remain
// immutable; no process-global Vulkan loader or GPU allocation policy changes.
struct GraphicsFailureProbe {
  explicit GraphicsFailureProbe(VulkanDevice &device)
      : table(const_cast<VolkDeviceTable &>(device.vk_device().functions())),
        create_module(table.vkCreateShaderModule),
        destroy_module(table.vkDestroyShaderModule),
        create_layout(table.vkCreatePipelineLayout),
        create_graphics(table.vkCreateGraphicsPipelines) {
    active = this;
    table.vkCreateShaderModule = tracked_create_module;
    table.vkDestroyShaderModule = tracked_destroy_module;
    table.vkCreatePipelineLayout = tracked_create_layout;
    table.vkCreateGraphicsPipelines = tracked_create_graphics;
  }
  ~GraphicsFailureProbe() {
    table.vkCreateShaderModule = create_module;
    table.vkDestroyShaderModule = destroy_module;
    table.vkCreatePipelineLayout = create_layout;
    table.vkCreateGraphicsPipelines = create_graphics;
    active = nullptr;
  }
  static VKAPI_ATTR VkResult VKAPI_CALL tracked_create_module(
      VkDevice device, const VkShaderModuleCreateInfo *info,
      const VkAllocationCallbacks *alloc, VkShaderModule *result) {
    auto &probe = *active;
    if (++probe.module_calls == probe.fail_module_call) {
      *result = VK_NULL_HANDLE;
      return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    auto status = probe.create_module(device, info, alloc, result);
    if (status == VK_SUCCESS)
      ++probe.created;
    return status;
  }
  static VKAPI_ATTR void VKAPI_CALL tracked_destroy_module(
      VkDevice device, VkShaderModule module, const VkAllocationCallbacks *alloc) {
    if (module)
      ++active->destroyed;
    active->destroy_module(device, module, alloc);
  }
  static VKAPI_ATTR VkResult VKAPI_CALL tracked_create_layout(
      VkDevice device, const VkPipelineLayoutCreateInfo *info,
      const VkAllocationCallbacks *alloc, VkPipelineLayout *result) {
    if (active->fail_layout) {
      *result = VK_NULL_HANDLE;
      return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    return active->create_layout(device, info, alloc, result);
  }
  static VKAPI_ATTR VkResult VKAPI_CALL tracked_create_graphics(
      VkDevice device, VkPipelineCache cache, uint32_t count,
      const VkGraphicsPipelineCreateInfo *info,
      const VkAllocationCallbacks *alloc, VkPipeline *result) {
    ++active->graphics_calls;
    if (active->fail_graphics) {
      std::fill(result, result + count, VK_NULL_HANDLE);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    return active->create_graphics(device, cache, count, info, alloc, result);
  }
  VolkDeviceTable &table;
  PFN_vkCreateShaderModule create_module;
  PFN_vkDestroyShaderModule destroy_module;
  PFN_vkCreatePipelineLayout create_layout;
  PFN_vkCreateGraphicsPipelines create_graphics;
  int module_calls{0}, created{0}, destroyed{0}, graphics_calls{0};
  int fail_module_call{-1};
  bool fail_layout{false}, fail_graphics{false};
  static thread_local GraphicsFailureProbe *active;
};
thread_local GraphicsFailureProbe *GraphicsFailureProbe::active = nullptr;

std::unique_ptr<Pipeline> failure_probe_pipeline(VulkanDevice &device,
                                                const RasterParams &raster = {}) {
  return device.create_raster_pipeline(
      {{PipelineSourceType::spirv_binary, vertex, sizeof(vertex), PipelineStageType::vertex},
       {PipelineSourceType::spirv_binary, fragment, sizeof(fragment), PipelineStageType::fragment}},
      raster, {{0, 20, false}},
      {{0, 0, BufferFormat::rg32f, 0}, {1, 0, BufferFormat::rgb32f, 8}},
      "failure_probe");
}
}  // namespace

TEST(VulkanGraphicsPipeline, ConstructionFailureReleasesCreatedShaderModules) {
  if (!is_vulkan_api_available())
    GTEST_SKIP();
  VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  GraphicsFailureProbe probe(device);
  for (int attempt = 0; attempt < 3; ++attempt) {
    probe.fail_module_call = probe.module_calls + 2;
    EXPECT_THROW(failure_probe_pipeline(device), std::runtime_error);
    EXPECT_EQ(probe.created, probe.destroyed);
  }
  probe.fail_module_call = -1;
  probe.fail_layout = true;
  EXPECT_THROW(failure_probe_pipeline(device), BackendRuntimeError);
  EXPECT_EQ(probe.created, probe.destroyed);
  probe.fail_layout = false;
  RasterParams invalid;
  invalid.depth.bias_constant = std::numeric_limits<float>::quiet_NaN();
  EXPECT_ANY_THROW(failure_probe_pipeline(device, invalid));
  EXPECT_EQ(probe.created, probe.destroyed);
  auto good = failure_probe_pipeline(device);
  EXPECT_EQ(probe.created - probe.destroyed, 2);
  good.reset();
  EXPECT_EQ(probe.created, probe.destroyed);
}

TEST(VulkanGraphicsPipeline, FailedVariantsAreNotCachedAndRetryUsesDriver) {
  if (!is_vulkan_api_available())
    GTEST_SKIP();
  VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  GraphicsFailureProbe probe(device);
  auto owner = failure_probe_pipeline(device);
  auto &pipeline = *static_cast<VulkanPipeline *>(owner.get());
  VulkanRenderPassDesc desc;
  desc.color_attachments.push_back({VK_FORMAT_R8G8B8A8_UNORM, true});
  auto renderpass = device.get_renderpass(desc);
  for (bool dynamic : {false, true}) {
    if (dynamic && !device.vk_caps().dynamic_rendering)
      continue;
    auto create = [&]() {
      return dynamic ? pipeline.graphics_pipeline_dynamic(desc)
                     : pipeline.graphics_pipeline(desc, renderpass);
    };
    const auto before = probe.graphics_calls;
    probe.fail_graphics = true;
    EXPECT_THROW(create(), BackendRuntimeError);
    probe.fail_graphics = false;
    auto first = create();
    ASSERT_NE(first->pipeline, VK_NULL_HANDLE);
    EXPECT_EQ(probe.graphics_calls, before + 2);
    EXPECT_EQ(create(), first);
    EXPECT_EQ(probe.graphics_calls, before + 2);
  }
  owner.reset();
  EXPECT_EQ(probe.created, probe.destroyed);
}

TEST(VulkanGraphicsPipeline, RenderPassVariantsShareOnlyCompatibleFormats) {
  if (!is_vulkan_api_available())
    GTEST_SKIP();
  VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  GraphicsFailureProbe probe(device);
  auto owner = failure_probe_pipeline(device);
  auto &pipeline = *static_cast<VulkanPipeline *>(owner.get());
  VulkanRenderPassDesc desc;
  desc.color_attachments = {{VK_FORMAT_R8G8B8A8_UNORM, true}};
  auto create = [&]() {
    return pipeline.graphics_pipeline(desc, device.get_renderpass(desc));
  };
  auto original = create();
  desc.color_attachments[0].second = false;
  desc.clear_depth = true;
  desc.color_final_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  EXPECT_EQ(create(), original);
  EXPECT_EQ(probe.graphics_calls, 1);
  desc.color_attachments[0].first = VK_FORMAT_B8G8R8A8_UNORM;
  EXPECT_NE(create(), original);
  desc.depth_attachment = VK_FORMAT_D32_SFLOAT;
  EXPECT_NE(create(), original);
  EXPECT_EQ(probe.graphics_calls, 3);
  // The actual VkPipeline/layout owns its lifetime beyond the template/cache.
  owner.reset();
  EXPECT_EQ(probe.created, probe.destroyed);
  EXPECT_NE(original->pipeline, VK_NULL_HANDLE);
  original.reset();

  // Format order and count stay significant, even without creating a driver
  // pipeline whose shader does not have matching outputs.
  GraphicsPipelineFormatsEqual equal;
  GraphicsPipelineFormatsHasher hash;
  desc.color_attachments = {{VK_FORMAT_R8G8B8A8_UNORM, false},
                            {VK_FORMAT_R32_SFLOAT, false}};
  auto other = desc;
  other.color_attachments[1].second = true;
  EXPECT_TRUE(equal(desc, other));
  EXPECT_EQ(hash(desc), hash(other));
  std::swap(other.color_attachments[0], other.color_attachments[1]);
  EXPECT_FALSE(equal(desc, other));
  other.color_attachments.pop_back();
  EXPECT_FALSE(equal(desc, other));
}

TEST(VulkanGraphicsPipeline, CommandBindingPropagatesFailureAndCanBeRebuilt) {
  if (!is_vulkan_api_available())
    GTEST_SKIP();
  VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  const bool supports_dynamic = device.vk_caps().dynamic_rendering;
  GraphicsFailureProbe probe(device);
  ImageParams image;
  image.dimension = ImageDimension::d2D;
  image.format = BufferFormat::rgba8;
  image.x = image.y = 8;
  image.z = 1;
  image.usage = ImageAllocUsage::Attachment;
  auto target = device.create_image_unique(image);
  auto targets = prepare_vulkan_graphics_pass_targets(
      device, {{*target}}, kDeviceNullAllocation, false, 0);
  auto *stream = device.get_graphics_stream();
  for (bool dynamic : {false, true}) {
    if (dynamic && !supports_dynamic)
      continue;
    device.vk_caps().dynamic_rendering = dynamic;
    auto owner = failure_probe_pipeline(device);
    auto [commands, result] = stream->new_command_list_unique();
    ASSERT_EQ(result, RhiResult::success);
    begin_vulkan_graphics_pass(commands.get(), targets);
    probe.fail_graphics = true;
    EXPECT_THROW(commands->bind_pipeline(owner.get()), BackendRuntimeError);
    commands.reset();  // an unfinished pass must also be safe to discard
    probe.fail_graphics = false;
    std::tie(commands, result) = stream->new_command_list_unique();
    ASSERT_EQ(result, RhiResult::success);
    commands->image_transition(*target, ImageLayout::undefined,
                               ImageLayout::color_attachment);
    begin_vulkan_graphics_pass(commands.get(), targets);
    EXPECT_NO_THROW(commands->bind_pipeline(owner.get()));
    commands->end_renderpass();
    EXPECT_NO_THROW(stream->submit_synced(commands.get()));
  }
  device.vk_caps().dynamic_rendering = supports_dynamic;
}

TEST(VulkanGraphicsPipeline, ConcurrentCompatibleVariantsPublishOnce) {
  if (!is_vulkan_api_available())
    GTEST_SKIP();
  VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  auto owner = failure_probe_pipeline(device);
  auto &pipeline = *static_cast<VulkanPipeline *>(owner.get());
  std::array<vkapi::IVkPipeline, 4> results;
  std::vector<std::thread> threads;
  for (size_t i = 0; i < results.size(); ++i) {
    threads.emplace_back([&, i] {
      VulkanRenderPassDesc desc;
      desc.color_attachments = {{VK_FORMAT_R8G8B8A8_UNORM, bool(i % 2)}};
      desc.color_final_layout = i % 2 ? VK_IMAGE_LAYOUT_GENERAL
                                     : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      results[i] = pipeline.graphics_pipeline(desc, device.get_renderpass(desc));
    });
  }
  for (auto &thread : threads)
    thread.join();
  for (const auto &result : results)
    EXPECT_EQ(result, results[0]);
}

TEST(VulkanGraphicsPipeline, StorageWritesRequireEnabledStageFeatures) {
  if (!is_vulkan_api_available())
    GTEST_SKIP() << "Vulkan unavailable";
  VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  auto &caps = device.vk_caps();
  const auto physical_caps = caps;
  auto create = [&](const uint32_t *vert, size_t vert_bytes,
                    const uint32_t *frag, size_t frag_bytes) {
    return prepare_vulkan_graphics_pipeline(
        device,
        {{PipelineSourceType::spirv_binary, vert, vert_bytes,
          PipelineStageType::vertex},
         {PipelineSourceType::spirv_binary, frag, frag_bytes,
          PipelineStageType::fragment}},
        RasterParams{}, {}, {}, "storage_write_features");
  };
  caps.vertex_pipeline_stores_and_atomics = false;
  caps.fragment_stores_and_atomics = false;
  EXPECT_NO_THROW(create(storage_vertex, sizeof(storage_vertex),
                         shared_storage_fragment,
                         sizeof(shared_storage_fragment)));
  EXPECT_THROW(
      create(write_vertex, sizeof(write_vertex), fragment, sizeof(fragment)),
      std::invalid_argument);
  // One readonly member must not make the whole storage block read-only.
  EXPECT_THROW(create(storage_vertex, sizeof(storage_vertex), mixed_fragment,
                      sizeof(mixed_fragment)),
               std::invalid_argument);
  caps.vertex_pipeline_stores_and_atomics =
      physical_caps.vertex_pipeline_stores_and_atomics;
  if (caps.vertex_pipeline_stores_and_atomics) {
    EXPECT_NO_THROW(
        create(write_vertex, sizeof(write_vertex), fragment, sizeof(fragment)));
    EXPECT_THROW(create(storage_vertex, sizeof(storage_vertex), mixed_fragment,
                        sizeof(mixed_fragment)),
                 std::invalid_argument);
  }
  caps = physical_caps;
  if (caps.fragment_stores_and_atomics)
    EXPECT_NO_THROW(create(storage_vertex, sizeof(storage_vertex),
                           mixed_fragment, sizeof(mixed_fragment)));
}

TEST(VulkanGraphicsPipeline, MeshAdmissionUsesIndependentOverflowSafeLimits) {
  VulkanCapabilities caps;
  caps.max_mesh_work_group_count = {8, 4, 2};
  caps.max_mesh_work_group_total_count = 32;
  caps.max_task_work_group_count = {2, 2, 2};
  caps.max_task_work_group_total_count = 4;
  EXPECT_TRUE(caps.mesh_group_count_valid(8, 4, 1, false));
  EXPECT_FALSE(caps.mesh_group_count_valid(8, 4, 2, false));
  EXPECT_FALSE(caps.mesh_group_count_valid(9, 1, 1, false));
  EXPECT_FALSE(caps.mesh_group_count_valid(0, 1, 1, false));
  EXPECT_TRUE(caps.mesh_group_count_valid(2, 2, 1, true));
  EXPECT_FALSE(caps.mesh_group_count_valid(2, 2, 2, true));
  EXPECT_FALSE(caps.mesh_group_count_valid(4, 1, 1, true));
  EXPECT_TRUE(caps.mesh_group_count_valid(4, 1, 1, false));
  caps.max_mesh_work_group_count = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
  caps.max_mesh_work_group_total_count = UINT32_MAX;
  // The old uint64 product wraps to zero for these dimensions. This is a
  // host arithmetic check only; no oversized dispatch or allocation occurs.
  EXPECT_FALSE(caps.mesh_group_count_valid(1u << 22, 1u << 22, 1u << 20, false));
  EXPECT_TRUE(caps.mesh_group_count_valid(UINT32_MAX, 1, 1, false));
}

TEST(VulkanGraphicsPipeline, MeshShaderDeclarationsRespectDeviceLimits) {
  VulkanCapabilities caps;
  caps.mesh_shader = caps.task_shader = true;
  caps.max_mesh_work_group_size = {8, 4, 2};
  caps.max_mesh_work_group_invocations = 16;
  caps.max_task_work_group_size = {2, 2, 1};
  caps.max_task_work_group_invocations = 2;
  caps.max_mesh_output_vertices = 3;
  caps.max_mesh_output_primitives = 1;
  // Declaration-only fixtures exercise cold admission, not driver validation.
  const auto declarations = [](std::array<uint32_t, 3> size) {
    return std::vector<uint32_t>{spv::MagicNumber, 0x00010600, 0, 10, 0,
        (6u << 16) | spv::OpExecutionMode, 1, spv::ExecutionModeLocalSize,
        size[0], size[1], size[2],
        (4u << 16) | spv::OpExecutionMode, 1, spv::ExecutionModeOutputVertices, 3,
        (4u << 16) | spv::OpExecutionMode, 1, spv::ExecutionModeOutputPrimitivesNV, 1};
  };
  const auto validate = [&](const std::vector<uint32_t> &code, bool task = false) {
    validate_mesh_shader_limits(code.data(), code.size() * 4, 1,
        task ? VK_SHADER_STAGE_TASK_BIT_EXT : VK_SHADER_STAGE_MESH_BIT_EXT, caps);
  };
  EXPECT_NO_THROW(validate(declarations({8, 2, 1})));
  EXPECT_THROW(validate(declarations({9, 1, 1})), std::invalid_argument);
  EXPECT_THROW(validate(declarations({1, 5, 1})), std::invalid_argument);
  EXPECT_THROW(validate(declarations({1, 1, 3})), std::invalid_argument);
  EXPECT_THROW(validate(declarations({8, 4, 1})), std::invalid_argument);
  EXPECT_THROW(validate(declarations({0, 1, 1})), std::invalid_argument);
  EXPECT_NO_THROW(validate(declarations({2, 1, 1}), true));
  EXPECT_THROW(validate(declarations({2, 2, 1}), true), std::invalid_argument);
  auto code = declarations({1, 1, 1});
  caps.max_mesh_output_vertices = 2;
  EXPECT_THROW(validate(code), std::invalid_argument);
  EXPECT_NO_THROW(validate(code, true));  // mesh output limits do not constrain task
  caps.max_mesh_output_vertices = 3;
  caps.max_mesh_output_primitives = 0;
  EXPECT_THROW(validate(code), std::invalid_argument);
  caps.max_mesh_output_primitives = 1;
  code.insert(code.end(), {(6u << 16) | spv::OpExecutionMode, 2,
      spv::ExecutionModeLocalSize, UINT32_MAX, UINT32_MAX, UINT32_MAX});
  EXPECT_NO_THROW(validate(code));  // only the selected entry point
  caps.max_mesh_work_group_size = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
  caps.max_mesh_work_group_invocations = UINT32_MAX;
  EXPECT_THROW(validate(declarations({1u << 22, 1u << 22, 1u << 20})), std::invalid_argument);
  EXPECT_NO_THROW(validate(declarations({UINT32_MAX, 1, 1})));
  code.push_back(0);  // malformed instruction cannot make the scan loop forever
  EXPECT_THROW(validate(code), std::invalid_argument);
}

TEST(VulkanGraphicsPipeline, MeshWorkgroupDefaultsAndBuiltinPrecedence) {
  VulkanCapabilities caps;
  caps.mesh_shader = true;
  caps.max_mesh_work_group_size = {4, 4, 4};
  caps.max_mesh_work_group_invocations = 4;
  const auto validate = [&](const std::vector<uint32_t> &code) {
    validate_mesh_shader_limits(code.data(), code.size() * 4, 1,
        VK_SHADER_STAGE_MESH_BIT_EXT, caps);
  };
  std::vector<uint32_t> code{
      spv::MagicNumber, 0x00010600, 0, 10, 0,
      (6u << 16) | spv::OpExecutionModeId, 1, spv::ExecutionModeLocalSizeId, 4, 5, 5,
      (4u << 16) | spv::OpTypeInt, 2, 32, 0,
      (4u << 16) | spv::OpSpecConstant, 2, 4, 4,
      (4u << 16) | spv::OpConstant, 2, 5, 1};
  EXPECT_NO_THROW(validate(code));
  caps.max_mesh_work_group_invocations = 3;
  EXPECT_THROW(validate(code), std::invalid_argument);
  // Builtin size (1,1,1) shadows the otherwise oversized LocalSizeId (4,1,1).
  code.insert(code.end(), {
      (4u << 16) | spv::OpDecorate, 6, spv::DecorationBuiltIn, spv::BuiltInWorkgroupSize,
      (6u << 16) | spv::OpSpecConstantComposite, 3, 6, 5, 5, 5});
  EXPECT_NO_THROW(validate(code));
  // Unknown constant expressions must not fall back to the shadowed mode.
  code.back() = 9;
  EXPECT_NO_THROW(validate(code));
  caps.mesh_shader = false;
  EXPECT_THROW(validate(code), std::invalid_argument);
}

TEST(VulkanGraphicsPipeline, StorageCapacityUsesMergedLayoutAndActualFlags) {
  if (!is_vulkan_api_available())
    GTEST_SKIP() << "Vulkan unavailable";
  VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  auto &caps = device.vk_caps();
  const auto physical_caps = caps;
  if (caps.max_per_stage_descriptor_storage_buffers < 8 ||
      caps.max_descriptor_set_storage_buffers < 8)
    GTEST_SKIP() << "Fixture requires eight storage descriptors";
  // Only tighten the test device's advertised limits. Rejection must occur
  // before creating a driver layout, without allocating oversized tables.
  caps.descriptor_update_after_bind = false;
  caps.descriptor_storage_buffer_update_after_bind = false;
  caps.max_per_stage_descriptor_storage_buffers = 4;
  caps.max_descriptor_set_storage_buffers = 4;
  caps.max_per_stage_descriptor_update_after_bind_storage_buffers = 0;
  caps.max_descriptor_set_update_after_bind_storage_buffers = 0;
  auto create = [&](const uint32_t *fragment_words, size_t bytes) {
    return prepare_vulkan_graphics_pipeline(
        device,
        {{PipelineSourceType::spirv_binary, storage_vertex,
          sizeof(storage_vertex), PipelineStageType::vertex},
         {PipelineSourceType::spirv_binary, fragment_words, bytes,
          PipelineStageType::fragment}},
        RasterParams{}, {}, {}, "merged_storage_capacity");
  };
  EXPECT_NO_THROW(
      create(shared_storage_fragment, sizeof(shared_storage_fragment)));
  EXPECT_THROW(
      create(mismatched_storage_fragment, sizeof(mismatched_storage_fragment)),
      std::invalid_argument);
  EXPECT_THROW(create(storage_fragment, sizeof(storage_fragment)),
               std::invalid_argument);
  caps.max_per_stage_descriptor_storage_buffers = 8;
  EXPECT_THROW(create(storage_fragment, sizeof(storage_fragment)),
               std::invalid_argument);
  caps.max_per_stage_descriptor_storage_buffers = 4;
  caps.max_descriptor_set_storage_buffers = 8;
  EXPECT_THROW(create(storage_fragment, sizeof(storage_fragment)),
               std::invalid_argument);
  caps.max_per_stage_descriptor_storage_buffers = 8;
  EXPECT_NO_THROW(create(storage_fragment, sizeof(storage_fragment)));
  caps = physical_caps;
}

TEST(VulkanGraphicsPipeline, StorageCapacityHonorsUpdateAfterBindSetFlags) {
  if (!is_vulkan_api_available())
    GTEST_SKIP() << "Vulkan unavailable";
  VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  auto &caps = device.vk_caps();
  const auto physical_caps = caps;
  if (!caps.descriptor_storage_buffer_update_after_bind ||
      caps.max_per_stage_descriptor_storage_buffers < 4 ||
      caps.max_descriptor_set_storage_buffers < 4 ||
      caps.max_per_stage_descriptor_update_after_bind_storage_buffers < 8 ||
      caps.max_descriptor_set_update_after_bind_storage_buffers < 8)
    GTEST_SKIP() << "Fixture requires eight UAB storage descriptors";
  auto create = [&]() {
    return prepare_vulkan_graphics_pipeline(
        device,
        {{PipelineSourceType::spirv_binary, storage_vertex,
          sizeof(storage_vertex), PipelineStageType::vertex},
         {PipelineSourceType::spirv_binary, storage_fragment,
          sizeof(storage_fragment), PipelineStageType::fragment}},
        RasterParams{}, {}, {}, "update_after_bind_storage_capacity");
  };
  caps.descriptor_update_after_bind = false;
  caps.descriptor_storage_buffer_update_after_bind = false;
  caps.max_per_stage_descriptor_storage_buffers = 4;
  caps.max_descriptor_set_storage_buffers = 4;
  caps.max_per_stage_descriptor_update_after_bind_storage_buffers = 8;
  caps.max_descriptor_set_update_after_bind_storage_buffers = 8;
  // A larger UAB limit must not exempt a normal set from its own limit.
  EXPECT_THROW(create(), std::invalid_argument);
  caps.descriptor_storage_buffer_update_after_bind = true;
  EXPECT_NO_THROW(create());
  caps.max_per_stage_descriptor_update_after_bind_storage_buffers = 7;
  EXPECT_THROW(create(), std::invalid_argument);
  caps.max_per_stage_descriptor_update_after_bind_storage_buffers = 8;
  caps.max_descriptor_set_update_after_bind_storage_buffers = 7;
  EXPECT_THROW(create(), std::invalid_argument);
  caps = physical_caps;
}

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
  auto ordinary = create();
  ASSERT_NE(ordinary, nullptr);
  auto *reflected = static_cast<VulkanPipeline *>(ordinary.get());
  EXPECT_NO_THROW(reflected->validate_acceleration_structure_bindings({}));
  EXPECT_ANY_THROW(reflected->validate_acceleration_structure_bindings({{0, 0}}));
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

TEST(VulkanGraphicsPipeline, DescriptorArraysKeepLogicalSubranges) {
  if (!is_vulkan_api_available())
    GTEST_SKIP() << "Vulkan unavailable";
  VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  auto pipeline = prepare_vulkan_graphics_pipeline(
      device,
      {{PipelineSourceType::spirv_binary, storage_vertex,
        sizeof(storage_vertex), PipelineStageType::vertex},
       {PipelineSourceType::spirv_binary, shared_storage_fragment,
        sizeof(shared_storage_fragment), PipelineStageType::fragment}},
      RasterParams{}, {}, {}, "storage_subranges");
  const size_t alignment =
      std::max<size_t>(16, device.get_vk_physical_device_props()
                               .limits.minStorageBufferOffsetAlignment);
  auto [allocation, status] = device.allocate_memory_unique(
      {alignment * 4, false, false, false, AllocUsage::Storage});
  ASSERT_EQ(status, RhiResult::success);
  VulkanGraphicsShaderBufferView binding;
  binding.storage = true;
  for (size_t i = 0; i < 4; ++i)
    binding.array_elements.push_back({allocation->get_ptr(i * alignment), 16});
  auto prepare = [&] {
    return prepare_vulkan_graphics_draw_resources(
        device, *static_cast<VulkanPipeline *>(pipeline.get()), {binding}, {},
        {}, kDeviceNullPtr, false, false);
  };
  EXPECT_NO_THROW(prepare());
  auto &range = binding.array_elements.back();
  range.bytes = alignment + 1;
  EXPECT_ANY_THROW(prepare());
  range.bytes = 0;
  EXPECT_ANY_THROW(prepare());
  range.bytes = 16;
  if (device.get_vk_physical_device_props()
          .limits.minStorageBufferOffsetAlignment > 1) {
    ++range.pointer.offset;
    EXPECT_ANY_THROW(prepare());
    --range.pointer.offset;
  }
  EXPECT_NO_THROW(prepare());
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
      device, *static_cast<VulkanPipeline *>(pipeline.get()), {uniform_view},
      {{0, 5, *sampled, {}}}, {{0, vertex_buffer->get_ptr(16)}}, {}, false,
      false);
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
  auto targets = prepare_vulkan_graphics_pass_targets(
      device, {{*target}}, kDeviceNullAllocation, false, 0);
  validate_vulkan_graphics_pipeline_targets(
      *static_cast<VulkanPipeline *>(pipeline.get()), targets);
  begin_vulkan_graphics_pass(commands.get(), targets);
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

TEST(VulkanGraphicsPipeline, IndirectAliasesUseWritablePhysicalRanges) {
  if (!is_vulkan_api_available())
    GTEST_SKIP() << "Vulkan unavailable";
  VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  if (!device.vk_caps().fragment_stores_and_atomics)
    GTEST_SKIP() << "Fragment writes unavailable";
  auto pipeline = prepare_vulkan_graphics_pipeline(
      device,
      {{PipelineSourceType::spirv_binary, storage_vertex,
        sizeof(storage_vertex), PipelineStageType::vertex},
       {PipelineSourceType::spirv_binary, mixed_fragment,
        sizeof(mixed_fragment), PipelineStageType::fragment}},
      RasterParams{}, {}, {}, "indirect_aliases");
  auto &vk_pipeline = *static_cast<VulkanPipeline *>(pipeline.get());
  auto [bank, result] = device.allocate_memory_unique(
      {64, false, false, false, AllocUsage::Storage | AllocUsage::Indirect});
  ASSERT_EQ(result, RhiResult::success);
  VulkanGraphicsIndirectParameters p;
  p.command_offset = 8;
  p.stride = 16;
  p.vertex_record_limit = 3;
  p.instance_record_limit = 1;
  std::optional<DeviceBufferRange> count;
  if (device.vk_caps().draw_indirect_count)
    count = DeviceBufferRange{bank->get_ptr(48), 4};
  const auto packet = prepare_vulkan_graphics_indirect(
      device, false, p, {bank->get_ptr(8), 24}, count);
  VulkanGraphicsShaderBufferView view;
  view.allocation = *bank;
  view.storage = true;
  view.offset = 16;
  view.bytes = 16;
  auto validate = [&] {
    VulkanGraphicsIndirectAccesses accesses;
    accesses.add_indirect(false, packet);  // Physical commands [16,32).
    accesses.add_shader_buffers(vk_pipeline,
                                {view, view});  // Repeated aliases merge.
    accesses.validate(device);
  };
  view.binding = 0;  // Same command span, but reflected readonly across stages.
  EXPECT_NO_THROW(validate());
  view.binding =
      1;  // The fragment block has both readonly and writable members.
  EXPECT_ANY_THROW(validate());
  view.offset = 0;
  EXPECT_NO_THROW(validate());  // [0,16) abuts the command range.
  view.offset = 32;
  EXPECT_NO_THROW(validate());  // [32,48) lies between commands and count.
  view.offset = 48;
  view.bytes = 4;
  if (count)
    EXPECT_ANY_THROW(validate());
  view.offset = 0;
  view.bytes = kBufferSizeEntireSize;
  EXPECT_ANY_THROW(validate());
  auto [other, status] = device.allocate_memory_unique(
      {64, false, false, false, AllocUsage::Storage});
  ASSERT_EQ(status, RhiResult::success);
  view.allocation = *other;
  EXPECT_NO_THROW(validate());
  view.array_elements = {{other->get_ptr(0), 16}, {bank->get_ptr(16), 16}};
  EXPECT_ANY_THROW(
      validate());  // Array elements preserve their own view offsets.
}

TEST(VulkanGraphicsPipeline,
     SharedIndirectRangesRespectViewsAndDeviceFeatures) {
  if (!is_vulkan_api_available())
    GTEST_SKIP() << "Vulkan unavailable";
  VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  auto [buffer, status] = device.allocate_memory_unique(
      {64, false, false, false, AllocUsage::Indirect});
  ASSERT_EQ(status, RhiResult::success);
  VulkanGraphicsIndirectParameters p;
  p.command_offset = 4;
  p.stride = 16;
  p.vertex_record_limit = 3;
  p.instance_record_limit = 1;
  DeviceBufferRange commands{buffer->get_ptr(8), 20};
  std::optional<DeviceBufferRange> count;
  bool indexed = false;
  auto prepare = [&] {
    return prepare_vulkan_graphics_indirect(device, indexed, p, commands, count);
  };
  auto packet = prepare();
  EXPECT_EQ(packet.command_buffer, *buffer);
  EXPECT_EQ(packet.command_offset, 12);
  EXPECT_EQ(packet.count_buffer, kDeviceNullAllocation);
  commands.bytes = 19;
  EXPECT_ANY_THROW(prepare());  // The allocation is large enough; the view isn't.
  commands.bytes = 20;
  commands.pointer.offset = 9;
  EXPECT_ANY_THROW(prepare());
  commands.pointer.offset = 8;
  p.command_offset = (uint64_t(1) << 32) + 4;
  EXPECT_ANY_THROW(prepare());  // Must not truncate to the original valid offset.
  p.command_offset = 4;
  indexed = true;
  p.index_element_limit = 3;
  EXPECT_ANY_THROW(prepare());  // Indexed command needs a 20-byte stride.
  p.stride = 20;
  EXPECT_ANY_THROW(prepare());  // 20 bytes plus offset exceeds this view.
  commands.bytes = 24;
  EXPECT_NO_THROW(prepare());
  auto [storage, sr] = device.allocate_memory_unique(
      {64, false, false, false, AllocUsage::Storage});
  ASSERT_EQ(sr, RhiResult::success);
  commands.pointer = storage->get_ptr(8);
  EXPECT_ANY_THROW(prepare());
  commands.pointer = buffer->get_ptr(8);

  auto &caps = device.vk_caps();
  const auto original = caps;
  p.first_instance_may_be_nonzero = true;
  caps.draw_indirect_first_instance = false;
  EXPECT_ANY_THROW(prepare());
  p.first_instance_may_be_nonzero = false;
  p.max_draw_count = 2;
  caps.multi_draw_indirect = false;
  EXPECT_ANY_THROW(prepare());
  p.max_draw_count = 1;
  caps.max_draw_indirect_count = 0;
  EXPECT_ANY_THROW(prepare());
  caps = original;
  count = DeviceBufferRange{buffer->get_ptr(40), 8};
  p.count_offset = 4;
  if (caps.draw_indirect_count) {
    packet = prepare();
    EXPECT_EQ(packet.count_buffer, *buffer);
    EXPECT_EQ(packet.count_offset, 44);
    count->bytes = 7;
    EXPECT_ANY_THROW(prepare());
    count->bytes = 8;
  }
  caps.draw_indirect_count = false;
  EXPECT_ANY_THROW(prepare());
  caps = original;
  count.reset();
  EXPECT_ANY_THROW(prepare());  // Offset without a count owner is ambiguous.
}

TEST(VulkanGraphicsPipeline, SharedVertexRangesRespectLogicalSubranges) {
  if (!is_vulkan_api_available())
    GTEST_SKIP() << "Vulkan unavailable";
  VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  auto [vertices, vr] = device.allocate_memory_unique(
      {76, false, false, false, AllocUsage::Vertex});
  auto [indices, ir] = device.allocate_memory_unique(
      {16, false, false, false, AllocUsage::Index});
  ASSERT_EQ(vr, RhiResult::success);
  ASSERT_EQ(ir, RhiResult::success);
  std::vector<VertexInputBinding> layout{{0, 20, false}};
  std::vector<VulkanGraphicsVertexRange> views{
      {0, {vertices->get_ptr(16), 60}}};
  std::optional<VulkanGraphicsBufferRange> index;
  VulkanGraphicsDrawInfo draw;
  draw.element_count = 3;
  std::optional<VulkanGraphicsIndirectPacket> indirect;
  auto validate = [&] {
    validate_vulkan_graphics_vertex_ranges(device, layout, views, index, draw,
                                           indirect);
  };
  EXPECT_NO_THROW(validate());
  views[0].range.bytes = 59;
  EXPECT_ANY_THROW(validate());
  views[0].range.bytes = 61;  // Cannot borrow allocator padding.
  EXPECT_ANY_THROW(validate());
  views[0].range.bytes = 60;
  views.push_back(views.front());
  EXPECT_ANY_THROW(validate());
  views.pop_back();
  views[0].binding = 1;
  EXPECT_ANY_THROW(validate());
  views[0].binding = 0;
  draw.first_vertex = UINT32_MAX;
  EXPECT_ANY_THROW(validate());
  draw.first_vertex = 0;
  draw.indexed = true;
  draw.index_min = 0;
  draw.index_max = 2;
  EXPECT_ANY_THROW(validate());
  index = VulkanGraphicsBufferRange{indices->get_ptr(4), 12};
  EXPECT_NO_THROW(validate());
  index->pointer.offset = 2;
  EXPECT_ANY_THROW(validate());
  index->pointer.offset = 4;
  draw.vertex_offset = -1;
  EXPECT_ANY_THROW(validate());
  draw.vertex_offset = 0;
  draw.first_index = 1;
  EXPECT_ANY_THROW(validate());
  draw.first_index = 0;
  indirect.emplace();
  indirect->vertex_record_limit = 3;
  indirect->instance_record_limit = 1;
  indirect->index_element_limit = 3;
  EXPECT_NO_THROW(validate());
  indirect->vertex_record_limit = 4;
  EXPECT_ANY_THROW(validate());
  indirect.reset();
  layout[0].instance = true;
  draw.instance_count = 3;
  EXPECT_NO_THROW(validate());
  draw.first_instance = 1;
  EXPECT_ANY_THROW(validate());
}

TEST(VulkanGraphicsPipeline,
     SharedDescriptorsRejectMismatchBeforeFinalization) {
  if (!is_vulkan_api_available())
    GTEST_SKIP() << "Vulkan unavailable";
  VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  auto pipeline = prepare_vulkan_graphics_pipeline(
      device,
      {{PipelineSourceType::spirv_binary, sampled_vertex,
        sizeof(sampled_vertex), PipelineStageType::vertex},
       {PipelineSourceType::spirv_binary, sampled_fragment,
        sizeof(sampled_fragment), PipelineStageType::fragment}},
      RasterParams{}, {{0, 28, false}},
      {{0, 0, BufferFormat::rg32f, 0},
       {1, 0, BufferFormat::rgb32f, 8},
       {2, 0, BufferFormat::rg32f, 20}},
      "descriptor_validation");
  auto [buffer, status] = device.allocate_memory_unique(
      {20, false, false, false, AllocUsage::Uniform | AllocUsage::Storage});
  ASSERT_EQ(status, RhiResult::success);
  ImageParams params;
  params.dimension = ImageDimension::d2D;
  params.format = BufferFormat::r32f;
  params.x = params.y = params.z = 1;
  params.usage = ImageAllocUsage::Sampled;
  auto image = device.create_image_unique(params);
  std::vector<VulkanGraphicsShaderBufferView> buffers{{0, 3, *buffer}};
  std::vector<VulkanGraphicsShaderImageView> images{{0, 5, *image, {}}};
  auto prepare = [&] {
    return prepare_vulkan_graphics_draw_resources(
        device, *static_cast<VulkanPipeline *>(pipeline.get()), buffers, images,
        {}, {}, false, false);
  };
  EXPECT_NO_THROW(prepare());
  buffers[0].bytes = 21;
  EXPECT_ANY_THROW(prepare());
  buffers[0].bytes = 0;
  EXPECT_ANY_THROW(prepare());
  buffers[0].bytes = 16;
  buffers[0].offset = 1;
  if (device.get_vk_physical_device_props()
          .limits.minUniformBufferOffsetAlignment > 1)
    EXPECT_ANY_THROW(prepare());
  buffers[0].offset = 20;
  EXPECT_ANY_THROW(prepare());
  buffers[0].offset = 0;
  buffers[0].storage = true;
  EXPECT_ANY_THROW(prepare());
  buffers[0].storage = false;
  buffers[0].set_index = 63;
  EXPECT_ANY_THROW(prepare());
  buffers[0].set_index = 0;
  buffers[0].binding = 4;
  EXPECT_ANY_THROW(prepare());
  buffers[0].binding = 3;
  buffers.push_back(buffers.front());
  EXPECT_ANY_THROW(prepare());
  buffers.pop_back();
  images.clear();
  EXPECT_ANY_THROW(prepare());
  images = {{0, 5, *image, {}}};
  buffers.clear();
  EXPECT_ANY_THROW(prepare());
  buffers = {{0, 3, *buffer}};
  EXPECT_NO_THROW(prepare());
}

TEST(VulkanGraphicsPipeline, SharedAttachmentContractRejectsInvalidViewsCold) {
  if (!is_vulkan_api_available())
    GTEST_SKIP() << "Vulkan unavailable";
  VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  ImageParams params;
  params.dimension = ImageDimension::d2D;
  params.format = BufferFormat::rgba8;
  params.x = params.y = 8;
  params.z = 1;
  params.usage = ImageAllocUsage::Attachment;
  auto color = device.create_image_unique(params);
  params.format = BufferFormat::depth32f;
  auto depth = device.create_image_unique(params);
  auto prepare = [&](std::vector<VulkanGraphicsColorTarget> colors,
                     DeviceAllocation depth_target = kDeviceNullAllocation,
                     std::array<uint32_t, 4> viewport = {},
                     float clear_depth = 0.25f) {
    return prepare_vulkan_graphics_pass_targets(device, colors, depth_target,
                                                true, clear_depth, viewport);
  };
  auto targets = prepare({{*color}}, *depth, {2, 1, 4, 5});
  EXPECT_EQ(targets.width, 8);
  EXPECT_EQ(targets.height, 8);
  EXPECT_EQ(targets.viewport, (std::array<uint32_t, 4>{2, 1, 4, 5}));
  EXPECT_EQ(targets.color_types, std::vector<int>{1});
  EXPECT_TRUE(targets.color_clears[0]);
  EXPECT_FLOAT_EQ(targets.clear_depth, 0.25f);
  EXPECT_EQ(prepare({}, *depth).viewport,
            (std::array<uint32_t, 4>{0, 0, 8, 8}));
  EXPECT_ANY_THROW(prepare({}));
  EXPECT_ANY_THROW(prepare({{kDeviceNullAllocation}}));
  EXPECT_ANY_THROW(prepare({{*color}, {*color}}));
  EXPECT_ANY_THROW(prepare({{*depth}}, *depth));
  EXPECT_ANY_THROW(prepare({}, *color));
  EXPECT_ANY_THROW(prepare({{*color}}, *depth, {7, 1, 2, 1}));
  EXPECT_ANY_THROW(prepare({{*color}}, *depth, {0xffffffffu, 0, 2, 1}));
  EXPECT_ANY_THROW(prepare({{*color}}, *depth, {0, 0, 0, 1}));
  EXPECT_ANY_THROW(
      prepare({{*color}}, *depth, {}, std::numeric_limits<float>::quiet_NaN()));
  EXPECT_ANY_THROW(prepare({{*color}}, *depth, {}, 1.01f));

  params.mip_levels = 2;
  auto depth_mips = device.create_image_unique(params);
  EXPECT_ANY_THROW(prepare({{*color}}, *depth_mips));
  params.format = BufferFormat::rgba8;
  auto color_mips = device.create_image_unique(params);
  EXPECT_ANY_THROW(prepare({{*color_mips}}));
  params.mip_levels = 1;
  params.usage = ImageAllocUsage::Sampled;
  auto sampled = device.create_image_unique(params);
  EXPECT_ANY_THROW(prepare({{*sampled}}));
  params.usage = ImageAllocUsage::Attachment;
  params.y = 1;
  auto different_shape = device.create_image_unique(params);
  EXPECT_ANY_THROW(prepare({{*color}, {*different_shape}}));
  params.dimension = ImageDimension::d1D;
  auto one_dimensional = device.create_image_unique(params);
  EXPECT_ANY_THROW(prepare({{*one_dimensional}}));

  params.dimension = ImageDimension::d2D;
  params.y = 8;
  params.format = BufferFormat::r32u;
  auto unsigned_color = device.create_image_unique(params);
  params.format = BufferFormat::r32i;
  auto signed_color = device.create_image_unique(params);
  targets = prepare({{*color},
                     {*unsigned_color, true, {0xf1234567u}},
                     {*signed_color, false, {0xfedcba98u}}});
  EXPECT_EQ(targets.color_types, (std::vector<int>{1, 3, 2}));
  EXPECT_EQ(targets.clear_colors[1].uint32[0], 0xf1234567u);
  EXPECT_EQ(targets.clear_colors[2].uint32[0], 0xfedcba98u);
  EXPECT_FALSE(targets.color_clears[2]);
  EXPECT_FALSE(targets.color_blendable[1]);
  EXPECT_FALSE(targets.color_blendable[2]);
}
