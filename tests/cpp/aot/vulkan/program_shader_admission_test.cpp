#include "gtest/gtest.h"
#include "taichi/program/program.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/storage_view.h"
#include "taichi/program/texture.h"
#include "taichi/program/vulkan_ray_program.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/runtime/gfx/graph_recording.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/runtime/native_vulkan_ray_access.h"

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace taichi::lang {
namespace {
const uint32_t kVertex[] =
#include "tests/python/assets/hardware_graphics_depth.vert.spv.h"
    ;
const uint32_t kRaygen[] =
#include "tests/cpp/aot/vulkan/shaders/instance_mapping.rgen.spv.h"
    ;
const uint32_t kMiss[] =
#include "tests/cpp/aot/vulkan/shaders/instance_mapping.rmiss.spv.h"
    ;
const uint32_t kHit[] =
#include "tests/cpp/aot/vulkan/shaders/instance_mapping.rchit.spv.h"
    ;

std::vector<uint32_t> query_fragment() {
  const auto tests = std::filesystem::path(__FILE__).parent_path().parent_path()
                         .parent_path().parent_path();
  std::ifstream input(tests / "python/assets/hardware_graphics_ray_query.frag.spv",
                      std::ios::binary | std::ios::ate);
  if (!input || input.tellg() <= 0 || input.tellg() % 4 != 0)
    throw std::runtime_error("Missing ray-query fragment test artifact");
  std::vector<uint32_t> words(static_cast<size_t>(input.tellg()) / 4);
  input.seekg(0);
  input.read(reinterpret_cast<char *>(words.data()), words.size() * 4);
  if (!input)
    throw std::runtime_error("Incomplete ray-query fragment test artifact");
  return words;
}
}  // namespace

TEST(VulkanProgramShaderAdmission, OrdinaryCachedAndFixedReadersHonorASLoans) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  // Exercise both explicit AS close and Program destruction with live packet,
  // command and fixed-recording facades. No new public importer is needed.
  for (bool explicit_close : {false, true}) {
    auto program = std::make_unique<Program>(Arch::vulkan);
    program->materialize_runtime();
    auto &device = static_cast<vulkan::VulkanDevice &>(*program->get_compute_device());
    if (!device.vk_caps().ray_tracing_pipeline || !device.vk_caps().ray_query ||
        !device.vk_caps().timeline_semaphore ||
        !gfx::FixedGraphRecording::supports_graphics_queue(*program))
      GTEST_SKIP();
    auto *runtime = program->get_gfx_runtime_for_borrowing();
    auto allocate = [&](DataType type, std::vector<int> shape) {
      return program->create_ndarray(type, shape, ExternalArrayLayout::kNull, false);
    };
    const auto upload = [&](Ndarray *array, const auto &value) {
      auto pointer = array->get_device_allocation().get_ptr();
      const void *data = value.data();
      size_t bytes = sizeof(value);
      ASSERT_EQ(device.upload_data(&pointer, &data, &bytes), RhiResult::success);
    };
    auto *vertices = allocate(PrimitiveType::f32, {3, 3});
    auto *indices = allocate(PrimitiveType::i32, {1, 3});
    auto *raster_vertices = allocate(PrimitiveType::f32, {3, 6});
    auto *output = allocate(PrimitiveType::u32, {1, 8});
    auto *pixels = allocate(PrimitiveType::f32, {4, 4, 4});
    upload(vertices, std::array<float, 9>{-1, -1, .5f, 1, -1, .5f, 0, 1, .5f});
    upload(indices, std::array<int32_t, 3>{0, 1, 2});
    upload(raster_vertices, std::array<float, 18>{
        -1, -1, 0, .25f, .25f, 0, 3, -1, 0, .25f, .25f, 0,
        -1, 3, 0, .25f, .25f, 0});
    const auto v = storage::describe_ndarray_storage(*vertices);
    const auto ix = storage::describe_ndarray_storage(*indices);
    const auto out = storage::describe_ndarray_storage(*output);
    const auto blas = program->create_vulkan_triangle_blas_resource(3, 1);
    program->execute_vulkan_ray_geometry(program->prepare_vulkan_ray_geometry(
        blas, true, *v.descriptor, &*ix.descriptor, 3, 1));
    const auto tlas = program->create_vulkan_instance_tlas_resource({blas});
    VulkanRayInstanceInfo instance;
    instance.custom_index = 37;
    program->vulkan_instance_tlas_build(tlas, {instance}, false);

    auto *image = program->create_texture(BufferFormat::rgba32f, {4, 4});
    const auto pipeline = program->create_vulkan_graphics_pipeline(
        {std::begin(kVertex), std::end(kVertex)}, query_fragment(), {{0, 24, false}},
        {{0, 0, BufferFormat::rgb32f, 0}, {1, 0, BufferFormat::rgb32f, 12}},
        0, 0, false, false, false, false, false, "AS admission", {});
    VulkanGraphicsDrawCommand draw;
    draw.pipeline_handle = pipeline;
    draw.vertex_buffers = {{0, raster_vertices}};
    draw.draw.element_count = 3;
    draw.shader_acceleration_structures = {{0, 0, tlas}};
    VulkanGraphicsPassInfo pass;
    pass.retained_replay = true;
    auto graphics = program->prepare_vulkan_graphics_pass({{image}}, nullptr, {draw}, pass);

    using Stage = VulkanRayProgramShader::Stage;
    using Kind = VulkanRayProgramGroup::Kind;
    const auto ray_pipeline = program->create_vulkan_ray_program(
        {{Stage::raygen, {std::begin(kRaygen), std::end(kRaygen)}},
         {Stage::miss, {std::begin(kMiss), std::end(kMiss)}},
         {Stage::closest_hit, {std::begin(kHit), std::end(kHit)}}},
        {{Kind::raygen, 0}, {Kind::miss, 1}, {Kind::triangles, UINT32_MAX, 2}}, 1, false);
    const auto record = [](uint32_t group, uint32_t marker) {
      VulkanRayProgramRecord value{group, std::vector<uint8_t>(4)};
      std::memcpy(value.data.data(), &marker, 4);
      return value;
    };
    VulkanRayProgramLaunch info;
    info.raygen.group = 0;
    info.miss = {record(1, 99), record(1, 99)};
    info.hit = {record(2, 17), record(2, 17)};
    info.buffers = {{0, 1, &*out.descriptor, false, true}};
    info.scenes = {{0, 0, tlas}};
    auto rays = program->prepare_vulkan_ray_launch(ray_pipeline, info);
    program->initialize_vulkan_ray_launch(rays);
    auto graphics_command = program->vulkan_graphics_graph_command(graphics);
    auto ray_command = program->vulkan_ray_program_graph_command(rays);
    ASSERT_EQ(graphics_command->stateful_resources().size(), 2);
    ASSERT_EQ(ray_command->stateful_resources().size(), 2);
    // Invalidation also names pipeline/launch IDs; those must not be treated
    // as acceleration structures by the new access projection.
    ASSERT_GT(ray_command->ray_resource_dependencies().size(), 1);
    auto fixed = program->create_vulkan_graph_recording(
        {{graphics_command}, {ray_command}}, {}, false, {});
    const auto read_outputs = [&] {
      program->vulkan_copy_texture_to_ndarray(pixels, image, 0, 0, 0,
                                              {0, 0, 0}, {4, 4, 1}, 0, 0, 1);
      program->synchronize();
      std::array<uint32_t, 8> values{};
      auto pointer = output->get_device_allocation().get_ptr();
      void *data = values.data();
      size_t bytes = sizeof(values);
      ASSERT_EQ(device.readback_data(&pointer, &data, &bytes), RhiResult::success);
      EXPECT_EQ((std::array<uint32_t, 4>{values[0], values[1], values[2], values[3]}),
                (std::array<uint32_t, 4>{0, 0, 37, 17}));
      std::array<float, 4> geometry;
      std::memcpy(geometry.data(), values.data() + 4, sizeof(geometry));
      EXPECT_EQ(geometry, (std::array<float, 4>{1.5f, .3125f, .625f, 1}));
      std::array<float, 64> colors{};
      pointer = pixels->get_device_allocation().get_ptr();
      data = colors.data();
      bytes = sizeof(colors);
      ASSERT_EQ(device.readback_data(&pointer, &data, &bytes), RhiResult::success);
      for (size_t i = 0; i < colors.size(); i += 4)
        EXPECT_EQ((std::array<float, 4>{colors[i], colors[i+1], colors[i+2], colors[i+3]}),
                  (std::array<float, 4>{1, 0, 37, 1}));
    };
    const auto ordinary = [&] {
      program->execute_vulkan_graphics_pass(graphics);
      program->execute_vulkan_ray_launch(rays);
      read_outputs();
    };
    const auto warm_graphics_cache = [&] {
      const auto before = runtime->debug_graphics_command_replay_stats().at("retained_replay_replays");
      for (int i = 0; i < 3; ++i) {
        program->execute_vulkan_graphics_pass(graphics);
        program->synchronize();
      }
      EXPECT_GT(runtime->debug_graphics_command_replay_stats().at("retained_replay_replays"), before);
    };
    ordinary();
    warm_graphics_cache();
    fixed->run();
    read_outputs();
    if (!explicit_close) {
      for (const auto handle : {tlas, blas}) {
        for (bool writes : {true, false}) {
          // Fixed recording/readback may alter image layouts. Prove that the
          // ordinary graphics cache is populated again before reserving AS.
          warm_graphics_cache();
          auto workspace = runtime->create_fixed_graph_workspace();
          runtime->prepare_fixed_shared_ray_resources(
              workspace, collect_native_ray_uses(program->acquire_vulkan_ray_native_resource(handle), writes),
              program->runtime_fault_domain());
          for (const auto &dependency : workspace->shared_resource_dependencies())
            dependency.wait();
          if (writes) {
            const auto before = device.queue_submission_snapshot().queue_submit_calls;
            EXPECT_ANY_THROW(program->execute_vulkan_graphics_pass(graphics));
            EXPECT_ANY_THROW(program->execute_vulkan_ray_launch(rays));
            EXPECT_ANY_THROW(fixed->run());
            EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, before);
          } else {
            ordinary();
            fixed->run();
            read_outputs();
          }
          // This fixture reserves access, but submits no selected-domain work.
          workspace->retire_shared_ray_loans({});
          ordinary();
          fixed->run();
          read_outputs();
        }
      }
    }
    std::weak_ptr<NativeResource> weak_tlas = program->acquire_vulkan_ray_native_resource(tlas);
    std::weak_ptr<NativeResource> weak_blas = program->acquire_vulkan_ray_native_resource(blas);
    program->destroy_vulkan_ray_resource(blas);
    ordinary();
    fixed->run();
    read_outputs();
    if (explicit_close) {
      program->destroy_vulkan_ray_resource(tlas);
      EXPECT_ANY_THROW(program->execute_vulkan_graphics_pass(graphics));
      EXPECT_ANY_THROW(program->execute_vulkan_ray_launch(rays));
      EXPECT_ANY_THROW(fixed->run());
      EXPECT_TRUE(weak_tlas.expired());
      EXPECT_TRUE(weak_blas.expired());
    }
    program.reset();
    EXPECT_TRUE(weak_tlas.expired());
    EXPECT_TRUE(weak_blas.expired());
    EXPECT_ANY_THROW(graphics_command->stateful_resources());
    EXPECT_ANY_THROW(ray_command->stateful_resources());
    fixed->close();
    // Keep every facade across reset, then release them independently. Their
    // host storage leases must already be gone before the device is destroyed.
    ray_command.reset();
    graphics_command.reset();
    rays.reset();
    graphics.reset();
  }
}
}  // namespace taichi::lang
