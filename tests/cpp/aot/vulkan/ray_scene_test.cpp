#include "gtest/gtest.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/runtime/vulkan_ray_query.h"
#include "taichi/runtime/vulkan_ray_scene.h"
#include "taichi/runtime/vulkan_ray_tlas.h"
#include "taichi/runtime/native_vulkan_ray_access.h"
#include "taichi/runtime/native_ray_admission.h"
#include "taichi/runtime/native_operation_registry.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/runtime/gfx/aot_module_loader_impl.h"
#include "taichi/program/program.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/storage_view.h"
#include "taichi/common/operation_json.h"

#include <array>
#include <cstdlib>
#include <limits>
#include <vector>

namespace taichi::lang {

TEST(VulkanRayScene, ProgramKernelReadersHonorLoansAndRetireRecordings) {
  const auto *folder = std::getenv("TAICHI_VULKAN_RAY_DOMAIN_AOT_DIR");
  if (!folder || !vulkan::is_vulkan_api_available())
    GTEST_SKIP() << "Requires the compiled ray query fixture";
  Program program(Arch::vulkan);
  program.materialize_runtime();
  auto &device = static_cast<vulkan::VulkanDevice &>(*program.get_compute_device());
  if (!device.vk_caps().acceleration_structure || !device.vk_caps().ray_query ||
      !device.vk_caps().timeline_semaphore)
    GTEST_SKIP();
  auto *runtime = program.get_gfx_runtime_for_borrowing();
  auto allocate = [&](DataType dtype, std::vector<int> shape) {
    return program.create_ndarray(dtype, shape, ExternalArrayLayout::kNull, false);
  };
  auto *vertices = allocate(PrimitiveType::f32, {3, 3});
  auto *indices = allocate(PrimitiveType::i32, {1, 3});
  auto *rays = allocate(PrimitiveType::f32, {1, 8});
  auto *hits = allocate(PrimitiveType::f32, {1, 4});
  auto *ids = allocate(PrimitiveType::i32, {1, 4});
  auto upload = [&](Ndarray *array, const auto &values) {
    auto pointer = array->get_device_allocation().get_ptr();
    const void *source = values.data();
    std::size_t bytes = sizeof(values);
    ASSERT_EQ(device.upload_data(&pointer, &source, &bytes), RhiResult::success);
  };
  upload(vertices, std::array<float, 9>{0, 0, 0, 2, 0, 0, 0, 2, 0});
  upload(indices, std::array<int32_t, 3>{0, 1, 2});
  upload(rays, std::array<float, 8>{.5f, .5f, 3, .01f, 0, 0, -1, 10});
  const auto v = storage::describe_ndarray_storage(*vertices);
  const auto ix = storage::describe_ndarray_storage(*indices);
  const auto blas_id = program.create_vulkan_triangle_blas_resource(3, 1);
  program.execute_vulkan_ray_geometry(program.prepare_vulkan_ray_geometry(
      blas_id, true, *v.descriptor, &*ix.descriptor, 3, 1));
  const auto tlas_id = program.create_vulkan_instance_tlas_resource({blas_id});
  VulkanRayInstanceInfo metadata;
  metadata.custom_index = 37;
  program.vulkan_instance_tlas_build(tlas_id, {metadata}, false);
  program.synchronize();

  gfx::AotModuleParams params;
  params.runtime = runtime;
  params.module_path = folder;
  auto module = aot::Module::load(Arch::vulkan, params);
  ASSERT_NE(module, nullptr);
  auto *kernel = dynamic_cast<gfx::KernelImpl *>(module->get_kernel("query_kernel"));
  ASSERT_NE(kernel, nullptr);
  LaunchContextBuilder context(kernel);
  context.set_arg_acceleration_structure({0}, &program, tlas_id);
  context.set_arg_ndarray({1}, *rays);
  context.set_arg_ndarray({2}, *hits);
  context.set_arg_ndarray({3}, *ids);
  program.resolve_ndarray_launch_context(context);
  const std::vector<gfx::GfxRuntime::GraphRecordingOperation> operations{
      {{kernel->handle(), &context}, {}}};
  // Materialize before any loan. Permission must be rechecked on replay, not
  // merely at descriptor binding or command recording time.
  auto fixed = runtime->prepare_fixed_graph(operations, {});
  const auto read = [&] {
    program.synchronize();
    std::array<float, 4> result{};
    auto pointer = hits->get_device_allocation().get_ptr();
    void *destination = result.data();
    std::size_t bytes = sizeof(result);
    EXPECT_EQ(device.readback_data(&pointer, &destination, &bytes), RhiResult::success);
    return result;
  };
  const auto run_both = [&] {
    kernel->launch(context);
    program.mark_runtime_submission_pending();
    EXPECT_EQ(read(), (std::array<float, 4>{3, .25f, .25f, 1}));
    fixed->launch_prepared();
    program.mark_runtime_submission_pending();
    EXPECT_EQ(read(), (std::array<float, 4>{3, .25f, .25f, 1}));
  };
  run_both();
  for (const auto id : {tlas_id, blas_id}) {
    auto root = program.acquire_vulkan_ray_native_resource(id);
    for (bool writes : {true, false}) {
      auto workspace = runtime->create_fixed_graph_workspace();
      runtime->prepare_fixed_shared_ray_resources(
          workspace, collect_native_ray_uses(root, writes), program.runtime_fault_domain());
      for (const auto &dependency : workspace->shared_resource_dependencies())
        dependency.wait();
      if (writes) {
        upload(hits, std::array<float, 4>{-42, -42, -42, -42});
        const auto before = device.queue_submission_snapshot().queue_submit_calls;
        EXPECT_ANY_THROW(kernel->launch(context));
        EXPECT_ANY_THROW(fixed->launch_prepared());
        EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, before);
        // Flush to catch rejected kernels appended without a queue submit.
        EXPECT_EQ(read(), (std::array<float, 4>{-42, -42, -42, -42}));
      } else {
        run_both();
      }
      // This admission-only fixture issued no selected-domain GPU commands.
      // Exact in-flight retirement remains covered by the domain tests.
      workspace->retire_shared_ray_loans({});
      run_both();
    }
  }
  std::weak_ptr<NativeResource> retained =
      program.acquire_vulkan_ray_native_resource(tlas_id);
  program.destroy_vulkan_ray_resource(tlas_id);
  EXPECT_ANY_THROW(fixed->launch_prepared());
  EXPECT_ANY_THROW(kernel->launch(context));
  program.synchronize();
  // An old public recording facade must not retain the AS core after close.
  EXPECT_TRUE(retained.expired());
  program.destroy_vulkan_ray_resource(blas_id);
}

TEST(VulkanRayScene, ProgramOwnersUseTheNativeOperationContracts) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  Program program(Arch::vulkan);
  program.materialize_runtime();
  auto &device = static_cast<vulkan::VulkanDevice &>(*program.get_compute_device());
  if (!device.vk_caps().acceleration_structure || !device.vk_caps().ray_query)
    GTEST_SKIP() << "Acceleration structures and ray queries required";
  auto *runtime = program.get_gfx_runtime_for_borrowing();
  auto allocate = [&](DataType dtype, std::vector<int> shape) {
    return program.create_ndarray(dtype, shape, ExternalArrayLayout::kNull, false);
  };
  auto *vertices = allocate(PrimitiveType::f32, {3, 3});
  auto *indices = allocate(PrimitiveType::i32, {1, 3});
  auto *rays = allocate(PrimitiveType::f32, {1, 8});
  auto *hits = allocate(PrimitiveType::f32, {1, 4});
  auto *ids = allocate(PrimitiveType::i32, {1, 4});
  auto upload = [&](Ndarray *array, const auto &data) {
    auto pointer = array->get_device_allocation().get_ptr();
    const void *source = data.data();
    std::size_t bytes = sizeof(data);
    ASSERT_EQ(device.upload_data(&pointer, &source, &bytes), RhiResult::success);
  };
  auto binding = [](Ndarray *array, DataType dtype, std::vector<uint64_t> shape) {
    std::size_t bytes = 4;
    for (auto extent : shape)
      bytes *= extent;
    return NativeArrayBinding{array->get_device_allocation().get_ptr(), bytes,
                              dtype, std::move(shape), {}};
  };
  const auto v = storage::describe_ndarray_storage(*vertices);
  const auto ix = storage::describe_ndarray_storage(*indices);
  ASSERT_TRUE(v);
  ASSERT_TRUE(ix);
  upload(indices, std::array<int32_t, 3>{0, 1, 2});
  upload(vertices, std::array<float, 9>{0, 0, 0, 2, 0, 0, 0, 2, 0});
  upload(rays, std::array<float, 8>{0.5f, 0.5f, 3, 0.01f, 0, 0, -1, 10});
  const auto scene_id = program.create_vulkan_triangle_ray_scene(
      *v.descriptor, *ix.descriptor, 3, 1);
  const auto blas_id = program.create_vulkan_triangle_blas_resource(3, 1);
  program.execute_vulkan_ray_geometry(program.prepare_vulkan_ray_geometry(
      blas_id, true, *v.descriptor, &*ix.descriptor, 3, 1));
  const auto tlas_id = program.create_vulkan_instance_tlas_resource({blas_id});
  VulkanRayInstanceInfo metadata;
  metadata.custom_index = 37;
  program.vulkan_instance_tlas_build(tlas_id, {metadata}, false);
  program.synchronize();
  {
    auto scene = program.acquire_vulkan_ray_native_resource(scene_id, true);
    auto blas = program.acquire_vulkan_ray_native_resource(blas_id);
    auto tlas = program.acquire_vulkan_ray_native_resource(tlas_id);
    EXPECT_EQ(tlas, program.acquire_vulkan_ray_native_resource(tlas_id));
    EXPECT_TRUE(dynamic_cast<native_vulkan_ray::QueryAccess *>(scene.get()));
    EXPECT_TRUE(dynamic_cast<native_vulkan_ray::GeometryAccess *>(scene.get()));
    EXPECT_FALSE(dynamic_cast<NativeAccelerationStructure *>(scene.get()));
    EXPECT_TRUE(dynamic_cast<native_vulkan_ray::BlasAccess *>(blas.get()));
    EXPECT_FALSE(dynamic_cast<native_vulkan_ray::QueryAccess *>(blas.get()));
    EXPECT_TRUE(dynamic_cast<NativeShaderAccelerationStructure *>(tlas.get()));
    auto uses = collect_native_ray_uses(tlas, true);
    ASSERT_EQ(uses.size(), 2u);
    EXPECT_EQ(uses[0].resource, tlas);
    EXPECT_TRUE(uses[0].writes);
    EXPECT_EQ(uses[1].resource, blas);
    EXPECT_FALSE(uses[1].writes);
    const auto original = dynamic_cast<NativeRayResourceAccess &>(*blas).ray_registration();
    EXPECT_EQ(original, dynamic_cast<NativeRayResourceAccess &>(*uses[1].resource).ray_registration());
    EXPECT_EQ(tlas->workspace_bytes(), 0u);  // The Program owns the AS allocation.

    NativePreparationContext context(Arch::vulkan, device, {}, program.runtime_fault_domain());
    context.gfx_runtime = runtime;
    const auto &registry = native_operation_registry();
    auto prepare = [&](const char *operation, NativeOperationBindings bindings,
                       const char *parameters = "{}") {
      NativeOperationDescription description;
      description.provider = "vulkan_ray";
      description.operation = operation;
      description.version = 1;
      description.parameters = liong::json::parse_complete(parameters);
      for (const auto &[name, value] : bindings)
        description.bindings.emplace(name, name);
      return registry.prepare(context, registry.decode(description), bindings);
    };
    const auto vertex_binding = binding(vertices, PrimitiveType::f32, {3, 3});
    auto scene_refit = prepare("refit", {{"scene", scene}, {"vertices", vertex_binding}});
    auto blas_refit = prepare("refit", {{"scene", blas}, {"vertices", vertex_binding}});
    auto query = [&](const std::shared_ptr<NativeResource> &resource) {
      return prepare("query", {{"scene", resource},
                     {"rays", binding(rays, PrimitiveType::f32, {1, 8})},
                     {"hits", binding(hits, PrimitiveType::f32, {1, 4})},
                     {"hit_indices", binding(ids, PrimitiveType::i32, {1, 4})}},
                     "{\"ray_count\":1}");
    };
    auto scene_query = query(scene);
    auto tlas_query = query(tlas);
    auto run = [&](const std::shared_ptr<NativeOperation> &operation) {
      NativeRecordingState state;
      operation->validate_recording(state);
      runtime->enqueue_compute_op_lambda(
          [operation](Device *device, CommandList *commands) {
            operation->record_vulkan(device, commands);
          }, {});
    };
    auto expect_hit = [&](float distance, bool instance) {
      program.synchronize();
      std::array<float, 4> result{};
      auto pointer = hits->get_device_allocation().get_ptr();
      void *destination = result.data();
      std::size_t bytes = sizeof(result);
      ASSERT_EQ(device.readback_data(&pointer, &destination, &bytes), RhiResult::success);
      EXPECT_FLOAT_EQ(result[0], distance);
      if (distance >= 0) {
        EXPECT_FLOAT_EQ(result[1], 0.25f);
        EXPECT_FLOAT_EQ(result[2], 0.25f);
      } else {
        EXPECT_EQ(result, (std::array<float, 4>{-1, 0, 0, 0}));
      }
      std::array<int32_t, 4> indices{};
      pointer = ids->get_device_allocation().get_ptr();
      destination = indices.data();
      bytes = sizeof(indices);
      ASSERT_EQ(device.readback_data(&pointer, &destination, &bytes), RhiResult::success);
      EXPECT_EQ(indices[0], distance < 0 ? -1 : 0);
      if (distance < 0)
        EXPECT_EQ(indices, (std::array<int32_t, 4>{-1, -1, -1, 0}));
      if (instance && distance >= 0)
        EXPECT_EQ(indices[2], 37);
    };
    for (int step = 0; step < 4; ++step) {
      const float x = step == 2 ? 8 : 0;
      const float z = step == 1 ? 1 : 0;
      upload(vertices, std::array<float, 9>{x, 0, z, x + 2, 0, z, x, 2, z});
      const float distance = step == 2 ? -1 : 3 - z;
      run(scene_refit);
      run(blas_refit);
      program.vulkan_instance_tlas_build(tlas_id, {metadata}, true);
      run(tlas_query);
      expect_hit(distance, true);
      // A native mutation is visible through the original Program API.
      program.vulkan_triangle_ray_query(scene_id, rays, hits, 1, ids);
      expect_hit(distance, false);
      program.execute_vulkan_ray_geometry(program.prepare_vulkan_ray_geometry(
          scene_id, false, *v.descriptor, nullptr, 3, 1));
      run(scene_query);
      expect_hit(distance, false);
    }
    // A TLAS retains the original BLAS owner after its public handle closes.
    program.destroy_vulkan_ray_resource(blas_id);
    EXPECT_ANY_THROW(program.acquire_vulkan_ray_native_resource(blas_id));
    EXPECT_EQ(original, dynamic_cast<NativeRayResourceAccess &>(*uses[1].resource).ray_registration());
    run(tlas_query);
    expect_hit(3, true);
  }
  program.destroy_vulkan_ray_resource(tlas_id);
  program.destroy_vulkan_triangle_ray_scene(scene_id);
}

TEST(VulkanRayShaderRead, SeparatesDomainBuildsFromRuntimeGraphicsSemaphores) {
  using Order = VulkanRayShaderReadOrder;
  const auto both = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
  const auto graphics = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  for (bool default_queues_match : {false, true}) {
    EXPECT_TRUE(vulkan_ray_shader_read_needs_barrier(
        both, graphics, Order::execution_domain, default_queues_match));
    EXPECT_TRUE(vulkan_ray_shader_read_needs_barrier(
        VK_QUEUE_COMPUTE_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        Order::compute, default_queues_match));
    EXPECT_EQ(vulkan_ray_shader_read_needs_barrier(
                  both, graphics, Order::runtime_graphics, default_queues_match),
              default_queues_match);
  }
  // A real graphics-only family can consume the ordinary runtime semaphore,
  // but cannot record an AS-build source stage for a selected domain.
  EXPECT_FALSE(vulkan_ray_shader_read_needs_barrier(
      VK_QUEUE_GRAPHICS_BIT, graphics, Order::runtime_graphics, false));
  EXPECT_THROW(vulkan_ray_shader_read_needs_barrier(
                   VK_QUEUE_GRAPHICS_BIT, graphics, Order::execution_domain, false),
               std::invalid_argument);
  EXPECT_THROW(vulkan_ray_shader_read_needs_barrier(
                   VK_QUEUE_COMPUTE_BIT, graphics, Order::execution_domain, false),
               std::invalid_argument);
  EXPECT_THROW(vulkan_ray_shader_read_needs_barrier(
                   VK_QUEUE_GRAPHICS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                   Order::runtime_graphics, false), std::invalid_argument);
  EXPECT_THROW(vulkan_ray_shader_read_needs_barrier(
                   both, VK_PIPELINE_STAGE_TRANSFER_BIT, Order::compute, true),
               std::invalid_argument);
}

TEST(VulkanRayScene, RejectsPhysicalPrimitiveOverflowBeforeAllocation) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator(vulkan::VulkanDeviceCreator::Params{});
  auto *device = creator.device();
  if (!device->vk_caps().acceleration_structure || !device->vk_caps().ray_query)
    GTEST_SKIP();
  VkPhysicalDeviceAccelerationStructurePropertiesKHR as_properties{};
  as_properties.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
  VkPhysicalDeviceProperties2 properties{};
  properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  properties.pNext = &as_properties;
  device->vk_device().instance_functions().vkGetPhysicalDeviceProperties2(
      device->vk_physical_device(), &properties);
  if (as_properties.maxPrimitiveCount >=
      (std::numeric_limits<std::uint32_t>::max)())
    GTEST_SKIP() << "The uint32 ABI already bounds this device";
  const auto count =
      static_cast<std::size_t>(as_properties.maxPrimitiveCount + 1);
  auto rejected = [&](auto construct) {
    try {
      construct();
      FAIL() << "Physical AS capacity overflow was accepted";
    } catch (const std::string &message) {
      EXPECT_NE(message.find("capacity"), std::string::npos) << message;
    }
  };
  // Only metadata is supplied. Both owners must reject before allocating the
  // geometry buffers or asking the driver to size an unsupported AS.
  rejected([&] { VulkanTriangleBlas blas(device, 3, count); });
  rejected([&] { VulkanTriangleRayScene scene(device, 3, count); });
}

TEST(VulkanRayScene, BuildRefitAndTypedQueryWithoutProgram) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator(vulkan::VulkanDeviceCreator::Params{});
  auto *device = creator.device();
  if (!device->vk_caps().acceleration_structure || !device->vk_caps().ray_query)
    GTEST_SKIP() << "Acceleration structures and ray queries required";

  std::vector<DeviceAllocationUnique> arrays;
  auto allocate = [&](std::size_t bytes) {
    auto [allocation, result] = device->allocate_memory_unique(
        {bytes, false, false, false, AllocUsage::Storage});
    EXPECT_EQ(result, RhiResult::success);
    const auto handle = static_cast<DeviceAllocation>(*allocation);
    arrays.push_back(std::move(allocation));
    return handle;
  };
  auto upload = [&](DeviceAllocation allocation, const auto &values) {
    auto pointer = allocation.get_ptr();
    const void *source = values.data();
    std::size_t bytes = sizeof(values);
    ASSERT_EQ(device->upload_data(&pointer, &source, &bytes),
              RhiResult::success);
  };
  auto download = [&](DeviceAllocation allocation, auto &values) {
    auto pointer = allocation.get_ptr();
    void *destination = values.data();
    std::size_t bytes = sizeof(values);
    ASSERT_EQ(device->readback_data(&pointer, &destination, &bytes),
              RhiResult::success);
  };
  const auto vertices = allocate(9 * sizeof(float));
  const auto indices = allocate(3 * sizeof(std::int32_t));
  const auto rays = allocate(8 * sizeof(float));
  const auto hits = allocate(4 * sizeof(float));
  const auto ids = allocate(4 * sizeof(std::int32_t));
  upload(indices, std::array<std::int32_t, 3>{0, 1, 2});
  upload(rays, std::array<float, 8>{0.5f, 0.5f, 2, 0.01f, 0, 0, -1, 10});

  EXPECT_ANY_THROW(VulkanTriangleRayScene(device, 0, 1));
  VulkanTriangleRayScene scene(device, 3, 1);
  EXPECT_EQ(scene.vertex_count(), 3u);
  EXPECT_EQ(scene.triangle_count(), 1u);
  EXPECT_EQ(scene.memory_statistics().known_allocation_count, 7u);
  const std::array<VulkanRayQueryPacket::Binding, 3> bindings{{
      {rays.get_ptr(), 8 * sizeof(float)},
      {hits.get_ptr(), 4 * sizeof(float)},
      {ids.get_ptr(), 4 * sizeof(std::int32_t)},
  }};
  const auto packet =
      prepare_vulkan_ray_query_packet(device, 1, true, bindings);
  scene.prepare_query_variant(packet.variant);
  auto *stream = device->get_compute_stream();
  for (int step = 0; step < 4; ++step) {
    const float x = step == 2 ? 8.0f : 0.0f;
    const float z = step == 1 ? 1.0f : 0.0f;
    upload(vertices, std::array<float, 9>{x, 0, z, x + 2, 0, z, x, 2, z});
    auto [commands, result] = stream->new_command_list_unique();
    ASSERT_EQ(result, RhiResult::success);
    if (step == 0)
      scene.record_build(commands.get(), vertices.get_ptr(), indices.get_ptr());
    else
      scene.record_refit(commands.get(), vertices.get_ptr());
    scene.record_query(commands.get(), packet);
    stream->submit_synced(commands.get());
    std::array<float, 4> values;
    std::array<std::int32_t, 4> hit_ids;
    download(hits, values);
    download(ids, hit_ids);
    if (step == 2) {
      EXPECT_EQ(values, (std::array<float, 4>{-1, 0, 0, 0}));
      EXPECT_EQ(hit_ids, (std::array<std::int32_t, 4>{-1, -1, -1, 0}));
    } else {
      EXPECT_EQ(values, (std::array<float, 4>{2 - z, 0.25f, 0.25f, 0}));
      EXPECT_EQ(hit_ids, (std::array<std::int32_t, 4>{0, 0, 0, 1}));
    }
  }
}

TEST(VulkanRayScene, SharedBlasAndDeviceTransformsWithoutProgram) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator(vulkan::VulkanDeviceCreator::Params{});
  auto *device = creator.device();
  if (!device->vk_caps().acceleration_structure || !device->vk_caps().ray_query)
    GTEST_SKIP();
  std::vector<DeviceAllocationUnique> arrays;
  auto allocate = [&](size_t bytes) {
    auto [allocation, result] = device->allocate_memory_unique(
        {bytes, false, false, false, AllocUsage::Storage});
    EXPECT_EQ(result, RhiResult::success);
    const auto handle = static_cast<DeviceAllocation>(*allocation);
    arrays.push_back(std::move(allocation));
    return handle;
  };
  auto upload = [&](DeviceAllocation allocation, const auto &values) {
    auto pointer = allocation.get_ptr();
    const void *source = values.data();
    size_t bytes = sizeof(values);
    ASSERT_EQ(device->upload_data(&pointer, &source, &bytes),
              RhiResult::success);
  };
  auto download = [&](DeviceAllocation allocation, auto &values) {
    auto pointer = allocation.get_ptr();
    void *target = values.data();
    size_t bytes = sizeof(values);
    ASSERT_EQ(device->readback_data(&pointer, &target, &bytes),
              RhiResult::success);
  };
  const auto vertices = allocate(9 * 4);
  const auto indices = allocate(3 * 4);
  const auto matrices = allocate(25 * 4);
  const auto rays = allocate(16 * 4);
  const auto hits = allocate(8 * 4);
  const auto ids = allocate(8 * 4);
  upload(indices, std::array<int32_t, 3>{0, 1, 2});
  upload(rays, std::array<float, 16>{0, 0, 0, .01f, 0, 0, 1, 3.5f, 4, 0, 0,
                                     .01f, 0, 0, 1, 3.5f});
  auto blas = std::make_shared<VulkanTriangleBlas>(device, 3, 1);
  std::weak_ptr<VulkanTriangleBlas> weak = blas;
  EXPECT_ANY_THROW((VulkanInstanceTlas(device, {nullptr}, {0})));
  auto tlas = std::make_unique<VulkanInstanceTlas>(
      device, std::vector<std::shared_ptr<VulkanTriangleBlas>>{blas, blas},
      std::vector<uint32_t>{3, 0xffffff});
  EXPECT_EQ(tlas->max_sbt_record_offset(), 0xffffffu);
  EXPECT_EQ(tlas->memory_statistics().known_allocation_count, 3u);
  std::vector<VulkanRayInstanceInfo> metadata(2);
  metadata[0].custom_index = 17;
  metadata[1].custom_index = 21;
  const auto transform = prepare_vulkan_ray_transform_packet(
      device, 2, {matrices.get_ptr(4), 24 * 4});
  EXPECT_EQ(transform.parameters[0], 2u);
  EXPECT_ANY_THROW(prepare_vulkan_ray_transform_packet(
      device, 0, {matrices.get_ptr(4), 24 * 4}));
  EXPECT_ANY_THROW(prepare_vulkan_ray_transform_packet(
      device, 2, {matrices.get_ptr(2), 24 * 4}));
  const auto query =
      prepare_vulkan_ray_query_packet(device, 2, true,
                                      {{{rays.get_ptr(), 16 * 4},
                                        {hits.get_ptr(), 8 * 4},
                                        {ids.get_ptr(), 8 * 4}}});
  tlas->prepare_transforms();
  tlas->prepare_query_variant(query.variant);
  auto *stream = device->get_compute_stream();
  for (int step = 0; step < 4; ++step) {
    const float z = step == 1 ? 3.f : 1.f;
    const float x = step == 2 ? 8.f : 0.f;
    upload(vertices, std::array<float, 9>{-1, -1, z, 1, -1, z, 0, 1, z});
    // One leading scalar forces the transform descriptor's subrange variant.
    upload(matrices,
           std::array<float, 25>{99, 1, 0, 0,     x, 0, 1, 0, 0, 0, 0, 1, 0,
                                 1,  0, 0, x + 4, 0, 1, 0, 0, 0, 0, 1, 1});
    auto [commands, result] = stream->new_command_list_unique();
    ASSERT_EQ(result, RhiResult::success);
    weak.lock()->record_build(commands.get(), vertices.get_ptr(),
                              indices.get_ptr(), step != 0);
    if (step == 0)
      tlas->record_build_device(commands.get(), metadata, transform);
    else
      tlas->record_transforms(commands.get(), transform);
    tlas->record_query(commands.get(), query);
    stream->submit_synced(commands.get());
    blas.reset();  // TLAS retains one shared geometry across both ordinals.
    EXPECT_EQ(tlas->max_sbt_record_offset(), 0xffffffu);
    ASSERT_FALSE(weak.expired());
    std::array<float, 8> values;
    std::array<int32_t, 8> hit_ids;
    download(hits, values);
    download(ids, hit_ids);
    for (int i = 0; i < 2; ++i) {
      const bool hit = step != 2 && (step != 1 || i == 0);
      EXPECT_EQ(values[4 * i], hit ? z + i : -1);
      EXPECT_EQ(values[4 * i + 1], hit ? .25f : 0);
      EXPECT_EQ(values[4 * i + 2], hit ? .5f : 0);
      EXPECT_EQ(values[4 * i + 3], 0);
      EXPECT_EQ(hit_ids[4 * i], hit ? 0 : -1);
      EXPECT_EQ(hit_ids[4 * i + 1], hit ? i : -1);
      EXPECT_EQ(hit_ids[4 * i + 2], hit ? (i ? 21 : 17) : -1);
      EXPECT_EQ(hit_ids[4 * i + 3], hit ? 1 : 0);
    }
  }
  tlas.reset();
  EXPECT_TRUE(weak.expired());
}

}  // namespace taichi::lang
