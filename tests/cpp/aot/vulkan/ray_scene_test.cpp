#include "gtest/gtest.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/runtime/vulkan_ray_query.h"
#include "taichi/runtime/vulkan_ray_scene.h"

#include <array>
#include <vector>

namespace taichi::lang {

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

}  // namespace taichi::lang
