#include "gtest/gtest.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/runtime/vulkan_parallel_sort_plan.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <tuple>

namespace taichi::lang {
namespace {
class VulkanParallelSortPlanTest
    : public ::testing::TestWithParam<std::tuple<bool, bool, bool>> {
 protected:
  void SetUp() override {
    if (!vulkan::is_vulkan_api_available())
      GTEST_SKIP() << "Vulkan is unavailable";
    const auto *path = std::getenv("TI_TEST_PARALLEL_SORT_SHADERS");
    if (!path || !*path)
      GTEST_SKIP()
          << "Set TI_TEST_PARALLEL_SORT_SHADERS to compiled provider fixtures";
    directory = path;
    vulkan::VulkanDeviceCreator::Params params;
    params.enable_validation_layer = true;
    creator = std::make_unique<vulkan::VulkanDeviceCreator>(params);
    device = creator->device();
  }
  std::vector<std::vector<uint32_t>> shaders(bool payload, bool fused) {
    std::vector<std::vector<uint32_t>> result;
    for (int i = 0; i < (fused ? 3 : 5); ++i) {
      const auto name = std::to_string(payload) + "_" + std::to_string(fused) +
                        "_" + std::to_string(i) + ".spv";
      std::ifstream input(directory / name, std::ios::binary | std::ios::ate);
      if (!input)
        throw std::runtime_error("Missing Parallel Sort shader fixture");
      auto bytes = static_cast<size_t>(input.tellg());
      if (bytes % 4)
        throw std::runtime_error("Invalid Parallel Sort shader fixture size");
      auto &words = result.emplace_back(bytes / 4);
      input.seekg(0);
      input.read(reinterpret_cast<char *>(words.data()), bytes);
      if (!input)
        throw std::runtime_error("Truncated Parallel Sort shader fixture");
    }
    return result;
  }
  std::shared_ptr<DeviceAllocationGuard> allocate(size_t count) {
    Device::AllocParams params;
    params.size = count * 4;
    params.usage = AllocUsage::Storage;
    auto [allocation, status] = device->allocate_memory_unique(params);
    if (status != RhiResult::success)
      throw std::runtime_error("Test allocation failed");
    return std::shared_ptr<DeviceAllocationGuard>(std::move(allocation));
  }
  void upload(DeviceAllocation allocation,
              const std::vector<uint32_t> &values) {
    auto ptr = allocation.get_ptr();
    const void *host = values.data();
    size_t bytes = values.size() * 4;
    ASSERT_EQ(device->upload_data(&ptr, &host, &bytes), RhiResult::success);
  }
  std::vector<uint32_t> read(DeviceAllocation allocation, size_t count) {
    std::vector<uint32_t> values(count);
    auto ptr = allocation.get_ptr();
    void *host = values.data();
    size_t bytes = count * 4;
    EXPECT_EQ(device->readback_data(&ptr, &host, &bytes), RhiResult::success);
    return values;
  }
  std::filesystem::path directory;
  std::unique_ptr<vulkan::VulkanDeviceCreator> creator;
  vulkan::VulkanDevice *device{};
};

TEST_P(VulkanParallelSortPlanTest, InlineAndRetainedCommandsOwnStorage) {
  const auto [payload, fused, inline_recording] = GetParam();
  // One case crosses the second reduced-prefix block; the rest isolate small
  // exact tails and subviews without repeating a large sorting matrix.
  const uint32_t n = payload && !fused && inline_recording    ? 262147
                     : !payload && fused && !inline_recording ? 1
                                                              : 513;
  const size_t prefix =
      inline_recording
          ? std::max<size_t>(4, device->get_vk_physical_device_props()
                                    .limits.minStorageBufferOffsetAlignment) /
                4
          : 0;
  const auto capacity = prefix + n + 5;
  auto keys = allocate(capacity);
  auto values = payload ? allocate(capacity) : nullptr;
  const DeviceAllocation key_allocation = *keys;
  const DeviceAllocation value_allocation = payload ? *values : key_allocation;
  auto leases =
      std::make_shared<std::vector<std::shared_ptr<DeviceAllocationGuard>>>();
  leases->push_back(keys);
  if (values)
    leases->push_back(values);
  std::weak_ptr<DeviceAllocationGuard> storage = keys;
  auto plan = make_vulkan_parallel_sort_plan(
      *device, keys->get_ptr(prefix * 4),
      payload ? std::optional<DevicePtr>(values->get_ptr(prefix * 4))
              : std::nullopt,
      leases, n, shaders(payload, fused), !inline_recording);
  auto facts = plan->statistics();
  EXPECT_EQ(facts.at("dispatch_count"), fused ? 24 : 40);
  EXPECT_EQ(facts.at("barrier_count"), facts.at("dispatch_count") + 1);
  EXPECT_EQ(facts.at("device_copy_count"), 0);
  auto *stream = device->get_compute_stream();
  auto [commands, status] = stream->new_command_list_unique();
  ASSERT_EQ(status, RhiResult::success);
  plan->record(commands.get(), inline_recording);
  plan.reset();
  keys.reset();
  values.reset();
  leases.reset();
  ASSERT_FALSE(storage.expired());
  for (uint32_t replay = 0; replay < 2; ++replay) {
    std::vector<uint32_t> input(capacity, 0xfefefefe),
        indices(capacity, 0xcdcdcdcd), order(n);
    std::iota(order.begin(), order.end(), 0);
    for (uint32_t i = 0; i < n; ++i) {
      input[prefix + i] = replay ? (i * 2654435761u) : ((n - i) % 37);
      indices[prefix + i] = i;
    }
    input[prefix] = std::numeric_limits<uint32_t>::max();
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
      return input[prefix + a] < input[prefix + b];
    });
    upload(key_allocation, input);
    if (payload)
      upload(value_allocation, indices);
    auto completion = stream->submit(commands.get());
    ASSERT_TRUE(completion);
    ASSERT_TRUE(completion->wait());
    const auto output = read(key_allocation, capacity);
    for (uint32_t i = 0; i < n; ++i)
      EXPECT_EQ(output[prefix + i], input[prefix + order[i]]);
    for (size_t i = 0; i < capacity; ++i)
      if (i < prefix || i >= prefix + n)
        EXPECT_EQ(output[i], input[i]);
    if (payload) {
      const auto sorted_indices = read(value_allocation, capacity);
      for (uint32_t i = 0; i < n; ++i)
        EXPECT_EQ(sorted_indices[prefix + i], order[i]);
      for (size_t i = 0; i < capacity; ++i)
        if (i < prefix || i >= prefix + n)
          EXPECT_EQ(sorted_indices[i], indices[i]);
    }
  }
  commands.reset();
  stream->command_sync();
  EXPECT_TRUE(storage.expired());
}

TEST_F(VulkanParallelSortPlanTest,
       RejectsInvalidStorageBeforePipelineCreation) {
  auto keys = allocate(513);
  auto code = shaders(false, false);
  auto make = [&](DevicePtr key, std::optional<DevicePtr> values, uint64_t n) {
    return make_vulkan_parallel_sort_plan(*device, key, values, keys, n, code,
                                          false);
  };
  EXPECT_THROW(make(keys->get_ptr(), keys->get_ptr(), 513),
               std::invalid_argument);
  EXPECT_THROW(make(keys->get_ptr(1), {}, 512), std::invalid_argument);
  // The allocator may pad its physical VkBuffer. Typed callers separately
  // check logical ndarray extents; this core validates the actual allocation.
  const auto beyond_buffer = device->get_vkbuffer_size(*keys) / 4 + 1;
  EXPECT_THROW(make(keys->get_ptr(), {}, beyond_buffer), std::invalid_argument);
  EXPECT_THROW(make(keys->get_ptr(), {}, 0), std::invalid_argument);
  EXPECT_THROW(make(keys->get_ptr(), {}, uint64_t(1) << 32),
               std::invalid_argument);
  code[0].clear();
  EXPECT_THROW(make(keys->get_ptr(), {}, 513), std::invalid_argument);
  EXPECT_EQ(
      vulkan_parallel_sort_storage_bytes(std::numeric_limits<uint32_t>::max()),
      uint64_t(std::numeric_limits<uint32_t>::max()) * 4);
}

INSTANTIATE_TEST_SUITE_P(SharedCore,
                         VulkanParallelSortPlanTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Bool()));
}  // namespace
}  // namespace taichi::lang
