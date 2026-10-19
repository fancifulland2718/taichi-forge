#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <limits>
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/runtime/fixed_execution.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/runtime/vulkan_grouped_reduce.h"

namespace taichi::lang {
namespace {
template <typename T, size_t N>
void write_grouped_words(Device &device,
                         DeviceAllocation allocation,
                         const std::array<T, N> &words) {
  void *mapped = nullptr;
  ASSERT_EQ(device.map(allocation, &mapped), RhiResult::success);
  std::memcpy(mapped, words.data(), sizeof(words));
  device.unmap(allocation);
}
template <typename T, size_t N>
void expect_grouped_words(Device &device,
                          DeviceAllocation allocation,
                          const std::array<T, N> &words) {
  void *mapped = nullptr;
  ASSERT_EQ(device.map(allocation, &mapped), RhiResult::success);
  const auto *actual = static_cast<const T *>(mapped);
  for (size_t i = 0; i < N; ++i)
    EXPECT_EQ(actual[i], words[i]) << "element " << i;
  device.unmap(allocation);
}

template <typename T>
void exercise_grouped(Device &device, gfx::GfxRuntime &runtime, int type) {
  constexpr size_t count = 513, groups = 7;
  // Aligned packed, descriptor-misaligned packed, and strided members.
  for (int layout : {0, 1, 2}) {
    SCOPED_TRACE(type);
    SCOPED_TRACE(layout);
    const size_t offset = layout ? 1 : 0, step = layout == 2 ? 2 : 1;
    const size_t output_offset = 1100 + offset;
    std::array<int32_t, 1050> keys;
    std::array<T, 1140> data;
    keys.fill(29);
    data.fill(37);
    auto [key_alloc, kr] = device.allocate_memory_unique(
        {sizeof(keys), true, true, false, AllocUsage::Storage});
    auto [data_alloc, vr] = device.allocate_memory_unique(
        {sizeof(data), true, true, false, AllocUsage::Storage});
    ASSERT_EQ(kr, RhiResult::success);
    ASSERT_EQ(vr, RhiResult::success);
    write_grouped_words(device, *key_alloc, keys);
    write_grouped_words(device, *data_alloc, data);
    const gfx::VulkanGroupedReduceView key_view{key_alloc->get_ptr(offset * 4),
                                                step * 4};
    const gfx::VulkanGroupedReduceView value_view{
        data_alloc->get_ptr(offset * sizeof(T)), step * sizeof(T)};
    const gfx::VulkanGroupedReduceView output_view{
        data_alloc->get_ptr(output_offset * sizeof(T)), step * sizeof(T)};
    auto op = gfx::prepare_vulkan_grouped_reduce_operation(
        device, key_view, value_view, output_view, count, groups, type);
    auto full = std::make_unique<FixedExecution>(
        Arch::vulkan, device, &runtime,
        std::vector<FixedExecution::Operation>{op});
    FixedExecution empty(Arch::vulkan, device, &runtime,
                         {gfx::prepare_vulkan_grouped_reduce_operation(
                             device, {}, {}, output_view, 0, groups, type)});
    EXPECT_EQ(full->workspace_bytes(), 0);
    expect_grouped_words(device, *data_alloc,
                         data);  // Cold preparation is read-only.
    for (int replay = 0; replay < 3; ++replay) {
      SCOPED_TRACE(replay);
      std::array<T, groups> expected{};
      for (size_t i = 0; i < count; ++i) {
        const int32_t key = replay == 1  ? (i % 2 ? INT32_MIN : INT32_MAX)
                            : i % 5 == 0 ? -1
                            : i % 5 == 1
                                ? static_cast<int32_t>(groups)
                                : static_cast<int32_t>((i + replay) % groups);
        keys[offset + step * i] = key;
        data[offset + step * i] = static_cast<T>(i % 7 + 1);
        if (key >= 0 && key < groups)
          expected[key] += data[offset + step * i];
      }
      write_grouped_words(device, *key_alloc, keys);
      write_grouped_words(device, *data_alloc, data);
      full->run();
      runtime.synchronize();
      for (size_t i = 0; i < groups; ++i)
        data[output_offset + step * i] = expected[i];
      expect_grouped_words(device, *key_alloc, keys);
      expect_grouped_words(device, *data_alloc, data);
    }
    empty.run();
    runtime.synchronize();
    const auto nonempty = data;
    for (size_t i = 0; i < groups; ++i)
      data[output_offset + step * i] = 0;
    expect_grouped_words(device, *data_alloc, data);
    op.reset();
    full->run();
    full.reset();  // Queued work retains pipelines and descriptors until
                   // completion.
    runtime.synchronize();
    expect_grouped_words(device, *data_alloc, nonempty);
  }
}
}  // namespace

TEST(VulkanGroupedReduce, FixedReplayEmptyAndStridedNeighbors) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator(vulkan::VulkanDeviceCreator::Params{});
  auto &device = *creator.device();
  gfx::GfxRuntime::Params params;
  params.device = &device;
  gfx::GfxRuntime runtime(params);
  exercise_grouped<int32_t>(device, runtime, 0);
  exercise_grouped<uint32_t>(device, runtime, 2);
  const auto &caps = device.get_caps();
  if (caps.get(DeviceCapability::spirv_has_atomic_float_add))
    exercise_grouped<float>(device, runtime, 1);
  if (caps.get(DeviceCapability::spirv_has_int64) &&
      caps.get(DeviceCapability::spirv_has_atomic_int64)) {
    exercise_grouped<uint64_t>(device, runtime, 3);
    exercise_grouped<int64_t>(device, runtime, 4);
  }
  if (caps.get(DeviceCapability::spirv_has_float64) &&
      caps.get(DeviceCapability::spirv_has_atomic_float64_add))
    exercise_grouped<double>(device, runtime, 5);
}

TEST(VulkanGroupedReduce, ColdValidationRejectsInvalidStorageBeforeWriting) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator(vulkan::VulkanDeviceCreator::Params{});
  auto &device = *creator.device();
  std::array<int32_t, 64> words;
  words.fill(37);
  auto [allocation, result] = device.allocate_memory_unique(
      {sizeof(words), true, true, false, AllocUsage::Storage});
  ASSERT_EQ(result, RhiResult::success);
  write_grouped_words(device, *allocation, words);
  using View = gfx::VulkanGroupedReduceView;
  const View keys{allocation->get_ptr(0), 4};
  const View values{allocation->get_ptr(64), 4};
  const View output{allocation->get_ptr(128), 4};
  auto prepare = [&](View k, View v, View out, size_t n, size_t groups,
                     int type) {
    return gfx::prepare_vulkan_grouped_reduce_operation(device, k, v, out, n,
                                                        groups, type);
  };
  EXPECT_ANY_THROW(prepare(keys, values, output, 7, 0, 0));
  EXPECT_ANY_THROW(prepare(keys, values, output, 7, 7, 99));
  EXPECT_ANY_THROW(prepare(keys, values, output, SIZE_MAX, 7, 0));
  EXPECT_ANY_THROW(
      prepare(keys, values, {allocation->get_ptr(252), 4}, 7, 7, 0));
  EXPECT_ANY_THROW(prepare(keys, values, values, 7, 7, 0));
  EXPECT_ANY_THROW(
      prepare(keys, {allocation->get_ptr(64), 0}, output, 7, 7, 0));
  EXPECT_ANY_THROW(
      prepare({allocation->get_ptr(2), 4}, values, output, 7, 7, 0));
  EXPECT_ANY_THROW(
      prepare(keys, values, {allocation->get_ptr(130), 4}, 7, 7, 0));
  EXPECT_ANY_THROW(
      prepare(keys, values, {allocation->get_ptr(128), SIZE_MAX}, 7, 7, 0));
  expect_grouped_words(device, *allocation, words);
}
}  // namespace taichi::lang
