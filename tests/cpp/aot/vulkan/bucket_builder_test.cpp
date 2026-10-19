#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>
#include <vector>
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/runtime/fixed_execution.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/runtime/vulkan_bucket_builder.h"

namespace taichi::lang {
namespace {
template <typename T>
void write_bucket_words(Device &device,
                        DeviceAllocation allocation,
                        const std::vector<T> &values) {
  void *mapped = nullptr;
  ASSERT_EQ(device.map(allocation, &mapped), RhiResult::success);
  std::memcpy(mapped, values.data(), values.size() * sizeof(T));
  device.unmap(allocation);
}
template <typename T>
std::vector<T> read_bucket_words(Device &device,
                                 DeviceAllocation allocation,
                                 size_t count) {
  void *mapped = nullptr;
  if (device.map(allocation, &mapped) != RhiResult::success)
    throw std::runtime_error("Could not map bucket fixture");
  std::vector<T> result(count);
  std::memcpy(result.data(), mapped, count * sizeof(T));
  device.unmap(allocation);
  return result;
}

void exercise_bucket(Device &device,
                     gfx::GfxRuntime &runtime,
                     bool private_path,
                     int type,
                     size_t words,
                     int layout,
                     size_t bins = 7) {
  SCOPED_TRACE(private_path);
  SCOPED_TRACE(type);
  SCOPED_TRACE(layout);
  SCOPED_TRACE(bins);
  constexpr size_t n = 2051;
  constexpr uint32_t sentinel = 0xbad0bad0;
  const size_t scalar_words = type >= 3 ? 2 : 1;
  const size_t source_base = layout ? scalar_words : 0;
  const size_t source_step = words * (layout == 2 ? 2 : 1);
  const size_t target_step = words * (layout == 2 ? 3 : 1);
  const size_t target_base =
      ((source_base + n * source_step + 7 + scalar_words - 1) / scalar_words) *
      scalar_words;
  const size_t key_base = layout ? 1 : 0, key_step = layout == 2 ? 2 : 1;
  const size_t offset_base = layout ? 1 : 0, offset_step = layout == 2 ? 2 : 1;
  std::vector<int32_t> keys(key_base + n * key_step + 3, -77),
      offsets(offset_base + (bins + 1) * offset_step + 3, -77);
  std::vector<uint32_t> data(target_base + n * target_step + 3, sentinel);
  auto [key_alloc, kr] = device.allocate_memory_unique(
      {keys.size() * 4, true, true, false, AllocUsage::Storage});
  auto [offset_alloc, orr] = device.allocate_memory_unique(
      {offsets.size() * 4, true, true, false, AllocUsage::Storage});
  auto [data_alloc, dr] = device.allocate_memory_unique(
      {data.size() * 4, true, true, false, AllocUsage::Storage});
  ASSERT_EQ(kr, RhiResult::success);
  ASSERT_EQ(orr, RhiResult::success);
  ASSERT_EQ(dr, RhiResult::success);
  write_bucket_words(device, *key_alloc, keys);
  write_bucket_words(device, *offset_alloc, offsets);
  write_bucket_words(device, *data_alloc, data);
  using View = gfx::VulkanBucketBuilderView;
  const View key_view{key_alloc->get_ptr(key_base * 4), key_step * 4};
  const View values{data_alloc->get_ptr(source_base * 4), source_step * 4};
  const View output{data_alloc->get_ptr(target_base * 4), target_step * 4};
  const View offset_view{offset_alloc->get_ptr(offset_base * 4),
                         offset_step * 4};
  gfx::VulkanBucketBuilderPolicy policy{private_path, 0, UINT64_MAX,
                                        UINT64_MAX};
  auto make_full = [&] {
    return std::make_unique<FixedExecution>(
        Arch::vulkan, device, &runtime,
        std::vector<FixedExecution::Operation>{
            gfx::prepare_vulkan_bucket_builder_operation(
                device, key_view, values, offset_view, output, n, bins,
                words * 4, type, policy)});
  };
  auto full = make_full(), peer = make_full();
  FixedExecution empty(
      Arch::vulkan, device, &runtime,
      {gfx::prepare_vulkan_bucket_builder_operation(
          device, {}, {}, offset_view, {}, 0, bins, words * 4, type, policy)});
  EXPECT_EQ(full->workspace_bytes(), private_path && bins <= 4096
                                         ? ((n + 2047) / 2048) * bins * 4
                                         : bins * 4);
  EXPECT_EQ(empty.workspace_bytes(), 0);
  EXPECT_EQ(read_bucket_words<int32_t>(device, *offset_alloc, offsets.size()),
            offsets);
  EXPECT_EQ(read_bucket_words<uint32_t>(device, *data_alloc, data.size()),
            data);
  for (int replay = 0; replay < 3; ++replay) {
    std::vector<int32_t> expected_offsets = offsets, counts(bins), seen(n);
    for (size_t i = 0; i < n; ++i) {
      const auto key = replay == 1 || i % 7 == 0 ? -1
                       : i % 7 == 1              ? INT32_MAX
                                                 : int32_t((i + replay) % bins);
      keys[key_base + i * key_step] = key;
      if (key >= 0 && size_t(key) < bins)
        ++counts[key];
      data[source_base + i * source_step] = 100 + i;
      for (size_t lane = 1; lane < words; ++lane)
        data[source_base + i * source_step + lane] =
            (lane == 1 ? 0x7fc01234u : 0x80000000u) + uint32_t(i + replay);
    }
    expected_offsets[offset_base] = 0;
    for (size_t b = 0; b < bins; ++b)
      expected_offsets[offset_base + (b + 1) * offset_step] =
          expected_offsets[offset_base + b * offset_step] + counts[b];
    write_bucket_words(device, *key_alloc, keys);
    write_bucket_words(device, *data_alloc, data);
    (replay == 1 ? peer : full)->run();
    runtime.synchronize();
    auto actual = read_bucket_words<uint32_t>(device, *data_alloc, data.size());
    auto expected_data = data;
    ASSERT_EQ(read_bucket_words<int32_t>(device, *offset_alloc, offsets.size()),
              expected_offsets);
    for (size_t b = 0; b < bins; ++b) {
      for (int pos = expected_offsets[offset_base + b * offset_step];
           pos < expected_offsets[offset_base + (b + 1) * offset_step]; ++pos) {
        const auto id = actual[target_base + pos * target_step] - 100;
        ASSERT_LT(id, n);
        EXPECT_EQ(keys[key_base + id * key_step], b);
        ++seen[id];
        for (size_t lane = 0; lane < words; ++lane)
          expected_data[target_base + pos * target_step + lane] =
              data[source_base + id * source_step + lane];
      }
    }
    for (size_t i = 0; i < n; ++i)
      EXPECT_EQ(seen[i], keys[key_base + i * key_step] >= 0 &&
                                 size_t(keys[key_base + i * key_step]) < bins
                             ? 1
                             : 0);
    EXPECT_EQ(actual,
              expected_data);  // Includes source, gaps and unwritten tail.
    EXPECT_EQ(read_bucket_words<int32_t>(device, *key_alloc, keys.size()),
              keys);
  }
  full->run();
  full.reset();
  peer->run();
  runtime.synchronize();
  const auto previous =
      read_bucket_words<uint32_t>(device, *data_alloc, data.size());
  empty.run();
  runtime.synchronize();
  for (size_t b = 0; b <= bins; ++b)
    offsets[offset_base + b * offset_step] = 0;
  EXPECT_EQ(read_bucket_words<int32_t>(device, *offset_alloc, offsets.size()),
            offsets);
  EXPECT_EQ(read_bucket_words<uint32_t>(device, *data_alloc, data.size()),
            previous);
}
}  // namespace

TEST(VulkanBucketBuilder, DirectPrivateAlignedStridedAndFallbackReplay) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator(vulkan::VulkanDeviceCreator::Params{});
  auto &device = *creator.device();
  gfx::GfxRuntime::Params params;
  params.device = &device;
  gfx::GfxRuntime runtime(params);
  for (bool private_path : {false, true}) {
    exercise_bucket(device, runtime, private_path, 0, 1, 0);
    exercise_bucket(device, runtime, private_path, 1, 1, 1);
    exercise_bucket(device, runtime, private_path, 3, 2, 2);
    exercise_bucket(device, runtime, private_path, 2, 3, 2);
  }
  exercise_bucket(device, runtime, true, 2, 1, 1, 4097);
}

TEST(VulkanBucketBuilder, InvalidStorageRejectedBeforeWriting) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator(vulkan::VulkanDeviceCreator::Params{});
  auto &device = *creator.device();
  std::vector<uint32_t> data(128, 37);
  auto [allocation, result] = device.allocate_memory_unique(
      {data.size() * 4, true, true, false, AllocUsage::Storage});
  ASSERT_EQ(result, RhiResult::success);
  write_bucket_words(device, *allocation, data);
  using View = gfx::VulkanBucketBuilderView;
  const View keys{allocation->get_ptr(0), 4},
      values{allocation->get_ptr(64), 4}, offsets{allocation->get_ptr(128), 4},
      output{allocation->get_ptr(192), 4};
  auto prepare = [&](View key, View value, View off, View out, size_t n,
                     size_t bins) {
    return gfx::prepare_vulkan_bucket_builder_operation(device, key, value, off,
                                                        out, n, bins, 4, 0);
  };
  EXPECT_ANY_THROW(prepare(keys, values, offsets, output, 7, 0));
  EXPECT_ANY_THROW(prepare(keys, values, offsets, output, SIZE_MAX, 7));
  EXPECT_ANY_THROW(prepare(keys, values, offsets, values, 7, 7));
  EXPECT_ANY_THROW(prepare(keys, values, output, output, 7, 7));
  EXPECT_ANY_THROW(
      prepare({allocation->get_ptr(2), 4}, values, offsets, output, 7, 7));
  EXPECT_ANY_THROW(
      prepare(keys, values, {allocation->get_ptr(508), 4}, output, 7, 7));
  EXPECT_ANY_THROW(
      prepare(keys, values, offsets, {allocation->get_ptr(192), 0}, 7, 7));
  EXPECT_EQ(read_bucket_words<uint32_t>(device, *allocation, data.size()),
            data);
}
}  // namespace taichi::lang
