#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <limits>
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/runtime/fixed_execution.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/runtime/vulkan_diagnostic.h"

namespace taichi::lang {
namespace {
template <typename T, size_t N>
void write_words(Device &device,
           DeviceAllocation allocation,
           const std::array<T, N> &data) {
  void *mapped = nullptr;
  ASSERT_EQ(device.map(allocation, &mapped), RhiResult::success);
  std::memcpy(mapped, data.data(), sizeof(data));
  device.unmap(allocation);
}
template <typename T, size_t N>
void expect_bytes(Device &device,
                  DeviceAllocation allocation,
                  const std::array<T, N> &expected) {
  void *mapped = nullptr;
  ASSERT_EQ(device.map(allocation, &mapped), RhiResult::success);
  const auto *actual = static_cast<const unsigned char *>(mapped);
  const auto *wanted = reinterpret_cast<const unsigned char *>(expected.data());
  for (size_t i = 0; i < sizeof(expected); ++i)
    EXPECT_EQ(actual[i], wanted[i]) << "byte offset " << i;
  device.unmap(allocation);
}
}  // namespace

TEST(VulkanDiagnostic, FixedReplayPreservesStridedViewsAndOverwritesNonfinite) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator(vulkan::VulkanDeviceCreator::Params{});
  auto &device = *creator.device();
  gfx::GfxRuntime::Params params;
  params.device = &device;
  gfx::GfxRuntime runtime(params);
  std::array<float, 64> words;
  words.fill(37);
  auto [allocation, result] = device.allocate_memory_unique(
      {sizeof(words), true, true, false, AllocUsage::Storage});
  ASSERT_EQ(result, RhiResult::success);
  write_words(device, *allocation, words);
  const gfx::VulkanDiagnosticInput values{allocation->get_ptr(4), 8};
  const gfx::VulkanDiagnosticInput other{allocation->get_ptr(17 * 4), 4};
  auto check = gfx::prepare_vulkan_check_operation(
      device, values, allocation->get_ptr(33 * 4), 6, 1, 0, 0, 0);
  auto metric = gfx::prepare_vulkan_metric_operation(
      device, values, other, allocation->get_ptr(35 * 4), 6, 1, 1);
  auto magnitude = gfx::prepare_vulkan_metric_operation(
      device, values, {}, allocation->get_ptr(37 * 4), 6, 1, 0);
  auto full = std::make_unique<FixedExecution>(
      Arch::vulkan, device, &runtime,
      std::vector<FixedExecution::Operation>{check, metric, magnitude});
  FixedExecution empty(
      Arch::vulkan, device, &runtime,
      {gfx::prepare_vulkan_check_operation(
           device, {}, allocation->get_ptr(33 * 4), 0, 1, 0, 0, 0),
       gfx::prepare_vulkan_metric_operation(
           device, {}, {}, allocation->get_ptr(35 * 4), 0, 1, 1),
       gfx::prepare_vulkan_metric_operation(
           device, {}, {}, allocation->get_ptr(37 * 4), 0, 1, 0)});
  expect_bytes(device, *allocation, words);  // Preparation does not write.
  EXPECT_EQ(full->workspace_bytes(), 0);
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::array<float, 6> initial{-0.0f, 0.0f, 2, -3, nan, inf};
  for (int replay = 0; replay < 3; ++replay) {
    SCOPED_TRACE(replay);
    for (size_t i = 0; i < initial.size(); ++i) {
      words[1 + 2 * i] =
          replay == 0 ? initial[i] : (replay == 2 && i == 0 ? 4 : 0);
      words[17 + i] = replay == 0 ? initial[i] : 0;
    }
    write_words(device, *allocation, words);
    full->run();
    runtime.synchronize();
    const int32_t count = replay == 0 ? 4 : replay == 1 ? 0 : 1;
    std::memcpy(&words[33], &count, sizeof(count));
    words[35] = replay == 0 ? inf : replay == 1 ? 0 : 4;
    words[37] = words[35];
    expect_bytes(device, *allocation, words);
  }
  empty.run();
  runtime.synchronize();
  words[33] = words[35] = words[37] = 0;
  expect_bytes(device, *allocation, words);
  check.reset();
  metric.reset();
  magnitude.reset();
  full->run();
  full.reset();  // In-flight graph closure retains its native
                 // commands/resources.
  runtime.synchronize();  // CPU observation still requires completion.
  const int32_t one = 1;
  std::memcpy(&words[33], &one, sizeof(one));
  words[35] = words[37] = 4;
  expect_bytes(device, *allocation, words);
}

TEST(VulkanDiagnostic, WideInputAndDifferentGridPreferencesShareSemantics) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator(vulkan::VulkanDeviceCreator::Params{});
  auto &device = *creator.device();
  if (!device.get_caps().get(DeviceCapability::spirv_has_int64))
    GTEST_SKIP();
  gfx::GfxRuntime::Params params;
  params.device = &device;
  gfx::GfxRuntime runtime(params);
  // 1025 elements exercise more than one group, far below a costly GPU test.
  std::array<uint64_t, 2056> words;
  words.fill(37);
  for (size_t i = 0; i < 1025; ++i)
    words[1 + 2 * i] = i % 3 ? i : UINT64_MAX;
  auto [allocation, result] = device.allocate_memory_unique(
      {sizeof(words), true, true, false, AllocUsage::Storage});
  ASSERT_EQ(result, RhiResult::success);
  for (auto items : {uint64_t(0), uint64_t(16384), UINT64_MAX}) {
    write_words(device, *allocation, words);
    auto op = gfx::prepare_vulkan_check_operation(
        device, {allocation->get_ptr(8), 16}, allocation->get_ptr(2053 * 8),
        1025, 3, 5, -3, -1, items);
    FixedExecution execution(Arch::vulkan, device, &runtime, {op});
    execution.run();
    runtime.synchronize();
    const int32_t count = 1025;
    std::memcpy(&words[2053], &count, sizeof(count));
    expect_bytes(device, *allocation, words);
  }
}

TEST(VulkanDiagnostic, ColdValidationRejectsOverlapsAndInvalidViews) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator(vulkan::VulkanDeviceCreator::Params{});
  auto &device = *creator.device();
  auto [allocation, result] = device.allocate_memory_unique(
      {256, false, false, false, AllocUsage::Storage});
  ASSERT_EQ(result, RhiResult::success);
  const auto output = allocation->get_ptr(128);
  const gfx::VulkanDiagnosticInput values{allocation->get_ptr(4), 8};
  EXPECT_ANY_THROW(gfx::prepare_vulkan_check_operation(
      device, values, allocation->get_ptr(12), 6, 0, 0, 0, 0));
  EXPECT_ANY_THROW(gfx::prepare_vulkan_check_operation(
      device, {allocation->get_ptr(2), 8}, output, 6, 0, 0, 0, 0));
  EXPECT_ANY_THROW(gfx::prepare_vulkan_check_operation(
      device, {allocation->get_ptr(4), 3}, output, 6, 0, 0, 0, 0));
  EXPECT_ANY_THROW(gfx::prepare_vulkan_check_operation(device, values, output,
                                                       6, 1, 5, 0, 4));
  EXPECT_ANY_THROW(gfx::prepare_vulkan_check_operation(
      device, {allocation->get_ptr(), SIZE_MAX}, output, SIZE_MAX, 0, 0, 0, 0));
  EXPECT_ANY_THROW(gfx::prepare_vulkan_metric_operation(device, values, values,
                                                        output, 6, 5, 0));
  EXPECT_ANY_THROW(gfx::prepare_vulkan_metric_operation(device, values, {},
                                                        output, 6, 1, 1));
}
}  // namespace taichi::lang
