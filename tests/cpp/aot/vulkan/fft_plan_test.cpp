#include "gtest/gtest.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/runtime/vulkan_fft_plan.h"

#include <cmath>
#include <complex>
#include <cstdlib>
#include <limits>

namespace taichi::lang {
namespace {

class VulkanFftPlanTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    if (!vulkan::is_vulkan_api_available())
      GTEST_SKIP() << "Vulkan is unavailable";
    const char *path = std::getenv("TI_VKFFT_LIBRARY_PATH");
    if (!path || !*path)
      GTEST_SKIP() << "Set TI_VKFFT_LIBRARY_PATH for the real FFT adapter";
    adapter = path;
    creator = std::make_unique<vulkan::VulkanDeviceCreator>(
        vulkan::VulkanDeviceCreator::Params{});
    device = creator->device();
  }

  void upload(DeviceAllocation allocation, const std::vector<float> &values) {
    auto ptr = allocation.get_ptr();
    const void *host = values.data();
    std::size_t bytes = values.size() * sizeof(float);
    ASSERT_EQ(device->upload_data(&ptr, &host, &bytes), RhiResult::success);
  }

  std::vector<float> read(DeviceAllocation allocation, std::size_t count) {
    std::vector<float> values(count);
    auto ptr = allocation.get_ptr();
    void *host = values.data();
    std::size_t bytes = count * sizeof(float);
    EXPECT_EQ(device->readback_data(&ptr, &host, &bytes), RhiResult::success);
    return values;
  }

  void check_dft(const std::vector<float> &input,
                 const std::vector<float> &actual) {
    // Independent eight-point DFT, three batches. No FFT provider reference.
    for (int batch = 0; batch < 3; ++batch) {
      for (int k = 0; k < 8; ++k) {
        std::complex<double> expected{};
        for (int n = 0; n < 8; ++n) {
          const auto index = (batch * 8 + n) * 2;
          expected += std::complex<double>(input[index], input[index + 1]) *
                      std::polar(1.0, -2 * std::acos(-1.0) * k * n / 8);
        }
        const auto index = (batch * 8 + k) * 2;
        EXPECT_NEAR(actual[index], expected.real(), 2e-5);
        EXPECT_NEAR(actual[index + 1], expected.imag(), 2e-5);
      }
    }
  }

  std::string adapter;
  std::unique_ptr<vulkan::VulkanDeviceCreator> creator;
  vulkan::VulkanDevice *device{nullptr};
};

TEST_P(VulkanFftPlanTest, IndependentBindingsReplayAndCommandRetirement) {
  constexpr std::size_t count = 3 * 8 * 2;
  constexpr std::size_t bytes = count * sizeof(float);
  auto allocate = [&] {
    auto [allocation, result] = device->allocate_memory_unique(
        {bytes, false, false, false, AllocUsage::Storage});
    EXPECT_EQ(result, RhiResult::success);
    return std::shared_ptr<DeviceAllocationGuard>(std::move(allocation));
  };
  auto first = allocate(), second = allocate();
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  const DeviceAllocation first_allocation = *first, second_allocation = *second;
  std::vector<float> input(count), other(count);
  for (std::size_t i = 0; i < count; ++i) {
    input[i] = static_cast<float>(static_cast<int>(i % 11) - 5) / 32;
    other[i] = static_cast<float>(static_cast<int>(i % 7) - 2) / 16;
  }
  upload(first_allocation, input);
  upload(second_allocation, other);
  VulkanFftPlanConfig config{{8}, 3, -1, false, 2};
  auto forward =
      make_vulkan_fft_plan(*device, *first, bytes, first, config, adapter);
  config.batch_tile = 0;
  auto independent =
      make_vulkan_fft_plan(*device, *second, bytes, second, config, adapter);
  config.direction = 1;
  config.normalize_inverse = true;
  config.batch_tile = 2;
  auto inverse =
      make_vulkan_fft_plan(*device, *first, bytes, first, config, adapter);
  const auto facts = forward->statistics();
  EXPECT_EQ(facts.at("batch_tile"), 2);
  EXPECT_EQ(facts.at("application_count"), 2);
  EXPECT_GT(facts.at("shader_module_count"), 0);
  if (GetParam()) {
    ASSERT_TRUE(forward->supports_inline_recording());
    ASSERT_TRUE(independent->supports_inline_recording());
    ASSERT_TRUE(inverse->supports_inline_recording());
  }
  EXPECT_EQ(read(first_allocation, count), input);
  EXPECT_EQ(read(second_allocation, count), other);

  auto *stream = device->get_compute_stream();
  auto [forward_commands, forward_status] = stream->new_command_list_unique();
  auto [inverse_commands, inverse_status] = stream->new_command_list_unique();
  ASSERT_EQ(forward_status, RhiResult::success);
  ASSERT_EQ(inverse_status, RhiResult::success);
  forward_commands->memory_barrier();
  forward->record(forward_commands.get(), GetParam());
  forward_commands->memory_barrier();
  independent->record(forward_commands.get(), GetParam());
  forward_commands->memory_barrier();
  inverse_commands->memory_barrier();
  inverse->record(inverse_commands.get(), GetParam());
  inverse_commands->memory_barrier();

  std::weak_ptr<VulkanFftPlanCore> plan_lease = forward;
  std::weak_ptr<DeviceAllocationGuard> storage_lease = first;
  forward.reset();
  independent.reset();
  inverse.reset();
  first.reset();
  second.reset();
  EXPECT_FALSE(plan_lease.expired());
  EXPECT_FALSE(storage_lease.expired());
  for (int replay = 0; replay < 2; ++replay) {
    upload(first_allocation, input);
    upload(second_allocation, other);
    auto completion = stream->submit(forward_commands.get());
    ASSERT_TRUE(completion);
    EXPECT_FALSE(plan_lease.expired());
    ASSERT_TRUE(completion->wait());
    check_dft(input, read(first_allocation, count));
    check_dft(other, read(second_allocation, count));
    ASSERT_TRUE(stream->submit_synced(inverse_commands.get()));
    const auto restored = read(first_allocation, count);
    for (std::size_t i = 0; i < count; ++i) {
      EXPECT_NEAR(restored[i], input[i], 1e-6);
      input[i] *= -0.5f;
      other[i] *= 1.5f;
    }
  }
  forward_commands.reset();
  inverse_commands.reset();
  stream->command_sync();
  EXPECT_TRUE(plan_lease.expired());
  EXPECT_TRUE(storage_lease.expired());
}

TEST_F(VulkanFftPlanTest, RejectsInvalidStorageAndConfigBeforeAdapterLoad) {
  auto [allocation, result] = device->allocate_memory_unique(
      {64, false, false, false, AllocUsage::Storage});
  ASSERT_EQ(result, RhiResult::success);
  auto lease = std::shared_ptr<DeviceAllocationGuard>(std::move(allocation));
  VulkanFftPlanConfig config{{8}};
  const auto reject = [&](std::size_t bytes, std::shared_ptr<void> owner,
                          const char *expected) {
    try {
      make_vulkan_fft_plan(*device, *lease, bytes, owner, config,
                           "missing-fft-test-adapter");
      FAIL() << "Invalid FFT configuration was accepted";
    } catch (const std::string &error) {
      EXPECT_NE(error.find(expected), std::string::npos) << error;
    }
  };
  reject(64, nullptr, "lifetime lease");
  reject(32, lease, "complete compact transform");
  config.dimensions = {16};
  reject(128, lease, "complete compact transform");
  config.dimensions = {17};
  reject(136, lease, "prime factors");
  config.dimensions = {8};
  config.batches = std::numeric_limits<std::uint64_t>::max();
  reject(64, lease, "size overflow");
  config.batches = 1;
  config.batch_tile = 2;
  reject(64, lease, "batch tile");
}

INSTANTIATE_TEST_SUITE_P(RecordingModes, VulkanFftPlanTest, ::testing::Bool());
}  // namespace
}  // namespace taichi::lang
