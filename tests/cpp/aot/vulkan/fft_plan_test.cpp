#include "gtest/gtest.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/runtime/vulkan_fft_plan.h"
#include "taichi/runtime/native_provider.h"
#include "taichi/runtime/native_operation_registry.h"
#include "taichi/vkfft/forge_vkfft_runtime_provider.h"

#include <cmath>
#include <complex>
#include <cstdlib>
#include <limits>
#include <tuple>

namespace taichi::lang {
namespace {

class VulkanFftPlanTest
    : public ::testing::TestWithParam<std::tuple<bool, bool>> {
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
  const auto [inline_recording, common_provider] = GetParam();
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
  auto session =
      std::make_unique<NativeProviderSession>(NativeProviderLocations{
          {"vkfft", {adapter, ""}}, {"unused", {"not-loaded", ""}}});
  auto runtime = common_provider
                     ? session->runtime(vulkan_fft_provider_definition())
                     : nullptr;
  std::weak_ptr<NativeProviderRuntime> runtime_lease = runtime;
  auto prepare = [&](std::shared_ptr<DeviceAllocationGuard> storage,
                     const VulkanFftPlanConfig &description) {
    return common_provider
               ? make_vulkan_fft_plan(*device, *storage, bytes, storage,
                                      description, runtime)
               : make_vulkan_fft_plan(*device, *storage, bytes, storage,
                                      description, adapter);
  };
  VulkanFftPlanConfig config{{8}, 3, -1, false, 2};
  auto forward = prepare(first, config);
  config.batch_tile = 0;
  auto independent = prepare(second, config);
  config.direction = 1;
  config.normalize_inverse = true;
  config.batch_tile = 2;
  auto inverse = prepare(first, config);
  const auto facts = forward->statistics();
  EXPECT_EQ(facts.at("batch_tile"), 2);
  EXPECT_EQ(facts.at("application_count"), 2);
  EXPECT_GT(facts.at("shader_module_count"), 0);
  if (inline_recording) {
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
  forward->record(forward_commands.get(), inline_recording);
  forward_commands->memory_barrier();
  independent->record(forward_commands.get(), inline_recording);
  forward_commands->memory_barrier();
  inverse_commands->memory_barrier();
  inverse->record(inverse_commands.get(), inline_recording);
  inverse_commands->memory_barrier();

  std::weak_ptr<VulkanFftPlanCore> plan_lease = forward;
  std::weak_ptr<DeviceAllocationGuard> storage_lease = first;
  forward.reset();
  independent.reset();
  inverse.reset();
  first.reset();
  second.reset();
  runtime.reset();
  session.reset();
  EXPECT_FALSE(plan_lease.expired());
  EXPECT_FALSE(storage_lease.expired());
  if (common_provider)
    EXPECT_FALSE(runtime_lease.expired());
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
  EXPECT_TRUE(runtime_lease.expired());
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
    } catch (const std::invalid_argument &error) {
      EXPECT_NE(std::string(error.what()).find(expected), std::string::npos)
          << error.what();
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

TEST_F(VulkanFftPlanTest, AotRejectsInvalidBindingsBeforeProviderLoad) {
  const auto &registry = native_operation_registry();
  const auto plan = registry.decode(
      NativeOperationDescription::parse(liong::json::parse_complete(
          R"({"kind":"native","provider":"vkfft","operation":"plan_f32","version":1,
          "parameters":{"dimensions":[8]},"bindings":{"data":"x"}})")),
      NativeObjectKind::resource);
  auto [allocation, result] = device->allocate_memory_unique(
      {64, false, false, false, AllocUsage::Storage});
  ASSERT_EQ(result, RhiResult::success);
  NativeArrayBinding valid{
      allocation->get_ptr(), 64, PrimitiveType::f32, {8, 2}};
  const auto reject = [&](NativeArrayBinding array, const char *expected) {
    try {
      registry.prepare_resource(Arch::vulkan, *device, plan, {{"data", array}});
      FAIL() << "Invalid FFT binding was accepted";
    } catch (const std::invalid_argument &error) {
      EXPECT_NE(std::string(error.what()).find(expected), std::string::npos)
          << error.what();
    }
  };
  auto invalid = valid;
  invalid.pointer.offset = 4;
  reject(invalid, "subview offsets");
  invalid = valid;
  invalid.scalar_type = PrimitiveType::i32;
  reject(invalid, "dtype");
  invalid = valid;
  invalid.shape = {16};
  reject(invalid, "shape");
  invalid = valid;
  invalid.bytes = 32;
  reject(invalid, "complete compact transform");
}

INSTANTIATE_TEST_SUITE_P(RecordingModes,
                         VulkanFftPlanTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool()));

TEST(VulkanFftProvider, MetadataAndSessionWithoutDeviceCreation) {
  const char *path = std::getenv("TI_VKFFT_LIBRARY_PATH");
  if (!path || !*path)
    GTEST_SKIP() << "Set TI_VKFFT_LIBRARY_PATH for the real FFT adapter";
  auto module =
      NativeProviderModule::load(vulkan_fft_provider_definition(), path);
  EXPECT_TRUE(module->info().features &
              TI_FORGE_RUNTIME_PROVIDER_FEATURE_EMBEDDED_RUNTIME);
  EXPECT_EQ(module->info().required_symbol_count, 0);
  const auto facts = module->probe();
  EXPECT_EQ(facts.version_major, 1);
  EXPECT_EQ(facts.version_minor, 3);
  EXPECT_EQ(facts.version_patch, 4);
  EXPECT_EQ(facts.cuda_runtime_version, 0);
  EXPECT_TRUE(facts.library_path.empty());
  EXPECT_NE(facts.build_version.find("glslang"), std::string::npos);
  EXPECT_THROW(module->probe("unused-vendor-path"), NativeProviderError);
  EXPECT_THROW(module->create_runtime("unused-vendor-path"),
               NativeProviderError);
  auto runtime = module->create_runtime();
  const auto api = runtime->execution_api<TiForgeVkfftExecutionApi>();
  EXPECT_EQ(api.provider.vkfft_version, 10304);
  EXPECT_NE(api.recipe.create, nullptr);
  EXPECT_NE(api.record_inline, nullptr);
  EXPECT_THROW(runtime->execution_api<TiForgeVkfftExecutionApi>(2),
               NativeProviderError);
  std::weak_ptr<NativeProviderModule> retained = module;
  module.reset();
  EXPECT_FALSE(retained.expired());
  runtime.reset();
  EXPECT_TRUE(retained.expired());
}
}  // namespace
}  // namespace taichi::lang
