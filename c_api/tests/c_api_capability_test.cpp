#include "c_api/src/taichi_core_impl.h"
#include "taichi/cpp/taichi.hpp"
#include "gtest/gtest.h"
#include "c_api/tests/gtest_fixture.h"

using taichi::lang::DeviceCapability;

class CapabilityMappingTest : public CapiTest,
                              public ::testing::WithParamInterface<TiArch> {};

TEST_P(CapabilityMappingTest,
       PublicNamesMatchDeviceAndOverridesAreTransactional) {
  if (!ti::is_arch_available(GetParam()))
    GTEST_SKIP();
  ti::Runtime runtime(GetParam());
  auto &device = reinterpret_cast<Runtime *>(runtime.runtime())->get();
  auto original = device.get_caps();
  auto level = [](const taichi::lang::DeviceCapabilityConfig &caps,
                  DeviceCapability cap) {
    const auto found = caps.devcaps.find(cap);
    return found == caps.devcaps.end() ? 0u : found->second;
  };
  const auto public_caps = runtime.get_capabilities();
#define PER_CAPABILITY(public_name, internal_name)        \
  EXPECT_EQ(public_caps.get(TI_CAPABILITY_##public_name), \
            level(original, DeviceCapability::internal_name));
#include "c_api/src/device_capability.inc.h"
#undef PER_CAPABILITY

  // Round trips alone missed the old symmetric ordinal-cast bug. Check the
  // actual internal storage/width facts, then independently check the getter.
  const TiCapabilityLevelInfo values[] = {
      {TI_CAPABILITY_SPIRV_HAS_INT64, 7},
      {TI_CAPABILITY_SPIRV_HAS_INT8_STORAGE, 3},
      {TI_CAPABILITY_SPIRV_HAS_RAY_QUERY, 1}};
  ti_set_runtime_capabilities_ext(runtime, 3, values);
  ASSERT_TAICHI_SUCCESS();
  EXPECT_EQ(level(device.get_caps(), DeviceCapability::spirv_has_int64), 7);
  EXPECT_EQ(level(device.get_caps(), DeviceCapability::spirv_has_int8_storage),
            3);
  EXPECT_EQ(level(device.get_caps(), DeviceCapability::spirv_has_ray_query), 1);
  EXPECT_EQ(level(device.get_caps(), DeviceCapability::spirv_has_float16), 0);
  const auto changed = runtime.get_capabilities();
  EXPECT_EQ(changed.get(TI_CAPABILITY_SPIRV_HAS_INT64), 7);
  EXPECT_EQ(changed.get(TI_CAPABILITY_SPIRV_HAS_INT8_STORAGE), 3);

  const TiCapabilityLevelInfo invalid[] = {{TI_CAPABILITY_SPIRV_HAS_INT64, 99},
                                           {TI_CAPABILITY_MAX_ENUM, 1}};
  ti_set_runtime_capabilities_ext(runtime, 2, invalid);
  EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_OUT_OF_RANGE, "capability");
  EXPECT_EQ(level(device.get_caps(), DeviceCapability::spirv_has_int64), 7);
  ti_set_runtime_capabilities_ext(runtime, 1, nullptr);
  EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_NULL, "capabilities");

  // A short output must report total count without writing past capacity.
  TiCapabilityLevelInfo output[2] = {{TI_CAPABILITY_RESERVED, 0},
                                     {TI_CAPABILITY_MAX_ENUM, 123}};
  uint32_t count = 1;
  ti_get_runtime_capabilities(runtime, &count, output);
  ASSERT_TAICHI_SUCCESS();
  EXPECT_EQ(count, 3);
  EXPECT_EQ(output[1].level, 123);
  device.set_caps(std::move(original));
}

INSTANTIATE_TEST_SUITE_P(Backends,
                         CapabilityMappingTest,
                         ::testing::Values(TI_ARCH_X64,
                                           TI_ARCH_CUDA,
                                           TI_ARCH_VULKAN));
