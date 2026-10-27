#include "gtest/gtest.h"

#ifdef TI_WITH_VULKAN
#include "spirv-tools/libspirv.hpp"
#include "taichi/codegen/spirv/spirv_ir_builder.h"

namespace taichi::lang {

TEST(SpirvRandom, SharedCounterIsOnlyAccessedAtomically) {
  for (auto version : {0x10000u, 0x10300u, 0x10500u}) {
    SCOPED_TRACE(version);
    DeviceCapabilityConfig caps;
    caps.set(DeviceCapability::spirv_version, version);
    spirv::IRBuilder builder(Arch::vulkan, &caps);
    builder.init_header();
    const auto temporary =
        builder.buffer_argument(builder.u32_type(), 0, 0, "global_tmp");
    const auto function = builder.new_function();
    builder.start_function(function);
    builder.get_global_invocation_id(0);
    builder.rand_u32(temporary);
    builder.rand_f32(temporary);
    builder.make_inst(spv::OpReturn);
    builder.make_inst(spv::OpFunctionEnd);
    std::vector<spirv::Value> interfaces;
    if (version >= 0x10400u) {
      interfaces.push_back(temporary);
    }
    builder.commit_kernel_function(function, "main", interfaces, {128, 1, 1});
    const auto module = builder.finalize();
    std::string diagnostic;
    spvtools::SpirvTools validator(SPV_ENV_VULKAN_1_2);
    validator.SetMessageConsumer(
        [&](spv_message_level_t, const char *, const spv_position_t &,
            const char *message) { diagnostic += message; });
    ASSERT_TRUE(validator.Validate(module)) << diagnostic;

    uint32_t counter = 0;
    int claims = 0;
    for (size_t pos = 5; pos < module.size(); pos += module[pos] >> 16) {
      if ((module[pos] & 0xffffu) == spv::OpAtomicIIncrement) {
        counter = module[pos + 3];
        ++claims;
      }
    }
    ASSERT_EQ(claims, 1);
    ASSERT_NE(counter, 0u);
    // These checks fail for both the old non-atomic Vulkan update and a
    // partial fix that retains a non-atomic read before an atomic update.
    for (size_t pos = 5; pos < module.size(); pos += module[pos] >> 16) {
      const auto op = module[pos] & 0xffffu;
      if (op == spv::OpLoad) {
        EXPECT_NE(module[pos + 3], counter);
      }
      if (op == spv::OpStore) {
        EXPECT_NE(module[pos + 1], counter);
      }
    }
  }
}

}  // namespace taichi::lang
#endif
