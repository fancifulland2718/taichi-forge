#include "gtest/gtest.h"

#ifdef TI_WITH_VULKAN
#include "spirv-tools/libspirv.hpp"
#include "taichi/codegen/spirv/spirv_ir_builder.h"

namespace taichi::lang {

TEST(SpirvArrayLayout, LocalSharedAndBufferArraysValidateTogether) {
  for (auto version : {0x10000u, 0x10300u, 0x10500u}) {
    for (bool boolean_elements : {false, true}) {
      SCOPED_TRACE(version);
      SCOPED_TRACE(boolean_elements);
      DeviceCapabilityConfig caps;
      caps.set(DeviceCapability::spirv_version, version);
      spirv::IRBuilder builder(Arch::vulkan, &caps);
      builder.init_header();
      const auto element =
          boolean_elements ? builder.bool_type() : builder.i32_type();
      // The same element type and length need distinct layout contracts.
      const auto array = builder.get_array_type(element, 4);
      const auto shared = builder.alloca_workgroup_array(array);
      const auto fixed_struct = builder.get_struct_array_type(element, 4);
      if (version < 0x10300u) {
        builder.decorate(spv::OpDecorate, fixed_struct,
                         spv::DecorationBufferBlock);
      }
      const auto buffer_ptr = builder.get_storage_pointer_type(fixed_struct);
      const auto fixed_buffer =
          builder.new_value(buffer_ptr, spirv::ValueKind::kStructArrayPtr);
      builder.declare_global(spv::OpVariable, buffer_ptr, fixed_buffer,
                             buffer_ptr.storage_class);
      builder.decorate(spv::OpDecorate, fixed_buffer,
                       spv::DecorationDescriptorSet, 0);
      builder.decorate(spv::OpDecorate, fixed_buffer, spv::DecorationBinding,
                       0);
      const auto runtime_buffer =
          builder.buffer_argument(element, 0, 1, "runtime");
      const auto function = builder.new_function();
      builder.start_function(function);
      builder.alloca_variable(array);
      builder.make_inst(spv::OpReturn);
      builder.make_inst(spv::OpFunctionEnd);
      std::vector<spirv::Value> interfaces;
      if (version > 0x10300u) {
        interfaces = {shared, fixed_buffer, runtime_buffer};
      }
      builder.commit_kernel_function(function, "main", std::move(interfaces),
                                     {1, 1, 1});
      const auto module = builder.finalize();

      std::string diagnostic;
      spvtools::SpirvTools validator(SPV_ENV_VULKAN_1_2);
      validator.SetMessageConsumer(
          [&](spv_message_level_t, const char *, const spv_position_t &,
              const char *message) { diagnostic += message; });
      EXPECT_TRUE(validator.Validate(module)) << diagnostic;

      // Both buffer arrays retain their four-byte physical element stride,
      // including bool arrays (stored as i32). Locals have no ArrayStride.
      int strides = 0;
      for (size_t pos = 5; pos < module.size(); pos += module[pos] >> 16) {
        if ((module[pos] & 0xffffu) == spv::OpDecorate &&
            module[pos + 2] == spv::DecorationArrayStride) {
          EXPECT_NE(module[pos + 1], array.id);
          EXPECT_EQ(module[pos + 3], 4u);
          ++strides;
        }
      }
      EXPECT_EQ(strides, 2);
    }
  }
}

}  // namespace taichi::lang
#endif
