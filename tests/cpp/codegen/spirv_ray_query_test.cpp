#include "gtest/gtest.h"

#ifdef TI_WITH_VULKAN
#include "spirv-tools/libspirv.hpp"
#include "taichi/codegen/spirv/spirv_ir_builder.h"
#include "taichi/ir/type_factory.h"

namespace taichi::lang {

TEST(SpirvRayQuery, ClosestTraversalValidates) {
  const DataType hit_type = TypeFactory::get_instance().get_struct_type(
      {{PrimitiveType::i32, "hit"},
       {PrimitiveType::f32, "t"},
       {PrimitiveType::u32, "primitive_index"},
       {PrimitiveType::u32, "instance_id"},
       {PrimitiveType::u32, "instance_custom_index"},
       {PrimitiveType::u32, "geometry_index"},
       {PrimitiveType::f32, "barycentric_u"},
       {PrimitiveType::f32, "barycentric_v"},
       {PrimitiveType::i32, "front_face"}});
  for (auto version : {0x10300u, 0x10500u}) {
    // Validate both full closest-hit and hit-only (occlusion) results.
    for (auto member_mask : {0x1ffu, 1u}) {
      SCOPED_TRACE(version);
      SCOPED_TRACE(member_mask);
      DeviceCapabilityConfig caps;
      caps.set(DeviceCapability::spirv_version, version);
      caps.set(DeviceCapability::spirv_has_ray_query, 1);
      spirv::IRBuilder builder(Arch::vulkan, &caps);
      builder.init_header();
      const auto scene = builder.acceleration_structure_argument(0, 0);
      const auto zero = builder.float_immediate_number(builder.f32_type(), 0);
      const auto one = builder.float_immediate_number(builder.f32_type(), 1);
      const auto flags = builder.uint_immediate_number(
          builder.u32_type(), spv::RayFlagsOpaqueKHRMask);
      const auto mask = builder.uint_immediate_number(builder.u32_type(), 0xff);
      const auto function = builder.new_function();
      builder.start_function(function);
      builder.ray_query_closest(
          scene, {zero, zero, zero, zero, zero, one, zero, one, flags, mask},
          hit_type, member_mask);
      builder.make_inst(spv::OpReturn);
      builder.make_inst(spv::OpFunctionEnd);
      builder.commit_kernel_function(function, "main", {}, {1, 1, 1});
      const auto module = builder.finalize();

      std::string diagnostic;
      spvtools::SpirvTools validator(SPV_ENV_VULKAN_1_2);
      validator.SetMessageConsumer(
          [&](spv_message_level_t, const char *, const spv_position_t &,
              const char *message) { diagnostic += message; });
      EXPECT_TRUE(validator.Validate(module)) << diagnostic;
    }
  }
}

}  // namespace taichi::lang
#endif
