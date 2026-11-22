#include "gtest/gtest.h"

#ifdef TI_WITH_VULKAN
#include "spirv-tools/libspirv.hpp"
#include "taichi/codegen/spirv/spirv_ir_builder.h"
#include "taichi/rhi/vulkan/shader_specialization.h"

namespace taichi::lang {

TEST(SpirvArrayLayout, PreparePrunesGuardedWorkgroupAccess) {
  // The bound count determines both the guard and the last valid index.
  // Keep the result observable through a separate storage buffer.
  spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
  std::vector<uint32_t> input;
  ASSERT_TRUE(tools.Assemble(R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %args %shared %out
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %args DescriptorSet 0
OpDecorate %args Binding 0
OpDecorate %out DescriptorSet 0
OpDecorate %out Binding 1
OpDecorate %argsType Block
OpMemberDecorate %argsType 0 Offset 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%i32 = OpTypeInt 32 1
%zero = OpConstant %i32 0
%one = OpConstant %i32 1
%four = OpConstant %i32 4
%argsType = OpTypeStruct %i32
%argsPtr = OpTypePointer Uniform %argsType
%uniformInt = OpTypePointer Uniform %i32
%localBool = OpTypePointer Function %bool
%array = OpTypeArray %i32 %four
%arrayPtr = OpTypePointer Workgroup %array
%elementPtr = OpTypePointer Workgroup %i32
%outPtr = OpTypePointer StorageBuffer %argsType
%outInt = OpTypePointer StorageBuffer %i32
%out = OpVariable %outPtr StorageBuffer
%args = OpVariable %argsPtr Uniform
%shared = OpVariable %arrayPtr Workgroup
%main = OpFunction %void None %fn
%entry = OpLabel
%guard = OpVariable %localBool Function
%member = OpAccessChain %uniformInt %args %zero
%n = OpLoad %i32 %member
%enabled = OpSGreaterThan %bool %n %zero
OpStore %guard %enabled
%active = OpLoad %bool %guard
OpSelectionMerge %end None
OpBranchConditional %active %body %end
%body = OpLabel
%idx = OpISub %i32 %n %one
%ptr = OpAccessChain %elementPtr %shared %idx
%val = OpLoad %i32 %ptr
%to = OpAccessChain %outInt %out %zero
OpStore %to %val
OpBranch %end
%end = OpLabel
OpReturn
OpFunctionEnd
)", &input));
  ASSERT_TRUE(tools.Validate(input));
  // Layout-only binding keeps the branch and the workgroup/output addresses.
  // The folded count must no longer leave an unused UBO address behind.
  size_t layout_folds = 0;
  const auto bound = vulkan::bind_spirv_layout_words(input, 0, {{0, 32, 2}}, &layout_folds);
  ASSERT_EQ(layout_folds, 1u);
  ASSERT_TRUE(tools.Validate(bound));
  size_t addresses = 0;
  for (size_t pos = 5; pos < bound.size(); pos += bound[pos] >> 16) {
    if ((bound[pos] & 0xffffu) == spv::OpAccessChain)
      ++addresses;
  }
  EXPECT_EQ(addresses, 2u);
  for (uint32_t count : {0u, 2u}) {
    size_t folded = 0;
    const auto shader = vulkan::specialize_spirv_arguments(
        input, 0, {{0, 32, count}}, &folded);
    ASSERT_EQ(folded, 1u);
    EXPECT_TRUE(tools.Validate(shader));
    size_t loads = 0;
    for (size_t pos = 5; pos < shader.size(); pos += shader[pos] >> 16) {
      if ((shader[pos] & 0xffffu) == spv::OpLoad)
        ++loads;
    }
    EXPECT_EQ(loads, count ? 1u : 0u);
  }
}

TEST(SpirvArrayLayout, DispatchUniformReuseIsLocalToOneBlockAndBinding) {
  DeviceCapabilityConfig caps;
  caps.set(DeviceCapability::spirv_version, 0x10300u);
  spirv::IRBuilder builder(Arch::vulkan, &caps);
  builder.init_header();
  const auto type = builder.i32_type();
  std::vector<std::tuple<spirv::SType, std::string, size_t>> members = {
      {type, "x", 0}, {type, "y", 4}};
  auto structure = builder.create_struct_type(members);
  auto args = builder.uniform_struct_argument(structure, 0, 0, "args");
  auto other = builder.uniform_struct_argument(structure, 0, 1, "other");
  builder.reuse_dispatch_arguments(args);
  builder.start_function(builder.new_function());
  auto read = [&](spirv::Value base, int index) {
    auto pointer = builder.make_access_chain(
        builder.get_pointer_type(type, spv::StorageClassUniform), base,
        std::vector<int>{index});
    return builder.load_variable(pointer, type);
  };
  auto first = read(args, 0);
  EXPECT_EQ(first.id, read(args, 0).id);
  EXPECT_NE(first.id, read(args, 1).id);
  auto uncached = read(other, 0);
  EXPECT_NE(uncached.id, read(other, 0).id);
  auto next = builder.new_label();
  builder.make_inst(spv::OpBranch, next);
  builder.start_label(next);
  EXPECT_NE(first.id, read(args, 0).id);
}

TEST(SpirvArrayLayout, RandomInitializationAllowsLaterFunctionLocals) {
  DeviceCapabilityConfig caps;
  caps.set(DeviceCapability::spirv_version, 0x10300u);
  spirv::IRBuilder builder(Arch::vulkan, &caps);
  builder.init_header();
  const auto function = builder.new_function();
  const auto scratch = builder.buffer_argument(builder.u32_type(), 0, 0, "scratch");
  builder.start_function(function);
  builder.get_global_invocation_id(0);
  const auto entry = builder.current_label();
  const auto body = builder.new_label();
  builder.make_inst(spv::OpBranch, body);
  builder.start_label(body);
  auto phi = builder.make_phi(builder.u32_type(), 1);
  const auto value = builder.rand_u32(scratch);
  phi.set_incoming(0, builder.uint_immediate_number(builder.u32_type(), 0), entry);
  builder.store_variable(builder.alloca_variable(builder.u32_type()), value);
  builder.make_inst(spv::OpReturn);
  builder.make_inst(spv::OpFunctionEnd);
  builder.commit_kernel_function(function, "main", {}, {1, 1, 1});
  const auto module = builder.finalize();
  spvtools::SpirvTools validator(SPV_ENV_VULKAN_1_2);
  EXPECT_TRUE(validator.Validate(module));
}

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
