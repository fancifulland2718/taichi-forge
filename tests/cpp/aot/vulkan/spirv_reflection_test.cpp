#include "gtest/gtest.h"
#include "external/SPIRV-Reflect/spirv_reflect.h"
#include "spirv-tools/libspirv.hpp"

#include <future>
#include <regex>
#include <sstream>
#include <vector>

namespace {

std::vector<uint32_t> reflection_shader(uint32_t stride) {
  const std::string source = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %1 "main" %3
OpExecutionMode %1 LocalSize 8 1 1
OpDecorate %11 ArrayStride 4
OpDecorate %12 Block
OpMemberDecorate %12 0 Offset 0
OpDecorate %2 DescriptorSet 2
OpDecorate %2 Binding 7
OpDecorate %3 BuiltIn GlobalInvocationId
%4 = OpTypeVoid
%5 = OpTypeFunction %4
%10 = OpTypeInt 32 0
%11 = OpTypeRuntimeArray %10
%12 = OpTypeStruct %11
%13 = OpTypePointer StorageBuffer %12
%14 = OpTypePointer StorageBuffer %10
%15 = OpTypeVector %10 3
%16 = OpTypePointer Input %15
%17 = OpConstant %10 0
%18 = OpConstant %10 42
%2 = OpVariable %13 StorageBuffer
%3 = OpVariable %16 Input
%1 = OpFunction %4 None %5
%19 = OpLabel
%20 = OpLoad %15 %3
%21 = OpCompositeExtract %10 %20 0
%22 = OpAccessChain %14 %2 %17 %21
OpStore %22 %18
OpReturn
OpFunctionEnd
)";
  // Power-of-two strides deliberately collide in the private ID table. The
  // runtime index also references an instruction the reflector does not store.
  const std::regex id("%([0-9]+)");
  std::ostringstream remapped;
  size_t end = 0;
  for (auto it = std::sregex_iterator(source.begin(), source.end(), id);
       it != std::sregex_iterator(); ++it) {
    remapped << source.substr(end, it->position() - end) << '%'
             << std::stoul((*it)[1]) * stride;
    end = it->position() + it->length();
  }
  remapped << source.substr(end);
  spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_1);
  std::vector<uint32_t> words;
  EXPECT_TRUE(tools.Assemble(remapped.str(), &words,
                            SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS));
  EXPECT_TRUE(tools.Validate(words));
  return words;
}

bool reflect_expected(const std::vector<uint32_t> &words) {
  SpvReflectShaderModule module{};
  if (spvReflectCreateShaderModule(words.size() * sizeof(uint32_t),
                                  words.data(), &module) !=
      SPV_REFLECT_RESULT_SUCCESS) {
    return false;
  }
  const bool valid = module.descriptor_binding_count == 1 &&
                     module.descriptor_bindings[0].set == 2 &&
                     module.descriptor_bindings[0].binding == 7 &&
                     module.descriptor_bindings[0].descriptor_type ==
                         SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER &&
                     module.descriptor_bindings[0].count == 1 &&
                     module.entry_point_count == 1 &&
                     module.entry_points[0].local_size.x == 8;
  spvReflectDestroyShaderModule(&module);
  return valid;
}

}  // namespace

TEST(SpirvReflection, PreservesDynamicIndicesAndCollidingSparseIds) {
  for (uint32_t stride : {1u, 128u, 65536u}) {
    auto words = reflection_shader(stride);
    ASSERT_FALSE(words.empty());
    EXPECT_TRUE(reflect_expected(words));
    // Parsing storage must depend on actual nodes, not this untrusted bound.
    words[3] = 0xfffffff0u;
    EXPECT_TRUE(reflect_expected(words));
  }
}

TEST(SpirvReflection, KeepsConcurrentParserIndexesIndependent) {
  const auto words = reflection_shader(128);
  std::vector<std::future<bool>> workers;
  for (int i = 0; i < 8; ++i) {
    workers.emplace_back(std::async(std::launch::async, [&]() {
      for (int j = 0; j < 16; ++j) {
        if (!reflect_expected(words))
          return false;
      }
      return true;
    }));
  }
  for (auto &worker : workers)
    EXPECT_TRUE(worker.get());
}

TEST(SpirvReflection, RejectsMalformedInstructionsBeforeIndexConstruction) {
  auto words = reflection_shader(1);
  words[5] = 0;
  SpvReflectShaderModule module{};
  EXPECT_EQ(spvReflectCreateShaderModule(words.size() * sizeof(uint32_t),
                                         words.data(), &module),
            SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_INSTRUCTION);
}
