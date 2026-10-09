#include "gtest/gtest.h"

#ifdef TI_WITH_VULKAN
#include "spirv/unified1/spirv.h"
#include "spirv-tools/libspirv.hpp"
#include "taichi/codegen/spirv/spirv_redundancy_elimination.h"

namespace taichi::lang {
namespace {

const std::string header = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%int = OpTypeInt 32 1
%bool = OpTypeBool
%ptr = OpTypePointer Function %int
%zero = OpConstant %int 0
%one = OpConstant %int 1
%two = OpConstant %int 2
%main = OpFunction %void None %fn
%entry = OpLabel
%input = OpVariable %ptr Function
%x = OpLoad %int %input
)";

std::vector<uint32_t> compare_with_upstream(const std::string &text) {
  spvtools::SpirvTools tools(SPV_ENV_UNIVERSAL_1_3);
  std::vector<uint32_t> input, expected, actual;
  EXPECT_TRUE(tools.Assemble(text, &input));
  EXPECT_TRUE(tools.Validate(input));
  spvtools::Optimizer reference(SPV_ENV_UNIVERSAL_1_3);
  reference.RegisterPass(spvtools::CreateRedundancyEliminationPass());
  EXPECT_TRUE(reference.Run(input.data(), input.size(), &expected));
  spvtools::Optimizer candidate(SPV_ENV_UNIVERSAL_1_3);
  candidate.RegisterPass(spirv::create_scoped_redundancy_elimination_pass());
  EXPECT_TRUE(candidate.Run(input.data(), input.size(), &actual));
  EXPECT_TRUE(tools.Validate(actual));
  EXPECT_EQ(actual, expected);
  return actual;
}

}  // namespace

TEST(SpirvRedundancyElimination, SiblingDefinitionsDoNotEscapeTheirScope) {
  const auto output = compare_with_upstream(header + R"(
%common = OpIAdd %int %x %one
%condition = OpIEqual %bool %x %zero
OpSelectionMerge %merge None
OpBranchConditional %condition %left %right
%left = OpLabel
%left_add = OpIAdd %int %x %one
%left_mul = OpIMul %int %left_add %two
OpBranch %merge
%right = OpLabel
%right_add = OpIAdd %int %x %one
%right_mul = OpIMul %int %right_add %two
OpBranch %merge
%merge = OpLabel
%result = OpPhi %int %left_mul %left %right_mul %right
OpStore %input %result
OpReturn
OpFunctionEnd
)");
  int adds = 0, multiplies = 0;
  for (size_t i = 5; i < output.size(); i += output[i] >> 16) {
    adds += (output[i] & 0xffffu) == SpvOpIAdd;
    multiplies += (output[i] & 0xffffu) == SpvOpIMul;
  }
  EXPECT_EQ(adds, 1);
  EXPECT_EQ(multiplies, 2);
}

TEST(SpirvRedundancyElimination, LoopPhisAndBackedgesKeepTheirDefinitions) {
  compare_with_upstream(header + R"(
%common = OpIAdd %int %x %one
OpBranch %header
%header = OpLabel
%i = OpPhi %int %zero %entry %next %continue
%condition = OpSLessThan %bool %i %two
OpLoopMerge %exit %continue None
OpBranchConditional %condition %body %exit
%body = OpLabel
%duplicate = OpIAdd %int %x %one
%product = OpIMul %int %duplicate %i
OpStore %input %product
OpBranch %continue
%continue = OpLabel
%next = OpIAdd %int %i %one
OpBranch %header
%exit = OpLabel
OpReturn
OpFunctionEnd
)");
}

TEST(SpirvRedundancyElimination, DeepDominatorChain) {
  std::string text = header + "%common = OpIAdd %int %x %one\n";
  for (int i = 0; i < 256; ++i) {
    const auto index = std::to_string(i);
    text += "OpBranch %block" + index + "\n%block" + index + " = OpLabel\n";
    text += "%add" + index + " = OpIAdd %int %x %one\n";
  }
  compare_with_upstream(text + "OpReturn\nOpFunctionEnd\n");
}

}  // namespace taichi::lang
#endif
