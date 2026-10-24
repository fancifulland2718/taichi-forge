#include "gtest/gtest.h"

#include "taichi/runtime/gfx/cooperative_matrix.h"
#include "taichi/rhi/common/spirv_capability.h"
#ifdef TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_device.h"
#endif

namespace taichi::lang {
namespace {

const std::vector<uint32_t> kCooperativeHeader{
    0x07230203, 0x10300, 0, 1, 0, 0x20011, 1, 0x20011, 6022};

spirv::TaskAttributes matrix_task() {
  spirv::TaskAttributes task;
  task.task_type = OffloadedTaskType::range_for;
  task.advisory_num_threads_per_group = 32;
  task.range_for_attribs = spirv::TaskAttributes::RangeForAttributes{};
  task.range_for_attribs->begin = RangeBound::constant(0);
  task.range_for_attribs->end = RangeBound::constant(64);
  task.cooperative_matrix_requirements.push_back({16, 16, 16, 32});
  return task;
}

TEST(CooperativeMatrixRequirements, CapabilityPrefixIsBounded) {
  EXPECT_TRUE(spirv_declares_capability(kCooperativeHeader.data(),
                                       kCooperativeHeader.size(), 6022));
  EXPECT_FALSE(spirv_declares_capability(nullptr, 0, 6022));
  auto code = kCooperativeHeader;
  code[5] = 0xffff0011;
  EXPECT_FALSE(spirv_declares_capability(code.data(), code.size(), 6022));
  code = kCooperativeHeader;
  code[5] = 0x2000e;  // A later integer is not a capability declaration.
  EXPECT_FALSE(spirv_declares_capability(code.data(), code.size(), 6022));
}

TEST(CooperativeMatrixRequirements, MetadataRequiresExactLaunchAndVersion) {
  auto task = matrix_task();
  EXPECT_NO_THROW(gfx::validate_cooperative_matrix_metadata(
      task, kCooperativeHeader, 8));
  EXPECT_THROW(gfx::validate_cooperative_matrix_metadata(
                   task, kCooperativeHeader, 7),
               std::invalid_argument);
  auto original = task;
  task.cooperative_matrix_requirements.clear();
  EXPECT_THROW(gfx::validate_cooperative_matrix_metadata(
                   task, kCooperativeHeader, 8),
               std::invalid_argument);
  task = original;
  task.cooperative_matrix_requirements[0].k = 0;
  EXPECT_THROW(gfx::validate_cooperative_matrix_metadata(
                   task, kCooperativeHeader, 8),
               std::invalid_argument);
  task = original;
  task.advisory_num_threads_per_group = 31;
  EXPECT_THROW(gfx::validate_cooperative_matrix_metadata(
                   task, kCooperativeHeader, 8),
               std::invalid_argument);
  task = original;
  task.range_for_attribs->end = RangeBound::constant(63);
  EXPECT_THROW(gfx::validate_cooperative_matrix_metadata(
                   task, kCooperativeHeader, 8),
               std::invalid_argument);
  EXPECT_THROW(gfx::validate_cooperative_matrix_device(
                   nullptr, original, kCooperativeHeader),
               std::invalid_argument);
  auto ordinary = original;
  ordinary.cooperative_matrix_requirements.clear();
  std::vector<uint32_t> code{0x07230203, 0x10300, 0, 1, 0};
  EXPECT_NO_THROW(
      gfx::validate_cooperative_matrix_metadata(ordinary, code, 2));
  EXPECT_THROW(gfx::validate_cooperative_matrix_metadata(original, code, 8),
               std::invalid_argument);
  auto encoded = liong::json::serialize(original);
  spirv::TaskAttributes decoded;
  liong::json::deserialize(encoded, decoded);
  EXPECT_EQ(decoded.cooperative_matrix_requirements,
            original.cooperative_matrix_requirements);
}

#ifdef TI_WITH_VULKAN
TEST(CooperativeMatrixRequirements, DeviceTupleAndSubgroupAreIndependent) {
  vulkan::VulkanCapabilities caps;
  caps.cooperative_matrix = true;
  caps.compute_full_subgroups = true;
  caps.subgroup_size = 32;
  caps.cooperative_matrix_supported_stages = VK_SHADER_STAGE_COMPUTE_BIT;
  caps.cooperative_matrix_properties.push_back(
      {16, 16, 16, VK_COMPONENT_TYPE_FLOAT16_KHR, VK_COMPONENT_TYPE_FLOAT16_KHR,
       VK_COMPONENT_TYPE_FLOAT32_KHR, VK_COMPONENT_TYPE_FLOAT32_KHR,
       VK_SCOPE_SUBGROUP_KHR, false});
  EXPECT_TRUE(caps.cooperative_matrix_f16_f32_valid(16, 16, 16, 32));
  EXPECT_FALSE(caps.cooperative_matrix_f16_f32_valid(16, 16, 16, 64));
  EXPECT_FALSE(caps.cooperative_matrix_f16_f32_valid(8, 16, 16, 32));
  caps.cooperative_matrix_properties[0].saturating_accumulation = true;
  EXPECT_FALSE(caps.cooperative_matrix_f16_f32_valid(16, 16, 16, 32));
  caps.cooperative_matrix_properties[0].saturating_accumulation = false;
  caps.compute_full_subgroups = false;
  EXPECT_FALSE(caps.cooperative_matrix_f16_f32_valid(16, 16, 16, 32));
  caps.compute_full_subgroups = true;
  caps.cooperative_matrix_supported_stages = VK_SHADER_STAGE_FRAGMENT_BIT;
  EXPECT_FALSE(caps.cooperative_matrix_f16_f32_valid(16, 16, 16, 32));
}
#endif

}  // namespace
}  // namespace taichi::lang
