#include <gtest/gtest.h>
#include "taichi/aot/execution_plan.h"

namespace taichi::lang::aot {
namespace {
const std::string source = R"({"version":21,
  "operations":[{"kind":"kernel","entry":"assemble","arguments":["x","scratch","temp"]},
                {"kind":"copy","source":"temp","destination":"out"}],
  "resources":[
    {"kind":"native","name":"pool","provider":"memory","operation":"pool","version":1,"parameters":{},"bindings":{}},
    {"kind":"ndarray","name":"scratch","dtype":"f32","shape":[{"array":"shape","axis":0}],"allocator":"pool","zero_initialize":true},
    {"kind":"ndarray","name":"temp","dtype":"f32","shape":[{"array":"scratch","axis":0}],"allocator":"pool"},
    {"kind":"ndarray","name":"shape","dtype":"f32","shape":[3]},
    {"kind":"scalar","name":"__forge_aot_ring_0","dtype":"i32","bits":"00000000"}],
  "gradients":{"x":"dx"},
  "constraints":[
    {"kind":"array_layout","array":"scratch","exact_shape":[3],"record_bytes":4,"alignment":4},
    {"kind":"disjoint","left":"scratch","right":"temp"},
    {"kind":"disjoint_writes","reads":["scratch"],"writes":[{"array":"temp"}]},
    {"kind":"resource_alias","left":"scratch","right":"scratch","left_path":[],"right_path":[]},
    {"kind":"disjoint","left":"x","right":"out"}],
  "storage_recipe":{"kind":"ordered_temporary_ring","version":1,"capacity":3,"arrays":["scratch","temp"]}})";

std::string changed(const std::string &before, const std::string &after) {
  auto result = source;
  const auto position = result.find(before);
  if (position == std::string::npos)
    throw std::logic_error("Test replacement not found");
  result.replace(position, before.size(), after);
  return result;
}
}  // namespace

TEST(OrderedStoragePlan, ExpandsBindingsAndConstraintsButKeepsCodeAndAllocator) {
  const auto plan = ExecutionPlan::parse(source);
  ASSERT_TRUE(plan.storage_ring);
  EXPECT_EQ(plan.storage_ring->capacity, 3);
  EXPECT_EQ(plan.arguments, (std::vector<std::string>{"dx", "out", "x"}));
  EXPECT_EQ(plan.operations.size(), 6);
  EXPECT_EQ(plan.resources.declarations.size(), 9);
  EXPECT_EQ(plan.constraints.size(), 13);
  const auto &kernel = std::get<ExecutionPlan::Kernel>(plan.operations[2]);
  EXPECT_EQ(kernel.entry, "assemble");
  EXPECT_EQ(kernel.arguments,
            (std::vector<std::string>{"x", "__forge_aot_ring_1", "__forge_aot_ring_2"}));
  EXPECT_EQ(std::get<ExecutionPlan::Copy>(plan.operations[3]).source, kernel.arguments[2]);
  const auto &scratch = std::get<ExecutionResources::Array>(plan.resources.declarations[5].value);
  const auto &temp = std::get<ExecutionResources::Array>(plan.resources.declarations[6].value);
  EXPECT_TRUE(scratch.zero_initialize);
  EXPECT_EQ(scratch.allocator, "pool");
  EXPECT_EQ(temp.allocator, "pool");
  EXPECT_EQ(std::get<ExecutionResources::Extent>(scratch.shape[0]).array, "shape");
  EXPECT_EQ(std::get<ExecutionResources::Extent>(temp.shape[0]).array, kernel.arguments[1]);
  EXPECT_TRUE(plan.replay_bindings.count(kernel.arguments[1]));
  EXPECT_EQ(std::get<ExecutionPlan::ArrayLayout>(plan.constraints[5]).array, kernel.arguments[1]);
  EXPECT_EQ(std::get<ExecutionPlan::Disjoint>(plan.constraints[6]).right, kernel.arguments[2]);
  EXPECT_EQ(std::get<ExecutionPlan::DisjointWrites>(plan.constraints[7]).reads[0], kernel.arguments[1]);
  EXPECT_EQ(std::get<ExecutionPlan::ResourceAlias>(plan.constraints[8]).right, kernel.arguments[1]);
  const auto wide = ExecutionPlan::parse(changed("\"capacity\":3", "\"capacity\":65"));
  EXPECT_EQ(wide.operations.size(), 130);
  EXPECT_EQ(wide.resources.declarations.size(), 133);
}

TEST(OrderedStoragePlan, RejectsInvalidCapacityAndUnprovenSharedAliases) {
  for (const auto &[before, after] :
       std::vector<std::pair<std::string, std::string>>{
         {"\"version\":21", "\"version\":20"},
         {"\"version\":21", "\"version\":21,\"initialization\":[]"},
         {"\"version\":21", "\"version\":21,\"binding_strategy\":\"cuda_immutable_argument_frames\""},
         {"\"capacity\":3", "\"capacity\":0"},
         {"\"capacity\":3", "\"capacity\":true"},
         {"\"capacity\":3", "\"capacity\":3.0"},
         {"\"capacity\":3", "\"capacity\":9223372036854775807"},
         {"\"capacity\":3", "\"capacity\":3,\"unknown\":1"},
         {"ordered_temporary_ring", "unknown"},
         {"\"arrays\":[\"scratch\",\"temp\"]", "\"arrays\":[]"},
         {"\"arrays\":[\"scratch\",\"temp\"]", "\"arrays\":[\"scratch\",\"scratch\"]"},
         {"\"arrays\":[\"scratch\",\"temp\"]", "\"arrays\":[\"scratch\",\"x\"]"},
         {"\"arrays\":[\"scratch\",\"temp\"]", "\"arrays\":[\"scratch\"]"},
         {"\"arrays\":[\"scratch\",\"temp\"]", "\"arrays\":[\"scratch\",\"temp\",\"shape\"]"},
         {"\"gradients\":{\"x\":\"dx\"}", "\"gradients\":{\"x\":\"scratch\"}"}}) {
    const auto invalid = changed(before, after);
    EXPECT_THROW(ExecutionPlan::parse(invalid), std::invalid_argument) << invalid;
  }
}
}  // namespace taichi::lang::aot
