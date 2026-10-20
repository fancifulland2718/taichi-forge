#include <gtest/gtest.h>
#include <algorithm>
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

const ExecutionResources::Array &array(const ExecutionPlan &plan,
                                      const std::string &name) {
  const auto found = std::find_if(plan.resources.declarations.begin(),
                                  plan.resources.declarations.end(),
                                  [&](const auto &value) { return value.name == name; });
  if (found == plan.resources.declarations.end())
    throw std::logic_error("Expected a declared array: " + name);
  return std::get<ExecutionResources::Array>(found->value);
}

std::string grouped(const std::string &groups) {
  auto result = changed(
      R"("version":1,"capacity":3,"arrays":["scratch","temp"])",
      R"("version":2,"groups":)" + groups);
  result.replace(result.find(R"("version":21)"), 12, R"("version":22)");
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
  EXPECT_EQ(kernel.arguments[0], "x");
  EXPECT_NE(kernel.arguments[1], "scratch");
  EXPECT_NE(kernel.arguments[2], "temp");
  EXPECT_NE(kernel.arguments[1], kernel.arguments[2]);
  EXPECT_NE(kernel.arguments[1], "__forge_aot_ring_0");
  EXPECT_EQ(std::get<ExecutionPlan::Copy>(plan.operations[3]).source, kernel.arguments[2]);
  const auto &scratch = array(plan, kernel.arguments[1]);
  const auto &temp = array(plan, kernel.arguments[2]);
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

TEST(OrderedStoragePlan, StructureConstraintsFollowEachPrivateSlot) {
  const auto plan = ExecutionPlan::parse(R"({"version":24,
    "operations":[{"kind":"copy","source":"state","destination":"output"}],
    "resources":[{"kind":"ndarray","name":"state","layout":"particle","shape":[17]}],
    "constraints":[{"kind":"struct_array_layout","array":"state","layout":"particle","exact_shape":[17]}],
    "storage_recipe":{"kind":"ordered_temporary_ring","version":1,"capacity":3,"arrays":["state"]}})");
  ASSERT_EQ(plan.operations.size(), 3);
  ASSERT_EQ(plan.constraints.size(), 3);
  for (std::size_t i = 0; i < 3; ++i) {
    const auto &required =
        std::get<ExecutionPlan::StructArrayLayout>(plan.constraints[i]);
    EXPECT_EQ(required.array, std::get<ExecutionPlan::Copy>(plan.operations[i]).source);
    EXPECT_EQ(required.layout, "particle");
    EXPECT_EQ(required.exact_shape, (std::vector<std::uint64_t>{17}));
  }
}

TEST(OrderedStoragePlan, GroupsKeepTheirAllocationsWithoutProductFrames) {
  const auto plan = ExecutionPlan::parse(grouped(R"([
    {"capacity":3,"arrays":["scratch"]},
    {"capacity":5,"arrays":["temp"]}])"));
  ASSERT_TRUE(plan.storage_ring);
  EXPECT_EQ(plan.storage_ring->capacity, 5);
  EXPECT_EQ(plan.operations.size(), 10);
  EXPECT_EQ(plan.resources.declarations.size(), 11);
  EXPECT_EQ(plan.constraints.size(), 21);
  EXPECT_EQ(plan.arguments, (std::vector<std::string>{"dx", "out", "x"}));
  std::vector<std::string> scratch, temp;
  for (std::size_t frame = 0; frame < 5; ++frame) {
    const auto &call = std::get<ExecutionPlan::Kernel>(plan.operations[frame * 2]);
    EXPECT_EQ(call.entry, "assemble");
    scratch.push_back(call.arguments[1]);
    temp.push_back(call.arguments[2]);
    EXPECT_EQ(std::get<ExecutionPlan::Copy>(plan.operations[frame * 2 + 1]).source,
              temp.back());
    EXPECT_EQ(array(plan, scratch.back()).allocator, "pool");
    EXPECT_EQ(std::get<ExecutionResources::Extent>(array(plan, temp.back()).shape[0]).array,
              scratch.back());
  }
  EXPECT_EQ(std::set<std::string>(scratch.begin(), scratch.end()).size(), 3);
  EXPECT_EQ(std::set<std::string>(temp.begin(), temp.end()).size(), 5);
  EXPECT_EQ(scratch[3], scratch[0]);
  EXPECT_EQ(scratch[4], scratch[1]);
  for (std::size_t frame = 1; frame < 5; ++frame) {
    const auto &constraint = std::get<ExecutionPlan::Disjoint>(plan.constraints[5 + (frame - 1) * 4 + 1]);
    EXPECT_EQ(constraint.left, scratch[frame]);
    EXPECT_EQ(constraint.right, temp[frame]);
  }
  const auto wide = ExecutionPlan::parse(grouped(R"([
    {"capacity":3,"arrays":["scratch"]},
    {"capacity":65,"arrays":["temp"]}])"));
  EXPECT_EQ(wide.storage_ring->capacity, 65);
  EXPECT_EQ(wide.operations.size(), 130);
  EXPECT_EQ(wide.resources.declarations.size(), 71);
}

TEST(OrderedStoragePlan, GroupedDeclarationsRejectAmbiguousOwnership) {
  for (const auto *groups : {
        R"([])", R"({})", R"([{"capacity":3,"arrays":[]}])",
        R"([{"capacity":3,"arrays":["scratch","temp"]},{"capacity":5,"arrays":["temp"]}])",
        R"([{"capacity":0,"arrays":["scratch","temp"]}])",
        R"([{"capacity":true,"arrays":["scratch","temp"]}])",
        R"([{"capacity":3,"arrays":["scratch","temp"],"extra":1}])",
        R"([{"capacity":9223372036854775807,"arrays":["scratch","temp"]}])"}) {
    EXPECT_THROW(ExecutionPlan::parse(grouped(groups)), std::invalid_argument) << groups;
  }
  auto legacy = grouped(R"([{"capacity":3,"arrays":["scratch","temp"]}])");
  legacy.replace(legacy.find(R"("version":22)"), 12, R"("version":21)");
  EXPECT_THROW(ExecutionPlan::parse(legacy), std::invalid_argument);
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
