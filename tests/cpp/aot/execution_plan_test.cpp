#include "gtest/gtest.h"
#include "taichi/aot/execution_plan.h"

namespace taichi::lang::aot {
TEST(ExecutionPlan, PreservesOrderedOperationsAndSymbolicBindings) {
  const auto plan = ExecutionPlan::parse(R"({"version":1,"operations":[
    {"kind":"kernel","entry":"forward","arguments":["x","scratch"]},
    {"kind":"copy","source":"scratch","destination":"y","source_offset":4},
    {"kind":"fill32","destination":"scratch","value":1065353216},
    {"kind":"kernel","entry":"backward","arguments":["y","x"]}]})");
  ASSERT_EQ(plan.operations.size(), 4);
  EXPECT_EQ(plan.arguments, (std::vector<std::string>{"scratch", "x", "y"}));
  EXPECT_EQ(std::get<ExecutionPlan::Kernel>(plan.operations[0]).entry,
            "forward");
  const auto &copy = std::get<ExecutionPlan::Copy>(plan.operations[1]);
  EXPECT_EQ(copy.source_offset, 4);
  EXPECT_FALSE(copy.bytes);
  EXPECT_EQ(std::get<ExecutionPlan::Fill>(plan.operations[2]).value,
            1065353216);
}

TEST(ExecutionPlan, RejectsUnsupportedOrAmbiguousDescriptions) {
  for (
      const auto *document :
      {R"({"version":2,"operations":[]})", R"({"version":1,"operations":[]})",
       R"({"version":1,"operations":[{"kind":"callback"}]})",
       R"({"version":1,"operations":[{"kind":"kernel","entry":"x","arguments":[],"gated":true}]})",
       R"({"version":1,"operations":[{"kind":"kernel","entry":"x","arguments":[""]}]})",
       R"({"version":1,"operations":[{"kind":"copy","source":"x","destination":"y","size_bytes":-1}]})",
       R"({"version":1,"operations":[{"kind":"fill32","destination":"y","value":4294967296}]})",
       R"({"version":1,"version":1,"operations":[{"kind":"fill32","destination":"y","value":0}]})",
       R"({"version":1,"operations":[{"kind":"fill32","destination":"y","value":1,"value":0}]})",
       R"({"version":1,"operations":[{"kind":"kernel","entry":"x","arguments":[]}] } {})"}) {
    EXPECT_THROW(ExecutionPlan::parse(document), std::invalid_argument)
        << document;
  }
}

TEST(ExecutionPlan, NativeCompactRetainsAllSymbolicDependencies) {
  const auto plan = ExecutionPlan::parse(R"({"version":1,"operations":[
    {"kind":"compact","values":"x","flags":"active","output":"selected","count":"n"}]})");
  ASSERT_EQ(plan.operations.size(), 1);
  const auto &compact = std::get<ExecutionPlan::Compact>(plan.operations[0]);
  ASSERT_EQ(compact.columns.size(), 1);
  EXPECT_EQ(compact.columns[0].values, "x");
  EXPECT_EQ(compact.flags, "active");
  EXPECT_EQ(compact.columns[0].output, "selected");
  EXPECT_EQ(compact.count, "n");
  EXPECT_EQ(plan.arguments,
            (std::vector<std::string>{"active", "n", "selected", "x"}));
  EXPECT_THROW(ExecutionPlan::parse(R"({"version":1,"operations":[
    {"kind":"compact","values":"x","flags":"active","output":"selected"}]})"),
               std::invalid_argument);
}

TEST(ExecutionPlan, SortHasOptionalPayloadWithoutImplicitArguments) {
  const auto plan = ExecutionPlan::parse(R"({"version":1,"operations":[
    {"kind":"sort","keys":"x"},
    {"kind":"sort","keys":"x","values":"indices"}]})");
  ASSERT_EQ(plan.operations.size(), 2);
  EXPECT_FALSE(std::get<ExecutionPlan::Sort>(plan.operations[0]).values);
  EXPECT_EQ(*std::get<ExecutionPlan::Sort>(plan.operations[1]).values,
            "indices");
  EXPECT_EQ(plan.arguments, (std::vector<std::string>{"indices", "x"}));
  EXPECT_THROW(ExecutionPlan::parse(R"({"version":1,"operations":[
    {"kind":"sort","keys":"x","descending":true}]})"),
               std::invalid_argument);
}

TEST(ExecutionPlan, CompactColumnsShareOnePredicateAndCount) {
  const auto plan = ExecutionPlan::parse(R"({"version":1,"operations":[
    {"kind":"compact","flags":"active","count":"n","columns":[
      {"values":"keys","output":"selected_keys"},
      {"values":"values","output":"selected_values"}]}]})");
  const auto &compact = std::get<ExecutionPlan::Compact>(plan.operations[0]);
  ASSERT_EQ(compact.columns.size(), 2);
  EXPECT_EQ(compact.columns[1].values, "values");
  EXPECT_EQ(compact.columns[1].output, "selected_values");
  EXPECT_EQ(plan.arguments.size(), 6);
  for (
      const auto *description :
      {R"({"version":1,"operations":[{"kind":"compact","flags":"f","count":"n","columns":[]}]})",
       R"({"version":1,"operations":[{"kind":"compact","flags":"f","count":"n","columns":[{"values":"x"}]}]})",
       R"({"version":1,"operations":[{"kind":"compact","flags":"f","count":"n","columns":[{"values":"x","output":"y"}],"values":"z","output":"w"}]})"})
    EXPECT_THROW(ExecutionPlan::parse(description), std::invalid_argument);
}
TEST(ExecutionPlan, ReconstructsOnlyLiveResourceStorageAndShapeClosure) {
  const auto plan = ExecutionPlan::parse(R"({"version":1,
    "resources":[
      {"kind":"ndarray","name":"scratch","dtype":"f32","shape":[{"array":"shape_only","axis":0}]},
      {"kind":"extent","name":"n","dtype":"i32","array":"scratch","axis":0},
      {"kind":"ndarray","name":"shape_only","dtype":"f32","shape":[{"array":"x","axis":0},3]},
      {"kind":"ndarray","name":"unused","dtype":"f32","shape":[{"array":"not_required","axis":0}]},
      {"kind":"ndarray","name":"dx","dtype":"f32","shape":[{"array":"scratch","axis":0}]}],
    "gradients":{"scratch":"dx"},
    "operations":[{"kind":"kernel","entry":"forward","arguments":["scratch","n","y"]}]})");
  EXPECT_EQ(plan.arguments, (std::vector<std::string>{"x", "y"}));
  EXPECT_EQ(plan.resources.order, (std::vector<std::size_t>{2, 0, 1, 4}));
  EXPECT_EQ(plan.resources.materialize,
            (std::vector<bool>{true, true, false, false, true}));
}

TEST(ExecutionPlan, RejectsInvalidResourceGraphsBeforeBinding) {
  const std::string prefix =
      R"({"version":1,"operations":[{"kind":"fill32","destination":"x","value":0}],"resources":)";
  for (
      const auto *resources :
      {R"(null)",
       R"([{"kind":"ndarray","name":"x","dtype":"f32","shape":[true]}])",
       R"([{"kind":"ndarray","name":"x","dtype":"f32","shape":[],"element_shape":[0]}])",
       R"([{"kind":"extent","name":"n","dtype":"f32","array":"x","axis":0}])",
       R"([{"kind":"ndarray","name":"x","dtype":"f32","shape":[]},{"kind":"ndarray","name":"x","dtype":"f32","shape":[]}])",
       R"([{"kind":"ndarray","name":"x","dtype":"f32","shape":[{"array":"y","axis":0}]},{"kind":"ndarray","name":"y","dtype":"f32","shape":[{"array":"x","axis":0}]}])",
       R"([{"kind":"extent","name":"n","dtype":"i32","array":"input","axis":0},{"kind":"ndarray","name":"x","dtype":"f32","shape":[{"array":"n","axis":0}]}])"})
    EXPECT_THROW(ExecutionPlan::parse(prefix + resources + "}"),
                 std::invalid_argument);
}
}  // namespace taichi::lang::aot
