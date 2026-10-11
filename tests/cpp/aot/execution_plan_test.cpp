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
  EXPECT_EQ(compact.values, "x");
  EXPECT_EQ(compact.flags, "active");
  EXPECT_EQ(compact.output, "selected");
  EXPECT_EQ(compact.count, "n");
  EXPECT_EQ(plan.arguments,
            (std::vector<std::string>{"active", "n", "selected", "x"}));
  EXPECT_THROW(ExecutionPlan::parse(R"({"version":1,"operations":[
    {"kind":"compact","values":"x","flags":"active","output":"selected"}]})"),
               std::invalid_argument);
}
}  // namespace taichi::lang::aot
