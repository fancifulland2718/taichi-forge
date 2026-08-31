#include "gtest/gtest.h"

#include "taichi/program/callable.h"
#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST
#include "taichi/program/kernel.h"
#include "taichi/program/launch_context_builder.h"
#include "taichi/program/program.h"

namespace taichi::lang {
namespace {

TEST(LaunchContextBuilderTest, HasNoCudaTaskExecutionPlanByDefault) {
  CallableBase callable;
  LaunchContextBuilder builder(&callable);

  EXPECT_FALSE(builder.has_cuda_task_execution_plan());
  EXPECT_TRUE(builder.cuda_task_execution_plan_identity().empty());
  EXPECT_TRUE(builder.cuda_task_execution_plan_kinds().empty());
  EXPECT_TRUE(builder.cuda_task_grid_residency_waves().empty());
  EXPECT_TRUE(
      builder.cuda_task_range_work_per_thread_targets().empty());
}

void set_test_execution_plan(Kernel &kernel,
                             const std::string &execution_identity,
                             int work_per_thread) {
  kernel.set_offload_execution_plan(
      "compilation:test", execution_identity, {0}, {"range_for"}, {0},
      {"auto"}, {2}, {-1}, {0}, {work_per_thread}, {"direct"});
}

TEST(LaunchContextBuilderTest, FreezesBorrowedKernelExecutionPlan) {
  Program program(Arch::x64);
  Kernel kernel(program, [] {}, "execution_plan_freeze_test");

  // Replacing a cold materialization is valid because no launch context can
  // yet hold references into the plan.
  set_test_execution_plan(kernel, "execution:test:one", 1);
  set_test_execution_plan(kernel, "execution:test:two", 2);

  auto builder = kernel.make_launch_context();
  ASSERT_TRUE(builder.has_cuda_task_execution_plan());
  EXPECT_EQ(builder.cuda_task_execution_plan_identity(),
            "execution:test:two");
  ASSERT_EQ(builder.cuda_task_range_work_per_thread_targets().size(), 1);
  EXPECT_EQ(builder.cuda_task_range_work_per_thread_targets()[0], 2);

  EXPECT_THROW(set_test_execution_plan(kernel, "execution:test:three", 4),
               std::string);
  EXPECT_EQ(builder.cuda_task_execution_plan_identity(),
            "execution:test:two");
  EXPECT_EQ(builder.cuda_task_range_work_per_thread_targets()[0], 2);
}

}  // namespace
}  // namespace taichi::lang
