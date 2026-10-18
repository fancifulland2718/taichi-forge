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

TEST(LaunchContextBuilderTest, DescriptorPointerWritesPreserveAdjacentScalar) {
  auto &types = TypeFactory::get_instance();
  const auto pointer = types.get_pointer_type(PrimitiveType::f32);
  for (const auto *layout : {"1-", "1b", "none"}) {
    const bool narrow = std::string(layout) == "1-";
    CallableBase callable;
    callable.args_type =
        types
            .get_struct_type(
                {{pointer, "pointer", 0},
                 {PrimitiveType::u32, "sentinel", narrow ? 4u : 8u}},
                layout)
            ->as<StructType>();
    callable.args_size = 16;
    LaunchContextBuilder builder(&callable);
    builder.set_struct_arg_impl<uint32_t>({1}, 0xabcdef01);
    builder.set_struct_arg<uint64_t>({0}, 0x1234567800000004ull);
    EXPECT_EQ(builder.get_struct_arg<uint32_t>({1}), 0xabcdef01);
    if (narrow)
      EXPECT_EQ(builder.get_struct_arg<uint32_t>({0}), 4);
    else
      EXPECT_EQ(builder.get_struct_arg<uint64_t>({0}), 0x1234567800000004ull);
  }
}

void set_test_execution_plan(Kernel &kernel,
                             const std::string &execution_identity,
                             int work_per_thread) {
  kernel.set_offload_execution_plan("compilation:test", execution_identity, {0},
                                    {"range_for"}, {0}, {"auto"}, {2}, {-1},
                                    {0}, {work_per_thread}, {"saturating"},
                                    {"direct"}, {{}}, {{}}, {{}}, {{}}, {});
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
