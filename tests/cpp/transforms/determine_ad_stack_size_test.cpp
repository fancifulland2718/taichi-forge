#include "gtest/gtest.h"

#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/ir/transforms.h"
#include "taichi/program/program.h"

namespace taichi::lang {

TEST(AdStackLayoutTest, IncludesCounterAndBothValuesAtFullCapacity) {
  AdStackAllocaStmt f32_stack(PrimitiveType::f32, 2);
  AdStackAllocaStmt f64_stack(PrimitiveType::f64, 3);
  AdStackAllocaStmt tensor_stack(
      TypeFactory::create_tensor_type({2, 3}, PrimitiveType::f32), 2);
  EXPECT_EQ(f32_stack.size_in_bytes(), 24);
  EXPECT_EQ(f64_stack.size_in_bytes(), 56);
  EXPECT_EQ(tensor_stack.size_in_bytes(), 104);
  EXPECT_EQ(AdStackLayout::header_size, sizeof(AdStackLayout::Counter));
  EXPECT_EQ(AdStackLayout::header_size % alignof(double), 0);
}

class DetermineAdStackSizeTest
    : public ::testing::TestWithParam<std::tuple<int, int>> {
 protected:
  void SetUp() override {
    prog_ = std::make_unique<Program>();
    prog_->materialize_runtime();
  }

  std::unique_ptr<Program> prog_;
};

TEST_F(DetermineAdStackSizeTest, Basic) {
  IRBuilder builder;
  auto *stack =
      builder.create_ad_stack(get_data_type<int>(), 0 /*adaptive size*/);
  builder.ad_stack_push(stack, builder.get_int32(1));
  builder.ad_stack_push(stack, builder.get_int32(2));
  builder.ad_stack_push(stack, builder.get_int32(3));
  builder.ad_stack_pop(stack);
  builder.ad_stack_pop(stack);
  builder.ad_stack_push(stack, builder.get_int32(4));
  builder.ad_stack_push(stack, builder.get_int32(5));
  builder.ad_stack_push(stack, builder.get_int32(6));
  // stack contains [1, 4, 5, 6] now
  builder.ad_stack_pop(stack);
  builder.ad_stack_pop(stack);
  builder.ad_stack_push(stack, builder.get_int32(7));

  auto *stack2 =
      builder.create_ad_stack(get_data_type<int>(), 0 /*adaptive size*/);
  builder.ad_stack_push(stack2, builder.get_int32(8));

  auto ir = builder.extract_ir();
  ASSERT_TRUE(ir->is<Block>());
  auto *ir_block = ir->as<Block>();
  irpass::type_check(ir_block, CompileConfig());

  EXPECT_EQ(stack->max_size, 0);
  EXPECT_EQ(stack2->max_size, 0);
  irpass::determine_ad_stack_size(ir_block, CompileConfig());
  EXPECT_EQ(stack->max_size, 4);
  EXPECT_EQ(stack->size_in_bytes(), 40);
  EXPECT_EQ(stack2->max_size, 1);
}

TEST_F(DetermineAdStackSizeTest, Loop) {
  IRBuilder builder;
  auto *stack =
      builder.create_ad_stack(get_data_type<int>(), 0 /*adaptive size*/);
  auto *loop = builder.create_range_for(/*begin=*/builder.get_int32(0),
                                        /*end=*/builder.get_int32(10));
  {
    auto _ = builder.get_loop_guard(loop);
    builder.ad_stack_push(stack, builder.get_int32(1));
    builder.ad_stack_pop(stack);
  }

  auto ir = builder.extract_ir();
  ASSERT_TRUE(ir->is<Block>());
  auto *ir_block = ir->as<Block>();
  irpass::type_check(ir_block, CompileConfig());

  EXPECT_EQ(stack->max_size, 0);
  irpass::determine_ad_stack_size(ir_block, CompileConfig());
  EXPECT_EQ(stack->max_size, 1);
}

TEST_F(DetermineAdStackSizeTest, FinitePositiveLoop) {
  IRBuilder builder;
  auto *stack =
      builder.create_ad_stack(get_data_type<int>(), 0 /*adaptive size*/);
  auto *loop = builder.create_range_for(/*begin=*/builder.get_int32(0),
                                        /*end=*/builder.get_int32(100));
  {
    auto _ = builder.get_loop_guard(loop);
    builder.ad_stack_push(stack, builder.get_int32(1));
  }

  auto ir = builder.extract_ir();
  ASSERT_TRUE(ir->is<Block>());
  auto *ir_block = ir->as<Block>();
  irpass::type_check(ir_block, CompileConfig());

  CompileConfig config;
  constexpr int kDefaultAdStackSize = 32;
  config.default_ad_stack_size = kDefaultAdStackSize;
  EXPECT_EQ(stack->max_size, 0);
  irpass::determine_ad_stack_size(ir_block, config);
  EXPECT_EQ(stack->max_size, 100);
}

TEST_F(DetermineAdStackSizeTest, NestedLoopsAndStackLifetime) {
  IRBuilder builder;
  auto *stack = builder.create_ad_stack(get_data_type<float>(), 0);
  builder.ad_stack_push(stack, builder.get_float32(0));
  auto *outer = builder.create_range_for(builder.get_int32(0), builder.get_int32(2));
  AdStackAllocaStmt *local;
  {
    auto outer_guard = builder.get_loop_guard(outer);
    local = builder.create_ad_stack(get_data_type<float>(), 0);
    auto *inner = builder.create_range_for(builder.get_int32(0), builder.get_int32(2));
    {
      auto inner_guard = builder.get_loop_guard(inner);
      for (int i = 0; i < 10; ++i) {
        builder.ad_stack_push(stack, builder.get_float32(1));
        builder.ad_stack_push(local, builder.get_float32(1));
      }
    }
  }
  auto ir = builder.extract_ir();
  irpass::type_check(ir.get(), CompileConfig());
  irpass::determine_ad_stack_size(ir.get(), CompileConfig());
  EXPECT_EQ(stack->max_size, 41);
  EXPECT_EQ(local->max_size, 20);  // reinitialized in each outer iteration
}

TEST_F(DetermineAdStackSizeTest, EmptyLoopAndDrainedStack) {
  IRBuilder builder;
  auto *stack = builder.create_ad_stack(get_data_type<int>(), 0);
  auto *empty = builder.create_range_for(builder.get_int32(4), builder.get_int32(2));
  {
    auto guard = builder.get_loop_guard(empty);
    for (int i = 0; i < 10; ++i)
      builder.ad_stack_push(stack, builder.get_int32(1));
  }
  auto *push = builder.create_range_for(builder.get_int32(0), builder.get_int32(64));
  {
    auto guard = builder.get_loop_guard(push);
    builder.ad_stack_push(stack, builder.get_int32(1));
  }
  auto *pop = builder.create_range_for(builder.get_int32(0), builder.get_int32(64));
  pop->reversed = true;
  {
    auto guard = builder.get_loop_guard(pop);
    builder.ad_stack_pop(stack);
  }
  builder.ad_stack_push(stack, builder.get_int32(2));
  auto ir = builder.extract_ir();
  irpass::type_check(ir.get(), CompileConfig());
  irpass::determine_ad_stack_size(ir.get(), CompileConfig());
  EXPECT_EQ(stack->max_size, 64);
}

TEST_F(DetermineAdStackSizeTest, DynamicFallbackPreservesResolvedAndFixedStacks) {
  IRBuilder builder;
  auto *fixed = builder.create_ad_stack(get_data_type<int>(), 17);
  auto *bounded = builder.create_ad_stack(get_data_type<int>(), 0);
  auto *dynamic = builder.create_ad_stack(get_data_type<int>(), 0);
  builder.ad_stack_push(fixed, builder.get_int32(1));
  builder.ad_stack_push(bounded, builder.get_int32(1));
  auto *end = builder.create_arg_load({0}, get_data_type<int>(), false, 0);
  auto *loop = builder.create_range_for(builder.get_int32(0), end);
  {
    auto guard = builder.get_loop_guard(loop);
    builder.ad_stack_push(dynamic, builder.get_int32(1));
  }
  auto ir = builder.extract_ir();
  CompileConfig config;
  config.default_ad_stack_size = 7;
  irpass::type_check(ir.get(), config);
  irpass::determine_ad_stack_size(ir.get(), config);
  EXPECT_EQ(fixed->max_size, 17);
  EXPECT_EQ(bounded->max_size, 1);
  EXPECT_EQ(dynamic->max_size, 7);
}

TEST_F(DetermineAdStackSizeTest, ContinueCanSkipStackPop) {
  IRBuilder builder;
  auto *stack = builder.create_ad_stack(get_data_type<int>(), 0);
  auto *cond = builder.create_arg_load({0}, get_data_type<int>(), false, 0);
  auto *loop = builder.create_range_for(builder.get_int32(0), builder.get_int32(4));
  {
    auto guard = builder.get_loop_guard(loop);
    builder.ad_stack_push(stack, builder.get_int32(1));
    auto *branch = builder.create_if(cond);
    {
      auto branch_guard = builder.get_if_guard(branch, true);
      builder.create_continue();
    }
    builder.ad_stack_pop(stack);
  }
  auto ir = builder.extract_ir();
  CompileConfig config;
  config.default_ad_stack_size = 7;
  irpass::type_check(ir.get(), config);
  irpass::determine_ad_stack_size(ir.get(), config);
  // The structured net delta looks balanced, but continue skips the pop.
  // Leave early-exit paths to the CFG fallback until they are modeled.
  EXPECT_EQ(stack->max_size, 7);
}

TEST_F(DetermineAdStackSizeTest, StackFreeHelperLoopPreservesOuterCapacity) {
  IRBuilder builder;
  auto *stack = builder.create_ad_stack(get_data_type<int>(), 0);
  auto *loop = builder.create_range_for(builder.get_int32(0), builder.get_int32(40));
  {
    auto guard = builder.get_loop_guard(loop);
    builder.ad_stack_push(stack, builder.get_int32(1));
    auto *helper = builder.create_while_true();
    {
      auto helper_guard = builder.get_loop_guard(helper);
      builder.create_break();
    }
  }
  auto ir = builder.extract_ir();
  CompileConfig config;
  config.default_ad_stack_size = 7;
  irpass::type_check(ir.get(), config);
  irpass::determine_ad_stack_size(ir.get(), config);
  EXPECT_EQ(stack->max_size, 40);
}

TEST_P(DetermineAdStackSizeTest, If) {
  constexpr int kCommonPushes = 1;
  const int kTrueBranchPushes = std::get<0>(GetParam());
  const int kFalseBranchPushes = std::get<1>(GetParam());
  bool has_true_branch = (kTrueBranchPushes > 0);
  bool has_false_branch = (kFalseBranchPushes > 0);

  IRBuilder builder;
  auto *arg = builder.create_arg_load({0}, get_data_type<int>(), false, 0);
  auto *stack =
      builder.create_ad_stack(get_data_type<int>(), 0 /*adaptive size*/);
  auto *if_stmt = builder.create_if(arg);
  auto *one = builder.get_int32(1);
  for (int i = 1; i <= kCommonPushes; i++) {
    builder.ad_stack_push(stack, one);  // Make sure the stack is not unused
  }
  if (has_true_branch) {
    auto _ = builder.get_if_guard(if_stmt, true);
    for (int i = 1; i <= kTrueBranchPushes; i++) {
      builder.ad_stack_push(stack, one);
    }
  }
  if (has_false_branch) {
    auto _ = builder.get_if_guard(if_stmt, false);
    for (int i = 1; i <= kFalseBranchPushes; i++) {
      builder.ad_stack_push(stack, one);
    }
  }

  auto ir = builder.extract_ir();
  ASSERT_TRUE(ir->is<Block>());
  auto *ir_block = ir->as<Block>();
  irpass::type_check(ir_block, CompileConfig());
  EXPECT_EQ(irpass::analysis::count_statements(ir_block),
            4 /*arg_load, stack, if, one*/ + kCommonPushes +
                has_true_branch * kTrueBranchPushes +
                has_false_branch * kFalseBranchPushes);

  EXPECT_EQ(stack->max_size, 0);
  irpass::determine_ad_stack_size(ir_block, CompileConfig());
  EXPECT_EQ(stack->max_size,
            kCommonPushes + std::max(has_true_branch * kTrueBranchPushes,
                                     has_false_branch * kFalseBranchPushes));
}

INSTANTIATE_TEST_SUITE_P(
    Parameterized,
    DetermineAdStackSizeTest,
    testing::Combine(testing::Values(0, 3), testing::Values(0, 4)),
    [](const testing::TestParamInfo<DetermineAdStackSizeTest::ParamType>
           &info) {
      return fmt::format("True{}_False{}", std::get<0>(info.param),
                         std::get<1>(info.param));
    });

TEST_F(DetermineAdStackSizeTest, EmptyNodes) {
  IRBuilder builder;
  auto *arg = builder.create_arg_load({0}, get_data_type<int>(), false, 0);
  auto *stack =
      builder.create_ad_stack(get_data_type<int>(), 0 /*adaptive size*/);
  auto *one = builder.get_int32(1);
  builder.ad_stack_push(stack, one);  // stack contains [1] now
  auto *if_stmt = builder.create_if(arg);
  {
    auto _ = builder.get_if_guard(if_stmt, true);
    builder.get_int32(2);  // avoid CFGNode being deleted
  }
  {
    auto _ = builder.get_if_guard(if_stmt, false);
    builder.get_int32(3);  // avoid CFGNode being deleted
  }
  builder.ad_stack_push(stack, one);  // stack contains [1, 1] now

  auto ir = builder.extract_ir();
  ASSERT_TRUE(ir->is<Block>());
  auto *ir_block = ir->as<Block>();
  irpass::type_check(ir_block, CompileConfig());

  EXPECT_EQ(stack->max_size, 0);
  irpass::determine_ad_stack_size(ir_block, CompileConfig());
  EXPECT_EQ(stack->max_size, 2);
}

}  // namespace taichi::lang
