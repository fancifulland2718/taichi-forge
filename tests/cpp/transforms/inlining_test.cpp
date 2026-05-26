#include "gtest/gtest.h"

#include "taichi/ir/analysis.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/program/program.h"

namespace taichi::lang {

class InliningTest : public ::testing::Test {
 protected:
  void SetUp() override {
    prog_ = std::make_unique<Program>();
    prog_->materialize_runtime();
  }

  std::unique_ptr<Program> prog_;
};

namespace {

int count_func_calls(IRNode *node) {
  return static_cast<int>(irpass::analysis::gather_statements(
                              node, [](Stmt *s) { return s->is<FuncCallStmt>(); })
                              .size());
}

}  // namespace
TEST_F(InliningTest, ArgLoadOfArgLoad) {
  IRBuilder builder;
  // def test_func(x: ti.i32) -> ti.i32:
  //     return x + 1
  auto *arg = builder.create_arg_load(/*arg_id=*/{0}, get_data_type<int>(),
                                      /*is_ptr=*/false, /*arg_depth=*/0);
  auto *sum = builder.create_add(arg, builder.get_int32(1));
  builder.create_return(sum);
  auto func_body = builder.extract_ir();
  EXPECT_TRUE(func_body->is<Block>());
  auto *func_block = func_body->as<Block>();
  EXPECT_EQ(func_block->size(), 4);

  auto *func = prog_->create_function(
      FunctionKey("test_func", /*func_id=*/0, /*instance_id=*/0));
  func->insert_scalar_param(get_data_type<int>());
  func->insert_ret(get_data_type<int>());
  func->set_function_body(std::move(func_body));
  func->finalize_params();
  func->finalize_rets();

  // def kernel(x: ti.i32) -> ti.i32:
  //     return test_func(x)
  auto *kernel_arg =
      builder.create_arg_load(/*arg_id=*/{0}, get_data_type<int>(),
                              /*is_ptr=*/false, /*arg_depth=*/0);
  auto *func_call = builder.create_func_call(func, {kernel_arg});
  builder.create_return(func_call);
  auto kernel_body = builder.extract_ir();
  EXPECT_TRUE(kernel_body->is<Block>());
  auto *kernel_block = kernel_body->as<Block>();
  EXPECT_EQ(kernel_block->size(), 3);
  irpass::type_check(kernel_block, CompileConfig());

  irpass::inlining(kernel_block, CompileConfig(), {});
  irpass::full_simplify(kernel_block, CompileConfig(), {false, false});

  EXPECT_EQ(kernel_block->size(), 4);
  EXPECT_TRUE(irpass::analysis::same_statements(func_block, kernel_block));
}

TEST_F(InliningTest, BudgetUsesRecursiveStatementCount) {
  IRBuilder builder;
  auto *arg = builder.create_arg_load(/*arg_id=*/{0}, get_data_type<int>(),
                                      /*is_ptr=*/false, /*arg_depth=*/0);
  auto *cond = builder.create_cmp_gt(arg, builder.get_int32(0));
  auto *if_stmt = builder.create_if(cond);
  {
    auto guard = builder.get_if_guard(if_stmt, /*true_branch=*/true);
    builder.create_return(builder.create_add(arg, builder.get_int32(1)));
  }
  builder.create_return(arg);
  auto func_body = builder.extract_ir();
  EXPECT_TRUE(func_body->is<Block>());
  auto *func_block = func_body->as<Block>();
  EXPECT_EQ(func_block->size(), 4);
  EXPECT_GT(irpass::analysis::count_statements(func_block), 4);

  auto *func = prog_->create_function(
      FunctionKey("nested_func", /*func_id=*/1, /*instance_id=*/0));
  func->insert_scalar_param(get_data_type<int>());
  func->insert_ret(get_data_type<int>());
  func->set_function_body(std::move(func_body));
  func->finalize_params();
  func->finalize_rets();

  auto *kernel_arg =
      builder.create_arg_load(/*arg_id=*/{0}, get_data_type<int>(),
                              /*is_ptr=*/false, /*arg_depth=*/0);
  auto *func_call = builder.create_func_call(func, {kernel_arg});
  builder.create_return(func_call);
  auto kernel_body = builder.extract_ir();
  EXPECT_TRUE(kernel_body->is<Block>());
  auto *kernel_block = kernel_body->as<Block>();
  irpass::type_check(kernel_block, CompileConfig());

  InliningPass::Args inl_args;
  inl_args.budget = 4;
  EXPECT_FALSE(irpass::inlining(kernel_block, CompileConfig(), inl_args));
  EXPECT_EQ(count_func_calls(kernel_block), 1);

  inl_args.budget = irpass::analysis::count_statements(func->ir.get());
  EXPECT_TRUE(irpass::inlining(kernel_block, CompileConfig(), inl_args));
  EXPECT_EQ(count_func_calls(kernel_block), 0);
}
}  // namespace taichi::lang
