#include "gtest/gtest.h"

#include "taichi/ir/analysis.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"

namespace taichi::lang {

TEST(LowerAST, RebindsLocalsInBothBranchesAndPreservesParents) {
  auto root = std::make_unique<Block>();
  auto old_alloca =
      root->push_back<FrontendAllocaStmt>(Identifier(0), PrimitiveType::i32);
  auto cond = root->push_back<ConstStmt>(TypedConstant(1));
  auto branch = root->push_back<IfStmt>(cond)->as<IfStmt>();
  branch->set_true_statements(std::make_unique<Block>());
  branch->set_false_statements(std::make_unique<Block>());
  auto left = branch->true_statements->push_back<LocalLoadStmt>(old_alloca)
                  ->as<LocalLoadStmt>();
  auto right = branch->false_statements->push_back<LocalLoadStmt>(old_alloca)
                   ->as<LocalLoadStmt>();
  auto after = root->push_back<LocalLoadStmt>(old_alloca)->as<LocalLoadStmt>();

  irpass::lower_ast(root.get());
  ASSERT_EQ(root->size(), 4);
  auto allocation = root->statements[0].get();
  ASSERT_TRUE(allocation->is<AllocaStmt>());
  EXPECT_EQ(root->lookup_var(Identifier(0)), allocation);
  EXPECT_EQ(left->src, allocation);
  EXPECT_EQ(right->src, allocation);
  EXPECT_EQ(after->src, allocation);
  EXPECT_EQ(branch->true_statements->parent_block(), root.get());
  EXPECT_EQ(branch->false_statements->parent_block(), root.get());
  EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
}

TEST(LowerAST, PreservesLargeBlockOrderAndOperandIdentity) {
  auto root = std::make_unique<Block>();
  for (int i = 0; i < 4096; ++i) {
    auto allocation =
        root->push_back<FrontendAllocaStmt>(Identifier(i), PrimitiveType::i32);
    root->push_back<LocalLoadStmt>(allocation);
  }
  irpass::lower_ast(root.get());
  ASSERT_EQ(root->size(), 8192);
  for (int i = 0; i < 4096; ++i) {
    auto allocation = root->statements[2 * i].get();
    ASSERT_TRUE(allocation->is<AllocaStmt>());
    EXPECT_EQ(root->statements[2 * i + 1]->as<LocalLoadStmt>()->src,
              allocation);
    EXPECT_EQ(root->lookup_var(Identifier(i)), allocation);
  }
  EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
}

TEST(LowerAST, MaterializedCallsBelongToEachLowering) {
  auto call = Expr::make<InternalFuncCallExpression>(
      Operations::get(InternalOp::do_nothing), std::vector<Expr>{}, true);
  call->type_check(nullptr);
  auto source = std::make_unique<Block>();
  source->push_back<FrontendExprStmt>(call);
  source->push_back<FrontendExprStmt>(call);
  auto one = Expr::make<ConstExpression>(int32(1));
  one->type_check(nullptr);
  auto conditional = Expr::make<TernaryOpExpression>(
      TernaryOpType::ifte, one, call, one);
  conditional->type_check(nullptr);
  source->push_back<FrontendExprStmt>(conditional);
  auto logical = Expr::make<BinaryOpExpression>(
      BinaryOpType::logical_and, one, call);
  logical->type_check(nullptr);
  source->push_back<FrontendExprStmt>(logical);

  // Frontend clones share expressions, but must not share lowered statements.
  // Destroy each lowered tree before compiling the next target/request.
  for (int i = 0; i < 3; ++i) {
    auto lowered = irpass::analysis::clone(source.get());
    irpass::lower_ast(lowered.get());
    auto calls = irpass::analysis::gather_statements(
        lowered.get(), [](Stmt *stmt) { return stmt->is<InternalFuncStmt>(); });
    ASSERT_EQ(calls.size(), 1);
    EXPECT_NO_THROW(irpass::analysis::verify(lowered.get()));
  }
}

}  // namespace taichi::lang
