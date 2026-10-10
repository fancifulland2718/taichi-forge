#include "gtest/gtest.h"

#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/transforms/batch_stmt_replacer.h"

namespace taichi::lang {

TEST(BatchStmtReplacer, RebindsNewStatementsAndRepeatedOperands) {
  auto root = std::make_unique<Block>();
  auto a = root->push_back<ConstStmt>(TypedConstant(1));
  auto b = root->push_back<BinaryOpStmt>(BinaryOpType::add, a, a);
  auto result = root->push_back<ReturnStmt>(std::vector<Stmt *>{b});
  BatchStmtReplacer modifier(root.get());
  VecStatement replacement;
  auto sum = replacement.push_back<BinaryOpStmt>(BinaryOpType::sub, a, a);
  modifier.replace(b, std::move(replacement), sum);
  VecStatement constant;
  auto value = constant.push_back<ConstStmt>(TypedConstant(2));
  modifier.replace(a, std::move(constant), value);
  EXPECT_TRUE(modifier.apply());
  EXPECT_EQ(sum->lhs, value);
  EXPECT_EQ(sum->rhs, value);
  EXPECT_EQ(result->operand(0), sum);
  EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
}

TEST(BatchStmtReplacer, NoopThenMultipleBatchesUpdateContainerOperands) {
  auto root = std::make_unique<Block>();
  auto one = root->push_back<ConstStmt>(TypedConstant(1));
  auto condition = root->push_back<BinaryOpStmt>(BinaryOpType::add, one, one);
  auto branch = root->push_back<IfStmt>(condition)->as<IfStmt>();
  branch->set_true_statements(std::make_unique<Block>());
  auto result = branch->true_statements->push_back<ReturnStmt>(
      std::vector<Stmt *>{condition, one});
  BatchStmtReplacer modifier(root.get());
  EXPECT_FALSE(modifier.apply());

  VecStatement first;
  auto sum = first.push_back<BinaryOpStmt>(BinaryOpType::sub, one, one);
  modifier.replace(condition, std::move(first), sum);
  EXPECT_TRUE(modifier.apply());
  EXPECT_FALSE(modifier.apply());

  VecStatement second;
  auto two = second.push_back<ConstStmt>(TypedConstant(2));
  modifier.replace(one, std::move(second), two);
  EXPECT_TRUE(modifier.apply());
  EXPECT_EQ(sum->lhs, two);
  EXPECT_EQ(sum->rhs, two);
  EXPECT_EQ(branch->cond, sum);
  EXPECT_EQ(result->operand(0), sum);
  EXPECT_EQ(result->operand(1), two);
  EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
}

TEST(BatchStmtReplacer, ConstantFoldPropagatesLongChainsIntoBranches) {
  auto root = std::make_unique<Block>();
  auto one = root->push_back<ConstStmt>(TypedConstant(1));
  Stmt *value = one;
  for (int i = 0; i < 4096; ++i) {
    value = root->push_back<BinaryOpStmt>(BinaryOpType::add, value, one);
  }
  auto branch = root->push_back<IfStmt>(one)->as<IfStmt>();
  branch->set_true_statements(std::make_unique<Block>());
  auto result = branch->true_statements->push_back<ReturnStmt>(
      std::vector<Stmt *>{value});
  irpass::type_check(root.get(), CompileConfig());
  EXPECT_TRUE(irpass::constant_fold(root.get()));
  ASSERT_TRUE(result->operand(0)->is<ConstStmt>());
  EXPECT_EQ(result->operand(0)->as<ConstStmt>()->val.val_int(), 4097);
  EXPECT_FALSE(irpass::constant_fold(root.get()));
  EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
}

TEST(BatchStmtReplacer, DemotedAtomicsReturnOldValuesInOrder) {
  auto root = std::make_unique<Block>();
  auto allocation = root->push_back<AllocaStmt>(PrimitiveType::i32);
  auto one = root->push_back<ConstStmt>(TypedConstant(1));
  auto a = root->push_back<AtomicOpStmt>(AtomicOpType::add, allocation, one);
  auto b = root->push_back<AtomicOpStmt>(AtomicOpType::add, allocation, a);
  auto result = root->push_back<ReturnStmt>(std::vector<Stmt *>{a, b});
  irpass::type_check(root.get(), CompileConfig());
  EXPECT_TRUE(irpass::demote_atomics(root.get(), CompileConfig()));
  ASSERT_EQ(root->size(), 9);
  EXPECT_EQ(result->operand(0), root->statements[2].get());
  EXPECT_EQ(result->operand(1), root->statements[5].get());
  EXPECT_EQ(root->statements[6]->as<BinaryOpStmt>()->rhs,
            root->statements[2].get());
  EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
}

TEST(BatchStmtReplacer, ImmutableLocalChainsKeepTheirValues) {
  auto root = std::make_unique<Block>();
  auto value = root->push_back<ConstStmt>(TypedConstant(7));
  Stmt *last = value;
  for (int i = 0; i < 4096; ++i) {
    auto allocation = root->push_back<AllocaStmt>(PrimitiveType::i32);
    root->push_back<LocalStoreStmt>(allocation, last);
    last = root->push_back<LocalLoadStmt>(allocation);
  }
  auto result = root->push_back<ReturnStmt>(std::vector<Stmt *>{last});
  irpass::type_check(root.get(), CompileConfig());
  irpass::eliminate_immutable_local_vars(root.get());
  EXPECT_EQ(root->size(), 2);
  EXPECT_EQ(result->operand(0), value);
  EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
}

}  // namespace taichi::lang
