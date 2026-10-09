#include "gtest/gtest.h"

#include "taichi/ir/statements.h"
#include "taichi/ir/stmt_use_index.h"

namespace taichi::lang {
namespace {

TEST(StmtUseIndex, ReplacementChainsUpdateRepeatedAndContainerOperands) {
  auto root = std::make_unique<Block>();
  auto first = root->push_back<ConstStmt>(TypedConstant(1));
  auto second = root->push_back<ConstStmt>(TypedConstant(2));
  auto final = root->push_back<ConstStmt>(TypedConstant(3));
  auto sum = root->push_back<BinaryOpStmt>(BinaryOpType::add, first, first);
  auto other = root->push_back<UnaryOpStmt>(UnaryOpType::neg, second);
  auto branch = root->push_back<IfStmt>(first)->as<IfStmt>();
  branch->set_true_statements(std::make_unique<Block>());
  auto result = branch->true_statements->push_back<ReturnStmt>(
      std::vector<Stmt *>{first, sum});
  StmtUseIndex index(root.get());
  EXPECT_EQ(index.users(first).size(), 3);
  std::unordered_set<Stmt *> changed;
  index.replace_uses(first, second, [&](Stmt *stmt) {
    EXPECT_TRUE(changed.insert(stmt).second);
  });
  EXPECT_EQ(changed.size(), 3);
  EXPECT_TRUE(index.users(first).empty());
  EXPECT_EQ(index.users(second).size(), 4);
  index.replace_uses(second, final);
  EXPECT_EQ(sum->operand(0), final);
  EXPECT_EQ(sum->operand(1), final);
  EXPECT_EQ(other->operand(0), final);
  EXPECT_EQ(branch->operand(0), final);
  EXPECT_EQ(result->operand(0), final);
  EXPECT_TRUE(index.users(second).empty());
  index.replace_uses(final, final);
  EXPECT_EQ(index.users(final).size(), 4);
}

TEST(StmtUseIndex, InsertEraseAndOperandChangesMaintainUsers) {
  auto root = std::make_unique<Block>();
  auto first = root->push_back<ConstStmt>(TypedConstant(1));
  auto second = root->push_back<ConstStmt>(TypedConstant(2));
  auto final = root->push_back<ConstStmt>(TypedConstant(3));
  auto user = root->push_back<BinaryOpStmt>(BinaryOpType::add, first, first);
  StmtUseIndex index(root.get());
  // Pending insertion already needs replacement when its operands change.
  auto pending = Stmt::make<BinaryOpStmt>(BinaryOpType::sub, first, second);
  auto *inserted = pending.get();
  index.add_user(inserted);
  index.add_user(inserted);  // idempotent
  index.replace_uses(first, second);
  EXPECT_EQ(inserted->operand(0), second);
  root->insert(std::move(pending));
  index.remove_user(user);
  user->set_operand(0, final);
  user->set_operand(1, final);
  index.add_user(user);
  EXPECT_EQ(index.users(second).size(), 1);
  EXPECT_EQ(index.users(final).size(), 1);
  index.remove_user(inserted);
  root->erase(inserted);
  EXPECT_TRUE(index.users(second).empty());
  index.replace_uses(second, first);
  EXPECT_EQ(user->operand(0), final);
  index.replace_uses(final, first);
  EXPECT_EQ(user->operand(0), first);
  EXPECT_EQ(user->operand(1), first);
}

TEST(StmtUseIndex, SubtreeScopeAndMovedStatementsStayExplicit) {
  auto root = std::make_unique<Block>();
  auto first = root->push_back<ConstStmt>(TypedConstant(1));
  auto second = root->push_back<ConstStmt>(TypedConstant(2));
  auto outside = root->push_back<UnaryOpStmt>(UnaryOpType::neg, first);
  auto branch = root->push_back<IfStmt>(first)->as<IfStmt>();
  branch->set_true_statements(std::make_unique<Block>());
  auto inside =
      branch->true_statements->push_back<UnaryOpStmt>(UnaryOpType::neg, first);
  StmtUseIndex index(branch->true_statements.get());
  index.replace_uses(first, second);
  EXPECT_EQ(outside->operand(0), first);
  EXPECT_EQ(branch->operand(0), first);
  EXPECT_EQ(inside->operand(0), second);
  auto moved = branch->true_statements->extract(inside);
  root->insert(std::move(moved));
  // A moved user stays registered until explicitly removed or rebuilt.
  index.replace_uses(second, first);
  EXPECT_EQ(inside->operand(0), first);
  index.rebuild(branch->true_statements.get());
  EXPECT_TRUE(index.users(first).empty());
}

TEST(StmtUseIndex, UnknownMutationRequiresRebuildAndNullOperandsAreIgnored) {
  auto root = std::make_unique<Block>();
  auto first = root->push_back<ConstStmt>(TypedConstant(1));
  auto second = root->push_back<ConstStmt>(TypedConstant(2));
  auto user = root->push_back<UnaryOpStmt>(UnaryOpType::neg, first);
  StmtUseIndex index(root.get());
  user->set_operand(0, second);
  index.invalidate();
  EXPECT_FALSE(index.valid());
  index.rebuild(root.get());
  EXPECT_TRUE(index.valid());
  EXPECT_TRUE(index.users(first).empty());
  EXPECT_EQ(index.users(second).size(), 1);
  index.replace_uses(second, nullptr);
  EXPECT_EQ(user->operand(0), nullptr);
  EXPECT_TRUE(index.users(nullptr).empty());
  index.rebuild(root.get());
  EXPECT_TRUE(index.users(nullptr).empty());
}

}  // namespace
}  // namespace taichi::lang
