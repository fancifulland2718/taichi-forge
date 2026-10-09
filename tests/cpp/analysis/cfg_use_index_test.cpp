#include "gtest/gtest.h"

#include "taichi/ir/analysis.h"
#include "taichi/ir/control_flow_graph.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/stmt_use_index.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"

namespace taichi::lang {
namespace {

class CheckUseIndex : public BasicStmtVisitor {
 public:
  CheckUseIndex(IRNode *root, const StmtUseIndex &actual)
      : expected_(root), actual_(actual) {
    invoke_default_visitor = true;
    root->accept(this);
  }

  void visit(Stmt *stmt) override {
    EXPECT_EQ(actual_.users(stmt), expected_.users(stmt));
    for (int i = 0; i < stmt->num_operands(); ++i) {
      if (auto *operand = stmt->operand(i))
        EXPECT_EQ(actual_.users(operand), expected_.users(operand));
    }
  }

  void preprocess_container_stmt(Stmt *stmt) override {
    visit(stmt);
  }

 private:
  StmtUseIndex expected_;
  const StmtUseIndex &actual_;
};

TEST(CFGUseIndex, ForwardingTracksZeroInsertionChainsAndBranchOperands) {
  auto root = std::make_unique<Block>();
  auto one = root->push_back<ConstStmt>(TypedConstant(1));
  auto local = root->push_back<AllocaStmt>(PrimitiveType::i32);
  auto other = root->push_back<AllocaStmt>(PrimitiveType::i32);
  auto zero = root->push_back<LocalLoadStmt>(local);
  root->push_back<LocalStoreStmt>(local, one);
  auto load = root->push_back<LocalLoadStmt>(local);
  root->push_back<LocalStoreStmt>(other, load);
  auto chained = root->push_back<LocalLoadStmt>(other);
  auto branch = root->push_back<IfStmt>(chained)->as<IfStmt>();
  branch->set_true_statements(std::make_unique<Block>());
  auto result = branch->true_statements->push_back<ReturnStmt>(
      std::vector<Stmt *>{zero, load, chained});
  irpass::type_check(root.get(), CompileConfig{});
  auto cfg = irpass::analysis::build_cfg(root.get());
  cfg->simplify_graph();
  StmtUseIndex uses(root.get());
  EXPECT_TRUE(cfg->store_to_load_forwarding(true, false, &uses));
  ASSERT_TRUE(result->operand(0)->is<ConstStmt>());
  EXPECT_TRUE(result->operand(0)->as<ConstStmt>()->val.equal_value(0));
  EXPECT_EQ(result->operand(1), one);
  EXPECT_EQ(result->operand(2), one);
  EXPECT_EQ(branch->operand(0), one);
  EXPECT_TRUE(uses.users(zero).empty());
  EXPECT_TRUE(uses.users(load).empty());
  EXPECT_TRUE(uses.users(chained).empty());
  CheckUseIndex(root.get(), uses);
  cfg->dead_store_elimination(true, std::nullopt, &uses);
  CheckUseIndex(root.get(), uses);
  EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
}

TEST(CFGUseIndex, DeadStoresAndAtomicWeakeningUpdateOperandUses) {
  auto root = std::make_unique<Block>();
  auto one = root->push_back<ConstStmt>(TypedConstant(1));
  auto local = root->push_back<AllocaStmt>(PrimitiveType::i32);
  auto store = root->push_back<LocalStoreStmt>(local, one);
  auto atomic = root->push_back<AtomicOpStmt>(AtomicOpType::add, local, one);
  auto result = root->push_back<ReturnStmt>(std::vector<Stmt *>{atomic});
  irpass::type_check(root.get(), CompileConfig{});
  auto cfg = irpass::analysis::build_cfg(root.get());
  cfg->simplify_graph();
  StmtUseIndex uses(root.get());
  EXPECT_TRUE(cfg->dead_store_elimination(true, std::nullopt, &uses));
  EXPECT_FALSE(store->erased);
  ASSERT_TRUE(result->operand(0)->is<LocalLoadStmt>());
  EXPECT_TRUE(uses.users(atomic).empty());
  CheckUseIndex(root.get(), uses);
  EXPECT_TRUE(cfg->store_to_load_forwarding(true, false, &uses));
  EXPECT_EQ(result->operand(0), one);
  cfg->dead_store_elimination(true, std::nullopt, &uses);
  EXPECT_TRUE(store->erased);
  CheckUseIndex(root.get(), uses);
  EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
}

TEST(CFGUseIndex, IdenticalLoadsTransferUsersBeforeErasure) {
  auto root = std::make_unique<Block>();
  auto local = root->push_back<AllocaStmt>(PrimitiveType::i32);
  auto first = root->push_back<LocalLoadStmt>(local);
  auto second = root->push_back<LocalLoadStmt>(local);
  auto third = root->push_back<LocalLoadStmt>(local);
  auto result =
      root->push_back<ReturnStmt>(std::vector<Stmt *>{first, second, third});
  irpass::type_check(root.get(), CompileConfig{});
  auto cfg = irpass::analysis::build_cfg(root.get());
  cfg->simplify_graph();
  StmtUseIndex uses(root.get());
  EXPECT_TRUE(cfg->dead_store_elimination(true, std::nullopt, &uses));
  EXPECT_EQ(result->operand(0), first);
  EXPECT_EQ(result->operand(1), first);
  EXPECT_EQ(result->operand(2), first);
  EXPECT_TRUE(uses.users(second).empty());
  EXPECT_TRUE(uses.users(third).empty());
  CheckUseIndex(root.get(), uses);
  EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
}

}  // namespace
}  // namespace taichi::lang
