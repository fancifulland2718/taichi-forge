#include "gtest/gtest.h"

#include "taichi/ir/analysis.h"
#include "taichi/ir/control_flow_graph.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"

namespace taichi::lang {
namespace {

TEST(CFGDeadStore, LocalUseSurvivesLaterOverwrite) {
  for (bool lowered : {false, true}) {
    auto root = std::make_unique<Block>();
    auto one = root->push_back<ConstStmt>(TypedConstant(1));
    auto two = root->push_back<ConstStmt>(TypedConstant(2));
    auto local = root->push_back<AllocaStmt>(PrimitiveType::i32);
    auto first = root->push_back<LocalStoreStmt>(local, one);
    auto load = root->push_back<LocalLoadStmt>(local);
    auto last = root->push_back<LocalStoreStmt>(local, two);
    root->push_back<ReturnStmt>(std::vector<Stmt *>{load});
    irpass::type_check(root.get(), CompileConfig{});
    auto cfg = irpass::analysis::build_cfg(root.get());
    cfg->simplify_graph();
    EXPECT_TRUE(cfg->dead_store_elimination(lowered, std::nullopt));
    EXPECT_FALSE(first->erased);
    EXPECT_TRUE(last->erased);
    EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
  }
}

TEST(CFGDeadStore, FullOverwriteKillsStoreEvenWithSuccessorUse) {
  for (bool lowered : {false, true}) {
    auto root = std::make_unique<Block>();
    auto cond = root->push_back<ArgLoadStmt>(
        std::vector<int>{0}, PrimitiveType::i32, false, true, 0);
    auto one = root->push_back<ConstStmt>(TypedConstant(1));
    auto two = root->push_back<ConstStmt>(TypedConstant(2));
    auto local = root->push_back<AllocaStmt>(PrimitiveType::i32);
    auto first = root->push_back<LocalStoreStmt>(local, one);
    auto last = root->push_back<LocalStoreStmt>(local, two);
    auto branch = root->push_back<IfStmt>(cond)->as<IfStmt>();
    branch->set_true_statements(std::make_unique<Block>());
    auto load = branch->true_statements->push_back<LocalLoadStmt>(local);
    branch->true_statements->push_back<ReturnStmt>(std::vector<Stmt *>{load});
    irpass::type_check(root.get(), CompileConfig{});
    auto cfg = irpass::analysis::build_cfg(root.get());
    cfg->simplify_graph();
    EXPECT_TRUE(cfg->dead_store_elimination(lowered, std::nullopt));
    EXPECT_TRUE(first->erased);
    EXPECT_FALSE(last->erased);
    EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
  }
}

TEST(CFGDeadStore, PartialOverwritePreservesWholeTensorForSuccessor) {
  for (bool lowered : {false, true}) {
    for (bool dynamic : {false, true}) {
      auto root = std::make_unique<Block>();
      auto cond = root->push_back<ArgLoadStmt>(
          std::vector<int>{0}, PrimitiveType::i32, false, true, 0);
      auto zero = root->push_back<ConstStmt>(TypedConstant(0));
      auto one = root->push_back<ConstStmt>(TypedConstant(1));
      auto type =
          TypeFactory::get_instance().get_tensor_type({2}, PrimitiveType::i32);
      auto local = root->push_back<AllocaStmt>(type);
      auto component =
          root->push_back<MatrixPtrStmt>(local, dynamic ? cond : zero);
      auto value =
          root->push_back<MatrixInitStmt>(std::vector<Stmt *>{one, one});
      value->ret_type = type;
      auto whole = root->push_back<LocalStoreStmt>(local, value);
      auto part = root->push_back<LocalStoreStmt>(component, zero);
      auto branch = root->push_back<IfStmt>(cond)->as<IfStmt>();
      branch->set_true_statements(std::make_unique<Block>());
      auto load = branch->true_statements->push_back<LocalLoadStmt>(local);
      branch->true_statements->push_back<ReturnStmt>(std::vector<Stmt *>{load});
      irpass::type_check(root.get(), CompileConfig{});
      auto cfg = irpass::analysis::build_cfg(root.get());
      cfg->simplify_graph();
      cfg->dead_store_elimination(lowered, std::nullopt);
      EXPECT_FALSE(whole->erased);
      EXPECT_FALSE(part->erased);
      EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
    }
  }
}

TEST(CFGDeadStore, WeakenedAtomicRetainsItsInputStore) {
  auto root = std::make_unique<Block>();
  auto one = root->push_back<ConstStmt>(TypedConstant(1));
  auto local = root->push_back<AllocaStmt>(PrimitiveType::i32);
  auto store = root->push_back<LocalStoreStmt>(local, one);
  auto atomic = root->push_back<AtomicOpStmt>(AtomicOpType::add, local, one);
  auto result = root->push_back<ReturnStmt>(std::vector<Stmt *>{atomic});
  irpass::type_check(root.get(), CompileConfig{});
  auto cfg = irpass::analysis::build_cfg(root.get());
  cfg->simplify_graph();
  EXPECT_TRUE(cfg->dead_store_elimination(true, std::nullopt));
  EXPECT_FALSE(store->erased);
  ASSERT_TRUE(result->operand(0)->is<LocalLoadStmt>());
  EXPECT_EQ(result->operand(0)->as<LocalLoadStmt>()->src, local);
  EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
}

TEST(CFGDeadStore, LiveAliasUpdatePreservesPartialTensorReads) {
  for (bool lowered : {false, true}) {
    auto root = std::make_unique<Block>();
    auto zero = root->push_back<ConstStmt>(TypedConstant(0));
    auto one = root->push_back<ConstStmt>(TypedConstant(1));
    auto type = TypeFactory::get_instance().get_tensor_type(
        {2}, PrimitiveType::i32);
    auto local = root->push_back<AllocaStmt>(type);
    auto first = root->push_back<MatrixPtrStmt>(local, zero);
    auto alias = root->push_back<MatrixPtrStmt>(local, zero);
    auto second = root->push_back<MatrixPtrStmt>(local, one);
    auto values = root->push_back<MatrixInitStmt>(std::vector<Stmt *>{one, one});
    values->ret_type = type;
    auto whole = root->push_back<LocalStoreStmt>(local, values);
    auto before = root->push_back<LocalLoadStmt>(second);
    auto part = root->push_back<LocalStoreStmt>(first, zero);
    auto after = root->push_back<LocalLoadStmt>(alias);
    root->push_back<ReturnStmt>(std::vector<Stmt *>{before, after});
    irpass::type_check(root.get(), CompileConfig{});
    auto cfg = irpass::analysis::build_cfg(root.get());
    cfg->simplify_graph();
    cfg->dead_store_elimination(lowered, std::nullopt);
    EXPECT_FALSE(whole->erased);
    EXPECT_FALSE(part->erased);
    EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
  }
}

}  // namespace
}  // namespace taichi::lang
