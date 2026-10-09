#include "gtest/gtest.h"

#include "taichi/ir/analysis.h"
#include "taichi/ir/control_flow_graph.h"
#include "taichi/ir/local_storage.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"

namespace taichi::lang {
namespace {

TEST(CFGForwarding, WholeTensorJoinsKeepValuesVisibilityAndPartialWrites) {
  for (bool lowered : {false, true}) {
    for (bool same : {false, true}) {
      for (bool visible : {false, true}) {
        for (bool partial : {false, true}) {
          SCOPED_TRACE(lowered);
          SCOPED_TRACE(same);
          SCOPED_TRACE(visible);
          SCOPED_TRACE(partial);
          auto root = std::make_unique<Block>();
          auto cond = root->push_back<ArgLoadStmt>(
              std::vector<int>{0}, PrimitiveType::i32, false, true, 0);
          auto zero = root->push_back<ConstStmt>(TypedConstant(0));
          auto one = root->push_back<ConstStmt>(TypedConstant(1));
          auto type = TypeFactory::get_instance().get_tensor_type(
              {2}, PrimitiveType::i32);
          auto local = root->push_back<AllocaStmt>(type);
          auto part = root->push_back<MatrixPtrStmt>(local, zero);
          auto first =
              root->push_back<MatrixInitStmt>(std::vector<Stmt *>{one, one});
          auto second =
              root->push_back<MatrixInitStmt>(std::vector<Stmt *>{one, zero});
          first->ret_type = second->ret_type = type;
          auto branch = root->push_back<IfStmt>(cond)->as<IfStmt>();
          branch->set_true_statements(std::make_unique<Block>());
          branch->set_false_statements(std::make_unique<Block>());
          auto lhs = visible
                         ? first
                         : branch->true_statements->push_back<MatrixInitStmt>(
                               std::vector<Stmt *>{one, one});
          auto rhs = visible
                         ? (same ? first : second)
                         : branch->false_statements->push_back<MatrixInitStmt>(
                               std::vector<Stmt *>{one, same ? one : zero});
          lhs->ret_type = rhs->ret_type = type;
          branch->true_statements->push_back<LocalStoreStmt>(local, lhs);
          branch->false_statements->push_back<LocalStoreStmt>(local, rhs);
          if (partial)
            branch->true_statements->push_back<LocalStoreStmt>(part, zero);
          // Use a separate consumer block. The existing final interval check
          // conservatively includes parent-block allocas if the load is there.
          auto consumer = root->push_back<IfStmt>(cond)->as<IfStmt>();
          consumer->set_true_statements(std::make_unique<Block>());
          auto load =
              consumer->true_statements->push_back<LocalLoadStmt>(local);
          auto result = consumer->true_statements->push_back<ReturnStmt>(
              std::vector<Stmt *>{load});
          irpass::type_check(root.get(), CompileConfig{});
          auto cfg = irpass::analysis::build_cfg(root.get());
          cfg->simplify_graph();
          cfg->store_to_load_forwarding(lowered, false);
          EXPECT_EQ(result->operand(0),
                    same && visible && !partial ? first : load);
          EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
        }
      }
    }
  }
}

TEST(CFGForwarding, EntryFactsDoNotOverrideNearestLocalTensorStore) {
  auto root = std::make_unique<Block>();
  auto zero = root->push_back<ConstStmt>(TypedConstant(0));
  auto one = root->push_back<ConstStmt>(TypedConstant(1));
  auto type =
      TypeFactory::get_instance().get_tensor_type({2}, PrimitiveType::i32);
  auto local = root->push_back<AllocaStmt>(type);
  auto part = root->push_back<MatrixPtrStmt>(local, zero);
  auto unknown = root->push_back<LocalLoadStmt>(part);
  root->push_back<LocalStoreStmt>(part, one);
  auto known = root->push_back<LocalLoadStmt>(part);
  auto scalar = root->push_back<AllocaStmt>(PrimitiveType::i32);
  auto initialized = root->push_back<LocalLoadStmt>(scalar);
  auto result = root->push_back<ReturnStmt>(
      std::vector<Stmt *>{unknown, known, initialized});
  irpass::type_check(root.get(), CompileConfig{});
  auto cfg = irpass::analysis::build_cfg(root.get());
  cfg->simplify_graph();
  cfg->store_to_load_forwarding(true, false);
  EXPECT_EQ(result->operand(0), unknown);
  EXPECT_EQ(result->operand(1), one);
  ASSERT_TRUE(result->operand(2)->is<ConstStmt>());
  EXPECT_TRUE(result->operand(2)->as<ConstStmt>()->val.equal_value(0));
  EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
}

TEST(CFGForwarding, TensorDirectoryKeepsRewrittenValuesAcrossFactWords) {
  auto root = std::make_unique<Block>();
  auto cond = root->push_back<ArgLoadStmt>(std::vector<int>{0},
                                           PrimitiveType::i32, false, true, 0);
  auto one = root->push_back<ConstStmt>(TypedConstant(1));
  auto input = root->push_back<AllocaStmt>(PrimitiveType::i32);
  root->push_back<LocalStoreStmt>(input, one);
  auto type =
      TypeFactory::get_instance().get_tensor_type({2}, PrimitiveType::i32);
  std::vector<Stmt *> locals, values, loads;
  for (int i = 0; i < 70; ++i) {
    locals.push_back(root->push_back<AllocaStmt>(type));
    auto from_input = root->push_back<LocalLoadStmt>(input);
    auto value =
        root->push_back<MatrixInitStmt>(std::vector<Stmt *>{from_input, one});
    value->ret_type = type;
    values.push_back(value);
    root->push_back<LocalStoreStmt>(locals.back(), value);
  }
  auto branch = root->push_back<IfStmt>(cond)->as<IfStmt>();
  branch->set_true_statements(std::make_unique<Block>());
  for (int i = 0; i < 70; ++i)
    branch->true_statements->push_back<LocalStoreStmt>(locals[i], values[i]);
  auto consumer = root->push_back<IfStmt>(cond)->as<IfStmt>();
  consumer->set_true_statements(std::make_unique<Block>());
  for (auto local : locals)
    loads.push_back(consumer->true_statements->push_back<LocalLoadStmt>(local));
  auto result = consumer->true_statements->push_back<ReturnStmt>(loads);
  irpass::type_check(root.get(), CompileConfig{});
  auto cfg = irpass::analysis::build_cfg(root.get());
  cfg->simplify_graph();
  EXPECT_TRUE(cfg->store_to_load_forwarding(true, false));
  for (int i = 0; i < 70; ++i) {
    EXPECT_EQ(result->operand(i), values[i]);
    EXPECT_EQ(values[i]->operand(0), one);
  }
  cfg->store_to_load_forwarding(true, false);
  EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
}

TEST(CFGForwarding, MixedOpaqueDestinationsKeepUnknownTensorValue) {
  auto root = std::make_unique<Block>();
  auto cond = root->push_back<ArgLoadStmt>(std::vector<int>{0},
                                           PrimitiveType::i32, false, true, 0);
  auto one = root->push_back<ConstStmt>(TypedConstant(1));
  auto type =
      TypeFactory::get_instance().get_tensor_type({2}, PrimitiveType::i32);
  auto local = root->push_back<AllocaStmt>(type);
  auto global = root->push_back<GlobalTemporaryStmt>(0, PrimitiveType::i32);
  auto values = root->push_back<MatrixInitStmt>(std::vector<Stmt *>{one, one});
  values->ret_type = type;
  root->push_back<LocalStoreStmt>(local, values);
  auto branch = root->push_back<IfStmt>(cond)->as<IfStmt>();
  branch->set_true_statements(std::make_unique<Block>());
  branch->true_statements->push_back<ExternalFuncCallStmt>(
      ExternalFuncCallStmt::SHARED_OBJECT, nullptr, "", "", "",
      std::vector<Stmt *>{}, std::vector<Stmt *>{local, global, local});
  auto load = root->push_back<LocalLoadStmt>(local);
  auto result = root->push_back<ReturnStmt>(std::vector<Stmt *>{load});
  irpass::type_check(root.get(), CompileConfig{});
  auto cfg = irpass::analysis::build_cfg(root.get());
  cfg->simplify_graph();
  cfg->store_to_load_forwarding(false, false);
  EXPECT_EQ(result->operand(0), load);
  EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
}

TEST(CFGForwarding, LocalStorageGroupingRejectsNestedAndNonlocalPointers) {
  auto root = std::make_unique<Block>();
  auto zero = root->push_back<ConstStmt>(TypedConstant(0));
  auto type =
      TypeFactory::get_instance().get_tensor_type({2}, PrimitiveType::i32);
  auto nested_type = TypeFactory::get_instance().get_tensor_type({2}, type);
  auto local = root->push_back<AllocaStmt>(nested_type);
  auto part = root->push_back<MatrixPtrStmt>(local, zero);
  auto nested = root->push_back<MatrixPtrStmt>(part, zero);
  auto global = root->push_back<GlobalTemporaryStmt>(0, type);
  auto global_part = root->push_back<MatrixPtrStmt>(global, zero);
  EXPECT_EQ(direct_local_allocation(local), local);
  EXPECT_EQ(direct_local_allocation(part), local);
  EXPECT_EQ(direct_local_allocation(nested), nullptr);
  EXPECT_EQ(direct_local_allocation(global_part), nullptr);
}

}  // namespace
}  // namespace taichi::lang
