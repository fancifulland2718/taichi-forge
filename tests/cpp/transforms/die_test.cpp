#include "gtest/gtest.h"

#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"

namespace taichi::lang {
namespace {

TEST(DeadInstructionElimination, RemovesDependencyChainsInOneInvocation) {
  auto root = std::make_unique<Block>();
  Stmt *value = root->push_back<ConstStmt>(TypedConstant(1));
  for (int i = 0; i < 1024; ++i) {
    value = root->push_back<UnaryOpStmt>(UnaryOpType::neg, value);
  }
  EXPECT_TRUE(irpass::die(root.get()));
  EXPECT_EQ(root->size(), 0);
  EXPECT_EQ(root->trash_bin.size(), 1025);
  EXPECT_TRUE(value->erased);
  EXPECT_FALSE(irpass::die(root.get()));
}

TEST(DeadInstructionElimination, RepeatedOperandsAndLiveDiamond) {
  auto root = std::make_unique<Block>();
  auto value = root->push_back<ConstStmt>(TypedConstant(1));
  auto left = root->push_back<UnaryOpStmt>(UnaryOpType::neg, value);
  auto right = root->push_back<BinaryOpStmt>(BinaryOpType::add, value, value);
  auto dead = root->push_back<BinaryOpStmt>(BinaryOpType::add, left, right);
  auto live = root->push_back<ReturnStmt>(std::vector<Stmt *>{right});
  EXPECT_TRUE(irpass::die(root.get()));
  ASSERT_EQ(root->size(), 3);
  EXPECT_EQ(root->statements[0].get(), value);
  EXPECT_EQ(root->statements[1].get(), right);
  EXPECT_EQ(root->statements[2].get(), live);
  EXPECT_TRUE(left->erased);
  EXPECT_TRUE(dead->erased);
  EXPECT_FALSE(irpass::die(root.get()));
}

TEST(DeadInstructionElimination, PreservesLocalStoresAndDecorations) {
  auto root = std::make_unique<Block>();
  auto allocation = root->push_back<AllocaStmt>(PrimitiveType::i32);
  auto value = root->push_back<ConstStmt>(TypedConstant(7));
  auto stored = root->push_back<UnaryOpStmt>(UnaryOpType::neg, value);
  auto store = root->push_back<LocalStoreStmt>(allocation, stored);
  auto decorated = root->push_back<ConstStmt>(TypedConstant(2));
  auto decoration =
      root->push_back<DecorationStmt>(decorated, std::vector<uint32_t>{1});
  auto dead = root->push_back<ConstStmt>(TypedConstant(3));
  EXPECT_TRUE(irpass::die(root.get()));
  EXPECT_EQ(root->size(), 6);
  EXPECT_FALSE(allocation->erased);
  EXPECT_FALSE(stored->erased);
  EXPECT_FALSE(store->erased);
  EXPECT_FALSE(decorated->erased);
  EXPECT_FALSE(decoration->erased);
  EXPECT_TRUE(dead->erased);
}

TEST(DeadInstructionElimination, SubtreeDoesNotDeleteOutsideDefinitions) {
  auto root = std::make_unique<Block>();
  auto condition = root->push_back<ConstStmt>(TypedConstant(1));
  auto outside = root->push_back<ConstStmt>(TypedConstant(2));
  auto branch = root->push_back<IfStmt>(condition)->as<IfStmt>();
  branch->set_true_statements(std::make_unique<Block>());
  branch->true_statements->push_back<UnaryOpStmt>(UnaryOpType::neg, outside);
  EXPECT_TRUE(irpass::die(branch->true_statements.get()));
  EXPECT_FALSE(outside->erased);
  EXPECT_EQ(root->size(), 3);
  EXPECT_EQ(branch->true_statements->size(), 0);
  EXPECT_TRUE(irpass::die(root.get()));
  EXPECT_TRUE(outside->erased);
  EXPECT_FALSE(condition->erased);
  EXPECT_FALSE(branch->erased);
  EXPECT_EQ(root->size(), 2);
}

TEST(DeadInstructionElimination, RetainsRangeBoundsAndLoopOperands) {
  auto root = std::make_unique<Block>();
  auto begin = root->push_back<ConstStmt>(TypedConstant(0));
  auto end = root->push_back<ConstStmt>(TypedConstant(4));
  auto loop = root->push_back<RangeForStmt>(
                      begin, end, std::make_unique<Block>(), false, 1, 1, true)
                  ->as<RangeForStmt>();
  auto dead = loop->body->push_back<UnaryOpStmt>(UnaryOpType::neg, begin);
  EXPECT_TRUE(irpass::die(root.get()));
  EXPECT_TRUE(dead->erased);
  EXPECT_EQ(loop->body->size(), 0);
  EXPECT_EQ(root->size(), 3);
  EXPECT_FALSE(irpass::die(root.get()));
}

TEST(DeadInstructionElimination, PreservesOffloadSpecialReferencesAndScope) {
  auto root = std::make_unique<Block>();
  auto base = root->push_back<ConstStmt>(TypedConstant(3));
  auto end = root->push_back<UnaryOpStmt>(UnaryOpType::neg, base);
  auto offload = root->push_back<OffloadedStmt>(OffloadedTaskType::range_for,
                                                Arch::vulkan, nullptr)
                     ->as<OffloadedStmt>();
  offload->end_stmt = end;  // Deliberately not a registered operand.
  offload->mesh_prologue = std::make_unique<Block>();
  auto untouched =
      offload->mesh_prologue->push_back<ConstStmt>(TypedConstant(5));
  offload->tls_prologue = std::make_unique<Block>();
  auto tls_dead = offload->tls_prologue->push_back<ConstStmt>(TypedConstant(6));
  offload->body->push_back<ConstStmt>(TypedConstant(7));
  offload->tls_epilogue = std::make_unique<Block>();
  auto epilogue_dead =
      offload->tls_epilogue->push_back<ConstStmt>(TypedConstant(8));
  EXPECT_TRUE(irpass::die(root.get()));
  EXPECT_FALSE(end->erased);
  EXPECT_FALSE(base->erased);
  EXPECT_FALSE(untouched->erased);
  EXPECT_EQ(offload->mesh_prologue->size(), 1);
  EXPECT_TRUE(tls_dead->erased);
  EXPECT_TRUE(epilogue_dead->erased);
  EXPECT_EQ(offload->body->size(), 0);
  EXPECT_FALSE(irpass::die(root.get()));
}

TEST(DeadInstructionElimination, NullOperandsAndCyclesStayConservative) {
  auto root = std::make_unique<Block>();
  auto first = root->push_back<UnaryOpStmt>(UnaryOpType::neg, nullptr);
  auto second = root->push_back<UnaryOpStmt>(UnaryOpType::neg, first);
  // A synthetic dependency cycle checks the legacy "has a user" contract.
  // DIE does not attempt a reachability proof for cyclic operand relations.
  first->set_operand(0, second);
  auto unused = root->push_back<UnaryOpStmt>(UnaryOpType::neg, nullptr);
  EXPECT_TRUE(irpass::die(root.get()));
  EXPECT_TRUE(unused->erased);
  EXPECT_FALSE(first->erased);
  EXPECT_FALSE(second->erased);
  EXPECT_EQ(root->size(), 2);
  EXPECT_FALSE(irpass::die(root.get()));
}

}  // namespace
}  // namespace taichi::lang
