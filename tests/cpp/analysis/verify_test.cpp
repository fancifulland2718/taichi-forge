#include "gtest/gtest.h"

#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"

namespace taichi::lang {

TEST(IRVerifier, RejectsValuesOutsideTheirBranch) {
  for (bool sibling : {false, true}) {
    SCOPED_TRACE(sibling);
    auto root = std::make_unique<Block>();
    auto cond = root->push_back<ConstStmt>(TypedConstant(1));
    auto branch = root->push_back<IfStmt>(cond)->as<IfStmt>();
    branch->set_true_statements(std::make_unique<Block>());
    auto value =
        branch->true_statements->push_back<ConstStmt>(TypedConstant(7));
    auto use_block = root.get();
    if (sibling) {
      branch->set_false_statements(std::make_unique<Block>());
      use_block = branch->false_statements.get();
    }
    use_block->push_back<UnaryOpStmt>(UnaryOpType::neg, value);
    EXPECT_ANY_THROW(irpass::analysis::verify(root.get()));
  }
}

TEST(IRVerifier, AcceptsAncestorValuesAndSubsequentSiblingBlocks) {
  auto root = std::make_unique<Block>();
  auto cond = root->push_back<ConstStmt>(TypedConstant(1));
  for (int i = 0; i < 2; ++i) {
    auto branch = root->push_back<IfStmt>(cond)->as<IfStmt>();
    branch->set_true_statements(std::make_unique<Block>());
    branch->set_false_statements(std::make_unique<Block>());
    branch->true_statements->push_back<UnaryOpStmt>(UnaryOpType::neg, cond);
    branch->false_statements->push_back<UnaryOpStmt>(UnaryOpType::neg, cond);
  }
  root->push_back<UnaryOpStmt>(UnaryOpType::neg, cond);
  EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
}

TEST(IRVerifier, IsolatesTasksButPreservesPrologueVisibility) {
  auto root = std::make_unique<Block>();
  Stmt *previous = nullptr;
  OffloadedStmt *last = nullptr;
  for (int i = 0; i < 2; ++i) {
    auto task = root->push_back<OffloadedStmt>(OffloadedTaskType::range_for,
                                               Arch::vulkan, nullptr)
                    ->as<OffloadedStmt>();
    task->tls_prologue = std::make_unique<Block>();
    task->tls_prologue->set_parent_stmt(task);
    auto value = task->tls_prologue->push_back<ConstStmt>(TypedConstant(i));
    task->body->push_back<UnaryOpStmt>(UnaryOpType::neg, value);
    task->body->push_back<LoopIndexStmt>(task, 0);
    if (!previous)
      previous = value;
    last = task;
  }
  EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
  last->body->push_back<UnaryOpStmt>(UnaryOpType::neg, previous);
  EXPECT_ANY_THROW(irpass::analysis::verify(root.get()));
}

}  // namespace taichi::lang
