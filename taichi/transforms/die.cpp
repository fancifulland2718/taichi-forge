// Dead Instruction Elimination

#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/system/profiler.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace taichi::lang {

// Dead Instruction Elimination
class DIE : public IRVisitor {
 public:
  bool modified_ir{false};

  explicit DIE(IRNode *node) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    node->accept(this);

    // A statement can become dead only when one of its users is removed.
    // Count operand edges (including repeated operands), then propagate those
    // removals without rescanning the tree for every dependency layer.
    std::vector<Stmt *> worklist;
    for (auto *stmt : candidates_) {
      if (usage_.at(stmt).count == 0) {
        worklist.push_back(stmt);
      }
    }
    std::unordered_map<Block *, std::unordered_set<Stmt *>> dead;
    for (size_t i = 0; i < worklist.size(); ++i) {
      auto *stmt = worklist[i];
      dead[stmt->parent].insert(stmt);
      for (int j = 0; j < stmt->num_operands(); ++j) {
        if (auto *operand = stmt->operand(j)) {
          auto &use = usage_.at(operand);
          TI_ASSERT(use.count > 0);
          if (--use.count == 0 && use.eliminable) {
            worklist.push_back(operand);
          }
        }
      }
    }
    // Keep pointers stable during propagation, then compact each affected
    // block once. Block::erase retains the existing trash-bin ownership.
    for (auto &[block, statements] : dead) {
      block->erase(std::move(statements));
    }
    modified_ir = !worklist.empty();
  }

  void register_usage(Stmt *stmt) {
    for (int i = 0; i < stmt->num_operands(); ++i) {
      if (auto *operand = stmt->operand(i)) {
        ++usage_[operand].count;
      }
    }
  }

  void visit(Stmt *stmt) override {
    TI_ASSERT(!stmt->erased);
    register_usage(stmt);
    if (stmt->dead_instruction_eliminable()) {
      usage_[stmt].eliminable = true;
      candidates_.push_back(stmt);
    }
  }

  void visit(Block *stmt_list) override {
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
  }

  void visit(IfStmt *if_stmt) override {
    register_usage(if_stmt);
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
  }

  void visit(WhileStmt *stmt) override {
    register_usage(stmt);
    stmt->body->accept(this);
  }

  void visit(RangeForStmt *for_stmt) override {
    register_usage(for_stmt);
    for_stmt->body->accept(this);
  }

  void visit(StructForStmt *for_stmt) override {
    register_usage(for_stmt);
    for_stmt->body->accept(this);
  }

  void visit(MeshForStmt *for_stmt) override {
    register_usage(for_stmt);
    for_stmt->body->accept(this);
  }

  void visit(OffloadedStmt *stmt) override {
    // TODO: A hack to make sure end_stmt is registered.
    // Ideally end_stmt should be its own Block instead.
    if (stmt->end_stmt) {
      ++usage_[stmt->end_stmt].count;
    }
    stmt->all_blocks_accept(this, true);
  }

 private:
  struct Usage {
    size_t count{0};
    // Only ordinary statements visited in this invocation can be removed.
    // Operands outside the subtree and container statements remain pinned.
    bool eliminable{false};
  };
  std::unordered_map<Stmt *, Usage> usage_;
  std::vector<Stmt *> candidates_;
};

namespace irpass {

bool die(IRNode *root) {
  TI_AUTO_PROF;
  DIE instance(root);
  return instance.modified_ir;
}

}  // namespace irpass

}  // namespace taichi::lang
