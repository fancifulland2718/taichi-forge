#pragma once

#include "taichi/ir/analysis.h"
#include "taichi/ir/ir.h"

namespace taichi::lang {

// For passes replacing non-container statements. Update indexed uses eagerly
// so later decisions see the new values, but rebuild each affected block once.
// New statements are also indexed: subsequent replacements must update their
// operands as well as operands that existed when the pass began.
class BatchStmtReplacer {
 public:
  explicit BatchStmtReplacer(IRNode *root)
      : usages_(irpass::analysis::gather_statement_usages(root)) {
  }

  void replace(Stmt *old_stmt, VecStatement stmts, Stmt *value) {
    TI_ASSERT(!old_stmt->is_container_statement());
    auto &edits = edits_[old_stmt->parent];
    TI_ASSERT(edits.find(old_stmt) == edits.end());
    for (auto &stmt : stmts.stmts) {
      TI_ASSERT(!stmt->is_container_statement());
      stmt->parent = old_stmt->parent;
      for (int i = 0; i < stmt->num_operands(); ++i) {
        if (auto operand = stmt->operand(i)) {
          usages_[operand].emplace_back(stmt.get(), i);
        }
      }
    }
    if (value != old_stmt) {
      auto found = usages_.find(old_stmt);
      if (found != usages_.end()) {
        auto users = std::move(found->second);
        usages_.erase(found);
        for (auto [user, index] : users) {
          // An earlier rewrite may have changed this slot already.
          if (user->operand(index) == old_stmt) {
            TI_ASSERT(value != nullptr);
            user->set_operand(index, value);
            usages_[value].emplace_back(user, index);
          }
        }
      }
    }
    edits.emplace(old_stmt, std::move(stmts));
  }

  bool apply() {
    const bool modified = !edits_.empty();
    for (auto &[block, edits] : edits_) {
      stmt_vector rebuilt;
      size_t size = block->size() - edits.size();
      for (const auto &[stmt, replacement] : edits) {
        size += replacement.size();
      }
      rebuilt.reserve(size);
      for (auto &stmt : block->statements) {
        auto found = edits.find(stmt.get());
        if (found == edits.end()) {
          rebuilt.push_back(std::move(stmt));
        } else {
          for (auto &replacement : found->second.stmts) {
            rebuilt.push_back(std::move(replacement));
          }
          stmt->erased = true;
          block->trash_bin.push_back(std::move(stmt));
        }
      }
      block->statements = std::move(rebuilt);
    }
    edits_.clear();
    return modified;
  }

 private:
  std::unordered_map<Stmt *, std::vector<std::pair<Stmt *, int>>> usages_;
  std::unordered_map<Block *, std::unordered_map<Stmt *, VecStatement>> edits_;
};

}  // namespace taichi::lang
