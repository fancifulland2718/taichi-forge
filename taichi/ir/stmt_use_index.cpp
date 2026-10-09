#include "taichi/ir/stmt_use_index.h"

#include "taichi/ir/ir.h"
#include "taichi/ir/visitors.h"

namespace taichi::lang {
namespace {

class IndexBuilder : public BasicStmtVisitor {
 public:
  explicit IndexBuilder(StmtUseIndex &index) : index_(index) {
    invoke_default_visitor = true;
  }

  void visit(Stmt *stmt) override {
    index_.add_user(stmt);
  }

  void preprocess_container_stmt(Stmt *stmt) override {
    index_.add_user(stmt);
  }

 private:
  StmtUseIndex &index_;
};

}  // namespace

StmtUseIndex::StmtUseIndex(IRNode *root) {
  rebuild(root);
}

void StmtUseIndex::rebuild(IRNode *root) {
  uses_.clear();
  valid_ = true;
  IndexBuilder builder(*this);
  root->accept(&builder);
}

void StmtUseIndex::invalidate() {
  uses_.clear();
  valid_ = false;
}

bool StmtUseIndex::valid() const {
  return valid_;
}

void StmtUseIndex::add_user(Stmt *stmt) {
  TI_ASSERT(valid_);
  for (int i = 0; i < stmt->num_operands(); ++i) {
    if (auto *operand = stmt->operand(i))
      uses_[operand].insert(stmt);
  }
}

void StmtUseIndex::remove_user(Stmt *stmt) {
  TI_ASSERT(valid_);
  for (int i = 0; i < stmt->num_operands(); ++i) {
    auto found = uses_.find(stmt->operand(i));
    if (found != uses_.end()) {
      found->second.erase(stmt);
      if (found->second.empty())
        uses_.erase(found);
    }
  }
}

const StmtUseIndex::Users &StmtUseIndex::users(Stmt *definition) const {
  TI_ASSERT(valid_);
  static const Users empty;
  auto found = uses_.find(definition);
  return found == uses_.end() ? empty : found->second;
}

void StmtUseIndex::replace_uses(Stmt *old_stmt,
                                Stmt *new_stmt,
                                const std::function<void(Stmt *)> &on_change) {
  TI_ASSERT(valid_);
  if (old_stmt == new_stmt)
    return;
  auto found = uses_.find(old_stmt);
  if (found == uses_.end())
    return;
  auto old_users = std::move(found->second);
  uses_.erase(found);
  for (auto *user : old_users) {
    if (user->erased)
      continue;
    user->replace_operand_with(old_stmt, new_stmt);
    if (new_stmt)
      uses_[new_stmt].insert(user);
    if (on_change)
      on_change(user);
  }
}

}  // namespace taichi::lang
