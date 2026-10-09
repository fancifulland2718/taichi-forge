#pragma once

#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace taichi::lang {

class IRNode;
class Stmt;

// A pass-scoped index of registered operands, including container operands.
// The owner must register inserted statements, remove users before changing
// their operands/erasing them, and register them again after operand changes.
// Moving a statement within the indexed tree does not change its uses. Unknown
// mutations invalidate the index; rebuild it before the next query. This is not
// a dominance, alias or liveness analysis, and does not index non-operand
// fields such as OffloadedStmt::end_stmt. Those retain their pass-specific
// contracts.
class StmtUseIndex {
 public:
  using Users = std::unordered_set<Stmt *>;

  StmtUseIndex() = default;
  explicit StmtUseIndex(IRNode *root);

  void rebuild(IRNode *root);
  void invalidate();
  bool valid() const;

  // These operations affect only this statement's outgoing operand uses, not
  // its children or the users of its result. Registration is idempotent.
  void add_user(Stmt *stmt);
  void remove_user(Stmt *stmt);
  const Users &users(Stmt *definition) const;

  // Updates the index immediately, including A -> B -> C replacement chains.
  // Callback receives each changed live user once and must not mutate operands.
  // The caller proves visibility/type legality and manages statement ownership.
  void replace_uses(Stmt *old_stmt,
                    Stmt *new_stmt,
                    const std::function<void(Stmt *)> &on_change = {});

 private:
  std::unordered_map<Stmt *, Users> uses_;
  bool valid_{false};
};

}  // namespace taichi::lang
