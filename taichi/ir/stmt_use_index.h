#pragma once

#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace taichi::lang {

class IRNode;
class Stmt;
class Block;

// A pass-scoped index of registered operands, including container operands.
// The owner must register inserted statements, remove users before changing
// their operands/erasing them, and register them again after operand changes.
// Moving a statement within the indexed tree does not change its uses. Unknown
// mutations invalidate the index; rebuild it before the next query. This is not
// a dominance, alias or liveness analysis, and does not index non-operand
// fields. replace_uses also updates direct task range-bound references in
// their owning body; these forward references are not ordinary SSA users.
class StmtUseIndex {
 public:
  using Users = std::unordered_set<Stmt *>;

  StmtUseIndex() = default;
  explicit StmtUseIndex(IRNode *root);

  // A pass rewriting only definitions in one block can omit all other
  // definitions from its index. Users may still be in descendant blocks.
  void rebuild(IRNode *root, Block *definition_block = nullptr);
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
  IRNode *scope_{nullptr};
  Block *definition_block_{nullptr};
  bool valid_{false};
};

}  // namespace taichi::lang
