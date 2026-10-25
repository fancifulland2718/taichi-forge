#pragma once

#include "taichi/ir/statements.h"

namespace taichi::lang {

// These references point into a task body, so registering them as ordinary
// operands would introduce forward SSA edges. Rewriters update them alongside
// ordinary uses without rescanning the surrounding IR. A subtree rewrite must
// not retarget a bound whose definition lies outside its replacement scope.
inline void replace_direct_range_bound_uses(Stmt *old_stmt, Stmt *replacement,
                                            IRNode *scope = nullptr) {
  auto *owner = old_stmt->parent ? old_stmt->parent->parent_stmt() : nullptr;
  auto *task = owner ? owner->cast<OffloadedStmt>() : nullptr;
  if (!task)
    return;
  for (auto **bound : {&task->begin_stmt, &task->end_stmt}) {
    if (*bound == old_stmt) {
      if (scope && scope != old_stmt) {
        bool included = false;
        for (auto *block = old_stmt->parent; block; block = block->parent_block()) {
          if (scope == block || scope == block->parent_stmt()) {
            included = true;
            break;
          }
        }
        if (!included)
          return;
      }
      TI_ASSERT(replacement);
      *bound = replacement;
    }
  }
}

}  // namespace taichi::lang
