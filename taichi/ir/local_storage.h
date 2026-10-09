#pragma once

#include "taichi/ir/statements.h"

namespace taichi::lang {

// A conservative grouping key, not an alias or coverage decision. Nested and
// byte-offset pointers retain the caller's unclassified-address fallback.
inline Stmt *direct_local_allocation(Stmt *address) {
  if (address->is<AllocaStmt>())
    return address;
  auto *component = address->cast<MatrixPtrStmt>();
  if (component && component->origin->is<AllocaStmt>() &&
      component->offset_used_as_index())
    return component->origin;
  return nullptr;
}

}  // namespace taichi::lang
