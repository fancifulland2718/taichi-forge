#pragma once

#include "taichi/analysis/range_for_bounds.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/snode.h"

namespace taichi::lang {

// A code-generation fact, independent of whether a caller selects device range
// dispatch. Keep unknown intrinsics/calls conservative: their physical-thread
// or workspace contract is not described by an ordinary grid-stride loop.
inline bool supports_device_range_dispatch(OffloadedStmt &task) {
  const auto bounds = describe_range_for_bounds(task);
  if (!bounds || bounds->begin.kind == RangeBoundKind::unknown ||
      bounds->end.kind == RangeBoundKind::unknown || bounds->const_range() ||
      task.one_to_one ||
      task.reversed || task.is_bit_vectorized || task.external_shared_staged ||
      task.bls_size || task.tls_prologue || task.tls_epilogue ||
      task.bls_prologue || task.bls_epilogue || task.mesh_prologue || !task.body)
    return false;
  return !irpass::analysis::any_statement(task.body.get(), [](Stmt *stmt) {
    if (stmt->is<InternalFuncStmt>() || stmt->is<ExternalFuncCallStmt>() ||
        stmt->is<FuncCallStmt>() || stmt->is<GlobalThreadIndexStmt>() ||
        stmt->is<BlockCornerIndexStmt>() || stmt->is<LoopLinearIndexStmt>() ||
        stmt->is<ThreadLocalPtrStmt>() ||
        stmt->is<BlockLocalPtrStmt>() || stmt->is<RandStmt>() ||
        stmt->is<SNodeOpStmt>())
      return true;
    if (auto *allocation = stmt->cast<AllocaStmt>())
      return allocation->is_shared;
    if (auto *lookup = stmt->cast<SNodeLookupStmt>())
      return lookup->snode->type != SNodeType::dense &&
             lookup->snode->type != SNodeType::root;
    return false;
  });
}

}  // namespace taichi::lang
