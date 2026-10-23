#pragma once

#include <unordered_set>

#include "taichi/inc/constants.h"

namespace taichi::lang {
class IRNode;
class Stmt;

namespace irpass {
// Run only after Offloader and type_check, before global-temporary promotion.
// range_bound_statements must contain BOTH OffloadedRanges maps. These uses
// are not registered operands. Other OffloadedStmt references are read here.
// CUDA/Vulkan dense scalar i32 field loads additionally require a closed
// consumer proof excluding writes to that leaf, external storage, raw pointers
// and calls. CPU retains field snapshots to avoid repeated address work.
bool sink_serial_integer_preludes(
    IRNode *root,
    AutodiffMode autodiff_mode,
    const std::unordered_set<Stmt *> &range_bound_statements);
}  // namespace irpass
}  // namespace taichi::lang
