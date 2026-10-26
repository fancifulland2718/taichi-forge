#pragma once

#include "taichi/ir/statements.h"

namespace taichi::lang {

inline bool is_demotable_dense_struct_for(const OffloadedStmt &task) {
  return task.task_type == OffloadedStmt::TaskType::struct_for && task.snode &&
         task.snode->is_path_all_dense;
}

// Execution plans describe emitted tasks, including this existing one-to-one
// lowering. Early consumers need the same proof as the lowering pass itself.
inline OffloadedTaskType task_type_after_dense_demotion(
    const OffloadedStmt &task, bool demote_dense) {
  return demote_dense && is_demotable_dense_struct_for(task)
             ? OffloadedTaskType::range_for
             : task.task_type;
}

}  // namespace taichi::lang
