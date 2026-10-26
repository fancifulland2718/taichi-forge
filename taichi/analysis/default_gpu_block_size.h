#pragma once

#include "taichi/analysis/range_dispatch.h"
#include "taichi/analysis/host_range_interval.h"

namespace taichi::lang {

// Spread a known range over otherwise idle compute units, without increasing
// the number of scheduling waves. Also trim idle lanes in a single block when
// unit count is unknown. This is a compile-time heuristic, not occupancy or a
// workload limit; no argument-value specialization or trial launch is involved.
inline int select_default_gpu_block_size(OffloadedStmt &task,
                                         uint32_t compute_units = 0,
                                         uint32_t subgroup_size = 32) {
  const int original = task.block_dim;
  const auto granularity = subgroup_size ? subgroup_size : 32;
  if ((task.device != Arch::cuda && task.device != Arch::vulkan) ||
      task.task_type != OffloadedTaskType::range_for ||
      task.source_block_dim_explicit ||
      original <= 0 || static_cast<uint32_t>(original) <= granularity ||
      (granularity & (granularity - 1)) != 0 ||
      (original & (original - 1)) != 0)
    return original;

  std::int64_t count;
  if (task.const_begin && task.const_end) {
    count = std::int64_t(task.end_value) - task.begin_value;
  } else if (task.host_launch_range) {
    const auto begin = static_host_range_interval(task.host_launch_range->begin);
    const auto end = static_host_range_interval(task.host_launch_range->end);
    if (!begin || !end)
      return original;
    count = end->upper - begin->lower;  // Upper bound, not the actual workload.
  } else {
    return original;
  }
  if (count <= 0)
    return original;

  const auto units = std::max(uint32_t{1}, compute_units);
  int selected = original;
  while (static_cast<uint32_t>(selected / 2) >= granularity &&
         (count + selected / 2 - 1) / (selected / 2) <= units)
    selected /= 2;
  if (selected == original)
    return original;  // No body walk for large or unknown workloads.

  // Eliding empty groups alone does NOT prove that changing group size is
  // legal. In particular, keep all TLS/BLS preparation and reductions on their
  // existing geometry, including atomics that have not been promoted to TLS.
  // OffloadedStmt reserves one sentinel byte even when it has no TLS users.
  if (task.tls_size > 1 || task.tls_prologue || task.tls_epilogue ||
      describe_empty_range_group_elision(task) !=
          EmptyRangeGroupProof::independent_iterations ||
      irpass::analysis::any_statement(task.body.get(), [](Stmt *stmt) {
        return stmt->is<AtomicOpStmt>();
      }))
    return original;

  return selected;
}

}  // namespace taichi::lang
