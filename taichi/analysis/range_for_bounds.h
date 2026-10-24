#pragma once

#include "taichi/codegen/range_for_bounds.h"
#include "taichi/ir/statements.h"

namespace taichi::lang {

// Run after offload/demotion. Each task describes its own bound snapshot even
// when subsequent tasks reuse the same global temporary offset.
inline std::optional<RangeForBounds> describe_range_for_bounds(
    const OffloadedStmt &task) {
  if (task.task_type != OffloadedTaskType::range_for) {
    return std::nullopt;
  }
  RangeForBounds result;
  const auto temporary = [](std::size_t offset) {
    // OffloadedStmt uses size_t; never truncate an address into a valid-looking
    // offset. i32 describes the loaded value, not an artificial address cap.
    TI_ASSERT(offset <= std::numeric_limits<std::int64_t>::max());
    return RangeBound::global_tmp(static_cast<std::int64_t>(offset));
  };
  result.begin = task.const_begin ? RangeBound::constant(task.begin_value)
                                  : temporary(task.begin_offset);
  // SPIR-V can lower argument/shape expressions directly in the range shader.
  // Their end_offset is not a global temporary address.
  if (task.end_stmt == nullptr) {
    result.end = task.const_end ? RangeBound::constant(task.end_value)
                               : temporary(task.end_offset);
  }
  TI_ASSERT(result.valid());
  return result;
}

}  // namespace taichi::lang
