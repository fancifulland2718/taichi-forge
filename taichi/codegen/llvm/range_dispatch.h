#pragma once

#include <stdexcept>
#include "taichi/codegen/llvm/llvm_compiled_data.h"

namespace taichi::lang::LLVM {

// Shared cold proof validation for persisted entries and native recordings.
// Runtime address ownership and driver availability belong to the recorder.
template <typename Index>
void validate_device_range_selection(const std::vector<OffloadedTask> &tasks,
                                     const std::vector<Index> &selection,
                                     std::size_t temporary_bytes) {
  for (std::size_t i = 0; i < selection.size(); ++i) {
    const auto index = selection[i];
    if (index >= tasks.size() || (i && index <= selection[i - 1]))
      throw std::invalid_argument(
          "CUDA range task indices must be in topology order and unique");
    const auto &task = tasks[index];
    if (task.device_range_dispatch_version != 1 ||
        !task.valid_device_range_dispatch_contract())
      throw std::invalid_argument("CUDA range task has no final IR qualification");
    for (const auto &bound : {task.range_bounds->begin, task.range_bounds->end}) {
      if (!(bound.kind == RangeBoundKind::constant_i32 && bound.valid()) &&
          !bound.fits_global_tmps(temporary_bytes))
        throw std::invalid_argument("CUDA range bound exceeds its workspace");
    }
  }
}

}  // namespace taichi::lang::LLVM
