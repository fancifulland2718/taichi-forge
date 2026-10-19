#pragma once

#include "taichi/runtime/native_operation_description.h"

namespace taichi::lang {

// Packed signed-i32 groups with numeric sum payloads. CPU preserves the frozen
// producer partition policy; CUDA/Vulkan use their direct atomic sum strategy.
struct GroupedReduceOperation {
  static constexpr const char *kind = "grouped_reduce";
  static constexpr uint32_t version = 1;
  size_t size{0}, num_groups{0};
  int value_type{0}, cpu_max_threads{0};
  NativeArrayBinding keys, values, output;
};

}  // namespace taichi::lang
