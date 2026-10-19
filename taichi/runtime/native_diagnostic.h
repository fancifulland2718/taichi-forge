#pragma once

#include <optional>
#include "taichi/runtime/native_operation_description.h"

namespace taichi::lang {

struct DiagnosticOperation {
  size_t size{0};
  int value_type{0}, op{0}, cpu_max_threads{0};
  uint64_t vulkan_items_per_group{16384};
  NativeArrayBinding values, output;
};

// The output remains a count. Negative i32 represents overflow, preserving
// zero/nonzero truth; host Boolean interpretations do not change this ABI.
struct CheckCountOperation : DiagnosticOperation {
  static constexpr const char *kind = "check_count";
  static constexpr uint32_t version = 1;
  int32_t lower{0}, upper{0};
};

struct MetricReduceOperation : DiagnosticOperation {
  static constexpr const char *kind = "metric_reduce";
  static constexpr uint32_t version = 1;
  std::optional<NativeArrayBinding> other;
};

}  // namespace taichi::lang
