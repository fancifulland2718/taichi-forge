#pragma once

#include "taichi/runtime/native_operation_description.h"

namespace taichi::lang {

// Packed, nonempty reduction. Output may be scalar or a vector; only its first
// element is written. Read and write ranges must be disjoint.
struct ReduceOperation {
  static constexpr const char *kind = "reduce";
  static constexpr std::uint32_t version = 1;
  std::size_t size{0};
  int value_type{0};
  int op{0};  // sum, min, max
  // Zero selects the consumer configuration. Exported CPU recordings freeze
  // the producer configuration to preserve its partition and floating order.
  int cpu_max_threads{0};
  NativeArrayBinding values;
  NativeArrayBinding output;
};

class NativeOperation;
namespace gfx {
std::shared_ptr<NativeOperation> prepare_vulkan_reduce_operation(
    Device &device,
    const ReduceOperation &operation);
}
}  // namespace taichi::lang
