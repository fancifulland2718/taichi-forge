#pragma once

#include "taichi/runtime/native_operation_description.h"

namespace taichi::lang {

enum class ReduceStrategy {
  packed_array,
  scalar_field,
  strided_scalar,
  packed_field
};

// Nonempty reduction. Output may be scalar or a vector; only its first
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
  // Scalar Fields select packed arithmetic when their actual record strides
  // are contiguous; member ndarrays retain the explicit strided method.
  ReduceStrategy strategy{ReduceStrategy::packed_array};
  // Packed tensor Fields reduce every lane of the first output record, using
  // the existing lane order and one shared workspace.
  std::vector<std::uint64_t> element_shape;
};

struct ReduceAddressing {
  std::uint64_t values_offset{0}, values_stride{0}, output_offset{0}, output_stride{0};
  std::size_t lanes{1};
};
std::size_t reduce_item_bytes(const ReduceOperation &operation);
ReduceAddressing qualify_reduce_addressing(
    Arch arch, Device &device, const ReduceOperation &operation);

class NativeOperation;
namespace gfx {
std::shared_ptr<NativeOperation> prepare_vulkan_reduce_operation(
    Device &device,
    const ReduceOperation &operation);
}
}  // namespace taichi::lang
