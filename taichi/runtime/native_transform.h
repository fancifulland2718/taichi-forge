#pragma once

#include "taichi/runtime/native_operation_description.h"

namespace taichi::lang {

enum class TransformStrategy { packed_array, packed_field };

// Packed scalar/tensor affine transform. Exact in-place or disjoint ranges;
// only the selected prefix is written. Coefficients retain the native double
// argument contract and convert to the selected arithmetic type at preparation.
struct TransformOperation {
  static constexpr const char *kind = "transform";
  static constexpr std::uint32_t version = 1;
  std::size_t size{0};
  int value_type{0};
  double scale{1};
  double bias{0};
  NativeArrayBinding src;
  NativeArrayBinding dst;
  std::vector<std::uint64_t> element_shape;
  // Preserve the producer's scheduler/launch selection independently of the
  // consumer binding type: packed Fields use one flat CUDA launch and the
  // scalar CPU policy, while Vulkan uses the same packed shader for both.
  TransformStrategy strategy{TransformStrategy::packed_array};
};

std::size_t transform_item_bytes(const TransformOperation &operation);

// One transform/gather/scatter dispatch, including both visible intermediate
// outputs. Writable ranges must be disjoint from each other and both inputs.
struct IndexedTransformOperation {
  static constexpr const char *kind = "indexed_transform";
  static constexpr std::uint32_t version = 1;
  std::size_t src_size{0}, index_size{0}, dst_size{0};
  int value_type{0};
  double scale{1}, bias{0};
  NativeArrayBinding src, transformed, indices, gathered, dst;
};

void validate_indexed_transform_operation(
    Device &device, const IndexedTransformOperation &operation);

class NativeOperation;
namespace gfx {
std::shared_ptr<NativeOperation> prepare_vulkan_transform_operation(
    Device &device,
    const TransformOperation &operation);
std::shared_ptr<NativeOperation> prepare_vulkan_indexed_transform_operation(
    Device &device, const IndexedTransformOperation &operation);
}
}  // namespace taichi::lang
