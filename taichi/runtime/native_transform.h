#pragma once

#include "taichi/runtime/native_operation_description.h"

namespace taichi::lang {

// Packed scalar affine transform. Exact in-place or physically disjoint ranges;
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
};

class NativeOperation;
namespace gfx {
std::shared_ptr<NativeOperation> prepare_vulkan_transform_operation(
    Device &device,
    const TransformOperation &operation);
}
}  // namespace taichi::lang
