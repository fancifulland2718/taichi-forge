#pragma once

#include "taichi/runtime/native_operation_description.h"

namespace taichi::lang {

// Packed in-place inclusive sum. Each prepared operation owns its scratch;
// the enclosing executor retains values through completion.
struct ScanOperation {
  static constexpr const char *kind = "scan";
  static constexpr std::uint32_t version = 1;
  std::size_t size{0};
  int value_type{0};  // Same numeric type IDs as the shared JIT primitives.
  NativeArrayBinding values;
};

class NativeOperation;
namespace gfx {
std::shared_ptr<NativeOperation> prepare_vulkan_scan_operation(
    Device &device,
    const ScanOperation &operation);
}

}  // namespace taichi::lang
