#pragma once

#include "taichi/runtime/native_operation_description.h"

namespace taichi::lang {

// The graph owns scratch explicitly; replay never consults the Program arena.
struct RetainedScanOperation {
  static constexpr const char *kind = "cuda_retained_scan";
  static constexpr std::uint32_t version = 1;
  int size{0};
  DataType dtype{PrimitiveType::i32};
  NativeArrayBinding values, scratch;
};

}  // namespace taichi::lang
