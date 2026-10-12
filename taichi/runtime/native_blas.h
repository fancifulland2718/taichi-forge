#pragma once

#include "taichi/runtime/native_operation.h"

namespace taichi::lang {

struct BlasContextDescription {
  static constexpr const char *kind = "cublas_context";
  static constexpr std::uint32_t version = 1;
};

// Compact scalar f32 vectors. The device scalar output is overwritten on every
// execution and must be disjoint from both inputs. Inputs may alias each other.
struct DotF32Operation {
  static constexpr const char *kind = "dot_f32";
  static constexpr std::uint32_t version = 1;
  DevicePtr x;
  DevicePtr y;
  DevicePtr output;
  std::size_t size;
  std::shared_ptr<NativeResource> context;
};

}  // namespace taichi::lang
