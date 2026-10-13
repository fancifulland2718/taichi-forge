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

// Alpha is either a finite host constant or a live device scalar. Mutation
// operands are explicit; output overlap with read-only inputs is rejected.
struct AxpyF32Operation {
  static constexpr const char *kind = "axpy_f32";
  static constexpr std::uint32_t version = 1;
  DevicePtr x;
  DevicePtr y;
  std::size_t size{0};
  float alpha{1};
  DevicePtr device_alpha;
  std::shared_ptr<NativeResource> context;
};

struct ScalF32Operation {
  static constexpr const char *kind = "scal_f32";
  static constexpr std::uint32_t version = 1;
  DevicePtr x;
  std::size_t size{0};
  float alpha{1};
  DevicePtr device_alpha;
  std::shared_ptr<NativeResource> context;
};

struct Nrm2F32Operation {
  static constexpr const char *kind = "nrm2_f32";
  static constexpr std::uint32_t version = 1;
  DevicePtr x;
  DevicePtr output;
  std::size_t size{0};
  std::shared_ptr<NativeResource> context;
};

}  // namespace taichi::lang
