#pragma once

#include <array>

namespace taichi::lang::cuda::detail {

enum class MemoryAllocationRoute {
  kSynchronous,
  kAsyncMemoryPool,
};

constexpr bool supports_memory_pool(int driver_major,
                                    int driver_minor,
                                    int device_supports_memory_pool) {
  return device_supports_memory_pool != 0 &&
         (driver_major > 11 ||
          (driver_major == 11 && driver_minor >= 2));
}

constexpr MemoryAllocationRoute memory_allocation_route(
    bool supports_memory_pool) {
  return supports_memory_pool ? MemoryAllocationRoute::kAsyncMemoryPool
                             : MemoryAllocationRoute::kSynchronous;
}

// Keep this table in sync with the bundled LLVM NVPTX backend. LLVM 20.1.7
// defines these generic (non-architecture-suffix) processors and their
// minimum PTX ISA versions in llvm/lib/Target/NVPTX/NVPTX.td. A physical CUDA
// device can be newer than the bundled compiler, so select the newest target
// that LLVM can emit without ever selecting a target newer than the device.
struct ComputeCapabilityTarget {
  int compute_capability;
  int ptx_version;
};

inline constexpr std::array<ComputeCapabilityTarget, 23>
    kSupportedComputeCapabilityTargets{{
        {20, 32}, {21, 32}, {30, 32}, {32, 40}, {35, 32}, {37, 41},
        {50, 40}, {52, 41}, {53, 42}, {60, 50}, {61, 50}, {62, 50},
        {70, 60}, {72, 61}, {75, 63}, {80, 70}, {86, 71}, {87, 74},
        {89, 78}, {90, 78}, {100, 86}, {101, 86}, {120, 87},
    }};

struct ComputeCapabilityResolution {
  int device_compute_capability;
  int codegen_compute_capability;
  int ptx_version;
  bool uses_fallback;
};

constexpr ComputeCapabilityResolution resolve_compute_capability_target(
    int device_compute_capability) {
  auto selected = kSupportedComputeCapabilityTargets.front();
  for (const auto &candidate : kSupportedComputeCapabilityTargets) {
    if (candidate.compute_capability > device_compute_capability) {
      break;
    }
    selected = candidate;
  }
  return {device_compute_capability, selected.compute_capability,
          selected.ptx_version,
          selected.compute_capability != device_compute_capability};
}

}  // namespace taichi::lang::cuda::detail
