#pragma once

#include <array>
#include <cstddef>

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

// CUDA minor-version compatibility is defined within a Toolkit major.  Do not
// reject, for example, a CUDA 12.8-built runtime solely because the driver
// reports CUDA 12.0.  Individual APIs and embedded PTX/cubins remain subject to
// their own capability checks and launch-time errors.
constexpr bool supports_cuda_toolkit_major(int driver_major,
                                           int toolkit_major) {
  return driver_major >= toolkit_major;
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

// cp.async first appears in PTX 7.0 and requires sm_80.  Keep the mechanism
// internal: this admission describes the compiler-generated block-local fetch
// read-only pattern, not a public CUDA instruction API.  The 8 KiB floor is
// the first qualified workload slice; smaller or read-write prologues retain
// the ordinary synchronous load/store lowering because they are outside the
// qualified performance and correctness envelope.
inline constexpr std::size_t kCudaAsyncTileMinBlsBytes = 8 * 1024;

enum class CudaAsyncTileAdmissionReason : std::uint8_t {
  kAdmitted,
  kBelowSize,
  kReadWriteBls,
  kUnsupportedWidth,
  kNonDirectAddress,
  kAliasUnknown,
  kSharedMemoryPressure,
  kTargetCapability,
  kCostGate,
};

constexpr CudaAsyncTileAdmissionReason cuda_async_tile_copy_admission(
    int compute_capability,
    int ptx_version,
    std::size_t bls_bytes,
    int copy_bytes,
    bool direct_global_to_bls_copy,
    bool read_only_bls) {
  if (compute_capability < 80 || ptx_version < 70) {
    return CudaAsyncTileAdmissionReason::kTargetCapability;
  }
  if (bls_bytes < kCudaAsyncTileMinBlsBytes) {
    return CudaAsyncTileAdmissionReason::kBelowSize;
  }
  if (!read_only_bls) {
    return CudaAsyncTileAdmissionReason::kReadWriteBls;
  }
  if (copy_bytes != 4 && copy_bytes != 8 && copy_bytes != 16) {
    return CudaAsyncTileAdmissionReason::kUnsupportedWidth;
  }
  if (!direct_global_to_bls_copy) {
    return CudaAsyncTileAdmissionReason::kNonDirectAddress;
  }
  return CudaAsyncTileAdmissionReason::kAdmitted;
}

constexpr bool cuda_async_tile_copy_admitted(int compute_capability,
                                             int ptx_version,
                                             std::size_t bls_bytes,
                                             int copy_bytes,
                                             bool direct_global_to_bls_copy,
                                             bool read_only_bls) {
  return cuda_async_tile_copy_admission(
             compute_capability, ptx_version, bls_bytes, copy_bytes,
             direct_global_to_bls_copy, read_only_bls) ==
         CudaAsyncTileAdmissionReason::kAdmitted;
}

}  // namespace taichi::lang::cuda::detail
