#pragma once

#include <cstdint>
#include <string>

#include "taichi/program/compile_config.h"
#include "taichi/runtime/cuda/jit_cuda.h"

namespace taichi::lang::cuda {

struct CUDAArtifactProviderTelemetrySnapshot {
  std::uint64_t external_requests{0};
  std::uint64_t cache_hits{0};
  std::uint64_t cache_misses{0};
  std::uint64_t compile_calls{0};
  std::uint64_t compile_failures{0};
  std::uint64_t compile_wall_ns{0};
  std::uint64_t cubin_loads{0};
  std::uint64_t cubin_unloads{0};
  std::uint64_t cubin_bytes{0};
  std::uint64_t cubin_current_bytes{0};
  std::uint64_t cubin_peak_bytes{0};
  std::uint64_t entry_points_loaded{0};
  std::uint64_t multi_entry_artifacts{0};
};

CUDAKernelArtifact select_cuda_kernel_artifact(CUDAKernelArtifact artifact,
                                               const CompileConfig &config);

std::string cuda_artifact_provider_configuration_identity();

void record_cuda_artifact_load(std::size_t entry_count,
                               bool is_cubin,
                               std::size_t bytes) noexcept;
void record_cuda_artifact_unload(bool is_cubin, std::size_t bytes) noexcept;

CUDAArtifactProviderTelemetrySnapshot
get_cuda_artifact_provider_telemetry_snapshot();

}  // namespace taichi::lang::cuda
