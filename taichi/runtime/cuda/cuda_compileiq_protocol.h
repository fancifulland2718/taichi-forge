#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

#include "taichi/runtime/cuda/jit_cuda.h"

namespace taichi::lang::cuda {

struct CUDAAdvancedControls {
  std::filesystem::path path;
  std::string sha256;
  std::string source_identity;
};

struct CUDACompileIQProtocolRequest {
  const CUDAKernelArtifact &artifact;
  std::string base_artifact_key;
  std::filesystem::path cache_root;
  std::string ptxas_path;
  std::string ptxas_sha256;
  std::string ptxas_version;
};

struct CUDACompileIQProtocolTelemetrySnapshot {
  std::uint64_t requests{0};
  std::uint64_t cache_hits{0};
  std::uint64_t worker_calls{0};
  std::uint64_t worker_failures{0};
  std::uint64_t worker_wall_ns{0};
  std::uint64_t acf_responses{0};
  std::uint64_t pass_responses{0};
  std::uint64_t fail_open_responses{0};
};

std::optional<CUDAAdvancedControls> resolve_cuda_advanced_controls(
    const CUDACompileIQProtocolRequest &request);

CUDACompileIQProtocolTelemetrySnapshot
get_cuda_compileiq_protocol_telemetry_snapshot();

}  // namespace taichi::lang::cuda
