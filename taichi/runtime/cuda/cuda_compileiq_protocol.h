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

enum class CUDAAdvancedControlsMode {
  baseline,
  apply_explicit_acf,
  request_tuning,
};

struct CUDAAdvancedControlsConfiguration {
  CUDAAdvancedControlsMode mode{CUDAAdvancedControlsMode::baseline};
  std::string explicit_acf_path;
  std::string worker_path;
  std::string python_path;
  bool nested_tuning_request_rejected{false};
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
  std::uint64_t nested_requests_rejected{0};
};

CUDAAdvancedControlsConfiguration
cuda_advanced_controls_configuration_from_environment();

const char *cuda_advanced_controls_mode_name(
    CUDAAdvancedControlsMode mode) noexcept;

std::optional<CUDAAdvancedControls> resolve_cuda_advanced_controls(
    const CUDACompileIQProtocolRequest &request,
    const CUDAAdvancedControlsConfiguration &configuration);

CUDACompileIQProtocolTelemetrySnapshot
get_cuda_compileiq_protocol_telemetry_snapshot();

}  // namespace taichi::lang::cuda
