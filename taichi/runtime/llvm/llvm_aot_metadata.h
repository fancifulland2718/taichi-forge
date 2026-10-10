#pragma once

#include <cstdint>
#include <map>
#include <string>

#include "taichi/common/core.h"
#include "taichi/common/serialization.h"

namespace taichi::lang::LLVM {

inline constexpr char kLlvmAotMetadataFilename[] = "aot_metadata.json";
// Schema 2 introduced the exact Graph image format. Schema 3 declares the
// kernel representation; old runtimes must reject it instead of reading a
// stale .ll beside a newer .bc. Binary task/Graph layouts are unchanged.
inline constexpr std::uint32_t kLlvmAotSchemaVersion = 3;

struct LlvmAotMetadata {
  std::uint32_t schema_version{0};
  std::map<std::string, uint32_t> required_caps;
  // Empty is only valid for legacy schema-2 text artifacts.
  std::string kernel_format;

  TI_IO_DEF(schema_version, required_caps, kernel_format);
};

inline void validate_llvm_aot_metadata(const LlvmAotMetadata &metadata) {
  TI_ERROR_IF(metadata.schema_version != 2 &&
                  metadata.schema_version != kLlvmAotSchemaVersion,
              "LLVM AOT artifact schema {} is incompatible with runtime "
              "schema {}. Rebuild the artifact with the current Forge AOT "
              "compiler.",
              metadata.schema_version, kLlvmAotSchemaVersion);
  if (metadata.schema_version == 2) {
    TI_ERROR_IF(!metadata.kernel_format.empty() && metadata.kernel_format != "ll",
                "Legacy LLVM AOT schema 2 requires text kernel format.");
  } else {
    TI_ERROR_IF(metadata.kernel_format != "ll" && metadata.kernel_format != "bc",
                "Unsupported or missing LLVM AOT kernel format '{}'.",
                metadata.kernel_format);
  }
}

inline void validate_cuda_aot_metadata(const LlvmAotMetadata &metadata,
                                       int device_compute_capability,
                                       int device_ptx_version) {
  const auto cc_it =
      metadata.required_caps.find("cuda_compute_capability");
  const auto ptx_it = metadata.required_caps.find("cuda_ptx_version");
  TI_ERROR_IF(cc_it == metadata.required_caps.end() ||
                  ptx_it == metadata.required_caps.end(),
              "CUDA AOT metadata is missing required compute capability or "
              "PTX version.");
  TI_ERROR_IF(
      device_compute_capability < static_cast<int>(cc_it->second),
      "CUDA AOT artifact requires compute capability {}, but the active "
      "device provides {}.",
      cc_it->second, device_compute_capability);
  TI_ERROR_IF(device_ptx_version < static_cast<int>(ptx_it->second),
              "CUDA AOT artifact requires PTX {}, but the active CUDA target "
              "provides PTX {}.",
              ptx_it->second, device_ptx_version);
}

}  // namespace taichi::lang::LLVM
