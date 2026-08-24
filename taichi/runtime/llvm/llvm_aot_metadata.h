#pragma once

#include <cstdint>
#include <map>
#include <string>

#include "taichi/common/core.h"
#include "taichi/common/serialization.h"

namespace taichi::lang::LLVM {

inline constexpr char kLlvmAotMetadataFilename[] = "aot_metadata.json";
inline constexpr std::uint32_t kLlvmAotSchemaVersion = 1;

struct LlvmAotMetadata {
  std::uint32_t schema_version{0};
  std::map<std::string, uint32_t> required_caps;

  TI_IO_DEF(schema_version, required_caps);
};

inline void validate_llvm_aot_metadata(const LlvmAotMetadata &metadata) {
  TI_ERROR_IF(metadata.schema_version != kLlvmAotSchemaVersion,
              "LLVM AOT artifact schema {} is incompatible with runtime "
              "schema {}. Rebuild the artifact with the current Forge AOT "
              "compiler.",
              metadata.schema_version, kLlvmAotSchemaVersion);
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
