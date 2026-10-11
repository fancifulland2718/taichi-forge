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
inline constexpr std::uint32_t kCudaPtxAotSchemaVersion = 4;
inline constexpr std::uint32_t kRelocatableLlvmAotSchemaVersion = 5;

struct CudaAotArtifactMetadata {
  std::string sha256;
  uint64_t size{0};
  std::vector<std::string> entry_names;
  std::string target_identity;
  int max_registers{-1};
  bool fast_math{false};
  int llvm_opt_level{-1};
  std::vector<int> tree_relocations;
  std::string contract_sha256;

  TI_IO_DEF(sha256,
            size,
            entry_names,
            target_identity,
            max_registers,
            fast_math,
            llvm_opt_level,
            tree_relocations,
            contract_sha256);
};

struct LlvmAotMetadata {
  std::uint32_t schema_version{0};
  std::map<std::string, uint32_t> required_caps;
  // Empty is only valid for legacy schema-2 text artifacts.
  std::string kernel_format;
  // Schema 4 binds PTX to the serialized launch ABI, with per-kernel hashes.
  std::string abi_sha256;
  std::map<std::string, CudaAotArtifactMetadata> cuda_artifacts;
  std::string resource_binding;

  TI_IO_DEF(schema_version,
            required_caps,
            kernel_format,
            abi_sha256,
            cuda_artifacts,
            resource_binding);
};

inline void validate_llvm_aot_metadata(const LlvmAotMetadata &metadata) {
  TI_ERROR_IF(metadata.schema_version != 2 &&
                  metadata.schema_version != kLlvmAotSchemaVersion &&
                  metadata.schema_version != kCudaPtxAotSchemaVersion &&
                  metadata.schema_version != kRelocatableLlvmAotSchemaVersion,
              "LLVM AOT artifact schema {} is incompatible with this runtime. "
              "Rebuild the artifact with the current Forge AOT "
              "compiler.",
              metadata.schema_version);
  if (metadata.schema_version == kRelocatableLlvmAotSchemaVersion) {
    TI_ERROR_IF(
        metadata.resource_binding != "dense_v1" ||
            (metadata.kernel_format != "bc" &&
             metadata.kernel_format != "ptx") ||
            metadata.abi_sha256.size() != 64,
        "Relocatable LLVM AOT requires dense_v1, BC/PTX and an ABI checksum");
  } else if (metadata.schema_version == 2) {
    TI_ERROR_IF(!metadata.kernel_format.empty() && metadata.kernel_format != "ll",
                "Legacy LLVM AOT schema 2 requires text kernel format.");
  } else if (metadata.schema_version == kCudaPtxAotSchemaVersion) {
    TI_ERROR_IF(metadata.kernel_format != "ptx" ||
                    metadata.abi_sha256.size() != 64,
                "CUDA PTX AOT requires PTX format and a launch ABI checksum.");
  } else {
    TI_ERROR_IF(metadata.kernel_format != "ll" && metadata.kernel_format != "bc",
                "Unsupported or missing LLVM AOT kernel format '{}'.",
                metadata.kernel_format);
  }
  TI_ERROR_IF(metadata.schema_version != kRelocatableLlvmAotSchemaVersion &&
                  !metadata.resource_binding.empty(),
              "Legacy LLVM AOT cannot declare resource relocation");
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
