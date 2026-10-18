#pragma once

#include <cstdint>
#include <map>
#include <string>

#include "taichi/common/core.h"
#include "taichi/common/serialization.h"
#include "taichi/aot/initial_root_data.h"
#include "taichi/runtime/cuda/kernel_launch_abi.h"

namespace taichi::lang::LLVM {

inline constexpr char kLlvmAotMetadataFilename[] = "aot_metadata.json";
// Schema 2 introduced the exact Graph image format. Schema 3 declares the
// kernel representation; old runtimes must reject it instead of reading a
// stale .ll beside a newer .bc. Binary task/Graph layouts are unchanged.
inline constexpr std::uint32_t kLlvmAotSchemaVersion = 3;
inline constexpr std::uint32_t kCudaPtxAotSchemaVersion = 4;
inline constexpr std::uint32_t kRelocatableLlvmAotSchemaVersion = 5;
inline constexpr std::uint32_t kCudaCubinAotSchemaVersion = 6;
inline constexpr std::uint32_t kInitialRootDataAotSchemaVersion = 7;
inline constexpr std::uint32_t kKernelLaunchAbiAotSchemaVersion = 8;

inline bool is_cuda_aot_artifact_format(const std::string &format) {
  return format == "ptx" || format == "cubin";
}

inline bool is_sparse_aot_resource_binding(const std::string &binding) {
  return binding == "sparse_v1" || binding == "sparse_v2";
}

inline bool is_relocatable_aot_resource_binding(const std::string &binding) {
  // list_v1 certifies exact task-to-tree list ownership and CUDA's current
  // device list protocol. Node layouts, not this name, govern root snapshots.
  return binding == "dense_v1" || binding == "list_v1" ||
         is_sparse_aot_resource_binding(binding);
}

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
  // Separate from the version-5 per-kernel contract hash for compatibility.
  std::map<std::string, std::string> cuda_cubin_providers;
  std::string resource_binding;
  std::vector<aot::InitialRootData> initial_root_data;
  // Non-legacy call contracts, including an appended RuntimeContext binding.
  // Schema 8 and the exact strings prevent old readers from launching code
  // with an insufficient context or a missing argument prefix.
  std::map<std::string, std::string> kernel_launch_abis;
  std::string launch_abi_sha256;

  TI_IO_DEF(schema_version,
            required_caps,
            kernel_format,
            abi_sha256,
            cuda_artifacts,
            cuda_cubin_providers,
            resource_binding,
            initial_root_data,
            kernel_launch_abis,
            launch_abi_sha256);
};

inline void validate_llvm_aot_metadata(const LlvmAotMetadata &metadata) {
  auto base_schema = metadata.schema_version;
  const bool launch_abi = base_schema == kKernelLaunchAbiAotSchemaVersion;
  TI_ERROR_IF(launch_abi != !metadata.kernel_launch_abis.empty() ||
                  (!launch_abi && !metadata.launch_abi_sha256.empty()),
              "AOT kernel launch ABI declarations require schema 8");
  if (launch_abi) {
    TI_ERROR_IF(metadata.abi_sha256.size() != 64 ||
                    metadata.launch_abi_sha256.size() != 64 ||
                    (metadata.kernel_format != "bc" &&
                     !is_cuda_aot_artifact_format(metadata.kernel_format)),
                "AOT kernel launch ABI requires BC/PTX/cubin and checksums");
    for (const auto &[name, abi] : metadata.kernel_launch_abis) {
      TI_ERROR_IF(name.empty() || name.find('\0') != std::string::npos ||
                      !cuda::is_declared_launch_abi(abi),
                  "Unsupported AOT kernel launch ABI declaration: {}", name);
    }
    // The launch declaration extends each existing representation/resource
    // contract; it does not replace those checks or duplicate their payloads.
    base_schema = !metadata.initial_root_data.empty()
                      ? kInitialRootDataAotSchemaVersion
                  : metadata.kernel_format == "cubin"
                      ? kCudaCubinAotSchemaVersion
                  : !metadata.resource_binding.empty()
                      ? kRelocatableLlvmAotSchemaVersion
                  : metadata.kernel_format == "ptx" ? kCudaPtxAotSchemaVersion
                                                    : kLlvmAotSchemaVersion;
  }
  const bool initial_state =
      base_schema == kInitialRootDataAotSchemaVersion;
  TI_ERROR_IF(initial_state != !metadata.initial_root_data.empty() ||
                  (initial_state && metadata.resource_binding != "dense_v1" &&
                   metadata.resource_binding != "list_v1"),
              "AOT initial root data requires schema 7 and dense root binding.");
  // Reuse the exact BC/PTX or cubin contract without copying payload metadata.
  const auto schema = initial_state ? (metadata.kernel_format == "cubin"
                                           ? kCudaCubinAotSchemaVersion
                                           : kRelocatableLlvmAotSchemaVersion)
                                    : base_schema;
  TI_ERROR_IF(schema != 2 && schema != kLlvmAotSchemaVersion &&
                  schema != kCudaPtxAotSchemaVersion &&
                  schema != kRelocatableLlvmAotSchemaVersion &&
                  schema != kCudaCubinAotSchemaVersion,
              "LLVM AOT artifact schema {} is incompatible with this runtime. "
              "Rebuild the artifact with the current Forge AOT "
              "compiler.",
              metadata.schema_version);
  if (schema == kCudaCubinAotSchemaVersion) {
    TI_ERROR_IF(metadata.kernel_format != "cubin" ||
                    metadata.abi_sha256.size() != 64 ||
                    (!metadata.resource_binding.empty() &&
                     !is_relocatable_aot_resource_binding(
                         metadata.resource_binding)),
                "CUDA cubin AOT requires cubin, an ABI checksum and a supported "
                "resource binding");
  } else if (schema == kRelocatableLlvmAotSchemaVersion) {
    TI_ERROR_IF(
        !is_relocatable_aot_resource_binding(metadata.resource_binding) ||
            (metadata.kernel_format != "bc" &&
             metadata.kernel_format != "ptx") ||
            metadata.abi_sha256.size() != 64,
        "Relocatable LLVM AOT requires dense_v1/list_v1/sparse_v1/sparse_v2, BC/PTX and an ABI "
        "checksum");
  } else if (schema == 2) {
    TI_ERROR_IF(!metadata.kernel_format.empty() && metadata.kernel_format != "ll",
                "Legacy LLVM AOT schema 2 requires text kernel format.");
  } else if (schema == kCudaPtxAotSchemaVersion) {
    TI_ERROR_IF(metadata.kernel_format != "ptx" ||
                    metadata.abi_sha256.size() != 64,
                "CUDA PTX AOT requires PTX format and a launch ABI checksum.");
  } else {
    TI_ERROR_IF(metadata.kernel_format != "ll" && metadata.kernel_format != "bc",
                "Unsupported or missing LLVM AOT kernel format '{}'.",
                metadata.kernel_format);
  }
  TI_ERROR_IF(schema != kRelocatableLlvmAotSchemaVersion &&
                  schema != kCudaCubinAotSchemaVersion &&
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
  // Cubins have binary compatibility within a compute-capability major, in
  // the forward minor direction. PTX retains its broader JIT compatibility.
  if (metadata.kernel_format == "cubin") {
    TI_ERROR_IF(device_compute_capability / 10 !=
                    static_cast<int>(cc_it->second) / 10,
                "CUDA cubin target {} is incompatible with device {}: "
                "compute-capability major versions must match",
                cc_it->second, device_compute_capability);
    return;  // Producer PTX version does not constrain loading machine code.
  }
  TI_ERROR_IF(device_ptx_version < static_cast<int>(ptx_it->second),
              "CUDA AOT artifact requires PTX {}, but the active CUDA target "
              "provides PTX {}.",
              ptx_it->second, device_ptx_version);
}

}  // namespace taichi::lang::LLVM
