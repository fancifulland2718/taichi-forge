#pragma once

#include <fstream>
#include <iterator>
#include <string>

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include "taichi/common/core.h"
#include "taichi/runtime/llvm/llvm_aot_metadata.h"

namespace taichi::lang::LLVM {

inline std::string aot_artifact_sha256(llvm::StringRef bytes) {
  llvm::SHA256 hash;
  hash.update(bytes);
  return llvm::toHex(hash.final(), /*LowerCase=*/true);
}

inline std::string read_aot_artifact(const std::string &path) {
  std::ifstream stream(path, std::ios::binary);
  TI_ERROR_IF(!stream, "Cannot read AOT artifact {}", path);
  std::string bytes((std::istreambuf_iterator<char>(stream)),
                     std::istreambuf_iterator<char>());
  TI_ERROR_IF(stream.bad(), "Failed to read AOT artifact {}", path);
  return bytes;
}

inline std::string aot_cuda_contract_sha256(CudaAotArtifactMetadata metadata) {
  metadata.contract_sha256.clear();
  return aot_artifact_sha256(
      liong::json::print(liong::json::serialize(metadata)));
}

inline std::string aot_launch_abi_sha256(const LlvmAotMetadata &metadata) {
  return aot_artifact_sha256(
      metadata.abi_sha256 + "\n" +
      liong::json::print(liong::json::serialize(metadata.kernel_launch_abis)));
}

inline std::string aot_runtime_state_sha256(const LlvmAotMetadata &metadata) {
  return aot_artifact_sha256(metadata.abi_sha256 + "\n" +
                             liong::json::print(liong::json::serialize(
                                 metadata.kernel_runtime_state)));
}

}  // namespace taichi::lang::LLVM
