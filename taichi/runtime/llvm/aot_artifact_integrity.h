#pragma once

#include <fstream>
#include <iterator>
#include <string>

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include "taichi/common/core.h"

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

}  // namespace taichi::lang::LLVM
