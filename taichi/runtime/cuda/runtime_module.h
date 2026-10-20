#pragma once

#include <string>

namespace llvm {
class Module;
}

namespace taichi::lang {

// Preparation phases of the internal LLVM runtime, not user kernel policies.
enum class CUDARuntimeGroup { core, sparse, statistics };

const char *cuda_runtime_group_name(CUDARuntimeGroup group);
CUDARuntimeGroup cuda_runtime_group(const std::string &entry);

// Partitioning must not duplicate mutable state or observable symbol identity.
// A module which fails this structural check keeps the ordinary eager path.
bool cuda_runtime_can_partition(const llvm::Module &module);

// Retain one set of entry points and its transitive device-call dependencies.
// Other entry points may survive as internal functions, never as kernels.
void retain_cuda_runtime_group(llvm::Module &module, CUDARuntimeGroup group);

}  // namespace taichi::lang
