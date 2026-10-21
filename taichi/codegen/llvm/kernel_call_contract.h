#pragma once

#include "taichi/analysis/graph_kernel_metadata.h"
#include "taichi/program/callable.h"

namespace taichi::lang::LLVM {

// The compiler owns these facts. Cache and deployment containers carry this
// same contract instead of rebuilding it from the frontend Kernel. Backend
// task records remain in the shared LLVMCompiledKernel; module flags carry
// its context/gated launch requirements. This value owns no executable or IR.
// Containers explicitly enumerate fields to preserve their existing wire
// layouts; this base itself does not introduce another serialized envelope.
struct KernelCallContract {
  std::vector<std::pair<std::vector<int>, Callable::Parameter>> args;
  std::vector<Callable::Ret> rets;
  std::vector<int> used_snode_tree_ids;
  GraphKernelMetadata graph_metadata;

  const StructType *ret_type = nullptr;
  size_t ret_size{0};
  const StructType *args_type = nullptr;
  size_t args_size{0};
};

}  // namespace taichi::lang::LLVM
