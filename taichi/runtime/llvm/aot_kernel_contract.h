#pragma once

#include "taichi/codegen/llvm/cuda_graph_gate.h"
#include "taichi/codegen/llvm/kernel_call_contract.h"
#include "taichi/runtime/cuda/kernel_launch_abi.h"
#include "taichi/runtime/llvm/aot_snode_relocation.h"
#include "taichi/runtime/llvm/runtime_context_abi.h"

namespace taichi::lang::LLVM {

// Project the transformed code's hidden requirements into the existing wire
// declaration. The caller's requested variant is not a second ABI authority.
inline const char *compiled_kernel_launch_abi(const llvm::Module &module) {
  const bool gated = has_cuda_graph_gate(module);
  if (runtime_context_abi(module) == kRuntimeContextABI)
    return gated ? cuda::kGraphGateContextLaunchABI : cuda::kContextLaunchABI;
  return gated ? cuda::kGraphGateLaunchABI : "ordinary";
}

// Apply a deployment projection to both relocatable code and its actual tree
// dependencies. Work on the export clone; never mutate the producer contract.
// Source-analysis graph metadata is not the deployed storage closure.
inline void remap_aot_kernel_tree_ids(llvm::Module &module,
                                      KernelCallContract &contract,
                                      const std::map<int, int> &mapping) {
  std::vector<int> dependencies;
  dependencies.reserve(contract.used_snode_tree_ids.size());
  for (int id : contract.used_snode_tree_ids) {
    const auto found = mapping.find(id);
    TI_ERROR_IF(found == mapping.end() || found->second < 0,
                "LLVM AOT tree dependency {} has no declared resource", id);
    dependencies.push_back(found->second);
  }
  for (const auto &[id, global] : aot_tree_id_globals(module)) {
    TI_ERROR_IF(std::find(contract.used_snode_tree_ids.begin(),
                          contract.used_snode_tree_ids.end(), id) ==
                    contract.used_snode_tree_ids.end(),
                "LLVM AOT tree relocation {} is absent from the kernel contract",
                id);
  }
  remap_aot_tree_id_symbols(module, mapping);
  contract.used_snode_tree_ids = std::move(dependencies);
}

}  // namespace taichi::lang::LLVM
