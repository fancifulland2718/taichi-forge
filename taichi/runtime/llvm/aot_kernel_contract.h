#pragma once

#include "taichi/codegen/llvm/cuda_graph_gate.h"
#include "taichi/codegen/llvm/kernel_call_contract.h"
#include "taichi/program/kernel.h"
#include "taichi/runtime/cuda/kernel_launch_abi.h"
#include "taichi/runtime/cuda/persisted_task_launch_policy.h"
#include "taichi/runtime/llvm/aot_snode_relocation.h"
#include "taichi/runtime/llvm/runtime_context_abi.h"

namespace taichi::lang::LLVM {

// Launch-only plans intentionally share compiled code. Project each selected
// entry onto its private AOT contract after cloning the cached payload.
inline void project_aot_task_launch_policy(
    LLVMCompiledKernel &compiled,
    const Kernel::OffloadExecutionPlan &plan,
    Arch arch) {
  TI_ERROR_IF(plan.tasks.size() != compiled.tasks.size(),
              "AOT task launch policy topology differs from compiled code");
  for (std::size_t index = 0; index < compiled.tasks.size(); ++index) {
    const auto &spec = plan.tasks[index];
    const auto &task = compiled.tasks[index];
    TI_ERROR_IF(spec.task_index != index ||
                    spec.task_kind != offloaded_task_type_name(task.task_type),
                "AOT task launch policy topology mismatch at task {}", index);
    TI_ERROR_IF(spec.sparse_list_policy != "saturating",
                "AOT task launch policy cannot persist nonbaseline sparse "
                "list policy at task {}", index);
    TI_ERROR_IF(arch != Arch::cuda &&
                    (spec.grid_residency_waves != 0 ||
                     spec.range_work_per_thread_target != 1),
                "AOT task grid policy requires CUDA");
    auto projected = task;
    projected.requested_grid_residency_waves = spec.grid_residency_waves;
    projected.requested_range_work_per_thread_target =
        spec.range_work_per_thread_target;
    cuda::validate_persisted_task_launch_policy(projected, index);
  }
  // Complete validation before changing even the private contract.
  for (std::size_t index = 0; index < compiled.tasks.size(); ++index) {
    compiled.tasks[index].requested_grid_residency_waves =
        plan.tasks[index].grid_residency_waves;
    compiled.tasks[index].requested_range_work_per_thread_target =
        plan.tasks[index].range_work_per_thread_target;
  }
}

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
