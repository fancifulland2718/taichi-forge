#include "taichi/runtime/cuda/runtime_module.h"

#include <unordered_set>

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "taichi/runtime/llvm/llvm_context.h"

namespace taichi::lang {

const char *cuda_runtime_group_name(CUDARuntimeGroup group) {
  switch (group) {
    case CUDARuntimeGroup::core:
      return "core";
    case CUDARuntimeGroup::sparse:
      return "sparse";
    case CUDARuntimeGroup::statistics:
      return "statistics";
  }
  TI_NOT_IMPLEMENTED;
}

CUDARuntimeGroup cuda_runtime_group(const std::string &entry) {
  // Explicit membership preserves eager preparation for future unclassified
  // entries. These functions are called at tree materialization boundaries.
  static const std::unordered_set<std::string> sparse{
      "runtime_NodeAllocator_initialize_ex",
      "runtime_NodeAllocator_set_dedicated_pool",
      "runtime_NodeAllocator_set_deterministic_capacity",
      "runtime_allocate_ambient_direct",
      "runtime_initialize_snode_element_list",
      "runtime_element_list_set_backing_pool",
      "runtime_element_lists_prepare_backing_pool",
      "runtime_element_lists_finalize_backing_pool"};
  static const std::unordered_set<std::string> statistics{
      "runtime_get_snode_element_list",
      "runtime_get_snode_node_allocator",
      "runtime_hash_probe_stats_get",
      "runtime_hash_probe_stats_reset",
      "runtime_ListManager_get_element_size",
      "runtime_ListManager_get_max_num_elements_per_chunk",
      "runtime_ListManager_get_num_active_chunks",
      "runtime_ListManager_get_num_elements",
      "runtime_LLVMRuntime_get_total_requested_memory",
      "runtime_NodeManager_get_data_list",
      "runtime_NodeManager_get_deterministic_capacity",
      "runtime_NodeManager_get_deterministic_peak",
      "runtime_NodeManager_get_free_list",
      "runtime_NodeManager_get_free_list_used",
      "runtime_NodeManager_get_recycled_list",
      "runtime_sparse_snode_statistics_collect",
      "runtime_sparse_tree_statistics_reset"};
  if (sparse.count(entry))
    return CUDARuntimeGroup::sparse;
  if (statistics.count(entry))
    return CUDARuntimeGroup::statistics;
  return CUDARuntimeGroup::core;
}

bool cuda_runtime_can_partition(const llvm::Module &module) {
  if (!module.getModuleInlineAsm().empty() || !module.alias_empty() ||
      !module.ifunc_empty())
    return false;
  for (const auto &global : module.globals()) {
    if (!global.isConstant() || !global.hasGlobalUnnamedAddr())
      return false;
  }
  for (const auto &function : module) {
    if (function.hasAddressTaken())
      return false;
  }
  return true;
}

void retain_cuda_runtime_group(llvm::Module &module, CUDARuntimeGroup group) {
  std::unordered_set<std::string> roots;
  auto *annotations = module.getNamedMetadata("nvvm.annotations");
  if (annotations) {
    std::vector<llvm::MDNode *> retained;
    for (auto *node : annotations->operands()) {
      const auto *value =
          node->getNumOperands() > 0
              ? llvm::dyn_cast_if_present<llvm::ValueAsMetadata>(
                    node->getOperand(0).get())
              : nullptr;
      const auto *function =
          value ? llvm::dyn_cast<llvm::Function>(value->getValue()) : nullptr;
      if (function && cuda_runtime_group(function->getName().str()) != group)
        continue;
      retained.push_back(node);
      const auto *key = node->getNumOperands() > 1
                            ? llvm::dyn_cast_if_present<llvm::MDString>(
                                  node->getOperand(1).get())
                            : nullptr;
      if (function && key && key->getString() == "kernel")
        roots.insert(function->getName().str());
    }
    annotations->clearOperands();
    for (auto *node : retained)
      annotations->addOperand(node);
  }
  TaichiLLVMContext::eliminate_unused_functions(
      &module, [&](const std::string &name) { return roots.count(name) != 0; });
}

}  // namespace taichi::lang
