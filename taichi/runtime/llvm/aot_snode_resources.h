#pragma once

#include <limits>
#include <unordered_map>
#include <vector>

#include "taichi/ir/offloaded_task_type.h"
#include "taichi/ir/snode_types.h"
#include "taichi/runtime/llvm/llvm_offline_cache.h"

namespace taichi::lang::LLVM {

inline bool is_dense_aot_snode(SNodeType type) {
  return type == SNodeType::root || type == SNodeType::dense ||
         type == SNodeType::place;
}

inline bool is_relocatable_aot_snode(SNodeType type, bool sparse) {
  return is_dense_aot_snode(type) || (sparse && (type == SNodeType::pointer ||
                                                 type == SNodeType::bitmasked ||
                                                 type == SNodeType::dynamic));
}

inline bool is_relocatable_aot_task(OffloadedTaskType type, bool sparse) {
  return type == OffloadedTaskType::serial ||
         type == OffloadedTaskType::range_for ||
         (sparse && (type == OffloadedTaskType::struct_for ||
                     type == OffloadedTaskType::listgen ||
                     type == OffloadedTaskType::gc));
}

inline void validate_relocatable_aot_field(
    const LlvmOfflineCache::FieldCacheData &field,
    bool sparse) {
  TI_ERROR_IF(field.tree_generation == 0 || field.snode_metas.empty() ||
                  field.snode_metas.size() >
                      std::size_t(std::numeric_limits<int>::max()),
              "Invalid relocatable LLVM AOT tree {}", field.tree_id);
  std::vector<int> parents(field.snode_metas.size(), -1);
  std::vector<bool> seen(parents.size(), false);
  std::vector<bool> can_parent(parents.size(), false);
  std::unordered_map<int, int> local_ids;
  int roots = 0;
  for (const auto &node : field.snode_metas) {
    TI_ERROR_IF(!is_relocatable_aot_snode(node.type, sparse) || node.id < 0 ||
                    node.runtime_local_id < 0 ||
                    std::size_t(node.runtime_local_id) >= seen.size() ||
                    seen[node.runtime_local_id] ||
                    !local_ids.emplace(node.id, node.runtime_local_id).second,
                "Invalid relocatable LLVM AOT node in tree {}", field.tree_id);
    seen[node.runtime_local_id] = true;
    can_parent[node.runtime_local_id] = node.type != SNodeType::place;
    if (node.type == SNodeType::root) {
      ++roots;
      TI_ERROR_IF(node.id != field.root_id ||
                      node.runtime_local_id != field.root_runtime_local_id ||
                      node.parent_id != -1,
                  "Invalid relocatable LLVM AOT root in tree {}",
                  field.tree_id);
    }
  }
  TI_ERROR_IF(roots != 1,
              "Missing or duplicate relocatable LLVM AOT root in tree {}",
              field.tree_id);
  for (const auto &node : field.snode_metas) {
    if (node.type == SNodeType::root)
      continue;
    const auto parent = local_ids.find(node.parent_id);
    TI_ERROR_IF(parent == local_ids.end() || !can_parent[parent->second],
                "Missing relocatable LLVM AOT parent in tree {}",
                field.tree_id);
    parents[node.runtime_local_id] = parent->second;
  }
  // Validate the parent closure in linear time, without recursion or an
  // artificial depth limit. Every node must lead to the single declared root.
  std::vector<unsigned char> state(parents.size(), 0);
  for (std::size_t start = 0; start < parents.size(); ++start) {
    int node = static_cast<int>(start);
    while (node >= 0 && state[node] == 0) {
      state[node] = 1;
      node = parents[node];
    }
    TI_ERROR_IF(node >= 0 && state[node] == 1,
                "Cyclic relocatable LLVM AOT parents in tree {}",
                field.tree_id);
    node = static_cast<int>(start);
    while (node >= 0 && state[node] == 1) {
      state[node] = 2;
      node = parents[node];
    }
  }
}

}  // namespace taichi::lang::LLVM
