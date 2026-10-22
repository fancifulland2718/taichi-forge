#pragma once

#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "taichi/ir/offloaded_task_type.h"
#include "taichi/ir/snode_types.h"
#include "taichi/ir/snode_hash_utils.h"
#include "taichi/runtime/llvm/llvm_offline_cache.h"

namespace taichi::lang::LLVM {

inline bool is_dense_aot_snode(SNodeType type) {
  return type == SNodeType::root || type == SNodeType::dense ||
         type == SNodeType::place;
}

inline bool is_relocatable_aot_snode(SNodeType type, bool sparse) {
  return is_dense_aot_snode(type) || (sparse && (type == SNodeType::pointer ||
                                                 type == SNodeType::bitmasked ||
                                                 type == SNodeType::dynamic ||
                                                 type == SNodeType::hash ||
                                                 type == SNodeType::bit_struct ||
                                                 type == SNodeType::quant_array));
}

inline bool is_relocatable_aot_task(OffloadedTaskType type, bool lists) {
  return type == OffloadedTaskType::serial ||
         type == OffloadedTaskType::range_for ||
         // Mesh lowering turns mappings/relations into ordinary SNode loads.
         // Patch counts are constants; no live Mesh owner survives codegen.
         type == OffloadedTaskType::mesh_for ||
         (lists && (type == OffloadedTaskType::struct_for ||
                    type == OffloadedTaskType::listgen ||
                    type == OffloadedTaskType::gc));
}

// Cold metadata only: no LLVM decoding, kernel registration or root allocation.
// A list task identifies one tree even when its kernel reads other dense roots.
// Older artifacts can lack task identities; conservatively cover their declared
// dependencies without weakening the exact contract of new list_v1 exports.
inline std::unordered_set<int> aot_element_list_tree_ids(
    const LlvmOfflineCache &cache,
    bool exact) {
  std::unordered_map<int, int> node_trees;
  std::unordered_set<int> result;
  for (const auto &[id, field] : cache.fields) {
    TI_ERROR_IF(field.tree_id != id, "Inconsistent LLVM AOT tree {}", id);
    for (const auto &node : field.snode_metas) {
      TI_ERROR_IF(!node_trees.emplace(node.id, id).second,
                  "Duplicate LLVM AOT SNode id {}", node.id);
      if (!is_dense_aot_snode(node.type))
        result.insert(id);
    }
  }
  for (const auto &[name, kernel] : cache.kernels) {
    std::unordered_set<int> dependencies;
    for (int id : kernel.used_snode_tree_ids) {
      TI_ERROR_IF(!cache.fields.count(id),
                  "LLVM AOT kernel {} is missing SNodeTree {}", name, id);
      dependencies.insert(id);
    }
    for (const auto &task : kernel.compiled_data.tasks) {
      if (task.task_type != OffloadedTaskType::listgen &&
          task.task_type != OffloadedTaskType::struct_for &&
          task.sparse_list_op == OffloadedTask::kSparseListOpNone)
        continue;
      if (task.sparse_list_snode_id < 0) {
        TI_ERROR_IF(exact || dependencies.empty(),
                    "LLVM AOT list task {} has no tree identity", task.name);
        result.insert(dependencies.begin(), dependencies.end());
        continue;
      }
      const auto owner = node_trees.find(task.sparse_list_snode_id);
      TI_ERROR_IF(owner == node_trees.end() ||
                      !dependencies.count(owner->second),
                  "LLVM AOT list task {} has an undeclared tree", task.name);
      result.insert(owner->second);
    }
  }
  return result;
}

inline void validate_relocatable_aot_field(const LLVM::SNodeTreeLayout &field,
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
    if (node.type == SNodeType::hash) {
      validate_hash_snode_capacity(node.vk_max_active_hint > 0
                                      ? node.vk_max_active_hint
                                      : node.num_cells_per_container);
    }
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
