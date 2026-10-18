#include "taichi/runtime/gfx/snode_tree_resources.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "taichi/codegen/spirv/snode_struct_compiler.h"

namespace taichi::lang::gfx {
namespace {

std::size_t checked_product(std::size_t lhs, std::size_t rhs) {
  if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs) {
    throw std::overflow_error("SNode list capacity overflows host size_t");
  }
  return lhs * rhs;
}

std::size_t checked_offset(std::size_t base, std::size_t offset) {
  if (offset > std::numeric_limits<std::size_t>::max() - base) {
    throw std::overflow_error("SNode resource offset overflows host size_t");
  }
  return base + offset;
}

bool needs_list(const SNode *node) {
  if (!node || node->type == SNodeType::root || node->is_place() ||
      node->is_path_all_dense) {
    return false;
  }
  return node->type == SNodeType::dense || node->need_activation();
}

}  // namespace

SNodeTreeResources describe_snode_tree_resources(
    const spirv::CompiledSNodeStructs &compiled) {
  SNodeTreeResources result;
  result.root_size = compiled.root_size;
  result.snode_ids.reserve(compiled.snode_descriptors.size());
  for (const auto &[id, desc] : compiled.snode_descriptors) {
    result.snode_ids.push_back(id);
    if (needs_list(desc.snode)) {
      std::size_t entries = desc.total_num_cells_from_root;
      std::size_t suffix = 1;
      for (auto *node = desc.snode; node && node->type != SNodeType::root;
           node = node->parent) {
        const auto it = compiled.snode_descriptors.find(node->id);
        if (it != compiled.snode_descriptors.end() &&
            node->type == SNodeType::hash &&
            it->second.hash_table_capacity > 0) {
          entries = checked_product(it->second.hash_table_capacity, suffix);
          break;
        }
        suffix = checked_product(suffix, node->num_cells_per_container);
      }
      result.listgen_capacity_entries =
          std::max(result.listgen_capacity_entries, entries);
    }
    // Preserve the existing root-level counter watches. Nested overflows use
    // the shared device diagnostic buffer instead of host topology walks.
    if (desc.snode && desc.snode->type == SNodeType::hash &&
        desc.hash_table_capacity > 0 && desc.snode->parent &&
        desc.snode->parent->type == SNodeType::root) {
      const auto base = desc.mem_offset_in_parent_cell;
      const auto no_offset = static_cast<std::size_t>(-1);
      result.hash_watches.push_back(
          {id, checked_offset(base, desc.hash_overflow_count_offset),
           checked_offset(base, desc.hash_active_count_offset),
           desc.hash_tombstone_count_offset == no_offset
               ? no_offset
               : checked_offset(base, desc.hash_tombstone_count_offset)});
    }
  }
  for (const auto &[id, contract] : compiled.pointer_contracts) {
    TI_ASSERT_INFO(contract.snode_id == id,
                   "Pointer resource identity does not match SNode {}", id);
    std::size_t size = 0;
    if (contract.pool_buffer_binding_id >= 0) {
      const auto it = compiled.pool_buffer_sizes.find(id);
      TI_ASSERT_INFO(it != compiled.pool_buffer_sizes.end() && it->second > 0,
                     "Pool buffer size missing or zero for pointer SNode {}",
                     id);
      size = it->second;
    }
    result.pointer_pools.push_back({contract, size});
  }
  std::sort(result.snode_ids.begin(), result.snode_ids.end());
  std::sort(result.pointer_pools.begin(), result.pointer_pools.end(),
            [](const auto &a, const auto &b) {
              return a.contract.snode_id < b.contract.snode_id;
            });
  std::sort(
      result.hash_watches.begin(), result.hash_watches.end(),
      [](const auto &a, const auto &b) { return a.snode_id < b.snode_id; });
  return result;
}

}  // namespace taichi::lang::gfx
