#pragma once

#include <algorithm>

#include "taichi/ir/snode.h"

namespace taichi::lang {

// The LLVM listgen helpers have two different mappings. Root listgen assigns
// one output slice to each thread; nonroot listgen assigns one parent-list
// entry (a slice of a parent container) to each block. Use declared capacities,
// never the current active set or allocator sizing hints. Zero means unproven.
inline int llvm_listgen_grid_bound(const SNode *child,
                                  int block_dim,
                                  int saturating_grid) {
  if (!child || !child->parent || block_dim <= 0 || saturating_grid <= 0) {
    return 0;
  }
  const SNode *root = child;
  for (auto *node = child; node->type != SNodeType::root;
       node = node->parent) {
    // Hash has dedicated bucket/active-slot enumeration. Its logical shape
    // does not describe the ordinary sliced-container list used below.
    if ((node->type != SNodeType::dense &&
         node->type != SNodeType::bitmasked &&
         node->type != SNodeType::pointer &&
         node->type != SNodeType::dynamic) ||
        node->num_cells_per_container <= 0 || !node->parent) {
      return 0;
    }
    root = node->parent;
  }
  const auto slices = [](int64 cells) {
    return (cells - 1) / taichi_listgen_max_element_size + 1;
  };
  const auto *parent = child->parent;
  if (parent == root) {
    const int64 groups = (slices(child->num_cells_per_container) - 1) /
                            block_dim +
                        1;
    return static_cast<int>(std::min<int64>(groups, saturating_grid));
  }
  int64 entries = slices(parent->num_cells_per_container);
  for (auto *node = parent->parent; node != root; node = node->parent) {
    const int64 cells = node->num_cells_per_container;
    // Saturate before multiplying, including layouts whose theoretical
    // capacity does not fit in int64. This limits launch width, not work.
    if (entries >= saturating_grid ||
        cells > saturating_grid / entries) {
      return saturating_grid;
    }
    entries *= cells;
  }
  return static_cast<int>(std::min<int64>(entries, saturating_grid));
}

}  // namespace taichi::lang
