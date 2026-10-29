#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include "taichi/common/core.h"
#include "taichi/common/serialization.h"
#include "taichi/ir/snode_types.h"

namespace taichi::lang::LLVM {

// Common tree layout consumed by JIT materialization, AOT reconstruction and
// runtime retirement. It owns neither LLVM IR nor allocated root storage.
// Tree/node IDs and generation are supplied by the owning binding; AOT projects
// module-local IDs before runtime use. Keep the serialized field order stable.
struct SNodeTreeLayout {
  struct Node {
    int id{0};
    int parent_id{-1};
    int runtime_local_id{-1};
    SNodeType type = SNodeType::undefined;
    std::size_t cell_size_bytes{0};
    std::size_t chunk_size{0};
    // Compiler capacity hints and per-node dimensions used by runtime pool
    // sizing. These are layout facts, not allocations or live device state.
    std::int64_t vk_max_active_hint{-1};
    std::int64_t hash_expected_active_hint{-1};
    std::int64_t num_cells_per_container{1};
    std::int64_t total_num_cells_from_root{1};

    TI_IO_DEF(id,
              parent_id,
              runtime_local_id,
              type,
              cell_size_bytes,
              chunk_size,
              vk_max_active_hint,
              hash_expected_active_hint,
              num_cells_per_container,
              total_num_cells_from_root);
  };

  int tree_id{0};
  std::uint64_t tree_generation{1};
  int root_id{0};
  int root_runtime_local_id{0};
  std::size_t root_size{0};
  std::vector<Node> snode_metas;

  SNodeTreeLayout normalized() const {
    auto result = *this;
    std::map<int, int> local;
    for (const auto &node : snode_metas)
      local.emplace(node.id, node.runtime_local_id);
    result.tree_id = 0;
    result.tree_generation = 0;
    result.root_id = root_runtime_local_id;
    for (auto &node : result.snode_metas) {
      node.id = node.runtime_local_id;
      if (node.parent_id >= 0) node.parent_id = local.at(node.parent_id);
    }
    return result;
  }

  TI_IO_DEF(tree_id,
            tree_generation,
            root_id,
            root_runtime_local_id,
            root_size,
            snode_metas);
};

}  // namespace taichi::lang::LLVM
