#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "taichi/program/runtime_statistics.h"
#include "taichi/rhi/arch.h"

namespace taichi::lang {

// Diagnostic, tree-scoped sparse runtime inventory.  A measured zero is
// distinct from an unavailable value.  "Tree-owned" counts only allocations
// whose lifetime is exactly the SNodeTree lifetime; logical runtime resources
// may live in a Program-owned reusable arena and are reported separately.
struct SparseSNodeTreeMemoryStatistics {
  RuntimeOptionalCounter tree_owned_reserved_bytes;
  RuntimeOptionalCounter root_reserved_bytes;
  RuntimeOptionalCounter sparse_pool_reserved_bytes;

  RuntimeOptionalCounter runtime_metadata_requested_bytes;
  RuntimeOptionalCounter direct_ambient_requested_bytes;
  RuntimeOptionalCounter allocator_payload_reserved_bytes;
  RuntimeOptionalCounter allocator_payload_used_bytes;
  RuntimeOptionalCounter allocator_bookkeeping_reserved_bytes;
  RuntimeOptionalCounter active_list_reserved_bytes;
  RuntimeOptionalCounter active_list_used_bytes;

  RuntimeOptionalCounter allocator_in_use_elements;
  RuntimeOptionalCounter allocator_free_elements;
  RuntimeOptionalCounter allocator_recycled_elements;

  // Some backends use a Program-wide listgen arena.  It is intentionally not
  // included in tree_owned_reserved_bytes.
  RuntimeOptionalCounter shared_listgen_workspace_reserved_bytes;

  std::string tree_owned_scope{"unavailable"};
  std::string runtime_resource_scope{"unavailable"};
  std::string shared_listgen_workspace_scope{"unavailable"};
};

struct SparseListgenNodeStatistics {
  int snode_id{-1};
  int parent_snode_id{-1};
  std::uint64_t requests{0};
  std::uint64_t rebuilds{0};
  std::uint64_t reuse_hits{0};
  std::uint64_t invalidations{0};
  // GFX-only today: requests whose otherwise-current list was displaced from
  // the single resident listgen workspace by another traversal SNode.
  RuntimeOptionalCounter resident_evictions;
  RuntimeOptionalCounter candidate_slots_dispatched;
  RuntimeOptionalCounter scanned_elements;
  RuntimeOptionalCounter emitted_elements;
  RuntimeOptionalCounter serial_rebuilds;
  RuntimeOptionalCounter parallel_rebuilds;
  std::string last_rebuild_reason{"none"};
};

struct SparseSNodeTreeListgenStatistics {
  bool available{false};
  std::vector<SparseListgenNodeStatistics> nodes;
};

struct SparseSNodeTreeStatistics {
  int tree_id{-1};
  std::uint64_t generation{0};
  std::uint64_t layout_fingerprint{0};
  Arch backend{Arch::x64};
  SparseSNodeTreeMemoryStatistics memory;
  SparseSNodeTreeListgenStatistics listgen;
};

}  // namespace taichi::lang
