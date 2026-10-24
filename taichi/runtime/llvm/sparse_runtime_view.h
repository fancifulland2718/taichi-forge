#pragma once

#include <cstdint>

// Cold device-side transaction input. The executor must hold
// generation-qualified tree loans and order these calls after all previous
// users and before replay. Entries are sorted by tree_id. Neither the view nor
// its directory owns trees, pools, temporaries, or the source runtime; those
// outlive successful release. A failed release leaves every manager attached:
// callers must retain the view and its dependencies until retry or owner-wide
// retirement, never free it.
struct LlvmSparseRuntimeViewTree {
  std::int32_t tree_id;
  std::int32_t reserved{0};
  std::uint64_t generation;
};

enum class LlvmSparseRuntimeViewError : std::int32_t {
  success = 0,
  invalid_view,
  invalid_request,
  unavailable_tree,
  stale_generation,
  foreign_manager,
};

struct LlvmSparseRuntimeViewStatus {
  LlvmSparseRuntimeViewError error{LlvmSparseRuntimeViewError::success};
  std::int32_t tree_id{-1};
};
