#pragma once

#include <cstdint>
#include "taichi/inc/constants.h"

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

// An immutable code-slot -> instance-state binding. The host executor verifies
// common construction provenance and both generations before lending storage.
// The embedded runtime validates the state generation before publishing slots.
struct LlvmSparseRuntimeViewBinding {
  std::int32_t tree_id;
  std::int32_t source_tree_id;
  std::uint64_t generation;
  std::uint64_t code_generation;
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

// One immutable submission result, copied after its kernels on the same
// stream. Device layout has no pointers and is identical on the host.
// Message and arguments are only meaningful for a failing submission. CUDA
// success snapshots may retain a preceding payload; the CPU observer preserves
// the error code retrieved and cleared before requesting the snapshot.
struct LlvmSparseRuntimeDiagnostic {
  std::int64_t code;
  char message[taichi_error_message_max_length];
  std::uint64_t arguments[taichi_error_message_max_num_arguments];
};
