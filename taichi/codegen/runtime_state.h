#pragma once

#include <cstdint>

namespace taichi::lang {

// Compiler facts about internal shared state, separate from application array
// aliases. Zero denotes legacy/unknown code; only an explicit known, empty
// inventory admits independent execution. No instruction scan at launch.
enum RuntimeStateUse : std::uint32_t {
  runtime_state_known = 1u << 0,
  runtime_state_random = 1u << 1,
  runtime_state_error = 1u << 2,
  runtime_state_sparse = 1u << 3,
  runtime_state_external = 1u << 4,
  runtime_state_mesh = 1u << 5,
  runtime_state_print = 1u << 6,
};

inline const char *independent_runtime_state_blocker(std::uint32_t state) {
  if (!(state & runtime_state_known))
    return "kernel has no compiled runtime-state inventory";
  if (state & runtime_state_random)
    return "kernel uses shared random-number state";
  if (state & runtime_state_error)
    return "kernel uses the shared runtime error slot";
  if (state & runtime_state_sparse)
    return "kernel uses shared sparse list or allocator state";
  if (state & runtime_state_external)
    return "kernel calls code with unqualified runtime state";
  if (state & runtime_state_mesh)
    return "kernel uses unqualified mesh runtime state";
  if (state & runtime_state_print)
    return "kernel uses shared device print state";
  if (state != runtime_state_known)
    return "kernel declares unknown runtime-state uses";
  return nullptr;
}

}  // namespace taichi::lang
