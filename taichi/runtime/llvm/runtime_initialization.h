#pragma once

#include <cstdint>

namespace taichi::lang {

// Returned once in the existing runtime result buffer. These are borrowed
// addresses owned by this runtime, never persisted in a compiled artifact.
// Publishing both in one response avoids a later device query or extra sync
// when an ordinary recording needs the same temporaries as its payload.
struct LlvmRuntimeInitialization {
  std::uint64_t runtime;
  std::uint64_t temporaries;
};
static_assert(sizeof(LlvmRuntimeInitialization) == 2 * sizeof(std::uint64_t));

}  // namespace taichi::lang
