#pragma once

#include <cstddef>
#include <cstdint>

namespace taichi::lang {

// Cold mapping from a declared submission value to the compiled argument ABI.
// The first supported scalar contract is u64; no device address or owner is
// stored here. Backends translate the byte offset to their argument storage.
struct SubmissionParameterPatch {
  std::size_t slot{0};
  std::size_t argument_offset{0};
};

using SubmissionParameterValue = std::uint64_t;

}  // namespace taichi::lang
