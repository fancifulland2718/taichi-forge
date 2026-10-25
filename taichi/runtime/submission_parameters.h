#pragma once

#include <cstddef>
#include <cstdint>

namespace taichi::lang {

// Cold mapping from a declared submission value to the compiled argument ABI.
// Values occupy uint64 host slots; only the compiled scalar's low 4/8 bytes
// are copied. No device address or owner is stored here.
struct SubmissionParameterPatch {
  std::size_t slot{0};
  std::size_t argument_offset{0};
  std::size_t bytes{8};
};

using SubmissionParameterValue = std::uint64_t;

}  // namespace taichi::lang
