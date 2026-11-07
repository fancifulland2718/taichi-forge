#pragma once

#include "taichi/codegen/range_for_bounds.h"

namespace taichi::lang {

// One allocation in an offloaded task. Extents are proven expressions over
// the same immutable argument image used by its device code. They are not
// producer values or a list of supported model sizes.
struct DynamicSharedArray {
  int allocation_id{0};
  int element_bytes{0};
  std::vector<HostRangeBound> extents;

  bool operator==(const DynamicSharedArray &other) const {
    return allocation_id == other.allocation_id &&
           element_bytes == other.element_bytes && extents == other.extents;
  }

  TI_IO_DEF(allocation_id, element_bytes, extents);
};

struct DynamicSharedLayout {
  std::vector<std::uint32_t> offsets;
  std::uint32_t bytes{0};
};

// Alignment matches the native shared allocations. The supplied limit is a
// backend/device admission bound, never a finite model-size specialization.
inline std::optional<DynamicSharedLayout> evaluate_dynamic_shared_layout(
    const std::vector<DynamicSharedArray> &arrays, const void *arguments,
    std::uint64_t argument_bytes, std::uint32_t limit,
    const std::vector<std::pair<std::uint64_t, std::uint64_t>>
        *mutable_arguments = nullptr) {
  DynamicSharedLayout result;
  int previous_id = -1;
  for (const auto &array : arrays) {
    if (array.allocation_id <= previous_id || array.element_bytes <= 0 ||
        array.element_bytes > 8 ||
        (array.element_bytes & (array.element_bytes - 1)) ||
        array.extents.empty())
      return std::nullopt;
    previous_id = array.allocation_id;
    std::uint64_t bytes = array.element_bytes;
    for (const auto &extent : array.extents) {
      if (!extent.valid(argument_bytes))
        return std::nullopt;
      if (mutable_arguments) {
        for (const auto &[offset, size] : *mutable_arguments)
          if (extent.reads(offset, size))
            return std::nullopt;
      }
      auto size = extent.evaluate(arguments, argument_bytes);
      if (!size || *size <= 0 || bytes > limit / std::uint64_t(*size))
        return std::nullopt;
      bytes *= *size;
    }
    const auto offset = (std::uint64_t(result.bytes) + 7) & ~std::uint64_t(7);
    if (offset > limit || bytes > limit - offset)
      return std::nullopt;
    result.offsets.push_back(static_cast<std::uint32_t>(offset));
    result.bytes = static_cast<std::uint32_t>(offset + bytes);
  }
  return result;
}

}  // namespace taichi::lang
