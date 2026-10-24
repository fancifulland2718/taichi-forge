#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

#include "taichi/common/serialization.h"

namespace taichi::lang {

// Facts about the final physical task, not permission to change its grid.
// Unknown expressions must never be interpreted as global temporary offset 0.
enum class RangeBoundKind { unknown, constant_i32, global_tmp_i32 };

struct RangeBound {
  RangeBoundKind kind{RangeBoundKind::unknown};
  std::int64_t value{0};

  static RangeBound constant(std::int32_t value) {
    return {RangeBoundKind::constant_i32, value};
  }
  static RangeBound global_tmp(std::int64_t byte_offset) {
    return {RangeBoundKind::global_tmp_i32, byte_offset};
  }
  bool valid() const {
    switch (kind) {
      case RangeBoundKind::unknown:
        return value == 0;
      case RangeBoundKind::constant_i32:
        return value >= std::numeric_limits<std::int32_t>::min() &&
               value <= std::numeric_limits<std::int32_t>::max();
      case RangeBoundKind::global_tmp_i32:
        return value >= 0 && value % sizeof(std::int32_t) == 0;
    }
    return false;
  }
  // A consumer must also check its concrete instance's workspace allocation.
  bool fits_global_tmps(std::uint64_t bytes) const {
    return valid() && kind == RangeBoundKind::global_tmp_i32 &&
           bytes >= sizeof(std::int32_t) &&
           static_cast<std::uint64_t>(value) <= bytes - sizeof(std::int32_t);
  }

  TI_IO_DEF(kind, value);
};

struct RangeForBounds {
  RangeBound begin;
  RangeBound end;

  bool valid() const {
    return begin.valid() && end.valid();
  }
  bool const_range() const {
    return valid() && begin.kind == RangeBoundKind::constant_i32 &&
           end.kind == RangeBoundKind::constant_i32;
  }
  std::optional<std::int64_t> constant_size() const {
    if (!const_range()) {
      return std::nullopt;
    }
    return std::max<std::int64_t>(end.value - begin.value, 0);
  }

  TI_IO_DEF(begin, end);
};

}  // namespace taichi::lang
