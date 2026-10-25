#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "taichi/common/serialization.h"

namespace taichi::lang {

// Facts about the final physical task, not permission to change its grid.
// Unknown expressions must never be interpreted as global temporary offset 0.
enum class RangeBoundKind { unknown, constant_i32, global_tmp_i32 };

// Permission for one transformation only: remove whole groups beyond the
// logical range, keeping the block size and range mapping unchanged. This is
// not permission to coarsen productive groups, change TLS, or specialize code.
enum class EmptyRangeGroupProof {
  unknown,
  independent_iterations,
  fixed_mapping,
  auxiliary_region,
  unproven_call,
  physical_thread,
  local_storage,
  sparse_topology,
  compiler_tls_reduction,
};

inline const char *empty_range_group_proof_name(EmptyRangeGroupProof proof) {
  switch (proof) {
    case EmptyRangeGroupProof::unknown:
      return "unknown";
    case EmptyRangeGroupProof::independent_iterations:
      return "independent_iterations";
    case EmptyRangeGroupProof::fixed_mapping:
      return "fixed_mapping";
    case EmptyRangeGroupProof::auxiliary_region:
      return "auxiliary_region";
    case EmptyRangeGroupProof::unproven_call:
      return "unproven_call";
    case EmptyRangeGroupProof::physical_thread:
      return "physical_thread";
    case EmptyRangeGroupProof::local_storage:
      return "local_storage";
    case EmptyRangeGroupProof::sparse_topology:
      return "sparse_topology";
    case EmptyRangeGroupProof::compiler_tls_reduction:
      return "compiler_tls_reduction";
  }
  return nullptr;
}

// A small, proven expression over the immutable launch-argument image. It is
// either a constant, one i32 scalar, or a product of ndarray dimensions and a
// nonnegative constant, followed by a normalized affine transform. The input
// interval preserves every intermediate i32 overflow check without replaying
// the source expression at launch. No device reads or expression bytecode.
struct HostRangeBound {
  std::int32_t factor{1};
  std::vector<std::uint64_t> offsets;
  bool shape_product{false};
  std::int32_t scale{1}, bias{0};
  std::int32_t input_min{INT32_MIN}, input_max{INT32_MAX};

  bool reads(std::uint64_t offset, std::uint64_t bytes) const {
    // Difference-based overlap avoids overflow for untrusted byte ranges.
    for (auto input : offsets) {
      if (bytes && (input >= offset ? input - offset < bytes
                                   : offset - input < sizeof(std::int32_t)))
        return true;
    }
    return false;
  }

  bool operator==(const HostRangeBound &other) const {
    return factor == other.factor && offsets == other.offsets &&
           shape_product == other.shape_product && scale == other.scale &&
           bias == other.bias && input_min == other.input_min &&
           input_max == other.input_max;
  }

  bool valid(std::uint64_t argument_bytes) const {
    if ((!shape_product && !offsets.empty() &&
         (factor != 1 || offsets.size() != 1)) ||
        (shape_product && factor < 0) || input_min > input_max)
      return false;
    for (auto endpoint : {input_min, input_max}) {
      const auto value = std::int64_t(scale) * endpoint + bias;
      if (value < INT32_MIN || value > INT32_MAX)
        return false;
    }
    for (auto offset : offsets)
      if (argument_bytes < sizeof(std::int32_t) ||
          offset > argument_bytes - sizeof(std::int32_t) ||
          offset % alignof(std::int32_t))
        return false;
    return true;
  }

  std::optional<std::int32_t> evaluate(const void *arguments,
                                      std::uint64_t argument_bytes) const {
    if (!valid(argument_bytes) || (!arguments && !offsets.empty()))
      return std::nullopt;
    std::int64_t value = factor;
    for (auto offset : offsets) {
      std::int32_t operand;
      std::memcpy(&operand, static_cast<const char *>(arguments) + offset,
                  sizeof(operand));
      if (shape_product && operand < 0)
        return std::nullopt;
      value *= operand;
      // Preserve the backend's original path for overflow instead of making
      // a different host-side interpretation of the i32 range expression.
      if (value < std::numeric_limits<std::int32_t>::min() ||
          value > std::numeric_limits<std::int32_t>::max())
        return std::nullopt;
    }
    if (value < input_min || value > input_max)
      return std::nullopt;
    return static_cast<std::int32_t>(value * scale + bias);
  }

  TI_IO_DEF(factor, offsets, shape_product, scale, bias, input_min, input_max);
};

struct HostRangeBounds {
  HostRangeBound begin, end;

  bool operator==(const HostRangeBounds &other) const {
    return begin == other.begin && end == other.end;
  }

  bool valid(std::uint64_t argument_bytes) const {
    return begin.valid(argument_bytes) && end.valid(argument_bytes);
  }

  int grid(int baseline, int block, const void *arguments,
           std::uint64_t argument_bytes) const {
    if (baseline <= 0 || block <= 0)
      return baseline;
    const auto first = begin.evaluate(arguments, argument_bytes);
    const auto last = end.evaluate(arguments, argument_bytes);
    if (!first || !last)
      return baseline;
    const auto count = std::max<std::int64_t>(0, std::int64_t(*last) - *first);
    // Keep one block for empty ranges: existing kernel/Graph contracts remain
    // intact. Grid-stride execution still covers arbitrarily larger workloads.
    return static_cast<int>(std::min<std::int64_t>(
        baseline, std::max<std::int64_t>(1, (count + block - 1) / block)));
  }

  TI_IO_DEF(begin, end);
};

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
  std::optional<HostRangeBounds> host_launch;
  EmptyRangeGroupProof empty_group_elision{EmptyRangeGroupProof::unknown};

  bool can_elide_empty_groups() const {
    return empty_group_elision == EmptyRangeGroupProof::independent_iterations ||
           empty_group_elision == EmptyRangeGroupProof::compiler_tls_reduction;
  }
  bool supports_device_dispatch() const {
    // The current persisted device controller contract excludes TLS. Host
    // trimming can be qualified independently without widening that contract.
    return empty_group_elision == EmptyRangeGroupProof::independent_iterations &&
           valid() && !const_range() &&
           begin.kind != RangeBoundKind::unknown &&
           end.kind != RangeBoundKind::unknown;
  }
  bool uses_host_launch_grid() const {
    return host_launch.has_value() && can_elide_empty_groups();
  }

  bool valid() const {
    return begin.valid() && end.valid() &&
           empty_range_group_proof_name(empty_group_elision) != nullptr;
  }
  bool valid(std::uint64_t argument_bytes) const {
    return valid() && (!host_launch || host_launch->valid(argument_bytes));
  }
  int launch_grid(int baseline, int block, const void *arguments,
                  std::uint64_t argument_bytes,
                  const std::vector<std::pair<std::uint64_t, std::uint64_t>>
                      *mutable_arguments = nullptr) const {
    if (!uses_host_launch_grid())
      return baseline;
    // A prepared graph cannot resize its launch at each parameter upload. Do
    // not size it from the initial value of an explicitly mutable bound. Other
    // mutable arguments do not disable shape-based selection for this task.
    if (mutable_arguments) {
      for (const auto &[offset, bytes] : *mutable_arguments) {
        if (host_launch->begin.reads(offset, bytes) ||
            host_launch->end.reads(offset, bytes))
          return baseline;
      }
    }
    return host_launch->grid(baseline, block, arguments, argument_bytes);
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

  TI_IO_DEF(begin, end, host_launch, empty_group_elision);
};

}  // namespace taichi::lang
