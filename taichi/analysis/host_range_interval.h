#pragma once

#include "taichi/codegen/range_for_bounds.h"

namespace taichi::lang {

// A compile-time envelope over ALL valid argument values, never a sampled
// shape or the input interval of a guarded host evaluator. Reject possible
// intermediate overflow even if a later clamp would hide it. This analysis
// neither evaluates live arguments nor changes the runtime range program.
struct HostRangeInterval {
  std::int64_t lower, upper;

  bool fits_i32() const {
    return lower >= INT32_MIN && upper <= INT32_MAX && lower <= upper;
  }
};

inline HostRangeInterval multiply_host_range_intervals(HostRangeInterval a,
                                                       HostRangeInterval b) {
  const std::array<std::int64_t, 4> values{
      a.lower * b.lower, a.lower * b.upper,
      a.upper * b.lower, a.upper * b.upper};
  const auto ends = std::minmax_element(values.begin(), values.end());
  return {*ends.first, *ends.second};
}

inline std::optional<HostRangeInterval> static_host_range_interval(
    const HostRangeBound &bound) {
  // The description is compiler-owned, but keep malformed node references
  // fail-closed and do not read any argument bytes while checking its shape.
  if (!bound.valid(std::numeric_limits<std::uint64_t>::max()))
    return std::nullopt;
  if (bound.program.empty()) {
    HostRangeInterval value{bound.factor, bound.factor};
    for (std::size_t i = 0; i < bound.offsets.size(); ++i) {
      value = multiply_host_range_intervals(
          value, {bound.shape_product ? 0 : INT32_MIN, INT32_MAX});
      if (!value.fits_i32())
        return std::nullopt;
    }
    // input_min/max validate host evaluation; they do not constrain what a
    // caller may submit. A fallback input must not invalidate this envelope.
    if (value.lower < bound.input_min || value.upper > bound.input_max)
      return std::nullopt;
    value = multiply_host_range_intervals(value, {bound.scale, bound.scale});
    value.lower += bound.bias;
    value.upper += bound.bias;
    return value.fits_i32() ? std::optional<HostRangeInterval>(value)
                           : std::nullopt;
  }

  std::array<HostRangeInterval, 16> inline_values;
  std::vector<HostRangeInterval> spill;
  auto *values = inline_values.data();
  if (bound.program.size() > inline_values.size()) {
    spill.resize(bound.program.size());
    values = spill.data();
  }
  for (std::size_t i = 0; i < bound.program.size(); ++i) {
    const auto &node = bound.program[i];
    HostRangeInterval result{node.immediate, node.immediate};
    switch (node.op) {
      case HostRangeOp::constant_i32:
        break;
      case HostRangeOp::argument_i32:
        result = {INT32_MIN, INT32_MAX};
        break;
      case HostRangeOp::shape_i32:
        result = {0, INT32_MAX};
        break;
      case HostRangeOp::add:
        result = {values[node.lhs].lower + values[node.rhs].lower,
                  values[node.lhs].upper + values[node.rhs].upper};
        break;
      case HostRangeOp::subtract:
        result = {values[node.lhs].lower - values[node.rhs].upper,
                  values[node.lhs].upper - values[node.rhs].lower};
        break;
      case HostRangeOp::multiply:
        result = multiply_host_range_intervals(values[node.lhs], values[node.rhs]);
        break;
      case HostRangeOp::minimum:
        result = {std::min(values[node.lhs].lower, values[node.rhs].lower),
                  std::min(values[node.lhs].upper, values[node.rhs].upper)};
        break;
      case HostRangeOp::maximum:
        result = {std::max(values[node.lhs].lower, values[node.rhs].lower),
                  std::max(values[node.lhs].upper, values[node.rhs].upper)};
        break;
      case HostRangeOp::divide:
      case HostRangeOp::floor_divide: {
        const auto divide = [&](std::int64_t value) {
          return value / node.immediate -
                 (node.op == HostRangeOp::floor_divide && value % node.immediate < 0);
        };
        result = {divide(values[node.lhs].lower), divide(values[node.lhs].upper)};
        break;
      }
    }
    if (!result.fits_i32())
      return std::nullopt;
    values[i] = result;
  }
  return values[bound.program.size() - 1];
}

}  // namespace taichi::lang
