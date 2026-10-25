#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

#include "taichi/common/serialization.h"

namespace taichi::lang {

enum class HostRangeOp {
  constant_i32,
  argument_i32,
  shape_i32,
  add,
  subtract,
  multiply,
  minimum,
  maximum,
  divide,
  floor_divide,
};

// A topologically ordered DAG. Leaves carry an argument byte offset or an i32
// constant; arithmetic carries prior node indices. Division has a positive
// compile-time divisor. No live values, pointers, or runtime specialization.
struct HostRangeInstruction {
  HostRangeOp op{HostRangeOp::constant_i32};
  std::int32_t immediate{0};
  std::uint64_t lhs{0}, rhs{0}, offset{0};

  bool is_input() const {
    return op == HostRangeOp::argument_i32 || op == HostRangeOp::shape_i32;
  }

  bool operator==(const HostRangeInstruction &other) const {
    return op == other.op && immediate == other.immediate && lhs == other.lhs &&
           rhs == other.rhs && offset == other.offset;
  }

  bool valid(std::uint64_t index, std::uint64_t argument_bytes) const {
    switch (op) {
      case HostRangeOp::constant_i32:
        return lhs == 0 && rhs == 0 && offset == 0;
      case HostRangeOp::argument_i32:
      case HostRangeOp::shape_i32:
        return lhs == 0 && rhs == 0 && immediate == 0 &&
               argument_bytes >= sizeof(std::int32_t) &&
               offset <= argument_bytes - sizeof(std::int32_t) &&
               offset % alignof(std::int32_t) == 0;
      case HostRangeOp::add:
      case HostRangeOp::subtract:
      case HostRangeOp::multiply:
      case HostRangeOp::minimum:
      case HostRangeOp::maximum:
        return lhs < index && rhs < index && immediate == 0 && offset == 0;
      case HostRangeOp::divide:
      case HostRangeOp::floor_divide:
        return lhs < index && rhs == 0 && immediate > 0 && offset == 0;
    }
    return false;
  }

  TI_IO_DEF(op, immediate, lhs, rhs, offset);
};

inline bool valid_host_range_program(
    const std::vector<HostRangeInstruction> &program,
    std::uint64_t argument_bytes) {
  for (std::size_t i = 0; i < program.size(); ++i)
    if (!program[i].valid(i, argument_bytes))
      return false;
  return true;
}

// The inline storage is an allocation optimization, not an expression limit.
// Larger DAGs have a dynamically sized spill; neither workload nor expression
// depth is capped. Common affine bounds do not enter this path at all.
inline std::optional<std::int32_t> evaluate_host_range_program(
    const std::vector<HostRangeInstruction> &program, const void *arguments,
    std::uint64_t argument_bytes) {
  if (program.empty() || !valid_host_range_program(program, argument_bytes))
    return std::nullopt;
  std::array<std::int32_t, 16> inline_values;
  std::vector<std::int32_t> spill;
  auto *values = inline_values.data();
  if (program.size() > inline_values.size()) {
    spill.resize(program.size());
    values = spill.data();
  }
  for (std::size_t i = 0; i < program.size(); ++i) {
    const auto &node = program[i];
    std::int64_t value = 0;
    switch (node.op) {
      case HostRangeOp::constant_i32:
        value = node.immediate;
        break;
      case HostRangeOp::argument_i32:
      case HostRangeOp::shape_i32: {
        if (!arguments)
          return std::nullopt;
        std::int32_t input;
        std::memcpy(&input, static_cast<const char *>(arguments) + node.offset,
                    sizeof(input));
        if (node.op == HostRangeOp::shape_i32 && input < 0)
          return std::nullopt;
        value = input;
        break;
      }
      case HostRangeOp::add:
        value = std::int64_t(values[node.lhs]) + values[node.rhs];
        break;
      case HostRangeOp::subtract:
        value = std::int64_t(values[node.lhs]) - values[node.rhs];
        break;
      case HostRangeOp::multiply:
        value = std::int64_t(values[node.lhs]) * values[node.rhs];
        break;
      case HostRangeOp::minimum:
        value = std::min(values[node.lhs], values[node.rhs]);
        break;
      case HostRangeOp::maximum:
        value = std::max(values[node.lhs], values[node.rhs]);
        break;
      case HostRangeOp::divide:
        value = values[node.lhs] / node.immediate;
        break;
      case HostRangeOp::floor_divide:
        value = values[node.lhs] / node.immediate -
                (values[node.lhs] % node.immediate < 0);
        break;
    }
    // Check every original arithmetic node, before a later min/max can hide
    // overflow. On failure the caller keeps the backend's original grid.
    if (value < INT32_MIN || value > INT32_MAX)
      return std::nullopt;
    values[i] = static_cast<std::int32_t>(value);
  }
  return values[program.size() - 1];
}

}  // namespace taichi::lang
