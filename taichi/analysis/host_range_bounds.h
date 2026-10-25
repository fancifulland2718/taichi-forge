#pragma once

#include "taichi/codegen/range_for_bounds.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/type_factory.h"

namespace taichi::lang {

// Compose k * expression + c at compile time, retaining the input interval
// where ALL original intermediate expressions fit i32. Cancellation must not
// turn an overflowing source expression into an apparently safe launch bound.
inline bool compose_host_range_affine(HostRangeBound &bound,
                                      std::int64_t k, std::int64_t c) {
  const auto scale = k * bound.scale;
  const auto bias = k * bound.bias + c;
  if (scale < INT32_MIN || scale > INT32_MAX ||
      bias < INT32_MIN || bias > INT32_MAX)
    return false;
  auto lower = std::int64_t(bound.input_min);
  auto upper = std::int64_t(bound.input_max);
  const auto floor_div = [](std::int64_t n, std::int64_t d) {
    return n / d - (n % d < 0);
  };
  const auto ceil_div = [](std::int64_t n, std::int64_t d) {
    return n / d + (n % d > 0);
  };
  if (scale > 0) {
    lower = std::max(lower, ceil_div(std::int64_t(INT32_MIN) - bias, scale));
    upper = std::min(upper, floor_div(std::int64_t(INT32_MAX) - bias, scale));
  } else if (scale < 0) {
    lower = std::max(lower, ceil_div(bias - INT32_MAX, -scale));
    upper = std::min(upper, floor_div(bias - INT32_MIN, -scale));
  }
  if (lower > upper)
    return false;
  bound.scale = static_cast<std::int32_t>(scale);
  bound.bias = static_cast<std::int32_t>(bias);
  bound.input_min = static_cast<std::int32_t>(lower);
  bound.input_max = static_cast<std::int32_t>(upper);
  return true;
}

// Capture provenance before cross-offload promotion turns argument expressions
// into global-temporary loads. Argument layouts have already been finalized.
inline std::optional<HostRangeBound> describe_host_range_bound(
    Stmt *stmt, const StructType *arguments) {
  if (!stmt || stmt->ret_type != PrimitiveType::i32 || !arguments)
    return std::nullopt;
  HostRangeBound result;
  if (auto *constant = stmt->cast<ConstStmt>()) {
    result.factor = constant->val.val_int32();
  } else if (auto *shape = stmt->cast<ExternalTensorShapeAlongAxisStmt>()) {
    // Nested argpacks have device-owned storage. Only ordinary top-level
    // ndarray metadata is part of this immutable host argument image.
    if (shape->arg_id.size() != 1)
      return std::nullopt;
    auto path = shape->arg_id;
    path.push_back(TypeFactory::SHAPE_POS_IN_NDARRAY);
    path.push_back(shape->axis);
    result.offsets.push_back(arguments->get_element_offset(path));
    result.shape_product = true;
  } else if (auto *arg = stmt->cast<ArgLoadStmt>()) {
    if (arg->is_ptr || !arg->create_load || arg->arg_depth != 0 ||
        arg->arg_id.size() != 1)
      return std::nullopt;
    result.offsets.push_back(arguments->get_element_offset(arg->arg_id));
  } else if (auto *binary = stmt->cast<BinaryOpStmt>()) {
    if (binary->is_bit_vectorized)
      return std::nullopt;
    auto *constant = binary->rhs->cast<ConstStmt>();
    auto *variable = binary->lhs;
    bool constant_left = false;
    if (!constant) {
      constant = binary->lhs->cast<ConstStmt>();
      variable = binary->rhs;
      constant_left = true;
    }
    if (constant && constant->ret_type == PrimitiveType::i32) {
      auto input = describe_host_range_bound(variable, arguments);
      if (!input)
        return std::nullopt;
      const std::int64_t c = constant->val.val_int32();
      bool accepted = false;
      switch (binary->op_type) {
        case BinaryOpType::add:
          accepted = compose_host_range_affine(*input, 1, c);
          break;
        case BinaryOpType::sub:
          accepted = compose_host_range_affine(*input, constant_left ? -1 : 1,
                                               constant_left ? c : -c);
          break;
        case BinaryOpType::mul:
          if (c >= 0 && input->shape_product && input->scale == 1 &&
              input->bias == 0 && input->input_min == INT32_MIN &&
              input->input_max == INT32_MAX &&
              c * input->factor <= INT32_MAX) {
            input->factor = static_cast<std::int32_t>(c * input->factor);
            accepted = true;
          } else {
            accepted = compose_host_range_affine(*input, c, 0);
          }
          break;
        default:
          break;
      }
      return accepted ? input : std::nullopt;
    }
    if (binary->op_type != BinaryOpType::mul)
      return std::nullopt;
    auto left = describe_host_range_bound(binary->lhs, arguments);
    auto right = describe_host_range_bound(binary->rhs, arguments);
    if (!left || !right)
      return std::nullopt;
    // Keep arbitrary scalar products out of this contract. Flatten only the
    // existing nonnegative shape product; affine expressions are not reordered.
    for (const auto *part : {&*left, &*right})
      if ((!part->shape_product && !part->offsets.empty()) || part->factor < 0 ||
          part->scale != 1 || part->bias != 0 || part->input_min != INT32_MIN ||
          part->input_max != INT32_MAX)
        return std::nullopt;
    const auto factor = std::int64_t(left->factor) * right->factor;
    if (factor > std::numeric_limits<std::int32_t>::max())
      return std::nullopt;
    result.factor = static_cast<std::int32_t>(factor);
    result.offsets = std::move(left->offsets);
    result.offsets.insert(result.offsets.end(), right->offsets.begin(),
                           right->offsets.end());
    result.shape_product = true;
  } else {
    return std::nullopt;
  }
  return result;
}

inline std::optional<HostRangeBounds> describe_host_range_bounds(
    Stmt *begin, Stmt *end, const StructType *arguments) {
  auto first = describe_host_range_bound(begin, arguments);
  auto last = describe_host_range_bound(end, arguments);
  if (!first || !last || (first->offsets.empty() && last->offsets.empty()))
    return std::nullopt;  // Constant ranges already have codegen grid selection.
  return HostRangeBounds{std::move(*first), std::move(*last)};
}

}  // namespace taichi::lang
