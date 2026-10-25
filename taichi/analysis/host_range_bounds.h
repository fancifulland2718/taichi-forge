#pragma once

#include "taichi/codegen/range_for_bounds.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/type_factory.h"

namespace taichi::lang {

// Capture provenance before cross-offload promotion turns argument expressions
// into global-temporary loads. Argument layouts have already been finalized.
inline std::optional<HostRangeBound> describe_host_range_bound(
    Stmt *stmt, const StructType *arguments, bool product = false) {
  if (!stmt || stmt->ret_type != PrimitiveType::i32 || !arguments)
    return std::nullopt;
  HostRangeBound result;
  if (auto *constant = stmt->cast<ConstStmt>()) {
    result.factor = constant->val.val_int32();
    if (product && result.factor < 0)
      return std::nullopt;
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
    if (product || arg->is_ptr || !arg->create_load || arg->arg_depth != 0 ||
        arg->arg_id.size() != 1)
      return std::nullopt;
    result.offsets.push_back(arguments->get_element_offset(arg->arg_id));
  } else if (auto *binary = stmt->cast<BinaryOpStmt>();
             binary && binary->op_type == BinaryOpType::mul) {
    auto left = describe_host_range_bound(binary->lhs, arguments, true);
    auto right = describe_host_range_bound(binary->rhs, arguments, true);
    if (!left || !right)
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
