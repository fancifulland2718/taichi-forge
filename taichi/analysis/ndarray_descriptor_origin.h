#pragma once

#include "taichi/ir/statements.h"
#include "taichi/program/callable.h"

namespace taichi::lang::irpass::analysis {

// A real-function call expands each array into its shape, strides, affine
// flag and data/gradient addresses. Recover a single descriptor only when all
// of those fields provably belong to the same source argument.
template <typename Leaf>
bool forwarded_ndarray(const CallableBase::Parameter &parameter, Leaf leaf,
                       ExternalTensorBasePtrStmt *&pointer,
                       ExternalTensorBasePtrStmt *&gradient) {
  const int ndim = parameter.total_dim - parameter.element_shape.size();
  Stmt *data = leaf({TypeFactory::DATA_PTR_POS_IN_NDARRAY});
  pointer = data ? data->cast<ExternalTensorBasePtrStmt>() : nullptr;
  if (!pointer || pointer->is_grad)
    return false;
  const auto &origin = pointer->arg_id;
  auto descriptor_type = parameter.get_dtype();
  // Callable stores a scalar data pointer after ndarray scalarization, but
  // the caller's descriptor ArgLoad keeps the original tensor element type.
  // Both are the same ABI; preserve the exact rank/element/gradient layout.
  const Type *tensor_descriptor_type = descriptor_type;
  if (!parameter.element_shape.empty()) {
    auto *element_type = descriptor_type->as<StructType>()
                             ->get_element_type({TypeFactory::DATA_PTR_POS_IN_NDARRAY})
                             ->as<PointerType>()->get_pointee_type();
    auto &factory = TypeFactory::get_instance();
    tensor_descriptor_type = factory.get_ndarray_struct_type(
        factory.get_tensor_type(parameter.element_shape, element_type), ndim,
        parameter.needs_grad);
  }
  bool valid = true;
  for (int axis = 0; axis < TypeFactory::ndarray_shape_member_count(ndim); ++axis) {
    Stmt *value = leaf({TypeFactory::SHAPE_POS_IN_NDARRAY, axis});
    if (ndim == 0) {
      auto *padding = value ? value->cast<ConstStmt>() : nullptr;
      valid &= padding && padding->val.val_int() == 0;
    } else {
      auto *shape = value ? value->cast<ExternalTensorShapeAlongAxisStmt>() : nullptr;
      valid &= shape && shape->arg_id == origin && shape->axis == axis;
    }
  }
  for (int field = TypeFactory::ndarray_shape_member_count(ndim);
       field <= TypeFactory::affine_mode_pos_in_ndarray(ndim); ++field) {
    Stmt *value = leaf({TypeFactory::SHAPE_POS_IN_NDARRAY, field});
    auto *member = value ? value->cast<GetElementStmt>() : nullptr;
    auto *source = member ? member->src->cast<ArgLoadStmt>() : nullptr;
    valid &= member && !member->is_ptr && source && source->is_ptr &&
             !source->create_load && source->arg_id == origin &&
             (source->ret_type.ptr_removed() == descriptor_type ||
              (const Type *)source->ret_type.ptr_removed() == tensor_descriptor_type) &&
             member->index == std::vector<int>{TypeFactory::SHAPE_POS_IN_NDARRAY, field};
  }
  gradient = nullptr;
  if (parameter.needs_grad) {
    Stmt *value = leaf({TypeFactory::GRAD_PTR_POS_IN_NDARRAY});
    gradient = value ? value->cast<ExternalTensorBasePtrStmt>() : nullptr;
    valid &= gradient && gradient->is_grad && gradient->arg_id == origin;
  }
  return valid;
}

}  // namespace taichi::lang::irpass::analysis
