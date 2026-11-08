#pragma once

#include <algorithm>
#include <cstring>
#include <map>

#include "taichi/aot/graph_data.h"
#include "taichi/ir/type_utils.h"
#include "taichi/program/launch_context_builder.h"

namespace taichi::lang::aot {

struct ArgumentConstant {
  unsigned bits;
  uint64_t value;
};
using ArgumentConstants = std::map<int, ArgumentConstant>;

// Facts belong to an immutable prepared invocation. Only ABI descriptor words
// and explicitly selected integer scalars qualify, never pointers or contents.
inline ArgumentConstants specialization_constants(
    const Kernel &kernel, LaunchContextBuilder &context,
    const std::vector<uint32_t> &constant_args) {
  ArgumentConstants constants;
  auto add = [&](const std::vector<int> &path, unsigned bits) {
    const int offset = kernel.args_type->get_element_offset(path);
    TI_ERROR_IF(offset < 0 || offset + bits / 8 > kernel.args_size,
                "Specialization argument is outside the loaded ABI");
    uint64_t value = 0;
    std::memcpy(&value, static_cast<const char *>(context.get_context().arg_buffer) + offset, bits / 8);
    constants.emplace(offset, ArgumentConstant{bits, value});
  };
  for (const auto &[path, parameter] : kernel.nested_parameters) {
    if (parameter.ptype != ParameterType::kNdarray)
      continue;
    auto member = path;
    member.push_back(TypeFactory::SHAPE_POS_IN_NDARRAY);
    member.push_back(0);
    const int rank = parameter.total_dim - parameter.element_shape.size();
    for (int axis = 0; axis < rank; ++axis) {
      member.back() = axis;
      add(member, 32);
    }
    if (std::find(kernel.runtime_affine_arrays.begin(), kernel.runtime_affine_arrays.end(), path) !=
        kernel.runtime_affine_arrays.end()) {
      for (int axis = 0; axis < rank; ++axis) {
        member.back() = TypeFactory::stride_pos_in_ndarray(rank, axis);
        add(member, 32);
      }
      member.back() = TypeFactory::affine_mode_pos_in_ndarray(rank);
      add(member, 32);
    }
  }
  for (uint32_t index : constant_args) {
    const auto parameter = kernel.nested_parameters.find({static_cast<int>(index)});
    TI_ERROR_IF(parameter == kernel.nested_parameters.end() || parameter->second.is_array ||
                    parameter->second.ptype != ParameterType::kScalar ||
                    !is_integral(parameter->second.get_dtype()),
                "Specialization constants must name integer scalar arguments");
    const auto bits = data_type_size(parameter->second.get_dtype()) * 8;
    TI_ERROR_IF(bits != 8 && bits != 16 && bits != 32 && bits != 64,
                "Unsupported specialization integer width");
    add({static_cast<int>(index)}, bits);
  }
  return constants;
}

inline std::string specialization_key(const std::string &code,
                                      const ArgumentConstants &constants) {
  std::string key = code;
  for (const auto &[offset, value] : constants)
    key += fmt::format("|{}:{}:{}", offset, value.bits, value.value);
  return key;
}

}  // namespace taichi::lang::aot
