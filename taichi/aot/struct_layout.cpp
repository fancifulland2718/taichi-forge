#include "taichi/aot/struct_layout.h"

#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace taichi::lang::aot {
namespace {

void reflect(DataType type,
             const std::string &path,
             std::size_t base,
             std::vector<StructLayoutMember> &members) {
  if (const auto *structure = type->cast<StructType>()) {
    std::unordered_set<std::string> names;
    std::size_t end = 0;
    if (structure->elements().empty()) {
      throw std::invalid_argument(
          "Empty StructNdarray elements are unsupported");
    }
    for (int i = 0; i < structure->elements().size(); ++i) {
      const auto &member = structure->elements()[i];
      if (member.name.empty() || member.name.find('.') != std::string::npos ||
          !names.insert(member.name).second) {
        throw std::invalid_argument("Invalid StructNdarray member path");
      }
      // Validate types before invoking the native layout helpers (pointer and
      // unresolved primitive types do not have a portable AoS layout).
      const auto first_child = members.size();
      reflect(member.type,
              path.empty() ? member.name : path + "." + member.name, 0,
              members);
      const auto offset = structure->get_element_offset({i});
      const auto size = data_type_size(member.type);
      const auto alignment = data_type_alignment(member.type);
      if (offset < end || offset % alignment != 0 ||
          offset > std::numeric_limits<int>::max() - size) {
        throw std::invalid_argument("Invalid StructNdarray member layout");
      }
      // Child offsets were relative to this member. No second padding
      // algorithm is used; the compiler's StructType supplies every offset.
      for (auto child = first_child; child < members.size(); ++child) {
        members[child].offset += base + offset;
      }
      end = offset + size;
    }
    const auto alignment = data_type_alignment(type);
    if (end > std::numeric_limits<int>::max() - (alignment - 1)) {
      throw std::invalid_argument(
          "StructNdarray stride exceeds native type size");
    }
    return;
  }
  auto scalar = type;
  std::vector<int> shape;
  if (const auto *tensor = type->cast<TensorType>()) {
    shape = tensor->get_shape();
    scalar = tensor->get_element_type();
    if (shape.empty() || shape.size() > 2) {
      throw std::invalid_argument(
          "StructNdarray supports vector/matrix members");
    }
    uint64_t count = 1;
    for (int extent : shape) {
      if (extent <= 0 || count > std::numeric_limits<int>::max() / extent) {
        throw std::invalid_argument("Invalid StructNdarray tensor extent");
      }
      count *= extent;
    }
    if (!scalar->is<PrimitiveType>() || data_type_size(scalar) <= 0 ||
        count > std::numeric_limits<int>::max() / data_type_size(scalar)) {
      throw std::invalid_argument("Invalid StructNdarray tensor scalar type");
    }
  }
  if (!scalar->is<PrimitiveType>() || data_type_size(scalar) <= 0) {
    throw std::invalid_argument(
        "StructNdarray requires concrete scalar members");
  }
  members.push_back({path, scalar, shape, base,
                     static_cast<std::size_t>(data_type_size(type)),
                     static_cast<std::size_t>(data_type_alignment(type))});
}

}  // namespace

StructLayout::StructLayout(DataType type) : type(type) {
  if (!type->is<StructType>()) {
    throw std::invalid_argument(
        "StructNdarray layout requires a structure type");
  }
  reflect(type, "", 0, members);
  stride = data_type_size(type);
  alignment = data_type_alignment(type);
  if (stride == 0 || stride > std::numeric_limits<int>::max() ||
      stride % alignment != 0) {
    throw std::invalid_argument("Invalid StructNdarray element stride");
  }
}

}  // namespace taichi::lang::aot
