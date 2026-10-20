#include "taichi/aot/struct_layout.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
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

namespace {
constexpr const char *kLayoutFilename = "array_layouts.json";

void validate_layout_name(const std::string &name) {
  if (name.empty() || name.find('\0') != std::string::npos)
    throw std::invalid_argument("Invalid AOT ndarray layout name");
}
}  // namespace

StructLayoutArtifact make_struct_layout_artifact(
    const NamedStructLayouts &layouts) {
  StructLayoutArtifact result;
  std::map<const Type *, std::uint64_t> indices;
  for (const auto &[name, type] : layouts) {
    validate_layout_name(name);
    const auto [it, added] = indices.emplace(static_cast<const Type *>(type),
                                            result.types.size());
    if (added) {
      StructLayout checked(type);
      result.types.push_back(type);
    }
    result.layouts.emplace(name, it->second);
  }
  return result;
}

NamedStructLayouts parse_struct_layout_artifact(
    const liong::json::JsonValue &json) {
  StructLayoutArtifact artifact;
  liong::json::deserialize(json, artifact);
  if (artifact.version != 1)
    throw std::invalid_argument("Unsupported AOT ndarray layout artifact version");
  NamedStructLayouts result;
  std::vector<bool> used(artifact.types.size());
  std::unordered_set<const Type *> unique;
  for (const auto &[name, index] : artifact.layouts) {
    validate_layout_name(name);
    if (index >= artifact.types.size())
      throw std::invalid_argument("AOT ndarray layout references an unknown type");
    if (!used[index]) {
      if (!unique.insert(static_cast<const Type *>(artifact.types[index])).second)
        throw std::invalid_argument("Duplicate AOT ndarray layout type");
      StructLayout checked(artifact.types[index]);
      used[index] = true;
    }
    result.emplace(name, artifact.types[index]);
  }
  if (std::find(used.begin(), used.end(), false) != used.end())
    throw std::invalid_argument("Unused AOT ndarray layout type");
  return result;
}

NamedStructLayouts load_struct_layout_artifact(const io::VirtualDir *directory) {
  std::size_t size = 0;
  if (!directory || !directory->get_file_size(kLayoutFilename, size))
    return {};
  std::vector<std::uint8_t> bytes;
  if (!directory->load_file(kLayoutFilename, bytes) || bytes.empty())
    throw std::invalid_argument("Unreadable AOT ndarray layout metadata");
  return parse_struct_layout_artifact(liong::json::parse(
      reinterpret_cast<const char *>(bytes.data()),
      reinterpret_cast<const char *>(bytes.data() + bytes.size())));
}

void write_struct_layout_artifact(const NamedStructLayouts &layouts,
                                 const std::string &directory) {
  const auto path = std::filesystem::path(directory) / kLayoutFilename;
  if (layouts.empty()) {
    std::filesystem::remove(path);
    return;
  }
  const auto json = liong::json::print(
      liong::json::serialize(make_struct_layout_artifact(layouts)));
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(json.data(), json.size());
  if (!stream)
    throw std::runtime_error("Unable to write AOT ndarray layout metadata");
}

}  // namespace taichi::lang::aot
