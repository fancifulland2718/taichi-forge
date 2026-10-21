#include "taichi/aot/struct_layout.h"
#include "taichi/aot/value_layout.h"
#include "taichi/aot/value_parameters.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace taichi::lang::aot {
namespace {

struct ReflectionExtent {
  std::size_t size;
  std::size_t alignment;
};

ReflectionExtent reflect(DataType type,
                         const std::string &path,
                         std::vector<int> &indices,
                         std::vector<ValueLayoutMember> &members,
                         ValueStorage storage,
                         bool array_element) {
  if (const auto *structure = type->cast<StructType>()) {
    std::unordered_set<std::string> names;
    std::size_t end = 0;
    std::size_t alignment = 1;
    if (array_element && structure->elements().empty()) {
      throw std::invalid_argument(
          "Empty StructNdarray elements are unsupported");
    }
    const bool compiled_offsets = structure->get_layout() != "none";
    for (int i = 0; i < structure->elements().size(); ++i) {
      const auto &member = structure->elements()[i];
      if (array_element &&
          (member.name.empty() || member.name.find('.') != std::string::npos ||
           !names.insert(member.name).second)) {
        throw std::invalid_argument("Invalid StructNdarray member path");
      }
      // Validate types before invoking the native layout helpers (pointer and
      // unresolved primitive types do not have a portable AoS layout).
      const auto first_child = members.size();
      indices.push_back(i);
      const auto child = reflect(
          member.type, path.empty() ? member.name : path + "." + member.name,
          indices, members, storage, array_element);
      indices.pop_back();
      // An explicit compiled layout with all-zero offsets must not be
      // reinterpreted as an implicit natural layout (which can hide overlap).
      const auto offset = compiled_offsets ? member.offset
                                           : structure->get_element_offset({i});
      if (offset < end || offset % child.alignment != 0 ||
          offset > std::numeric_limits<std::size_t>::max() - child.size) {
        throw std::invalid_argument("Invalid value member layout");
      }
      if (array_element &&
          offset > std::numeric_limits<int>::max() - child.size) {
        throw std::invalid_argument("StructNdarray member exceeds native type size");
      }
      // Child offsets were relative to this member. No second padding
      // algorithm is used; the compiler's StructType supplies every offset.
      for (auto index = first_child; index < members.size(); ++index) {
        members[index].offset += offset;
      }
      end = offset + child.size;
      alignment = std::max(alignment, child.alignment);
    }
    if (array_element) {
      if (end > std::numeric_limits<int>::max() - (alignment - 1)) {
        throw std::invalid_argument(
            "StructNdarray stride exceeds native type size");
      }
      // Only array elements use the existing natural AoS stride. Compiled
      // value buffers supply their size externally and never get repacked.
      return {static_cast<std::size_t>(data_type_size(type)), alignment};
    }
    return {end, alignment};
  }
  auto scalar = type;
  std::vector<int> shape;
  std::size_t count = 1;
  if (const auto *tensor = type->cast<TensorType>()) {
    shape = tensor->get_shape();
    scalar = tensor->get_element_type();
    if (shape.empty() || (array_element && shape.size() > 2)) {
      throw std::invalid_argument(
          "Invalid value tensor shape; StructNdarray supports vector/matrix members");
    }
    for (int extent : shape) {
      // TensorType represents its element count and native byte size as int.
      if (extent <= 0 || count > std::numeric_limits<int>::max() / extent) {
        throw std::invalid_argument("Invalid value tensor extent");
      }
      count *= extent;
    }
  }
  if (!scalar->is<PrimitiveType>() || data_type_size(scalar) <= 0) {
    throw std::invalid_argument(
        "Value layouts require concrete scalar members");
  }
  const auto scalar_size = static_cast<std::size_t>(
      storage == ValueStorage::Gfx ? data_type_size_gfx(scalar)
                                    : data_type_size(scalar));
  if (count > std::numeric_limits<int>::max() / scalar_size) {
    throw std::invalid_argument("Value tensor exceeds native type size");
  }
  members.push_back({path, indices, scalar, shape, 0, count * scalar_size,
                     scalar_size, scalar_size});
  return {count * scalar_size, scalar_size};
}

}  // namespace

ValueLayout::ValueLayout(DataType type,
                         std::size_t buffer_size,
                         ValueStorage storage)
    : type(type), buffer_size(buffer_size) {
  std::vector<int> indices;
  const auto extent = reflect(type, "", indices, members, storage, false);
  if (extent.size > buffer_size) {
    throw std::invalid_argument("Value members exceed compiled buffer size");
  }
}

ValueLayout::ValueLayout(DataType logical,
                         DataType physical,
                         std::size_t buffer_size,
                         ValueStorage storage)
    : ValueLayout(logical, std::numeric_limits<std::size_t>::max(),
                   ValueStorage::Host) {
  validate_value_parameter_type(logical, physical);
  ValueLayout compiled(physical, buffer_size, storage);
  this->buffer_size = buffer_size;
  std::size_t leaf = 0, component = 0;
  for (auto &member : members) {
    const auto count = member.size / member.scalar_size;
    std::size_t end = 0;
    for (std::size_t i = 0; i < count; ++i) {
      if (leaf >= compiled.members.size())
        throw std::invalid_argument("Missing compiled value component");
      const auto &source = compiled.members[leaf];
      if (member.scalar_type != source.scalar_type)
        throw std::invalid_argument("Compiled value component type mismatch");
      const auto offset = source.offset + component * source.scalar_size;
      if (i == 0) {
        member.offset = offset;
        member.scalar_size = source.scalar_size;
        member.scalar_alignment = source.scalar_alignment;
      }
      if (offset != member.offset + i * member.scalar_size &&
          member.component_offsets.empty()) {
        member.component_offsets.reserve(count);
        for (std::size_t j = 0; j < i; ++j)
          member.component_offsets.push_back(member.offset + j * member.scalar_size);
      }
      if (!member.component_offsets.empty())
        member.component_offsets.push_back(offset);
      end = offset + source.scalar_size;
      if (++component == source.size / source.scalar_size) {
        ++leaf;
        component = 0;
      }
    }
    member.size = end - member.offset;
  }
  if (leaf != compiled.members.size() || component != 0)
    throw std::invalid_argument("Unexpected compiled value components");
}

StructLayout::StructLayout(DataType type) : type(type) {
  if (!type->is<StructType>()) {
    throw std::invalid_argument(
        "StructNdarray layout requires a structure type");
  }
  std::vector<ValueLayoutMember> values;
  std::vector<int> indices;
  const auto extent =
      reflect(type, "", indices, values, ValueStorage::Host, true);
  stride = extent.size;
  alignment = extent.alignment;
  members.reserve(values.size());
  for (auto &value : values) {
    members.push_back({std::move(value.path), value.scalar_type,
                       std::move(value.shape), value.offset, value.size,
                       value.scalar_alignment});
  }
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
  return parse_struct_layout_artifact(liong::json::parse_complete(
      std::string(bytes.begin(), bytes.end())));
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
