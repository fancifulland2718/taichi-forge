#include "taichi/aot/value_parameters.h"
#include "taichi/aot/value_layout.h"
#include "taichi/common/virtual_dir.h"
#include "taichi/program/callable.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace taichi::lang::aot {
namespace {
constexpr char kFilename[] = "value_parameters.json";

void validate_name(const std::string &name) {
  if (name.empty() || name.find('\0') != std::string::npos)
    throw std::invalid_argument("Invalid AOT value parameter kernel name");
}

void validate_indices(const std::vector<int> &indices) {
  if (indices.empty() ||
      std::any_of(indices.begin(), indices.end(), [](int i) { return i < 0; }))
    throw std::invalid_argument("Invalid AOT value parameter indices");
}

void validate_type(DataType type) {
  // Reuse the common pure-value type/extent checks; this is not a new layout.
  ValueLayout checked(type, std::numeric_limits<std::size_t>::max(),
                      ValueStorage::Host);
}
}  // namespace

DataType compiled_value_parameter_type(const CallableBase &kernel,
                                      const std::vector<int> &indices) {
  validate_indices(indices);
  if (!kernel.args_type)
    throw std::invalid_argument("Missing compiled value parameter ABI");
  DataType type(kernel.args_type);
  std::vector<int> prefix;
  for (int index : indices) {
    const auto *structure = type->cast<StructType>();
    if (!structure || index >= structure->elements().size())
      throw std::invalid_argument("Value parameter exceeds compiled ABI");
    type = structure->elements()[index].type;
    prefix.push_back(index);
    const auto parameter = kernel.nested_parameters.find(prefix);
    if (prefix.size() < indices.size() &&
        parameter != kernel.nested_parameters.end() &&
        parameter->second.is_argpack) {
      const auto pack = kernel.argpack_types.find(prefix);
      if (pack != kernel.argpack_types.end()) {
        if (!pack->second)
          throw std::invalid_argument("Missing compiled argument pack type");
        type = pack->second;
      } else {
        const auto *outer = type->cast<StructType>();
        if (!outer || outer->elements().size() != 1 ||
            !outer->elements()[0].type->is<PointerType>())
          throw std::invalid_argument("Invalid compiled argument pack ABI");
        type = DataType(outer->elements()[0].type).ptr_removed();
      }
    }
  }
  return type;
}

void validate_value_parameter_type(DataType logical, DataType lowered) {
  if (const auto *tensor = logical->cast<TensorType>()) {
    validate_type(logical);
    if (lowered->is<TensorType>()) {
      if (logical == lowered)
        return;
    } else if (const auto *structure = lowered->cast<StructType>()) {
      if (structure->elements().size() == tensor->get_num_elements() &&
          std::all_of(structure->elements().begin(),
                      structure->elements().end(), [&](const auto &member) {
                        return member.type == tensor->get_element_type();
                      }))
        return;
    }
    throw std::invalid_argument("Logical tensor differs from its lowered ABI");
  }
  if (const auto *structure = logical->cast<StructType>()) {
    const auto *physical = lowered->cast<StructType>();
    if (!physical || structure->elements().size() != physical->elements().size())
      throw std::invalid_argument("Logical structure differs from its lowered ABI");
    std::unordered_set<std::string> names;
    for (std::size_t i = 0; i < structure->elements().size(); ++i) {
      const auto &member = structure->elements()[i];
      const auto &target = physical->elements()[i];
      if (member.name.empty() || !names.insert(member.name).second ||
          member.name != target.name)
        throw std::invalid_argument("Logical value structure member mismatch");
      validate_value_parameter_type(member.type, target.type);
    }
    return;
  }
  validate_type(logical);
  if (logical != lowered)
    throw std::invalid_argument("Logical scalar differs from its lowered ABI");
}

ValueParameterArtifact make_value_parameter_artifact(
    const NamedValueParameters &parameters) {
  ValueParameterArtifact result;
  std::map<const Type *, std::uint64_t> types;
  for (const auto &[name, values] : parameters) {
    validate_name(name);
    if (values.empty())
      throw std::invalid_argument("Empty AOT value parameter entry");
    auto &entry = result.kernels[name];
    for (const auto &[indices, type] : values) {
      validate_indices(indices);
      const auto [found, inserted] =
          types.emplace(static_cast<const Type *>(type), result.types.size());
      if (inserted) {
        validate_type(type);
        result.types.push_back(type);
      }
      entry.push_back({indices, found->second});
    }
  }
  return result;
}

NamedValueParameters parse_value_parameter_artifact(
    const liong::json::JsonValue &json) {
  ValueParameterArtifact artifact;
  liong::json::deserialize(json, artifact, true);
  if (artifact.version != 1)
    throw std::invalid_argument("Unsupported AOT value parameter version");
  NamedValueParameters result;
  std::vector<bool> used(artifact.types.size());
  std::unordered_set<const Type *> unique;
  for (const auto &[name, values] : artifact.kernels) {
    validate_name(name);
    if (values.empty())
      throw std::invalid_argument("Empty AOT value parameter entry");
    for (const auto &value : values) {
      validate_indices(value.indices);
      if (value.type >= artifact.types.size())
        throw std::invalid_argument("Unknown AOT value parameter type");
      const auto type = artifact.types[value.type];
      if (!used[value.type]) {
        if (!unique.insert(static_cast<const Type *>(type)).second)
          throw std::invalid_argument("Duplicate AOT value parameter type");
        validate_type(type);
        used[value.type] = true;
      }
      if (!result[name].emplace(value.indices, type).second)
        throw std::invalid_argument("Duplicate AOT value parameter indices");
    }
  }
  if (std::find(used.begin(), used.end(), false) != used.end())
    throw std::invalid_argument("Unused AOT value parameter type");
  return result;
}

NamedValueParameters load_value_parameter_artifact(const io::VirtualDir *source) {
  std::size_t size = 0;
  if (!source || !source->get_file_size(kFilename, size))
    return {};
  std::vector<std::uint8_t> bytes;
  if (!source->load_file(kFilename, bytes) || bytes.empty())
    throw std::invalid_argument("Unreadable AOT value parameter metadata");
  return parse_value_parameter_artifact(liong::json::parse_complete(
      std::string(bytes.begin(), bytes.end())));
}

void write_value_parameter_artifact(const NamedValueParameters &parameters,
                                    const std::string &directory) {
  const auto path = std::filesystem::path(directory) / kFilename;
  if (parameters.empty()) {
    std::filesystem::remove(path);
    return;
  }
  const auto json = liong::json::print(
      liong::json::serialize(make_value_parameter_artifact(parameters)));
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(json.data(), json.size());
  if (!stream)
    throw std::runtime_error("Unable to write AOT value parameter metadata");
}
}  // namespace taichi::lang::aot
