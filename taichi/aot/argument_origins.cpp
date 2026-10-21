#include "taichi/aot/argument_origins.h"
#include "taichi/aot/value_layout.h"
#include "taichi/aot/value_parameters.h"
#include "taichi/common/virtual_dir.h"
#include "taichi/program/callable.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

namespace taichi::lang::aot {
namespace {
constexpr char kFilename[] = "argument_origins.json";

void require(bool condition, const char *message) {
  if (!condition)
    throw std::invalid_argument(message);
}
bool valid_name(const std::string &name) {
  return !name.empty() && name.find('\0') == std::string::npos;
}
bool valid_indices(const std::vector<int> &indices) {
  return !indices.empty() &&
         std::all_of(indices.begin(), indices.end(), [](int i) { return i >= 0; });
}
std::vector<int> parent(std::vector<int> indices) {
  indices.pop_back();
  return indices;
}
bool is_resource(const std::string &kind) {
  static const std::set<std::string> kinds{
      "ndarray", "texture", "rw_texture", "texture_collection",
      "acceleration_structure", "sparse_matrix_builder"};
  return kinds.count(kind) != 0;
}
}  // namespace

ArgumentOrigins canonical_argument_origins(ArgumentOrigins origins) {
  require(!origins.empty(), "Empty AOT argument origin layout");
  std::sort(origins.begin(), origins.end());
  std::map<std::vector<int>, const ArgumentOrigin *> logical;
  std::set<std::vector<int>> physical;
  std::map<std::vector<int>, std::set<std::string>> names;
  std::map<std::vector<int>, std::size_t> next_child;
  bool has_pack = false;
  for (const auto &entry : origins) {
    require(valid_indices(entry.logical) && valid_indices(entry.physical),
            "Invalid AOT argument origin indices");
    require(valid_name(entry.name), "Invalid AOT argument origin name");
    require(entry.kind == "value" || entry.kind == "argpack" ||
                is_resource(entry.kind),
            "Unknown AOT argument origin kind");
    require(logical.emplace(entry.logical, &entry).second &&
                physical.insert(entry.physical).second,
            "Duplicate AOT argument origin path");
    const auto container = parent(entry.logical);
    require(names[container].insert(entry.name).second,
            "Duplicate AOT argument member name");
    require(std::size_t(entry.logical.back()) == next_child[container]++,
            "Noncontiguous AOT argument member indices");
    if (!container.empty()) {
      const auto found = logical.find(container);
      require(found != logical.end() && found->second->kind == "argpack",
              "AOT argument member has no pack parent");
      if (!is_resource(entry.kind))
        require(parent(entry.physical) == found->second->physical,
                "AOT argument member has a different physical parent");
    } else {
      require(entry.physical.size() == 1,
              "AOT public argument is not a physical root");
    }
    if (is_resource(entry.kind))
      require(entry.physical.size() == 1,
              "AOT resource argument was not lifted to a root");
    has_pack |= entry.kind == "argpack";
  }
  require(has_pack, "AOT argument origins require an argument pack");
  return origins;
}

void validate_argument_origins(const CallableBase &kernel,
                               const ArgumentOrigins &origins) {
  const auto checked = canonical_argument_origins(origins);
  require(checked.size() == kernel.nested_parameters.size(),
          "AOT argument origins do not cover compiled parameters");
  for (const auto &entry : checked) {
    const auto found = kernel.nested_parameters.find(entry.physical);
    require(found != kernel.nested_parameters.end(),
            "AOT argument origin has no compiled parameter");
    const auto &parameter = found->second;
    const auto type = compiled_value_parameter_type(kernel, entry.physical);
    if (entry.kind == "argpack") {
      require(parameter.is_argpack && !parameter.is_array,
              "AOT argument origin does not name a pack");
      const auto *outer = type->cast<StructType>();
      require(outer && outer->elements().size() == 1 &&
                  outer->elements()[0].type->is<PointerType>(),
              "Invalid compiled argument pack pointer");
      const auto storage = kernel.argpack_types.find(entry.physical);
      if (storage != kernel.argpack_types.end()) {
        require(storage->second != nullptr,
                "Missing compiled argument pack storage");
      } else {
        require(DataType(outer->elements()[0].type).ptr_removed()->is<StructType>(),
                "Missing compiled argument pack storage");
      }
    } else if (entry.kind == "value") {
      require(!parameter.is_argpack && !parameter.is_array &&
                  parameter.ptype == ParameterType::kScalar,
              "AOT value origin names a resource");
      ValueLayout layout(type, std::numeric_limits<std::size_t>::max(),
                         ValueStorage::Host);
    } else {
      require(!parameter.is_argpack,
              "AOT resource origin names an argument pack");
      bool valid = false;
      if (entry.kind == "sparse_matrix_builder") {
        valid = (parameter.is_array &&
                 parameter.ptype == ParameterType::kNdarray) ||
                (!parameter.is_array &&
                 parameter.ptype == ParameterType::kScalar &&
                 type == PrimitiveType::u64);
      } else if (parameter.is_array) {
        valid = (entry.kind == "ndarray" &&
                 parameter.ptype == ParameterType::kNdarray) ||
                (entry.kind == "texture" &&
                 parameter.ptype == ParameterType::kTexture) ||
                (entry.kind == "rw_texture" &&
                 parameter.ptype == ParameterType::kRWTexture) ||
                (entry.kind == "texture_collection" &&
                 parameter.ptype == ParameterType::kTextureCollection) ||
                (entry.kind == "acceleration_structure" &&
                 parameter.ptype == ParameterType::kAccelerationStructure);
      }
      require(valid, "AOT resource origin differs from its compiled kind");
    }
  }
}

ArgumentOriginArtifact make_argument_origin_artifact(
    const NamedArgumentOrigins &origins) {
  ArgumentOriginArtifact artifact;
  std::map<ArgumentOrigins, std::uint64_t> layouts;
  for (const auto &[name, entries] : origins) {
    require(valid_name(name), "Invalid AOT argument origin kernel name");
    auto canonical = canonical_argument_origins(entries);
    const auto [it, inserted] = layouts.emplace(canonical, artifact.layouts.size());
    if (inserted)
      artifact.layouts.push_back(std::move(canonical));
    artifact.kernels.emplace(name, it->second);
  }
  return artifact;
}

NamedArgumentOrigins parse_argument_origin_artifact(
    const liong::json::JsonValue &json) {
  ArgumentOriginArtifact artifact;
  liong::json::deserialize(json, artifact, true);
  require(artifact.version == 1, "Unsupported AOT argument origin version");
  std::set<ArgumentOrigins> unique;
  for (auto &layout : artifact.layouts) {
    layout = canonical_argument_origins(std::move(layout));
    require(unique.insert(layout).second, "Duplicate AOT argument origin layout");
  }
  std::vector<bool> used(artifact.layouts.size());
  NamedArgumentOrigins result;
  for (const auto &[name, index] : artifact.kernels) {
    require(valid_name(name), "Invalid AOT argument origin kernel name");
    require(index < artifact.layouts.size(), "Unknown AOT argument origin layout");
    used[index] = true;
    result.emplace(name, artifact.layouts[index]);
  }
  require(std::find(used.begin(), used.end(), false) == used.end(),
          "Unused AOT argument origin layout");
  return result;
}

NamedArgumentOrigins load_argument_origin_artifact(const io::VirtualDir *source) {
  std::size_t size = 0;
  if (!source || !source->get_file_size(kFilename, size))
    return {};
  std::vector<std::uint8_t> bytes;
  require(source->load_file(kFilename, bytes) && !bytes.empty(),
          "Unreadable AOT argument origin metadata");
  return parse_argument_origin_artifact(liong::json::parse_complete(
      std::string(bytes.begin(), bytes.end())));
}

void write_argument_origin_artifact(const NamedArgumentOrigins &origins,
                                     const std::string &directory) {
  const auto path = std::filesystem::path(directory) / kFilename;
  if (origins.empty()) {
    std::filesystem::remove(path);
    return;
  }
  const auto text = liong::json::print(
      liong::json::serialize(make_argument_origin_artifact(origins)));
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(text.data(), text.size());
  if (!stream)
    throw std::runtime_error("Unable to write AOT argument origin metadata");
}

}  // namespace taichi::lang::aot
