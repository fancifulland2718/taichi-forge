#include "taichi/aot/graph_artifact.h"
#include "taichi/aot/struct_layout.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unordered_set>

namespace taichi::lang::aot {
namespace {
constexpr const char *kFilename = "graphs.v2.json";
constexpr const char *kLogicalFilename = "graphs.v3.json";
}

GraphArgumentSchema GraphArgumentSchema::from_arg(const Arg &arg) {
  GraphArgumentSchema schema;
  schema.argument = arg;
  schema.argument.pack_dtype = nullptr;
  schema.argument.struct_dtype = nullptr;
  schema.argument.members.clear();
  schema.argument.needs_grad.reset();
  if (arg.pack_dtype || arg.struct_dtype)
    schema.aggregate_type = arg.dtype();
  if (arg.needs_grad)
    schema.needs_grad = *arg.needs_grad ? 1 : 0;
  for (const auto &member : arg.members)
    schema.members.push_back(from_arg(member));
  return schema;
}

Arg GraphArgumentSchema::to_arg() const {
  Arg arg = argument;
  if (needs_grad < -1 || needs_grad > 1)
    throw std::invalid_argument("Invalid Graph gradient requirement");
  if (needs_grad != -1)
    arg.needs_grad = needs_grad != 0;
  if (arg.tag == ArgKind::kArgPack) {
    if (!aggregate_type->is<ArgPackType>())
      throw std::invalid_argument("Missing Graph ArgPack type");
    arg.pack_dtype = aggregate_type->as<ArgPackType>();
    std::unordered_set<std::string> names;
    for (const auto &member : members) {
      auto restored = member.to_arg();
      if (!names.insert(restored.name).second)
        throw std::invalid_argument("Duplicate Graph ArgPack member");
      arg.members.push_back(std::move(restored));
    }
  } else {
    if (!members.empty())
      throw std::invalid_argument("Only Graph ArgPacks have logical members");
    if (aggregate_type != PrimitiveType::unknown) {
      if (!aggregate_type->is<StructType>() ||
          (arg.tag != ArgKind::kNdarray && arg.tag != ArgKind::kStruct))
        throw std::invalid_argument("Invalid Graph aggregate type");
      arg.struct_dtype = aggregate_type->as<StructType>();
    }
  }
  return arg;
}

bool has_argpack_graph_arguments(const GraphMap &graphs) {
  for (const auto &[name, graph] : graphs)
    for (const auto &dispatch : graph.dispatches)
      for (const auto &arg : dispatch.symbolic_args)
        if (arg.tag == ArgKind::kArgPack)
          return true;
  return false;
}

bool has_struct_graph_arguments(const GraphMap &graphs) {
  for (const auto &[name, graph] : graphs) {
    for (const auto &dispatch : graph.dispatches) {
      for (const auto &arg : dispatch.symbolic_args) {
        if (arg.struct_dtype)
          return true;
      }
    }
  }
  return false;
}

void write_struct_graph_artifact(const GraphMap &graphs,
                                 const std::string &directory) {
  const auto path = std::filesystem::path(directory) / kFilename;
  const auto logical_path = std::filesystem::path(directory) / kLogicalFilename;
  if (has_argpack_graph_arguments(graphs)) {
    LogicalGraphArtifact artifact;
    artifact.graphs = graphs;
    for (const auto &[name, graph] : graphs)
      for (const auto &[key, arg] : graph.args)
        artifact.arguments[name].emplace(key, GraphArgumentSchema::from_arg(arg));
    const auto json = liong::json::print(liong::json::serialize(artifact));
    std::ofstream out(logical_path, std::ios::binary | std::ios::trunc);
    out.write(json.data(), json.size());
    if (!out)
      throw std::runtime_error("Unable to write AOT Graph v3 metadata");
    std::filesystem::remove(path);
    return;
  }
  std::filesystem::remove(logical_path);
  if (!has_struct_graph_arguments(graphs)) {
    // A save may replace a previous generation in the same directory.
    std::filesystem::remove(path);
    return;
  }
  GraphArtifact artifact;
  artifact.version = 2;
  artifact.graphs = graphs;
  for (const auto &[name, graph] : graphs) {
    for (const auto &dispatch : graph.dispatches) {
      for (const auto &arg : dispatch.symbolic_args) {
        if (arg.struct_dtype) {
          StructLayout layout(arg.struct_dtype);
          auto &types = artifact.struct_types[name];
          const auto [it, added] = types.emplace(arg.name, arg.struct_dtype);
          if (!added && it->second != DataType(arg.struct_dtype)) {
            throw std::invalid_argument(
                "Conflicting AOT Graph structure types");
          }
        }
      }
    }
  }
  const auto json = liong::json::print(liong::json::serialize(artifact));
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(json.data(), json.size());
  if (!out)
    throw std::runtime_error("Unable to write AOT Graph v2 metadata");
}

bool load_struct_graph_artifact(const io::VirtualDir &directory,
                                GraphMap &graphs) {
  std::vector<uint8_t> bytes;
  size_t size = 0;
  if (directory.get_file_size(kLogicalFilename, size)) {
    if (!directory.load_file(kLogicalFilename, bytes) || bytes.empty())
      throw std::invalid_argument("Unreadable AOT Graph v3 metadata");
    const auto json = liong::json::parse(
        reinterpret_cast<const char *>(bytes.data()),
        reinterpret_cast<const char *>(bytes.data() + bytes.size()));
    LogicalGraphArtifact artifact;
    liong::json::deserialize(json, artifact);
    if (artifact.version != 3)
      throw std::invalid_argument("Unsupported AOT Graph artifact version");
    for (const auto &[name, arguments] : artifact.arguments)
      if (!artifact.graphs.count(name))
        throw std::invalid_argument("AOT Graph schema names an unknown graph");
    for (auto &[name, graph] : artifact.graphs) {
      const auto found = artifact.arguments.find(name);
      std::unordered_set<std::string> used;
      for (auto &dispatch : graph.dispatches) {
        for (auto &arg : dispatch.symbolic_args) {
          if (found == artifact.arguments.end() || !found->second.count(arg.name))
            throw std::invalid_argument("Missing AOT Graph logical argument");
          const auto &schema = found->second.at(arg.name);
          if (schema.argument != arg)
            throw std::invalid_argument("Conflicting AOT Graph logical argument");
          used.insert(arg.name);
          arg = schema.to_arg();
        }
      }
      if (found != artifact.arguments.end() && used.size() != found->second.size())
        throw std::invalid_argument("Unused AOT Graph logical argument");
      graph = CompiledGraph(std::move(graph.dispatches));
    }
    graphs = std::move(artifact.graphs);
    return true;
  }
  if (!directory.get_file_size(kFilename, size))
    return false;
  if (!directory.load_file(kFilename, bytes) || bytes.empty()) {
    throw std::invalid_argument("Unreadable AOT Graph v2 metadata");
  }
  const auto json = liong::json::parse(
      reinterpret_cast<const char *>(bytes.data()),
      reinterpret_cast<const char *>(bytes.data() + bytes.size()));
  GraphArtifact artifact;
  liong::json::deserialize(json, artifact);
  if (artifact.version != 2) {
    throw std::invalid_argument("Unsupported AOT Graph artifact version");
  }
  for (const auto &[name, types] : artifact.struct_types) {
    if (!artifact.graphs.count(name)) {
      throw std::invalid_argument("AOT Graph layout names an unknown graph");
    }
    for (const auto &[arg, type] : types) {
      StructLayout layout(type);
    }
  }
  for (auto &[name, graph] : artifact.graphs) {
    const auto types = artifact.struct_types.find(name);
    std::unordered_set<std::string> used;
    for (auto &dispatch : graph.dispatches) {
      for (auto &arg : dispatch.symbolic_args) {
        if (types != artifact.struct_types.end() &&
            types->second.count(arg.name)) {
          if (arg.tag != ArgKind::kNdarray ||
              arg.dtype_id != PrimitiveTypeID::unknown ||
              !arg.element_shape.empty()) {
            throw std::invalid_argument(
                "Conflicting AOT Graph layout metadata");
          }
          arg.struct_dtype = types->second.at(arg.name)->as<StructType>();
          used.insert(arg.name);
        } else if (arg.tag == ArgKind::kNdarray &&
                   arg.dtype_id == PrimitiveTypeID::unknown) {
          throw std::invalid_argument("Missing AOT Graph structure layout");
        }
      }
    }
    if (types != artifact.struct_types.end() &&
        used.size() != types->second.size()) {
      throw std::invalid_argument("Unused AOT Graph structure layout");
    }
    // Deserialization stores dispatches only. Rebuild the named-argument map
    // once, also checking that repeated names have an identical contract.
    graph = CompiledGraph(std::move(graph.dispatches));
  }
  graphs = std::move(artifact.graphs);
  return true;
}

}  // namespace taichi::lang::aot
