#pragma once

#include "taichi/aot/graph_data.h"
#include "taichi/common/virtual_dir.h"

namespace taichi::lang::aot {

using GraphMap = std::unordered_map<std::string, CompiledGraph>;

// Legacy Arg/CompiledGraph binary layouts remain unchanged. Version 3 adds a
// logical schema dictionary for packs; compiled kernels remain authoritative
// for physical layout, lifted resources and launch ABI.
struct GraphArgumentSchema {
  Arg argument{};
  DataType aggregate_type{PrimitiveType::unknown};
  std::vector<GraphArgumentSchema> members;
  int needs_grad{-1};
  TI_IO_DEF(argument, aggregate_type, members, needs_grad);

  static GraphArgumentSchema from_arg(const Arg &arg);
  Arg to_arg() const;
};

struct GraphArtifact {
  uint32_t version{0};
  GraphMap graphs;
  std::unordered_map<std::string, std::unordered_map<std::string, DataType>>
      struct_types;
  TI_IO_DEF(version, graphs, struct_types);
};

struct LogicalGraphArtifact {
  uint32_t version{3};
  GraphMap graphs;
  std::unordered_map<std::string,
                     std::unordered_map<std::string, GraphArgumentSchema>>
      arguments;
  TI_IO_DEF(version, graphs, arguments);
};

bool has_struct_graph_arguments(const GraphMap &graphs);
bool has_argpack_graph_arguments(const GraphMap &graphs);
void write_struct_graph_artifact(const GraphMap &graphs,
                                 const std::string &directory);
bool load_struct_graph_artifact(const io::VirtualDir &directory,
                                GraphMap &graphs);

}  // namespace taichi::lang::aot
