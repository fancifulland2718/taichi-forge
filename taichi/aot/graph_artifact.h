#pragma once

#include "taichi/aot/graph_data.h"
#include "taichi/common/virtual_dir.h"

namespace taichi::lang::aot {

using GraphMap = std::unordered_map<std::string, CompiledGraph>;

// Legacy Arg/CompiledGraph binary layouts are unchanged. Only graphs that use
// structure elements need the v2 envelope and its shared type dictionary.
struct GraphArtifact {
  uint32_t version{0};
  GraphMap graphs;
  std::unordered_map<std::string, std::unordered_map<std::string, DataType>>
      struct_types;
  TI_IO_DEF(version, graphs, struct_types);
};

bool has_struct_graph_arguments(const GraphMap &graphs);
void write_struct_graph_artifact(const GraphMap &graphs,
                                 const std::string &directory);
bool load_struct_graph_artifact(const io::VirtualDir &directory,
                                GraphMap &graphs);

}  // namespace taichi::lang::aot
