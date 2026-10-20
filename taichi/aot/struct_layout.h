#pragma once

#include <map>
#include <string>
#include <vector>

#include "taichi/ir/type_utils.h"
#include "taichi/common/json.h"
#include "taichi/common/serialization.h"
#include "taichi/common/virtual_dir.h"

namespace taichi::lang::aot {

// Reflection of the layout used by existing StructNdarray lowering. This is
// constructed at the cold module/argument boundary, never on each dispatch.
struct StructLayoutMember {
  std::string path;
  DataType scalar_type;
  std::vector<int> shape;
  std::size_t offset;
  std::size_t size;
  std::size_t alignment;
};

struct StructLayout {
  static constexpr uint32_t kVersion = 1;
  DataType type;
  std::size_t stride;
  std::size_t alignment;
  std::vector<StructLayoutMember> members;

  explicit StructLayout(DataType type);
};

// Named layouts are independent of executable entries. Several public names
// may refer to one type; the artifact stores each structural type only once.
using NamedStructLayouts = std::map<std::string, DataType>;
struct StructLayoutArtifact {
  std::uint32_t version{1};
  std::vector<DataType> types;
  std::map<std::string, std::uint64_t> layouts;
  TI_IO_DEF(version, types, layouts);
};

StructLayoutArtifact make_struct_layout_artifact(
    const NamedStructLayouts &layouts);
NamedStructLayouts parse_struct_layout_artifact(
    const liong::json::JsonValue &json);
NamedStructLayouts load_struct_layout_artifact(const io::VirtualDir *directory);
void write_struct_layout_artifact(const NamedStructLayouts &layouts,
                                 const std::string &directory);

}  // namespace taichi::lang::aot
