#pragma once

#include <string>
#include <vector>

#include "taichi/ir/type_utils.h"

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

}  // namespace taichi::lang::aot
