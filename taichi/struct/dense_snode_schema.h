#pragma once

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "taichi/common/serialization.h"
#include "taichi/ir/snode.h"

namespace taichi::lang {

// A parameterized construction recipe, NOT a second byte-layout calculator.
// instantiate() calls the same SNode constructors as FieldsBuilder. The owner
// then runs its normal StructCompiler and allocator on the resulting tree.
// Node indices are stable within the recipe; live SNode IDs/offsets are absent.
struct DenseSNodeSchema {
  struct Extent {
    int axis{0};
    int constant{0};
    std::string parameter;
    TI_IO_DEF(axis, constant, parameter);
  };
  struct Node {
    int parent{-1};
    SNodeType type{SNodeType::undefined};
    DataType dtype{PrimitiveType::gen};
    std::vector<Extent> extents;
    TI_IO_DEF(parent, type, dtype, extents);
  };

  int version{1};
  std::vector<Node> nodes;
  TI_IO_DEF(version, nodes);

  using Parameter = std::tuple<const SNode *, int, std::string>;
  static DenseSNodeSchema capture(const SNode &root,
                                  const std::vector<Parameter> &parameters);
  void validate() const;
  std::unique_ptr<SNode> instantiate(
      const std::map<std::string, int> &dimensions) const;
  std::string to_json() const;
  static DenseSNodeSchema from_json(const std::string &text);
};

}  // namespace taichi::lang
