#pragma once

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "taichi/common/serialization.h"
#include "taichi/ir/snode.h"

namespace taichi::lang {

// Affine projection of a compiled Dense tree. Each logical axis must be
// introduced exactly once along the leaf path; tiled/repeated axes need a
// different addressing contract. Byte facts come from the StructCompiler.
struct DenseSNodeMemberLayout {
  std::uint64_t offset{0};
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> strides;
};
DenseSNodeMemberLayout dense_snode_member_layout(const SNode &leaf);

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
    // Dense: index axes. Version 2 place: logical tensor axes, expanded as
    // adjacent scalar places by the ordinary SNode construction API.
    std::vector<Extent> extents;
    TI_IO_DEF(parent, type, dtype, extents);
  };

  int version{1};
  std::vector<Node> nodes;
  TI_IO_DEF(version, nodes);

  using Parameter = std::tuple<const SNode *, int, std::string>;
  static DenseSNodeSchema capture(const SNode &root,
                                  const std::vector<Parameter> &parameters);
  using TensorParameter = std::tuple<std::vector<const SNode *>,
                                     std::vector<int>,
                                     std::vector<std::string>>;
  static DenseSNodeSchema capture_tensors(
      const SNode &root,
      const std::vector<Parameter> &parameters,
      const std::vector<TensorParameter> &tensors);
  // Producer-only live SNode ID -> persistent recipe node. Compacted tensor
  // components share one node; source IDs are never serialized in the recipe.
  static std::pair<DenseSNodeSchema, std::map<int, int>> capture_tensor_members(
      const SNode &root,
      const std::vector<Parameter> &parameters,
      const std::vector<TensorParameter> &tensors);
  void validate() const;
  std::unique_ptr<SNode> instantiate(
      const std::map<std::string, int> &dimensions) const;
  std::unique_ptr<SNode> instantiate_projected(
      const std::map<std::string, int> &dimensions,
      std::vector<std::vector<SNode *>> &recipe_nodes) const;
  std::string to_json() const;
  static DenseSNodeSchema from_json(const std::string &text);
};

}  // namespace taichi::lang
