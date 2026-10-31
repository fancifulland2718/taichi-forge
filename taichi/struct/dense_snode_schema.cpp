#include "taichi/struct/dense_snode_schema.h"

#include <functional>
#include <limits>
#include <set>
#include <stdexcept>

#include "taichi/common/json.h"
#include "taichi/ir/type_utils.h"

namespace taichi::lang {
namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::invalid_argument(message);
}

bool supported_scalar(DataType dtype) {
  return dtype == PrimitiveType::i8 || dtype == PrimitiveType::u8 ||
         dtype == PrimitiveType::i16 || dtype == PrimitiveType::u16 ||
         dtype == PrimitiveType::i32 || dtype == PrimitiveType::u32 ||
         dtype == PrimitiveType::i64 || dtype == PrimitiveType::u64 ||
         dtype == PrimitiveType::f16 || dtype == PrimitiveType::f32 ||
         dtype == PrimitiveType::f64;
}

}  // namespace

DenseSNodeMemberLayout dense_snode_member_layout(const SNode &leaf) {
  require(leaf.type == SNodeType::place && leaf.parent,
          "Dense member requires a placed scalar leaf");
  DenseSNodeMemberLayout layout;
  std::map<int, std::int64_t> strides;
  for (auto *node = &leaf; node->type != SNodeType::root;
       node = node->parent) {
    require(node->parent && !node->is_bit_level && !node->_morton &&
                node->index_offsets.empty() && !node->has_ambient,
            "Dense member requires plain Dense storage");
    const auto limit = std::uint64_t(std::numeric_limits<std::int64_t>::max());
    require(node->offset_bytes_in_parent_cell <= limit - layout.offset,
            "Dense member offset exceeds native address range");
    layout.offset += node->offset_bytes_in_parent_cell;
    if (node == &leaf)
      continue;
    require(node->type == SNodeType::dense,
            "Dense member requires only Dense ancestors");
    for (int axis = 0; axis < taichi_max_num_indices; ++axis) {
      const auto &extractor = node->extractors[axis];
      if (!extractor.active)
        continue;
      require(extractor.acc_shape > 0 &&
                  node->cell_size_bytes <= limit / extractor.acc_shape,
              "Dense member stride exceeds native address range");
      require(strides.emplace(axis, node->cell_size_bytes * extractor.acc_shape)
                  .second,
              "Dense member repeated axes require non-affine addressing");
    }
  }
  for (int i = 0; i < leaf.num_active_indices; ++i) {
    const auto found = strides.find(leaf.physical_index_position[i]);
    require(found != strides.end(), "Dense member axis lacks a layout source");
    layout.shape.push_back(leaf.shape_along_axis(i));
    layout.strides.push_back(found->second);
  }
  return layout;
}

DenseSNodeSchema DenseSNodeSchema::capture(
    const SNode &root,
    const std::vector<Parameter> &parameters) {
  require(root.type == SNodeType::root, "Dense schema requires an SNode root");
  std::map<std::pair<const SNode *, int>, std::string> requested;
  for (const auto &[node, axis, name] : parameters) {
    require(node && axis >= 0 && axis < taichi_max_num_indices &&
                !name.empty(),
            "Invalid Dense schema dimension declaration");
    require(requested.emplace(std::make_pair(node, axis), name).second,
            "Duplicate Dense schema dimension declaration");
  }
  DenseSNodeSchema schema;
  std::map<std::string, int> sample_dimensions;
  std::function<void(const SNode &, int)> visit =
      [&](const SNode &source, int parent) {
        require(!source.is_bit_level && !source._morton &&
                    source.index_offsets.empty() && !source.has_ambient,
                "Dense schema does not support quantized, Morton, offset or ambient layouts");
        require(schema.nodes.size() < std::numeric_limits<int>::max(),
                "Dense schema node count exceeds native indexing");
        const auto index = static_cast<int>(schema.nodes.size());
        Node node;
        node.parent = parent;
        node.type = source.type;
        if (source.type == SNodeType::place)
          node.dtype = source.dt;
        if (source.type == SNodeType::dense) {
          for (int axis = 0; axis < taichi_max_num_indices; ++axis) {
            if (!source.extractors[axis].active)
              continue;
            Extent extent;
            extent.axis = axis;
            const auto found = requested.find({&source, axis});
            if (found == requested.end()) {
              extent.constant = source.extractors[axis].shape;
            } else {
              extent.parameter = found->second;
              const int value = source.extractors[axis].shape;
              const auto [sample, inserted] =
                  sample_dimensions.emplace(extent.parameter, value);
              require(inserted || sample->second == value,
                      "One declared dimension has inconsistent producer extents");
              requested.erase(found);
            }
            node.extents.push_back(std::move(extent));
          }
        }
        schema.nodes.push_back(std::move(node));
        for (const auto &child : source.ch)
          visit(*child, index);
      };
  visit(root, -1);
  require(requested.empty(),
          "Dimension declaration does not identify a dense axis in this root");
  schema.validate();
  return schema;
}

void DenseSNodeSchema::validate() const {
  require(version == 1, "Unsupported Dense schema version");
  // Parent references and the normal SNode construction API use signed ints.
  // Do not impose a smaller deployment-only node-count policy.
  require(!nodes.empty() && nodes.size() <= std::numeric_limits<int>::max(),
          "Dense schema requires a nonempty tree within native node indexing");
  require(nodes[0].type == SNodeType::root && nodes[0].parent == -1,
          "Dense schema must start with one root");
  std::vector<int> ancestors;
  for (size_t i = 0; i < nodes.size(); ++i) {
    const auto &node = nodes[i];
    if (i != 0) {
      require(node.parent >= 0 && node.parent < i,
              "Dense schema parent must precede its child");
      while (!ancestors.empty() && ancestors.back() != node.parent)
        ancestors.pop_back();
      require(!ancestors.empty(),
              "Dense schema nodes must preserve preorder placement order");
      require(node.type == SNodeType::dense || node.type == SNodeType::place,
              "Dense schema supports only dense and scalar place nodes");
      const auto parent_type = nodes[node.parent].type;
      require(parent_type == SNodeType::root || parent_type == SNodeType::dense,
              "Dense schema requires root or Dense parents");
    }
    require(node.type != SNodeType::place || supported_scalar(node.dtype),
            "Dense schema place requires a supported scalar dtype");
    require(node.type == SNodeType::place || node.dtype == PrimitiveType::gen,
            "Dense schema container must not have a scalar dtype");
    require(node.type == SNodeType::dense || node.extents.empty(),
            "Only dense nodes may declare extents");
    int previous_axis = -1;
    for (const auto &extent : node.extents) {
      require(extent.axis > previous_axis && extent.axis < taichi_max_num_indices,
              "Dense schema axes must be unique and ordered");
      previous_axis = extent.axis;
      require(extent.parameter.empty() ? extent.constant > 0
                                       : extent.constant == 0,
              "Dense schema extent must be either a positive constant or a parameter");
      for (int ancestor : ancestors) {
        for (const auto &outer : nodes[ancestor].extents)
          require(outer.axis != extent.axis,
                  "Dense schema repeated axes require non-affine addressing");
      }
    }
    ancestors.push_back(static_cast<int>(i));
  }
}

std::unique_ptr<SNode> DenseSNodeSchema::instantiate(
    const std::map<std::string, int> &dimensions) const {
  validate();
  // Validate all bounds before constructing nodes. SNode's index extractors
  // use i32; allowing an overflowing product would corrupt address mapping.
  std::set<std::string> used_dimensions;
  std::vector<std::vector<int>> sizes(nodes.size());
  std::vector<int64_t> path_products(nodes.size(), 1);
  for (size_t i = 0; i < nodes.size(); ++i) {
    int64_t product = nodes[i].parent < 0 ? 1 : path_products[nodes[i].parent];
    for (const auto &extent : nodes[i].extents) {
      int value = extent.constant;
      if (!extent.parameter.empty()) {
        const auto found = dimensions.find(extent.parameter);
        require(found != dimensions.end(), "Missing Dense schema dimension");
        used_dimensions.insert(extent.parameter);
        value = found->second;
      }
      require(value > 0 && product <= std::numeric_limits<int>::max() / value,
              "Dense schema extent/product exceeds the positive i32 indexing range");
      product *= value;
      sizes[i].push_back(value);
    }
    path_products[i] = product;
  }
  require(used_dimensions.size() == dimensions.size(),
          "Unknown Dense schema dimension binding");
  auto root = std::make_unique<SNode>(0, SNodeType::root);
  std::vector<SNode *> materialized{root.get()};
  for (size_t i = 1; i < nodes.size(); ++i) {
    const auto &node = nodes[i];
    auto *parent = materialized[node.parent];
    if (node.type == SNodeType::dense) {
      std::vector<Axis> axes;
      for (const auto &extent : node.extents)
        axes.emplace_back(extent.axis);
      materialized.push_back(&parent->dense(axes, sizes[i]));
    } else {
      auto &place = parent->insert_children(SNodeType::place);
      place.dt = node.dtype;
      materialized.push_back(&place);
    }
  }
  return root;
}

std::string DenseSNodeSchema::to_json() const {
  validate();
  return liong::json::print(liong::json::serialize(*this));
}

DenseSNodeSchema DenseSNodeSchema::from_json(const std::string &text) {
  DenseSNodeSchema schema;
  liong::json::deserialize(liong::json::parse_complete(text), schema);
  schema.validate();
  return schema;
}

}  // namespace taichi::lang
