#include "taichi/struct/dense_snode_schema.h"

#include <algorithm>
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

DenseSNodeTensorLayout dense_snode_tensor_layout(
    const std::vector<SNode *> &leaves) {
  require(!leaves.empty() && leaves.front(),
          "Tensor layout requires scalar leaves");
  const auto *anchor = leaves.front();
  DenseSNodeTensorLayout result{
      dense_snode_member_layout(*anchor),
      static_cast<std::uint64_t>(data_type_size(anchor->dt))};
  for (size_t lane = 0; lane < leaves.size(); ++lane) {
    const auto *leaf = leaves[lane];
    require(leaf && leaf->type == SNodeType::place && leaf->dt == anchor->dt &&
                leaf->get_snode_tree_id() == anchor->get_snode_tree_id(),
            "Tensor components require one root and scalar type");
    std::uint64_t delta = 0;
    if (leaf->parent == anchor->parent) {
      // Preserve the packed-field fast path: siblings inherit one index map.
      require(leaf->offset_bytes_in_parent_cell >=
                  anchor->offset_bytes_in_parent_cell,
              "Tensor components have reversed offsets");
      delta = leaf->offset_bytes_in_parent_cell -
              anchor->offset_bytes_in_parent_cell;
    } else {
      const auto layout = dense_snode_member_layout(*leaf);
      require(layout.shape == result.member.shape &&
                  layout.strides == result.member.strides &&
                  layout.offset >= result.member.offset,
              "Tensor components have different index mappings or reversed "
              "offsets");
      delta = layout.offset - result.member.offset;
    }
    if (lane == 1)
      result.lane_stride = delta;
    require(
        result.lane_stride > 0 &&
            result.lane_stride <= std::numeric_limits<int64_t>::max() &&
            lane <= std::numeric_limits<uint64_t>::max() / result.lane_stride &&
            delta == lane * result.lane_stride,
        "Tensor components do not have a positive affine lane stride");
  }
  return result;
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

DenseSNodeSchema DenseSNodeSchema::capture_tensors(
    const SNode &root,
    const std::vector<Parameter> &parameters,
    const std::vector<TensorParameter> &tensors) {
  return capture_tensor_members(root, parameters, tensors).first;
}

std::pair<DenseSNodeSchema, std::map<int, int>>
DenseSNodeSchema::capture_tensor_members(
    const SNode &root,
    const std::vector<Parameter> &parameters,
    const std::vector<TensorParameter> &tensors) {
  return capture_tensor_layout_members(
      root, parameters, tensors, std::vector<bool>(tensors.size(), false));
}

std::pair<DenseSNodeSchema, std::map<int, int>>
DenseSNodeSchema::capture_tensor_layout_members(
    const SNode &root,
    const std::vector<Parameter> &parameters,
    const std::vector<TensorParameter> &tensors,
    const std::vector<bool> &separated) {
  require(separated.size() == tensors.size(),
          "Tensor layout roles must match tensor declarations");
  auto schema = capture(root, parameters);
  std::vector<const SNode *> source_nodes;
  std::map<const SNode *, int> indices;
  std::function<void(const SNode &)> collect = [&](const SNode &node) {
    indices.emplace(&node, static_cast<int>(source_nodes.size()));
    source_nodes.push_back(&node);
    for (const auto &child : node.ch)
      collect(*child);
  };
  collect(root);
  std::map<std::string, int> dimension_values;
  for (const auto &[node, axis, name] : parameters)
    dimension_values.emplace(name, node->extractors[axis].shape);
  std::set<int> claimed, removed;
  std::map<int, int> prototypes;
  size_t tensor_index = 0;
  bool has_separated = false;
  for (const auto &[leaves, shape, names] : tensors) {
    const bool soa = separated[tensor_index++];
    has_separated |= soa;
    require(!leaves.empty() && !shape.empty() && shape.size() <= 2 &&
                names.size() == shape.size(),
            "Tensor schema requires scalar components and vector/matrix axes");
    std::uint64_t lanes = 1;
    std::vector<Extent> extents;
    for (size_t axis = 0; axis < shape.size(); ++axis) {
      require(shape[axis] > 0 &&
                  lanes <= std::numeric_limits<int>::max() / shape[axis],
              "Tensor schema component product exceeds native indexing");
      lanes *= shape[axis];
      Extent extent;
      extent.axis = static_cast<int>(axis);
      extent.parameter = names[axis];
      if (extent.parameter.empty()) {
        extent.constant = shape[axis];
      } else {
        const auto [entry, inserted] =
            dimension_values.emplace(extent.parameter, shape[axis]);
        require(inserted || entry->second == shape[axis],
                "Tensor dimension disagrees with another declared extent");
      }
      extents.push_back(std::move(extent));
    }
    require(lanes == leaves.size(),
            "Tensor schema shape differs from its scalar components");
    const auto first = indices.find(leaves.front());
    require(first != indices.end(),
            "Tensor component is outside the source root");
    const int start = first->second;
    if (soa) {
      // A SOA lane is an entire Dense chain ending in one scalar. Preserve its
      // real common parent and sibling order, and prove each chain identical.
      std::vector<int> prototype;
      const SNode *common_parent = nullptr;
      int previous_end = -1;
      for (const auto *leaf : leaves) {
        require(indices.count(leaf) && leaf->type == SNodeType::place,
                "SOA tensor component must be a scalar in the source root");
        std::vector<int> chain{indices.at(leaf)};
        auto *branch = leaf;
        while (branch->parent && branch->parent->type == SNodeType::dense &&
               branch->parent->ch.size() == 1) {
          branch = branch->parent;
          chain.push_back(indices.at(branch));
        }
        std::reverse(chain.begin(), chain.end());
        require(chain.size() >= 2,
                "SOA tensor lanes require a separate single-leaf Dense branch");
        if (prototype.empty()) {
          prototype = chain;
          common_parent = branch->parent;
        } else {
          require(
              branch->parent == common_parent &&
                  chain.size() == prototype.size() &&
                  chain.front() == previous_end + 1,
              "SOA tensor branches must be adjacent and have the same parent");
        }
        for (size_t depth = 0; depth < chain.size(); ++depth) {
          const auto index = chain[depth];
          const auto &actual = schema.nodes[index];
          const auto &expected = schema.nodes[prototype[depth]];
          require(actual.type == expected.type &&
                      actual.dtype == expected.dtype &&
                      actual.extents.size() == expected.extents.size() &&
                      std::equal(actual.extents.begin(), actual.extents.end(),
                                 expected.extents.begin(),
                                 [](const Extent &a, const Extent &b) {
                                   return a.axis == b.axis &&
                                          a.constant == b.constant &&
                                          a.parameter == b.parameter;
                                 }),
                  "SOA tensor branches have different index or scalar layouts");
          require(claimed.insert(index).second,
                  "Tensor schema components overlap another declaration");
          if (index != prototype[depth])
            removed.insert(index);
          prototypes[index] = prototype[depth];
        }
        previous_end = chain.back();
      }
      schema.nodes[prototype.front()].tensor_extents = std::move(extents);
      continue;
    }
    for (size_t lane = 0; lane < leaves.size(); ++lane) {
      const auto found = indices.find(leaves[lane]);
      require(found != indices.end(),
              "Tensor component is outside the source root");
      const auto *leaf = leaves[lane];
      require(leaf->type == SNodeType::place && leaf->parent &&
                  leaf->parent == leaves.front()->parent &&
                  leaf->dt == leaves.front()->dt &&
                  found->second == start + static_cast<int>(lane),
              "Tensor components must be adjacent same-type scalar siblings");
      require(claimed.insert(found->second).second,
              "Tensor schema components overlap another declaration");
      if (lane)
        removed.insert(found->second);
      prototypes[found->second] = start;
    }
    schema.nodes[start].extents = std::move(extents);
  }
  std::vector<Node> compact;
  std::vector<int> remap(schema.nodes.size(), -1);
  for (size_t i = 0; i < schema.nodes.size(); ++i) {
    if (removed.count(static_cast<int>(i)))
      continue;
    auto node = schema.nodes[i];
    if (node.parent >= 0)
      node.parent = remap[node.parent];
    remap[i] = static_cast<int>(compact.size());
    compact.push_back(std::move(node));
  }
  schema.nodes = std::move(compact);
  schema.version = has_separated ? 3 : tensors.empty() ? 1 : 2;
  schema.validate();
  std::map<int, int> members;
  for (size_t i = 0; i < source_nodes.size(); ++i) {
    const auto prototype = prototypes.find(static_cast<int>(i));
    members.emplace(source_nodes[i]->id,
                    remap[prototype == prototypes.end() ? i : prototype->second]);
  }
  return {std::move(schema), std::move(members)};
}

void DenseSNodeSchema::validate() const {
  require(version >= 1 && version <= 3, "Unsupported Dense schema version");
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
    const bool tensor = version >= 2 && node.type == SNodeType::place;
    require(node.tensor_extents.empty() ||
                (version == 3 && node.type == SNodeType::dense),
            "SOA tensor repetition requires a version 3 Dense branch");
    require(node.tensor_extents.size() <= 2,
            "SOA tensor schema requires vector/matrix axes");
    for (size_t axis = 0; axis < node.tensor_extents.size(); ++axis) {
      const auto &extent = node.tensor_extents[axis];
      require(
          extent.axis == axis &&
              (extent.parameter.empty() ? extent.constant > 0
                                        : extent.constant == 0),
          "SOA tensor axes must be consecutive positive extents or parameters");
    }
    if (!node.tensor_extents.empty()) {
      int parent = static_cast<int>(i);
      size_t next = i + 1;
      while (next < nodes.size() && nodes[next].parent == parent) {
        require(nodes[next].tensor_extents.empty() &&
                    (nodes[next].type != SNodeType::place ||
                     nodes[next].extents.empty()),
                "SOA tensor branch cannot contain another tensor repetition");
        parent = static_cast<int>(next++);
        if (nodes[parent].type == SNodeType::place)
          break;
      }
      require(nodes[parent].type == SNodeType::place &&
                  (next == nodes.size() || nodes[next].parent < i),
              "SOA tensor repetition requires a single-leaf Dense chain");
    }
    require(node.type == SNodeType::dense || tensor || node.extents.empty(),
            "Only dense nodes or version 2 tensor places may declare extents");
    require(!tensor || node.extents.size() <= 2,
            "Tensor schema requires vector/matrix axes");
    int previous_axis = -1;
    for (const auto &extent : node.extents) {
      require(
          extent.axis > previous_axis && extent.axis < taichi_max_num_indices,
          "Dense schema axes must be unique and ordered");
      require(!tensor || extent.axis == previous_axis + 1,
              "Tensor schema axes must be consecutive from zero");
      previous_axis = extent.axis;
      require(
          extent.parameter.empty() ? extent.constant > 0 : extent.constant == 0,
          "Dense schema extent must be either a positive constant or a "
          "parameter");
      if (!tensor) {
        for (int ancestor : ancestors) {
          for (const auto &outer : nodes[ancestor].extents)
            require(outer.axis != extent.axis,
                    "Dense schema repeated axes require non-affine addressing");
        }
      }
    }
    ancestors.push_back(static_cast<int>(i));
  }
}

std::unique_ptr<SNode> DenseSNodeSchema::instantiate(
    const std::map<std::string, int> &dimensions) const {
  std::vector<std::vector<SNode *>> ignored;
  return instantiate_projected(dimensions, ignored);
}

std::unique_ptr<SNode> DenseSNodeSchema::instantiate_projected(
    const std::map<std::string, int> &dimensions,
    std::vector<std::vector<SNode *>> &recipe_nodes) const {
  validate();
  // Validate all bounds before constructing nodes. SNode's index extractors
  // use i32; allowing an overflowing product would corrupt address mapping.
  std::set<std::string> used_dimensions;
  std::vector<std::vector<int>> sizes(nodes.size());
  std::vector<int64_t> path_products(nodes.size(), 1);
  std::vector<int64_t> lane_counts(nodes.size(), 1);
  std::vector<int64_t> branch_counts(nodes.size(), 1);
  std::vector<int64_t> instances(nodes.size(), 1);
  std::uint64_t expanded_nodes = 0;
  const auto extent_value = [&](const Extent &extent) {
    int value = extent.constant;
    if (!extent.parameter.empty()) {
      const auto found = dimensions.find(extent.parameter);
      require(found != dimensions.end(), "Missing Dense schema dimension");
      used_dimensions.insert(extent.parameter);
      value = found->second;
    }
    return value;
  };
  for (size_t i = 0; i < nodes.size(); ++i) {
    const bool tensor = nodes[i].type == SNodeType::place;
    const int64_t inherited =
        nodes[i].parent < 0 ? 1 : path_products[nodes[i].parent];
    int64_t product = tensor ? 1 : inherited;
    for (const auto &extent : nodes[i].extents) {
      const int value = extent_value(extent);
      require(value > 0 && product <= std::numeric_limits<int>::max() / value,
              "Dense schema extent/product exceeds the positive i32 indexing "
              "range");
      product *= value;
      sizes[i].push_back(value);
    }
    path_products[i] = tensor ? inherited : product;
    if (tensor) {
      lane_counts[i] = product;
    }
    for (const auto &extent : nodes[i].tensor_extents) {
      const int value = extent_value(extent);
      require(value > 0 &&
                  branch_counts[i] <= std::numeric_limits<int>::max() / value,
              "SOA tensor component product exceeds native indexing");
      branch_counts[i] *= value;
    }
    const auto parent_instances =
        nodes[i].parent < 0 ? 1 : instances[nodes[i].parent];
    const auto copies = tensor ? lane_counts[i] : branch_counts[i];
    require(parent_instances <= std::numeric_limits<int>::max() / copies,
            "Expanded tensor schema exceeds native node indexing");
    instances[i] = parent_instances * copies;
    expanded_nodes += instances[i];
    require(expanded_nodes <= std::numeric_limits<int>::max(),
            "Expanded tensor schema exceeds native node indexing");
  }
  require(used_dimensions.size() == dimensions.size(),
          "Unknown Dense schema dimension binding");
  auto root = std::make_unique<SNode>(0, SNodeType::root);
  std::vector<std::vector<int>> children(nodes.size());
  for (size_t i = 1; i < nodes.size(); ++i)
    children[nodes[i].parent].push_back(static_cast<int>(i));
  std::vector<std::vector<SNode *>> projection(nodes.size());
  projection[0].push_back(root.get());
  std::function<void(int, SNode *)> construct = [&](int i, SNode *parent) {
    const auto &node = nodes[i];
    if (node.type == SNodeType::dense) {
      std::vector<Axis> axes;
      for (const auto &extent : node.extents)
        axes.emplace_back(extent.axis);
      for (int64_t lane = 0; lane < branch_counts[i]; ++lane) {
        auto *branch = &parent->dense(axes, sizes[i]);
        projection[i].push_back(branch);
        for (int child : children[i])
          construct(child, branch);
      }
    } else {
      for (int64_t lane = 0; lane < lane_counts[i]; ++lane) {
        auto &place = parent->insert_children(SNodeType::place);
        place.dt = node.dtype;
        projection[i].push_back(&place);
      }
    }
  };
  for (int child : children[0])
    construct(child, root.get());
  recipe_nodes = std::move(projection);
  return root;
}

std::string DenseSNodeSchema::to_json() const {
  validate();
  auto data = liong::json::serialize(*this);
  if (version < 3)
    for (auto &node : data.obj.inner.at("nodes").arr.inner)
      node.obj.inner.erase("tensor_extents");
  return liong::json::print(data);
}

DenseSNodeSchema DenseSNodeSchema::from_json(const std::string &text) {
  DenseSNodeSchema schema;
  auto data = liong::json::parse_complete(text);
  if (data["version"].num_int == 1 || data["version"].num_int == 2)
    for (auto &node : data.obj.inner.at("nodes").arr.inner)
      node.obj.inner.emplace("tensor_extents",
                             liong::json::JsonValue(liong::json::JsonArray()));
  liong::json::deserialize(data, schema);
  schema.validate();
  return schema;
}

}  // namespace taichi::lang
