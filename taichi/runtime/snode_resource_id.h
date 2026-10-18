#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace taichi::lang {

// Host resource identity. The tree slot is physical to this runtime; node IDs
// retain their compiler/module-local meaning. Tree retirement must erase keys
// before a physical slot can be reused.
struct SNodeResourceId {
  int tree_id{-1};
  int snode_id{-1};

  bool valid() const noexcept {
    return tree_id >= 0 && snode_id >= 0;
  }
  bool operator==(const SNodeResourceId &other) const noexcept {
    return tree_id == other.tree_id && snode_id == other.snode_id;
  }
  bool operator!=(const SNodeResourceId &other) const noexcept {
    return !(*this == other);
  }
};

struct SNodeResourceIdHash {
  std::size_t operator()(const SNodeResourceId &id) const noexcept {
    return std::hash<std::uint64_t>{}(
        (std::uint64_t(std::uint32_t(id.tree_id)) << 32) |
        std::uint32_t(id.snode_id));
  }
};

using SNodeResourceTreeMap = std::unordered_map<int, std::vector<int>>;

// Cold, per-kernel resolver. Only declared trees and requested node IDs are
// inspected; repeated task references reuse the small local result map.
class SNodeResourceResolver {
 public:
  SNodeResourceResolver(const SNodeResourceTreeMap &trees,
                        const std::vector<int> &used_trees)
      : trees_(trees), used_trees_(used_trees) {
    if (!std::is_sorted(used_trees.begin(), used_trees.end()) ||
        std::adjacent_find(used_trees.begin(), used_trees.end()) !=
            used_trees.end() ||
        (!used_trees.empty() && used_trees.front() < 0)) {
      throw std::invalid_argument("SNode resource tree set must be canonical");
    }
  }

  SNodeResourceId resolve(int snode_id) {
    if (snode_id < 0)
      return {};
    const auto cached = resolved_.find(snode_id);
    if (cached != resolved_.end())
      return cached->second;
    SNodeResourceId result;
    for (int tree_id : used_trees_) {
      const auto tree = trees_.find(tree_id);
      if (tree == trees_.end() ||
          !std::binary_search(tree->second.begin(), tree->second.end(),
                              snode_id)) {
        continue;
      }
      if (result.valid()) {
        throw std::invalid_argument(
            "SNode resource has ambiguous tree ownership");
      }
      result = {tree_id, snode_id};
    }
    if (!result.valid()) {
      throw std::invalid_argument(
          "SNode resource is missing from declared trees");
    }
    resolved_.emplace(snode_id, result);
    return result;
  }

 private:
  const SNodeResourceTreeMap &trees_;
  const std::vector<int> &used_trees_;
  std::unordered_map<int, SNodeResourceId> resolved_;
};

}  // namespace taichi::lang
