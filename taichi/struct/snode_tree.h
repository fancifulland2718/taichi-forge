#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>

#include "taichi/ir/snode.h"

namespace taichi::lang {

struct SNodeTreeDependency {
  int tree_id{-1};
  std::uint64_t generation{0};

  bool operator==(const SNodeTreeDependency &other) const {
    return tree_id == other.tree_id && generation == other.generation;
  }

  bool operator<(const SNodeTreeDependency &other) const {
    return tree_id < other.tree_id ||
           (tree_id == other.tree_id && generation < other.generation);
  }
};

/**
 * Represents a tree of SNodes.
 *
 * An SNodeTree will be backed by a contiguous chunk of memory.
 */
class SNodeTree {
 public:
  constexpr static int kFirstID = 0;

  /**
   * Constructor.
   *
   * @param id Id of the tree
   * @param root Root of the tree
   */
  explicit SNodeTree(int id,
                     std::uint64_t generation,
                     std::unique_ptr<SNode> root);

  int id() const {
    return id_;
  }

  std::uint64_t generation() const {
    return generation_;
  }

  const SNode *root() const {
    return root_.get();
  }

  SNode *root() {
    return root_.get();
  }

 private:
  int id_{0};
  std::uint64_t generation_{0};
  std::unique_ptr<SNode> root_{nullptr};

  void check_tree_validity(SNode &node);
};

/**
 * Returns the mapping from each SNode under @param root to itself.
 *
 * @param root Root SNode
 * @returns The ID mapping
 */
std::unordered_map<int, int> get_snodes_to_root_id(const SNode &root);

}  // namespace taichi::lang
