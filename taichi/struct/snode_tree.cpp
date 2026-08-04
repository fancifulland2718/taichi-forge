#include "taichi/struct/snode_tree.h"

namespace taichi::lang {
namespace {

constexpr std::uint64_t kFnvOffsetBasis = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

void fingerprint_bytes(std::uint64_t &fingerprint,
                       const void *data,
                       std::size_t size) {
  const auto *bytes = static_cast<const unsigned char *>(data);
  for (std::size_t i = 0; i < size; ++i) {
    fingerprint ^= bytes[i];
    fingerprint *= kFnvPrime;
  }
}

template <typename T>
void fingerprint_integer(std::uint64_t &fingerprint, T value) {
  const std::uint64_t bits = static_cast<std::uint64_t>(value);
  for (std::size_t i = 0; i < sizeof(bits); ++i) {
    const unsigned char byte =
        static_cast<unsigned char>((bits >> (i * 8)) & 0xffu);
    fingerprint_bytes(fingerprint, &byte, 1);
  }
}

void fingerprint_string(std::uint64_t &fingerprint,
                        const std::string &value) {
  fingerprint_integer(fingerprint, value.size());
  fingerprint_bytes(fingerprint, value.data(), value.size());
}

void fingerprint_snode(std::uint64_t &fingerprint, const SNode &node) {
  fingerprint_integer(fingerprint, node.type);
  fingerprint_integer(fingerprint, node.ch.size());
  fingerprint_integer(fingerprint, node.num_active_indices);
  for (int i = 0; i < taichi_max_num_indices; ++i) {
    const auto &extractor = node.extractors[i];
    fingerprint_integer(fingerprint, extractor.active);
    if (extractor.active) {
      fingerprint_integer(fingerprint, extractor.shape);
      fingerprint_integer(fingerprint, extractor.acc_shape);
      fingerprint_integer(fingerprint, extractor.num_elements_from_root);
    }
  }
  for (int i = 0; i < node.num_active_indices; ++i) {
    fingerprint_integer(fingerprint, node.physical_index_position[i]);
  }
  fingerprint_integer(fingerprint, node.index_offsets.size());
  for (int offset : node.index_offsets) {
    fingerprint_integer(fingerprint, offset);
  }
  fingerprint_integer(fingerprint, node.num_cells_per_container);
  fingerprint_integer(fingerprint, node.vk_max_active_hint);
  fingerprint_integer(fingerprint, node.hash_expected_active_hint);
  fingerprint_integer(fingerprint, node.chunk_size);
  fingerprint_integer(fingerprint, node.cell_size_bytes);
  fingerprint_integer(fingerprint, node.offset_bytes_in_parent_cell);
  fingerprint_string(fingerprint, node.dt.to_string());
  fingerprint_integer(fingerprint, node.has_ambient);
  fingerprint_integer(fingerprint, node.id_in_bit_struct);
  fingerprint_integer(fingerprint, node.is_bit_level);
  fingerprint_integer(fingerprint, node.is_path_all_dense);
  fingerprint_integer(fingerprint, node._morton);
  for (const auto &child : node.ch) {
    fingerprint_snode(fingerprint, *child);
  }
}

void get_snodes_to_root_id_impl(const SNode &node,
                                const int root_id,
                                std::unordered_map<int, int> *map) {
  (*map)[node.id] = root_id;
  for (auto &ch : node.ch) {
    get_snodes_to_root_id_impl(*ch, root_id, map);
  }
}

}  // namespace

std::uint64_t snode_tree_layout_fingerprint(const SNodeTree &tree) {
  return tree.layout_fingerprint();
}

void SNodeTree::refresh_layout_fingerprint() {
  std::uint64_t fingerprint = kFnvOffsetBasis;
  fingerprint_snode(fingerprint, *root());
  layout_fingerprint_ = fingerprint;
}

SNodeTree::SNodeTree(int id,
                     std::uint64_t generation,
                     std::unique_ptr<SNode> root)
    : id_(id), generation_(generation), root_(std::move(root)) {
  TI_ASSERT(generation_ != 0);
  check_tree_validity(*root_);
  assign_runtime_local_ids(*root_);
}

void SNodeTree::assign_runtime_local_ids(SNode &node) {
  TI_ASSERT(node.runtime_local_id < 0);
  node.runtime_local_id = num_snodes_++;
  for (auto &ch : node.ch) {
    assign_runtime_local_ids(*ch);
  }
}

void SNodeTree::check_tree_validity(SNode &node) {
  if (node.ch.empty()) {
    if (node.type != SNodeType::place && node.type != SNodeType::root) {
      TI_ERROR("{} node must have at least one child.",
               snode_type_name(node.type));
    }
  }
  for (auto &ch : node.ch) {
    check_tree_validity(*ch);
  }
}

std::unordered_map<int, int> get_snodes_to_root_id(const SNode &root) {
  // TODO: Consider generalizing this SNode visiting method
  std::unordered_map<int, int> res;
  get_snodes_to_root_id_impl(root, root.id, &res);
  return res;
}

}  // namespace taichi::lang
