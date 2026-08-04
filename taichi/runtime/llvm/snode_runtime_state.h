#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

// These POD layouts are shared by the host LLVM executor and the embedded
// LLVM runtime module. They deliberately contain only pointer-sized values and
// fixed-width integers so the host can append an exact-sized runtime-state
// block to each SNodeTree root allocation without knowing backend internals.
struct ListManager;
struct NodeManager;

struct LlvmSNodeRuntimeState {
  ListManager *element_list{nullptr};
  NodeManager *node_allocator{nullptr};
  std::uint8_t *ambient_element{nullptr};
  std::int32_t element_list_dirty_epoch{1};
  std::int32_t element_list_dirty_flag{1};
  std::int32_t element_list_version{0};
  std::int32_t element_list_clean_epoch{0};
  std::int32_t element_list_clean_parent_version{0};
};

struct LlvmSNodeTreeRuntimeState {
  std::uint8_t *root{nullptr};
  std::size_t root_mem_size{0};
  LlvmSNodeRuntimeState *nodes{nullptr};
  std::int32_t node_count{0};
  std::int32_t reserved{0};
  std::uint64_t generation{0};
};

constexpr std::size_t llvm_snode_runtime_align_up(std::size_t value,
                                                  std::size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

constexpr std::size_t llvm_snode_tree_runtime_nodes_offset() {
  return llvm_snode_runtime_align_up(sizeof(LlvmSNodeTreeRuntimeState),
                                     alignof(LlvmSNodeRuntimeState));
}

inline bool llvm_snode_tree_runtime_state_bytes(std::size_t node_count,
                                                std::size_t *bytes) {
  constexpr std::size_t offset = llvm_snode_tree_runtime_nodes_offset();
  if (node_count >
      (std::numeric_limits<std::size_t>::max() - offset) /
          sizeof(LlvmSNodeRuntimeState)) {
    return false;
  }
  *bytes = offset + node_count * sizeof(LlvmSNodeRuntimeState);
  return true;
}
