// Codegen for the hierarchical data structure
#pragma once

#include <unordered_map>

#include "taichi/ir/snode.h"

#include "spirv_types.h"

namespace taichi::lang {
namespace spirv {

struct SNodeDescriptor {
  const SNode *snode = nullptr;
  // Stride (bytes) of a single cell.
  size_t cell_stride = 0;

  // Bytes of a single container.
  size_t container_stride = 0;

  // Total number of CELLS of this SNode
  // For example, for a layout of
  // ti.root
  //   .dense(ti.ij, (3, 2))  // S1
  //   .dense(ti.ij, (5, 3))  // S2
  // |total_num_cells_from_root| for S2 is 3x2x5x3 = 90. That is, S2 has a total
  // of 90 cells. Note that the number of S2 (container) itself is 3x2=6!
  size_t total_num_cells_from_root = 0;
  // An SNode can have multiple number of components, where each component
  // starts at a fixed offset in its parent cell's memory.
  size_t mem_offset_in_parent_cell = 0;

  // Phase 2b (vulkan_sparse_experimental, pointer SNode only):
  // For pointer SNodes, the slot array lives at the regular SNode container
  // position in the parent cell (size = 4 * num_cells_per_container, each u32
  // slot stores 0 = inactive, otherwise (pool_index + 1)). The actual child
  // cells live in a per-pointer-SNode pool appended to the end of the root
  // buffer; the bump watermark is a single u32 also appended at the end.
  // These three fields are only meaningful when snode->type == pointer; for
  // all other SNode types they remain zero.
  size_t pointer_pool_offset_in_root = 0;
  size_t pointer_watermark_offset_in_root = 0;
  size_t pointer_pool_capacity = 0;
  // G1.b (vulkan_sparse_experimental, pointer SNode only,
  // gated by TI_VULKAN_POINTER_FREELIST at codegen time):
  //   freelist_head:  u32 sentinel; 0 = empty, otherwise (pool_index + 1).
  //   freelist_links: u32[pool_capacity]; per-pool-slot "next" pointer in
  //                   the same encoding (0 = tail, otherwise pool_index+1).
  // Both are zero-initialized by GfxRuntime::add_root_buffer (every root
  // buffer is memset(0) on construction). Layout is stable across
  // alloc/deactivate; activate either pops freelist (if non-empty) or
  // bumps watermark; deactivate pushes onto freelist.
  size_t pointer_freelist_head_offset_in_root = 0;
  size_t pointer_freelist_links_offset_in_root = 0;

  SNode *get_child(int ch_i) const {
    return snode->ch[ch_i].get();
  }
};

using SNodeDescriptorsMap = std::unordered_map<int, SNodeDescriptor>;

struct CompiledSNodeStructs {
  // Root buffer size in bytes.
  size_t root_size{0};
  // Root SNode
  const SNode *root{nullptr};
  // Map from SNode ID to its descriptor.
  SNodeDescriptorsMap snode_descriptors;

  // TODO: Use the new type compiler
  // tinyir::Block *type_factory;
  // const tinyir::Type *root_type;
};

CompiledSNodeStructs compile_snode_structs(SNode &root);

}  // namespace spirv
}  // namespace taichi::lang
