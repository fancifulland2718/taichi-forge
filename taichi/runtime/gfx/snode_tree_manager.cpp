#include "taichi/runtime/gfx/snode_tree_manager.h"

#include "taichi/runtime/gfx/runtime.h"
#if defined(TI_WITH_VULKAN_POINTER)
#include "taichi/runtime/gfx/snode_allocator.h"
#endif

namespace taichi::lang {
namespace gfx {

SNodeTreeManager::SNodeTreeManager(GfxRuntime *rtm) : runtime_(rtm) {
}

void SNodeTreeManager::materialize_snode_tree(
    SNodeTree *tree,
    const taichi::lang::spirv::PointerLayoutPolicy &policy) {
  auto *const root = tree->root();
  CompiledSNodeStructs compiled_structs = compile_snode_structs(*root, policy);
  runtime_->update_listgen_buffer_for_snode_tree(compiled_structs);
  const int root_id = tree->id();
  runtime_->add_root_buffer(root_id, compiled_structs.root_size);
  runtime_->register_hash_overflow_checks(root_id, compiled_structs);
#if defined(TI_WITH_VULKAN_POINTER)
  // 路线 B B-1（2026-04-30）：用 contracts 在该 root_buffer 上构造 BumpOnly
  // allocator，与 codegen 端 contract 字节等价。当前 root_buffer 已由
  // add_root_buffer() memset(0)，allocator::clear_all() 暂未被调用。
  std::unordered_map<int, std::unique_ptr<DeviceNodeAllocator>>
      allocators_for_tree;
  if (!compiled_structs.pointer_contracts.empty()) {
    DeviceAllocation root_alloc = *runtime_->root_buffers_[root_id];
    for (const auto &[sid, contract] : compiled_structs.pointer_contracts) {
      BumpOnlyDeviceNodeAllocator::Params p;
      p.device = runtime_->device_;
      p.snode_id = sid;
      p.pool_capacity = contract.pool_capacity;
      p.cell_payload_bytes = contract.cell_stride_bytes;
      p.watermark_offset = contract.watermark_offset;
      p.pool_data_offset = contract.pool_data_offset;
      p.has_freelist = contract.has_freelist;
      p.freelist_head_offset = contract.freelist_head_offset;
      p.freelist_links_offset = contract.freelist_links_offset;
      p.has_ambient_zone = contract.has_ambient_zone;
      p.ambient_offset = contract.ambient_offset;
      p.alloc_protocol = contract.alloc_protocol;
      p.pool_fraction = contract.pool_fraction;
      // C-2.1 (2026-05): allocator_kind / chunk_* 透传到 Params。Bump 路径
      // 这些字段保持默认 0/-1，BumpOnly 不读取，byte-equivalent。
      p.allocator_kind = contract.allocator_kind;
      p.chunk_log2_capacity = contract.chunk_log2_capacity;
      p.max_chunks = contract.max_chunks;
      p.chunk_descriptor_array_first_binding =
          contract.chunk_descriptor_array_first_binding;
      p.root_buffer_alloc = root_alloc;
      // B-3.b (2026-05): contract.pool_buffer_binding_id >= 0 表示 layout 端
      // 已选定该 pointer SNode 走独立 NodeAllocatorPool；此处申请独立
      // DeviceAllocation 并交由 allocator 管理生命周期。size 来自 layout pass
      // 计算的 footprint（覆盖 watermark + freelist + pool_data + ambient）。
      if (contract.pool_buffer_binding_id >= 0) {
        const auto sz_it = compiled_structs.pool_buffer_sizes.find(sid);
        TI_ASSERT_INFO(
            sz_it != compiled_structs.pool_buffer_sizes.end() &&
                sz_it->second > 0,
            "B-3.b: pool_buffer_size missing or zero for pointer SNode {}",
            sid);
        p.use_independent_pool = true;
        p.independent_pool_size = sz_it->second;
      }
      allocators_for_tree.emplace(sid, create_device_node_allocator(p));
    }
  }
  runtime_->node_allocators_[root_id] = std::move(allocators_for_tree);
#endif
  if (static_cast<std::size_t>(root_id) == compiled_snode_structs_.size()) {
    compiled_snode_structs_.push_back(std::move(compiled_structs));
  } else {
    TI_ASSERT(root_id >= 0 &&
              static_cast<std::size_t>(root_id) <
                  compiled_snode_structs_.size());
    TI_ASSERT(compiled_snode_structs_[root_id].root == nullptr);
    compiled_snode_structs_[root_id] = std::move(compiled_structs);
  }
}

void SNodeTreeManager::destroy_snode_tree(SNodeTree *snode_tree) {
  const int root_id = snode_tree->id();
  TI_ERROR_IF(
      root_id < 0 ||
          static_cast<std::size_t>(root_id) >= compiled_snode_structs_.size() ||
          compiled_snode_structs_[root_id].root != snode_tree->root(),
      "the tree to be destroyed cannot be found");
  runtime_->remove_root_buffer(root_id);
#if defined(TI_WITH_VULKAN_POINTER)
  runtime_->node_allocators_.erase(root_id);
#endif
  compiled_snode_structs_[root_id] = {};
}

size_t SNodeTreeManager::get_field_in_tree_offset(int tree_id,
                                                  const SNode *child) {
  auto &snode_struct = compiled_snode_structs_[tree_id];
  TI_ASSERT_INFO(
      snode_struct.snode_descriptors.find(child->id) !=
              snode_struct.snode_descriptors.end() &&
          snode_struct.snode_descriptors.at(child->id).snode == child,
      "Requested SNode not found in compiled SNodeTree");

  size_t offset = 0;
  for (const SNode *sn = child; sn; sn = sn->parent) {
    offset +=
        snode_struct.snode_descriptors.at(sn->id).mem_offset_in_parent_cell;
  }

  return offset;
}

DevicePtr SNodeTreeManager::get_snode_tree_device_ptr(int tree_id) {
  TI_ERROR_IF(tree_id < 0 ||
                  static_cast<std::size_t>(tree_id) >=
                      compiled_snode_structs_.size() ||
                  compiled_snode_structs_[tree_id].root == nullptr,
              "Requested SNodeTree id {} is not active.", tree_id);
  return runtime_->root_buffers_[tree_id]->get_ptr();
}

SparseSNodeTreeMemoryStatistics SNodeTreeManager::get_memory_statistics(
    SNodeTree *snode_tree) const {
  TI_ASSERT(snode_tree != nullptr);
  const int root_id = snode_tree->id();
  TI_ERROR_IF(
      root_id < 0 ||
          static_cast<std::size_t>(root_id) >= compiled_snode_structs_.size() ||
          compiled_snode_structs_[root_id].root != snode_tree->root(),
      "SNodeTree id={} has no live GFX sparse layout.", root_id);

  const auto exact = [](std::uint64_t value) {
    return RuntimeOptionalCounter{value, true};
  };
  const auto &compiled = compiled_snode_structs_[root_id];
  const std::uint64_t root_bytes = runtime_->get_root_buffer_size(root_id);
  std::uint64_t independent_pool_bytes = 0;
  for (const auto &[snode_id, bytes] : compiled.pool_buffer_sizes) {
    const auto contract_it = compiled.pointer_contracts.find(snode_id);
    if (contract_it != compiled.pointer_contracts.end() &&
        contract_it->second.pool_buffer_binding_id >= 0) {
      independent_pool_bytes += bytes;
    }
  }

  SparseSNodeTreeMemoryStatistics result;
  result.root_reserved_bytes = exact(root_bytes);
  result.sparse_pool_reserved_bytes = exact(independent_pool_bytes);
  result.tree_owned_reserved_bytes =
      exact(root_bytes + independent_pool_bytes);
  result.shared_listgen_workspace_reserved_bytes =
      exact(runtime_->listgen_buffer_size_);
  result.tree_owned_scope =
      "exclusive_root_and_independent_pointer_pools";
  result.runtime_resource_scope =
      "tree_owned_static_layout_split_unavailable";
  result.shared_listgen_workspace_scope =
      "program_shared_capacity_not_tree_owned";
  return result;
}

}  // namespace gfx
}  // namespace taichi::lang
