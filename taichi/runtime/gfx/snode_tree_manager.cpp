#include "taichi/runtime/gfx/snode_tree_manager.h"

#include "taichi/runtime/gfx/runtime.h"
#include "taichi/runtime/gfx/snode_tree_resources.h"

namespace taichi::lang {
namespace gfx {

SNodeTreeManager::SNodeTreeManager(GfxRuntime *rtm) : runtime_(rtm) {
}

void SNodeTreeManager::materialize_snode_tree(
    SNodeTree *tree,
    const taichi::lang::spirv::PointerLayoutPolicy &policy) {
  auto *const root = tree->root();
  CompiledSNodeStructs compiled_structs = compile_snode_structs(*root, policy);
  const int root_id = tree->id();
  TI_ASSERT(root_id >= 0);
  if (static_cast<std::size_t>(root_id) >= compiled_snode_structs_.size())
    compiled_snode_structs_.resize(static_cast<std::size_t>(root_id) + 1);
  TI_ASSERT(compiled_snode_structs_[root_id].root == nullptr);
  runtime_->materialize_snode_tree_resources(
      root_id, describe_snode_tree_resources(compiled_structs));
  compiled_snode_structs_[root_id] = std::move(compiled_structs);
}

void SNodeTreeManager::destroy_snode_tree(SNodeTree *snode_tree) {
  const int root_id = snode_tree->id();
  TI_ERROR_IF(
      root_id < 0 ||
          static_cast<std::size_t>(root_id) >= compiled_snode_structs_.size() ||
          compiled_snode_structs_[root_id].root != snode_tree->root(),
      "the tree to be destroyed cannot be found");
  runtime_->remove_root_buffer(root_id);
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
