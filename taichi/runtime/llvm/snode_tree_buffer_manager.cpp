#include "snode_tree_buffer_manager.h"
#include "taichi/runtime/llvm/llvm_runtime_executor.h"

namespace taichi::lang {

SNodeTreeBufferManager::SNodeTreeBufferManager(
    LlvmRuntimeExecutor *runtime_exec)
    : runtime_exec_(runtime_exec) {
  TI_TRACE("SNode tree buffer manager created.");
}

Ptr SNodeTreeBufferManager::allocate(std::size_t size,
                                     const int snode_tree_id,
                                     uint64 *result_buffer) {
  auto devalloc = runtime_exec_->allocate_memory_on_device(size, result_buffer);
  snode_tree_id_to_device_alloc_[snode_tree_id] = devalloc;
  snode_tree_id_to_size_[snode_tree_id] = size;
  return (Ptr)runtime_exec_->get_device_alloc_info_ptr(devalloc);
}

void SNodeTreeBufferManager::destroy(SNodeTree *snode_tree) {
  const int tree_id = snode_tree->id();
  auto alloc_it = snode_tree_id_to_device_alloc_.find(tree_id);
  TI_ASSERT(alloc_it != snode_tree_id_to_device_alloc_.end());
  auto devalloc = alloc_it->second;
  runtime_exec_->deallocate_memory_on_device(devalloc);
  snode_tree_id_to_device_alloc_.erase(alloc_it);
  snode_tree_id_to_size_.erase(tree_id);
}

std::size_t SNodeTreeBufferManager::get_size(int snode_tree_id) const {
  auto it = snode_tree_id_to_size_.find(snode_tree_id);
  TI_ERROR_IF(it == snode_tree_id_to_size_.end(),
              "SNodeTree id={} has no live LLVM root allocation.",
              snode_tree_id);
  return it->second;
}

}  // namespace taichi::lang
