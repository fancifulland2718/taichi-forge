// Included after LLVMRuntime and the sparse managers are defined. These cold
// entry points never allocate and never publish a partially rebound tree set.
// Invalid input is reported separately from the source runtime's diagnostics.

void runtime_sparse_view_get_size(uint64 *result) {
  *result = sizeof(LLVMRuntime);
}

void runtime_sparse_view_snapshot(LLVMRuntime *view,
                                  LlvmSparseRuntimeDiagnostic *snapshot) {
  const auto error_code = view->error_code;
  snapshot->code = error_code;
#if ARCH_cuda
  // Avoid copying an unused payload through mapped host memory on every
  // successful CUDA submission.
  const bool copy_payload = error_code != 0;
#else
  // The CPU observer has already retrieved and reset the code before asking
  // for the payload. It only calls this function for a failing submission.
  constexpr bool copy_payload = true;
#endif
  if (copy_payload) {
    std::memcpy(snapshot->message, view->error_message_template,
                sizeof(snapshot->message));
    std::memcpy(snapshot->arguments, view->error_message_arguments,
                sizeof(snapshot->arguments));
  }
#if ARCH_cuda
  // Snapshots may be mapped host storage. Publish all stores to the system
  // before the stream's completion event makes this diagnostic observable.
  asm volatile("membar.sys;");
#endif
  // The stream orders this after all readers/writers of the selected view.
  // Other views and the source runtime have independent diagnostic slots.
  view->error_code = 0;
  view->error_message_lock = 0;
}

void runtime_sparse_view_initialize(LLVMRuntime *source,
                                    LLVMRuntime *view,
                                    LlvmSNodeTreeRuntimeState **directory,
                                    int capacity,
                                    Ptr temporaries,
                                    Ptr result_buffer,
                                    LlvmSparseRuntimeViewStatus *status) {
  *status = {LlvmSparseRuntimeViewError::invalid_view, -1};
  // The caller supplies zeroed, correctly aligned view storage. Reinitializing
  // an active view would strand managers that still refer to that view.
  if (!source || !view || source == view || source->sparse_allocation_source ||
      view->sparse_allocation_source || capacity < 0 ||
      (capacity && !directory) ||
      (directory && directory == source->snode_tree_states) || !temporaries ||
      temporaries == source->temporaries || !result_buffer ||
      result_buffer == source->result_buffer)
    return;
  std::memset(view, 0, sizeof(LLVMRuntime));
  for (int i = 0; i < capacity; ++i)
    directory[i] = nullptr;
  view->host_allocator = source->host_allocator;
  view->assert_failed = source->assert_failed;
  view->host_printf = source->host_printf;
  view->host_vsnprintf = source->host_vsnprintf;
  view->memory_pool = source->memory_pool;
  view->thread_pool = source->thread_pool;
  view->parallel_for = source->parallel_for;
  view->snode_tree_states = directory;
  view->snode_tree_state_capacity = capacity;
  view->temporaries = temporaries;
  view->result_buffer = result_buffer;
  view->sparse_allocation_source = source;
  view->sparse_view_tree_count = -1;
  *status = {};
}

bool sparse_view_managers_match(LlvmSNodeTreeRuntimeState *tree,
                                LLVMRuntime *owner) {
  for (int i = 0; i < tree->node_count; ++i) {
    const auto &node = tree->nodes[i];
    if (node.element_list && node.element_list->runtime != owner)
      return false;
    const auto *manager = node.node_allocator;
    if (manager && (manager->runtime != owner || !manager->free_list ||
                    !manager->recycled_list || !manager->data_list ||
                    manager->free_list->runtime != owner ||
                    manager->recycled_list->runtime != owner ||
                    manager->data_list->runtime != owner))
      return false;
  }
  return true;
}

void sparse_view_rebind_managers(LlvmSNodeTreeRuntimeState *tree,
                                 LLVMRuntime *owner) {
  for (int i = 0; i < tree->node_count; ++i) {
    auto &node = tree->nodes[i];
    if (node.element_list)
      node.element_list->runtime = owner;
    if (auto *manager = node.node_allocator) {
      manager->runtime = owner;
      manager->free_list->runtime = owner;
      manager->recycled_list->runtime = owner;
      manager->data_list->runtime = owner;
    }
  }
}

extern "C++" {
inline int sparse_view_source(const LlvmSparseRuntimeViewTree &tree) {
  return tree.reserved == 0 ? tree.tree_id : -1;
}
inline int sparse_view_source(const LlvmSparseRuntimeViewBinding &tree) {
  return tree.source_tree_id;
}

template <typename Entry>
bool sparse_view_preflight(LLVMRuntime *view,
                           const Entry *trees,
                           int count,
                           bool release,
                           LlvmSparseRuntimeViewStatus *status) {
  *status = {LlvmSparseRuntimeViewError::invalid_view, -1};
  if (!view || !view->sparse_allocation_source ||
      (release ? (view->sparse_view_tree_count != count &&
                  !(count == 0 && view->sparse_view_tree_count == -1))
               : view->sparse_view_tree_count != -1))
    return false;
  auto *source = view->sparse_allocation_source;
  *status = {LlvmSparseRuntimeViewError::invalid_request, -1};
  if (count < 0 || (count && !trees))
    return false;
  int previous = -1;
  for (int i = 0; i < count; ++i) {
    const auto id = trees[i].tree_id;
    const auto source_id = sparse_view_source(trees[i]);
    status->tree_id = id;
    status->error = LlvmSparseRuntimeViewError::invalid_request;
    if (id <= previous || source_id < 0 ||
        id >= view->snode_tree_state_capacity)
      return false;
    if constexpr (std::is_same_v<Entry, LlvmSparseRuntimeViewBinding>) {
      // A manager has one view owner. Do not publish one state in two slots.
      for (int j = 0; j < i; ++j)
        if (sparse_view_source(trees[j]) == source_id)
          return false;
    }
    previous = id;
    status->error = LlvmSparseRuntimeViewError::unavailable_tree;
    if (source_id >= source->snode_tree_state_capacity || !source->snode_tree_states)
      return false;
    auto *tree = source->snode_tree_states[source_id];
    if (!tree || tree->node_count < 0 || (tree->node_count && !tree->nodes))
      return false;
    status->error = LlvmSparseRuntimeViewError::stale_generation;
    if (tree->generation != trees[i].generation ||
        (release && view->snode_tree_states[id] != tree))
      return false;
    status->error = LlvmSparseRuntimeViewError::foreign_manager;
    if ((!release && view->snode_tree_states[id]) ||
        !sparse_view_managers_match(tree, release ? view : source))
      return false;
  }
  *status = {};
  return true;
}

template <typename Entry>
void sparse_view_transfer(LLVMRuntime *view,
                          const Entry *trees,
                          int count,
                          bool release,
                          LlvmSparseRuntimeViewStatus *status) {
  if (!sparse_view_preflight(view, trees, count, release, status))
    return;
  for (int i = 0; i < count; ++i) {
    const int id = trees[i].tree_id;
    auto *source = view->sparse_allocation_source;
    auto *tree = source->snode_tree_states[sparse_view_source(trees[i])];
    sparse_view_rebind_managers(tree, release ? source : view);
    view->snode_tree_states[id] = release ? nullptr : tree;
  }
  if (release)
    view->sparse_allocation_source = nullptr;
  view->sparse_view_tree_count = release ? -1 : count;
}
}  // extern "C++"

void runtime_sparse_view_bind(LLVMRuntime *view,
                              const LlvmSparseRuntimeViewTree *trees,
                              int count,
                              LlvmSparseRuntimeViewStatus *status) {
  sparse_view_transfer(view, trees, count, false, status);
}

void runtime_sparse_view_release(LLVMRuntime *view,
                                 const LlvmSparseRuntimeViewTree *trees,
                                 int count,
                                 LlvmSparseRuntimeViewStatus *status) {
  sparse_view_transfer(view, trees, count, true, status);
}

void runtime_sparse_view_bind_mapped(LLVMRuntime *view,
                                     const LlvmSparseRuntimeViewBinding *trees,
                                     int count,
                                     LlvmSparseRuntimeViewStatus *status) {
  sparse_view_transfer(view, trees, count, false, status);
}

void runtime_sparse_view_release_mapped(LLVMRuntime *view,
                                        const LlvmSparseRuntimeViewBinding *trees,
                                        int count,
                                        LlvmSparseRuntimeViewStatus *status) {
  sparse_view_transfer(view, trees, count, true, status);
}
