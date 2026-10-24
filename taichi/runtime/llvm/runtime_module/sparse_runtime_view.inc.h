// Included after LLVMRuntime and the sparse managers are defined. These cold
// entry points never allocate and never publish a partially rebound tree set.
// Invalid input is reported separately from the source runtime's diagnostics.

void runtime_sparse_view_get_size(uint64 *result) {
  *result = sizeof(LLVMRuntime);
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

bool sparse_view_preflight(LLVMRuntime *view,
                           const LlvmSparseRuntimeViewTree *trees,
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
    status->tree_id = id;
    status->error = LlvmSparseRuntimeViewError::invalid_request;
    if (id <= previous || trees[i].reserved != 0 ||
        id >= view->snode_tree_state_capacity)
      return false;
    previous = id;
    status->error = LlvmSparseRuntimeViewError::unavailable_tree;
    if (id >= source->snode_tree_state_capacity || !source->snode_tree_states)
      return false;
    auto *tree = source->snode_tree_states[id];
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

void runtime_sparse_view_bind(LLVMRuntime *view,
                              const LlvmSparseRuntimeViewTree *trees,
                              int count,
                              LlvmSparseRuntimeViewStatus *status) {
  if (!sparse_view_preflight(view, trees, count, false, status))
    return;
  for (int i = 0; i < count; ++i) {
    const int id = trees[i].tree_id;
    auto *tree = view->sparse_allocation_source->snode_tree_states[id];
    sparse_view_rebind_managers(tree, view);
    view->snode_tree_states[id] = tree;
  }
  view->sparse_view_tree_count = count;
}

void runtime_sparse_view_release(LLVMRuntime *view,
                                 const LlvmSparseRuntimeViewTree *trees,
                                 int count,
                                 LlvmSparseRuntimeViewStatus *status) {
  if (!sparse_view_preflight(view, trees, count, true, status))
    return;
  for (int i = 0; i < count; ++i) {
    const int id = trees[i].tree_id;
    sparse_view_rebind_managers(view->snode_tree_states[id],
                                view->sparse_allocation_source);
    view->snode_tree_states[id] = nullptr;
  }
  view->sparse_allocation_source = nullptr;
  view->sparse_view_tree_count = -1;
}
