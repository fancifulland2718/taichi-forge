#pragma once

struct HashMeta : public StructMeta {
  i64 table_capacity;
  std::size_t state_offset;
  std::size_t key_offset;
  std::size_t payload_offset;
  std::size_t active_count_offset;
  std::size_t overflow_count_offset;
  std::size_t active_slots_offset;
  std::size_t active_slots_count_offset;
  std::size_t tombstone_count_offset;
  i32 extract_shape[taichi_max_num_indices];
  i32 extract_acc_shape[taichi_max_num_indices];
};

STRUCT_FIELD(HashMeta, table_capacity);
STRUCT_FIELD(HashMeta, state_offset);
STRUCT_FIELD(HashMeta, key_offset);
STRUCT_FIELD(HashMeta, payload_offset);
STRUCT_FIELD(HashMeta, active_count_offset);
STRUCT_FIELD(HashMeta, overflow_count_offset);
STRUCT_FIELD(HashMeta, active_slots_offset);
STRUCT_FIELD(HashMeta, active_slots_count_offset);
STRUCT_FIELD(HashMeta, tombstone_count_offset);
STRUCT_FIELD_ARRAY(HashMeta, extract_shape);
STRUCT_FIELD_ARRAY(HashMeta, extract_acc_shape);

constexpr std::size_t hash_no_offset = (std::size_t)-1;
constexpr i32 hash_state_empty = 0;
constexpr i32 hash_state_busy = 1;
constexpr i32 hash_state_occupied = 2;
constexpr i32 hash_state_tombstone = 3;

inline i32 *Hash_states(HashMeta *meta, Ptr node) {
  return (i32 *)(node + meta->state_offset);
}

inline i32 *Hash_keys(HashMeta *meta, Ptr node) {
  return (i32 *)(node + meta->key_offset);
}

inline Ptr Hash_payload(HashMeta *meta, Ptr node, i64 bucket) {
  return node + meta->payload_offset + meta->element_size * bucket;
}

inline i32 *Hash_active_count(HashMeta *meta, Ptr node) {
  return (i32 *)(node + meta->active_count_offset);
}

inline i32 *Hash_overflow_count(HashMeta *meta, Ptr node) {
  return (i32 *)(node + meta->overflow_count_offset);
}

inline bool Hash_has_active_slots(HashMeta *meta) {
  return meta->active_slots_offset != hash_no_offset &&
         meta->active_slots_count_offset != hash_no_offset;
}

inline bool Hash_has_tombstone_count(HashMeta *meta) {
  return meta->tombstone_count_offset != hash_no_offset;
}

inline i32 *Hash_active_slots(HashMeta *meta, Ptr node) {
  return (i32 *)(node + meta->active_slots_offset);
}

inline i32 *Hash_active_slots_count(HashMeta *meta, Ptr node) {
  return (i32 *)(node + meta->active_slots_count_offset);
}

inline i32 *Hash_tombstone_count(HashMeta *meta, Ptr node) {
  return (i32 *)(node + meta->tombstone_count_offset);
}

inline u32 Hash_mix_u32(u32 x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

inline i32 Hash_load_i32(volatile i32 *ptr) {
#if ARCH_cuda || ARCH_amdgpu
  return *ptr;
#else
  return __atomic_load_n(ptr, std::memory_order::memory_order_seq_cst);
#endif
}

inline bool Hash_compare_exchange_i32(volatile i32 *ptr,
                                      i32 expected,
                                      i32 desired) {
  return __atomic_compare_exchange(
      ptr, &expected, &desired, false, std::memory_order::memory_order_seq_cst,
      std::memory_order::memory_order_seq_cst);
}

inline i32 Hash_find_bucket(HashMeta *meta, Ptr node, i32 key) {
  auto states = Hash_states(meta, node);
  auto keys = Hash_keys(meta, node);
  auto mask = (u32)(meta->table_capacity - 1);
  auto start = Hash_mix_u32((u32)key) & mask;
  for (i64 step = 0; step < meta->table_capacity; step++) {
    auto bucket = (i64)((start + (u32)step) & mask);
    i32 state = Hash_load_i32(&states[bucket]);
    while (state == hash_state_busy) {
      state = Hash_load_i32(&states[bucket]);
    }
    if (state == hash_state_empty) {
      return -1;
    }
    if (state == hash_state_occupied && keys[bucket] == key) {
      return (i32)bucket;
    }
  }
  return -1;
}

inline void Hash_publish_bucket(HashMeta *meta,
                                Ptr node,
                                i64 bucket,
                                i32 key,
                                bool reused_tombstone) {
  auto states = Hash_states(meta, node);
  auto keys = Hash_keys(meta, node);
  keys[bucket] = key;
  std::memset(Hash_payload(meta, node, bucket), 0, meta->element_size);
  atomic_exchange_i32(&states[bucket], hash_state_occupied);
  atomic_add_i32(Hash_active_count(meta, node), 1);
  if (Hash_has_active_slots(meta) && !reused_tombstone) {
    auto slot = atomic_add_i32(Hash_active_slots_count(meta, node), 1);
    if (slot >= 0 && slot < meta->table_capacity) {
      Hash_active_slots(meta, node)[slot] = (i32)bucket;
    }
  }
  if (reused_tombstone && Hash_has_tombstone_count(meta)) {
    atomic_add_i32(Hash_tombstone_count(meta, node), -1);
  }
  mark_element_lists_dirty_if_reuse(meta);
}

inline i32 Hash_find_or_insert_bucket(HashMeta *meta, Ptr node, i32 key) {
  auto states = Hash_states(meta, node);
  auto keys = Hash_keys(meta, node);
  auto mask = (u32)(meta->table_capacity - 1);
  auto start = Hash_mix_u32((u32)key) & mask;
  i64 first_tombstone = -1;
  for (i64 step = 0; step < meta->table_capacity; step++) {
    auto bucket = (i64)((start + (u32)step) & mask);
    i32 state = Hash_load_i32(&states[bucket]);
    while (state == hash_state_busy) {
      state = Hash_load_i32(&states[bucket]);
    }
    if (state == hash_state_occupied) {
      if (keys[bucket] == key) {
        return (i32)bucket;
      }
      continue;
    }
    if (state == hash_state_tombstone) {
      if (first_tombstone < 0) {
        first_tombstone = bucket;
      }
      continue;
    }
    if (state == hash_state_empty) {
      auto target = first_tombstone >= 0 ? first_tombstone : bucket;
      auto expected =
          first_tombstone >= 0 ? hash_state_tombstone : hash_state_empty;
      if (Hash_compare_exchange_i32(&states[target], expected,
                                    hash_state_busy)) {
        Hash_publish_bucket(meta, node, target, key,
                            first_tombstone >= 0);
        return (i32)target;
      }
      first_tombstone = -1;
      step = -1;
    }
  }
  if (first_tombstone >= 0 &&
      Hash_compare_exchange_i32(&states[first_tombstone],
                                hash_state_tombstone, hash_state_busy)) {
    Hash_publish_bucket(meta, node, first_tombstone, key,
                        /*reused_tombstone=*/true);
    return (i32)first_tombstone;
  }
  atomic_add_i32(Hash_overflow_count(meta, node), 1);
  return -1;
}

i32 Hash_get_num_elements(Ptr meta_, Ptr node) {
  return (i32)((HashMeta *)meta_)->max_num_elements;
}

void Hash_activate(Ptr meta_, Ptr node, int i) {
  auto meta = (HashMeta *)meta_;
  auto bucket = Hash_find_or_insert_bucket(meta, node, i);
  if (bucket < 0) {
    taichi_assert_runtime(meta->context->runtime, false,
                          "Hash SNode table overflow.");
  }
}

void Hash_deactivate(Ptr meta_, Ptr node, int i) {
  auto meta = (HashMeta *)meta_;
  auto bucket = Hash_find_bucket(meta, node, i);
  if (bucket < 0) {
    return;
  }
  auto states = Hash_states(meta, node);
  if (Hash_compare_exchange_i32(&states[bucket], hash_state_occupied,
                                hash_state_busy)) {
    std::memset(Hash_payload(meta, node, bucket), 0, meta->element_size);
    atomic_exchange_i32(&states[bucket], hash_state_tombstone);
    atomic_add_i32(Hash_active_count(meta, node), -1);
    if (Hash_has_tombstone_count(meta)) {
      atomic_add_i32(Hash_tombstone_count(meta, node), 1);
    }
    mark_element_lists_dirty_if_reuse(meta);
  }
}

u1 Hash_is_active(Ptr meta_, Ptr node, int i) {
  auto meta = (HashMeta *)meta_;
  return Hash_find_bucket(meta, node, i) >= 0;
}

Ptr Hash_lookup_element(Ptr meta_, Ptr node, int i) {
  auto meta = (HashMeta *)meta_;
  auto bucket = Hash_find_bucket(meta, node, i);
  if (bucket < 0) {
    return meta->context->runtime->ambient_elements[meta->snode_id];
  }
  return Hash_payload(meta, node, bucket);
}

inline void Hash_refine_key(HashMeta *meta,
                            i32 key,
                            PhysicalCoordinates *coord) {
  for (int i = 0; i < taichi_max_num_indices; i++) {
    auto shape = meta->extract_shape[i];
    if (shape == 1) {
      continue;
    }
    i32 value = 0;
    auto acc_shape = meta->extract_acc_shape[i];
    if (shape > 1) {
      value = (key % (acc_shape * shape)) / acc_shape;
    }
    coord->val[i] = coord->val[i] * shape + value;
  }
}

inline void Hash_append_list_element(HashMeta *child,
                                     Ptr child_container,
                                     i32 key,
                                     const PhysicalCoordinates &parent_coord,
                                     ListManager *child_list) {
  auto elem = (Element *)child_list->allocate();
  elem->element = child_container;
  elem->loop_bounds[0] = key;
  elem->loop_bounds[1] = key + 1;
  elem->pcoord = parent_coord;
  Hash_refine_key(child, key, &elem->pcoord);
}

void element_listgen_root_hash(LLVMRuntime *runtime,
                               StructMeta *parent,
                               StructMeta *child_) {
  if (child_->listgen_reuse &&
      element_list_is_current(runtime, parent, child_)) {
    return;
  }
  auto child = (HashMeta *)child_;
  auto parent_list = runtime->element_lists[parent->snode_id];
  auto child_list = runtime->element_lists[child->snode_id];
  auto parent_lookup_element = parent->lookup_element;
  auto child_from_parent_element = child->from_parent_element;

#if ARCH_cuda || ARCH_amdgpu
  int b_start = block_dim() * block_idx() + thread_idx();
  int b_step = grid_dim() * block_dim();
#else
  int b_start = 0;
  int b_step = 1;
#endif

  auto element = parent_list->get<Element>(0);
  auto ch_element = parent_lookup_element((Ptr)parent, element.element, 0);
  ch_element = child_from_parent_element((Ptr)ch_element);
  auto states = Hash_states(child, ch_element);
  auto keys = Hash_keys(child, ch_element);
  auto scan_count = child->table_capacity;
  bool use_active_slots = false;
  if (Hash_has_active_slots(child) && Hash_has_tombstone_count(child)) {
    auto active_slot_count = Hash_load_i32(Hash_active_slots_count(child, ch_element));
    auto active_count = Hash_load_i32(Hash_active_count(child, ch_element));
    auto tombstone_count = Hash_load_i32(Hash_tombstone_count(child, ch_element));
    if (active_slot_count >= 0 && active_slot_count <= child->table_capacity &&
        active_slot_count == active_count && tombstone_count == 0) {
      scan_count = active_slot_count;
      use_active_slots = true;
    }
  }

  for (i64 scan_i = b_start; scan_i < scan_count; scan_i += b_step) {
    i64 bucket = use_active_slots ? Hash_active_slots(child, ch_element)[scan_i]
                                  : scan_i;
    if (Hash_load_i32(&states[bucket]) != hash_state_occupied) {
      continue;
    }
    auto key = keys[bucket];
    Hash_append_list_element(child, ch_element, key, element.pcoord,
                             child_list);
  }
  if (child->listgen_reuse) {
    mark_element_list_current(runtime, parent, child_);
  }
}

void element_listgen_nonroot_hash(LLVMRuntime *runtime,
                                  StructMeta *parent,
                                  StructMeta *child_) {
  if (child_->listgen_reuse &&
      element_list_is_current(runtime, parent, child_)) {
    return;
  }
  auto child = (HashMeta *)child_;
  auto parent_list = runtime->element_lists[parent->snode_id];
  int num_parent_elements = parent_list->size();
  auto child_list = runtime->element_lists[child->snode_id];
  auto parent_refine_coordinates = parent->refine_coordinates;
  auto parent_is_active = parent->is_active;
  auto parent_lookup_element = parent->lookup_element;
  auto child_from_parent_element = child->from_parent_element;

#if ARCH_cuda || ARCH_amdgpu
  int i_start = block_idx();
  int i_step = grid_dim();
  int j_start = thread_idx();
  int j_step = block_dim();
#else
  int i_start = 0;
  int i_step = 1;
  int j_start = 0;
  int j_step = 1;
#endif

  for (int i = i_start; i < num_parent_elements; i += i_step) {
    auto element = parent_list->get<Element>(i);
    int j_lower = element.loop_bounds[0] + j_start;
    int j_higher = element.loop_bounds[1];
    for (int j = j_lower; j < j_higher; j += j_step) {
      PhysicalCoordinates refined_coord;
      parent_refine_coordinates(&element.pcoord, &refined_coord, j);
      if (!parent_is_active((Ptr)parent, element.element, j)) {
        continue;
      }

      auto ch_element = parent_lookup_element((Ptr)parent, element.element, j);
      ch_element = child_from_parent_element((Ptr)ch_element);
      auto states = Hash_states(child, ch_element);
      auto keys = Hash_keys(child, ch_element);
      auto scan_count = child->table_capacity;
      bool use_active_slots = false;
      if (Hash_has_active_slots(child) && Hash_has_tombstone_count(child)) {
        auto active_slot_count =
            Hash_load_i32(Hash_active_slots_count(child, ch_element));
        auto active_count = Hash_load_i32(Hash_active_count(child, ch_element));
        auto tombstone_count =
            Hash_load_i32(Hash_tombstone_count(child, ch_element));
        if (active_slot_count >= 0 &&
            active_slot_count <= child->table_capacity &&
            active_slot_count == active_count && tombstone_count == 0) {
          scan_count = active_slot_count;
          use_active_slots = true;
        }
      }

      for (i64 scan_i = 0; scan_i < scan_count; scan_i++) {
        i64 bucket = use_active_slots
                         ? Hash_active_slots(child, ch_element)[scan_i]
                         : scan_i;
        if (Hash_load_i32(&states[bucket]) != hash_state_occupied) {
          continue;
        }
        auto key = keys[bucket];
        Hash_append_list_element(child, ch_element, key, refined_coord,
                                 child_list);
      }
    }
  }
  if (child->listgen_reuse) {
    mark_element_list_current(runtime, parent, child_);
  }
}

void element_listgen_nonroot_hash_parent_hash(LLVMRuntime *runtime,
                                              StructMeta *parent,
                                              StructMeta *child_) {
  if (child_->listgen_reuse &&
      element_list_is_current(runtime, parent, child_)) {
    return;
  }
  auto child = (HashMeta *)child_;
  auto parent_list = runtime->element_lists[parent->snode_id];
  int num_parent_elements = parent_list->size();
  auto child_list = runtime->element_lists[child->snode_id];
  auto parent_lookup_element = parent->lookup_element;
  auto child_from_parent_element = child->from_parent_element;

#if ARCH_cuda || ARCH_amdgpu
  int i_start = block_dim() * block_idx() + thread_idx();
  int i_step = grid_dim() * block_dim();
#else
  int i_start = 0;
  int i_step = 1;
#endif

  for (int i = i_start; i < num_parent_elements; i += i_step) {
    auto element = parent_list->get<Element>(i);
    // Parent hash listgen emits singleton ranges for occupied keys.
    int j = element.loop_bounds[0];
    auto ch_element = parent_lookup_element((Ptr)parent, element.element, j);
    ch_element = child_from_parent_element((Ptr)ch_element);
    auto states = Hash_states(child, ch_element);
    auto keys = Hash_keys(child, ch_element);
    auto scan_count = child->table_capacity;
    bool use_active_slots = false;
    if (Hash_has_active_slots(child) && Hash_has_tombstone_count(child)) {
      auto active_slot_count =
          Hash_load_i32(Hash_active_slots_count(child, ch_element));
      auto active_count = Hash_load_i32(Hash_active_count(child, ch_element));
      auto tombstone_count =
          Hash_load_i32(Hash_tombstone_count(child, ch_element));
      if (active_slot_count >= 0 &&
          active_slot_count <= child->table_capacity &&
          active_slot_count == active_count && tombstone_count == 0) {
        scan_count = active_slot_count;
        use_active_slots = true;
      }
    }

    for (i64 scan_i = 0; scan_i < scan_count; scan_i++) {
      i64 bucket = use_active_slots
                       ? Hash_active_slots(child, ch_element)[scan_i]
                       : scan_i;
      if (Hash_load_i32(&states[bucket]) != hash_state_occupied) {
        continue;
      }
      auto key = keys[bucket];
      Hash_append_list_element(child, ch_element, key, element.pcoord,
                               child_list);
    }
  }
  if (child->listgen_reuse) {
    mark_element_list_current(runtime, parent, child_);
  }
}
