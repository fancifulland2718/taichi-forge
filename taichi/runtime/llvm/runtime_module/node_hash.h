#pragma once

struct HashMeta : public StructMeta {
  i64 table_capacity;
  std::size_t state_offset;
  std::size_t key_offset;
  std::size_t payload_offset;
  std::size_t active_count_offset;
  std::size_t overflow_count_offset;
  i32 extract_shape[taichi_max_num_indices];
  i32 extract_acc_shape[taichi_max_num_indices];
};

STRUCT_FIELD(HashMeta, table_capacity);
STRUCT_FIELD(HashMeta, state_offset);
STRUCT_FIELD(HashMeta, key_offset);
STRUCT_FIELD(HashMeta, payload_offset);
STRUCT_FIELD(HashMeta, active_count_offset);
STRUCT_FIELD(HashMeta, overflow_count_offset);
STRUCT_FIELD_ARRAY(HashMeta, extract_shape);
STRUCT_FIELD_ARRAY(HashMeta, extract_acc_shape);

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

inline void Hash_publish_bucket(HashMeta *meta, Ptr node, i64 bucket, i32 key) {
  auto states = Hash_states(meta, node);
  auto keys = Hash_keys(meta, node);
  keys[bucket] = key;
  std::memset(Hash_payload(meta, node, bucket), 0, meta->element_size);
  atomic_exchange_i32(&states[bucket], hash_state_occupied);
  atomic_add_i32(Hash_active_count(meta, node), 1);
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
        Hash_publish_bucket(meta, node, target, key);
        return (i32)target;
      }
      first_tombstone = -1;
      step = -1;
    }
  }
  if (first_tombstone >= 0 &&
      Hash_compare_exchange_i32(&states[first_tombstone],
                                hash_state_tombstone, hash_state_busy)) {
    Hash_publish_bucket(meta, node, first_tombstone, key);
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

inline void Hash_decode_key(HashMeta *meta,
                            i32 key,
                            PhysicalCoordinates *coord) {
  for (int i = 0; i < taichi_max_num_indices; i++) {
    i32 value = 0;
    auto shape = meta->extract_shape[i];
    auto acc_shape = meta->extract_acc_shape[i];
    if (shape > 1) {
      value = (key % (acc_shape * shape)) / acc_shape;
    }
    coord->val[i] = value;
  }
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

  for (i64 bucket = b_start; bucket < child->table_capacity;
       bucket += b_step) {
    if (Hash_load_i32(&states[bucket]) != hash_state_occupied) {
      continue;
    }
    auto key = keys[bucket];
    Element elem;
    elem.element = ch_element;
    elem.loop_bounds[0] = key;
    elem.loop_bounds[1] = key + 1;
    Hash_decode_key(child, key, &elem.pcoord);
    child_list->append(&elem);
  }
  if (child->listgen_reuse) {
    mark_element_list_current(runtime, parent, child_);
  }
}
