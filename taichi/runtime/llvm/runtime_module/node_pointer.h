#pragma once

// Specialized Attributes and functions
struct PointerMeta : public StructMeta {
  bool deterministic_slot;   // CS-2: set at codegen time from CompileConfig
};

STRUCT_FIELD(PointerMeta, deterministic_slot);

constexpr u64 pointer_slot_busy = ~0ULL;

inline bool Pointer_compare_exchange_u64(volatile u64 *dest,
                                         u64 expected,
                                         u64 desired) {
  return __atomic_compare_exchange(
      dest, &expected, &desired, false, std::memory_order::memory_order_seq_cst,
      std::memory_order::memory_order_seq_cst);
}

i32 Pointer_get_num_elements(Ptr meta, Ptr node) {
  return ((StructMeta *)meta)->max_num_elements;
}

bool cuda_warp_is_representative(uint32 mask, uint64 value) {
#if defined(ARCH_cuda)
  // If many threads in the mask share the same value, simply
  // elect one thread to return true and let others return false.
  if (cuda_compute_capability() < 70) {
    // <= Pascal
    bool has_following_eqiv = false;
    for (int s = 1; s < 32; s++) {
      auto cond = warp_idx() + s < 32 && ((mask >> (warp_idx() + s)) & 1);
#define TEST_PEER(x) ((x) == cuda_shfl_down_sync_i32(mask, (x), s, 31))
      auto equiv = cond && TEST_PEER(i32(i64(value))) &&
                   TEST_PEER(i32((u64)value >> 32));
#undef TEST_PEER
      has_following_eqiv = has_following_eqiv || equiv;
    }
    return !has_following_eqiv;
  } else {
    // >= Volta > Pascal
    i32 equiv_mask = cuda_match_any_sync_i64(mask, i64(value));
    auto leader = cttz_i32(equiv_mask);
    return warp_idx() == leader;
  }
#else
  return true;
#endif
}

void Pointer_activate(Ptr meta_, Ptr node, int i) {
  auto meta = (StructMeta *)meta_;
  auto num_elements = Pointer_get_num_elements(meta_, node);
  volatile Ptr *data_ptr = (Ptr *)(node + 8 * (num_elements + i));

  // CS-2 (2026-05): deterministic-slot fast path. When the pointer SNode
  // has a single instance in the tree, each child index i ∈ [0,
  // num_cells_per_container) maps to a unique pool slot. The host has
  // pre-allocated and zero-filled a contiguous pool. We compute the target
  // address deterministically and publish it with a single atomicCAS.
  // Duplicate lanes first elect one representative; the cross-warp CAS
  // winner publishes the address and all contenders observe the same value.
  // This keeps the legacy allocator lock and recycle lists out of the path.
  if (((PointerMeta *)meta)->deterministic_slot) {
    // The common topology-stable path only needs a load. On a cold slot,
    // elect one lane per target slot so duplicate writers in the same warp do
    // not all issue the same CAS. Cross-warp publication remains protected by
    // the original strong CAS + busy sentinel protocol.
    u64 published = *(volatile u64 *)data_ptr;
    if (published != 0) {
      while (published == pointer_slot_busy) {
        published = *(volatile u64 *)data_ptr;
      }
      return;
    }
    u32 mask = cuda_active_mask();
    auto rt = meta->context->runtime;
    auto nm = rt->node_allocators[meta->snode_id];
    // CS-2: deterministic pool carved from the TAIL of the dedicated chunk.
    // The head is bumped by ListManager data allocations; the tail is stable.
    // reserve = max_num_elements × element_size bytes at the end.
    uint64_t det_bytes =
        (uint64_t)meta->max_num_elements * meta->element_size;
    Ptr pool_base =
        (Ptr)((uint64_t)nm->dedicated_chunk.preallocated_tail - det_bytes);
    Ptr pool_cell =
        (Ptr)((uint64_t)pool_base + (uint64_t)i * meta->element_size);
    if (cuda_warp_is_representative(mask, (u64)data_ptr)) {
      if (Pointer_compare_exchange_u64((volatile u64 *)data_ptr, 0,
                                       pointer_slot_busy)) {
        std::memset(pool_cell, 0, meta->element_size);
        auto active = atomic_add_i32(&nm->deterministic_active, 1) + 1;
        atomic_max_i32(&nm->deterministic_peak, active);
        grid_memfence();
        atomic_exchange_u64((u64 *)data_ptr, (u64)pool_cell);
        mark_element_lists_dirty_if_reuse(meta);
      } else {
        while (*(volatile u64 *)data_ptr == pointer_slot_busy) {
        }
      }
    }
    warp_barrier(mask);
    return;
  }

  // Legacy path: allocate through NodeManager with lock + warp coordination.
  volatile Ptr lock = node + 8 * i;
  if (*data_ptr == nullptr) {
    // The cuda_ calls will return 0 or do noop on CPUs
    u32 mask = cuda_active_mask();
    if (cuda_warp_is_representative(mask, (u64)lock)) {
      locked_task(
          lock,
          [&] {
            auto rt = meta->context->runtime;
            auto alloc = rt->node_allocators[meta->snode_id];
            auto allocated = (u64)alloc->allocate();
            std::memset((Ptr)allocated, 0, meta->element_size);
            grid_memfence();
            // TODO: Not sure if we really need atomic_exchange here,
            // just to be safe.
            atomic_exchange_u64((u64 *)data_ptr, allocated);
            mark_element_lists_dirty_if_reuse(meta);
          },
          [&]() { return *data_ptr == nullptr; });
    }
    warp_barrier(mask);
  }
}

void Pointer_deactivate(Ptr meta, Ptr node, int i) {
  auto num_elements = Pointer_get_num_elements(meta, node);
  Ptr lock = node + 8 * i;
  Ptr &data_ptr = *(Ptr *)(node + 8 * (num_elements + i));
  // CS-1 (2026-05): deterministic-slot SNodes skip recycle — the GC chain
  // is bypassed entirely. Generic deactivate calls clear this slot here;
  // pure deactivate_all kernels may instead be lowered to Pointer_reset_all.
  if (data_ptr != nullptr) {
    auto smeta = (StructMeta *)meta;
    if (((PointerMeta *)smeta)->deterministic_slot) {
      // Fast path: clear once and maintain current/peak allocation telemetry.
      auto previous = atomic_exchange_u64((u64 *)&data_ptr, 0);
      if (previous != 0 && previous != pointer_slot_busy) {
        auto rt = smeta->context->runtime;
        auto nm = rt->node_allocators[smeta->snode_id];
        atomic_add_i32(&nm->deterministic_active, -1);
        mark_element_lists_dirty_if_reuse(smeta);
      }
      return;
    }
    // Legacy path: lock + recycle for GC
    locked_task(lock, [&] {
      if (data_ptr != nullptr) {
        auto rt = smeta->context->runtime;
        auto alloc = rt->node_allocators[smeta->snode_id];
        alloc->recycle(data_ptr);
        data_ptr = nullptr;
        mark_element_lists_dirty_if_reuse(smeta);
      }
    });
  }
}

// CS-1 (2026-05): bulk reset all pointer slots to null for deterministic-slot
// SNodes. Called from a single kernel launch replacing the 3-stage GC chain.
// Each thread handles a chunk of slots; all threads write null in parallel
// (no locks — the slots are already individually cleared by
// Pointer_deactivate's fast path, and this just ensures a clean sweep).
void Pointer_reset_all(Ptr meta, Ptr node) {
  auto smeta = (StructMeta *)meta;
  auto num_elements = Pointer_get_num_elements(meta, node);
  Ptr *slot_base = (Ptr *)(node + 8 * num_elements);
  if (block_idx() == 0 && thread_idx() == 0) {
    mark_element_lists_dirty_if_reuse(smeta);
    auto rt = smeta->context->runtime;
    rt->node_allocators[smeta->snode_id]->deterministic_active = 0;
  }
  int linear = block_idx() * block_dim() + thread_idx();
  for (int i = linear; i < num_elements; i += block_dim() * grid_dim()) {
    slot_base[i] = nullptr;
  }
}

u1 Pointer_is_active(Ptr meta, Ptr node, int i) {
  auto num_elements = Pointer_get_num_elements(meta, node);
  auto data_ptr = *(Ptr *)(node + 8 * (num_elements + i));
  return data_ptr != nullptr;
}

Ptr Pointer_lookup_element(Ptr meta, Ptr node, int i) {
  auto num_elements = Pointer_get_num_elements(meta, node);
  auto data_ptr = *(Ptr *)(node + 8 * (num_elements + i));
  if (data_ptr == nullptr) {
    auto smeta = (StructMeta *)meta;
    auto context = smeta->context;
    data_ptr = (context->runtime)->ambient_elements[smeta->snode_id];
  }
  return data_ptr;
}
