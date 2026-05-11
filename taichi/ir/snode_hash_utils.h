#pragma once

#include <cstddef>
#include <limits>

#include "taichi/common/logging.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/type_utils.h"

namespace taichi::lang {

constexpr int kHashAuxStateIndex = 0;
constexpr int kHashAuxKeyIndex = 1;
constexpr int kHashAuxActiveCountIndex = 2;
constexpr int kHashAuxOverflowCountIndex = 3;
constexpr int kHashAuxActiveSlotsIndex = 4;
constexpr int kHashAuxActiveSlotsCountIndex = 5;
constexpr int kHashAuxTombstoneCountIndexWithActiveList = 6;
constexpr int kHashAuxTombstoneCountIndexNoActiveList = 4;
constexpr std::size_t kHashSNodeNoOffset =
    std::numeric_limits<std::size_t>::max();
constexpr int32 kHashCompactSlotEmpty = 0;
constexpr int32 kHashCompactSlotBusy = -1;

struct HashSNodeFlatLayout {
  std::size_t table_capacity{0};
  std::size_t state_offset{0};
  std::size_t key_offset{0};
  std::size_t payload_offset{0};
  std::size_t active_count_offset{0};
  std::size_t overflow_count_offset{0};
  std::size_t ambient_offset{kHashSNodeNoOffset};
  std::size_t container_stride{0};

  // H4.0: reserved metadata slots for future opt-in extensions. These are not
  // allocated in the default layout yet; future phases must set them explicitly.
  std::size_t active_slots_offset{kHashSNodeNoOffset};
  std::size_t active_slots_count_offset{kHashSNodeNoOffset};
  std::size_t tombstone_count_offset{kHashSNodeNoOffset};
  std::size_t probe_stats_offset{kHashSNodeNoOffset};
  std::size_t compact_child_pool_capacity{0};
  std::size_t compact_child_pool_offset{kHashSNodeNoOffset};
  std::size_t compact_child_pool_next_offset{kHashSNodeNoOffset};
  std::size_t compact_child_pool_overflow_offset{kHashSNodeNoOffset};
  std::size_t compact_child_pool_stride{0};
};

inline int64 get_hash_snode_capacity(const SNode &snode) {
  int64 capacity =
      snode.vk_max_active_hint > 0 ? snode.vk_max_active_hint
                                   : snode.max_num_elements();
  TI_ERROR_IF(capacity <= 0, "Hash SNode capacity must be positive.");
  TI_ERROR_IF((capacity & (capacity - 1)) != 0,
              "Hash SNode capacity must be a power of two, got {}.",
              capacity);
  TI_ERROR_IF(capacity > std::numeric_limits<int32>::max(),
              "Hash SNode capacity {} exceeds the 32-bit index limit.",
              capacity);
  return capacity;
}

inline int64 get_hash_snode_expected_active(const SNode &snode) {
  if (snode.hash_expected_active_hint > 0) {
    return snode.hash_expected_active_hint;
  }
  return get_hash_snode_capacity(snode);
}

inline bool hash_snode_uses_compact_child_pool(const SNode &snode,
                                               bool enabled) {
  return enabled && snode.type == SNodeType::hash && snode.ch.size() == 1 &&
         !snode.ch[0]->is_bit_level &&
         snode.ch[0]->type == SNodeType::hash &&
         get_hash_snode_expected_active(snode) < get_hash_snode_capacity(snode);
}

inline HashSNodeFlatLayout compute_hash_snode_flat_layout(
    const SNode &snode,
    std::size_t payload_stride,
    bool include_ambient_payload,
    bool include_active_slots = false,
    bool include_tombstone_count = false,
    bool include_compact_child_pool = false) {
  TI_ERROR_IF(payload_stride == 0 || payload_stride % 4 != 0,
              "Hash SNode requires a positive 4-byte aligned payload cell "
              "size, got {} bytes.",
              payload_stride);

  HashSNodeFlatLayout layout;
  layout.table_capacity =
      static_cast<std::size_t>(get_hash_snode_capacity(snode));
  layout.state_offset = 0;
  layout.key_offset = align_up(layout.state_offset + layout.table_capacity * 4,
                               static_cast<std::size_t>(4));
  const std::size_t payload_unit_stride =
      include_compact_child_pool ? static_cast<std::size_t>(4)
                                 : payload_stride;
  layout.payload_offset =
      align_up(layout.key_offset + layout.table_capacity * 4,
               static_cast<std::size_t>(4));
  layout.active_count_offset =
      align_up(layout.payload_offset +
                   payload_unit_stride * layout.table_capacity,
               static_cast<std::size_t>(4));
  layout.overflow_count_offset = layout.active_count_offset + 4;

  std::size_t cursor = layout.overflow_count_offset + 4;
  if (include_active_slots) {
    layout.active_slots_offset =
        align_up(cursor, static_cast<std::size_t>(4));
    cursor = layout.active_slots_offset + layout.table_capacity * 4;
    layout.active_slots_count_offset =
        align_up(cursor, static_cast<std::size_t>(4));
    cursor = layout.active_slots_count_offset + 4;
  }
  if (include_tombstone_count) {
    layout.tombstone_count_offset =
        align_up(cursor, static_cast<std::size_t>(4));
    cursor = layout.tombstone_count_offset + 4;
  }
  if (include_compact_child_pool) {
    layout.compact_child_pool_capacity =
        static_cast<std::size_t>(get_hash_snode_expected_active(snode));
    TI_ERROR_IF(layout.compact_child_pool_capacity == 0,
                "Hash compact child pool capacity must be positive.");
    layout.compact_child_pool_next_offset =
        align_up(cursor, static_cast<std::size_t>(4));
    cursor = layout.compact_child_pool_next_offset + 4;
    layout.compact_child_pool_overflow_offset =
        align_up(cursor, static_cast<std::size_t>(4));
    cursor = layout.compact_child_pool_overflow_offset + 4;
    layout.compact_child_pool_offset =
        align_up(cursor, static_cast<std::size_t>(4));
    layout.compact_child_pool_stride = payload_stride;
    cursor = layout.compact_child_pool_offset +
             layout.compact_child_pool_stride *
                 layout.compact_child_pool_capacity;
  }

  if (include_ambient_payload) {
    layout.ambient_offset = align_up(cursor, static_cast<std::size_t>(4));
    layout.container_stride =
        align_up(layout.ambient_offset + payload_stride,
                 static_cast<std::size_t>(4));
  } else {
    layout.container_stride = align_up(cursor, static_cast<std::size_t>(4));
  }
  return layout;
}

}  // namespace taichi::lang
