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

inline HashSNodeFlatLayout compute_hash_snode_flat_layout(
    const SNode &snode,
    std::size_t payload_stride,
    bool include_ambient_payload,
    bool include_active_slots = false,
    bool include_tombstone_count = false) {
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
  layout.payload_offset =
      align_up(layout.key_offset + layout.table_capacity * 4,
               static_cast<std::size_t>(4));
  layout.active_count_offset =
      align_up(layout.payload_offset + payload_stride * layout.table_capacity,
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
