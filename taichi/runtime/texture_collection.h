#pragma once

#include <cstddef>
#include <cstdint>
#include "taichi/rhi/device.h"

namespace taichi::lang {

// Shared cold facts for immutable sampled-image tables. Callers retain the
// image owners and validate device identity; no Program or submission is owned.
void validate_texture_collection_capacity(Arch arch,
                                          const DeviceCapabilityConfig &caps,
                                          std::size_t capacity);
void validate_texture_collection_member(int dimensions,
                                        ImageDimension dimension,
                                        BufferFormat format);

// One process-monotonic identity space for both JIT and native resource tables.
// Object addresses may be reused after retirement and are not cache identities.
std::uint64_t allocate_texture_collection_snapshot_id();

}  // namespace taichi::lang
