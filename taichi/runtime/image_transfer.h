#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace taichi::lang {

// Cold image/view facts shared by Program-owned and reconstructed transfers.
// This resource model has one array layer; coordinates use the signed RHI ABI.
void validate_image_transfer_region(const std::array<uint32_t, 3> &size,
                                    uint32_t mip_levels,
                                    const std::array<uint32_t, 3> &offset,
                                    const std::array<uint32_t, 3> &extent,
                                    uint32_t mip_level,
                                    uint32_t base_layer,
                                    uint32_t layer_count,
                                    const char *name);

// End of the finite byte range, relative to the supplied buffer view. Padding
// after the final row/slice is not accessed and is not charged as capacity.
std::size_t required_buffer_image_bytes(std::size_t buffer_offset,
                                        uint32_t buffer_row_length,
                                        uint32_t buffer_image_height,
                                        const std::array<uint32_t, 3> &extent,
                                        std::size_t texel_bytes);

// Offset in the actual VkBuffer, after composing all view/region offsets.
void validate_buffer_image_alignment(std::size_t offset,
                                     std::size_t texel_bytes,
                                     bool depth_aspect);

}  // namespace taichi::lang
