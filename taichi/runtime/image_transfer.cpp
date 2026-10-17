#include "taichi/runtime/image_transfer.h"

#include <algorithm>
#include <limits>
#include "taichi/common/core.h"

namespace taichi::lang {
void validate_image_transfer_region(const std::array<uint32_t, 3> &size,
                                    uint32_t mip_levels,
                                    const std::array<uint32_t, 3> &offset,
                                    const std::array<uint32_t, 3> &extent,
                                    uint32_t mip_level,
                                    uint32_t base_layer,
                                    uint32_t layer_count,
                                    const char *name) {
  // A uint32 extent has at most 32 mip levels; reject before shifting.
  TI_ERROR_IF(!mip_levels || mip_levels > 32 || mip_level >= mip_levels,
              "Vulkan {} mip level is outside the allocated Texture chain.",
              name);
  TI_ERROR_IF(base_layer != 0 || layer_count != 1,
              "Vulkan {} array layers are unsupported by the current Texture "
              "resource model.",
              name);
  for (size_t axis = 0; axis < 3; ++axis) {
    TI_ERROR_IF(!size[axis] || !extent[axis] || offset[axis] > INT32_MAX ||
                    uint64_t(offset[axis]) + extent[axis] >
                        std::max(1u, size[axis] >> mip_level),
                "Vulkan {} region exceeds the texture extent or coordinate "
                "representation.",
                name);
  }
}

std::size_t required_buffer_image_bytes(std::size_t buffer_offset,
                                        uint32_t buffer_row_length,
                                        uint32_t buffer_image_height,
                                        const std::array<uint32_t, 3> &extent,
                                        std::size_t texel_bytes) {
  TI_ERROR_IF(!extent[0] || !extent[1] || !extent[2],
              "Vulkan buffer-image extent must be positive.");
  const size_t row = buffer_row_length ? buffer_row_length : extent[0];
  const size_t height = buffer_image_height ? buffer_image_height : extent[1];
  TI_ERROR_IF(row < extent[0] || height < extent[1],
              "Vulkan buffer-image row length and image height must cover the "
              "copy extent.");
  TI_ERROR_IF(!texel_bytes || buffer_offset % texel_bytes,
              "Vulkan buffer-image offset must be aligned to the image texel "
              "block size.");
  // Vulkan VUID-vkCmdCopyBufferToImage-bufferRowLength-09108 (also
  // image-to-buffer).
  TI_ERROR_IF(row > size_t(INT32_MAX) / texel_bytes,
              "Vulkan buffer-image row pitch exceeds the 2^31-1 byte limit.");
  const auto limit = (std::numeric_limits<size_t>::max)();
  const auto multiply = [limit](size_t a, size_t b) {
    TI_ERROR_IF(b && a > limit / b,
                "Vulkan buffer-image storage size overflow.");
    return a * b;
  };
  const auto add = [limit](size_t a, size_t b) {
    TI_ERROR_IF(a > limit - b, "Vulkan buffer-image byte range overflow.");
    return a + b;
  };
  const size_t slices = multiply(multiply(size_t(extent[2] - 1), height), row);
  const size_t rows = multiply(size_t(extent[1] - 1), row);
  return add(buffer_offset,
             multiply(add(add(slices, rows), extent[0]), texel_bytes));
}
}  // namespace taichi::lang
