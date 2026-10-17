#include "taichi/runtime/texture_collection.h"

#include <atomic>
#include <limits>
#include "taichi/program/texture.h"

namespace taichi::lang {
namespace {
std::atomic<std::uint64_t> next_snapshot_id{1};
}

void validate_texture_collection_capacity(Arch arch,
                                          const DeviceCapabilityConfig &caps,
                                          std::size_t capacity) {
  TI_ERROR_IF(arch != Arch::vulkan,
              "TextureCollection requires managed Vulkan textures");
  TI_ERROR_IF(!capacity, "TextureCollection requires live Texture members");
  TI_ERROR_IF(
      !caps.get(
          DeviceCapability::spirv_has_sampled_image_array_non_uniform_indexing),
      "TextureCollection requires sampled-image non-uniform indexing");
  TI_ERROR_IF(
      capacity >
              caps.get(DeviceCapability::max_sampled_texture_collection_size) ||
          capacity > (std::numeric_limits<int>::max)(),
      "TextureCollection exceeds the device sampled-image descriptor "
      "limit or native capacity representation");
}

void validate_texture_collection_member(int dimensions,
                                        ImageDimension dimension,
                                        BufferFormat format) {
  const int actual = dimension == ImageDimension::d1D   ? 1
                     : dimension == ImageDimension::d2D ? 2
                     : dimension == ImageDimension::d3D ? 3
                                                        : 0;
  TI_ERROR_IF(!actual || dimensions != actual,
              "TextureCollection members must have the same dimensionality");
  TI_ERROR_IF(!is_float_sampled_texture_format(format),
              "TextureCollection requires floating-point or normalized "
              "sampled formats");
}

std::uint64_t allocate_texture_collection_snapshot_id() {
  auto next = next_snapshot_id.load(std::memory_order_relaxed);
  for (;;) {
    TI_ERROR_IF(
        next == 0 || next == (std::numeric_limits<std::uint64_t>::max)(),
        "TextureCollection snapshot identity space was exhausted");
    if (next_snapshot_id.compare_exchange_weak(next, next + 1,
                                               std::memory_order_relaxed,
                                               std::memory_order_relaxed))
      return next;
  }
}
}  // namespace taichi::lang
