#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace taichi::lang {

struct VulkanTriangleRaySceneMemoryStatistics {
  std::size_t geometry_input_requested_bytes{0};
  std::size_t acceleration_structure_requested_bytes{0};
  std::size_t build_scratch_requested_bytes{0};
  std::size_t known_requested_bytes{0};
  std::size_t known_allocation_count{0};
};

struct VulkanRayInstanceInfo {
  std::array<float, 12> transform{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  std::uint32_t mask{0xff};
  std::uint32_t custom_index{0};
};

}  // namespace taichi::lang
