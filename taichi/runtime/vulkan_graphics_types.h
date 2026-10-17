#pragma once

#include <array>
#include <cstdint>

namespace taichi::lang {
struct VulkanGraphicsDrawInfo {
  float clear_depth{0.0f};
  std::uint32_t element_count{0};
  std::uint32_t instance_count{1};
  std::uint32_t first_vertex{0};
  std::uint32_t first_index{0};
  std::uint32_t first_instance{0};
  std::int32_t vertex_offset{0};
  std::uint32_t index_min{0};
  std::uint32_t index_max{0};
  bool indexed{false};
  std::array<float, 4> clear_color{0.0f, 0.0f, 0.0f, 1.0f};
  std::array<std::uint32_t, 4> viewport{0, 0, 0, 0};
};

struct VulkanGraphicsMeshDrawInfo {
  std::uint32_t group_count_x{1};
  std::uint32_t group_count_y{1};
  std::uint32_t group_count_z{1};
};
}  // namespace taichi::lang
