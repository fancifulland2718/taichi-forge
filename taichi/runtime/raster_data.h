#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <glm/glm.hpp>

namespace taichi::lang {

// The existing GGUI shader ABI, independent of Window, Program and staging.
// Explicit padding is initialized so frozen payloads never contain host bytes.
struct alignas(16) RasterPointLight {
  glm::vec4 pos{0.0f};
  glm::vec4 color{0.0f};
};

struct RasterSceneData {
  alignas(16) glm::vec3 camera_pos{0.0f};
  uint32_t reserved{0};
  alignas(16) glm::mat4 view{0.0f};
  alignas(16) glm::mat4 projection{0.0f};
  alignas(16) glm::vec3 ambient_light{0.0f};
  int32_t point_light_count{0};
};
struct RasterSceneUniforms {
  RasterSceneData scene;
  float window_width{0}, window_height{0}, tan_half_fov{0}, aspect_ratio{0};
};
struct RasterMeshUniforms {
  alignas(16) glm::vec3 color{0.0f};
  int32_t use_per_vertex_color{0};
  int32_t two_sided{0};
  float has_attribute{0};
  std::array<uint32_t, 2> reserved{};
};
struct RasterParticleUniforms {
  alignas(16) glm::vec3 color{0.0f};
  int32_t use_per_vertex_color{0};
  int32_t use_per_vertex_radius{0};
  float radius{0};
  std::array<uint32_t, 2> reserved{};
};
struct RasterLineUniforms {
  alignas(16) glm::vec3 color{0.0f};
  float line_width{0};
  int32_t per_vertex_color_offset{0};
  int32_t vertex_stride{0};
  int32_t start_vertex{0}, start_index{0}, num_vertices{0}, is_indexed{0};
  std::array<uint32_t, 2> reserved{};
};

static_assert(std::is_standard_layout_v<RasterSceneUniforms>);
static_assert(std::is_trivially_copyable_v<RasterSceneUniforms>);
static_assert(sizeof(RasterPointLight) == 32);
static_assert(sizeof(RasterSceneData) == 160);
static_assert(offsetof(RasterSceneData, view) == 16);
static_assert(offsetof(RasterSceneData, projection) == 80);
static_assert(offsetof(RasterSceneData, ambient_light) == 144);
static_assert(offsetof(RasterSceneData, point_light_count) == 156);
static_assert(sizeof(RasterSceneUniforms) == 176);
static_assert(offsetof(RasterSceneUniforms, window_width) == 160);
static_assert(sizeof(RasterMeshUniforms) == 32);
static_assert(offsetof(RasterMeshUniforms, use_per_vertex_color) == 12);
static_assert(offsetof(RasterMeshUniforms, has_attribute) == 20);
static_assert(sizeof(RasterParticleUniforms) == 32);
static_assert(offsetof(RasterParticleUniforms, radius) == 20);
static_assert(sizeof(RasterLineUniforms) == 48);
static_assert(offsetof(RasterLineUniforms, line_width) == 12);
static_assert(offsetof(RasterLineUniforms, is_indexed) == 36);
}  // namespace taichi::lang
