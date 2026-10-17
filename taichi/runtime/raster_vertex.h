#pragma once

#include <cstddef>

namespace taichi::lang {
// Packed device vertex storage. Keep this independent of math configuration;
// including a vertex declaration must not choose a camera clip-space
// convention.
struct RasterVertex {
  struct vec3 {
    float x, y, z;
  };
  struct vec4 {
    float x, y, z, w;
  };
  struct vec2 {
    float x, y;
  };
  vec3 pos;
  vec3 normal;
  vec2 tex_coord;
  vec4 color;
};
static_assert(sizeof(RasterVertex) == 48);
static_assert(offsetof(RasterVertex, normal) == 12);
static_assert(offsetof(RasterVertex, tex_coord) == 24);
static_assert(offsetof(RasterVertex, color) == 32);
}  // namespace taichi::lang
