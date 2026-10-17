#pragma once

#include "taichi/ui/utils/utils.h"
#include "taichi/runtime/raster_data.h"

namespace taichi::ui {

enum class ProjectionMode : int { Perspective = 0, Orthogonal = 1 };

struct Camera {
  glm::vec3 position;
  glm::vec3 lookat;
  glm::vec3 up;
  ProjectionMode projection_mode = ProjectionMode::Perspective;

  float fov{45};

  float left{-1};
  float right{1};
  float top{-1};
  float bottom{1};
  float z_near{0.1};
  float z_far{1000};

  glm::mat4 get_view_matrix() {
    return glm::lookAt(position, lookat, up);
  }
  glm::mat4 get_projection_matrix(float aspect_ratio) {
    if (projection_mode == ProjectionMode::Perspective) {
      return glm::perspective(glm::radians(fov), aspect_ratio, z_far, z_near);
    } else if (projection_mode == ProjectionMode::Orthogonal) {
      return glm::ortho(left, right, top, bottom, z_far, z_near);
    } else {
      throw std::runtime_error("invalid camera projection mode");
    }
  }

  lang::RasterSceneData raster_scene_data(float aspect_ratio,
                                          glm::vec3 ambient_light,
                                          int32_t point_light_count) {
    return lang::make_raster_scene_data(position, get_view_matrix(),
                                        get_projection_matrix(aspect_ratio),
                                        ambient_light, point_light_count);
  }
};

}  // namespace taichi::ui
