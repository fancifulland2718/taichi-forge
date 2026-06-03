#pragma once

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <array>
#include <optional>
#include <set>
#include "taichi/ui/utils/utils.h"
#include "taichi/ui/ggui/vertex.h"

#include "taichi/ui/ggui/app_context.h"
#include "taichi/ui/ggui/swap_chain.h"
#include "taichi/ui/ggui/renderable.h"
#include "taichi/program/field_info.h"
#include "taichi/ui/common/canvas_base.h"
#include "taichi/rhi/device.h"

namespace taichi::ui {

namespace vulkan {

class SetImage;
void erase_direct_set_image_state(const SetImage *set_image);

class SetImage final : public Renderable {
 public:
  struct UniformBufferObject {
    glm::vec2 lower_bound;
    glm::vec2 upper_bound;
    // in non_packed_mode,
    // the actual image is only a corner of the whole image
    float x_factor{1.0};
    float y_factor{1.0};
    int transpose{0};
  };

  struct DirectUniformBufferObject {
    glm::vec2 lower_bound;
    glm::vec2 upper_bound;
    float x_factor{1.0};
    float y_factor{1.0};
    int transpose{0};
    int width{0};
    int height{0};
    int padding0{0};
    int padding1{0};
    int padding2{0};
  };

  SetImage(AppContext *app_context, VertexAttributes vbo_attrs);

  ~SetImage() override {
    erase_direct_set_image_state(this);
  }

  void record_this_frame_commands(
      taichi::lang::CommandList *command_list) final;

  void record_prepass_this_frame_commands(
      taichi::lang::CommandList *command_list) final;

  void update_data(const SetImageInfo &info);

  void update_data(const DisplayFrameInfo &info);

  void update_data(taichi::lang::Texture *tex);

 private:
  int width_{0};
  int height_{0};

  taichi::lang::DeviceImageUnique texture_{nullptr};
  taichi::lang::DeviceAllocationUnique host_staging_{nullptr};
  uint64_t upload_staging_size_{0};
  bool upload_staging_host_write_{false};
  bool upload_staging_export_sharing_{false};
  taichi::lang::DevicePtr pending_upload_buffer_{taichi::lang::kDeviceNullPtr};
  bool pending_upload_{false};

  taichi::lang::BufferFormat format_;

 private:
  void resize_texture(int width, int height, taichi::lang::BufferFormat format);

  void update_ubo(float x_factor, float y_factor, bool transpose);

  void update_direct_buffer_ubo();

  bool can_use_direct_buffer(taichi::lang::DevicePtr ptr) const;

  void use_direct_buffer_pipeline();

  void use_texture_pipeline();

  taichi::lang::DevicePtr upload_host_rgba8(const void *host_ptr,
                                            int width,
                                            int height,
                                            int row_stride_bytes);

  taichi::lang::DevicePtr stage_device_rgba8(taichi::lang::DevicePtr src,
                                             uint64_t size_bytes);

  taichi::lang::DevicePtr ensure_upload_staging(uint64_t size_bytes,
                                                bool host_write,
                                                bool export_sharing);

  void reset_upload_staging();
};

}  // namespace vulkan

}  // namespace taichi::ui
