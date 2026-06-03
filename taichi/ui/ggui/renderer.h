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
#include <memory>
#include <deque>

#include "taichi/ui/utils/utils.h"
#include "taichi/ui/ggui/vertex.h"
#include "taichi/ui/ggui/scene.h"
#include "taichi/ui/ggui/app_context.h"
#include "taichi/ui/ggui/swap_chain.h"
#include "taichi/ui/ggui/renderable.h"
#include "taichi/ui/common/canvas_base.h"

#include "gui.h"
#include "gui_metal.h"

#include "renderables/set_image.h"
#include "renderables/triangles.h"
#include "renderables/mesh.h"
#include "renderables/particles.h"
#include "renderables/circles.h"
#include "renderables/lines.h"
#include "renderables/scene_lines.h"

#ifdef TI_WITH_METAL
#include "nswindow_adapter.h"
#endif

namespace taichi::lang {
class Program;
}  // namespace taichi::lang

namespace taichi::ui {

namespace vulkan {

class TI_DLL_EXPORT Renderer {
 public:
  void init(lang::Program *prog, TaichiWindow *window, const AppConfig &config);
  ~Renderer();

  void prepare_for_next_frame();
  void wait_for_frame_slot();
  void wait_for_in_flight_frames();
  bool can_accept_frame();
  bool has_render_work() const;
  void discard_pending_frame();

  void set_background_color(const glm::vec3 &color);

  void set_image(const SetImageInfo &info);

  void set_image(const DisplayFrameInfo &info);

  void set_image(taichi::lang::Texture *tex);

  void triangles(const TrianglesInfo &info);

  void circles(const CirclesInfo &info);

  void lines(const LinesInfo &info);

  void mesh(const MeshInfo &info);

  void particles(const ParticlesInfo &info);

  void scene_lines(const SceneLinesInfo &info);

  void scene(SceneBase *scene);

  void scene_v2(SceneBase *scene);

  bool draw_frame(GuiBase *gui, bool blocking_acquire = false);

  const AppContext &app_context() const;
  AppContext &app_context();
  const SwapChain &swap_chain() const;
  SwapChain &swap_chain();

  taichi::lang::StreamSemaphore get_render_complete_semaphore();
  const taichi::lang::SurfaceImage &get_render_surface_image() const;

 private:
  struct InFlightFrame {
    taichi::lang::StreamSemaphore complete;
    // Kept alive until this frame completes because its acquire wait proves the
    // previous present using the same swapchain image has finished.
    std::vector<taichi::lang::StreamSemaphore> present_waits_after_acquire;
    std::vector<std::unique_ptr<Renderable>> renderables;
  };

  void resize_lights_ssbo(int new_ssbo_size);
  void update_scene_data(SceneBase *scene);
  void init_scene_ubo();
  void retire_completed_frames();
  void wait_oldest_frame();
  size_t max_frames_in_flight();
  SetImage *get_set_image_renderable();
  void recycle_renderable_list(std::vector<std::unique_ptr<Renderable>> &list);
  void recycle_renderables(InFlightFrame &frame);

  glm::vec3 background_color_ = glm::vec3(0.f, 0.f, 0.f);

  AppContext app_context_;
  SwapChain swap_chain_;

  std::vector<std::unique_ptr<Renderable>> renderables_;
  std::vector<Renderable *> render_queue_;
  std::deque<InFlightFrame> in_flight_frames_;
  std::vector<std::unique_ptr<SetImage>> reusable_set_images_;
  SetImage *pending_set_image_{nullptr};

  DeviceAllocationUnique lights_ssbo_{nullptr};
  unsigned long long lights_ssbo_size{0};
  DeviceAllocationUnique scene_ubo_{nullptr};

  taichi::lang::StreamSemaphore render_complete_semaphore_{nullptr};
  taichi::lang::SurfaceImage render_surface_image_;

  template <typename T>
  T *get_renderable_of_type(VertexAttributes vbo_attrs);
};

}  // namespace vulkan

}  // namespace taichi::ui
