#include "taichi/ui/ggui/window.h"
#include "taichi/program/callable.h"

#include <chrono>
#include <cmath>
#include <thread>

#include "taichi/program/program.h"
#include "taichi/ui/utils/utils.h"
#include "taichi/rhi/common/window_system.h"

using taichi::lang::Program;

namespace taichi::ui {

namespace vulkan {

namespace {
using Clock = std::chrono::high_resolution_clock;
constexpr double kUnpacedFpsLimit = 65535.0;

bool should_limit_frame_rate(double fps_limit) {
  return std::isfinite(fps_limit) && fps_limit > 0.0 &&
         fps_limit < kUnpacedFpsLimit;
}

}  // namespace

Window::Window(Program *prog, const AppConfig &config) : WindowBase(config) {
  init(prog, config);
}

void Window::init(Program *prog, const AppConfig &config) {
  if (config_.show_window) {
    glfwSetFramebufferSizeCallback(glfw_window_, framebuffer_resize_callback);
  }

  renderer_ = std::make_unique<Renderer>();
  renderer_->init(prog, glfw_window_, config);
  canvas_ = std::make_unique<Canvas>(renderer_.get());
  switch (config.ggui_arch) {
    case Arch::vulkan:
      gui_ = std::make_unique<Gui>(&renderer_->app_context(),
                                   &renderer_->swap_chain(), glfw_window_);
      break;
#ifdef TI_WITH_METAL
    case Arch::metal:
      gui_ =
          std::make_unique<GuiMetal>(&renderer_->app_context(), glfw_window_);
      break;
#endif
    default:
      TI_NOT_IMPLEMENTED;
  }

  fps_limit_ = config.fps_limit;

  if (config_.show_window) {
    resize();
  }
  prepare_for_next_frame();
}

bool Window::show() {
  if (config_.show_window) {
    WindowBase::show();
    if (!is_running()) {
      return false;
    }
    if (renderer_->swap_chain().device_lost()) {
      return false;
    }
    if (renderer_->swap_chain().needs_swapchain_recreate()) {
      resize();
    }
  }
  if (!drawn_frame_) {
    const bool has_gui_work = gui_ && gui_->has_widgets();
    if (config_.show_window && !renderer_->can_accept_frame()) {
      renderer_->discard_pending_frame();
      record_display_frame_dropped();
      prepare_for_next_frame();
      return false;
    }
    if (config_.show_window && !renderer_->has_render_work() &&
        !has_gui_work) {
      prepare_for_next_frame();
      return false;
    }
    if (!draw_frame()) {
      renderer_->discard_pending_frame();
      record_display_frame_dropped();
      prepare_for_next_frame();
      return false;
    }
  }
  if (!config_.show_window) {
    prepare_for_next_frame();
    record_display_frame_submitted();
    return true;
  }
  present_frame();
  prepare_for_next_frame();
  record_display_frame_submitted();
  return true;
}

void Window::prepare_for_next_frame() {
  renderer_->prepare_for_next_frame();
  if (gui_) {
    gui_->prepare_for_next_frame();
  }
  drawn_frame_ = false;
}

bool Window::can_render_frame() {
  if (!config_.show_window) {
    return true;
  }
  return renderer_ && !drawn_frame_ && !renderer_->has_render_work() &&
         !(gui_ && gui_->has_widgets()) && renderer_->can_accept_frame();
}

CanvasBase *Window::get_canvas() {
  return canvas_.get();
}

SceneBase *Window::get_scene() {
  if (!scene_) {
    scene_ = std::make_unique<SceneV2>(renderer_.get());
  }
  return scene_.get();
}

GuiBase *Window::gui() {
  return gui_.get();
}

void Window::framebuffer_resize_callback(GLFWwindow *glfw_window_,
                                         int width,
                                         int height) {
  auto window =
      reinterpret_cast<Window *>(glfwGetWindowUserPointer(glfw_window_));
  window->resize();
}

void Window::resize() {
  int width = 0, height = 0;
  glfwGetFramebufferSize(glfw_window_, &width, &height);
  while (width == 0 || height == 0) {
    glfwGetFramebufferSize(glfw_window_, &width, &height);
    glfwWaitEvents();
  }
  renderer_->wait_for_in_flight_frames();
  renderer_->app_context().config.width = width;
  renderer_->app_context().config.height = height;

  renderer_->swap_chain().resize(width, height);

  // config_.width and config_.height are used for computing relative mouse
  // positions, so they need to be updated once the window is resized.
  config_.width = width;
  config_.height = height;
}

bool Window::draw_frame(bool blocking_acquire) {
  if (!renderer_->draw_frame(gui_.get(), blocking_acquire)) {
    return false;
  }
  drawn_frame_ = true;
  return true;
}

void Window::present_frame() {
  if (should_limit_frame_rate(fps_limit_)) {
    const double target = 1000.0 / fps_limit_ - limiter_overshoot_;
    const auto time_now = Clock::now();
    const double time_diff =
        std::chrono::duration<double, std::milli>(time_now - last_frame_time_)
            .count();
    if (target > 0.0 && time_diff <= target) {
      std::this_thread::sleep_until(
          last_frame_time_ + std::chrono::duration<double, std::milli>(target));
      const auto after_sleep = Clock::now();
      const double sleep_diff = std::chrono::duration<double, std::milli>(
                                    after_sleep - last_frame_time_)
                                    .count();
      limiter_overshoot_ = sleep_diff - target;
      last_frame_time_ = after_sleep;
    } else {
      last_frame_time_ = time_now;
      limiter_overshoot_ *= 0.9;
    }
  }
  renderer_->swap_chain().surface().present_surface_image(
      renderer_->get_render_surface_image(),
      {renderer_->get_render_complete_semaphore()});
}

Window::~Window() {
  if (renderer_) {
    renderer_->discard_pending_frame();
    renderer_->wait_for_in_flight_frames();
    renderer_->app_context().device().wait_idle();
  }
  gui_.reset();
  renderer_.reset();
}

std::pair<uint32_t, uint32_t> Window::get_window_shape() {
  return {renderer_->swap_chain().width(), renderer_->swap_chain().height()};
}

void Window::write_image(const std::string &filename) {
  if (!drawn_frame_) {
    if (!draw_frame(/*blocking_acquire=*/true)) {
      return;
    }
  }
  renderer_->swap_chain().write_image(filename);
  if (!config_.show_window) {
    prepare_for_next_frame();
  }
}

void Window::copy_depth_buffer_to_ndarray(
    const taichi::lang::Ndarray &depth_arr) {
  if (!drawn_frame_) {
    if (!draw_frame(/*blocking_acquire=*/true)) {
      return;
    }
  }

  if (depth_arr.dtype != taichi::lang::PrimitiveType::f32) {
    TI_ERROR("Data type of depth field must be ti.f32!");
  }
  int w = renderer_->swap_chain().width();
  int h = renderer_->swap_chain().height();

  int size = depth_arr.shape[0];

  if (size != w * h) {
    TI_ERROR("Size of Depth-Ndarray not matched with the window!");
  }

  // We might not have a current program if GGUI is used in external apps to
  // load AOT modules
  Program *prog = renderer_->app_context().prog();

  if (prog) {
    prog->flush();
  }

  // If there is no current program, VBO information should be provided directly
  // instead of accessing through the current SNode
  if (depth_arr.ndarray_alloc_ == taichi::lang::kDeviceNullAllocation) {
    TI_ERROR("Null Allocation for Depth-Ndarray!");
  }

  auto arr_dev_ptr = depth_arr.ndarray_alloc_.get_ptr();
  renderer_->swap_chain().copy_depth_buffer_to_ndarray(arr_dev_ptr);

  if (!config_.show_window) {
    prepare_for_next_frame();
  }
}

std::vector<uint32_t> &Window::get_image_buffer(uint32_t &w, uint32_t &h) {
  if (!drawn_frame_) {
    draw_frame(/*blocking_acquire=*/true);
  }
  w = renderer_->swap_chain().width();
  h = renderer_->swap_chain().height();
  auto &img_buffer = renderer_->swap_chain().dump_image_buffer();
  if (!config_.show_window) {
    prepare_for_next_frame();
  }
  return img_buffer;
}

}  // namespace vulkan

}  // namespace taichi::ui
