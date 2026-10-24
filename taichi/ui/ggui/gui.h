#pragma once

#include "taichi/ui/utils/utils.h"
#include <optional>

#ifndef IMGUI_IMPL_VULKAN_NO_PROTOTYPES
#define IMGUI_IMPL_VULKAN_NO_PROTOTYPES
#endif

#include <imgui.h>
#ifdef ANDROID
#include <imgui_impl_android.h>
#else
#include <imgui_impl_glfw.h>
#endif
#include <imgui_impl_vulkan.h>
#include "taichi/ui/ggui/app_context.h"
#include "taichi/ui/common/gui_base.h"
#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::ui {

namespace vulkan {

class TI_DLL_EXPORT Gui final : public GuiBase {
 public:
  Gui(AppContext *app_context,
      SwapChain *swap_chain,
      TaichiWindow *window,
      WindowLayoutState *window_layout);
  ~Gui() override;

  struct DrawReservation {
    uint64_t generation;
    uint64_t serial;
    uint32_t slot;
  };

  // Cold preparation is separate from pure command recording.
  void prepare_render_resources(VkRenderPass render_pass, uint32_t image_count);
  void validate_draw_resources(VkRenderPass render_pass, uint32_t image_count);
  std::optional<DrawReservation> reserve_draw_slot(bool blocking);
  bool draw(taichi::lang::CommandList *commands, DrawReservation reservation);
  void validate_recorded_draw(DrawReservation reservation);
  void mark_draw_submitting(DrawReservation reservation) noexcept;
  void seal_draw(DrawReservation reservation,
                 const taichi::lang::StreamSemaphore &complete) noexcept;
  void cancel_recorded_draw(DrawReservation reservation) noexcept;
  void quarantine_draw() noexcept;
  bool retirement_failed() const noexcept;
  void validate_retirement();
  void retire_draw_readers();
  void init_render_resources(VkRenderPass render_pass, uint32_t image_count);
  void cleanup_render_resources();

  void begin(const std::string &name,
             float x,
             float y,
             float width,
             float height) override;
  void begin_auto(const std::string &name,
                  float x,
                  float y,
                  float width) override;
  bool begin_collapsible_section(const std::string &name,
                                 bool default_open) override;
  void end_collapsible_section() override;
  bool begin_edge_region(const std::string &name, WindowEdge edge) override;
  void end_edge_region(WindowEdge edge) override;
  void end() override;
  void text(const std::string &text) override;
  void text(const std::string &text, glm::vec3 color) override;
  bool checkbox(const std::string &name, bool old_value) override;
  int slider_int(const std::string &name,
                 int old_value,
                 int minimum,
                 int maximum) override;
  float slider_float(const std::string &name,
                     float old_value,
                     float minimum,
                     float maximum) override;
  // TODO: consider renaming this?
  glm::vec3 color_edit_3(const std::string &name, glm::vec3 old_value) override;
  bool button(const std::string &text) override;


  void prepare_for_next_frame() override;
  bool has_widgets() const override;
  bool wants_capture_mouse() const override;
  bool wants_capture_keyboard() const override;
  void end_frame() override;

  VkRenderPass render_pass() {
    return render_pass_;
  }

  bool is_empty() const;

 private:
  bool is_empty_{true};
  bool frame_started_{false};
  AppContext *app_context_{nullptr};
  SwapChain *swap_chain_{nullptr};
  WindowLayoutState *window_layout_{nullptr};
  ImGuiContext *imgui_context_{nullptr};
  int widthBeforeDPIScale{0};
  int heightBeforeDPIScale{0};

  VkRenderPass render_pass_{VK_NULL_HANDLE};

  VkDescriptorPool descriptor_pool_{VK_NULL_HANDLE};
  enum class DrawPhase { idle, reserved, recorded, unsealed, in_flight };
  struct DrawSlot {
    taichi::lang::StreamSemaphore complete;
    DrawPhase phase{DrawPhase::idle};
  };
  std::vector<DrawSlot> draw_slots_;
  uint32_t next_frame_slot_{0};
  uint64_t generation_{0};
  uint64_t next_serial_{0};
  std::optional<DrawReservation> active_reservation_;
  bool matches(DrawReservation reservation, DrawPhase phase) const noexcept;
  bool backend_resources_failed_{false};

  void create_descriptor_pool();

  float abs_x(float x);

  float abs_y(float y);

  bool initialized();

  void mark_used();
};

}  // namespace vulkan

}  // namespace taichi::ui
