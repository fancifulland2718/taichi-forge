#include "gui_metal.h"
#include "taichi/ui/ggui/app_context.h"
#include "taichi/ui/ggui/edge_layout_imgui.h"
#include "taichi/ui/ggui/swap_chain.h"
#include <imgui_impl_metal.h>

using namespace taichi::lang::metal;
using namespace taichi::lang;

namespace taichi::ui {

namespace vulkan {

namespace {

float default_font_size(const ImGuiIO &io) {
  if (io.FontDefault != nullptr) {
    return io.FontDefault->FontSize * io.FontDefault->Scale;
  }
  if (io.Fonts != nullptr && io.Fonts->Fonts.Size > 0 &&
      io.Fonts->Fonts[0] != nullptr) {
    return io.Fonts->Fonts[0]->FontSize * io.Fonts->Fonts[0]->Scale;
  }
  return 13.0f;
}

bool pointer_over_edge_region(const ImGuiIO &io,
                              const WindowLayoutState *layout) {
  if (layout == nullptr) {
    return false;
  }
  for (std::size_t i = 0; i < kWindowEdgeCount; ++i) {
    const auto edge = static_cast<WindowEdge>(i);
    const auto &config = layout->region(edge);
    if (config.enabled && !config.collapsed &&
        layout->snapshot().edge_regions[i].contains(io.MousePos.x,
                                                    io.MousePos.y)) {
      return true;
    }
  }
  return false;
}

} // namespace

GuiMetal::GuiMetal(AppContext *app_context,
                   TaichiWindow *window,
                   WindowLayoutState *window_layout) {
  app_context_ = app_context;
  window_layout_ = window_layout;

  IMGUI_CHECKVERSION();
  imgui_context_ = ImGui::CreateContext();
  [[maybe_unused]] ImGuiIO &io = ImGui::GetIO();

  ImGui::StyleColorsDark();

  if (app_context->config.show_window) {
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    glfwGetWindowSize(window, &widthBeforeDPIScale, &heightBeforeDPIScale);
  } else {
    widthBeforeDPIScale = app_context->config.width;
    heightBeforeDPIScale = app_context->config.height;
  }
  auto &device =
      static_cast<taichi::lang::metal::MetalDevice &>(app_context_->device());

  ImGui_ImplMetal_Init(device.mtl_device());
}

void GuiMetal::init_render_resources(void *rpd) {
  current_rpd_ = (__bridge MTLRenderPassDescriptor *)rpd;
}

void GuiMetal::prepare_for_next_frame() {
  ImGui::SetCurrentContext(imgui_context_);
  end_frame();
  if (app_context_->config.show_window) {
    ImGui_ImplGlfw_NewFrame();
  } else {
    // io.DisplaySize is set during ImGui_ImplGlfw_NewFrame()
    // but since we're headless, we do it explicitly here
    auto w = app_context_->config.width;
    auto h = app_context_->config.height;
    ImGuiIO &io = ImGui::GetIO();
    io.DisplaySize = ImVec2((float)w, (float)h);
  }
  ImGuiIO &io = ImGui::GetIO();
  io.FontGlobalScale =
      update_font_scale(io.DisplaySize.y, default_font_size(io));
  if (window_layout_ != nullptr) {
    window_layout_->apply_pending_updates();
    window_layout_->update_dimensions(
        io.DisplaySize.x, io.DisplaySize.y, app_context_->config.width,
        app_context_->config.height);
  }
  if (io.DisplaySize.x > 0.0f && io.DisplaySize.y > 0.0f) {
    widthBeforeDPIScale = static_cast<int>(io.DisplaySize.x);
    heightBeforeDPIScale = static_cast<int>(io.DisplaySize.y);
  }
  ImGui::NewFrame();
  const bool pointer_over_edge = pointer_over_edge_region(io, window_layout_);
  if (font_shortcuts_enabled() && io.KeyCtrl) {
    if (pointer_over_edge && io.MouseWheel != 0.0f) {
      adjust_font_zoom(io.MouseWheel * 0.1f);
    }
    if (pointer_over_edge || io.WantCaptureKeyboard) {
      if (ImGui::IsKeyPressed(ImGuiKey_Equal)) {
        adjust_font_zoom(0.1f);
      }
      if (ImGui::IsKeyPressed(ImGuiKey_Minus)) {
        adjust_font_zoom(-0.1f);
      }
      if (ImGui::IsKeyPressed(ImGuiKey_0)) {
        reset_font_zoom();
      }
    }
  }
  frame_started_ = true;
  is_empty_ = true;
}

float GuiMetal::abs_x(float x) { return x * widthBeforeDPIScale; }
float GuiMetal::abs_y(float y) { return y * heightBeforeDPIScale; }

void GuiMetal::begin(const std::string &name, float x, float y, float width,
                     float height) {
  ImGui::SetNextWindowPos(ImVec2(abs_x(x), abs_y(y)), ImGuiCond_Once);
  ImGui::SetNextWindowSize(ImVec2(abs_x(width), abs_y(height)), ImGuiCond_Once);
  ImGui::Begin(name.c_str());
  is_empty_ = false;
}
void GuiMetal::begin_auto(const std::string &name, float x, float y,
                          float width) {
  ImGui::SetNextWindowPos(ImVec2(abs_x(x), abs_y(y)), ImGuiCond_Once);
  const float fixed_width = abs_x(width);
  ImGui::SetNextWindowSizeConstraints(ImVec2(fixed_width, 0.0f),
                                      ImVec2(fixed_width, FLT_MAX));
  ImGui::Begin(name.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize);
  is_empty_ = false;
}
bool GuiMetal::begin_collapsible_section(const std::string &name,
                                         bool default_open) {
  const ImGuiTreeNodeFlags flags =
      default_open ? ImGuiTreeNodeFlags_DefaultOpen : ImGuiTreeNodeFlags_None;
  const bool expanded = ImGui::CollapsingHeader(name.c_str(), flags);
  if (expanded) {
    ImGui::PushID(name.c_str());
    ImGui::Indent();
  }
  return expanded;
}
void GuiMetal::end_collapsible_section() {
  ImGui::Unindent();
  ImGui::PopID();
}
bool GuiMetal::begin_edge_region(const std::string &name, WindowEdge edge) {
  is_empty_ = false;
  return taichi::ui::vulkan::begin_edge_region(window_layout_, name, edge);
}
void GuiMetal::end_edge_region(WindowEdge edge) {
  taichi::ui::vulkan::end_edge_region(window_layout_, edge);
}
void GuiMetal::end() { ImGui::End(); }
void GuiMetal::text(const std::string &text) {
  ImGui::Text("%s", text.c_str());
}
void GuiMetal::text(const std::string &text, glm::vec3 color) {
  ImGui::TextColored(ImVec4(color[0], color[1], color[2], 1.0f), "%s",
                     text.c_str());
}
bool GuiMetal::checkbox(const std::string &name, bool old_value) {
  ImGui::Checkbox(name.c_str(), &old_value);
  return old_value;
}
int GuiMetal::slider_int(const std::string &name, int old_value, int minimum,
                         int maximum) {
  ImGui::SliderInt(name.c_str(), &old_value, minimum, maximum);
  return old_value;
}
float GuiMetal::slider_float(const std::string &name, float old_value,
                             float minimum, float maximum) {
  ImGui::SliderFloat(name.c_str(), &old_value, minimum, maximum);
  return old_value;
}
glm::vec3 GuiMetal::color_edit_3(const std::string &name, glm::vec3 old_value) {
  ImGui::ColorEdit3(name.c_str(), (float *)&old_value);
  return old_value;
}
bool GuiMetal::button(const std::string &text) {
  return ImGui::Button(text.c_str());
}

void GuiMetal::draw(taichi::lang::CommandList *cmd_list) {
  if (!frame_started_) {
    return;
  }
  ImGui_ImplMetal_NewFrame(current_rpd_);

  // Rendering
  ImGui::Render();
  frame_started_ = false;

  @autoreleasepool {
    MTLCommandBuffer_id buffer =
        static_cast<MetalCommandList *>(cmd_list)->finalize();

    MTLRenderCommandEncoder_id rce =
        [buffer renderCommandEncoderWithDescriptor:current_rpd_];
    ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), buffer, rce);
    [rce endEncoding];
  }
}
void GuiMetal::cleanup_render_resources() {
  end_frame();
  ImGui_ImplMetal_Shutdown();
  current_rpd_ = nullptr;
}

GuiMetal::~GuiMetal() {
  if (app_context_->config.show_window) {
    ImGui_ImplGlfw_Shutdown();
  }
  cleanup_render_resources();
  ImGui::DestroyContext(imgui_context_);
}

bool GuiMetal::has_widgets() const { return !is_empty_; }

bool GuiMetal::wants_capture_mouse() const {
  ImGui::SetCurrentContext(imgui_context_);
  return ImGui::GetIO().WantCaptureMouse;
}

bool GuiMetal::wants_capture_keyboard() const {
  ImGui::SetCurrentContext(imgui_context_);
  return ImGui::GetIO().WantCaptureKeyboard;
}

void GuiMetal::end_frame() {
  ImGui::SetCurrentContext(imgui_context_);
  if (!frame_started_) {
    return;
  }
  ImGui::EndFrame();
  frame_started_ = false;
}

bool GuiMetal::is_empty() const { return is_empty_; }

} // namespace vulkan

} // namespace taichi::ui
