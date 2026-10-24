#include "gui.h"
#include <cassert>
#include "taichi/ui/ggui/imgui_vulkan_forge.h"
#include "taichi/ui/ggui/swap_chain.h"
#include "taichi/ui/ggui/app_context.h"
#include "taichi/ui/ggui/edge_layout_imgui.h"
#ifndef ANDROID
#include "taichi/ui/common/glfw_input.h"
#endif

using namespace taichi::lang::vulkan;
using namespace taichi::lang;

namespace taichi::ui {

namespace vulkan {

namespace {

thread_local VulkanDevice *gui_vulkan_device = nullptr;

// ImGui's result callback has no user data. Bind it only while invoking the
// backend, restoring both callback owner and ImGui context on exception.
class GuiVulkanCall final {
 public:
  GuiVulkanCall(ImGuiContext *context, VulkanDevice &device)
      : previous_context_(ImGui::GetCurrentContext()),
        previous_device_(gui_vulkan_device) {
    ImGui::SetCurrentContext(context);
    gui_vulkan_device = &device;
  }
  ~GuiVulkanCall() {
    gui_vulkan_device = previous_device_;
    ImGui::SetCurrentContext(previous_context_);
  }

 private:
  ImGuiContext *previous_context_;
  VulkanDevice *previous_device_;
};

void check_gui_vulkan_result(VkResult result) {
  if (result == VK_SUCCESS)
    return;
  TI_ERROR_IF(gui_vulkan_device == nullptr,
              "ImGui Vulkan error has no bound device");
  // Report after the caller's queue lock unwinds, matching VulkanStream.
  throw BackendRuntimeError(Arch::vulkan, static_cast<std::int64_t>(result),
                            "ImGui Vulkan", "ImGui Vulkan resource operation failed");
}

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

}  // namespace

PFN_vkVoidFunction load_vk_function_for_gui(const char *name, void *userData) {
  const auto *device = static_cast<VulkanDevice *>(userData);
  return vkGetInstanceProcAddr(device->vk_instance(), name);
}

Gui::Gui(AppContext *app_context,
         SwapChain *swap_chain,
         TaichiWindow *window,
         WindowLayoutState *window_layout) {
  app_context_ = app_context;
  swap_chain_ = swap_chain;
  window_layout_ = window_layout;

  create_descriptor_pool();

  IMGUI_CHECKVERSION();
  imgui_context_ = ImGui::CreateContext();
  ImGui::SetCurrentContext(imgui_context_);
  ImGuiIO &io = ImGui::GetIO();
  // GGUI owns GLFW cursor visibility through InputHandler. The ImGui GLFW
  // backend otherwise writes the native cursor every frame, which can add
  // visible Windows event-pump stalls when mouse enter/leave events are dense.
  io.ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;

  ImGui::StyleColorsDark();

  if (app_context->config.show_window) {
#ifdef ANDROID
    ImGui_ImplAndroid_Init(window);
    widthBeforeDPIScale = (int)ANativeWindow_getWidth(window);
    heightBeforeDPIScale = (int)ANativeWindow_getHeight(window);
#else
    ImGui_ImplGlfw_InitForVulkan(window, true);
    attach_glfw_imgui_context(window, imgui_context_);
    glfwGetWindowSize(window, &widthBeforeDPIScale, &heightBeforeDPIScale);
#endif
  } else {
    widthBeforeDPIScale = app_context->config.width;
    heightBeforeDPIScale = app_context->config.height;
  }
}

void Gui::init_render_resources(VkRenderPass render_pass, uint32_t image_count) {
  auto &device = static_cast<VulkanDevice &>(app_context_->device());
  GuiVulkanCall call(imgui_context_, device);
  TI_ERROR_IF(backend_resources_failed_ ||
                  ImGui::GetIO().BackendRendererUserData != nullptr,
              "ImGui Vulkan resources require proven cleanup before initialization");
  device.throw_if_backend_submission_disallowed("initialize ImGui Vulkan");
  TI_ERROR_IF(!ImGui_ImplVulkan_LoadFunctions(device.vk_caps().vk_api_version,
                                              load_vk_function_for_gui, &device),
              "Unable to load ImGui Vulkan functions");

  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.ApiVersion = device.vk_caps().vk_api_version;
  init_info.Instance = device.vk_instance();
  init_info.PhysicalDevice = device.vk_physical_device();
  init_info.Device = device.vk_device();
  init_info.QueueFamily = device.graphics_queue_family_index();
  init_info.Queue = device.graphics_queue();
  init_info.DescriptorPool = descriptor_pool_;
  init_info.MinImageCount = image_count;
  init_info.ImageCount = init_info.MinImageCount;
  init_info.RenderPass = render_pass;
  init_info.CheckVkResultFn = check_gui_vulkan_result;
  try {
    TI_ERROR_IF(!ImGui_ImplVulkan_Init(&init_info),
                "Unable to initialize ImGui Vulkan backend");
    // This remains a cold queue-wide font upload. Its owner and checked proof
    // are explicit; NewFrame must never submit a hidden upload afterward.
    {
      auto queue_lock = device.acquire_external_graphics_queue_lock();
      TI_ERROR_IF(!ImGui_ImplVulkan_CreateFontsTexture() ||
                      ImGui_ImplVulkan_ForgeGetFontState() !=
                          ImGui_ImplVulkan_ForgeFontState::ready,
                  "ImGui font upload has no completion proof");
    }
    next_frame_slot_ = 0;
    render_pass_ = render_pass;
    prepare_for_next_frame();
  } catch (const BackendRuntimeError &error) {
    backend_resources_failed_ = true;
    device.report_backend_error(error);
    throw;
  } catch (...) {
    backend_resources_failed_ = true;
    throw;
  }
}


void Gui::prepare_render_resources(VkRenderPass pass, uint32_t image_count) {
  validate_retirement();
  TI_ERROR_IF(pass == VK_NULL_HANDLE || image_count < 2,
              "ImGui Vulkan requires a renderpass and at least two draw slots");
  if (render_pass_ == pass && draw_slots_.size() == image_count) {
    validate_draw_resources(pass, image_count);
    ImGui::SetCurrentContext(imgui_context_);
    return;  // Do not end or restart an already populated CPU frame.
  }
  std::vector<DrawSlot> prepared(image_count);
  if (render_pass_ != VK_NULL_HANDLE)
    cleanup_render_resources();
  TI_ERROR_IF(generation_ == UINT64_MAX, "ImGui draw generation exhausted");
  draw_slots_ = std::move(prepared);
  ++generation_;
  init_render_resources(pass, image_count);
  ImGui::SetCurrentContext(imgui_context_);
}

void Gui::validate_draw_resources(VkRenderPass pass, uint32_t image_count) {
  auto &device = static_cast<VulkanDevice &>(app_context_->device());
  GuiVulkanCall call(imgui_context_, device);
  TI_ERROR_IF(backend_resources_failed_ || !device.backend_calls_safe() ||
                  render_pass_ != pass || draw_slots_.size() != image_count ||
                  pass == VK_NULL_HANDLE ||
                  ImGui_ImplVulkan_ForgeGetFontState() !=
                      ImGui_ImplVulkan_ForgeFontState::ready,
              "ImGui Vulkan requires compatible cold-prepared resources");
}

bool Gui::matches(DrawReservation r, DrawPhase phase) const noexcept {
  return active_reservation_ && r.generation == generation_ &&
         active_reservation_->generation == r.generation &&
         active_reservation_->serial == r.serial &&
         active_reservation_->slot == r.slot && r.slot < draw_slots_.size() &&
         draw_slots_[r.slot].phase == phase;
}

std::optional<Gui::DrawReservation> Gui::reserve_draw_slot(bool blocking) {
  validate_draw_resources(render_pass_, static_cast<uint32_t>(draw_slots_.size()));
  TI_ERROR_IF(active_reservation_, "ImGui already has an active draw reservation");
  if (!has_widgets())
    return std::nullopt;
  TI_ERROR_IF(next_serial_ == UINT64_MAX, "ImGui draw serial exhausted");
  try {
    for (;;) {
      for (uint32_t offset = 0; offset < draw_slots_.size(); ++offset) {
        const auto index = (next_frame_slot_ + offset) % draw_slots_.size();
        auto &slot = draw_slots_[index];
        if (slot.phase == DrawPhase::in_flight) {
          TI_ERROR_IF(!slot.complete, "ImGui reader is missing its completion");
          if (!slot.complete->is_ready())
            continue;
          slot.complete.reset();
          slot.phase = DrawPhase::idle;
        }
        TI_ERROR_IF(slot.phase != DrawPhase::idle, "ImGui slot is not reusable");
        active_reservation_ = DrawReservation{generation_, ++next_serial_,
                                               static_cast<uint32_t>(index)};
        slot.phase = DrawPhase::reserved;
        return active_reservation_;
      }
      if (!blocking)
        return std::nullopt;
      const auto &reader = draw_slots_[next_frame_slot_].complete;
      TI_ERROR_IF(!reader || !reader->wait(), "ImGui reader wait has no proof");
    }
  } catch (...) {
    quarantine_draw();
    throw;
  }
}

void Gui::validate_recorded_draw(DrawReservation r) {
  TI_ERROR_IF(backend_resources_failed_ || !matches(r, DrawPhase::recorded),
              "ImGui submission requires the recorded reservation");
}

void Gui::mark_draw_submitting(DrawReservation r) noexcept {
  assert(matches(r, DrawPhase::recorded));
  draw_slots_[r.slot].phase = DrawPhase::unsealed;
}

void Gui::seal_draw(DrawReservation r, const StreamSemaphore &complete) noexcept {
  assert(complete && matches(r, DrawPhase::unsealed));
  draw_slots_[r.slot].complete = complete;
  draw_slots_[r.slot].phase = DrawPhase::in_flight;
  active_reservation_.reset();
  next_frame_slot_ = (r.slot + 1) % draw_slots_.size();
}

void Gui::cancel_recorded_draw(DrawReservation r) noexcept {
  // Cancellation is never a substitute for a missing submission proof.
  if (!backend_resources_failed_ &&
      (matches(r, DrawPhase::reserved) || matches(r, DrawPhase::recorded))) {
    draw_slots_[r.slot].phase = DrawPhase::idle;
    active_reservation_.reset();
  }
}

void Gui::quarantine_draw() noexcept {
  backend_resources_failed_ = true;
}

bool Gui::retirement_failed() const noexcept {
  return backend_resources_failed_;
}

void Gui::validate_retirement() {
  auto &device = static_cast<VulkanDevice &>(app_context_->device());
  GuiVulkanCall call(imgui_context_, device);
  TI_ERROR_IF(backend_resources_failed_ || !device.backend_calls_safe() ||
                  active_reservation_ ||
                  ImGui_ImplVulkan_ForgeGetFontState() ==
                      ImGui_ImplVulkan_ForgeFontState::unsealed,
              "Cannot release unproven ImGui Vulkan resources");
  for (const auto &slot : draw_slots_)
    TI_ERROR_IF(slot.phase != DrawPhase::idle &&
                    (slot.phase != DrawPhase::in_flight || !slot.complete),
                "ImGui slot has no exact retirement proof");
}

void Gui::retire_draw_readers() {
  validate_retirement();
  // Do not clear any owner until every exact reader wait succeeds.
  try {
    for (const auto &slot : draw_slots_)
      if (slot.complete)
        TI_ERROR_IF(!slot.complete->wait(), "ImGui reader retirement failed");
  } catch (...) {
    quarantine_draw();
    throw;
  }
  for (auto &slot : draw_slots_) {
    slot.complete.reset();
    slot.phase = DrawPhase::idle;
  }
}

void Gui::create_descriptor_pool() {
  VkDescriptorPoolSize pool_sizes[] = {
      {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
      {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};
  VkDescriptorPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  pool_info.maxSets = 1000 * IM_ARRAYSIZE(pool_sizes);
  pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
  pool_info.pPoolSizes = pool_sizes;
  auto &device = static_cast<VulkanDevice &>(app_context_->device());
  const auto &vk_device = device.vk_device();
  const VkResult result = vk_device.functions().vkCreateDescriptorPool(
      vk_device, &pool_info, VK_NULL_HANDLE, &descriptor_pool_);
  if (result != VK_SUCCESS)
    device.raise_backend_error(static_cast<std::int64_t>(result),
                               "vkCreateDescriptorPool", "ImGui descriptor pool creation failed");
}

void Gui::prepare_for_next_frame() {
  ImGui::SetCurrentContext(imgui_context_);
  TI_ERROR_IF(backend_resources_failed_ || active_reservation_,
              "Cannot start an ImGui frame with unretired draw state");
  end_frame();
  if (render_pass_ == VK_NULL_HANDLE) {
    return;
  }
  auto &device = static_cast<VulkanDevice &>(app_context_->device());
  GuiVulkanCall call(imgui_context_, device);
  TI_ERROR_IF(backend_resources_failed_ ||
                  ImGui_ImplVulkan_ForgeGetFontState() !=
                      ImGui_ImplVulkan_ForgeFontState::ready,
              "ImGui font resources require cold preparation");
  ImGui_ImplVulkan_NewFrame();
  if (app_context_->config.show_window) {
#ifdef ANDROID
    ImGui_ImplAndroid_NewFrame();
#else
    ImGui_ImplGlfw_NewFrame();
#endif
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
    const int framebuffer_width =
        swap_chain_ != nullptr ? static_cast<int>(swap_chain_->width())
                               : app_context_->config.width;
    const int framebuffer_height =
        swap_chain_ != nullptr ? static_cast<int>(swap_chain_->height())
                               : app_context_->config.height;
    window_layout_->update_dimensions(io.DisplaySize.x, io.DisplaySize.y,
                                      framebuffer_width,
                                      framebuffer_height);
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

bool Gui::initialized() {
  return render_pass_ != VK_NULL_HANDLE && !backend_resources_failed_;
}

void Gui::mark_used() {
  ImGui::SetCurrentContext(imgui_context_);
  TI_ERROR_IF(backend_resources_failed_, "ImGui Vulkan resources are quarantined");
  is_empty_ = false;
}

float Gui::abs_x(float x) {
  return x * widthBeforeDPIScale;
}
float Gui::abs_y(float y) {
  return y * heightBeforeDPIScale;
}

void Gui::begin(const std::string &name,
                float x,
                float y,
                float width,
                float height) {
  mark_used();
  if (!initialized()) {
    return;
  }
  ImGui::SetNextWindowPos(ImVec2(abs_x(x), abs_y(y)), ImGuiCond_Once);
  ImGui::SetNextWindowSize(ImVec2(abs_x(width), abs_y(height)), ImGuiCond_Once);
  ImGui::Begin(name.c_str());
}
void Gui::begin_auto(const std::string &name, float x, float y, float width) {
  mark_used();
  if (!initialized()) {
    return;
  }
  const float fixed_width = abs_x(width);
  ImGui::SetNextWindowPos(ImVec2(abs_x(x), abs_y(y)), ImGuiCond_Once);
  ImGui::SetNextWindowSizeConstraints(ImVec2(fixed_width, 0.0f),
                                      ImVec2(fixed_width, FLT_MAX));
  ImGui::Begin(name.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize);
}
bool Gui::begin_collapsible_section(const std::string &name,
                                    bool default_open) {
  mark_used();
  if (!initialized()) {
    return default_open;
  }
  const ImGuiTreeNodeFlags flags =
      default_open ? ImGuiTreeNodeFlags_DefaultOpen : ImGuiTreeNodeFlags_None;
  const bool expanded = ImGui::CollapsingHeader(name.c_str(), flags);
  if (expanded) {
    ImGui::PushID(name.c_str());
    ImGui::Indent();
  }
  return expanded;
}
void Gui::end_collapsible_section() {
  ImGui::SetCurrentContext(imgui_context_);
  if (!initialized()) {
    return;
  }
  ImGui::Unindent();
  ImGui::PopID();
}
bool Gui::begin_edge_region(const std::string &name, WindowEdge edge) {
  mark_used();
  if (!initialized()) {
    return window_layout_ != nullptr &&
           window_layout_->region(edge).enabled &&
           !window_layout_->region(edge).collapsed;
  }
  return taichi::ui::vulkan::begin_edge_region(window_layout_, name, edge);
}
void Gui::end_edge_region(WindowEdge edge) {
  ImGui::SetCurrentContext(imgui_context_);
  if (!initialized()) {
    return;
  }
  taichi::ui::vulkan::end_edge_region(window_layout_, edge);
}
void Gui::end() {
  ImGui::SetCurrentContext(imgui_context_);
  if (!initialized()) {
    return;
  }
  ImGui::End();
}
void Gui::text(const std::string &text) {
  mark_used();
  if (!initialized()) {
    return;
  }
  ImGui::Text("%s", text.c_str());
}
void Gui::text(const std::string &text, glm::vec3 color) {
  mark_used();
  if (!initialized()) {
    return;
  }
  ImGui::TextColored(ImVec4(color[0], color[1], color[2], 1.0f), "%s",
                     text.c_str());
}
bool Gui::checkbox(const std::string &name, bool old_value) {
  mark_used();
  if (!initialized()) {
    return old_value;
  }
  ImGui::Checkbox(name.c_str(), &old_value);
  return old_value;
}
int Gui::slider_int(const std::string &name,
                    int old_value,
                    int minimum,
                    int maximum) {
  mark_used();
  if (!initialized()) {
    return old_value;
  }
  ImGui::SliderInt(name.c_str(), &old_value, minimum, maximum);
  return old_value;
}
float Gui::slider_float(const std::string &name,
                        float old_value,
                        float minimum,
                        float maximum) {
  mark_used();
  if (!initialized()) {
    return old_value;
  }
  ImGui::SliderFloat(name.c_str(), &old_value, minimum, maximum);
  return old_value;
}
glm::vec3 Gui::color_edit_3(const std::string &name, glm::vec3 old_value) {
  mark_used();
  if (!initialized()) {
    return old_value;
  }
  ImGui::ColorEdit3(name.c_str(), (float *)&old_value);
  return old_value;
}
bool Gui::button(const std::string &text) {
  mark_used();
  if (!initialized()) {
    return false;
  }
  return ImGui::Button(text.c_str());
}

bool Gui::draw(CommandList *cmd_list, DrawReservation reservation) {
  TI_ERROR_IF(!frame_started_ || !matches(reservation, DrawPhase::reserved),
              "ImGui draw requires its current reservation");
  auto &device = static_cast<VulkanDevice &>(app_context_->device());
  GuiVulkanCall call(imgui_context_, device);
  device.throw_if_backend_submission_disallowed("record ImGui Vulkan");
  try {
    ImGui::Render();
    frame_started_ = false;
    ImDrawData *draw_data = ImGui::GetDrawData();
    const int width = int(draw_data->DisplaySize.x * draw_data->FramebufferScale.x);
    const int height = int(draw_data->DisplaySize.y * draw_data->FramebufferScale.y);
    if (width <= 0 || height <= 0) {
      cancel_recorded_draw(reservation);
      return false;
    }
    auto buffer = static_cast<VulkanCommandList *>(cmd_list)
                      ->vk_command_buffer()->buffer;
    // The ledger selected this exact retired slot before backend map/growth.
    ImGui_ImplVulkan_ForgeRenderDrawData(draw_data, buffer, reservation.slot);
    draw_slots_[reservation.slot].phase = DrawPhase::recorded;
    return true;
  } catch (const BackendRuntimeError &error) {
    quarantine_draw();
    device.report_backend_error(error);
    throw;
  } catch (...) {
    quarantine_draw();
    throw;
  }
}

void Gui::cleanup_render_resources() {
  retire_draw_readers();
  auto &device = static_cast<VulkanDevice &>(app_context_->device());
  GuiVulkanCall call(imgui_context_, device);
  end_frame();
  if (ImGui::GetIO().BackendRendererUserData != nullptr)
    ImGui_ImplVulkan_Shutdown();
  render_pass_ = VK_NULL_HANDLE;
  draw_slots_.clear();
  next_frame_slot_ = 0;
}

Gui::~Gui() {
  auto *previous_context = ImGui::GetCurrentContext();
  ImGuiContext *remaining_context = nullptr;
  try {
    cleanup_render_resources();
  } catch (...) {
    // Keep the backend owner, upload allocations, descriptor pool and context.
    // Window preflight also retains the enclosing renderer/device closure.
    TI_WARN("ImGui Vulkan cleanup lacks proof; retaining backend resources");
    ImGui::SetCurrentContext(previous_context == imgui_context_
                                 ? nullptr : previous_context);
    return;
  }
  ImGui::SetCurrentContext(imgui_context_);
#ifndef ANDROID
  remaining_context = detach_glfw_imgui_context(imgui_context_);
#endif
  if (app_context_->config.show_window) {
#ifdef ANDROID
    ImGui_ImplAndroid_Shutdown();
#else
    ImGui_ImplGlfw_Shutdown();
#endif
  }
  if (descriptor_pool_ != VK_NULL_HANDLE) {
    const auto &device = static_cast<taichi::lang::vulkan::VulkanDevice &>(
                             app_context_->device())
                             .vk_device();
    device.functions().vkDestroyDescriptorPool(device, descriptor_pool_,
                                               nullptr);
    descriptor_pool_ = VK_NULL_HANDLE;
  }
  ImGui::DestroyContext(imgui_context_);
  ImGui::SetCurrentContext(previous_context == imgui_context_
                               ? remaining_context
                               : previous_context);
}

bool Gui::has_widgets() const {
  return !is_empty_;
}

bool Gui::wants_capture_mouse() const {
  ImGui::SetCurrentContext(imgui_context_);
  return ImGui::GetIO().WantCaptureMouse;
}

bool Gui::wants_capture_keyboard() const {
  ImGui::SetCurrentContext(imgui_context_);
  return ImGui::GetIO().WantCaptureKeyboard;
}

void Gui::end_frame() {
  ImGui::SetCurrentContext(imgui_context_);
  if (!frame_started_) {
    return;
  }
  ImGui::EndFrame();
  frame_started_ = false;
}

bool Gui::is_empty() const {
  return is_empty_;
}

}  // namespace vulkan

}  // namespace taichi::ui
