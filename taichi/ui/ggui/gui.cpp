#include "gui.h"
#include "taichi/ui/ggui/swap_chain.h"
#include "taichi/ui/ggui/app_context.h"
#include "taichi/ui/ggui/edge_layout_imgui.h"

using namespace taichi::lang::vulkan;
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

}  // namespace

PFN_vkVoidFunction load_vk_function_for_gui(const char *name, void *userData) {
  auto result = VulkanLoader::instance().load_function(name);

  return result;
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
    glfwGetWindowSize(window, &widthBeforeDPIScale, &heightBeforeDPIScale);
#endif
  } else {
    widthBeforeDPIScale = app_context->config.width;
    heightBeforeDPIScale = app_context->config.height;
  }
}

void Gui::init_render_resources(VkRenderPass render_pass) {
  auto &device =
      static_cast<taichi::lang::vulkan::VulkanDevice &>(app_context_->device());

  // imgui 1.91+: ImGui_ImplVulkan_LoadFunctions takes api_version as first arg.
  ImGui_ImplVulkan_LoadFunctions(
      device.vk_caps().vk_api_version,
      load_vk_function_for_gui);  // this is because we're using volk.

  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.ApiVersion = device.vk_caps().vk_api_version;
  init_info.Instance = device.vk_instance();
  init_info.PhysicalDevice = device.vk_physical_device();
  init_info.Device = device.vk_device();
  init_info.QueueFamily = device.graphics_queue_family_index();
  init_info.Queue = device.graphics_queue();
  init_info.PipelineCache = VK_NULL_HANDLE;
  init_info.DescriptorPool = descriptor_pool_;
  init_info.Allocator = VK_NULL_HANDLE;
  init_info.MinImageCount = swap_chain_->surface().get_image_count();
  init_info.ImageCount = swap_chain_->surface().get_image_count();
  // imgui 1.90+: RenderPass moved from Init() arg into InitInfo;
  // CreateFontsTexture is now self-managing (called automatically in NewFrame),
  // and ImGui_ImplVulkan_DestroyFontUploadObjects has been removed.
  init_info.RenderPass = render_pass;
  ImGui_ImplVulkan_Init(&init_info);
  render_pass_ = render_pass;

  prepare_for_next_frame();
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
  [[maybe_unused]] VkResult err = vkCreateDescriptorPool(
      static_cast<taichi::lang::vulkan::VulkanDevice &>(app_context_->device())
          .vk_device(),
      &pool_info, VK_NULL_HANDLE, &descriptor_pool_);
}

void Gui::prepare_for_next_frame() {
  ImGui::SetCurrentContext(imgui_context_);
  end_frame();
  if (render_pass_ == VK_NULL_HANDLE) {
    return;
  }
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
  return render_pass_ != VK_NULL_HANDLE;
}

void Gui::mark_used() {
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
  if (!initialized()) {
    return;
  }
  taichi::ui::vulkan::end_edge_region(window_layout_, edge);
}
void Gui::end() {
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

void Gui::draw(taichi::lang::CommandList *cmd_list) {
  if (!frame_started_) {
    return;
  }

  // Rendering
  ImGui::Render();
  frame_started_ = false;
  ImDrawData *draw_data = ImGui::GetDrawData();

  VkCommandBuffer buffer =
      static_cast<VulkanCommandList *>(cmd_list)->vk_command_buffer()->buffer;

  ImGui_ImplVulkan_RenderDrawData(draw_data, buffer);
}

void Gui::cleanup_render_resources() {
  end_frame();
  if (initialized()) {
    ImGui_ImplVulkan_Shutdown();
  }
  render_pass_ = VK_NULL_HANDLE;
}

Gui::~Gui() {
  cleanup_render_resources();
  if (app_context_->config.show_window) {
#ifdef ANDROID
    ImGui_ImplAndroid_Shutdown();
#else
    ImGui_ImplGlfw_Shutdown();
#endif
  }
  if (descriptor_pool_ != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(
        static_cast<taichi::lang::vulkan::VulkanDevice &>(app_context_->device())
            .vk_device(),
        descriptor_pool_, nullptr);
    descriptor_pool_ = VK_NULL_HANDLE;
  }
  ImGui::DestroyContext(imgui_context_);
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
