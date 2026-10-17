#include "taichi/ui/common/glfw_input.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui.h>

#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif

#include <stdexcept>
#include <unordered_map>

namespace taichi::ui {
namespace {

struct WindowInput {
  std::pair<double, double> scroll{0.0, 0.0};
  GLFWscrollfun previous_scroll{nullptr};
  ImGuiContext *context{nullptr};
  GLFWwindowfocusfun focus{nullptr};
  GLFWcursorenterfun enter{nullptr};
  GLFWcursorposfun position{nullptr};
  GLFWmousebuttonfun button{nullptr};
  GLFWscrollfun imgui_scroll{nullptr};
  GLFWkeyfun key{nullptr};
  GLFWcharfun character{nullptr};
#ifdef _WIN32
  HWND native_window{nullptr};
  WNDPROC previous_wndproc{nullptr};
#endif
};

std::unordered_map<GLFWwindow *, WindowInput> inputs;

void scroll_callback(GLFWwindow *window, double x, double y) {
  auto &input = inputs.at(window);
  input.scroll.first += x;
  input.scroll.second += y;
  if (input.previous_scroll) {
    input.previous_scroll(window, x, y);
  }
}

struct ContextScope {
  ImGuiContext *previous{ImGui::GetCurrentContext()};
  explicit ContextScope(ImGuiContext *context) {
    ImGui::SetCurrentContext(context);
  }
  ~ContextScope() {
    ImGui::SetCurrentContext(previous);
  }
};

#ifdef _WIN32
constexpr wchar_t kInputProperty[] = L"TaichiForge.ImGuiWindowInput";

LRESULT CALLBACK forward_wndproc(HWND window,
                                UINT message,
                                WPARAM wparam,
                                LPARAM lparam) {
  auto *input = static_cast<WindowInput *>(GetPropW(window, kInputProperty));
  ContextScope scope(input->context);
  return CallWindowProcW(input->previous_wndproc, window, message, wparam, lparam);
}
#endif

template <typename Callback, typename... Args>
void forward(GLFWwindow *window, Callback WindowInput::*member, Args... args) {
  auto &input = inputs.at(window);
  ContextScope scope(input.context);
  if (auto callback = input.*member) {
    callback(window, args...);
  }
}

}  // namespace

void initialize_glfw_input(GLFWwindow *window) {
  auto &input = inputs.emplace(window, WindowInput{}).first->second;
  input.previous_scroll = glfwSetScrollCallback(window, scroll_callback);
}

void destroy_glfw_input(GLFWwindow *window) {
  auto iter = inputs.find(window);
  if (iter != inputs.end()) {
    glfwSetScrollCallback(window, iter->second.previous_scroll);
    inputs.erase(iter);
  }
}

std::pair<double, double> consume_glfw_scroll(GLFWwindow *window) {
  auto iter = inputs.find(window);
  if (iter == inputs.end()) {
    return {0.0, 0.0};
  }
  return std::exchange(iter->second.scroll, {0.0, 0.0});
}

void attach_glfw_imgui_context(GLFWwindow *window, ImGuiContext *context) {
  auto &input = inputs.at(window);
  input.context = context;
  // ImGui's installed callbacks still own their original user callback chain.
  // Select the event window's context before entering that chain, since one
  // glfwPollEvents() dispatches events for every window in the process.
  input.focus = glfwSetWindowFocusCallback(window, [](GLFWwindow *w, int focused) {
    forward(w, &WindowInput::focus, focused);
  });
  input.enter = glfwSetCursorEnterCallback(window, [](GLFWwindow *w, int entered) {
    forward(w, &WindowInput::enter, entered);
  });
  input.position = glfwSetCursorPosCallback(
      window, [](GLFWwindow *w, double x, double y) {
        forward(w, &WindowInput::position, x, y);
      });
  input.button = glfwSetMouseButtonCallback(
      window, [](GLFWwindow *w, int button, int action, int mods) {
        forward(w, &WindowInput::button, button, action, mods);
      });
  input.imgui_scroll = glfwSetScrollCallback(
      window, [](GLFWwindow *w, double x, double y) {
        forward(w, &WindowInput::imgui_scroll, x, y);
      });
  input.key = glfwSetKeyCallback(
      window, [](GLFWwindow *w, int key, int scan, int action, int mods) {
        forward(w, &WindowInput::key, key, scan, action, mods);
      });
  input.character = glfwSetCharCallback(
      window, [](GLFWwindow *w, unsigned int codepoint) {
        forward(w, &WindowInput::character, codepoint);
      });
#ifdef _WIN32
  // The GLFW ImGui backend also installs a Win32 mouse-source hook. Route it
  // before it reads BackendPlatformUserData, including when a headless ImGui
  // context happens to be current. Wheel collection remains entirely in GLFW.
  input.native_window = glfwGetWin32Window(window);
  if (!SetPropW(input.native_window, kInputProperty, &input)) {
    throw std::runtime_error("Cannot associate the GGUI window input context");
  }
  input.previous_wndproc = reinterpret_cast<WNDPROC>(SetWindowLongPtrW(
      input.native_window, GWLP_WNDPROC,
      reinterpret_cast<LONG_PTR>(forward_wndproc)));
#endif
}

ImGuiContext *detach_glfw_imgui_context(ImGuiContext *context) {
  ImGuiContext *remaining = nullptr;
  for (auto &[window, input] : inputs) {
    if (input.context == context) {
#ifdef _WIN32
      SetWindowLongPtrW(input.native_window, GWLP_WNDPROC,
                       reinterpret_cast<LONG_PTR>(input.previous_wndproc));
      RemovePropW(input.native_window, kInputProperty);
#endif
      glfwSetWindowFocusCallback(window, input.focus);
      glfwSetCursorEnterCallback(window, input.enter);
      glfwSetCursorPosCallback(window, input.position);
      glfwSetMouseButtonCallback(window, input.button);
      glfwSetScrollCallback(window, input.imgui_scroll);
      glfwSetKeyCallback(window, input.key);
      glfwSetCharCallback(window, input.character);
      input.context = nullptr;
    } else if (input.context) {
      remaining = input.context;
    }
  }
  return remaining;
}

}  // namespace taichi::ui
