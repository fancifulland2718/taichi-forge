#pragma once

#include <utility>

struct GLFWwindow;
struct ImGuiContext;

namespace taichi::ui {

// GLFW window and input operations run on the event thread. Keep this private
// state outside the WindowBase layout shared by the runtime and Python shim.
void initialize_glfw_input(GLFWwindow *window);
void destroy_glfw_input(GLFWwindow *window);
std::pair<double, double> consume_glfw_scroll(GLFWwindow *window);

void attach_glfw_imgui_context(GLFWwindow *window, ImGuiContext *context);
ImGuiContext *detach_glfw_imgui_context(ImGuiContext *context);

}  // namespace taichi::ui
