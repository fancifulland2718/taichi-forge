#include <gtest/gtest.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>

#include <cstdlib>
#include <string>

#include "taichi/ui/common/glfw_input.h"

namespace taichi::ui {
namespace {

class GlfwInputTest : public testing::Test {
 protected:
  GLFWwindow *windows[2]{};
  ImGuiContext *contexts[2]{};
  int key_events[2]{};
  bool initialized{false};

  void create(int index) {
    windows[index] =
        glfwCreateWindow(100, 100, "Forge input test", nullptr, nullptr);
    ASSERT_NE(windows[index], nullptr);
    glfwSetWindowUserPointer(windows[index], &key_events[index]);
    glfwSetKeyCallback(windows[index], [](GLFWwindow *w, int, int, int, int) {
      ++*static_cast<int *>(glfwGetWindowUserPointer(w));
    });
    initialize_glfw_input(windows[index]);
    contexts[index] = ImGui::CreateContext();
    ImGui::SetCurrentContext(contexts[index]);
    ImGui::GetIO().IniFilename = nullptr;
    ASSERT_TRUE(ImGui_ImplGlfw_InitForOther(windows[index], true));
    attach_glfw_imgui_context(windows[index], contexts[index]);
  }

  void close(int index) {
    if (!windows[index]) {
      return;
    }
    auto *previous = ImGui::GetCurrentContext();
    ImGui::SetCurrentContext(contexts[index]);
    auto *remaining = detach_glfw_imgui_context(contexts[index]);
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext(contexts[index]);
    ImGui::SetCurrentContext(previous == contexts[index] ? remaining : previous);
    contexts[index] = nullptr;
    destroy_glfw_input(windows[index]);
    glfwDestroyWindow(windows[index]);
    windows[index] = nullptr;
  }

  void SetUp() override {
    const auto *enabled = std::getenv("TI_RUN_HEADED_GGUI_TESTS");
    if (!enabled || std::string(enabled) != "1") {
      GTEST_SKIP() << "Requires an opt-in desktop GLFW session";
    }
    ASSERT_TRUE(glfwInit());
    initialized = true;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    create(0);
    create(1);
  }

  void TearDown() override {
    close(0);
    close(1);
    if (initialized) {
      glfwTerminate();
    }
  }

  void scroll(int index, double x, double y) {
    auto callback = glfwSetScrollCallback(windows[index], nullptr);
    ASSERT_NE(callback, nullptr);
    glfwSetScrollCallback(windows[index], callback);
    callback(windows[index], x, y);
  }

  int wheel_events(int index) {
    int count = 0;
    for (const auto &event : contexts[index]->InputEventsQueue) {
      count += event.Type == ImGuiInputEventType_MouseWheel;
    }
    return count;
  }
};

TEST_F(GlfwInputTest, FractionalAxesAccumulateAndDrainOnce) {
  const double fine = 0.123456789012345;
  scroll(0, fine, -0.25);
  scroll(0, 0.5, 1.0);
  EXPECT_EQ(consume_glfw_scroll(windows[0]),
            (std::pair<double, double>{fine + 0.5, 0.75}));
  EXPECT_EQ(consume_glfw_scroll(windows[0]), (std::pair<double, double>{0.0, 0.0}));
  EXPECT_EQ(consume_glfw_scroll(nullptr), (std::pair<double, double>{0.0, 0.0}));
}

TEST_F(GlfwInputTest, RoutesEachWindowToItsOwnImGuiContext) {
  ImGui::SetCurrentContext(contexts[1]);
  scroll(0, 0.5, 1.25);
  EXPECT_EQ(ImGui::GetCurrentContext(), contexts[1]);
  EXPECT_EQ(wheel_events(0), 1);
  EXPECT_EQ(wheel_events(1), 0);
  scroll(1, -1.0, -0.5);
  EXPECT_EQ(consume_glfw_scroll(windows[0]), (std::pair<double, double>{0.5, 1.25}));
  EXPECT_EQ(consume_glfw_scroll(windows[1]), (std::pair<double, double>{-1.0, -0.5}));
  EXPECT_EQ(wheel_events(0), 1);
  EXPECT_EQ(wheel_events(1), 1);
}

TEST_F(GlfwInputTest, ConsumptionDoesNotStealImGuiEvents) {
  ImGui::SetCurrentContext(contexts[0]);
  ImGui::GetIO().WantCaptureMouse = true;
  scroll(0, 0.0, -0.5);
  EXPECT_EQ(consume_glfw_scroll(windows[0]).second, -0.5);
  EXPECT_EQ(wheel_events(0), 1);
  EXPECT_TRUE(ImGui::GetIO().WantCaptureMouse);
  EXPECT_EQ(consume_glfw_scroll(windows[0]).second, 0.0);
}

TEST_F(GlfwInputTest, PreservesExistingKeyboardCallbackChain) {
  ImGui::SetCurrentContext(contexts[1]);
  auto callback = glfwSetKeyCallback(windows[0], nullptr);
  glfwSetKeyCallback(windows[0], callback);
  ASSERT_NE(callback, nullptr);
  callback(windows[0], GLFW_KEY_A, 0, GLFW_PRESS, 0);
  EXPECT_EQ(key_events[0], 1);
  EXPECT_EQ(key_events[1], 0);
  EXPECT_EQ(ImGui::GetCurrentContext(), contexts[1]);
  EXPECT_FALSE(contexts[0]->InputEventsQueue.empty());
}

TEST_F(GlfwInputTest, CloseAndRecreateCannotReplayPendingScroll) {
  scroll(0, 1.0, 3.0);
  ImGui::SetCurrentContext(contexts[0]);
  close(0);
  EXPECT_EQ(ImGui::GetCurrentContext(), contexts[1]);
  scroll(1, -0.25, -0.75);
  EXPECT_EQ(consume_glfw_scroll(windows[1]),
            (std::pair<double, double>{-0.25, -0.75}));
  create(0);
  EXPECT_EQ(consume_glfw_scroll(windows[0]), (std::pair<double, double>{0.0, 0.0}));
  scroll(0, 0.0, 0.5);
  EXPECT_EQ(consume_glfw_scroll(windows[0]).second, 0.5);
}

}  // namespace
}  // namespace taichi::ui
