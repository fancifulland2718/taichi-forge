#include "gtest/gtest.h"

#if defined(TI_WITH_VULKAN) || defined(TI_TEST_DISPLAY_COMPLETION)
#include "taichi/rhi/vulkan/vulkan_present_policy.h"

namespace taichi::lang::vulkan {
namespace {

TEST(VulkanPresentPolicy, ExplicitVsyncOffIsImmediateRegardlessOfDriverOrder) {
  std::vector<VkPresentModeKHR> modes{
      VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_MAILBOX_KHR,
      VK_PRESENT_MODE_FIFO_KHR, VK_PRESENT_MODE_FIFO_RELAXED_KHR,
      VK_PRESENT_MODE_FIFO_LATEST_READY_KHR};
  std::sort(modes.begin(), modes.end());
  do {
    for (bool adaptive : {false, true}) {
      for (bool latest_ready : {false, true}) {
        EXPECT_EQ(detail::select_present_mode(modes, false, adaptive,
                                             latest_ready),
                  VK_PRESENT_MODE_IMMEDIATE_KHR);
        EXPECT_EQ(detail::select_present_mode(modes, true, adaptive,
                                             latest_ready),
                  VK_PRESENT_MODE_MAILBOX_KHR);
      }
    }
  } while (std::next_permutation(modes.begin(), modes.end()));
}

TEST(VulkanPresentPolicy, VsyncOffFallsBackWhenImmediateIsUnavailable) {
  EXPECT_EQ(detail::select_present_mode(
                {VK_PRESENT_MODE_FIFO_KHR, VK_PRESENT_MODE_MAILBOX_KHR},
                false, true, false),
            VK_PRESENT_MODE_MAILBOX_KHR);
  EXPECT_EQ(detail::select_present_mode({VK_PRESENT_MODE_FIFO_KHR}, false,
                                       true, false),
            VK_PRESENT_MODE_FIFO_KHR);
}

TEST(VulkanPresentPolicy, LatestReadyRequiresEnabledFeatureAndSurfaceMode) {
  const std::vector<VkPresentModeKHR> modes{
      VK_PRESENT_MODE_FIFO_KHR, VK_PRESENT_MODE_FIFO_LATEST_READY_KHR};
  for (bool vsync : {false, true}) {
    EXPECT_EQ(detail::select_present_mode(modes, vsync, false, true),
              VK_PRESENT_MODE_FIFO_LATEST_READY_KHR);
    EXPECT_EQ(detail::select_present_mode(modes, vsync, false, false),
              VK_PRESENT_MODE_FIFO_KHR);
    EXPECT_EQ(detail::select_present_mode({VK_PRESENT_MODE_FIFO_KHR}, vsync,
                                         false, true),
              VK_PRESENT_MODE_FIFO_KHR);
  }
}

TEST(VulkanPresentPolicy, AdaptiveFallbackDoesNotOverrideStrictVsync) {
  const std::vector<VkPresentModeKHR> modes{
      VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_FIFO_RELAXED_KHR,
      VK_PRESENT_MODE_FIFO_KHR};
  EXPECT_EQ(detail::select_present_mode(modes, true, true, false),
            VK_PRESENT_MODE_FIFO_RELAXED_KHR);
  EXPECT_EQ(detail::select_present_mode(modes, true, false, false),
            VK_PRESENT_MODE_FIFO_KHR);
}

TEST(VulkanPresentPolicy, MissingSurfaceModesFailBeforeSwapchainCreation) {
  EXPECT_THROW(detail::select_present_mode({}, false, false, false),
               std::runtime_error);
}

}  // namespace
}  // namespace taichi::lang::vulkan
#endif
