#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace taichi::lang::vulkan::detail {

// Internal policy, shared with the host-only tests. Keep the existing RHI
// entry point and exported ABI in vulkan_device.cpp.
static inline VkPresentModeKHR select_present_mode(
    const std::vector<VkPresentModeKHR> &available,
    bool vsync,
    bool adaptive,
    bool fifo_latest_ready_supported) {
  const auto supports = [&](VkPresentModeKHR mode) {
    return std::find(available.begin(), available.end(), mode) != available.end();
  };
  // MAILBOX still presents at vertical blank. Explicitly disabling vsync
  // should use immediate presentation when the surface supports it.
  if (!vsync && supports(VK_PRESENT_MODE_IMMEDIATE_KHR)) {
    return VK_PRESENT_MODE_IMMEDIATE_KHR;
  }
  if (supports(VK_PRESENT_MODE_MAILBOX_KHR)) {
    return VK_PRESENT_MODE_MAILBOX_KHR;
  }
  if (fifo_latest_ready_supported &&
      supports(VK_PRESENT_MODE_FIFO_LATEST_READY_KHR)) {
    return VK_PRESENT_MODE_FIFO_LATEST_READY_KHR;
  }
  if (vsync && adaptive && supports(VK_PRESENT_MODE_FIFO_RELAXED_KHR)) {
    return VK_PRESENT_MODE_FIFO_RELAXED_KHR;
  }
  // FIFO is the portable fallback and is required for ordinary WSI surfaces.
  if (supports(VK_PRESENT_MODE_FIFO_KHR)) {
    return VK_PRESENT_MODE_FIFO_KHR;
  }
  if (available.empty()) {
    throw std::runtime_error("no available present modes");
  }
  return available.front();
}

}  // namespace taichi::lang::vulkan::detail
