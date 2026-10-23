#pragma once

#include "gtest/gtest.h"
#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::lang::test_detail {
class TimelineGate {
 public:
  TimelineGate(vulkan::VulkanDevice &device, uint32_t queue_index)
      : device_(device.vk_device()) {
    VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    semaphore_ = vkapi::create_semaphore(device_, 0, &type);
    const uint64_t value = 1;
    VkTimelineSemaphoreSubmitInfo values{
        VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    values.waitSemaphoreValueCount = 1;
    values.pWaitSemaphoreValues = &value;
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo wait{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    wait.pNext = &values;
    wait.waitSemaphoreCount = 1;
    wait.pWaitSemaphores = &semaphore_->semaphore;
    wait.pWaitDstStageMask = &stage;
    fence_ = vkapi::create_fence(device_, 0);
    if (device_.functions().vkQueueSubmit(device.compute_queue_at(queue_index),
                                          1, &wait, fence_->fence) != VK_SUCCESS)
      throw std::runtime_error("Cannot enqueue test timeline gate");
  }
  ~TimelineGate() {
    release();
    device_.functions().vkWaitForFences(device_, 1, &fence_->fence, VK_TRUE,
                                       UINT64_MAX);
  }
  void release() {
    if (!released_) {
      VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
      signal.semaphore = semaphore_->semaphore;
      signal.value = 1;
      EXPECT_EQ(device_.functions().vkSignalSemaphore(device_, &signal), VK_SUCCESS);
      released_ = true;
    }
  }
 private:
  vkapi::DeviceHandle device_;
  vkapi::IVkSemaphore semaphore_;
  vkapi::IVkFence fence_;
  bool released_{false};
};

}  // namespace taichi::lang::test_detail
