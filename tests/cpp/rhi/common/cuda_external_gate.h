#pragma once

// Test-only, real pending device dependency. Vulkan queues a binary signal
// behind a host event before CUDA queues its wait. No CUDA spin kernel or
// unrecorded event is used. A watchdog only prevents a broken test hanging;
// every positive assertion must happen before release and without timeout.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#ifdef _WIN32
#include "taichi/platform/windows/windows.h"
#else
#include <unistd.h>
#endif

namespace taichi::lang::testing {

class CudaExternalGate {
 public:
  CudaExternalGate() : creator_({}), device_(*creator_.device()) {
    try {
      auto context = CUDAContext::get_instance().get_guard();
      auto &driver = CUDADriver::get_instance();
      CUuuid cuda_uuid{};
      driver.device_get_uuid(&cuda_uuid,
                             CUDAContext::get_instance().get_device());
      VkPhysicalDeviceIDProperties id{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
      VkPhysicalDeviceProperties2 properties{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
      properties.pNext = &id;
      const auto &vk = device_.vk_device();
      vk.instance_functions().vkGetPhysicalDeviceProperties2(
          device_.vk_physical_device(), &properties);
      if (std::memcmp(cuda_uuid.bytes, id.deviceUUID, VK_UUID_SIZE) != 0)
        throw std::runtime_error("Test gate requires the same CUDA/Vulkan GPU");

      VkExportSemaphoreCreateInfo external{
          VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
#ifdef _WIN32
      external.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
      external.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
      VkSemaphoreCreateInfo semaphore_info{
          VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
      semaphore_info.pNext = &external;
      check(vk.functions().vkCreateSemaphore(vk, &semaphore_info, nullptr,
                                             &semaphore_));
      CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC import{};
#ifdef _WIN32
      VkSemaphoreGetWin32HandleInfoKHR handle_info{
          VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
      handle_info.semaphore = semaphore_;
      handle_info.handleType =
          VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
      HANDLE handle = nullptr;
      check(vk.functions().vkGetSemaphoreWin32HandleKHR(vk, &handle_info,
                                                        &handle));
      import.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32;
      import.handle.win32.handle = handle;
      const auto result =
          driver.import_external_semaphore.call(&cuda_semaphore_, &import);
      CloseHandle(handle);
#else
      VkSemaphoreGetFdInfoKHR handle_info{
          VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
      handle_info.semaphore = semaphore_;
      handle_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
      int fd = -1;
      check(vk.functions().vkGetSemaphoreFdKHR(vk, &handle_info, &fd));
      import.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD;
      import.handle.fd = fd;
      const auto result =
          driver.import_external_semaphore.call(&cuda_semaphore_, &import);
      if (result != CUDA_SUCCESS)
        ::close(fd);  // Successful CUDA import takes FD ownership.
#endif
      if (result != CUDA_SUCCESS)
        throw std::runtime_error(
            driver.import_external_semaphore.get_error_message(result));
      VkEventCreateInfo event_info{VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
      check(vk.functions().vkCreateEvent(vk, &event_info, nullptr, &event_));
      VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
      check(vk.functions().vkCreateFence(vk, &fence_info, nullptr, &fence_));
      VkCommandPoolCreateInfo pool_info{
          VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
      pool_info.queueFamilyIndex = device_.compute_queue_family_index();
      check(
          vk.functions().vkCreateCommandPool(vk, &pool_info, nullptr, &pool_));
      VkCommandBufferAllocateInfo allocation{
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
      allocation.commandPool = pool_;
      allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      allocation.commandBufferCount = 1;
      VkCommandBuffer command{};
      check(vk.functions().vkAllocateCommandBuffers(vk, &allocation, &command));
      VkCommandBufferBeginInfo begin{
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      check(vk.functions().vkBeginCommandBuffer(command, &begin));
      vk.functions().vkCmdWaitEvents(command, 1, &event_,
                                     VK_PIPELINE_STAGE_HOST_BIT,
                                     VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                                     nullptr, 0, nullptr, 0, nullptr);
      check(vk.functions().vkEndCommandBuffer(command));
      VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      submit.commandBufferCount = 1;
      submit.pCommandBuffers = &command;
      submit.signalSemaphoreCount = 1;
      submit.pSignalSemaphores = &semaphore_;
      {
        auto queue_lock = device_.acquire_external_compute_queue_lock();
        check(vk.functions().vkQueueSubmit(device_.compute_queue(), 1, &submit,
                                           fence_));
        submitted_ = true;
      }
      watchdog_ = std::thread([this] {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!changed_.wait_for(lock, std::chrono::seconds(5),
                               [this] { return released_; })) {
          timed_out_.store(true);
          lock.unlock();
          release();
        }
      });
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~CudaExternalGate() {
    cleanup();
  }
  CudaExternalGate(const CudaExternalGate &) = delete;
  CudaExternalGate &operator=(const CudaExternalGate &) = delete;

  void enqueue_wait(void *stream) {
    if (!streams_.empty())
      throw std::logic_error("Binary gate can be consumed only once");
    streams_.push_back(stream);  // Allocate before the device side effect.
    auto context = CUDAContext::get_instance().get_guard();
    CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS params{};
    CUDADriver::get_instance().wait_external_semaphore_async(
        &cuda_semaphore_, &params, 1, static_cast<CUstream>(stream));
  }

  void release() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!released_ && event_) {
      device_.vk_device().functions().vkSetEvent(device_.vk_device(), event_);
      released_ = true;
      changed_.notify_all();
    }
  }

  bool timed_out() const {
    return timed_out_.load();
  }

 private:
  static void check(VkResult result) {
    if (result != VK_SUCCESS)
      throw std::runtime_error("Vulkan test gate setup failed: " +
                               std::to_string(result));
  }
  void cleanup() noexcept {
    release();
    if (watchdog_.joinable())
      watchdog_.join();
    const auto &vk = device_.vk_device();
    if (submitted_)
      vk.functions().vkWaitForFences(vk, 1, &fence_, VK_TRUE, UINT64_MAX);
    if (cuda_semaphore_) {
      auto context = CUDAContext::get_instance().get_guard();
      auto &driver = CUDADriver::get_instance();
      for (auto stream : streams_)
        driver.stream_synchronize.call_concurrent(stream);
      driver.external_semaphore_destroy.call(cuda_semaphore_);
    }
    if (pool_)
      vk.functions().vkDestroyCommandPool(vk, pool_, nullptr);
    if (fence_)
      vk.functions().vkDestroyFence(vk, fence_, nullptr);
    if (event_)
      vk.functions().vkDestroyEvent(vk, event_, nullptr);
    if (semaphore_)
      vk.functions().vkDestroySemaphore(vk, semaphore_, nullptr);
  }

  vulkan::VulkanDeviceCreator creator_;
  vulkan::VulkanDevice &device_;
  VkSemaphore semaphore_{};
  VkEvent event_{};
  VkFence fence_{};
  VkCommandPool pool_{};
  CUexternalSemaphore cuda_semaphore_{};
  bool submitted_{false};
  std::vector<void *> streams_;
  std::mutex mutex_;
  std::condition_variable changed_;
  bool released_{false};
  std::atomic<bool> timed_out_{false};
  std::thread watchdog_;
};

}  // namespace taichi::lang::testing
