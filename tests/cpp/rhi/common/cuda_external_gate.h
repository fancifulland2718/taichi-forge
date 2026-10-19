#pragma once

// Test-only, real pending device dependency. Vulkan queues a binary signal
// behind a host-signaled timeline before CUDA queues its wait. No GPU spin or
// unrecorded event is used. A watchdog only prevents a broken test hanging;
// every positive assertion must happen before release and without timeout.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>

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
      const auto &original = device_.vk_device();
      original.instance_functions().vkGetPhysicalDeviceProperties2(
          device_.vk_physical_device(), &properties);
      if (std::memcmp(cuda_uuid.bytes, id.deviceUUID, VK_UUID_SIZE) != 0)
        throw std::runtime_error("Test gate requires the same CUDA/Vulkan GPU");

      // A vkCmdWaitEvents gate can occupy an engine and be released by the
      // Windows driver under cross-context pressure. Queue-level timeline
      // waits provide a real unsignaled dependency without running GPU work.
      VkPhysicalDeviceTimelineSemaphoreFeatures timeline_feature{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
      VkPhysicalDeviceFeatures2 features{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
      features.pNext = &timeline_feature;
      original.instance_functions().vkGetPhysicalDeviceFeatures2(
          device_.vk_physical_device(), &features);
      if (!timeline_feature.timelineSemaphore)
        throw std::runtime_error(
            "Test gate requires Vulkan timeline semaphores");
      float priority = 1;
      VkDeviceQueueCreateInfo queue_info{
          VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
      queue_info.queueFamilyIndex = device_.compute_queue_family_index();
      queue_info.queueCount = 1;
      queue_info.pQueuePriorities = &priority;
      const char *extensions[] = {VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
                                  VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
#ifdef _WIN32
                                  VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME
#else
                                  VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME
#endif
      };
      VkDeviceCreateInfo create{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
      create.pNext = &timeline_feature;
      create.queueCreateInfoCount = 1;
      create.pQueueCreateInfos = &queue_info;
      create.enabledExtensionCount = 3;
      create.ppEnabledExtensionNames = extensions;
      VkDevice logical = VK_NULL_HANDLE;
      check(original.instance_functions().vkCreateDevice(
          device_.vk_physical_device(), &create, nullptr, &logical));
      timeline_device_ = vkapi::DeviceHandle(device_.vk_instance(), logical);
      const auto &vk = timeline_device_;
      vk.functions().vkGetDeviceQueue(vk, queue_info.queueFamilyIndex, 0,
                                      &queue_);

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
      VkSemaphoreTypeCreateInfo type{
          VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
      type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
      VkSemaphoreCreateInfo timeline_info{
          VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
      timeline_info.pNext = &type;
      check(vk.functions().vkCreateSemaphore(vk, &timeline_info, nullptr,
                                             &timeline_));
      VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
      check(vk.functions().vkCreateFence(vk, &fence_info, nullptr, &fence_));
      const std::uint64_t wait_value = 1, binary_value = 0;
      VkTimelineSemaphoreSubmitInfo values{
          VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
      values.waitSemaphoreValueCount = values.signalSemaphoreValueCount = 1;
      values.pWaitSemaphoreValues = &wait_value;
      values.pSignalSemaphoreValues = &binary_value;
      VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
      VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      submit.pNext = &values;
      submit.waitSemaphoreCount = 1;
      submit.pWaitSemaphores = &timeline_;
      submit.pWaitDstStageMask = &stage;
      submit.signalSemaphoreCount = 1;
      submit.pSignalSemaphores = &semaphore_;
      check(vk.functions().vkQueueSubmit(queue_, 1, &submit, fence_));
      submitted_ = true;
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
    if (wait_completion_)
      throw std::logic_error("Binary gate can be consumed only once");
    auto context = CUDAContext::get_instance().get_guard();
    auto &driver = CUDADriver::get_instance();
    driver.event_create(&wait_completion_, CU_EVENT_DISABLE_TIMING);
    try {
      CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS params{};
      driver.wait_external_semaphore_async(&cuda_semaphore_, &params, 1,
                                           static_cast<CUstream>(stream));
      driver.event_record(wait_completion_, stream);
    } catch (...) {
      release();
      driver.stream_synchronize.call_concurrent(stream);
      throw;
    }
  }

  void release() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!released_ && timeline_) {
      VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
      signal.semaphore = timeline_;
      signal.value = 1;
      timeline_device_.functions().vkSignalSemaphore(timeline_device_, &signal);
      released_ = true;
      changed_.notify_all();
    }
  }

  bool timed_out() const {
    return timed_out_.load();
  }

  bool producer_pending() const {
    const auto &vk = timeline_device_;
    return vk.functions().vkGetFenceStatus(vk, fence_) == VK_NOT_READY;
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
    if (!timeline_device_)
      return;
    const auto &vk = timeline_device_;
    if (submitted_)
      vk.functions().vkWaitForFences(vk, 1, &fence_, VK_TRUE, UINT64_MAX);
    if (cuda_semaphore_) {
      auto context = CUDAContext::get_instance().get_guard();
      auto &driver = CUDADriver::get_instance();
      if (wait_completion_) {
        driver.event_synchronize.call_concurrent(wait_completion_);
        driver.event_destroy.call(wait_completion_);
      }
      driver.external_semaphore_destroy.call(cuda_semaphore_);
    }
    if (fence_)
      vk.functions().vkDestroyFence(vk, fence_, nullptr);
    if (timeline_)
      vk.functions().vkDestroySemaphore(vk, timeline_, nullptr);
    if (semaphore_)
      vk.functions().vkDestroySemaphore(vk, semaphore_, nullptr);
    vk.functions().vkDestroyDevice(vk, nullptr);
  }

  vulkan::VulkanDeviceCreator creator_;
  vulkan::VulkanDevice &device_;
  VkSemaphore semaphore_{};
  vkapi::DeviceHandle timeline_device_;
  VkQueue queue_{};
  VkSemaphore timeline_{};
  VkFence fence_{};
  CUexternalSemaphore cuda_semaphore_{};
  bool submitted_{false};
  void *wait_completion_{nullptr};
  std::mutex mutex_;
  std::condition_variable changed_;
  bool released_{false};
  std::atomic<bool> timed_out_{false};
  std::thread watchdog_;
};

}  // namespace taichi::lang::testing
