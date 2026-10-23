#pragma once

#include <shared_mutex>

#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::lang::vulkan {

// Shared implementation for ordinary-stream prefixes and independent domains.
// The Device retirement registry freezes native state after all queues drain.
class VulkanTimelineState {
 public:
  explicit VulkanTimelineState(VulkanDevice &device);
  std::uint64_t completed() const;
  void check_retired() const;  // mutex held
  void check(VkResult result, const char *operation) const;
  vkapi::DeviceHandle device;
  std::shared_ptr<BackendFaultReporter> reporter;
  mutable std::mutex mutex;
  mutable std::shared_mutex backend_mutex;
  vkapi::IVkSemaphore semaphore;
  std::uint64_t retired_value{0};
  bool retirement_failed{false};
  std::uint64_t max_difference{0};
  std::atomic<std::uint64_t> highest_enqueued{0};
};

// Immutable, reusable device dependency. A host observation never consumes it.
class VulkanTimelinePoint final : public ReusableStreamDependency {
 public:
  bool is_ready() const override;
  bool wait() const override;
  std::uint64_t value() const noexcept { return value_; }

 private:
  friend class VulkanExecutionQueue;
  friend class VulkanStream;
  VulkanTimelinePoint(std::shared_ptr<VulkanTimelineState> state,
                      std::uint64_t value);
  std::shared_ptr<VulkanTimelineState> state_;
  std::uint64_t value_;
};

}  // namespace taichi::lang::vulkan
