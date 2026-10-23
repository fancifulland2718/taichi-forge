#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_timeline.h"

namespace taichi::lang::vulkan {

class VulkanExternalAccess;

// Internal RHI owner used by fixed execution domains. Queue zero remains with
// the ordinary runtime. Admission capacity is chosen by the caller; queue count
// comes from the device. No Vulkan object is allocated during submit.
class VulkanExecutionQueue {
 public:
  struct Commands {
   private:
    friend class VulkanExecutionQueue;
    Commands() = default;
    VulkanDevice *device{nullptr};
    std::vector<vkapi::IVkCommandBuffer> buffers;
    std::vector<VkCommandBuffer> handles;
    // Cold, owned binary semaphore payloads. Ready/done epoch admission is
    // the execution owner's responsibility; no wait or signal is submitted
    // until this queue has admitted the entire command sequence.
    std::vector<vkapi::IVkSemaphore> binary_waits;
    std::vector<vkapi::IVkSemaphore> binary_signals;
    std::shared_ptr<void> operands;
    std::shared_ptr<const VulkanExternalAccess> external_access;
  };
  using Point = std::shared_ptr<VulkanTimelinePoint>;
  using CommandSelector =
      std::function<std::shared_ptr<const Commands>(std::size_t)>;
  class Busy : public std::runtime_error {
   public:
    Busy() : std::runtime_error("Vulkan execution queue is full") {}
  };

  VulkanExecutionQueue(VulkanDevice &device, std::size_t capacity);
  ~VulkanExecutionQueue();
  VulkanExecutionQueue(const VulkanExecutionQueue &) = delete;
  VulkanExecutionQueue &operator=(const VulkanExecutionQueue &) = delete;

  // Cold recording. Commands must be immutable, from this device/family, and
  // primary; operands own every allocation referenced by them.
  std::unique_ptr<CommandList> new_command_list();
  StreamGpuTiming prepare_timing(CommandList &begin, CommandList &end);
  // External semaphores must be binary. The caller proves that each ready
  // signal is already enqueued and owns the matching wait/signal epoch;
  // Vulkan has no query that can infer those facts from a semaphore handle.
  // They are retained with the commands through submission retirement.
  std::shared_ptr<const Commands> prepare(
      const std::vector<CommandList *> &commands,
      std::shared_ptr<void> operands = {},
      const std::vector<vkapi::IVkSemaphore> &binary_waits = {},
      const std::vector<vkapi::IVkSemaphore> &binary_signals = {});
  // Cold composition around an already selected immutable body. Reuses its
  // exact command buffers and operand closure; no kernel recording or submit.
  // The body must not already carry an external handshake. The caller supplies
  // matching whole-resource barriers and retains epoch admission responsibility.
  std::shared_ptr<const Commands> prepare_external_access(
      std::shared_ptr<const Commands> body,
      CommandList &acquire,
      CommandList &release,
      const std::vector<vkapi::IVkSemaphore> &binary_waits,
      const std::vector<vkapi::IVkSemaphore> &binary_signals,
      std::shared_ptr<void> operands = {});
  // Managed cold binding: records matching ownership boundaries and retains
  // the exact epoch plan in Commands. Admission cannot use a different plan.
  std::shared_ptr<const Commands> prepare_external_access(
      std::shared_ptr<const Commands> body,
      std::shared_ptr<const VulkanExternalAccess> access);
  // All validation and admission precede this host-only callback. It may copy
  // into the selected preallocated slot and reserve a completion wrapper for
  // the unpublished point, but cannot submit/wait/reenter/publish the point.
  // Every fallible host allocation must finish here, before device enqueue.
  Point submit(std::shared_ptr<const Commands> commands,
               const std::vector<Point> &dependencies = {},
               const std::function<void(std::size_t, const Point &)> &admit = {});
  // A runtime owner collects before taking its wider admission lock, keeping
  // operand destruction outside both locks. This variant never collects.
  Point submit_precollected(
      std::shared_ptr<const Commands> commands,
      const std::vector<Point> &dependencies = {},
      const std::function<void(std::size_t, const Point &)> &admit = {},
      // Select a pre-recorded variant after capacity/dependency admission and
      // before publication or device effects. This callback must not enqueue.
      const CommandSelector &select = {});
  std::size_t collect(std::size_t budget);
  std::size_t pending() const;
  std::uint32_t queue_index() const;
  Point latest() const;
  void stop();
  void close();
  // Runtime retirement only: every queue that waited on our points must have
  // drained first. Keeps old points queryable without native handles.
  void retire_timeline();

 private:
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace taichi::lang::vulkan
