#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_timeline.h"

namespace taichi::lang::vulkan {

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
    std::shared_ptr<void> operands;
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
  std::shared_ptr<const Commands> prepare(
      const std::vector<CommandList *> &commands,
      std::shared_ptr<void> operands = {});
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
