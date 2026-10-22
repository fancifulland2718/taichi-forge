#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/reusable_stream_dependency.h"

namespace taichi::lang::vulkan {

class VulkanTimelineState;

// A reusable device dependency, unlike a consumable binary semaphore. The
// owning runtime must retire all queue users before retiring the timeline.
// Completion wrappers may then retain the frozen host result after device exit.
class VulkanTimelinePoint final : public ReusableStreamDependency {
 public:
  bool is_ready() const override;
  bool wait() const override;
  std::uint64_t value() const noexcept { return value_; }

 private:
  friend class VulkanExecutionQueue;
  VulkanTimelinePoint(std::shared_ptr<VulkanTimelineState> state,
                      std::uint64_t value);
  std::shared_ptr<VulkanTimelineState> state_;
  std::uint64_t value_;
};

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
      const std::function<void(std::size_t, const Point &)> &admit = {});
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
