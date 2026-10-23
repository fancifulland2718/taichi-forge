#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "taichi/rhi/vulkan/vulkan_external_buffer.h"
#include "taichi/rhi/vulkan/vulkan_timeline.h"

namespace taichi::lang::vulkan {

class VulkanExternalAccess;

// One producer allocation and one binary handshake. Views/instances must share
// this owner. The producer serializes epochs through consumed and promises that
// ready(epoch) means its matching signal is already enqueued (not merely
// planned).
class VulkanExternalBufferOwner final {
 public:
  // Called under the owner lock: nonblocking, side-effect-free, and no Forge
  // reentry, GPU enqueue, or ownership transfer. False consumes nothing.
  using ReadyQuery = std::function<bool(std::uint64_t)>;
  struct Snapshot {
    std::uint64_t next_epoch;
    bool accepting, closed, poisoned;
  };
  static std::shared_ptr<VulkanExternalBufferOwner> import_borrowed(
      VulkanDevice &device,
      const VulkanExternalBufferDescription &description,
      const VulkanExternalBufferHandles &handles,
      ReadyQuery ready,
      std::shared_ptr<void> producer_retention = {});
  ~VulkanExternalBufferOwner();
  VulkanExternalBufferOwner(const VulkanExternalBufferOwner &) = delete;
  VulkanExternalBufferOwner &operator=(const VulkanExternalBufferOwner &) =
      delete;
  std::uint64_t identity() const noexcept;
  Snapshot snapshot() const;
  // Cold trusted recording only. Managed consumers retain the owner/plan too.
  std::shared_ptr<VulkanExternalBuffer> buffer() const;
  // Stops admission, waits only the last proven prefix outside the owner lock,
  // then drops its import. Command references can retain imports longer; their
  // producer retention is released only after the actual import destruction.
  void close();

 private:
  friend class VulkanExternalAccess;
  struct State;
  explicit VulkanExternalBufferOwner(std::unique_ptr<State> state);
  std::unique_ptr<State> state_;
};

// Immutable cold binding, deduplicated and sorted by owner identity. No runtime
// registry or backend callback is queried by the command-body replay itself.
class VulkanExternalAccess final {
 public:
  explicit VulkanExternalAccess(
      std::vector<std::shared_ptr<VulkanExternalBufferOwner>> owners);
  std::size_t size() const noexcept {
    return owners_.size();
  }

 private:
  friend class VulkanExecutionQueue;
  void record(VulkanCommandList &acquire,
              VulkanCommandList &release,
              std::vector<vkapi::IVkSemaphore> &waits,
              std::vector<vkapi::IVkSemaphore> &signals) const;
  // Stack transaction. reserve() rolls back its own partial locking on throw.
  // No allocation, callback, or other fallible action is allowed after arm().
  class Transaction {
   public:
    explicit Transaction(const VulkanExternalAccess *plan) : plan_(plan) {
    }
    ~Transaction();
    Transaction(const Transaction &) = delete;
    Transaction &operator=(const Transaction &) = delete;
    void reserve();
    void arm(const std::shared_ptr<VulkanTimelinePoint> &point) noexcept;
    void seal() noexcept;

   private:
    void unlock() noexcept;
    const VulkanExternalAccess *plan_;
    std::size_t locked_{0};
    std::shared_ptr<VulkanTimelinePoint> point_;
  };
  // Only the queue's successful native drain can resolve uncertain enqueue.
  // Owners remain permanently poisoned; this permits safe retirement only.
  void confirm_drained(const std::shared_ptr<VulkanTimelinePoint> &point) const;
  const std::vector<std::shared_ptr<VulkanExternalBufferOwner>> owners_;
};

}  // namespace taichi::lang::vulkan
