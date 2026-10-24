#pragma once

#include <algorithm>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <type_traits>
#include <variant>
#include <vector>

#include "taichi/rhi/vulkan/vulkan_external_buffer.h"
#include "taichi/rhi/vulkan/vulkan_timeline.h"
#include "taichi/rhi/interop/external_sync.h"
#include "taichi/rhi/interop/native_ready_gate.h"

namespace taichi::lang::vulkan {

class VulkanExternalAccess;

// Canonical cold import and finite view. Transport adapters copy facts here;
// all Vulkan flags/UUID/requirements rules remain in the RHI importer.
struct VulkanExternalBufferImport {
  VulkanExternalBufferDescription buffer;
  VulkanExternalBufferHandles handles;
  std::uint64_t byte_offset{0}, byte_size{0};
  void validate_ndarray(VulkanDevice &device) const;
};

// Accept the exact public C ABI structure without depending on its header.
// Both the C API and JIT use this decoder; no function-pointer reinterpretation.
template <typename Info>
VulkanExternalBufferImport decode_vulkan_external_buffer_import(const Info &info) {
  static_assert(std::is_standard_layout_v<Info> &&
                std::is_trivially_copyable_v<Info>);
  static_assert(sizeof(info.device_uuid) == VK_UUID_SIZE &&
                sizeof(info.driver_uuid) == VK_UUID_SIZE);
  if (info.struct_size != sizeof(Info) || info.schema_version != 1)
    throw std::invalid_argument("Invalid external Vulkan import size or version");
  constexpr auto max_handle = (std::numeric_limits<std::uintptr_t>::max)();
  if (info.dedicated != 1 || info.memory_handle >= max_handle ||
      info.ready_for_consumer >= max_handle ||
      info.ready_for_external >= max_handle)
    throw std::invalid_argument(
        "Invalid external Vulkan dedicated or handle facts");
  VulkanExternalBufferImport result;
  auto &d = result.buffer;
  d.size = info.buffer_size;
  d.allocation_size = info.allocation_size;
  d.bind_offset = info.bind_offset;
  d.memory_type_index = info.memory_type_index;
  d.flags = info.buffer_flags;
  d.usage = info.buffer_usage;
  d.allocation_flags = info.allocation_flags;
  d.sharing_mode = static_cast<VkSharingMode>(info.sharing_mode);
  d.dedicated = true;
  std::copy(std::begin(info.device_uuid), std::end(info.device_uuid),
            d.device_uuid.begin());
  std::copy(std::begin(info.driver_uuid), std::end(info.driver_uuid),
            d.driver_uuid.begin());
  result.handles = {static_cast<std::uintptr_t>(info.memory_handle),
                    static_cast<std::uintptr_t>(info.ready_for_consumer),
                    static_cast<std::uintptr_t>(info.ready_for_external)};
  result.byte_offset = info.byte_offset;
  result.byte_size = info.byte_size;
  return result;
}

// One producer allocation and one binary handshake. Views/instances must share
// this owner. The producer serializes epochs through consumed and promises that
// ready(epoch) means its matching signal is already enqueued (not merely
// planned).
class VulkanExternalBufferOwner final : public ExternalSynchronizationDomain {
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
  ~VulkanExternalBufferOwner() override;
  VulkanExternalBufferOwner(const VulkanExternalBufferOwner &) = delete;
  VulkanExternalBufferOwner &operator=(const VulkanExternalBufferOwner &) =
      delete;
  std::uint64_t identity() const noexcept override;
  bool retirement_waits_for_consumer() const noexcept override {
    return true;
  }
  bool has_consumer_ready_query() const noexcept override {
    return true;
  }
  bool consumer_ready() const override;
  void acquire_for_consumer(const ExternalStreamDomain &) override;
  void release_from_consumer(const ExternalStreamDomain &) override;
  Snapshot snapshot() const;
  // Cold trusted recording only. Managed consumers retain the owner/plan too.
  std::shared_ptr<VulkanExternalBuffer> buffer() const;
  // Cold, idempotent view for the existing managed ndarray registry. Requires
  // the producer's actual storage/address flags; never changes imported flags.
  DeviceAllocation ndarray_allocation();
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

// One local packed display allocation. A cold prepared recipe may borrow it
// before acquisition, but admission requires one outstanding write lease and
// consumes that lease's one complete producer submission. This owner stores
// only native points, never RuntimeCompletion/Commands/instance closures.
// Canvas graphics readers and last-image pinning are added at the UI boundary;
// this native contract alone does not enable writable Canvas frames.
class VulkanDisplayBufferOwner final : public ExternalSynchronizationDomain {
 public:
  struct Selection {
    std::uint64_t program_owner{0}, domain{0};
    std::uint32_t queue_index{0};
  };
  struct Snapshot {
    std::uint64_t next_epoch, lease;
    bool accepting, closed, poisoned, acquired, has_writer, sealed;
  };
  // Real local storage, not an external-handle round trip. The caller gets a
  // finite allocation for Program's existing external-managed registry.
  static std::shared_ptr<VulkanDisplayBufferOwner> create(
      VulkanDevice &device, std::size_t bytes, Selection selected);
  ~VulkanDisplayBufferOwner() override;
  VulkanDisplayBufferOwner(const VulkanDisplayBufferOwner &) = delete;
  VulkanDisplayBufferOwner &operator=(const VulkanDisplayBufferOwner &) = delete;
  std::uint64_t identity() const noexcept override;
  bool retirement_waits_for_consumer() const noexcept override { return true; }
  bool has_consumer_ready_query() const noexcept override { return true; }
  bool consumer_ready() const override;
  void acquire_for_consumer(const ExternalStreamDomain &) override;
  void release_from_consumer(const ExternalStreamDomain &) override;
  Snapshot snapshot() const;
  Selection selection() const noexcept;
  DeviceAllocation ndarray_allocation() const;
  void validate_binding(VulkanDevice &device, Selection selected) const;
  // A fresh lease nonce rejects stale host calls even if an unwritten lease
  // was cancelled. Cancellation does not consume a producer epoch.
  std::uint64_t acquire_write();
  void cancel_write(std::uint64_t lease);
  // Identity equality, not timeline value equality: only the actual writer
  // accepted for this lease can seal it. Failure leaves the lease retryable.
  std::shared_ptr<VulkanTimelinePoint> seal_write(
      std::uint64_t lease, const std::shared_ptr<VulkanTimelinePoint> &producer);
  void stop();
  // Cold retirement only. Stop admission, wait the exact proven writer outside
  // the lock, and release device storage now, even if ticket/loan shells live.
  void close();

 private:
  friend class VulkanExternalAccess;
  struct State;
  explicit VulkanDisplayBufferOwner(std::unique_ptr<State> state);
  std::unique_ptr<State> state_;
};

// Complete native cold transaction; frontends own their runtime submission
// guard and register the returned owner's idempotent ndarray wrapper once.
template <typename Gate>
std::shared_ptr<VulkanExternalBufferOwner> import_vulkan_external_ndarray(
    VulkanDevice &device, const VulkanExternalBufferImport &info,
    const Gate &gate) {
  validate_native_ready_gate(gate);
  info.validate_ndarray(device);
  auto owner = VulkanExternalBufferOwner::import_borrowed(
      device, info.buffer, info.handles, retain_native_ready_gate(gate));
  owner->ndarray_allocation();
  return owner;
}

// Immutable cold binding, deduplicated and sorted by owner identity. No runtime
// registry or backend callback is queried by the command-body replay itself.
class VulkanExternalAccess final {
 public:
  using Owner = std::variant<std::shared_ptr<VulkanExternalBufferOwner>,
                             std::shared_ptr<VulkanDisplayBufferOwner>>;
  explicit VulkanExternalAccess(
      std::vector<std::shared_ptr<VulkanExternalBufferOwner>> owners);
  explicit VulkanExternalAccess(std::vector<Owner> owners);
  std::size_t size() const noexcept {
    return owners_.size();
  }

 private:
  friend class VulkanExecutionQueue;
  bool requires_import_boundaries() const;
  void validate_queue(VulkanDevice &device, std::uint32_t queue_index) const;
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
    void reserve(VulkanDevice &device, std::uint32_t queue_index);
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
  const std::vector<Owner> owners_;
};

}  // namespace taichi::lang::vulkan
