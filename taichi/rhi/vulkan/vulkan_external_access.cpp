#include "taichi/rhi/vulkan/vulkan_external_access.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <stdexcept>

#include "taichi/rhi/vulkan/vulkan_execution_queue.h"

namespace taichi::lang::vulkan {
namespace {
std::atomic<std::uint64_t> next_identity{1};

auto normalize(std::vector<std::shared_ptr<VulkanExternalBufferOwner>> owners) {
  if (owners.empty())
    throw std::invalid_argument("Vulkan external access requires an owner");
  for (const auto &owner : owners)
    if (!owner)
      throw std::invalid_argument("Null Vulkan external buffer owner");
  std::sort(owners.begin(), owners.end(), [](const auto &a, const auto &b) {
    return a->identity() < b->identity();
  });
  owners.erase(std::unique(owners.begin(), owners.end()), owners.end());
  return owners;
}
}  // namespace

struct VulkanExternalBufferOwner::State {
  VulkanDevice &device;
  const std::uint64_t identity;
  std::shared_ptr<const ReadyQuery> ready;
  mutable std::mutex mutex;
  std::shared_ptr<VulkanExternalBuffer> buffer;
  DeviceAllocationUnique allocation;
  std::shared_ptr<VulkanTimelinePoint> latest, uncertain;
  std::uint64_t next_epoch{1};
  bool accepting{true}, closed{false}, poisoned{false};

  State(VulkanDevice &device, ReadyQuery ready)
      : device(device),
        identity(next_identity.fetch_add(1, std::memory_order_relaxed)),
        ready(std::make_shared<ReadyQuery>(std::move(ready))) {
    if (!identity)
      throw std::overflow_error("Vulkan external owner identity exhausted");
    if (!*this->ready)
      throw std::invalid_argument(
          "Vulkan external buffer requires an epoch ready query");
  }
  void check() const {
    if (!accepting || closed || poisoned)
      throw std::runtime_error(
          "Vulkan external buffer owner is closed or poisoned");
  }
};

VulkanExternalBufferOwner::VulkanExternalBufferOwner(
    std::unique_ptr<State> state)
    : state_(std::move(state)) {
}

std::shared_ptr<VulkanExternalBufferOwner>
VulkanExternalBufferOwner::import_borrowed(
    VulkanDevice &device,
    const VulkanExternalBufferDescription &description,
    const VulkanExternalBufferHandles &handles,
    ReadyQuery ready,
    std::shared_ptr<void> producer_retention) {
  auto state = std::make_unique<State>(device, std::move(ready));
  struct ProducerRetention {
    std::shared_ptr<void> producer;
    std::shared_ptr<const ReadyQuery> ready;
  };
  // The callback can itself retain its native producer gate. It must survive
  // every import, too, even when the public owner has already been closed.
  auto retention = std::make_shared<ProducerRetention>(
      ProducerRetention{std::move(producer_retention), state->ready});
  state->buffer = VulkanExternalBuffer::import_borrowed(
      device, description, handles, std::move(retention));
  return std::shared_ptr<VulkanExternalBufferOwner>(
      new VulkanExternalBufferOwner(std::move(state)));
}

VulkanExternalBufferOwner::~VulkanExternalBufferOwner() {
  try {
    close();
  } catch (...) {
    // An unproven enqueue must retain its finite import/producer closure.
    (void)state_.release();
  }
}

std::uint64_t VulkanExternalBufferOwner::identity() const noexcept {
  return state_->identity;
}

VulkanExternalBufferOwner::Snapshot VulkanExternalBufferOwner::snapshot()
    const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return {state_->next_epoch, state_->accepting, state_->closed,
          state_->poisoned};
}

std::shared_ptr<VulkanExternalBuffer> VulkanExternalBufferOwner::buffer()
    const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->check();
  return state_->buffer;
}

bool VulkanExternalBufferOwner::consumer_ready() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->check();
  return (*state_->ready)(state_->next_epoch);
}

void VulkanExternalBufferOwner::acquire_for_consumer(
    const ExternalStreamDomain &) {
  throw std::invalid_argument(
      "External Vulkan access requires prepared atomic submission");
}

void VulkanExternalBufferOwner::release_from_consumer(
    const ExternalStreamDomain &) {
  throw std::invalid_argument(
      "External Vulkan access requires prepared atomic submission");
}

DeviceAllocation VulkanExternalBufferOwner::ndarray_allocation() {
  auto &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  state.check();
  if (!state.allocation) {
    const auto &description = state.buffer->description();
    if (!(description.usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT))
      throw std::invalid_argument(
          "External Vulkan ndarray requires producer storage usage");
    if (state.device.get_caps().get(
            DeviceCapability::spirv_has_physical_storage_buffer) &&
        (!(description.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) ||
         !(description.allocation_flags &
           VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)))
      throw std::invalid_argument(
          "External Vulkan ndarray requires producer device-address flags");
    // new allocates its guard before evaluating the import. No fallible host
    // allocation is left between a successful wrapper import and RAII
    // ownership.
    state.allocation.reset(
        new DeviceAllocationGuard(state.device.import_vkbuffer(
            state.buffer, description.size, state.buffer->memory(), 0,
            AllocUsage::Storage)));
  }
  return *state.allocation;
}

void VulkanExternalBufferOwner::close() {
  auto &state = *state_;
  std::shared_ptr<VulkanTimelinePoint> prefix;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    state.accepting = false;
    if (state.closed)
      return;
    if (state.uncertain)
      throw std::runtime_error(
          "Vulkan external buffer enqueue has no retirement proof");
    prefix = state.latest;
  }
  if (prefix)
    prefix->wait();
  std::shared_ptr<const ReadyQuery> retired_ready;
  std::shared_ptr<VulkanExternalBuffer> retired;
  DeviceAllocationUnique retired_allocation;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    retired = std::move(state.buffer);
    retired_allocation = std::move(state.allocation);
    retired_ready = std::move(state.ready);
    state.closed = true;
  }
  // Both native import destruction and producer callbacks run outside locks.
}

VulkanExternalAccess::VulkanExternalAccess(
    std::vector<std::shared_ptr<VulkanExternalBufferOwner>> owners)
    : owners_(normalize(std::move(owners))) {
}

void VulkanExternalAccess::record(
    VulkanCommandList &acquire,
    VulkanCommandList &release,
    std::vector<vkapi::IVkSemaphore> &waits,
    std::vector<vkapi::IVkSemaphore> &signals) const {
  for (const auto &owner : owners_) {
    std::lock_guard<std::mutex> lock(owner->state_->mutex);
    owner->state_->check();
    auto &buffer = *owner->state_->buffer;
    buffer.record_acquire(acquire);
    buffer.record_release(release);
    waits.push_back(buffer.ready());
    signals.push_back(buffer.consumed());
  }
}

void VulkanExternalAccess::Transaction::reserve() {
  if (!plan_)
    return;
  try {
    for (const auto &owner : plan_->owners_) {
      auto &state = *owner->state_;
      state.mutex.lock();
      ++locked_;
      state.check();
      if (state.next_epoch == UINT64_MAX)
        throw std::overflow_error("Vulkan external buffer epoch exhausted");
      if (!(*state.ready)(state.next_epoch))
        throw VulkanExecutionQueue::Busy();
    }
  } catch (...) {
    unlock();
    throw;
  }
}

void VulkanExternalAccess::Transaction::arm(
    const std::shared_ptr<VulkanTimelinePoint> &point) noexcept {
  if (plan_)
    point_ = point;
}

void VulkanExternalAccess::Transaction::seal() noexcept {
  if (!plan_)
    return;
  for (const auto &owner : plan_->owners_) {
    auto &state = *owner->state_;
    state.latest = point_;
    ++state.next_epoch;
  }
  point_.reset();
  unlock();
}

VulkanExternalAccess::Transaction::~Transaction() {
  if (point_) {
    for (const auto &owner : plan_->owners_) {
      auto &state = *owner->state_;
      state.poisoned = true;
      state.accepting = false;
      state.uncertain = point_;
    }
  }
  unlock();
}

void VulkanExternalAccess::Transaction::unlock() noexcept {
  while (locked_)
    plan_->owners_[--locked_]->state_->mutex.unlock();
}

void VulkanExternalAccess::confirm_drained(
    const std::shared_ptr<VulkanTimelinePoint> &point) const {
  for (const auto &owner : owners_) {
    std::lock_guard<std::mutex> lock(owner->state_->mutex);
    if (owner->state_->uncertain == point)
      owner->state_->uncertain.reset();
  }
}

}  // namespace taichi::lang::vulkan
