#include "taichi/rhi/vulkan/vulkan_external_access.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <stdexcept>

#include "taichi/rhi/vulkan/vulkan_execution_queue.h"

namespace taichi::lang::vulkan {
namespace {
std::atomic<std::uint64_t> next_identity{1};

auto normalize(std::vector<VulkanExternalAccess::Owner> owners) {
  if (owners.empty())
    throw std::invalid_argument("Vulkan external access requires an owner");
  const auto identity = [](const auto &owner) {
    return std::visit([](const auto &value) { return value->identity(); }, owner);
  };
  for (const auto &owner : owners)
    if (!std::visit([](const auto &value) { return bool(value); }, owner))
      throw std::invalid_argument("Null Vulkan external buffer owner");
  std::sort(owners.begin(), owners.end(), [&](const auto &a, const auto &b) {
    return identity(a) < identity(b);
  });
  owners.erase(std::unique(owners.begin(), owners.end()), owners.end());
  return owners;
}

auto imported_entries(
    std::vector<std::shared_ptr<VulkanExternalBufferOwner>> owners) {
  std::vector<VulkanExternalAccess::Owner> result;
  result.reserve(owners.size());
  for (auto &owner : owners)
    result.emplace_back(std::move(owner));
  return result;
}
}  // namespace

void VulkanExternalBufferImport::validate_ndarray(VulkanDevice &device) const {
  if (buffer.allocation_size > (std::numeric_limits<std::size_t>::max)() ||
      !byte_size || byte_offset > buffer.size ||
      byte_size > buffer.size - byte_offset ||
      !(buffer.usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT))
    throw std::invalid_argument(
        "Invalid external Vulkan ndarray range or storage usage");
  // One source for flags, handle distinctness, UUID and native import support.
  // These queries touch only the local physical device, before retaining producer.
  VulkanExternalBuffer::validate_import(device, buffer, handles);
  if (device.get_caps().get(DeviceCapability::spirv_has_physical_storage_buffer) &&
      (!(buffer.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) ||
       !(buffer.allocation_flags & VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)))
    throw std::invalid_argument(
        "External Vulkan ndarray requires producer device-address flags");
}

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

struct VulkanDisplayBufferOwner::State {
  VulkanDevice &device;
  const std::uint64_t identity;
  const Selection selected;
  mutable std::mutex mutex;
  DeviceAllocationUnique allocation;
  std::vector<StreamSemaphore> readers;
  std::shared_ptr<VulkanTimelinePoint> latest, uncertain;
  std::uint64_t next_epoch{1}, next_lease{1}, lease{0}, writer_lease{0};
  bool accepting{true}, closed{false}, poisoned{false};
  bool acquired{false}, sealed{false};
  bool graphics_unsealed{false};

  State(VulkanDevice &device, Selection selected)
      : device(device),
        identity(next_identity.fetch_add(1, std::memory_order_relaxed)),
        selected(selected) {
    if (!identity)
      throw std::overflow_error("Vulkan display owner identity exhausted");
  }
  void check() const {
    if (!accepting || closed || poisoned || graphics_unsealed)
      throw std::runtime_error("Vulkan display buffer owner is closed or poisoned");
  }
  bool ready_for_write() const {
    check();
    if (acquired || (latest && !latest->is_ready()))
      return false;
    for (const auto &reader : readers)
      if (reader && !reader->is_ready())
        return false;
    return true;
  }
  void check_lease(std::uint64_t value) const {
    check();
    if (!acquired || !value || value != lease)
      throw std::invalid_argument("Vulkan display write lease is not current");
  }
};

VulkanDisplayBufferOwner::VulkanDisplayBufferOwner(std::unique_ptr<State> state)
    : state_(std::move(state)) {
}

std::shared_ptr<VulkanDisplayBufferOwner> VulkanDisplayBufferOwner::create(
    VulkanDevice &device, std::size_t bytes, Selection selected,
    std::size_t graphics_capacity) {
  if (!bytes || bytes % sizeof(std::uint32_t) ||
      bytes > device.get_storage_buffer_binding_limits().max_range)
    throw std::invalid_argument("Invalid local Vulkan display storage extent");
  if (!selected.program_owner || !selected.domain || !selected.queue_index ||
      selected.queue_index >= device.created_compute_queue_count())
    throw std::invalid_argument("Vulkan display storage requires a selected independent domain");
  auto state = std::make_unique<State>(device, selected);
  state->readers.resize(graphics_capacity);
  auto allocation = device.allocate_memory_unique(
      {bytes, false, false, false, AllocUsage::Storage});
  if (allocation.second != RhiResult::success)
    throw std::runtime_error("Cannot allocate local Vulkan display storage");
  state->allocation = std::move(allocation.first);
  return std::shared_ptr<VulkanDisplayBufferOwner>(
      new VulkanDisplayBufferOwner(std::move(state)));
}

VulkanDisplayBufferOwner::~VulkanDisplayBufferOwner() {
  try {
    close();
  } catch (...) {
    // An enqueue without proof keeps its finite device allocation quarantined.
    (void)state_.release();
  }
}

std::uint64_t VulkanDisplayBufferOwner::identity() const noexcept {
  return state_->identity;
}

VulkanDisplayBufferOwner::Selection VulkanDisplayBufferOwner::selection()
    const noexcept {
  return state_->selected;
}

VulkanDisplayBufferOwner::Snapshot VulkanDisplayBufferOwner::snapshot() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return {state_->next_epoch, state_->lease,
          state_->accepting && !state_->graphics_unsealed, state_->closed,
          state_->poisoned || state_->graphics_unsealed, state_->acquired,
          state_->writer_lease != 0,
          state_->sealed};
}

bool VulkanDisplayBufferOwner::consumer_ready() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->check();
  return state_->acquired && !state_->writer_lease;
}

void VulkanDisplayBufferOwner::acquire_for_consumer(const ExternalStreamDomain &) {
  throw std::invalid_argument("Vulkan display access requires prepared atomic submission");
}

void VulkanDisplayBufferOwner::release_from_consumer(const ExternalStreamDomain &) {
  throw std::invalid_argument("Vulkan display access requires prepared atomic submission");
}

DeviceAllocation VulkanDisplayBufferOwner::ndarray_allocation() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->check();
  return *state_->allocation;
}

void VulkanDisplayBufferOwner::validate_binding(VulkanDevice &device,
                                                Selection selected) const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->check();
  const auto &actual = state_->selected;
  if (&device != &state_->device || selected.program_owner != actual.program_owner ||
      selected.domain != actual.domain || selected.queue_index != actual.queue_index)
    throw std::invalid_argument("Vulkan display writer belongs to another device or domain");
}

std::uint64_t VulkanDisplayBufferOwner::acquire_write() {
  auto &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  state.check();
  if (!state.ready_for_write())
    throw VulkanExecutionQueue::Busy();
  if (state.next_lease == UINT64_MAX)
    throw std::overflow_error("Vulkan display lease counter exhausted");
  state.lease = state.next_lease++;
  state.writer_lease = 0;
  state.acquired = true;
  state.sealed = false;
  return state.lease;
}

bool VulkanDisplayBufferOwner::ready_for_write() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->ready_for_write();
}

void VulkanDisplayBufferOwner::cancel_write(std::uint64_t lease) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->check_lease(lease);
  // Keep latest after an accepted writer. Its exact prefix, not the number of
  // facade references, determines when a cancelled slot may be acquired again.
  state_->acquired = false;
}

std::shared_ptr<VulkanTimelinePoint> VulkanDisplayBufferOwner::seal_write(
    std::uint64_t lease, const std::shared_ptr<VulkanTimelinePoint> &producer) {
  auto &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  state.check_lease(lease);
  if (!producer || state.writer_lease != lease || producer != state.latest)
    throw std::invalid_argument("Vulkan display seal requires this lease's exact producer");
  state.acquired = false;
  state.sealed = true;
  return state.latest;
}

StreamSemaphore VulkanDisplayBufferOwner::submit_graphics(
    VulkanStream &stream, CommandList *commands,
    const std::vector<StreamSemaphore> &additional_waits,
    const std::shared_ptr<VulkanTimelinePoint> &producer) {
  auto &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  state.check();
  if (!state.sealed || state.acquired || !producer || producer != state.latest)
    throw std::invalid_argument("Vulkan display reader requires the sealed producer");
  auto available = std::find_if(state.readers.begin(), state.readers.end(),
      [](const auto &reader) { return !reader || reader->is_ready(); });
  if (available == state.readers.end())
    throw std::runtime_error("Vulkan display reader capacity is full");
  auto waits = additional_waits;
  waits.push_back(producer);
  // Every fallible host preparation precedes enqueue. The RHI stream can still
  // throw after vkQueueSubmit, so no returned proof means sticky quarantine.
  try {
    auto complete = stream.submit(commands, waits);
    if (!complete)
      throw std::runtime_error("Vulkan display reader returned no completion");
    *available = complete;
    return complete;
  } catch (...) {
    state.graphics_unsealed = true;
    throw;
  }
}

void VulkanDisplayBufferOwner::stop() {
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->accepting = false;
}

void VulkanDisplayBufferOwner::close() {
  auto &state = *state_;
  std::shared_ptr<VulkanTimelinePoint> prefix;
  std::vector<StreamSemaphore> readers;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    state.accepting = false;
    if (state.closed)
      return;
    if (state.uncertain || state.graphics_unsealed)
      throw std::runtime_error("Vulkan display enqueue has no retirement proof");
    prefix = state.latest;
    readers = state.readers;
  }
  if (prefix)
    prefix->wait();
  for (const auto &reader : readers)
    if (reader && !reader->wait())
      throw std::runtime_error("Vulkan display reader has no retirement proof");
  DeviceAllocationUnique retired;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    retired = std::move(state.allocation);
    readers = std::move(state.readers);
    state.latest.reset();
    state.acquired = false;
    state.closed = true;
  }
  // DeviceAllocationGuard must retire before the Program destroys its device,
  // irrespective of any surviving borrowed owner or completion-query shell.
}

VulkanExternalAccess::VulkanExternalAccess(
    std::vector<std::shared_ptr<VulkanExternalBufferOwner>> owners)
    : VulkanExternalAccess(imported_entries(std::move(owners))) {
}

VulkanExternalAccess::VulkanExternalAccess(std::vector<Owner> owners)
    : owners_(normalize(std::move(owners))) {
}

bool VulkanExternalAccess::requires_import_boundaries() const {
  return std::any_of(owners_.begin(), owners_.end(), [](const auto &owner) {
    return std::holds_alternative<std::shared_ptr<VulkanExternalBufferOwner>>(owner);
  });
}

void VulkanExternalAccess::validate_queue(VulkanDevice &device,
                                         std::uint32_t queue_index) const {
  for (const auto &entry : owners_) {
    if (const auto *local =
            std::get_if<std::shared_ptr<VulkanDisplayBufferOwner>>(&entry)) {
      auto selected = (*local)->selection();
      selected.queue_index = queue_index;
      (*local)->validate_binding(device, selected);
    }
  }
}

void VulkanExternalAccess::record(
    VulkanCommandList &acquire,
    VulkanCommandList &release,
    std::vector<vkapi::IVkSemaphore> &waits,
    std::vector<vkapi::IVkSemaphore> &signals) const {
  for (const auto &entry : owners_) {
    const auto *imported =
        std::get_if<std::shared_ptr<VulkanExternalBufferOwner>>(&entry);
    if (!imported)
      continue;
    const auto &owner = *imported;
    std::lock_guard<std::mutex> lock(owner->state_->mutex);
    owner->state_->check();
    auto &buffer = *owner->state_->buffer;
    buffer.record_acquire(acquire);
    buffer.record_release(release);
    waits.push_back(buffer.ready());
    signals.push_back(buffer.consumed());
  }
}

void VulkanExternalAccess::Transaction::reserve(VulkanDevice &device,
                                                std::uint32_t queue_index) {
  if (!plan_)
    return;
  try {
    for (const auto &entry : plan_->owners_) {
      std::visit([&](const auto &owner) {
        auto &state = *owner->state_;
        state.mutex.lock();
        ++locked_;
        state.check();
        if (state.next_epoch == UINT64_MAX)
          throw std::overflow_error("Vulkan managed buffer epoch exhausted");
        if constexpr (std::is_same_v<std::decay_t<decltype(*owner)>,
                                     VulkanExternalBufferOwner>) {
          if (!(*state.ready)(state.next_epoch))
            throw VulkanExecutionQueue::Busy();
        } else {
          if (&state.device != &device || state.selected.queue_index != queue_index)
            throw std::invalid_argument("Vulkan display access queue mismatch");
          if (!state.acquired || state.writer_lease)
            throw std::invalid_argument(
                "Vulkan display lease requires one complete prepared writer");
        }
      }, entry);
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
  for (const auto &entry : plan_->owners_) {
    std::visit([&](const auto &owner) {
      auto &state = *owner->state_;
      state.latest = point_;
      ++state.next_epoch;
      if constexpr (std::is_same_v<std::decay_t<decltype(*owner)>,
                                   VulkanDisplayBufferOwner>)
        state.writer_lease = state.lease;
    }, entry);
  }
  point_.reset();
  unlock();
}

VulkanExternalAccess::Transaction::~Transaction() {
  if (point_) {
    for (const auto &entry : plan_->owners_) {
      std::visit([&](const auto &owner) {
        auto &state = *owner->state_;
        state.poisoned = true;
        state.accepting = false;
        state.uncertain = point_;
      }, entry);
    }
  }
  unlock();
}

void VulkanExternalAccess::Transaction::unlock() noexcept {
  while (locked_)
    std::visit([](const auto &owner) { owner->state_->mutex.unlock(); },
               plan_->owners_[--locked_]);
}

void VulkanExternalAccess::confirm_drained(
    const std::shared_ptr<VulkanTimelinePoint> &point) const {
  for (const auto &entry : owners_) {
    std::visit([&](const auto &owner) {
      std::lock_guard<std::mutex> lock(owner->state_->mutex);
      if (owner->state_->uncertain == point)
        owner->state_->uncertain.reset();
    }, entry);
  }
}

}  // namespace taichi::lang::vulkan
