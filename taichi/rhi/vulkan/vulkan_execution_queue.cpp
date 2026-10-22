#include "taichi/rhi/vulkan/vulkan_execution_queue.h"

#include <algorithm>
#include <condition_variable>
#include <limits>
#include <shared_mutex>

namespace taichi::lang::vulkan {

class VulkanTimelineState {
 public:
  explicit VulkanTimelineState(VulkanDevice &device)
      : device(device.vk_device()), reporter(device.backend_fault_reporter()) {
    VkPhysicalDeviceTimelineSemaphoreProperties timeline{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties.pNext = &timeline;
    this->device.instance_functions().vkGetPhysicalDeviceProperties2(
        device.vk_physical_device(), &properties);
    max_difference = timeline.maxTimelineSemaphoreValueDifference;
    VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    semaphore = vkapi::create_semaphore(this->device, 0, &type);
  }
  std::uint64_t completed() const {
    std::lock_guard<std::mutex> lock(mutex);
    if (!semaphore)
      return retired_value;
    if (reporter)
      reporter->throw_if_submission_disallowed("query Vulkan timeline");
    std::uint64_t value = 0;
    check(device.functions().vkGetSemaphoreCounterValue(
              device, semaphore->semaphore, &value),
          "vkGetSemaphoreCounterValue");
    return value;
  }
  void check(VkResult result, const char *operation) const {
    if (result == VK_SUCCESS)
      return;
    BackendRuntimeError error(Arch::vulkan, result, operation,
                              "Vulkan execution timeline failed");
    if (reporter)
      reporter->report_backend_error(error, 0);
    throw error;
  }
  vkapi::DeviceHandle device;
  std::shared_ptr<BackendFaultReporter> reporter;
  mutable std::mutex mutex;
  mutable std::shared_mutex backend_mutex;
  vkapi::IVkSemaphore semaphore;
  std::uint64_t retired_value{0};
  std::uint64_t max_difference{0};
  std::atomic<std::uint64_t> highest_enqueued{0};
};

VulkanTimelinePoint::VulkanTimelinePoint(
    std::shared_ptr<VulkanTimelineState> state, std::uint64_t value)
    : state_(std::move(state)), value_(value) {}

bool VulkanTimelinePoint::is_ready() const {
  return state_->completed() >= value_;
}

bool VulkanTimelinePoint::wait() const {
  std::shared_lock<std::shared_mutex> backend_lock(state_->backend_mutex);
  vkapi::IVkSemaphore semaphore;
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (!state_->semaphore)
      return state_->retired_value >= value_;
    semaphore = state_->semaphore;
  }
  if (state_->reporter)
    state_->reporter->throw_if_submission_disallowed("wait Vulkan timeline");
  VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  wait.semaphoreCount = 1;
  wait.pSemaphores = &semaphore->semaphore;
  wait.pValues = &value_;
  state_->check(state_->device.functions().vkWaitSemaphores(
                    state_->device, &wait, UINT64_MAX),
                "vkWaitSemaphores");
  return true;
}

struct VulkanExecutionQueue::State {
  struct Slot {
    Point point;
    std::shared_ptr<const Commands> commands;
    std::vector<Point> dependencies;
  };
  VulkanDevice &device;
  std::uint32_t index{0};
  std::unique_ptr<VulkanStream> stream;
  std::shared_ptr<VulkanTimelineState> timeline;
  mutable std::mutex mutex;
  std::condition_variable changed;
  std::vector<Slot> slots;
  std::size_t head{0}, count{0};
  Point tail;
  bool accepting{true}, closing{false}, closed{false};
  std::exception_ptr error;
  State(VulkanDevice &device, std::size_t capacity)
      : device(device), slots(capacity) {}
};

VulkanExecutionQueue::VulkanExecutionQueue(VulkanDevice &device,
                                           std::size_t capacity) {
  if (!capacity || !device.vk_caps().timeline_semaphore)
    throw std::invalid_argument(
        "Vulkan execution queues require positive capacity and timeline support");
  device.throw_if_backend_submission_disallowed("create Vulkan execution queue");
  state_ = std::make_unique<State>(device, capacity);
  auto &state = *state_;
  {
    std::lock_guard<std::mutex> lock(device.execution_queue_mutex_);
    for (std::size_t i = 0; i < device.additional_compute_queues_.size(); ++i) {
      auto &queue = *device.additional_compute_queues_[i];
      if (!queue.leased) {
        queue.leased = true;
        state.index = static_cast<std::uint32_t>(i + 1);
        break;
      }
    }
  }
  if (!state.index)
    throw std::runtime_error("No independent Vulkan compute queue is available");
  try {
    state.stream = device.create_compute_stream(state.index);
    state.timeline = std::make_shared<VulkanTimelineState>(device);
    if (capacity > state.timeline->max_difference)
      throw std::invalid_argument(
          "Capacity exceeds the device timeline value-difference limit");
  } catch (...) {
    std::lock_guard<std::mutex> lock(device.execution_queue_mutex_);
    device.additional_compute_queues_[state.index - 1]->leased = false;
    throw;
  }
}

VulkanExecutionQueue::~VulkanExecutionQueue() {
  try {
    close();
  } catch (...) {
    // No completion proof: retain the finite in-flight closure. Device loss
    // must not turn wrapper destruction into use-after-free of GPU operands.
    (void)state_.release();
  }
}

std::unique_ptr<CommandList> VulkanExecutionQueue::new_command_list() {
  auto &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  if (!state.accepting)
    throw std::runtime_error("Vulkan execution queue is closed");
  auto [commands, status] = state.stream->new_command_list_unique();
  if (status != RhiResult::success)
    throw std::runtime_error("Vulkan execution command allocation failed");
  return std::move(commands);
}

std::shared_ptr<const VulkanExecutionQueue::Commands>
VulkanExecutionQueue::prepare(const std::vector<CommandList *> &commands,
                               std::shared_ptr<void> operands) {
  auto &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  if (!state.accepting)
    throw std::runtime_error("Vulkan execution queue is closed");
  if (commands.size() > UINT32_MAX)
    throw std::overflow_error("Vulkan command count exceeds uint32");
  auto result = std::shared_ptr<Commands>(new Commands);
  result->device = &state.device;
  // Keep the parameter's reference until after the lock on every failure.
  result->operands = operands;
  for (auto *command : commands) {
    auto *native = dynamic_cast<VulkanCommandList *>(command);
    if (!native)
      throw std::invalid_argument("Vulkan execution requires native commands");
    auto buffer = native->finalize();
    if (buffer->device != state.device.vk_device() ||
        buffer->level != VK_COMMAND_BUFFER_LEVEL_PRIMARY ||
        buffer->ref_pool->queue_family_index != state.device.compute_queue_family_index())
      throw std::invalid_argument("Vulkan execution command device/family mismatch");
    result->handles.push_back(buffer->buffer);
    result->buffers.push_back(std::move(buffer));
  }
  return result;
}

VulkanExecutionQueue::Point VulkanExecutionQueue::submit(
    std::shared_ptr<const Commands> commands,
    const std::vector<Point> &dependencies,
    const std::function<void(std::size_t)> &admit) {
  if (!commands || dependencies.size() > UINT32_MAX)
    throw std::invalid_argument("Invalid Vulkan execution submission");
  collect(1);
  auto &state = *state_;
  State::Slot pending;
  std::lock_guard<std::mutex> lock(state.mutex);
  state.device.throw_if_backend_submission_disallowed("submit Vulkan execution queue");
  if (state.error)
    std::rethrow_exception(state.error);
  if (!state.accepting)
    throw std::runtime_error("Vulkan execution queue is closed");
  if (commands->device != &state.device)
    throw std::invalid_argument("Vulkan execution command device mismatch");
  std::vector<std::pair<VkSemaphore, std::uint64_t>> waits;
  waits.reserve(dependencies.size());
  for (const auto &point : dependencies) {
    if (!point || point->state_->device != state.device.vk_device() ||
        !point->state_->semaphore)
      throw std::invalid_argument("Vulkan dependency device is foreign or retired");
    const auto highest =
        point->state_->highest_enqueued.load(std::memory_order_acquire);
    if (highest - point->value() > point->state_->max_difference)
      throw std::invalid_argument(
          "Dependency exceeds the device timeline value-difference limit");
    waits.emplace_back(point->state_->semaphore->semaphore, point->value());
  }
  // Several exact prefixes of one timeline need only the greatest wait.
  std::sort(waits.begin(), waits.end(), [](const auto &a, const auto &b) {
    return std::less<VkSemaphore>{}(a.first, b.first);
  });
  std::vector<VkSemaphore> semaphores;
  std::vector<std::uint64_t> values;
  for (const auto &[semaphore, value] : waits) {
    if (!semaphores.empty() && semaphores.back() == semaphore) {
      values.back() = std::max(values.back(), value);
    } else {
      semaphores.push_back(semaphore);
      values.push_back(value);
    }
  }
  if (state.count == state.slots.size())
    throw Busy();
  const auto previous = state.tail ? state.tail->value() : 0;
  if (previous == UINT64_MAX)
    throw std::overflow_error("Vulkan timeline counter exhausted");
  auto point = Point(new VulkanTimelinePoint(state.timeline, previous + 1));
  pending.dependencies = dependencies;
  pending.commands = std::move(commands);
  pending.point = point;
  std::vector<VkPipelineStageFlags> stages(semaphores.size(),
                                          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
  const auto index = (state.head + state.count) % state.slots.size();
  if (admit)
    admit(index);
  auto &slot = state.slots[index];
  slot = std::move(pending);
  state.tail = point;
  ++state.count;
  VkTimelineSemaphoreSubmitInfo timeline{
      VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
  timeline.waitSemaphoreValueCount = static_cast<uint32_t>(values.size());
  timeline.pWaitSemaphoreValues = values.data();
  timeline.signalSemaphoreValueCount = 1;
  timeline.pSignalSemaphoreValues = &point->value_;
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.pNext = &timeline;
  submit.waitSemaphoreCount = static_cast<uint32_t>(semaphores.size());
  submit.pWaitSemaphores = semaphores.data();
  submit.pWaitDstStageMask = stages.data();
  submit.commandBufferCount = static_cast<uint32_t>(slot.commands->handles.size());
  submit.pCommandBuffers = slot.commands->handles.data();
  submit.signalSemaphoreCount = 1;
  submit.pSignalSemaphores = &state.timeline->semaphore->semaphore;
  try {
    const auto queue = state.device.compute_queue_at(state.index);
    auto queue_lock = state.device.acquire_queue_lock(queue);
    state.timeline->check(state.device.vk_device().functions().vkQueueSubmit(
                              queue, 1, &submit, VK_NULL_HANDLE),
                          "vkQueueSubmit");
    state.timeline->highest_enqueued.store(point->value(), std::memory_order_release);
    state.device.queue_tail_fence_locked(queue).reset();
    state.device.queue_submit_calls_.fetch_add(1, std::memory_order_relaxed);
    state.device.submitted_command_buffers_.fetch_add(
        slot.commands->handles.size(), std::memory_order_relaxed);
  } catch (...) {
    state.error = std::current_exception();
    state.accepting = false;
    throw;
  }
  return point;
}

std::size_t VulkanExecutionQueue::collect(std::size_t budget) {
  auto &state = *state_;
  std::size_t collected = 0;
  while (collected < budget) {
    State::Slot retired;
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      if (!state.count || state.closing || state.closed || state.error)
        break;
      auto &slot = state.slots[state.head];
      if (!slot.point->is_ready())
        break;
      retired = std::move(slot);
      slot = {};
      state.head = (state.head + 1) % state.slots.size();
      --state.count;
      ++collected;
    }
    // Resource/provider destruction never runs under admission locks.
  }
  return collected;
}

std::size_t VulkanExecutionQueue::pending() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->count;
}
std::uint32_t VulkanExecutionQueue::queue_index() const { return state_->index; }
VulkanExecutionQueue::Point VulkanExecutionQueue::latest() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->tail;
}
void VulkanExecutionQueue::stop() {
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->accepting = false;
}
void VulkanExecutionQueue::close() {
  auto &state = *state_;
  Point tail;
  {
    std::unique_lock<std::mutex> lock(state.mutex);
    state.changed.wait(lock, [&] { return !state.closing; });
    if (state.closed)
      return;
    state.accepting = false;
    state.closing = true;
    tail = state.tail;
  }
  std::vector<State::Slot> retired;
  try {
    if (state.error) {
      const auto queue = state.device.compute_queue_at(state.index);
      auto queue_lock = state.device.acquire_queue_lock(queue);
      state.timeline->check(state.device.vk_device().functions().vkQueueWaitIdle(queue),
                            "vkQueueWaitIdle");
    } else if (tail) {
      tail->wait();
    }
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      retired = std::move(state.slots);
      state.count = 0;
    }
    retired.clear();
    state.stream.reset();
    {
      std::lock_guard<std::mutex> lock(state.device.execution_queue_mutex_);
      state.device.additional_compute_queues_[state.index - 1]->leased = false;
    }
  } catch (...) {
    std::lock_guard<std::mutex> lock(state.mutex);
    state.closing = false;
    state.changed.notify_all();
    throw;
  }
  std::lock_guard<std::mutex> lock(state.mutex);
  state.closed = true;
  state.closing = false;
  state.changed.notify_all();
}

void VulkanExecutionQueue::retire_timeline() {
  close();
  auto &timeline = *state_->timeline;
  std::unique_lock<std::shared_mutex> backend_lock(timeline.backend_mutex);
  const auto completed = timeline.completed();
  std::lock_guard<std::mutex> lock(timeline.mutex);
  timeline.retired_value = completed;
  timeline.semaphore.reset();
}

}  // namespace taichi::lang::vulkan
