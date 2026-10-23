#include "taichi/rhi/vulkan/vulkan_execution_queue.h"

#include <algorithm>
#include <condition_variable>
#include <limits>
#include <shared_mutex>

namespace taichi::lang::vulkan {

VulkanTimelineState::VulkanTimelineState(VulkanDevice &owner)
    : device(owner.vk_device()), reporter(owner.backend_fault_reporter()) {
  VkPhysicalDeviceTimelineSemaphoreProperties timeline{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_PROPERTIES};
  VkPhysicalDeviceProperties2 properties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  properties.pNext = &timeline;
  device.instance_functions().vkGetPhysicalDeviceProperties2(
      owner.vk_physical_device(), &properties);
  max_difference = timeline.maxTimelineSemaphoreValueDifference;
  VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  semaphore = vkapi::create_semaphore(device, 0, &type);
}

std::uint64_t VulkanTimelineState::completed() const {
  std::lock_guard<std::mutex> lock(mutex);
  if (!semaphore) {
    check_retired();
    return retired_value;
  }
  if (reporter)
    reporter->throw_if_submission_disallowed("query Vulkan timeline");
  std::uint64_t value = 0;
  check(device.functions().vkGetSemaphoreCounterValue(
            device, semaphore->semaphore, &value),
        "vkGetSemaphoreCounterValue");
  return value;
}

void VulkanTimelineState::check_retired() const {
  if (!retirement_failed)
    return;
  if (reporter)
    reporter->throw_if_submission_disallowed("query retired Vulkan timeline");
  throw std::runtime_error(
      "Vulkan timeline retired without device completion proof");
}

void VulkanTimelineState::check(VkResult result, const char *operation) const {
  if (result == VK_SUCCESS)
    return;
  BackendRuntimeError error(Arch::vulkan, result, operation,
                            "Vulkan execution timeline failed");
  if (reporter)
    reporter->report_backend_error(error, 0);
  throw error;
}

std::shared_ptr<VulkanTimelineState> VulkanDevice::create_execution_timeline() {
  auto timeline = std::make_shared<VulkanTimelineState>(*this);
  std::lock_guard<std::mutex> lock(execution_queue_mutex_);
  execution_timelines_.erase(
      std::remove_if(execution_timelines_.begin(), execution_timelines_.end(),
                     [](const auto &entry) { return entry.use_count() == 1; }),
      execution_timelines_.end());
  execution_timelines_.push_back(timeline);
  return timeline;
}

void VulkanDevice::retire_execution_timelines(bool backend_complete) noexcept {
  // Device teardown already excludes new domain creation/submission. Keep the
  // registry's references until every native semaphore has been retired, so
  // concurrent host ticket destruction can never destroy one after the device.
  std::lock_guard<std::mutex> registry_lock(execution_queue_mutex_);
  for (const auto &timeline : execution_timelines_) {
    std::unique_lock<std::shared_mutex> backend_lock(timeline->backend_mutex);
    std::lock_guard<std::mutex> lock(timeline->mutex);
    if (!timeline->semaphore)
      continue;
    timeline->retirement_failed = !backend_complete;
    if (backend_complete)
      timeline->retired_value =
          timeline->highest_enqueued.load(std::memory_order_acquire);
    timeline->semaphore.reset();
  }
  execution_timelines_.clear();
}

VulkanTimelinePoint::VulkanTimelinePoint(
    std::shared_ptr<VulkanTimelineState> state, std::uint64_t value)
    : state_(std::move(state)), value_(value) {}

bool VulkanTimelinePoint::is_ready() const {
  return state_->completed() >= value_;
}

bool VulkanTimelinePoint::wait() const {
  if (value_ > state_->highest_enqueued.load(std::memory_order_acquire))
    throw std::invalid_argument("Vulkan timeline prefix has not been enqueued");
  std::shared_lock<std::shared_mutex> backend_lock(state_->backend_mutex);
  vkapi::IVkSemaphore semaphore;
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (!state_->semaphore) {
      state_->check_retired();
      return state_->retired_value >= value_;
    }
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
    state.timeline = device.create_execution_timeline();
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

StreamGpuTiming VulkanExecutionQueue::prepare_timing(CommandList &begin,
                                                     CommandList &end) {
  auto &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  if (!state.accepting)
    throw std::runtime_error("Vulkan execution queue is closed");
  auto *boundary = dynamic_cast<VulkanCommandList *>(&begin);
  if (!boundary)
    throw std::invalid_argument("Vulkan timing requires native commands");
  boundary->runtime_timing_barrier();
  auto timing = state.stream->begin_gpu_timing_inline(&begin);
  boundary->runtime_timing_barrier();
  state.stream->end_gpu_timing_inline(timing, &end);
  return timing;
}

std::shared_ptr<const VulkanExecutionQueue::Commands>
VulkanExecutionQueue::prepare(
    const std::vector<CommandList *> &commands,
    std::shared_ptr<void> operands,
    const std::vector<vkapi::IVkSemaphore> &binary_waits,
    const std::vector<vkapi::IVkSemaphore> &binary_signals) {
  auto &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  if (!state.accepting)
    throw std::runtime_error("Vulkan execution queue is closed");
  if (commands.size() > UINT32_MAX)
    throw std::overflow_error("Vulkan command count exceeds uint32");
  auto result = std::shared_ptr<Commands>(new Commands);
  result->device = &state.device;
  if (binary_waits.size() > UINT32_MAX || binary_signals.size() >= UINT32_MAX)
    throw std::overflow_error("Vulkan external semaphore count exceeds uint32");
  std::vector<VkSemaphore> seen;
  const auto retain = [&](const auto &input, auto &output) {
    for (const auto &semaphore : input) {
      if (!semaphore || !semaphore->semaphore ||
          semaphore->device != state.device.vk_device() ||
          semaphore->semaphore == state.timeline->semaphore->semaphore ||
          std::find(seen.begin(), seen.end(), semaphore->semaphore) !=
              seen.end())
        throw std::invalid_argument(
            "Vulkan external semaphore is foreign, duplicate, or invalid");
      seen.push_back(semaphore->semaphore);
      output.push_back(semaphore);
    }
  };
  retain(binary_waits, result->binary_waits);
  retain(binary_signals, result->binary_signals);
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

std::shared_ptr<const VulkanExecutionQueue::Commands>
VulkanExecutionQueue::prepare_external_access(
    std::shared_ptr<const Commands> body,
    CommandList &acquire,
    CommandList &release,
    const std::vector<vkapi::IVkSemaphore> &binary_waits,
    const std::vector<vkapi::IVkSemaphore> &binary_signals,
    std::shared_ptr<void> operands) {
  if (!body || body->device != &state_->device ||
      !body->binary_waits.empty() || !body->binary_signals.empty())
    throw std::invalid_argument(
        "External access requires an unbound body from this Vulkan device");
  if (body->handles.size() > UINT32_MAX - 2u)
    throw std::overflow_error("Vulkan command count exceeds uint32");
  // Keep both original closures, including on partial construction failure.
  // prepare() remains the single cold validator for family and semaphore facts.
  const auto boundaries =
      prepare({&acquire, &release}, std::move(operands), binary_waits,
              binary_signals);
  struct Closure {
    std::shared_ptr<const Commands> body, boundaries;
  };
  auto result = std::shared_ptr<Commands>(new Commands);
  result->device = body->device;
  result->operands = std::make_shared<Closure>(Closure{body, boundaries});
  result->binary_waits = boundaries->binary_waits;
  result->binary_signals = boundaries->binary_signals;
  result->buffers.reserve(body->buffers.size() + 2);
  result->handles.reserve(body->handles.size() + 2);
  result->buffers.push_back(boundaries->buffers.front());
  result->buffers.insert(result->buffers.end(), body->buffers.begin(),
                          body->buffers.end());
  result->buffers.push_back(boundaries->buffers.back());
  for (const auto &buffer : result->buffers)
    result->handles.push_back(buffer->buffer);
  return result;
}

VulkanExecutionQueue::Point VulkanExecutionQueue::submit(
    std::shared_ptr<const Commands> commands,
    const std::vector<Point> &dependencies,
    const std::function<void(std::size_t, const Point &)> &admit) {
  collect(1);
  return submit_precollected(std::move(commands), dependencies, admit);
}

VulkanExecutionQueue::Point VulkanExecutionQueue::submit_precollected(
    std::shared_ptr<const Commands> commands,
    const std::vector<Point> &dependencies,
    const std::function<void(std::size_t, const Point &)> &admit,
    const CommandSelector &select) {
  if ((!commands && !select) || dependencies.size() > UINT32_MAX)
    throw std::invalid_argument("Invalid Vulkan execution submission");
  auto &state = *state_;
  State::Slot pending;
  std::lock_guard<std::mutex> lock(state.mutex);
  state.device.throw_if_backend_submission_disallowed("submit Vulkan execution queue");
  if (state.error)
    std::rethrow_exception(state.error);
  if (!state.accepting)
    throw std::runtime_error("Vulkan execution queue is closed");
  if (commands && commands->device != &state.device)
    throw std::invalid_argument("Vulkan execution command device mismatch");
  std::vector<std::pair<VkSemaphore, std::uint64_t>> waits;
  waits.reserve(dependencies.size());
  for (const auto &point : dependencies) {
    if (!point || point->state_->device != state.device.vk_device() ||
        !point->state_->semaphore)
      throw std::invalid_argument("Vulkan dependency device is foreign or retired");
    const auto highest =
        point->state_->highest_enqueued.load(std::memory_order_acquire);
    if (point->value() > highest ||
        highest - point->value() > point->state_->max_difference)
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
  const auto index = (state.head + state.count) % state.slots.size();
  if (select)
    commands = select(index);
  if (!commands || commands->device != &state.device)
    throw std::invalid_argument("Vulkan execution command device mismatch");
  if (commands->binary_waits.size() > UINT32_MAX - semaphores.size())
    throw std::overflow_error("Vulkan submission wait count exceeds uint32");
  for (const auto &wait : commands->binary_waits) {
    if (std::find(semaphores.begin(), semaphores.end(), wait->semaphore) !=
        semaphores.end())
      throw std::invalid_argument(
          "Vulkan binary wait aliases a timeline dependency");
    semaphores.push_back(wait->semaphore);
    values.push_back(0);
  }
  const auto previous = state.tail ? state.tail->value() : 0;
  if (previous == UINT64_MAX)
    throw std::overflow_error("Vulkan timeline counter exhausted");
  auto point = Point(new VulkanTimelinePoint(state.timeline, previous + 1));
  std::vector<VkSemaphore> signals;
  std::vector<std::uint64_t> signal_values;
  if (!commands->binary_signals.empty()) {
    signals.reserve(commands->binary_signals.size() + 1);
    signal_values.reserve(commands->binary_signals.size() + 1);
    signals.push_back(state.timeline->semaphore->semaphore);
    signal_values.push_back(point->value());
  }
  for (const auto &signal : commands->binary_signals) {
    if (signal->semaphore == signals.front() ||
        std::find(semaphores.begin(), semaphores.end(), signal->semaphore) !=
            semaphores.end())
      throw std::invalid_argument(
          "Vulkan binary signal aliases a submission dependency");
    signals.push_back(signal->semaphore);
    signal_values.push_back(0);
  }
  pending.dependencies = dependencies;
  pending.commands = std::move(commands);
  pending.point = point;
  std::vector<VkPipelineStageFlags> stages(semaphores.size(),
                                          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
  if (admit)
    admit(index, point);
  auto &slot = state.slots[index];
  slot = std::move(pending);
  state.tail = point;
  ++state.count;
  VkTimelineSemaphoreSubmitInfo timeline{
      VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
  timeline.waitSemaphoreValueCount = static_cast<uint32_t>(values.size());
  timeline.pWaitSemaphoreValues = values.data();
  timeline.signalSemaphoreValueCount =
      signals.empty() ? 1 : static_cast<uint32_t>(signal_values.size());
  timeline.pSignalSemaphoreValues =
      signals.empty() ? &point->value_ : signal_values.data();
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.pNext = &timeline;
  submit.waitSemaphoreCount = static_cast<uint32_t>(semaphores.size());
  submit.pWaitSemaphores = semaphores.data();
  submit.pWaitDstStageMask = stages.data();
  submit.commandBufferCount = static_cast<uint32_t>(slot.commands->handles.size());
  submit.pCommandBuffers = slot.commands->handles.data();
  submit.signalSemaphoreCount = timeline.signalSemaphoreValueCount;
  submit.pSignalSemaphores =
      signals.empty() ? &state.timeline->semaphore->semaphore : signals.data();
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
