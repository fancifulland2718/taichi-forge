#include "taichi/rhi/vulkan/vulkan_execution_queue.h"

#include <algorithm>
#include <condition_variable>
#include <limits>
#include <shared_mutex>

#include "taichi/rhi/vulkan/vulkan_external_access.h"

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
  const VulkanExecutionQueueKind kind;
  const std::uint32_t family;
  std::uint32_t index{0};
  std::unique_ptr<VulkanStream> stream;
  std::shared_ptr<VulkanTimelineState> timeline;
  mutable std::mutex mutex;
  std::condition_variable changed;
  std::vector<Slot> slots;
  std::vector<std::weak_ptr<ForkJoin>> fork_joins;
  std::size_t head{0}, count{0};
  Point tail;
  Point retirement_prefix;
  bool uncertain_submission{false};
  bool retirement_proven{false};
  bool accepting{true}, closing{false}, closed{false};
  std::exception_ptr error;
  State(VulkanDevice &device, std::size_t capacity, VulkanExecutionQueueKind kind)
      : device(device), kind(kind), family(device.execution_queue_family_index(kind)), slots(capacity) {}
};

struct VulkanExecutionQueue::ForkJoin::State {
  std::shared_ptr<VulkanTimelineState> anchor;
  std::uint32_t anchor_index;
  // These queues are private to the topology. Their ordinary slot rings are
  // unused: the anchor retains every operand in one logical compound slot.
  std::vector<std::unique_ptr<VulkanExecutionQueue>> auxiliaries;
};

struct VulkanExecutionQueue::Commands::Parallel {
  std::shared_ptr<ForkJoin> topology;
  std::shared_ptr<const Commands> prefix, suffix;
  std::vector<std::shared_ptr<const Commands>> branches;
};

VulkanExecutionQueue::ForkJoin::ForkJoin(std::unique_ptr<State> state)
    : state_(std::move(state)) {}
VulkanExecutionQueue::ForkJoin::~ForkJoin() = default;

std::vector<std::uint32_t>
VulkanExecutionQueue::ForkJoin::queue_indices() const {
  std::vector<std::uint32_t> result{state_->anchor_index};
  for (const auto &queue : state_->auxiliaries)
    result.push_back(queue->queue_index());
  return result;
}

VulkanExecutionQueue::VulkanExecutionQueue(VulkanDevice &device,
                                           std::size_t capacity,
                                           VulkanExecutionQueueKind kind) {
  if (!capacity || !device.vk_caps().timeline_semaphore)
    throw std::invalid_argument(
        "Vulkan execution queues require positive capacity and timeline support");
  device.throw_if_backend_submission_disallowed("create Vulkan execution queue");
  state_ = std::make_unique<State>(device, capacity, kind);
  auto &state = *state_;
  {
    std::lock_guard<std::mutex> lock(device.execution_queue_mutex_);
    auto &pool = device.execution_queue_pool(kind);
    for (std::size_t i = 0; i < pool.size(); ++i) {
      auto &queue = *pool[i];
      if (!queue.leased) {
        queue.leased = true;
        state.index = static_cast<std::uint32_t>(i + 1);
        break;
      }
    }
  }
  if (!state.index)
    throw std::runtime_error(kind == VulkanExecutionQueueKind::graphics
        ? "No independent Vulkan graphics/compute queue is available"
        : "No independent Vulkan compute queue is available");
  try {
    state.stream = device.create_execution_stream(kind, state.index);
    state.timeline = device.create_execution_timeline();
    if (capacity > state.timeline->max_difference)
      throw std::invalid_argument(
          "Capacity exceeds the device timeline value-difference limit");

  } catch (...) {
    std::lock_guard<std::mutex> lock(device.execution_queue_mutex_);
    device.execution_queue_pool(kind)[state.index - 1]->leased = false;
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
  result->family = state.family;
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
        buffer->ref_pool->queue_family_index != state.family)
      throw std::invalid_argument("Vulkan execution command device/family mismatch");
    result->handles.push_back(buffer->buffer);
    result->buffers.push_back(std::move(buffer));
  }
  return result;
}

std::shared_ptr<const VulkanExecutionQueue::Commands>
VulkanExecutionQueue::prepare_fork_join_commands(
    std::shared_ptr<ForkJoin> topology,
    std::shared_ptr<const Commands> prefix,
    const std::vector<std::shared_ptr<const Commands>> &branches,
    std::shared_ptr<const Commands> suffix,
    std::shared_ptr<void> operands) {
  // All caller-owned references outlive the lock, including rejected operands.
  auto result = std::shared_ptr<Commands>(new Commands);
  auto parallel = std::make_shared<Commands::Parallel>();
  parallel->topology = topology;
  parallel->prefix = prefix;
  parallel->suffix = suffix;
  parallel->branches = branches;
  result->operands = operands;
  result->parallel = parallel;
  auto &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  if (!state.accepting)
    throw std::runtime_error("Vulkan execution queue is closed");
  if (!topology || topology->state_->anchor != state.timeline ||
      branches.size() != topology->state_->auxiliaries.size() + 1)
    throw std::invalid_argument("Vulkan fork/join topology belongs to another anchor or width");
  const auto validate = [&](const auto &part) {
    if (!part || part->device != &state.device || part->family != state.family || part->parallel ||
        part->external_access || !part->binary_waits.empty() ||
        !part->binary_signals.empty())
      throw std::invalid_argument("Vulkan fork/join requires flat commands without external handshakes");
  };
  validate(prefix);
  validate(suffix);
  for (const auto &branch : branches) {
    validate(branch);
    if (branch->handles.empty())
      throw std::invalid_argument("Vulkan fork/join branches must be nonempty");
  }
  result->device = &state.device;
  result->family = state.family;
  return result;
}

std::shared_ptr<VulkanExecutionQueue::ForkJoin>
VulkanExecutionQueue::prepare_fork_join(std::size_t branch_count) {
  auto &state = *state_;
  auto topology = std::make_unique<ForkJoin::State>();
  std::size_t capacity;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.accepting)
      throw std::runtime_error("Vulkan execution queue is closed");
    if (branch_count < 2 || branch_count > UINT32_MAX)
      throw std::invalid_argument("Vulkan fork/join requires at least two branches");
    capacity = state.slots.size();
    // Prefix, branch zero and final join advance the anchor timeline three
    // times. The public capacity still counts complete invocations, not phases.
    if (capacity > state.timeline->max_difference / 3)
      throw std::invalid_argument("Vulkan fork/join capacity exceeds timeline value-difference limit");
    topology->anchor = state.timeline;
    topology->anchor_index = state.index;
  }
  // Acquire the complete width before publishing the topology. Any failure
  // destroys these unused owners and returns all acquired physical leases.
  // Construction/rollback never occurs under the anchor admission lock.
  {
    std::lock_guard<std::mutex> lock(state.device.execution_queue_mutex_);
    const auto &pool = state.device.execution_queue_pool(state.kind);
    const auto available = std::count_if(
        pool.begin(), pool.end(),
        [](const auto &queue) { return !queue->leased; });
    if (branch_count - 1 > static_cast<std::size_t>(available))
      throw std::runtime_error("Insufficient independent Vulkan compute queues for fork/join: requested auxiliary=" +
                               std::to_string(branch_count - 1) + ", available=" +
                               std::to_string(available));
  }
  topology->auxiliaries.reserve(branch_count - 1);
  for (std::size_t i = 1; i < branch_count; ++i)
    topology->auxiliaries.push_back(
        std::make_unique<VulkanExecutionQueue>(state.device, capacity, state.kind));
  auto result = std::shared_ptr<ForkJoin>(new ForkJoin(std::move(topology)));
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.accepting)
      throw std::runtime_error("Vulkan execution queue closed during fork/join preparation");
    state.fork_joins.erase(
        std::remove_if(state.fork_joins.begin(), state.fork_joins.end(),
                       [](const auto &entry) { return entry.expired(); }),
        state.fork_joins.end());
    state.fork_joins.push_back(result);
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
  if (!body || body->device != &state_->device || body->family != state_->family ||
      body->parallel || body->external_access || !body->binary_waits.empty() ||
      !body->binary_signals.empty())
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
  result->family = body->family;
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

std::shared_ptr<const VulkanExecutionQueue::Commands>
VulkanExecutionQueue::prepare_external_access(
    std::shared_ptr<const Commands> body,
    std::shared_ptr<const VulkanExternalAccess> access) {
  if (!access)
    throw std::invalid_argument("Null Vulkan external access plan");
  // Managed external epochs currently identify queues in the compute pool.
  // Do not reinterpret a family-local graphics index as a compute identity.
  if (state_->kind != VulkanExecutionQueueKind::compute)
    throw std::invalid_argument("Managed Vulkan external access requires a compute domain");
  if (!body || body->device != &state_->device || body->family != state_->family || body->parallel || body->external_access ||
      !body->binary_waits.empty() || !body->binary_signals.empty())
    throw std::invalid_argument(
        "Managed access requires an unbound body from this Vulkan device");
  access->validate_queue(state_->device, state_->index);
  if (!access->requires_import_boundaries()) {
    // Local storage has no OS ownership transition or binary handshake. Cold
    // host composition retains the same body and adds only atomic admission.
    auto result = std::shared_ptr<Commands>(new Commands(*body));
    result->external_access = std::move(access);
    return result;
  }
  auto acquire = new_command_list();
  auto release = new_command_list();
  std::vector<vkapi::IVkSemaphore> waits, signals;
  access->record(*static_cast<VulkanCommandList *>(acquire.get()),
                 *static_cast<VulkanCommandList *>(release.get()), waits,
                 signals);
  auto result = std::const_pointer_cast<Commands>(prepare_external_access(
      std::move(body), *acquire, *release, waits, signals));
  result->external_access = std::move(access);
  return result;
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
  if (commands && (commands->device != &state.device || commands->family != state.family))
    throw std::invalid_argument("Vulkan execution command device/family mismatch");
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
  if (!commands || commands->device != &state.device || commands->family != state.family)
    throw std::invalid_argument("Vulkan execution command device/family mismatch");
  if (commands->parallel)
    return submit_fork_join_locked(commands, dependencies,
                                  std::move(semaphores), std::move(values), admit);
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
  VulkanExternalAccess::Transaction external(pending.commands->external_access.get());
  // In particular, not-ready rejects before the runtime's admit constructs a
  // completion/sequence or writes its selected parameter slot.
  external.reserve(state.device, state.index);
  if (admit)
    admit(index, point);
  external.arm(point);
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
    const auto queue = state.device.execution_queue_at(state.kind, state.index);
    auto queue_lock = state.device.acquire_queue_lock(queue);
    state.uncertain_submission = true;
    state.retirement_proven = false;
    state.timeline->check(state.device.vk_device().functions().vkQueueSubmit(
                              queue, 1, &submit, VK_NULL_HANDLE),
                          "vkQueueSubmit");
    state.timeline->highest_enqueued.store(point->value(), std::memory_order_release);
    state.retirement_prefix = point;
    state.uncertain_submission = false;
    state.device.queue_tail_fence_locked(queue).reset();
    state.device.queue_submit_calls_.fetch_add(1, std::memory_order_relaxed);
    state.device.submitted_command_buffers_.fetch_add(
        slot.commands->handles.size(), std::memory_order_relaxed);
  } catch (...) {
    state.error = std::current_exception();
    state.accepting = false;
    throw;
  }
  external.seal();  // highest_enqueued is visible before owner close can wait.
  return point;
}

VulkanExecutionQueue::Point VulkanExecutionQueue::submit_fork_join_locked(
    std::shared_ptr<const Commands> commands,
    const std::vector<Point> &dependencies,
    std::vector<VkSemaphore> semaphores,
    std::vector<std::uint64_t> values,
    const std::function<void(std::size_t, const Point &)> &admit) {
  auto &state = *state_;
  const auto &parallel = *commands->parallel;
  auto &topology = *parallel.topology->state_;
  if (topology.anchor != state.timeline)
    throw std::invalid_argument("Vulkan fork/join submitted on a foreign anchor");
  const auto previous = state.tail ? state.tail->value() : 0;
  if (previous > UINT64_MAX - 3)
    throw std::overflow_error("Vulkan fork/join timeline counter exhausted");
  if (semaphores.size() == UINT32_MAX)
    throw std::overflow_error("Vulkan fork/join dependency count exceeds uint32");
  // Queue order alone is not a memory dependency. Every invocation depends
  // explicitly on the prior complete anchor prefix before reusing operands.
  if (state.tail) {
    semaphores.push_back(state.timeline->semaphore->semaphore);
    values.push_back(previous);
    // The user's dependency may already name this timeline. Keep its greatest
    // value and never submit the same semaphore twice in one wait list.
    for (std::size_t i = 0; i + 1 < semaphores.size(); ++i)
      if (semaphores[i] == semaphores.back()) {
        values[i] = std::max(values[i], values.back());
        semaphores.pop_back();
        values.pop_back();
        break;
      }
  }
  const auto prefix = Point(new VulkanTimelinePoint(state.timeline, previous + 1));
  const auto branch_zero = Point(new VulkanTimelinePoint(state.timeline, previous + 2));
  const auto final = Point(new VulkanTimelinePoint(state.timeline, previous + 3));
  std::vector<Point> branch_points{branch_zero};
  std::vector<VkSemaphore> join_semaphores{state.timeline->semaphore->semaphore};
  std::vector<std::uint64_t> join_values{branch_zero->value()};
  for (const auto &queue : topology.auxiliaries) {
    auto &aux = *queue->state_;
    if (aux.error)
      std::rethrow_exception(aux.error);
    if (!aux.accepting || aux.closed || aux.closing)
      throw std::runtime_error("Vulkan fork/join auxiliary queue is closed");
    const auto tail = aux.tail ? aux.tail->value() : 0;
    if (tail == UINT64_MAX)
      throw std::overflow_error("Vulkan fork/join branch timeline counter exhausted");
    branch_points.push_back(Point(new VulkanTimelinePoint(aux.timeline, tail + 1)));
    join_semaphores.push_back(aux.timeline->semaphore->semaphore);
    join_values.push_back(tail + 1);
  }
  std::vector<VkPipelineStageFlags> prefix_stages(semaphores.size(),
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
  std::vector<VkPipelineStageFlags> join_stages(join_semaphores.size(),
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
  const auto stage = VkPipelineStageFlags(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
  const auto anchor_semaphore = state.timeline->semaphore->semaphore;
  // All submit structures and backing arrays are stable before admit. The
  // internal points never escape; only the final point can be published.
  struct Submission {
    VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  };
  std::vector<Submission> prepared(parallel.branches.size() + 2);
  const auto configure = [&](std::size_t i, const Commands &part,
                             std::uint32_t wait_count, const VkSemaphore *waits,
                             const std::uint64_t *wait_values,
                             const VkPipelineStageFlags *stages,
                             const VkSemaphore *signal, const std::uint64_t *value) {
    auto &submission = prepared[i];
    submission.timeline.waitSemaphoreValueCount = wait_count;
    submission.timeline.pWaitSemaphoreValues = wait_values;
    submission.timeline.signalSemaphoreValueCount = 1;
    submission.timeline.pSignalSemaphoreValues = value;
    submission.info.pNext = &submission.timeline;
    submission.info.waitSemaphoreCount = wait_count;
    submission.info.pWaitSemaphores = waits;
    submission.info.pWaitDstStageMask = stages;
    submission.info.commandBufferCount = static_cast<std::uint32_t>(part.handles.size());
    submission.info.pCommandBuffers = part.handles.data();
    submission.info.signalSemaphoreCount = 1;
    submission.info.pSignalSemaphores = signal;
  };
  configure(0, *parallel.prefix, static_cast<std::uint32_t>(semaphores.size()),
            semaphores.data(), values.data(), prefix_stages.data(),
            &anchor_semaphore, &prefix->value_);
  for (std::size_t i = 0; i < parallel.branches.size(); ++i)
    configure(i + 1, *parallel.branches[i], 1, &anchor_semaphore,
              &prefix->value_, &stage, &join_semaphores[i], &join_values[i]);
  configure(prepared.size() - 1, *parallel.suffix,
            static_cast<std::uint32_t>(join_semaphores.size()),
            join_semaphores.data(), join_values.data(), join_stages.data(),
            &anchor_semaphore, &final->value_);
  State::Slot pending;
  pending.commands = std::move(commands);
  pending.dependencies = dependencies;
  pending.point = final;
  const auto index = (state.head + state.count) % state.slots.size();
  if (admit)
    admit(index, final);
  state.slots[index] = std::move(pending);
  state.tail = final;
  ++state.count;

  const auto enqueue = [&](State &target, const Submission &submission,
                           const Point &point) {
    // A native error can leave device effects uncertain. Retain its attempted
    // point and mark the exact queue before rethrow; close must drain it rather
    // than wait a signal that may never have been enqueued.
    target.tail = point;
    try {
      const auto queue = target.device.execution_queue_at(target.kind, target.index);
      auto queue_lock = target.device.acquire_queue_lock(queue);
      target.uncertain_submission = true;
      target.retirement_proven = false;
      target.timeline->check(target.device.vk_device().functions().vkQueueSubmit(
          queue, 1, &submission.info, VK_NULL_HANDLE), "vkQueueSubmit");
      target.timeline->highest_enqueued.store(point->value(), std::memory_order_release);
      target.retirement_prefix = point;
      target.uncertain_submission = false;
      target.device.queue_tail_fence_locked(queue).reset();
      target.device.queue_submit_calls_.fetch_add(1, std::memory_order_relaxed);
      target.device.submitted_command_buffers_.fetch_add(
          submission.info.commandBufferCount, std::memory_order_relaxed);
    } catch (...) {
      target.error = std::current_exception();
      target.accepting = false;
      throw;
    }
  };
  try {
    enqueue(state, prepared[0], prefix);
    enqueue(state, prepared[1], branch_zero);
    for (std::size_t i = 0; i < topology.auxiliaries.size(); ++i)
      enqueue(*topology.auxiliaries[i]->state_, prepared[i + 2], branch_points[i + 1]);
    enqueue(state, prepared.back(), final);
  } catch (...) {
    // The retained slot still owns the final *unsealed* point and every branch
    // operand. Do not publish it, do not submit a join for a failed branch, and
    // do not let collection treat an earlier successful anchor point as final.
    state.tail = final;
    state.error = std::current_exception();
    state.accepting = false;
    throw;
  }
  return final;
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
std::uint32_t VulkanExecutionQueue::queue_family_index() const { return state_->family; }
VulkanExecutionQueue::Point VulkanExecutionQueue::latest() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->tail;
}
void VulkanExecutionQueue::stop() {
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->accepting = false;
}

void VulkanExecutionQueue::wait_retirement_prefix(const Point &point) {
  if (!point)
    return;
  const auto &timeline = point->state_;
  if (point->value() > timeline->highest_enqueued.load(std::memory_order_acquire))
    throw std::runtime_error("Cannot retire an unsealed Vulkan prefix");
  std::shared_lock<std::shared_mutex> backend_lock(timeline->backend_mutex);
  vkapi::IVkSemaphore semaphore;
  {
    std::lock_guard<std::mutex> lock(timeline->mutex);
    if (!timeline->semaphore) {
      timeline->check_retired();
      if (timeline->retired_value < point->value())
        throw std::runtime_error("Vulkan retired prefix has no completion proof");
      return;
    }
    semaphore = timeline->semaphore;
  }
  // Public queries/submissions stay fail-stop after a sibling native failure.
  // Internal teardown may still obtain a native proof for already-enqueued
  // work. Keep the same semaphore/backend lifetime locks as Point::wait().
  VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  wait.semaphoreCount = 1;
  wait.pSemaphores = &semaphore->semaphore;
  wait.pValues = &point->value_;
  timeline->check(timeline->device.functions().vkWaitSemaphores(
                     timeline->device, &wait, UINT64_MAX), "vkWaitSemaphores");
}

void VulkanExecutionQueue::close() {
  auto &state = *state_;
  {
    std::unique_lock<std::mutex> lock(state.mutex);
    state.changed.wait(lock, [&] { return !state.closing; });
    if (state.closed)
      return;
    state.accepting = false;
    state.closing = true;
  }
  std::vector<State::Slot> retired;
  try {
    prove_retirement();
    // An unsuccessful compound join proves nothing about auxiliary queues.
    // Keep the anchor slots intact until all relevant queue owners prove their
    // own tails/drains. A failed proof leaves the entire operand closure here.
    for (const auto &entry : state.fork_joins)
      if (const auto topology = entry.lock())
        for (const auto &aux : topology->state_->auxiliaries)
          aux->prove_retirement();
    // Prove the entire closure before returning any topology lease. A failed
    // proof leaves even successfully proven siblings owned until close retries.
    for (const auto &entry : state.fork_joins)
      if (const auto topology = entry.lock())
        for (const auto &aux : topology->state_->auxiliaries)
          aux->close();
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      retired = std::move(state.slots);
      state.count = 0;
    }
    // A failed enqueue has no signal proof. Only the successful queue drain
    // above permits its external owners to retire; they remain fail-stop.
    if (state.error)
      for (const auto &slot : retired)
        if (slot.commands && slot.commands->external_access)
          slot.commands->external_access->confirm_drained(slot.point);
    retired.clear();
    state.stream.reset();
    {
      std::lock_guard<std::mutex> lock(state.device.execution_queue_mutex_);
      state.device.execution_queue_pool(state.kind)[state.index - 1]->leased = false;
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

void VulkanExecutionQueue::prove_retirement() {
  auto &state = *state_;
  if (state.retirement_proven)
    return;
  if (state.uncertain_submission) {
    const auto queue = state.device.execution_queue_at(state.kind, state.index);
    auto queue_lock = state.device.acquire_queue_lock(queue);
    state.timeline->check(state.device.vk_device().functions().vkQueueWaitIdle(queue),
                          "vkQueueWaitIdle");
  } else if (state.error) {
    wait_retirement_prefix(state.retirement_prefix);
  } else {
    wait_retirement_prefix(state.tail);
  }
  state.retirement_proven = true;
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
