#include "gtest/gtest.h"
#include "tests/cpp/aot/vulkan/timeline_gate.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_execution_queue.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/codegen/spirv/spirv_ir_builder.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/program/runtime_completion.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/runtime/completion_wait_set.h"
#include "taichi/runtime/vulkan_execution_domain.h"
#include "taichi/runtime/fixed_execution_vulkan.h"
#include "taichi/runtime/gfx/aot_graph_data.h"
#include "taichi/runtime/gfx/aot_module_loader_impl.h"
#include "taichi/runtime/native_operation.h"
#include "taichi/program/ndarray.h"

#include <chrono>
#include <set>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <array>
#include <atomic>
#include <algorithm>

namespace taichi::lang {

TEST(VulkanExecutionQueueFamilies, GraphicsLeaseBusyAndCrossFamilyAdmission) {
  using Kind = vulkan::VulkanExecutionQueueKind;
  using Queue = vulkan::VulkanExecutionQueue;
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  if (device.created_execution_queue_count(Kind::graphics) < 3 ||
      device.created_compute_queue_count() < 2 || !device.vk_caps().timeline_semaphore)
    GTEST_SKIP() << "Requires two additional graphics/compute queues and one compute queue";
  Queue graphics(device, 1, Kind::graphics), peer(device, 1, Kind::graphics);
  Queue compute(device, 1);
  RecordProperty("graphics_family", graphics.queue_family_index());
  RecordProperty("compute_family", compute.queue_family_index());
  EXPECT_EQ(graphics.queue_family_index(), device.graphics_queue_family_index());
  EXPECT_NE(device.execution_queue_at(Kind::graphics, graphics.queue_index()), device.graphics_queue());
  EXPECT_NE(device.execution_queue_at(Kind::graphics, graphics.queue_index()),
            device.execution_queue_at(Kind::graphics, peer.queue_index()));
  auto [storage, status] = device.allocate_memory_unique(
      {4, false, true, false, AllocUsage::Storage});
  ASSERT_EQ(status, RhiResult::success);
  auto command = graphics.new_command_list();
  command->buffer_fill(storage->get_ptr(), 4, 73);
  command->memory_barrier();
  auto commands = graphics.prepare({command.get()});
  if (graphics.queue_family_index() != compute.queue_family_index()) {
    EXPECT_THROW(compute.prepare({command.get()}), std::invalid_argument);
    bool admitted = false;
    EXPECT_THROW(compute.submit(commands, {}, [&](auto, const auto &) { admitted = true; }),
                 std::invalid_argument);
    EXPECT_FALSE(admitted);
    EXPECT_EQ(compute.pending(), 0u);
  }
  {
    test_detail::TimelineGate gate(device, graphics.queue_index(), Kind::graphics);
    auto blocked = graphics.submit(commands);
    EXPECT_FALSE(blocked->is_ready());
    bool admitted = false;
    EXPECT_THROW(graphics.submit(commands, {}, [&](auto, const auto &) { admitted = true; }), Queue::Busy);
    EXPECT_FALSE(admitted);
    EXPECT_EQ(graphics.latest(), blocked);
    auto peer_done = peer.submit(peer.prepare({}));
    auto compute_done = compute.submit(compute.prepare({}));
    // Host deadline is only a test failure bound, never a workload limit.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while ((!peer_done->is_ready() || !compute_done->is_ready()) &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    EXPECT_TRUE(peer_done->is_ready());
    EXPECT_TRUE(compute_done->is_ready());
    EXPECT_FALSE(blocked->is_ready());
    gate.release();
    EXPECT_TRUE(blocked->wait());
  }
  void *mapped = nullptr;
  ASSERT_EQ(device.map(*storage, &mapped), RhiResult::success);
  EXPECT_EQ(*static_cast<const uint32_t *>(mapped), 73u);
  device.unmap(*storage);
  const auto released = graphics.queue_index();
  graphics.close();
  Queue reused(device, 1, Kind::graphics);
  EXPECT_EQ(reused.queue_index(), released);
  EXPECT_TRUE(reused.submit(reused.prepare({}))->wait());
}

TEST(VulkanExecutionQueueFamilies, ImportedSharedFamilyUsesOnePhysicalLeasePool) {
  using Kind = vulkan::VulkanExecutionQueueKind;
  using Queue = vulkan::VulkanExecutionQueue;
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  if (device.created_execution_queue_count(Kind::graphics) < 3 || !device.vk_caps().timeline_semaphore)
    GTEST_SKIP();
  // A real imported VkDevice with only the explicitly supplied queues. Both
  // roles use its universal family; no queue is fetched behind the owner's back.
  vulkan::VulkanDevice::Params params;
  params.instance = device.vk_instance();
  params.physical_device = device.vk_physical_device();
  params.device = device.vk_device();
  params.compute_queue = params.graphics_queue = device.graphics_queue();
  params.compute_queue_family_index = params.graphics_queue_family_index = device.graphics_queue_family_index();
  params.timeline_semaphore = true;
  params.additional_compute_queues = {
      device.execution_queue_at(Kind::graphics, 1), device.execution_queue_at(Kind::graphics, 2)};
  vulkan::VulkanDevice imported;
  imported.vk_caps() = device.vk_caps();
  imported.init_vulkan_structs(params);
  EXPECT_EQ(imported.created_execution_queue_count(Kind::graphics), 3u);
  Queue compute(imported, 1), graphics(imported, 1, Kind::graphics);
  EXPECT_EQ(compute.queue_family_index(), graphics.queue_family_index());
  EXPECT_NE(compute.queue_index(), graphics.queue_index());
  EXPECT_THROW(Queue(imported, 1), std::runtime_error);
  EXPECT_THROW(Queue(imported, 1, Kind::graphics), std::runtime_error);
  auto command = compute.new_command_list();
  auto shared = compute.prepare({command.get()});
  EXPECT_TRUE(graphics.submit(shared)->wait());
  const auto released = compute.queue_index();
  compute.close();
  Queue replacement(imported, 1, Kind::graphics);
  EXPECT_EQ(replacement.queue_index(), released);
  EXPECT_TRUE(replacement.submit(shared)->wait());
}

TEST(VulkanExecutionQueues, UsesCreatedSameFamilyCapacityAndIndependentTails) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  const auto count = device.created_compute_queue_count();
  RecordProperty("created_compute_queues", count);
  EXPECT_GE(count, 1u);
  EXPECT_EQ(device.compute_queue_at(0), device.compute_queue());
  EXPECT_THROW(device.create_compute_stream(count), std::out_of_range);
  std::set<VkQueue> queues;
  for (uint32_t index = 0; index < count; ++index)
    EXPECT_TRUE(queues.insert(device.compute_queue_at(index)).second);
  if (count < 2 || !device.vk_caps().timeline_semaphore)
    GTEST_SKIP() << "Requires an additional queue and enabled timeline semaphores";

  auto additional = device.create_compute_stream(count - 1);
  auto *ordinary = static_cast<vulkan::VulkanStream *>(device.get_compute_stream());
  auto [storage, allocation] = device.allocate_memory_unique(
      {4, false, true, false, AllocUsage::Storage});
  ASSERT_EQ(allocation, RhiResult::success);
  // A host-signaled timeline keeps one queue pending without occupying an
  // execution engine. Binary semaphore waits cannot precede their signal
  // submission, so they cannot implement this controlled dependency.
  for (bool block_ordinary : {true, false}) {
    auto *blocked = block_ordinary ? ordinary : additional.get();
    auto *peer = block_ordinary ? additional.get() : ordinary;
    auto [commands, result] = peer->new_command_list_unique();
    ASSERT_EQ(result, RhiResult::success);
    const uint32_t expected = block_ordinary ? 37 : 91;
    commands->buffer_fill(storage->get_ptr(), 4, expected);
    commands->memory_barrier();
    const auto &vk = device.vk_device();
    VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo create{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    create.pNext = &type;
    VkSemaphore timeline = VK_NULL_HANDLE;
    ASSERT_EQ(vk.functions().vkCreateSemaphore(vk, &create, nullptr, &timeline),
              VK_SUCCESS);
    // Cleanup releases the dependency before any stream or device teardown,
    // including on assertions/exceptions after a successful queue submission.
    auto gate = std::shared_ptr<void>(reinterpret_cast<void *>(1), [&](void *) {
      VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
      signal.semaphore = timeline;
      signal.value = 1;
      EXPECT_EQ(vk.functions().vkSignalSemaphore(vk, &signal), VK_SUCCESS);
      blocked->command_sync();
      peer->command_sync();
      vk.functions().vkDestroySemaphore(vk, timeline, nullptr);
    });
    const uint64_t value = 1;
    VkTimelineSemaphoreSubmitInfo values{
        VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    values.waitSemaphoreValueCount = 1;
    values.pWaitSemaphoreValues = &value;
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo wait{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    wait.pNext = &values;
    wait.waitSemaphoreCount = 1;
    wait.pWaitSemaphores = &timeline;
    wait.pWaitDstStageMask = &stage;
    const auto queue = device.compute_queue_at(block_ordinary ? 0 : count - 1);
    ASSERT_EQ(vk.functions().vkQueueSubmit(queue, 1, &wait, VK_NULL_HANDLE),
              VK_SUCCESS);
    const auto pending = blocked->submit(nullptr);
    StreamSemaphore completed;
    bool peer_ready = false, still_pending = false, own_tail = false;
    {
      completed = peer->submit(commands.get());
      const auto deadline = std::chrono::steady_clock::now() +
                            std::chrono::seconds(3);
      while (!(peer_ready = completed->is_ready()) &&
             std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      still_pending = !pending->is_ready();
      own_tail = peer->is_last_submission(completed) &&
                 blocked->is_last_submission(pending) &&
                 !peer->is_last_submission(pending);
    }
    gate.reset();
    EXPECT_TRUE(peer_ready);
    EXPECT_TRUE(still_pending);
    EXPECT_TRUE(own_tail);
    void *mapped = nullptr;
    ASSERT_EQ(device.map(*storage, &mapped), RhiResult::success);
    EXPECT_EQ(*static_cast<const uint32_t *>(mapped), expected);
    device.unmap(*storage);
  }
}

}  // namespace taichi::lang

namespace taichi::lang {
namespace {
class DeferredForkGate {
 public:
  explicit DeferredForkGate(vulkan::VulkanDevice &device)
      : device_(device.vk_device()) {
    VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    semaphore_ = vkapi::create_semaphore(device_, 0, &type);
    fence_ = vkapi::create_fence(device_, 0);
  }
  VkResult arm(VkQueue queue, PFN_vkQueueSubmit submit) {
    const std::uint64_t value = 1;
    VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    timeline.waitSemaphoreValueCount = 1;
    timeline.pWaitSemaphoreValues = &value;
    const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    info.pNext = &timeline;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &semaphore_->semaphore;
    info.pWaitDstStageMask = &stage;
    const auto result = submit(queue, 1, &info, fence_->fence);
    armed_ = result == VK_SUCCESS;
    return result;
  }
  void release() {
    if (released_)
      return;
    VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
    signal.semaphore = semaphore_->semaphore;
    signal.value = 1;
    EXPECT_EQ(device_.functions().vkSignalSemaphore(device_, &signal), VK_SUCCESS);
    released_ = true;
  }
  ~DeferredForkGate() {
    release();
    if (armed_)
      device_.functions().vkWaitForFences(device_, 1, &fence_->fence, VK_TRUE, UINT64_MAX);
  }
 private:
  vkapi::DeviceHandle device_;
  vkapi::IVkSemaphore semaphore_;
  vkapi::IVkFence fence_;
  bool armed_{false}, released_{false};
};

// Observe actual native signals and optionally reject one native submission.
// No production dispatch/completion is replaced with a synthetic success.
class ForkJoinSubmitProbe {
 public:
  struct Signal { VkSemaphore semaphore; std::uint64_t value; };
  explicit ForkJoinSubmitProbe(vulkan::VulkanDevice &device, int fail_at = 0,
                               VkResult failure = VK_ERROR_OUT_OF_HOST_MEMORY)
      : api_(const_cast<VolkDeviceTable &>(device.vk_device().functions())),
        original_(api_.vkQueueSubmit), fail_at_(fail_at), failure_(failure) {
    if (active_)
      throw std::runtime_error("Nested fork/join submit probe");
    active_ = this;
    api_.vkQueueSubmit = hook;
  }
  ~ForkJoinSubmitProbe() { api_.vkQueueSubmit = original_; active_ = nullptr; }
  int calls{0};
  DeferredForkGate *gate_after_prefix{nullptr};
  std::array<Signal, 32> signals{};
 private:
  static VKAPI_ATTR VkResult VKAPI_CALL hook(
      VkQueue queue, std::uint32_t count, const VkSubmitInfo *info, VkFence fence) {
    auto &probe = *active_;
    const auto index = probe.calls++;
    if (probe.fail_at_ && probe.calls == probe.fail_at_)
      return probe.failure_;
    const auto result = probe.original_(queue, count, info, fence);
    if (result == VK_SUCCESS && index < int(probe.signals.size()) && count == 1 &&
        info->signalSemaphoreCount == 1 && info->pNext) {
      const auto *timeline = static_cast<const VkTimelineSemaphoreSubmitInfo *>(info->pNext);
      if (timeline->sType == VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO &&
          timeline->signalSemaphoreValueCount == 1)
        probe.signals[index] = {info->pSignalSemaphores[0], timeline->pSignalSemaphoreValues[0]};
    }
    if (result == VK_SUCCESS && index == 0 && probe.gate_after_prefix)
      return probe.gate_after_prefix->arm(queue, probe.original_);
    return result;
  }
  inline static ForkJoinSubmitProbe *active_{nullptr};
  VolkDeviceTable &api_;
  PFN_vkQueueSubmit original_;
  int fail_at_;
  VkResult failure_;
};

class ForkJoinDrainProbe {
 public:
  explicit ForkJoinDrainProbe(vulkan::VulkanDevice &device, VkQueue failed)
      : api_(const_cast<VolkDeviceTable &>(device.vk_device().functions())),
        original_(api_.vkQueueWaitIdle), failed_(failed) {
    active_ = this;
    api_.vkQueueWaitIdle = hook;
  }
  ~ForkJoinDrainProbe() { api_.vkQueueWaitIdle = original_; active_ = nullptr; }
  int calls{0};
  std::array<VkQueue, 16> queues{};
 private:
  static VKAPI_ATTR VkResult VKAPI_CALL hook(VkQueue queue) {
    auto &probe = *active_;
    if (probe.calls < int(probe.queues.size()))
      probe.queues[probe.calls] = queue;
    ++probe.calls;
    return queue == probe.failed_ ? VK_ERROR_DEVICE_LOST : probe.original_(queue);
  }
  inline static ForkJoinDrainProbe *active_{nullptr};
  VolkDeviceTable &api_;
  PFN_vkQueueWaitIdle original_;
  VkQueue failed_;
};

bool fork_signal_ready(vulkan::VulkanDevice &device,
                       const ForkJoinSubmitProbe::Signal &signal) {
  if (!signal.semaphore)
    return false;
  std::uint64_t value = 0;
  const auto &vk = device.vk_device();
  return vk.functions().vkGetSemaphoreCounterValue(vk, signal.semaphore, &value) == VK_SUCCESS &&
         value >= signal.value;
}

bool wait_fork_signal(vulkan::VulkanDevice &device,
                      const ForkJoinSubmitProbe::Signal &signal) {
  const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (!fork_signal_ready(device, signal) && std::chrono::steady_clock::now() < end)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  return fork_signal_ready(device, signal);
}
}  // namespace

TEST(VulkanForkJoin, OneLogicalSlotAllBranchesAndSharedVariants) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  const auto count = device.created_compute_queue_count();
  if (!device.vk_caps().timeline_semaphore || count < 3)
    GTEST_SKIP() << "Requires two non-default compute queues";
  using Queue = vulkan::VulkanExecutionQueue;
  Queue queue(device, 1);
  const auto width = std::min<std::uint32_t>(3, count - 1);
  RecordProperty("branches", width);
  auto topology = queue.prepare_fork_join(width);
  const auto indices = topology->queue_indices();
  ASSERT_EQ(indices.size(), width);
  std::set<std::uint32_t> unique(indices.begin(), indices.end());
  EXPECT_EQ(unique.size(), width);
  EXPECT_FALSE(unique.count(0));
  auto [storage, status] = device.allocate_memory_unique(
      {8 * width, false, true, false, AllocUsage::Storage});
  ASSERT_EQ(status, RhiResult::success);
  std::vector<std::shared_ptr<const Queue::Commands>> branches;
  for (std::uint32_t i = 0; i < width; ++i) {
    auto commands = queue.new_command_list();
    commands->buffer_fill(storage->get_ptr(4 * i), 4, 30 + i);
    commands->memory_barrier();
    branches.push_back(queue.prepare({commands.get()}));
  }
  auto suffix = queue.new_command_list();
  suffix->buffer_copy(storage->get_ptr(4 * width), storage->get_ptr(), 4 * width);
  suffix->memory_barrier();
  auto body = queue.prepare_fork_join_commands(
      topology, queue.prepare({}), branches, queue.prepare({suffix.get()}));
  auto variant = queue.prepare_fork_join_commands(
      topology, queue.prepare({}), branches, queue.prepare({suffix.get()}));
  // Variant creation must not lease another branch width.
  std::vector<std::unique_ptr<Queue>> remaining;
  for (std::uint32_t i = width + 1; i < count; ++i)
    remaining.push_back(std::make_unique<Queue>(device, 1));
  EXPECT_THROW(Queue(device, 1), std::runtime_error);
  remaining.clear();
  for (std::uint32_t blocked = 1; blocked < width; ++blocked) {
    test_detail::TimelineGate ordinary(device, 0);
    test_detail::TimelineGate gate(device, indices[blocked]);
    ForkJoinSubmitProbe probe(device);
    int admitted = 0;
    const auto point = queue.submit(body, {}, [&](std::size_t slot, const auto &) {
      EXPECT_EQ(slot, 0u);
      ++admitted;
    });
    EXPECT_EQ(admitted, 1);
    EXPECT_EQ(queue.pending(), 1u);
    EXPECT_EQ(probe.calls, int(width + 2));
    EXPECT_TRUE(wait_fork_signal(device, probe.signals[1])); // real branch-zero tail
    for (std::uint32_t i = 1; i < width; ++i)
      if (i != blocked)
        EXPECT_TRUE(wait_fork_signal(device, probe.signals[i + 1]));
    EXPECT_FALSE(point->is_ready());
    const auto calls = probe.calls;
    EXPECT_THROW(queue.submit(variant, {}, [&](std::size_t, const auto &) { ++admitted; }), Queue::Busy);
    EXPECT_EQ(probe.calls, calls);
    EXPECT_EQ(admitted, 1);
    gate.release();
    EXPECT_TRUE(wait_fork_signal(device, probe.signals[width + 1]));
    ordinary.release(); // rescue before any blocking wait; a rescue never passes the check above
    ASSERT_TRUE(point->wait());
    EXPECT_EQ(queue.collect(1), 1u);
    // The final native signal was observed while queue zero remained blocked.
    void *mapped = nullptr;
    ASSERT_EQ(device.map(*storage, &mapped), RhiResult::success);
    for (std::uint32_t i = 0; i < width; ++i)
      EXPECT_EQ(static_cast<const std::uint32_t *>(mapped)[width + i], 30u + i);
    device.unmap(*storage);
  }
  // Mix ordinary and compound commands without changing logical capacity.
  auto serial = queue.submit(queue.prepare({}));
  ASSERT_TRUE(serial->wait());
  auto final = queue.submit(variant, {serial, serial});
  ASSERT_TRUE(final->wait());
  queue.close();
  EXPECT_TRUE(final->is_ready());
  // Even retained cold variants must not keep physical queues leased after
  // their anchor closes. Their old immutable completion remains queryable.
  for (std::uint32_t i = 1; i < count; ++i)
    remaining.push_back(std::make_unique<Queue>(device, 1));
  EXPECT_EQ(remaining.size(), count - 1);
}

TEST(VulkanForkJoin, ColdRejectionAndHostAdmissionHaveNoDeviceEffects) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  if (!device.vk_caps().timeline_semaphore || device.created_compute_queue_count() < 3)
    GTEST_SKIP();
  using Queue = vulkan::VulkanExecutionQueue;
  Queue queue(device, 1);
  EXPECT_THROW(queue.prepare_fork_join(1), std::invalid_argument);
  EXPECT_THROW(queue.prepare_fork_join(device.created_compute_queue_count()), std::runtime_error);
  auto topology = queue.prepare_fork_join(2);
  auto list = queue.new_command_list();
  // Empty branch is rejected. These nonempty barrier-only lists have no
  // conflicting writes; this test targets admission, not arithmetic.
  list->memory_barrier();
  auto empty = queue.prepare({});
  auto branch = queue.prepare({list.get()});
  EXPECT_THROW(queue.prepare_fork_join_commands(topology, empty, {branch, empty}, empty), std::invalid_argument);
  auto body = queue.prepare_fork_join_commands(topology, empty, {branch, branch}, empty);
  EXPECT_THROW(queue.prepare_fork_join_commands(topology, empty, {body, branch}, empty), std::invalid_argument);
  auto binary = vkapi::create_semaphore(device.vk_device(), 0);
  auto handshake = queue.prepare({list.get()}, {}, {binary}, {});
  EXPECT_THROW(queue.prepare_fork_join_commands(topology, empty, {handshake, branch}, empty), std::invalid_argument);
  bool retired = false;
  auto rejected = queue.prepare_fork_join_commands(topology, empty, {branch, branch}, empty,
      std::shared_ptr<void>(reinterpret_cast<void *>(1), [&](void *) {
        EXPECT_EQ(queue.pending(), 0u); // must not run under admission lock
        retired = true;
      }));
  ForkJoinSubmitProbe probe(device);
  const auto before = queue.latest();
  EXPECT_THROW(queue.submit(std::move(rejected), {}, [](std::size_t, const auto &) {
    throw std::invalid_argument("reject parameters");
  }), std::invalid_argument);
  EXPECT_TRUE(retired);
  EXPECT_EQ(queue.latest(), before);
  EXPECT_EQ(probe.calls, 0);
  ASSERT_TRUE(queue.submit(body)->wait());
  queue.close();
}

TEST(VulkanForkJoin, FanoutDoesNotWaitForAnchorBranchZero) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  if (!device.vk_caps().timeline_semaphore || device.created_compute_queue_count() < 3)
    GTEST_SKIP();
  using Queue = vulkan::VulkanExecutionQueue;
  Queue queue(device, 1);
  auto topology = queue.prepare_fork_join(2);
  auto [storage, status] = device.allocate_memory_unique({16, false, true, false, AllocUsage::Storage});
  ASSERT_EQ(status, RhiResult::success);
  auto prefix = queue.new_command_list();
  prefix->buffer_fill(storage->get_ptr(), 4, 61);
  auto first = queue.new_command_list();
  first->buffer_copy(storage->get_ptr(4), storage->get_ptr(), 4);
  auto second = queue.new_command_list();
  second->buffer_copy(storage->get_ptr(8), storage->get_ptr(), 4);
  auto suffix = queue.new_command_list();
  suffix->buffer_copy(storage->get_ptr(12), storage->get_ptr(4), 4);
  suffix->memory_barrier();
  auto commands = queue.prepare_fork_join_commands(topology, queue.prepare({prefix.get()}),
      {queue.prepare({first.get()}), queue.prepare({second.get()})}, queue.prepare({suffix.get()}));
  DeferredForkGate gate(device); // native objects allocated cold
  ForkJoinSubmitProbe probe(device);
  probe.gate_after_prefix = &gate; // inject only between actual prefix and branch-zero enqueue
  auto point = queue.submit(commands);
  EXPECT_EQ(probe.calls, 4);
  EXPECT_TRUE(wait_fork_signal(device, probe.signals[2])); // auxiliary sees prefix writes
  EXPECT_FALSE(fork_signal_ready(device, probe.signals[1]));
  EXPECT_FALSE(point->is_ready());
  gate.release(); // release on every path before blocking waits / queue teardown
  ASSERT_TRUE(point->wait());
  void *mapped = nullptr;
  ASSERT_EQ(device.map(*storage, &mapped), RhiResult::success);
  for (int i = 0; i < 4; ++i)
    EXPECT_EQ(static_cast<const std::uint32_t *>(mapped)[i], 61u);
  device.unmap(*storage);
  queue.close();
}

TEST(VulkanForkJoin, PrefixAndJoinFailureNeverPublishPartialCompletion) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  // Prefix failure must enqueue no branch. Join failure has real accepted
  // branches, but neither failure can publish a completed logical invocation.
  for (const int fail_at : {1, 4}) {
    vulkan::VulkanDeviceCreator creator({});
    auto &device = *creator.device();
    if (!device.vk_caps().timeline_semaphore ||
        device.created_compute_queue_count() < 3)
      GTEST_SKIP();
    using Queue = vulkan::VulkanExecutionQueue;
    auto fault = std::make_shared<RuntimeFaultDomain>(Arch::vulkan, 74);
    device.set_backend_fault_reporter(fault);
    Queue queue(device, 1);
    auto topology = queue.prepare_fork_join(2);
    auto list = queue.new_command_list();
    list->memory_barrier();
    auto branch = queue.prepare({list.get()});
    auto owner = std::make_shared<int>(fail_at);
    std::weak_ptr<int> retained = owner;
    auto commands = queue.prepare_fork_join_commands(
        topology, queue.prepare({}), {branch, branch}, queue.prepare({}), owner);
    owner.reset();
    {
      ForkJoinSubmitProbe probe(device, fail_at, VK_ERROR_DEVICE_LOST);
      EXPECT_THROW(queue.submit(std::move(commands)), BackendRuntimeError);
      EXPECT_EQ(probe.calls, fail_at);
    }
    EXPECT_FALSE(retained.expired());
    EXPECT_EQ(queue.pending(), 1u);
    EXPECT_THROW(queue.latest()->wait(), std::invalid_argument);
    EXPECT_NO_THROW(queue.close());
    EXPECT_TRUE(retained.expired());
    EXPECT_FALSE(fault->submission_allowed());
  }
}

TEST(VulkanForkJoin, LateNativeFailureRetainsAllBranchesUntilRelatedDrain) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  if (!device.vk_caps().timeline_semaphore || device.created_compute_queue_count() < 4)
    GTEST_SKIP() << "Requires three non-default compute queues";
  using Queue = vulkan::VulkanExecutionQueue;
  auto fault = std::make_shared<RuntimeFaultDomain>(Arch::vulkan, 73);
  device.set_backend_fault_reporter(fault);
  Queue queue(device, 1);
  auto topology = queue.prepare_fork_join(3);
  auto list = queue.new_command_list();
  list->memory_barrier();
  auto branch = queue.prepare({list.get()});
  auto marker = std::make_shared<int>(19);
  std::weak_ptr<int> retained = marker;
  auto body = queue.prepare_fork_join_commands(topology, queue.prepare({}),
      {branch, branch, branch}, queue.prepare({}), marker);
  marker.reset();
  test_detail::TimelineGate gate(device, topology->queue_indices()[1]);
  {
    ForkJoinSubmitProbe probe(device, 4, VK_ERROR_DEVICE_LOST);
    // Fault injection: prefix, anchor and first aux are real accepted work;
    // the late auxiliary reports a native error, with a real fault reporter.
    EXPECT_THROW(queue.submit(std::move(body)), BackendRuntimeError);
    EXPECT_EQ(probe.calls, 4); // no join waiting a never-enqueued signal
  }
  EXPECT_EQ(queue.pending(), 1u);
  EXPECT_TRUE(fault->has_fatal_fault());
  EXPECT_FALSE(retained.expired());
  // The internal final point is not an accepted completion. Its wait rejects
  // instead of hanging on a signal that no native submit ever accepted.
  EXPECT_THROW(queue.latest()->wait(), std::invalid_argument);
  EXPECT_THROW(queue.submit(branch), TaichiRuntimeError);
  std::atomic<bool> entered{false}, closed{false};
  std::exception_ptr failure;
  {
    const auto failed_queue = device.compute_queue_at(topology->queue_indices()[2]);
    ForkJoinDrainProbe drain(device, failed_queue);
    std::thread retire([&] {
      entered.store(true);
      try { queue.close(); } catch (...) { failure = std::current_exception(); }
      closed.store(true);
    });
    while (!entered.load())
      std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_FALSE(closed.load());
    EXPECT_FALSE(retained.expired());
    gate.release();
    retire.join();
    EXPECT_TRUE(failure); // injected failure of the only uncertain native queue
    EXPECT_EQ(drain.calls, 1);
    EXPECT_EQ(drain.queues[0], failed_queue);
    EXPECT_FALSE(retained.expired());
    EXPECT_EQ(queue.pending(), 1u);
  } // restore native wait API before close retries
  EXPECT_THROW(queue.submit(branch), TaichiRuntimeError);
  EXPECT_NO_THROW(queue.close());
  EXPECT_TRUE(retained.expired());
  EXPECT_FALSE(fault->submission_allowed()); // retirement never repairs public health
  EXPECT_NO_THROW(queue.close());
}
}  // namespace taichi::lang

namespace taichi::lang {
namespace {
using test_detail::TimelineGate;

bool ready_before_deadline(const vulkan::VulkanExecutionQueue::Point &point) {
  const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (!point->is_ready() && std::chrono::steady_clock::now() < end)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  return point->is_ready();
}

gfx::GfxRuntime::KernelHandle temporary_counter(gfx::GfxRuntime &runtime,
                                               int root,
                                               bool diagnostic = false) {
  DeviceCapabilityConfig caps;
  caps.set(DeviceCapability::spirv_version, 0x10300);
  spirv::IRBuilder builder(Arch::vulkan, &caps);
  builder.init_header();
  const auto temp = builder.buffer_argument(builder.u32_type(), 0, 0, "temporary");
  const auto out = builder.buffer_argument(builder.u32_type(), 0, 1, "output");
  const auto function = builder.new_function();
  builder.start_function(function);
  const auto pointer = builder.struct_array_access(
      builder.u32_type(), temp, builder.const_i32_zero_);
  const auto value = builder.add(builder.load_variable(pointer, builder.u32_type()),
                                  builder.uint_immediate_number(builder.u32_type(), 1));
  builder.store_variable(pointer, value);
  builder.store_variable(builder.struct_array_access(
                             builder.u32_type(), out, builder.const_i32_zero_), value);
  builder.make_inst(spv::OpReturn);
  builder.make_inst(spv::OpFunctionEnd);
  builder.commit_kernel_function(function, "main", {}, {1, 1, 1});
  gfx::GfxRuntime::RegisterParams params;
  params.num_snode_trees = 1;
  params.root_buffer_bindings = {root};
  params.snode_tree_ids = {root};
  spirv::TaskAttributes task;
  task.name = "private_temporary_counter";
  task.task_type = OffloadedTaskType::serial;
  task.advisory_total_num_threads = task.advisory_num_threads_per_group = 1;
  using Buffer = spirv::TaskAttributes::BufferType;
  task.buffer_binds = {{diagnostic ? Buffer::HashOverflow : Buffer::GlobalTmps, 0},
                       {{Buffer::Root, 0}, 1}};
  params.kernel_attribs.tasks_attribs = {task};
  params.task_spirv_source_codes = {builder.finalize()};
  return runtime.register_taichi_kernel(std::move(params));
}
}  // namespace

TEST(VulkanFixedCommands, PrivateTemporariesPersistWithoutDefaultQueueProgress) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  if (!device.vk_caps().timeline_semaphore || device.created_compute_queue_count() < 3)
    GTEST_SKIP();
  gfx::GfxRuntime::Params params;
  params.device = &device;
  gfx::GfxRuntime runtime(params);
  for (int i = 0; i < 3; ++i)
    runtime.add_root_buffer(4);
  const auto ordinary = temporary_counter(runtime, 2);
  CallableBase callable;
  LaunchContextBuilder context(&callable);
  runtime.launch_kernel(ordinary, context);
  runtime.synchronize();
  using Queue = vulkan::VulkanExecutionQueue;
  Queue left(device, 2), right(device, 2);
  std::array<Queue *, 2> queues{&left, &right};
  std::array<std::shared_ptr<const Queue::Commands>, 2> prepared;
  std::array<std::shared_ptr<DeviceAllocationGuard>, 2> outputs;
  std::shared_ptr<const Queue::Commands> continuation;
  for (int i = 0; i < 2; ++i) {
    auto workspace = runtime.create_fixed_graph_workspace();
    EXPECT_EQ(workspace->bytes(), 0u);
    const auto kernel = temporary_counter(runtime, i);
    auto [output, status] = device.allocate_memory_unique(
        {4, false, true, false, AllocUsage::Storage});
    ASSERT_EQ(status, RhiResult::success);
    outputs[i] = std::move(output);
    gfx::GfxRuntime::GraphRecordingOperation copy;
    copy.inline_recording = true;
    const auto root = *runtime.get_root_buffer(i);
    copy.buffers = std::vector<DeviceAllocation>{root, *outputs[i]};
    copy.external = [root, output = outputs[i]](Device *, CommandList *commands) {
      commands->buffer_copy(output->get_ptr(), root.get_ptr(), 4);
      commands->memory_barrier();
    };
    const std::vector<gfx::GfxRuntime::GraphRecordingOperation> operations{
        {{kernel, &context}, {}}, copy};
    const auto before = device.queue_submission_snapshot();
    auto payload = runtime.prepare_fixed_commands(
        operations, queues[i]->new_command_list(), workspace, {outputs[i]}, {i});
    EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls,
              before.queue_submit_calls);
    EXPECT_GT(workspace->bytes(), 0u);
    prepared[i] = queues[i]->prepare({payload->cmdlist.get()}, payload);
    if (i == 0) {
      auto next = runtime.prepare_fixed_commands(
          operations, queues[i]->new_command_list(), workspace, {outputs[i]}, {i});
      continuation = queues[i]->prepare({next->cmdlist.get()}, next);
    }
    auto initialize = queues[i]->new_command_list();
    workspace->record_initialization(*initialize);
    EXPECT_ANY_THROW(workspace->record_initialization(*initialize));
    ASSERT_TRUE(queues[i]->submit(queues[i]->prepare({initialize.get()}, workspace))->wait());
  }
  const auto read = [&](int index) {
    void *mapped = nullptr;
    if (device.map(*outputs[index], &mapped) != RhiResult::success) {
      ADD_FAILURE() << "Cannot map readback";
      return uint32_t(0);
    }
    const auto value = *static_cast<const uint32_t *>(mapped);
    device.unmap(*outputs[index]);
    return value;
  };
  {
    TimelineGate ordinary_gate(device, 0);
    TimelineGate left_gate(device, left.queue_index());
    const auto pending = left.submit(prepared[0]);
    for (uint32_t expected : {1, 2}) {
      ASSERT_TRUE(ready_before_deadline(right.submit(prepared[1])));
      EXPECT_EQ(read(1), expected);
      EXPECT_FALSE(pending->is_ready());
    }
    left_gate.release();
    ASSERT_TRUE(ready_before_deadline(pending));
    EXPECT_EQ(read(0), 1u);
    ASSERT_TRUE(ready_before_deadline(left.submit(continuation)));
    EXPECT_EQ(read(0), 2u);
  }
  // Both instances leave the original runtime's mutable temporary untouched.
  runtime.launch_kernel(ordinary, context);
  runtime.synchronize();
  auto source = runtime.get_root_buffer(2)->get_ptr();
  uint32_t value = 0;
  void *host = &value;
  std::size_t bytes = sizeof(value);
  ASSERT_EQ(device.readback_data(&source, &host, &bytes, 1), RhiResult::success);
  EXPECT_EQ(value, 2u);
  left.close();
  right.close();
}

TEST(VulkanFixedCommands, RejectsSharedStateAndUnqualifiedCallbacksBeforeRecording) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  if (!device.vk_caps().timeline_semaphore || device.created_compute_queue_count() < 2)
    GTEST_SKIP();
  gfx::GfxRuntime::Params params;
  params.device = &device;
  gfx::GfxRuntime runtime(params);
  runtime.add_root_buffer(4);
  CallableBase callable;
  LaunchContextBuilder context(&callable);
  const auto shared = temporary_counter(runtime, 0, true);
  const auto normal = temporary_counter(runtime, 0);
  vulkan::VulkanExecutionQueue queue(device, 1);
  auto workspace = runtime.create_fixed_graph_workspace();
  bool called = false;
  gfx::GfxRuntime::GraphRecordingOperation external;
  external.external = [&](Device *, CommandList *) { called = true; };
  const auto before = device.queue_submission_snapshot();
  EXPECT_ANY_THROW(runtime.prepare_fixed_commands(
      {external}, queue.new_command_list(), workspace, {}));
  EXPECT_FALSE(called);
  EXPECT_ANY_THROW(runtime.prepare_fixed_commands(
      {{{shared, &context}, {}}}, queue.new_command_list(), workspace, {}, {0}));
  EXPECT_EQ(workspace->bytes(), 0u);
  auto initialization = queue.new_command_list();
  workspace->record_initialization(*initialization);
  EXPECT_ANY_THROW(runtime.prepare_fixed_commands(
      {{{normal, &context}, {}}}, queue.new_command_list(), workspace, {}, {0}));
  EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls,
            before.queue_submit_calls);
  EXPECT_FALSE(queue.latest());
}

TEST(VulkanFixedCommands, PrivateImagePhasesDoNotUseTheDefaultQueue) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  if (!device.vk_caps().timeline_semaphore || device.created_compute_queue_count() < 2)
    GTEST_SKIP();
  gfx::GfxRuntime::Params params;
  params.device = &device;
  gfx::GfxRuntime runtime(params);
  vulkan::VulkanExecutionQueue queue(device, 1);
  ImageParams image_params;
  image_params.dimension = ImageDimension::d2D;
  image_params.format = BufferFormat::r32u;
  image_params.x = image_params.y = image_params.z = 1;
  image_params.usage = ImageAllocUsage::Storage;
  const auto image = runtime.create_image(image_params);
  auto owner = std::shared_ptr<void>(reinterpret_cast<void *>(1), [&](void *) {
    runtime.untrack_image(image);
    device.destroy_image(image);
  });
  auto [pixels, result] = device.allocate_memory_unique(
      {4, false, true, false, AllocUsage::Storage});
  ASSERT_EQ(result, RhiResult::success);
  BufferImageCopyParams copy;
  copy.buffer_row_length = copy.buffer_image_height = 1;
  copy.image_extent = {1, 1, 1};
  gfx::GfxRuntime::GraphRecordingOperation upload, download;
  upload.inline_recording = download.inline_recording = true;
  upload.buffers = download.buffers = std::vector<DeviceAllocation>{*pixels};
  upload.images = {{image, ImageLayout::transfer_dst}};
  download.images = {{image, ImageLayout::transfer_src}};
  upload.external = [image, ptr = pixels->get_ptr(), copy](Device *, CommandList *commands) {
    commands->buffer_fill(ptr, 4, 0xFEDCBA98u);
    commands->buffer_barrier(ptr);
    commands->buffer_to_image(image, ptr, ImageLayout::transfer_dst, copy);
  };
  download.external = [image, ptr = pixels->get_ptr(), copy](Device *, CommandList *commands) {
    commands->image_to_buffer(ptr, image, ImageLayout::transfer_src, copy);
    commands->memory_barrier();
  };
  auto workspace = runtime.create_fixed_graph_workspace();
  EXPECT_ANY_THROW(runtime.prepare_fixed_commands(
      {upload}, queue.new_command_list(), workspace, {}));
  runtime.prepare_fixed_private_image(workspace, image, ImageLayout::shader_read_write, owner);
  EXPECT_ANY_THROW(runtime.prepare_fixed_private_image(
      workspace, image, ImageLayout::shader_read, owner));
  auto first = runtime.prepare_fixed_commands(
      {upload}, queue.new_command_list(), workspace, {});
  auto second = runtime.prepare_fixed_commands(
      {download}, queue.new_command_list(), workspace, {});
  auto initialize = queue.new_command_list();
  workspace->record_initialization(*initialize);
  EXPECT_ANY_THROW(runtime.prepare_fixed_private_image(
      workspace, image, ImageLayout::shader_read_write, owner));
  {
    TimelineGate gate(device, 0);
    // First layout transition and both separately recorded phases must proceed
    // while the default queue is pending. Their boundary layout is identical.
    auto init = queue.prepare({initialize.get()}, workspace);
    ASSERT_TRUE(ready_before_deadline(queue.submit(init)));
    auto commands = queue.prepare({first->cmdlist.get(), second->cmdlist.get()}, workspace);
    for (int replay = 0; replay < 3; ++replay) {
      const auto before = device.queue_submission_snapshot();
      ASSERT_TRUE(ready_before_deadline(queue.submit(commands)));
      EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, before.queue_submit_calls + 1);
      void *mapped = nullptr;
      ASSERT_EQ(device.map(*pixels, &mapped), RhiResult::success);
      EXPECT_EQ(*static_cast<const uint32_t *>(mapped), 0xFEDCBA98u);
      device.unmap(*pixels);
    }
  }
  queue.close();
}

TEST(VulkanControlDescriptors, ScalarViewsRespectStorageAlignment) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  const auto *folder = std::getenv("TAICHI_AOT_FOLDER_PATH");
  ASSERT_NE(folder, nullptr);
  vulkan::VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  gfx::GfxRuntime::Params params;
  params.device = &device;
  gfx::GfxRuntime runtime(params);
  gfx::AotModuleParams module_params;
  module_params.module_path = folder;
  module_params.runtime = &runtime;
  auto module = aot::Module::load(Arch::vulkan, module_params);
  ASSERT_TRUE(module);
  auto *condition = dynamic_cast<gfx::KernelImpl *>(module->get_kernel("condition_at"));
  auto *advance = dynamic_cast<gfx::KernelImpl *>(module->get_kernel("advance_at"));
  ASSERT_TRUE(condition && advance);
  auto [storage, status] = device.allocate_memory_unique(
      {16 * sizeof(int), true, true, false, AllocUsage::Storage});
  ASSERT_EQ(status, RhiResult::success);
  Ndarray array(*storage, PrimitiveType::i32, {16});
  LaunchContextBuilder cond(condition), body(advance);
  cond.set_arg_ndarray({0}, array);
  body.set_arg_ndarray({0}, array);
  const std::vector<gfx::GfxRuntime::GraphDispatch> dispatches{
      {condition->handle(), &cond}, {advance->handle(), &body},
      {condition->handle(), &cond}};
  using Strategy = gfx::GfxRuntime::GraphStructuredStrategy;
  for (auto strategy : {Strategy::compact, Strategy::chained,
                        Strategy::conditional, Strategy::coarse_conditional}) {
    if ((strategy == Strategy::conditional || strategy == Strategy::coarse_conditional) &&
        !device.supports_conditional_commands())
      continue;
    auto registration = runtime.register_graph_replay(1);
    // More than one slot rotation, with different scalar biases and argument
    // images, protects the controller pipeline's specialization identity.
    for (int replay = 0; replay < 16; ++replay) {
      const int base = 1 + replay % 10;
      const bool observe = replay % 2 == 0;
      std::array<int, 16> values{};
      void *mapped = nullptr;
      ASSERT_EQ(device.map(*storage, &mapped), RhiResult::success);
      std::memcpy(mapped, values.data(), sizeof(values));
      device.unmap(*storage);
      cond.set_arg_int({1}, base);
      body.set_arg_int({1}, base);
      gfx::GfxRuntime::GraphStructuredControl control;
      control.predicate = storage->get_ptr(base * sizeof(int));
      control.counter = storage->get_ptr((base + 1) * sizeof(int));
      control.status = storage->get_ptr((base + 2) * sizeof(int));
      control.has_status = true;
      control.initial_dispatch_count = 1;
      control.max_iterations = 3;
      control.strategy = strategy;
      gfx::GfxRuntime::GraphStructuredResult result;
      ASSERT_TRUE(runtime.try_launch_graph(dispatches, registration->replay_key(),
                                          nullptr, &control, observe ? &result : nullptr));
      runtime.synchronize();
      ASSERT_EQ(device.map(*storage, &mapped), RhiResult::success);
      std::memcpy(values.data(), mapped, sizeof(values));
      device.unmap(*storage);
      EXPECT_EQ(values[base], 0);
      EXPECT_EQ(values[base + 1], 2);
      EXPECT_EQ(values[base + 2], 2);
      if (observe) {
        EXPECT_TRUE(result.submitted);
        EXPECT_EQ(result.logical_iterations, 2);
        EXPECT_EQ(result.predicate, 0);
        EXPECT_EQ(result.counter, 2);
        EXPECT_EQ(result.status, 2);
        EXPECT_EQ(result.initial_status, 1);
      }
    }
  }
}

TEST(VulkanFixedCommands, StructuredControlOwnsQueueAndTemporaryStorage) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  const auto *folder = std::getenv("TAICHI_AOT_FOLDER_PATH");
  ASSERT_NE(folder, nullptr);
  vulkan::VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  if (!device.vk_caps().timeline_semaphore || device.created_compute_queue_count() < 3)
    GTEST_SKIP();
  gfx::GfxRuntime::Params params;
  params.device = &device;
  gfx::GfxRuntime runtime(params);
  runtime.add_root_buffer(4);
  runtime.add_root_buffer(4);
  gfx::AotModuleParams module_params;
  module_params.module_path = folder;
  module_params.runtime = &runtime;
  std::shared_ptr<aot::Module> module = aot::Module::load(Arch::vulkan, module_params);
  ASSERT_TRUE(module);
  auto *condition = dynamic_cast<gfx::KernelImpl *>(module->get_kernel("condition"));
  auto *advance = dynamic_cast<gfx::KernelImpl *>(module->get_kernel("advance"));
  auto *adjoint = dynamic_cast<gfx::KernelImpl *>(module->get_kernel("adjoint"));
  ASSERT_TRUE(condition && advance && adjoint);
  using Queue = vulkan::VulkanExecutionQueue;
  Queue left(device, 1), right(device, 1);
  using Strategy = gfx::GfxRuntime::GraphStructuredStrategy;
  for (auto strategy : {Strategy::compact, Strategy::conditional}) {
    if (strategy == Strategy::conditional && !device.supports_conditional_commands())
      continue;
    for (uint32_t budget : {0u, 3u}) {
      struct Frame {
        std::shared_ptr<DeviceAllocationGuard> state, count;
        std::shared_ptr<gfx::GfxRuntime::FixedGraphWorkspace> workspace;
        std::shared_ptr<gfx::GfxRuntime::GraphReplayExecutable::Slot> loop, finish;
        std::shared_ptr<const Queue::Commands> first, replay;
      };
      std::array<Frame, 2> frames;
      for (int lane = 0; lane < 2; ++lane) {
        auto &frame = frames[lane];
        auto &queue = lane == 0 ? left : right;
        auto [state, status] = device.allocate_memory_unique(
            {7 * sizeof(int), true, true, false, AllocUsage::Storage});
        ASSERT_EQ(status, RhiResult::success);
        frame.state = std::move(state);
        auto [count, count_status] = device.allocate_memory_unique(
            {4, false, true, false, AllocUsage::Storage});
        ASSERT_EQ(count_status, RhiResult::success);
        frame.count = std::move(count);
        frame.workspace = runtime.create_fixed_graph_workspace();
        Ndarray array(*frame.state, PrimitiveType::i32, {7});
        LaunchContextBuilder cond(condition), body(advance), grad(adjoint);
        for (auto *context : {&cond, &body, &grad})
          context->set_arg_ndarray({0}, array);
        CallableBase callable;
        LaunchContextBuilder temporary(&callable);
        const auto counter = temporary_counter(runtime, lane);
        const std::vector<gfx::GfxRuntime::GraphDispatch> dispatches{
            {condition->handle(), &cond}, {counter, &temporary},
            {advance->handle(), &body}, {condition->handle(), &cond}};
        gfx::GfxRuntime::GraphStructuredControl control;
        control.predicate = frame.state->get_ptr();
        control.counter = frame.state->get_ptr(4);
        control.status = frame.state->get_ptr(8);
        control.has_status = true;
        control.initial_dispatch_count = 1;
        control.max_iterations = budget;
        control.strategy = budget ? strategy : Strategy::compact;
        const auto before = device.queue_submission_snapshot().queue_submit_calls;
        auto invalid = control;
        invalid.initial_dispatch_count = dispatches.size();
        EXPECT_ANY_THROW(runtime.prepare_fixed_control_commands(
            dispatches, &invalid, nullptr, queue.new_command_list(),
            frame.workspace, {module, frame.state}, {lane}));
        frame.loop = runtime.prepare_fixed_control_commands(
            dispatches, &control, nullptr, queue.new_command_list(),
            frame.workspace, {module, frame.state}, {lane});
        frame.finish = runtime.prepare_fixed_commands(
            {{{adjoint->handle(), &grad}, {}}}, queue.new_command_list(),
            frame.workspace, {module, frame.state});
        auto initialize = queue.new_command_list();
        frame.workspace->record_initialization(*initialize);
        auto copy = queue.new_command_list();
        copy->memory_barrier();
        copy->buffer_copy(frame.count->get_ptr(), runtime.get_root_buffer(lane)->get_ptr(), 4);
        copy->memory_barrier();
        // Initialize observable output when a zero-trip loop never writes it.
        initialize->buffer_fill(runtime.get_root_buffer(lane)->get_ptr(), 4, 0);
        initialize->memory_barrier();
        frame.first = queue.prepare({initialize.get(), frame.loop->cmdlist.get(),
                                     frame.finish->cmdlist.get(), copy.get()});
        frame.replay = queue.prepare({frame.loop->cmdlist.get(), frame.finish->cmdlist.get(), copy.get()});
        EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, before);
      }
      runtime.synchronize();
      std::array<int, 2> cumulative{};
      {
        TimelineGate default_gate(device, 0);
        bool first = true;
        for (int target : {0, 2, 7, 0}) {
          for (int lane = 0; lane < 2; ++lane) {
            auto &frame = frames[lane];
            auto &queue = lane == 0 ? left : right;
            const int initial = lane + 3;
            const std::array<int, 7> values{0, 0, 0, initial, target, 0, 0};
            void *mapped = nullptr;
            ASSERT_EQ(device.map(*frame.state, &mapped), RhiResult::success);
            std::memcpy(mapped, values.data(), sizeof(values));
            device.unmap(*frame.state);
            const auto before = device.queue_submission_snapshot().queue_submit_calls;
            auto point = queue.submit(first ? frame.first : frame.replay);
            ASSERT_TRUE(ready_before_deadline(point));
            EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, before + 1);
            const int rounds = std::min(target, static_cast<int>(budget));
            cumulative[lane] += rounds;
            ASSERT_EQ(device.map(*frame.state, &mapped), RhiResult::success);
            std::array<int, 7> observed{};
            std::memcpy(observed.data(), mapped, sizeof(observed));
            device.unmap(*frame.state);
            EXPECT_EQ(observed, (std::array<int, 7>{target > rounds, rounds,
                target > rounds ? 1 : 2, initial << rounds, target, rounds + 1,
                2 * initial * (1 << (2 * rounds))}));
            ASSERT_EQ(device.map(*frame.count, &mapped), RhiResult::success);
            EXPECT_EQ(*static_cast<const int *>(mapped), cumulative[lane]);
            device.unmap(*frame.count);
          }
          first = false;
        }
      }
    }
  }
  left.close();
  right.close();
}

namespace {
struct FixedOperands final : RuntimeCompletionResources {
  std::vector<std::shared_ptr<DeviceAllocationGuard>> allocations;
  std::size_t retained_resource_count(std::uint32_t) const noexcept override {
    return allocations.size();
  }
};

struct FixedParts final : RuntimeClient {
  std::vector<std::shared_ptr<RuntimeClient>> parts;
  void retire() override {
    for (const auto &part : parts)
      part->retire();
    parts.clear();
  }
};
}  // namespace

TEST(VulkanFixedExecution, PrivateReplayAndNativeSortIgnoreBlockedDefaultQueue) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  if (!device.vk_caps().timeline_semaphore || device.created_compute_queue_count() < 3)
    GTEST_SKIP();
  auto owner = std::make_shared<RuntimeFaultDomain>(Arch::vulkan, 51);
  device.set_backend_fault_reporter(owner);
  gfx::GfxRuntime::Params params;
  params.device = &device;
  gfx::GfxRuntime runtime(params);
  runtime.add_root_buffer(4);
  runtime.add_root_buffer(4);
  runtime.synchronize();
  vulkan::ExecutionDomainRegistry registry(device, owner);
  std::array<std::shared_ptr<vulkan::ExecutionDomain>, 2> domains{
      registry.create({1, 1}), registry.create({1, 1})};
  std::array<std::unique_ptr<FixedExecution>, 2> calls;
  std::array<std::shared_ptr<FixedOperands>, 2> operands;
  CallableBase callable;
  LaunchContextBuilder context(&callable);
  for (int i = 0; i < 2; ++i) {
    auto resources = std::make_shared<FixedOperands>();
    auto [out, out_status] = device.allocate_memory_unique(
        {4, false, true, false, AllocUsage::Storage});
    auto [keys, key_status] = device.allocate_memory_unique(
        {32, true, true, false, AllocUsage::Storage});
    ASSERT_EQ(out_status, RhiResult::success);
    ASSERT_EQ(key_status, RhiResult::success);
    resources->allocations.push_back(std::move(out));
    resources->allocations.push_back(std::move(keys));
    void *mapped = nullptr;
    ASSERT_EQ(device.map(*resources->allocations[1], &mapped), RhiResult::success);
    for (uint32_t j = 0; j < 8; ++j)
      static_cast<uint32_t *>(mapped)[j] = 8 - j;
    device.unmap(*resources->allocations[1]);
    const auto handle = temporary_counter(runtime, i);
    gfx::GfxRuntime::RegisterParams metadata;
    metadata.snode_tree_ids = {i};
    gfx::KernelImpl kernel(&runtime, std::move(metadata), &handle);
    auto sort = prepare_sort_operation(Arch::vulkan, device,
        {resources->allocations[1]->get_ptr(), kDeviceNullPtr, 8, NativeSortKeyType::u32});
    FixedExecution::VulkanExecutionBinding binding{domains[i], resources};
    const auto before = device.queue_submission_snapshot().queue_submit_calls;
    calls[i] = std::make_unique<FixedExecution>(
        Arch::vulkan, device, &runtime,
        std::vector<FixedExecution::Operation>{
            FixedExecution::Dispatch{&kernel, &context}, sort,
            FixedExecution::BufferCopy{resources->allocations[0]->get_ptr(),
                                      runtime.get_root_buffer(i)->get_ptr(), 4}},
        nullptr, owner, nullptr, nullptr, nullptr, nullptr, nullptr, &binding);
    EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, before);
    operands[i] = std::move(resources);
  }
  const auto read = [&](int index) {
    void *mapped = nullptr;
    EXPECT_EQ(device.map(*operands[index]->allocations[0], &mapped), RhiResult::success);
    const auto value = *static_cast<uint32_t *>(mapped);
    device.unmap(*operands[index]->allocations[0]);
    EXPECT_EQ(device.map(*operands[index]->allocations[1], &mapped), RhiResult::success);
    for (uint32_t j = 0; j < 8; ++j)
      EXPECT_EQ(static_cast<uint32_t *>(mapped)[j], j + 1);
    device.unmap(*operands[index]->allocations[1]);
    return value;
  };
  RuntimeCompletion completed;
  {
    TimelineGate default_gate(device, 0);
    TimelineGate left_gate(device, domains[0]->snapshot().queue_index);
    const auto pending = domains[0]->submit(domains[0]->prepare_commands({}));
    const auto before = device.queue_submission_snapshot().queue_submit_calls;
    EXPECT_THROW(calls[0]->submit(), vulkan::ExecutionDomain::Busy);
    EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, before);
    for (uint32_t expected : {1, 2}) {
      const auto before_peer = device.queue_submission_snapshot().queue_submit_calls;
      completed = calls[1]->submit();
      CompletionWaitSet ready({completed}, std::chrono::milliseconds(1));
      ASSERT_EQ(ready.wait(std::chrono::seconds(3)).status, CompletionWaitSet::Status::ready);
      EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, before_peer + 1);
      EXPECT_EQ(read(1), expected);
      EXPECT_FALSE(pending.done());
    }
    left_gate.release();
    pending.wait();
    for (uint32_t expected : {1, 2}) {
      calls[0]->submit().wait();
      EXPECT_EQ(read(0), expected);
    }
    domains[0]->close();
    EXPECT_THROW(calls[0]->submit(), std::exception);
    calls[1]->submit().wait();
    EXPECT_EQ(read(1), 3u);
    domains[1]->close();
  }
  registry.retire();
  EXPECT_TRUE(completed.done());
}

TEST(VulkanFixedExecution, SharedPhaseOwnerRetiresAndRejectedPreparationReleasesBudget) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  if (!device.vk_caps().timeline_semaphore || device.created_compute_queue_count() < 2)
    GTEST_SKIP();
  auto owner = std::make_shared<RuntimeFaultDomain>(Arch::vulkan, 52);
  device.set_backend_fault_reporter(owner);
  gfx::GfxRuntime::Params params;
  params.device = &device;
  gfx::GfxRuntime runtime(params);
  runtime.add_root_buffer(4);
  runtime.synchronize();
  vulkan::ExecutionDomainRegistry registry(device, owner);
  auto domain = registry.create({2, 1});
  auto operands = std::make_shared<FixedOperands>();
  CallableBase callable;
  LaunchContextBuilder context(&callable);
  const auto handle = temporary_counter(runtime, 0);
  gfx::GfxRuntime::RegisterParams metadata;
  metadata.snode_tree_ids = {0};
  gfx::KernelImpl kernel(&runtime, std::move(metadata), &handle);
  const std::vector<FixedExecution::Operation> operations{
      FixedExecution::Dispatch{&kernel, &context}};
  FixedExecution::VulkanExecutionBinding binding{domain, operands};
  auto parts = std::make_shared<FixedParts>();
  auto workspace = runtime.create_fixed_graph_workspace();
  binding.workspace = workspace;
  binding.retirement_parts = &parts->parts;
  std::unique_ptr<FixedExecution> first, second;
  auto registration = domain->prepare_owned_instance(parts, [&](const auto &instance) {
    binding.instance = instance;
    first = std::make_unique<FixedExecution>(Arch::vulkan, device, &runtime,
        operations, nullptr, owner, nullptr, nullptr, nullptr, nullptr, nullptr, &binding);
    second = std::make_unique<FixedExecution>(Arch::vulkan, device, &runtime,
        operations, nullptr, owner, nullptr, nullptr, nullptr, nullptr, nullptr, &binding);
  });
  auto initialize = domain->new_command_list();
  workspace->record_initialization(*initialize);
  domain->submit_instance(registration,
      domain->prepare_commands({initialize.get()}, workspace)).wait();
  first->submit().wait();
  auto completed = second->submit();
  domain->retire_instance(registration);
  EXPECT_TRUE(completed.done());
  EXPECT_TRUE(parts->parts.empty());
  EXPECT_THROW(first->submit(), std::exception);
  auto source = runtime.get_root_buffer(0)->get_ptr();
  uint32_t value = 0;
  void *host = &value;
  std::size_t bytes = sizeof(value);
  ASSERT_EQ(device.readback_data(&source, &host, &bytes, 1), RhiResult::success);
  EXPECT_EQ(value, 2u);
  // A subsequent rejected detached recording must release the one-instance
  // reservation. Its caller cannot accidentally publish a partial execution.
  const auto bad_handle = temporary_counter(runtime, 0, true);
  gfx::GfxRuntime::RegisterParams bad_metadata;
  bad_metadata.snode_tree_ids = {0};
  gfx::KernelImpl bad(&runtime, std::move(bad_metadata), &bad_handle);
  binding = {domain, operands};
  EXPECT_ANY_THROW(FixedExecution(Arch::vulkan, device, &runtime,
      {FixedExecution::Dispatch{&bad, &context}}, nullptr, owner,
      nullptr, nullptr, nullptr, nullptr, nullptr, &binding));
  FixedExecution replacement(Arch::vulkan, device, &runtime, operations,
      nullptr, owner, nullptr, nullptr, nullptr, nullptr, nullptr, &binding);
  replacement.submit().wait();
  registry.retire();
}

TEST(VulkanFixedExecution, PrefixChunkedControlAndSuffixShareOneSubmission) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  const auto *folder = std::getenv("TAICHI_AOT_FOLDER_PATH");
  ASSERT_NE(folder, nullptr);
  vulkan::VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  if (!device.vk_caps().timeline_semaphore || device.created_compute_queue_count() < 2)
    GTEST_SKIP();
  auto owner = std::make_shared<RuntimeFaultDomain>(Arch::vulkan, 54);
  device.set_backend_fault_reporter(owner);
  gfx::GfxRuntime::Params params;
  params.device = &device;
  gfx::GfxRuntime runtime(params);
  runtime.add_root_buffer(4);
  gfx::AotModuleParams module_params;
  module_params.module_path = folder;
  module_params.runtime = &runtime;
  auto module = aot::Module::load(Arch::vulkan, module_params);
  ASSERT_TRUE(module);
  auto *condition = dynamic_cast<gfx::KernelImpl *>(module->get_kernel("condition"));
  auto *advance = dynamic_cast<gfx::KernelImpl *>(module->get_kernel("advance"));
  auto *adjoint = dynamic_cast<gfx::KernelImpl *>(module->get_kernel("adjoint"));
  ASSERT_TRUE(condition && advance && adjoint);
  auto resources = std::make_shared<FixedOperands>();
  auto [state, status] = device.allocate_memory_unique(
      {7 * sizeof(int), true, true, false, AllocUsage::Storage});
  ASSERT_EQ(status, RhiResult::success);
  resources->allocations.push_back(std::move(state));
  auto &storage = *resources->allocations[0];
  Ndarray array(storage, PrimitiveType::i32, {7});
  LaunchContextBuilder cond(condition), body(advance), grad(adjoint);
  for (auto *context : {&cond, &body, &grad})
    context->set_arg_ndarray({0}, array);
  vulkan::ExecutionDomainRegistry registry(device, owner);
  auto domain = registry.create({1, 1});
  FixedExecution::VulkanExecutionBinding binding{domain, resources};
  auto loop = std::make_shared<FixedExecution::While>();
  loop->predicate = storage.get_ptr();
  loop->max_iterations = 3;
  loop->condition = {FixedExecution::Dispatch{condition, &cond}};
  loop->body = {FixedExecution::Dispatch{advance, &body}};
  loop->vulkan.emplace();
  loop->vulkan->counter = storage.get_ptr(4);
  loop->vulkan->status = storage.get_ptr(8);
  loop->vulkan->recipe.chunks = {
      {1, VulkanControlStrategy::compact}, {2, VulkanControlStrategy::compact}};
  // Reject a shared-state loop only after recording the valid prefix. The
  // failed transaction must return the one instance reservation without submit.
  CallableBase callable;
  LaunchContextBuilder invalid_context(&callable);
  const auto invalid_handle = temporary_counter(runtime, 0, true);
  gfx::GfxRuntime::RegisterParams metadata;
  metadata.snode_tree_ids = {0};
  gfx::KernelImpl invalid_kernel(&runtime, std::move(metadata), &invalid_handle);
  auto invalid_loop = std::make_shared<FixedExecution::While>(*loop);
  invalid_loop->body = {FixedExecution::Dispatch{&invalid_kernel, &invalid_context}};
  const auto before_prepare = device.queue_submission_snapshot().queue_submit_calls;
  EXPECT_ANY_THROW(FixedExecution(Arch::vulkan, device, &runtime,
      {FixedExecution::BufferFill{storage.get_ptr(4), 4, 0},
       std::shared_ptr<const FixedExecution::While>(invalid_loop)},
      nullptr, owner, nullptr, nullptr, nullptr, nullptr, nullptr, &binding));
  EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, before_prepare);
  FixedExecution call(Arch::vulkan, device, &runtime,
      {FixedExecution::BufferFill{storage.get_ptr(4), 4, 0},
       std::shared_ptr<const FixedExecution::While>(loop),
       FixedExecution::Dispatch{adjoint, &grad}},
      nullptr, owner, nullptr, nullptr, nullptr, nullptr, nullptr, &binding);
  EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, before_prepare);
  runtime.synchronize();
  {
    TimelineGate default_gate(device, 0);
    for (int target : {0, 1, 7}) {
      const std::array<int, 7> values{0, 99, 0, 3, target, 0, 0};
      void *mapped = nullptr;
      ASSERT_EQ(device.map(storage, &mapped), RhiResult::success);
      std::memcpy(mapped, values.data(), sizeof(values));
      device.unmap(storage);
      const auto before = device.queue_submission_snapshot().queue_submit_calls;
      auto done = call.submit();
      CompletionWaitSet ready({done}, std::chrono::milliseconds(1));
      ASSERT_EQ(ready.wait(std::chrono::seconds(3)).status, CompletionWaitSet::Status::ready);
      EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, before + 1);
      const int rounds = std::min(target, 3);
      std::array<int, 7> observed{};
      ASSERT_EQ(device.map(storage, &mapped), RhiResult::success);
      std::memcpy(observed.data(), mapped, sizeof(observed));
      device.unmap(storage);
      EXPECT_EQ(observed, (std::array<int, 7>{target > rounds, rounds,
          target > rounds ? 1 : 2, 3 << rounds, target, rounds + 1,
          6 * (1 << (2 * rounds))}));
    }
    domain->close();
  }
  registry.retire();
}

TEST(VulkanFixedExecution, SubmissionParametersAndSparseTimingUseReusableSlots) {
  const auto *folder = std::getenv("TAICHI_VULKAN_PARAMETERS_AOT_DIR");
  if (!folder || !vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  if (!device.vk_caps().timeline_semaphore || device.created_compute_queue_count() < 2)
    GTEST_SKIP();
  auto owner = std::make_shared<RuntimeFaultDomain>(Arch::vulkan, 53);
  device.set_backend_fault_reporter(owner);
  gfx::GfxRuntime::Params params;
  params.device = &device;
  gfx::GfxRuntime runtime(params);
  gfx::AotModuleParams module_params;
  module_params.runtime = &runtime;
  module_params.module_path = folder;
  auto module = aot::Module::load(Arch::vulkan, module_params);
  ASSERT_NE(module, nullptr);
  auto *kernel = module->get_kernel("publish");
  ASSERT_NE(kernel, nullptr);
  vulkan::ExecutionDomainRegistry registry(device, owner);
  auto domain = registry.create({2, 2, true});
  std::array<std::unique_ptr<FixedExecution>, 2> calls;
  std::array<std::shared_ptr<FixedOperands>, 2> operands;
  for (int i = 0; i < 2; ++i) {
    auto resources = std::make_shared<FixedOperands>();
    auto [header, status] = device.allocate_memory_unique(
        {24, false, true, false, AllocUsage::Storage});
    ASSERT_EQ(status, RhiResult::success);
    resources->allocations.push_back(std::move(header));
    Ndarray array(*resources->allocations[0], PrimitiveType::u64, {3});
    LaunchContextBuilder context(kernel);
    for (int parameter = 0; parameter < 3; ++parameter)
      context.set_arg_uint({parameter}, 0);
    context.set_arg_ndarray({3}, array);
    FixedExecution::Dispatch dispatch{kernel, &context};
    for (int parameter = 0; parameter < 3; ++parameter)
      dispatch.parameters.push_back({static_cast<std::size_t>(parameter),
          kernel->args_type->get_element_offset({parameter})});
    FixedExecution::VulkanExecutionBinding binding{domain, resources};
    binding.parameter_count = 3;
    const auto before = device.queue_submission_snapshot().queue_submit_calls;
    calls[i] = std::make_unique<FixedExecution>(Arch::vulkan, device, &runtime,
        std::vector<FixedExecution::Operation>{dispatch}, nullptr, owner,
        nullptr, nullptr, nullptr, nullptr, nullptr, &binding);
    EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, before);
    operands[i] = std::move(resources);
  }
  const auto values = [](std::uint64_t seed) {
    return std::vector<std::uint64_t>{(std::uint64_t{1} << 63) + seed,
        (std::uint64_t{1} << 53) + seed * 3, (std::uint64_t{1} << 61) + seed * 7};
  };
  const auto check = [&](int i, const std::vector<std::uint64_t> &expected) {
    void *mapped = nullptr;
    ASSERT_EQ(device.map(*operands[i]->allocations[0], &mapped), RhiResult::success);
    std::array<std::uint64_t, 3> actual;
    std::memcpy(actual.data(), mapped, 24);
    device.unmap(*operands[i]->allocations[0]);
    EXPECT_TRUE(std::equal(actual.begin(), actual.end(), expected.begin()));
  };
  RuntimeCompletion sampled;
  {
    TimelineGate default_gate(device, 0);
    TimelineGate selected_gate(device, domain->snapshot().queue_index);
    auto changed = values(1);
    sampled = calls[0]->submit({}, changed, true);
    changed.assign(3, 0);  // Accepted upload no longer observes caller storage.
    const auto other = calls[1]->submit({}, values(2));
    EXPECT_FALSE(sampled.done());
    EXPECT_EQ(sampled.gpu_timing_snapshot().status, "pending");
    const auto before = device.queue_submission_snapshot().queue_submit_calls;
    EXPECT_THROW(calls[0]->submit({}, values(99), true), vulkan::ExecutionDomain::Busy);
    EXPECT_THROW(calls[0]->submit({}, {1, 2}), std::invalid_argument);
    EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, before);
    selected_gate.release();
    other.wait();
    sampled.wait();
    check(0, values(1));
    check(1, values(2));
    EXPECT_EQ(other.gpu_timing_snapshot().status, "unavailable");
    const auto original = sampled.gpu_timing_snapshot();
    if (device.queue_timestamp_valid_bits(device.compute_queue_family_index()))
      EXPECT_TRUE(original.available);
    else
      EXPECT_EQ(original.status, "unsupported");
    for (std::uint64_t iteration = 0; iteration < 1000; ++iteration) {
      const auto left = values(10 + iteration * 2);
      const auto right = values(11 + iteration * 2);
      const auto before_pair = device.queue_submission_snapshot().queue_submit_calls;
      auto a = calls[0]->submit({}, left, iteration % 113 == 0);
      auto b = calls[1]->submit({}, right, iteration % 127 == 0);
      b.wait();
      // Do not poll a: domain collection must freeze its query before reuse.
      check(0, left);
      check(1, right);
      EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, before_pair + 2);
    }
    const auto preserved = sampled.gpu_timing_snapshot();
    EXPECT_EQ(preserved.available, original.available);
    EXPECT_EQ(preserved.duration_ns, original.duration_ns);
    domain->close();
    EXPECT_EQ(sampled.gpu_timing_snapshot().duration_ns, original.duration_ns);
  }
  registry.retire();
}

TEST(VulkanExecutionQueue, OrdinaryPrefixesSupportBidirectionalDependenciesAndFanout) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  if (!device.vk_caps().timeline_semaphore || device.created_compute_queue_count() < 3)
    GTEST_SKIP();
  auto owner = std::make_shared<RuntimeFaultDomain>(Arch::vulkan, 41);
  device.set_backend_fault_reporter(owner);
  auto *ordinary = static_cast<vulkan::VulkanStream *>(device.get_compute_stream());
  using Queue = vulkan::VulkanExecutionQueue;
  Queue first(device, 2), second(device, 2);
  auto [storage, result] = device.allocate_memory_unique(
      {12, false, true, false, AllocUsage::Storage});
  ASSERT_EQ(result, RhiResult::success);
  auto [publish, status] = ordinary->new_command_list_unique();
  ASSERT_EQ(status, RhiResult::success);
  publish->buffer_fill(storage->get_ptr(), 4, 17);
  auto copy1 = first.new_command_list();
  copy1->buffer_copy(storage->get_ptr(4), storage->get_ptr(), 4);
  auto copy2 = second.new_command_list();
  copy2->buffer_copy(storage->get_ptr(8), storage->get_ptr(), 4);
  auto commands1 = first.prepare({copy1.get()});
  auto commands2 = second.prepare({copy2.get()});
  RuntimeCompletion ticket;
  Queue::Point prefix;
  {
    TimelineGate gate(device, 0);
    const auto before = device.queue_submission_snapshot().queue_submit_calls;
    auto binary = ordinary->submit(publish.get());
    EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, before + 1);
    ticket = RuntimeCompletion::from_stream_semaphore(Arch::vulkan, 41, 1, binary, owner);
    prefix = std::dynamic_pointer_cast<vulkan::VulkanTimelinePoint>(
        ticket.reusable_stream_dependency(Arch::vulkan, 41, owner));
    ASSERT_TRUE(prefix);
    const auto a = first.submit(commands1, {prefix});
    const auto b = second.submit(commands2, {prefix});
    EXPECT_FALSE(a->is_ready());
    EXPECT_FALSE(b->is_ready());
    EXPECT_FALSE(ticket.done());
    gate.release();
    // Binary waits also require the producer's transitive signal operations
    // to be submitted. Release the artificial host gate before consuming the
    // legacy token; timeline fanout above can safely precede that release.
    auto consumed = ordinary->submit_dependency({binary});
    EXPECT_TRUE(a->wait());
    EXPECT_TRUE(b->wait());
    ticket.wait();
    EXPECT_TRUE(consumed->wait());
  }
  EXPECT_EQ(ticket.reusable_stream_dependency(Arch::vulkan, 41, owner), prefix);
  void *mapped = nullptr;
  ASSERT_EQ(device.map(*storage, &mapped), RhiResult::success);
  const auto *values = static_cast<const uint32_t *>(mapped);
  EXPECT_EQ(values[1], 17u);
  EXPECT_EQ(values[2], 17u);
  device.unmap(*storage);
  // Reverse direction: ordinary submission waits on the exact independent
  // prefix, including two values from one timeline (deduplicated to the tail).
  auto update = first.new_command_list();
  update->buffer_fill(storage->get_ptr(), 4, 91);
  {
    TimelineGate gate(device, first.queue_index());
    auto a = first.submit(first.prepare({update.get()}));
    auto b = first.submit(first.prepare({}), {a});
    auto [consume, created] = ordinary->new_command_list_unique();
    ASSERT_EQ(created, RhiResult::success);
    consume->buffer_copy(storage->get_ptr(8), storage->get_ptr(), 4);
    const auto before = device.queue_submission_snapshot().queue_submit_calls;
    auto observed = ordinary->submit(consume.get(), {a, b});
    EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, before + 1);
    EXPECT_FALSE(observed->is_ready());
    gate.release();
    EXPECT_TRUE(observed->wait());
  }
  ASSERT_EQ(device.map(*storage, &mapped), RhiResult::success);
  EXPECT_EQ(static_cast<const uint32_t *>(mapped)[2], 91u);
  device.unmap(*storage);
  // A host-observed prefix can still be waited on repeatedly.
  EXPECT_TRUE(first.submit(commands1, {prefix})->wait());
  EXPECT_TRUE(second.submit(commands2, {prefix})->wait());
  ordinary->command_sync();
}

TEST(VulkanExecutionQueue, OrdinaryBatchPublishesOnlyEnqueuedPrefixesAndRetiresThem) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  RuntimeCompletion ticket;
  std::shared_ptr<vulkan::VulkanTimelinePoint> retained;
  auto owner = std::make_shared<RuntimeFaultDomain>(Arch::vulkan, 42);
  {
    vulkan::VulkanDeviceCreator creator({});
    auto &device = *creator.device();
    if (!device.vk_caps().timeline_semaphore || device.created_compute_queue_count() < 2)
      GTEST_SKIP();
    device.set_backend_fault_reporter(owner);
    auto *ordinary = static_cast<vulkan::VulkanStream *>(device.get_compute_stream());
    vulkan::VulkanExecutionQueue consumer(device, 2);
    auto commands = consumer.prepare({});
    const auto before = device.queue_submission_snapshot();
    ordinary->begin_submission_batch();
    auto first = ordinary->submit(nullptr);
    auto last = ordinary->submit(nullptr);
    ticket = RuntimeCompletion::from_stream_semaphore(Arch::vulkan, 42, 1, last, owner);
    retained = std::dynamic_pointer_cast<vulkan::VulkanTimelinePoint>(
        ticket.reusable_stream_dependency(Arch::vulkan, 42, owner));
    ASSERT_TRUE(retained);
    EXPECT_FALSE(retained->is_ready());
    EXPECT_THROW(retained->wait(), std::invalid_argument);
    EXPECT_THROW(consumer.submit(commands, {retained}), std::invalid_argument);
    EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, before.queue_submit_calls);
    EXPECT_FALSE(consumer.latest());
    EXPECT_EQ(ordinary->end_submission_batch(), last);
    auto submitted = device.queue_submission_snapshot();
    EXPECT_EQ(submitted.queue_submit_calls, before.queue_submit_calls + 1);
    EXPECT_EQ(submitted.batched_queue_submit_calls, before.batched_queue_submit_calls + 1);
    EXPECT_EQ(submitted.batched_command_buffers, before.batched_command_buffers + 2);
    EXPECT_TRUE(consumer.submit(commands, {retained})->wait());
    ticket.wait();
    EXPECT_TRUE(ordinary->is_last_submission(first));
    EXPECT_TRUE(ordinary->is_last_submission(last));
    ordinary->command_sync();
  }
  // Device retirement freezes the prefix without requiring surviving streams.
  EXPECT_TRUE(retained->is_ready());
  EXPECT_TRUE(retained->wait());
  EXPECT_TRUE(ticket.done());
  EXPECT_NO_THROW(ticket.wait());
}

TEST(VulkanExecutionQueue, PhysicalLeasesAreExclusiveAndReusable) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  if (!device.vk_caps().timeline_semaphore || device.created_compute_queue_count() < 2)
    GTEST_SKIP();
  using Queue = vulkan::VulkanExecutionQueue;
  EXPECT_THROW(Queue(device, 0), std::invalid_argument);
  std::vector<std::unique_ptr<Queue>> queues;
  for (uint32_t i = 1; i < device.created_compute_queue_count(); ++i) {
    queues.push_back(std::make_unique<Queue>(device, 1));
    EXPECT_EQ(queues.back()->queue_index(), i);
  }
  EXPECT_THROW(Queue(device, 1), std::runtime_error);
  auto &old = *queues.back();
  auto empty = old.prepare({});
  old.close();
  EXPECT_THROW(old.submit(empty), std::runtime_error);
  Queue reused(device, 1);
  EXPECT_EQ(reused.queue_index(), old.queue_index());
  auto point = reused.submit(reused.prepare({}));
  ASSERT_TRUE(point->wait());
  reused.retire_timeline();
  EXPECT_TRUE(point->is_ready());
  EXPECT_TRUE(point->wait());
}

TEST(VulkanExecutionQueue, RuntimeCompletionPreservesReusablePrefixAndOwnerChecks) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  RuntimeCompletion ticket;
  std::shared_ptr<ReusableStreamDependency> retired_origin;
  auto owner = std::make_shared<RuntimeFaultDomain>(Arch::vulkan, 19);
  auto other = std::make_shared<RuntimeFaultDomain>(Arch::vulkan, 19);
  {
    vulkan::VulkanDeviceCreator creator({});
    auto &device = *creator.device();
    if (!device.vk_caps().timeline_semaphore || device.created_compute_queue_count() < 3)
      GTEST_SKIP();
    using Queue = vulkan::VulkanExecutionQueue;
    Queue producer(device, 1), consumer(device, 2);
    auto commands = producer.prepare({});
    const auto before = device.queue_submission_snapshot();
    EXPECT_THROW(producer.submit(commands, {}, [&](std::size_t, const auto &point) {
      auto reservation = RuntimeCompletion::from_reusable_stream_dependency(
          Arch::vulkan, 19, 1, point, owner);
      throw std::bad_alloc();
    }), std::bad_alloc);
    EXPECT_FALSE(producer.latest());
    EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, before.queue_submit_calls);

    TimelineGate gate(device, producer.queue_index());
    const auto point = producer.submit(commands, {}, [&](std::size_t, const auto &prefix) {
      ticket = RuntimeCompletion::from_reusable_stream_dependency(
          Arch::vulkan, 19, 1, prefix, owner);
    });
    auto retained = ticket.reusable_stream_dependency(Arch::vulkan, 19, owner);
    EXPECT_EQ(retained.get(), point.get());
    EXPECT_THROW(ticket.reusable_stream_dependency(Arch::vulkan, 19, other),
                  std::invalid_argument);
    EXPECT_THROW(ticket.reusable_stream_dependency(Arch::vulkan, 20, owner),
                  std::invalid_argument);
    EXPECT_THROW(ticket.reusable_stream_dependency(Arch::cuda, 19, owner),
                  std::invalid_argument);
    bool admitted = false;
    EXPECT_THROW(producer.submit(commands, {}, [&](std::size_t, const auto &) {
      admitted = true;
    }), Queue::Busy);
    EXPECT_FALSE(admitted);
    CompletionWaitSet progress({ticket}, std::chrono::milliseconds(1));
    EXPECT_EQ(progress.wait(std::chrono::nanoseconds(0)).status,
              CompletionWaitSet::Status::timeout);
    gate.release();
    EXPECT_EQ(progress.wait(std::chrono::seconds(3)).status,
              CompletionWaitSet::Status::ready);
    EXPECT_TRUE(ticket.done());
    EXPECT_FALSE(ticket.has_backend_work());
    producer.close();
    // Host observation and producer close do not erase the device dependency.
    auto dependency = std::dynamic_pointer_cast<vulkan::VulkanTimelinePoint>(
        ticket.reusable_stream_dependency(Arch::vulkan, 19, owner));
    ASSERT_EQ(dependency.get(), point.get());
    auto consumer_commands = consumer.prepare({});
    const auto first = consumer.submit(consumer_commands, {dependency});
    const auto second = consumer.submit(consumer_commands, {dependency});
    ASSERT_TRUE(first->wait());
    ASSERT_TRUE(second->wait());

    auto ordinary = RuntimeCompletion::from_stream_semaphore(
        Arch::vulkan, 19, 2, point, owner);
    ordinary.wait();
    EXPECT_THROW(ordinary.reusable_stream_dependency(Arch::vulkan, 19, owner),
                  std::invalid_argument);
    EXPECT_FALSE(RuntimeCompletion::completed(Arch::vulkan, 19, 0, owner)
                     .reusable_stream_dependency(Arch::vulkan, 19, owner));
    auto failed = RuntimeCompletion::from_reusable_stream_dependency(
        Arch::vulkan, 19, 3, point, owner);
    failed.invalidate("test failed submission");
    failed.mark_completed();
    EXPECT_THROW(failed.reusable_stream_dependency(Arch::vulkan, 19, owner),
                  std::runtime_error);
    consumer.close();
    producer.retire_timeline();
    consumer.retire_timeline();
    retired_origin = ticket.reusable_stream_dependency(Arch::vulkan, 19, owner);
  }
  // Retired dependency/query state has no live device access.
  EXPECT_TRUE(retired_origin->is_ready());
  EXPECT_TRUE(retired_origin->wait());
  EXPECT_TRUE(ticket.done());
  EXPECT_NO_THROW(ticket.wait());
  owner->begin_finalizing();
  owner->mark_finalized();
  EXPECT_ANY_THROW(ticket.reusable_stream_dependency(Arch::vulkan, 19, owner));
  EXPECT_TRUE(ticket.done());
}

TEST(VulkanExecutionQueue, DeviceRetiresPrefixesAfterQueueOwnersAreGone) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  using Queue = vulkan::VulkanExecutionQueue;
  auto creator = std::make_unique<vulkan::VulkanDeviceCreator>(
      vulkan::VulkanDeviceCreator::Params{});
  auto &device = *creator->device();
  if (!device.vk_caps().timeline_semaphore || device.created_compute_queue_count() < 2)
    GTEST_SKIP();
  auto fault = std::make_shared<RuntimeFaultDomain>(Arch::vulkan, 29);
  device.set_backend_fault_reporter(fault);
  std::vector<RuntimeCompletion> tickets;
  std::vector<Queue::Point> points;
  std::weak_ptr<vulkan::VulkanTimelinePoint> released;
  // Reuse one physical queue; keep only selected old completion prefixes.
  // The retirement registry must not keep a Point, command, or queue owner.
  for (int i = 0; i < 24; ++i) {
    Queue queue(device, 1);
    auto point = queue.submit(queue.prepare({}), points);
    ASSERT_TRUE(point->wait());
    queue.close();
    if (i % 3 == 0) {
      points.push_back(point);
      tickets.push_back(RuntimeCompletion::from_reusable_stream_dependency(
          Arch::vulkan, 29, i + 1, point, fault));
    } else {
      released = point;
    }
  }
  EXPECT_TRUE(released.expired());
  std::atomic<bool> stop{false};
  std::exception_ptr failure;
  std::thread observer([&] {
    try {
      while (!stop.load(std::memory_order_acquire)) {
        for (const auto &point : points) {
          if (!point->is_ready() || !point->wait())
            throw std::runtime_error("completed prefix became pending");
        }
        std::this_thread::yield();
      }
    } catch (...) {
      failure = std::current_exception();
    }
  });
  // No manual retire_timeline call and no surviving Queue owner. Queries may
  // race teardown, but native semaphore operations must finish before exit.
  creator.reset();
  stop.store(true, std::memory_order_release);
  observer.join();
  EXPECT_FALSE(failure);
  for (std::size_t i = 0; i < points.size(); ++i) {
    EXPECT_TRUE(points[i]->is_ready());
    EXPECT_TRUE(points[i]->wait());
    EXPECT_TRUE(tickets[i].done());
    EXPECT_NO_THROW(tickets[i].wait());
  }
}

TEST(VulkanExecutionQueue, DeviceRetirementDoesNotInventCompletionAfterFault) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  using Queue = vulkan::VulkanExecutionQueue;
  Queue::Point point;
  auto fault = std::make_shared<RuntimeFaultDomain>(Arch::vulkan, 30);
  {
    vulkan::VulkanDeviceCreator creator({});
    auto &device = *creator.device();
    if (!device.vk_caps().timeline_semaphore || device.created_compute_queue_count() < 2)
      GTEST_SKIP();
    device.set_backend_fault_reporter(fault);
    {
      Queue queue(device, 1);
      point = queue.submit(queue.prepare({}));
      ASSERT_TRUE(point->wait());
      queue.close();
    }
    // Inject lifecycle failure only after real device work drained. Teardown
    // cannot use a successful vkDeviceWaitIdle as its completion proof now.
    fault->report_fatal({Arch::vulkan, VK_ERROR_DEVICE_LOST, 7,
                        "test retirement", "retirement first failure"});
  }
  for (bool wait : {false, true}) {
    try {
      wait ? point->wait() : point->is_ready();
      FAIL() << "failed timeline was reported complete";
    } catch (const std::exception &error) {
      EXPECT_NE(std::string(error.what()).find("retirement first failure"),
                std::string::npos);
    }
  }
}

TEST(VulkanExecutionQueue, BusyIsAtomicAndTimelineDependenciesCanFanOut) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  if (!device.vk_caps().timeline_semaphore || device.created_compute_queue_count() < 3)
    GTEST_SKIP();
  using Queue = vulkan::VulkanExecutionQueue;
  Queue producer(device, 1), consumer(device, 2);
  auto [storage, status] = device.allocate_memory_unique(
      {8, false, true, false, AllocUsage::Storage});
  ASSERT_EQ(status, RhiResult::success);
  auto produce = producer.new_command_list();
  produce->buffer_fill(storage->get_ptr(), 4, 13);
  produce->memory_barrier();
  auto peer = consumer.new_command_list();
  peer->buffer_fill(storage->get_ptr(4), 4, 17);
  peer->memory_barrier();
  auto copy = consumer.new_command_list();
  copy->buffer_copy(storage->get_ptr(4), storage->get_ptr(), 4);
  copy->memory_barrier();
  const auto produced_commands = producer.prepare({produce.get()});
  const auto peer_commands = consumer.prepare({peer.get()});
  const auto copy_commands = consumer.prepare({copy.get()});
  TimelineGate gate(device, producer.queue_index());
  const auto pending = producer.submit(produced_commands);
  ASSERT_FALSE(pending->is_ready());
  bool admitted = false;
  EXPECT_THROW(producer.submit(produced_commands, {},
                               [&](std::size_t, const auto &) { admitted = true; }), Queue::Busy);
  EXPECT_FALSE(admitted);
  EXPECT_EQ(producer.latest(), pending);
  const auto independent = consumer.submit(peer_commands);
  EXPECT_TRUE(ready_before_deadline(independent));
  EXPECT_FALSE(pending->is_ready());
  EXPECT_EQ(consumer.collect(2), 1u);
  EXPECT_THROW(consumer.submit(peer_commands, {nullptr},
                               [&](std::size_t, const auto &) { admitted = true; }),
               std::invalid_argument);
  EXPECT_FALSE(admitted);
  TimelineGate peer_gate(device, consumer.queue_index());
  const auto first = consumer.submit(copy_commands, {pending, pending});
  const auto second = consumer.submit(copy_commands, {pending});
  EXPECT_FALSE(first->is_ready());
  EXPECT_FALSE(second->is_ready());
  gate.release();
  ASSERT_TRUE(pending->wait());
  producer.close();
  EXPECT_FALSE(first->is_ready());
  EXPECT_FALSE(second->is_ready());
  peer_gate.release();
  ASSERT_TRUE(second->wait());
  EXPECT_TRUE(first->is_ready());
  // A completed producer prefix remains a device memory dependency after
  // close; a binary semaphore could not be consumed for a third time.
  ASSERT_TRUE(consumer.submit(copy_commands, {pending})->wait());
  consumer.close();
  producer.retire_timeline();
  consumer.retire_timeline();
  EXPECT_TRUE(pending->is_ready());
  EXPECT_TRUE(second->wait());
  void *mapped = nullptr;
  ASSERT_EQ(device.map(*storage, &mapped), RhiResult::success);
  EXPECT_EQ(static_cast<uint32_t *>(mapped)[0], 13u);
  EXPECT_EQ(static_cast<uint32_t *>(mapped)[1], 13u);
  device.unmap(*storage);
}

TEST(VulkanRuntimeDomain, AdmissionIsAtomicAndPeersKeepProgress) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  if (!device.vk_caps().timeline_semaphore || device.created_compute_queue_count() < 3)
    GTEST_SKIP();
  auto owner = std::make_shared<RuntimeFaultDomain>(Arch::vulkan, 39);
  device.set_backend_fault_reporter(owner);
  vulkan::ExecutionDomainRegistry registry(device, owner);
  auto left = registry.create({1, 2});
  auto right = registry.create({1, 2});
  auto foreign = std::make_shared<RuntimeFaultDomain>(Arch::vulkan, 39);
  EXPECT_TRUE(left->belongs_to(owner));
  EXPECT_FALSE(left->belongs_to(foreign));
  auto [storage, result] = device.allocate_memory_unique(
      {8, false, true, false, AllocUsage::Storage});
  ASSERT_EQ(result, RhiResult::success);
  auto left_cmd = left->new_command_list();
  left_cmd->buffer_fill(storage->get_ptr(), 4, 37);
  left_cmd->memory_barrier();
  auto right_cmd = right->new_command_list();
  right_cmd->buffer_fill(storage->get_ptr(4), 4, 91);
  right_cmd->memory_barrier();
  auto left_work = left->prepare_commands({left_cmd.get()});
  auto right_work = right->prepare_commands({right_cmd.get()});
  auto before = device.queue_submission_snapshot().queue_submit_calls;
  EXPECT_THROW(left->submit(left_work, {}, {}, [](auto, const auto &) {
    throw std::invalid_argument("rejected parameter snapshot");
  }), std::invalid_argument);
  EXPECT_EQ(left->snapshot().pending, 0u);
  EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, before);
  EXPECT_THROW(left->submit(left_work,
      {RuntimeCompletion::completed(Arch::vulkan, 39, 0, foreign)}),
      std::invalid_argument);
  {
    TimelineGate default_gate(device, 0);
    TimelineGate gate(device, left->snapshot().queue_index);
    const auto pending = left->submit(left_work);
    bool admitted = false;
    before = device.queue_submission_snapshot().queue_submit_calls;
    EXPECT_THROW(left->submit(left_work, {}, {}, [&](auto, const auto &) {
      admitted = true;
    }), vulkan::ExecutionDomain::Busy);
    EXPECT_FALSE(admitted);
    EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, before);
    auto peer = right->submit(right_work);
    CompletionWaitSet progress({peer}, std::chrono::milliseconds(1));
    EXPECT_EQ(progress.wait(std::chrono::seconds(3)).status,
              CompletionWaitSet::Status::ready);
    EXPECT_FALSE(pending.done());
    auto dependent = right->submit(right_work, {pending});
    EXPECT_FALSE(dependent.done());
    gate.release();
    dependent.wait();
    EXPECT_TRUE(pending.done());
    EXPECT_GT(dependent.sequence(), peer.sequence());
    EXPECT_EQ(dependent.program_domain(), owner->program_domain());
    left->close();
    // Completion observation and domain close preserve cross-queue ordering.
    right->submit(right_work, {pending}).wait();
    right->submit(right_work, {pending}).wait();
    right->close();
  }
  void *mapped = nullptr;
  ASSERT_EQ(device.map(*storage, &mapped), RhiResult::success);
  EXPECT_EQ(static_cast<std::uint32_t *>(mapped)[0], 37u);
  EXPECT_EQ(static_cast<std::uint32_t *>(mapped)[1], 91u);
  device.unmap(*storage);
  registry.retire();
  EXPECT_FALSE(registry.accepting());
  EXPECT_THROW(registry.create({1, 1}), std::runtime_error);
}

TEST(VulkanRuntimeDomain, InstanceRetirementReusesBudgetWhilePeerIsPending) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  if (!device.vk_caps().timeline_semaphore || device.created_compute_queue_count() < 2)
    GTEST_SKIP();
  auto owner = std::make_shared<RuntimeFaultDomain>(Arch::vulkan, 40);
  device.set_backend_fault_reporter(owner);
  vulkan::ExecutionDomainRegistry registry(device, owner);
  auto domain = registry.create({2, 2});
  struct Client : RuntimeClient {
    std::function<void()> callback;
    void retire() override { callback(); }
  };
  int retirements = 0;
  auto client = std::make_shared<Client>();
  client->callback = [&] {
    (void)domain->snapshot();  // No admission mutex during resource callbacks.
    ++retirements;
  };
  const auto prepare = [](const auto &) {};
  auto first = domain->prepare_owned_instance(client, prepare);
  auto second = domain->prepare_owned_instance(client, prepare);
  auto work = domain->prepare_commands({});
  auto completed = domain->submit_instance(first, work);
  completed.wait();
  EXPECT_THROW(domain->prepare_owned_instance(client, prepare), std::runtime_error);
  EXPECT_EQ(retirements, 0);
  {
    TimelineGate gate(device, domain->snapshot().queue_index);
    auto pending = domain->submit_instance(second, work);
    EXPECT_FALSE(pending.done());
    domain->retire_instance(first);
    EXPECT_EQ(retirements, 1);
    EXPECT_TRUE(domain->instance_retired(first));
    EXPECT_FALSE(pending.done());
    EXPECT_THROW(domain->submit_instance(first, work), std::invalid_argument);
    auto reused = domain->prepare_owned_instance(client, prepare);
    domain->retire_instance(reused);
    EXPECT_EQ(retirements, 2);
    std::shared_ptr<vulkan::ExecutionInstance> rejected;
    EXPECT_THROW(domain->prepare_owned_instance(client, [&](const auto &instance) {
      rejected = instance;
      throw std::invalid_argument("partial preparation rejected");
    }), std::invalid_argument);
    EXPECT_EQ(retirements, 3);
    EXPECT_TRUE(domain->instance_retired(rejected));
    EXPECT_TRUE(domain->snapshot().accepting);
    auto recovered = domain->prepare_owned_instance(client, prepare);
    EXPECT_FALSE(pending.done());
    gate.release();
    pending.wait();
    domain->submit_instance(recovered, work).wait();
  }
  domain->close();
  EXPECT_EQ(retirements, 5);
  EXPECT_TRUE(domain->instance_retired(second));
  EXPECT_TRUE(completed.done());
  const auto previous_id = domain->snapshot().identity;
  // Releasing the native queue permits registry-slot reuse without revoking
  // old completion identities or requiring a runtime-wide synchronize.
  auto replacement = registry.create({1, 1});
  EXPECT_NE(replacement->snapshot().identity, previous_id);
  replacement->submit(replacement->prepare_commands({}), {completed}).wait();
  registry.retire();
  EXPECT_TRUE(replacement->snapshot().closed);
}

TEST(VulkanRuntimeDomain, RejectedAndCollectedOperandsRetireOutsideAdmission) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  if (!device.vk_caps().timeline_semaphore || device.created_compute_queue_count() < 2)
    GTEST_SKIP();
  auto owner = std::make_shared<RuntimeFaultDomain>(Arch::vulkan, 41);
  device.set_backend_fault_reporter(owner);
  vulkan::ExecutionDomain domain(device, owner, {1, 1});
  int released = 0;
  const auto operand = [&] {
    return std::shared_ptr<void>(reinterpret_cast<void *>(1), [&](void *) {
      (void)domain.snapshot();
      ++released;
    });
  };
  EXPECT_THROW(domain.submit(domain.prepare_commands({}, operand()), {}, {},
      [](auto, const auto &) { throw std::invalid_argument("reject"); }),
      std::invalid_argument);
  EXPECT_EQ(released, 1);
  auto first = domain.submit(domain.prepare_commands({}, operand()));
  first.wait();
  EXPECT_EQ(released, 1);
  domain.submit(domain.prepare_commands({})).wait();
  EXPECT_EQ(released, 2);
  domain.close();
}

TEST(VulkanExecutionQueue, CompletedOperandReleaseDoesNotHoldAdmissionLock) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator({});
  auto &device = *creator.device();
  if (!device.vk_caps().timeline_semaphore || device.created_compute_queue_count() < 2)
    GTEST_SKIP();
  vulkan::VulkanExecutionQueue queue(device, 1);
  bool retired = false;
  auto owner = std::shared_ptr<void>(reinterpret_cast<void *>(1), [&](void *) {
    EXPECT_EQ(queue.pending(), 0u);
    retired = true;
  });
  auto commands = queue.prepare({}, std::move(owner));
  auto point = queue.submit(std::move(commands));
  ASSERT_TRUE(point->wait());
  EXPECT_FALSE(retired);
  EXPECT_EQ(queue.collect(1), 1u);
  EXPECT_TRUE(retired);
  // A rejected host admission also destroys its private operands outside the
  // lock, does not publish a timeline value, and leaves the queue reusable.
  const auto before = queue.latest();
  retired = false;
  auto rejected = queue.prepare({}, std::shared_ptr<void>(
      reinterpret_cast<void *>(1), [&](void *) {
        EXPECT_EQ(queue.pending(), 0u);
        retired = true;
      }));
  EXPECT_THROW(queue.submit(std::move(rejected), {}, [](std::size_t, const auto &) {
                 throw std::invalid_argument("invalid parameter snapshot");
               }), std::invalid_argument);
  EXPECT_TRUE(retired);
  EXPECT_EQ(queue.latest(), before);
  EXPECT_TRUE(queue.submit(queue.prepare({}))->wait());
  queue.retire_timeline();
}

}  // namespace taichi::lang
