#include "gtest/gtest.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_execution_queue.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"

#include <chrono>
#include <set>
#include <thread>

namespace taichi::lang {

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
class TimelineGate {
 public:
  TimelineGate(vulkan::VulkanDevice &device, uint32_t queue_index)
      : device_(device.vk_device()) {
    VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    semaphore_ = vkapi::create_semaphore(device_, 0, &type);
    const uint64_t value = 1;
    VkTimelineSemaphoreSubmitInfo values{
        VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    values.waitSemaphoreValueCount = 1;
    values.pWaitSemaphoreValues = &value;
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo wait{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    wait.pNext = &values;
    wait.waitSemaphoreCount = 1;
    wait.pWaitSemaphores = &semaphore_->semaphore;
    wait.pWaitDstStageMask = &stage;
    fence_ = vkapi::create_fence(device_, 0);
    if (device_.functions().vkQueueSubmit(device.compute_queue_at(queue_index),
                                          1, &wait, fence_->fence) != VK_SUCCESS)
      throw std::runtime_error("Cannot enqueue test timeline gate");
  }
  ~TimelineGate() {
    release();
    device_.functions().vkWaitForFences(device_, 1, &fence_->fence, VK_TRUE,
                                       UINT64_MAX);
  }
  void release() {
    if (!released_) {
      VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
      signal.semaphore = semaphore_->semaphore;
      signal.value = 1;
      EXPECT_EQ(device_.functions().vkSignalSemaphore(device_, &signal), VK_SUCCESS);
      released_ = true;
    }
  }
 private:
  vkapi::DeviceHandle device_;
  vkapi::IVkSemaphore semaphore_;
  vkapi::IVkFence fence_;
  bool released_{false};
};

bool ready_before_deadline(const vulkan::VulkanExecutionQueue::Point &point) {
  const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (!point->is_ready() && std::chrono::steady_clock::now() < end)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  return point->is_ready();
}
}  // namespace

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
                               [&](std::size_t) { admitted = true; }), Queue::Busy);
  EXPECT_FALSE(admitted);
  EXPECT_EQ(producer.latest(), pending);
  const auto independent = consumer.submit(peer_commands);
  EXPECT_TRUE(ready_before_deadline(independent));
  EXPECT_FALSE(pending->is_ready());
  EXPECT_EQ(consumer.collect(2), 1u);
  EXPECT_THROW(consumer.submit(peer_commands, {nullptr},
                               [&](std::size_t) { admitted = true; }),
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
  EXPECT_THROW(queue.submit(std::move(rejected), {}, [](std::size_t) {
                 throw std::invalid_argument("invalid parameter snapshot");
               }), std::invalid_argument);
  EXPECT_TRUE(retired);
  EXPECT_EQ(queue.latest(), before);
  EXPECT_TRUE(queue.submit(queue.prepare({}))->wait());
  queue.retire_timeline();
}

}  // namespace taichi::lang
