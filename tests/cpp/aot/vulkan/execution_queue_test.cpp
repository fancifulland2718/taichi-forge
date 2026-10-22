#include "gtest/gtest.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_execution_queue.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/codegen/spirv/spirv_ir_builder.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/program/runtime_completion.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/runtime/completion_wait_set.h"

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
