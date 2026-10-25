#include "gtest/gtest.h"
#include "taichi/analysis/range_dispatch.h"

namespace taichi::lang {
TEST(DeviceRangeProof, RejectsUnknownBoundsAndPhysicalThreadBehavior) {
  const auto make = [] {
    auto task = std::make_unique<OffloadedStmt>(
        OffloadedTaskType::range_for, Arch::vulkan, nullptr);
    task->const_begin = true;
    task->begin_value = -3;
    task->end_offset = 4;
    task->body = std::make_unique<Block>();
    return task;
  };
  auto ordinary = make();
  ordinary->host_launch_range =
      HostRangeBounds{{0, {}, false}, {1, {0}, true}};
  EXPECT_TRUE(describe_launch_range_for_bounds(*ordinary)->host_launch);
  EXPECT_TRUE(supports_device_range_dispatch(*ordinary));
  ordinary->one_to_one = true;
  const auto fixed = describe_launch_range_for_bounds(*ordinary);
  EXPECT_TRUE(fixed->host_launch);
  EXPECT_EQ(fixed->empty_group_elision, EmptyRangeGroupProof::fixed_mapping);
  EXPECT_FALSE(fixed->uses_host_launch_grid());
  EXPECT_FALSE(supports_device_range_dispatch(*ordinary));
  auto unknown = make();
  auto expression = Stmt::make_typed<ConstStmt>(TypedConstant(5));
  unknown->end_stmt = expression.get();
  EXPECT_FALSE(supports_device_range_dispatch(*unknown));
  for (const auto &name : {"vkGlobalThreadIdx", "workgroupBarrier", "future_intrinsic"}) {
    auto task = make();
    task->body->insert(std::make_unique<InternalFuncStmt>(
        name, std::vector<Stmt *>{}));
    EXPECT_FALSE(supports_device_range_dispatch(*task));
  }
  auto physical = make();
  physical->body->insert(std::make_unique<GlobalThreadIndexStmt>());
  EXPECT_FALSE(supports_device_range_dispatch(*physical));
  auto shared = make();
  shared->body->insert(std::make_unique<AllocaStmt>(
      std::vector<int>{32}, PrimitiveType::i32, true));
  EXPECT_FALSE(supports_device_range_dispatch(*shared));
  auto cpu = make();
  cpu->device = Arch::x64;
  cpu->host_launch_range = ordinary->host_launch_range;
  const auto cpu_bounds = describe_launch_range_for_bounds(*cpu);
  EXPECT_TRUE(cpu_bounds->host_launch);
  EXPECT_EQ(cpu_bounds->empty_group_elision, EmptyRangeGroupProof::fixed_mapping);
  EXPECT_FALSE(cpu_bounds->supports_device_dispatch());
}
}  // namespace taichi::lang

#ifdef TI_WITH_VULKAN
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>
#include "taichi/runtime/gfx/range_dispatch.h"
#include "taichi/runtime/gfx/aot_module_loader_impl.h"
#include "taichi/runtime/gfx/aot_graph_data.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/launch_context_builder.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/rhi/vulkan/vulkan_execution_queue.h"
#include "tests/cpp/aot/vulkan/timeline_gate.h"

namespace taichi::lang {
namespace {
class VulkanRangeDispatch : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!vulkan::is_vulkan_api_available())
      GTEST_SKIP();
    creator = std::make_unique<vulkan::VulkanDeviceCreator>(
        vulkan::VulkanDeviceCreator::Params{});
    device = creator->device();
  }
  std::unique_ptr<DeviceAllocationGuard> allocate(std::size_t bytes) {
    auto [buffer, status] = device->allocate_memory_unique(
        {bytes, true, true, false, AllocUsage::Storage});
    TI_ASSERT(status == RhiResult::success);
    return std::move(buffer);
  }
  template <typename T, std::size_t N>
  void write(DeviceAllocation buffer, const std::array<T, N> &values) {
    void *mapped = nullptr;
    ASSERT_EQ(device->map(buffer, &mapped), RhiResult::success);
    std::memcpy(mapped, values.data(), sizeof(values));
    device->unmap(buffer);
  }
  template <typename T, std::size_t N>
  std::array<T, N> read(DeviceAllocation buffer) {
    std::array<T, N> values{};
    auto ptr = buffer.get_ptr();
    void *host = values.data();
    std::size_t bytes = sizeof(values);
    TI_ASSERT(device->readback_data(&ptr, &host, &bytes, 1) == RhiResult::success);
    return values;
  }
  std::unique_ptr<vulkan::VulkanDeviceCreator> creator;
  vulkan::VulkanDevice *device{nullptr};
};

TEST_F(VulkanRangeDispatch, SignedControllerDoesNotOverflowOrReadHostCounts) {
  auto bounds = allocate(8);
  spirv::TaskAttributes task;
  task.name = "signed_range";
  task.task_type = OffloadedTaskType::range_for;
  task.device_range_dispatch_version = 1;
  task.range_for_attribs = RangeForBounds{RangeBound::global_tmp(0),
                                        RangeBound::global_tmp(4)};
  task.range_for_attribs->empty_group_elision = EmptyRangeGroupProof::independent_iterations;
  task.advisory_total_num_threads = 128;
  task.advisory_num_threads_per_group = 32;
  gfx::RangeDispatch controller(*device, task, *bounds, 8);
  auto [commands, status] = device->get_compute_stream()->new_command_list_unique();
  ASSERT_EQ(status, RhiResult::success);
  controller.record_controller(*commands);
  commands->buffer_transition(
      controller.packet(), gfx::RangeDispatch::kPacketBytes,
      {BufferBarrierStage::Compute, BufferBarrierAccess::ShaderWrite,
       BufferBarrierStage::Transfer, BufferBarrierAccess::TransferRead});
  const std::array<std::array<int32_t, 2>, 7> cases{{
      {{0, 0}}, {{-3, 2}}, {{-3, 62}}, {{9, -2}},
      {{INT32_MIN, INT32_MAX}}, {{INT32_MAX - 1, INT32_MAX}}, {{0, 0}}}};
  const std::array<uint32_t, 7> expected{1, 1, 3, 1, 4, 1, 1};
  for (std::size_t i = 0; i < cases.size(); ++i) {
    write(*bounds, cases[i]);
    device->get_compute_stream()->submit_synced(commands.get());
    EXPECT_EQ((read<uint32_t, 3>(controller.packet())),
              (std::array<uint32_t, 3>{expected[i], 1, 1}));
  }
  // Two serial producers overwrite the same temporary words in one command
  // stream. Each following controller must consume its own current snapshot.
  gfx::RangeDispatch second(*device, task, *bounds, 8);
  auto observed = allocate(6 * sizeof(uint32_t));
  auto [sequence, sequence_status] =
      device->get_compute_stream()->new_command_list_unique();
  ASSERT_EQ(sequence_status, RhiResult::success);
  int index = 0;
  for (auto *range : {&controller, &second}) {
    sequence->memory_barrier();
    sequence->buffer_fill(bounds->get_ptr(), 4, 0);
    sequence->buffer_fill(bounds->get_ptr(4), 4, index == 0 ? 5 : 65);
    range->record_controller(*sequence);
    sequence->buffer_transition(
        range->packet(), gfx::RangeDispatch::kPacketBytes,
        {BufferBarrierStage::Compute, BufferBarrierAccess::ShaderWrite,
         BufferBarrierStage::Transfer, BufferBarrierAccess::TransferRead});
    sequence->buffer_copy(observed->get_ptr(index++ * 12), range->packet(), 12);
  }
  sequence->memory_barrier();
  device->get_compute_stream()->submit_synced(sequence.get());
  EXPECT_EQ((read<uint32_t, 6>(*observed)),
            (std::array<uint32_t, 6>{1, 1, 1, 3, 1, 1}));
  auto bad = task;
  bad.range_for_attribs->end = RangeBound::global_tmp(8);
  EXPECT_ANY_THROW(gfx::RangeDispatch::validate(bad, 8));
  bad = task;
  bad.device_range_dispatch_version = 0;
  EXPECT_ANY_THROW(gfx::RangeDispatch::validate(bad, 8));
}

TEST_F(VulkanRangeDispatch, CompiledMultiTaskSnapshotsRetainResetAndFinalization) {
  const auto *folder = std::getenv("TAICHI_VULKAN_RANGE_AOT_DIR");
  if (!folder)
    GTEST_SKIP() << "Requires the compiled multi-task range fixture";
  gfx::GfxRuntime::Params runtime_params;
  runtime_params.device = device;
  gfx::GfxRuntime runtime(runtime_params);
  gfx::AotModuleParams module_params;
  module_params.module_path = folder;
  module_params.runtime = &runtime;
  auto module = aot::Module::load(Arch::vulkan, module_params);
  ASSERT_TRUE(module);
  auto *kernel = dynamic_cast<gfx::KernelImpl *>(module->get_kernel("sequence"));
  ASSERT_TRUE(kernel);
  auto input = allocate(65 * sizeof(float));
  auto output = allocate(65 * sizeof(float));
  auto limits = allocate(2 * sizeof(int32_t));
  auto result = allocate(2 * sizeof(int32_t));
  Ndarray x(*input, PrimitiveType::f32, {65});
  Ndarray y(*output, PrimitiveType::f32, {65});
  Ndarray bounds(*limits, PrimitiveType::i32, {2});
  Ndarray status(*result, PrimitiveType::i32, {2});
  LaunchContextBuilder context(kernel);
  context.set_arg_ndarray({0}, x);
  context.set_arg_ndarray({1}, y);
  context.set_arg_ndarray({2}, bounds);
  context.set_arg_ndarray({3}, status);
  gfx::GfxRuntime::GraphDispatch dispatch{kernel->handle(), &context};
  const auto &tasks = kernel->params().kernel_attribs.tasks_attribs;
  for (uint32_t i = 0; i < tasks.size(); ++i)
    if (tasks[i].device_range_dispatch_version == 1)
      dispatch.device_range_tasks.push_back(i);
  ASSERT_EQ(dispatch.device_range_tasks.size(), 2);
  ASSERT_GT(tasks.size(), 2);
  auto fixed = runtime.prepare_fixed_graph({{dispatch, {}}}, {});
  auto replay = runtime.register_graph_replay(0xD151);
  std::array<float, 65> inputs{}, sentinel{};
  for (int i = 0; i < 65; ++i) {
    inputs[i] = float(i + 1);
    sentinel[i] = -777.f;
  }
  for (bool retained : {false, true}) {
    for (int count : {0, 5, 65, 0}) {
      runtime.synchronize();
      write(*input, inputs);
      write(*output, sentinel);
      write(*limits, std::array<int32_t, 2>{-3, count - 3});
      write(*result, std::array<int32_t, 2>{-1, -1});
      if (retained)
        ASSERT_TRUE(runtime.try_launch_graph({dispatch}, replay->replay_key(), nullptr));
      else
        fixed->launch_prepared();
      runtime.synchronize();
      const auto actual = read<float, 65>(*output);
      for (int i = 0; i < 65; ++i) {
        const float expected = i >= count ? -777.f :
            inputs[i] * inputs[i] + (i > 0 && i < count - 1 ? 2 * inputs[i] : 0);
        EXPECT_FLOAT_EQ(actual[i], expected);
      }
      EXPECT_EQ((read<int32_t, 2>(*result)), (std::array<int32_t, 2>{17, 22}));
    }
  }
  EXPECT_EQ(fixed->snapshot_stats().known_persistent_control_bytes, 24);
  auto invalid = dispatch;
  invalid.device_range_tasks.push_back(invalid.device_range_tasks.back());
  EXPECT_ANY_THROW(runtime.prepare_fixed_graph({{invalid, {}}}, {}));
  EXPECT_ANY_THROW(runtime.try_launch_graph({invalid}, replay->replay_key(), nullptr));
  runtime.synchronize();
}

TEST_F(VulkanRangeDispatch, PrivateWorkspacesAdjointsBusyAndRetirement) {
  const auto *folder = std::getenv("TAICHI_VULKAN_RANGE_AOT_DIR");
  if (!folder)
    GTEST_SKIP() << "Requires the compiled multi-task range fixture";
  if (!device->vk_caps().timeline_semaphore ||
      device->created_compute_queue_count() < 3)
    GTEST_SKIP() << "Requires two independent compute queues";
  gfx::GfxRuntime::Params params;
  params.device = device;
  gfx::GfxRuntime runtime(params);
  gfx::AotModuleParams module_params;
  module_params.module_path = folder;
  module_params.runtime = &runtime;
  std::shared_ptr<aot::Module> module = aot::Module::load(Arch::vulkan, module_params);
  ASSERT_TRUE(module);
  auto *forward = dynamic_cast<gfx::KernelImpl *>(module->get_kernel("square"));
  auto *backward = dynamic_cast<gfx::KernelImpl *>(module->get_kernel("square_grad"));
  ASSERT_TRUE(forward && backward);
  using Queue = vulkan::VulkanExecutionQueue;
  Queue left(*device, 1), right(*device, 1);
  struct Frame {
    std::array<std::shared_ptr<DeviceAllocationGuard>, 5> buffers;
    std::shared_ptr<gfx::GfxRuntime::FixedGraphWorkspace> workspace;
    std::shared_ptr<gfx::GfxRuntime::GraphReplayExecutable::Slot> payload;
    std::shared_ptr<const Queue::Commands> commands;
  };
  std::array<Frame, 2> frames;
  for (int lane = 0; lane < 2; ++lane) {
    auto &frame = frames[lane];
    auto &queue = lane == 0 ? left : right;
    for (int j = 0; j < 5; ++j)
      frame.buffers[j] = allocate((j == 4 ? 2 : 65) * sizeof(float));
    Ndarray x(*frame.buffers[0], PrimitiveType::f32, {65});
    Ndarray y(*frame.buffers[1], PrimitiveType::f32, {65});
    Ndarray dx(*frame.buffers[2], PrimitiveType::f32, {65});
    Ndarray dy(*frame.buffers[3], PrimitiveType::f32, {65});
    Ndarray bounds(*frame.buffers[4], PrimitiveType::i32, {2});
    LaunchContextBuilder primal(forward), adjoint(backward);
    std::vector<gfx::GfxRuntime::GraphRecordingOperation> operations;
    for (auto *context : {&primal, &adjoint}) {
      context->set_arg_ndarray_with_grad({0}, x, dx);
      context->set_arg_ndarray_with_grad({1}, y, dy);
      context->set_arg_ndarray({2}, bounds);
      auto *kernel = context == &primal ? forward : backward;
      gfx::GfxRuntime::GraphDispatch dispatch{kernel->handle(), context};
      const auto &tasks = kernel->params().kernel_attribs.tasks_attribs;
      for (uint32_t i = 0; i < tasks.size(); ++i)
        if (tasks[i].device_range_dispatch_version == 1)
          dispatch.device_range_tasks.push_back(i);
      ASSERT_EQ(dispatch.device_range_tasks.size(), 1);
      operations.push_back({dispatch, {}});
    }
    frame.workspace = runtime.create_fixed_graph_workspace();
    std::vector<std::shared_ptr<void>> owners{module};
    owners.insert(owners.end(), frame.buffers.begin(), frame.buffers.end());
    const auto before = device->queue_submission_snapshot().queue_submit_calls;
    frame.payload = runtime.prepare_fixed_commands(
        operations, queue.new_command_list(), frame.workspace, std::move(owners));
    EXPECT_EQ(device->queue_submission_snapshot().queue_submit_calls, before);
    frame.commands = queue.prepare({frame.payload->cmdlist.get()}, frame.payload);
    auto initialize = queue.new_command_list();
    frame.workspace->record_initialization(*initialize);
    ASSERT_TRUE(queue.submit(queue.prepare({initialize.get()}, frame.workspace))->wait());
  }
  const auto first_packet = [](const Frame &frame) {
    for (const auto &range : frame.payload->range_dispatches)
      if (range)
        return range->packet().alloc_id;
    return DeviceAllocationId{0};
  };
  EXPECT_NE(first_packet(frames[0]), first_packet(frames[1]));
  EXPECT_GT(frames[0].workspace->bytes(), 0);
  EXPECT_GT(frames[1].workspace->bytes(), 0);
  runtime.synchronize();
  for (int iteration = 0; iteration < 4; ++iteration) {
    const std::array<int, 4> sizes{0, 5, 65, 0};
    std::array<int, 2> counts{sizes[iteration], sizes[(iteration + 1) % 4]};
    for (int lane = 0; lane < 2; ++lane) {
      std::array<float, 65> values{};
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 65; ++i)
          values[i] = j == 0 ? float(i + lane + 1) : j == 1 ? -777.f : j == 2 ? 0.f : 2.f;
        write(*frames[lane].buffers[j], values);
      }
      write(*frames[lane].buffers[4], std::array<int32_t, 2>{-3, counts[lane] - 3});
    }
    {
      test_detail::TimelineGate ordinary(*device, 0);
      test_detail::TimelineGate blocked(*device, left.queue_index());
      const auto pending = left.submit(frames[0].commands);
      const auto submissions = device->queue_submission_snapshot().queue_submit_calls;
      bool admitted = false;
      EXPECT_THROW(left.submit(frames[0].commands, {},
                               [&](auto, const auto &) { admitted = true; }), Queue::Busy);
      EXPECT_FALSE(admitted);
      EXPECT_EQ(device->queue_submission_snapshot().queue_submit_calls, submissions);
      const auto other = right.submit(frames[1].commands);
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (!other->is_ready() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
      const bool independent = other->is_ready();
      EXPECT_FALSE(pending->is_ready());
      blocked.release();
      ordinary.release();
      ASSERT_TRUE(pending->wait());
      ASSERT_TRUE(other->wait());
      ASSERT_TRUE(independent);
    }
    for (int lane = 0; lane < 2; ++lane) {
      const auto y = read<float, 65>(*frames[lane].buffers[1]);
      const auto dx = read<float, 65>(*frames[lane].buffers[2]);
      for (int i = 0; i < 65; ++i) {
        const float x = float(i + lane + 1);
        EXPECT_FLOAT_EQ(y[i], i < counts[lane] ? x * x : -777.f);
        EXPECT_FLOAT_EQ(dx[i], i < counts[lane] ? 4 * x : 0.f);
      }
    }
  }
  left.close();
  frames[0] = {};
  // Releasing one instance's packet, descriptors and workspace cannot retire
  // the other's shared compiled code or independent resources.
  ASSERT_TRUE(right.submit(frames[1].commands)->wait());
  right.close();
}
}  // namespace
}  // namespace taichi::lang
#endif
