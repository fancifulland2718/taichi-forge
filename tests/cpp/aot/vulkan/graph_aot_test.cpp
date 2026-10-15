#include "gtest/gtest.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "tests/cpp/aot/gfx_utils.h"
#include "taichi/runtime/gfx/aot_module_loader_impl.h"
#include <limits>

using namespace taichi;
using namespace lang;

TEST(CGraphAotTest, VulkanRunCGraph2) {
  // Otherwise will segfault on macOS VM,
  // where Vulkan is installed but no devices are present
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  // Create Taichi Device for computation
  lang::vulkan::VulkanDeviceCreator::Params evd_params;
  evd_params.api_version = std::nullopt;
  auto embedded_device =
      std::make_unique<taichi::lang::vulkan::VulkanDeviceCreator>(evd_params);
  taichi::lang::vulkan::VulkanDevice *device_ =
      static_cast<taichi::lang::vulkan::VulkanDevice *>(
          embedded_device->device());
  aot_test_utils::run_cgraph2(Arch::vulkan, device_);
}

TEST(CGraphAotTest, VulkanRunCGraph1) {
  // Otherwise will segfault on macOS VM,
  // where Vulkan is installed but no devices are present
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  // Create Taichi Device for computation
  lang::vulkan::VulkanDeviceCreator::Params evd_params;
  evd_params.api_version = std::nullopt;
  auto embedded_device =
      std::make_unique<taichi::lang::vulkan::VulkanDeviceCreator>(evd_params);
  taichi::lang::vulkan::VulkanDevice *device_ =
      static_cast<taichi::lang::vulkan::VulkanDevice *>(
          embedded_device->device());
  aot_test_utils::run_cgraph1(Arch::vulkan, device_);
}

TEST(CGraphAotTest, VulkanMpm88) {
  // Otherwise will segfault on macOS VM,
  // where Vulkan is installed but no devices are present
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  // Create Taichi Device for computation
  lang::vulkan::VulkanDeviceCreator::Params evd_params;
  evd_params.api_version = std::nullopt;
  auto embedded_device =
      std::make_unique<taichi::lang::vulkan::VulkanDeviceCreator>(evd_params);
  taichi::lang::vulkan::VulkanDevice *device_ =
      static_cast<taichi::lang::vulkan::VulkanDevice *>(
          embedded_device->device());
  aot_test_utils::run_mpm88_graph(Arch::vulkan, device_);
}

TEST(CGraphAotTest, VulkanFixedControl) {
  if (!vulkan::is_vulkan_api_available()) {
    GTEST_SKIP() << "Vulkan is unavailable";
  }
  const auto *folder = std::getenv("TAICHI_AOT_FOLDER_PATH");
  ASSERT_NE(folder, nullptr);
  vulkan::VulkanDeviceCreator::Params params;
  vulkan::VulkanDeviceCreator creator(params);
  auto *device = creator.device();
  gfx::GfxRuntime::Params runtime_params;
  runtime_params.device = device;
  gfx::GfxRuntime runtime(std::move(runtime_params));
  using Strategy = gfx::GfxRuntime::GraphStructuredStrategy;
  for (auto strategy :
       {Strategy::automatic, Strategy::compact, Strategy::chained,
        Strategy::conditional, Strategy::coarse_conditional}) {
    if ((strategy == Strategy::conditional ||
         strategy == Strategy::coarse_conditional) &&
        !device->supports_conditional_commands()) {
      continue;
    }
    SCOPED_TRACE(static_cast<int>(strategy));
    gfx::AotModuleParams module_params;
    module_params.module_path = folder;
    module_params.runtime = &runtime;
    std::shared_ptr<aot::Module> module =
        aot::Module::load(Arch::vulkan, module_params);
    ASSERT_TRUE(module && !module->is_corrupted());
    auto *condition =
        dynamic_cast<gfx::KernelImpl *>(module->get_kernel("condition"));
    auto *advance =
        dynamic_cast<gfx::KernelImpl *>(module->get_kernel("advance"));
    auto *adjoint =
        dynamic_cast<gfx::KernelImpl *>(module->get_kernel("adjoint"));
    ASSERT_TRUE(condition && advance && adjoint);
    struct Frame {
      std::shared_ptr<DeviceAllocationGuard> storage;
      std::unique_ptr<gfx::GraphReplayRegistration> loop;
      std::unique_ptr<gfx::GraphReplayRegistration> finish;
      gfx::GfxRuntime::GraphStructuredControl control;
    };
    std::array<Frame, 2> frames;
    const auto write = [&](Frame &frame, const std::array<int, 7> &values) {
      void *mapped = nullptr;
      ASSERT_EQ(device->map(*frame.storage, &mapped), RhiResult::success);
      std::memcpy(mapped, values.data(), sizeof(values));
      device->unmap(*frame.storage);
    };
    const auto read = [&](Frame &frame) {
      std::array<int, 7> result{};
      void *mapped = nullptr;
      if (device->map(*frame.storage, &mapped) != RhiResult::success) {
        ADD_FAILURE() << "Cannot map control storage";
        return result;
      }
      std::memcpy(result.data(), mapped, sizeof(result));
      device->unmap(*frame.storage);
      return result;
    };
    for (auto &frame : frames) {
      auto [storage, result] = device->allocate_memory_unique(
          {7 * sizeof(int), true, true, false, AllocUsage::Storage});
      ASSERT_EQ(result, RhiResult::success);
      frame.storage = std::move(storage);
      write(frame, {0, 0, 0, 3, 2, 0, 0});
      Ndarray array(*frame.storage, PrimitiveType::i32, {7});
      LaunchContextBuilder cond_ctx(condition), body_ctx(advance),
          grad_ctx(adjoint);
      for (auto *context : {&cond_ctx, &body_ctx, &grad_ctx}) {
        context->set_arg_ndarray({0}, array);
      }
      const std::vector<gfx::GfxRuntime::GraphDispatch> dispatches{
          {condition->handle(), &cond_ctx},
          {advance->handle(), &body_ctx},
          {condition->handle(), &cond_ctx}};
      auto &control = frame.control;
      control.predicate = frame.storage->get_ptr();
      control.counter = frame.storage->get_ptr(sizeof(int));
      control.status = frame.storage->get_ptr(2 * sizeof(int));
      control.has_status = &frame == &frames.front();
      control.initial_dispatch_count = 1;
      control.max_iterations = 3;
      control.strategy = strategy;
      auto invalid = control;
      invalid.initial_dispatch_count = dispatches.size();
      EXPECT_ANY_THROW(runtime.prepare_fixed_structured_graph(
          dispatches, invalid, {module, frame.storage}));
      const auto before = device->queue_submission_snapshot();
      frame.loop = runtime.prepare_fixed_structured_graph(
          dispatches, control, {module, frame.storage});
      frame.finish = runtime.prepare_fixed_graph(
          {{{adjoint->handle(), &grad_ctx}, {}}}, {module, frame.storage});
      EXPECT_EQ(device->queue_submission_snapshot().queue_submit_calls,
                before.queue_submit_calls);
      runtime.synchronize();
      EXPECT_EQ(read(frame), (std::array<int, 7>{0, 0, 0, 3, 2, 0, 0}));
      // A compact zero-budget plan runs the initial condition only.
      auto zero = control;
      zero.max_iterations = 0;
      zero.strategy = Strategy::compact;
      frame.finish->launch_prepared();
      const auto pending = device->queue_submission_snapshot();
      auto empty = runtime.prepare_fixed_structured_graph(
          dispatches, zero, {module, frame.storage});
      EXPECT_EQ(device->queue_submission_snapshot().queue_submit_calls,
                pending.queue_submit_calls);
      runtime.synchronize();
      EXPECT_EQ(read(frame), (std::array<int, 7>{0, 0, 0, 3, 2, 0, 6}));
      empty->launch_prepared();
      runtime.synchronize();
      EXPECT_EQ(read(frame), (std::array<int, 7>{1, 0, 1, 3, 2, 1, 6}));
      // Later compound chunks consume the preceding device predicate without
      // repeating the initial condition (which has an observable side effect).
      auto continuation = control;
      continuation.execute_initial_dispatches = false;
      auto chunk = runtime.prepare_fixed_structured_graph(
          dispatches, continuation, {module, frame.storage});
      chunk->launch_prepared();
      runtime.synchronize();
      EXPECT_EQ(read(frame), (std::array<int, 7>{0, 2, 2, 12, 2, 3, 6}));
      chunk->launch_prepared();
      runtime.synchronize();
      EXPECT_EQ(read(frame), (std::array<int, 7>{0, 2, 2, 12, 2, 3, 6}));
      auto precomputed = control;
      precomputed.initial_dispatch_count = 0;
      auto single = runtime.prepare_fixed_structured_graph(
          {{condition->handle(), &cond_ctx}}, precomputed,
          {module, frame.storage});
      write(frame, {1, 0, 0, 3, 0, 0, 0});
      single->launch_prepared();
      runtime.synchronize();
      EXPECT_EQ(read(frame), (std::array<int, 7>{0, 0, 2, 3, 0, 1, 0}));
    }  // All launch contexts and borrowed Ndarray wrappers are now destroyed.
    for (int target : {0, 2, 7, 0}) {
      for (std::size_t i = 0; i < frames.size(); ++i) {
        write(frames[i], {0, 0, 0, 3 + static_cast<int>(i), target, 0, 0});
        frames[i].loop->launch_prepared();
        frames[i].finish->launch_prepared();
      }
      runtime.synchronize();
      for (std::size_t i = 0; i < frames.size(); ++i) {
        const int rounds = std::min(target, 3);
        const int initial = 3 + static_cast<int>(i);
        EXPECT_EQ(read(frames[i]),
                  (std::array<int, 7>{target > 3, rounds, target > 3 ? 1 : 2,
                                      initial << rounds, target, rounds + 1,
                                      2 * initial * (1 << (2 * rounds))}));
      }
    }
    // Keep actual pending work at close, so premature owner release cannot
    // accidentally pass merely because this tiny kernel already completed.
    std::weak_ptr<aot::Module> retired_module = module;
    std::weak_ptr<DeviceAllocationGuard> retired_storage = frames[0].storage;
    runtime.begin_submission_batch();
    for (auto &frame : frames) {
      frame.loop->launch_prepared();
      frame.loop.reset();
      frame.finish.reset();
      frame.storage.reset();
    }
    module.reset();
    EXPECT_FALSE(retired_module.expired());
    EXPECT_FALSE(retired_storage.expired());
    runtime.end_submission_batch();
    runtime.synchronize();
    EXPECT_TRUE(retired_module.expired());
    EXPECT_TRUE(retired_storage.expired());
  }
}

TEST(CGraphAotTest, VulkanFixedNestedControl) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP() << "Vulkan is unavailable";
  const auto *folder = std::getenv("TAICHI_AOT_FOLDER_PATH");
  ASSERT_NE(folder, nullptr);
  vulkan::VulkanDeviceCreator::Params params;
  vulkan::VulkanDeviceCreator creator(params);
  auto *device = creator.device();
  if (!device->supports_conditional_commands())
    GTEST_SKIP() << "Conditional commands are unavailable";
  gfx::GfxRuntime::Params runtime_params;
  runtime_params.device = device;
  gfx::GfxRuntime runtime(std::move(runtime_params));

  // Test zero budgets and go past each historical JIT route policy without
  // large data or long logical execution. No arbitrary bounds in this API.
  for (const auto shape : {std::array<int, 3>{3, 4, 2},
                           {0, 4, 2},
                           {3, 0, 2},
                           {65, 1, 1},
                           {1, 65, 1},
                           {1, 1, 9},
                           {65, 65, 1}}) {
    SCOPED_TRACE(::testing::Message()
                 << shape[0] << "/" << shape[1] << "/" << shape[2]);
    gfx::AotModuleParams module_params;
    module_params.module_path = folder;
    module_params.runtime = &runtime;
    std::shared_ptr<aot::Module> module =
        aot::Module::load(Arch::vulkan, module_params);
    ASSERT_TRUE(module && !module->is_corrupted());
    const int elements = (shape[2] + 1) * 8;
    struct Frame {
      std::shared_ptr<DeviceAllocationGuard> storage;
      std::unique_ptr<gfx::GraphReplayRegistration> loop, finish;
    };
    std::array<Frame, 2> frames;
    const auto write = [&](Frame &frame, const std::vector<int> &values) {
      void *mapped = nullptr;
      ASSERT_EQ(device->map(*frame.storage, &mapped), RhiResult::success);
      std::memcpy(mapped, values.data(), values.size() * sizeof(int));
      device->unmap(*frame.storage);
    };
    const auto read = [&](Frame &frame) {
      std::vector<int> values(elements);
      void *mapped = nullptr;
      if (device->map(*frame.storage, &mapped) != RhiResult::success) {
        ADD_FAILURE() << "Cannot map nested control storage";
        return values;
      }
      std::memcpy(values.data(), mapped, values.size() * sizeof(int));
      device->unmap(*frame.storage);
      return values;
    };
    for (auto &frame : frames) {
      auto [storage, result] = device->allocate_memory_unique(
          {elements * sizeof(int), true, true, false, AllocUsage::Storage});
      ASSERT_EQ(result, RhiResult::success);
      frame.storage = std::move(storage);
      std::vector<int> initial(elements, 0);
      initial[3] = 3;
      write(frame, initial);
      Ndarray array(*frame.storage, PrimitiveType::i32, {elements});
      std::vector<std::unique_ptr<LaunchContextBuilder>> contexts;
      std::vector<gfx::GfxRuntime::GraphDispatch> dispatches;
      const auto append = [&](const char *name, int base) {
        auto *kernel =
            dynamic_cast<gfx::KernelImpl *>(module->get_kernel(name));
        TI_ASSERT(kernel);
        auto context = std::make_unique<LaunchContextBuilder>(kernel);
        context->set_arg_ndarray({0}, array);
        context->set_arg_int({1}, base);
        dispatches.push_back({kernel->handle(), context.get()});
        contexts.push_back(std::move(context));
      };
      gfx::GfxRuntime::GraphNestedStructuredControl control;
      control.outer_predicate = frame.storage->get_ptr();
      control.outer_counter = frame.storage->get_ptr(sizeof(int));
      control.outer_status = frame.storage->get_ptr(2 * sizeof(int));
      control.outer_has_status = &frame == &frames.front();
      control.outer_max_iterations = shape[0];
      append("condition", 0);
      control.outer_condition_dispatch_count = dispatches.size();
      for (int child = 0; child < shape[2]; ++child) {
        const int base = (child + 1) * 8;
        gfx::GfxRuntime::GraphNestedStructuredInnerControl inner;
        inner.inner_predicate = frame.storage->get_ptr(base * sizeof(int));
        inner.inner_counter = frame.storage->get_ptr((base + 1) * sizeof(int));
        inner.inner_status = frame.storage->get_ptr((base + 2) * sizeof(int));
        inner.inner_has_status = (child % 2 == 0) && control.outer_has_status;
        inner.inner_max_iterations = shape[1];
        inner.inner_chunk_size = shape[1];
        append("reset", base);
        inner.inner_condition_dispatch_begin = dispatches.size();
        append("condition", base);
        inner.inner_body_dispatch_begin = dispatches.size();
        append("advance", base);
        append("condition", base);
        inner.inner_dispatch_end = dispatches.size();
        control.inner_controls.push_back(inner);
      }
      append("outer_advance", 0);
      append("condition", 0);
      auto invalid = control;
      invalid.inner_controls[0].inner_counter = control.outer_counter;
      EXPECT_ANY_THROW(runtime.prepare_fixed_nested_structured_graph(
          dispatches, invalid, {module, frame.storage}));
      invalid = control;
      invalid.inner_controls[0].inner_dispatch_end = dispatches.size() + 1;
      EXPECT_ANY_THROW(runtime.prepare_fixed_nested_structured_graph(
          dispatches, invalid, {module, frame.storage}));
      invalid = control;
      invalid.outer_max_iterations = std::numeric_limits<std::uint32_t>::max();
      invalid.inner_controls[0].inner_max_iterations =
          std::numeric_limits<std::uint32_t>::max();
      invalid.inner_controls[0].inner_chunk_size =
          invalid.inner_controls[0].inner_max_iterations;
      EXPECT_ANY_THROW(runtime.prepare_fixed_nested_structured_graph(
          dispatches, invalid, {module, frame.storage}));
      const auto before = device->queue_submission_snapshot();
      frame.loop = runtime.prepare_fixed_nested_structured_graph(
          dispatches, control, {module, frame.storage});
      // Two serial dispatches per inner body: only their indirect packets and
      // the stable outer predicate remain, independent of iteration budgets.
      EXPECT_EQ(frame.loop->snapshot_stats().known_persistent_control_bytes,
                (shape[2] * 2 * 3 + 1) * sizeof(std::uint32_t));
      append("adjoint", 0);
      frame.finish = runtime.prepare_fixed_graph({{dispatches.back(), {}}},
                                                 {module, frame.storage});
      EXPECT_EQ(device->queue_submission_snapshot().queue_submit_calls,
                before.queue_submit_calls);
      runtime.synchronize();
      EXPECT_EQ(read(frame), initial);
      if (shape == (std::array<int, 3>{3, 4, 2}) && &frame == &frames.front()) {
        const std::vector<gfx::GfxRuntime::GraphDispatch> nested_dispatches(
            dispatches.begin(), dispatches.end() - 1);
        auto dynamic = runtime.register_graph_replay(1);
        for (bool observe : {false, true, false, true}) {
          auto input = initial;
          input[4] = 1;
          input[12] = 1;
          input[20] = 1;
          write(frame, input);
          gfx::GfxRuntime::GraphNestedStructuredResult result;
          ASSERT_TRUE(runtime.try_launch_graph(
              nested_dispatches, dynamic->replay_key(), nullptr, nullptr,
              nullptr, &control, observe ? &result : nullptr));
          runtime.synchronize();
          const auto output = read(frame);
          EXPECT_EQ(output[1], 1);
          EXPECT_EQ(output[3], 12);
          EXPECT_EQ(output[7], 2);
          if (observe) {
            EXPECT_TRUE(result.submitted);
            EXPECT_EQ(result.outer_logical_iterations, 1);
            EXPECT_EQ(result.outer_initial_counter, 0);
            EXPECT_EQ(result.outer_final_counter, 1);
            EXPECT_EQ(result.inner_logical_iterations,
                      (std::vector<std::uint32_t>{1, 1}));
          }
        }
        write(frame, initial);
        runtime.begin_submission_batch();
        frame.finish->launch_prepared();
        const auto pending = device->queue_submission_snapshot();
        auto additional = runtime.prepare_fixed_nested_structured_graph(
            nested_dispatches, control, {module, frame.storage});
        EXPECT_EQ(device->queue_submission_snapshot().queue_submit_calls,
                  pending.queue_submit_calls);
        runtime.end_submission_batch();
        runtime.synchronize();
        initial[6] = 6;
        EXPECT_EQ(read(frame), initial);
      }
    }  // Borrowed arrays and launch contexts are gone before the first replay.

    std::vector<std::array<int, 2>> cases{
        {0, 1}, {1, 1}, {1, 0}, {2, 1}, {0, 1}};
    if (shape == (std::array<int, 3>{3, 4, 2})) {
      cases.push_back({5, 1});
      cases.push_back({1, 5});
    }
    for (const auto targets : cases) {
      for (std::size_t index = 0; index < frames.size(); ++index) {
        auto &frame = frames[index];
        std::vector<int> initial(elements, 0);
        initial[3] = 3 + static_cast<int>(index);
        initial[4] = targets[0];
        for (int child = 0; child < shape[2]; ++child)
          initial[(child + 1) * 8 + 4] = targets[1];
        write(frame, initial);
        const auto before = device->queue_submission_snapshot();
        runtime.begin_submission_batch();
        frame.loop->launch_prepared();
        frame.finish->launch_prepared();
        runtime.end_submission_batch();
        EXPECT_EQ(device->queue_submission_snapshot().queue_submit_calls,
                  before.queue_submit_calls + 1);
      }
      runtime.synchronize();
      for (std::size_t index = 0; index < frames.size(); ++index) {
        const auto values = read(frames[index]);
        const int outer = std::min(shape[0], targets[0]);
        const int inner = std::min(shape[1], targets[1]);
        const int doublings = outer * inner * shape[2];
        const int initial = 3 + static_cast<int>(index);
        EXPECT_EQ(values[0], outer < targets[0]);
        EXPECT_EQ(values[1], outer);
        EXPECT_EQ(values[2], outer < targets[0] ? 1 : 2);
        EXPECT_EQ(values[3], initial << doublings);
        EXPECT_EQ(values[5], outer + 1);
        EXPECT_EQ(values[6], 2 * initial * (1 << (2 * doublings)));
        EXPECT_EQ(values[7], doublings);
        for (int child = 0; child < shape[2]; ++child) {
          const int base = (child + 1) * 8;
          EXPECT_EQ(values[base], outer && inner < targets[1]);
          EXPECT_EQ(values[base + 1], outer ? inner : 0);
          EXPECT_EQ(values[base + 2], outer ? (inner < targets[1] ? 1 : 2) : 0);
          EXPECT_EQ(values[base + 5], outer * (inner + 1));
        }
      }
    }
    std::weak_ptr<aot::Module> retired_module = module;
    std::weak_ptr<DeviceAllocationGuard> retired_storage = frames[0].storage;
    runtime.begin_submission_batch();
    for (auto &frame : frames) {
      frame.loop->launch_prepared();
      frame.loop.reset();
      frame.finish.reset();
      frame.storage.reset();
    }
    module.reset();
    EXPECT_FALSE(retired_module.expired());
    EXPECT_FALSE(retired_storage.expired());
    runtime.end_submission_batch();
    runtime.synchronize();
    EXPECT_TRUE(retired_module.expired());
    EXPECT_TRUE(retired_storage.expired());
  }
}
