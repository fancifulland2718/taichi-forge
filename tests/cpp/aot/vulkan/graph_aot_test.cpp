#include "gtest/gtest.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "tests/cpp/aot/gfx_utils.h"
#include "taichi/runtime/gfx/aot_module_loader_impl.h"

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
