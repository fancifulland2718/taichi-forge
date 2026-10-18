#include <gtest/gtest.h>

#include <cstring>
#include <functional>
#include "taichi/runtime/fixed_execution.h"
#ifdef TI_WITH_LLVM
#include "taichi/rhi/cpu/cpu_device.h"
#endif
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#endif
#ifdef TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/runtime/gfx/runtime.h"
#endif

namespace taichi::lang {
namespace {
using Write = std::function<void(DevicePtr, std::uint32_t)>;
using Read = std::function<std::uint32_t(DevicePtr)>;

void check_ring(Arch arch,
               Device &device,
               gfx::GfxRuntime *runtime,
               const Write &write,
               const Read &read) {
  for (const std::size_t capacity : {3, 65}) {
    auto memory = device.allocate_memory_unique({(capacity + 1) * 4});
    ASSERT_EQ(memory.second, RhiResult::success);
    const auto output = memory.first->get_ptr();
    const auto slot = [&](std::size_t index) {
      return memory.first->get_ptr((index + 1) * 4);
    };
    write(output, 999);
    std::vector<FixedExecution::Operation> operations;
    for (std::size_t i = 0; i < capacity; ++i) {
      write(slot(i), 100 + i);
      operations.emplace_back(FixedExecution::BufferCopy{output, slot(i), 4});
      operations.emplace_back(FixedExecution::BufferFill{
          slot(i), 4, static_cast<std::uint32_t>(300 + i)});
    }
    const OrderedStorageRing recipe{capacity};
    const auto prepare = [&](const auto &steps) {
      return std::make_unique<FixedExecution>(
          arch, device, runtime, steps, nullptr, nullptr, nullptr, nullptr,
          nullptr, &recipe);
    };
    // The first frame records successfully before a later malformed fill fails.
    // Capture/prepare must not execute a frame or change its initial bytes.
    auto invalid = operations;
    std::get<FixedExecution::BufferFill>(invalid[3]).bytes = 3;
    EXPECT_THROW(prepare(invalid), std::invalid_argument);
    EXPECT_EQ(read(output), 999);
    EXPECT_EQ(read(slot(0)), 100);
    auto execution = prepare(operations);
    EXPECT_EQ(read(output), 999);
    EXPECT_EQ(read(slot(capacity - 1)), 100 + capacity - 1);
    operations.clear();  // No source descriptors are retained by warm replay.
    for (std::size_t i = 0; i < 3; ++i) {
      execution->run();
      EXPECT_EQ(read(output), 100 + i);
    }
    // Queue through two wraps, without a host completion check between calls.
    for (std::size_t i = 3; i < capacity * 2 + 1; ++i)
      execution->run();
    EXPECT_EQ(read(output), 300);
    EXPECT_EQ(read(slot(0)), 300);
    EXPECT_EQ(read(slot(capacity - 1)), 300 + capacity - 1);
    // Recordings may retire while their caller retains the submitted storage.
    execution->run();
    execution.reset();
    EXPECT_EQ(read(output), 301);
  }
}

TEST(OrderedStorageRing, RequiresPositiveCapacityAndCompleteFrames) {
  EXPECT_THROW(OrderedStorageRing{0}.frame_size(2), std::invalid_argument);
  EXPECT_THROW(OrderedStorageRing{3}.frame_size(0), std::invalid_argument);
  EXPECT_THROW(OrderedStorageRing{3}.frame_size(5), std::invalid_argument);
  EXPECT_EQ(OrderedStorageRing{65}.frame_size(130), 2);
}

#ifdef TI_WITH_LLVM
TEST(OrderedStorageRing, CpuOwnsDistinctFramesAndRecoversFromColdFailure) {
  cpu::CpuDevice device;
  auto address = [&](DevicePtr ptr) {
    void *value = nullptr;
    EXPECT_EQ(device.map_range_for_cpu_native(ptr, 4, &value), RhiResult::success);
    return value;
  };
  check_ring(host_arch(), device, nullptr,
             [&](DevicePtr ptr, std::uint32_t value) {
               std::memcpy(address(ptr), &value, 4);
             },
             [&](DevicePtr ptr) {
               std::uint32_t value;
               std::memcpy(&value, address(ptr), 4);
               return value;
             });
}
#endif

#ifdef TI_WITH_CUDA
TEST(OrderedStorageRing, CudaOwnsDistinctFramesAndRecoversFromColdFailure) {
  if (!CUDADriver::get_instance_without_context().detected())
    GTEST_SKIP();
  cuda::CudaDevice device;
  auto guard = CUDAContext::get_instance().get_guard();
  auto lock = CUDAContext::get_instance().get_submission_lock_guard();
  auto &driver = CUDADriver::get_instance();
  auto address = [&](DevicePtr ptr) {
    return static_cast<char *>(device.get_memory_addr(ptr)) + ptr.offset;
  };
  check_ring(Arch::cuda, device, nullptr,
             [&](DevicePtr ptr, std::uint32_t value) {
               driver.memcpy_host_to_device(address(ptr), &value, 4);
             },
             [&](DevicePtr ptr) {
               std::uint32_t value;
               driver.memcpy_device_to_host(&value, address(ptr), 4);
               return value;
             });
}
#endif

#ifdef TI_WITH_VULKAN
TEST(OrderedStorageRing, VulkanOwnsDistinctFramesAndRecoversFromColdFailure) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator::Params params;
  vulkan::VulkanDeviceCreator creator(params);
  auto &device = *creator.device();
  gfx::GfxRuntime::Params runtime_params;
  runtime_params.device = &device;
  gfx::GfxRuntime runtime(std::move(runtime_params));
  check_ring(Arch::vulkan, device, &runtime,
             [&](DevicePtr ptr, std::uint32_t value) {
               (void)runtime.flush_if_pending();
               const void *source = &value;
               std::size_t bytes = 4;
               EXPECT_EQ(device.upload_data(&ptr, &source, &bytes),
                         RhiResult::success);
             },
             [&](DevicePtr ptr) {
               (void)runtime.flush_if_pending();
               std::uint32_t value;
               void *target = &value;
               std::size_t bytes = 4;
               EXPECT_EQ(device.readback_data(&ptr, &target, &bytes),
                         RhiResult::success);
               return value;
             });
}
#endif
}  // namespace
}  // namespace taichi::lang
