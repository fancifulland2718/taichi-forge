#include <gtest/gtest.h>

#include <taichi/rhi/cuda/cuda_capability.h>
#include <taichi/rhi/cuda/cuda_context.h>
#include <taichi/rhi/cuda/cuda_device.h>

#include <atomic>
#include <cstdint>
#include <thread>

namespace taichi::lang {

TEST(CUDAContext, MemoryPoolSupportRequiresDriverAndDeviceCapability) {
  using cuda::detail::MemoryAllocationRoute;
  using cuda::detail::memory_allocation_route;
  using cuda::detail::supports_memory_pool;

  EXPECT_EQ(memory_allocation_route(supports_memory_pool(11, 1, 1)),
            MemoryAllocationRoute::kSynchronous);
  EXPECT_EQ(memory_allocation_route(supports_memory_pool(11, 2, 0)),
            MemoryAllocationRoute::kSynchronous);
  EXPECT_EQ(memory_allocation_route(supports_memory_pool(11, 2, 1)),
            MemoryAllocationRoute::kAsyncMemoryPool);
  EXPECT_EQ(memory_allocation_route(supports_memory_pool(12, 0, 0)),
            MemoryAllocationRoute::kSynchronous);
  EXPECT_EQ(memory_allocation_route(supports_memory_pool(12, 0, 1)),
            MemoryAllocationRoute::kAsyncMemoryPool);
}

TEST(CUDADevice, MapLifecycleRejectsInvalidTransitions) {
  if (!CUDADriver::get_instance_without_context().detected()) {
    GTEST_SKIP();
  }

  cuda::CudaDevice device;
  Device::AllocParams params;
  params.size = sizeof(uint32_t);
  DeviceAllocation allocation;
  ASSERT_EQ(device.allocate_memory(params, &allocation), RhiResult::success);
  DeviceAllocationGuard allocation_guard(allocation);

  void *mapped = nullptr;
  ASSERT_EQ(device.map(allocation, &mapped), RhiResult::success);
  ASSERT_NE(mapped, nullptr);
  *static_cast<uint32_t *>(mapped) = 42;
  EXPECT_EQ(device.map(allocation, &mapped), RhiResult::invalid_usage);
  device.unmap(allocation);

  ASSERT_EQ(device.map(allocation, &mapped), RhiResult::success);
  EXPECT_EQ(*static_cast<uint32_t *>(mapped), 42u);
  device.unmap(allocation);

  DeviceAllocation invalid = allocation;
  ++invalid.alloc_id;
  EXPECT_EQ(device.map(invalid, &mapped), RhiResult::invalid_usage);

  ASSERT_EQ(device.map(allocation, &mapped), RhiResult::success);
  device.unmap(allocation);
}

TEST(CUDAContext, GraphCaptureStreamIsThreadLocal) {
  if (!CUDADriver::get_instance_without_context().detected()) {
    GTEST_SKIP();
  }

  auto &context = CUDAContext::get_instance();
  void *original_stream = context.get_stream();
  void *main_stream = reinterpret_cast<void *>(uintptr_t{1});
  void *worker_stream = reinterpret_cast<void *>(uintptr_t{2});
  std::atomic<void *> worker_initial{nullptr};
  std::atomic<void *> worker_observed{nullptr};

  context.set_stream(main_stream);
  std::thread worker([&] {
    worker_initial.store(context.get_stream(), std::memory_order_relaxed);
    context.set_stream(worker_stream);
    worker_observed.store(context.get_stream(), std::memory_order_relaxed);
  });
  worker.join();

  EXPECT_EQ(context.get_stream(), main_stream);
  EXPECT_EQ(worker_initial.load(std::memory_order_relaxed), nullptr);
  EXPECT_EQ(worker_observed.load(std::memory_order_relaxed), worker_stream);
  context.set_stream(original_stream);
}

}  // namespace taichi::lang
