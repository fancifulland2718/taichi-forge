#include <gtest/gtest.h>

#include <taichi/rhi/cuda/cuda_capability.h>
#include <taichi/rhi/cuda/cuda_context.h>
#include <taichi/rhi/cuda/cuda_device.h>

#include <atomic>
#include <array>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

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

TEST(CUDACapability, ResolvesNativeAndCompilerFallbackTargets) {
  using cuda::detail::resolve_compute_capability_target;

  const auto ampere = resolve_compute_capability_target(86);
  EXPECT_EQ(ampere.codegen_compute_capability, 86);
  EXPECT_EQ(ampere.ptx_version, 71);
  EXPECT_FALSE(ampere.uses_fallback);

  const auto blackwell = resolve_compute_capability_target(120);
  EXPECT_EQ(blackwell.codegen_compute_capability, 120);
  EXPECT_EQ(blackwell.ptx_version, 87);
  EXPECT_FALSE(blackwell.uses_fallback);

  const auto future = resolve_compute_capability_target(121);
  EXPECT_EQ(future.codegen_compute_capability, 120);
  EXPECT_EQ(future.ptx_version, 87);
  EXPECT_TRUE(future.uses_fallback);
}

TEST(CUDAContext, SeparatesDeviceCapabilityFromCodegenTarget) {
  if (!CUDADriver::get_instance_without_context().detected()) {
    GTEST_SKIP();
  }

  auto &context = CUDAContext::get_instance();
  const auto target = cuda::detail::resolve_compute_capability_target(
      context.get_compute_capability());
  EXPECT_EQ(context.get_codegen_compute_capability(),
            target.codegen_compute_capability);
  EXPECT_EQ(context.get_mcpu(),
            "sm_" + std::to_string(target.codegen_compute_capability));
  EXPECT_EQ(context.get_mattrs(),
            "+ptx" + std::to_string(target.ptx_version));
}

TEST(CUDADiagnostics, SamplesLockContentionWithoutChangingLockOwnership) {
  CUDASampledLockTelemetry telemetry;
  std::mutex mutex;
  for (uint32_t i = 0; i < CUDASampledLockTelemetry::kSamplingPeriod; ++i) {
    auto lock = telemetry.acquire(mutex);
  }
  EXPECT_EQ(telemetry.snapshot().sampled_acquisitions, 1u);
  EXPECT_EQ(telemetry.snapshot().contended_acquisitions, 0u);

  std::atomic<bool> ready{false};
  std::atomic<bool> start_contention{false};
  std::thread contender([&] {
    for (uint32_t i = 1; i < CUDASampledLockTelemetry::kSamplingPeriod; ++i) {
      auto lock = telemetry.acquire(mutex);
    }
    ready.store(true, std::memory_order_release);
    while (!start_contention.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    auto lock = telemetry.acquire(mutex);
  });
  while (!ready.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  mutex.lock();
  start_contention.store(true, std::memory_order_release);
  while (telemetry.snapshot().sampled_acquisitions < 2) {
    std::this_thread::yield();
  }
  mutex.unlock();
  contender.join();
  EXPECT_EQ(telemetry.snapshot().contended_acquisitions, 1u);
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

TEST(CUDADevice, RejectsStaleWrongDeviceAndOutOfRangeAllocations) {
  if (!CUDADriver::get_instance_without_context().detected()) {
    GTEST_SKIP();
  }

  cuda::CudaDevice device;
  cuda::CudaDevice other_device;
  Device::AllocParams params;
  params.size = 32;
  DeviceAllocation allocation;
  ASSERT_EQ(device.allocate_memory(params, &allocation), RhiResult::success);

  std::array<uint8_t, 8> input{};
  input.fill(42);
  const void *input_ptr = input.data();
  size_t input_size = input.size();
  DevicePtr ptr = allocation.get_ptr(8);
  ASSERT_EQ(device.upload_data(&ptr, &input_ptr, &input_size),
            RhiResult::success);
  DevicePtr out_of_range = allocation.get_ptr(31);
  EXPECT_EQ(device.upload_data(&out_of_range, &input_ptr, &input_size),
            RhiResult::invalid_usage);

  void *mapped = nullptr;
  DeviceAllocation wrong_device = allocation;
  wrong_device.device = &other_device;
  EXPECT_EQ(other_device.map(wrong_device, &mapped), RhiResult::invalid_usage);

  device.dealloc_memory(allocation);
  EXPECT_EQ(device.map(allocation, &mapped), RhiResult::invalid_usage);
  EXPECT_EQ(device.upload_data(&ptr, &input_ptr, &input_size),
            RhiResult::invalid_usage);

  DeviceAllocation replacement;
  ASSERT_EQ(device.allocate_memory(params, &replacement), RhiResult::success);
  EXPECT_NE(replacement.alloc_id, allocation.alloc_id);
  device.dealloc_memory(replacement);
}

TEST(CUDADevice, ConcurrentAllocateCopyReadbackAndRelease) {
  if (!CUDADriver::get_instance_without_context().detected()) {
    GTEST_SKIP();
  }

  cuda::CudaDevice device;
  constexpr int kThreads = 4;
  constexpr int kIterations = 32;
  constexpr size_t kBytes = 128;
  std::atomic<bool> succeeded{true};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (int thread_index = 0; thread_index < kThreads; ++thread_index) {
    threads.emplace_back([&, thread_index] {
      Device::AllocParams params;
      params.size = kBytes;
      for (int iteration = 0; iteration < kIterations; ++iteration) {
        DeviceAllocation src;
        DeviceAllocation dst;
        if (device.allocate_memory(params, &src) != RhiResult::success ||
            device.allocate_memory(params, &dst) != RhiResult::success) {
          succeeded.store(false, std::memory_order_relaxed);
          return;
        }

        std::array<uint8_t, 128> input{};
        input.fill(static_cast<uint8_t>(thread_index + iteration));
        const void *input_ptr = input.data();
        size_t input_size = input.size();
        DevicePtr src_ptr = src.get_ptr();
        DevicePtr dst_ptr = dst.get_ptr();
        if (device.upload_data(&src_ptr, &input_ptr, &input_size) !=
            RhiResult::success) {
          succeeded.store(false, std::memory_order_relaxed);
          return;
        }

        device.memcpy_internal(dst_ptr, src_ptr, kBytes);
        std::array<uint8_t, 128> output{};
        void *output_ptr = output.data();
        size_t output_size = output.size();
        if (device.readback_data(&dst_ptr, &output_ptr, &output_size) !=
                RhiResult::success ||
            output != input) {
          succeeded.store(false, std::memory_order_relaxed);
          return;
        }
        device.dealloc_memory(src);
        device.dealloc_memory(dst);
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }
  EXPECT_TRUE(succeeded.load(std::memory_order_relaxed));
}

TEST(CUDADevice, RuntimeAsyncAllocationUsesItsAllocationStream) {
  if (!CUDADriver::get_instance_without_context().detected()) {
    GTEST_SKIP();
  }

  auto &driver = CUDADriver::get_instance();
  const auto before = driver.get_telemetry_snapshot();
  cuda::CudaDevice device;
  LlvmDevice::LlvmRuntimeAllocParams params;
  params.size = 256;
  params.use_memory_pool = true;
  DeviceAllocation allocation = device.allocate_memory_runtime(params);
  ASSERT_EQ(allocation.device, &device);
  device.dealloc_memory(allocation);
  CUDADriver::get_instance().stream_synchronize(nullptr);
  const auto after = driver.get_telemetry_snapshot();
  if (CUDAContext::get_instance().supports_mem_pool()) {
    EXPECT_GE(after.async_allocation_calls, before.async_allocation_calls + 1);
    EXPECT_GE(after.async_free_calls, before.async_free_calls + 1);
  } else {
    EXPECT_GE(after.sync_allocation_fallback_calls,
              before.sync_allocation_fallback_calls + 1);
    EXPECT_GE(after.sync_free_fallback_calls,
              before.sync_free_fallback_calls + 1);
  }
}

TEST(CUDADevice, RuntimeAddressQueryTreatsRetiredAllocationAsNull) {
  if (!CUDADriver::get_instance_without_context().detected()) {
    GTEST_SKIP();
  }

  cuda::CudaDevice device;
  LlvmDevice::LlvmRuntimeAllocParams params;
  params.size = 256;
  params.use_memory_pool = true;
  DeviceAllocation allocation = device.allocate_memory_runtime(params);
  ASSERT_NE(device.get_memory_addr(allocation), nullptr);
  device.dealloc_memory(allocation);
  EXPECT_EQ(device.get_memory_addr(allocation), nullptr);
  CUDADriver::get_instance().stream_synchronize(nullptr);
}

}  // namespace taichi::lang
