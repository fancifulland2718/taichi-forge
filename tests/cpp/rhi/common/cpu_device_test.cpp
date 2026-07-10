#include <gtest/gtest.h>

#include "taichi/rhi/cpu/cpu_device.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

namespace taichi::lang {
namespace cpu {
namespace {

TEST(CpuDeviceTest, RejectsStaleWrongDeviceAndOutOfRangeAllocations) {
  CpuDevice device;
  CpuDevice other_device;
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

  void *mapped = nullptr;
  ASSERT_EQ(device.map_range(ptr, input.size(), &mapped), RhiResult::success);
  EXPECT_EQ(std::memcmp(mapped, input.data(), input.size()), 0);
  EXPECT_EQ(device.map_range(allocation.get_ptr(31), 2, &mapped),
            RhiResult::invalid_usage);

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

TEST(CpuDeviceTest, ConcurrentAllocateCopyReadbackAndRelease) {
  CpuDevice device;
  constexpr int kThreads = 4;
  constexpr int kIterations = 128;
  constexpr size_t kBytes = 256;
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

        std::array<uint8_t, 256> input{};
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
        std::array<uint8_t, 256> output{};
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

}  // namespace
}  // namespace cpu
}  // namespace taichi::lang
