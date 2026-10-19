#include <gtest/gtest.h>

#if defined(TI_WITH_CUDA) && defined(TI_WITH_VULKAN)
#include "taichi/rhi/cuda/cuda_retained_memory.h"
#include "tests/cpp/rhi/common/cuda_external_gate.h"

namespace taichi::lang {

TEST(CudaRetainedMemory, ExternalGateRetainsPendingStateAcrossSchedulerSlices) {
  auto context = CUDAContext::get_instance().get_guard();
  auto &driver = CUDADriver::get_instance();
  testing::CudaExternalGate gate;
  void *stream = nullptr, *event = nullptr;
  driver.stream_create(&stream, CU_STREAM_NON_BLOCKING);
  driver.event_create(&event, CU_EVENT_DISABLE_TIMING);
  gate.enqueue_wait(stream);
  driver.event_record(event, stream);
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 30; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const auto result = driver.event_query.call(event);
    if (result != CUDA_ERROR_NOT_READY) {
      ADD_FAILURE() << "Gate completed after "
                    << std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - start)
                           .count()
                    << "s, CUDA result=" << result
                    << ", Vulkan producer pending=" << gate.producer_pending()
                    << ", watchdog=" << gate.timed_out();
      break;
    }
  }
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  driver.event_synchronize(event);
  driver.event_destroy(event);
  driver.stream_destroy(stream);
}

TEST(CudaRetainedMemory, PendingPrefixIsNotReusedOrWaitedOn) {
  auto context = CUDAContext::get_instance().get_guard();
  auto &driver = CUDADriver::get_instance();
  cuda::RetainedMemoryPool pool;
  void *stream = nullptr;
  driver.stream_create(&stream, CU_STREAM_NON_BLOCKING);
  {
    testing::CudaExternalGate gate;
    void *first = pool.allocate(4096);
    gate.enqueue_wait(stream);
    driver.memsetd32_async(first, 13, 1, stream);
    pool.release(first, stream);
    // An actual pending GPU prefix is the oracle. A sleep-only test cannot
    // establish whether storage is safely retained or a stream was waited on.
    void *second = pool.allocate(4096);
    EXPECT_NE(first, second);
    EXPECT_FALSE(gate.timed_out());
    EXPECT_EQ(pool.snapshot().at("native_allocation_calls"), 2u);
    gate.release();
    driver.stream_synchronize(stream);
    std::uint32_t value = 0;
    driver.memcpy_device_to_host(&value, first, sizeof(value));
    EXPECT_EQ(value, 13u);
    void *reused = pool.allocate(4096);
    EXPECT_EQ(first, reused);
    pool.release(reused, stream);
    pool.release(second, stream);
  }
  driver.stream_synchronize(stream);
  EXPECT_EQ(pool.snapshot().at("used_current_bytes"), 0u);
  EXPECT_EQ(pool.snapshot().at("native_free_calls"), 0u);
  pool.trim();
  EXPECT_EQ(pool.snapshot().at("reserved_current_bytes"), 0u);
  EXPECT_EQ(pool.snapshot().at("native_free_calls"), 2u);
  driver.stream_destroy(stream);
}

TEST(CudaRetainedMemory, TenThousandCompletedReusesKeepStorageBounded) {
  auto context = CUDAContext::get_instance().get_guard();
  auto &driver = CUDADriver::get_instance();
  cuda::RetainedMemoryPool pool;
  void *event = nullptr;
  driver.event_create(&event, CU_EVENT_DISABLE_TIMING);
  for (std::uint32_t index = 0; index < 10000; ++index) {
    void *pointer = pool.allocate(4096);
    driver.memsetd32_async(pointer, index, 1, nullptr);
    driver.event_record(event, nullptr);
    // Qualification test only: make completion known before each reuse. This
    // is not an application scheduling or throughput benchmark.
    driver.event_synchronize(event);
    std::uint32_t value = 0;
    driver.memcpy_device_to_host(&value, pointer, sizeof(value));
    ASSERT_EQ(value, index);
    pool.release(pointer, nullptr);
    driver.event_record(event, nullptr);
    driver.event_synchronize(event);
  }
  const auto stats = pool.snapshot();
  EXPECT_EQ(stats.at("native_allocation_calls"), 1u);
  EXPECT_EQ(stats.at("reuse_calls"), 9999u);
  EXPECT_EQ(stats.at("native_free_calls"), 0u);
  EXPECT_EQ(stats.at("reserved_current_bytes"), 4096u);
  EXPECT_EQ(stats.at("used_current_bytes"), 0u);
  driver.event_destroy(event);
}

}  // namespace taichi::lang
#endif
