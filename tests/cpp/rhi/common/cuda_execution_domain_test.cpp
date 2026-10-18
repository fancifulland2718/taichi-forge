#include <gtest/gtest.h>

#if defined(TI_WITH_CUDA) && defined(TI_WITH_VULKAN)
#include <array>
#include <future>
#include "taichi/program/runtime_completion.h"
#include "taichi/program/runtime_fault.h"
#include "tests/cpp/rhi/common/cuda_external_gate.h"

namespace taichi::lang {
namespace {

class CudaExecutionDependency : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CUDADriver::get_instance_without_context()
             .nvidia_extensions_available())
      GTEST_SKIP() << "Requires NVIDIA CUDA and Vulkan external semaphores";
    auto context = CUDAContext::get_instance().get_guard();
    auto &driver = CUDADriver::get_instance();
    for (auto &stream : streams)
      driver.stream_create(&stream, CU_STREAM_NON_BLOCKING);
    driver.malloc(&data, 64);
    driver.memsetd32(data, 0, 16);
    driver.stream_synchronize(nullptr);  // Cold initialization only.
  }
  void TearDown() override {
    if (!data)
      return;
    auto context = CUDAContext::get_instance().get_guard();
    auto &driver = CUDADriver::get_instance();
    for (auto stream : streams) {
      driver.stream_synchronize.call_concurrent(stream);
      driver.stream_destroy.call(stream);
    }
    pool->clear();
    driver.mem_free.call(data);
  }
  void *at(int index) {
    return static_cast<std::uint32_t *>(data) + index;
  }
  RuntimeCompletion record(int index) {
    return RuntimeCompletion::from_cuda_stream(71, ++sequence, streams[index],
                                               fault, pool);
  }
  static bool poll_until(const RuntimeCompletion &completion) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!completion.done()) {
      if (std::chrono::steady_clock::now() >= deadline)
        return false;
      std::this_thread::yield();
    }
    return true;
  }
  std::shared_ptr<RuntimeFaultDomain> fault =
      std::make_shared<RuntimeFaultDomain>(Arch::cuda, 71);
  std::shared_ptr<RuntimeCompletionCudaEventPool> pool =
      std::make_shared<RuntimeCompletionCudaEventPool>(fault, 8);
  std::array<void *, 3> streams{};
  void *data{};
  std::uint64_t sequence{};
};

TEST_F(CudaExecutionDependency,
       PendingProducerWaitsOnDeviceAndLeavesOtherStreamFree) {
  testing::CudaExternalGate gate;
  auto context = CUDAContext::get_instance().get_guard();
  auto &driver = CUDADriver::get_instance();
  gate.enqueue_wait(streams[0]);
  driver.memsetd32_async(at(0), 37, 1, streams[0]);
  auto producer = record(0);
  ASSERT_FALSE(producer.done());
  auto waiter = std::async(std::launch::async, [&] { producer.wait(); });
  ASSERT_TRUE(producer.enqueue_cuda_wait(streams[1], 71, fault));
  driver.memcpy_device_to_device_async(at(1), at(0), 4, streams[1]);
  auto consumer = record(1);
  // Same-domain negative control must remain behind the actual pending gate.
  driver.memsetd32_async(at(2), 41, 1, streams[0]);
  auto same_domain = record(0);
  driver.memsetd32_async(at(3), 53, 1, streams[2]);
  auto independent = record(2);
  EXPECT_TRUE(poll_until(independent));
  EXPECT_FALSE(producer.done());
  EXPECT_FALSE(consumer.done());
  EXPECT_FALSE(same_domain.done());
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  waiter.get();
  consumer.wait();
  same_domain.wait();
  std::array<std::uint32_t, 4> values{};
  driver.memcpy_device_to_host(values.data(), data, sizeof(values));
  EXPECT_EQ(values, (std::array<std::uint32_t, 4>{37, 37, 41, 53}));
}

TEST_F(CudaExecutionDependency, EventReuseCannotMoveAnAlreadyQueuedDependency) {
  testing::CudaExternalGate consumer_gate;
  testing::CudaExternalGate later_producer_gate;
  auto context = CUDAContext::get_instance().get_guard();
  auto &driver = CUDADriver::get_instance();
  driver.memsetd32_async(at(0), 67, 1, streams[0]);
  auto first = record(0);
  consumer_gate.enqueue_wait(streams[1]);
  ASSERT_TRUE(first.enqueue_cuda_wait(streams[1], 71, fault));
  driver.memcpy_device_to_device_async(at(1), at(0), 4, streams[1]);
  auto consumer = record(1);
  first.wait();  // Return the exact captured event to its pool.
  const auto before = pool->snapshot();
  later_producer_gate.enqueue_wait(streams[0]);
  auto later = record(0);  // Re-record the returned event behind another gate.
  EXPECT_GT(pool->snapshot().reused, before.reused);
  consumer_gate.release();
  EXPECT_TRUE(poll_until(consumer));
  EXPECT_FALSE(later.done());
  EXPECT_FALSE(consumer_gate.timed_out());
  EXPECT_FALSE(later_producer_gate.timed_out());
  later_producer_gate.release();
  later.wait();
  std::uint32_t value = 0;
  driver.memcpy_device_to_host(&value, at(1), sizeof(value));
  EXPECT_EQ(value, 67u);
}

}  // namespace
}  // namespace taichi::lang
#endif
