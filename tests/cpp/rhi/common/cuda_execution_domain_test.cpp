#include <gtest/gtest.h>

#if defined(TI_WITH_CUDA) && defined(TI_WITH_VULKAN)
#include <array>
#include <algorithm>
#include <cstdio>
#include <future>
#include "taichi/program/runtime_completion.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/runtime/cuda/execution_domain.h"
#include "taichi/system/profiler_annotation.h"
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

class DomainResources final : public RuntimeCompletionResources {
 public:
  DomainResources(cuda::ExecutionDomain &domain, std::atomic<int> &released)
      : domain_(domain), released_(released) {
  }
  ~DomainResources() override {
    // Reentry catches resource callbacks made under the admission mutex.
    (void)domain_.snapshot();
    ++released_;
  }
  std::size_t retained_resource_count(std::uint32_t) const noexcept override {
    return 1;
  }

 private:
  cuda::ExecutionDomain &domain_;
  std::atomic<int> &released_;
};

TEST_F(CudaExecutionDependency, DomainCapacityAndDependenciesAreIndependent) {
  ASSERT_TRUE(cuda::ExecutionDomain::available());
  cuda::ExecutionDomain output(fault, {1, 0});
  cuda::ExecutionDomain physics(fault, {2, -1});
  cuda::ExecutionDomain input(fault, {1, 0});
  testing::CudaExternalGate gate;
  const auto output_info = output.snapshot();
  const auto physics_info = physics.snapshot();
  EXPECT_NE(output_info.identity, physics_info.identity);
  EXPECT_NE(output_info.stream, physics_info.stream);
  EXPECT_EQ(output_info.context, physics_info.context);
  EXPECT_EQ(output_info.owner, physics_info.owner);
  EXPECT_LE(physics_info.effective_priority, physics_info.least_priority);
  EXPECT_GE(physics_info.effective_priority, physics_info.greatest_priority);
  std::atomic<int> released{0};
  auto blocked = output.submit(
      1, {}, std::make_shared<DomainResources>(output, released),
      [&](void *stream) {
        gate.enqueue_wait(stream);
        CUDADriver::get_instance().memsetd32_async(at(0), 73, 1, stream);
      });
  bool attempted = false;
  EXPECT_THROW(output.submit(2, {}, nullptr, [&](void *) { attempted = true; }),
               cuda::ExecutionDomainBusy);
  EXPECT_FALSE(attempted);
  auto dependency = input.submit(3, {blocked}, nullptr, [&](void *stream) {
    CUDADriver::get_instance().memcpy_device_to_device_async(at(1), at(0), 4,
                                                             stream);
  });
  auto sentinel = physics.submit(4, {}, nullptr, [&](void *stream) {
    CUDADriver::get_instance().memsetd32_async(at(2), 79, 1, stream);
  });
  EXPECT_TRUE(poll_until(sentinel));
  EXPECT_FALSE(blocked.done());
  EXPECT_FALSE(dependency.done());
  EXPECT_EQ(released.load(), 0);
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  dependency.wait();
  EXPECT_EQ(output.collect_ready(1), 1);
  EXPECT_EQ(released.load(), 1);
  EXPECT_EQ(output.snapshot().pending, 0);
  auto reused = output.submit(5, {}, nullptr, [](void *) {});
  reused.wait();
  output.close();
  EXPECT_TRUE(output.snapshot().closed);
  EXPECT_EQ(output.snapshot().capacity, 1);
  EXPECT_EQ(released.load(), 1);
  std::array<std::uint32_t, 3> values{};
  auto context = CUDAContext::get_instance().get_guard();
  CUDADriver::get_instance().memcpy_device_to_host(values.data(), data,
                                                   sizeof(values));
  EXPECT_EQ(values, (std::array<std::uint32_t, 3>{73, 73, 79}));
}

TEST_F(CudaExecutionDependency,
       DomainCloseLeavesQueriesAndOtherAdmissionAvailable) {
  cuda::ExecutionDomain output(fault, {1, 0});
  cuda::ExecutionDomain physics(fault, {1, 0});
  testing::CudaExternalGate gate;
  auto blocked = output.submit(
      1, {}, nullptr, [&](void *stream) { gate.enqueue_wait(stream); });
  auto closer = std::async(std::launch::async, [&] { output.close(); });
  auto admission = std::async(std::launch::async, [&] {
    EXPECT_FALSE(blocked.done());
    (void)output.snapshot();
    auto sentinel = physics.submit(2, {}, nullptr, [](void *) {});
    return poll_until(sentinel);
  });
  const bool responsive =
      admission.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
  EXPECT_TRUE(responsive);
  if (responsive)
    EXPECT_TRUE(admission.get());
  EXPECT_EQ(closer.wait_for(std::chrono::milliseconds(0)),
            std::future_status::timeout);
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  closer.get();
  if (!responsive)
    admission.get();
  EXPECT_TRUE(blocked.done());
  EXPECT_TRUE(output.snapshot().closed);
  EXPECT_THROW(output.submit(3, {}, nullptr, [](void *) {}),
               std::runtime_error);
}

TEST_F(CudaExecutionDependency,
       PartialEnqueueFailureRetainsLeasesUntilCompletion) {
  cuda::ExecutionDomain output(fault, {1, 0});
  cuda::ExecutionDomain physics(fault, {1, 0});
  testing::CudaExternalGate gate;
  std::atomic<int> released{0};
  EXPECT_THROW(
      output.submit(1, {}, std::make_shared<DomainResources>(output, released),
                    [&](void *stream) {
                      gate.enqueue_wait(stream);
                      CUDADriver::get_instance().memsetd32_async(at(0), 83, 1,
                                                                 stream);
                      throw std::runtime_error("failure after enqueue");
                    }),
      std::runtime_error);
  EXPECT_EQ(released.load(), 0);
  EXPECT_EQ(output.collect_ready(1), 0);
  EXPECT_FALSE(output.snapshot().accepting);
  EXPECT_THROW(output.submit(2, {}, nullptr, [](void *) {}),
               std::runtime_error);
  auto other = physics.submit(3, {}, nullptr, [](void *) {});
  EXPECT_TRUE(poll_until(other));
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  output.close();
  EXPECT_EQ(released.load(), 1);
  output.close();
  EXPECT_EQ(released.load(), 1);
  std::uint32_t value = 0;
  auto context = CUDAContext::get_instance().get_guard();
  CUDADriver::get_instance().memcpy_device_to_host(&value, at(0),
                                                   sizeof(value));
  EXPECT_EQ(value, 83u);
}

TEST_F(CudaExecutionDependency, InvalidDomainInputsHaveNoEnqueueSideEffects) {
  auto wrong_backend = std::make_shared<RuntimeFaultDomain>(Arch::x64, 71);
  EXPECT_THROW((cuda::ExecutionDomain{wrong_backend, {1, 0}}),
               std::invalid_argument);
  EXPECT_THROW((cuda::ExecutionDomain{fault, {0, 0}}), std::invalid_argument);
  cuda::ExecutionDomain domain(fault, {1, 0});
  auto other = std::make_shared<RuntimeFaultDomain>(Arch::cuda, 71);
  auto foreign = RuntimeCompletion::completed(Arch::cuda, 71, 1, other);
  int calls = 0;
  const auto enqueue = [&](void *) { ++calls; };
  EXPECT_THROW(domain.submit(1, {foreign}, nullptr, enqueue),
               std::invalid_argument);
  EXPECT_THROW(domain.submit(0, {}, nullptr, enqueue), std::invalid_argument);
  EXPECT_EQ(domain.snapshot().pending, 0);
  EXPECT_EQ(calls, 0);
  auto valid = domain.submit(1, {}, nullptr, enqueue);
  valid.wait();
  EXPECT_THROW(domain.submit(1, {}, nullptr, enqueue), std::invalid_argument);
  fault->begin_finalizing();
  EXPECT_THROW(domain.submit(2, {}, nullptr, enqueue), std::exception);
  EXPECT_EQ(calls, 1);
  domain.close();
  fault->mark_finalized();
  EXPECT_TRUE(valid.done());
}

TEST_F(CudaExecutionDependency, WarmAdmissionProgressesBeforeExternalRelease) {
  cuda::ExecutionDomain output(fault, {1, 0});
  cuda::ExecutionDomain physics(fault, {16, -1});
  const auto fill = [&](void *stream) {
    CUDADriver::get_instance().memsetd32_async(at(0), 89, 1, stream);
  };
  RuntimeCompletion last;
  for (int i = 1; i <= 32; ++i) {
    // Warm driver paths outside the measured blocked interval.
    last = physics.submit(i, {}, nullptr, fill);
    last.wait();
  }
  physics.collect_ready(16);
  testing::CudaExternalGate gate;
  auto blocked = output.submit(
      33, {}, nullptr, [&](void *stream) { gate.enqueue_wait(stream); });
  std::vector<double> duration_us;
  duration_us.reserve(1000);
  std::size_t busy = 0;
  {
    ScopedExternalProfilerAnnotation range(
        "F3/isolated_progress_before_release");
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    for (std::uint64_t sequence = 34; sequence < 1034;) {
      const auto start = std::chrono::steady_clock::now();
      try {
        last = physics.submit(sequence, {}, nullptr, fill);
        duration_us.push_back(std::chrono::duration<double, std::micro>(
                                  std::chrono::steady_clock::now() - start)
                                  .count());
        ++sequence;
      } catch (const cuda::ExecutionDomainBusy &) {
        ++busy;
        std::this_thread::yield();
      }
      if (std::chrono::steady_clock::now() >= deadline)
        break;
    }
    EXPECT_TRUE(poll_until(last));
    EXPECT_FALSE(blocked.done());
    EXPECT_FALSE(gate.timed_out());
  }
  {
    ScopedExternalProfilerAnnotation range("F3/release_external_producer");
    gate.release();
  }
  blocked.wait();
  ASSERT_EQ(duration_us.size(), 1000);
  std::sort(duration_us.begin(), duration_us.end());
  const auto priority = physics.snapshot();
  RecordProperty("submit_p50_us", std::to_string(duration_us[499]));
  RecordProperty("submit_p95_us", std::to_string(duration_us[949]));
  RecordProperty("busy_retries", std::to_string(busy));
  RecordProperty("effective_priority",
                 std::to_string(priority.effective_priority));
  std::printf(
      "F3 native warm admission: n=1000 p50=%.3f us p95=%.3f us busy=%zu "
      "priority=%d\n",
      duration_us[499], duration_us[949], busy, priority.effective_priority);
}

}  // namespace
}  // namespace taichi::lang
#endif
