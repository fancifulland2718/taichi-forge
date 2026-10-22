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
    return RuntimeCompletion::from_cuda_stream(
        71, fault->allocate_submission_sequence(), streams[index], fault, pool);
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

TEST_F(CudaExecutionDependency,
       EventDestructionLeavesUnrelatedDefaultWorkPending) {
  auto context = CUDAContext::get_instance().get_guard();
  auto &driver = CUDADriver::get_instance();
  std::array<void *, 128> events{};
  for (auto &event : events) {
    driver.event_create(&event, CU_EVENT_DISABLE_TIMING);
    driver.event_record(event, streams[1]);
  }
  driver.stream_synchronize(streams[1]);
  testing::CudaExternalGate gate;
  gate.enqueue_wait(nullptr);
  void *pending = nullptr;
  driver.event_create(&pending, CU_EVENT_DISABLE_TIMING);
  driver.event_record(pending, nullptr);
  ASSERT_EQ(driver.event_query.call(pending), CUDA_ERROR_NOT_READY);
  const auto start = std::chrono::steady_clock::now();
  for (auto event : events)
    driver.event_destroy(event);
  std::cout << "event destruction_us="
            << std::chrono::duration<double, std::micro>(
                   std::chrono::steady_clock::now() - start)
                   .count()
            << std::endl;
  EXPECT_EQ(driver.event_query.call(pending), CUDA_ERROR_NOT_READY);
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  driver.event_synchronize(pending);
  driver.event_destroy(pending);
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
      {}, std::make_shared<DomainResources>(output, released),
      [&](void *stream) {
        gate.enqueue_wait(stream);
        CUDADriver::get_instance().memsetd32_async(at(0), 73, 1, stream);
      });
  bool attempted = false;
  EXPECT_THROW(output.submit({}, nullptr, [&](void *) { attempted = true; }),
               cuda::ExecutionDomainBusy);
  EXPECT_FALSE(attempted);
  EXPECT_EQ(fault->next_submission_sequence(), blocked.sequence() + 1);
  auto dependency = input.submit({blocked}, nullptr, [&](void *stream) {
    CUDADriver::get_instance().memcpy_device_to_device_async(at(1), at(0), 4,
                                                             stream);
  });
  auto sentinel = physics.submit({}, nullptr, [&](void *stream) {
    CUDADriver::get_instance().memsetd32_async(at(2), 79, 1, stream);
  });
  EXPECT_EQ(dependency.sequence(), blocked.sequence() + 1);
  EXPECT_EQ(sentinel.sequence(), dependency.sequence() + 1);
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
  auto reused = output.submit({}, nullptr, [](void *) {});
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
      {}, nullptr, [&](void *stream) { gate.enqueue_wait(stream); });
  auto closer = std::async(std::launch::async, [&] { output.close(); });
  auto admission = std::async(std::launch::async, [&] {
    EXPECT_FALSE(blocked.done());
    (void)output.snapshot();
    auto sentinel = physics.submit({}, nullptr, [](void *) {});
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
  EXPECT_THROW(output.submit({}, nullptr, [](void *) {}), std::runtime_error);
}

TEST_F(CudaExecutionDependency,
       RejectedProviderAdmissionDoesNotEnqueueDependenciesOrPoisonDomain) {
  cuda::ExecutionDomain output(fault, {1, 0});
  cuda::ExecutionDomain physics(fault, {1, 0});
  testing::CudaExternalGate gate;
  auto blocked = output.submit(
      {}, nullptr, [&](void *stream) { gate.enqueue_wait(stream); });
  ASSERT_FALSE(blocked.done());
  const auto next = fault->next_submission_sequence();
  std::atomic<int> cancelled{0};
  int admissions = 0, enqueues = 0;
  const auto enqueue = [&](void *stream) {
    ++enqueues;
    CUDADriver::get_instance().memsetd32_async(at(0), 113, 1, stream);
  };
  const auto reject = [&] {
    ++admissions;
    throw std::runtime_error("Provider capacity exhausted before enqueue");
  };
  EXPECT_THROW(
      physics.submit({blocked},
                     std::make_shared<DomainResources>(physics, cancelled),
                     enqueue, reject),
      std::runtime_error);
  EXPECT_EQ(cancelled.load(), 1);
  EXPECT_EQ(admissions, 1);
  EXPECT_EQ(enqueues, 0);
  EXPECT_EQ(physics.snapshot().pending, 0);
  EXPECT_TRUE(physics.snapshot().accepting);
  EXPECT_EQ(fault->next_submission_sequence(), next);
  // A subsequent submit without the rejected dependency must make progress
  // before the producer releases its real external wait.
  auto sentinel = physics.submit({}, nullptr, enqueue, [&] { ++admissions; });
  EXPECT_EQ(sentinel.sequence(), next);
  EXPECT_TRUE(poll_until(sentinel));
  EXPECT_FALSE(blocked.done());
  EXPECT_EQ(admissions, 2);
  EXPECT_EQ(enqueues, 1);
  // Busy and invalid predecessor checks precede provider admission itself.
  EXPECT_THROW(output.submit({}, nullptr, enqueue, reject),
               cuda::ExecutionDomainBusy);
  auto other_owner = std::make_shared<RuntimeFaultDomain>(Arch::cuda, 71);
  auto foreign = RuntimeCompletion::completed(Arch::cuda, 71, 1, other_owner);
  EXPECT_THROW(physics.submit({foreign}, nullptr, enqueue, reject),
               std::invalid_argument);
  EXPECT_EQ(admissions, 2);
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  blocked.wait();
  std::uint32_t value = 0;
  auto context = CUDAContext::get_instance().get_guard();
  CUDADriver::get_instance().memcpy_device_to_host(&value, at(0),
                                                   sizeof(value));
  EXPECT_EQ(value, 113u);
}

TEST_F(CudaExecutionDependency,
       PartialEnqueueFailureRetainsLeasesUntilCompletion) {
  cuda::ExecutionDomain output(fault, {1, 0});
  cuda::ExecutionDomain physics(fault, {1, 0});
  testing::CudaExternalGate gate;
  std::atomic<int> released{0};
  EXPECT_THROW(
      output.submit({}, std::make_shared<DomainResources>(output, released),
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
  EXPECT_THROW(output.submit({}, nullptr, [](void *) {}), std::runtime_error);
  auto other = physics.submit({}, nullptr, [](void *) {});
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
  EXPECT_THROW(domain.submit({foreign}, nullptr, enqueue),
               std::invalid_argument);
  EXPECT_EQ(domain.snapshot().pending, 0);
  EXPECT_EQ(calls, 0);
  auto valid = domain.submit({}, nullptr, enqueue);
  EXPECT_EQ(valid.sequence(), 1);
  valid.wait();
  fault->begin_finalizing();
  EXPECT_THROW(domain.submit({}, nullptr, enqueue), std::exception);
  EXPECT_EQ(calls, 1);
  domain.close();
  fault->mark_finalized();
  EXPECT_TRUE(valid.done());
}

TEST_F(CudaExecutionDependency,
       SampledTimingExcludesPreludeAndFreezesBeforeReuse) {
  cuda::ExecutionDomain domain(fault, {2, 0, 4, true});
  testing::CudaExternalGate gate;
  auto pending = domain.submit(
      {}, nullptr,
      [&](void *stream) {
        CUDADriver::get_instance().memsetd32_async(at(0), 19, 1, stream);
      },
      {}, [&](void *stream) { gate.enqueue_wait(stream); }, {}, true);
  EXPECT_FALSE(pending.done());
  EXPECT_EQ(pending.gpu_timing_snapshot().status, "pending");
  // This deliberate external wait is outside the instrumented envelope.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(pending.done());
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  pending.wait();
  const auto snapshot = pending.gpu_timing_snapshot();
  ASSERT_TRUE(snapshot.available);
  EXPECT_LT(snapshot.duration_ns, 50000000u);
  EXPECT_EQ(snapshot.stream_id, domain.snapshot().stream);
  auto &driver = CUDADriver::get_instance();
  // A warm sampled submit must reuse both completion and timestamp events.
  {
    auto saved_create = driver.event_create;
    auto restore = std::shared_ptr<void>(nullptr, [&](void *) {
      driver.event_create = std::move(saved_create);
    });
    driver.event_create.set(
        reinterpret_cast<void *>(+[](void **, uint32) -> uint32 { return 2; }));
    for (int i = 0; i < 8; ++i) {
      auto current = domain.submit(
          {}, nullptr,
          [&](void *stream) { driver.memsetd32_async(at(1), 23, 1, stream); },
          {}, {}, {}, i % 2 == 0);
      current.wait();
      EXPECT_EQ(current.gpu_timing_snapshot().available, i % 2 == 0);
    }
  }
  domain.close();
  EXPECT_EQ(pending.gpu_timing_snapshot().duration_ns, snapshot.duration_ns);
}

TEST_F(CudaExecutionDependency, TimingFinalizerRunsBeforeFailedPrefixRetires) {
  for (bool fail_in_prelude : {false, true}) {
    cuda::ExecutionDomain domain(fault, {1, 0, 4, true});
    int finalized = 0;
    const auto fail = [] {
      throw std::runtime_error("test operation failure");
    };
    EXPECT_THROW(domain.submit(
                     {}, nullptr,
                     [&](void *) {
                       if (!fail_in_prelude)
                         fail();
                     },
                     {},
                     [&](void *) {
                       if (fail_in_prelude)
                         fail();
                     },
                     [&](void *stream) {
                       ++finalized;
                       CUDADriver::get_instance().memsetd32_async(at(0), 97, 1,
                                                                  stream);
                     },
                     true),
                 std::runtime_error);
    EXPECT_EQ(finalized, 1);
    domain.close();
    std::uint32_t result = 0;
    auto current = CUDAContext::get_instance().get_guard();
    CUDADriver::get_instance().memcpy_device_to_host(&result, at(0),
                                                     sizeof(result));
    EXPECT_EQ(result, 97);
  }
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
    last = physics.submit({}, nullptr, fill);
    last.wait();
  }
  physics.collect_ready(16);
  testing::CudaExternalGate gate;
  auto blocked = output.submit(
      {}, nullptr, [&](void *stream) { gate.enqueue_wait(stream); });
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
        last = physics.submit({}, nullptr, fill);
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

TEST_F(CudaExecutionDependency,
       RegistryRetiresAllDomainsWithoutHoldingItsLock) {
  cuda::ExecutionDomainRegistry registry(fault, 2);
  EXPECT_TRUE(registry.belongs_to(fault));
  EXPECT_FALSE(registry.belongs_to(
      std::make_shared<RuntimeFaultDomain>(Arch::cuda, 71)));
  auto output = registry.create({1, 0});
  auto physics = registry.create({1, 0});
  auto extra = registry.create({1, 0});
  testing::CudaExternalGate gate;
  std::atomic<int> released{0};
  auto blocked =
      output->submit({}, std::make_shared<DomainResources>(*output, released),
                     [&](void *stream) { gate.enqueue_wait(stream); });
  auto sentinel = physics->submit({}, nullptr, [&](void *stream) {
    CUDADriver::get_instance().memsetd32_async(at(0), 97, 1, stream);
  });
  ASSERT_TRUE(poll_until(sentinel));
  auto retirement = std::async(std::launch::async, [&] { registry.retire(); });
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (physics->snapshot().accepting &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  EXPECT_FALSE(registry.accepting());
  EXPECT_FALSE(physics->snapshot().accepting);
  EXPECT_ANY_THROW(registry.create({1, 0}));
  EXPECT_ANY_THROW(physics->submit({}, nullptr, [](void *) {}));
  EXPECT_FALSE(blocked.done());
  EXPECT_EQ(released.load(), 0);
  EXPECT_EQ(retirement.wait_for(std::chrono::milliseconds(0)),
            std::future_status::timeout);
  auto second_retirement =
      std::async(std::launch::async, [&] { registry.retire(); });
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  retirement.get();
  second_retirement.get();
  EXPECT_TRUE(output->snapshot().closed);
  EXPECT_TRUE(physics->snapshot().closed);
  EXPECT_TRUE(extra->snapshot().closed);
  EXPECT_TRUE(blocked.done());
  EXPECT_EQ(released.load(), 1);
}

TEST_F(CudaExecutionDependency, RegistryReusesOnlyClosedDomainSlots) {
  cuda::ExecutionDomainRegistry registry(fault, 1);
  auto previous = registry.create({1, 0});
  const auto original_identity = previous->snapshot().identity;
  for (int i = 0; i < 32; ++i) {
    previous->close();
    std::weak_ptr<cuda::ExecutionDomain> retired = previous;
    previous = registry.create({1, 0});
    EXPECT_TRUE(retired.expired());
    EXPECT_NE(previous->snapshot().identity, original_identity);
  }
  // The runtime retains an open domain even when its application wrapper is
  // discarded, so owner teardown can still retire its storage before code.
  std::weak_ptr<cuda::ExecutionDomain> retained = previous;
  previous.reset();
  EXPECT_FALSE(retained.expired());
  registry.retire();
  EXPECT_TRUE(retained.lock()->snapshot().closed);
}

TEST_F(CudaExecutionDependency, RegistryGrowthPreservesDomainsAndCompletions) {
  cuda::ExecutionDomainRegistry registry(fault);
  std::vector<std::shared_ptr<cuda::ExecutionDomain>> domains;
  std::vector<RuntimeCompletion> completions;
  // Cross the former default registry limit with minimal one-event domains.
  // No large kernel or simulation is needed to exercise this cold ownership.
  for (int i = 0; i < 65; ++i) {
    auto domain = registry.create({1, 0, 1});
    completions.push_back(domain->submit({}, nullptr, [](void *) {}));
    domains.push_back(std::move(domain));
  }
  registry.synchronize();
  for (const auto &completion : completions)
    EXPECT_TRUE(completion.done());
  const auto first_identity = domains.front()->snapshot().identity;
  domains.front()->close();
  auto replacement = registry.create({1, 0, 1});
  EXPECT_NE(replacement->snapshot().identity, first_identity);
  EXPECT_TRUE(completions.front().done());
  // Failed creation neither publishes an owner nor poisons later growth.
  EXPECT_THROW(registry.create({0, 0}), std::invalid_argument);
  auto after_failure = registry.create({1, 0, 1});
  registry.retire();
  for (const auto &domain : domains)
    EXPECT_TRUE(domain->snapshot().closed);
  EXPECT_TRUE(replacement->snapshot().closed);
  EXPECT_TRUE(after_failure->snapshot().closed);
  EXPECT_THROW(registry.create({1, 0}), std::runtime_error);
  for (const auto &completion : completions)
    EXPECT_TRUE(completion.done());
}

struct DomainInstance final : RuntimeClient {
  std::function<void()> on_retire;
  std::atomic<int> retire_calls{0};
  void retire() override {
    ++retire_calls;
    if (on_retire)
      on_retire();
  }
};

TEST_F(CudaExecutionDependency, InstanceRetirementWaitsOnlyItsOwnPrefixAndReusesBudget) {
  cuda::ExecutionDomain domain(fault, {3, 0, 1});
  auto client = std::make_shared<DomainInstance>();
  auto instance = domain.prepare_owned_instance(client, [](const auto &) {});
  auto first = domain.submit_instance(instance, {}, nullptr, [&](void *stream) {
    CUDADriver::get_instance().memsetd32_async(at(0), 17, 1, stream);
  });
  testing::CudaExternalGate gate;
  auto later = domain.submit({}, nullptr, [&](void *stream) { gate.enqueue_wait(stream); });
  client->on_retire = [&] {
    EXPECT_TRUE(first.done());
    EXPECT_FALSE(later.done());
    EXPECT_TRUE(domain.snapshot().accepting);
  };
  auto closing = std::async(std::launch::async, [&] { domain.retire_instance(instance); });
  const bool independent = closing.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
  EXPECT_TRUE(independent);
  EXPECT_FALSE(gate.timed_out());
  if (!independent)
    gate.release();  // Keep a failed assertion from deadlocking teardown.
  closing.get();
  EXPECT_EQ(client->retire_calls.load(), 1);
  domain.retire_instance(instance);
  int enqueued = 0;
  EXPECT_THROW(domain.submit_instance(instance, {}, nullptr, [&](void *) { ++enqueued; }),
               std::invalid_argument);
  EXPECT_EQ(enqueued, 0);
  for (int i = 0; i < 3; ++i) {
    auto replacement_client = std::make_shared<DomainInstance>();
    auto replacement = domain.prepare_owned_instance(replacement_client, [](const auto &) {});
    domain.retire_instance(replacement);
    EXPECT_EQ(replacement_client->retire_calls.load(), 1);
  }
  if (independent) {
    EXPECT_FALSE(later.done());
    gate.release();
  }
  later.wait();
  domain.close();
  EXPECT_TRUE(first.done());
  first.wait();
  EXPECT_EQ(client->retire_calls.load(), 1);
}

TEST_F(CudaExecutionDependency, InstanceRetirementKeepsPendingWorkAndOtherDomainProgress) {
  cuda::ExecutionDomain domain(fault, {2, 0, 1});
  cuda::ExecutionDomain peer(fault, {1, 0, 1});
  auto client = std::make_shared<DomainInstance>();
  auto instance = domain.prepare_owned_instance(client, [](const auto &) {});
  testing::CudaExternalGate gate;
  auto pending = domain.submit_instance(instance, {}, nullptr, [&](void *stream) {
    gate.enqueue_wait(stream);
    CUDADriver::get_instance().memsetd32_async(at(0), 23, 1, stream);
  });
  EXPECT_THROW(peer.submit_instance(instance, {}, nullptr, [](void *) {}), std::invalid_argument);
  EXPECT_THROW(peer.retire_instance(instance), std::invalid_argument);
  auto closing = std::async(std::launch::async, [&] { domain.retire_instance(instance); });
  EXPECT_EQ(closing.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
  EXPECT_EQ(client->retire_calls.load(), 0);
  auto independent = peer.submit({}, nullptr, [&](void *stream) {
    CUDADriver::get_instance().memsetd32_async(at(1), 29, 1, stream);
  });
  EXPECT_TRUE(poll_until(independent));
  EXPECT_FALSE(pending.done());
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  closing.get();
  EXPECT_TRUE(pending.done());
  EXPECT_EQ(client->retire_calls.load(), 1);
}

TEST_F(CudaExecutionDependency, IndividualRetirementAndDomainCloseDoNotDoubleRetire) {
  cuda::ExecutionDomain domain(fault, {1, 0, 1});
  auto client = std::make_shared<DomainInstance>();
  auto instance = domain.prepare_owned_instance(client, [](const auto &) {});
  std::promise<void> entered, release;
  auto released = release.get_future().share();
  client->on_retire = [&] {
    entered.set_value();
    released.wait();
  };
  auto individual = std::async(std::launch::async, [&] { domain.retire_instance(instance); });
  entered.get_future().wait();
  EXPECT_THROW(domain.prepare_owned_instance(std::make_shared<DomainInstance>(), [](const auto &) {}),
               std::runtime_error);
  auto closing = std::async(std::launch::async, [&] { domain.close(); });
  EXPECT_EQ(closing.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
  release.set_value();
  individual.get();
  closing.get();
  domain.retire_instance(instance);
  EXPECT_TRUE(domain.snapshot().closed);
  EXPECT_EQ(client->retire_calls.load(), 1);
}

TEST_F(CudaExecutionDependency, FailedIndividualRetirementPreservesPeersAndQuarantinesItsBudget) {
  cuda::ExecutionDomain domain(fault, {1, 0, 2});
  auto broken_client = std::make_shared<DomainInstance>();
  broken_client->on_retire = [] { throw std::runtime_error("retirement failed"); };
  auto broken = domain.prepare_owned_instance(broken_client, [](const auto &) {});
  auto healthy_client = std::make_shared<DomainInstance>();
  auto healthy = domain.prepare_owned_instance(healthy_client, [](const auto &) {});
  EXPECT_THROW(domain.retire_instance(broken), std::runtime_error);
  EXPECT_THROW(domain.retire_instance(broken), std::runtime_error);
  EXPECT_EQ(broken_client->retire_calls.load(), 1);
  EXPECT_THROW(domain.prepare_owned_instance(std::make_shared<DomainInstance>(), [](const auto &) {}),
               std::runtime_error);
  EXPECT_TRUE(poll_until(domain.submit_instance(healthy, {}, nullptr, [](void *) {})));
  domain.retire_instance(healthy);
  auto replacement_client = std::make_shared<DomainInstance>();
  auto replacement = domain.prepare_owned_instance(replacement_client, [](const auto &) {});
  EXPECT_TRUE(poll_until(domain.submit_instance(replacement, {}, nullptr, [](void *) {})));
  EXPECT_THROW(domain.close(), std::runtime_error);
  EXPECT_EQ(healthy_client->retire_calls.load(), 1);
  EXPECT_EQ(replacement_client->retire_calls.load(), 1);
  EXPECT_EQ(broken_client->retire_calls.load(), 1);
}

TEST_F(CudaExecutionDependency, FatalIndividualRetirementFaultsRuntime) {
  cuda::ExecutionDomain domain(fault, {1, 0, 1});
  auto client = std::make_shared<DomainInstance>();
  // Inject the provider report, without deliberately faulting the actual GPU.
  client->on_retire = [] {
    throw BackendRuntimeError(Arch::cuda, CUDA_ERROR_ILLEGAL_ADDRESS,
                              "retire test provider", "injected device fault");
  };
  auto instance = domain.prepare_owned_instance(client, [](const auto &) {});
  auto ticket = domain.submit_instance(instance, {}, nullptr, [](void *) {});
  ticket.wait();
  EXPECT_THROW(domain.retire_instance(instance), BackendRuntimeError);
  const auto observed = fault->snapshot();
  ASSERT_TRUE(observed.first_fault.has_value());
  EXPECT_EQ(observed.first_fault->submission_sequence, ticket.sequence());
  EXPECT_EQ(observed.state, RuntimeLifecycleState::kFaulted);
  EXPECT_ANY_THROW(domain.submit({}, nullptr, [](void *) {}));
  EXPECT_THROW(domain.retire_instance(instance), BackendRuntimeError);
  EXPECT_EQ(client->retire_calls.load(), 1);
  // The finite failed owner is quarantined; no backend frees run after fault.
}

TEST_F(CudaExecutionDependency, RegistryDrainsPeersBeforeProviderDestruction) {
  cuda::ExecutionDomainRegistry registry(fault, 2);
  auto ready_domain = registry.create({1, 0});
  auto blocked_domain = registry.create({1, 0});
  testing::CudaExternalGate gate;
  auto pending = blocked_domain->submit(
      {}, nullptr, [&](void *stream) { gate.enqueue_wait(stream); });
  auto instance = std::make_shared<DomainInstance>();
  instance->on_retire = [&] { EXPECT_TRUE(pending.done()); };
  ready_domain->prepare_instance(instance, [] {});
  auto retirement = std::async(std::launch::async, [&] { registry.retire(); });
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (blocked_domain->snapshot().accepting &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  EXPECT_FALSE(blocked_domain->snapshot().accepting);
  EXPECT_FALSE(pending.done());
  EXPECT_EQ(instance->retire_calls.load(), 0);
  EXPECT_EQ(retirement.wait_for(std::chrono::milliseconds(0)),
            std::future_status::timeout);
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  retirement.get();
  EXPECT_EQ(instance->retire_calls.load(), 1);
}

TEST_F(CudaExecutionDependency, InstanceCapacityPrecedesPreparationEffects) {
  cuda::ExecutionDomain domain(fault, {1, 0, 1});
  auto failed = std::make_shared<DomainInstance>();
  failed->on_retire = [&] { (void)domain.snapshot(); };
  EXPECT_THROW(
      domain.prepare_instance(
          failed, [] { throw std::invalid_argument("invalid recipe"); }),
      std::invalid_argument);
  EXPECT_EQ(failed->retire_calls.load(), 1);
  auto instance = std::make_shared<DomainInstance>();
  instance->on_retire = [&] { domain.close(); };
  int prepared = 0;
  domain.prepare_instance(instance, [&] { ++prepared; });
  EXPECT_THROW(domain.prepare_instance(std::make_shared<DomainInstance>(),
                                       [&] { ++prepared; }),
               std::runtime_error);
  EXPECT_EQ(prepared, 1);
  std::weak_ptr<DomainInstance> retained = instance;
  instance.reset();
  EXPECT_FALSE(retained.expired());
  domain.close();
  EXPECT_TRUE(retained.expired());
  EXPECT_EQ(failed->retire_calls.load(), 1);
  EXPECT_THROW(domain.prepare_instance(failed, [&] { ++prepared; }),
               std::runtime_error);
  EXPECT_EQ(prepared, 1);
}

TEST_F(CudaExecutionDependency, CloseWaitsForPreparationWithoutAdmissionLock) {
  cuda::ExecutionDomain domain(fault, {1, 0, 1});
  cuda::ExecutionDomain other(fault, {1, 0});
  std::promise<void> entered, release;
  auto released = release.get_future().share();
  auto instance = std::make_shared<DomainInstance>();
  auto preparing = std::async(std::launch::async, [&] {
    domain.prepare_instance(instance, [&] {
      entered.set_value();
      released.wait();
    });
  });
  entered.get_future().wait();
  auto close = std::async(std::launch::async, [&] { domain.close(); });
  auto query = std::async(std::launch::async, [&] {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (domain.snapshot().accepting &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    EXPECT_FALSE(domain.snapshot().closed);
    EXPECT_FALSE(domain.snapshot().accepting);
    EXPECT_ANY_THROW(domain.submit({}, nullptr, [](void *) {}));
    return poll_until(other.submit({}, nullptr, [](void *) {}));
  });
  const bool responsive =
      query.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
  EXPECT_TRUE(responsive);
  EXPECT_EQ(close.wait_for(std::chrono::milliseconds(0)),
            std::future_status::timeout);
  EXPECT_EQ(instance->retire_calls.load(), 0);
  release.set_value();
  preparing.get();
  close.get();
  EXPECT_TRUE(query.get());
  EXPECT_EQ(instance->retire_calls.load(), 1);
  EXPECT_TRUE(domain.snapshot().closed);
}

TEST_F(CudaExecutionDependency, ConcurrentCloseWaitsForInstanceRetirement) {
  cuda::ExecutionDomain domain(fault, {1, 0, 1});
  std::promise<void> entered, release;
  auto released = release.get_future().share();
  auto instance = std::make_shared<DomainInstance>();
  instance->on_retire = [&] {
    EXPECT_FALSE(domain.snapshot().closed);
    entered.set_value();
    released.wait();
  };
  domain.prepare_instance(instance, [] {});
  auto first = std::async(std::launch::async, [&] { domain.close(); });
  entered.get_future().wait();
  auto second = std::async(std::launch::async, [&] { domain.close(); });
  EXPECT_EQ(second.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  EXPECT_FALSE(domain.snapshot().closed);
  release.set_value();
  first.get();
  second.get();
  EXPECT_EQ(instance->retire_calls.load(), 1);
  EXPECT_TRUE(domain.snapshot().closed);
}

TEST_F(CudaExecutionDependency, FailedInstanceRetirementIsSticky) {
  cuda::ExecutionDomainRegistry registry(fault, 1);
  auto retained_domain = registry.create({1, 0, 1});
  auto &domain = *retained_domain;
  auto instance = std::make_shared<DomainInstance>();
  instance->on_retire = [] {
    throw std::runtime_error("instance retire error");
  };
  domain.prepare_instance(instance, [] {});
  EXPECT_THROW(domain.close(), std::runtime_error);
  EXPECT_THROW(domain.close(), std::runtime_error);
  EXPECT_EQ(instance->retire_calls.load(), 1);
  EXPECT_FALSE(domain.snapshot().accepting);
  EXPECT_FALSE(domain.snapshot().closed);
  auto peer = registry.create({1, 0, 1});
  EXPECT_NE(peer->snapshot().identity, domain.snapshot().identity);
  EXPECT_ANY_THROW(domain.submit({}, nullptr, [](void *) {}));
  std::weak_ptr<cuda::ExecutionDomain> retained = retained_domain;
  retained_domain.reset();
  EXPECT_FALSE(retained.expired());
  EXPECT_THROW(registry.retire(), std::runtime_error);
  EXPECT_TRUE(peer->snapshot().closed);
  EXPECT_FALSE(retained.lock()->snapshot().closed);
  EXPECT_EQ(instance->retire_calls.load(), 1);
  // Cold destruction quarantines this bounded instance instead of freeing
  // storage after a failed retirement. The exception must remain observable.
}

}  // namespace
}  // namespace taichi::lang
#endif
