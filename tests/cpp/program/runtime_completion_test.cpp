#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>

#include "taichi/program/runtime_completion.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/rhi/backend_error.h"
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_driver.h"
#endif
#ifdef TI_TEST_DISPLAY_COMPLETION
struct GLFWwindow;
#include "taichi/ui/ggui/renderables/set_image.h"
#endif

namespace taichi::lang {
namespace {

class FakeSemaphore final : public StreamSemaphoreObject {
 public:
  bool is_ready() const override {
    ++polls;
    if (fail) {
      throw std::runtime_error("fake completion failure");
    }
    return ready;
  }

  bool wait() const override {
    ++waits;
    if (fail) {
      throw std::runtime_error("fake completion failure");
    }
    ready = true;
    return true;
  }

  mutable bool ready{false};
  mutable bool fail{false};
  mutable int polls{0};
  mutable int waits{0};
};

class CountedResources final : public RuntimeCompletionResources {
 public:
  CountedResources(std::atomic<int> *released, std::size_t count)
      : released_(released), count_(count) {
  }

  ~CountedResources() override {
    released_->fetch_add(1, std::memory_order_relaxed);
  }

  std::size_t retained_resource_count(
      std::uint32_t kind) const noexcept override {
    return kind == 7 ? count_ : 0;
  }

 private:
  std::atomic<int> *released_;
  std::size_t count_;
};

// Completion is independently observable while the host waiter is still
// inside its backend call, including the interval before it returns to State.
class GatedSemaphore final : public StreamSemaphoreObject {
 public:
  bool is_ready() const override {
    if (fail_query.load()) {
      throw std::runtime_error("first query failure");
    }
    return ready.load();
  }

  bool wait() const override {
    std::unique_lock<std::mutex> lock(mutex);
    ++entered;
    changed.notify_all();
    changed.wait(lock, [this] { return released; });
    if (fail_wait.load()) {
      throw std::runtime_error("later wait failure");
    }
    ready.store(true);
    return true;
  }

  bool await_waiters(int count) {
    std::unique_lock<std::mutex> lock(mutex);
    return changed.wait_for(lock, std::chrono::seconds(2),
                            [&] { return entered >= count; });
  }

  void release() {
    std::lock_guard<std::mutex> lock(mutex);
    released = true;
    changed.notify_all();
  }

  mutable std::atomic<bool> ready{false};
  std::atomic<bool> fail_query{false};
  std::atomic<bool> fail_wait{false};

 private:
  mutable std::mutex mutex;
  mutable std::condition_variable changed;
  mutable int entered{0};
  bool released{false};
};

TEST(RuntimeCompletion, WaitDoesNotBlockPendingQueryAndSubmission) {
  auto semaphore = std::make_shared<GatedSemaphore>();
  auto completion = RuntimeCompletion::from_stream_semaphore(
      Arch::vulkan, 17, 1, semaphore);
  auto waiter = std::async(std::launch::async, [&] { completion.wait(); });
  EXPECT_TRUE(semaphore->await_waiters(1));
  auto admission = std::async(std::launch::async, [&] {
    const bool ready = completion.done();
    auto next = RuntimeCompletion::completed(Arch::vulkan, 17, 2);
    next.wait();
    return ready;
  });
  const bool responsive = admission.wait_for(std::chrono::milliseconds(200)) ==
                          std::future_status::ready;
  semaphore->release();
  waiter.get();
  EXPECT_TRUE(responsive);
  if (responsive) {
    EXPECT_FALSE(admission.get());
  } else {
    admission.get();
  }
}

#ifdef TI_WITH_CUDA
TEST(RuntimeCompletion, BlockingAndDefaultEventCachesDoNotMix) {
  RuntimeCompletionCudaEventPool pool({}, 2);
  // These sentinels stay in the cache and are checked out before destruction;
  // the test does not call CUDA with synthetic handles or require a GPU.
  auto *default_event = reinterpret_cast<void *>(std::uintptr_t{1});
  auto *blocking_event = reinterpret_cast<void *>(std::uintptr_t{2});
  for (int iteration = 0; iteration < 8; ++iteration) {
    pool.release(default_event, true);
    pool.release(blocking_event, true, true);
    EXPECT_EQ(pool.snapshot().cached, 2);
    EXPECT_EQ(pool.acquire(), default_event);
    EXPECT_EQ(pool.acquire(true), blocking_event);
  }
  EXPECT_EQ(pool.snapshot().cached, 0);
  EXPECT_EQ(pool.snapshot().reused, 16);
  EXPECT_EQ(pool.snapshot().created, 0);
}

TEST(RuntimeCompletion, ConcurrentDriverWaitLeavesSubmissionMutexAvailable) {
  GatedSemaphore gate;
  std::mutex driver_mutex;
  CUDADriverFunction<void *> wait;
  wait.set(reinterpret_cast<void *>(+[](void *value) -> uint32 {
    return static_cast<GatedSemaphore *>(value)->wait() ? 0 : 1;
  }));
  wait.set_lock(&driver_mutex);
  CUDADriverFunction<> query;
  query.set(reinterpret_cast<void *>(+[]() -> uint32 { return 0; }));
  query.set_lock(&driver_mutex);
  auto waiter = std::async(std::launch::async,
                          [&] { return wait.call_concurrent(&gate); });
  EXPECT_TRUE(gate.await_waiters(1));
  auto submission = std::async(std::launch::async, [&] { return query.call(); });
  EXPECT_EQ(submission.wait_for(std::chrono::milliseconds(200)),
            std::future_status::ready);
  gate.release();
  EXPECT_EQ(submission.get(), 0);
  EXPECT_EQ(waiter.get(), 0);
}
#endif

TEST(RuntimeCompletion, QueryRetiresResourcesWhileWaitCallReturns) {
  auto semaphore = std::make_shared<GatedSemaphore>();
  auto completion = RuntimeCompletion::from_stream_semaphore(
      Arch::vulkan, 17, 1, semaphore);
  std::atomic<int> released{0};
  completion.attach_resources(std::make_shared<CountedResources>(&released, 1));
  auto waiter = std::async(std::launch::async, [&] { completion.wait(); });
  EXPECT_TRUE(semaphore->await_waiters(1));
  semaphore->ready.store(true);
  auto query = std::async(std::launch::async, [&] { return completion.done(); });
  const bool responsive = query.wait_for(std::chrono::milliseconds(200)) ==
                          std::future_status::ready;
  EXPECT_TRUE(responsive);
  if (responsive) {
    EXPECT_EQ(released.load(), 1);
    EXPECT_TRUE(completion.has_backend_work());
  }
  semaphore->release();
  EXPECT_TRUE(query.get());
  waiter.get();
  completion.mark_completed();
  EXPECT_FALSE(completion.has_backend_work());
  EXPECT_EQ(released.load(), 1);
}

TEST(RuntimeCompletion, InvalidationAndFirstErrorWinAgainstPendingWait) {
  for (bool query_failure : {false, true}) {
    auto semaphore = std::make_shared<GatedSemaphore>();
    auto completion = RuntimeCompletion::from_stream_semaphore(
        Arch::vulkan, 17, 1, semaphore);
    std::atomic<int> released{0};
    completion.attach_resources(std::make_shared<CountedResources>(&released, 1));
    auto waiter = std::async(std::launch::async, [&] {
      try {
        completion.wait();
        return std::string{};
      } catch (const std::exception &error) {
        return std::string(error.what());
      }
    });
    EXPECT_TRUE(semaphore->await_waiters(1));
    auto invalidator = std::async(std::launch::async, [&] {
      if (query_failure) {
        semaphore->fail_query.store(true);
        try {
          completion.done();
        } catch (const std::exception &) {
        }
      } else {
        completion.invalidate("runtime reset");
      }
    });
    const bool responsive = invalidator.wait_for(std::chrono::milliseconds(200)) ==
                            std::future_status::ready;
    semaphore->fail_wait.store(true);
    semaphore->release();
    invalidator.get();
    EXPECT_TRUE(responsive);
    const auto expected = query_failure ? "first query failure" : "runtime reset";
    EXPECT_EQ(waiter.get(), expected);
    completion.mark_completed();
    EXPECT_EQ(released.load(), 1);
    EXPECT_EQ(completion.first_error_message(), expected);
  }
}

TEST(RuntimeCompletion, TeardownDrainsConcurrentBackendWaiters) {
  auto semaphore = std::make_shared<GatedSemaphore>();
  auto completion = RuntimeCompletion::from_stream_semaphore(
      Arch::vulkan, 17, 1, semaphore);
  std::atomic<int> released{0};
  completion.attach_resources(std::make_shared<CountedResources>(&released, 1));
  auto first = std::async(std::launch::async, [&] { completion.wait(); });
  auto second = std::async(std::launch::async, [&] { completion.wait(); });
  EXPECT_TRUE(semaphore->await_waiters(2));
  auto teardown = std::async(std::launch::async, [&] { completion.mark_completed(); });
  EXPECT_EQ(teardown.wait_for(std::chrono::milliseconds(50)),
            std::future_status::timeout);
  semaphore->release();
  first.get();
  second.get();
  teardown.get();
  EXPECT_TRUE(completion.done());
  EXPECT_EQ(released.load(), 1);
}

#ifdef TI_TEST_DISPLAY_COMPLETION
TEST(DisplayCompletion, QueryAndStatusDoNotWaitForHostWaiter) {
  auto semaphore = std::make_shared<GatedSemaphore>();
  taichi::ui::vulkan::DisplayCompletion completion;
  completion.attach(semaphore);
  auto waiter = std::async(std::launch::async, [&] { completion.wait(); });
  EXPECT_TRUE(semaphore->await_waiters(1));
  auto query = std::async(std::launch::async, [&] {
    EXPECT_EQ(completion.status(), "submitted");
    return completion.done();
  });
  const bool responsive = query.wait_for(std::chrono::milliseconds(200)) ==
                          std::future_status::ready;
  EXPECT_TRUE(responsive);
  if (responsive)
    EXPECT_FALSE(query.get());
  semaphore->release();
  waiter.get();
  if (!responsive)
    query.get();
  completion.retire();
  EXPECT_TRUE(completion.done());
}

TEST(DisplayCompletion, TeardownDrainsWaitEvenAfterReadyQuery) {
  auto semaphore = std::make_shared<GatedSemaphore>();
  taichi::ui::vulkan::DisplayCompletion completion;
  completion.attach(semaphore);
  auto waiter = std::async(std::launch::async, [&] { completion.wait(); });
  EXPECT_TRUE(semaphore->await_waiters(1));
  semaphore->ready.store(true);
  auto query = std::async(std::launch::async, [&] { return completion.done(); });
  EXPECT_EQ(query.wait_for(std::chrono::milliseconds(200)), std::future_status::ready);
  auto teardown = std::async(std::launch::async, [&] { completion.retire(); });
  EXPECT_EQ(teardown.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
  semaphore->release();
  EXPECT_TRUE(query.get());
  waiter.get();
  teardown.get();
  EXPECT_EQ(completion.status(), "complete");
  completion.wait();
}
#endif

class FatalBackendSemaphore final : public StreamSemaphoreObject {
 public:
  bool is_ready() const override {
    throw BackendRuntimeError(Arch::vulkan, -4, "vkGetFenceStatus",
                              "injected Vulkan device loss");
  }

  bool wait() const override {
    throw BackendRuntimeError(Arch::vulkan, -4, "vkWaitForFences",
                              "injected Vulkan device loss");
  }
};

class FakeGpuTiming final : public StreamGpuTimingObject {
 public:
  explicit FakeGpuTiming(std::atomic<int> *released) : released_(released) {
  }

  ~FakeGpuTiming() override {
    released_->fetch_add(1, std::memory_order_relaxed);
  }

  StreamGpuTimingSnapshot snapshot() const override {
    ++queries;
    StreamGpuTimingSnapshot result;
    result.available = true;
    result.duration_ns = 1234;
    result.exact = true;
    result.measurement_path_changed = true;
    result.stream_id = 7;
    result.status = "instrumented_exact";
    return result;
  }

  mutable int queries{0};

 private:
  std::atomic<int> *released_;
};

TEST(RuntimeCompletion, CompletedTokenUsesNoBackendWork) {
  auto domain = std::make_shared<RuntimeFaultDomain>(Arch::x64, 11);
  auto completion =
      RuntimeCompletion::completed(Arch::x64, 11, 3, domain);
  EXPECT_TRUE(completion.valid());
  EXPECT_TRUE(completion.done());
  completion.wait();
  EXPECT_FALSE(completion.has_backend_work());
  EXPECT_EQ(completion.program_domain(), 11);
  EXPECT_EQ(completion.sequence(), 3);
  const auto statistics = domain->statistics().snapshot();
  EXPECT_EQ(statistics.synchronization.completion_polls, 1u);
  EXPECT_EQ(statistics.synchronization.completion_waits, 1u);
}

TEST(RuntimeCompletion, WaitReleasesResourcesExactlyOnce) {
  auto domain = std::make_shared<RuntimeFaultDomain>(Arch::vulkan, 17);
  auto semaphore = std::make_shared<FakeSemaphore>();
  auto completion = RuntimeCompletion::from_stream_semaphore(
      Arch::vulkan, 17, 9, semaphore, domain);
  std::atomic<int> released{0};
  completion.attach_resources(
      std::make_shared<CountedResources>(&released, 4));

  EXPECT_FALSE(completion.done());
  EXPECT_EQ(completion.retained_resource_count(7), 4);
  completion.wait();
  EXPECT_TRUE(completion.done());
  EXPECT_EQ(completion.retained_resource_count(7), 0);
  EXPECT_EQ(released.load(std::memory_order_relaxed), 1);
  completion.wait();
  EXPECT_EQ(released.load(std::memory_order_relaxed), 1);
  EXPECT_EQ(semaphore->waits, 1);
  const auto statistics = domain->statistics().snapshot();
  EXPECT_EQ(statistics.synchronization.completion_polls, 2u);
  EXPECT_EQ(statistics.synchronization.completion_waits, 2u);
}

TEST(RuntimeCompletion, GpuTimingIsFrozenAndReadableAfterCompletion) {
  auto semaphore = std::make_shared<FakeSemaphore>();
  std::atomic<int> released{0};
  auto timing = std::make_shared<FakeGpuTiming>(&released);
  auto region_timing = std::make_shared<FakeGpuTiming>(&released);
  auto completion = RuntimeCompletion::from_stream_semaphore(
      Arch::vulkan, 17, 10, semaphore, nullptr, timing,
      {{"outer/solve", region_timing}});

  auto pending = completion.gpu_timing_snapshot();
  EXPECT_FALSE(pending.available);
  EXPECT_EQ(pending.status, "pending");
  EXPECT_EQ(timing->queries, 0);
  auto pending_regions = completion.gpu_region_timing_snapshots();
  ASSERT_EQ(pending_regions.size(), 1u);
  EXPECT_EQ(pending_regions[0].path_id, "outer/solve");
  EXPECT_EQ(pending_regions[0].timing.status, "pending");
  EXPECT_EQ(region_timing->queries, 0);

  semaphore->ready = true;
  EXPECT_TRUE(completion.done());
  auto completed = completion.gpu_timing_snapshot();
  EXPECT_TRUE(completed.available);
  EXPECT_EQ(completed.duration_ns, 1234u);
  EXPECT_TRUE(completed.exact);
  EXPECT_TRUE(completed.measurement_path_changed);
  EXPECT_EQ(completed.stream_id, 7u);
  EXPECT_EQ(completed.status, "instrumented_exact");
  EXPECT_EQ(timing->queries, 1);
  auto completed_regions = completion.gpu_region_timing_snapshots();
  ASSERT_EQ(completed_regions.size(), 1u);
  EXPECT_EQ(completed_regions[0].path_id, "outer/solve");
  EXPECT_TRUE(completed_regions[0].timing.available);
  EXPECT_EQ(completed_regions[0].timing.duration_ns, 1234u);
  EXPECT_EQ(region_timing->queries, 1);

  timing.reset();
  region_timing.reset();
  EXPECT_EQ(released.load(std::memory_order_relaxed), 2);
  auto cached = completion.gpu_timing_snapshot();
  EXPECT_TRUE(cached.available);
  EXPECT_EQ(cached.duration_ns, 1234u);
  auto cached_regions = completion.gpu_region_timing_snapshots();
  ASSERT_EQ(cached_regions.size(), 1u);
  EXPECT_EQ(cached_regions[0].timing.duration_ns, 1234u);
  completion = RuntimeCompletion{};
  EXPECT_EQ(released.load(std::memory_order_relaxed), 2);
}

TEST(RuntimeCompletion, FirstBackendErrorIsSticky) {
  auto semaphore = std::make_shared<FakeSemaphore>();
  semaphore->fail = true;
  auto completion = RuntimeCompletion::from_stream_semaphore(
      Arch::vulkan, 23, 5, semaphore);

  EXPECT_THROW(completion.done(), std::runtime_error);
  EXPECT_EQ(completion.first_error_message(), "fake completion failure");
  semaphore->fail = false;
  semaphore->ready = true;
  EXPECT_THROW(completion.done(), std::runtime_error);
  EXPECT_THROW(completion.wait(), std::runtime_error);
  EXPECT_EQ(completion.first_error_message(), "fake completion failure");
}

TEST(RuntimeCompletion, ProgramSyncRetiresFaultedResourcesWithoutHidingError) {
  auto semaphore = std::make_shared<FakeSemaphore>();
  semaphore->fail = true;
  auto completion = RuntimeCompletion::from_stream_semaphore(
      Arch::vulkan, 11, 7, semaphore);
  std::atomic<int> released{0};
  completion.attach_resources(
      std::make_shared<CountedResources>(&released, 1));
  EXPECT_THROW(completion.done(), std::runtime_error);
  EXPECT_EQ(released.load(std::memory_order_relaxed), 0);

  completion.mark_completed();
  EXPECT_EQ(released.load(std::memory_order_relaxed), 1);
  EXPECT_THROW(completion.wait(), std::runtime_error);
  EXPECT_NE(completion.first_error_message().find("fake completion failure"),
            std::string::npos);
}

TEST(RuntimeCompletion, FatalBackendErrorPoisonsOwningDomainWithSequence) {
  auto domain = std::make_shared<RuntimeFaultDomain>(Arch::vulkan, 81);
  auto semaphore = std::make_shared<FatalBackendSemaphore>();
  auto completion = RuntimeCompletion::from_stream_semaphore(
      Arch::vulkan, 81, 19, semaphore, domain);
  std::atomic<int> released{0};
  completion.attach_resources(
      std::make_shared<CountedResources>(&released, 2));

  EXPECT_THROW(completion.done(), BackendRuntimeError);
  const RuntimeFaultSnapshot snapshot = domain->snapshot();
  ASSERT_TRUE(snapshot.first_fault.has_value());
  EXPECT_EQ(snapshot.state, RuntimeLifecycleState::kFaulted);
  EXPECT_EQ(snapshot.first_fault->backend_code, -4);
  EXPECT_EQ(snapshot.first_fault->submission_sequence, 19u);
  EXPECT_EQ(snapshot.first_fault->operation, "vkGetFenceStatus");
  EXPECT_EQ(released.load(std::memory_order_relaxed), 0);

  completion.invalidate_and_release("runtime faulted");
  EXPECT_EQ(released.load(std::memory_order_relaxed), 1);
  EXPECT_THROW(completion.wait(), BackendRuntimeError);
  const auto statistics = domain->statistics().snapshot();
  EXPECT_EQ(statistics.synchronization.completion_polls, 1u);
  EXPECT_EQ(statistics.synchronization.completion_waits, 1u);
}

}  // namespace
}  // namespace taichi::lang
