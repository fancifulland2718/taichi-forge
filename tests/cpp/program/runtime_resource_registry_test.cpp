#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <taichi/program/runtime_resource_registry.h>

namespace taichi::lang {
namespace {

struct ResourceTracker {
  std::atomic<std::uint64_t> destructions{0};
  std::mutex mutex;
  std::vector<int> finalized;
};

struct FakeResource {
  FakeResource(int id,
               std::shared_ptr<ResourceTracker> tracker,
               bool fail_finalizer = false,
               std::function<void()> on_destroy = {})
      : id(id),
        tracker(std::move(tracker)),
        fail_finalizer(fail_finalizer),
        on_destroy(std::move(on_destroy)) {
  }

  ~FakeResource() noexcept {
    if (on_destroy) {
      on_destroy();
    }
    if (tracker) {
      tracker->destructions.fetch_add(1, std::memory_order_relaxed);
    }
  }

  int id;
  std::shared_ptr<ResourceTracker> tracker;
  bool fail_finalizer{false};
  std::function<void()> on_destroy;
  std::atomic<std::uint64_t> value{0};
};

struct ConstructorResource {
  enum class Failure {
    kNone,
    kBadAlloc,
    kOther,
  };

  explicit ConstructorResource(Failure failure) {
    if (failure == Failure::kBadAlloc) {
      throw std::bad_alloc();
    }
    if (failure == Failure::kOther) {
      throw std::runtime_error(std::string{});
    }
  }

  ~ConstructorResource() noexcept = default;
};

struct ChurnResource {
  explicit ChurnResource(std::uint64_t value) : value(value) {
  }
  ~ChurnResource() noexcept = default;

  std::uint64_t value;
};

TEST(RuntimeResourceRegistryTest,
     GenerationDomainAndKindRejectStaleHandles) {
  using Registry = RuntimeResourceRegistry<FakeResource>;
  auto tracker = std::make_shared<ResourceTracker>();
  Registry registry(101);
  Registry invalid_domain(0);

  EXPECT_EQ(invalid_domain.emplace(1, 6, tracker).first,
            Registry::Result::kInvalidArgument);
  EXPECT_EQ(registry.emplace(0, 6, tracker).first,
            Registry::Result::kInvalidArgument);
  EXPECT_EQ(tracker->destructions.load(std::memory_order_relaxed), 0u);

  auto [create_result, first] = registry.emplace(1, 7, tracker);
  ASSERT_EQ(create_result, Registry::Result::kSuccess);
  ASSERT_TRUE(first);

  auto [acquire_result, lease] = registry.acquire(first);
  ASSERT_EQ(acquire_result, Registry::Result::kSuccess);
  ASSERT_TRUE(lease);
  lease->value.store(42, std::memory_order_relaxed);

  auto wrong_domain = first;
  ++wrong_domain.domain;
  EXPECT_EQ(registry.acquire(wrong_domain).first,
            Registry::Result::kInvalidHandle);
  auto wrong_kind = first;
  ++wrong_kind.kind;
  EXPECT_EQ(registry.acquire(wrong_kind).first,
            Registry::Result::kInvalidHandle);

  ASSERT_EQ(registry.retire(first), Registry::Result::kSuccess);
  EXPECT_EQ(registry.state(first), Registry::State::kRetiring);
  EXPECT_EQ(registry.acquire(first).first, Registry::Result::kInvalidHandle);
  EXPECT_EQ(lease->value.load(std::memory_order_relaxed), 42u);
  EXPECT_EQ(tracker->destructions.load(std::memory_order_relaxed), 0u);

  lease.reset();
  EXPECT_EQ(registry.state(first), Registry::State::kReleased);
  EXPECT_EQ(tracker->destructions.load(std::memory_order_relaxed), 1u);

  auto [second_result, second] = registry.emplace(1, 8, tracker);
  ASSERT_EQ(second_result, Registry::Result::kSuccess);
  EXPECT_EQ(second.index, first.index);
  EXPECT_NE(second.generation, first.generation);
  EXPECT_FALSE(registry.state(first).has_value());
  EXPECT_EQ(registry.acquire(first).first, Registry::Result::kInvalidHandle);
  EXPECT_EQ(registry.acquire(second).first, Registry::Result::kSuccess);
}

TEST(RuntimeResourceRegistryTest, ResourceDestructorRunsOutsideRegistryLock) {
  using Registry = RuntimeResourceRegistry<FakeResource>;
  auto tracker = std::make_shared<ResourceTracker>();
  std::atomic<int> reentrant_calls{0};
  Registry registry(102);

  auto [result, handle] = registry.emplace(
      1, 1, tracker, false, [&] {
        const auto stats = registry.stats();
        if (stats.released == 1) {
          reentrant_calls.fetch_add(1, std::memory_order_relaxed);
        }
      });
  ASSERT_EQ(result, Registry::Result::kSuccess);
  EXPECT_EQ(registry.retire(handle), Registry::Result::kSuccess);
  EXPECT_EQ(reentrant_calls.load(std::memory_order_relaxed), 1);
  EXPECT_EQ(tracker->destructions.load(std::memory_order_relaxed), 1u);
}

TEST(RuntimeResourceRegistryTest, LeaseCloneSharesDelayedOwnership) {
  using Registry = RuntimeResourceRegistry<FakeResource>;
  auto tracker = std::make_shared<ResourceTracker>();
  Registry registry(110);

  auto [create_result, handle] = registry.emplace(1, 17, tracker);
  ASSERT_EQ(create_result, Registry::Result::kSuccess);
  auto [acquire_result, owner] = registry.acquire(handle);
  ASSERT_EQ(acquire_result, Registry::Result::kSuccess);
  auto clone = owner.clone();
  ASSERT_TRUE(clone);
  EXPECT_EQ(registry.stats().leases, 2u);

  ASSERT_EQ(registry.retire(handle), Registry::Result::kSuccess);
  EXPECT_EQ(registry.acquire(handle).first, Registry::Result::kInvalidHandle);
  auto retiring_clone = owner.clone();
  ASSERT_TRUE(retiring_clone);
  EXPECT_EQ(registry.stats().leases, 3u);

  owner.reset();
  clone.reset();
  EXPECT_EQ(tracker->destructions.load(std::memory_order_relaxed), 0u);
  retiring_clone.reset();
  EXPECT_EQ(tracker->destructions.load(std::memory_order_relaxed), 1u);
  EXPECT_FALSE(owner.clone());
  EXPECT_EQ(registry.state(handle), Registry::State::kReleased);
}

TEST(RuntimeResourceRegistryTest,
     FinalizerExceptionDoesNotCorruptRegistryState) {
  using Finalizer = std::function<void(FakeResource &)>;
  using Registry = RuntimeResourceRegistry<FakeResource, Finalizer>;
  auto tracker = std::make_shared<ResourceTracker>();
  Registry *registry_ptr = nullptr;
  Finalizer finalizer = [&](FakeResource &resource) {
    const auto stats = registry_ptr->stats();
    EXPECT_EQ(stats.live, 0u);
    EXPECT_EQ(stats.released, 1u);
    {
      std::lock_guard<std::mutex> lock(tracker->mutex);
      tracker->finalized.push_back(resource.id);
    }
    if (resource.fail_finalizer) {
      throw std::runtime_error(std::string{});
    }
  };
  Registry registry(103, std::move(finalizer));
  registry_ptr = &registry;

  auto [first_result, first] = registry.emplace(1, 11, tracker, true);
  ASSERT_EQ(first_result, Registry::Result::kSuccess);
  EXPECT_EQ(registry.retire(first), Registry::Result::kSuccess);
  auto stats = registry.stats();
  EXPECT_EQ(stats.released, 1u);
  EXPECT_EQ(stats.release_errors, 1u);
  EXPECT_EQ(tracker->destructions.load(std::memory_order_relaxed), 1u);

  auto [second_result, second] = registry.emplace(1, 12, tracker, false);
  ASSERT_EQ(second_result, Registry::Result::kSuccess);
  EXPECT_EQ(registry.retire(second), Registry::Result::kSuccess);
  stats = registry.stats();
  EXPECT_EQ(stats.released_total, 2u);
  EXPECT_EQ(stats.release_errors, 1u);
  EXPECT_EQ(tracker->destructions.load(std::memory_order_relaxed), 2u);
}

TEST(RuntimeResourceRegistryTest,
     ConstructorFailureAndOutOfMemoryLeaveRegistryUsable) {
  using Registry = RuntimeResourceRegistry<ConstructorResource>;
  Registry registry(104);

  EXPECT_EQ(registry.emplace(1, ConstructorResource::Failure::kBadAlloc).first,
            Registry::Result::kOutOfMemory);
  EXPECT_EQ(registry.emplace(1, ConstructorResource::Failure::kOther).first,
            Registry::Result::kResourceError);
  auto stats = registry.stats();
  EXPECT_EQ(stats.slots, 0u);
  EXPECT_EQ(stats.created_total, 0u);

  auto [result, handle] =
      registry.emplace(1, ConstructorResource::Failure::kNone);
  ASSERT_EQ(result, Registry::Result::kSuccess);
  EXPECT_EQ(registry.retire(handle), Registry::Result::kSuccess);
  stats = registry.stats();
  EXPECT_EQ(stats.slots, 1u);
  EXPECT_EQ(stats.released_total, 1u);
}

TEST(RuntimeResourceRegistryTest, FinalizeUsesExplicitKindDependencyOrder) {
  using Finalizer = std::function<void(FakeResource &)>;
  using Registry = RuntimeResourceRegistry<FakeResource, Finalizer>;
  auto tracker = std::make_shared<ResourceTracker>();
  Finalizer finalizer = [tracker](FakeResource &resource) {
    std::lock_guard<std::mutex> lock(tracker->mutex);
    tracker->finalized.push_back(resource.id);
  };
  Registry registry(105, std::move(finalizer));

  ASSERT_EQ(registry.emplace(1, 10, tracker).first,
            Registry::Result::kSuccess);
  ASSERT_EQ(registry.emplace(4, 40, tracker).first,
            Registry::Result::kSuccess);
  ASSERT_EQ(registry.emplace(2, 20, tracker).first,
            Registry::Result::kSuccess);
  ASSERT_EQ(registry.emplace(3, 30, tracker).first,
            Registry::Result::kSuccess);

  registry.finalize({3, 2, 1});
  const std::vector<int> expected{30, 20, 10, 40};
  EXPECT_EQ(tracker->finalized, expected);
  const auto stats = registry.stats();
  EXPECT_TRUE(stats.closed);
  EXPECT_EQ(stats.live, 0u);
  EXPECT_EQ(stats.retiring, 0u);
  EXPECT_EQ(stats.released, 4u);
  EXPECT_EQ(registry.emplace(1, 50, tracker).first,
            Registry::Result::kClosed);
  EXPECT_EQ(tracker->destructions.load(std::memory_order_relaxed), 4u);
}

TEST(RuntimeResourceRegistryTest, LeaseCanOutliveRegistryObject) {
  using Registry = RuntimeResourceRegistry<FakeResource>;
  auto tracker = std::make_shared<ResourceTracker>();
  Registry::Lease lease;

  {
    auto registry = std::make_unique<Registry>(106);
    auto [create_result, handle] = registry->emplace(1, 1, tracker);
    ASSERT_EQ(create_result, Registry::Result::kSuccess);
    auto [acquire_result, acquired] = registry->acquire(handle);
    ASSERT_EQ(acquire_result, Registry::Result::kSuccess);
    lease = std::move(acquired);
    registry->finalize();
    EXPECT_EQ(registry->stats().retiring, 1u);
    registry.reset();
  }

  EXPECT_TRUE(lease);
  EXPECT_EQ(tracker->destructions.load(std::memory_order_relaxed), 0u);
  lease.reset();
  EXPECT_EQ(tracker->destructions.load(std::memory_order_relaxed), 1u);
}

TEST(RuntimeResourceRegistryTest, ConcurrentAcquireAndRetireIsRaceFree) {
  using Registry = RuntimeResourceRegistry<FakeResource>;
  auto tracker = std::make_shared<ResourceTracker>();
  Registry registry(107);
  auto [create_result, handle] = registry.emplace(1, 1, tracker);
  ASSERT_EQ(create_result, Registry::Result::kSuccess);

  constexpr int kThreads = 4;
  std::atomic<bool> start{false};
  std::atomic<std::uint64_t> acquisitions{0};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (;;) {
        auto [result, lease] = registry.acquire(handle);
        if (result != Registry::Result::kSuccess) {
          break;
        }
        lease->value.fetch_add(1, std::memory_order_relaxed);
        acquisitions.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  start.store(true, std::memory_order_release);
  while (acquisitions.load(std::memory_order_relaxed) < 1000) {
    std::this_thread::yield();
  }
  EXPECT_EQ(registry.retire(handle), Registry::Result::kSuccess);
  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_GE(acquisitions.load(std::memory_order_relaxed), 1000u);
  EXPECT_EQ(tracker->destructions.load(std::memory_order_relaxed), 1u);
  const auto stats = registry.stats();
  EXPECT_EQ(stats.live, 0u);
  EXPECT_EQ(stats.retiring, 0u);
  EXPECT_EQ(stats.released, 1u);
  EXPECT_EQ(stats.leases, 0u);
}

TEST(RuntimeResourceRegistryTest, ConcurrentAcquireAndFinalizeIsRaceFree) {
  using Registry = RuntimeResourceRegistry<FakeResource>;
  auto tracker = std::make_shared<ResourceTracker>();
  Registry registry(109);
  auto [create_result, handle] = registry.emplace(1, 1, tracker);
  ASSERT_EQ(create_result, Registry::Result::kSuccess);

  constexpr int kThreads = 4;
  std::atomic<bool> start{false};
  std::atomic<std::uint64_t> acquisitions{0};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (;;) {
        auto [result, lease] = registry.acquire(handle);
        if (result != Registry::Result::kSuccess) {
          break;
        }
        lease->value.fetch_add(1, std::memory_order_relaxed);
        acquisitions.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  start.store(true, std::memory_order_release);
  while (acquisitions.load(std::memory_order_relaxed) < 1000) {
    std::this_thread::yield();
  }
  registry.finalize({1});
  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_GE(acquisitions.load(std::memory_order_relaxed), 1000u);
  EXPECT_EQ(tracker->destructions.load(std::memory_order_relaxed), 1u);
  const auto stats = registry.stats();
  EXPECT_TRUE(stats.closed);
  EXPECT_EQ(stats.live, 0u);
  EXPECT_EQ(stats.retiring, 0u);
  EXPECT_EQ(stats.released, 1u);
  EXPECT_EQ(stats.leases, 0u);
}

TEST(RuntimeResourceRegistryTest, MillionEntryChurnReusesMetadataSlot) {
  using Registry = RuntimeResourceRegistry<ChurnResource>;
  constexpr std::uint64_t kIterations = 1000000;
  Registry registry(108);
  Registry::Handle first;
  Registry::Handle last;
  bool succeeded = true;

  for (std::uint64_t i = 0; i < kIterations; ++i) {
    auto [create_result, handle] = registry.emplace(1, i);
    if (create_result != Registry::Result::kSuccess) {
      succeeded = false;
      break;
    }
    if (i == 0) {
      first = handle;
    }
    last = handle;
    if (registry.retire(handle) != Registry::Result::kSuccess) {
      succeeded = false;
      break;
    }
  }

  ASSERT_TRUE(succeeded);
  const auto stats = registry.stats();
  EXPECT_EQ(stats.slots, 1u);
  EXPECT_EQ(stats.live, 0u);
  EXPECT_EQ(stats.retiring, 0u);
  EXPECT_EQ(stats.released, 1u);
  EXPECT_EQ(stats.created_total, kIterations);
  EXPECT_EQ(stats.retired_total, kIterations);
  EXPECT_EQ(stats.released_total, kIterations);
  EXPECT_NE(first.generation, last.generation);
  EXPECT_EQ(registry.acquire(first).first, Registry::Result::kInvalidHandle);
  EXPECT_EQ(registry.state(last), Registry::State::kReleased);
}

}  // namespace
}  // namespace taichi::lang
