#include <gtest/gtest.h>

#include "taichi/rhi/common/allocation_registry.h"

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

namespace taichi::lang {
namespace {

struct FakeRecord {
  explicit FakeRecord(uint64_t size) : size(size) {
  }

  uint64_t size;
  uint64_t payload{0};
};

struct TrackedRecord {
  TrackedRecord(uint64_t size, std::shared_ptr<std::atomic<int>> destructions)
      : size(size), destructions(std::move(destructions)) {
  }

  ~TrackedRecord() {
    destructions->fetch_add(1, std::memory_order_relaxed);
  }

  uint64_t size;
  std::shared_ptr<std::atomic<int>> destructions;
};

TEST(AllocationRegistryTest, ValidatesGenerationAndOverflowSafeRanges) {
  AllocationRegistry<FakeRecord> registry;
  auto [create_result, handle] = registry.emplace(16);
  ASSERT_EQ(create_result, RhiResult::success);
  ASSERT_NE(handle, 0u);

  auto [range_result, lease] = registry.acquire(handle, 8, 8);
  ASSERT_EQ(range_result, RhiResult::success);
  ASSERT_TRUE(lease);
  EXPECT_EQ(lease->size, 16u);
  EXPECT_EQ(registry.acquire(handle, 17, 0).first, RhiResult::invalid_usage);
  EXPECT_EQ(registry.acquire(handle, 8, 9).first, RhiResult::invalid_usage);
  EXPECT_EQ(registry.acquire(handle, std::numeric_limits<uint64_t>::max(), 1)
                .first,
            RhiResult::invalid_usage);
  EXPECT_EQ(registry.acquire(handle ^ (uint64_t{1} << 32)).first,
            RhiResult::invalid_usage);
}

TEST(AllocationRegistryTest, RetiredRecordsWaitForLeasesAndReuseGeneration) {
  AllocationRegistry<TrackedRecord> registry;
  auto destructions = std::make_shared<std::atomic<int>>(0);
  auto [create_result, first] = registry.emplace(8, destructions);
  ASSERT_EQ(create_result, RhiResult::success);
  auto [lease_result, lease] = registry.acquire(first);
  ASSERT_EQ(lease_result, RhiResult::success);

  EXPECT_EQ(registry.retire(first), RhiResult::success);
  EXPECT_EQ(registry.state(first),
            AllocationRegistry<TrackedRecord>::State::kRetiring);
  EXPECT_EQ(registry.acquire(first).first, RhiResult::invalid_usage);
  EXPECT_EQ(registry.collect_retired(), 0u);
  EXPECT_EQ(destructions->load(std::memory_order_relaxed), 0);

  lease = {};
  EXPECT_EQ(registry.collect_retired(), 1u);
  EXPECT_EQ(destructions->load(std::memory_order_relaxed), 1);
  EXPECT_EQ(registry.state(first),
            AllocationRegistry<TrackedRecord>::State::kReleased);

  auto [second_result, second] = registry.emplace(8, destructions);
  ASSERT_EQ(second_result, RhiResult::success);
  EXPECT_NE(second, first);
  EXPECT_EQ(registry.acquire(first).first, RhiResult::invalid_usage);
  EXPECT_EQ(registry.acquire(second).first, RhiResult::success);
}

TEST(AllocationRegistryTest, ConcurrentAllocateLeaseAndRetireIsRaceFree) {
  AllocationRegistry<FakeRecord> registry;
  constexpr int kThreads = 4;
  constexpr int kIterations = 256;
  std::atomic<bool> succeeded{true};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (int thread_index = 0; thread_index < kThreads; ++thread_index) {
    threads.emplace_back([&] {
      for (int iteration = 0; iteration < kIterations; ++iteration) {
        auto [create_result, handle] = registry.emplace(64);
        if (create_result != RhiResult::success) {
          succeeded.store(false, std::memory_order_relaxed);
          return;
        }
        auto [lease_result, lease] = registry.acquire(handle, 16, 32);
        if (lease_result != RhiResult::success || !lease) {
          succeeded.store(false, std::memory_order_relaxed);
          return;
        }
        lease->payload = static_cast<uint64_t>(iteration);
        if (registry.retire(handle) != RhiResult::success) {
          succeeded.store(false, std::memory_order_relaxed);
          return;
        }
        lease = {};
        registry.collect_retired();
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_TRUE(succeeded.load(std::memory_order_relaxed));
  registry.collect_retired();
  const auto stats = registry.stats();
  EXPECT_EQ(stats.live, 0u);
  EXPECT_EQ(stats.retiring, 0u);
}

}  // namespace
}  // namespace taichi::lang
