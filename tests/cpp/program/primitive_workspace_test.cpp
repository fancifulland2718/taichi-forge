#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>

#include "taichi/program/primitive_workspace.h"

namespace taichi::lang {
namespace {

struct FakeWorkspace {
  explicit FakeWorkspace(std::uint64_t identity = 0) : identity(identity) {
  }

  ~FakeWorkspace() noexcept {
    if (destructions) {
      destructions->fetch_add(1, std::memory_order_relaxed);
    }
  }

  std::size_t allocated_bytes() const noexcept {
    return bytes;
  }

  std::uint64_t identity{0};
  std::size_t bytes{0};
  std::shared_ptr<std::atomic<std::uint64_t>> destructions;
};

struct OtherWorkspace {
  std::size_t allocated_bytes() const noexcept {
    return 0;
  }
};

PrimitiveWorkspaceKey cuda_scan_key(std::uint64_t stream = 0,
                                    std::uint64_t variant = 0) {
  return {PrimitiveWorkspaceBackend::cuda, PrimitiveWorkspaceFamily::scan,
          stream, variant};
}

TEST(PrimitiveWorkspaceArena, ReusesExactKeyAndAccountsGrowth) {
  PrimitiveWorkspaceArena arena;
  std::uint64_t first_identity = 0;
  {
    auto lease = arena.acquire<FakeWorkspace>(
        cuda_scan_key(), [] { return std::make_shared<FakeWorkspace>(7); });
    first_identity = lease->identity;
    lease->bytes = 4096;
  }

  auto after_first = arena.snapshot();
  EXPECT_EQ(after_first.entries, 1);
  EXPECT_EQ(after_first.reserved_bytes, 4096);
  EXPECT_EQ(after_first.reclaimable_bytes, 4096);
  EXPECT_EQ(after_first.cache_misses, 1);
  EXPECT_EQ(after_first.cache_hits, 0);
  EXPECT_EQ(after_first.active_leases, 0);

  {
    auto lease = arena.acquire<FakeWorkspace>(
        cuda_scan_key(), [] { return std::make_shared<FakeWorkspace>(99); });
    EXPECT_EQ(lease->identity, first_identity);
    EXPECT_EQ(lease->bytes, 4096);
  }
  const auto after_reuse = arena.snapshot();
  EXPECT_EQ(after_reuse.entries, 1);
  EXPECT_EQ(after_reuse.cache_hits, 1);
  EXPECT_EQ(after_reuse.cache_misses, 1);
  EXPECT_GE(after_reuse.peak_in_use_bytes, 4096);
}

TEST(PrimitiveWorkspaceArena, SerializesOneKeyWithoutGlobalSerialization) {
  PrimitiveWorkspaceArena arena;
  std::atomic<bool> first_has_lease{false};
  std::atomic<bool> release_first{false};
  std::atomic<bool> second_has_lease{false};

  std::thread first([&] {
    auto lease = arena.acquire<FakeWorkspace>(
        cuda_scan_key(1), [] { return std::make_shared<FakeWorkspace>(); });
    first_has_lease.store(true, std::memory_order_release);
    while (!release_first.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  });
  while (!first_has_lease.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  std::thread second([&] {
    auto lease = arena.acquire<FakeWorkspace>(
        cuda_scan_key(1), [] { return std::make_shared<FakeWorkspace>(); });
    second_has_lease.store(true, std::memory_order_release);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  EXPECT_FALSE(second_has_lease.load(std::memory_order_acquire));

  // A different execution domain must not wait for the busy key.
  {
    auto independent = arena.acquire<FakeWorkspace>(
        cuda_scan_key(2), [] { return std::make_shared<FakeWorkspace>(); });
    EXPECT_TRUE(independent);
  }
  release_first.store(true, std::memory_order_release);
  first.join();
  second.join();
  EXPECT_TRUE(second_has_lease.load(std::memory_order_acquire));
  const auto stats = arena.snapshot();
  EXPECT_EQ(stats.entries, 2);
  EXPECT_GE(stats.lock_contentions, 1);
}

TEST(PrimitiveWorkspaceArena, ExplicitTrimHonorsBudgetAndPersistence) {
  PrimitiveWorkspaceArena arena;
  {
    auto lease = arena.acquire<FakeWorkspace>(
        cuda_scan_key(1), [] { return std::make_shared<FakeWorkspace>(); },
        true);
    lease->bytes = 8;
  }
  {
    auto lease = arena.acquire<FakeWorkspace>(
        cuda_scan_key(2), [] { return std::make_shared<FakeWorkspace>(); });
    lease->bytes = 8;
  }
  arena.set_budget_bytes(8);
  EXPECT_EQ(arena.snapshot().over_budget_bytes, 8);
  arena.trim_to_budget();

  const auto trimmed = arena.snapshot();
  EXPECT_EQ(trimmed.entries, 1);
  EXPECT_EQ(trimmed.reserved_bytes, 8);
  EXPECT_EQ(trimmed.persistent_bytes, 8);
  EXPECT_EQ(trimmed.evictions, 1);
  EXPECT_EQ(trimmed.over_budget_bytes, 0);
}

TEST(PrimitiveWorkspaceArena, ClearDestroysOutsideMetadataDomain) {
  PrimitiveWorkspaceArena arena;
  auto destructions = std::make_shared<std::atomic<std::uint64_t>>(0);
  {
    auto lease = arena.acquire<FakeWorkspace>(cuda_scan_key(), [&] {
      auto resource = std::make_shared<FakeWorkspace>();
      resource->destructions = destructions;
      return resource;
    });
    lease->bytes = 32;
  }
  arena.clear();
  const auto cleared = arena.snapshot();
  EXPECT_EQ(cleared.entries, 0);
  EXPECT_EQ(cleared.reserved_bytes, 0);
  EXPECT_EQ(cleared.cleared_entries, 1);
  EXPECT_EQ(destructions->load(std::memory_order_relaxed), 1);

  // Reacquiring the same key after retirement creates a new generation.
  {
    auto lease = arena.acquire<FakeWorkspace>(
        cuda_scan_key(), [] { return std::make_shared<FakeWorkspace>(2); });
    EXPECT_EQ(lease->identity, 2);
  }
}

TEST(PrimitiveWorkspaceArena, RejectsNullFactoryWithoutPublishingEntry) {
  PrimitiveWorkspaceArena arena;
  EXPECT_THROW(
      arena.acquire<FakeWorkspace>(cuda_scan_key(), [] {
        return std::shared_ptr<FakeWorkspace>();
      }),
      std::invalid_argument);

  const auto stats = arena.snapshot();
  EXPECT_EQ(stats.entries, 0);
  EXPECT_EQ(stats.active_leases, 0);
  EXPECT_EQ(stats.cache_misses, 0);
}

TEST(PrimitiveWorkspaceArena, TypeMismatchDoesNotCorruptExistingEntry) {
  PrimitiveWorkspaceArena arena;
  {
    auto lease = arena.acquire<FakeWorkspace>(
        cuda_scan_key(), [] { return std::make_shared<FakeWorkspace>(17); });
    EXPECT_EQ(lease->identity, 17);
  }

  EXPECT_THROW(
      arena.acquire<OtherWorkspace>(
          cuda_scan_key(), [] { return std::make_shared<OtherWorkspace>(); }),
      std::logic_error);

  {
    auto lease = arena.acquire<FakeWorkspace>(
        cuda_scan_key(), [] { return std::make_shared<FakeWorkspace>(99); });
    EXPECT_EQ(lease->identity, 17);
  }
  const auto stats = arena.snapshot();
  EXPECT_EQ(stats.entries, 1);
  EXPECT_EQ(stats.active_leases, 0);
}

TEST(PrimitiveWorkspaceArena, ClearWaitsForAnActiveLease) {
  PrimitiveWorkspaceArena arena;
  std::atomic<bool> lease_acquired{false};
  std::atomic<bool> release_lease{false};
  std::atomic<bool> clear_started{false};
  std::atomic<bool> clear_finished{false};

  std::thread user([&] {
    auto lease = arena.acquire<FakeWorkspace>(
        cuda_scan_key(), [] { return std::make_shared<FakeWorkspace>(); });
    lease_acquired.store(true, std::memory_order_release);
    while (!release_lease.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  });
  while (!lease_acquired.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  std::thread clearer([&] {
    clear_started.store(true, std::memory_order_release);
    arena.clear();
    clear_finished.store(true, std::memory_order_release);
  });
  while (!clear_started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  EXPECT_FALSE(clear_finished.load(std::memory_order_acquire));

  release_lease.store(true, std::memory_order_release);
  user.join();
  clearer.join();
  EXPECT_TRUE(clear_finished.load(std::memory_order_acquire));
  EXPECT_EQ(arena.snapshot().entries, 0);
}

}  // namespace
}  // namespace taichi::lang
