#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "taichi/rhi/common/host_memory_pool.h"

namespace taichi::lang {

class HostMemoryPoolTestHelper {
 public:
  struct ChunkSnapshot {
    bool is_exclusive;
    std::uintptr_t data;
    std::uintptr_t allocation;
    std::uintptr_t head;
    std::uintptr_t tail;
  };

  static void setDefaultAllocatorSize(std::size_t size) {
    UnifiedAllocator::default_allocator_size = size;
  }
  static size_t getDefaultAllocatorSize() {
    return UnifiedAllocator::default_allocator_size;
  }
  static void setInitialAllocatorSize(std::size_t size) {
    UnifiedAllocator::initial_allocator_size = size;
  }
  static size_t getInitialAllocatorSize() {
    return UnifiedAllocator::initial_allocator_size;
  }
  static std::unique_ptr<HostMemoryPool> createPool(
      bool adaptive_chunk_policy) {
    return std::unique_ptr<HostMemoryPool>(
        new HostMemoryPool(adaptive_chunk_policy));
  }
  static std::vector<std::pair<std::uintptr_t, std::size_t>> getRawChunks(
      HostMemoryPool &pool) {
    std::lock_guard<std::mutex> lock(pool.mut_allocation_);
    std::vector<std::pair<std::uintptr_t, std::size_t>> result;
    for (const auto &[ptr, size] : pool.raw_memory_chunks_) {
      result.emplace_back(reinterpret_cast<std::uintptr_t>(ptr), size);
    }
    return result;
  }
  static std::vector<ChunkSnapshot> getUnifiedChunks(HostMemoryPool &pool) {
    std::lock_guard<std::mutex> lock(pool.mut_allocation_);
    std::vector<ChunkSnapshot> result;
    for (const auto &chunk : pool.allocator_->chunks_) {
      result.push_back(
          {chunk.is_exclusive, reinterpret_cast<std::uintptr_t>(chunk.data),
           reinterpret_cast<std::uintptr_t>(chunk.allocation),
           reinterpret_cast<std::uintptr_t>(chunk.head),
           reinterpret_cast<std::uintptr_t>(chunk.tail)});
    }
    return result;
  }
  static bool checkedAddSize(std::size_t lhs,
                             std::size_t rhs,
                             std::size_t *result) {
    return UnifiedAllocator::checked_add_size(lhs, rhs, result);
  }
  static bool checkedAddAddress(std::uintptr_t lhs,
                                std::size_t rhs,
                                std::uintptr_t *result) {
    return UnifiedAllocator::checked_add_address(lhs, rhs, result);
  }
  static bool alignUp(std::uintptr_t value,
                      std::size_t alignment,
                      std::uintptr_t *result) {
    return UnifiedAllocator::align_up(value, alignment, result);
  }
};

class DefaultAllocatorSizeGuard {
 public:
  explicit DefaultAllocatorSizeGuard(std::size_t size)
      : old_size_(HostMemoryPoolTestHelper::getDefaultAllocatorSize()) {
    HostMemoryPoolTestHelper::setDefaultAllocatorSize(size);
  }
  ~DefaultAllocatorSizeGuard() {
    HostMemoryPoolTestHelper::setDefaultAllocatorSize(old_size_);
  }

 private:
  std::size_t old_size_;
};

class AdaptiveAllocatorSizeGuard {
 public:
  AdaptiveAllocatorSizeGuard(std::size_t initial_size,
                             std::size_t maximum_size)
      : old_initial_(
            HostMemoryPoolTestHelper::getInitialAllocatorSize()),
        old_maximum_(HostMemoryPoolTestHelper::getDefaultAllocatorSize()) {
    HostMemoryPoolTestHelper::setInitialAllocatorSize(initial_size);
    HostMemoryPoolTestHelper::setDefaultAllocatorSize(maximum_size);
  }
  ~AdaptiveAllocatorSizeGuard() {
    HostMemoryPoolTestHelper::setInitialAllocatorSize(old_initial_);
    HostMemoryPoolTestHelper::setDefaultAllocatorSize(old_maximum_);
  }

 private:
  std::size_t old_initial_;
  std::size_t old_maximum_;
};

void expectAllocationInsideRawChunk(HostMemoryPool &pool,
                                    void *ptr,
                                    std::size_t size) {
  const auto address = reinterpret_cast<std::uintptr_t>(ptr);
  bool found = false;
  for (const auto &[base, chunk_size] :
       HostMemoryPoolTestHelper::getRawChunks(pool)) {
    const auto tail = base + chunk_size;
    if (address >= base && address <= tail && size <= tail - address) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found) << "allocation [" << address << ", " << address + size
                     << ") is outside every raw chunk";
  if (found && size > 0) {
    auto *bytes = static_cast<volatile std::uint8_t *>(ptr);
    bytes[0] = 0xa5;
    bytes[size - 1] = 0x5a;
  }
}

void expectUnifiedChunksConsistent(HostMemoryPool &pool) {
  for (const auto &chunk :
       HostMemoryPoolTestHelper::getUnifiedChunks(pool)) {
    EXPECT_LE(chunk.data, chunk.allocation);
    EXPECT_LE(chunk.allocation, chunk.head);
    EXPECT_LE(chunk.head, chunk.tail);
  }
}

TEST(HostMemoryPool, AllocateMemory) {
  DefaultAllocatorSizeGuard allocator_size(102400);  // 100 KiB

  HostMemoryPool pool;

  void *ptr1 = pool.allocate(1024, 16);
  void *ptr2 = pool.allocate(1024, 16);
  void *ptr3 = pool.allocate(1024, 16);

  EXPECT_NE(ptr1, ptr2);
  EXPECT_NE(ptr1, ptr3);
  EXPECT_NE(ptr2, ptr3);

  EXPECT_EQ((std::size_t)ptr2, (std::size_t)ptr1 + 1024);
  EXPECT_EQ((std::size_t)ptr3, (std::size_t)ptr2 + 1024);
  EXPECT_EQ(pool.get_stats().raw_chunks, 1);
  expectAllocationInsideRawChunk(pool, ptr1, 1024);
  expectAllocationInsideRawChunk(pool, ptr2, 1024);
  expectAllocationInsideRawChunk(pool, ptr3, 1024);
  expectUnifiedChunksConsistent(pool);
}

TEST(HostMemoryPool, ExactFillCreatesSecondChunk) {
  DefaultAllocatorSizeGuard allocator_size(64);
  HostMemoryPool pool;

  void *exact = pool.allocate(64, 1);
  EXPECT_EQ(pool.get_stats().raw_chunks, 1);
  expectAllocationInsideRawChunk(pool, exact, 64);

  void *next = pool.allocate(1, 1);
  const auto stats = pool.get_stats();
  EXPECT_EQ(stats.raw_chunks, 2);
  EXPECT_EQ(stats.raw_bytes, 128);
  expectAllocationInsideRawChunk(pool, next, 1);
  expectUnifiedChunksConsistent(pool);
}

TEST(HostMemoryPool, AlignmentPaddingCannotCrossChunkTail) {
  DefaultAllocatorSizeGuard allocator_size(64);
  HostMemoryPool pool;

  void *first = pool.allocate(63, 1);
  void *aligned = pool.allocate(1, 2);

  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(aligned) % 2, 0);
  EXPECT_EQ(pool.get_stats().raw_chunks, 2);
  expectAllocationInsideRawChunk(pool, first, 63);
  expectAllocationInsideRawChunk(pool, aligned, 1);
  expectUnifiedChunksConsistent(pool);
}

TEST(HostMemoryPool, RequestLargerThanDefaultStaysInsideMapping) {
  DefaultAllocatorSizeGuard allocator_size(64);
  HostMemoryPool pool;

  void *large = pool.allocate(65, 16);
  EXPECT_EQ(pool.get_stats().raw_bytes, 65);
  expectAllocationInsideRawChunk(pool, large, 65);

  void *next = pool.allocate(1, 1);
  const auto stats = pool.get_stats();
  EXPECT_EQ(stats.raw_chunks, 2);
  EXPECT_EQ(stats.raw_bytes, 129);
  expectAllocationInsideRawChunk(pool, next, 1);
  expectUnifiedChunksConsistent(pool);
}

TEST(HostMemoryPool, ExclusiveAlignedAllocationReleasesRawMapping) {
  DefaultAllocatorSizeGuard allocator_size(64);
  HostMemoryPool pool;
  const std::size_t alignment = HostMemoryPool::page_size * 2;

  void *ptr = pool.allocate(17, alignment, true);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % alignment, 0);
  EXPECT_EQ(pool.get_stats().raw_chunks, 1);
  expectAllocationInsideRawChunk(pool, ptr, 17);
  expectUnifiedChunksConsistent(pool);

  pool.release(17, ptr);
  const auto stats = pool.get_stats();
  EXPECT_EQ(stats.raw_chunks, 0);
  EXPECT_EQ(stats.raw_bytes, 0);
  EXPECT_EQ(stats.unified_chunks, 0);
}

TEST(HostMemoryPool, NonExclusiveZeroSizeKeepsExistingContract) {
  DefaultAllocatorSizeGuard allocator_size(64);
  HostMemoryPool pool;

  void *zero1 = pool.allocate(0, 16);
  void *zero2 = pool.allocate(0, 16);
  void *first_byte = pool.allocate(1, 16);

  EXPECT_EQ(zero1, zero2);
  EXPECT_EQ(zero1, first_byte);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(zero1) % 16, 0);
  EXPECT_EQ(pool.get_stats().raw_chunks, 1);
  expectAllocationInsideRawChunk(pool, zero1, 0);
  expectAllocationInsideRawChunk(pool, first_byte, 1);
  expectUnifiedChunksConsistent(pool);
}

TEST(HostMemoryPool, AllocatorTelemetrySeparatesCapacityAndWaste) {
  DefaultAllocatorSizeGuard allocator_size(64);
  HostMemoryPool pool;

  void *first = pool.allocate(3, 1);
  void *aligned = pool.allocate(1, 8);
  auto stats = pool.get_stats();

  EXPECT_EQ(stats.requested_live_bytes, 4);
  EXPECT_EQ(stats.reserved_bytes, 64);
  EXPECT_EQ(stats.capacity_bytes, 64);
  EXPECT_EQ(stats.used_bytes, 9);
  EXPECT_EQ(stats.available_bytes, 55);
  EXPECT_EQ(stats.alignment_waste_bytes, 5);
  EXPECT_EQ(stats.unreclaimed_released_bytes, 0);
  EXPECT_EQ(stats.wasted_bytes, 5);
  EXPECT_EQ(stats.slab_chunks, 1);
  EXPECT_EQ(stats.large_chunks, 0);
  EXPECT_EQ(stats.exclusive_chunks, 0);
  EXPECT_EQ(stats.peak_requested_live_bytes, 4);
  EXPECT_EQ(stats.peak_reserved_bytes, 64);
  EXPECT_EQ(stats.peak_used_bytes, 9);
  EXPECT_EQ(stats.peak_wasted_bytes, 5);
  EXPECT_EQ(stats.peak_chunks, 1);
  EXPECT_EQ(stats.raw_bytes, stats.reserved_bytes);
  EXPECT_EQ(stats.unified_chunks, stats.slab_chunks + stats.large_chunks +
                                      stats.exclusive_chunks);
#if defined(TI_PLATFORM_UNIX)
  EXPECT_FALSE(stats.committed_bytes_available);
#else
  EXPECT_TRUE(stats.committed_bytes_available);
  EXPECT_EQ(stats.committed_bytes, stats.reserved_bytes);
#endif

  pool.release(3, first);
  stats = pool.get_stats();
  EXPECT_EQ(stats.requested_live_bytes, 1);
  EXPECT_EQ(stats.used_bytes, 9);
  EXPECT_EQ(stats.unreclaimed_released_bytes, 3);
  EXPECT_EQ(stats.wasted_bytes, 8);
  EXPECT_EQ(stats.peak_wasted_bytes, 8);

  pool.release(1, aligned);
  stats = pool.get_stats();
  EXPECT_EQ(stats.requested_live_bytes, 0);
  EXPECT_EQ(stats.unreclaimed_released_bytes, 4);
  EXPECT_EQ(stats.wasted_bytes, 9);
  EXPECT_EQ(stats.used_bytes, stats.wasted_bytes);
}

TEST(HostMemoryPool, TelemetryClassifiesLargeAndExclusiveChunks) {
  DefaultAllocatorSizeGuard allocator_size(64);
  HostMemoryPool pool;

  pool.allocate(65, 1);
  auto stats = pool.get_stats();
  EXPECT_EQ(stats.slab_chunks, 0);
  EXPECT_EQ(stats.large_chunks, 1);
  EXPECT_EQ(stats.exclusive_chunks, 0);
  EXPECT_EQ(stats.capacity_bytes, 65);
  EXPECT_EQ(stats.used_bytes, 65);

  void *exclusive =
      pool.allocate(17, HostMemoryPool::page_size * 2, true);
  stats = pool.get_stats();
  EXPECT_EQ(stats.large_chunks, 1);
  EXPECT_EQ(stats.exclusive_chunks, 1);
  EXPECT_EQ(stats.requested_live_bytes, 82);
  const auto peak_reserved = stats.peak_reserved_bytes;
  const auto peak_used = stats.peak_used_bytes;
  const auto peak_chunks = stats.peak_chunks;

  pool.release(17, exclusive);
  stats = pool.get_stats();
  EXPECT_EQ(stats.large_chunks, 1);
  EXPECT_EQ(stats.exclusive_chunks, 0);
  EXPECT_EQ(stats.requested_live_bytes, 65);
  EXPECT_LT(stats.reserved_bytes, peak_reserved);
  EXPECT_LT(stats.used_bytes, peak_used);
  EXPECT_EQ(stats.peak_reserved_bytes, peak_reserved);
  EXPECT_EQ(stats.peak_used_bytes, peak_used);
  EXPECT_EQ(stats.peak_chunks, peak_chunks);

  pool.reset();
  stats = pool.get_stats();
  EXPECT_EQ(stats.reserved_bytes, 0);
  EXPECT_EQ(stats.capacity_bytes, 0);
  EXPECT_EQ(stats.used_bytes, 0);
  EXPECT_EQ(stats.unified_chunks, 0);
  EXPECT_EQ(stats.peak_reserved_bytes, peak_reserved);
  EXPECT_EQ(stats.peak_used_bytes, peak_used);
  EXPECT_EQ(stats.peak_chunks, peak_chunks);
}

TEST(HostMemoryPool, AdaptiveSlabsGrowGeometricallyAndStopAtMaximum) {
  AdaptiveAllocatorSizeGuard allocator_size(64, 256);
  auto pool = HostMemoryPoolTestHelper::createPool(true);

  pool->allocate(64, 1);
  pool->allocate(128, 1);
  pool->allocate(256, 1);
  pool->allocate(256, 1);

  const auto stats = pool->get_stats();
  EXPECT_EQ(stats.capacity_bytes, 64 + 128 + 256 + 256);
  EXPECT_EQ(stats.used_bytes, stats.capacity_bytes);
  EXPECT_EQ(stats.slab_chunks, 4);
  EXPECT_EQ(stats.large_chunks, 0);

  auto raw_chunks = HostMemoryPoolTestHelper::getRawChunks(*pool);
  std::vector<std::size_t> sizes;
  for (const auto &[address, size] : raw_chunks) {
    (void)address;
    sizes.push_back(size);
  }
  std::sort(sizes.begin(), sizes.end());
  EXPECT_EQ(sizes, (std::vector<std::size_t>{64, 128, 256, 256}));
}

TEST(HostMemoryPool, AdaptiveLargeRequestDoesNotInflateNextSlab) {
  AdaptiveAllocatorSizeGuard allocator_size(64, 256);
  auto pool = HostMemoryPoolTestHelper::createPool(true);

  pool->allocate(65, 1);
  auto stats = pool->get_stats();
  EXPECT_EQ(stats.capacity_bytes, 65);
  EXPECT_EQ(stats.slab_chunks, 0);
  EXPECT_EQ(stats.large_chunks, 1);

  pool->allocate(1, 1);
  stats = pool->get_stats();
  EXPECT_EQ(stats.capacity_bytes, 65 + 64);
  EXPECT_EQ(stats.slab_chunks, 1);
  EXPECT_EQ(stats.large_chunks, 1);
}

TEST(HostMemoryPool, LegacyPolicyKeepsFixedMaximumSlabsAcrossReset) {
  AdaptiveAllocatorSizeGuard allocator_size(64, 256);
  auto pool = HostMemoryPoolTestHelper::createPool(false);

  pool->allocate(1, 1);
  pool->allocate(255, 1);
  pool->allocate(1, 1);
  auto stats = pool->get_stats();
  EXPECT_EQ(stats.capacity_bytes, 512);
  EXPECT_EQ(stats.slab_chunks, 2);
  EXPECT_EQ(stats.large_chunks, 0);

  pool->reset();
  pool->allocate(1, 1);
  stats = pool->get_stats();
  EXPECT_EQ(stats.capacity_bytes, 256);
  EXPECT_EQ(stats.slab_chunks, 1);
}

TEST(HostMemoryPool, ExclusiveSwapEraseKeepsReleaseAddressIndexValid) {
  DefaultAllocatorSizeGuard allocator_size(64);
  HostMemoryPool pool;

  void *first = pool.allocate(64, 1);
  void *exclusive = pool.allocate(17, 1, true);
  void *second = pool.allocate(1, 1);
  pool.release(17, exclusive);
  pool.release(1, second);

  auto stats = pool.get_stats();
  EXPECT_EQ(stats.requested_live_bytes, 64);
  EXPECT_EQ(stats.unreclaimed_released_bytes, 1);
  EXPECT_EQ(stats.slab_chunks, 2);
  EXPECT_EQ(stats.exclusive_chunks, 0);

  pool.release(64, first);
  stats = pool.get_stats();
  EXPECT_EQ(stats.requested_live_bytes, 0);
  EXPECT_EQ(stats.unreclaimed_released_bytes, 65);
}

TEST(HostMemoryPool, AdaptiveResetChurnRestartsInitialSlab) {
  AdaptiveAllocatorSizeGuard allocator_size(64, 256);
  auto pool = HostMemoryPoolTestHelper::createPool(true);

  for (int cycle = 0; cycle < 100; ++cycle) {
    pool->allocate(64, 1);
    pool->allocate(128, 1);
    pool->allocate(300, 1);
    const auto live = pool->get_stats();
    EXPECT_EQ(live.capacity_bytes, 64 + 128 + 300);
    EXPECT_EQ(live.slab_chunks, 2);
    EXPECT_EQ(live.large_chunks, 1);

    pool->reset();
    const auto reset = pool->get_stats();
    EXPECT_EQ(reset.reserved_bytes, 0);
    EXPECT_EQ(reset.capacity_bytes, 0);
    EXPECT_EQ(reset.unified_chunks, 0);
    EXPECT_GE(reset.peak_reserved_bytes, 64 + 128 + 300);
  }

  pool->allocate(1, 1);
  const auto restarted = pool->get_stats();
  EXPECT_EQ(restarted.capacity_bytes, 64);
  EXPECT_EQ(restarted.slab_chunks, 1);
}

TEST(HostMemoryPool, ConcurrentTelemetrySnapshotsStayConsistent) {
  DefaultAllocatorSizeGuard allocator_size(1 << 20);
  HostMemoryPool pool;
  constexpr int kThreadCount = 8;
  constexpr int kAllocationsPerThread = 1000;
  std::atomic<bool> start{false};
  std::atomic<int> active{kThreadCount};
  std::atomic<int> invariant_failures{0};
  std::vector<std::thread> workers;

  for (int thread = 0; thread < kThreadCount; ++thread) {
    workers.emplace_back([&, thread] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      std::vector<std::pair<void *, std::size_t>> allocations;
      allocations.reserve(kAllocationsPerThread);
      for (int index = 0; index < kAllocationsPerThread; ++index) {
        const std::size_t size = 1 + ((thread * 17 + index) % 64);
        allocations.emplace_back(pool.allocate(size, 8), size);
      }
      for (const auto &[ptr, size] : allocations) {
        pool.release(size, ptr);
      }
      active.fetch_sub(1, std::memory_order_release);
    });
  }

  start.store(true, std::memory_order_release);
  while (active.load(std::memory_order_acquire) != 0) {
    const auto stats = pool.get_stats();
    if (stats.used_bytes > stats.capacity_bytes ||
        stats.available_bytes != stats.capacity_bytes - stats.used_bytes ||
        stats.wasted_bytes > stats.used_bytes ||
        stats.requested_live_bytes + stats.wasted_bytes !=
            stats.used_bytes ||
        stats.unified_chunks !=
            stats.slab_chunks + stats.large_chunks +
                stats.exclusive_chunks) {
      invariant_failures.fetch_add(1, std::memory_order_relaxed);
    }
  }
  for (auto &worker : workers) {
    worker.join();
  }

  const auto stats = pool.get_stats();
  EXPECT_EQ(invariant_failures.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(stats.allocate_count,
            kThreadCount * kAllocationsPerThread);
  EXPECT_EQ(stats.release_count,
            kThreadCount * kAllocationsPerThread);
  EXPECT_EQ(stats.requested_live_bytes, 0);
  EXPECT_EQ(stats.wasted_bytes, stats.used_bytes);
  EXPECT_GT(stats.peak_requested_live_bytes, 0);
  EXPECT_GE(stats.peak_used_bytes, stats.used_bytes);
}

TEST(HostMemoryPool, CheckedArithmeticRejectsOverflowAndZeroAlignment) {
  std::size_t size_result = 0;
  std::uintptr_t address_result = 0;

  EXPECT_TRUE(HostMemoryPoolTestHelper::checkedAddSize(13, 7, &size_result));
  EXPECT_EQ(size_result, 20);
  EXPECT_FALSE(HostMemoryPoolTestHelper::checkedAddSize(
      std::numeric_limits<std::size_t>::max(), 1, &size_result));

  EXPECT_TRUE(HostMemoryPoolTestHelper::checkedAddAddress(
      13, 7, &address_result));
  EXPECT_EQ(address_result, 20);
  EXPECT_FALSE(HostMemoryPoolTestHelper::checkedAddAddress(
      std::numeric_limits<std::uintptr_t>::max(), 1, &address_result));

  EXPECT_TRUE(HostMemoryPoolTestHelper::alignUp(13, 8, &address_result));
  EXPECT_EQ(address_result, 16);
  EXPECT_FALSE(HostMemoryPoolTestHelper::alignUp(
      std::numeric_limits<std::uintptr_t>::max() - 3, 8, &address_result));
  EXPECT_FALSE(HostMemoryPoolTestHelper::alignUp(13, 0, &address_result));
}

}  // namespace taichi::lang
