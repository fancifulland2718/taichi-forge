#include <cstdint>
#include <limits>
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
