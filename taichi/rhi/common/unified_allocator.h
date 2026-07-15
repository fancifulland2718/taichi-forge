#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "taichi/rhi/arch.h"
#include "taichi/rhi/device.h"

namespace taichi::lang {

class HostMemoryPool;

// This class can only be accessed by MemoryPool
class UnifiedAllocator {
 public:
  struct MemoryChunk {
    bool is_exclusive;
    bool is_large;
    void *data;
    void *allocation;
    void *head;
    void *tail;
    std::size_t requested_size;
    std::size_t released_size;
  };

  struct Statistics {
    std::uint64_t capacity_bytes{0};
    std::uint64_t used_bytes{0};
    std::uint64_t available_bytes{0};
    std::uint64_t alignment_waste_bytes{0};
    std::uint64_t unreclaimed_released_bytes{0};
    std::uint64_t wasted_bytes{0};
    std::uint64_t requested_live_bytes{0};
    std::uint64_t chunk_count{0};
    std::uint64_t slab_chunk_count{0};
    std::uint64_t large_chunk_count{0};
    std::uint64_t exclusive_chunk_count{0};
  };

 private:
  static std::size_t default_allocator_size;

  explicit UnifiedAllocator(HostMemoryPool *owner);

  void *allocate(std::size_t size,
                 std::size_t alignment,
                 bool exclusive = false);

  void *release(std::size_t size, void *ptr);

  Statistics get_stats() const;

  static bool checked_add_size(std::size_t lhs,
                               std::size_t rhs,
                               std::size_t *result);
  static bool checked_add_address(std::uintptr_t lhs,
                                  std::size_t rhs,
                                  std::uintptr_t *result);
  static bool align_up(std::uintptr_t value,
                       std::size_t alignment,
                       std::uintptr_t *result);

  HostMemoryPool *owner_{nullptr};

  std::vector<MemoryChunk> chunks_;
  std::uint64_t capacity_bytes_{0};
  std::uint64_t used_bytes_{0};
  std::uint64_t alignment_waste_bytes_{0};
  std::uint64_t unreclaimed_released_bytes_{0};
  std::uint64_t requested_live_bytes_{0};
  std::uint64_t slab_chunk_count_{0};
  std::uint64_t large_chunk_count_{0};
  std::uint64_t exclusive_chunk_count_{0};

  friend class HostMemoryPool;
  friend class HostMemoryPoolTestHelper;
};

}  // namespace taichi::lang
