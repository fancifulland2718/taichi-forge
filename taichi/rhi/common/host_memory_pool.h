#pragma once
#include "taichi/common/core.h"
#include "taichi/rhi/common/unified_allocator.h"
#include "taichi/rhi/device.h"
#include <mutex>
#include <vector>
#include <memory>
#include <thread>

namespace taichi::lang {

// A memory pool that runs on the host

// R1.c: read-only diagnostic snapshot. All values are sampled under the
// HostMemoryPool's own mutex, so the snapshot is internally consistent.
struct HostMemoryPoolStats {
  uint64_t allocate_count{0};      // # of HostMemoryPool::allocate calls
  uint64_t release_count{0};       // # of HostMemoryPool::release calls
  uint64_t bytes_allocated_total{0};  // sum of all `size` requested ever
  uint64_t bytes_released_total{0};   // sum of all `size` released ever
  uint64_t raw_chunks{0};          // # OS-level chunks currently alive
  uint64_t raw_bytes{0};           // sum of OS-level chunk sizes
  uint64_t unified_chunks{0};      // # UnifiedAllocator slab chunks
  uint64_t requested_live_bytes{0};
  uint64_t reserved_bytes{0};
  uint64_t committed_bytes{0};
  bool committed_bytes_available{false};
  uint64_t capacity_bytes{0};
  uint64_t used_bytes{0};
  uint64_t available_bytes{0};
  uint64_t alignment_waste_bytes{0};
  uint64_t unreclaimed_released_bytes{0};
  uint64_t wasted_bytes{0};
  uint64_t slab_chunks{0};
  uint64_t large_chunks{0};
  uint64_t exclusive_chunks{0};
  uint64_t peak_requested_live_bytes{0};
  uint64_t peak_reserved_bytes{0};
  uint64_t peak_used_bytes{0};
  uint64_t peak_wasted_bytes{0};
  uint64_t peak_chunks{0};
};

class TI_DLL_EXPORT HostMemoryPool {
 public:
  static const size_t page_size;

  static HostMemoryPool &get_instance();

  void *allocate(std::size_t size,
                 std::size_t alignment,
                 bool exclusive = false);
  void release(std::size_t size, void *ptr);
  void reset();
  HostMemoryPool();
  ~HostMemoryPool();

  // R1.c diagnostic stats. Read-only, takes the same lock allocate/release
  // take, so adds only one extra mutex acquire per call (off the hot path).
  HostMemoryPoolStats get_stats();

 protected:
  void *allocate_raw_memory(std::size_t size);
  void deallocate_raw_memory(void *ptr);
  void update_allocator_peaks_locked();

  // All the raw memory allocated from OS/Driver
  // We need to keep track of them to guarantee that they are freed
  std::map<void *, std::size_t> raw_memory_chunks_;

  std::unique_ptr<UnifiedAllocator> allocator_;
  std::mutex mut_allocation_;

  // R1.c counters; updated under mut_allocation_, no extra atomics.
  uint64_t allocate_count_{0};
  uint64_t release_count_{0};
  uint64_t bytes_allocated_total_{0};
  uint64_t bytes_released_total_{0};
  uint64_t raw_bytes_{0};
  uint64_t peak_requested_live_bytes_{0};
  uint64_t peak_reserved_bytes_{0};
  uint64_t peak_used_bytes_{0};
  uint64_t peak_wasted_bytes_{0};
  uint64_t peak_chunks_{0};

  friend class UnifiedAllocator;
  friend class HostMemoryPoolTestHelper;
};

}  // namespace taichi::lang
