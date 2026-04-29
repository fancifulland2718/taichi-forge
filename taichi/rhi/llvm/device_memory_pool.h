#pragma once
#include "taichi/common/core.h"
#include "taichi/rhi/device.h"
#include "taichi/rhi/llvm/llvm_device.h"
#include "taichi/rhi/llvm/allocator.h"
#include <mutex>
#include <vector>
#include <memory>
#include <thread>

namespace taichi::lang {

// A memory pool that runs on the host

// R1.c: read-only diagnostic snapshot for the device-side memory pool.
struct DeviceMemoryPoolStats {
  uint64_t allocate_count{0};       // # allocate + allocate_with_cache calls
  uint64_t release_count{0};        // # release calls
  uint64_t bytes_allocated_total{0};
  uint64_t bytes_released_total{0};
  uint64_t cache_hit_count{0};      // hits served from CachingAllocator
  uint64_t cache_miss_count{0};     // forwarded to device->allocate_*
  uint64_t raw_chunks{0};           // # device-side raw blocks alive
  uint64_t raw_bytes{0};            // sum of device-side raw block sizes
  uint64_t cached_blocks{0};        // # blocks parked in caching free-list
  uint64_t cached_bytes{0};         // total bytes parked in free-list
};

class TI_DLL_EXPORT DeviceMemoryPool {
 public:
  std::unique_ptr<CachingAllocator> allocator_{nullptr};
  static const size_t page_size;

  static DeviceMemoryPool &get_instance(bool merge_upon_release = true);

  void *allocate_with_cache(LlvmDevice *device,
                            const LlvmDevice::LlvmRuntimeAllocParams &params);
  void *allocate(std::size_t size, std::size_t alignment, bool managed = false);
  void release(std::size_t size, void *ptr, bool release_raw = false);
  void reset();
  explicit DeviceMemoryPool(bool merge_upon_release);
  ~DeviceMemoryPool();

  // R1.c: returns a consistent snapshot under the pool's own mutex.
  DeviceMemoryPoolStats get_stats();

 protected:
  void *allocate_raw_memory(std::size_t size, bool managed = false);
  void deallocate_raw_memory(void *ptr);

  // All the raw memory allocated from OS/Driver
  // We need to keep track of them to guarantee that they are freed
  std::map<void *, std::size_t> raw_memory_chunks_;

  std::mutex mut_allocation_;
  bool merge_upon_release_ = true;

  // R1.c counters; updated under mut_allocation_.
  uint64_t allocate_count_{0};
  uint64_t release_count_{0};
  uint64_t bytes_allocated_total_{0};
  uint64_t bytes_released_total_{0};
};

}  // namespace taichi::lang
