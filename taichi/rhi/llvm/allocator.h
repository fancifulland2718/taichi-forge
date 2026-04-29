#pragma once

#include "taichi/common/core.h"
#include "taichi/math/arithmetic.h"
#include "taichi/rhi/llvm/llvm_device.h"
#include "taichi/inc/constants.h"
#include <stdint.h>
#include <map>
#include <set>

namespace taichi::lang {

class CachingAllocator {
 public:
  explicit CachingAllocator(bool merge_upon_release = true);

  uint64_t *allocate(LlvmDevice *device,
                     const LlvmDevice::LlvmRuntimeAllocParams &params);
  void release(size_t sz, uint64_t *ptr);

  // R1.c read-only diagnostic accessors. Caller must hold the parent
  // DeviceMemoryPool's mutex; no extra synchronization here.
  std::size_t cached_block_count() const { return mem_blocks_.size(); }
  std::size_t cached_bytes() const {
    std::size_t total = 0;
    for (const auto &kv : ptr_map_) {
      total += kv.second;
    }
    return total;
  }
  uint64_t cache_hit_count() const { return cache_hit_count_; }
  uint64_t cache_miss_count() const { return cache_miss_count_; }

 private:
  void merge_and_insert(uint8_t *ptr, std::size_t size);

  std::set<std::pair<std::size_t, uint8_t *>> mem_blocks_;
  std::map<uint8_t *, std::size_t> ptr_map_;

  // Allocator options
  bool merge_upon_release_ = true;

  // R1.c counters; protected by parent DeviceMemoryPool::mut_allocation_.
  uint64_t cache_hit_count_{0};
  uint64_t cache_miss_count_{0};
};

}  // namespace taichi::lang
