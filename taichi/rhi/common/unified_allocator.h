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
    void *data;
    void *allocation;
    void *head;
    void *tail;
  };

 private:
  static std::size_t default_allocator_size;

  explicit UnifiedAllocator(HostMemoryPool *owner);

  void *allocate(std::size_t size,
                 std::size_t alignment,
                 bool exclusive = false);

  void *release(std::size_t size, void *ptr);

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

  friend class HostMemoryPool;
  friend class HostMemoryPoolTestHelper;
};

}  // namespace taichi::lang
