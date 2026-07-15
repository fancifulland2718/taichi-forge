// Virtual memory allocator for CPU/GPU

#include "taichi/rhi/common/unified_allocator.h"
#include "taichi/rhi/common/host_memory_pool.h"
#include <algorithm>
#include <limits>
#include <string>

namespace taichi::lang {

std::size_t UnifiedAllocator::default_allocator_size =
    1 << 30;  // 1 GB per allocator

template <typename T>
static void swap_erase_vector(std::vector<T> &vec, size_t idx) {
  bool is_last = idx == vec.size() - 1;
  TI_ASSERT(idx < vec.size());

  if (!is_last) {
    std::swap(vec[idx], vec.back());
  }

  vec.pop_back();

  // There's no need to swap back since we'll iterate the memory chunks to
  // search for reusable memory
}

UnifiedAllocator::UnifiedAllocator(HostMemoryPool *owner) : owner_(owner) {
  TI_ASSERT(owner_ != nullptr);
}

bool UnifiedAllocator::checked_add_size(std::size_t lhs,
                                        std::size_t rhs,
                                        std::size_t *result) {
  if (result == nullptr ||
      rhs > std::numeric_limits<std::size_t>::max() - lhs) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

bool UnifiedAllocator::checked_add_address(std::uintptr_t lhs,
                                           std::size_t rhs,
                                           std::uintptr_t *result) {
  if (result == nullptr ||
      rhs > std::numeric_limits<std::uintptr_t>::max() - lhs) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

bool UnifiedAllocator::align_up(std::uintptr_t value,
                                std::size_t alignment,
                                std::uintptr_t *result) {
  if (alignment == 0 || result == nullptr) {
    return false;
  }
  const auto remainder = value % alignment;
  const auto padding = remainder == 0 ? 0 : alignment - remainder;
  return checked_add_address(value, padding, result);
}

void *UnifiedAllocator::allocate(std::size_t size,
                                 std::size_t alignment,
                                 bool exclusive) {
  // UnifiedAllocator never reuses the previously allocated memory
  // just move the head forward util depleting all the free memory

  // Note: put mutex on MemoryPool instead of Allocator, since Allocators are
  // transparent to user code
  TI_ERROR_IF(alignment == 0, "Memory allocation alignment must be non-zero");
  TI_ERROR_IF(exclusive && size == 0,
              "Exclusive zero-sized host allocation is not supported");

  if (!chunks_.empty() && !exclusive) {
    // Search for a non-exclusive chunk that has enough space
    for (size_t chunk_id = 0; chunk_id < chunks_.size(); chunk_id++) {
      auto &chunk = chunks_[chunk_id];
      if (chunk.is_exclusive) {
        continue;
      }
      auto head = reinterpret_cast<std::uintptr_t>(chunk.head);
      auto tail = reinterpret_cast<std::uintptr_t>(chunk.tail);
      auto data = reinterpret_cast<std::uintptr_t>(chunk.data);
      std::uintptr_t ret = 0;
      TI_TRACE("UM [data={}] allocate() request={} remain={}", (intptr_t)data,
               size, (tail - head));
      if (!align_up(head, alignment, &ret)) {
        continue;
      }
      if (ret < data || ret > tail || size > tail - ret) {
        continue;
      }

      // The subtraction check above proves this addition cannot overflow.
      TI_ASSERT(checked_add_address(ret, size, &head));
      TI_ASSERT(ret % alignment == 0);
      chunk.head = reinterpret_cast<void *>(head);
      return reinterpret_cast<void *>(ret);
    }
  }

  // Allocate a new chunk
  MemoryChunk chunk;

  // Raw host mappings are page-aligned. When the requested alignment does not
  // divide the page size, reserve worst-case padding so that the aligned user
  // range still fits entirely inside the mapping.
  const std::size_t max_alignment_padding =
      HostMemoryPool::page_size % alignment == 0 ? 0 : alignment - 1;
  std::size_t minimum_allocation_size = 0;
  TI_ERROR_IF(!checked_add_size(size, max_alignment_padding,
                                &minimum_allocation_size),
              "Host allocation size overflow: size={}, alignment={}", size,
              alignment);

  std::size_t allocation_size = minimum_allocation_size;
  if (!exclusive) {
    // Do not allocate large memory chunks for "exclusive" allocation
    // to improve memory and allocation efficiency.
    allocation_size = std::max(allocation_size, default_allocator_size);
  }
  TI_ERROR_IF(allocation_size == 0,
              "Host allocation mapping size must be non-zero");

  TI_TRACE("Allocating virtual address space of size {} MB",
           allocation_size / 1024 / 1024);

  void *ptr = owner_->allocate_raw_memory(allocation_size);
  const auto data = reinterpret_cast<std::uintptr_t>(ptr);
  std::uintptr_t tail = 0;
  std::uintptr_t ret = 0;
  std::uintptr_t head = 0;
  if (!checked_add_address(data, allocation_size, &tail) ||
      !align_up(data, alignment, &ret) || ret < data || ret > tail ||
      size > tail - ret || !checked_add_address(ret, size, &head)) {
    owner_->deallocate_raw_memory(ptr);
    TI_ERROR("Host allocation range overflow: size={}, alignment={}", size,
             alignment);
  }

  chunk.data = ptr;
  chunk.allocation = reinterpret_cast<void *>(ret);
  chunk.head = reinterpret_cast<void *>(head);
  chunk.tail = reinterpret_cast<void *>(tail);
  chunk.is_exclusive = exclusive;

  TI_ASSERT(chunk.data != nullptr);
  TI_ASSERT(uint64(chunk.data) % HostMemoryPool::page_size == 0);
  TI_ASSERT(ret % alignment == 0);

  chunks_.emplace_back(std::move(chunk));
  return reinterpret_cast<void *>(ret);
}

void *UnifiedAllocator::release(std::size_t size, void *ptr) {
  // UnifiedAllocator is special in that it never reuses the previously
  // allocated memory We have to release the entire memory chunk to avoid memory
  // leak
  (void)size;
  int remove_idx = -1;
  void *raw_ptr = nullptr;
  for (size_t chunk_idx = 0; chunk_idx < chunks_.size(); chunk_idx++) {
    auto &chunk = chunks_[chunk_idx];

    if (chunk.is_exclusive && chunk.allocation == ptr) {
      remove_idx = chunk_idx;
      raw_ptr = chunk.data;
      break;
    }
  }

  if (remove_idx != -1) {
    swap_erase_vector<MemoryChunk>(chunks_, remove_idx);
    // MemoryPool is responsible for releasing the raw memory
    return raw_ptr;
  }

  return nullptr;
}

}  // namespace taichi::lang
