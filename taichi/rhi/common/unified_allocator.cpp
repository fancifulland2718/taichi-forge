// Virtual memory allocator for CPU/GPU

#include "taichi/rhi/common/unified_allocator.h"
#include "taichi/rhi/common/host_memory_pool.h"
#include <algorithm>
#include <limits>
#include <string>

namespace taichi::lang {

std::size_t UnifiedAllocator::default_allocator_size =
    1 << 30;  // 1 GiB maximum slab
std::size_t UnifiedAllocator::initial_allocator_size =
    16 << 20;  // 16 MiB initial adaptive slab

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

UnifiedAllocator::UnifiedAllocator(HostMemoryPool *owner,
                                   bool adaptive_chunk_policy)
    : owner_(owner), adaptive_chunk_policy_(adaptive_chunk_policy) {
  TI_ASSERT(owner_ != nullptr);
  TI_ERROR_IF(default_allocator_size == 0,
              "Host allocator maximum slab size must be non-zero");
  TI_ERROR_IF(initial_allocator_size == 0,
              "Host allocator initial slab size must be non-zero");
  next_slab_size_ =
      adaptive_chunk_policy_
          ? std::min(initial_allocator_size, default_allocator_size)
          : default_allocator_size;
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

void UnifiedAllocator::grow_next_slab_size() {
  if (!adaptive_chunk_policy_ ||
      next_slab_size_ >= default_allocator_size) {
    return;
  }
  if (next_slab_size_ > default_allocator_size / 2) {
    next_slab_size_ = default_allocator_size;
  } else {
    next_slab_size_ *= 2;
  }
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
    // Recent slabs are overwhelmingly likely to own the remaining bump tail.
    // Large chunks remain request-owned instead of becoming implicit slabs.
    for (size_t chunk_id = chunks_.size(); chunk_id > 0; --chunk_id) {
      auto &chunk = chunks_[chunk_id - 1];
      if (chunk.is_exclusive || chunk.is_large) {
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

      TI_ERROR_IF(
          size > std::numeric_limits<std::size_t>::max() -
                     chunk.requested_size,
          "Host allocator per-chunk requested size overflowed");
      // The subtraction check above proves this addition cannot overflow.
      const auto old_head = head;
      TI_ASSERT(checked_add_address(ret, size, &head));
      TI_ASSERT(ret % alignment == 0);
      chunk.head = reinterpret_cast<void *>(head);
      const auto alignment_waste = ret - old_head;
      const auto consumed = head - old_head;
      TI_ASSERT(consumed <= capacity_bytes_ - used_bytes_);
      used_bytes_ += consumed;
      alignment_waste_bytes_ += alignment_waste;
      chunk.alignment_waste_size += alignment_waste;
      requested_live_bytes_ += size;
      chunk.requested_size += size;
      chunk.has_zero_size_allocation |= size == 0;
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
  bool is_large = false;
  if (!exclusive) {
    if (minimum_allocation_size > next_slab_size_) {
      // A request larger than the next slab gets a request-sized,
      // alignment-safe mapping and does not inflate the geometric slab
      // sequence used by later small requests.
      is_large = true;
    } else {
      allocation_size = next_slab_size_;
    }
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
  chunk.is_large = is_large;
  chunk.has_zero_size_allocation = size == 0;
  chunk.requested_size = size;
  chunk.released_size = 0;
  chunk.alignment_waste_size = ret - data;

  TI_ASSERT(chunk.data != nullptr);
  TI_ASSERT(uint64(chunk.data) % HostMemoryPool::page_size == 0);
  TI_ASSERT(ret % alignment == 0);

  chunks_.emplace_back(std::move(chunk));
  const auto chunk_index = chunks_.size() - 1;
  const auto [index_it, index_inserted] =
      chunk_indices_by_base_.emplace(data, chunk_index);
  (void)index_it;
  TI_ASSERT(index_inserted);
  if (!exclusive && !is_large) {
    // Advance the policy only after the mapping and chunk insertion succeed.
    grow_next_slab_size();
  }
  const auto consumed = head - data;
  const auto alignment_waste = ret - data;
  capacity_bytes_ += allocation_size;
  used_bytes_ += consumed;
  alignment_waste_bytes_ += alignment_waste;
  requested_live_bytes_ += size;
  if (exclusive) {
    ++exclusive_chunk_count_;
  } else if (is_large) {
    ++large_chunk_count_;
  } else {
    ++slab_chunk_count_;
  }
  return reinterpret_cast<void *>(ret);
}

void *UnifiedAllocator::release_chunk(std::size_t chunk_index) {
  TI_ASSERT(chunk_index < chunks_.size());
  const auto &chunk = chunks_[chunk_index];
  void *raw_ptr = chunk.data;
  const auto data = reinterpret_cast<std::uintptr_t>(chunk.data);
  const auto head = reinterpret_cast<std::uintptr_t>(chunk.head);
  const auto tail = reinterpret_cast<std::uintptr_t>(chunk.tail);
  const auto capacity = tail - data;
  const auto consumed = head - data;
  const auto remaining_request =
      chunk.requested_size - chunk.released_size;

  TI_ASSERT(capacity <= capacity_bytes_);
  TI_ASSERT(consumed <= used_bytes_);
  TI_ASSERT(chunk.alignment_waste_size <= alignment_waste_bytes_);
  TI_ASSERT(chunk.released_size <= unreclaimed_released_bytes_);
  TI_ASSERT(remaining_request <= requested_live_bytes_);
  capacity_bytes_ -= capacity;
  used_bytes_ -= consumed;
  alignment_waste_bytes_ -= chunk.alignment_waste_size;
  unreclaimed_released_bytes_ -= chunk.released_size;
  requested_live_bytes_ -= remaining_request;
  if (chunk.is_exclusive) {
    TI_ASSERT(exclusive_chunk_count_ > 0);
    --exclusive_chunk_count_;
  } else if (chunk.is_large) {
    TI_ASSERT(large_chunk_count_ > 0);
    --large_chunk_count_;
  } else {
    TI_ASSERT(slab_chunk_count_ > 0);
    --slab_chunk_count_;
  }

  const auto location = chunk_indices_by_base_.find(data);
  TI_ASSERT(location != chunk_indices_by_base_.end());
  const auto last_index = chunks_.size() - 1;
  chunk_indices_by_base_.erase(location);
  if (chunk_index != last_index) {
    const auto swapped_base =
        reinterpret_cast<std::uintptr_t>(chunks_[last_index].data);
    auto swapped_location = chunk_indices_by_base_.find(swapped_base);
    TI_ASSERT(swapped_location != chunk_indices_by_base_.end());
    swapped_location->second = chunk_index;
  }
  swap_erase_vector<MemoryChunk>(chunks_, chunk_index);
  // HostMemoryPool releases the raw mapping after this metadata is removed.
  return raw_ptr;
}

void *UnifiedAllocator::release(std::size_t size, void *ptr) {
  // UnifiedAllocator is special in that it never reuses the previously
  // allocated memory We have to release the entire memory chunk to avoid memory
  // leak
  const auto released_address = reinterpret_cast<std::uintptr_t>(ptr);
  auto location = chunk_indices_by_base_.upper_bound(released_address);
  if (location == chunk_indices_by_base_.begin()) {
    return nullptr;
  }
  --location;
  const auto chunk_index = location->second;
  TI_ASSERT(chunk_index < chunks_.size());
  auto &chunk = chunks_[chunk_index];
  const auto data = reinterpret_cast<std::uintptr_t>(chunk.data);
  const auto head = reinterpret_cast<std::uintptr_t>(chunk.head);

  if (chunk.is_exclusive && chunk.allocation == ptr) {
    return release_chunk(chunk_index);
  }

  if (!chunk.is_exclusive && released_address >= data &&
      (released_address < head ||
       (size == 0 && released_address == head))) {
    const auto unreclaimed_capacity =
        chunk.requested_size -
        std::min(chunk.requested_size, chunk.released_size);
    const auto released =
        std::min<std::uint64_t>(size, unreclaimed_capacity);
    chunk.released_size += released;
    unreclaimed_released_bytes_ += released;
    requested_live_bytes_ -=
        std::min(requested_live_bytes_, released);
    if (chunk.requested_size != 0 &&
        chunk.released_size == chunk.requested_size &&
        !chunk.has_zero_size_allocation) {
      return release_chunk(chunk_index);
    }
  }

  return nullptr;
}

UnifiedAllocator::Statistics UnifiedAllocator::get_stats() const {
  Statistics stats;
  stats.capacity_bytes = capacity_bytes_;
  stats.used_bytes = used_bytes_;
  stats.available_bytes = capacity_bytes_ - used_bytes_;
  stats.alignment_waste_bytes = alignment_waste_bytes_;
  stats.unreclaimed_released_bytes = unreclaimed_released_bytes_;
  stats.wasted_bytes =
      alignment_waste_bytes_ + unreclaimed_released_bytes_;
  stats.requested_live_bytes = requested_live_bytes_;
  stats.chunk_count = chunks_.size();
  stats.slab_chunk_count = slab_chunk_count_;
  stats.large_chunk_count = large_chunk_count_;
  stats.exclusive_chunk_count = exclusive_chunk_count_;
  return stats;
}

}  // namespace taichi::lang
