#include "taichi/rhi/common/host_memory_pool.h"

#include <algorithm>
#include <limits>
#include <memory>

#if defined(TI_PLATFORM_UNIX)
#include <sys/mman.h>
#else
#include "taichi/platform/windows/windows.h"
#endif

namespace taichi::lang {

namespace {

uint64_t saturating_add(uint64_t lhs, uint64_t rhs) {
  return rhs > std::numeric_limits<uint64_t>::max() - lhs
             ? std::numeric_limits<uint64_t>::max()
             : lhs + rhs;
}

}  // namespace

HostMemoryPool::HostMemoryPool() {
  allocator_ = std::unique_ptr<UnifiedAllocator>(new UnifiedAllocator(this));

  TI_TRACE("Memory pool created. Default buffer size per allocator = {} MB",
           UnifiedAllocator::default_allocator_size / 1024 / 1024);
}

void *HostMemoryPool::allocate(std::size_t size,
                               std::size_t alignment,
                               bool exclusive) {
  std::lock_guard<std::mutex> _(mut_allocation_);

  if (!allocator_) {
    TI_ERROR("Memory pool is already destroyed");
  }
  void *ret = allocator_->allocate(size, alignment, exclusive);
  // R1.c counters: update under existing lock (no extra cost on hot path).
  allocate_count_ = saturating_add(allocate_count_, 1);
  bytes_allocated_total_ = saturating_add(bytes_allocated_total_, size);
  update_allocator_peaks_locked();
  return ret;
}

void HostMemoryPool::release(std::size_t size, void *ptr) {
  std::lock_guard<std::mutex> _(mut_allocation_);

  if (!allocator_) {
    TI_ERROR("Memory pool is already destroyed");
  }

  if (void *raw_ptr = allocator_->release(size, ptr)) {
    deallocate_raw_memory(raw_ptr);  // release raw memory as well
  }
  release_count_ = saturating_add(release_count_, 1);
  bytes_released_total_ = saturating_add(bytes_released_total_, size);
  update_allocator_peaks_locked();
}

HostMemoryPoolStats HostMemoryPool::get_stats() {
  std::lock_guard<std::mutex> _(mut_allocation_);
  HostMemoryPoolStats s;
  s.allocate_count = allocate_count_;
  s.release_count = release_count_;
  s.bytes_allocated_total = bytes_allocated_total_;
  s.bytes_released_total = bytes_released_total_;
  s.raw_chunks = raw_memory_chunks_.size();
  s.raw_bytes = raw_bytes_;
  s.reserved_bytes = raw_bytes_;
#if defined(TI_PLATFORM_UNIX)
  s.committed_bytes_available = false;
#else
  s.committed_bytes = raw_bytes_;
  s.committed_bytes_available = true;
#endif
  if (allocator_) {
    const auto allocator = allocator_->get_stats();
    s.unified_chunks = allocator.chunk_count;
    s.requested_live_bytes = allocator.requested_live_bytes;
    s.capacity_bytes = allocator.capacity_bytes;
    s.used_bytes = allocator.used_bytes;
    s.available_bytes = allocator.available_bytes;
    s.alignment_waste_bytes = allocator.alignment_waste_bytes;
    s.unreclaimed_released_bytes =
        allocator.unreclaimed_released_bytes;
    s.wasted_bytes = allocator.wasted_bytes;
    s.slab_chunks = allocator.slab_chunk_count;
    s.large_chunks = allocator.large_chunk_count;
    s.exclusive_chunks = allocator.exclusive_chunk_count;
  }
  s.peak_requested_live_bytes = peak_requested_live_bytes_;
  s.peak_reserved_bytes = peak_reserved_bytes_;
  s.peak_used_bytes = peak_used_bytes_;
  s.peak_wasted_bytes = peak_wasted_bytes_;
  s.peak_chunks = peak_chunks_;
  return s;
}

void HostMemoryPool::update_allocator_peaks_locked() {
  if (!allocator_) {
    return;
  }
  const auto stats = allocator_->get_stats();
  peak_requested_live_bytes_ =
      std::max(peak_requested_live_bytes_, stats.requested_live_bytes);
  peak_used_bytes_ = std::max(peak_used_bytes_, stats.used_bytes);
  peak_wasted_bytes_ = std::max(peak_wasted_bytes_, stats.wasted_bytes);
  peak_chunks_ = std::max(peak_chunks_, stats.chunk_count);
}

void *HostMemoryPool::allocate_raw_memory(std::size_t size) {
  /*
    Be aware that this methods is not protected by the mutex.

    allocate_raw_memory() is designed to be a private method, and
    should only be called by its Allocators friends.

    The caller ensures that no other thread is accessing the memory pool
    when calling this method.
  */

  void *ptr = nullptr;
#if defined(TI_PLATFORM_UNIX)
  ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
             -1, 0);
  TI_ERROR_IF(ptr == MAP_FAILED, "Virtual memory allocation ({} B) failed.",
              size);
#else
  MEMORYSTATUSEX stat;
  stat.dwLength = sizeof(stat);
  GlobalMemoryStatusEx(&stat);
  if (stat.ullAvailVirtual < size) {
    TI_P(stat.ullAvailVirtual);
    TI_P(size);
    TI_ERROR("Insufficient virtual memory space");
  }
  ptr = VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  TI_ERROR_IF(ptr == nullptr, "Virtual memory allocation ({} B) failed.", size);
#endif
  TI_ERROR_IF(((uint64_t)ptr) % page_size != 0,
              "Allocated address ({:}) is not aligned by page size {}", ptr,
              page_size);

  if (raw_memory_chunks_.count(ptr)) {
    TI_ERROR("Memory address ({:}) is already allocated", ptr);
  }

  raw_memory_chunks_[ptr] = size;
  raw_bytes_ = saturating_add(raw_bytes_, size);
  peak_reserved_bytes_ = std::max(peak_reserved_bytes_, raw_bytes_);
  return ptr;
}

void HostMemoryPool::deallocate_raw_memory(void *ptr) {
  /*
    Be aware that this methods is not protected by the mutex.

    deallocate_raw_memory() is designed to be a private method, and
    should only be called by its Allocators friends.

    The caller ensures that no other thread is accessing the memory pool
    when calling this method.
  */
  if (!raw_memory_chunks_.count(ptr)) {
    TI_ERROR("Memory address ({:}) is not allocated", ptr);
  }

  std::size_t size = raw_memory_chunks_[ptr];
#if defined(TI_PLATFORM_UNIX)
  if (munmap(ptr, size) != 0)
#else
  // https://docs.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualfree
  // According to MS Doc: size must be when using MEM_RELEASE
  if (!VirtualFree(ptr, 0, MEM_RELEASE))
#endif
    TI_ERROR("Failed to free virtual memory ({} B)", size);

  raw_memory_chunks_.erase(ptr);
  TI_ASSERT(size <= raw_bytes_);
  raw_bytes_ -= size;
}

void HostMemoryPool::reset() {
  std::lock_guard<std::mutex> _(mut_allocation_);
  allocator_ = std::unique_ptr<UnifiedAllocator>(new UnifiedAllocator(this));

  const auto ptr_map_copied = raw_memory_chunks_;
  for (auto &ptr : ptr_map_copied) {
    deallocate_raw_memory(ptr.first);
  }
}

HostMemoryPool::~HostMemoryPool() {
  reset();
}

const size_t HostMemoryPool::page_size{1 << 12};  // 4 KB page size by default

HostMemoryPool &HostMemoryPool::get_instance() {
  static HostMemoryPool *memory_pool = new HostMemoryPool();
  return *memory_pool;
}

}  // namespace taichi::lang
