#pragma once

namespace taichi::lang::cuda::detail {

enum class MemoryAllocationRoute {
  kSynchronous,
  kAsyncMemoryPool,
};

constexpr bool supports_memory_pool(int driver_major,
                                    int driver_minor,
                                    int device_supports_memory_pool) {
  return device_supports_memory_pool != 0 &&
         (driver_major > 11 ||
          (driver_major == 11 && driver_minor >= 2));
}

constexpr MemoryAllocationRoute memory_allocation_route(
    bool supports_memory_pool) {
  return supports_memory_pool ? MemoryAllocationRoute::kAsyncMemoryPool
                             : MemoryAllocationRoute::kSynchronous;
}

}  // namespace taichi::lang::cuda::detail
