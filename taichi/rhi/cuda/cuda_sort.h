#pragma once

#include <cstddef>

namespace taichi::lang::cuda {

enum class CubSortKeyType : int {
  u32 = 0,
  i32 = 1,
  f32 = 2,
  u64 = 3,
  i64 = 4,
  f64 = 5,
};

enum class CubSortMode : int {
  native = 0,
  split32 = 1,
};

enum class CubSortNanPolicy : int {
  last = 0,
  bitwise = 1,
};

enum class CubScanValueType : int {
  i32 = 0,
};

bool cub_radix_sort_available();

std::size_t cub_radix_sort(void *keys,
                           void *values,
                           int num_items,
                           CubSortKeyType key_type,
                           CubSortMode mode,
                           CubSortNanPolicy nan_policy,
                           bool has_values,
                           void *stream,
                           void *owner);

void cub_radix_sort_clear_cache(void *owner);

std::size_t cub_radix_sort_cached_bytes(void *owner);

bool cub_inclusive_scan_available();

std::size_t cub_inclusive_scan(void *data,
                               int num_items,
                               CubScanValueType value_type,
                               void *stream,
                               void *owner);

void cub_inclusive_scan_clear_cache(void *owner);

std::size_t cub_inclusive_scan_cached_bytes(void *owner);

}  // namespace taichi::lang::cuda
