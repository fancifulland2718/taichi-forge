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

enum class CubSelectValueType : int {
  i32 = 0,
};

enum class CubHistogramValueType : int {
  i32 = 0,
};

enum class CubReduceValueType : int {
  i32 = 0,
  f32 = 1,
};

enum class CubReduceOp : int {
  sum = 0,
  min = 1,
  max = 2,
};

enum class CudaTransformValueType : int {
  i32 = 0,
  f32 = 1,
};

enum class CudaIndexedCopyOp : int {
  gather = 0,
  scatter = 1,
};

enum class CudaScatterAddValueType : int {
  i32 = 0,
  f32 = 1,
};

bool driver_transform_available();

std::size_t driver_transform_affine(void *src,
                                    void *dst,
                                    int num_items,
                                    CudaTransformValueType value_type,
                                    double scale,
                                    double bias);

bool driver_indexed_copy_available();

std::size_t driver_indexed_copy(void *src,
                                void *indices,
                                void *dst,
                                int num_items,
                                int index_bound,
                                CudaIndexedCopyOp op);

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

bool cub_select_available();

std::size_t cub_select_flagged(void *values,
                               void *flags,
                               void *output,
                               void *count,
                               int num_items,
                               CubSelectValueType value_type,
                               void *stream,
                               void *owner);

void cub_select_clear_cache(void *owner);

std::size_t cub_select_cached_bytes(void *owner);

bool cub_histogram_available();

std::size_t cub_histogram_even(void *values,
                               void *bins,
                               int num_items,
                               int num_bins,
                               CubHistogramValueType value_type,
                               void *stream,
                               void *owner);

void cub_histogram_clear_cache(void *owner);

std::size_t cub_histogram_cached_bytes(void *owner);

bool cub_reduce_available();

std::size_t cub_reduce(void *values,
                       void *output,
                       int num_items,
                       CubReduceValueType value_type,
                       CubReduceOp op,
                       void *stream,
                       void *owner);

void cub_reduce_clear_cache(void *owner);

std::size_t cub_reduce_cached_bytes(void *owner);

bool cub_scatter_add_available();

std::size_t cub_scatter_add(void *src,
                            void *indices,
                            void *dst,
                            int num_items,
                            int index_bound,
                            CudaScatterAddValueType value_type,
                            void *stream);

bool cub_bucket_builder_available();

std::size_t cub_bucket_builder_i32(void *keys,
                                   void *values,
                                   void *offsets,
                                   void *output,
                                   void *cursor,
                                   int num_items,
                                   int num_bins,
                                   void *stream,
                                   void *owner);

void cub_bucket_builder_clear_cache(void *owner);

std::size_t cub_bucket_builder_cached_bytes(void *owner);

}  // namespace taichi::lang::cuda
