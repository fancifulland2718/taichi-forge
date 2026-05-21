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

enum class CubSortValueType : int {
  i32 = 0,
  f32 = 1,
  u32 = 2,
  u64 = 3,
  i64 = 4,
  f64 = 5,
};

enum class CubScanValueType : int {
  i32 = 0,
  f32 = 1,
  u32 = 2,
  u64 = 3,
  i64 = 4,
  f64 = 5,
};

enum class CubSelectValueType : int {
  i32 = 0,
  f32 = 1,
  u32 = 2,
  u64 = 3,
  i64 = 4,
  f64 = 5,
};

enum class CubHistogramValueType : int {
  i32 = 0,
  u32 = 2,
};

enum class CubHistogramBinType : int {
  i32 = 0,
  i64 = 4,
};

enum class CubReduceValueType : int {
  i32 = 0,
  f32 = 1,
  u32 = 2,
  u64 = 3,
  i64 = 4,
  f64 = 5,
};

enum class CubReduceOp : int {
  sum = 0,
  min = 1,
  max = 2,
};

enum class CudaTransformValueType : int {
  i32 = 0,
  f32 = 1,
  u32 = 2,
  u64 = 3,
  i64 = 4,
  f64 = 5,
};

enum class CudaIndexedCopyOp : int {
  gather = 0,
  scatter = 1,
};

enum class CudaScatterAddValueType : int {
  i32 = 0,
  f32 = 1,
  u32 = 2,
  u64 = 3,
  i64 = 4,
  f64 = 5,
};

enum class CudaGroupedReduceValueType : int {
  i32 = 0,
  f32 = 1,
  u32 = 2,
  u64 = 3,
  i64 = 4,
  f64 = 5,
};

enum class CudaBucketBuilderValueType : int {
  i32 = 0,
  f32 = 1,
  u32 = 2,
  u64 = 3,
  i64 = 4,
  f64 = 5,
};

bool driver_transform_available();

std::size_t driver_transform_affine(void *src,
                                    void *dst,
                                    int num_items,
                                    CudaTransformValueType value_type,
                                    double scale,
                                    double bias);

bool cub_transform_available();

std::size_t cub_transform_affine(void *src,
                                 void *dst,
                                 int num_items,
                                 CudaTransformValueType value_type,
                                 double scale,
                                 double bias,
                                 void *stream);

std::size_t cub_transform_affine_strided(void *src,
                                         void *dst,
                                         int num_items,
                                         CudaTransformValueType value_type,
                                         std::size_t offset,
                                         std::size_t stride,
                                         double scale,
                                         double bias,
                                         void *stream);

bool driver_indexed_copy_available();

std::size_t driver_indexed_copy(void *src,
                                void *indices,
                                void *dst,
                                int num_items,
                                int index_bound,
                                CudaIndexedCopyOp op);

bool cub_indexed_copy_available();

std::size_t cub_indexed_copy(void *src,
                             void *indices,
                             void *dst,
                             int num_items,
                             int index_bound,
                             int item_words,
                             CudaIndexedCopyOp op,
                             void *stream);

bool cub_radix_sort_available();

std::size_t cub_radix_sort(void *keys,
                           void *values,
                           int num_items,
                           CubSortKeyType key_type,
                           CubSortValueType value_type,
                           CubSortMode mode,
                           CubSortNanPolicy nan_policy,
                           bool has_values,
                           int value_words,
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

std::size_t cub_inclusive_scan_strided(void *data,
                                       int num_items,
                                       CubScanValueType value_type,
                                       std::size_t offset,
                                       std::size_t stride,
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
                               int item_words,
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
                               CubHistogramBinType bin_type,
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

std::size_t cub_reduce_strided(void *values,
                               void *output,
                               int num_items,
                               CubReduceValueType value_type,
                               std::size_t offset,
                               std::size_t stride,
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

std::size_t cub_scatter_add_strided(void *src,
                                    void *indices,
                                    void *dst,
                                    int num_items,
                                    int index_bound,
                                    CudaScatterAddValueType value_type,
                                    std::size_t offset,
                                    std::size_t stride,
                                    void *stream);

std::size_t cub_scatter_add_strided_io(void *src,
                                       void *indices,
                                       void *dst,
                                       int num_items,
                                       int index_bound,
                                       CudaScatterAddValueType value_type,
                                       std::size_t src_offset,
                                       std::size_t src_stride,
                                       std::size_t dst_offset,
                                       std::size_t dst_stride,
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

std::size_t cub_bucket_builder(void *keys,
                               void *values,
                               void *offsets,
                               void *output,
                               void *cursor,
                               int num_items,
                               int num_bins,
                               CudaBucketBuilderValueType value_type,
                               int item_words,
                               void *stream,
                               void *owner);

void cub_bucket_builder_clear_cache(void *owner);

std::size_t cub_bucket_builder_cached_bytes(void *owner);

bool cub_grouped_reduce_available();

std::size_t cub_grouped_reduce_atomic(void *keys,
                                      void *values,
                                      void *output,
                                      int num_items,
                                      int num_groups,
                                      CudaGroupedReduceValueType value_type,
                                      int op,
                                      void *stream);

std::size_t cub_grouped_reduce_atomic_strided(
    void *keys,
    void *values,
    void *output,
    int num_items,
    int num_groups,
    CudaGroupedReduceValueType value_type,
    std::size_t offset,
    std::size_t stride,
    int op,
    void *stream);

std::size_t cub_grouped_reduce_atomic_strided_io(
    void *keys,
    void *values,
    void *output,
    int num_items,
    int num_groups,
    CudaGroupedReduceValueType value_type,
    std::size_t keys_offset,
    std::size_t keys_stride,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t output_offset,
    std::size_t output_stride,
    int op,
    void *stream);

std::size_t cub_grouped_reduce_i32_atomic(void *keys,
                                          void *values,
                                          void *output,
                                          int num_items,
                                          int num_groups,
                                          int op,
                                          void *stream);

std::size_t cub_grouped_reduce_i32(void *keys,
                                   void *values,
                                   void *output,
                                   void *offsets,
                                   void *scratch,
                                   void *cursor,
                                   int num_items,
                                   int num_groups,
                                   int op,
                                   void *stream,
                                   void *owner);

std::size_t cub_grouped_reduce(void *keys,
                               void *values,
                               void *output,
                               void *offsets,
                               void *scratch,
                               void *cursor,
                               int num_items,
                               int num_groups,
                               CudaGroupedReduceValueType value_type,
                               int op,
                               void *stream,
                               void *owner);

void cub_grouped_reduce_clear_cache(void *owner);

std::size_t cub_grouped_reduce_cached_bytes(void *owner);

}  // namespace taichi::lang::cuda
