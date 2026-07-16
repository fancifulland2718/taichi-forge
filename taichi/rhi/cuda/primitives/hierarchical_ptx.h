#pragma once

#include <cstddef>

#include "taichi/rhi/cuda/primitives/linear_ptx.h"

namespace taichi::lang {
class PrimitiveWorkspaceArena;
}

namespace taichi::lang::cuda {

enum class CudaHierarchicalReduceOp : int {
  sum = 0,
  min = 1,
  max = 2,
};

bool driver_hierarchical_available();

std::size_t driver_inclusive_scan_strided(
    void *data,
    int num_items,
    CudaTransformValueType value_type,
    std::size_t offset,
    std::size_t stride,
    bool reverse,
    void *stream,
    PrimitiveWorkspaceArena *workspace_arena);

std::size_t driver_reduce_strided(void *values,
                                  void *output,
                                  int num_items,
                                  CudaTransformValueType value_type,
                                  std::size_t values_offset,
                                  std::size_t values_stride,
                                  std::size_t output_offset,
                                  std::size_t output_stride,
                                  CudaHierarchicalReduceOp op,
                                  void *stream,
                                  PrimitiveWorkspaceArena *workspace_arena);

std::size_t driver_histogram_strided(void *values,
                                     void *bins,
                                     int num_items,
                                     int num_bins,
                                     CudaTransformValueType value_type,
                                     CudaTransformValueType bin_type,
                                     std::size_t values_offset,
                                     std::size_t values_stride,
                                     std::size_t bins_offset,
                                     std::size_t bins_stride,
                                     void *stream);

std::size_t driver_add_strided(void *src,
                               void *dst,
                               int num_items,
                               CudaTransformValueType value_type,
                               std::size_t src_offset,
                               std::size_t src_stride,
                               std::size_t dst_offset,
                               std::size_t dst_stride,
                               void *stream);

std::size_t driver_add_scaled_strided(void *src,
                                      void *dst,
                                      int num_items,
                                      CudaTransformValueType value_type,
                                      std::size_t src_offset,
                                      std::size_t src_stride,
                                      std::size_t dst_offset,
                                      std::size_t dst_stride,
                                      double scale,
                                      void *stream);

std::size_t driver_scatter_add_strided(void *src,
                                       void *indices,
                                       void *dst,
                                       int num_items,
                                       int index_bound,
                                       CudaTransformValueType value_type,
                                       std::size_t src_offset,
                                       std::size_t src_stride,
                                       std::size_t indices_offset,
                                       std::size_t indices_stride,
                                       std::size_t dst_offset,
                                       std::size_t dst_stride,
                                       void *stream);

std::size_t driver_gather_add_strided(void *src,
                                      void *indices,
                                      void *dst,
                                      int num_items,
                                      int index_bound,
                                      CudaTransformValueType value_type,
                                      std::size_t src_offset,
                                      std::size_t src_stride,
                                      std::size_t indices_offset,
                                      std::size_t indices_stride,
                                      std::size_t dst_offset,
                                      std::size_t dst_stride,
                                      void *stream);

std::size_t driver_zero_strided(void *dst,
                                int num_items,
                                CudaTransformValueType value_type,
                                std::size_t dst_offset,
                                std::size_t dst_stride,
                                void *stream);

std::size_t driver_compact_strided(void *values,
                                   void *flags,
                                   void *output,
                                   void *count,
                                   int num_items,
                                   int item_words,
                                   std::size_t values_offset,
                                   std::size_t values_stride,
                                   std::size_t flags_offset,
                                   std::size_t flags_stride,
                                   std::size_t output_offset,
                                   std::size_t output_stride,
                                   std::size_t count_offset,
                                   void *stream,
                                   PrimitiveWorkspaceArena *workspace_arena);

std::size_t driver_bucket_builder_strided(
    void *keys,
    void *values,
    void *offsets,
    void *output,
    void *cursor,
    int num_items,
    int num_bins,
    int item_words,
    std::size_t keys_offset,
    std::size_t keys_stride,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t offsets_offset,
    std::size_t offsets_stride,
    std::size_t output_offset,
    std::size_t output_stride,
    void *stream,
    PrimitiveWorkspaceArena *workspace_arena);

std::size_t driver_grouped_reduce_strided(void *keys,
                                          void *values,
                                          void *output,
                                          int num_items,
                                          int num_groups,
                                          CudaTransformValueType value_type,
                                          std::size_t keys_offset,
                                          std::size_t keys_stride,
                                          std::size_t values_offset,
                                          std::size_t values_stride,
                                          std::size_t output_offset,
                                          std::size_t output_stride,
                                          void *stream);

}  // namespace taichi::lang::cuda
