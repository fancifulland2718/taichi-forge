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

}  // namespace taichi::lang::cuda
