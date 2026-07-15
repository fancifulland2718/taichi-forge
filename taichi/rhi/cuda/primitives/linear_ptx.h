#pragma once

#include <cstddef>

namespace taichi::lang::cuda {

// Stable value identifiers shared by the low-PTX-ISA Driver provider and the
// optional Toolkit reference bridge. Keep numeric values aligned with
// PrimitiveValueType in Program.
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

bool driver_scatter_add_available();

std::size_t driver_scatter_add(void *src,
                               void *indices,
                               void *dst,
                               int num_items,
                               int index_bound,
                               CudaScatterAddValueType value_type);

}  // namespace taichi::lang::cuda
