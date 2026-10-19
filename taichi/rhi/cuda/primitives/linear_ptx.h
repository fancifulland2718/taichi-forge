#pragma once

#include <cstddef>
#include <cstdint>

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

enum class CudaCheckOp : int {
  nonzero = 0,
  zero = 1,
  nan = 2,
  inf = 3,
  not_finite = 4,
  index_oob = 5,
};

enum class CudaMetricOp : int {
  max_abs = 0,
  max_abs_delta = 1,
};

bool driver_transform_available();

// Resolve immutable code before CUDA capture. Fixed recordings use the explicit
// stream entry; ordinary Program submission retains the default-stream wrapper.
void driver_prepare_transform(CudaTransformValueType value_type);
void driver_prepare_transform_strided(CudaTransformValueType value_type);

std::size_t driver_transform_affine_strided_on_stream(
    void *src, void *dst, std::uint32_t num_items,
    CudaTransformValueType value_type, std::size_t src_offset,
    std::size_t src_stride, std::size_t dst_offset, std::size_t dst_stride,
    double scale, double bias, void *stream);

std::size_t driver_transform_affine_on_stream(void *src,
                                             void *dst,
                                             std::uint32_t num_items,
                                             CudaTransformValueType value_type,
                                             double scale,
                                             double bias,
                                             void *stream);

std::size_t driver_transform_affine(void *src,
                                    void *dst,
                                    int num_items,
                                    CudaTransformValueType value_type,
                                    double scale,
                                    double bias);

std::size_t driver_transform_affine_strided(
    void *src,
    void *dst,
    int num_items,
    CudaTransformValueType value_type,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride,
    double scale,
    double bias);

std::size_t driver_transform_affine_packed_strided(
    void *src,
    void *dst,
    int num_items,
    int lane_count,
    CudaTransformValueType value_type,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride,
    double scale,
    double bias);

bool driver_indexed_copy_available();

// Resolve immutable code before capture. The fixed entry uses the shader's
// unsigned count ABI and the owner's stream; ordinary callers keep their APIs.
void driver_prepare_indexed_copy(CudaIndexedCopyOp op, std::uint32_t item_words);

std::size_t driver_indexed_copy_on_stream(void *src,
                                        void *indices,
                                        void *dst,
                                        std::uint32_t num_items,
                                        std::uint32_t index_bound,
                                        std::uint32_t item_words,
                                        CudaIndexedCopyOp op,
                                        void *stream);

std::size_t driver_indexed_copy(void *src,
                                void *indices,
                                void *dst,
                                int num_items,
                                int index_bound,
                                int item_words,
                                CudaIndexedCopyOp op,
                                void *stream = nullptr);

std::size_t driver_indexed_copy(void *src,
                                void *indices,
                                void *dst,
                                int num_items,
                                int index_bound,
                                CudaIndexedCopyOp op);

std::size_t driver_indexed_copy_strided(void *src,
                                        void *indices,
                                        void *dst,
                                        int num_items,
                                        int index_bound,
                                        int item_words,
                                        std::size_t src_offset_words,
                                        std::size_t src_stride_words,
                                        std::size_t dst_offset_words,
                                        std::size_t dst_stride_words,
                                        CudaIndexedCopyOp op,
                                        void *stream = nullptr);

bool driver_scatter_add_available();

std::size_t driver_scatter_add(void *src,
                               void *indices,
                               void *dst,
                               int num_items,
                               int index_bound,
                               CudaScatterAddValueType value_type);

bool driver_check_count_available();

// Prepare the existing diagnostic module before capture. Fixed recordings use
// the kernel's uint32 count ABI and the owner's explicit stream; ordinary
// signed-count callers retain their entry points below.
void driver_prepare_check_count(CudaTransformValueType value_type,
                                CudaCheckOp op);

std::size_t driver_check_count_on_stream(void *values,
                                         void *output,
                                         std::uint32_t num_items,
                                         CudaTransformValueType value_type,
                                         std::size_t offset,
                                         std::size_t stride,
                                         CudaCheckOp op,
                                         int lower,
                                         int upper,
                                         void *stream);

std::size_t driver_check_count(void *values,
                               void *output,
                               int num_items,
                               CudaTransformValueType value_type,
                               std::size_t offset,
                               std::size_t stride,
                               CudaCheckOp op,
                               int lower,
                               int upper,
                               void *stream = nullptr);

bool driver_metric_reduce_available();

void driver_prepare_metric_reduce(CudaTransformValueType value_type,
                                  CudaMetricOp op);

std::size_t driver_metric_reduce_on_stream(void *values,
                                           void *other,
                                           void *output,
                                           std::uint32_t num_items,
                                           CudaTransformValueType value_type,
                                           std::size_t values_offset,
                                           std::size_t values_stride,
                                           std::size_t other_offset,
                                           std::size_t other_stride,
                                           CudaMetricOp op,
                                           void *stream);

std::size_t driver_metric_reduce(void *values,
                                 void *other,
                                 void *output,
                                 int num_items,
                                 CudaTransformValueType value_type,
                                 std::size_t values_offset,
                                 std::size_t values_stride,
                                 std::size_t other_offset,
                                 std::size_t other_stride,
                                 CudaMetricOp op,
                                 void *stream = nullptr);

}  // namespace taichi::lang::cuda
