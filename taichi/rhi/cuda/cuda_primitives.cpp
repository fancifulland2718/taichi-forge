#include "taichi/rhi/cuda/cuda_primitives.h"

#include "taichi/common/core.h"
#include "taichi/common/dynamic_loader.h"
#include "taichi/rhi/cuda/cuda_capability.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_driver.h"

#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace taichi::lang::cuda {

#if defined(TI_WITH_CUDA_TOOLKIT)
std::size_t cub_radix_sort_impl(void *keys,
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
void cub_radix_sort_clear_cache_impl(void *owner);
std::size_t cub_radix_sort_cached_bytes_impl(void *owner);
std::size_t cub_inclusive_scan_impl(void *data,
                                    int num_items,
                                    CubScanValueType value_type,
                                    void *stream,
                                    void *owner);
std::size_t cub_inclusive_reverse_scan_impl(void *data,
                                            int num_items,
                                            CubScanValueType value_type,
                                            void *stream,
                                            void *owner);
std::size_t cub_inclusive_scan_strided_impl(void *data,
                                            int num_items,
                                            CubScanValueType value_type,
                                            std::size_t offset,
                                            std::size_t stride,
                                            void *stream,
                                            void *owner);
std::size_t cub_inclusive_reverse_scan_strided_impl(void *data,
                                                    int num_items,
                                                    CubScanValueType value_type,
                                                    std::size_t offset,
                                                    std::size_t stride,
                                                    void *stream,
                                                    void *owner);
void cub_inclusive_scan_clear_cache_impl(void *owner);
std::size_t cub_inclusive_scan_cached_bytes_impl(void *owner);
std::size_t cub_select_flagged_impl(void *values,
                                    void *flags,
                                    void *output,
                                    void *count,
                                    int num_items,
                                    CubSelectValueType value_type,
                                    int item_words,
                                    void *stream,
                                    void *owner);
void cub_select_clear_cache_impl(void *owner);
std::size_t cub_select_cached_bytes_impl(void *owner);
std::size_t cub_histogram_even_impl(void *values,
                                    void *bins,
                                    int num_items,
                                    int num_bins,
                                    CubHistogramValueType value_type,
                                    CubHistogramBinType bin_type,
                                    void *stream,
                                    void *owner);
void cub_histogram_clear_cache_impl(void *owner);
std::size_t cub_histogram_cached_bytes_impl(void *owner);
std::size_t cub_reduce_impl(void *values,
                            void *output,
                            int num_items,
                            CubReduceValueType value_type,
                            CubReduceOp op,
                            void *stream,
                            void *owner);
std::size_t cub_reduce_strided_impl(void *values,
                                    void *output,
                                    int num_items,
                                    CubReduceValueType value_type,
                                    std::size_t offset,
                                    std::size_t stride,
                                    CubReduceOp op,
                                    void *stream,
                                    void *owner);
void cub_reduce_clear_cache_impl(void *owner);
std::size_t cub_reduce_cached_bytes_impl(void *owner);
std::size_t cub_check_count_impl(void *values,
                                 void *output,
                                 int num_items,
                                 CubReduceValueType value_type,
                                 std::size_t offset,
                                 std::size_t stride,
                                 CudaCheckOp op,
                                 int lower,
                                 int upper,
                                 void *stream,
                                 void *owner);
void cub_check_count_clear_cache_impl(void *owner);
std::size_t cub_check_count_cached_bytes_impl(void *owner);
std::size_t cub_metric_reduce_impl(void *values,
                                   void *other,
                                   void *output,
                                   int num_items,
                                   CubReduceValueType value_type,
                                   std::size_t values_offset,
                                   std::size_t values_stride,
                                   std::size_t other_offset,
                                   std::size_t other_stride,
                                   CudaMetricOp op,
                                   void *stream,
                                   void *owner);
void cub_metric_reduce_clear_cache_impl(void *owner);
std::size_t cub_metric_reduce_cached_bytes_impl(void *owner);
std::size_t cub_scatter_add_impl(void *src,
                                 void *indices,
                                 void *dst,
                                 int num_items,
                                 int index_bound,
                                 CudaScatterAddValueType value_type,
                                 void *stream);
std::size_t cub_scatter_add_strided_impl(void *src,
                                         void *indices,
                                         void *dst,
                                         int num_items,
                                         int index_bound,
                                         CudaScatterAddValueType value_type,
                                         std::size_t offset,
                                         std::size_t stride,
                                         void *stream);
std::size_t cub_scatter_add_strided_io_impl(void *src,
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
std::size_t cub_scatter_add_packed_strided_io_impl(
    void *src,
    void *indices,
    void *dst,
    int num_items,
    int index_bound,
    int lane_count,
    CudaScatterAddValueType value_type,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride,
    void *stream);
std::size_t cub_indexed_copy_impl(void *src,
                                  void *indices,
                                  void *dst,
                                  int num_items,
                                  int index_bound,
                                  int item_words,
                                  CudaIndexedCopyOp op,
                                  void *stream);
std::size_t cub_indexed_copy_strided_impl(void *src,
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
                                          void *stream);
std::size_t cub_gather_add_impl(void *src,
                                void *indices,
                                void *dst,
                                int num_items,
                                int index_bound,
                                CudaTransformValueType value_type,
                                void *stream);
std::size_t cub_gather_add_strided_impl(void *src,
                                        void *indices,
                                        void *dst,
                                        int num_items,
                                        int index_bound,
                                        CudaTransformValueType value_type,
                                        std::size_t src_offset,
                                        std::size_t src_stride,
                                        std::size_t dst_offset,
                                        std::size_t dst_stride,
                                        void *stream);
std::size_t cub_transform_affine_impl(void *src,
                                      void *dst,
                                      int num_items,
                                      CudaTransformValueType value_type,
                                      double scale,
                                      double bias,
                                      void *stream);
std::size_t cub_transform_affine_strided_impl(void *src,
                                              void *dst,
                                              int num_items,
                                              CudaTransformValueType value_type,
                                              std::size_t offset,
                                              std::size_t stride,
                                              double scale,
                                              double bias,
                                              void *stream);
std::size_t cub_transform_affine_strided_to_strided_impl(
    void *src,
    void *dst,
    int num_items,
    CudaTransformValueType value_type,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride,
    double scale,
    double bias,
    void *stream);
std::size_t cub_transform_affine_packed_strided_impl(
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
    double bias,
    void *stream);
std::size_t cub_add_merge_impl(void *src,
                               void *dst,
                               int num_items,
                               CudaTransformValueType value_type,
                               void *stream);
std::size_t cub_add_merge_strided_impl(void *src,
                                       void *dst,
                                       int num_items,
                                       CudaTransformValueType value_type,
                                       std::size_t src_offset,
                                       std::size_t src_stride,
                                       std::size_t dst_offset,
                                       std::size_t dst_stride,
                                       void *stream);
std::size_t cub_add_scaled_impl(void *src,
                                void *dst,
                                int num_items,
                                CudaTransformValueType value_type,
                                double scale,
                                void *stream);
std::size_t cub_add_scaled_strided_impl(void *src,
                                        void *dst,
                                        int num_items,
                                        CudaTransformValueType value_type,
                                        std::size_t src_offset,
                                        std::size_t src_stride,
                                        std::size_t dst_offset,
                                        std::size_t dst_stride,
                                        double scale,
                                        void *stream);
std::size_t cub_bucket_builder_i32_impl(void *keys,
                                        void *values,
                                        void *offsets,
                                        void *output,
                                        void *cursor,
                                        int num_items,
                                        int num_bins,
                                        void *stream,
                                        void *owner);
std::size_t cub_bucket_builder_impl(void *keys,
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
void cub_bucket_builder_clear_cache_impl(void *owner);
std::size_t cub_bucket_builder_cached_bytes_impl(void *owner);
std::size_t cub_grouped_reduce_i32_impl(void *keys,
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
std::size_t cub_grouped_reduce_impl(void *keys,
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
std::size_t cub_grouped_reduce_strided_io_impl(
    void *keys,
    void *values,
    void *output,
    void *offsets,
    void *scratch,
    void *cursor,
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
    void *stream,
    void *owner);
std::size_t cub_grouped_reduce_atomic_impl(
    void *keys,
    void *values,
    void *output,
    int num_items,
    int num_groups,
    CudaGroupedReduceValueType value_type,
    int op,
    void *stream);
std::size_t cub_grouped_reduce_atomic_strided_impl(
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
std::size_t cub_grouped_reduce_atomic_strided_io_impl(
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
void cub_grouped_reduce_clear_cache_impl(void *owner);
std::size_t cub_grouped_reduce_cached_bytes_impl(void *owner);
#endif

namespace {

#if defined(TI_WITH_CUDA_TOOLKIT)

#if defined(TI_CUDA_CUB_SORT_DYNAMIC_CUDART)

std::unique_ptr<taichi::DynamicLoader> cudart_loader;
std::once_flag cudart_load_once;
bool cudart_loaded{false};
std::string cudart_load_error;

bool cuda_toolkit_driver_compatible_for_cub_sort() {
#if defined(TI_CUDA_TOOLKIT_VERSION_MAJOR) && \
    defined(TI_CUDA_TOOLKIT_VERSION_MINOR)
  auto &driver = CUDADriver::get_instance_without_context();
  if (!driver.detected()) {
    cudart_load_error =
        "CUDA CUB native primitives require a detectable CUDA driver.";
    return false;
  }
  constexpr int kToolkitMajor = TI_CUDA_TOOLKIT_VERSION_MAJOR;
  const int driver_major = driver.get_version_major();
  const int driver_minor = driver.get_version_minor();
  if (!detail::supports_cuda_toolkit_major(driver_major, kToolkitMajor)) {
    cudart_load_error = fmt::format(
        "CUDA CUB native primitives were built with CUDA {}.{}, but the "
        "NVIDIA driver reports CUDA {}.{}. A CUDA {}-compatible driver or "
        "newer is required.",
        kToolkitMajor, TI_CUDA_TOOLKIT_VERSION_MINOR, driver_major,
        driver_minor, kToolkitMajor);
    TI_TRACE("{}", cudart_load_error);
    return false;
  }
#endif
  return true;
}

bool try_load_bundled_cudart(const std::string &candidate) {
  if (taichi::DynamicLoader::check_lib_loaded(candidate)) {
    cudart_loader = std::make_unique<taichi::DynamicLoader>(candidate);
    return cudart_loader->loaded();
  }
  auto loader = std::make_unique<taichi::DynamicLoader>(candidate);
  if (!loader->loaded()) {
    return false;
  }
  cudart_loader = std::move(loader);
  return true;
}

void load_cudart_for_cub_sort_once() {
  const char *bundled_path =
      std::getenv("TI_CUDA_CUB_SORT_BUNDLED_CUDART_PATH");
  if (bundled_path != nullptr && bundled_path[0] != '\0') {
    if (try_load_bundled_cudart(bundled_path)) {
      cudart_loaded = true;
      TI_TRACE("CUDA CUB sort runtime loaded from {}", bundled_path);
      return;
    }
  }

#if defined(TI_PLATFORM_WINDOWS)
  cudart_load_error = fmt::format(
      "CUDA CUB sort could not load the bundled {} from "
      "taichi-forge-runtime.",
      TI_CUDA_CUB_SORT_CUDART_DLL);
#else
  cudart_load_error =
      "CUDA CUB sort could not load the bundled libcudart.so from "
      "taichi-forge-runtime.";
#endif
}

bool ensure_cudart_for_cub_sort() {
  if (!cuda_toolkit_driver_compatible_for_cub_sort()) {
    return false;
  }
  std::call_once(cudart_load_once, load_cudart_for_cub_sort_once);
  return cudart_loaded;
}

const std::string &cudart_error() {
  return cudart_load_error;
}

#else

bool ensure_cudart_for_cub_sort() {
  return true;
}

const std::string &cudart_error() {
  static const std::string empty;
  return empty;
}

#endif

#endif

}  // namespace

bool cub_transform_available() {
#if defined(TI_WITH_CUDA_TOOLKIT)
  return ensure_cudart_for_cub_sort();
#else
  return false;
#endif
}

std::size_t cub_transform_affine(void *src,
                                 void *dst,
                                 int num_items,
                                 CudaTransformValueType value_type,
                                 double scale,
                                 double bias,
                                 void *stream) {
  TI_ERROR_IF(num_items < 0,
              "CUDA toolkit transform expects non-negative num_items.");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_transform_affine_impl(src, dst, num_items, value_type, scale, bias,
                                   stream);
#else
  TI_ERROR(
      "CUDA transform requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t cub_transform_affine_strided(void *src,
                                         void *dst,
                                         int num_items,
                                         CudaTransformValueType value_type,
                                         std::size_t offset,
                                         std::size_t stride,
                                         double scale,
                                         double bias,
                                         void *stream) {
  TI_ERROR_IF(num_items < 0,
              "CUDA toolkit strided transform expects non-negative num_items.");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_transform_affine_strided_impl(src, dst, num_items, value_type,
                                           offset, stride, scale, bias, stream);
#else
  TI_ERROR(
      "CUDA strided transform requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t cub_transform_affine_strided_to_strided(
    void *src,
    void *dst,
    int num_items,
    CudaTransformValueType value_type,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride,
    double scale,
    double bias,
    void *stream) {
  TI_ERROR_IF(num_items < 0,
              "CUDA toolkit strided transform expects non-negative num_items.");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_transform_affine_strided_to_strided_impl(
      src, dst, num_items, value_type, src_offset, src_stride, dst_offset,
      dst_stride, scale, bias, stream);
#else
  TI_ERROR(
      "CUDA strided transform requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t cub_transform_affine_packed_strided(
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
    double bias,
    void *stream) {
  TI_ERROR_IF(num_items < 0,
              "CUDA toolkit packed transform expects non-negative num_items.");
  TI_ERROR_IF(lane_count <= 0,
              "CUDA toolkit packed transform expects positive lane_count.");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_transform_affine_packed_strided_impl(
      src, dst, num_items, lane_count, value_type, src_offset, src_stride,
      dst_offset, dst_stride, scale, bias, stream);
#else
  TI_ERROR(
      "CUDA packed strided transform requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

bool cub_add_merge_available() {
#if defined(TI_WITH_CUDA_TOOLKIT)
  return ensure_cudart_for_cub_sort();
#else
  return false;
#endif
}

std::size_t cub_add_merge(void *src,
                          void *dst,
                          int num_items,
                          CudaTransformValueType value_type,
                          void *stream) {
  TI_ERROR_IF(num_items < 0,
              "CUDA toolkit add-merge expects non-negative num_items.");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_add_merge_impl(src, dst, num_items, value_type, stream);
#else
  TI_ERROR(
      "CUDA add-merge requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t cub_add_merge_strided(void *src,
                                  void *dst,
                                  int num_items,
                                  CudaTransformValueType value_type,
                                  std::size_t src_offset,
                                  std::size_t src_stride,
                                  std::size_t dst_offset,
                                  std::size_t dst_stride,
                                  void *stream) {
  TI_ERROR_IF(num_items < 0,
              "CUDA toolkit strided add-merge expects non-negative num_items.");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_add_merge_strided_impl(src, dst, num_items, value_type, src_offset,
                                    src_stride, dst_offset, dst_stride, stream);
#else
  TI_ERROR(
      "CUDA strided add-merge requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t cub_add_scaled(void *src,
                           void *dst,
                           int num_items,
                           CudaTransformValueType value_type,
                           double scale,
                           void *stream) {
  TI_ERROR_IF(num_items < 0,
              "CUDA toolkit scaled-add expects non-negative num_items.");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_add_scaled_impl(src, dst, num_items, value_type, scale, stream);
#else
  TI_ERROR(
      "CUDA scaled-add requires building Taichi with TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t cub_add_scaled_strided(void *src,
                                   void *dst,
                                   int num_items,
                                   CudaTransformValueType value_type,
                                   std::size_t src_offset,
                                   std::size_t src_stride,
                                   std::size_t dst_offset,
                                   std::size_t dst_stride,
                                   double scale,
                                   void *stream) {
  TI_ERROR_IF(num_items < 0,
              "CUDA toolkit strided scaled-add expects non-negative "
              "num_items.");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_add_scaled_strided_impl(src, dst, num_items, value_type,
                                     src_offset, src_stride, dst_offset,
                                     dst_stride, scale, stream);
#else
  TI_ERROR(
      "CUDA strided scaled-add requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

bool cub_indexed_copy_available() {
#if defined(TI_WITH_CUDA_TOOLKIT)
  return ensure_cudart_for_cub_sort();
#else
  return false;
#endif
}

std::size_t cub_indexed_copy(void *src,
                             void *indices,
                             void *dst,
                             int num_items,
                             int index_bound,
                             int item_words,
                             CudaIndexedCopyOp op,
                             void *stream) {
  TI_ERROR_IF(num_items < 0,
              "CUDA toolkit indexed-copy expects non-negative num_items.");
  TI_ERROR_IF(index_bound < 0,
              "CUDA toolkit indexed-copy expects non-negative index_bound.");
  TI_ERROR_IF(item_words <= 0,
              "CUDA toolkit indexed-copy expects at least one 32-bit word per "
              "item.");
  TI_ERROR_IF(num_items > std::numeric_limits<int>::max() / item_words,
              "CUDA toolkit indexed-copy word count exceeds INT_MAX.");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_indexed_copy_impl(src, indices, dst, num_items, index_bound,
                               item_words, op, stream);
#else
  TI_ERROR(
      "CUDA indexed-copy requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t cub_indexed_copy_strided(void *src,
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
                                     void *stream) {
  TI_ERROR_IF(num_items < 0,
              "CUDA toolkit strided indexed-copy expects non-negative "
              "num_items.");
  TI_ERROR_IF(index_bound < 0,
              "CUDA toolkit strided indexed-copy expects non-negative "
              "index_bound.");
  TI_ERROR_IF(item_words <= 0,
              "CUDA toolkit strided indexed-copy expects at least one "
              "32-bit word per item.");
  TI_ERROR_IF(num_items > std::numeric_limits<int>::max() / item_words,
              "CUDA toolkit strided indexed-copy word count exceeds INT_MAX.");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_indexed_copy_strided_impl(
      src, indices, dst, num_items, index_bound, item_words, src_offset_words,
      src_stride_words, dst_offset_words, dst_stride_words, op, stream);
#else
  TI_ERROR(
      "CUDA strided indexed-copy requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t cub_gather_add(void *src,
                           void *indices,
                           void *dst,
                           int num_items,
                           int index_bound,
                           CudaTransformValueType value_type,
                           void *stream) {
  TI_ERROR_IF(num_items < 0,
              "CUDA toolkit gather-add expects non-negative num_items.");
  TI_ERROR_IF(index_bound < 0,
              "CUDA toolkit gather-add expects non-negative index_bound.");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_gather_add_impl(src, indices, dst, num_items, index_bound,
                             value_type, stream);
#else
  TI_ERROR(
      "CUDA gather-add requires building Taichi with TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t cub_gather_add_strided(void *src,
                                   void *indices,
                                   void *dst,
                                   int num_items,
                                   int index_bound,
                                   CudaTransformValueType value_type,
                                   std::size_t src_offset,
                                   std::size_t src_stride,
                                   std::size_t dst_offset,
                                   std::size_t dst_stride,
                                   void *stream) {
  TI_ERROR_IF(num_items < 0,
              "CUDA toolkit strided gather-add expects non-negative "
              "num_items.");
  TI_ERROR_IF(index_bound < 0,
              "CUDA toolkit strided gather-add expects non-negative "
              "index_bound.");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_gather_add_strided_impl(src, indices, dst, num_items, index_bound,
                                     value_type, src_offset, src_stride,
                                     dst_offset, dst_stride, stream);
#else
  TI_ERROR(
      "CUDA strided gather-add requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

bool cub_radix_sort_available() {
#if defined(TI_WITH_CUDA_TOOLKIT)
  return ensure_cudart_for_cub_sort();
#else
  return false;
#endif
}

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
                           void *owner) {
  TI_ERROR_IF(num_items < 0, "CUB sort expects non-negative num_items");
  TI_ERROR_IF(has_values && value_words <= 0,
              "CUB sort expects positive value_words when values are present");
  TI_ERROR_IF(has_values && num_items > 0 &&
                  num_items > std::numeric_limits<int>::max() / value_words,
              "CUB sort value word count exceeds INT_MAX");
  if (num_items <= 1) {
    return 0;
  }
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_radix_sort_impl(keys, values, num_items, key_type, value_type, mode,
                             nan_policy, has_values, value_words, stream,
                             owner);
#else
  TI_ERROR(
      "CUDA CUB sort requires building Taichi with TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

void cub_radix_sort_clear_cache(void *owner) {
#if defined(TI_WITH_CUDA_TOOLKIT)
  cub_radix_sort_clear_cache_impl(owner);
#endif
}

std::size_t cub_radix_sort_cached_bytes(void *owner) {
#if defined(TI_WITH_CUDA_TOOLKIT)
  return cub_radix_sort_cached_bytes_impl(owner);
#else
  return 0;
#endif
}

bool cub_inclusive_scan_available() {
#if defined(TI_WITH_CUDA_TOOLKIT)
  return ensure_cudart_for_cub_sort();
#else
  return false;
#endif
}

std::size_t cub_inclusive_scan(void *data,
                               int num_items,
                               CubScanValueType value_type,
                               void *stream,
                               void *owner) {
  TI_ERROR_IF(num_items < 0, "CUB scan expects non-negative num_items");
  if (num_items <= 1) {
    return 0;
  }
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_inclusive_scan_impl(data, num_items, value_type, stream, owner);
#else
  TI_ERROR(
      "CUDA CUB scan requires building Taichi with TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t cub_inclusive_reverse_scan(void *data,
                                       int num_items,
                                       CubScanValueType value_type,
                                       void *stream,
                                       void *owner) {
  TI_ERROR_IF(num_items < 0, "CUB reverse scan expects non-negative num_items");
  if (num_items <= 1) {
    return 0;
  }
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_inclusive_reverse_scan_impl(data, num_items, value_type, stream,
                                         owner);
#else
  TI_ERROR(
      "CUDA CUB reverse scan requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t cub_inclusive_scan_strided(void *data,
                                       int num_items,
                                       CubScanValueType value_type,
                                       std::size_t offset,
                                       std::size_t stride,
                                       void *stream,
                                       void *owner) {
  TI_ERROR_IF(num_items < 0, "CUB strided scan expects non-negative num_items");
  if (num_items <= 1) {
    return 0;
  }
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_inclusive_scan_strided_impl(data, num_items, value_type, offset,
                                         stride, stream, owner);
#else
  TI_ERROR(
      "CUDA CUB strided scan requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t cub_inclusive_reverse_scan_strided(void *data,
                                               int num_items,
                                               CubScanValueType value_type,
                                               std::size_t offset,
                                               std::size_t stride,
                                               void *stream,
                                               void *owner) {
  TI_ERROR_IF(num_items < 0,
              "CUB reverse strided scan expects non-negative num_items");
  if (num_items <= 1) {
    return 0;
  }
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_inclusive_reverse_scan_strided_impl(
      data, num_items, value_type, offset, stride, stream, owner);
#else
  TI_ERROR(
      "CUDA CUB reverse strided scan requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

void cub_inclusive_scan_clear_cache(void *owner) {
#if defined(TI_WITH_CUDA_TOOLKIT)
  cub_inclusive_scan_clear_cache_impl(owner);
#endif
}

std::size_t cub_inclusive_scan_cached_bytes(void *owner) {
#if defined(TI_WITH_CUDA_TOOLKIT)
  return cub_inclusive_scan_cached_bytes_impl(owner);
#else
  return 0;
#endif
}

bool cub_select_available() {
#if defined(TI_WITH_CUDA_TOOLKIT)
  return ensure_cudart_for_cub_sort();
#else
  return false;
#endif
}

std::size_t cub_select_flagged(void *values,
                               void *flags,
                               void *output,
                               void *count,
                               int num_items,
                               CubSelectValueType value_type,
                               int item_words,
                               void *stream,
                               void *owner) {
  TI_ERROR_IF(num_items < 0, "CUB select expects non-negative num_items");
  TI_ERROR_IF(item_words <= 0, "CUB select expects positive item_words");
  TI_ERROR_IF(num_items > 0 &&
                  num_items > std::numeric_limits<int>::max() / item_words,
              "CUB select word count exceeds INT_MAX");
  if (num_items <= 0) {
    return 0;
  }
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_select_flagged_impl(values, flags, output, count, num_items,
                                 value_type, item_words, stream, owner);
#else
  TI_ERROR(
      "CUDA CUB select requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

void cub_select_clear_cache(void *owner) {
#if defined(TI_WITH_CUDA_TOOLKIT)
  cub_select_clear_cache_impl(owner);
#endif
}

std::size_t cub_select_cached_bytes(void *owner) {
#if defined(TI_WITH_CUDA_TOOLKIT)
  return cub_select_cached_bytes_impl(owner);
#else
  return 0;
#endif
}

bool cub_histogram_available() {
#if defined(TI_WITH_CUDA_TOOLKIT)
  return ensure_cudart_for_cub_sort();
#else
  return false;
#endif
}

std::size_t cub_histogram_even(void *values,
                               void *bins,
                               int num_items,
                               int num_bins,
                               CubHistogramValueType value_type,
                               CubHistogramBinType bin_type,
                               void *stream,
                               void *owner) {
  TI_ERROR_IF(num_items < 0, "CUB histogram expects non-negative num_items");
  TI_ERROR_IF(num_bins <= 0, "CUB histogram expects positive num_bins");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_histogram_even_impl(values, bins, num_items, num_bins, value_type,
                                 bin_type, stream, owner);
#else
  TI_ERROR(
      "CUDA CUB histogram requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

void cub_histogram_clear_cache(void *owner) {
#if defined(TI_WITH_CUDA_TOOLKIT)
  cub_histogram_clear_cache_impl(owner);
#endif
}

std::size_t cub_histogram_cached_bytes(void *owner) {
#if defined(TI_WITH_CUDA_TOOLKIT)
  return cub_histogram_cached_bytes_impl(owner);
#else
  return 0;
#endif
}

bool cub_reduce_available() {
#if defined(TI_WITH_CUDA_TOOLKIT)
  return ensure_cudart_for_cub_sort();
#else
  return false;
#endif
}

std::size_t cub_reduce(void *values,
                       void *output,
                       int num_items,
                       CubReduceValueType value_type,
                       CubReduceOp op,
                       void *stream,
                       void *owner) {
  TI_ERROR_IF(num_items <= 0, "CUB reduce expects positive num_items");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_reduce_impl(values, output, num_items, value_type, op, stream,
                         owner);
#else
  TI_ERROR(
      "CUDA CUB reduce requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t cub_reduce_strided(void *values,
                               void *output,
                               int num_items,
                               CubReduceValueType value_type,
                               std::size_t offset,
                               std::size_t stride,
                               CubReduceOp op,
                               void *stream,
                               void *owner) {
  TI_ERROR_IF(num_items <= 0, "CUB strided reduce expects positive num_items");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_reduce_strided_impl(values, output, num_items, value_type, offset,
                                 stride, op, stream, owner);
#else
  TI_ERROR(
      "CUDA CUB strided reduce requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

void cub_reduce_clear_cache(void *owner) {
#if defined(TI_WITH_CUDA_TOOLKIT)
  cub_reduce_clear_cache_impl(owner);
#endif
}

std::size_t cub_reduce_cached_bytes(void *owner) {
#if defined(TI_WITH_CUDA_TOOLKIT)
  return cub_reduce_cached_bytes_impl(owner);
#else
  return 0;
#endif
}

bool cub_check_count_available() {
#if defined(TI_WITH_CUDA_TOOLKIT)
  return ensure_cudart_for_cub_sort();
#else
  return false;
#endif
}

std::size_t cub_reduce_value_type_size(CubReduceValueType value_type);

std::size_t cub_check_count(void *values,
                            void *output,
                            int num_items,
                            CubReduceValueType value_type,
                            CudaCheckOp op,
                            int lower,
                            int upper,
                            void *stream,
                            void *owner) {
  TI_ERROR_IF(num_items <= 0, "CUB check_count expects positive num_items");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  const std::size_t value_size = cub_reduce_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUB check_count received an unsupported value type.");
  return cub_check_count_impl(values, output, num_items, value_type, 0,
                              value_size, op, lower, upper, stream, owner);
#else
  TI_ERROR(
      "CUDA CUB check_count requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t cub_check_count_strided(void *values,
                                    void *output,
                                    int num_items,
                                    CubReduceValueType value_type,
                                    std::size_t offset,
                                    std::size_t stride,
                                    CudaCheckOp op,
                                    int lower,
                                    int upper,
                                    void *stream,
                                    void *owner) {
  TI_ERROR_IF(num_items <= 0,
              "CUB strided check_count expects positive num_items");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  const std::size_t value_size = cub_reduce_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUB strided check_count received an unsupported value type.");
  TI_ERROR_IF(stride < value_size || offset % value_size != 0 ||
                  stride % value_size != 0,
              "CUB strided check_count received invalid offset/stride.");
  return cub_check_count_impl(values, output, num_items, value_type, offset,
                              stride, op, lower, upper, stream, owner);
#else
  TI_ERROR(
      "CUDA CUB strided check_count requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

void cub_check_count_clear_cache(void *owner) {
#if defined(TI_WITH_CUDA_TOOLKIT)
  cub_check_count_clear_cache_impl(owner);
#endif
}

std::size_t cub_check_count_cached_bytes(void *owner) {
#if defined(TI_WITH_CUDA_TOOLKIT)
  return cub_check_count_cached_bytes_impl(owner);
#else
  return 0;
#endif
}

bool cub_metric_reduce_available() {
#if defined(TI_WITH_CUDA_TOOLKIT)
  return ensure_cudart_for_cub_sort();
#else
  return false;
#endif
}

bool cub_metric_reduce_value_type_available(CubReduceValueType value_type) {
  return value_type == CubReduceValueType::f32 ||
         value_type == CubReduceValueType::f64;
}

std::size_t cub_reduce_value_type_size(CubReduceValueType value_type) {
  switch (value_type) {
    case CubReduceValueType::i32:
    case CubReduceValueType::f32:
    case CubReduceValueType::u32:
      return sizeof(uint32_t);
    case CubReduceValueType::u64:
    case CubReduceValueType::i64:
    case CubReduceValueType::f64:
      return sizeof(uint64_t);
  }
  return 0;
}

std::size_t cub_metric_reduce(void *values,
                              void *other,
                              void *output,
                              int num_items,
                              CubReduceValueType value_type,
                              CudaMetricOp op,
                              void *stream,
                              void *owner) {
  TI_ERROR_IF(num_items <= 0, "CUB metric_reduce expects positive num_items");
  TI_ERROR_IF(!cub_metric_reduce_value_type_available(value_type),
              "CUB metric_reduce currently supports only f32/f64.");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  const std::size_t value_size = cub_reduce_value_type_size(value_type);
  return cub_metric_reduce_impl(values, other, output, num_items, value_type,
                                0, value_size, 0, value_size, op, stream,
                                owner);
#else
  TI_ERROR(
      "CUDA CUB metric_reduce requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t cub_metric_reduce_strided(void *values,
                                      void *other,
                                      void *output,
                                      int num_items,
                                      CubReduceValueType value_type,
                                      std::size_t values_offset,
                                      std::size_t values_stride,
                                      std::size_t other_offset,
                                      std::size_t other_stride,
                                      CudaMetricOp op,
                                      void *stream,
                                      void *owner) {
  TI_ERROR_IF(num_items <= 0,
              "CUB strided metric_reduce expects positive num_items");
  TI_ERROR_IF(!cub_metric_reduce_value_type_available(value_type),
              "CUB metric_reduce currently supports only f32/f64.");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  const std::size_t value_size = cub_reduce_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "CUB strided metric_reduce received an unsupported value type.");
  TI_ERROR_IF(values_stride < value_size ||
                  values_offset % value_size != 0 ||
                  values_stride % value_size != 0 ||
                  other_stride < value_size || other_offset % value_size != 0 ||
                  other_stride % value_size != 0,
              "CUB strided metric_reduce received invalid offset/stride.");
  return cub_metric_reduce_impl(values, other, output, num_items, value_type,
                                values_offset, values_stride, other_offset,
                                other_stride, op, stream, owner);
#else
  TI_ERROR(
      "CUDA CUB strided metric_reduce requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

void cub_metric_reduce_clear_cache(void *owner) {
#if defined(TI_WITH_CUDA_TOOLKIT)
  cub_metric_reduce_clear_cache_impl(owner);
#endif
}

std::size_t cub_metric_reduce_cached_bytes(void *owner) {
#if defined(TI_WITH_CUDA_TOOLKIT)
  return cub_metric_reduce_cached_bytes_impl(owner);
#else
  return 0;
#endif
}

bool cub_scatter_add_available() {
#if defined(TI_WITH_CUDA_TOOLKIT)
  return ensure_cudart_for_cub_sort();
#else
  return false;
#endif
}

std::size_t cub_scatter_add(void *src,
                            void *indices,
                            void *dst,
                            int num_items,
                            int index_bound,
                            CudaScatterAddValueType value_type,
                            void *stream) {
  TI_ERROR_IF(num_items < 0,
              "CUDA toolkit scatter-add expects non-negative num_items.");
  TI_ERROR_IF(index_bound < 0,
              "CUDA toolkit scatter-add expects non-negative index_bound.");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_scatter_add_impl(src, indices, dst, num_items, index_bound,
                              value_type, stream);
#else
  TI_ERROR(
      "CUDA scatter-add requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t cub_scatter_add_strided(void *src,
                                    void *indices,
                                    void *dst,
                                    int num_items,
                                    int index_bound,
                                    CudaScatterAddValueType value_type,
                                    std::size_t offset,
                                    std::size_t stride,
                                    void *stream) {
  TI_ERROR_IF(num_items < 0,
              "CUDA toolkit strided scatter-add expects non-negative num_items.");
  TI_ERROR_IF(index_bound < 0,
              "CUDA toolkit strided scatter-add expects non-negative index_bound.");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_scatter_add_strided_impl(src, indices, dst, num_items,
                                      index_bound, value_type, offset, stride,
                                      stream);
#else
  TI_ERROR(
      "CUDA strided scatter-add requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

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
                                       void *stream) {
  TI_ERROR_IF(num_items < 0,
              "CUDA toolkit strided scatter-add expects non-negative "
              "num_items.");
  TI_ERROR_IF(index_bound < 0,
              "CUDA toolkit strided scatter-add expects non-negative "
              "index_bound.");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_scatter_add_strided_io_impl(
      src, indices, dst, num_items, index_bound, value_type, src_offset,
      src_stride, dst_offset, dst_stride, stream);
#else
  TI_ERROR(
      "CUDA strided scatter-add requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t cub_scatter_add_packed_strided_io(void *src,
                                              void *indices,
                                              void *dst,
                                              int num_items,
                                              int index_bound,
                                              int lane_count,
                                              CudaScatterAddValueType value_type,
                                              std::size_t src_offset,
                                              std::size_t src_stride,
                                              std::size_t dst_offset,
                                              std::size_t dst_stride,
                                              void *stream) {
  TI_ERROR_IF(num_items < 0,
              "CUDA toolkit packed scatter-add expects non-negative "
              "num_items.");
  TI_ERROR_IF(index_bound < 0,
              "CUDA toolkit packed scatter-add expects non-negative "
              "index_bound.");
  TI_ERROR_IF(lane_count <= 0,
              "CUDA toolkit packed scatter-add expects a positive lane_count.");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_scatter_add_packed_strided_io_impl(
      src, indices, dst, num_items, index_bound, lane_count, value_type,
      src_offset, src_stride, dst_offset, dst_stride, stream);
#else
  TI_ERROR(
      "CUDA packed scatter-add requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

bool cub_bucket_builder_available() {
#if defined(TI_WITH_CUDA_TOOLKIT)
  return ensure_cudart_for_cub_sort();
#else
  return false;
#endif
}

std::size_t cub_bucket_builder_i32(void *keys,
                                   void *values,
                                   void *offsets,
                                   void *output,
                                   void *cursor,
                                   int num_items,
                                   int num_bins,
                                   void *stream,
                                   void *owner) {
  TI_ERROR_IF(num_items < 0,
              "CUDA bucket builder expects non-negative num_items.");
  TI_ERROR_IF(num_bins <= 0, "CUDA bucket builder expects positive num_bins.");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_bucket_builder_i32_impl(keys, values, offsets, output, cursor,
                                     num_items, num_bins, stream, owner);
#else
  TI_ERROR(
      "CUDA bucket builder requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

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
                               void *owner) {
  TI_ERROR_IF(num_items < 0,
              "CUDA bucket builder expects non-negative num_items.");
  TI_ERROR_IF(num_bins <= 0, "CUDA bucket builder expects positive num_bins.");
  TI_ERROR_IF(item_words <= 0,
              "CUDA bucket builder expects positive item_words.");
  TI_ERROR_IF(num_items > 0 &&
                  num_items > std::numeric_limits<int>::max() / item_words,
              "CUDA bucket builder word count exceeds INT_MAX.");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_bucket_builder_impl(keys, values, offsets, output, cursor,
                                 num_items, num_bins, value_type, item_words,
                                 stream, owner);
#else
  TI_ERROR(
      "CUDA bucket builder requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

void cub_bucket_builder_clear_cache(void *owner) {
#if defined(TI_WITH_CUDA_TOOLKIT)
  cub_bucket_builder_clear_cache_impl(owner);
#endif
}

std::size_t cub_bucket_builder_cached_bytes(void *owner) {
#if defined(TI_WITH_CUDA_TOOLKIT)
  return cub_bucket_builder_cached_bytes_impl(owner);
#else
  return 0;
#endif
}

bool cub_grouped_reduce_available() {
#if defined(TI_WITH_CUDA_TOOLKIT)
  return ensure_cudart_for_cub_sort();
#else
  return false;
#endif
}

std::size_t cub_grouped_reduce_i32_atomic(void *keys,
                                          void *values,
                                          void *output,
                                          int num_items,
                                          int num_groups,
                                          int op,
                                          void *stream) {
  return cub_grouped_reduce_atomic(keys, values, output, num_items, num_groups,
                                   CudaGroupedReduceValueType::i32, op, stream);
}

std::size_t cub_grouped_reduce_atomic(void *keys,
                                      void *values,
                                      void *output,
                                      int num_items,
                                      int num_groups,
                                      CudaGroupedReduceValueType value_type,
                                      int op,
                                      void *stream) {
  TI_ERROR_IF(num_items < 0,
              "CUDA grouped reduce expects non-negative num_items.");
  TI_ERROR_IF(num_groups <= 0,
              "CUDA grouped reduce expects positive num_groups.");
  TI_ERROR_IF(op != 0, "CUDA grouped reduce currently supports only sum.");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_grouped_reduce_atomic_impl(keys, values, output, num_items,
                                        num_groups, value_type, op, stream);
#else
  TI_ERROR(
      "CUDA grouped reduce requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

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
    void *stream) {
  TI_ERROR_IF(num_items < 0,
              "CUDA strided grouped reduce expects non-negative num_items.");
  TI_ERROR_IF(num_groups <= 0,
              "CUDA strided grouped reduce expects positive num_groups.");
  TI_ERROR_IF(op != 0,
              "CUDA strided grouped reduce currently supports only sum.");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_grouped_reduce_atomic_strided_impl(
      keys, values, output, num_items, num_groups, value_type, offset, stride,
      op, stream);
#else
  TI_ERROR(
      "CUDA strided grouped reduce requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

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
    void *stream) {
  TI_ERROR_IF(num_items < 0,
              "CUDA strided grouped reduce expects non-negative num_items.");
  TI_ERROR_IF(num_groups <= 0,
              "CUDA strided grouped reduce expects positive num_groups.");
  TI_ERROR_IF(op != 0,
              "CUDA strided grouped reduce currently supports only sum.");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_grouped_reduce_atomic_strided_io_impl(
      keys, values, output, num_items, num_groups, value_type, keys_offset,
      keys_stride, values_offset, values_stride, output_offset, output_stride,
      op, stream);
#else
  TI_ERROR(
      "CUDA strided grouped reduce requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

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
                                   void *owner) {
  TI_ERROR_IF(num_items < 0,
              "CUDA grouped reduce expects non-negative num_items.");
  TI_ERROR_IF(num_groups <= 0,
              "CUDA grouped reduce expects positive num_groups.");
  TI_ERROR_IF(op != 0, "CUDA grouped reduce currently supports only sum.");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_grouped_reduce_i32_impl(keys, values, output, offsets, scratch,
                                     cursor, num_items, num_groups, op, stream,
                                     owner);
#else
  TI_ERROR(
      "CUDA grouped reduce requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

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
                               void *owner) {
  TI_ERROR_IF(num_items < 0,
              "CUDA grouped reduce expects non-negative num_items.");
  TI_ERROR_IF(num_groups <= 0,
              "CUDA grouped reduce expects positive num_groups.");
  TI_ERROR_IF(op != 0, "CUDA grouped reduce currently supports only sum.");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_grouped_reduce_impl(keys, values, output, offsets, scratch,
                                 cursor, num_items, num_groups, value_type, op,
                                 stream, owner);
#else
  TI_ERROR(
      "CUDA grouped reduce requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t cub_grouped_reduce_strided_io(
    void *keys,
    void *values,
    void *output,
    void *offsets,
    void *scratch,
    void *cursor,
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
    void *stream,
    void *owner) {
  TI_ERROR_IF(num_items < 0,
              "CUDA strided grouped reduce expects non-negative num_items.");
  TI_ERROR_IF(num_groups <= 0,
              "CUDA strided grouped reduce expects positive num_groups.");
  TI_ERROR_IF(op != 0,
              "CUDA strided grouped reduce currently supports only sum.");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_grouped_reduce_strided_io_impl(
      keys, values, output, offsets, scratch, cursor, num_items, num_groups,
      value_type, keys_offset, keys_stride, values_offset, values_stride,
      output_offset, output_stride, op, stream, owner);
#else
  TI_ERROR(
      "CUDA strided grouped reduce requires building Taichi with "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

void cub_grouped_reduce_clear_cache(void *owner) {
#if defined(TI_WITH_CUDA_TOOLKIT)
  cub_grouped_reduce_clear_cache_impl(owner);
#endif
}

std::size_t cub_grouped_reduce_cached_bytes(void *owner) {
#if defined(TI_WITH_CUDA_TOOLKIT)
  return cub_grouped_reduce_cached_bytes_impl(owner);
#else
  return 0;
#endif
}

}  // namespace taichi::lang::cuda
