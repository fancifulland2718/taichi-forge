#include "taichi/rhi/cuda/cuda_sort.h"

#include "taichi/common/core.h"
#include "taichi/common/dynamic_loader.h"
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

#if defined(TI_WITH_CUDA_TOOLKIT) && defined(TI_CUDA_CUB_SORT_DYNAMIC_CUDART)

std::unique_ptr<taichi::DynamicLoader> cudart_loader;
std::once_flag cudart_load_once;
bool cudart_loaded{false};
std::string cudart_load_error;

void append_cuda_runtime_candidates(std::vector<std::string> &candidates,
                                    const char *root) {
  if (root == nullptr || root[0] == '\0') {
    return;
  }
  std::string base(root);
  while (!base.empty() && (base.back() == '/' || base.back() == '\\')) {
    base.pop_back();
  }
  if (base.empty()) {
    return;
  }
#if defined(TI_PLATFORM_WINDOWS)
  candidates.push_back(base + "\\bin\\" + TI_CUDA_CUB_SORT_CUDART_DLL);
  candidates.push_back(base + "\\bin\\x64\\" + TI_CUDA_CUB_SORT_CUDART_DLL);
#else
  candidates.push_back(base + "/lib64/libcudart.so");
  candidates.push_back(base + "/lib/libcudart.so");
#endif
}

bool try_load_cudart_candidate(const std::string &candidate) {
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
  std::vector<std::string> candidates;
  const char *explicit_path = std::getenv("TI_CUDA_CUB_SORT_CUDART_PATH");
  if (explicit_path != nullptr && explicit_path[0] != '\0') {
    candidates.emplace_back(explicit_path);
  }
#if defined(TI_PLATFORM_WINDOWS)
  candidates.emplace_back(TI_CUDA_CUB_SORT_CUDART_DLL);
#else
  candidates.emplace_back("libcudart.so");
#endif
  append_cuda_runtime_candidates(candidates, std::getenv("CUDA_PATH"));
  append_cuda_runtime_candidates(candidates, std::getenv("CUDA_HOME"));
  append_cuda_runtime_candidates(candidates, std::getenv("CUDA_ROOT"));
#if defined(TI_PLATFORM_WINDOWS)
  append_cuda_runtime_candidates(candidates, std::getenv("CUDA_PATH_V13_2"));
  append_cuda_runtime_candidates(candidates, std::getenv("CUDA_PATH_V13_1"));
  append_cuda_runtime_candidates(candidates, std::getenv("CUDA_PATH_V13_0"));
  append_cuda_runtime_candidates(candidates, std::getenv("CUDA_PATH_V12_9"));
  append_cuda_runtime_candidates(candidates, std::getenv("CUDA_PATH_V12_8"));
  append_cuda_runtime_candidates(candidates, std::getenv("CUDA_PATH_V12_7"));
  append_cuda_runtime_candidates(candidates, std::getenv("CUDA_PATH_V12_6"));
  append_cuda_runtime_candidates(candidates, std::getenv("CUDA_PATH_V12_5"));
  append_cuda_runtime_candidates(candidates, std::getenv("CUDA_PATH_V12_4"));
#endif

  for (const auto &candidate : candidates) {
    if (candidate.empty()) {
      continue;
    }
    if (try_load_cudart_candidate(candidate)) {
      cudart_loaded = true;
      TI_TRACE("CUDA CUB sort runtime loaded from {}", candidate);
      return;
    }
  }

#if defined(TI_PLATFORM_WINDOWS)
  cudart_load_error = fmt::format(
      "CUDA CUB sort could not load {}. Set CUDA_PATH or "
      "TI_CUDA_CUB_SORT_CUDART_PATH to the CUDA runtime DLL.",
      TI_CUDA_CUB_SORT_CUDART_DLL);
#else
  cudart_load_error =
      "CUDA CUB sort could not load libcudart.so. Set CUDA_PATH, CUDA_HOME, "
      "or TI_CUDA_CUB_SORT_CUDART_PATH.";
#endif
}

bool ensure_cudart_for_cub_sort() {
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

const char kCudaTransformPtx[] = R"ptx(
.version 6.0
.target sm_50
.address_size 64

.visible .entry transform_i32_affine(
    .param .u64 src_param,
    .param .u64 dst_param,
    .param .u32 n_param,
    .param .u32 scale_param,
    .param .u32 bias_param
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [src_param];
    ld.param.u64 %rd2, [dst_param];
    ld.param.u32 %r1, [n_param];
    ld.param.u32 %r2, [scale_param];
    ld.param.u32 %r3, [bias_param];

    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %ntid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.u32 %r7, %r4, %r5, %r6;
    setp.ge.u32 %p1, %r7, %r1;
    @%p1 bra DONE_I32;

    mul.wide.u32 %rd3, %r7, 4;
    add.u64 %rd4, %rd1, %rd3;
    add.u64 %rd5, %rd2, %rd3;
    ld.global.u32 %r8, [%rd4];
    mul.lo.u32 %r9, %r8, %r2;
    add.u32 %r9, %r9, %r3;
    st.global.u32 [%rd5], %r9;

DONE_I32:
    ret;
}

.visible .entry transform_f32_affine(
    .param .u64 src_param,
    .param .u64 dst_param,
    .param .u32 n_param,
    .param .f32 scale_param,
    .param .f32 bias_param
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<5>;

    ld.param.u64 %rd1, [src_param];
    ld.param.u64 %rd2, [dst_param];
    ld.param.u32 %r1, [n_param];
    ld.param.f32 %f1, [scale_param];
    ld.param.f32 %f2, [bias_param];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE_F32;

    mul.wide.u32 %rd3, %r5, 4;
    add.u64 %rd4, %rd1, %rd3;
    add.u64 %rd5, %rd2, %rd3;
    ld.global.f32 %f3, [%rd4];
    mul.rn.f32 %f4, %f3, %f1;
    add.rn.f32 %f4, %f4, %f2;
    st.global.f32 [%rd5], %f4;

DONE_F32:
    ret;
}
)ptx";

std::once_flag transform_module_once;
void *transform_module{nullptr};
void *transform_i32_func{nullptr};
void *transform_f32_func{nullptr};

const char kCudaIndexedCopyPtx[] = R"ptx(
.version 6.0
.target sm_50
.address_size 64

.visible .entry gather_u32_by_i32(
    .param .u64 src_param,
    .param .u64 indices_param,
    .param .u64 dst_param,
    .param .u32 n_param,
    .param .u32 bound_param
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<9>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [src_param];
    ld.param.u64 %rd2, [indices_param];
    ld.param.u64 %rd3, [dst_param];
    ld.param.u32 %r1, [n_param];
    ld.param.u32 %r8, [bound_param];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE_GATHER;

    mul.wide.u32 %rd4, %r5, 4;
    add.u64 %rd5, %rd2, %rd4;
    ld.global.u32 %r6, [%rd5];
    setp.ge.u32 %p1, %r6, %r8;
    @%p1 bra INVALID_GATHER;
    mul.wide.u32 %rd6, %r6, 4;
    add.u64 %rd7, %rd1, %rd6;
    ld.global.u32 %r7, [%rd7];
    add.u64 %rd8, %rd3, %rd4;
    st.global.u32 [%rd8], %r7;
    bra DONE_GATHER;

INVALID_GATHER:
    add.u64 %rd8, %rd3, %rd4;
    mov.u32 %r7, 0;
    st.global.u32 [%rd8], %r7;

DONE_GATHER:
    ret;
}

.visible .entry scatter_u32_by_i32(
    .param .u64 src_param,
    .param .u64 indices_param,
    .param .u64 dst_param,
    .param .u32 n_param,
    .param .u32 bound_param
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<9>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [src_param];
    ld.param.u64 %rd2, [indices_param];
    ld.param.u64 %rd3, [dst_param];
    ld.param.u32 %r1, [n_param];
    ld.param.u32 %r8, [bound_param];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE_SCATTER;

    mul.wide.u32 %rd4, %r5, 4;
    add.u64 %rd5, %rd1, %rd4;
    ld.global.u32 %r6, [%rd5];
    add.u64 %rd6, %rd2, %rd4;
    ld.global.u32 %r7, [%rd6];
    setp.ge.u32 %p1, %r7, %r8;
    @%p1 bra DONE_SCATTER;
    mul.wide.u32 %rd7, %r7, 4;
    add.u64 %rd8, %rd3, %rd7;
    st.global.u32 [%rd8], %r6;

DONE_SCATTER:
    ret;
}
)ptx";

std::once_flag indexed_copy_module_once;
void *indexed_copy_module{nullptr};
void *gather_u32_func{nullptr};
void *scatter_u32_func{nullptr};

const char kCudaScatterAddPtx[] = R"ptx(
.version 6.0
.target sm_50
.address_size 64

.visible .entry scatter_add_u32_by_i32(
    .param .u64 src_param,
    .param .u64 indices_param,
    .param .u64 dst_param,
    .param .u32 n_param,
    .param .u32 bound_param
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [src_param];
    ld.param.u64 %rd2, [indices_param];
    ld.param.u64 %rd3, [dst_param];
    ld.param.u32 %r1, [n_param];
    ld.param.u32 %r8, [bound_param];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE_U32;

    mul.wide.u32 %rd4, %r5, 4;
    add.u64 %rd5, %rd2, %rd4;
    ld.global.s32 %r6, [%rd5];
    setp.lt.s32 %p1, %r6, 0;
    @%p1 bra DONE_U32;
    setp.ge.s32 %p2, %r6, %r8;
    @%p2 bra DONE_U32;
    add.u64 %rd6, %rd1, %rd4;
    ld.global.u32 %r7, [%rd6];
    mul.wide.u32 %rd7, %r6, 4;
    add.u64 %rd8, %rd3, %rd7;
    atom.global.add.u32 %r9, [%rd8], %r7;

DONE_U32:
    ret;
}

.visible .entry scatter_add_f32_by_i32(
    .param .u64 src_param,
    .param .u64 indices_param,
    .param .u64 dst_param,
    .param .u32 n_param,
    .param .u32 bound_param
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<9>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<3>;

    ld.param.u64 %rd1, [src_param];
    ld.param.u64 %rd2, [indices_param];
    ld.param.u64 %rd3, [dst_param];
    ld.param.u32 %r1, [n_param];
    ld.param.u32 %r8, [bound_param];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE_F32;

    mul.wide.u32 %rd4, %r5, 4;
    add.u64 %rd5, %rd2, %rd4;
    ld.global.s32 %r6, [%rd5];
    setp.lt.s32 %p1, %r6, 0;
    @%p1 bra DONE_F32;
    setp.ge.s32 %p2, %r6, %r8;
    @%p2 bra DONE_F32;
    add.u64 %rd6, %rd1, %rd4;
    ld.global.f32 %f1, [%rd6];
    mul.wide.u32 %rd7, %r6, 4;
    add.u64 %rd8, %rd3, %rd7;
    atom.global.add.f32 %f2, [%rd8], %f1;

DONE_F32:
    ret;
}
)ptx";

std::once_flag scatter_add_module_once;
void *scatter_add_module{nullptr};
void *scatter_add_u32_func{nullptr};
void *scatter_add_f32_func{nullptr};

void load_transform_module_once() {
  auto &ctx = CUDAContext::get_instance();
  auto context_guard = ctx.get_guard();
  auto &driver = CUDADriver::get_instance();
  driver.module_load_data_ex(&transform_module, kCudaTransformPtx, 0, nullptr,
                             nullptr);
  driver.module_get_function(&transform_i32_func, transform_module,
                             "transform_i32_affine");
  driver.module_get_function(&transform_f32_func, transform_module,
                             "transform_f32_affine");
}

void *cuda_transform_function(CudaTransformValueType value_type) {
  std::call_once(transform_module_once, load_transform_module_once);
  switch (value_type) {
    case CudaTransformValueType::i32:
    case CudaTransformValueType::u32:
      return transform_i32_func;
    case CudaTransformValueType::f32:
      return transform_f32_func;
    case CudaTransformValueType::u64:
    case CudaTransformValueType::i64:
    case CudaTransformValueType::f64:
      TI_ERROR("64-bit CUDA transform requires CUDA toolkit runtime support.");
  }
  TI_ERROR("Unsupported CUDA transform value type.");
  return nullptr;
}

void load_indexed_copy_module_once() {
  auto &ctx = CUDAContext::get_instance();
  auto context_guard = ctx.get_guard();
  auto &driver = CUDADriver::get_instance();
  driver.module_load_data_ex(&indexed_copy_module, kCudaIndexedCopyPtx, 0,
                             nullptr, nullptr);
  driver.module_get_function(&gather_u32_func, indexed_copy_module,
                             "gather_u32_by_i32");
  driver.module_get_function(&scatter_u32_func, indexed_copy_module,
                             "scatter_u32_by_i32");
}

void load_scatter_add_module_once() {
  auto &ctx = CUDAContext::get_instance();
  auto context_guard = ctx.get_guard();
  auto &driver = CUDADriver::get_instance();
  driver.module_load_data_ex(&scatter_add_module, kCudaScatterAddPtx, 0,
                             nullptr, nullptr);
  driver.module_get_function(&scatter_add_u32_func, scatter_add_module,
                             "scatter_add_u32_by_i32");
  driver.module_get_function(&scatter_add_f32_func, scatter_add_module,
                             "scatter_add_f32_by_i32");
}

void *cuda_indexed_copy_function(CudaIndexedCopyOp op) {
  std::call_once(indexed_copy_module_once, load_indexed_copy_module_once);
  switch (op) {
    case CudaIndexedCopyOp::gather:
      return gather_u32_func;
    case CudaIndexedCopyOp::scatter:
      return scatter_u32_func;
  }
  TI_ERROR("Unsupported CUDA indexed copy op.");
  return nullptr;
}

void *cuda_scatter_add_function(CudaScatterAddValueType value_type) {
  std::call_once(scatter_add_module_once, load_scatter_add_module_once);
  switch (value_type) {
    case CudaScatterAddValueType::i32:
    case CudaScatterAddValueType::u32:
      return scatter_add_u32_func;
    case CudaScatterAddValueType::f32:
      return scatter_add_f32_func;
    case CudaScatterAddValueType::u64:
    case CudaScatterAddValueType::i64:
    case CudaScatterAddValueType::f64:
      return nullptr;
  }
  return nullptr;
}

}  // namespace

bool driver_transform_available() {
  return CUDADriver::get_instance_without_context().detected();
}

std::size_t driver_transform_affine(void *src,
                                    void *dst,
                                    int num_items,
                                    CudaTransformValueType value_type,
                                    double scale,
                                    double bias) {
  TI_ERROR_IF(num_items < 0,
              "CUDA driver transform expects non-negative num_items.");
  TI_ERROR_IF(!src || !dst, "CUDA driver transform received a null pointer.");
  if (num_items == 0) {
    return 0;
  }
  constexpr unsigned kBlockDim = 256;
  const unsigned grid_dim =
      static_cast<unsigned>((num_items + kBlockDim - 1) / kBlockDim);
  void *func = cuda_transform_function(value_type);
  void *src_arg = src;
  void *dst_arg = dst;
  uint32_t n_arg = static_cast<uint32_t>(num_items);
  std::vector<void *> args;
  args.reserve(5);
  args.push_back(&src_arg);
  args.push_back(&dst_arg);
  args.push_back(&n_arg);
  uint32_t scale_u32 = 0;
  uint32_t bias_u32 = 0;
  float scale_f32 = 0.0f;
  float bias_f32 = 0.0f;
  if (value_type == CudaTransformValueType::i32) {
    scale_u32 = static_cast<uint32_t>(static_cast<int32_t>(scale));
    bias_u32 = static_cast<uint32_t>(static_cast<int32_t>(bias));
    args.push_back(&scale_u32);
    args.push_back(&bias_u32);
  } else if (value_type == CudaTransformValueType::u32) {
    scale_u32 = static_cast<uint32_t>(scale);
    bias_u32 = static_cast<uint32_t>(bias);
    args.push_back(&scale_u32);
    args.push_back(&bias_u32);
  } else {
    scale_f32 = static_cast<float>(scale);
    bias_f32 = static_cast<float>(bias);
    args.push_back(&scale_f32);
    args.push_back(&bias_f32);
  }
  CUDAContext::get_instance().launch(func, "cuda_transform_affine", args, {},
                                     grid_dim, kBlockDim, 0);
  return 0;
}

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

bool driver_indexed_copy_available() {
  return CUDADriver::get_instance_without_context().detected();
}

std::size_t driver_indexed_copy(void *src,
                                void *indices,
                                void *dst,
                                int num_items,
                                int index_bound,
                                CudaIndexedCopyOp op) {
  TI_ERROR_IF(num_items < 0,
              "CUDA driver indexed copy expects non-negative num_items.");
  TI_ERROR_IF(index_bound < 0,
              "CUDA driver indexed copy expects non-negative index_bound.");
  TI_ERROR_IF(!src || !indices || !dst,
              "CUDA driver indexed copy received a null pointer.");
  if (num_items == 0) {
    return 0;
  }
  constexpr unsigned kBlockDim = 256;
  const unsigned grid_dim =
      static_cast<unsigned>((num_items + kBlockDim - 1) / kBlockDim);
  void *func = cuda_indexed_copy_function(op);
  void *src_arg = src;
  void *indices_arg = indices;
  void *dst_arg = dst;
  uint32_t n_arg = static_cast<uint32_t>(num_items);
  uint32_t bound_arg = static_cast<uint32_t>(index_bound);
  std::vector<void *> args;
  args.reserve(5);
  args.push_back(&src_arg);
  args.push_back(&indices_arg);
  args.push_back(&dst_arg);
  args.push_back(&n_arg);
  args.push_back(&bound_arg);
  CUDAContext::get_instance().launch(func, "cuda_indexed_copy", args, {},
                                     grid_dim, kBlockDim, 0);
  return 0;
}

bool driver_scatter_add_available() {
  return CUDADriver::get_instance_without_context().detected();
}

std::size_t driver_scatter_add(void *src,
                               void *indices,
                               void *dst,
                               int num_items,
                               int index_bound,
                               CudaScatterAddValueType value_type) {
  TI_ERROR_IF(num_items < 0,
              "CUDA driver scatter-add expects non-negative num_items.");
  TI_ERROR_IF(index_bound < 0,
              "CUDA driver scatter-add expects non-negative index_bound.");
  TI_ERROR_IF(!src || !indices || !dst,
              "CUDA driver scatter-add received a null pointer.");
  if (num_items == 0) {
    return 0;
  }
  void *func = cuda_scatter_add_function(value_type);
  TI_ERROR_IF(!func,
              "CUDA driver scatter-add currently supports i32, u32, and f32.");
  constexpr unsigned kBlockDim = 256;
  const unsigned grid_dim =
      static_cast<unsigned>((num_items + kBlockDim - 1) / kBlockDim);
  void *src_arg = src;
  void *indices_arg = indices;
  void *dst_arg = dst;
  uint32_t n_arg = static_cast<uint32_t>(num_items);
  uint32_t bound_arg = static_cast<uint32_t>(index_bound);
  std::vector<void *> args;
  args.reserve(5);
  args.push_back(&src_arg);
  args.push_back(&indices_arg);
  args.push_back(&dst_arg);
  args.push_back(&n_arg);
  args.push_back(&bound_arg);
  CUDAContext::get_instance().launch(func, "cuda_scatter_add", args, {},
                                     grid_dim, kBlockDim, 0);
  return 0;
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
