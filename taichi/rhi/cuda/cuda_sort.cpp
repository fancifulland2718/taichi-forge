#include "taichi/rhi/cuda/cuda_sort.h"

#include "taichi/common/core.h"
#include "taichi/common/dynamic_loader.h"

#include <cstdlib>
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
                                CubSortMode mode,
                                CubSortNanPolicy nan_policy,
                                bool has_values,
                                void *stream,
                                void *owner);
void cub_radix_sort_clear_cache_impl(void *owner);
std::size_t cub_radix_sort_cached_bytes_impl(void *owner);
std::size_t cub_inclusive_scan_impl(void *data,
                                    int num_items,
                                    CubScanValueType value_type,
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
                                    void *stream,
                                    void *owner);
void cub_select_clear_cache_impl(void *owner);
std::size_t cub_select_cached_bytes_impl(void *owner);
std::size_t cub_histogram_even_impl(void *values,
                                    void *bins,
                                    int num_items,
                                    int num_bins,
                                    CubHistogramValueType value_type,
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
void cub_reduce_clear_cache_impl(void *owner);
std::size_t cub_reduce_cached_bytes_impl(void *owner);
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

}  // namespace

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
                           CubSortMode mode,
                           CubSortNanPolicy nan_policy,
                           bool has_values,
                           void *stream,
                           void *owner) {
  TI_ERROR_IF(num_items < 0, "CUB sort expects non-negative num_items");
  if (num_items <= 1) {
    return 0;
  }
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_radix_sort_impl(keys, values, num_items, key_type, mode,
                             nan_policy, has_values, stream, owner);
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
                               void *stream,
                               void *owner) {
  TI_ERROR_IF(num_items < 0, "CUB select expects non-negative num_items");
  if (num_items <= 0) {
    return 0;
  }
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_select_flagged_impl(values, flags, output, count, num_items,
                                 value_type, stream, owner);
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
                               void *stream,
                               void *owner) {
  TI_ERROR_IF(num_items < 0, "CUB histogram expects non-negative num_items");
  TI_ERROR_IF(num_bins <= 0, "CUB histogram expects positive num_bins");
#if defined(TI_WITH_CUDA_TOOLKIT)
  TI_ERROR_IF(!ensure_cudart_for_cub_sort(), "{}", cudart_error());
  return cub_histogram_even_impl(values, bins, num_items, num_bins, value_type,
                                 stream, owner);
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

}  // namespace taichi::lang::cuda
