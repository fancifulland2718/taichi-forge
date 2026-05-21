#include "taichi/rhi/cuda/cuda_sort.h"

#include <cub/cub.cuh>
#include <cuda/iterator>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace taichi::lang::cuda {
namespace {

struct CubSortCache {
  void *keys_out{nullptr};
  void *values_out{nullptr};
  void *temp_storage{nullptr};
  void *key32_a{nullptr};
  void *key32_b{nullptr};
  void *key64_a{nullptr};
  void *key64_b{nullptr};
  void *index_a{nullptr};
  void *index_b{nullptr};
  void *high32{nullptr};
  std::size_t keys_out_bytes{0};
  std::size_t values_out_bytes{0};
  std::size_t temp_storage_bytes{0};
  std::size_t key32_a_bytes{0};
  std::size_t key32_b_bytes{0};
  std::size_t key64_a_bytes{0};
  std::size_t key64_b_bytes{0};
  std::size_t index_a_bytes{0};
  std::size_t index_b_bytes{0};
  std::size_t high32_bytes{0};
  int device_id{-1};

  ~CubSortCache() {
    release_noexcept();
  }

  void release_noexcept() {
    if (temp_storage) {
      cudaFree(temp_storage);
    }
    if (values_out) {
      cudaFree(values_out);
    }
    if (keys_out) {
      cudaFree(keys_out);
    }
    if (key32_a) {
      cudaFree(key32_a);
    }
    if (key32_b) {
      cudaFree(key32_b);
    }
    if (key64_a) {
      cudaFree(key64_a);
    }
    if (key64_b) {
      cudaFree(key64_b);
    }
    if (index_a) {
      cudaFree(index_a);
    }
    if (index_b) {
      cudaFree(index_b);
    }
    if (high32) {
      cudaFree(high32);
    }
    keys_out = nullptr;
    values_out = nullptr;
    temp_storage = nullptr;
    key32_a = nullptr;
    key32_b = nullptr;
    key64_a = nullptr;
    key64_b = nullptr;
    index_a = nullptr;
    index_b = nullptr;
    high32 = nullptr;
    keys_out_bytes = 0;
    values_out_bytes = 0;
    temp_storage_bytes = 0;
    key32_a_bytes = 0;
    key32_b_bytes = 0;
    key64_a_bytes = 0;
    key64_b_bytes = 0;
    index_a_bytes = 0;
    index_b_bytes = 0;
    high32_bytes = 0;
    device_id = -1;
  }

  std::size_t allocated_bytes() const {
    return keys_out_bytes + values_out_bytes + temp_storage_bytes +
           key32_a_bytes + key32_b_bytes + key64_a_bytes + key64_b_bytes +
           index_a_bytes + index_b_bytes + high32_bytes;
  }
};

struct CubScanCache {
  void *temp_storage{nullptr};
  std::size_t temp_storage_bytes{0};
  int device_id{-1};

  ~CubScanCache() {
    release_noexcept();
  }

  void release_noexcept() {
    if (temp_storage) {
      cudaFree(temp_storage);
    }
    temp_storage = nullptr;
    temp_storage_bytes = 0;
    device_id = -1;
  }

  std::size_t allocated_bytes() const {
    return temp_storage_bytes;
  }
};

struct CubSelectCache {
  void *temp_storage{nullptr};
  void *prefix{nullptr};
  std::size_t temp_storage_bytes{0};
  std::size_t prefix_bytes{0};
  int device_id{-1};

  ~CubSelectCache() {
    release_noexcept();
  }

  void release_noexcept() {
    if (temp_storage) {
      cudaFree(temp_storage);
    }
    if (prefix) {
      cudaFree(prefix);
    }
    temp_storage = nullptr;
    prefix = nullptr;
    temp_storage_bytes = 0;
    prefix_bytes = 0;
    device_id = -1;
  }

  std::size_t allocated_bytes() const {
    return temp_storage_bytes + prefix_bytes;
  }
};

struct CubHistogramCache {
  void *temp_storage{nullptr};
  std::size_t temp_storage_bytes{0};
  int device_id{-1};

  ~CubHistogramCache() {
    release_noexcept();
  }

  void release_noexcept() {
    if (temp_storage) {
      cudaFree(temp_storage);
    }
    temp_storage = nullptr;
    temp_storage_bytes = 0;
    device_id = -1;
  }

  std::size_t allocated_bytes() const {
    return temp_storage_bytes;
  }
};

struct CubReduceCache {
  void *temp_storage{nullptr};
  std::size_t temp_storage_bytes{0};
  int device_id{-1};

  ~CubReduceCache() {
    release_noexcept();
  }

  void release_noexcept() {
    if (temp_storage) {
      cudaFree(temp_storage);
    }
    temp_storage = nullptr;
    temp_storage_bytes = 0;
    device_id = -1;
  }

  std::size_t allocated_bytes() const {
    return temp_storage_bytes;
  }
};

struct CubBucketBuilderCache {
  void *temp_storage{nullptr};
  std::size_t temp_storage_bytes{0};
  int device_id{-1};

  ~CubBucketBuilderCache() {
    release_noexcept();
  }

  void release_noexcept() {
    if (temp_storage) {
      cudaFree(temp_storage);
    }
    temp_storage = nullptr;
    temp_storage_bytes = 0;
    device_id = -1;
  }

  std::size_t allocated_bytes() const {
    return temp_storage_bytes;
  }
};

struct CubGroupedReduceCache {
  void *temp_storage{nullptr};
  std::size_t temp_storage_bytes{0};
  int device_id{-1};

  ~CubGroupedReduceCache() {
    release_noexcept();
  }

  void release_noexcept() {
    if (temp_storage) {
      cudaFree(temp_storage);
    }
    temp_storage = nullptr;
    temp_storage_bytes = 0;
    device_id = -1;
  }

  std::size_t allocated_bytes() const {
    return temp_storage_bytes;
  }
};

std::mutex &get_cache_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<void *, std::unique_ptr<CubSortCache>> &get_caches() {
  static std::unordered_map<void *, std::unique_ptr<CubSortCache>> caches;
  return caches;
}

std::unordered_map<void *, std::unique_ptr<CubScanCache>> &get_scan_caches() {
  static std::unordered_map<void *, std::unique_ptr<CubScanCache>> caches;
  return caches;
}

std::unordered_map<void *, std::unique_ptr<CubSelectCache>> &
get_select_caches() {
  static std::unordered_map<void *, std::unique_ptr<CubSelectCache>> caches;
  return caches;
}

std::unordered_map<void *, std::unique_ptr<CubHistogramCache>> &
get_histogram_caches() {
  static std::unordered_map<void *, std::unique_ptr<CubHistogramCache>> caches;
  return caches;
}

std::unordered_map<void *, std::unique_ptr<CubReduceCache>> &
get_reduce_caches() {
  static std::unordered_map<void *, std::unique_ptr<CubReduceCache>> caches;
  return caches;
}

std::unordered_map<void *, std::unique_ptr<CubBucketBuilderCache>> &
get_bucket_builder_caches() {
  static std::unordered_map<void *, std::unique_ptr<CubBucketBuilderCache>>
      caches;
  return caches;
}

std::unordered_map<void *, std::unique_ptr<CubGroupedReduceCache>> &
get_grouped_reduce_caches() {
  static std::unordered_map<void *, std::unique_ptr<CubGroupedReduceCache>>
      caches;
  return caches;
}

CubSortCache &get_cache(void *owner) {
  static int fallback_owner = 0;
  if (!owner) {
    owner = &fallback_owner;
  }
  auto &caches = get_caches();
  auto it = caches.find(owner);
  if (it == caches.end()) {
    it = caches.emplace(owner, std::make_unique<CubSortCache>()).first;
  }
  return *it->second;
}

CubScanCache &get_scan_cache(void *owner) {
  static int fallback_owner = 0;
  if (!owner) {
    owner = &fallback_owner;
  }
  auto &caches = get_scan_caches();
  auto it = caches.find(owner);
  if (it == caches.end()) {
    it = caches.emplace(owner, std::make_unique<CubScanCache>()).first;
  }
  return *it->second;
}

CubSelectCache &get_select_cache(void *owner) {
  static int fallback_owner = 0;
  if (!owner) {
    owner = &fallback_owner;
  }
  auto &caches = get_select_caches();
  auto it = caches.find(owner);
  if (it == caches.end()) {
    it = caches.emplace(owner, std::make_unique<CubSelectCache>()).first;
  }
  return *it->second;
}

CubHistogramCache &get_histogram_cache(void *owner) {
  static int fallback_owner = 0;
  if (!owner) {
    owner = &fallback_owner;
  }
  auto &caches = get_histogram_caches();
  auto it = caches.find(owner);
  if (it == caches.end()) {
    it = caches.emplace(owner, std::make_unique<CubHistogramCache>()).first;
  }
  return *it->second;
}

CubReduceCache &get_reduce_cache(void *owner) {
  static int fallback_owner = 0;
  if (!owner) {
    owner = &fallback_owner;
  }
  auto &caches = get_reduce_caches();
  auto it = caches.find(owner);
  if (it == caches.end()) {
    it = caches.emplace(owner, std::make_unique<CubReduceCache>()).first;
  }
  return *it->second;
}

CubBucketBuilderCache &get_bucket_builder_cache(void *owner) {
  static int fallback_owner = 0;
  if (!owner) {
    owner = &fallback_owner;
  }
  auto &caches = get_bucket_builder_caches();
  auto it = caches.find(owner);
  if (it == caches.end()) {
    it = caches.emplace(owner, std::make_unique<CubBucketBuilderCache>()).first;
  }
  return *it->second;
}

CubGroupedReduceCache &get_grouped_reduce_cache(void *owner) {
  static int fallback_owner = 0;
  if (!owner) {
    owner = &fallback_owner;
  }
  auto &caches = get_grouped_reduce_caches();
  auto it = caches.find(owner);
  if (it == caches.end()) {
    it = caches.emplace(owner, std::make_unique<CubGroupedReduceCache>()).first;
  }
  return *it->second;
}

void check_cuda(cudaError_t err, const char *expr, const char *file, int line) {
  if (err == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string("CUDA CUB sort failed: ") +
                           cudaGetErrorString(err) + " at " + file + ":" +
                           std::to_string(line) + " (" + expr + ")");
}

#define TI_CUDA_SORT_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

void ensure_device_cache(CubSortCache &cache) {
  int device_id = 0;
  TI_CUDA_SORT_CHECK(cudaGetDevice(&device_id));
  if (cache.device_id == -1) {
    cache.device_id = device_id;
  } else if (cache.device_id != device_id) {
    cache.release_noexcept();
    cache.device_id = device_id;
  }
}

void ensure_device_cache(CubScanCache &cache) {
  int device_id = 0;
  TI_CUDA_SORT_CHECK(cudaGetDevice(&device_id));
  if (cache.device_id == -1) {
    cache.device_id = device_id;
  } else if (cache.device_id != device_id) {
    cache.release_noexcept();
    cache.device_id = device_id;
  }
}

void ensure_device_cache(CubSelectCache &cache) {
  int device_id = 0;
  TI_CUDA_SORT_CHECK(cudaGetDevice(&device_id));
  if (cache.device_id == -1) {
    cache.device_id = device_id;
  } else if (cache.device_id != device_id) {
    cache.release_noexcept();
    cache.device_id = device_id;
  }
}

void ensure_device_cache(CubHistogramCache &cache) {
  int device_id = 0;
  TI_CUDA_SORT_CHECK(cudaGetDevice(&device_id));
  if (cache.device_id == -1) {
    cache.device_id = device_id;
  } else if (cache.device_id != device_id) {
    cache.release_noexcept();
    cache.device_id = device_id;
  }
}

void ensure_device_cache(CubReduceCache &cache) {
  int device_id = 0;
  TI_CUDA_SORT_CHECK(cudaGetDevice(&device_id));
  if (cache.device_id == -1) {
    cache.device_id = device_id;
  } else if (cache.device_id != device_id) {
    cache.release_noexcept();
    cache.device_id = device_id;
  }
}

void ensure_device_cache(CubBucketBuilderCache &cache) {
  int device_id = 0;
  TI_CUDA_SORT_CHECK(cudaGetDevice(&device_id));
  if (cache.device_id == -1) {
    cache.device_id = device_id;
  } else if (cache.device_id != device_id) {
    cache.release_noexcept();
    cache.device_id = device_id;
  }
}

void ensure_device_cache(CubGroupedReduceCache &cache) {
  int device_id = 0;
  TI_CUDA_SORT_CHECK(cudaGetDevice(&device_id));
  if (cache.device_id == -1) {
    cache.device_id = device_id;
  } else if (cache.device_id != device_id) {
    cache.release_noexcept();
    cache.device_id = device_id;
  }
}

void ensure_buffer(void **ptr, std::size_t *capacity, std::size_t required) {
  if (required <= *capacity) {
    return;
  }
  if (*ptr) {
    TI_CUDA_SORT_CHECK(cudaFree(*ptr));
    *ptr = nullptr;
    *capacity = 0;
  }
  TI_CUDA_SORT_CHECK(cudaMalloc(ptr, required));
  *capacity = required;
}

template <typename T>
T *ensure_typed_buffer(void **ptr,
                       std::size_t *capacity,
                       std::size_t count) {
  ensure_buffer(ptr, capacity, sizeof(T) * count);
  return static_cast<T *>(*ptr);
}

__global__ void bucket_count_i32_kernel(const int32_t *keys,
                                        int32_t *offsets,
                                        int num_items,
                                        int num_bins) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_items) {
    return;
  }
  int key = keys[i];
  if (key >= 0 && key < num_bins) {
    atomicAdd(offsets + key + 1, 1);
  }
}

template <typename T>
__global__ void bucket_scatter_kernel(const int32_t *keys,
                                      const T *values,
                                      int32_t *cursor,
                                      T *output,
                                      int num_items,
                                      int num_bins) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_items) {
    return;
  }
  int key = keys[i];
  if (key < 0 || key >= num_bins) {
    return;
  }
  int out_idx = atomicAdd(cursor + key, 1);
  if (out_idx >= 0 && out_idx < num_items) {
    output[out_idx] = values[i];
  }
}

__global__ void bucket_scatter_words_kernel(const int32_t *keys,
                                            const uint32_t *values,
                                            int32_t *cursor,
                                            uint32_t *output,
                                            int num_items,
                                            int num_bins,
                                            int item_words) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_items) {
    return;
  }
  int key = keys[i];
  if (key < 0 || key >= num_bins) {
    return;
  }
  int out_idx = atomicAdd(cursor + key, 1);
  if (out_idx < 0 || out_idx >= num_items) {
    return;
  }
  const int src_base = i * item_words;
  const int dst_base = out_idx * item_words;
  for (int lane = 0; lane < item_words; ++lane) {
    output[dst_base + lane] = values[src_base + lane];
  }
}

__global__ void select_flags_to_prefix_kernel(const int32_t *flags,
                                              int32_t *prefix,
                                              int num_items) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_items) {
    prefix[i] = flags[i] != 0 ? 1 : 0;
  }
}

__global__ void select_scatter_words_kernel(const uint32_t *values,
                                            const int32_t *flags,
                                            const int32_t *prefix,
                                            uint32_t *output,
                                            int32_t *count,
                                            int num_items,
                                            int item_words,
                                            int total_words) {
  int word = blockIdx.x * blockDim.x + threadIdx.x;
  if (word >= total_words) {
    return;
  }
  const int item = word / item_words;
  if (word == total_words - 1) {
    count[0] = prefix[num_items - 1];
  }
  if (flags[item] == 0) {
    return;
  }
  const int out_item = prefix[item] - 1;
  if (out_item >= 0 && out_item < num_items) {
    output[out_item * item_words + (word - item * item_words)] = values[word];
  }
}

template <typename T>
__device__ void scatter_atomic_add(T *addr, T value) {
  atomicAdd(addr, value);
}

template <>
__device__ void scatter_atomic_add<uint64_t>(uint64_t *addr, uint64_t value) {
  atomicAdd(reinterpret_cast<unsigned long long *>(addr),
            static_cast<unsigned long long>(value));
}

template <>
__device__ void scatter_atomic_add<int64_t>(int64_t *addr, int64_t value) {
  atomicAdd(reinterpret_cast<unsigned long long *>(addr),
            static_cast<unsigned long long>(value));
}

template <>
__device__ void scatter_atomic_add<double>(double *addr, double value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
  atomicAdd(addr, value);
#else
  auto *addr_as_ull = reinterpret_cast<unsigned long long *>(addr);
  unsigned long long old = *addr_as_ull;
  unsigned long long assumed = 0;
  do {
    assumed = old;
    const double updated = value + __longlong_as_double(assumed);
    old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(updated));
  } while (assumed != old);
#endif
}

template <typename T>
__global__ void scatter_add_by_i32_kernel(const T *src,
                                          const int32_t *indices,
                                          T *dst,
                                          int num_items,
                                          int index_bound) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_items) {
    return;
  }
  int index = indices[i];
  if (index >= 0 && index < index_bound) {
    scatter_atomic_add(dst + index, src[i]);
  }
}

template <typename T>
__device__ T load_strided_value(const uint8_t *base,
                                int i,
                                std::size_t offset,
                                std::size_t stride) {
  const auto *ptr =
      reinterpret_cast<const T *>(base + offset + static_cast<std::size_t>(i) *
                                                    stride);
  return *ptr;
}

template <typename T>
__device__ T *strided_value_ptr(uint8_t *base,
                                int i,
                                std::size_t offset,
                                std::size_t stride) {
  return reinterpret_cast<T *>(base + offset + static_cast<std::size_t>(i) *
                                                   stride);
}

template <typename T>
__global__ void scatter_add_by_i32_strided_kernel(const uint8_t *src,
                                                  const int32_t *indices,
                                                  T *dst,
                                                  int num_items,
                                                  int index_bound,
                                                  std::size_t offset,
                                                  std::size_t stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_items) {
    return;
  }
  int index = indices[i];
  if (index >= 0 && index < index_bound) {
    scatter_atomic_add(dst + index,
                       load_strided_value<T>(src, i, offset, stride));
  }
}

template <typename T>
__global__ void scatter_add_by_i32_strided_io_kernel(
    const uint8_t *src,
    const int32_t *indices,
    uint8_t *dst,
    int num_items,
    int index_bound,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_items) {
    return;
  }
  int index = indices[i];
  if (index >= 0 && index < index_bound) {
    scatter_atomic_add(strided_value_ptr<T>(dst, index, dst_offset, dst_stride),
                       load_strided_value<T>(src, i, src_offset, src_stride));
  }
}

template <typename T>
__global__ void grouped_reduce_atomic_sum_kernel(const int32_t *keys,
                                                 const T *values,
                                                 T *output,
                                                 int num_items,
                                                 int num_groups) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_items) {
    return;
  }
  int key = keys[i];
  if (key >= 0 && key < num_groups) {
    scatter_atomic_add(output + key, values[i]);
  }
}

template <typename T>
__global__ void grouped_reduce_atomic_sum_strided_kernel(
    const int32_t *keys,
    const uint8_t *values,
    T *output,
    int num_items,
    int num_groups,
    std::size_t offset,
    std::size_t stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_items) {
    return;
  }
  int key = keys[i];
  if (key >= 0 && key < num_groups) {
    scatter_atomic_add(output + key,
                       load_strided_value<T>(values, i, offset, stride));
  }
}

template <typename T>
__global__ void zero_strided_kernel(uint8_t *output,
                                    int num_items,
                                    std::size_t offset,
                                    std::size_t stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_items) {
    return;
  }
  *strided_value_ptr<T>(output, i, offset, stride) = T{};
}

template <typename T>
__global__ void grouped_reduce_atomic_sum_strided_io_kernel(
    const uint8_t *keys,
    const uint8_t *values,
    uint8_t *output,
    int num_items,
    int num_groups,
    std::size_t keys_offset,
    std::size_t keys_stride,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t output_offset,
    std::size_t output_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_items) {
    return;
  }
  int key = load_strided_value<int32_t>(keys, i, keys_offset, keys_stride);
  if (key >= 0 && key < num_groups) {
    scatter_atomic_add(
        strided_value_ptr<T>(output, key, output_offset, output_stride),
        load_strided_value<T>(values, i, values_offset, values_stride));
  }
}

template <typename T>
__device__ bool histogram_bin_in_range(T bin, int num_bins);

template <>
__device__ bool histogram_bin_in_range<int32_t>(int32_t bin, int num_bins) {
  return bin >= 0 && bin < num_bins;
}

template <>
__device__ bool histogram_bin_in_range<uint32_t>(uint32_t bin, int num_bins) {
  return bin < static_cast<uint32_t>(num_bins);
}

template <typename T>
__global__ void histogram_i64_direct_kernel(const T *samples,
                                            int64_t *hist,
                                            int num_items,
                                            int num_bins) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_items) {
    return;
  }
  T bin = samples[i];
  if (histogram_bin_in_range(bin, num_bins)) {
    atomicAdd(reinterpret_cast<unsigned long long *>(hist +
                                                     static_cast<int>(bin)),
              1ull);
  }
}

__global__ void indexed_copy_words_by_i32_kernel(const uint32_t *src,
                                                 const int32_t *indices,
                                                 uint32_t *dst,
                                                 int num_items,
                                                 int index_bound,
                                                 int item_words,
                                                 int op) {
  int word_i = blockIdx.x * blockDim.x + threadIdx.x;
  int total_words = num_items * item_words;
  if (word_i >= total_words) {
    return;
  }
  int item = word_i / item_words;
  int lane = word_i - item * item_words;
  int index = indices[item];
  if (op == static_cast<int>(CudaIndexedCopyOp::gather)) {
    if (index >= 0 && index < index_bound) {
      dst[word_i] = src[index * item_words + lane];
    } else {
      dst[word_i] = 0u;
    }
  } else {
    if (index >= 0 && index < index_bound) {
      dst[index * item_words + lane] = src[word_i];
    }
  }
}

__global__ void transform_u32_affine_kernel(const uint32_t *src,
                                            uint32_t *dst,
                                            int num_items,
                                            uint32_t scale,
                                            uint32_t bias) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_items) {
    return;
  }
  dst[i] = src[i] * scale + bias;
}

__global__ void transform_f32_affine_kernel(const float *src,
                                            float *dst,
                                            int num_items,
                                            float scale,
                                            float bias) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_items) {
    return;
  }
  dst[i] = src[i] * scale + bias;
}

__global__ void transform_u64_affine_kernel(const uint64_t *src,
                                            uint64_t *dst,
                                            int num_items,
                                            uint64_t scale,
                                            uint64_t bias) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_items) {
    return;
  }
  dst[i] = src[i] * scale + bias;
}

__global__ void transform_f64_affine_kernel(const double *src,
                                            double *dst,
                                            int num_items,
                                            double scale,
                                            double bias) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_items) {
    return;
  }
  dst[i] = src[i] * scale + bias;
}

template <typename T>
__global__ void transform_strided_affine_kernel(const uint8_t *src,
                                                T *dst,
                                                int num_items,
                                                std::size_t offset,
                                                std::size_t stride,
                                                T scale,
                                                T bias) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_items) {
    return;
  }
  const auto *value =
      reinterpret_cast<const T *>(src + offset + static_cast<std::size_t>(i) *
                                                   stride);
  dst[i] = (*value) * scale + bias;
}

__device__ uint32_t sortable_f32_key(float value, int nan_policy) {
  const uint32_t bits = __float_as_uint(value);
  constexpr uint32_t kSign = 0x80000000u;
  const bool is_nan = (bits & 0x7fffffffu) > 0x7f800000u;
  if (nan_policy == static_cast<int>(CubSortNanPolicy::last) && is_nan) {
    return 0xffffffffu;
  }
  if (nan_policy == static_cast<int>(CubSortNanPolicy::last) &&
      (bits & 0x7fffffffu) == 0) {
    return kSign;
  }
  return (bits & kSign) ? ~bits : (bits ^ kSign);
}

__device__ uint64_t sortable_f64_key(double value, int nan_policy) {
  const uint64_t bits = static_cast<uint64_t>(__double_as_longlong(value));
  constexpr uint64_t kSign = 0x8000000000000000ull;
  const bool is_nan = (bits & 0x7fffffffffffffffull) > 0x7ff0000000000000ull;
  if (nan_policy == static_cast<int>(CubSortNanPolicy::last) && is_nan) {
    return 0xffffffffffffffffull;
  }
  if (nan_policy == static_cast<int>(CubSortNanPolicy::last) &&
      (bits & 0x7fffffffffffffffull) == 0) {
    return kSign;
  }
  return (bits & kSign) ? ~bits : (bits ^ kSign);
}

template <typename KeyT>
__device__ uint64_t sortable_u64_key(KeyT value, int nan_policy);

template <>
__device__ uint64_t sortable_u64_key<uint64_t>(uint64_t value,
                                               int nan_policy) {
  return value;
}

template <>
__device__ uint64_t sortable_u64_key<int64_t>(int64_t value, int nan_policy) {
  return static_cast<uint64_t>(value) ^ 0x8000000000000000ull;
}

template <>
__device__ uint64_t sortable_u64_key<double>(double value, int nan_policy) {
  return sortable_f64_key(value, nan_policy);
}

__global__ void init_sortable_f32_kernel(const float *keys,
                                         uint32_t *sort_keys,
                                         uint32_t *indices,
                                         int num_items,
                                         int nan_policy) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_items) {
    return;
  }
  sort_keys[i] = sortable_f32_key(keys[i], nan_policy);
  indices[i] = static_cast<uint32_t>(i);
}

template <typename KeyT>
__device__ uint32_t sortable_u32_key(KeyT value);

template <>
__device__ uint32_t sortable_u32_key<uint32_t>(uint32_t value) {
  return value;
}

template <>
__device__ uint32_t sortable_u32_key<int32_t>(int32_t value) {
  return static_cast<uint32_t>(value) ^ 0x80000000u;
}

template <typename KeyT>
__global__ void init_sortable32_kernel(const KeyT *keys,
                                       uint32_t *sort_keys,
                                       uint32_t *indices,
                                       int num_items) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_items) {
    return;
  }
  sort_keys[i] = sortable_u32_key<KeyT>(keys[i]);
  indices[i] = static_cast<uint32_t>(i);
}

template <typename KeyT>
__global__ void init_sortable64_kernel(const KeyT *keys,
                                       uint64_t *sort_keys,
                                       uint32_t *indices,
                                       int num_items,
                                       int nan_policy) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_items) {
    return;
  }
  sort_keys[i] = sortable_u64_key<KeyT>(keys[i], nan_policy);
  indices[i] = static_cast<uint32_t>(i);
}

template <typename KeyT>
__global__ void init_split32_kernel(const KeyT *keys,
                                    uint32_t *low_keys,
                                    uint32_t *high_keys,
                                    uint32_t *indices,
                                    int num_items,
                                    int nan_policy) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_items) {
    return;
  }
  const uint64_t sortable = sortable_u64_key<KeyT>(keys[i], nan_policy);
  low_keys[i] = static_cast<uint32_t>(sortable);
  high_keys[i] = static_cast<uint32_t>(sortable >> 32);
  indices[i] = static_cast<uint32_t>(i);
}

__global__ void gather_u32_by_index_kernel(const uint32_t *src,
                                           const uint32_t *indices,
                                           uint32_t *dst,
                                           int num_items) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_items) {
    return;
  }
  dst[i] = src[indices[i]];
}

template <typename KeyT, typename ValueT>
__global__ void scatter_by_index_kernel(const KeyT *keys,
                                        const ValueT *values,
                                        const uint32_t *indices,
                                        KeyT *keys_out,
                                        ValueT *values_out,
                                        int num_items,
                                        bool has_values) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_items) {
    return;
  }
  const uint32_t src = indices[i];
  keys_out[i] = keys[src];
  if (has_values) {
    values_out[i] = values[src];
  }
}

template <typename KeyT>
__global__ void scatter_raw_values_by_index_kernel(const KeyT *keys,
                                                   const uint32_t *values,
                                                   const uint32_t *indices,
                                                   KeyT *keys_out,
                                                   uint32_t *values_out,
                                                   int num_items,
                                                   int item_words,
                                                   int total_words) {
  const int word = blockIdx.x * blockDim.x + threadIdx.x;
  if (word >= total_words) {
    return;
  }
  const int item = word / item_words;
  const int lane = word - item * item_words;
  const uint32_t src = indices[item];
  if (lane == 0) {
    keys_out[item] = keys[src];
  }
  values_out[word] = values[src * item_words + lane];
}

void check_last_cuda_error(const char *) {
  TI_CUDA_SORT_CHECK(cudaGetLastError());
}

template <typename KeyT>
void launch_init_sortable32(const KeyT *keys,
                            uint32_t *sort_keys,
                            uint32_t *indices,
                            int num_items,
                            cudaStream_t stream) {
  constexpr int kBlockSize = 256;
  const int grid = (num_items + kBlockSize - 1) / kBlockSize;
  init_sortable32_kernel<KeyT><<<grid, kBlockSize, 0, stream>>>(
      keys, sort_keys, indices, num_items);
  check_last_cuda_error("init_sortable32_kernel");
}

void launch_init_sortable_f32(const float *keys,
                              uint32_t *sort_keys,
                              uint32_t *indices,
                              int num_items,
                              int nan_policy,
                              cudaStream_t stream) {
  constexpr int kBlockSize = 256;
  const int grid = (num_items + kBlockSize - 1) / kBlockSize;
  init_sortable_f32_kernel<<<grid, kBlockSize, 0, stream>>>(
      keys, sort_keys, indices, num_items, nan_policy);
  check_last_cuda_error("init_sortable_f32_kernel");
}

template <typename KeyT>
void launch_init_sortable64(const KeyT *keys,
                            uint64_t *sort_keys,
                            uint32_t *indices,
                            int num_items,
                            int nan_policy,
                            cudaStream_t stream) {
  constexpr int kBlockSize = 256;
  const int grid = (num_items + kBlockSize - 1) / kBlockSize;
  init_sortable64_kernel<KeyT><<<grid, kBlockSize, 0, stream>>>(
      keys, sort_keys, indices, num_items, nan_policy);
  check_last_cuda_error("init_sortable64_kernel");
}

template <typename KeyT>
void launch_init_split32(const KeyT *keys,
                         uint32_t *low_keys,
                         uint32_t *high_keys,
                         uint32_t *indices,
                         int num_items,
                         int nan_policy,
                         cudaStream_t stream) {
  constexpr int kBlockSize = 256;
  const int grid = (num_items + kBlockSize - 1) / kBlockSize;
  init_split32_kernel<KeyT><<<grid, kBlockSize, 0, stream>>>(
      keys, low_keys, high_keys, indices, num_items, nan_policy);
  check_last_cuda_error("init_split32_kernel");
}

void launch_gather_u32_by_index(const uint32_t *src,
                                const uint32_t *indices,
                                uint32_t *dst,
                                int num_items,
                                cudaStream_t stream) {
  constexpr int kBlockSize = 256;
  const int grid = (num_items + kBlockSize - 1) / kBlockSize;
  gather_u32_by_index_kernel<<<grid, kBlockSize, 0, stream>>>(
      src, indices, dst, num_items);
  check_last_cuda_error("gather_u32_by_index_kernel");
}

template <typename KeyT, typename ValueT>
void launch_scatter_by_index(const KeyT *keys,
                             const ValueT *values,
                             const uint32_t *indices,
                             KeyT *keys_out,
                             ValueT *values_out,
                             int num_items,
                             bool has_values,
                             cudaStream_t stream) {
  constexpr int kBlockSize = 256;
  const int grid = (num_items + kBlockSize - 1) / kBlockSize;
  scatter_by_index_kernel<KeyT, ValueT><<<grid, kBlockSize, 0, stream>>>(
      keys, values, indices, keys_out, values_out, num_items, has_values);
  check_last_cuda_error("scatter_by_index_kernel");
}

template <typename KeyT>
void launch_scatter_raw_values_by_index(const KeyT *keys,
                                        const uint32_t *values,
                                        const uint32_t *indices,
                                        KeyT *keys_out,
                                        uint32_t *values_out,
                                        int num_items,
                                        int item_words,
                                        cudaStream_t stream) {
  constexpr int kBlockSize = 256;
  const int total_words = num_items * item_words;
  const int grid = (total_words + kBlockSize - 1) / kBlockSize;
  scatter_raw_values_by_index_kernel<KeyT><<<grid, kBlockSize, 0, stream>>>(
      keys, values, indices, keys_out, values_out, num_items, item_words,
      total_words);
  check_last_cuda_error("scatter_raw_values_by_index_kernel");
}

template <typename KeyT, typename ValueT>
void cub_sort_pairs(CubSortCache &cache,
                    KeyT *keys_in,
                    KeyT *keys_out,
                    ValueT *values_in,
                    ValueT *values_out,
                    int num_items,
                    cudaStream_t stream) {
  const bool use_stream = stream != nullptr;
  std::size_t temp_storage_bytes = 0;
  if (use_stream) {
    TI_CUDA_SORT_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes, keys_in, keys_out, values_in, values_out,
        num_items, 0, sizeof(KeyT) * 8, stream));
  } else {
    TI_CUDA_SORT_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes, keys_in, keys_out, values_in, values_out,
        num_items));
  }
  ensure_buffer(&cache.temp_storage, &cache.temp_storage_bytes,
                temp_storage_bytes);
  if (use_stream) {
    TI_CUDA_SORT_CHECK(cub::DeviceRadixSort::SortPairs(
        cache.temp_storage, temp_storage_bytes, keys_in, keys_out, values_in,
        values_out, num_items, 0, sizeof(KeyT) * 8, stream));
  } else {
    TI_CUDA_SORT_CHECK(cub::DeviceRadixSort::SortPairs(
        cache.temp_storage, temp_storage_bytes, keys_in, keys_out, values_in,
        values_out, num_items));
  }
}

template <typename KeyT, typename ValueT>
std::size_t sort_typed(CubSortCache &cache,
                       void *keys,
                       void *values,
                       int num_items,
                       bool has_values,
                       void *stream_ptr) {
  KeyT *keys_in = static_cast<KeyT *>(keys);
  ValueT *values_in = static_cast<ValueT *>(values);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const bool use_stream = stream != nullptr;
  ensure_device_cache(cache);

  ensure_buffer(&cache.keys_out, &cache.keys_out_bytes,
                sizeof(KeyT) * num_items);
  KeyT *keys_out = static_cast<KeyT *>(cache.keys_out);
  ValueT *values_out = nullptr;
  if (has_values) {
    ensure_buffer(&cache.values_out, &cache.values_out_bytes,
                  sizeof(ValueT) * num_items);
    values_out = static_cast<ValueT *>(cache.values_out);
  } else if (cache.values_out) {
    TI_CUDA_SORT_CHECK(cudaFree(cache.values_out));
    cache.values_out = nullptr;
    cache.values_out_bytes = 0;
  }

  std::size_t temp_storage_bytes = 0;
  constexpr int kBeginBit = 0;
  constexpr int kEndBit = sizeof(KeyT) * 8;
  if (has_values) {
    if (use_stream) {
      TI_CUDA_SORT_CHECK(cub::DeviceRadixSort::SortPairs(
          nullptr, temp_storage_bytes, keys_in, keys_out, values_in, values_out,
          num_items, kBeginBit, kEndBit, stream));
    } else {
      TI_CUDA_SORT_CHECK(cub::DeviceRadixSort::SortPairs(
          nullptr, temp_storage_bytes, keys_in, keys_out, values_in, values_out,
          num_items));
    }
  } else {
    if (use_stream) {
      TI_CUDA_SORT_CHECK(cub::DeviceRadixSort::SortKeys(
          nullptr, temp_storage_bytes, keys_in, keys_out, num_items, kBeginBit,
          kEndBit, stream));
    } else {
      TI_CUDA_SORT_CHECK(cub::DeviceRadixSort::SortKeys(
          nullptr, temp_storage_bytes, keys_in, keys_out, num_items));
    }
  }

  ensure_buffer(&cache.temp_storage, &cache.temp_storage_bytes,
                temp_storage_bytes);
  if (has_values) {
    if (use_stream) {
      TI_CUDA_SORT_CHECK(cub::DeviceRadixSort::SortPairs(
          cache.temp_storage, temp_storage_bytes, keys_in, keys_out, values_in,
          values_out, num_items, kBeginBit, kEndBit, stream));
      TI_CUDA_SORT_CHECK(cudaMemcpyAsync(values_in, values_out,
                                         sizeof(ValueT) * num_items,
                                         cudaMemcpyDeviceToDevice, stream));
    } else {
      TI_CUDA_SORT_CHECK(cub::DeviceRadixSort::SortPairs(
          cache.temp_storage, temp_storage_bytes, keys_in, keys_out, values_in,
          values_out, num_items));
      TI_CUDA_SORT_CHECK(cudaMemcpy(values_in, values_out,
                                    sizeof(ValueT) * num_items,
                                    cudaMemcpyDeviceToDevice));
    }
  } else {
    if (use_stream) {
      TI_CUDA_SORT_CHECK(cub::DeviceRadixSort::SortKeys(
          cache.temp_storage, temp_storage_bytes, keys_in, keys_out, num_items,
          kBeginBit, kEndBit, stream));
    } else {
      TI_CUDA_SORT_CHECK(cub::DeviceRadixSort::SortKeys(
          cache.temp_storage, temp_storage_bytes, keys_in, keys_out,
          num_items));
    }
  }
  if (use_stream) {
    TI_CUDA_SORT_CHECK(cudaMemcpyAsync(keys_in, keys_out,
                                       sizeof(KeyT) * num_items,
                                       cudaMemcpyDeviceToDevice, stream));
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaMemcpy(keys_in, keys_out, sizeof(KeyT) * num_items,
                                  cudaMemcpyDeviceToDevice));
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }

  return cache.allocated_bytes();
}

template <typename ValueT>
std::size_t sort_f32_transformed(CubSortCache &cache,
                                 void *keys,
                                 void *values,
                                 int num_items,
                                 bool has_values,
                                 CubSortNanPolicy nan_policy,
                                 void *stream_ptr) {
  float *keys_in = static_cast<float *>(keys);
  ValueT *values_in = static_cast<ValueT *>(values);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const bool use_stream = stream != nullptr;
  ensure_device_cache(cache);

  auto *key_a = ensure_typed_buffer<uint32_t>(&cache.key32_a,
                                              &cache.key32_a_bytes, num_items);
  auto *key_b = ensure_typed_buffer<uint32_t>(&cache.key32_b,
                                              &cache.key32_b_bytes, num_items);
  auto *index_a = ensure_typed_buffer<uint32_t>(
      &cache.index_a, &cache.index_a_bytes, num_items);
  auto *index_b = ensure_typed_buffer<uint32_t>(
      &cache.index_b, &cache.index_b_bytes, num_items);
  auto *keys_out = ensure_typed_buffer<float>(&cache.keys_out,
                                              &cache.keys_out_bytes, num_items);
  ValueT *values_out = nullptr;
  if (has_values) {
    values_out = ensure_typed_buffer<ValueT>(
        &cache.values_out, &cache.values_out_bytes, num_items);
  } else if (cache.values_out) {
    TI_CUDA_SORT_CHECK(cudaFree(cache.values_out));
    cache.values_out = nullptr;
    cache.values_out_bytes = 0;
  }

  launch_init_sortable_f32(keys_in, key_a, index_a, num_items,
                           static_cast<int>(nan_policy), stream);
  cub_sort_pairs(cache, key_a, key_b, index_a, index_b, num_items, stream);
  launch_scatter_by_index<float, ValueT>(keys_in, values_in, index_b, keys_out, values_out,
                          num_items, has_values, stream);

  if (use_stream) {
    TI_CUDA_SORT_CHECK(cudaMemcpyAsync(keys_in, keys_out,
                                       sizeof(float) * num_items,
                                       cudaMemcpyDeviceToDevice, stream));
    if (has_values) {
      TI_CUDA_SORT_CHECK(cudaMemcpyAsync(values_in, values_out,
                                         sizeof(ValueT) * num_items,
                                         cudaMemcpyDeviceToDevice, stream));
    }
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaMemcpy(keys_in, keys_out, sizeof(float) * num_items,
                                  cudaMemcpyDeviceToDevice));
    if (has_values) {
      TI_CUDA_SORT_CHECK(cudaMemcpy(values_in, values_out,
                                    sizeof(ValueT) * num_items,
                                    cudaMemcpyDeviceToDevice));
    }
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }

  return cache.allocated_bytes();
}

template <typename KeyT, typename ValueT>
std::size_t sort_u64_transformed(CubSortCache &cache,
                                 void *keys,
                                 void *values,
                                 int num_items,
                                 bool has_values,
                                 CubSortNanPolicy nan_policy,
                                 void *stream_ptr) {
  KeyT *keys_in = static_cast<KeyT *>(keys);
  ValueT *values_in = static_cast<ValueT *>(values);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const bool use_stream = stream != nullptr;
  ensure_device_cache(cache);

  auto *key_a = ensure_typed_buffer<uint64_t>(&cache.key64_a,
                                              &cache.key64_a_bytes, num_items);
  auto *key_b = ensure_typed_buffer<uint64_t>(&cache.key64_b,
                                              &cache.key64_b_bytes, num_items);
  auto *index_a = ensure_typed_buffer<uint32_t>(
      &cache.index_a, &cache.index_a_bytes, num_items);
  auto *index_b = ensure_typed_buffer<uint32_t>(
      &cache.index_b, &cache.index_b_bytes, num_items);
  auto *keys_out = ensure_typed_buffer<KeyT>(&cache.keys_out,
                                             &cache.keys_out_bytes, num_items);
  ValueT *values_out = nullptr;
  if (has_values) {
    values_out = ensure_typed_buffer<ValueT>(
        &cache.values_out, &cache.values_out_bytes, num_items);
  } else if (cache.values_out) {
    TI_CUDA_SORT_CHECK(cudaFree(cache.values_out));
    cache.values_out = nullptr;
    cache.values_out_bytes = 0;
  }

  launch_init_sortable64(keys_in, key_a, index_a, num_items,
                         static_cast<int>(nan_policy), stream);
  cub_sort_pairs(cache, key_a, key_b, index_a, index_b, num_items, stream);
  launch_scatter_by_index<KeyT, ValueT>(keys_in, values_in, index_b, keys_out, values_out,
                          num_items, has_values, stream);

  if (use_stream) {
    TI_CUDA_SORT_CHECK(cudaMemcpyAsync(keys_in, keys_out,
                                       sizeof(KeyT) * num_items,
                                       cudaMemcpyDeviceToDevice, stream));
    if (has_values) {
      TI_CUDA_SORT_CHECK(cudaMemcpyAsync(values_in, values_out,
                                         sizeof(ValueT) * num_items,
                                         cudaMemcpyDeviceToDevice, stream));
    }
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaMemcpy(keys_in, keys_out, sizeof(KeyT) * num_items,
                                  cudaMemcpyDeviceToDevice));
    if (has_values) {
      TI_CUDA_SORT_CHECK(cudaMemcpy(values_in, values_out,
                                    sizeof(ValueT) * num_items,
                                    cudaMemcpyDeviceToDevice));
    }
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }

  return cache.allocated_bytes();
}

template <typename KeyT, typename ValueT>
std::size_t sort_split32(CubSortCache &cache,
                         void *keys,
                         void *values,
                         int num_items,
                         bool has_values,
                         CubSortNanPolicy nan_policy,
                         void *stream_ptr) {
  KeyT *keys_in = static_cast<KeyT *>(keys);
  ValueT *values_in = static_cast<ValueT *>(values);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const bool use_stream = stream != nullptr;
  ensure_device_cache(cache);

  auto *low_keys = ensure_typed_buffer<uint32_t>(
      &cache.key32_a, &cache.key32_a_bytes, num_items);
  auto *tmp_keys = ensure_typed_buffer<uint32_t>(
      &cache.key32_b, &cache.key32_b_bytes, num_items);
  auto *high_keys = ensure_typed_buffer<uint32_t>(
      &cache.high32, &cache.high32_bytes, num_items);
  auto *index_a = ensure_typed_buffer<uint32_t>(
      &cache.index_a, &cache.index_a_bytes, num_items);
  auto *index_b = ensure_typed_buffer<uint32_t>(
      &cache.index_b, &cache.index_b_bytes, num_items);
  auto *keys_out = ensure_typed_buffer<KeyT>(&cache.keys_out,
                                             &cache.keys_out_bytes, num_items);
  ValueT *values_out = nullptr;
  if (has_values) {
    values_out = ensure_typed_buffer<ValueT>(
        &cache.values_out, &cache.values_out_bytes, num_items);
  } else if (cache.values_out) {
    TI_CUDA_SORT_CHECK(cudaFree(cache.values_out));
    cache.values_out = nullptr;
    cache.values_out_bytes = 0;
  }

  launch_init_split32(keys_in, low_keys, high_keys, index_a, num_items,
                      static_cast<int>(nan_policy), stream);
  cub_sort_pairs(cache, low_keys, tmp_keys, index_a, index_b, num_items,
                 stream);
  launch_gather_u32_by_index(high_keys, index_b, low_keys, num_items, stream);
  cub_sort_pairs(cache, low_keys, tmp_keys, index_b, index_a, num_items,
                 stream);
  launch_scatter_by_index<KeyT, ValueT>(keys_in, values_in, index_a, keys_out, values_out,
                          num_items, has_values, stream);

  if (use_stream) {
    TI_CUDA_SORT_CHECK(cudaMemcpyAsync(keys_in, keys_out,
                                       sizeof(KeyT) * num_items,
                                       cudaMemcpyDeviceToDevice, stream));
    if (has_values) {
      TI_CUDA_SORT_CHECK(cudaMemcpyAsync(values_in, values_out,
                                         sizeof(ValueT) * num_items,
                                         cudaMemcpyDeviceToDevice, stream));
    }
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaMemcpy(keys_in, keys_out, sizeof(KeyT) * num_items,
                                  cudaMemcpyDeviceToDevice));
    if (has_values) {
      TI_CUDA_SORT_CHECK(cudaMemcpy(values_in, values_out,
                                    sizeof(ValueT) * num_items,
                                    cudaMemcpyDeviceToDevice));
    }
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }

  return cache.allocated_bytes();
}

template <typename KeyT>
std::size_t sort_raw32_index(CubSortCache &cache,
                             void *keys,
                             void *values,
                             int num_items,
                             int item_words,
                             void *stream_ptr) {
  KeyT *keys_in = static_cast<KeyT *>(keys);
  auto *values_in = static_cast<uint32_t *>(values);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const bool use_stream = stream != nullptr;
  ensure_device_cache(cache);

  auto *key_a = ensure_typed_buffer<uint32_t>(&cache.key32_a,
                                              &cache.key32_a_bytes, num_items);
  auto *key_b = ensure_typed_buffer<uint32_t>(&cache.key32_b,
                                              &cache.key32_b_bytes, num_items);
  auto *index_a = ensure_typed_buffer<uint32_t>(
      &cache.index_a, &cache.index_a_bytes, num_items);
  auto *index_b = ensure_typed_buffer<uint32_t>(
      &cache.index_b, &cache.index_b_bytes, num_items);
  auto *keys_out = ensure_typed_buffer<KeyT>(&cache.keys_out,
                                             &cache.keys_out_bytes, num_items);
  auto *values_out = ensure_typed_buffer<uint32_t>(
      &cache.values_out, &cache.values_out_bytes,
      static_cast<std::size_t>(num_items) * item_words);

  launch_init_sortable32(keys_in, key_a, index_a, num_items, stream);
  cub_sort_pairs(cache, key_a, key_b, index_a, index_b, num_items, stream);
  launch_scatter_raw_values_by_index(keys_in, values_in, index_b, keys_out,
                                     values_out, num_items, item_words,
                                     stream);
  const std::size_t key_bytes =
      static_cast<std::size_t>(num_items) * sizeof(KeyT);
  const std::size_t value_bytes =
      static_cast<std::size_t>(num_items) * item_words * sizeof(uint32_t);
  if (use_stream) {
    TI_CUDA_SORT_CHECK(cudaMemcpyAsync(keys_in, keys_out, key_bytes,
                                       cudaMemcpyDeviceToDevice, stream));
    TI_CUDA_SORT_CHECK(cudaMemcpyAsync(values_in, values_out, value_bytes,
                                       cudaMemcpyDeviceToDevice, stream));
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaMemcpy(keys_in, keys_out, key_bytes,
                                  cudaMemcpyDeviceToDevice));
    TI_CUDA_SORT_CHECK(cudaMemcpy(values_in, values_out, value_bytes,
                                  cudaMemcpyDeviceToDevice));
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }
  return cache.allocated_bytes();
}

template <typename KeyT>
std::size_t sort_raw64_index(CubSortCache &cache,
                             void *keys,
                             void *values,
                             int num_items,
                             int item_words,
                             CubSortNanPolicy nan_policy,
                             void *stream_ptr) {
  KeyT *keys_in = static_cast<KeyT *>(keys);
  auto *values_in = static_cast<uint32_t *>(values);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const bool use_stream = stream != nullptr;
  ensure_device_cache(cache);

  auto *key_a = ensure_typed_buffer<uint64_t>(&cache.key64_a,
                                              &cache.key64_a_bytes, num_items);
  auto *key_b = ensure_typed_buffer<uint64_t>(&cache.key64_b,
                                              &cache.key64_b_bytes, num_items);
  auto *index_a = ensure_typed_buffer<uint32_t>(
      &cache.index_a, &cache.index_a_bytes, num_items);
  auto *index_b = ensure_typed_buffer<uint32_t>(
      &cache.index_b, &cache.index_b_bytes, num_items);
  auto *keys_out = ensure_typed_buffer<KeyT>(&cache.keys_out,
                                             &cache.keys_out_bytes, num_items);
  auto *values_out = ensure_typed_buffer<uint32_t>(
      &cache.values_out, &cache.values_out_bytes,
      static_cast<std::size_t>(num_items) * item_words);

  launch_init_sortable64(keys_in, key_a, index_a, num_items,
                         static_cast<int>(nan_policy), stream);
  cub_sort_pairs(cache, key_a, key_b, index_a, index_b, num_items, stream);
  launch_scatter_raw_values_by_index(keys_in, values_in, index_b, keys_out,
                                     values_out, num_items, item_words,
                                     stream);
  const std::size_t key_bytes =
      static_cast<std::size_t>(num_items) * sizeof(KeyT);
  const std::size_t value_bytes =
      static_cast<std::size_t>(num_items) * item_words * sizeof(uint32_t);
  if (use_stream) {
    TI_CUDA_SORT_CHECK(cudaMemcpyAsync(keys_in, keys_out, key_bytes,
                                       cudaMemcpyDeviceToDevice, stream));
    TI_CUDA_SORT_CHECK(cudaMemcpyAsync(values_in, values_out, value_bytes,
                                       cudaMemcpyDeviceToDevice, stream));
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaMemcpy(keys_in, keys_out, key_bytes,
                                  cudaMemcpyDeviceToDevice));
    TI_CUDA_SORT_CHECK(cudaMemcpy(values_in, values_out, value_bytes,
                                  cudaMemcpyDeviceToDevice));
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }
  return cache.allocated_bytes();
}

std::size_t sort_f32_raw_index(CubSortCache &cache,
                               void *keys,
                               void *values,
                               int num_items,
                               int item_words,
                               CubSortNanPolicy nan_policy,
                               void *stream_ptr) {
  float *keys_in = static_cast<float *>(keys);
  auto *values_in = static_cast<uint32_t *>(values);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const bool use_stream = stream != nullptr;
  ensure_device_cache(cache);

  auto *key_a = ensure_typed_buffer<uint32_t>(&cache.key32_a,
                                              &cache.key32_a_bytes, num_items);
  auto *key_b = ensure_typed_buffer<uint32_t>(&cache.key32_b,
                                              &cache.key32_b_bytes, num_items);
  auto *index_a = ensure_typed_buffer<uint32_t>(
      &cache.index_a, &cache.index_a_bytes, num_items);
  auto *index_b = ensure_typed_buffer<uint32_t>(
      &cache.index_b, &cache.index_b_bytes, num_items);
  auto *keys_out = ensure_typed_buffer<float>(&cache.keys_out,
                                              &cache.keys_out_bytes, num_items);
  auto *values_out = ensure_typed_buffer<uint32_t>(
      &cache.values_out, &cache.values_out_bytes,
      static_cast<std::size_t>(num_items) * item_words);

  launch_init_sortable_f32(keys_in, key_a, index_a, num_items,
                           static_cast<int>(nan_policy), stream);
  cub_sort_pairs(cache, key_a, key_b, index_a, index_b, num_items, stream);
  launch_scatter_raw_values_by_index(keys_in, values_in, index_b, keys_out,
                                     values_out, num_items, item_words,
                                     stream);
  const std::size_t key_bytes =
      static_cast<std::size_t>(num_items) * sizeof(float);
  const std::size_t value_bytes =
      static_cast<std::size_t>(num_items) * item_words * sizeof(uint32_t);
  if (use_stream) {
    TI_CUDA_SORT_CHECK(cudaMemcpyAsync(keys_in, keys_out, key_bytes,
                                       cudaMemcpyDeviceToDevice, stream));
    TI_CUDA_SORT_CHECK(cudaMemcpyAsync(values_in, values_out, value_bytes,
                                       cudaMemcpyDeviceToDevice, stream));
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaMemcpy(keys_in, keys_out, key_bytes,
                                  cudaMemcpyDeviceToDevice));
    TI_CUDA_SORT_CHECK(cudaMemcpy(values_in, values_out, value_bytes,
                                  cudaMemcpyDeviceToDevice));
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }
  return cache.allocated_bytes();
}

template <typename KeyT>
std::size_t sort_split32_raw(CubSortCache &cache,
                             void *keys,
                             void *values,
                             int num_items,
                             int item_words,
                             CubSortNanPolicy nan_policy,
                             void *stream_ptr) {
  KeyT *keys_in = static_cast<KeyT *>(keys);
  auto *values_in = static_cast<uint32_t *>(values);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const bool use_stream = stream != nullptr;
  ensure_device_cache(cache);

  auto *low_keys = ensure_typed_buffer<uint32_t>(
      &cache.key32_a, &cache.key32_a_bytes, num_items);
  auto *tmp_keys = ensure_typed_buffer<uint32_t>(
      &cache.key32_b, &cache.key32_b_bytes, num_items);
  auto *high_keys = ensure_typed_buffer<uint32_t>(
      &cache.high32, &cache.high32_bytes, num_items);
  auto *index_a = ensure_typed_buffer<uint32_t>(
      &cache.index_a, &cache.index_a_bytes, num_items);
  auto *index_b = ensure_typed_buffer<uint32_t>(
      &cache.index_b, &cache.index_b_bytes, num_items);
  auto *keys_out = ensure_typed_buffer<KeyT>(&cache.keys_out,
                                             &cache.keys_out_bytes, num_items);
  auto *values_out = ensure_typed_buffer<uint32_t>(
      &cache.values_out, &cache.values_out_bytes,
      static_cast<std::size_t>(num_items) * item_words);

  launch_init_split32(keys_in, low_keys, high_keys, index_a, num_items,
                      static_cast<int>(nan_policy), stream);
  cub_sort_pairs(cache, low_keys, tmp_keys, index_a, index_b, num_items,
                 stream);
  launch_gather_u32_by_index(high_keys, index_b, low_keys, num_items, stream);
  cub_sort_pairs(cache, low_keys, tmp_keys, index_b, index_a, num_items,
                 stream);
  launch_scatter_raw_values_by_index(keys_in, values_in, index_a, keys_out,
                                     values_out, num_items, item_words,
                                     stream);
  const std::size_t key_bytes =
      static_cast<std::size_t>(num_items) * sizeof(KeyT);
  const std::size_t value_bytes =
      static_cast<std::size_t>(num_items) * item_words * sizeof(uint32_t);
  if (use_stream) {
    TI_CUDA_SORT_CHECK(cudaMemcpyAsync(keys_in, keys_out, key_bytes,
                                       cudaMemcpyDeviceToDevice, stream));
    TI_CUDA_SORT_CHECK(cudaMemcpyAsync(values_in, values_out, value_bytes,
                                       cudaMemcpyDeviceToDevice, stream));
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaMemcpy(keys_in, keys_out, key_bytes,
                                  cudaMemcpyDeviceToDevice));
    TI_CUDA_SORT_CHECK(cudaMemcpy(values_in, values_out, value_bytes,
                                  cudaMemcpyDeviceToDevice));
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }
  return cache.allocated_bytes();
}

std::size_t cub_radix_sort_raw_value_impl(CubSortCache &cache,
                                          void *keys,
                                          void *values,
                                          int num_items,
                                          CubSortKeyType key_type,
                                          CubSortMode mode,
                                          CubSortNanPolicy nan_policy,
                                          int value_words,
                                          void *stream) {
  if (value_words <= 0) {
    throw std::runtime_error("CUDA CUB sort expects positive raw value words");
  }
  if (mode == CubSortMode::split32) {
    switch (key_type) {
      case CubSortKeyType::u64:
        return sort_split32_raw<uint64_t>(
            cache, keys, values, num_items, value_words, nan_policy, stream);
      case CubSortKeyType::i64:
        return sort_split32_raw<int64_t>(
            cache, keys, values, num_items, value_words, nan_policy, stream);
      case CubSortKeyType::f64:
        return sort_split32_raw<double>(
            cache, keys, values, num_items, value_words, nan_policy, stream);
      default:
        throw std::runtime_error(
            "CUDA CUB split32 sort supports only u64/i64/f64 keys");
    }
  }

  switch (key_type) {
    case CubSortKeyType::u32:
      return sort_raw32_index<uint32_t>(
          cache, keys, values, num_items, value_words, stream);
    case CubSortKeyType::i32:
      return sort_raw32_index<int32_t>(
          cache, keys, values, num_items, value_words, stream);
    case CubSortKeyType::f32:
      return sort_f32_raw_index(cache, keys, values, num_items, value_words,
                                nan_policy, stream);
    case CubSortKeyType::u64:
      return sort_raw64_index<uint64_t>(
          cache, keys, values, num_items, value_words, nan_policy, stream);
    case CubSortKeyType::i64:
      return sort_raw64_index<int64_t>(
          cache, keys, values, num_items, value_words, nan_policy, stream);
    case CubSortKeyType::f64:
      return sort_raw64_index<double>(
          cache, keys, values, num_items, value_words, nan_policy, stream);
  }
  throw std::runtime_error("Unsupported CUB sort key type");
}

template <typename ValueT>
std::size_t cub_radix_sort_value_impl(CubSortCache &cache,
                                      void *keys,
                                      void *values,
                                      int num_items,
                                      CubSortKeyType key_type,
                                      CubSortMode mode,
                                      CubSortNanPolicy nan_policy,
                                      bool has_values,
                                      void *stream) {
  if (mode == CubSortMode::split32) {
    switch (key_type) {
      case CubSortKeyType::u64:
        return sort_split32<uint64_t, ValueT>(
            cache, keys, values, num_items, has_values, nan_policy, stream);
      case CubSortKeyType::i64:
        return sort_split32<int64_t, ValueT>(
            cache, keys, values, num_items, has_values, nan_policy, stream);
      case CubSortKeyType::f64:
        return sort_split32<double, ValueT>(
            cache, keys, values, num_items, has_values, nan_policy, stream);
      default:
        throw std::runtime_error(
            "CUDA CUB split32 sort supports only u64/i64/f64 keys");
    }
  }

  switch (key_type) {
    case CubSortKeyType::u32:
      return sort_typed<uint32_t, ValueT>(cache, keys, values, num_items,
                                          has_values, stream);
    case CubSortKeyType::i32:
      return sort_typed<int32_t, ValueT>(cache, keys, values, num_items,
                                         has_values, stream);
    case CubSortKeyType::f32:
      return sort_f32_transformed<ValueT>(cache, keys, values, num_items,
                                          has_values, nan_policy, stream);
    case CubSortKeyType::u64:
      return sort_typed<uint64_t, ValueT>(cache, keys, values, num_items,
                                          has_values, stream);
    case CubSortKeyType::i64:
      return sort_typed<int64_t, ValueT>(cache, keys, values, num_items,
                                         has_values, stream);
    case CubSortKeyType::f64:
      return sort_u64_transformed<double, ValueT>(
          cache, keys, values, num_items, has_values, nan_policy, stream);
  }
  throw std::runtime_error("Unsupported CUB sort key type");
}

}  // namespace

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
                                void *owner) {
  if (!keys) {
    throw std::runtime_error("CUB sort received a null key pointer");
  }
  if (has_values && !values) {
    throw std::runtime_error("CUB sort received a null value pointer");
  }
  std::lock_guard<std::mutex> lock(get_cache_mutex());
  CubSortCache &cache = get_cache(owner);
  if (!has_values) {
    return cub_radix_sort_value_impl<int32_t>(
        cache, keys, values, num_items, key_type, mode, nan_policy, has_values,
        stream);
  }
  int expected_value_words = 0;
  switch (value_type) {
    case CubSortValueType::i32:
    case CubSortValueType::f32:
    case CubSortValueType::u32:
      expected_value_words = 1;
      break;
    case CubSortValueType::u64:
    case CubSortValueType::i64:
    case CubSortValueType::f64:
      expected_value_words = 2;
      break;
  }
  if (expected_value_words == 0) {
    throw std::runtime_error("Unsupported CUB sort value type");
  }
  if (value_words != expected_value_words) {
    return cub_radix_sort_raw_value_impl(cache, keys, values, num_items,
                                         key_type, mode, nan_policy,
                                         value_words, stream);
  }
  switch (value_type) {
    case CubSortValueType::i32:
      return cub_radix_sort_value_impl<int32_t>(
          cache, keys, values, num_items, key_type, mode, nan_policy,
          has_values, stream);
    case CubSortValueType::f32:
      return cub_radix_sort_value_impl<float>(
          cache, keys, values, num_items, key_type, mode, nan_policy,
          has_values, stream);
    case CubSortValueType::u32:
      return cub_radix_sort_value_impl<uint32_t>(
          cache, keys, values, num_items, key_type, mode, nan_policy,
          has_values, stream);
    case CubSortValueType::u64:
      return cub_radix_sort_value_impl<uint64_t>(
          cache, keys, values, num_items, key_type, mode, nan_policy,
          has_values, stream);
    case CubSortValueType::i64:
      return cub_radix_sort_value_impl<int64_t>(
          cache, keys, values, num_items, key_type, mode, nan_policy,
          has_values, stream);
    case CubSortValueType::f64:
      return cub_radix_sort_value_impl<double>(
          cache, keys, values, num_items, key_type, mode, nan_policy,
          has_values, stream);
  }
  throw std::runtime_error("Unsupported CUB sort value type");
}

void cub_radix_sort_clear_cache_impl(void *owner) {
  static int fallback_owner = 0;
  if (!owner) {
    owner = &fallback_owner;
  }
  std::lock_guard<std::mutex> lock(get_cache_mutex());
  auto &caches = get_caches();
  auto it = caches.find(owner);
  if (it != caches.end()) {
    caches.erase(it);
  }
}

std::size_t cub_radix_sort_cached_bytes_impl(void *owner) {
  static int fallback_owner = 0;
  if (!owner) {
    owner = &fallback_owner;
  }
  std::lock_guard<std::mutex> lock(get_cache_mutex());
  auto &caches = get_caches();
  auto it = caches.find(owner);
  if (it == caches.end()) {
    return 0;
  }
  return it->second->allocated_bytes();
}

template <typename T>
std::size_t inclusive_scan_typed(CubScanCache &cache,
                                 void *data,
                                 int num_items,
                                 void *stream_ptr) {
  T *data_in_out = static_cast<T *>(data);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const bool use_stream = stream != nullptr;
  ensure_device_cache(cache);

  std::size_t temp_storage_bytes = 0;
  if (use_stream) {
    TI_CUDA_SORT_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, temp_storage_bytes, data_in_out, data_in_out, num_items,
        stream));
  } else {
    TI_CUDA_SORT_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, temp_storage_bytes, data_in_out, data_in_out, num_items));
  }
  ensure_buffer(&cache.temp_storage, &cache.temp_storage_bytes,
                temp_storage_bytes);
  if (use_stream) {
    TI_CUDA_SORT_CHECK(cub::DeviceScan::InclusiveSum(
        cache.temp_storage, temp_storage_bytes, data_in_out, data_in_out,
        num_items, stream));
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cub::DeviceScan::InclusiveSum(
        cache.temp_storage, temp_storage_bytes, data_in_out, data_in_out,
        num_items));
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }
  return cache.allocated_bytes();
}

template <typename T>
std::size_t inclusive_scan_strided_typed(CubScanCache &cache,
                                         void *data,
                                         int num_items,
                                         std::size_t offset,
                                         std::size_t stride,
                                         void *stream_ptr) {
  auto *data_in_out =
      reinterpret_cast<T *>(static_cast<uint8_t *>(data) + offset);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const bool use_stream = stream != nullptr;
  ensure_device_cache(cache);
  const auto stride_items = static_cast<std::ptrdiff_t>(stride / sizeof(T));
  auto strided_in_out =
      ::cuda::make_strided_iterator(data_in_out, stride_items);

  std::size_t temp_storage_bytes = 0;
  if (use_stream) {
    TI_CUDA_SORT_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, temp_storage_bytes, strided_in_out, strided_in_out, num_items,
        stream));
  } else {
    TI_CUDA_SORT_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, temp_storage_bytes, strided_in_out, strided_in_out,
        num_items));
  }
  ensure_buffer(&cache.temp_storage, &cache.temp_storage_bytes,
                temp_storage_bytes);
  if (use_stream) {
    TI_CUDA_SORT_CHECK(cub::DeviceScan::InclusiveSum(
        cache.temp_storage, temp_storage_bytes, strided_in_out, strided_in_out,
        num_items, stream));
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cub::DeviceScan::InclusiveSum(
        cache.temp_storage, temp_storage_bytes, strided_in_out, strided_in_out,
        num_items));
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }
  return cache.allocated_bytes();
}

std::size_t cub_inclusive_scan_impl(void *data,
                                    int num_items,
                                    CubScanValueType value_type,
                                    void *stream,
                                    void *owner) {
  if (!data) {
    throw std::runtime_error("CUB scan received a null data pointer");
  }
  std::lock_guard<std::mutex> lock(get_cache_mutex());
  CubScanCache &cache = get_scan_cache(owner);
  switch (value_type) {
    case CubScanValueType::i32:
      return inclusive_scan_typed<int32_t>(cache, data, num_items, stream);
    case CubScanValueType::f32:
      return inclusive_scan_typed<float>(cache, data, num_items, stream);
    case CubScanValueType::u32:
      return inclusive_scan_typed<uint32_t>(cache, data, num_items, stream);
    case CubScanValueType::u64:
      return inclusive_scan_typed<uint64_t>(cache, data, num_items, stream);
    case CubScanValueType::i64:
      return inclusive_scan_typed<int64_t>(cache, data, num_items, stream);
    case CubScanValueType::f64:
      return inclusive_scan_typed<double>(cache, data, num_items, stream);
  }
  throw std::runtime_error("Unsupported CUB scan value type");
}

std::size_t cub_inclusive_scan_strided_impl(void *data,
                                            int num_items,
                                            CubScanValueType value_type,
                                            std::size_t offset,
                                            std::size_t stride,
                                            void *stream,
                                            void *owner) {
  if (!data) {
    throw std::runtime_error("CUB strided scan received a null data pointer");
  }
  std::lock_guard<std::mutex> lock(get_cache_mutex());
  CubScanCache &cache = get_scan_cache(owner);
  switch (value_type) {
    case CubScanValueType::i32:
      return inclusive_scan_strided_typed<int32_t>(
          cache, data, num_items, offset, stride, stream);
    case CubScanValueType::f32:
      return inclusive_scan_strided_typed<float>(
          cache, data, num_items, offset, stride, stream);
    case CubScanValueType::u32:
      return inclusive_scan_strided_typed<uint32_t>(
          cache, data, num_items, offset, stride, stream);
    case CubScanValueType::u64:
      return inclusive_scan_strided_typed<uint64_t>(
          cache, data, num_items, offset, stride, stream);
    case CubScanValueType::i64:
      return inclusive_scan_strided_typed<int64_t>(
          cache, data, num_items, offset, stride, stream);
    case CubScanValueType::f64:
      return inclusive_scan_strided_typed<double>(
          cache, data, num_items, offset, stride, stream);
  }
  throw std::runtime_error("Unsupported CUB strided scan value type");
}

void cub_inclusive_scan_clear_cache_impl(void *owner) {
  static int fallback_owner = 0;
  if (!owner) {
    owner = &fallback_owner;
  }
  std::lock_guard<std::mutex> lock(get_cache_mutex());
  auto &caches = get_scan_caches();
  auto it = caches.find(owner);
  if (it != caches.end()) {
    caches.erase(it);
  }
}

std::size_t cub_inclusive_scan_cached_bytes_impl(void *owner) {
  static int fallback_owner = 0;
  if (!owner) {
    owner = &fallback_owner;
  }
  std::lock_guard<std::mutex> lock(get_cache_mutex());
  auto &caches = get_scan_caches();
  auto it = caches.find(owner);
  if (it == caches.end()) {
    return 0;
  }
  return it->second->allocated_bytes();
}

int scalar_words(CubSelectValueType value_type);

template <typename T>
std::size_t select_flagged_typed(CubSelectCache &cache,
                                 void *values,
                                 void *flags,
                                 void *output,
                                 void *count,
                                 int num_items,
                                 void *stream_ptr) {
  if (!values || !flags || !output || !count) {
    throw std::runtime_error("CUB select received a null pointer");
  }
  const T *values_in = static_cast<const T *>(values);
  const int32_t *flags_in = static_cast<const int32_t *>(flags);
  T *values_out = static_cast<T *>(output);
  int *count_out = static_cast<int *>(count);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const bool use_stream = stream != nullptr;
  ensure_device_cache(cache);

  std::size_t temp_storage_bytes = 0;
  if (use_stream) {
    TI_CUDA_SORT_CHECK(cub::DeviceSelect::Flagged(
        nullptr, temp_storage_bytes, values_in, flags_in, values_out, count_out,
        num_items, stream));
  } else {
    TI_CUDA_SORT_CHECK(cub::DeviceSelect::Flagged(
        nullptr, temp_storage_bytes, values_in, flags_in, values_out, count_out,
        num_items));
  }
  ensure_buffer(&cache.temp_storage, &cache.temp_storage_bytes,
                temp_storage_bytes);
  if (use_stream) {
    TI_CUDA_SORT_CHECK(cub::DeviceSelect::Flagged(
        cache.temp_storage, temp_storage_bytes, values_in, flags_in, values_out,
        count_out, num_items, stream));
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cub::DeviceSelect::Flagged(
        cache.temp_storage, temp_storage_bytes, values_in, flags_in, values_out,
        count_out, num_items));
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }
  return cache.allocated_bytes();
}

std::size_t select_flagged_words(CubSelectCache &cache,
                                 void *values,
                                 void *flags,
                                 void *output,
                                 void *count,
                                 int num_items,
                                 int item_words,
                                 void *stream_ptr) {
  if (!values || !flags || !output || !count) {
    throw std::runtime_error("CUB select received a null pointer");
  }
  if (item_words <= 0) {
    throw std::runtime_error("CUB select expects positive item_words");
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const bool use_stream = stream != nullptr;
  ensure_device_cache(cache);
  const int total_words = num_items * item_words;
  constexpr int kBlockDim = 256;
  const int item_grid = (num_items + kBlockDim - 1) / kBlockDim;
  const int word_grid = (total_words + kBlockDim - 1) / kBlockDim;
  ensure_buffer(&cache.prefix, &cache.prefix_bytes,
                static_cast<std::size_t>(num_items) * sizeof(int32_t));
  auto *prefix = static_cast<int32_t *>(cache.prefix);
  auto *flags_in = static_cast<const int32_t *>(flags);
  select_flags_to_prefix_kernel<<<item_grid, kBlockDim, 0, stream>>>(
      flags_in, prefix, num_items);
  TI_CUDA_SORT_CHECK(cudaGetLastError());

  std::size_t temp_storage_bytes = 0;
  if (use_stream) {
    TI_CUDA_SORT_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, temp_storage_bytes, prefix, prefix, num_items, stream));
  } else {
    TI_CUDA_SORT_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, temp_storage_bytes, prefix, prefix, num_items));
  }
  ensure_buffer(&cache.temp_storage, &cache.temp_storage_bytes,
                temp_storage_bytes);
  if (use_stream) {
    TI_CUDA_SORT_CHECK(cub::DeviceScan::InclusiveSum(
        cache.temp_storage, temp_storage_bytes, prefix, prefix, num_items,
        stream));
  } else {
    TI_CUDA_SORT_CHECK(cub::DeviceScan::InclusiveSum(
        cache.temp_storage, temp_storage_bytes, prefix, prefix, num_items));
  }

  select_scatter_words_kernel<<<word_grid, kBlockDim, 0, stream>>>(
      static_cast<const uint32_t *>(values), flags_in, prefix,
      static_cast<uint32_t *>(output), static_cast<int32_t *>(count), num_items,
      item_words, total_words);
  TI_CUDA_SORT_CHECK(cudaGetLastError());
  if (use_stream) {
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }
  return cache.allocated_bytes();
}

std::size_t cub_select_flagged_impl(void *values,
                                    void *flags,
                                    void *output,
                                    void *count,
                                    int num_items,
                                    CubSelectValueType value_type,
                                    int item_words,
                                    void *stream,
                                    void *owner) {
  std::lock_guard<std::mutex> lock(get_cache_mutex());
  CubSelectCache &cache = get_select_cache(owner);
  const int expected_words = scalar_words(value_type);
  if (expected_words == 0) {
    throw std::runtime_error("Unsupported CUB select value type");
  }
  if (item_words != expected_words) {
    return select_flagged_words(cache, values, flags, output, count, num_items,
                                item_words, stream);
  }
  switch (value_type) {
    case CubSelectValueType::i32:
      return select_flagged_typed<int32_t>(cache, values, flags, output, count,
                                           num_items, stream);
    case CubSelectValueType::f32:
      return select_flagged_typed<float>(cache, values, flags, output, count,
                                         num_items, stream);
    case CubSelectValueType::u32:
      return select_flagged_typed<uint32_t>(cache, values, flags, output, count,
                                            num_items, stream);
    case CubSelectValueType::u64:
      return select_flagged_typed<uint64_t>(cache, values, flags, output, count,
                                            num_items, stream);
    case CubSelectValueType::i64:
      return select_flagged_typed<int64_t>(cache, values, flags, output, count,
                                           num_items, stream);
    case CubSelectValueType::f64:
      return select_flagged_typed<double>(cache, values, flags, output, count,
                                          num_items, stream);
  }
  throw std::runtime_error("Unsupported CUB select value type");
}

void cub_select_clear_cache_impl(void *owner) {
  static int fallback_owner = 0;
  if (!owner) {
    owner = &fallback_owner;
  }
  std::lock_guard<std::mutex> lock(get_cache_mutex());
  auto &caches = get_select_caches();
  auto it = caches.find(owner);
  if (it != caches.end()) {
    caches.erase(it);
  }
}

std::size_t cub_select_cached_bytes_impl(void *owner) {
  static int fallback_owner = 0;
  if (!owner) {
    owner = &fallback_owner;
  }
  std::lock_guard<std::mutex> lock(get_cache_mutex());
  auto &caches = get_select_caches();
  auto it = caches.find(owner);
  if (it == caches.end()) {
    return 0;
  }
  return it->second->allocated_bytes();
}

template <typename T, typename CounterT>
std::size_t histogram_even_typed(CubHistogramCache &cache,
                                 void *values,
                                 void *bins,
                                 int num_items,
                                 int num_bins,
                                 void *stream_ptr) {
  if (!values || !bins) {
    throw std::runtime_error("CUB histogram received a null pointer");
  }
  const T *samples_in = static_cast<const T *>(values);
  CounterT *hist_out = static_cast<CounterT *>(bins);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const bool use_stream = stream != nullptr;
  ensure_device_cache(cache);
  const std::size_t bin_bytes = sizeof(CounterT) * num_bins;

  if (num_items == 0) {
    if (use_stream) {
      TI_CUDA_SORT_CHECK(cudaMemsetAsync(hist_out, 0, bin_bytes, stream));
      TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
    } else {
      TI_CUDA_SORT_CHECK(cudaMemset(hist_out, 0, bin_bytes));
      TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
    }
    return cache.allocated_bytes();
  }

  std::size_t temp_storage_bytes = 0;
  if (use_stream) {
    TI_CUDA_SORT_CHECK(cub::DeviceHistogram::HistogramEven(
        nullptr, temp_storage_bytes, samples_in, hist_out, num_bins + 1,
        static_cast<T>(0), static_cast<T>(num_bins), num_items, stream));
  } else {
    TI_CUDA_SORT_CHECK(cub::DeviceHistogram::HistogramEven(
        nullptr, temp_storage_bytes, samples_in, hist_out, num_bins + 1,
        static_cast<T>(0), static_cast<T>(num_bins), num_items));
  }
  ensure_buffer(&cache.temp_storage, &cache.temp_storage_bytes,
                temp_storage_bytes);
  if (use_stream) {
    TI_CUDA_SORT_CHECK(cub::DeviceHistogram::HistogramEven(
        cache.temp_storage, temp_storage_bytes, samples_in, hist_out,
        num_bins + 1, static_cast<T>(0), static_cast<T>(num_bins), num_items,
        stream));
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cub::DeviceHistogram::HistogramEven(
        cache.temp_storage, temp_storage_bytes, samples_in, hist_out,
        num_bins + 1, static_cast<T>(0), static_cast<T>(num_bins), num_items));
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }
  return cache.allocated_bytes();
}

template <typename CounterT>
std::size_t histogram_even_sample_dispatch(CubHistogramCache &cache,
                                           void *values,
                                           void *bins,
                                           int num_items,
                                           int num_bins,
                                           CubHistogramValueType value_type,
                                           void *stream) {
  switch (value_type) {
    case CubHistogramValueType::i32:
      return histogram_even_typed<int32_t, CounterT>(
          cache, values, bins, num_items, num_bins, stream);
    case CubHistogramValueType::u32:
      return histogram_even_typed<uint32_t, CounterT>(
          cache, values, bins, num_items, num_bins, stream);
  }
  throw std::runtime_error("Unsupported CUB histogram value type");
}

template <typename T>
std::size_t histogram_i64_direct_typed(CubHistogramCache &cache,
                                       void *values,
                                       void *bins,
                                       int num_items,
                                       int num_bins,
                                       void *stream_ptr) {
  if (!values || !bins) {
    throw std::runtime_error("CUDA histogram received a null pointer");
  }
  const T *samples_in = static_cast<const T *>(values);
  int64_t *hist_out = static_cast<int64_t *>(bins);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const bool use_stream = stream != nullptr;
  ensure_device_cache(cache);
  const std::size_t bin_bytes = sizeof(int64_t) * num_bins;
  if (use_stream) {
    TI_CUDA_SORT_CHECK(cudaMemsetAsync(hist_out, 0, bin_bytes, stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaMemset(hist_out, 0, bin_bytes));
  }
  if (num_items > 0) {
    constexpr int kBlockDim = 256;
    const int grid_dim = (num_items + kBlockDim - 1) / kBlockDim;
    histogram_i64_direct_kernel<T><<<grid_dim, kBlockDim, 0, stream>>>(
        samples_in, hist_out, num_items, num_bins);
    TI_CUDA_SORT_CHECK(cudaGetLastError());
  }
  if (use_stream) {
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }
  return cache.allocated_bytes();
}

std::size_t histogram_i64_sample_dispatch(CubHistogramCache &cache,
                                          void *values,
                                          void *bins,
                                          int num_items,
                                          int num_bins,
                                          CubHistogramValueType value_type,
                                          void *stream) {
  switch (value_type) {
    case CubHistogramValueType::i32:
      return histogram_i64_direct_typed<int32_t>(
          cache, values, bins, num_items, num_bins, stream);
    case CubHistogramValueType::u32:
      return histogram_i64_direct_typed<uint32_t>(
          cache, values, bins, num_items, num_bins, stream);
  }
  throw std::runtime_error("Unsupported CUDA histogram value type");
}

std::size_t cub_histogram_even_impl(void *values,
                                    void *bins,
                                    int num_items,
                                    int num_bins,
                                    CubHistogramValueType value_type,
                                    CubHistogramBinType bin_type,
                                    void *stream,
                                    void *owner) {
  std::lock_guard<std::mutex> lock(get_cache_mutex());
  CubHistogramCache &cache = get_histogram_cache(owner);
  switch (bin_type) {
    case CubHistogramBinType::i32:
      return histogram_even_sample_dispatch<int32_t>(
          cache, values, bins, num_items, num_bins, value_type, stream);
    case CubHistogramBinType::i64:
      return histogram_i64_sample_dispatch(
          cache, values, bins, num_items, num_bins, value_type, stream);
  }
  throw std::runtime_error("Unsupported CUB histogram bin type");
}

void cub_histogram_clear_cache_impl(void *owner) {
  static int fallback_owner = 0;
  if (!owner) {
    owner = &fallback_owner;
  }
  std::lock_guard<std::mutex> lock(get_cache_mutex());
  auto &caches = get_histogram_caches();
  auto it = caches.find(owner);
  if (it != caches.end()) {
    caches.erase(it);
  }
}

std::size_t cub_histogram_cached_bytes_impl(void *owner) {
  static int fallback_owner = 0;
  if (!owner) {
    owner = &fallback_owner;
  }
  std::lock_guard<std::mutex> lock(get_cache_mutex());
  auto &caches = get_histogram_caches();
  auto it = caches.find(owner);
  if (it == caches.end()) {
    return 0;
  }
  return it->second->allocated_bytes();
}

template <typename T>
std::size_t reduce_typed(CubReduceCache &cache,
                         void *values,
                         void *output,
                         int num_items,
                         CubReduceOp op,
                         void *stream_ptr) {
  if (!values || !output) {
    throw std::runtime_error("CUB reduce received a null pointer");
  }
  const T *values_in = static_cast<const T *>(values);
  T *output_out = static_cast<T *>(output);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const bool use_stream = stream != nullptr;
  ensure_device_cache(cache);

  std::size_t temp_storage_bytes = 0;
  auto query = [&]() {
    switch (op) {
      case CubReduceOp::sum:
        if (use_stream) {
          TI_CUDA_SORT_CHECK(cub::DeviceReduce::Sum(
              nullptr, temp_storage_bytes, values_in, output_out, num_items,
              stream));
        } else {
          TI_CUDA_SORT_CHECK(cub::DeviceReduce::Sum(
              nullptr, temp_storage_bytes, values_in, output_out, num_items));
        }
        break;
      case CubReduceOp::min:
        if (use_stream) {
          TI_CUDA_SORT_CHECK(cub::DeviceReduce::Min(
              nullptr, temp_storage_bytes, values_in, output_out, num_items,
              stream));
        } else {
          TI_CUDA_SORT_CHECK(cub::DeviceReduce::Min(
              nullptr, temp_storage_bytes, values_in, output_out, num_items));
        }
        break;
      case CubReduceOp::max:
        if (use_stream) {
          TI_CUDA_SORT_CHECK(cub::DeviceReduce::Max(
              nullptr, temp_storage_bytes, values_in, output_out, num_items,
              stream));
        } else {
          TI_CUDA_SORT_CHECK(cub::DeviceReduce::Max(
              nullptr, temp_storage_bytes, values_in, output_out, num_items));
        }
        break;
    }
  };
  query();
  ensure_buffer(&cache.temp_storage, &cache.temp_storage_bytes,
                temp_storage_bytes);

  switch (op) {
    case CubReduceOp::sum:
      if (use_stream) {
        TI_CUDA_SORT_CHECK(cub::DeviceReduce::Sum(
            cache.temp_storage, temp_storage_bytes, values_in, output_out,
            num_items, stream));
      } else {
        TI_CUDA_SORT_CHECK(cub::DeviceReduce::Sum(
            cache.temp_storage, temp_storage_bytes, values_in, output_out,
            num_items));
      }
      break;
    case CubReduceOp::min:
      if (use_stream) {
        TI_CUDA_SORT_CHECK(cub::DeviceReduce::Min(
            cache.temp_storage, temp_storage_bytes, values_in, output_out,
            num_items, stream));
      } else {
        TI_CUDA_SORT_CHECK(cub::DeviceReduce::Min(
            cache.temp_storage, temp_storage_bytes, values_in, output_out,
            num_items));
      }
      break;
    case CubReduceOp::max:
      if (use_stream) {
        TI_CUDA_SORT_CHECK(cub::DeviceReduce::Max(
            cache.temp_storage, temp_storage_bytes, values_in, output_out,
            num_items, stream));
      } else {
        TI_CUDA_SORT_CHECK(cub::DeviceReduce::Max(
            cache.temp_storage, temp_storage_bytes, values_in, output_out,
            num_items));
      }
      break;
  }

  if (use_stream) {
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }
  return cache.allocated_bytes();
}

template <typename T>
struct StridedLoadOp {
  const uint8_t *base{nullptr};
  std::size_t offset{0};
  std::size_t stride{0};

  __host__ __device__ T operator()(const int &i) const {
    const auto *value = reinterpret_cast<const T *>(
        base + offset + static_cast<std::size_t>(i) * stride);
    return *value;
  }
};

template <typename T>
std::size_t reduce_strided_typed(CubReduceCache &cache,
                                 void *values,
                                 void *output,
                                 int num_items,
                                 std::size_t offset,
                                 std::size_t stride,
                                 CubReduceOp op,
                                 void *stream_ptr) {
  if (!values || !output) {
    throw std::runtime_error("CUB strided reduce received a null pointer");
  }
  const auto *values_in = static_cast<const uint8_t *>(values);
  T *output_out = static_cast<T *>(output);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const bool use_stream = stream != nullptr;
  ensure_device_cache(cache);

  StridedLoadOp<T> load_op{values_in, offset, stride};
  auto counting = ::cuda::make_counting_iterator(0);
  auto strided_in = ::cuda::make_transform_iterator(counting, load_op);

  std::size_t temp_storage_bytes = 0;
  auto query = [&]() {
    switch (op) {
      case CubReduceOp::sum:
        if (use_stream) {
          TI_CUDA_SORT_CHECK(cub::DeviceReduce::Sum(
              nullptr, temp_storage_bytes, strided_in, output_out, num_items,
              stream));
        } else {
          TI_CUDA_SORT_CHECK(cub::DeviceReduce::Sum(
              nullptr, temp_storage_bytes, strided_in, output_out, num_items));
        }
        break;
      case CubReduceOp::min:
        if (use_stream) {
          TI_CUDA_SORT_CHECK(cub::DeviceReduce::Min(
              nullptr, temp_storage_bytes, strided_in, output_out, num_items,
              stream));
        } else {
          TI_CUDA_SORT_CHECK(cub::DeviceReduce::Min(
              nullptr, temp_storage_bytes, strided_in, output_out, num_items));
        }
        break;
      case CubReduceOp::max:
        if (use_stream) {
          TI_CUDA_SORT_CHECK(cub::DeviceReduce::Max(
              nullptr, temp_storage_bytes, strided_in, output_out, num_items,
              stream));
        } else {
          TI_CUDA_SORT_CHECK(cub::DeviceReduce::Max(
              nullptr, temp_storage_bytes, strided_in, output_out, num_items));
        }
        break;
    }
  };
  query();
  ensure_buffer(&cache.temp_storage, &cache.temp_storage_bytes,
                temp_storage_bytes);

  switch (op) {
    case CubReduceOp::sum:
      if (use_stream) {
        TI_CUDA_SORT_CHECK(cub::DeviceReduce::Sum(
            cache.temp_storage, temp_storage_bytes, strided_in, output_out,
            num_items, stream));
      } else {
        TI_CUDA_SORT_CHECK(cub::DeviceReduce::Sum(
            cache.temp_storage, temp_storage_bytes, strided_in, output_out,
            num_items));
      }
      break;
    case CubReduceOp::min:
      if (use_stream) {
        TI_CUDA_SORT_CHECK(cub::DeviceReduce::Min(
            cache.temp_storage, temp_storage_bytes, strided_in, output_out,
            num_items, stream));
      } else {
        TI_CUDA_SORT_CHECK(cub::DeviceReduce::Min(
            cache.temp_storage, temp_storage_bytes, strided_in, output_out,
            num_items));
      }
      break;
    case CubReduceOp::max:
      if (use_stream) {
        TI_CUDA_SORT_CHECK(cub::DeviceReduce::Max(
            cache.temp_storage, temp_storage_bytes, strided_in, output_out,
            num_items, stream));
      } else {
        TI_CUDA_SORT_CHECK(cub::DeviceReduce::Max(
            cache.temp_storage, temp_storage_bytes, strided_in, output_out,
            num_items));
      }
      break;
  }

  if (use_stream) {
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }
  return cache.allocated_bytes();
}

std::size_t cub_reduce_impl(void *values,
                            void *output,
                            int num_items,
                            CubReduceValueType value_type,
                            CubReduceOp op,
                            void *stream,
                            void *owner) {
  std::lock_guard<std::mutex> lock(get_cache_mutex());
  CubReduceCache &cache = get_reduce_cache(owner);
  switch (value_type) {
    case CubReduceValueType::i32:
      return reduce_typed<int32_t>(cache, values, output, num_items, op,
                                   stream);
    case CubReduceValueType::f32:
      return reduce_typed<float>(cache, values, output, num_items, op, stream);
    case CubReduceValueType::u32:
      return reduce_typed<uint32_t>(cache, values, output, num_items, op,
                                    stream);
    case CubReduceValueType::u64:
      return reduce_typed<uint64_t>(cache, values, output, num_items, op,
                                    stream);
    case CubReduceValueType::i64:
      return reduce_typed<int64_t>(cache, values, output, num_items, op,
                                   stream);
    case CubReduceValueType::f64:
      return reduce_typed<double>(cache, values, output, num_items, op,
                                  stream);
  }
  throw std::runtime_error("Unsupported CUB reduce value type");
}

std::size_t cub_reduce_strided_impl(void *values,
                                    void *output,
                                    int num_items,
                                    CubReduceValueType value_type,
                                    std::size_t offset,
                                    std::size_t stride,
                                    CubReduceOp op,
                                    void *stream,
                                    void *owner) {
  std::lock_guard<std::mutex> lock(get_cache_mutex());
  CubReduceCache &cache = get_reduce_cache(owner);
  switch (value_type) {
    case CubReduceValueType::i32:
      return reduce_strided_typed<int32_t>(
          cache, values, output, num_items, offset, stride, op, stream);
    case CubReduceValueType::f32:
      return reduce_strided_typed<float>(
          cache, values, output, num_items, offset, stride, op, stream);
    case CubReduceValueType::u32:
      return reduce_strided_typed<uint32_t>(
          cache, values, output, num_items, offset, stride, op, stream);
    case CubReduceValueType::u64:
      return reduce_strided_typed<uint64_t>(
          cache, values, output, num_items, offset, stride, op, stream);
    case CubReduceValueType::i64:
      return reduce_strided_typed<int64_t>(
          cache, values, output, num_items, offset, stride, op, stream);
    case CubReduceValueType::f64:
      return reduce_strided_typed<double>(
          cache, values, output, num_items, offset, stride, op, stream);
  }
  throw std::runtime_error("Unsupported CUB strided reduce value type");
}

void cub_reduce_clear_cache_impl(void *owner) {
  static int fallback_owner = 0;
  if (!owner) {
    owner = &fallback_owner;
  }
  std::lock_guard<std::mutex> lock(get_cache_mutex());
  auto &caches = get_reduce_caches();
  auto it = caches.find(owner);
  if (it != caches.end()) {
    caches.erase(it);
  }
}

std::size_t cub_reduce_cached_bytes_impl(void *owner) {
  static int fallback_owner = 0;
  if (!owner) {
    owner = &fallback_owner;
  }
  std::lock_guard<std::mutex> lock(get_cache_mutex());
  auto &caches = get_reduce_caches();
  auto it = caches.find(owner);
  if (it == caches.end()) {
    return 0;
  }
  return it->second->allocated_bytes();
}

std::size_t cub_scatter_add_impl(void *src,
                                 void *indices,
                                 void *dst,
                                 int num_items,
                                 int index_bound,
                                 CudaScatterAddValueType value_type,
                                 void *stream_ptr) {
  if (!src || !indices || !dst) {
    throw std::runtime_error("CUDA scatter-add received a null pointer");
  }
  if (num_items == 0) {
    return 0;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  constexpr int kBlockDim = 256;
  const int grid_dim = (num_items + kBlockDim - 1) / kBlockDim;
  const int32_t *indices_in = static_cast<const int32_t *>(indices);
  switch (value_type) {
    case CudaScatterAddValueType::i32:
      scatter_add_by_i32_kernel<int32_t><<<grid_dim, kBlockDim, 0, stream>>>(
          static_cast<const int32_t *>(src), indices_in,
          static_cast<int32_t *>(dst), num_items, index_bound);
      break;
    case CudaScatterAddValueType::f32:
      scatter_add_by_i32_kernel<float><<<grid_dim, kBlockDim, 0, stream>>>(
          static_cast<const float *>(src), indices_in, static_cast<float *>(dst),
          num_items, index_bound);
      break;
    case CudaScatterAddValueType::u32:
      scatter_add_by_i32_kernel<uint32_t><<<grid_dim, kBlockDim, 0, stream>>>(
          static_cast<const uint32_t *>(src), indices_in,
          static_cast<uint32_t *>(dst), num_items, index_bound);
      break;
    case CudaScatterAddValueType::u64:
      scatter_add_by_i32_kernel<uint64_t><<<grid_dim, kBlockDim, 0, stream>>>(
          static_cast<const uint64_t *>(src), indices_in,
          static_cast<uint64_t *>(dst), num_items, index_bound);
      break;
    case CudaScatterAddValueType::i64:
      scatter_add_by_i32_kernel<int64_t><<<grid_dim, kBlockDim, 0, stream>>>(
          static_cast<const int64_t *>(src), indices_in,
          static_cast<int64_t *>(dst), num_items, index_bound);
      break;
    case CudaScatterAddValueType::f64:
      scatter_add_by_i32_kernel<double><<<grid_dim, kBlockDim, 0, stream>>>(
          static_cast<const double *>(src), indices_in,
          static_cast<double *>(dst), num_items, index_bound);
      break;
    default:
      throw std::runtime_error("Unsupported CUDA scatter-add value type");
  }
  TI_CUDA_SORT_CHECK(cudaGetLastError());
  if (stream) {
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }
  return 0;
}

template <typename T>
void scatter_add_strided_launch(const uint8_t *src,
                                const int32_t *indices,
                                T *dst,
                                int num_items,
                                int index_bound,
                                std::size_t offset,
                                std::size_t stride,
                                cudaStream_t stream) {
  constexpr int kBlockDim = 256;
  const int grid_dim = (num_items + kBlockDim - 1) / kBlockDim;
  scatter_add_by_i32_strided_kernel<T><<<grid_dim, kBlockDim, 0, stream>>>(
      src, indices, dst, num_items, index_bound, offset, stride);
}

std::size_t cub_scatter_add_strided_impl(void *src,
                                         void *indices,
                                         void *dst,
                                         int num_items,
                                         int index_bound,
                                         CudaScatterAddValueType value_type,
                                         std::size_t offset,
                                         std::size_t stride,
                                         void *stream_ptr) {
  if (!src || !indices || !dst) {
    throw std::runtime_error(
        "CUDA strided scatter-add received a null pointer");
  }
  if (num_items == 0) {
    return 0;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const auto *src_in = static_cast<const uint8_t *>(src);
  const auto *indices_in = static_cast<const int32_t *>(indices);
  switch (value_type) {
    case CudaScatterAddValueType::i32:
      scatter_add_strided_launch(src_in, indices_in,
                                 static_cast<int32_t *>(dst), num_items,
                                 index_bound, offset, stride, stream);
      break;
    case CudaScatterAddValueType::f32:
      scatter_add_strided_launch(src_in, indices_in, static_cast<float *>(dst),
                                 num_items, index_bound, offset, stride,
                                 stream);
      break;
    case CudaScatterAddValueType::u32:
      scatter_add_strided_launch(src_in, indices_in,
                                 static_cast<uint32_t *>(dst), num_items,
                                 index_bound, offset, stride, stream);
      break;
    case CudaScatterAddValueType::u64:
      scatter_add_strided_launch(src_in, indices_in,
                                 static_cast<uint64_t *>(dst), num_items,
                                 index_bound, offset, stride, stream);
      break;
    case CudaScatterAddValueType::i64:
      scatter_add_strided_launch(src_in, indices_in,
                                 static_cast<int64_t *>(dst), num_items,
                                 index_bound, offset, stride, stream);
      break;
    case CudaScatterAddValueType::f64:
      scatter_add_strided_launch(src_in, indices_in, static_cast<double *>(dst),
                                 num_items, index_bound, offset, stride,
                                 stream);
      break;
    default:
      throw std::runtime_error(
          "Unsupported CUDA strided scatter-add value type");
  }
  TI_CUDA_SORT_CHECK(cudaGetLastError());
  if (stream) {
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }
  return 0;
}

template <typename T>
void scatter_add_strided_io_launch(const uint8_t *src,
                                   const int32_t *indices,
                                   uint8_t *dst,
                                   int num_items,
                                   int index_bound,
                                   std::size_t src_offset,
                                   std::size_t src_stride,
                                   std::size_t dst_offset,
                                   std::size_t dst_stride,
                                   cudaStream_t stream) {
  constexpr int kBlockDim = 256;
  const int grid_dim = (num_items + kBlockDim - 1) / kBlockDim;
  scatter_add_by_i32_strided_io_kernel<T>
      <<<grid_dim, kBlockDim, 0, stream>>>(
          src, indices, dst, num_items, index_bound, src_offset, src_stride,
          dst_offset, dst_stride);
}

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
                                            void *stream_ptr) {
  if (!src || !indices || !dst) {
    throw std::runtime_error(
        "CUDA strided scatter-add received a null pointer");
  }
  if (num_items == 0) {
    return 0;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const auto *src_in = static_cast<const uint8_t *>(src);
  const auto *indices_in = static_cast<const int32_t *>(indices);
  auto *dst_out = static_cast<uint8_t *>(dst);
  switch (value_type) {
    case CudaScatterAddValueType::i32:
      scatter_add_strided_io_launch<int32_t>(
          src_in, indices_in, dst_out, num_items, index_bound, src_offset,
          src_stride, dst_offset, dst_stride, stream);
      break;
    case CudaScatterAddValueType::f32:
      scatter_add_strided_io_launch<float>(
          src_in, indices_in, dst_out, num_items, index_bound, src_offset,
          src_stride, dst_offset, dst_stride, stream);
      break;
    case CudaScatterAddValueType::u32:
      scatter_add_strided_io_launch<uint32_t>(
          src_in, indices_in, dst_out, num_items, index_bound, src_offset,
          src_stride, dst_offset, dst_stride, stream);
      break;
    case CudaScatterAddValueType::u64:
      scatter_add_strided_io_launch<uint64_t>(
          src_in, indices_in, dst_out, num_items, index_bound, src_offset,
          src_stride, dst_offset, dst_stride, stream);
      break;
    case CudaScatterAddValueType::i64:
      scatter_add_strided_io_launch<int64_t>(
          src_in, indices_in, dst_out, num_items, index_bound, src_offset,
          src_stride, dst_offset, dst_stride, stream);
      break;
    case CudaScatterAddValueType::f64:
      scatter_add_strided_io_launch<double>(
          src_in, indices_in, dst_out, num_items, index_bound, src_offset,
          src_stride, dst_offset, dst_stride, stream);
      break;
    default:
      throw std::runtime_error(
          "Unsupported CUDA strided scatter-add value type");
  }
  TI_CUDA_SORT_CHECK(cudaGetLastError());
  if (stream) {
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }
  return 0;
}

std::size_t cub_indexed_copy_impl(void *src,
                                  void *indices,
                                  void *dst,
                                  int num_items,
                                  int index_bound,
                                  int item_words,
                                  CudaIndexedCopyOp op,
                                  void *stream_ptr) {
  if (!src || !indices || !dst) {
    throw std::runtime_error("CUDA indexed-copy received a null pointer");
  }
  if (num_items == 0 || index_bound == 0) {
    return 0;
  }
  if (item_words <= 0) {
    throw std::runtime_error("CUDA indexed-copy expects at least one word");
  }
  if (num_items > std::numeric_limits<int>::max() / item_words) {
    throw std::runtime_error("CUDA indexed-copy word count exceeds INT_MAX");
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  constexpr int kBlockDim = 256;
  const int total_words = num_items * item_words;
  const int grid_dim = (total_words + kBlockDim - 1) / kBlockDim;
  indexed_copy_words_by_i32_kernel<<<grid_dim, kBlockDim, 0, stream>>>(
      static_cast<const uint32_t *>(src), static_cast<const int32_t *>(indices),
      static_cast<uint32_t *>(dst), num_items, index_bound, item_words,
      static_cast<int>(op));
  TI_CUDA_SORT_CHECK(cudaGetLastError());
  if (stream) {
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }
  return 0;
}

std::size_t cub_transform_affine_impl(void *src,
                                      void *dst,
                                      int num_items,
                                      CudaTransformValueType value_type,
                                      double scale,
                                      double bias,
                                      void *stream_ptr) {
  if (!src || !dst) {
    throw std::runtime_error("CUDA transform received a null pointer");
  }
  if (num_items == 0) {
    return 0;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  constexpr int kBlockDim = 256;
  const int grid_dim = (num_items + kBlockDim - 1) / kBlockDim;
  switch (value_type) {
    case CudaTransformValueType::i32:
      transform_u32_affine_kernel<<<grid_dim, kBlockDim, 0, stream>>>(
          static_cast<const uint32_t *>(src), static_cast<uint32_t *>(dst),
          num_items,
          static_cast<uint32_t>(static_cast<int32_t>(scale)),
          static_cast<uint32_t>(static_cast<int32_t>(bias)));
      break;
    case CudaTransformValueType::u32:
      transform_u32_affine_kernel<<<grid_dim, kBlockDim, 0, stream>>>(
          static_cast<const uint32_t *>(src), static_cast<uint32_t *>(dst),
          num_items, static_cast<uint32_t>(scale),
          static_cast<uint32_t>(bias));
      break;
    case CudaTransformValueType::f32:
      transform_f32_affine_kernel<<<grid_dim, kBlockDim, 0, stream>>>(
          static_cast<const float *>(src), static_cast<float *>(dst), num_items,
          static_cast<float>(scale), static_cast<float>(bias));
      break;
    case CudaTransformValueType::u64:
      transform_u64_affine_kernel<<<grid_dim, kBlockDim, 0, stream>>>(
          static_cast<const uint64_t *>(src), static_cast<uint64_t *>(dst),
          num_items, static_cast<uint64_t>(scale),
          static_cast<uint64_t>(bias));
      break;
    case CudaTransformValueType::i64:
      transform_u64_affine_kernel<<<grid_dim, kBlockDim, 0, stream>>>(
          static_cast<const uint64_t *>(src), static_cast<uint64_t *>(dst),
          num_items,
          static_cast<uint64_t>(static_cast<int64_t>(scale)),
          static_cast<uint64_t>(static_cast<int64_t>(bias)));
      break;
    case CudaTransformValueType::f64:
      transform_f64_affine_kernel<<<grid_dim, kBlockDim, 0, stream>>>(
          static_cast<const double *>(src), static_cast<double *>(dst),
          num_items, scale, bias);
      break;
    default:
      throw std::runtime_error("Unsupported CUDA transform value type");
  }
  TI_CUDA_SORT_CHECK(cudaGetLastError());
  if (stream) {
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }
  return 0;
}

std::size_t cub_transform_affine_strided_impl(void *src,
                                              void *dst,
                                              int num_items,
                                              CudaTransformValueType value_type,
                                              std::size_t offset,
                                              std::size_t stride,
                                              double scale,
                                              double bias,
                                              void *stream_ptr) {
  if (!src || !dst) {
    throw std::runtime_error("CUDA strided transform received a null pointer");
  }
  if (num_items == 0) {
    return 0;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  constexpr int kBlockDim = 256;
  const int grid_dim = (num_items + kBlockDim - 1) / kBlockDim;
  const auto *src_bytes = static_cast<const uint8_t *>(src);
  switch (value_type) {
    case CudaTransformValueType::i32:
      transform_strided_affine_kernel<uint32_t>
          <<<grid_dim, kBlockDim, 0, stream>>>(
              src_bytes, static_cast<uint32_t *>(dst), num_items, offset,
              stride, static_cast<uint32_t>(static_cast<int32_t>(scale)),
              static_cast<uint32_t>(static_cast<int32_t>(bias)));
      break;
    case CudaTransformValueType::u32:
      transform_strided_affine_kernel<uint32_t>
          <<<grid_dim, kBlockDim, 0, stream>>>(
              src_bytes, static_cast<uint32_t *>(dst), num_items, offset,
              stride, static_cast<uint32_t>(scale), static_cast<uint32_t>(bias));
      break;
    case CudaTransformValueType::f32:
      transform_strided_affine_kernel<float>
          <<<grid_dim, kBlockDim, 0, stream>>>(
              src_bytes, static_cast<float *>(dst), num_items, offset, stride,
              static_cast<float>(scale), static_cast<float>(bias));
      break;
    case CudaTransformValueType::u64:
      transform_strided_affine_kernel<uint64_t>
          <<<grid_dim, kBlockDim, 0, stream>>>(
              src_bytes, static_cast<uint64_t *>(dst), num_items, offset,
              stride, static_cast<uint64_t>(scale), static_cast<uint64_t>(bias));
      break;
    case CudaTransformValueType::i64:
      transform_strided_affine_kernel<uint64_t>
          <<<grid_dim, kBlockDim, 0, stream>>>(
              src_bytes, static_cast<uint64_t *>(dst), num_items, offset,
              stride, static_cast<uint64_t>(static_cast<int64_t>(scale)),
              static_cast<uint64_t>(static_cast<int64_t>(bias)));
      break;
    case CudaTransformValueType::f64:
      transform_strided_affine_kernel<double>
          <<<grid_dim, kBlockDim, 0, stream>>>(
              src_bytes, static_cast<double *>(dst), num_items, offset, stride,
              scale, bias);
      break;
    default:
      throw std::runtime_error("Unsupported CUDA strided transform value type");
  }
  TI_CUDA_SORT_CHECK(cudaGetLastError());
  if (stream) {
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }
  return 0;
}

template <typename T>
void launch_bucket_scatter(const int32_t *keys_in,
                           const void *values,
                           int32_t *cursor_in_out,
                           void *output,
                           int num_items,
                           int num_bins,
                           int grid_dim,
                           int block_dim,
                           cudaStream_t stream) {
  bucket_scatter_kernel<T><<<grid_dim, block_dim, 0, stream>>>(
      keys_in, static_cast<const T *>(values), cursor_in_out,
      static_cast<T *>(output), num_items, num_bins);
}

int scalar_words(CudaBucketBuilderValueType value_type) {
  switch (value_type) {
    case CudaBucketBuilderValueType::i32:
    case CudaBucketBuilderValueType::f32:
    case CudaBucketBuilderValueType::u32:
      return 1;
    case CudaBucketBuilderValueType::u64:
    case CudaBucketBuilderValueType::i64:
    case CudaBucketBuilderValueType::f64:
      return 2;
  }
  return 0;
}

int scalar_words(CubSelectValueType value_type) {
  switch (value_type) {
    case CubSelectValueType::i32:
    case CubSelectValueType::f32:
    case CubSelectValueType::u32:
      return 1;
    case CubSelectValueType::u64:
    case CubSelectValueType::i64:
    case CubSelectValueType::f64:
      return 2;
  }
  return 0;
}

std::size_t cub_bucket_builder_impl(void *keys,
                                    void *values,
                                    void *offsets,
                                    void *output,
                                    void *cursor,
                                    int num_items,
                                    int num_bins,
                                    CudaBucketBuilderValueType value_type,
                                    int item_words,
                                    void *stream_ptr,
                                    void *owner) {
  if (!keys || !values || !offsets || !output || !cursor) {
    throw std::runtime_error("CUDA bucket builder received a null pointer");
  }
  if (item_words <= 0) {
    throw std::runtime_error("CUDA bucket builder expects positive item_words");
  }
  std::lock_guard<std::mutex> lock(get_cache_mutex());
  CubBucketBuilderCache &cache = get_bucket_builder_cache(owner);
  ensure_device_cache(cache);

  const int32_t *keys_in = static_cast<const int32_t *>(keys);
  int32_t *offsets_in_out = static_cast<int32_t *>(offsets);
  int32_t *cursor_in_out = static_cast<int32_t *>(cursor);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const bool use_stream = stream != nullptr;

  const std::size_t offsets_bytes =
      static_cast<std::size_t>(num_bins + 1) * sizeof(int32_t);
  if (use_stream) {
    TI_CUDA_SORT_CHECK(cudaMemsetAsync(offsets_in_out, 0, offsets_bytes, stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaMemset(offsets_in_out, 0, offsets_bytes));
  }

  constexpr int kBlockDim = 256;
  if (num_items > 0) {
    const int grid_dim = (num_items + kBlockDim - 1) / kBlockDim;
    bucket_count_i32_kernel<<<grid_dim, kBlockDim, 0, stream>>>(
        keys_in, offsets_in_out, num_items, num_bins);
    TI_CUDA_SORT_CHECK(cudaGetLastError());
  }

  std::size_t temp_storage_bytes = 0;
  const int scan_items = num_bins + 1;
  if (use_stream) {
    TI_CUDA_SORT_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, temp_storage_bytes, offsets_in_out, offsets_in_out, scan_items,
        stream));
  } else {
    TI_CUDA_SORT_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, temp_storage_bytes, offsets_in_out, offsets_in_out,
        scan_items));
  }
  ensure_buffer(&cache.temp_storage, &cache.temp_storage_bytes,
                temp_storage_bytes);
  if (use_stream) {
    TI_CUDA_SORT_CHECK(cub::DeviceScan::InclusiveSum(
        cache.temp_storage, temp_storage_bytes, offsets_in_out, offsets_in_out,
        scan_items, stream));
    TI_CUDA_SORT_CHECK(cudaMemcpyAsync(cursor_in_out, offsets_in_out,
                                       static_cast<std::size_t>(num_bins) *
                                           sizeof(int32_t),
                                       cudaMemcpyDeviceToDevice, stream));
  } else {
    TI_CUDA_SORT_CHECK(cub::DeviceScan::InclusiveSum(
        cache.temp_storage, temp_storage_bytes, offsets_in_out, offsets_in_out,
        scan_items));
    TI_CUDA_SORT_CHECK(cudaMemcpy(cursor_in_out, offsets_in_out,
                                  static_cast<std::size_t>(num_bins) *
                                      sizeof(int32_t),
                                  cudaMemcpyDeviceToDevice));
  }

  if (num_items > 0) {
    const int grid_dim = (num_items + kBlockDim - 1) / kBlockDim;
    const int expected_words = scalar_words(value_type);
    if (expected_words == 0) {
      throw std::runtime_error(
          "CUDA bucket builder received an unsupported value type");
    }
    if (item_words == expected_words) {
      switch (value_type) {
        case CudaBucketBuilderValueType::i32:
          launch_bucket_scatter<int32_t>(keys_in, values, cursor_in_out, output,
                                         num_items, num_bins, grid_dim,
                                         kBlockDim, stream);
          break;
        case CudaBucketBuilderValueType::f32:
          launch_bucket_scatter<float>(keys_in, values, cursor_in_out, output,
                                       num_items, num_bins, grid_dim, kBlockDim,
                                       stream);
          break;
        case CudaBucketBuilderValueType::u32:
          launch_bucket_scatter<uint32_t>(
              keys_in, values, cursor_in_out, output, num_items, num_bins,
              grid_dim, kBlockDim, stream);
          break;
        case CudaBucketBuilderValueType::u64:
          launch_bucket_scatter<uint64_t>(
              keys_in, values, cursor_in_out, output, num_items, num_bins,
              grid_dim, kBlockDim, stream);
          break;
        case CudaBucketBuilderValueType::i64:
          launch_bucket_scatter<int64_t>(keys_in, values, cursor_in_out, output,
                                         num_items, num_bins, grid_dim,
                                         kBlockDim, stream);
          break;
        case CudaBucketBuilderValueType::f64:
          launch_bucket_scatter<double>(keys_in, values, cursor_in_out, output,
                                        num_items, num_bins, grid_dim,
                                        kBlockDim, stream);
          break;
        default:
          break;
      }
    } else {
      bucket_scatter_words_kernel<<<grid_dim, kBlockDim, 0, stream>>>(
          keys_in, static_cast<const uint32_t *>(values), cursor_in_out,
          static_cast<uint32_t *>(output), num_items, num_bins, item_words);
    }
    TI_CUDA_SORT_CHECK(cudaGetLastError());
  }

  if (use_stream) {
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }
  return cache.allocated_bytes();
}

std::size_t cub_bucket_builder_i32_impl(void *keys,
                                        void *values,
                                        void *offsets,
                                        void *output,
                                        void *cursor,
                                        int num_items,
                                        int num_bins,
                                        void *stream_ptr,
                                        void *owner) {
  return cub_bucket_builder_impl(keys, values, offsets, output, cursor,
                                 num_items, num_bins,
                                 CudaBucketBuilderValueType::i32, 1,
                                 stream_ptr, owner);
}

void cub_bucket_builder_clear_cache_impl(void *owner) {
  static int fallback_owner = 0;
  if (!owner) {
    owner = &fallback_owner;
  }
  std::lock_guard<std::mutex> lock(get_cache_mutex());
  auto &caches = get_bucket_builder_caches();
  auto it = caches.find(owner);
  if (it != caches.end()) {
    caches.erase(it);
  }
}

std::size_t cub_bucket_builder_cached_bytes_impl(void *owner) {
  static int fallback_owner = 0;
  if (!owner) {
    owner = &fallback_owner;
  }
  std::lock_guard<std::mutex> lock(get_cache_mutex());
  auto &caches = get_bucket_builder_caches();
  auto it = caches.find(owner);
  if (it == caches.end()) {
    return 0;
  }
  return it->second->allocated_bytes();
}

template <typename T>
void grouped_reduce_segmented_sum_typed(T *scratch_values,
                                        T *output_out,
                                        int32_t *offsets_in,
                                        int num_groups,
                                        cudaStream_t stream,
                                        CubGroupedReduceCache &cache) {
  const bool use_stream = stream != nullptr;
  const std::size_t output_bytes =
      static_cast<std::size_t>(num_groups) * sizeof(T);
  if (use_stream) {
    TI_CUDA_SORT_CHECK(cudaMemsetAsync(output_out, 0, output_bytes, stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaMemset(output_out, 0, output_bytes));
  }

  std::size_t temp_storage_bytes = 0;
  if (use_stream) {
    TI_CUDA_SORT_CHECK(cub::DeviceSegmentedReduce::Sum(
        nullptr, temp_storage_bytes, scratch_values, output_out, num_groups,
        offsets_in, offsets_in + 1, stream));
  } else {
    TI_CUDA_SORT_CHECK(cub::DeviceSegmentedReduce::Sum(
        nullptr, temp_storage_bytes, scratch_values, output_out, num_groups,
        offsets_in, offsets_in + 1));
  }
  ensure_buffer(&cache.temp_storage, &cache.temp_storage_bytes,
                temp_storage_bytes);
  if (use_stream) {
    TI_CUDA_SORT_CHECK(cub::DeviceSegmentedReduce::Sum(
        cache.temp_storage, temp_storage_bytes, scratch_values, output_out,
        num_groups, offsets_in, offsets_in + 1, stream));
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cub::DeviceSegmentedReduce::Sum(
        cache.temp_storage, temp_storage_bytes, scratch_values, output_out,
        num_groups, offsets_in, offsets_in + 1));
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }
}

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
                                    void *stream_ptr,
                                    void *owner) {
  if (!keys || !values || !output || !offsets || !scratch || !cursor) {
    throw std::runtime_error("CUDA grouped reduce received a null pointer");
  }
  if (op != 0) {
    throw std::runtime_error("CUDA grouped reduce currently supports only sum");
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  std::size_t bucket_bytes = cub_bucket_builder_impl(
      keys, values, offsets, scratch, cursor, num_items, num_groups,
      static_cast<CudaBucketBuilderValueType>(
          static_cast<int>(value_type)),
      scalar_words(static_cast<CudaBucketBuilderValueType>(
          static_cast<int>(value_type))),
      stream, owner);

  auto *offsets_in = static_cast<int32_t *>(offsets);
  std::lock_guard<std::mutex> lock(get_cache_mutex());
  CubGroupedReduceCache &cache = get_grouped_reduce_cache(owner);
  ensure_device_cache(cache);
  switch (value_type) {
    case CudaGroupedReduceValueType::i32:
      grouped_reduce_segmented_sum_typed(
          static_cast<int32_t *>(scratch), static_cast<int32_t *>(output),
          offsets_in, num_groups, stream, cache);
      break;
    case CudaGroupedReduceValueType::f32:
      grouped_reduce_segmented_sum_typed(static_cast<float *>(scratch),
                                         static_cast<float *>(output),
                                         offsets_in, num_groups, stream, cache);
      break;
    case CudaGroupedReduceValueType::u32:
      grouped_reduce_segmented_sum_typed(
          static_cast<uint32_t *>(scratch), static_cast<uint32_t *>(output),
          offsets_in, num_groups, stream, cache);
      break;
    case CudaGroupedReduceValueType::u64:
      grouped_reduce_segmented_sum_typed(
          static_cast<uint64_t *>(scratch), static_cast<uint64_t *>(output),
          offsets_in, num_groups, stream, cache);
      break;
    case CudaGroupedReduceValueType::i64:
      grouped_reduce_segmented_sum_typed(
          static_cast<int64_t *>(scratch), static_cast<int64_t *>(output),
          offsets_in, num_groups, stream, cache);
      break;
    case CudaGroupedReduceValueType::f64:
      grouped_reduce_segmented_sum_typed(static_cast<double *>(scratch),
                                         static_cast<double *>(output),
                                         offsets_in, num_groups, stream, cache);
      break;
    default:
      throw std::runtime_error(
          "CUDA grouped reduce received an unsupported value type");
  }
  return bucket_bytes + cache.allocated_bytes();
}

std::size_t cub_grouped_reduce_i32_impl(void *keys,
                                        void *values,
                                        void *output,
                                        void *offsets,
                                        void *scratch,
                                        void *cursor,
                                        int num_items,
                                        int num_groups,
                                        int op,
                                        void *stream_ptr,
                                        void *owner) {
  return cub_grouped_reduce_impl(keys, values, output, offsets, scratch, cursor,
                                 num_items, num_groups,
                                 CudaGroupedReduceValueType::i32, op,
                                 stream_ptr, owner);
}

template <typename T>
void grouped_reduce_atomic_sum_launch(const int32_t *keys_in,
                                      const T *values_in,
                                      T *output_out,
                                      int num_items,
                                      int num_groups,
                                      cudaStream_t stream) {
  const std::size_t output_bytes =
      static_cast<std::size_t>(num_groups) * sizeof(T);
  if (stream) {
    TI_CUDA_SORT_CHECK(cudaMemsetAsync(output_out, 0, output_bytes, stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaMemset(output_out, 0, output_bytes));
  }
  if (num_items > 0) {
    constexpr int kBlockDim = 256;
    const int grid_dim = (num_items + kBlockDim - 1) / kBlockDim;
    grouped_reduce_atomic_sum_kernel<T><<<grid_dim, kBlockDim, 0, stream>>>(
        keys_in, values_in, output_out, num_items, num_groups);
    TI_CUDA_SORT_CHECK(cudaGetLastError());
  }
}

template <typename T>
void grouped_reduce_atomic_sum_strided_launch(const int32_t *keys_in,
                                              const uint8_t *values_in,
                                              T *output_out,
                                              int num_items,
                                              int num_groups,
                                              std::size_t offset,
                                              std::size_t stride,
                                              cudaStream_t stream) {
  const std::size_t output_bytes =
      static_cast<std::size_t>(num_groups) * sizeof(T);
  if (stream) {
    TI_CUDA_SORT_CHECK(cudaMemsetAsync(output_out, 0, output_bytes, stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaMemset(output_out, 0, output_bytes));
  }
  if (num_items > 0) {
    constexpr int kBlockDim = 256;
    const int grid_dim = (num_items + kBlockDim - 1) / kBlockDim;
    grouped_reduce_atomic_sum_strided_kernel<T>
        <<<grid_dim, kBlockDim, 0, stream>>>(
            keys_in, values_in, output_out, num_items, num_groups, offset,
            stride);
    TI_CUDA_SORT_CHECK(cudaGetLastError());
  }
}

template <typename T>
void grouped_reduce_atomic_sum_strided_io_launch(const uint8_t *keys_in,
                                                 const uint8_t *values_in,
                                                 uint8_t *output_out,
                                                 int num_items,
                                                 int num_groups,
                                                 std::size_t keys_offset,
                                                 std::size_t keys_stride,
                                                 std::size_t values_offset,
                                                 std::size_t values_stride,
                                                 std::size_t output_offset,
                                                 std::size_t output_stride,
                                                 cudaStream_t stream) {
  constexpr int kBlockDim = 256;
  if (num_groups > 0) {
    if (output_offset == 0 && output_stride == sizeof(T)) {
      const std::size_t output_bytes =
          static_cast<std::size_t>(num_groups) * sizeof(T);
      if (stream) {
        TI_CUDA_SORT_CHECK(
            cudaMemsetAsync(output_out, 0, output_bytes, stream));
      } else {
        TI_CUDA_SORT_CHECK(cudaMemset(output_out, 0, output_bytes));
      }
    } else {
      const int zero_grid = (num_groups + kBlockDim - 1) / kBlockDim;
      zero_strided_kernel<T><<<zero_grid, kBlockDim, 0, stream>>>(
          output_out, num_groups, output_offset, output_stride);
      TI_CUDA_SORT_CHECK(cudaGetLastError());
    }
  }
  if (num_items > 0) {
    const int grid_dim = (num_items + kBlockDim - 1) / kBlockDim;
    grouped_reduce_atomic_sum_strided_io_kernel<T>
        <<<grid_dim, kBlockDim, 0, stream>>>(
            keys_in, values_in, output_out, num_items, num_groups,
            keys_offset, keys_stride, values_offset, values_stride,
            output_offset, output_stride);
    TI_CUDA_SORT_CHECK(cudaGetLastError());
  }
}

std::size_t cub_grouped_reduce_atomic_impl(
    void *keys,
    void *values,
    void *output,
    int num_items,
    int num_groups,
    CudaGroupedReduceValueType value_type,
    int op,
    void *stream_ptr) {
  if (!keys || !values || !output) {
    throw std::runtime_error("CUDA grouped reduce received a null pointer");
  }
  if (op != 0) {
    throw std::runtime_error("CUDA grouped reduce currently supports only sum");
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const bool use_stream = stream != nullptr;
  auto *keys_in = static_cast<const int32_t *>(keys);
  switch (value_type) {
    case CudaGroupedReduceValueType::i32:
      grouped_reduce_atomic_sum_launch(
          keys_in, static_cast<const int32_t *>(values),
          static_cast<int32_t *>(output), num_items, num_groups, stream);
      break;
    case CudaGroupedReduceValueType::f32:
      grouped_reduce_atomic_sum_launch(
          keys_in, static_cast<const float *>(values),
          static_cast<float *>(output), num_items, num_groups, stream);
      break;
    case CudaGroupedReduceValueType::u32:
      grouped_reduce_atomic_sum_launch(
          keys_in, static_cast<const uint32_t *>(values),
          static_cast<uint32_t *>(output), num_items, num_groups, stream);
      break;
    case CudaGroupedReduceValueType::u64:
      grouped_reduce_atomic_sum_launch(
          keys_in, static_cast<const uint64_t *>(values),
          static_cast<uint64_t *>(output), num_items, num_groups, stream);
      break;
    case CudaGroupedReduceValueType::i64:
      grouped_reduce_atomic_sum_launch(
          keys_in, static_cast<const int64_t *>(values),
          static_cast<int64_t *>(output), num_items, num_groups, stream);
      break;
    case CudaGroupedReduceValueType::f64:
      grouped_reduce_atomic_sum_launch(
          keys_in, static_cast<const double *>(values),
          static_cast<double *>(output), num_items, num_groups, stream);
      break;
    default:
      throw std::runtime_error("Unsupported CUDA grouped reduce value type");
  }
  if (use_stream) {
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }
  return 0;
}

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
    void *stream_ptr) {
  if (!keys || !values || !output) {
    throw std::runtime_error(
        "CUDA strided grouped reduce received a null pointer");
  }
  if (op != 0) {
    throw std::runtime_error(
        "CUDA strided grouped reduce currently supports only sum");
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const bool use_stream = stream != nullptr;
  auto *keys_in = static_cast<const int32_t *>(keys);
  auto *values_in = static_cast<const uint8_t *>(values);
  switch (value_type) {
    case CudaGroupedReduceValueType::i32:
      grouped_reduce_atomic_sum_strided_launch(
          keys_in, values_in, static_cast<int32_t *>(output), num_items,
          num_groups, offset, stride, stream);
      break;
    case CudaGroupedReduceValueType::f32:
      grouped_reduce_atomic_sum_strided_launch(
          keys_in, values_in, static_cast<float *>(output), num_items,
          num_groups, offset, stride, stream);
      break;
    case CudaGroupedReduceValueType::u32:
      grouped_reduce_atomic_sum_strided_launch(
          keys_in, values_in, static_cast<uint32_t *>(output), num_items,
          num_groups, offset, stride, stream);
      break;
    case CudaGroupedReduceValueType::u64:
      grouped_reduce_atomic_sum_strided_launch(
          keys_in, values_in, static_cast<uint64_t *>(output), num_items,
          num_groups, offset, stride, stream);
      break;
    case CudaGroupedReduceValueType::i64:
      grouped_reduce_atomic_sum_strided_launch(
          keys_in, values_in, static_cast<int64_t *>(output), num_items,
          num_groups, offset, stride, stream);
      break;
    case CudaGroupedReduceValueType::f64:
      grouped_reduce_atomic_sum_strided_launch(
          keys_in, values_in, static_cast<double *>(output), num_items,
          num_groups, offset, stride, stream);
      break;
    default:
      throw std::runtime_error(
          "Unsupported CUDA strided grouped reduce value type");
  }
  if (use_stream) {
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }
  return 0;
}

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
    void *stream_ptr) {
  if (!keys || !values || !output) {
    throw std::runtime_error(
        "CUDA strided grouped reduce received a null pointer");
  }
  if (op != 0) {
    throw std::runtime_error(
        "CUDA strided grouped reduce currently supports only sum");
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const bool use_stream = stream != nullptr;
  auto *keys_in = static_cast<const uint8_t *>(keys);
  auto *values_in = static_cast<const uint8_t *>(values);
  auto *output_out = static_cast<uint8_t *>(output);
  switch (value_type) {
    case CudaGroupedReduceValueType::i32:
      grouped_reduce_atomic_sum_strided_io_launch<int32_t>(
          keys_in, values_in, output_out, num_items, num_groups, keys_offset,
          keys_stride, values_offset, values_stride, output_offset,
          output_stride, stream);
      break;
    case CudaGroupedReduceValueType::f32:
      grouped_reduce_atomic_sum_strided_io_launch<float>(
          keys_in, values_in, output_out, num_items, num_groups, keys_offset,
          keys_stride, values_offset, values_stride, output_offset,
          output_stride, stream);
      break;
    case CudaGroupedReduceValueType::u32:
      grouped_reduce_atomic_sum_strided_io_launch<uint32_t>(
          keys_in, values_in, output_out, num_items, num_groups, keys_offset,
          keys_stride, values_offset, values_stride, output_offset,
          output_stride, stream);
      break;
    case CudaGroupedReduceValueType::u64:
      grouped_reduce_atomic_sum_strided_io_launch<uint64_t>(
          keys_in, values_in, output_out, num_items, num_groups, keys_offset,
          keys_stride, values_offset, values_stride, output_offset,
          output_stride, stream);
      break;
    case CudaGroupedReduceValueType::i64:
      grouped_reduce_atomic_sum_strided_io_launch<int64_t>(
          keys_in, values_in, output_out, num_items, num_groups, keys_offset,
          keys_stride, values_offset, values_stride, output_offset,
          output_stride, stream);
      break;
    case CudaGroupedReduceValueType::f64:
      grouped_reduce_atomic_sum_strided_io_launch<double>(
          keys_in, values_in, output_out, num_items, num_groups, keys_offset,
          keys_stride, values_offset, values_stride, output_offset,
          output_stride, stream);
      break;
    default:
      throw std::runtime_error(
          "Unsupported CUDA strided grouped reduce value type");
  }
  if (use_stream) {
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }
  return 0;
}

void cub_grouped_reduce_clear_cache_impl(void *owner) {
  static int fallback_owner = 0;
  if (!owner) {
    owner = &fallback_owner;
  }
  std::lock_guard<std::mutex> lock(get_cache_mutex());
  auto &caches = get_grouped_reduce_caches();
  auto it = caches.find(owner);
  if (it != caches.end()) {
    caches.erase(it);
  }
}

std::size_t cub_grouped_reduce_cached_bytes_impl(void *owner) {
  static int fallback_owner = 0;
  if (!owner) {
    owner = &fallback_owner;
  }
  std::lock_guard<std::mutex> lock(get_cache_mutex());
  auto &caches = get_grouped_reduce_caches();
  auto it = caches.find(owner);
  if (it == caches.end()) {
    return 0;
  }
  return it->second->allocated_bytes();
}

}  // namespace taichi::lang::cuda
