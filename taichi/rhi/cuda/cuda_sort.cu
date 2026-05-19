#include "taichi/rhi/cuda/cuda_sort.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cstdint>
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
  std::size_t temp_storage_bytes{0};
  int device_id{-1};

  ~CubSelectCache() {
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

template <typename KeyT>
__global__ void scatter_by_index_kernel(const KeyT *keys,
                                        const int32_t *values,
                                        const uint32_t *indices,
                                        KeyT *keys_out,
                                        int32_t *values_out,
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

void check_last_cuda_error(const char *) {
  TI_CUDA_SORT_CHECK(cudaGetLastError());
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

template <typename KeyT>
void launch_scatter_by_index(const KeyT *keys,
                             const int32_t *values,
                             const uint32_t *indices,
                             KeyT *keys_out,
                             int32_t *values_out,
                             int num_items,
                             bool has_values,
                             cudaStream_t stream) {
  constexpr int kBlockSize = 256;
  const int grid = (num_items + kBlockSize - 1) / kBlockSize;
  scatter_by_index_kernel<KeyT><<<grid, kBlockSize, 0, stream>>>(
      keys, values, indices, keys_out, values_out, num_items, has_values);
  check_last_cuda_error("scatter_by_index_kernel");
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

template <typename KeyT>
std::size_t sort_typed(CubSortCache &cache,
                       void *keys,
                       void *values,
                       int num_items,
                       bool has_values,
                       void *stream_ptr) {
  KeyT *keys_in = static_cast<KeyT *>(keys);
  int32_t *values_in = static_cast<int32_t *>(values);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const bool use_stream = stream != nullptr;
  ensure_device_cache(cache);

  ensure_buffer(&cache.keys_out, &cache.keys_out_bytes,
                sizeof(KeyT) * num_items);
  KeyT *keys_out = static_cast<KeyT *>(cache.keys_out);
  int32_t *values_out = nullptr;
  if (has_values) {
    ensure_buffer(&cache.values_out, &cache.values_out_bytes,
                  sizeof(int32_t) * num_items);
    values_out = static_cast<int32_t *>(cache.values_out);
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
                                         sizeof(int32_t) * num_items,
                                         cudaMemcpyDeviceToDevice, stream));
    } else {
      TI_CUDA_SORT_CHECK(cub::DeviceRadixSort::SortPairs(
          cache.temp_storage, temp_storage_bytes, keys_in, keys_out, values_in,
          values_out, num_items));
      TI_CUDA_SORT_CHECK(cudaMemcpy(values_in, values_out,
                                    sizeof(int32_t) * num_items,
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

std::size_t sort_f32_transformed(CubSortCache &cache,
                                 void *keys,
                                 void *values,
                                 int num_items,
                                 bool has_values,
                                 CubSortNanPolicy nan_policy,
                                 void *stream_ptr) {
  float *keys_in = static_cast<float *>(keys);
  int32_t *values_in = static_cast<int32_t *>(values);
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
  int32_t *values_out = nullptr;
  if (has_values) {
    values_out = ensure_typed_buffer<int32_t>(
        &cache.values_out, &cache.values_out_bytes, num_items);
  } else if (cache.values_out) {
    TI_CUDA_SORT_CHECK(cudaFree(cache.values_out));
    cache.values_out = nullptr;
    cache.values_out_bytes = 0;
  }

  launch_init_sortable_f32(keys_in, key_a, index_a, num_items,
                           static_cast<int>(nan_policy), stream);
  cub_sort_pairs(cache, key_a, key_b, index_a, index_b, num_items, stream);
  launch_scatter_by_index(keys_in, values_in, index_b, keys_out, values_out,
                          num_items, has_values, stream);

  if (use_stream) {
    TI_CUDA_SORT_CHECK(cudaMemcpyAsync(keys_in, keys_out,
                                       sizeof(float) * num_items,
                                       cudaMemcpyDeviceToDevice, stream));
    if (has_values) {
      TI_CUDA_SORT_CHECK(cudaMemcpyAsync(values_in, values_out,
                                         sizeof(int32_t) * num_items,
                                         cudaMemcpyDeviceToDevice, stream));
    }
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaMemcpy(keys_in, keys_out, sizeof(float) * num_items,
                                  cudaMemcpyDeviceToDevice));
    if (has_values) {
      TI_CUDA_SORT_CHECK(cudaMemcpy(values_in, values_out,
                                    sizeof(int32_t) * num_items,
                                    cudaMemcpyDeviceToDevice));
    }
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }

  return cache.allocated_bytes();
}

template <typename KeyT>
std::size_t sort_u64_transformed(CubSortCache &cache,
                                 void *keys,
                                 void *values,
                                 int num_items,
                                 bool has_values,
                                 CubSortNanPolicy nan_policy,
                                 void *stream_ptr) {
  KeyT *keys_in = static_cast<KeyT *>(keys);
  int32_t *values_in = static_cast<int32_t *>(values);
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
  int32_t *values_out = nullptr;
  if (has_values) {
    values_out = ensure_typed_buffer<int32_t>(
        &cache.values_out, &cache.values_out_bytes, num_items);
  } else if (cache.values_out) {
    TI_CUDA_SORT_CHECK(cudaFree(cache.values_out));
    cache.values_out = nullptr;
    cache.values_out_bytes = 0;
  }

  launch_init_sortable64(keys_in, key_a, index_a, num_items,
                         static_cast<int>(nan_policy), stream);
  cub_sort_pairs(cache, key_a, key_b, index_a, index_b, num_items, stream);
  launch_scatter_by_index(keys_in, values_in, index_b, keys_out, values_out,
                          num_items, has_values, stream);

  if (use_stream) {
    TI_CUDA_SORT_CHECK(cudaMemcpyAsync(keys_in, keys_out,
                                       sizeof(KeyT) * num_items,
                                       cudaMemcpyDeviceToDevice, stream));
    if (has_values) {
      TI_CUDA_SORT_CHECK(cudaMemcpyAsync(values_in, values_out,
                                         sizeof(int32_t) * num_items,
                                         cudaMemcpyDeviceToDevice, stream));
    }
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaMemcpy(keys_in, keys_out, sizeof(KeyT) * num_items,
                                  cudaMemcpyDeviceToDevice));
    if (has_values) {
      TI_CUDA_SORT_CHECK(cudaMemcpy(values_in, values_out,
                                    sizeof(int32_t) * num_items,
                                    cudaMemcpyDeviceToDevice));
    }
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }

  return cache.allocated_bytes();
}

template <typename KeyT>
std::size_t sort_split32(CubSortCache &cache,
                         void *keys,
                         void *values,
                         int num_items,
                         bool has_values,
                         CubSortNanPolicy nan_policy,
                         void *stream_ptr) {
  KeyT *keys_in = static_cast<KeyT *>(keys);
  int32_t *values_in = static_cast<int32_t *>(values);
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
  int32_t *values_out = nullptr;
  if (has_values) {
    values_out = ensure_typed_buffer<int32_t>(
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
  launch_scatter_by_index(keys_in, values_in, index_a, keys_out, values_out,
                          num_items, has_values, stream);

  if (use_stream) {
    TI_CUDA_SORT_CHECK(cudaMemcpyAsync(keys_in, keys_out,
                                       sizeof(KeyT) * num_items,
                                       cudaMemcpyDeviceToDevice, stream));
    if (has_values) {
      TI_CUDA_SORT_CHECK(cudaMemcpyAsync(values_in, values_out,
                                         sizeof(int32_t) * num_items,
                                         cudaMemcpyDeviceToDevice, stream));
    }
    TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
  } else {
    TI_CUDA_SORT_CHECK(cudaMemcpy(keys_in, keys_out, sizeof(KeyT) * num_items,
                                  cudaMemcpyDeviceToDevice));
    if (has_values) {
      TI_CUDA_SORT_CHECK(cudaMemcpy(values_in, values_out,
                                    sizeof(int32_t) * num_items,
                                    cudaMemcpyDeviceToDevice));
    }
    TI_CUDA_SORT_CHECK(cudaDeviceSynchronize());
  }

  return cache.allocated_bytes();
}

}  // namespace

std::size_t cub_radix_sort_impl(void *keys,
                                void *values,
                                int num_items,
                                CubSortKeyType key_type,
                                CubSortMode mode,
                                CubSortNanPolicy nan_policy,
                                bool has_values,
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
  if (mode == CubSortMode::split32) {
    switch (key_type) {
      case CubSortKeyType::u64:
        return sort_split32<uint64_t>(cache, keys, values, num_items,
                                      has_values, nan_policy, stream);
      case CubSortKeyType::i64:
        return sort_split32<int64_t>(cache, keys, values, num_items, has_values,
                                     nan_policy, stream);
      case CubSortKeyType::f64:
        return sort_split32<double>(cache, keys, values, num_items, has_values,
                                    nan_policy, stream);
      default:
        throw std::runtime_error(
            "CUDA CUB split32 sort supports only u64/i64/f64 keys");
    }
  }

  switch (key_type) {
    case CubSortKeyType::u32:
      return sort_typed<uint32_t>(cache, keys, values, num_items, has_values,
                                  stream);
    case CubSortKeyType::i32:
      return sort_typed<int32_t>(cache, keys, values, num_items, has_values,
                                 stream);
    case CubSortKeyType::f32:
      return sort_f32_transformed(cache, keys, values, num_items, has_values,
                                  nan_policy, stream);
    case CubSortKeyType::u64:
      return sort_typed<uint64_t>(cache, keys, values, num_items, has_values,
                                  stream);
    case CubSortKeyType::i64:
      return sort_typed<int64_t>(cache, keys, values, num_items, has_values,
                                 stream);
    case CubSortKeyType::f64:
      return sort_u64_transformed<double>(cache, keys, values, num_items,
                                          has_values, nan_policy, stream);
  }
  throw std::runtime_error("Unsupported CUB sort key type");
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
  }
  throw std::runtime_error("Unsupported CUB scan value type");
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

std::size_t cub_select_flagged_impl(void *values,
                                    void *flags,
                                    void *output,
                                    void *count,
                                    int num_items,
                                    CubSelectValueType value_type,
                                    void *stream,
                                    void *owner) {
  std::lock_guard<std::mutex> lock(get_cache_mutex());
  CubSelectCache &cache = get_select_cache(owner);
  switch (value_type) {
    case CubSelectValueType::i32:
      return select_flagged_typed<int32_t>(cache, values, flags, output, count,
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

template <typename T>
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
  int32_t *hist_out = static_cast<int32_t *>(bins);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const bool use_stream = stream != nullptr;
  ensure_device_cache(cache);

  if (num_items == 0) {
    if (use_stream) {
      TI_CUDA_SORT_CHECK(
          cudaMemsetAsync(hist_out, 0, sizeof(int32_t) * num_bins, stream));
      TI_CUDA_SORT_CHECK(cudaStreamSynchronize(stream));
    } else {
      TI_CUDA_SORT_CHECK(cudaMemset(hist_out, 0, sizeof(int32_t) * num_bins));
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

std::size_t cub_histogram_even_impl(void *values,
                                    void *bins,
                                    int num_items,
                                    int num_bins,
                                    CubHistogramValueType value_type,
                                    void *stream,
                                    void *owner) {
  std::lock_guard<std::mutex> lock(get_cache_mutex());
  CubHistogramCache &cache = get_histogram_cache(owner);
  switch (value_type) {
    case CubHistogramValueType::i32:
      return histogram_even_typed<int32_t>(cache, values, bins, num_items,
                                           num_bins, stream);
  }
  throw std::runtime_error("Unsupported CUB histogram value type");
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
  }
  throw std::runtime_error("Unsupported CUB reduce value type");
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

}  // namespace taichi::lang::cuda
