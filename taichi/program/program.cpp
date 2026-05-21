// Program, context for Taichi program execution

#include "program.h"

#include "taichi/ir/statements.h"
#include "taichi/program/extension.h"
#include "taichi/codegen/cpu/codegen_cpu.h"
#include "taichi/struct/struct.h"
#include "taichi/runtime/program_impls/opengl/opengl_program.h"
#include "taichi/runtime/program_impls/metal/metal_program.h"
#include "taichi/platform/cuda/detect_cuda.h"
#include "taichi/system/timeline.h"
#include "taichi/system/threading.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/program/snode_expr_utils.h"
#include "taichi/math/arithmetic.h"
#include "taichi/rhi/common/host_memory_pool.h"
#include "taichi/program/parallel_executor.h"

#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/cuda/cuda_sort.h"
#endif

#ifdef TI_WITH_LLVM
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/codegen/llvm/struct_llvm.h"
#endif

#ifdef TI_WITH_VULKAN
#include "taichi/runtime/program_impls/vulkan/vulkan_program.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#endif
#ifdef TI_WITH_OPENGL
#include "taichi/runtime/program_impls/opengl/opengl_program.h"
#include "taichi/rhi/opengl/opengl_api.h"
#endif
#ifdef TI_WITH_DX11
#include "taichi/runtime/program_impls/dx/dx_program.h"
#include "taichi/rhi/dx/dx_api.h"
#endif
#ifdef TI_WITH_DX12
#include "taichi/runtime/program_impls/dx12/dx12_program.h"
#include "taichi/rhi/dx12/dx12_api.h"
#endif
#ifdef TI_WITH_METAL
#include "taichi/runtime/program_impls/metal/metal_program.h"
#include "taichi/rhi/metal/metal_api.h"
#endif  // TI_WITH_METAL

#if defined(_M_X64) || defined(__x86_64)
// For _MM_SET_FLUSH_ZERO_MODE
#include <xmmintrin.h>
#endif  // defined(_M_X64) || defined(__x86_64)

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <type_traits>
#include <vector>

namespace taichi::lang {
std::atomic<int> Program::num_instances_;

namespace {
std::atomic<std::size_t> cpu_scatter_add_workspace_bytes_peak{0};
std::atomic<std::size_t> cpu_grouped_reduce_workspace_bytes_peak{0};

void update_cpu_scatter_add_workspace_peak(std::size_t bytes) {
  auto current = cpu_scatter_add_workspace_bytes_peak.load(std::memory_order_relaxed);
  while (current < bytes &&
         !cpu_scatter_add_workspace_bytes_peak.compare_exchange_weak(
             current, bytes, std::memory_order_relaxed)) {
  }
}

void update_cpu_grouped_reduce_workspace_peak(std::size_t bytes) {
  auto current =
      cpu_grouped_reduce_workspace_bytes_peak.load(std::memory_order_relaxed);
  while (current < bytes &&
         !cpu_grouped_reduce_workspace_bytes_peak.compare_exchange_weak(
             current, bytes, std::memory_order_relaxed)) {
  }
}

uint32_t cpu_sortable_f32_key(float value, int nan_policy) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  constexpr uint32_t kSign = 0x80000000u;
  constexpr uint32_t kAbsMask = 0x7fffffffu;
  constexpr uint32_t kInfBits = 0x7f800000u;
  if (nan_policy == 0 && (bits & kAbsMask) > kInfBits) {
    return 0xffffffffu;
  }
  if (nan_policy == 0 && (bits & kAbsMask) == 0) {
    return kSign;
  }
  return (bits & kSign) ? ~bits : (bits ^ kSign);
}

uint64_t cpu_sortable_f64_key(double value, int nan_policy) {
  uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  constexpr uint64_t kSign = 0x8000000000000000ull;
  constexpr uint64_t kAbsMask = 0x7fffffffffffffffull;
  constexpr uint64_t kInfBits = 0x7ff0000000000000ull;
  if (nan_policy == 0 && (bits & kAbsMask) > kInfBits) {
    return 0xffffffffffffffffull;
  }
  if (nan_policy == 0 && (bits & kAbsMask) == 0) {
    return kSign;
  }
  return (bits & kSign) ? ~bits : (bits ^ kSign);
}

template <typename KeyT>
bool cpu_sort_key_before(KeyT lhs,
                         KeyT rhs,
                         bool descending,
                         int /*nan_policy*/) {
  if (lhs == rhs) {
    return false;
  }
  return descending ? lhs > rhs : lhs < rhs;
}

template <>
bool cpu_sort_key_before<float>(float lhs,
                                float rhs,
                                bool descending,
                                int nan_policy) {
  uint32_t lhs_bits = 0;
  uint32_t rhs_bits = 0;
  std::memcpy(&lhs_bits, &lhs, sizeof(lhs_bits));
  std::memcpy(&rhs_bits, &rhs, sizeof(rhs_bits));
  constexpr uint32_t kAbsMask = 0x7fffffffu;
  constexpr uint32_t kInfBits = 0x7f800000u;
  const bool lhs_nan = (lhs_bits & kAbsMask) > kInfBits;
  const bool rhs_nan = (rhs_bits & kAbsMask) > kInfBits;
  if (nan_policy == 0 && (lhs_nan || rhs_nan)) {
    return !lhs_nan && rhs_nan;
  }
  if (nan_policy == 0 && (lhs_bits & kAbsMask) == 0 &&
      (rhs_bits & kAbsMask) == 0) {
    return false;
  }
  const uint32_t lhs_key = cpu_sortable_f32_key(lhs, nan_policy);
  const uint32_t rhs_key = cpu_sortable_f32_key(rhs, nan_policy);
  if (lhs_key == rhs_key) {
    return false;
  }
  return descending ? lhs_key > rhs_key : lhs_key < rhs_key;
}

template <>
bool cpu_sort_key_before<double>(double lhs,
                                 double rhs,
                                 bool descending,
                                 int nan_policy) {
  uint64_t lhs_bits = 0;
  uint64_t rhs_bits = 0;
  std::memcpy(&lhs_bits, &lhs, sizeof(lhs_bits));
  std::memcpy(&rhs_bits, &rhs, sizeof(rhs_bits));
  constexpr uint64_t kAbsMask = 0x7fffffffffffffffull;
  constexpr uint64_t kInfBits = 0x7ff0000000000000ull;
  const bool lhs_nan = (lhs_bits & kAbsMask) > kInfBits;
  const bool rhs_nan = (rhs_bits & kAbsMask) > kInfBits;
  if (nan_policy == 0 && (lhs_nan || rhs_nan)) {
    return !lhs_nan && rhs_nan;
  }
  if (nan_policy == 0 && (lhs_bits & kAbsMask) == 0 &&
      (rhs_bits & kAbsMask) == 0) {
    return false;
  }
  const uint64_t lhs_key = cpu_sortable_f64_key(lhs, nan_policy);
  const uint64_t rhs_key = cpu_sortable_f64_key(rhs, nan_policy);
  if (lhs_key == rhs_key) {
    return false;
  }
  return descending ? lhs_key > rhs_key : lhs_key < rhs_key;
}

template <typename KeyT, typename ValueT>
std::size_t cpu_stable_sort_impl(KeyT *keys,
                                 ValueT *values,
                                 std::size_t n,
                                 bool descending,
                                 int nan_policy) {
  if (n <= 1) {
    return 0;
  }
  if (values) {
    struct Item {
      KeyT key;
      ValueT value;
    };
    std::vector<Item> items(n);
    for (std::size_t i = 0; i < n; ++i) {
      items[i] = {keys[i], values[i]};
    }
    std::stable_sort(items.begin(), items.end(), [&](const Item &lhs,
                                                     const Item &rhs) {
      return cpu_sort_key_before<KeyT>(
          lhs.key, rhs.key, descending, nan_policy);
    });
    for (std::size_t i = 0; i < n; ++i) {
      keys[i] = items[i].key;
      values[i] = items[i].value;
    }
    return items.size() * sizeof(Item);
  }

  std::vector<KeyT> sorted_keys(keys, keys + n);
  std::stable_sort(sorted_keys.begin(), sorted_keys.end(), [&](KeyT lhs,
                                                               KeyT rhs) {
    return cpu_sort_key_before<KeyT>(lhs, rhs, descending, nan_policy);
  });
  std::memcpy(keys, sorted_keys.data(), n * sizeof(KeyT));
  return sorted_keys.size() * sizeof(KeyT);
}

template <typename KeyT>
std::size_t cpu_stable_sort_value_dispatch(KeyT *keys,
                                           void *values,
                                           std::size_t n,
                                           int value_type,
                                           bool descending,
                                           int nan_policy) {
  if (!values) {
    return cpu_stable_sort_impl<KeyT, int32_t>(
        keys, nullptr, n, descending, nan_policy);
  }
  switch (value_type) {
    case 0:
      return cpu_stable_sort_impl<KeyT, int32_t>(
          keys, reinterpret_cast<int32_t *>(values), n, descending,
          nan_policy);
    case 1:
      return cpu_stable_sort_impl<KeyT, float>(
          keys, reinterpret_cast<float *>(values), n, descending, nan_policy);
    case 2:
      return cpu_stable_sort_impl<KeyT, uint32_t>(
          keys, reinterpret_cast<uint32_t *>(values), n, descending,
          nan_policy);
    case 3:
      return cpu_stable_sort_impl<KeyT, uint64_t>(
          keys, reinterpret_cast<uint64_t *>(values), n, descending,
          nan_policy);
    case 4:
      return cpu_stable_sort_impl<KeyT, int64_t>(
          keys, reinterpret_cast<int64_t *>(values), n, descending,
          nan_policy);
    case 5:
      return cpu_stable_sort_impl<KeyT, double>(
          keys, reinterpret_cast<double *>(values), n, descending, nan_policy);
    default:
      TI_ERROR("CPU native sort received an unsupported value type.");
  }
}

template <typename KeyT>
std::size_t cpu_stable_sort_raw_values(KeyT *keys,
                                       void *values,
                                       std::size_t n,
                                       std::size_t item_bytes,
                                       bool descending,
                                       int nan_policy) {
  if (n <= 1) {
    return 0;
  }
  auto *value_bytes = static_cast<uint8_t *>(values);
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](std::size_t lhs,
                                                   std::size_t rhs) {
    return cpu_sort_key_before<KeyT>(
        keys[lhs], keys[rhs], descending, nan_policy);
  });

  std::vector<KeyT> sorted_keys(n);
  std::vector<uint8_t> sorted_values(n * item_bytes);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t src = order[i];
    sorted_keys[i] = keys[src];
    std::memcpy(sorted_values.data() + i * item_bytes,
                value_bytes + src * item_bytes, item_bytes);
  }
  std::memcpy(keys, sorted_keys.data(), n * sizeof(KeyT));
  std::memcpy(value_bytes, sorted_values.data(), sorted_values.size());
  return order.size() * sizeof(std::size_t) + sorted_keys.size() * sizeof(KeyT) +
         sorted_values.size();
}

template <typename ValueT, typename CounterT>
struct CpuHistogramTaskContext {
  const ValueT *values{nullptr};
  CounterT *partial{nullptr};
  std::size_t n{0};
  std::size_t num_bins{0};
  int num_threads{1};
};

template <typename T>
struct CpuReduceTaskContext {
  const T *values{nullptr};
  T *partial{nullptr};
  std::size_t n{0};
  int num_threads{1};
  int op{0};
};

template <typename T>
struct CpuStridedReduceTaskContext {
  const uint8_t *values{nullptr};
  T *partial{nullptr};
  std::size_t n{0};
  std::size_t offset{0};
  std::size_t stride{0};
  int num_threads{1};
  int op{0};
};

struct CpuFillU32TaskContext {
  uint32_t *data{nullptr};
  std::size_t words{0};
  uint32_t value{0};
  int num_threads{1};
};

struct CpuCopyTaskContext {
  uint8_t *dst{nullptr};
  const uint8_t *src{nullptr};
  std::size_t bytes{0};
  int num_threads{1};
};

template <typename T>
struct CpuTransformTaskContext {
  const T *src{nullptr};
  T *dst{nullptr};
  std::size_t n{0};
  T scale{};
  T bias{};
  int num_threads{1};
};

template <typename T>
struct CpuStridedTransformTaskContext {
  const uint8_t *src{nullptr};
  T *dst{nullptr};
  std::size_t n{0};
  std::size_t offset{0};
  std::size_t stride{0};
  T scale{};
  T bias{};
  int num_threads{1};
};

template <typename T>
struct CpuStridedToStridedTransformTaskContext {
  const uint8_t *src{nullptr};
  uint8_t *dst{nullptr};
  std::size_t n{0};
  std::size_t src_offset{0};
  std::size_t src_stride{0};
  std::size_t dst_offset{0};
  std::size_t dst_stride{0};
  T scale{};
  T bias{};
  int num_threads{1};
};

struct CpuIndexedCopyTaskContext {
  const uint8_t *src{nullptr};
  const int32_t *indices{nullptr};
  uint8_t *dst{nullptr};
  std::size_t n{0};
  std::size_t index_bound{0};
  std::size_t item_bytes{0};
  bool scatter{false};
  int num_threads{1};
};

struct CpuStridedIndexedCopyTaskContext {
  const uint8_t *src{nullptr};
  const int32_t *indices{nullptr};
  uint8_t *dst{nullptr};
  std::size_t n{0};
  std::size_t index_bound{0};
  std::size_t item_bytes{0};
  std::size_t src_offset{0};
  std::size_t src_stride{0};
  std::size_t dst_offset{0};
  std::size_t dst_stride{0};
  bool scatter{false};
  int num_threads{1};
};

template <typename T>
struct CpuScatterAddTaskContext {
  const T *src{nullptr};
  const int32_t *indices{nullptr};
  T *partial{nullptr};
  T *dst{nullptr};
  std::size_t n{0};
  std::size_t dst_items{0};
  int num_threads{1};
};

template <typename T>
struct CpuStridedScatterAddTaskContext {
  const uint8_t *src{nullptr};
  const int32_t *indices{nullptr};
  T *partial{nullptr};
  T *dst{nullptr};
  std::size_t n{0};
  std::size_t dst_items{0};
  std::size_t offset{0};
  std::size_t stride{0};
  int num_threads{1};
};

template <typename T>
struct CpuStridedScatterAddIoTaskContext {
  const uint8_t *src{nullptr};
  const int32_t *indices{nullptr};
  T *partial{nullptr};
  uint8_t *dst{nullptr};
  std::size_t n{0};
  std::size_t dst_items{0};
  std::size_t src_offset{0};
  std::size_t src_stride{0};
  std::size_t dst_offset{0};
  std::size_t dst_stride{0};
  int num_threads{1};
};

struct CpuBucketCountTaskContext {
  const int32_t *keys{nullptr};
  int32_t *partial{nullptr};
  std::size_t n{0};
  std::size_t num_bins{0};
  int num_threads{1};
};

template <typename T>
struct CpuBucketScatterTaskContext {
  const int32_t *keys{nullptr};
  const T *values{nullptr};
  int32_t *thread_offsets{nullptr};
  T *output{nullptr};
  std::size_t n{0};
  std::size_t num_bins{0};
  int num_threads{1};
};

struct CpuBucketScatterRawTaskContext {
  const int32_t *keys{nullptr};
  const uint8_t *values{nullptr};
  int32_t *thread_offsets{nullptr};
  uint8_t *output{nullptr};
  std::size_t n{0};
  std::size_t num_bins{0};
  std::size_t item_bytes{0};
  int num_threads{1};
};

template <typename T>
struct CpuGroupedReduceTaskContext {
  const int32_t *keys{nullptr};
  const T *values{nullptr};
  T *partial{nullptr};
  T *output{nullptr};
  std::size_t n{0};
  std::size_t num_groups{0};
  int num_threads{1};
};

template <typename T>
struct CpuStridedGroupedReduceTaskContext {
  const int32_t *keys{nullptr};
  const uint8_t *values{nullptr};
  T *partial{nullptr};
  T *output{nullptr};
  std::size_t n{0};
  std::size_t num_groups{0};
  std::size_t offset{0};
  std::size_t stride{0};
  int num_threads{1};
};

template <typename T>
struct CpuStridedGroupedReduceIoTaskContext {
  const uint8_t *keys{nullptr};
  const uint8_t *values{nullptr};
  T *partial{nullptr};
  uint8_t *output{nullptr};
  std::size_t n{0};
  std::size_t num_groups{0};
  std::size_t keys_offset{0};
  std::size_t keys_stride{sizeof(int32_t)};
  std::size_t values_offset{0};
  std::size_t values_stride{0};
  std::size_t output_offset{0};
  std::size_t output_stride{0};
  int num_threads{1};
};

taichi::ThreadPool &get_cpu_primitive_thread_pool(int max_threads) {
  static std::mutex mutex;
  static std::unique_ptr<taichi::ThreadPool> pool;
  static int pool_threads = 0;
  std::lock_guard<std::mutex> lock(mutex);
  if (!pool || pool_threads < max_threads) {
    pool = std::make_unique<taichi::ThreadPool>(max_threads);
    pool_threads = max_threads;
  }
  return *pool;
}

template <typename T>
T cpu_reduce_identity(int op) {
  if (op == 1) {
    if constexpr (std::is_floating_point_v<T>) {
      return std::numeric_limits<T>::infinity();
    } else {
      return std::numeric_limits<T>::max();
    }
  }
  if (op == 2) {
    if constexpr (std::is_floating_point_v<T>) {
      return -std::numeric_limits<T>::infinity();
    } else {
      return std::numeric_limits<T>::lowest();
    }
  }
  return T{0};
}

template <typename T>
T cpu_reduce_combine(T a, T b, int op) {
  if (op == 1) {
    return std::min(a, b);
  }
  if (op == 2) {
    return std::max(a, b);
  }
  if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
    using U = std::make_unsigned_t<T>;
    U ua = 0;
    U ub = 0;
    std::memcpy(&ua, &a, sizeof(T));
    std::memcpy(&ub, &b, sizeof(T));
    U sum = ua + ub;
    T result{};
    std::memcpy(&result, &sum, sizeof(T));
    return result;
  } else {
    return a + b;
  }
}

template <typename T>
void cpu_reduce_task(void *raw_ctx, int /*thread_id*/, int task_id) {
  auto *ctx = static_cast<CpuReduceTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  T acc = cpu_reduce_identity<T>(ctx->op);
  for (std::size_t i = begin; i < end; ++i) {
    acc = cpu_reduce_combine(acc, ctx->values[i], ctx->op);
  }
  ctx->partial[tid] = acc;
}

template <typename T>
void cpu_strided_reduce_task(void *raw_ctx,
                             int /*thread_id*/,
                             int task_id) {
  auto *ctx = static_cast<CpuStridedReduceTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  T acc = cpu_reduce_identity<T>(ctx->op);
  for (std::size_t i = begin; i < end; ++i) {
    const auto *value = reinterpret_cast<const T *>(
        ctx->values + ctx->offset + i * ctx->stride);
    acc = cpu_reduce_combine(acc, *value, ctx->op);
  }
  ctx->partial[tid] = acc;
}

void cpu_fill_u32_task(void *raw_ctx, int /*thread_id*/, int task_id) {
  auto *ctx = static_cast<CpuFillU32TaskContext *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->words * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->words * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  std::fill(ctx->data + begin, ctx->data + end, ctx->value);
}

void cpu_copy_task(void *raw_ctx, int /*thread_id*/, int task_id) {
  auto *ctx = static_cast<CpuCopyTaskContext *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->bytes * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->bytes * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  std::memcpy(ctx->dst + begin, ctx->src + begin, end - begin);
}

template <typename T>
void cpu_transform_task(void *raw_ctx, int /*thread_id*/, int task_id) {
  auto *ctx = static_cast<CpuTransformTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    ctx->dst[i] = ctx->src[i] * ctx->scale + ctx->bias;
  }
}

template <typename T>
void cpu_strided_transform_task(void *raw_ctx,
                                int /*thread_id*/,
                                int task_id) {
  auto *ctx = static_cast<CpuStridedTransformTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    const auto *value = reinterpret_cast<const T *>(
        ctx->src + ctx->offset + i * ctx->stride);
    ctx->dst[i] = (*value) * ctx->scale + ctx->bias;
  }
}

template <typename T>
void cpu_strided_to_strided_transform_task(void *raw_ctx,
                                           int /*thread_id*/,
                                           int task_id) {
  auto *ctx =
      static_cast<CpuStridedToStridedTransformTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    const auto *value = reinterpret_cast<const T *>(
        ctx->src + ctx->src_offset + i * ctx->src_stride);
    auto *out = reinterpret_cast<T *>(
        ctx->dst + ctx->dst_offset + i * ctx->dst_stride);
    *out = (*value) * ctx->scale + ctx->bias;
  }
}

template <typename T>
void cpu_transform_run_typed(const T *src_ptr,
                             T *dst_ptr,
                             std::size_t n,
                             T scale,
                             T bias,
                             bool use_parallel,
                             int target_threads,
                             int max_threads) {
  if (use_parallel) {
    CpuTransformTaskContext<T> ctx;
    ctx.src = src_ptr;
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.scale = scale;
    ctx.bias = bias;
    ctx.num_threads = target_threads;
    auto &pool = get_cpu_primitive_thread_pool(max_threads);
    pool.run(target_threads, target_threads, &ctx, cpu_transform_task<T>);
    return;
  }
  for (std::size_t i = 0; i < n; ++i) {
    dst_ptr[i] = src_ptr[i] * scale + bias;
  }
}

template <typename T>
void cpu_transform_run_strided_typed(const uint8_t *src_ptr,
                                     T *dst_ptr,
                                     std::size_t n,
                                     std::size_t offset,
                                     std::size_t stride,
                                     T scale,
                                     T bias,
                                     bool use_parallel,
                                     int target_threads,
                                     int max_threads) {
  if (use_parallel) {
    CpuStridedTransformTaskContext<T> ctx;
    ctx.src = src_ptr;
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.offset = offset;
    ctx.stride = stride;
    ctx.scale = scale;
    ctx.bias = bias;
    ctx.num_threads = target_threads;
    auto &pool = get_cpu_primitive_thread_pool(max_threads);
    pool.run(target_threads, target_threads, &ctx,
             cpu_strided_transform_task<T>);
    return;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto *value =
        reinterpret_cast<const T *>(src_ptr + offset + i * stride);
    dst_ptr[i] = (*value) * scale + bias;
  }
}

template <typename T>
void cpu_transform_run_strided_to_strided_typed(const uint8_t *src_ptr,
                                                uint8_t *dst_ptr,
                                                std::size_t n,
                                                std::size_t src_offset,
                                                std::size_t src_stride,
                                                std::size_t dst_offset,
                                                std::size_t dst_stride,
                                                T scale,
                                                T bias,
                                                bool use_parallel,
                                                int target_threads,
                                                int max_threads) {
  if (use_parallel) {
    CpuStridedToStridedTransformTaskContext<T> ctx;
    ctx.src = src_ptr;
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.src_offset = src_offset;
    ctx.src_stride = src_stride;
    ctx.dst_offset = dst_offset;
    ctx.dst_stride = dst_stride;
    ctx.scale = scale;
    ctx.bias = bias;
    ctx.num_threads = target_threads;
    auto &pool = get_cpu_primitive_thread_pool(max_threads);
    pool.run(target_threads, target_threads, &ctx,
             cpu_strided_to_strided_transform_task<T>);
    return;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto *value =
        reinterpret_cast<const T *>(src_ptr + src_offset + i * src_stride);
    auto *out =
        reinterpret_cast<T *>(dst_ptr + dst_offset + i * dst_stride);
    *out = (*value) * scale + bias;
  }
}

std::size_t transform_value_size(int value_type) {
  TI_ERROR_IF(value_type < 0 || value_type > 5,
              "transform received an unsupported value type.");
  return (value_type == 3 || value_type == 4 || value_type == 5)
             ? sizeof(uint64_t)
             : sizeof(uint32_t);
}

void check_transform_member_request(const char *backend,
                                    Ndarray *src,
                                    Ndarray *dst,
                                    int value_type,
                                    std::size_t offset,
                                    std::size_t stride) {
  TI_ERROR_IF(!src || !dst, "{} strided transform received a null ndarray.",
              backend);
  TI_ERROR_IF(src->get_nelement() != dst->get_nelement(),
              "{} strided transform source and destination sizes differ.",
              backend);
  const std::size_t value_size = transform_value_size(value_type);
  TI_ERROR_IF(dst->get_element_size() != value_size,
              "{} strided transform destination dtype does not match value "
              "type.",
              backend);
  TI_ERROR_IF(stride < value_size,
              "{} strided transform source stride is smaller than value size.",
              backend);
  TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
              "{} strided transform source offset/stride must align to value "
              "size.",
              backend);
  const std::size_t n = src->get_nelement();
  if (n == 0) {
    return;
  }
  const std::size_t src_bytes = n * src->get_element_size();
  TI_ERROR_IF(src_bytes < value_size,
              "{} strided transform source buffer is smaller than value size.",
              backend);
  TI_ERROR_IF(offset > src_bytes - value_size,
              "{} strided transform source offset is out of bounds.", backend);
  const std::size_t last = offset + (n - 1) * stride + value_size;
  TI_ERROR_IF(last > src_bytes,
              "{} strided transform source range is out of bounds.", backend);
}

void check_transform_strided_range(const char *backend,
                                   const char *role,
                                   Ndarray *arr,
                                   std::size_t logical_items,
                                   std::size_t value_size,
                                   std::size_t offset,
                                   std::size_t stride) {
  TI_ERROR_IF(stride < value_size,
              "{} strided transform {} stride is smaller than value size.",
              backend, role);
  TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
              "{} strided transform {} offset/stride must align to value "
              "size.",
              backend, role);
  if (logical_items == 0) {
    return;
  }
  const std::size_t bytes = arr->get_nelement() * arr->get_element_size();
  TI_ERROR_IF(bytes < value_size,
              "{} strided transform {} buffer is smaller than value size.",
              backend, role);
  TI_ERROR_IF(offset > bytes - value_size,
              "{} strided transform {} offset is out of bounds.", backend,
              role);
  const std::size_t last = offset + (logical_items - 1) * stride + value_size;
  TI_ERROR_IF(last > bytes,
              "{} strided transform {} range is out of bounds.", backend,
              role);
}

void check_transform_strided_request(const char *backend,
                                     Ndarray *src,
                                     Ndarray *dst,
                                     int value_type,
                                     std::size_t src_offset,
                                     std::size_t src_stride,
                                     std::size_t dst_offset,
                                     std::size_t dst_stride) {
  TI_ERROR_IF(!src || !dst, "{} strided transform received a null ndarray.",
              backend);
  TI_ERROR_IF(src->get_nelement() != dst->get_nelement(),
              "{} strided transform source and destination sizes differ.",
              backend);
  const std::size_t value_size = transform_value_size(value_type);
  const std::size_t n = src->get_nelement();
  check_transform_strided_range(backend, "source", src, n, value_size,
                                src_offset, src_stride);
  check_transform_strided_range(backend, "destination", dst, n, value_size,
                                dst_offset, dst_stride);
}

void check_indexed_copy_strided_request(const char *backend,
                                        Ndarray *src,
                                        Ndarray *indices,
                                        Ndarray *dst,
                                        std::size_t item_bytes,
                                        std::size_t src_offset,
                                        std::size_t src_stride,
                                        std::size_t dst_offset,
                                        std::size_t dst_stride,
                                        bool scatter) {
  TI_ERROR_IF(!src || !indices || !dst,
              "{} strided indexed-copy received a null ndarray.", backend);
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "{} strided indexed-copy expects 1D ndarrays.", backend);
  TI_ERROR_IF(indices->get_element_size() != sizeof(int32_t),
              "{} strided indexed-copy expects i32 indices.", backend);
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(uint32_t) != 0,
              "{} strided indexed-copy item size must be a positive "
              "uint32-word multiple.",
              backend);
  if (scatter) {
    TI_ERROR_IF(src->get_nelement() != indices->get_nelement(),
                "{} strided scatter expects source and indices sizes to "
                "match.",
                backend);
  } else {
    TI_ERROR_IF(indices->get_nelement() != dst->get_nelement(),
                "{} strided gather expects indices and destination sizes to "
                "match.",
                backend);
  }
  auto check_range = [&](const char *role, Ndarray *arr,
                         std::size_t logical_items, std::size_t offset,
                         std::size_t stride) {
    TI_ERROR_IF(stride < item_bytes,
                "{} strided indexed-copy {} stride is smaller than item "
                "size.",
                backend, role);
    TI_ERROR_IF(offset % sizeof(uint32_t) != 0 ||
                    stride % sizeof(uint32_t) != 0,
                "{} strided indexed-copy {} offset/stride must be "
                "uint32-word aligned.",
                backend, role);
    if (logical_items == 0) {
      return;
    }
    const std::size_t bytes = arr->get_nelement() * arr->get_element_size();
    TI_ERROR_IF(bytes < item_bytes,
                "{} strided indexed-copy {} buffer is smaller than item "
                "size.",
                backend, role);
    TI_ERROR_IF(offset > bytes - item_bytes,
                "{} strided indexed-copy {} offset is out of bounds.",
                backend, role);
    const std::size_t last = offset + (logical_items - 1) * stride + item_bytes;
    TI_ERROR_IF(last > bytes,
                "{} strided indexed-copy {} range is out of bounds.",
                backend, role);
  };
  check_range("source", src, src->get_nelement(), src_offset, src_stride);
  check_range("destination", dst, dst->get_nelement(), dst_offset,
              dst_stride);
}

void cpu_indexed_copy_task(void *raw_ctx, int /*thread_id*/, int task_id) {
  auto *ctx = static_cast<CpuIndexedCopyTaskContext *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  if (ctx->scatter) {
    for (std::size_t i = begin; i < end; ++i) {
      const auto index = static_cast<std::size_t>(ctx->indices[i]);
      if (index < ctx->index_bound) {
        std::memcpy(ctx->dst + index * ctx->item_bytes,
                    ctx->src + i * ctx->item_bytes, ctx->item_bytes);
      }
    }
  } else {
    for (std::size_t i = begin; i < end; ++i) {
      const auto index = static_cast<std::size_t>(ctx->indices[i]);
      if (index < ctx->index_bound) {
        std::memcpy(ctx->dst + i * ctx->item_bytes,
                    ctx->src + index * ctx->item_bytes, ctx->item_bytes);
      } else {
        std::memset(ctx->dst + i * ctx->item_bytes, 0, ctx->item_bytes);
      }
    }
  }
}

void cpu_strided_indexed_copy_task(void *raw_ctx,
                                   int /*thread_id*/,
                                   int task_id) {
  auto *ctx = static_cast<CpuStridedIndexedCopyTaskContext *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  if (ctx->scatter) {
    for (std::size_t i = begin; i < end; ++i) {
      const auto index = static_cast<std::size_t>(ctx->indices[i]);
      if (index < ctx->index_bound) {
        std::memcpy(ctx->dst + ctx->dst_offset + index * ctx->dst_stride,
                    ctx->src + ctx->src_offset + i * ctx->src_stride,
                    ctx->item_bytes);
      }
    }
  } else {
    for (std::size_t i = begin; i < end; ++i) {
      const auto index = static_cast<std::size_t>(ctx->indices[i]);
      if (index < ctx->index_bound) {
        std::memcpy(ctx->dst + ctx->dst_offset + i * ctx->dst_stride,
                    ctx->src + ctx->src_offset + index * ctx->src_stride,
                    ctx->item_bytes);
      } else {
        std::memset(ctx->dst + ctx->dst_offset + i * ctx->dst_stride, 0,
                    ctx->item_bytes);
      }
    }
  }
}

template <typename T>
void cpu_scatter_add_count_task(void *raw_ctx,
                                int /*thread_id*/,
                                int task_id) {
  auto *ctx = static_cast<CpuScatterAddTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  T *local = ctx->partial + ctx->dst_items * static_cast<std::size_t>(tid);
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    const auto index = static_cast<std::size_t>(ctx->indices[i]);
    if (index < ctx->dst_items) {
      local[index] = cpu_reduce_combine(local[index], ctx->src[i], 0);
    }
  }
}

template <typename T>
void cpu_scatter_add_merge_task(void *raw_ctx,
                                int /*thread_id*/,
                                int task_id) {
  auto *ctx = static_cast<CpuScatterAddTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->dst_items * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->dst_items * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    T value{};
    for (int t = 0; t < ctx->num_threads; ++t) {
      value = cpu_reduce_combine(
          value,
          ctx->partial[ctx->dst_items * static_cast<std::size_t>(t) + i], 0);
    }
    ctx->dst[i] = cpu_reduce_combine(ctx->dst[i], value, 0);
  }
}

template <typename T>
void cpu_strided_scatter_add_count_task(void *raw_ctx,
                                        int /*thread_id*/,
                                        int task_id) {
  auto *ctx = static_cast<CpuStridedScatterAddTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  T *local = ctx->partial + ctx->dst_items * static_cast<std::size_t>(tid);
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    const auto index = static_cast<std::size_t>(ctx->indices[i]);
    if (index < ctx->dst_items) {
      const auto *value =
          reinterpret_cast<const T *>(ctx->src + ctx->offset + i * ctx->stride);
      local[index] = cpu_reduce_combine(local[index], *value, 0);
    }
  }
}

template <typename T>
void cpu_strided_scatter_add_merge_task(void *raw_ctx,
                                        int /*thread_id*/,
                                        int task_id) {
  auto *ctx = static_cast<CpuStridedScatterAddTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->dst_items * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->dst_items * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    T value{};
    for (int t = 0; t < ctx->num_threads; ++t) {
      value = cpu_reduce_combine(
          value,
          ctx->partial[ctx->dst_items * static_cast<std::size_t>(t) + i], 0);
    }
    ctx->dst[i] = cpu_reduce_combine(ctx->dst[i], value, 0);
  }
}

template <typename T>
void cpu_strided_scatter_add_io_count_task(void *raw_ctx,
                                           int /*thread_id*/,
                                           int task_id) {
  auto *ctx = static_cast<CpuStridedScatterAddIoTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  T *local = ctx->partial + ctx->dst_items * static_cast<std::size_t>(tid);
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    const auto index = static_cast<std::size_t>(ctx->indices[i]);
    if (index < ctx->dst_items) {
      const auto *value = reinterpret_cast<const T *>(
          ctx->src + ctx->src_offset + i * ctx->src_stride);
      local[index] = cpu_reduce_combine(local[index], *value, 0);
    }
  }
}

template <typename T>
void cpu_strided_scatter_add_io_merge_task(void *raw_ctx,
                                           int /*thread_id*/,
                                           int task_id) {
  auto *ctx = static_cast<CpuStridedScatterAddIoTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->dst_items * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->dst_items * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    T value{};
    for (int t = 0; t < ctx->num_threads; ++t) {
      value = cpu_reduce_combine(
          value,
          ctx->partial[ctx->dst_items * static_cast<std::size_t>(t) + i], 0);
    }
    auto *dst_value =
        reinterpret_cast<T *>(ctx->dst + ctx->dst_offset + i * ctx->dst_stride);
    *dst_value = cpu_reduce_combine(*dst_value, value, 0);
  }
}

template <typename T>
std::size_t cpu_scatter_add_typed(const T *src_ptr,
                                  const int32_t *indices_ptr,
                                  T *dst_ptr,
                                  std::size_t n,
                                  std::size_t dst_items,
                                  int max_threads,
                                  int target_threads) {
  TI_ERROR_IF(!src_ptr || !dst_ptr,
              "CPU native scatter-add received a null data pointer.");
  constexpr std::size_t kMaxWorkspaceBytes = 64ull << 20;
  const std::size_t workspace_bytes =
      static_cast<std::size_t>(target_threads) * dst_items * sizeof(T);
  if (target_threads > 1 && workspace_bytes <= kMaxWorkspaceBytes) {
    std::vector<T> partial(static_cast<std::size_t>(target_threads) * dst_items,
                           T{});
    CpuScatterAddTaskContext<T> ctx;
    ctx.src = src_ptr;
    ctx.indices = indices_ptr;
    ctx.partial = partial.data();
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.dst_items = dst_items;
    ctx.num_threads = target_threads;
    auto &pool = get_cpu_primitive_thread_pool(max_threads);
    pool.run(target_threads, target_threads, &ctx,
             cpu_scatter_add_count_task<T>);
    pool.run(target_threads, target_threads, &ctx,
             cpu_scatter_add_merge_task<T>);
    update_cpu_scatter_add_workspace_peak(workspace_bytes);
    return workspace_bytes;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto index = static_cast<std::size_t>(indices_ptr[i]);
    if (index < dst_items) {
      dst_ptr[index] = cpu_reduce_combine(dst_ptr[index], src_ptr[i], 0);
    }
  }
  return 0;
}

template <typename T>
std::size_t cpu_scatter_add_strided_typed(const uint8_t *src_ptr,
                                          std::size_t offset,
                                          std::size_t stride,
                                          const int32_t *indices_ptr,
                                          T *dst_ptr,
                                          std::size_t n,
                                          std::size_t dst_items,
                                          int max_threads,
                                          int target_threads) {
  TI_ERROR_IF(!src_ptr || !dst_ptr,
              "CPU native strided scatter-add received a null data pointer.");
  constexpr std::size_t kMaxWorkspaceBytes = 64ull << 20;
  const std::size_t workspace_bytes =
      static_cast<std::size_t>(target_threads) * dst_items * sizeof(T);
  if (target_threads > 1 && workspace_bytes <= kMaxWorkspaceBytes) {
    std::vector<T> partial(static_cast<std::size_t>(target_threads) * dst_items,
                           T{});
    CpuStridedScatterAddTaskContext<T> ctx;
    ctx.src = src_ptr;
    ctx.indices = indices_ptr;
    ctx.partial = partial.data();
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.dst_items = dst_items;
    ctx.offset = offset;
    ctx.stride = stride;
    ctx.num_threads = target_threads;
    auto &pool = get_cpu_primitive_thread_pool(max_threads);
    pool.run(target_threads, target_threads, &ctx,
             cpu_strided_scatter_add_count_task<T>);
    pool.run(target_threads, target_threads, &ctx,
             cpu_strided_scatter_add_merge_task<T>);
    update_cpu_scatter_add_workspace_peak(workspace_bytes);
    return workspace_bytes;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto index = static_cast<std::size_t>(indices_ptr[i]);
    if (index < dst_items) {
      const auto *value =
          reinterpret_cast<const T *>(src_ptr + offset + i * stride);
      dst_ptr[index] = cpu_reduce_combine(dst_ptr[index], *value, 0);
    }
  }
  return 0;
}

template <typename T>
std::size_t cpu_scatter_add_strided_io_typed(const uint8_t *src_ptr,
                                             std::size_t src_offset,
                                             std::size_t src_stride,
                                             const int32_t *indices_ptr,
                                             uint8_t *dst_ptr,
                                             std::size_t dst_offset,
                                             std::size_t dst_stride,
                                             std::size_t n,
                                             std::size_t dst_items,
                                             int max_threads,
                                             int target_threads) {
  TI_ERROR_IF(!src_ptr || !dst_ptr,
              "CPU native strided scatter-add received a null data pointer.");
  constexpr std::size_t kMaxWorkspaceBytes = 64ull << 20;
  const std::size_t workspace_bytes =
      static_cast<std::size_t>(target_threads) * dst_items * sizeof(T);
  if (target_threads > 1 && workspace_bytes <= kMaxWorkspaceBytes) {
    std::vector<T> partial(static_cast<std::size_t>(target_threads) *
                               dst_items,
                           T{});
    CpuStridedScatterAddIoTaskContext<T> ctx;
    ctx.src = src_ptr;
    ctx.indices = indices_ptr;
    ctx.partial = partial.data();
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.dst_items = dst_items;
    ctx.src_offset = src_offset;
    ctx.src_stride = src_stride;
    ctx.dst_offset = dst_offset;
    ctx.dst_stride = dst_stride;
    ctx.num_threads = target_threads;
    auto &pool = get_cpu_primitive_thread_pool(max_threads);
    pool.run(target_threads, target_threads, &ctx,
             cpu_strided_scatter_add_io_count_task<T>);
    pool.run(target_threads, target_threads, &ctx,
             cpu_strided_scatter_add_io_merge_task<T>);
    update_cpu_scatter_add_workspace_peak(workspace_bytes);
    return workspace_bytes;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto index = static_cast<std::size_t>(indices_ptr[i]);
    if (index < dst_items) {
      const auto *value =
          reinterpret_cast<const T *>(src_ptr + src_offset + i * src_stride);
      auto *dst_value = reinterpret_cast<T *>(dst_ptr + dst_offset +
                                              index * dst_stride);
      *dst_value = cpu_reduce_combine(*dst_value, *value, 0);
    }
  }
  return 0;
}

void cpu_bucket_count_task(void *raw_ctx, int /*thread_id*/, int task_id) {
  auto *ctx = static_cast<CpuBucketCountTaskContext *>(raw_ctx);
  const int tid = task_id;
  int32_t *local = ctx->partial + ctx->num_bins * static_cast<std::size_t>(tid);
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    int32_t key = ctx->keys[i];
    if (key >= 0 && static_cast<std::size_t>(key) < ctx->num_bins) {
      local[key] += 1;
    }
  }
}

template <typename T>
void cpu_bucket_scatter_task(void *raw_ctx, int /*thread_id*/, int task_id) {
  auto *ctx = static_cast<CpuBucketScatterTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  int32_t *local =
      ctx->thread_offsets + ctx->num_bins * static_cast<std::size_t>(tid);
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    int32_t key = ctx->keys[i];
    if (key >= 0 && static_cast<std::size_t>(key) < ctx->num_bins) {
      int32_t pos = local[key]++;
      if (pos >= 0 && static_cast<std::size_t>(pos) < ctx->n) {
        ctx->output[pos] = ctx->values[i];
      }
    }
  }
}

void cpu_bucket_scatter_raw_task(void *raw_ctx,
                                 int /*thread_id*/,
                                 int task_id) {
  auto *ctx = static_cast<CpuBucketScatterRawTaskContext *>(raw_ctx);
  const int tid = task_id;
  int32_t *local =
      ctx->thread_offsets + ctx->num_bins * static_cast<std::size_t>(tid);
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    int32_t key = ctx->keys[i];
    if (key >= 0 && static_cast<std::size_t>(key) < ctx->num_bins) {
      int32_t pos = local[key]++;
      if (pos >= 0 && static_cast<std::size_t>(pos) < ctx->n) {
        std::memcpy(ctx->output + static_cast<std::size_t>(pos) *
                                      ctx->item_bytes,
                    ctx->values + i * ctx->item_bytes, ctx->item_bytes);
      }
    }
  }
}

template <typename T>
std::size_t cpu_bucket_builder_typed(const int32_t *keys_ptr,
                                     const T *values_ptr,
                                     int32_t *offsets_ptr,
                                     T *output_ptr,
                                     std::size_t n,
                                     std::size_t num_bins,
                                     int max_threads) {
  TI_ERROR_IF(!keys_ptr || !values_ptr || !offsets_ptr || !output_ptr,
              "CPU native bucket builder received a null data pointer.");
  std::fill(offsets_ptr, offsets_ptr + num_bins + 1, 0);
  if (n == 0) {
    return 0;
  }

  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  constexpr std::size_t kMaxWorkspaceBytes = 64ull << 20;
  const std::size_t parallel_workspace =
      static_cast<std::size_t>(target_threads) * num_bins * sizeof(int32_t) *
      2;
  const bool use_parallel =
      n >= 65536 && target_threads > 1 && parallel_workspace <= kMaxWorkspaceBytes;

  if (use_parallel) {
    std::vector<int32_t> partial(
        static_cast<std::size_t>(target_threads) * num_bins, 0);
    CpuBucketCountTaskContext count_ctx;
    count_ctx.keys = keys_ptr;
    count_ctx.partial = partial.data();
    count_ctx.n = n;
    count_ctx.num_bins = num_bins;
    count_ctx.num_threads = target_threads;
    auto &pool = get_cpu_primitive_thread_pool(max_threads);
    pool.run(target_threads, target_threads, &count_ctx, cpu_bucket_count_task);

    std::vector<int32_t> thread_offsets(
        static_cast<std::size_t>(target_threads) * num_bins, 0);
    int64_t running = 0;
    offsets_ptr[0] = 0;
    for (std::size_t bin = 0; bin < num_bins; ++bin) {
      int64_t pos = running;
      for (int tid = 0; tid < target_threads; ++tid) {
        const std::size_t idx =
            static_cast<std::size_t>(tid) * num_bins + bin;
        thread_offsets[idx] = static_cast<int32_t>(pos);
        pos += partial[idx];
      }
      running = pos;
      TI_ERROR_IF(running > std::numeric_limits<int32_t>::max(),
                  "CPU native bucket builder valid item count exceeds i32 range.");
      offsets_ptr[bin + 1] = static_cast<int32_t>(running);
    }

    CpuBucketScatterTaskContext<T> scatter_ctx;
    scatter_ctx.keys = keys_ptr;
    scatter_ctx.values = values_ptr;
    scatter_ctx.thread_offsets = thread_offsets.data();
    scatter_ctx.output = output_ptr;
    scatter_ctx.n = n;
    scatter_ctx.num_bins = num_bins;
    scatter_ctx.num_threads = target_threads;
    pool.run(target_threads, target_threads, &scatter_ctx,
             cpu_bucket_scatter_task<T>);
    return parallel_workspace;
  }

  for (std::size_t i = 0; i < n; ++i) {
    int32_t key = keys_ptr[i];
    if (key >= 0 && static_cast<std::size_t>(key) < num_bins) {
      offsets_ptr[static_cast<std::size_t>(key) + 1] += 1;
    }
  }
  int64_t running = 0;
  for (std::size_t bin = 0; bin <= num_bins; ++bin) {
    running += offsets_ptr[bin];
    TI_ERROR_IF(running > std::numeric_limits<int32_t>::max(),
                "CPU native bucket builder valid item count exceeds i32 range.");
    offsets_ptr[bin] = static_cast<int32_t>(running);
  }
  std::vector<int32_t> cursor(offsets_ptr, offsets_ptr + num_bins);
  for (std::size_t i = 0; i < n; ++i) {
    int32_t key = keys_ptr[i];
    if (key >= 0 && static_cast<std::size_t>(key) < num_bins) {
      int32_t pos = cursor[key]++;
      output_ptr[pos] = values_ptr[i];
    }
  }
  return cursor.size() * sizeof(int32_t);
}

std::size_t cpu_bucket_builder_raw(const int32_t *keys_ptr,
                                   const uint8_t *values_ptr,
                                   int32_t *offsets_ptr,
                                   uint8_t *output_ptr,
                                   std::size_t n,
                                   std::size_t num_bins,
                                   std::size_t item_bytes,
                                   int max_threads) {
  TI_ERROR_IF(!keys_ptr || !values_ptr || !offsets_ptr || !output_ptr,
              "CPU native bucket builder received a null data pointer.");
  TI_ERROR_IF(item_bytes == 0,
              "CPU native bucket builder received empty payload items.");
  std::fill(offsets_ptr, offsets_ptr + num_bins + 1, 0);
  if (n == 0) {
    return 0;
  }

  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  constexpr std::size_t kMaxWorkspaceBytes = 64ull << 20;
  const std::size_t parallel_workspace =
      static_cast<std::size_t>(target_threads) * num_bins * sizeof(int32_t) *
      2;
  const bool use_parallel =
      n >= 65536 && target_threads > 1 && parallel_workspace <= kMaxWorkspaceBytes;

  if (use_parallel) {
    std::vector<int32_t> partial(
        static_cast<std::size_t>(target_threads) * num_bins, 0);
    CpuBucketCountTaskContext count_ctx;
    count_ctx.keys = keys_ptr;
    count_ctx.partial = partial.data();
    count_ctx.n = n;
    count_ctx.num_bins = num_bins;
    count_ctx.num_threads = target_threads;
    auto &pool = get_cpu_primitive_thread_pool(max_threads);
    pool.run(target_threads, target_threads, &count_ctx, cpu_bucket_count_task);

    std::vector<int32_t> thread_offsets(
        static_cast<std::size_t>(target_threads) * num_bins, 0);
    int64_t running = 0;
    offsets_ptr[0] = 0;
    for (std::size_t bin = 0; bin < num_bins; ++bin) {
      int64_t pos = running;
      for (int tid = 0; tid < target_threads; ++tid) {
        const std::size_t idx =
            static_cast<std::size_t>(tid) * num_bins + bin;
        thread_offsets[idx] = static_cast<int32_t>(pos);
        pos += partial[idx];
      }
      running = pos;
      TI_ERROR_IF(running > std::numeric_limits<int32_t>::max(),
                  "CPU native bucket builder valid item count exceeds i32 range.");
      offsets_ptr[bin + 1] = static_cast<int32_t>(running);
    }

    CpuBucketScatterRawTaskContext scatter_ctx;
    scatter_ctx.keys = keys_ptr;
    scatter_ctx.values = values_ptr;
    scatter_ctx.thread_offsets = thread_offsets.data();
    scatter_ctx.output = output_ptr;
    scatter_ctx.n = n;
    scatter_ctx.num_bins = num_bins;
    scatter_ctx.item_bytes = item_bytes;
    scatter_ctx.num_threads = target_threads;
    pool.run(target_threads, target_threads, &scatter_ctx,
             cpu_bucket_scatter_raw_task);
    return parallel_workspace;
  }

  for (std::size_t i = 0; i < n; ++i) {
    int32_t key = keys_ptr[i];
    if (key >= 0 && static_cast<std::size_t>(key) < num_bins) {
      offsets_ptr[static_cast<std::size_t>(key) + 1] += 1;
    }
  }
  int64_t running = 0;
  for (std::size_t bin = 0; bin <= num_bins; ++bin) {
    running += offsets_ptr[bin];
    TI_ERROR_IF(running > std::numeric_limits<int32_t>::max(),
                "CPU native bucket builder valid item count exceeds i32 range.");
    offsets_ptr[bin] = static_cast<int32_t>(running);
  }
  std::vector<int32_t> cursor(offsets_ptr, offsets_ptr + num_bins);
  for (std::size_t i = 0; i < n; ++i) {
    int32_t key = keys_ptr[i];
    if (key >= 0 && static_cast<std::size_t>(key) < num_bins) {
      int32_t pos = cursor[key]++;
      std::memcpy(output_ptr + static_cast<std::size_t>(pos) * item_bytes,
                  values_ptr + i * item_bytes, item_bytes);
    }
  }
  return cursor.size() * sizeof(int32_t);
}

template <typename T>
void cpu_grouped_reduce_count_task(void *raw_ctx,
                                   int /*thread_id*/,
                                   int task_id) {
  auto *ctx = static_cast<CpuGroupedReduceTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  T *local =
      ctx->partial + ctx->num_groups * static_cast<std::size_t>(tid);
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    int32_t key = ctx->keys[i];
    if (key >= 0 && static_cast<std::size_t>(key) < ctx->num_groups) {
      local[key] = cpu_reduce_combine(local[key], ctx->values[i], 0);
    }
  }
}

template <typename T>
void cpu_grouped_reduce_merge_task(void *raw_ctx,
                                   int /*thread_id*/,
                                   int task_id) {
  auto *ctx = static_cast<CpuGroupedReduceTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->num_groups * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->num_groups * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t group = begin; group < end; ++group) {
    T value{};
    for (int t = 0; t < ctx->num_threads; ++t) {
      value = cpu_reduce_combine(
          value,
          ctx->partial[ctx->num_groups * static_cast<std::size_t>(t) + group],
          0);
    }
    ctx->output[group] = value;
  }
}

template <typename T>
void cpu_strided_grouped_reduce_count_task(void *raw_ctx,
                                           int /*thread_id*/,
                                           int task_id) {
  auto *ctx = static_cast<CpuStridedGroupedReduceTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  T *local =
      ctx->partial + ctx->num_groups * static_cast<std::size_t>(tid);
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    int32_t key = ctx->keys[i];
    if (key >= 0 && static_cast<std::size_t>(key) < ctx->num_groups) {
      const auto *value =
          reinterpret_cast<const T *>(ctx->values + ctx->offset +
                                      i * ctx->stride);
      local[key] = cpu_reduce_combine(local[key], *value, 0);
    }
  }
}

template <typename T>
void cpu_strided_grouped_reduce_merge_task(void *raw_ctx,
                                           int /*thread_id*/,
                                           int task_id) {
  auto *ctx = static_cast<CpuStridedGroupedReduceTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->num_groups * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->num_groups * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t group = begin; group < end; ++group) {
    T value{};
    for (int t = 0; t < ctx->num_threads; ++t) {
      value = cpu_reduce_combine(
          value,
          ctx->partial[ctx->num_groups * static_cast<std::size_t>(t) + group],
          0);
    }
    ctx->output[group] = value;
  }
}

template <typename T>
void cpu_strided_grouped_reduce_io_count_task(void *raw_ctx,
                                              int /*thread_id*/,
                                              int task_id) {
  auto *ctx = static_cast<CpuStridedGroupedReduceIoTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  T *local =
      ctx->partial + ctx->num_groups * static_cast<std::size_t>(tid);
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t i = begin; i < end; ++i) {
    const int32_t key = *reinterpret_cast<const int32_t *>(
        ctx->keys + ctx->keys_offset + i * ctx->keys_stride);
    if (key >= 0 && static_cast<std::size_t>(key) < ctx->num_groups) {
      const auto *value = reinterpret_cast<const T *>(
          ctx->values + ctx->values_offset + i * ctx->values_stride);
      local[key] = cpu_reduce_combine(local[key], *value, 0);
    }
  }
}

template <typename T>
void cpu_strided_grouped_reduce_io_merge_task(void *raw_ctx,
                                              int /*thread_id*/,
                                              int task_id) {
  auto *ctx = static_cast<CpuStridedGroupedReduceIoTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->num_groups * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->num_groups * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  for (std::size_t group = begin; group < end; ++group) {
    T value{};
    for (int t = 0; t < ctx->num_threads; ++t) {
      value = cpu_reduce_combine(
          value,
          ctx->partial[ctx->num_groups * static_cast<std::size_t>(t) + group],
          0);
    }
    auto *out_value = reinterpret_cast<T *>(
        ctx->output + ctx->output_offset + group * ctx->output_stride);
    *out_value = value;
  }
}

template <typename T>
std::size_t cpu_grouped_reduce_typed(const int32_t *keys_ptr,
                                     const T *values_ptr,
                                     T *output_ptr,
                                     std::size_t n,
                                     std::size_t num_groups,
                                     int max_threads,
                                     int target_threads) {
  TI_ERROR_IF(!keys_ptr || !values_ptr || !output_ptr,
              "CPU native grouped reduce received a null data pointer.");
  std::fill(output_ptr, output_ptr + num_groups, T{});
  if (n == 0) {
    return 0;
  }
  constexpr std::size_t kMaxWorkspaceBytes = 64ull << 20;
  const std::size_t workspace_bytes =
      static_cast<std::size_t>(target_threads) * num_groups * sizeof(T);
  if (target_threads > 1 && workspace_bytes <= kMaxWorkspaceBytes) {
    std::vector<T> partial(static_cast<std::size_t>(target_threads) *
                               num_groups,
                           T{});
    CpuGroupedReduceTaskContext<T> ctx;
    ctx.keys = keys_ptr;
    ctx.values = values_ptr;
    ctx.partial = partial.data();
    ctx.output = output_ptr;
    ctx.n = n;
    ctx.num_groups = num_groups;
    ctx.num_threads = target_threads;
    auto &pool = get_cpu_primitive_thread_pool(max_threads);
    pool.run(target_threads, target_threads, &ctx,
             cpu_grouped_reduce_count_task<T>);
    pool.run(target_threads, target_threads, &ctx,
             cpu_grouped_reduce_merge_task<T>);
    update_cpu_grouped_reduce_workspace_peak(workspace_bytes);
    return workspace_bytes;
  }

  for (std::size_t i = 0; i < n; ++i) {
    int32_t key = keys_ptr[i];
    if (key >= 0 && static_cast<std::size_t>(key) < num_groups) {
      output_ptr[key] = cpu_reduce_combine(output_ptr[key], values_ptr[i], 0);
    }
  }
  return 0;
}

template <typename T>
std::size_t cpu_grouped_reduce_strided_typed(const int32_t *keys_ptr,
                                             const uint8_t *values_ptr,
                                             std::size_t offset,
                                             std::size_t stride,
                                             T *output_ptr,
                                             std::size_t n,
                                             std::size_t num_groups,
                                             int max_threads,
                                             int target_threads) {
  TI_ERROR_IF(!keys_ptr || !values_ptr || !output_ptr,
              "CPU native strided grouped reduce received a null data pointer.");
  std::fill(output_ptr, output_ptr + num_groups, T{});
  if (n == 0) {
    return 0;
  }
  constexpr std::size_t kMaxWorkspaceBytes = 64ull << 20;
  const std::size_t workspace_bytes =
      static_cast<std::size_t>(target_threads) * num_groups * sizeof(T);
  if (target_threads > 1 && workspace_bytes <= kMaxWorkspaceBytes) {
    std::vector<T> partial(static_cast<std::size_t>(target_threads) *
                               num_groups,
                           T{});
    CpuStridedGroupedReduceTaskContext<T> ctx;
    ctx.keys = keys_ptr;
    ctx.values = values_ptr;
    ctx.partial = partial.data();
    ctx.output = output_ptr;
    ctx.n = n;
    ctx.num_groups = num_groups;
    ctx.offset = offset;
    ctx.stride = stride;
    ctx.num_threads = target_threads;
    auto &pool = get_cpu_primitive_thread_pool(max_threads);
    pool.run(target_threads, target_threads, &ctx,
             cpu_strided_grouped_reduce_count_task<T>);
    pool.run(target_threads, target_threads, &ctx,
             cpu_strided_grouped_reduce_merge_task<T>);
    update_cpu_grouped_reduce_workspace_peak(workspace_bytes);
    return workspace_bytes;
  }

  for (std::size_t i = 0; i < n; ++i) {
    int32_t key = keys_ptr[i];
    if (key >= 0 && static_cast<std::size_t>(key) < num_groups) {
      const auto *value =
          reinterpret_cast<const T *>(values_ptr + offset + i * stride);
      output_ptr[key] = cpu_reduce_combine(output_ptr[key], *value, 0);
    }
  }
  return 0;
}

template <typename T>
std::size_t cpu_grouped_reduce_strided_io_typed(const uint8_t *keys_ptr,
                                                std::size_t keys_offset,
                                                std::size_t keys_stride,
                                                const uint8_t *values_ptr,
                                                std::size_t values_offset,
                                                std::size_t values_stride,
                                                uint8_t *output_ptr,
                                                std::size_t output_offset,
                                                std::size_t output_stride,
                                                std::size_t n,
                                                std::size_t num_groups,
                                                int max_threads,
                                                int target_threads) {
  TI_ERROR_IF(!keys_ptr || !values_ptr || !output_ptr,
              "CPU native strided grouped reduce received a null data pointer.");
  for (std::size_t group = 0; group < num_groups; ++group) {
    auto *out_value = reinterpret_cast<T *>(
        output_ptr + output_offset + group * output_stride);
    *out_value = T{};
  }
  if (n == 0) {
    return 0;
  }
  constexpr std::size_t kMaxWorkspaceBytes = 64ull << 20;
  const std::size_t workspace_bytes =
      static_cast<std::size_t>(target_threads) * num_groups * sizeof(T);
  if (target_threads > 1 && workspace_bytes <= kMaxWorkspaceBytes) {
    std::vector<T> partial(static_cast<std::size_t>(target_threads) *
                               num_groups,
                           T{});
    CpuStridedGroupedReduceIoTaskContext<T> ctx;
    ctx.keys = keys_ptr;
    ctx.values = values_ptr;
    ctx.partial = partial.data();
    ctx.output = output_ptr;
    ctx.n = n;
    ctx.num_groups = num_groups;
    ctx.keys_offset = keys_offset;
    ctx.keys_stride = keys_stride;
    ctx.values_offset = values_offset;
    ctx.values_stride = values_stride;
    ctx.output_offset = output_offset;
    ctx.output_stride = output_stride;
    ctx.num_threads = target_threads;
    auto &pool = get_cpu_primitive_thread_pool(max_threads);
    pool.run(target_threads, target_threads, &ctx,
             cpu_strided_grouped_reduce_io_count_task<T>);
    pool.run(target_threads, target_threads, &ctx,
             cpu_strided_grouped_reduce_io_merge_task<T>);
    update_cpu_grouped_reduce_workspace_peak(workspace_bytes);
    return workspace_bytes;
  }

  for (std::size_t i = 0; i < n; ++i) {
    const int32_t key = *reinterpret_cast<const int32_t *>(
        keys_ptr + keys_offset + i * keys_stride);
    if (key >= 0 && static_cast<std::size_t>(key) < num_groups) {
      const auto *value = reinterpret_cast<const T *>(
          values_ptr + values_offset + i * values_stride);
      auto *out_value = reinterpret_cast<T *>(
          output_ptr + output_offset + static_cast<std::size_t>(key) *
                                         output_stride);
      *out_value = cpu_reduce_combine(*out_value, *value, 0);
    }
  }
  return 0;
}

std::size_t primitive_value_type_size(int value_type) {
  if (value_type >= 0 && value_type <= 2) {
    return sizeof(uint32_t);
  }
  if (value_type >= 3 && value_type <= 5) {
    return sizeof(uint64_t);
  }
  return 0;
}

void check_reduce_member_request(const char *backend,
                                 Ndarray *values,
                                 Ndarray *output,
                                 int value_type,
                                 std::size_t offset,
                                 std::size_t stride,
                                 int op) {
  TI_ERROR_IF(!values || !output,
              "{} strided reduce received a null ndarray.", backend);
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "{} strided reduce expects 1D ndarrays.", backend);
  TI_ERROR_IF(values->get_nelement() == 0,
              "{} strided reduce expects at least one input item.", backend);
  TI_ERROR_IF(output->get_nelement() < 1,
              "{} strided reduce output must contain at least one item.",
              backend);
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "{} strided reduce received an unsupported value type.",
              backend);
  TI_ERROR_IF(output->get_element_size() != value_size,
              "{} strided reduce output dtype does not match value type.",
              backend);
  TI_ERROR_IF(op < 0 || op > 2,
              "{} strided reduce supports only sum/min/max operations.",
              backend);
  TI_ERROR_IF(stride < value_size,
              "{} strided reduce source stride is smaller than value size.",
              backend);
  TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
              "{} strided reduce source offset/stride must align to value "
              "size.",
              backend);
  const std::size_t n = values->get_nelement();
  const std::size_t src_bytes = n * values->get_element_size();
  TI_ERROR_IF(src_bytes < value_size,
              "{} strided reduce source buffer is smaller than value size.",
              backend);
  TI_ERROR_IF(offset > src_bytes - value_size,
              "{} strided reduce source offset is out of bounds.", backend);
  const std::size_t last = offset + (n - 1) * stride + value_size;
  TI_ERROR_IF(last > src_bytes,
              "{} strided reduce source range is out of bounds.", backend);
}

void check_reduce_strided_request(const char *backend,
                                  Ndarray *values,
                                  Ndarray *output,
                                  int value_type,
                                  std::size_t values_offset,
                                  std::size_t values_stride,
                                  std::size_t output_offset,
                                  std::size_t output_stride,
                                  int op) {
  TI_ERROR_IF(!values || !output,
              "{} strided reduce received a null ndarray.", backend);
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "{} strided reduce expects 1D ndarrays.", backend);
  TI_ERROR_IF(values->get_nelement() == 0,
              "{} strided reduce expects at least one input item.", backend);
  TI_ERROR_IF(output->get_nelement() < 1,
              "{} strided reduce output must contain at least one item.",
              backend);
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "{} strided reduce received an unsupported value type.",
              backend);
  TI_ERROR_IF(op < 0 || op > 2,
              "{} strided reduce supports only sum/min/max operations.",
              backend);
  auto check_range = [&](const char *role, Ndarray *arr,
                         std::size_t logical_items, std::size_t offset,
                         std::size_t stride) {
    TI_ERROR_IF(stride < value_size,
                "{} strided reduce {} stride is smaller than value size.",
                backend, role);
    TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
                "{} strided reduce {} offset/stride must align to value "
                "size.",
                backend, role);
    const std::size_t bytes = arr->get_nelement() * arr->get_element_size();
    TI_ERROR_IF(bytes < value_size,
                "{} strided reduce {} buffer is smaller than value size.",
                backend, role);
    TI_ERROR_IF(offset > bytes - value_size,
                "{} strided reduce {} offset is out of bounds.", backend,
                role);
    const std::size_t last =
        offset + (logical_items - 1) * stride + value_size;
    TI_ERROR_IF(last > bytes,
                "{} strided reduce {} range is out of bounds.", backend,
                role);
  };
  check_range("source", values, values->get_nelement(), values_offset,
              values_stride);
  check_range("destination", output, 1, output_offset, output_stride);
}

void check_scan_member_request(const char *backend,
                               Ndarray *data,
                               int value_type,
                               std::size_t offset,
                               std::size_t stride) {
  TI_ERROR_IF(!data, "{} strided scan received a null ndarray.", backend);
  TI_ERROR_IF(data->shape.size() != 1, "{} strided scan expects a 1D ndarray.",
              backend);
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "{} strided scan received an unsupported value type.", backend);
  TI_ERROR_IF(stride < value_size,
              "{} strided scan source stride is smaller than value size.",
              backend);
  TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
              "{} strided scan source offset/stride must align to value size.",
              backend);
  const std::size_t n = data->get_nelement();
  if (n == 0) {
    return;
  }
  const std::size_t src_bytes = n * data->get_element_size();
  TI_ERROR_IF(src_bytes < value_size,
              "{} strided scan source buffer is smaller than value size.",
              backend);
  TI_ERROR_IF(offset > src_bytes - value_size,
              "{} strided scan source offset is out of bounds.", backend);
  const std::size_t last = offset + (n - 1) * stride + value_size;
  TI_ERROR_IF(last > src_bytes,
              "{} strided scan source range is out of bounds.", backend);
}

void check_scatter_add_member_request(const char *backend,
                                      Ndarray *src,
                                      Ndarray *indices,
                                      Ndarray *dst,
                                      int value_type,
                                      std::size_t offset,
                                      std::size_t stride) {
  TI_ERROR_IF(!src || !indices || !dst,
              "{} strided scatter-add received a null ndarray.", backend);
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "{} strided scatter-add expects 1D ndarrays.", backend);
  TI_ERROR_IF(src->get_nelement() != indices->get_nelement(),
              "{} strided scatter-add source and indices sizes differ.",
              backend);
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "{} strided scatter-add received an unsupported value type.",
              backend);
  TI_ERROR_IF(dst->get_element_size() != value_size ||
                  indices->get_element_size() != sizeof(int32_t),
              "{} strided scatter-add destination dtype or i32 index size "
              "mismatch.",
              backend);
  TI_ERROR_IF(stride < value_size,
              "{} strided scatter-add source stride is smaller than value "
              "size.",
              backend);
  TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
              "{} strided scatter-add source offset/stride must align to "
              "value size.",
              backend);
  const std::size_t n = src->get_nelement();
  if (n == 0) {
    return;
  }
  const std::size_t src_bytes = n * src->get_element_size();
  TI_ERROR_IF(src_bytes < value_size,
              "{} strided scatter-add source buffer is smaller than value "
              "size.",
              backend);
  TI_ERROR_IF(offset > src_bytes - value_size,
              "{} strided scatter-add source offset is out of bounds.",
              backend);
  const std::size_t last = offset + (n - 1) * stride + value_size;
  TI_ERROR_IF(last > src_bytes,
              "{} strided scatter-add source range is out of bounds.",
              backend);
}

void check_grouped_reduce_member_request(const char *backend,
                                         Ndarray *keys,
                                         Ndarray *values,
                                         Ndarray *output,
                                         int value_type,
                                         std::size_t offset,
                                         std::size_t stride,
                                         int op) {
  TI_ERROR_IF(!keys || !values || !output,
              "{} strided grouped reduce received a null ndarray.", backend);
  TI_ERROR_IF(keys->shape.size() != 1 || values->shape.size() != 1 ||
                  output->shape.size() != 1,
              "{} strided grouped reduce expects 1D ndarrays.", backend);
  TI_ERROR_IF(keys->get_nelement() != values->get_nelement(),
              "{} strided grouped reduce keys and values sizes differ.",
              backend);
  TI_ERROR_IF(output->get_nelement() == 0,
              "{} strided grouped reduce output must contain at least one "
              "group.",
              backend);
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "{} strided grouped reduce received an unsupported value type.",
              backend);
  TI_ERROR_IF(keys->get_element_size() != sizeof(int32_t) ||
                  output->get_element_size() != value_size,
              "{} strided grouped reduce output dtype or i32 key size "
              "mismatch.",
              backend);
  TI_ERROR_IF(op != 0,
              "{} strided grouped reduce currently supports only sum.",
              backend);
  TI_ERROR_IF(stride < value_size,
              "{} strided grouped reduce source stride is smaller than value "
              "size.",
              backend);
  TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
              "{} strided grouped reduce source offset/stride must align to "
              "value size.",
              backend);
  const std::size_t n = values->get_nelement();
  if (n == 0) {
    return;
  }
  const std::size_t values_bytes = n * values->get_element_size();
  TI_ERROR_IF(values_bytes < value_size,
              "{} strided grouped reduce source buffer is smaller than value "
              "size.",
              backend);
  TI_ERROR_IF(offset > values_bytes - value_size,
              "{} strided grouped reduce source offset is out of bounds.",
              backend);
  const std::size_t last = offset + (n - 1) * stride + value_size;
  TI_ERROR_IF(last > values_bytes,
              "{} strided grouped reduce source range is out of bounds.",
              backend);
}

void check_strided_range(const char *backend,
                         const char *role,
                         Ndarray *arr,
                         std::size_t logical_items,
                         std::size_t value_size,
                         std::size_t offset,
                         std::size_t stride) {
  TI_ERROR_IF(stride < value_size,
              "{} strided {} stride is smaller than value size.", backend,
              role);
  TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
              "{} strided {} offset/stride must align to value size.", backend,
              role);
  if (logical_items == 0) {
    return;
  }
  const std::size_t bytes = arr->get_nelement() * arr->get_element_size();
  TI_ERROR_IF(bytes < value_size,
              "{} strided {} buffer is smaller than value size.", backend,
              role);
  TI_ERROR_IF(offset > bytes - value_size,
              "{} strided {} offset is out of bounds.", backend, role);
  const std::size_t last = offset + (logical_items - 1) * stride + value_size;
  TI_ERROR_IF(last > bytes, "{} strided {} range is out of bounds.", backend,
              role);
}

void check_scatter_add_strided_request(const char *backend,
                                       Ndarray *src,
                                       Ndarray *indices,
                                       Ndarray *dst,
                                       int value_type,
                                       std::size_t src_offset,
                                       std::size_t src_stride,
                                       std::size_t dst_offset,
                                       std::size_t dst_stride) {
  TI_ERROR_IF(!src || !indices || !dst,
              "{} strided scatter-add received a null ndarray.", backend);
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "{} strided scatter-add expects 1D ndarrays.", backend);
  TI_ERROR_IF(src->get_nelement() != indices->get_nelement(),
              "{} strided scatter-add source and indices sizes differ.",
              backend);
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "{} strided scatter-add received an unsupported value type.",
              backend);
  TI_ERROR_IF(indices->get_element_size() != sizeof(int32_t),
              "{} strided scatter-add expects i32 indices.", backend);
  const std::size_t n = src->get_nelement();
  const std::size_t dst_items = dst->get_nelement();
  check_strided_range(backend, "scatter-add source", src, n, value_size,
                      src_offset, src_stride);
  check_strided_range(backend, "scatter-add destination", dst, dst_items,
                      value_size, dst_offset, dst_stride);
}

void check_grouped_reduce_strided_keys_request(const char *backend,
                                               Ndarray *keys,
                                               Ndarray *values,
                                               Ndarray *output,
                                               int value_type,
                                               std::size_t keys_offset,
                                               std::size_t keys_stride,
                                               std::size_t values_offset,
                                               std::size_t values_stride,
                                               std::size_t output_offset,
                                               std::size_t output_stride,
                                               int op) {
  TI_ERROR_IF(!keys || !values || !output,
              "{} strided grouped reduce received a null ndarray.", backend);
  TI_ERROR_IF(keys->shape.size() != 1 || values->shape.size() != 1 ||
                  output->shape.size() != 1,
              "{} strided grouped reduce expects 1D ndarrays.", backend);
  TI_ERROR_IF(keys->get_nelement() != values->get_nelement(),
              "{} strided grouped reduce keys and values sizes differ.",
              backend);
  TI_ERROR_IF(output->get_nelement() == 0,
              "{} strided grouped reduce output must contain at least one "
              "group.",
              backend);
  const std::size_t value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "{} strided grouped reduce received an unsupported value type.",
              backend);
  TI_ERROR_IF(op != 0,
              "{} strided grouped reduce currently supports only sum.",
              backend);
  check_strided_range(backend, "grouped reduce keys", keys,
                      keys->get_nelement(), sizeof(int32_t), keys_offset,
                      keys_stride);
  check_strided_range(backend, "grouped reduce source", values,
                      values->get_nelement(), value_size, values_offset,
                      values_stride);
  check_strided_range(backend, "grouped reduce output", output,
                      output->get_nelement(), value_size, output_offset,
                      output_stride);
}

void check_grouped_reduce_strided_request(const char *backend,
                                          Ndarray *keys,
                                          Ndarray *values,
                                          Ndarray *output,
                                          int value_type,
                                          std::size_t values_offset,
                                          std::size_t values_stride,
                                          std::size_t output_offset,
                                          std::size_t output_stride,
                                          int op) {
  TI_ERROR_IF(keys && keys->get_element_size() != sizeof(int32_t),
              "{} strided grouped reduce expects i32 keys.", backend);
  check_grouped_reduce_strided_keys_request(
      backend, keys, values, output, value_type, 0, sizeof(int32_t),
      values_offset, values_stride, output_offset, output_stride, op);
}

std::size_t histogram_bin_type_size(int bin_type) {
  if (bin_type == 0) {
    return sizeof(int32_t);
  }
  if (bin_type == 4) {
    return sizeof(int64_t);
  }
  return 0;
}

template <typename ValueT>
bool cpu_histogram_valid_bin(ValueT bin, std::size_t num_bins) {
  if constexpr (std::is_unsigned_v<ValueT>) {
    return static_cast<std::size_t>(bin) < num_bins;
  } else {
    return bin >= 0 && static_cast<std::size_t>(bin) < num_bins;
  }
}

template <typename ValueT, typename CounterT>
void cpu_histogram_task(void *raw_ctx, int /*thread_id*/, int task_id) {
  auto *ctx = static_cast<CpuHistogramTaskContext<ValueT, CounterT> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      ctx->n * static_cast<std::size_t>(tid) /
      static_cast<std::size_t>(ctx->num_threads);
  const std::size_t end =
      ctx->n * static_cast<std::size_t>(tid + 1) /
      static_cast<std::size_t>(ctx->num_threads);
  CounterT *local =
      ctx->partial + static_cast<std::size_t>(tid) * ctx->num_bins;
  for (std::size_t i = begin; i < end; ++i) {
    ValueT bin = ctx->values[i];
    if (cpu_histogram_valid_bin(bin, ctx->num_bins)) {
      local[static_cast<std::size_t>(bin)] += 1;
    }
  }
}

template <typename ValueT, typename CounterT>
std::size_t cpu_histogram_typed(const ValueT *values_ptr,
                                CounterT *bins_ptr,
                                std::size_t n,
                                std::size_t num_bins,
                                int max_threads,
                                int target_threads,
                                bool use_parallel) {
  TI_ERROR_IF(!values_ptr || !bins_ptr,
              "CPU native histogram received a null data pointer.");
  std::fill(bins_ptr, bins_ptr + num_bins, CounterT{});

  if (use_parallel) {
    const int num_threads = target_threads;
    std::vector<CounterT> partial(
        static_cast<std::size_t>(num_threads) * num_bins, CounterT{});
    CpuHistogramTaskContext<ValueT, CounterT> ctx;
    ctx.values = values_ptr;
    ctx.partial = partial.data();
    ctx.n = n;
    ctx.num_bins = num_bins;
    ctx.num_threads = num_threads;
    auto &pool = get_cpu_primitive_thread_pool(max_threads);
    pool.run(num_threads, num_threads, &ctx,
             cpu_histogram_task<ValueT, CounterT>);
    for (std::size_t bin = 0; bin < num_bins; ++bin) {
      CounterT total{};
      for (int tid = 0; tid < num_threads; ++tid) {
        total += partial[static_cast<std::size_t>(tid) * num_bins + bin];
      }
      bins_ptr[bin] = total;
    }
    return partial.size() * sizeof(CounterT);
  }

  for (std::size_t i = 0; i < n; ++i) {
    ValueT bin = values_ptr[i];
    if (cpu_histogram_valid_bin(bin, num_bins)) {
      bins_ptr[static_cast<std::size_t>(bin)] += 1;
    }
  }
  return 0;
}

template <typename T>
std::size_t cpu_reduce_typed(T *values_ptr,
                             T *output_ptr,
                             int op,
                             std::size_t n,
                             int max_threads,
                             int target_threads,
                             bool use_parallel) {
  TI_ERROR_IF(!values_ptr || !output_ptr,
              "CPU native reduce received a null data pointer.");

  T result = cpu_reduce_identity<T>(op);
  if (use_parallel) {
    const int num_threads = target_threads;
    std::vector<T> partial(num_threads);
    CpuReduceTaskContext<T> ctx;
    ctx.values = values_ptr;
    ctx.partial = partial.data();
    ctx.n = n;
    ctx.num_threads = num_threads;
    ctx.op = op;
    auto &pool = get_cpu_primitive_thread_pool(max_threads);
    pool.run(num_threads, num_threads, &ctx, cpu_reduce_task<T>);
    for (int tid = 0; tid < num_threads; ++tid) {
      result = cpu_reduce_combine(result, partial[tid], op);
    }
    output_ptr[0] = result;
    return partial.size() * sizeof(T);
  }

  for (std::size_t i = 0; i < n; ++i) {
    result = cpu_reduce_combine(result, values_ptr[i], op);
  }
  output_ptr[0] = result;
  return 0;
}

template <typename T>
std::size_t cpu_reduce_strided_typed(const uint8_t *values_ptr,
                                     T *output_ptr,
                                     int op,
                                     std::size_t n,
                                     std::size_t offset,
                                     std::size_t stride,
                                     int max_threads,
                                     int target_threads,
                                     bool use_parallel) {
  TI_ERROR_IF(!values_ptr || !output_ptr,
              "CPU native strided reduce received a null data pointer.");

  T result = cpu_reduce_identity<T>(op);
  if (use_parallel) {
    const int num_threads = target_threads;
    std::vector<T> partial(num_threads);
    CpuStridedReduceTaskContext<T> ctx;
    ctx.values = values_ptr;
    ctx.partial = partial.data();
    ctx.n = n;
    ctx.offset = offset;
    ctx.stride = stride;
    ctx.num_threads = num_threads;
    ctx.op = op;
    auto &pool = get_cpu_primitive_thread_pool(max_threads);
    pool.run(num_threads, num_threads, &ctx, cpu_strided_reduce_task<T>);
    for (int tid = 0; tid < num_threads; ++tid) {
      result = cpu_reduce_combine(result, partial[tid], op);
    }
    output_ptr[0] = result;
    return partial.size() * sizeof(T);
  }

  for (std::size_t i = 0; i < n; ++i) {
    const auto *value =
        reinterpret_cast<const T *>(values_ptr + offset + i * stride);
    result = cpu_reduce_combine(result, *value, op);
  }
  output_ptr[0] = result;
  return 0;
}

template <typename T>
std::size_t cpu_scan_strided_typed(uint8_t *data_ptr,
                                   std::size_t n,
                                   std::size_t offset,
                                   std::size_t stride) {
  TI_ERROR_IF(!data_ptr, "CPU native strided scan received a null data pointer.");
  T prefix{};
  for (std::size_t i = 0; i < n; ++i) {
    auto *value = reinterpret_cast<T *>(data_ptr + offset + i * stride);
    prefix += *value;
    *value = prefix;
  }
  return 0;
}

bool snode_tree_contains_hash(const SNode *snode) {
  if (snode == nullptr) {
    return false;
  }
  if (snode->type == SNodeType::hash) {
    return true;
  }
  for (const auto &child : snode->ch) {
    if (snode_tree_contains_hash(child.get())) {
      return true;
    }
  }
  return false;
}
}  // namespace

Program::Program(Arch desired_arch) : snode_rw_accessors_bank_(this) {
  TI_TRACE("Program initializing...");

  // For performance considerations and correctness of QuantFloatType
  // operations, we force floating-point operations to flush to zero on all
  // backends (including CPUs).
#if defined(_M_X64) || defined(__x86_64)
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
#endif  // defined(_M_X64) || defined(__x86_64)
#if defined(__arm64__) || defined(__aarch64__)
  // Enforce flush to zero on arm64 CPUs
  // https://developer.arm.com/documentation/100403/0201/register-descriptions/advanced-simd-and-floating-point-registers/aarch64-register-descriptions/fpcr--floating-point-control-register?lang=en
  std::uint64_t fpcr;
  __asm__ __volatile__("");
  __asm__ __volatile__("MRS %0, FPCR" : "=r"(fpcr));
  __asm__ __volatile__("");
  __asm__ __volatile__("MSR FPCR, %0"
                       :
                       : "ri"(fpcr | (1 << 24)));  // Bit 24 is FZ
  __asm__ __volatile__("");
#endif  // defined(__arm64__) || defined(__aarch64__)
  auto &config = compile_config_;
  config = default_compile_config;
  config.arch = desired_arch;
  config.fit();

  profiler = make_profiler(config.arch, config.kernel_profiler);
  if (arch_uses_llvm(config.arch)) {
#ifdef TI_WITH_LLVM
    if (config.arch != Arch::dx12) {
      program_impl_ = std::make_unique<LlvmProgramImpl>(config, profiler.get());
    } else {
      // NOTE: use Dx12ProgramImpl to avoid using LlvmRuntimeExecutor for dx12.
#ifdef TI_WITH_DX12
      TI_ASSERT(directx12::is_dx12_api_available());
      program_impl_ = std::make_unique<Dx12ProgramImpl>(config);
#else
      TI_ERROR("This taichi is not compiled with DX12");
#endif
    }
#else
    TI_ERROR("This taichi is not compiled with LLVM");
#endif
  } else if (config.arch == Arch::metal) {
#ifdef TI_WITH_METAL
    TI_ASSERT(metal::is_metal_api_available());
    program_impl_ = std::make_unique<MetalProgramImpl>(config);
#else
    TI_ERROR("This taichi is not compiled with Metal")
#endif
  } else if (config.arch == Arch::vulkan) {
#ifdef TI_WITH_VULKAN
    TI_ASSERT(vulkan::is_vulkan_api_available());
    program_impl_ = std::make_unique<VulkanProgramImpl>(config);
#else
    TI_ERROR("This taichi is not compiled with Vulkan")
#endif
  } else if (config.arch == Arch::dx11) {
#ifdef TI_WITH_DX11
    TI_ASSERT(directx11::is_dx_api_available());
    program_impl_ = std::make_unique<Dx11ProgramImpl>(config);
#else
    TI_ERROR("This taichi is not compiled with DX11");
#endif
  } else if (config.arch == Arch::opengl) {
#ifdef TI_WITH_OPENGL
    TI_ASSERT(opengl::initialize_opengl(false));
    program_impl_ = std::make_unique<OpenglProgramImpl>(config);
#else
    TI_ERROR("This taichi is not compiled with OpenGL");
#endif
  } else if (config.arch == Arch::gles) {
#ifdef TI_WITH_OPENGL
    TI_ASSERT(opengl::initialize_opengl(true));
    program_impl_ = std::make_unique<OpenglProgramImpl>(config);
#else
    TI_ERROR("This taichi is not compiled with OpenGL");
#endif
  } else {
    TI_NOT_IMPLEMENTED
  }

  // program_impl_ should be set in the if-else branch above
  TI_ASSERT(program_impl_);

  // Phase 1c-D: propagate the user's vulkan_sparse_experimental opt-in to
  // the process-global extension table BEFORE any is_extension_supported()
  // query (the very next block already calls it for Extension::assertion).
  // Sticky once true; OR'd with the legacy TI_VULKAN_SPARSE env var.
  set_vulkan_sparse_experimental(config.vulkan_sparse_experimental);
  // §13 (2026-05-02): default for vulkan_sparse_experimental is now true.
  // Emit a one-shot informational warning whenever the experimental sparse
  // path is exercised on Arch::vulkan, so users can correlate any unexpected
  // behaviour with this opt-in path. Skipped on cpu/cuda (which never read
  // the flag) and skipped when the user explicitly disables it.
  if (config.arch == Arch::vulkan && config.vulkan_sparse_experimental) {
    static bool sparse_warn_emitted = false;
    if (!sparse_warn_emitted) {
      sparse_warn_emitted = true;
      TI_WARN(
          "Vulkan sparse SNode support is experimental and enabled by default "
          "as of taichi-forge 0.3.x; pass ti.init(vulkan_sparse_experimental="
          "False) to disable if you observe regressions vs. cuda/cpu.");
    }
  }
  // G9.1: same propagation pattern for quant_array / bit_struct on Vulkan.
  set_vulkan_quant_experimental(config.vulkan_quant_experimental);

  Device *compute_device = nullptr;
  compute_device = program_impl_->get_compute_device();
  // Must have handled all the arch fallback logic by this point.
  TI_ASSERT_INFO(num_instances_ == 0, "Only one instance at a time");
  total_compilation_time_ = 0;
  num_instances_ += 1;
  SNode::counter = 0;

  result_buffer = nullptr;
  finalized_ = false;

  if (!is_extension_supported(config.arch, Extension::assertion)) {
    if (config.check_out_of_bound) {
      TI_WARN("Out-of-bound access checking is not supported on arch={}",
              arch_name(config.arch));
      config.check_out_of_bound = false;
    }
  }

  Timelines::get_instance().set_enabled(config.timeline);

  TI_TRACE("Program ({}) arch={} initialized.", fmt::ptr(this),
           arch_name(config.arch));
}

TypeFactory &Program::get_type_factory() {
  TI_WARN(
      "Program::get_type_factory() will be deprecated, Please use "
      "TypeFactory::get_instance()");
  return TypeFactory::get_instance();
}

Function *Program::create_function(const FunctionKey &func_key) {
  TI_TRACE("Creating function {}...", func_key.get_full_name());
  functions_.emplace_back(std::make_unique<Function>(this, func_key));
  TI_ASSERT(function_map_.count(func_key) == 0);
  function_map_[func_key] = functions_.back().get();
  return functions_.back().get();
}

const CompiledKernelData &Program::compile_kernel(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &caps,
    const Kernel &kernel_def) {
  auto start_t = Time::get_time();
  TI_AUTO_PROF;
  auto &mgr = program_impl_->get_kernel_compilation_manager();
  // P-Compile-6: apply per-kernel compile_tier override (if set) by passing
  // an effective CompileConfig copy down. CompileConfig::compile_tier is
  // already part of the offline cache key (offline_cache_util.cpp), so cache
  // entries for the same kernel under different tiers are automatically
  // segregated.
  const auto &override = kernel_def.get_compile_tier_override();
  if (override.has_value() && *override != compile_config.compile_tier) {
    CompileConfig effective_config = compile_config;
    effective_config.compile_tier = *override;
    const auto &ckd = mgr.load_or_compile(effective_config, caps, kernel_def);
    total_compilation_time_ += Time::get_time() - start_t;
    return ckd;
  }
  const auto &ckd = mgr.load_or_compile(compile_config, caps, kernel_def);
  total_compilation_time_ += Time::get_time() - start_t;
  return ckd;
}

// P5.b — batch / parallel kernel compilation.
//
// Design:
// 1. Compilation is dispatched to a ParallelExecutor with
//    `compile_config.num_compile_threads` workers.
// 2. All heavy lifting (IR passes, LLVM opt, SPIR-V codegen, GPU module load)
//    runs on worker threads. The main thread only submits + flushes.
// 3. Ordering: kernel compilation is order-independent at the C++ level —
//    @ti.func inlining and template specialization are resolved in Python
//    before a `Kernel` object ever reaches C++. Each kernel is compiled as a
//    self-contained unit.
// 4. Thread-safety:
//    - `KernelCompilationManager::load_or_compile` is guarded by its own
//      cache_mutex_ (P5.a) so concurrent cache hits/inserts are safe.
//    - LLVM: TaichiLLVMContext maintains per-thread_id state under
//      thread_map_mut_; first-touch on a worker lazily clones the runtime
//      module + struct_modules from the main thread (which is already
//      quiescent after materialize_runtime).
//    - CUDA: `cuModuleLoadDataEx` is serialized by CUDAContext::get_lock_guard
//      inside JITSessionCUDA; all optimization runs in parallel.
//    - Vulkan: SPIR-V codegen touches no shared state.
// 5. Error propagation: the first exception from any worker is captured and
//    rethrown on the calling thread after flush(); remaining workers still
//    finish their in-flight tasks so we never leave the executor in a bad
//    state.
//
// V7 (2026-04-26) — thread-local depth counter that tracks whether the
// current thread is acting as a compile_kernels outer worker. The LLVM
// codegen path consults this via Program::in_compile_kernels_worker() to
// avoid double-pool oversubscription. Only incremented when
// compile_config.compile_dag_scheduler is true.
namespace {
thread_local int g_compile_kernels_worker_depth = 0;
}  // namespace

bool Program::in_compile_kernels_worker() {
  return g_compile_kernels_worker_depth > 0;
}

// Caller contract: do NOT destroy SNode trees concurrently with this call.
void Program::compile_kernels(
    const CompileConfig &compile_config,
    const std::vector<const Kernel *> &kernels) {
  if (kernels.empty()) {
    return;
  }
  auto start_t = Time::get_time();
  const auto caps = get_device_caps();

  const int n_compile_threads =
      std::max(1, compile_config.num_compile_threads);
  int n_workers = std::min<int>(n_compile_threads, (int)kernels.size());

  auto &mgr = program_impl_->get_kernel_compilation_manager();
  const bool dag_mode = compile_config.compile_dag_scheduler;
  // V8.a (2026-04-26): when dag_mode is on and there are fewer kernels than
  // compile threads, skip the outer ParallelExecutor entirely. The serial
  // outer loop lets each kernel's inner offload pool (LLVM
  // compilation_workers / SPIR-V V2 std::async) consume the full T-wide
  // worker budget on its own. With V7 enabled the previous behaviour would
  // create only N outer workers and force inner-serial, leaving (T-N) cores
  // idle. See compile_doc/优化总规划.md §3.5.
  const bool prefer_inner_parallelism =
      dag_mode && (int)kernels.size() < n_compile_threads;
  if (n_workers <= 1 || prefer_inner_parallelism) {
    // Fast path: honour the same serial path as compile_kernel.
    for (auto *k : kernels) {
      mgr.load_or_compile(compile_config, caps, *k);
    }
    total_compilation_time_ += Time::get_time() - start_t;
    return;
  }

  std::mutex err_mu;
  std::exception_ptr first_error;

  {
    ParallelExecutor exec("compile_kernels", n_workers);
    for (auto *k : kernels) {
      exec.enqueue([&, k]() {
        // V7: mark this worker so the LLVM inner pool stays serial.
        if (dag_mode) {
          ++g_compile_kernels_worker_depth;
        }
        try {
          mgr.load_or_compile(compile_config, caps, *k);
        } catch (...) {
          std::lock_guard<std::mutex> g(err_mu);
          if (!first_error) {
            first_error = std::current_exception();
          }
        }
        if (dag_mode) {
          --g_compile_kernels_worker_depth;
        }
      });
    }
    // ~ParallelExecutor runs flush() implicitly via its destructor.
    exec.flush();
  }

  total_compilation_time_ += Time::get_time() - start_t;
  if (first_error) {
    std::rethrow_exception(first_error);
  }
}

void Program::launch_kernel(const CompiledKernelData &compiled_kernel_data,
                            LaunchContextBuilder &ctx) {
  program_impl_->get_kernel_launcher().launch_kernel(compiled_kernel_data, ctx);
  const bool check_runtime_error =
      compile_config().debug || hash_snode_tree_count_ > 0;
  if (check_runtime_error && arch_uses_llvm(compiled_kernel_data.arch())) {
    program_impl_->check_runtime_error(result_buffer);
  }
}

void Program::materialize_runtime() {
  program_impl_->materialize_runtime(profiler.get(), &result_buffer);
}

static void remove_rw_accessor_cache(
    SNode *parent_snode,
    SNodeRwAccessorsBank *snode_rw_accessors_bank) {
  for (int i = 0; i < (int)parent_snode->ch.size(); i++) {
    auto child_snode = parent_snode->ch[i].get();
    if (child_snode->type == SNodeType::place) {
      snode_rw_accessors_bank->remove_cached_kernels(child_snode);
    }
    remove_rw_accessor_cache(child_snode, snode_rw_accessors_bank);
  }
}

void Program::destroy_snode_tree(SNodeTree *snode_tree) {
  TI_ASSERT(arch_uses_llvm(compile_config().arch) ||
            compile_config().arch == Arch::vulkan ||
            compile_config().arch == Arch::dx11 ||
            compile_config().arch == Arch::dx12);

  // When accessing a ti.field at Python scope, SNodeRwAccessorsBank creates
  // a Taichi Kernel to read/write the field in a JIT manner, which caches the
  // compiled JIT Kernel so as to avoid recompilation when accessing the same
  // field.

  // This cache uses the place-SNode's address (SNode*) as the key,
  // which becomes unsafe once the SNodeTree gets destroyed and that
  // place-SNode's address gets reused by another SNode. We have to remove all
  // cached kernels upon SNodeTree destruction.
  SNode *root = snode_tree->root();
  const bool contains_hash = snode_tree_contains_hash(root);

  // Traverse SNodeTree to remove all cached RWAccessor kernels
  remove_rw_accessor_cache(root, &snode_rw_accessors_bank_);

  program_impl_->destroy_snode_tree(snode_tree);
  if (contains_hash) {
    --hash_snode_tree_count_;
  }
  free_snode_tree_ids_.push(snode_tree->id());
}

SNodeTree *Program::add_snode_tree(std::unique_ptr<SNode> root,
                                   bool compile_only) {
  const int id = allocate_snode_tree_id();
  auto tree = std::make_unique<SNodeTree>(id, std::move(root));
  tree->root()->set_snode_tree_id(id);
  const bool contains_hash = snode_tree_contains_hash(tree->root());
  if (compile_only) {
    program_impl_->compile_snode_tree_types(tree.get());
  } else {
    program_impl_->materialize_snode_tree(tree.get(), result_buffer);
  }
  if (contains_hash) {
    ++hash_snode_tree_count_;
  }
  if (id < snode_trees_.size()) {
    snode_trees_[id] = std::move(tree);
  } else {
    TI_ASSERT(id == snode_trees_.size());
    snode_trees_.push_back(std::move(tree));
  }
  return snode_trees_[id].get();
}

SNode *Program::get_snode_root(int tree_id) {
  return snode_trees_[tree_id]->root();
}

void Program::synchronize() {
  program_impl_->synchronize();
}

StreamSemaphore Program::flush() {
  return program_impl_->flush();
}

int Program::get_snode_tree_size() {
  return snode_trees_.size();
}

Kernel &Program::get_snode_reader(SNode *snode) {
  TI_ASSERT(snode->type == SNodeType::place);
  auto kernel_name = fmt::format("snode_reader_{}", snode->id);
  auto &ker = kernel([snode, this](Kernel *kernel) {
    ExprGroup indices;
    for (int i = 0; i < snode->num_active_indices; i++) {
      auto argload_expr = Expr::make<ArgLoadExpression>(std::vector<int>{i},
                                                        PrimitiveType::i32);
      argload_expr->type_check(&this->compile_config());
      indices.push_back(std::move(argload_expr));
    }
    ASTBuilder &builder = kernel->context->builder();
    auto ret = Stmt::make<FrontendReturnStmt>(ExprGroup(
        builder.expr_subscript(Expr(snode_to_fields_.at(snode)), indices)));
    builder.insert(std::move(ret));
  });
  ker.name = kernel_name;
  ker.is_accessor = true;
  for (int i = 0; i < snode->num_active_indices; i++)
    ker.insert_scalar_param(PrimitiveType::i32);
  ker.insert_ret(snode->dt);
  ker.finalize_params();
  ker.finalize_rets();
  return ker;
}

Kernel &Program::get_snode_writer(SNode *snode) {
  TI_ASSERT(snode->type == SNodeType::place);
  auto kernel_name = fmt::format("snode_writer_{}", snode->id);
  auto &ker = kernel([snode, this](Kernel *kernel) {
    ExprGroup indices;
    for (int i = 0; i < snode->num_active_indices; i++) {
      auto argload_expr = Expr::make<ArgLoadExpression>(std::vector<int>{i},
                                                        PrimitiveType::i32);
      argload_expr->type_check(&this->compile_config());
      indices.push_back(std::move(argload_expr));
    }
    ASTBuilder &builder = kernel->context->builder();
    auto expr =
        builder.expr_subscript(Expr(snode_to_fields_.at(snode)), indices);
    expr.type_check(&this->compile_config());
    auto argload_expr = Expr::make<ArgLoadExpression>(
        std::vector<int>{snode->num_active_indices},
        snode->dt->get_compute_type());
    argload_expr->type_check(&this->compile_config());
    builder.insert_assignment(expr, argload_expr, expr->dbg_info);
  });
  ker.name = kernel_name;
  ker.is_accessor = true;
  for (int i = 0; i < snode->num_active_indices; i++)
    ker.insert_scalar_param(PrimitiveType::i32);
  ker.insert_scalar_param(snode->dt);
  ker.finalize_params();
  ker.finalize_rets();
  return ker;
}

uint64 Program::fetch_result_uint64(int i) {
  return program_impl_->fetch_result_uint64(i, result_buffer);
}

void Program::finalize() {
  if (finalized_) {
    return;
  }

  synchronize();
  TI_TRACE("Program finalizing...");

  synchronize();
  if (compile_config().arch == Arch::vulkan) {
    vulkan_radix_sort_clear_workspace();
    vulkan_scan_clear_workspace();
    vulkan_compact_clear_workspace();
    vulkan_histogram_clear_workspace();
    vulkan_reduce_clear_workspace();
    vulkan_transform_clear_workspace();
    vulkan_indexed_copy_clear_workspace();
    vulkan_bucket_builder_clear_workspace();
  }
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    cuda::cub_bucket_builder_clear_cache(this);
    cuda::cub_grouped_reduce_clear_cache(this);
  }
#endif
  textures_.clear();
  argpacks_.clear();
  ndarrays_.clear();
  if (arch_uses_llvm(compile_config().arch) ||
      compile_config().arch == Arch::vulkan) {
    program_impl_->finalize();
  }

  Stmt::reset_counter();

  finalized_ = true;
  num_instances_ -= 1;
  program_impl_->dump_cache_data_to_disk();
  compile_config_ = default_compile_config;
  TI_TRACE("Program ({}) finalized_.", fmt::ptr(this));

  // Reset memory pool
  HostMemoryPool::get_instance().reset();
}

int Program::default_block_dim(const CompileConfig &config) {
  if (arch_is_cpu(config.arch)) {
    return config.default_cpu_block_dim;
  } else {
    return config.default_gpu_block_dim;
  }
}

void Program::print_memory_profiler_info() {
  program_impl_->print_memory_profiler_info(snode_trees_, result_buffer);
}

std::size_t Program::get_snode_num_dynamically_allocated(SNode *snode) {
  return program_impl_->get_snode_num_dynamically_allocated(snode,
                                                            result_buffer);
}

void Program::reset_hash_snode_probe_stats() {
  program_impl_->reset_hash_snode_probe_stats(result_buffer);
}

std::vector<int64> Program::get_hash_snode_probe_stats() {
  return program_impl_->get_hash_snode_probe_stats(result_buffer);
}

Ndarray *Program::create_ndarray(const DataType type,
                                 const std::vector<int> &shape,
                                 ExternalArrayLayout layout,
                                 bool zero_fill,
                                 const DebugInfo &dbg_info) {
  auto arr = std::make_unique<Ndarray>(this, type, shape, layout, dbg_info);
  if (zero_fill) {
    Arch arch = compile_config().arch;
    if (arch_is_cpu(arch) || arch == Arch::cuda || arch == Arch::amdgpu) {
      fill_ndarray_fast_u32(arr.get(), /*data=*/0);
    } else if (arch != Arch::dx12) {
      // Device api support for dx12 backend are not complete yet
      Stream *stream =
          program_impl_->get_compute_device()->get_compute_stream();
      auto [cmdlist, res] = stream->new_command_list_unique();
      TI_ASSERT(res == RhiResult::success);
      cmdlist->buffer_fill(arr->ndarray_alloc_.get_ptr(0),
                           arr->get_element_size() * arr->get_nelement(),
                           /*data=*/0);
      stream->submit_synced(cmdlist.get());
    }
  }
  auto arr_ptr = arr.get();
  ndarrays_.insert({arr_ptr, std::move(arr)});
  return arr_ptr;
}

ArgPack *Program::create_argpack(const DataType dt) {
  auto pack = std::make_unique<ArgPack>(this, dt);
  auto pack_ptr = pack.get();
  argpacks_.insert({pack_ptr, std::move(pack)});
  return pack_ptr;
}

void Program::delete_ndarray(Ndarray *ndarray) {
  // [Note] Ndarray memory deallocation
  // Ndarray's memory allocation is managed by Taichi and Python can control
  // this via Taichi indirectly. For example, when an ndarray is GC-ed in
  // Python, it signals Taichi to free its memory allocation. But Taichi will
  // make sure **no pending kernels to be executed needs the ndarray** before it
  // actually frees the memory. When `ti.reset()` is called, all ndarrays
  // allocated in this program should be gone and no longer valid in Python.
  // This isn't the best implementation, ndarrays should be managed by taichi
  // runtime instead of this giant program and it should be freed when:
  // - Python GC signals taichi that it's no longer useful
  // - All kernels using it are executed.
  if (ndarrays_.count(ndarray) &&
      !program_impl_->used_in_kernel(ndarray->ndarray_alloc_.alloc_id)) {
    ndarrays_.erase(ndarray);
  }
}

void Program::delete_argpack(ArgPack *argpack) {
  // [Note] Argpack memory deallocation
  // Argpack's memory allocation is managed by Taichi and Python can control
  // this via Taichi indirectly. For example, when an argpack is GC-ed in
  // Python, it signals Taichi to free its memory allocation. But Taichi will
  // make sure **no pending kernels to be executed needs the argpack** before it
  // actually frees the memory. When `ti.reset()` is called, all argpack
  // allocated in this program should be gone and no longer valid in Python.
  // This isn't the best implementation, argpacks should be managed by taichi
  // runtime instead of this giant program and it should be freed when:
  // - Python GC signals taichi that it's no longer useful
  // - All kernels using it are executed.
  if (argpacks_.count(argpack) &&
      !program_impl_->used_in_kernel(argpack->argpack_alloc_.alloc_id)) {
    argpacks_.erase(argpack);
  }
}

Texture *Program::create_texture(BufferFormat buffer_format,
                                 const std::vector<int> &shape) {
  if (shape.size() == 1) {
    textures_.push_back(
        std::make_unique<Texture>(this, buffer_format, shape[0], 1, 1));
  } else if (shape.size() == 2) {
    textures_.push_back(
        std::make_unique<Texture>(this, buffer_format, shape[0], shape[1], 1));
  } else if (shape.size() == 3) {
    textures_.push_back(std::make_unique<Texture>(this, buffer_format, shape[0],
                                                  shape[1], shape[2]));
  } else {
    TI_ERROR("Texture shape invalid");
  }
  return textures_.back().get();
}

intptr_t Program::get_ndarray_data_ptr_as_int(const Ndarray *ndarray) {
  uint64_t *data_ptr{nullptr};
  if (arch_is_cpu(compile_config().arch) ||
      compile_config().arch == Arch::cuda ||
      compile_config().arch == Arch::amdgpu) {
    // For the LLVM backends, device allocation is a physical pointer.
    data_ptr =
        program_impl_->get_device_alloc_info_ptr(ndarray->ndarray_alloc_);
  }

  return reinterpret_cast<intptr_t>(data_ptr);
}

void Program::fill_ndarray_fast_u32(Ndarray *ndarray, uint32_t val) {
  TI_ERROR_IF(!ndarray, "fill_ndarray_fast_u32 received a null ndarray.");
  const std::size_t bytes =
      ndarray->get_nelement() * ndarray->get_element_size();
  if (bytes == 0) {
    return;
  }
  if (compile_config().arch == Arch::vulkan) {
    const DeviceAllocation alloc = ndarray->ndarray_alloc_;
    enqueue_compute_op_lambda(
        [alloc, bytes, val](Device * /*device*/, CommandList *cmdlist) {
          cmdlist->buffer_fill(alloc.get_ptr(0), bytes, val);
          cmdlist->buffer_barrier(alloc);
        },
        {});
    return;
  }
  if (compile_config().arch == Arch::cuda && val == 0 &&
      bytes % sizeof(uint32_t) != 0) {
#ifdef TI_WITH_CUDA
    auto *raw_ptr =
        program_impl_->get_device_alloc_info_ptr(ndarray->ndarray_alloc_);
    TI_ERROR_IF(!raw_ptr, "CUDA ndarray fill received a null data pointer.");
    CUDADriver::get_instance().memset(reinterpret_cast<void *>(raw_ptr), 0,
                                      bytes);
    return;
#else
    TI_NOT_IMPLEMENTED;
#endif
  }
  if (arch_is_cpu(compile_config().arch)) {
    auto *raw_ptr =
        program_impl_->get_device_alloc_info_ptr(ndarray->ndarray_alloc_);
    TI_ERROR_IF(!raw_ptr, "CPU ndarray fill received a null data pointer.");
    if (val == 0) {
      std::memset(raw_ptr, 0, bytes);
      return;
    }
    const std::size_t words = bytes / sizeof(uint32_t);
    auto *ptr = reinterpret_cast<uint32_t *>(raw_ptr);
    TI_ERROR_IF(!ptr, "CPU ndarray fill received a null data pointer.");
    const int max_threads =
        std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
    const int chunk_items = 32768;
    const int target_threads = static_cast<int>(
        std::min<std::size_t>((words + chunk_items - 1) / chunk_items,
                              static_cast<std::size_t>(max_threads)));
    if (words >= 65536 && target_threads > 1) {
      CpuFillU32TaskContext ctx;
      ctx.data = ptr;
      ctx.words = words;
      ctx.value = val;
      ctx.num_threads = target_threads;
      auto &pool = get_cpu_primitive_thread_pool(max_threads);
      pool.run(target_threads, target_threads, &ctx, cpu_fill_u32_task);
      return;
    }
    std::fill(ptr, ptr + words, val);
    return;
  }
  // This is a temporary solution to bypass device api on LLVM backends.
  program_impl_->fill_ndarray(
      ndarray->ndarray_alloc_, bytes / sizeof(uint32_t), val);
}

void Program::copy_ndarray_fast(Ndarray *dst, Ndarray *src) {
  TI_ERROR_IF(!dst || !src, "copy_ndarray_fast received a null ndarray.");
  const std::size_t dst_bytes = dst->get_nelement() * dst->get_element_size();
  const std::size_t src_bytes = src->get_nelement() * src->get_element_size();
  TI_ERROR_IF(dst_bytes != src_bytes,
              "copy_ndarray_fast requires source and destination to have the "
              "same byte size.");
  if (dst_bytes == 0 || dst == src) {
    return;
  }

  if (compile_config().arch == Arch::vulkan) {
    const DeviceAllocation dst_alloc = dst->ndarray_alloc_;
    const DeviceAllocation src_alloc = src->ndarray_alloc_;
    enqueue_compute_op_lambda(
        [dst_alloc, src_alloc, dst_bytes](Device * /*device*/,
                                          CommandList *cmdlist) {
          cmdlist->buffer_copy(dst_alloc.get_ptr(0), src_alloc.get_ptr(0),
                               dst_bytes);
          cmdlist->buffer_barrier(dst_alloc);
        },
        {});
    return;
  }

  if (arch_is_cpu(compile_config().arch)) {
    auto *dst_ptr = reinterpret_cast<uint8_t *>(
        program_impl_->get_device_alloc_info_ptr(dst->ndarray_alloc_));
    auto *src_ptr = reinterpret_cast<const uint8_t *>(
        program_impl_->get_device_alloc_info_ptr(src->ndarray_alloc_));
    TI_ERROR_IF(!dst_ptr || !src_ptr,
                "CPU ndarray copy received a null data pointer.");
    const int max_threads =
        std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
    const std::size_t chunk_bytes =
        dst_bytes <= (4 << 20) ? (1 << 20) : (256 << 10);
    const int target_threads = static_cast<int>(
        std::min<std::size_t>((dst_bytes + chunk_bytes - 1) / chunk_bytes,
                              static_cast<std::size_t>(max_threads)));
    if (dst_bytes >= (1 << 20) && target_threads > 1) {
      CpuCopyTaskContext ctx;
      ctx.dst = dst_ptr;
      ctx.src = src_ptr;
      ctx.bytes = dst_bytes;
      ctx.num_threads = target_threads;
      auto &pool = get_cpu_primitive_thread_pool(max_threads);
      pool.run(target_threads, target_threads, &ctx, cpu_copy_task);
      return;
    }
    std::memcpy(dst_ptr, src_ptr, dst_bytes);
    return;
  }

  if (compile_config().arch == Arch::cuda ||
      compile_config().arch == Arch::amdgpu) {
    Device::memcpy_direct(dst->ndarray_alloc_.get_ptr(0),
                          src->ndarray_alloc_.get_ptr(0), dst_bytes);
    return;
  }

  Stream *stream = program_impl_->get_compute_device()->get_compute_stream();
  auto [cmdlist, res] = stream->new_command_list_unique();
  TI_ASSERT(res == RhiResult::success);
  cmdlist->buffer_copy(dst->ndarray_alloc_.get_ptr(0),
                       src->ndarray_alloc_.get_ptr(0), dst_bytes);
  stream->submit_synced(cmdlist.get());
}

void Program::copy_ndarray_from_host(Ndarray *dst,
                                     const void *src,
                                     std::size_t bytes) {
  TI_ERROR_IF(!dst || !src,
              "copy_ndarray_from_host received a null pointer.");
  const std::size_t expected_bytes =
      dst->get_nelement() * dst->get_element_size();
  TI_ERROR_IF(bytes != expected_bytes,
              "copy_ndarray_from_host expected {} bytes, but received {}.",
              expected_bytes, bytes);
  if (bytes == 0) {
    return;
  }

  if (arch_is_cpu(compile_config().arch)) {
    auto *dst_ptr = reinterpret_cast<uint8_t *>(
        program_impl_->get_device_alloc_info_ptr(dst->ndarray_alloc_));
    TI_ERROR_IF(!dst_ptr,
                "CPU ndarray host upload received a null data pointer.");
    std::memcpy(dst_ptr, src, bytes);
    return;
  }

  auto *device = program_impl_->get_compute_device();
  DevicePtr dst_ptr = dst->ndarray_alloc_.get_ptr(0);
  const void *src_ptr = src;
  std::size_t size = bytes;
  const RhiResult res = device->upload_data(&dst_ptr, &src_ptr, &size, 1);
  TI_ERROR_IF(res != RhiResult::success,
              "copy_ndarray_from_host failed: {}", res);
}

void Program::copy_ndarray_to_host(Ndarray *src,
                                   void *dst,
                                   std::size_t bytes) {
  TI_ERROR_IF(!src || !dst, "copy_ndarray_to_host received a null pointer.");
  const std::size_t expected_bytes =
      src->get_nelement() * src->get_element_size();
  TI_ERROR_IF(bytes != expected_bytes,
              "copy_ndarray_to_host expected {} bytes, but received {}.",
              expected_bytes, bytes);
  if (bytes == 0) {
    return;
  }

  if (arch_is_cpu(compile_config().arch)) {
    auto *src_ptr = reinterpret_cast<const uint8_t *>(
        program_impl_->get_device_alloc_info_ptr(src->ndarray_alloc_));
    TI_ERROR_IF(!src_ptr,
                "CPU ndarray host readback received a null data pointer.");
    std::memcpy(dst, src_ptr, bytes);
    return;
  }

  auto *device = program_impl_->get_compute_device();
  DevicePtr src_ptr = src->ndarray_alloc_.get_ptr(0);
  void *dst_ptr = dst;
  std::size_t size = bytes;
  const RhiResult res = device->readback_data(&src_ptr, &dst_ptr, &size, 1);
  TI_ERROR_IF(res != RhiResult::success,
              "copy_ndarray_to_host failed: {}", res);
}

bool Program::cuda_device_transform_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         (cuda::driver_transform_available() ||
          cuda::cub_transform_available());
#else
  return false;
#endif
}

bool Program::cuda_toolkit_transform_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         cuda::cub_transform_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_device_transform_affine_ndarray(Ndarray *src,
                                                          Ndarray *dst,
                                                          int value_type,
                                                          double scale,
                                                          double bias) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA device transform is only available on CUDA.");
  TI_ERROR_IF(!src || !dst, "CUDA device transform received a null ndarray.");
  TI_ERROR_IF(src->get_nelement() != dst->get_nelement(),
              "CUDA device transform source and destination sizes differ.");
  TI_ERROR_IF(src->get_element_size() != dst->get_element_size(),
              "CUDA device transform source and destination dtypes differ.");
  TI_ERROR_IF(value_type < 0 || value_type > 5,
              "CUDA device transform received an unsupported value type.");
  const auto cuda_value_type =
      static_cast<cuda::CudaTransformValueType>(value_type);
  std::size_t expected_size = 0;
  switch (cuda_value_type) {
    case cuda::CudaTransformValueType::i32:
    case cuda::CudaTransformValueType::f32:
    case cuda::CudaTransformValueType::u32:
      expected_size = sizeof(uint32_t);
      break;
    case cuda::CudaTransformValueType::u64:
    case cuda::CudaTransformValueType::i64:
    case cuda::CudaTransformValueType::f64:
      expected_size = sizeof(uint64_t);
      break;
  }
  TI_ERROR_IF(src->get_element_size() != expected_size,
              "CUDA device transform dtype does not match value type.");
  TI_ERROR_IF(src->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device transform currently supports at most INT_MAX items.");
#ifdef TI_WITH_CUDA
  auto *src_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(src));
  auto *dst_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(dst));
  if (cuda_value_type == cuda::CudaTransformValueType::i32 ||
      cuda_value_type == cuda::CudaTransformValueType::u32 ||
      cuda_value_type == cuda::CudaTransformValueType::f32) {
    TI_ERROR_IF(!cuda::driver_transform_available(),
                "32-bit CUDA device transform requires CUDA driver API "
                "support.");
    return cuda::driver_transform_affine(
        src_ptr, dst_ptr, static_cast<int>(src->get_nelement()),
        cuda_value_type, scale, bias);
  }
  if (cuda::cub_transform_available()) {
    void *stream = CUDAContext::get_instance().get_stream();
    return cuda::cub_transform_affine(
        src_ptr, dst_ptr, static_cast<int>(src->get_nelement()),
        cuda_value_type, scale, bias, stream);
  }
  TI_ERROR_IF(cuda_value_type == cuda::CudaTransformValueType::u64 ||
                  cuda_value_type == cuda::CudaTransformValueType::i64 ||
                  cuda_value_type == cuda::CudaTransformValueType::f64,
              "64-bit CUDA device transform requires TI_WITH_CUDA_TOOLKIT=ON "
              "and a discoverable CUDA runtime.");
  return 0;
#else
  TI_ERROR("CUDA device transform requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_transform_affine_member_ndarray(
    Ndarray *src,
    Ndarray *dst,
    int value_type,
    std::size_t offset,
    std::size_t stride,
    double scale,
    double bias) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA strided transform is only available on CUDA.");
  check_transform_member_request("CUDA", src, dst, value_type, offset, stride);
  TI_ERROR_IF(src->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA strided transform currently supports at most INT_MAX "
              "items.");
#ifdef TI_WITH_CUDA
  TI_ERROR_IF(!cuda::cub_transform_available(),
              "CUDA strided transform requires TI_WITH_CUDA_TOOLKIT=ON and a "
              "discoverable CUDA runtime.");
  auto *src_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(src));
  auto *dst_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(dst));
  void *stream = CUDAContext::get_instance().get_stream();
  return cuda::cub_transform_affine_strided(
      src_ptr, dst_ptr, static_cast<int>(src->get_nelement()),
      static_cast<cuda::CudaTransformValueType>(value_type), offset, stride,
      scale, bias, stream);
#else
  TI_ERROR("CUDA strided transform requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_transform_affine_strided_ndarray(
    Ndarray *src,
    Ndarray *dst,
    int value_type,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride,
    double scale,
    double bias) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA strided transform is only available on CUDA.");
  check_transform_strided_request("CUDA", src, dst, value_type, src_offset,
                                  src_stride, dst_offset, dst_stride);
  TI_ERROR_IF(src->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA strided transform currently supports at most INT_MAX "
              "items.");
#ifdef TI_WITH_CUDA
  TI_ERROR_IF(!cuda::cub_transform_available(),
              "CUDA strided transform requires TI_WITH_CUDA_TOOLKIT=ON and a "
              "discoverable CUDA runtime.");
  auto *src_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(src));
  auto *dst_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(dst));
  void *stream = CUDAContext::get_instance().get_stream();
  return cuda::cub_transform_affine_strided_to_strided(
      src_ptr, dst_ptr, static_cast<int>(src->get_nelement()),
      static_cast<cuda::CudaTransformValueType>(value_type), src_offset,
      src_stride, dst_offset, dst_stride, scale, bias, stream);
#else
  TI_ERROR("CUDA strided transform requires TI_WITH_CUDA=ON.");
#endif
}

bool Program::cuda_device_indexed_copy_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         (cuda::cub_indexed_copy_available() ||
          cuda::driver_indexed_copy_available());
#else
  return false;
#endif
}

bool Program::cuda_device_indexed_copy_payload_available(
    std::size_t item_bytes) const {
#ifdef TI_WITH_CUDA
  if (compile_config().arch != Arch::cuda || item_bytes == 0 ||
      item_bytes % sizeof(uint32_t) != 0) {
    return false;
  }
  if (cuda::cub_indexed_copy_available()) {
    return true;
  }
  return item_bytes == sizeof(uint32_t) &&
         cuda::driver_indexed_copy_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_device_gather_ndarray(Ndarray *src,
                                                Ndarray *indices,
                                                Ndarray *dst) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA device gather is only available on CUDA.");
  TI_ERROR_IF(!src || !indices || !dst,
              "CUDA device gather received a null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "CUDA device gather currently expects 1D ndarrays.");
  TI_ERROR_IF(indices->get_nelement() != dst->get_nelement(),
              "CUDA device gather expects indices and destination sizes to "
              "match.");
  TI_ERROR_IF(src->get_element_size() != dst->get_element_size(),
              "CUDA device gather source and destination dtypes differ.");
  const std::size_t item_bytes = src->get_element_size();
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(uint32_t) != 0 ||
                  indices->get_element_size() != sizeof(int32_t),
              "CUDA device gather currently expects 4-byte aligned values and "
              "i32 indices.");
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device gather currently supports at most INT_MAX items.");
  TI_ERROR_IF(src->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device gather currently supports source sizes up to "
              "INT_MAX items.");
#ifdef TI_WITH_CUDA
  auto *src_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(src));
  auto *indices_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(indices));
  auto *dst_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(dst));
  const int item_words = static_cast<int>(item_bytes / sizeof(uint32_t));
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max() /
                                           item_words),
              "CUDA device gather word count exceeds INT_MAX.");
  if (cuda::cub_indexed_copy_available()) {
    void *stream = CUDAContext::get_instance().get_stream();
    return cuda::cub_indexed_copy(
        src_ptr, indices_ptr, dst_ptr,
        static_cast<int>(indices->get_nelement()),
        static_cast<int>(src->get_nelement()), item_words,
        cuda::CudaIndexedCopyOp::gather, stream);
  }
  TI_ERROR_IF(item_words != 1,
              "CUDA device gather for multi-word values requires "
              "TI_WITH_CUDA_TOOLKIT=ON and a discoverable CUDA runtime.");
  return cuda::driver_indexed_copy(src_ptr, indices_ptr, dst_ptr,
                                   static_cast<int>(indices->get_nelement()),
                                   static_cast<int>(src->get_nelement()),
                                   cuda::CudaIndexedCopyOp::gather);
#else
  TI_ERROR("CUDA device gather requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_gather_strided_ndarray(
    Ndarray *src,
    Ndarray *indices,
    Ndarray *dst,
    std::size_t item_bytes,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA device strided gather is only available on CUDA.");
  check_indexed_copy_strided_request("CUDA", src, indices, dst, item_bytes,
                                     src_offset, src_stride, dst_offset,
                                     dst_stride, false);
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device strided gather currently supports at most INT_MAX "
              "items.");
  TI_ERROR_IF(src->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device strided gather currently supports source sizes up "
              "to INT_MAX items.");
#ifdef TI_WITH_CUDA
  TI_ERROR_IF(!cuda::cub_indexed_copy_available(),
              "CUDA strided gather requires TI_WITH_CUDA_TOOLKIT=ON and a "
              "discoverable CUDA runtime.");
  auto *src_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(src));
  auto *indices_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(indices));
  auto *dst_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(dst));
  const int item_words = static_cast<int>(item_bytes / sizeof(uint32_t));
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max() /
                                           item_words),
              "CUDA device strided gather word count exceeds INT_MAX.");
  void *stream = CUDAContext::get_instance().get_stream();
  return cuda::cub_indexed_copy_strided(
      src_ptr, indices_ptr, dst_ptr, static_cast<int>(indices->get_nelement()),
      static_cast<int>(src->get_nelement()), item_words,
      src_offset / sizeof(uint32_t), src_stride / sizeof(uint32_t),
      dst_offset / sizeof(uint32_t), dst_stride / sizeof(uint32_t),
      cuda::CudaIndexedCopyOp::gather, stream);
#else
  TI_ERROR("CUDA device strided gather requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_scatter_ndarray(Ndarray *src,
                                                 Ndarray *indices,
                                                 Ndarray *dst) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA device scatter is only available on CUDA.");
  TI_ERROR_IF(!src || !indices || !dst,
              "CUDA device scatter received a null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "CUDA device scatter currently expects 1D ndarrays.");
  TI_ERROR_IF(src->get_nelement() != indices->get_nelement(),
              "CUDA device scatter expects source and indices sizes to match.");
  TI_ERROR_IF(src->get_element_size() != dst->get_element_size(),
              "CUDA device scatter source and destination dtypes differ.");
  const std::size_t item_bytes = src->get_element_size();
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(uint32_t) != 0 ||
                  indices->get_element_size() != sizeof(int32_t),
              "CUDA device scatter currently expects 4-byte aligned values and "
              "i32 indices.");
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device scatter currently supports at most INT_MAX items.");
  TI_ERROR_IF(dst->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device scatter currently supports destination sizes up to "
              "INT_MAX items.");
#ifdef TI_WITH_CUDA
  auto *src_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(src));
  auto *indices_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(indices));
  auto *dst_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(dst));
  const int item_words = static_cast<int>(item_bytes / sizeof(uint32_t));
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max() /
                                           item_words),
              "CUDA device scatter word count exceeds INT_MAX.");
  if (cuda::cub_indexed_copy_available()) {
    void *stream = CUDAContext::get_instance().get_stream();
    return cuda::cub_indexed_copy(
        src_ptr, indices_ptr, dst_ptr,
        static_cast<int>(indices->get_nelement()),
        static_cast<int>(dst->get_nelement()), item_words,
        cuda::CudaIndexedCopyOp::scatter, stream);
  }
  TI_ERROR_IF(item_words != 1,
              "CUDA device scatter for multi-word values requires "
              "TI_WITH_CUDA_TOOLKIT=ON and a discoverable CUDA runtime.");
  return cuda::driver_indexed_copy(src_ptr, indices_ptr, dst_ptr,
                                   static_cast<int>(indices->get_nelement()),
                                   static_cast<int>(dst->get_nelement()),
                                   cuda::CudaIndexedCopyOp::scatter);
#else
  TI_ERROR("CUDA device scatter requires TI_WITH_CUDA=ON.");
#endif
}

std::size_t Program::cuda_device_scatter_strided_ndarray(
    Ndarray *src,
    Ndarray *indices,
    Ndarray *dst,
    std::size_t item_bytes,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA device strided scatter is only available on CUDA.");
  check_indexed_copy_strided_request("CUDA", src, indices, dst, item_bytes,
                                     src_offset, src_stride, dst_offset,
                                     dst_stride, true);
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device strided scatter currently supports at most INT_MAX "
              "items.");
  TI_ERROR_IF(dst->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA device strided scatter currently supports destination "
              "sizes up to INT_MAX items.");
#ifdef TI_WITH_CUDA
  TI_ERROR_IF(!cuda::cub_indexed_copy_available(),
              "CUDA strided scatter requires TI_WITH_CUDA_TOOLKIT=ON and a "
              "discoverable CUDA runtime.");
  auto *src_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(src));
  auto *indices_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(indices));
  auto *dst_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(dst));
  const int item_words = static_cast<int>(item_bytes / sizeof(uint32_t));
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max() /
                                           item_words),
              "CUDA device strided scatter word count exceeds INT_MAX.");
  void *stream = CUDAContext::get_instance().get_stream();
  return cuda::cub_indexed_copy_strided(
      src_ptr, indices_ptr, dst_ptr, static_cast<int>(indices->get_nelement()),
      static_cast<int>(dst->get_nelement()), item_words,
      src_offset / sizeof(uint32_t), src_stride / sizeof(uint32_t),
      dst_offset / sizeof(uint32_t), dst_stride / sizeof(uint32_t),
      cuda::CudaIndexedCopyOp::scatter, stream);
#else
  TI_ERROR("CUDA device strided scatter requires TI_WITH_CUDA=ON.");
#endif
}

bool Program::cuda_device_scatter_add_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         cuda::cub_scatter_add_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_device_scatter_add_ndarray(Ndarray *src,
                                                     Ndarray *indices,
                                                     Ndarray *dst,
                                                     int value_type) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA toolkit scatter-add is only available on CUDA.");
  TI_ERROR_IF(!src || !indices || !dst,
              "CUDA toolkit scatter-add received a null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "CUDA toolkit scatter-add currently expects 1D ndarrays.");
  TI_ERROR_IF(src->get_nelement() != indices->get_nelement(),
              "CUDA toolkit scatter-add expects source and indices sizes to "
              "match.");
  TI_ERROR_IF(src->get_element_size() != dst->get_element_size(),
              "CUDA toolkit scatter-add source and destination dtypes differ.");
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(expected_size == 0,
              "CUDA toolkit scatter-add received an unsupported value type.");
  TI_ERROR_IF(src->get_element_size() != expected_size ||
                  indices->get_element_size() != sizeof(int32_t),
              "CUDA toolkit scatter-add dtype does not match value type or "
              "indices are not i32.");
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA toolkit scatter-add currently supports at most INT_MAX "
              "source items.");
  TI_ERROR_IF(dst->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA toolkit scatter-add currently supports destination sizes "
              "up to INT_MAX items.");
#ifdef TI_WITH_CUDA
  auto *src_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(src));
  auto *indices_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(indices));
  auto *dst_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(dst));
  void *stream = CUDAContext::get_instance().get_stream();
  return cuda::cub_scatter_add(
      src_ptr, indices_ptr, dst_ptr, static_cast<int>(indices->get_nelement()),
      static_cast<int>(dst->get_nelement()),
      static_cast<cuda::CudaScatterAddValueType>(value_type), stream);
#else
  TI_ERROR(
      "CUDA scatter-add requires building Taichi with TI_WITH_CUDA=ON and "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_device_scatter_add_member_ndarray(
    Ndarray *src,
    Ndarray *indices,
    Ndarray *dst,
    int value_type,
    std::size_t offset,
    std::size_t stride) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA toolkit strided scatter-add is only available on CUDA.");
  check_scatter_add_member_request("CUDA toolkit", src, indices, dst,
                                   value_type, offset, stride);
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA toolkit strided scatter-add currently supports at most "
              "INT_MAX source items.");
  TI_ERROR_IF(dst->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA toolkit strided scatter-add currently supports "
              "destination sizes up to INT_MAX items.");
#ifdef TI_WITH_CUDA
  void *stream = CUDAContext::get_instance().get_stream();
  return cuda::cub_scatter_add_strided(
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(src)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(indices)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(dst)),
      static_cast<int>(indices->get_nelement()),
      static_cast<int>(dst->get_nelement()),
      static_cast<cuda::CudaScatterAddValueType>(value_type), offset, stride,
      stream);
#else
  TI_ERROR(
      "CUDA strided scatter-add requires building Taichi with TI_WITH_CUDA=ON "
      "and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_device_scatter_add_strided_ndarray(
    Ndarray *src,
    Ndarray *indices,
    Ndarray *dst,
    int value_type,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA toolkit strided scatter-add is only available on CUDA.");
  check_scatter_add_strided_request("CUDA toolkit", src, indices, dst,
                                    value_type, src_offset, src_stride,
                                    dst_offset, dst_stride);
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA toolkit strided scatter-add currently supports at most "
              "INT_MAX source items.");
  TI_ERROR_IF(dst->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA toolkit strided scatter-add currently supports "
              "destination sizes up to INT_MAX items.");
#ifdef TI_WITH_CUDA
  void *stream = CUDAContext::get_instance().get_stream();
  return cuda::cub_scatter_add_strided_io(
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(src)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(indices)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(dst)),
      static_cast<int>(indices->get_nelement()),
      static_cast<int>(dst->get_nelement()),
      static_cast<cuda::CudaScatterAddValueType>(value_type), src_offset,
      src_stride, dst_offset, dst_stride, stream);
#else
  TI_ERROR(
      "CUDA strided scatter-add requires building Taichi with TI_WITH_CUDA=ON "
      "and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

bool Program::cuda_device_bucket_builder_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         cuda::cub_bucket_builder_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_device_bucket_builder_i32_ndarray(Ndarray *keys,
                                                            Ndarray *values,
                                                            Ndarray *offsets,
                                                            Ndarray *output,
                                                            Ndarray *cursor) {
  return cuda_device_bucket_builder_ndarray(keys, values, offsets, output,
                                            cursor, 0);
}

std::size_t Program::cuda_device_bucket_builder_ndarray(Ndarray *keys,
                                                        Ndarray *values,
                                                        Ndarray *offsets,
                                                        Ndarray *output,
                                                        Ndarray *cursor,
                                                        int value_type) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA toolkit bucket builder is only available on CUDA.");
  TI_ERROR_IF(!keys || !values || !offsets || !output || !cursor,
              "CUDA toolkit bucket builder received a null ndarray.");
  TI_ERROR_IF(keys->shape.size() != 1 || values->shape.size() != 1 ||
                  offsets->shape.size() != 1 || output->shape.size() != 1 ||
                  cursor->shape.size() != 1,
              "CUDA toolkit bucket builder expects 1D ndarrays.");
  TI_ERROR_IF(keys->get_nelement() != values->get_nelement(),
              "CUDA toolkit bucket builder keys and values sizes differ.");
  TI_ERROR_IF(offsets->get_nelement() < 2,
              "CUDA toolkit bucket builder offsets must contain num_bins + 1 items.");
  const std::size_t num_bins = offsets->get_nelement() - 1;
  TI_ERROR_IF(cursor->get_nelement() < num_bins,
              "CUDA toolkit bucket builder cursor is smaller than num_bins.");
  TI_ERROR_IF(output->get_nelement() < values->get_nelement(),
              "CUDA toolkit bucket builder output is smaller than input values.");
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(expected_size == 0,
              "CUDA toolkit bucket builder received an unsupported value type.");
  const std::size_t item_bytes = values->get_element_size();
  TI_ERROR_IF(keys->get_element_size() != sizeof(int32_t) ||
                  offsets->get_element_size() != sizeof(int32_t) ||
                  item_bytes == 0 ||
                  item_bytes % sizeof(uint32_t) != 0 ||
                  output->get_element_size() != item_bytes ||
                  cursor->get_element_size() != sizeof(int32_t),
              "CUDA toolkit bucket builder dtype does not match value type or "
              "keys/offsets/cursor are not i32, or payload is not 4-byte aligned.");
  TI_ERROR_IF(keys->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()) ||
                  num_bins > static_cast<std::size_t>(
                                 std::numeric_limits<uint32_t>::max()),
              "CUDA bucket builder input is too large for u32 launch parameters.");
  const int item_words = static_cast<int>(item_bytes / sizeof(uint32_t));
  TI_ERROR_IF(keys->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max() /
                                           item_words),
              "CUDA bucket builder word count exceeds INT_MAX.");
#ifdef TI_WITH_CUDA
  void *stream = CUDAContext::get_instance().get_stream();
  return cuda::cub_bucket_builder(
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(keys)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(offsets)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(cursor)),
      static_cast<int>(keys->get_nelement()), static_cast<int>(num_bins),
      static_cast<cuda::CudaBucketBuilderValueType>(value_type),
      item_words, stream, this);
#else
  TI_ERROR(
      "CUDA bucket builder requires building Taichi with TI_WITH_CUDA=ON and "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

bool Program::cuda_device_grouped_reduce_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         cuda::cub_grouped_reduce_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_device_grouped_reduce_i32_atomic_ndarray(
    Ndarray *keys,
    Ndarray *values,
    Ndarray *output,
    int op) {
  return cuda_device_grouped_reduce_atomic_ndarray(keys, values, output, 0, op);
}

std::size_t Program::cuda_device_grouped_reduce_atomic_ndarray(Ndarray *keys,
                                                               Ndarray *values,
                                                               Ndarray *output,
                                                               int value_type,
                                                               int op) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA grouped reduce is only available on CUDA.");
  TI_ERROR_IF(!keys || !values || !output,
              "CUDA grouped reduce received a null ndarray.");
  TI_ERROR_IF(keys->shape.size() != 1 || values->shape.size() != 1 ||
                  output->shape.size() != 1,
              "CUDA grouped reduce expects 1D ndarrays.");
  TI_ERROR_IF(keys->get_nelement() != values->get_nelement(),
              "CUDA grouped reduce keys and values sizes differ.");
  TI_ERROR_IF(output->get_nelement() == 0,
              "CUDA grouped reduce output must contain at least one group.");
  const std::size_t num_groups = output->get_nelement();
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(expected_size == 0,
              "CUDA grouped reduce received an unsupported value type.");
  TI_ERROR_IF(keys->get_element_size() != sizeof(int32_t) ||
                  values->get_element_size() != expected_size ||
                  output->get_element_size() != expected_size,
              "CUDA grouped reduce value type or i32 key size mismatch.");
  TI_ERROR_IF(op != 0, "CUDA grouped reduce currently supports only sum.");
  TI_ERROR_IF(keys->get_nelement() >
                      static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                  num_groups >
                      static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA grouped reduce input is too large for int launch parameters.");
#ifdef TI_WITH_CUDA
  void *stream = CUDAContext::get_instance().get_stream();
  return cuda::cub_grouped_reduce_atomic(
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(keys)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output)),
      static_cast<int>(keys->get_nelement()), static_cast<int>(num_groups),
      static_cast<cuda::CudaGroupedReduceValueType>(value_type), op, stream);
#else
  TI_ERROR(
      "CUDA grouped reduce requires building Taichi with TI_WITH_CUDA=ON and "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_device_grouped_reduce_atomic_member_ndarray(
    Ndarray *keys,
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t offset,
    std::size_t stride,
    int op) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA strided grouped reduce is only available on CUDA.");
  check_grouped_reduce_member_request("CUDA", keys, values, output, value_type,
                                      offset, stride, op);
  const std::size_t num_groups = output->get_nelement();
  TI_ERROR_IF(keys->get_nelement() >
                      static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                  num_groups >
                      static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA strided grouped reduce input is too large for int launch "
              "parameters.");
#ifdef TI_WITH_CUDA
  void *stream = CUDAContext::get_instance().get_stream();
  return cuda::cub_grouped_reduce_atomic_strided(
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(keys)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output)),
      static_cast<int>(keys->get_nelement()), static_cast<int>(num_groups),
      static_cast<cuda::CudaGroupedReduceValueType>(value_type), offset,
      stride, op, stream);
#else
  TI_ERROR(
      "CUDA strided grouped reduce requires building Taichi with "
      "TI_WITH_CUDA=ON and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_device_grouped_reduce_atomic_strided_ndarray(
    Ndarray *keys,
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t output_offset,
    std::size_t output_stride,
    int op) {
  return cuda_device_grouped_reduce_atomic_strided_keys_ndarray(
      keys, values, output, value_type, 0, sizeof(int32_t), values_offset,
      values_stride, output_offset, output_stride, op);
}

std::size_t Program::cuda_device_grouped_reduce_atomic_strided_keys_ndarray(
    Ndarray *keys,
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t keys_offset,
    std::size_t keys_stride,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t output_offset,
    std::size_t output_stride,
    int op) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA strided grouped reduce is only available on CUDA.");
  check_grouped_reduce_strided_keys_request(
      "CUDA", keys, values, output, value_type, keys_offset, keys_stride,
      values_offset, values_stride, output_offset, output_stride, op);
  const std::size_t num_groups = output->get_nelement();
  TI_ERROR_IF(keys->get_nelement() >
                      static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                  num_groups >
                      static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA strided grouped reduce input is too large for int launch "
              "parameters.");
#ifdef TI_WITH_CUDA
  void *stream = CUDAContext::get_instance().get_stream();
  return cuda::cub_grouped_reduce_atomic_strided_io(
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(keys)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output)),
      static_cast<int>(keys->get_nelement()), static_cast<int>(num_groups),
      static_cast<cuda::CudaGroupedReduceValueType>(value_type), keys_offset,
      keys_stride, values_offset, values_stride, output_offset, output_stride,
      op, stream);
#else
  TI_ERROR(
      "CUDA strided grouped reduce requires building Taichi with "
      "TI_WITH_CUDA=ON and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_device_grouped_reduce_i32_ndarray(Ndarray *keys,
                                                            Ndarray *values,
                                                            Ndarray *output,
                                                            Ndarray *offsets,
                                                            Ndarray *scratch,
                                                            Ndarray *cursor,
                                                            int op) {
  return cuda_device_grouped_reduce_ndarray(keys, values, output, offsets,
                                            scratch, cursor, 0, op);
}

std::size_t Program::cuda_device_grouped_reduce_ndarray(Ndarray *keys,
                                                        Ndarray *values,
                                                        Ndarray *output,
                                                        Ndarray *offsets,
                                                        Ndarray *scratch,
                                                        Ndarray *cursor,
                                                        int value_type,
                                                        int op) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA grouped reduce is only available on CUDA.");
  TI_ERROR_IF(!keys || !values || !output || !offsets || !scratch || !cursor,
              "CUDA grouped reduce received a null ndarray.");
  TI_ERROR_IF(keys->shape.size() != 1 || values->shape.size() != 1 ||
                  output->shape.size() != 1 || offsets->shape.size() != 1 ||
                  scratch->shape.size() != 1 || cursor->shape.size() != 1,
              "CUDA grouped reduce expects 1D ndarrays.");
  TI_ERROR_IF(keys->get_nelement() != values->get_nelement(),
              "CUDA grouped reduce keys and values sizes differ.");
  TI_ERROR_IF(output->get_nelement() == 0,
              "CUDA grouped reduce output must contain at least one group.");
  const std::size_t num_groups = output->get_nelement();
  TI_ERROR_IF(offsets->get_nelement() < num_groups + 1,
              "CUDA grouped reduce offsets must contain num_groups + 1 items.");
  TI_ERROR_IF(scratch->get_nelement() < values->get_nelement(),
              "CUDA grouped reduce scratch is smaller than input values.");
  TI_ERROR_IF(cursor->get_nelement() < num_groups,
              "CUDA grouped reduce cursor is smaller than num_groups.");
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(expected_size == 0,
              "CUDA grouped reduce received an unsupported value type.");
  TI_ERROR_IF(keys->get_element_size() != sizeof(int32_t) ||
                  values->get_element_size() != expected_size ||
                  output->get_element_size() != expected_size ||
                  offsets->get_element_size() != sizeof(int32_t) ||
                  scratch->get_element_size() != expected_size ||
                  cursor->get_element_size() != sizeof(int32_t),
              "CUDA grouped reduce value type or i32 metadata size mismatch.");
  TI_ERROR_IF(op != 0, "CUDA grouped reduce currently supports only sum.");
  TI_ERROR_IF(keys->get_nelement() >
                      static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                  num_groups >
                      static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA grouped reduce input is too large for int launch parameters.");
#ifdef TI_WITH_CUDA
  void *stream = CUDAContext::get_instance().get_stream();
  return cuda::cub_grouped_reduce(
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(keys)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(offsets)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(scratch)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(cursor)),
      static_cast<int>(keys->get_nelement()), static_cast<int>(num_groups),
      static_cast<cuda::CudaGroupedReduceValueType>(value_type), op,
      stream, this);
#else
  TI_ERROR(
      "CUDA grouped reduce requires building Taichi with TI_WITH_CUDA=ON and "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_device_grouped_reduce_segmented_strided_keys_ndarray(
    Ndarray *keys,
    Ndarray *values,
    Ndarray *output,
    Ndarray *offsets,
    Ndarray *scratch,
    Ndarray *cursor,
    int value_type,
    std::size_t keys_offset,
    std::size_t keys_stride,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t output_offset,
    std::size_t output_stride,
    int op) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA strided segmented grouped reduce is only available on "
              "CUDA.");
  TI_ERROR_IF(!offsets || !scratch || !cursor,
              "CUDA strided segmented grouped reduce received a null "
              "workspace ndarray.");
  check_grouped_reduce_strided_keys_request(
      "CUDA", keys, values, output, value_type, keys_offset, keys_stride,
      values_offset, values_stride, output_offset, output_stride, op);
  const std::size_t n = keys->get_nelement();
  const std::size_t num_groups = output->get_nelement();
  TI_ERROR_IF(offsets->shape.size() != 1 || scratch->shape.size() != 1 ||
                  cursor->shape.size() != 1,
              "CUDA strided segmented grouped reduce workspace expects 1D "
              "ndarrays.");
  TI_ERROR_IF(offsets->get_nelement() < num_groups + 1,
              "CUDA strided segmented grouped reduce offsets must contain "
              "num_groups + 1 items.");
  TI_ERROR_IF(scratch->get_nelement() < n,
              "CUDA strided segmented grouped reduce scratch is smaller than "
              "input values.");
  TI_ERROR_IF(cursor->get_nelement() < num_groups,
              "CUDA strided segmented grouped reduce cursor is smaller than "
              "num_groups.");
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(offsets->get_element_size() != sizeof(int32_t) ||
                  scratch->get_element_size() != expected_size ||
                  cursor->get_element_size() != sizeof(int32_t),
              "CUDA strided segmented grouped reduce workspace dtype mismatch.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                  num_groups >
                      static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA strided segmented grouped reduce input is too large for "
              "int launch parameters.");
#ifdef TI_WITH_CUDA
  void *stream = CUDAContext::get_instance().get_stream();
  return cuda::cub_grouped_reduce_strided_io(
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(keys)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(offsets)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(scratch)),
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(cursor)),
      static_cast<int>(n), static_cast<int>(num_groups),
      static_cast<cuda::CudaGroupedReduceValueType>(value_type), keys_offset,
      keys_stride, values_offset, values_stride, output_offset, output_stride,
      op, stream, this);
#else
  TI_ERROR(
      "CUDA strided segmented grouped reduce requires building Taichi with "
      "TI_WITH_CUDA=ON and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

bool Program::cuda_cub_radix_sort_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         cuda::cub_radix_sort_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_cub_radix_sort_ndarray(Ndarray *keys,
                                                 Ndarray *values,
                                                 int key_type,
                                                 int value_type,
                                                 int mode,
                                                 int nan_policy) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB sort is only available on the CUDA backend.");
  TI_ERROR_IF(!keys, "CUDA CUB sort received null keys ndarray.");
  TI_ERROR_IF(keys->shape.size() != 1,
              "CUDA CUB sort currently expects a 1D ndarray.");
  const bool has_values = values != nullptr;
  if (has_values) {
    TI_ERROR_IF(values->shape.size() != 1,
                "CUDA CUB sort values must be a 1D ndarray.");
    TI_ERROR_IF(values->get_nelement() != keys->get_nelement(),
                "CUDA CUB sort keys and values must have the same length.");
  }
#ifdef TI_WITH_CUDA
  std::size_t expected_key_size = 0;
  const std::size_t expected_value_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(mode < 0 || mode > 1,
              "CUDA CUB sort received an unsupported sort mode.");
  TI_ERROR_IF(nan_policy < 0 || nan_policy > 1,
              "CUDA CUB sort received an unsupported NaN policy.");
  TI_ERROR_IF(has_values && expected_value_size == 0,
              "CUDA CUB sort received an unsupported value type.");
  const std::size_t actual_value_size =
      has_values ? values->get_element_size() : expected_value_size;
  const auto cub_key_type = static_cast<cuda::CubSortKeyType>(key_type);
  const auto cub_value_type = static_cast<cuda::CubSortValueType>(value_type);
  const auto cub_mode = static_cast<cuda::CubSortMode>(mode);
  const auto cub_nan_policy =
      static_cast<cuda::CubSortNanPolicy>(nan_policy);
  switch (cub_key_type) {
    case cuda::CubSortKeyType::u32:
    case cuda::CubSortKeyType::i32:
    case cuda::CubSortKeyType::f32:
      expected_key_size = 4;
      break;
    case cuda::CubSortKeyType::u64:
    case cuda::CubSortKeyType::i64:
    case cuda::CubSortKeyType::f64:
      expected_key_size = 8;
      break;
  }
  TI_ERROR_IF(expected_key_size == 0,
              "CUDA CUB sort received an unsupported key type.");
  TI_ERROR_IF(keys->get_element_size() != expected_key_size,
              "CUDA CUB sort key dtype does not match the requested key type.");
  TI_ERROR_IF(has_values &&
                  (actual_value_size == 0 ||
                   actual_value_size % sizeof(uint32_t) != 0),
              "CUDA CUB sort value payload must be 4-byte aligned.");
  if (cub_mode == cuda::CubSortMode::split32) {
    TI_ERROR_IF(cub_key_type != cuda::CubSortKeyType::u64 &&
                    cub_key_type != cuda::CubSortKeyType::i64 &&
                    cub_key_type != cuda::CubSortKeyType::f64,
                "CUDA CUB split32 sort supports only u64/i64/f64 keys.");
  }
#endif
  auto key_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(keys));
  auto value_ptr = has_values
                       ? reinterpret_cast<void *>(
                             get_ndarray_data_ptr_as_int(values))
                       : nullptr;
#ifdef TI_WITH_CUDA
  void *stream = CUDAContext::get_instance().get_stream();
  return cuda::cub_radix_sort(
      key_ptr, value_ptr, static_cast<int>(keys->get_nelement()),
      cub_key_type, cub_value_type, cub_mode, cub_nan_policy, has_values,
      has_values ? static_cast<int>(actual_value_size / sizeof(uint32_t)) : 0,
      stream, this);
#else
  TI_ERROR(
      "CUDA CUB sort requires building Taichi with TI_WITH_CUDA=ON and "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

void Program::cuda_cub_radix_sort_clear_workspace() {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    cuda::cub_radix_sort_clear_cache(this);
  }
#endif
}

std::size_t Program::cuda_cub_radix_sort_workspace_bytes() const {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    return cuda::cub_radix_sort_cached_bytes(const_cast<Program *>(this));
  }
#endif
  return 0;
}

bool Program::cpu_stable_sort_available() const {
  return arch_is_cpu(compile_config().arch);
}

std::size_t Program::cpu_stable_sort_ndarray(Ndarray *keys,
                                             Ndarray *values,
                                             int key_type,
                                             int value_type,
                                             bool descending,
                                             int nan_policy) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native sort is only available on CPU backends.");
  TI_ERROR_IF(!keys, "CPU native sort received null keys ndarray.");
  TI_ERROR_IF(keys->shape.size() != 1,
              "CPU native sort currently expects a 1D ndarray.");
  TI_ERROR_IF(nan_policy < 0 || nan_policy > 1,
              "CPU native sort received an unsupported NaN policy.");
  const bool has_values = values != nullptr;
  if (has_values) {
    TI_ERROR_IF(values->shape.size() != 1,
                "CPU native sort values must be a 1D ndarray.");
    TI_ERROR_IF(values->get_nelement() != keys->get_nelement(),
                "CPU native sort keys and values must have the same length.");
    const std::size_t expected_value_size = primitive_value_type_size(value_type);
    TI_ERROR_IF(expected_value_size == 0,
                "CPU native sort received an unsupported value type.");
    TI_ERROR_IF(values->get_element_size() == 0 ||
                    values->get_element_size() % sizeof(uint32_t) != 0,
                "CPU native sort value payload must be 4-byte aligned.");
  }

  const std::size_t n = keys->get_nelement();
  void *value_ptr = has_values
                        ? reinterpret_cast<void *>(
                              get_ndarray_data_ptr_as_int(values))
                        : nullptr;
  auto key_ptr = get_ndarray_data_ptr_as_int(keys);
  TI_ERROR_IF(!key_ptr, "CPU native sort received a null key pointer.");
  TI_ERROR_IF(has_values && !value_ptr,
              "CPU native sort received a null value pointer.");
  const std::size_t expected_value_size =
      has_values ? primitive_value_type_size(value_type) : 0;
  const std::size_t value_item_bytes =
      has_values ? values->get_element_size() : 0;
  const bool raw_value_payload =
      has_values && value_item_bytes != expected_value_size;

  switch (key_type) {
    case 0:
      TI_ERROR_IF(keys->get_element_size() != sizeof(uint32_t),
                  "CPU native sort key dtype does not match ti.u32.");
      if (raw_value_payload) {
        return cpu_stable_sort_raw_values(reinterpret_cast<uint32_t *>(key_ptr),
                                          value_ptr, n, value_item_bytes,
                                          descending, nan_policy);
      }
      return cpu_stable_sort_value_dispatch(
          reinterpret_cast<uint32_t *>(key_ptr), value_ptr, n, value_type,
          descending, nan_policy);
    case 1:
      TI_ERROR_IF(keys->get_element_size() != sizeof(int32_t),
                  "CPU native sort key dtype does not match ti.i32.");
      if (raw_value_payload) {
        return cpu_stable_sort_raw_values(reinterpret_cast<int32_t *>(key_ptr),
                                          value_ptr, n, value_item_bytes,
                                          descending, nan_policy);
      }
      return cpu_stable_sort_value_dispatch(
          reinterpret_cast<int32_t *>(key_ptr), value_ptr, n, value_type,
          descending, nan_policy);
    case 2:
      TI_ERROR_IF(keys->get_element_size() != sizeof(float),
                  "CPU native sort key dtype does not match ti.f32.");
      if (raw_value_payload) {
        return cpu_stable_sort_raw_values(reinterpret_cast<float *>(key_ptr),
                                          value_ptr, n, value_item_bytes,
                                          descending, nan_policy);
      }
      return cpu_stable_sort_value_dispatch(
          reinterpret_cast<float *>(key_ptr), value_ptr, n, value_type,
          descending, nan_policy);
    case 3:
      TI_ERROR_IF(keys->get_element_size() != sizeof(uint64_t),
                  "CPU native sort key dtype does not match ti.u64.");
      if (raw_value_payload) {
        return cpu_stable_sort_raw_values(reinterpret_cast<uint64_t *>(key_ptr),
                                          value_ptr, n, value_item_bytes,
                                          descending, nan_policy);
      }
      return cpu_stable_sort_value_dispatch(
          reinterpret_cast<uint64_t *>(key_ptr), value_ptr, n, value_type,
          descending, nan_policy);
    case 4:
      TI_ERROR_IF(keys->get_element_size() != sizeof(int64_t),
                  "CPU native sort key dtype does not match ti.i64.");
      if (raw_value_payload) {
        return cpu_stable_sort_raw_values(reinterpret_cast<int64_t *>(key_ptr),
                                          value_ptr, n, value_item_bytes,
                                          descending, nan_policy);
      }
      return cpu_stable_sort_value_dispatch(
          reinterpret_cast<int64_t *>(key_ptr), value_ptr, n, value_type,
          descending, nan_policy);
    case 5:
      TI_ERROR_IF(keys->get_element_size() != sizeof(double),
                  "CPU native sort key dtype does not match ti.f64.");
      if (raw_value_payload) {
        return cpu_stable_sort_raw_values(reinterpret_cast<double *>(key_ptr),
                                          value_ptr, n, value_item_bytes,
                                          descending, nan_policy);
      }
      return cpu_stable_sort_value_dispatch(
          reinterpret_cast<double *>(key_ptr), value_ptr, n, value_type,
          descending, nan_policy);
    default:
      TI_ERROR("CPU native sort received an unsupported key type.");
  }
}

bool Program::cuda_cub_scan_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda &&
         cuda::cub_inclusive_scan_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_cub_inclusive_scan_ndarray(Ndarray *data,
                                                     int value_type) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB scan is only available on the CUDA backend.");
  TI_ERROR_IF(!data, "CUDA CUB scan received a null ndarray.");
  TI_ERROR_IF(data->shape.size() != 1,
              "CUDA CUB scan currently expects a 1D ndarray.");
#ifdef TI_WITH_CUDA
  const auto cub_value_type = static_cast<cuda::CubScanValueType>(value_type);
  std::size_t expected_value_size = 0;
  switch (cub_value_type) {
    case cuda::CubScanValueType::i32:
      expected_value_size = sizeof(int32_t);
      break;
    case cuda::CubScanValueType::f32:
      expected_value_size = sizeof(float);
      break;
    case cuda::CubScanValueType::u32:
      expected_value_size = sizeof(uint32_t);
      break;
    case cuda::CubScanValueType::u64:
      expected_value_size = sizeof(uint64_t);
      break;
    case cuda::CubScanValueType::i64:
      expected_value_size = sizeof(int64_t);
      break;
    case cuda::CubScanValueType::f64:
      expected_value_size = sizeof(double);
      break;
  }
  TI_ERROR_IF(expected_value_size == 0,
              "CUDA CUB scan received an unsupported value type.");
  TI_ERROR_IF(data->get_element_size() != expected_value_size,
              "CUDA CUB scan dtype does not match the requested value type.");
  auto data_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(data));
  void *stream = CUDAContext::get_instance().get_stream();
  return cuda::cub_inclusive_scan(
      data_ptr, static_cast<int>(data->get_nelement()), cub_value_type, stream,
      this);
#else
  TI_ERROR(
      "CUDA CUB scan requires building Taichi with TI_WITH_CUDA=ON and "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_cub_inclusive_scan_member_ndarray(
    Ndarray *data,
    int value_type,
    std::size_t offset,
    std::size_t stride) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB strided scan is only available on the CUDA backend.");
  check_scan_member_request("CUDA CUB", data, value_type, offset, stride);
  TI_ERROR_IF(data->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA CUB strided scan currently supports at most INT_MAX "
              "items.");
  if (data->get_nelement() <= 1) {
    return 0;
  }
#ifdef TI_WITH_CUDA
  auto data_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(data));
  void *stream = CUDAContext::get_instance().get_stream();
  return cuda::cub_inclusive_scan_strided(
      data_ptr, static_cast<int>(data->get_nelement()),
      static_cast<cuda::CubScanValueType>(value_type), offset, stride, stream,
      this);
#else
  TI_ERROR(
      "CUDA CUB strided scan requires building Taichi with TI_WITH_CUDA=ON "
      "and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

void Program::cuda_cub_scan_clear_workspace() {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    cuda::cub_inclusive_scan_clear_cache(this);
  }
#endif
}

std::size_t Program::cuda_cub_scan_workspace_bytes() const {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    return cuda::cub_inclusive_scan_cached_bytes(const_cast<Program *>(this));
  }
#endif
  return 0;
}

bool Program::cuda_cub_select_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda && cuda::cub_select_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_cub_select_ndarray(Ndarray *values,
                                             Ndarray *flags,
                                             Ndarray *output,
                                             Ndarray *count,
                                             int value_type) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB select is only available on the CUDA backend.");
  TI_ERROR_IF(!values || !flags || !output || !count,
              "CUDA CUB select received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || flags->shape.size() != 1 ||
                  output->shape.size() != 1 || count->shape.size() != 1,
              "CUDA CUB select currently expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() != flags->get_nelement() ||
                  values->get_nelement() > output->get_nelement(),
              "CUDA CUB select expects values/flags to have the same length "
              "and output to have at least that many elements.");
  TI_ERROR_IF(count->get_nelement() < 1,
              "CUDA CUB select count ndarray must have at least one element.");
  std::size_t expected_value_bytes = 0;
  switch (value_type) {
    case 0:
    case 1:
    case 2:
      expected_value_bytes = sizeof(uint32_t);
      break;
    case 3:
    case 4:
    case 5:
      expected_value_bytes = sizeof(uint64_t);
      break;
    default:
      TI_ERROR("CUDA CUB select received an unsupported value type.");
  }
  TI_ERROR_IF(expected_value_bytes == 0,
              "CUDA CUB select received an unsupported value type.");
  const std::size_t item_bytes = values->get_element_size();
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(uint32_t) != 0 ||
                  output->get_element_size() != item_bytes ||
                  flags->get_element_size() != sizeof(int32_t) ||
                  count->get_element_size() != sizeof(int32_t),
              "CUDA CUB select received mismatched value/flag/count dtypes or "
              "a non-4-byte-aligned payload.");
  TI_ERROR_IF(values->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA CUB select currently supports at most INT_MAX items.");
  const int item_words = static_cast<int>(item_bytes / sizeof(uint32_t));
  TI_ERROR_IF(values->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max() /
                                           item_words),
              "CUDA CUB select word count exceeds INT_MAX.");
#ifdef TI_WITH_CUDA
  auto values_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values));
  auto flags_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(flags));
  auto output_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output));
  auto count_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(count));
  void *stream = CUDAContext::get_instance().get_stream();
  return cuda::cub_select_flagged(
      values_ptr, flags_ptr, output_ptr, count_ptr,
      static_cast<int>(values->get_nelement()),
      static_cast<cuda::CubSelectValueType>(value_type), item_words, stream,
      this);
#else
  TI_ERROR(
      "CUDA CUB select requires building Taichi with TI_WITH_CUDA=ON and "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_cub_select_i32_ndarray(Ndarray *values,
                                                 Ndarray *flags,
                                                 Ndarray *output,
                                                 Ndarray *count) {
  return cuda_cub_select_ndarray(values, flags, output, count, 0);
}

void Program::cuda_cub_select_clear_workspace() {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    cuda::cub_select_clear_cache(this);
  }
#endif
}

std::size_t Program::cuda_cub_select_workspace_bytes() const {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    return cuda::cub_select_cached_bytes(const_cast<Program *>(this));
  }
#endif
  return 0;
}

bool Program::cuda_cub_histogram_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda && cuda::cub_histogram_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_cub_histogram_i32_ndarray(Ndarray *values,
                                                    Ndarray *bins) {
  return cuda_cub_histogram_ndarray(values, bins, 0, 0);
}

std::size_t Program::cuda_cub_histogram_ndarray(Ndarray *values,
                                                Ndarray *bins,
                                                int value_type,
                                                int bin_type) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB histogram is only available on CUDA.");
  TI_ERROR_IF(!values || !bins,
              "CUDA CUB histogram received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || bins->shape.size() != 1,
              "CUDA CUB histogram currently expects 1D ndarrays.");
  TI_ERROR_IF(value_type != 0 && value_type != 2,
              "CUDA CUB histogram currently supports only i32/u32 bin ids.");
  const std::size_t value_size = value_type == 2 ? sizeof(uint32_t)
                                                 : sizeof(int32_t);
  const std::size_t bin_size = histogram_bin_type_size(bin_type);
  TI_ERROR_IF(bin_size == 0,
              "CUDA CUB histogram currently supports only i32/i64 bins.");
  TI_ERROR_IF(values->get_element_size() != value_size ||
                  bins->get_element_size() != bin_size,
              "CUDA CUB histogram received mismatched value/bin dtypes.");
  TI_ERROR_IF(bins->get_nelement() == 0,
              "CUDA CUB histogram expects at least one bin.");
#ifdef TI_WITH_CUDA
  auto values_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values));
  auto bins_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(bins));
  void *stream = CUDAContext::get_instance().get_stream();
  return cuda::cub_histogram_even(
      values_ptr, bins_ptr, static_cast<int>(values->get_nelement()),
      static_cast<int>(bins->get_nelement()),
      static_cast<cuda::CubHistogramValueType>(value_type),
      static_cast<cuda::CubHistogramBinType>(bin_type),
      stream, this);
#else
  TI_ERROR(
      "CUDA CUB histogram requires building Taichi with TI_WITH_CUDA=ON and "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

void Program::cuda_cub_histogram_clear_workspace() {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    cuda::cub_histogram_clear_cache(this);
  }
#endif
}

std::size_t Program::cuda_cub_histogram_workspace_bytes() const {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    return cuda::cub_histogram_cached_bytes(const_cast<Program *>(this));
  }
#endif
  return 0;
}

bool Program::cuda_cub_reduce_available() const {
#ifdef TI_WITH_CUDA
  return compile_config().arch == Arch::cuda && cuda::cub_reduce_available();
#else
  return false;
#endif
}

std::size_t Program::cuda_cub_reduce_ndarray(Ndarray *values,
                                             Ndarray *output,
                                             int value_type,
                                             int op) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB reduce is only available on CUDA.");
  TI_ERROR_IF(!values || !output,
              "CUDA CUB reduce received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "CUDA CUB reduce currently expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() == 0,
              "CUDA CUB reduce expects at least one input item.");
  TI_ERROR_IF(output->get_nelement() < 1,
              "CUDA CUB reduce output ndarray must have at least one item.");
  TI_ERROR_IF(values->get_element_size() != output->get_element_size(),
              "CUDA CUB reduce expects matching input/output element sizes.");
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(expected_size == 0,
              "CUDA CUB reduce received an unsupported value type.");
  TI_ERROR_IF(values->get_element_size() != expected_size,
              "CUDA CUB reduce dtype does not match value type.");
  TI_ERROR_IF(op < 0 || op > 2,
              "CUDA CUB reduce received an unsupported op.");
#ifdef TI_WITH_CUDA
  auto values_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values));
  auto output_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output));
  void *stream = CUDAContext::get_instance().get_stream();
  return cuda::cub_reduce(values_ptr, output_ptr,
                          static_cast<int>(values->get_nelement()),
                          static_cast<cuda::CubReduceValueType>(value_type),
                          static_cast<cuda::CubReduceOp>(op), stream, this);
#else
  TI_ERROR(
      "CUDA CUB reduce requires building Taichi with TI_WITH_CUDA=ON and "
      "TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_cub_reduce_member_ndarray(Ndarray *values,
                                                    Ndarray *output,
                                                    int value_type,
                                                    std::size_t offset,
                                                    std::size_t stride,
                                                    int op) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB strided reduce is only available on CUDA.");
  check_reduce_member_request("CUDA CUB", values, output, value_type, offset,
                              stride, op);
  TI_ERROR_IF(values->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA CUB strided reduce currently supports at most INT_MAX "
              "items.");
#ifdef TI_WITH_CUDA
  auto values_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values));
  auto output_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output));
  void *stream = CUDAContext::get_instance().get_stream();
  return cuda::cub_reduce_strided(
      values_ptr, output_ptr, static_cast<int>(values->get_nelement()),
      static_cast<cuda::CubReduceValueType>(value_type), offset, stride,
      static_cast<cuda::CubReduceOp>(op), stream, this);
#else
  TI_ERROR(
      "CUDA CUB strided reduce requires building Taichi with TI_WITH_CUDA=ON "
      "and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

std::size_t Program::cuda_cub_reduce_strided_ndarray(
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t output_offset,
    std::size_t output_stride,
    int op) {
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA CUB strided reduce is only available on CUDA.");
  check_reduce_strided_request("CUDA CUB", values, output, value_type,
                               values_offset, values_stride, output_offset,
                               output_stride, op);
  TI_ERROR_IF(values->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "CUDA CUB strided reduce currently supports at most INT_MAX "
              "items.");
#ifdef TI_WITH_CUDA
  auto values_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values));
  auto output_ptr = reinterpret_cast<void *>(
      get_ndarray_data_ptr_as_int(output) + output_offset);
  void *stream = CUDAContext::get_instance().get_stream();
  return cuda::cub_reduce_strided(
      values_ptr, output_ptr, static_cast<int>(values->get_nelement()),
      static_cast<cuda::CubReduceValueType>(value_type), values_offset,
      values_stride, static_cast<cuda::CubReduceOp>(op), stream, this);
#else
  TI_ERROR(
      "CUDA CUB strided reduce requires building Taichi with TI_WITH_CUDA=ON "
      "and TI_WITH_CUDA_TOOLKIT=ON.");
#endif
}

void Program::cuda_cub_reduce_clear_workspace() {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    cuda::cub_reduce_clear_cache(this);
  }
#endif
}

std::size_t Program::cuda_cub_reduce_workspace_bytes() const {
#ifdef TI_WITH_CUDA
  if (compile_config().arch == Arch::cuda) {
    return cuda::cub_reduce_cached_bytes(const_cast<Program *>(this));
  }
#endif
  return 0;
}

bool Program::cpu_scan_available() const {
  return arch_is_cpu(compile_config().arch);
}

std::size_t Program::cpu_inclusive_scan_ndarray(Ndarray *data,
                                                int value_type) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native scan is only available on CPU backends.");
  TI_ERROR_IF(!data, "CPU native scan received a null ndarray.");
  TI_ERROR_IF(data->shape.size() != 1,
              "CPU native scan currently expects a 1D ndarray.");
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(expected_size == 0,
              "CPU native scan received an unsupported value type.");
  TI_ERROR_IF(data->get_element_size() != expected_size,
              "CPU native scan dtype does not match the requested value type.");
  auto ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(data));
  TI_ERROR_IF(!ptr, "CPU native scan received a null data pointer.");
  const std::size_t n = data->get_nelement();

  auto scan_typed = [n](auto *typed_ptr) {
    using T = std::remove_pointer_t<decltype(typed_ptr)>;
    T prefix{};
    for (std::size_t i = 0; i < n; ++i) {
      prefix += typed_ptr[i];
      typed_ptr[i] = prefix;
    }
  };

  switch (value_type) {
    case 0:
      scan_typed(reinterpret_cast<int32_t *>(ptr));
      break;
    case 1:
      scan_typed(reinterpret_cast<float *>(ptr));
      break;
    case 2:
      scan_typed(reinterpret_cast<uint32_t *>(ptr));
      break;
    case 3:
      scan_typed(reinterpret_cast<uint64_t *>(ptr));
      break;
    case 4:
      scan_typed(reinterpret_cast<int64_t *>(ptr));
      break;
    case 5:
      scan_typed(reinterpret_cast<double *>(ptr));
      break;
    default:
      TI_ERROR("CPU native scan received an unsupported value type.");
  }
  return 0;
}

std::size_t Program::cpu_inclusive_scan_member_ndarray(Ndarray *data,
                                                       int value_type,
                                                       std::size_t offset,
                                                       std::size_t stride) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native strided scan is only available on CPU backends.");
  check_scan_member_request("CPU native", data, value_type, offset, stride);
  auto ptr = reinterpret_cast<uint8_t *>(get_ndarray_data_ptr_as_int(data));
  TI_ERROR_IF(!ptr, "CPU native strided scan received a null data pointer.");
  const std::size_t n = data->get_nelement();

  switch (value_type) {
    case 0:
      return cpu_scan_strided_typed<int32_t>(ptr, n, offset, stride);
    case 1:
      return cpu_scan_strided_typed<float>(ptr, n, offset, stride);
    case 2:
      return cpu_scan_strided_typed<uint32_t>(ptr, n, offset, stride);
    case 3:
      return cpu_scan_strided_typed<uint64_t>(ptr, n, offset, stride);
    case 4:
      return cpu_scan_strided_typed<int64_t>(ptr, n, offset, stride);
    case 5:
      return cpu_scan_strided_typed<double>(ptr, n, offset, stride);
    default:
      TI_ERROR("CPU native strided scan received an unsupported value type.");
  }
}

std::size_t Program::cpu_scan_workspace_bytes() const {
  return 0;
}

bool Program::cpu_compact_available() const {
  return arch_is_cpu(compile_config().arch);
}

std::size_t Program::cpu_compact_ndarray(Ndarray *values,
                                         Ndarray *flags,
                                         Ndarray *output,
                                         Ndarray *count,
                                         int value_type) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native compact is only available on CPU backends.");
  TI_ERROR_IF(!values || !flags || !output || !count,
              "CPU native compact received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || flags->shape.size() != 1 ||
                  output->shape.size() != 1 || count->shape.size() != 1,
              "CPU native compact expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() != flags->get_nelement(),
              "CPU native compact values and flags must have the same length.");
  TI_ERROR_IF(output->get_nelement() < values->get_nelement(),
              "CPU native compact output must have at least input length.");
  TI_ERROR_IF(count->get_nelement() < 1,
              "CPU native compact count must contain at least one item.");
  const std::size_t value_bytes =
      (value_type == 0 || value_type == 1 || value_type == 2)
          ? sizeof(uint32_t)
          : (value_type == 3 || value_type == 4 || value_type == 5)
                ? sizeof(uint64_t)
                : 0;
  TI_ERROR_IF(value_bytes == 0,
              "CPU native compact received an unsupported value type.");
  const std::size_t item_bytes = values->get_element_size();
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(uint32_t) != 0 ||
                  output->get_element_size() != item_bytes ||
                  flags->get_element_size() != sizeof(int32_t) ||
                  count->get_element_size() != sizeof(int32_t),
              "CPU native compact received mismatched value/flag/count dtypes "
              "or a non-4-byte-aligned payload.");

  auto *values_ptr =
      reinterpret_cast<const uint8_t *>(get_ndarray_data_ptr_as_int(values));
  auto *flags_ptr =
      reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(flags));
  auto *output_ptr =
      reinterpret_cast<uint8_t *>(get_ndarray_data_ptr_as_int(output));
  auto *count_ptr =
      reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(count));
  TI_ERROR_IF(!values_ptr || !flags_ptr || !output_ptr || !count_ptr,
              "CPU native compact received a null data pointer.");

  std::size_t written = 0;
  const std::size_t n = values->get_nelement();
  for (std::size_t i = 0; i < n; ++i) {
    if (flags_ptr[i] != 0) {
      std::memcpy(output_ptr + written * item_bytes,
                  values_ptr + i * item_bytes, item_bytes);
      written++;
    }
  }
  TI_ERROR_IF(written > static_cast<std::size_t>(
                            std::numeric_limits<int32_t>::max()),
              "CPU native compact output count exceeds i32 range.");
  count_ptr[0] = static_cast<int32_t>(written);
  return 0;
}

std::size_t Program::cpu_compact_i32_ndarray(Ndarray *values,
                                             Ndarray *flags,
                                             Ndarray *output,
                                             Ndarray *count) {
  return cpu_compact_ndarray(values, flags, output, count, 0);
}

std::size_t Program::cpu_compact_workspace_bytes() const {
  return 0;
}

bool Program::cpu_histogram_available() const {
  return arch_is_cpu(compile_config().arch);
}

std::size_t Program::cpu_histogram_i32_ndarray(Ndarray *values,
                                               Ndarray *bins) {
  return cpu_histogram_ndarray(values, bins, 0, 0);
}

std::size_t Program::cpu_histogram_ndarray(Ndarray *values,
                                           Ndarray *bins,
                                           int value_type,
                                           int bin_type) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native histogram is only available on CPU backends.");
  TI_ERROR_IF(!values || !bins,
              "CPU native histogram received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || bins->shape.size() != 1,
              "CPU native histogram expects 1D ndarrays.");
  TI_ERROR_IF(value_type != 0 && value_type != 2,
              "CPU native histogram currently supports only i32/u32 bin ids.");
  const std::size_t value_size = value_type == 2 ? sizeof(uint32_t)
                                                 : sizeof(int32_t);
  const std::size_t bin_size = histogram_bin_type_size(bin_type);
  TI_ERROR_IF(bin_size == 0,
              "CPU native histogram currently supports only i32/i64 bins.");
  TI_ERROR_IF(values->get_element_size() != value_size ||
                  bins->get_element_size() != bin_size,
              "CPU native histogram received mismatched value/bin dtypes.");
  TI_ERROR_IF(bins->get_nelement() == 0,
              "CPU native histogram expects at least one bin.");
  if (bin_type == 0) {
    TI_ERROR_IF(values->get_nelement() >
                    static_cast<std::size_t>(
                        std::numeric_limits<int32_t>::max()),
                "CPU native histogram input is too large for i32 bin counts.");
  }

  auto *values_ptr =
      reinterpret_cast<const void *>(get_ndarray_data_ptr_as_int(values));
  auto *bins_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(bins));
  TI_ERROR_IF(!values_ptr || !bins_ptr,
              "CPU native histogram received a null data pointer.");

  const std::size_t n = values->get_nelement();
  const std::size_t num_bins = bins->get_nelement();
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = n >= 65536 && num_bins <= 4096 &&
                            target_threads > 1;
  if (value_type == 2 && bin_type == 4) {
    return cpu_histogram_typed(static_cast<const uint32_t *>(values_ptr),
                               static_cast<int64_t *>(bins_ptr), n, num_bins,
                               max_threads, target_threads, use_parallel);
  }
  if (value_type == 2) {
    return cpu_histogram_typed(static_cast<const uint32_t *>(values_ptr),
                               static_cast<int32_t *>(bins_ptr), n, num_bins,
                               max_threads, target_threads, use_parallel);
  }
  if (bin_type == 4) {
    return cpu_histogram_typed(static_cast<const int32_t *>(values_ptr),
                               static_cast<int64_t *>(bins_ptr), n, num_bins,
                               max_threads, target_threads, use_parallel);
  }
  return cpu_histogram_typed(static_cast<const int32_t *>(values_ptr),
                             static_cast<int32_t *>(bins_ptr), n, num_bins,
                             max_threads, target_threads, use_parallel);
}

std::size_t Program::cpu_histogram_workspace_bytes() const {
  return 0;
}

bool Program::cpu_reduce_available() const {
  return arch_is_cpu(compile_config().arch);
}

std::size_t Program::cpu_reduce_ndarray(Ndarray *values,
                                        Ndarray *output,
                                        int value_type,
                                        int op) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native reduce is only available on CPU backends.");
  TI_ERROR_IF(!values || !output,
              "CPU native reduce received a null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "CPU native reduce expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() == 0,
              "CPU native reduce expects at least one input item.");
  TI_ERROR_IF(output->get_nelement() < 1,
              "CPU native reduce output must contain at least one item.");
  TI_ERROR_IF(values->get_element_size() != output->get_element_size(),
              "CPU native reduce expects matching input/output element sizes.");
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(expected_size == 0,
              "CPU native reduce received an unsupported value type.");
  TI_ERROR_IF(values->get_element_size() != expected_size,
              "CPU native reduce dtype does not match value type.");
  TI_ERROR_IF(op < 0 || op > 2,
              "CPU native reduce received an unsupported op.");

  const std::size_t n = values->get_nelement();
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = n >= 65536 && target_threads > 1;

  switch (value_type) {
    case 0:
      return cpu_reduce_typed(
          reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(values)),
          reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(output)), op,
          n, max_threads, target_threads, use_parallel);
    case 1:
      return cpu_reduce_typed(
          reinterpret_cast<float *>(get_ndarray_data_ptr_as_int(values)),
          reinterpret_cast<float *>(get_ndarray_data_ptr_as_int(output)), op, n,
          max_threads, target_threads, use_parallel);
    case 2:
      return cpu_reduce_typed(
          reinterpret_cast<uint32_t *>(get_ndarray_data_ptr_as_int(values)),
          reinterpret_cast<uint32_t *>(get_ndarray_data_ptr_as_int(output)), op,
          n, max_threads, target_threads, use_parallel);
    case 3:
      return cpu_reduce_typed(
          reinterpret_cast<uint64_t *>(get_ndarray_data_ptr_as_int(values)),
          reinterpret_cast<uint64_t *>(get_ndarray_data_ptr_as_int(output)), op,
          n, max_threads, target_threads, use_parallel);
    case 4:
      return cpu_reduce_typed(
          reinterpret_cast<int64_t *>(get_ndarray_data_ptr_as_int(values)),
          reinterpret_cast<int64_t *>(get_ndarray_data_ptr_as_int(output)), op,
          n, max_threads, target_threads, use_parallel);
    case 5:
      return cpu_reduce_typed(
          reinterpret_cast<double *>(get_ndarray_data_ptr_as_int(values)),
          reinterpret_cast<double *>(get_ndarray_data_ptr_as_int(output)), op,
          n, max_threads, target_threads, use_parallel);
  }
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::cpu_reduce_member_ndarray(Ndarray *values,
                                               Ndarray *output,
                                               int value_type,
                                               std::size_t offset,
                                               std::size_t stride,
                                               int op) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native strided reduce is only available on CPU backends.");
  check_reduce_member_request("CPU native", values, output, value_type, offset,
                              stride, op);

  const std::size_t n = values->get_nelement();
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = n >= 65536 && target_threads > 1;

  const auto values_addr = get_ndarray_data_ptr_as_int(values);
  const auto output_addr = get_ndarray_data_ptr_as_int(output);
  TI_ERROR_IF(!values_addr || !output_addr,
              "CPU native strided reduce received a null data pointer.");
  const auto *values_ptr = reinterpret_cast<const uint8_t *>(values_addr);
  switch (value_type) {
    case 0:
      return cpu_reduce_strided_typed<int32_t>(
          values_ptr, reinterpret_cast<int32_t *>(output_addr), op, n, offset,
          stride, max_threads, target_threads, use_parallel);
    case 1:
      return cpu_reduce_strided_typed<float>(
          values_ptr, reinterpret_cast<float *>(output_addr), op, n, offset,
          stride, max_threads, target_threads, use_parallel);
    case 2:
      return cpu_reduce_strided_typed<uint32_t>(
          values_ptr, reinterpret_cast<uint32_t *>(output_addr), op, n, offset,
          stride, max_threads, target_threads, use_parallel);
    case 3:
      return cpu_reduce_strided_typed<uint64_t>(
          values_ptr, reinterpret_cast<uint64_t *>(output_addr), op, n, offset,
          stride, max_threads, target_threads, use_parallel);
    case 4:
      return cpu_reduce_strided_typed<int64_t>(
          values_ptr, reinterpret_cast<int64_t *>(output_addr), op, n, offset,
          stride, max_threads, target_threads, use_parallel);
    case 5:
      return cpu_reduce_strided_typed<double>(
          values_ptr, reinterpret_cast<double *>(output_addr), op, n, offset,
          stride, max_threads, target_threads, use_parallel);
  }
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::cpu_reduce_strided_ndarray(Ndarray *values,
                                                Ndarray *output,
                                                int value_type,
                                                std::size_t values_offset,
                                                std::size_t values_stride,
                                                std::size_t output_offset,
                                                std::size_t output_stride,
                                                int op) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native strided reduce is only available on CPU backends.");
  check_reduce_strided_request("CPU native", values, output, value_type,
                               values_offset, values_stride, output_offset,
                               output_stride, op);

  const std::size_t n = values->get_nelement();
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = n >= 65536 && target_threads > 1;

  const auto values_addr = get_ndarray_data_ptr_as_int(values);
  const auto output_addr = get_ndarray_data_ptr_as_int(output);
  TI_ERROR_IF(!values_addr || !output_addr,
              "CPU native strided reduce received a null data pointer.");
  const auto *values_ptr = reinterpret_cast<const uint8_t *>(values_addr);
  auto *output_ptr = reinterpret_cast<uint8_t *>(output_addr + output_offset);
  switch (value_type) {
    case 0:
      return cpu_reduce_strided_typed<int32_t>(
          values_ptr, reinterpret_cast<int32_t *>(output_ptr), op, n,
          values_offset, values_stride, max_threads, target_threads,
          use_parallel);
    case 1:
      return cpu_reduce_strided_typed<float>(
          values_ptr, reinterpret_cast<float *>(output_ptr), op, n,
          values_offset, values_stride, max_threads, target_threads,
          use_parallel);
    case 2:
      return cpu_reduce_strided_typed<uint32_t>(
          values_ptr, reinterpret_cast<uint32_t *>(output_ptr), op, n,
          values_offset, values_stride, max_threads, target_threads,
          use_parallel);
    case 3:
      return cpu_reduce_strided_typed<uint64_t>(
          values_ptr, reinterpret_cast<uint64_t *>(output_ptr), op, n,
          values_offset, values_stride, max_threads, target_threads,
          use_parallel);
    case 4:
      return cpu_reduce_strided_typed<int64_t>(
          values_ptr, reinterpret_cast<int64_t *>(output_ptr), op, n,
          values_offset, values_stride, max_threads, target_threads,
          use_parallel);
    case 5:
      return cpu_reduce_strided_typed<double>(
          values_ptr, reinterpret_cast<double *>(output_ptr), op, n,
          values_offset, values_stride, max_threads, target_threads,
          use_parallel);
  }
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::cpu_reduce_workspace_bytes() const {
  return 0;
}

bool Program::cpu_transform_available() const {
  return arch_is_cpu(compile_config().arch);
}

std::size_t Program::cpu_transform_affine_ndarray(Ndarray *src,
                                                  Ndarray *dst,
                                                  int value_type,
                                                  double scale,
                                                  double bias) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native transform is only available on CPU backends.");
  TI_ERROR_IF(!src || !dst, "CPU native transform received a null ndarray.");
  TI_ERROR_IF(src->get_nelement() != dst->get_nelement(),
              "CPU native transform source and destination sizes differ.");
  TI_ERROR_IF(src->get_element_size() != dst->get_element_size(),
              "CPU native transform source and destination dtypes differ.");
  TI_ERROR_IF(value_type < 0 || value_type > 5,
              "CPU native transform received an unsupported value type.");
  const bool is_64bit = value_type == 3 || value_type == 4 || value_type == 5;
  TI_ERROR_IF(src->get_element_size() !=
                  (is_64bit ? sizeof(uint64_t) : sizeof(uint32_t)),
              "CPU native transform dtype does not match value type.");

  const std::size_t n = src->get_nelement();
  if (n == 0) {
    return 0;
  }
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = n >= 65536 && target_threads > 1;

  const auto src_addr = get_ndarray_data_ptr_as_int(src);
  const auto dst_addr = get_ndarray_data_ptr_as_int(dst);
  TI_ERROR_IF(!src_addr || !dst_addr,
              "CPU native transform received a null data pointer.");
  switch (value_type) {
    case 0:
      cpu_transform_run_typed<uint32_t>(
          reinterpret_cast<const uint32_t *>(src_addr),
          reinterpret_cast<uint32_t *>(dst_addr), n,
          static_cast<uint32_t>(static_cast<int32_t>(scale)),
          static_cast<uint32_t>(static_cast<int32_t>(bias)), use_parallel,
          target_threads, max_threads);
      return 0;
    case 2:
      cpu_transform_run_typed<uint32_t>(
          reinterpret_cast<const uint32_t *>(src_addr),
          reinterpret_cast<uint32_t *>(dst_addr), n,
          static_cast<uint32_t>(scale), static_cast<uint32_t>(bias),
          use_parallel, target_threads, max_threads);
      return 0;
    case 1:
      cpu_transform_run_typed<float>(
          reinterpret_cast<const float *>(src_addr),
          reinterpret_cast<float *>(dst_addr), n, static_cast<float>(scale),
          static_cast<float>(bias), use_parallel, target_threads, max_threads);
      return 0;
    case 3:
      cpu_transform_run_typed<uint64_t>(
          reinterpret_cast<const uint64_t *>(src_addr),
          reinterpret_cast<uint64_t *>(dst_addr), n,
          static_cast<uint64_t>(scale), static_cast<uint64_t>(bias),
          use_parallel, target_threads, max_threads);
      return 0;
    case 4:
      cpu_transform_run_typed<uint64_t>(
          reinterpret_cast<const uint64_t *>(src_addr),
          reinterpret_cast<uint64_t *>(dst_addr), n,
          static_cast<uint64_t>(static_cast<int64_t>(scale)),
          static_cast<uint64_t>(static_cast<int64_t>(bias)), use_parallel,
          target_threads, max_threads);
      return 0;
    case 5:
      cpu_transform_run_typed<double>(
          reinterpret_cast<const double *>(src_addr),
          reinterpret_cast<double *>(dst_addr), n, scale, bias, use_parallel,
          target_threads, max_threads);
      return 0;
    default:
      TI_ERROR("CPU native transform received an unsupported value type.");
  }
  return 0;
}

std::size_t Program::cpu_transform_affine_member_ndarray(Ndarray *src,
                                                         Ndarray *dst,
                                                         int value_type,
                                                         std::size_t offset,
                                                         std::size_t stride,
                                                         double scale,
                                                         double bias) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native strided transform is only available on CPU "
              "backends.");
  check_transform_member_request("CPU native", src, dst, value_type, offset,
                                 stride);
  const std::size_t n = src->get_nelement();
  if (n == 0) {
    return 0;
  }
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = n >= 65536 && target_threads > 1;

  const auto src_addr = get_ndarray_data_ptr_as_int(src);
  const auto dst_addr = get_ndarray_data_ptr_as_int(dst);
  TI_ERROR_IF(!src_addr || !dst_addr,
              "CPU native strided transform received a null data pointer.");
  const auto *src_bytes = reinterpret_cast<const uint8_t *>(src_addr);
  switch (value_type) {
    case 0:
      cpu_transform_run_strided_typed<uint32_t>(
          src_bytes, reinterpret_cast<uint32_t *>(dst_addr), n, offset, stride,
          static_cast<uint32_t>(static_cast<int32_t>(scale)),
          static_cast<uint32_t>(static_cast<int32_t>(bias)), use_parallel,
          target_threads, max_threads);
      return 0;
    case 2:
      cpu_transform_run_strided_typed<uint32_t>(
          src_bytes, reinterpret_cast<uint32_t *>(dst_addr), n, offset, stride,
          static_cast<uint32_t>(scale), static_cast<uint32_t>(bias),
          use_parallel, target_threads, max_threads);
      return 0;
    case 1:
      cpu_transform_run_strided_typed<float>(
          src_bytes, reinterpret_cast<float *>(dst_addr), n, offset, stride,
          static_cast<float>(scale), static_cast<float>(bias), use_parallel,
          target_threads, max_threads);
      return 0;
    case 3:
      cpu_transform_run_strided_typed<uint64_t>(
          src_bytes, reinterpret_cast<uint64_t *>(dst_addr), n, offset, stride,
          static_cast<uint64_t>(scale), static_cast<uint64_t>(bias),
          use_parallel, target_threads, max_threads);
      return 0;
    case 4:
      cpu_transform_run_strided_typed<uint64_t>(
          src_bytes, reinterpret_cast<uint64_t *>(dst_addr), n, offset, stride,
          static_cast<uint64_t>(static_cast<int64_t>(scale)),
          static_cast<uint64_t>(static_cast<int64_t>(bias)), use_parallel,
          target_threads, max_threads);
      return 0;
    case 5:
      cpu_transform_run_strided_typed<double>(
          src_bytes, reinterpret_cast<double *>(dst_addr), n, offset, stride,
          scale, bias, use_parallel, target_threads, max_threads);
      return 0;
    default:
      TI_ERROR("CPU native strided transform received an unsupported value "
               "type.");
  }
  return 0;
}

std::size_t Program::cpu_transform_affine_strided_ndarray(
    Ndarray *src,
    Ndarray *dst,
    int value_type,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride,
    double scale,
    double bias) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native strided transform is only available on CPU "
              "backends.");
  check_transform_strided_request("CPU native", src, dst, value_type,
                                  src_offset, src_stride, dst_offset,
                                  dst_stride);
  const std::size_t n = src->get_nelement();
  if (n == 0) {
    return 0;
  }
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  const bool use_parallel = n >= 65536 && target_threads > 1;

  const auto src_addr = get_ndarray_data_ptr_as_int(src);
  const auto dst_addr = get_ndarray_data_ptr_as_int(dst);
  TI_ERROR_IF(!src_addr || !dst_addr,
              "CPU native strided transform received a null data pointer.");
  const auto *src_bytes = reinterpret_cast<const uint8_t *>(src_addr);
  auto *dst_bytes = reinterpret_cast<uint8_t *>(dst_addr);
  switch (value_type) {
    case 0:
      cpu_transform_run_strided_to_strided_typed<uint32_t>(
          src_bytes, dst_bytes, n, src_offset, src_stride, dst_offset,
          dst_stride, static_cast<uint32_t>(static_cast<int32_t>(scale)),
          static_cast<uint32_t>(static_cast<int32_t>(bias)), use_parallel,
          target_threads, max_threads);
      return 0;
    case 2:
      cpu_transform_run_strided_to_strided_typed<uint32_t>(
          src_bytes, dst_bytes, n, src_offset, src_stride, dst_offset,
          dst_stride, static_cast<uint32_t>(scale), static_cast<uint32_t>(bias),
          use_parallel, target_threads, max_threads);
      return 0;
    case 1:
      cpu_transform_run_strided_to_strided_typed<float>(
          src_bytes, dst_bytes, n, src_offset, src_stride, dst_offset,
          dst_stride, static_cast<float>(scale), static_cast<float>(bias),
          use_parallel, target_threads, max_threads);
      return 0;
    case 3:
      cpu_transform_run_strided_to_strided_typed<uint64_t>(
          src_bytes, dst_bytes, n, src_offset, src_stride, dst_offset,
          dst_stride, static_cast<uint64_t>(scale), static_cast<uint64_t>(bias),
          use_parallel, target_threads, max_threads);
      return 0;
    case 4:
      cpu_transform_run_strided_to_strided_typed<uint64_t>(
          src_bytes, dst_bytes, n, src_offset, src_stride, dst_offset,
          dst_stride, static_cast<uint64_t>(static_cast<int64_t>(scale)),
          static_cast<uint64_t>(static_cast<int64_t>(bias)), use_parallel,
          target_threads, max_threads);
      return 0;
    case 5:
      cpu_transform_run_strided_to_strided_typed<double>(
          src_bytes, dst_bytes, n, src_offset, src_stride, dst_offset,
          dst_stride, scale, bias, use_parallel, target_threads, max_threads);
      return 0;
    default:
      TI_ERROR("CPU native strided transform received an unsupported value "
               "type.");
  }
  return 0;
}

std::size_t Program::cpu_transform_workspace_bytes() const {
  return 0;
}

bool Program::cpu_indexed_copy_available() const {
  return arch_is_cpu(compile_config().arch);
}

std::size_t Program::cpu_gather_ndarray(Ndarray *src,
                                        Ndarray *indices,
                                        Ndarray *dst) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native gather is only available on CPU backends.");
  TI_ERROR_IF(!src || !indices || !dst,
              "CPU native gather received a null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "CPU native gather currently expects 1D ndarrays.");
  TI_ERROR_IF(indices->get_nelement() != dst->get_nelement(),
              "CPU native gather expects indices and destination sizes to "
              "match.");
  TI_ERROR_IF(src->get_element_size() != dst->get_element_size(),
              "CPU native gather source and destination dtypes differ.");
  const std::size_t item_bytes = src->get_element_size();
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(uint32_t) != 0 ||
                  indices->get_element_size() != sizeof(int32_t),
              "CPU native gather currently expects 4-byte aligned values and "
              "i32 indices.");
  const std::size_t n = indices->get_nelement();
  const std::size_t src_items = src->get_nelement();
  if (n == 0) {
    return 0;
  }
  auto *src_ptr =
      reinterpret_cast<const uint8_t *>(get_ndarray_data_ptr_as_int(src));
  auto *indices_ptr =
      reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(indices));
  auto *dst_ptr = reinterpret_cast<uint8_t *>(get_ndarray_data_ptr_as_int(dst));
  TI_ERROR_IF(!src_ptr || !indices_ptr || !dst_ptr,
              "CPU native gather received a null data pointer.");
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  if (n >= 65536 && target_threads > 1) {
    CpuIndexedCopyTaskContext ctx;
    ctx.src = src_ptr;
    ctx.indices = indices_ptr;
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.index_bound = src_items;
    ctx.item_bytes = item_bytes;
    ctx.scatter = false;
    ctx.num_threads = target_threads;
    auto &pool = get_cpu_primitive_thread_pool(max_threads);
    pool.run(target_threads, target_threads, &ctx, cpu_indexed_copy_task);
    return 0;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto index = static_cast<std::size_t>(indices_ptr[i]);
    if (index < src_items) {
      std::memcpy(dst_ptr + i * item_bytes, src_ptr + index * item_bytes,
                  item_bytes);
    } else {
      std::memset(dst_ptr + i * item_bytes, 0, item_bytes);
    }
  }
  return 0;
}

std::size_t Program::cpu_gather_strided_ndarray(Ndarray *src,
                                                Ndarray *indices,
                                                Ndarray *dst,
                                                std::size_t item_bytes,
                                                std::size_t src_offset,
                                                std::size_t src_stride,
                                                std::size_t dst_offset,
                                                std::size_t dst_stride) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native strided gather is only available on CPU backends.");
  check_indexed_copy_strided_request("CPU native", src, indices, dst,
                                     item_bytes, src_offset, src_stride,
                                     dst_offset, dst_stride, false);
  const std::size_t n = indices->get_nelement();
  const std::size_t src_items = src->get_nelement();
  if (n == 0) {
    return 0;
  }
  auto *src_ptr =
      reinterpret_cast<const uint8_t *>(get_ndarray_data_ptr_as_int(src));
  auto *indices_ptr =
      reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(indices));
  auto *dst_ptr = reinterpret_cast<uint8_t *>(get_ndarray_data_ptr_as_int(dst));
  TI_ERROR_IF(!src_ptr || !indices_ptr || !dst_ptr,
              "CPU native strided gather received a null data pointer.");
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  if (n >= 65536 && target_threads > 1) {
    CpuStridedIndexedCopyTaskContext ctx;
    ctx.src = src_ptr;
    ctx.indices = indices_ptr;
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.index_bound = src_items;
    ctx.item_bytes = item_bytes;
    ctx.src_offset = src_offset;
    ctx.src_stride = src_stride;
    ctx.dst_offset = dst_offset;
    ctx.dst_stride = dst_stride;
    ctx.scatter = false;
    ctx.num_threads = target_threads;
    auto &pool = get_cpu_primitive_thread_pool(max_threads);
    pool.run(target_threads, target_threads, &ctx,
             cpu_strided_indexed_copy_task);
    return 0;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto index = static_cast<std::size_t>(indices_ptr[i]);
    if (index < src_items) {
      std::memcpy(dst_ptr + dst_offset + i * dst_stride,
                  src_ptr + src_offset + index * src_stride, item_bytes);
    } else {
      std::memset(dst_ptr + dst_offset + i * dst_stride, 0, item_bytes);
    }
  }
  return 0;
}

std::size_t Program::cpu_scatter_ndarray(Ndarray *src,
                                         Ndarray *indices,
                                         Ndarray *dst) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native scatter is only available on CPU backends.");
  TI_ERROR_IF(!src || !indices || !dst,
              "CPU native scatter received a null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "CPU native scatter currently expects 1D ndarrays.");
  TI_ERROR_IF(src->get_nelement() != indices->get_nelement(),
              "CPU native scatter expects source and indices sizes to match.");
  TI_ERROR_IF(src->get_element_size() != dst->get_element_size(),
              "CPU native scatter source and destination dtypes differ.");
  const std::size_t item_bytes = src->get_element_size();
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(uint32_t) != 0 ||
                  indices->get_element_size() != sizeof(int32_t),
              "CPU native scatter currently expects 4-byte aligned values and "
              "i32 indices.");
  const std::size_t n = indices->get_nelement();
  const std::size_t dst_items = dst->get_nelement();
  if (n == 0) {
    return 0;
  }
  auto *src_ptr =
      reinterpret_cast<const uint8_t *>(get_ndarray_data_ptr_as_int(src));
  auto *indices_ptr =
      reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(indices));
  auto *dst_ptr = reinterpret_cast<uint8_t *>(get_ndarray_data_ptr_as_int(dst));
  TI_ERROR_IF(!src_ptr || !indices_ptr || !dst_ptr,
              "CPU native scatter received a null data pointer.");
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  if (n >= 65536 && target_threads > 1) {
    CpuIndexedCopyTaskContext ctx;
    ctx.src = src_ptr;
    ctx.indices = indices_ptr;
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.index_bound = dst_items;
    ctx.item_bytes = item_bytes;
    ctx.scatter = true;
    ctx.num_threads = target_threads;
    auto &pool = get_cpu_primitive_thread_pool(max_threads);
    pool.run(target_threads, target_threads, &ctx, cpu_indexed_copy_task);
    return 0;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto index = static_cast<std::size_t>(indices_ptr[i]);
    if (index < dst_items) {
      std::memcpy(dst_ptr + index * item_bytes, src_ptr + i * item_bytes,
                  item_bytes);
    }
  }
  return 0;
}

std::size_t Program::cpu_scatter_strided_ndarray(Ndarray *src,
                                                 Ndarray *indices,
                                                 Ndarray *dst,
                                                 std::size_t item_bytes,
                                                 std::size_t src_offset,
                                                 std::size_t src_stride,
                                                 std::size_t dst_offset,
                                                 std::size_t dst_stride) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native strided scatter is only available on CPU backends.");
  check_indexed_copy_strided_request("CPU native", src, indices, dst,
                                     item_bytes, src_offset, src_stride,
                                     dst_offset, dst_stride, true);
  const std::size_t n = indices->get_nelement();
  const std::size_t dst_items = dst->get_nelement();
  if (n == 0) {
    return 0;
  }
  auto *src_ptr =
      reinterpret_cast<const uint8_t *>(get_ndarray_data_ptr_as_int(src));
  auto *indices_ptr =
      reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(indices));
  auto *dst_ptr = reinterpret_cast<uint8_t *>(get_ndarray_data_ptr_as_int(dst));
  TI_ERROR_IF(!src_ptr || !indices_ptr || !dst_ptr,
              "CPU native strided scatter received a null data pointer.");
  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  const int chunk_items = 32768;
  const int target_threads = static_cast<int>(
      std::min<std::size_t>((n + chunk_items - 1) / chunk_items,
                            static_cast<std::size_t>(max_threads)));
  if (n >= 65536 && target_threads > 1) {
    CpuStridedIndexedCopyTaskContext ctx;
    ctx.src = src_ptr;
    ctx.indices = indices_ptr;
    ctx.dst = dst_ptr;
    ctx.n = n;
    ctx.index_bound = dst_items;
    ctx.item_bytes = item_bytes;
    ctx.src_offset = src_offset;
    ctx.src_stride = src_stride;
    ctx.dst_offset = dst_offset;
    ctx.dst_stride = dst_stride;
    ctx.scatter = true;
    ctx.num_threads = target_threads;
    auto &pool = get_cpu_primitive_thread_pool(max_threads);
    pool.run(target_threads, target_threads, &ctx,
             cpu_strided_indexed_copy_task);
    return 0;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto index = static_cast<std::size_t>(indices_ptr[i]);
    if (index < dst_items) {
      std::memcpy(dst_ptr + dst_offset + index * dst_stride,
                  src_ptr + src_offset + i * src_stride, item_bytes);
    }
  }
  return 0;
}

std::size_t Program::cpu_indexed_copy_workspace_bytes() const {
  return 0;
}

bool Program::cpu_scatter_add_available() const {
  return arch_is_cpu(compile_config().arch);
}

std::size_t Program::cpu_scatter_add_ndarray(Ndarray *src,
                                             Ndarray *indices,
                                             Ndarray *dst,
                                             int value_type) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native scatter-add is only available on CPU backends.");
  TI_ERROR_IF(!src || !indices || !dst,
              "CPU native scatter-add received a null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "CPU native scatter-add currently expects 1D ndarrays.");
  TI_ERROR_IF(src->get_nelement() != indices->get_nelement(),
              "CPU native scatter-add expects source and indices sizes to "
              "match.");
  TI_ERROR_IF(src->get_element_size() != dst->get_element_size(),
              "CPU native scatter-add source and destination dtypes differ.");
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(expected_size == 0,
              "CPU native scatter-add received an unsupported value type.");
  TI_ERROR_IF(src->get_element_size() != expected_size ||
                  indices->get_element_size() != sizeof(int32_t),
              "CPU native scatter-add value type or i32 index size mismatch.");
  const std::size_t n = indices->get_nelement();
  const std::size_t dst_items = dst->get_nelement();
  if (n == 0 || dst_items == 0) {
    return 0;
  }
  auto *indices_ptr =
      reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(indices));
  TI_ERROR_IF(!indices_ptr,
              "CPU native scatter-add received a null index pointer.");
  const int max_threads = std::max(1, compile_config().cpu_max_num_threads);
  const int target_threads =
      std::min(max_threads, std::max(1, static_cast<int>(n / 65536)));
  switch (value_type) {
    case 0:
      return cpu_scatter_add_typed(
          reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(src)),
          indices_ptr,
          reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(dst)), n,
          dst_items, max_threads, target_threads);
    case 1:
      return cpu_scatter_add_typed(
          reinterpret_cast<const float *>(get_ndarray_data_ptr_as_int(src)),
          indices_ptr, reinterpret_cast<float *>(get_ndarray_data_ptr_as_int(dst)),
          n, dst_items, max_threads, target_threads);
    case 2:
      return cpu_scatter_add_typed(
          reinterpret_cast<const uint32_t *>(get_ndarray_data_ptr_as_int(src)),
          indices_ptr,
          reinterpret_cast<uint32_t *>(get_ndarray_data_ptr_as_int(dst)), n,
          dst_items, max_threads, target_threads);
    case 3:
      return cpu_scatter_add_typed(
          reinterpret_cast<const uint64_t *>(get_ndarray_data_ptr_as_int(src)),
          indices_ptr,
          reinterpret_cast<uint64_t *>(get_ndarray_data_ptr_as_int(dst)), n,
          dst_items, max_threads, target_threads);
    case 4:
      return cpu_scatter_add_typed(
          reinterpret_cast<const int64_t *>(get_ndarray_data_ptr_as_int(src)),
          indices_ptr,
          reinterpret_cast<int64_t *>(get_ndarray_data_ptr_as_int(dst)), n,
          dst_items, max_threads, target_threads);
    case 5:
      return cpu_scatter_add_typed(
          reinterpret_cast<const double *>(get_ndarray_data_ptr_as_int(src)),
          indices_ptr,
          reinterpret_cast<double *>(get_ndarray_data_ptr_as_int(dst)), n,
          dst_items, max_threads, target_threads);
    default:
      TI_ERROR("CPU native scatter-add received an unsupported value type.");
  }
  return 0;
}

std::size_t Program::cpu_scatter_add_member_ndarray(Ndarray *src,
                                                    Ndarray *indices,
                                                    Ndarray *dst,
                                                    int value_type,
                                                    std::size_t offset,
                                                    std::size_t stride) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native strided scatter-add is only available on CPU "
              "backends.");
  check_scatter_add_member_request("CPU native", src, indices, dst, value_type,
                                   offset, stride);
  const std::size_t n = indices->get_nelement();
  const std::size_t dst_items = dst->get_nelement();
  if (n == 0 || dst_items == 0) {
    return 0;
  }
  auto *src_ptr =
      reinterpret_cast<const uint8_t *>(get_ndarray_data_ptr_as_int(src));
  auto *indices_ptr =
      reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(indices));
  TI_ERROR_IF(!src_ptr || !indices_ptr,
              "CPU native strided scatter-add received a null data pointer.");
  const int max_threads = std::max(1, compile_config().cpu_max_num_threads);
  const int target_threads =
      std::min(max_threads, std::max(1, static_cast<int>(n / 65536)));
  switch (value_type) {
    case 0:
      return cpu_scatter_add_strided_typed<int32_t>(
          src_ptr, offset, stride, indices_ptr,
          reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(dst)), n,
          dst_items, max_threads, target_threads);
    case 1:
      return cpu_scatter_add_strided_typed<float>(
          src_ptr, offset, stride, indices_ptr,
          reinterpret_cast<float *>(get_ndarray_data_ptr_as_int(dst)), n,
          dst_items, max_threads, target_threads);
    case 2:
      return cpu_scatter_add_strided_typed<uint32_t>(
          src_ptr, offset, stride, indices_ptr,
          reinterpret_cast<uint32_t *>(get_ndarray_data_ptr_as_int(dst)), n,
          dst_items, max_threads, target_threads);
    case 3:
      return cpu_scatter_add_strided_typed<uint64_t>(
          src_ptr, offset, stride, indices_ptr,
          reinterpret_cast<uint64_t *>(get_ndarray_data_ptr_as_int(dst)), n,
          dst_items, max_threads, target_threads);
    case 4:
      return cpu_scatter_add_strided_typed<int64_t>(
          src_ptr, offset, stride, indices_ptr,
          reinterpret_cast<int64_t *>(get_ndarray_data_ptr_as_int(dst)), n,
          dst_items, max_threads, target_threads);
    case 5:
      return cpu_scatter_add_strided_typed<double>(
          src_ptr, offset, stride, indices_ptr,
          reinterpret_cast<double *>(get_ndarray_data_ptr_as_int(dst)), n,
          dst_items, max_threads, target_threads);
    default:
      TI_ERROR(
          "CPU native strided scatter-add received an unsupported value type.");
  }
  return 0;
}

std::size_t Program::cpu_scatter_add_strided_ndarray(
    Ndarray *src,
    Ndarray *indices,
    Ndarray *dst,
    int value_type,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native strided scatter-add is only available on CPU "
              "backends.");
  check_scatter_add_strided_request("CPU native", src, indices, dst,
                                    value_type, src_offset, src_stride,
                                    dst_offset, dst_stride);
  const std::size_t n = indices->get_nelement();
  const std::size_t dst_items = dst->get_nelement();
  if (n == 0 || dst_items == 0) {
    return 0;
  }
  auto *src_ptr =
      reinterpret_cast<const uint8_t *>(get_ndarray_data_ptr_as_int(src));
  auto *indices_ptr =
      reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(indices));
  auto *dst_ptr = reinterpret_cast<uint8_t *>(get_ndarray_data_ptr_as_int(dst));
  TI_ERROR_IF(!src_ptr || !indices_ptr || !dst_ptr,
              "CPU native strided scatter-add received a null data pointer.");
  const int max_threads = std::max(1, compile_config().cpu_max_num_threads);
  const int target_threads =
      std::min(max_threads, std::max(1, static_cast<int>(n / 65536)));
  switch (value_type) {
    case 0:
      return cpu_scatter_add_strided_io_typed<int32_t>(
          src_ptr, src_offset, src_stride, indices_ptr, dst_ptr, dst_offset,
          dst_stride, n, dst_items, max_threads, target_threads);
    case 1:
      return cpu_scatter_add_strided_io_typed<float>(
          src_ptr, src_offset, src_stride, indices_ptr, dst_ptr, dst_offset,
          dst_stride, n, dst_items, max_threads, target_threads);
    case 2:
      return cpu_scatter_add_strided_io_typed<uint32_t>(
          src_ptr, src_offset, src_stride, indices_ptr, dst_ptr, dst_offset,
          dst_stride, n, dst_items, max_threads, target_threads);
    case 3:
      return cpu_scatter_add_strided_io_typed<uint64_t>(
          src_ptr, src_offset, src_stride, indices_ptr, dst_ptr, dst_offset,
          dst_stride, n, dst_items, max_threads, target_threads);
    case 4:
      return cpu_scatter_add_strided_io_typed<int64_t>(
          src_ptr, src_offset, src_stride, indices_ptr, dst_ptr, dst_offset,
          dst_stride, n, dst_items, max_threads, target_threads);
    case 5:
      return cpu_scatter_add_strided_io_typed<double>(
          src_ptr, src_offset, src_stride, indices_ptr, dst_ptr, dst_offset,
          dst_stride, n, dst_items, max_threads, target_threads);
    default:
      TI_ERROR(
          "CPU native strided scatter-add received an unsupported value type.");
  }
  return 0;
}

std::size_t Program::cpu_scatter_add_workspace_bytes() const {
  return cpu_scatter_add_workspace_bytes_peak.load(std::memory_order_relaxed);
}

bool Program::cpu_bucket_builder_available() const {
  return arch_is_cpu(compile_config().arch);
}

std::size_t Program::cpu_bucket_builder_i32_ndarray(Ndarray *keys,
                                                    Ndarray *values,
                                                    Ndarray *offsets,
                                                    Ndarray *output) {
  return cpu_bucket_builder_ndarray(keys, values, offsets, output, 0);
}

std::size_t Program::cpu_bucket_builder_ndarray(Ndarray *keys,
                                                Ndarray *values,
                                                Ndarray *offsets,
                                                Ndarray *output,
                                                int value_type) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native bucket builder is only available on CPU backends.");
  TI_ERROR_IF(!keys || !values || !offsets || !output,
              "CPU native bucket builder received a null ndarray.");
  TI_ERROR_IF(keys->shape.size() != 1 || values->shape.size() != 1 ||
                  offsets->shape.size() != 1 || output->shape.size() != 1,
              "CPU native bucket builder expects 1D ndarrays.");
  TI_ERROR_IF(keys->get_nelement() != values->get_nelement(),
              "CPU native bucket builder keys and values sizes differ.");
  TI_ERROR_IF(offsets->get_nelement() < 2,
              "CPU native bucket builder offsets must contain num_bins + 1 items.");
  const std::size_t n = keys->get_nelement();
  const std::size_t num_bins = offsets->get_nelement() - 1;
  TI_ERROR_IF(output->get_nelement() < n,
              "CPU native bucket builder output is smaller than input values.");
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(expected_size == 0,
              "CPU native bucket builder received an unsupported value type.");
  const std::size_t item_bytes = values->get_element_size();
  TI_ERROR_IF(keys->get_element_size() != sizeof(int32_t) ||
                  offsets->get_element_size() != sizeof(int32_t) ||
                  item_bytes == 0 ||
                  item_bytes % sizeof(uint32_t) != 0 ||
                  output->get_element_size() != item_bytes,
              "CPU native bucket builder dtype does not match value type, "
              "keys/offsets are not i32, or payload is not 4-byte aligned.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()),
              "CPU native bucket builder input count exceeds i32 range.");

  auto *keys_ptr =
      reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(keys));
  auto *offsets_ptr =
      reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(offsets));
  TI_ERROR_IF(!keys_ptr || !offsets_ptr,
              "CPU native bucket builder received a null data pointer.");

  const int max_threads =
      std::max(1, static_cast<int>(compile_config().cpu_max_num_threads));
  if (item_bytes != expected_size) {
    return cpu_bucket_builder_raw(
        keys_ptr,
        reinterpret_cast<const uint8_t *>(get_ndarray_data_ptr_as_int(values)),
        offsets_ptr,
        reinterpret_cast<uint8_t *>(get_ndarray_data_ptr_as_int(output)), n,
        num_bins, item_bytes, max_threads);
  }
  switch (value_type) {
    case 0:
      return cpu_bucket_builder_typed(
          keys_ptr,
          reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(values)),
          offsets_ptr,
          reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(output)), n,
          num_bins, max_threads);
    case 1:
      return cpu_bucket_builder_typed(
          keys_ptr,
          reinterpret_cast<const float *>(get_ndarray_data_ptr_as_int(values)),
          offsets_ptr,
          reinterpret_cast<float *>(get_ndarray_data_ptr_as_int(output)), n,
          num_bins, max_threads);
    case 2:
      return cpu_bucket_builder_typed(
          keys_ptr,
          reinterpret_cast<const uint32_t *>(get_ndarray_data_ptr_as_int(values)),
          offsets_ptr,
          reinterpret_cast<uint32_t *>(get_ndarray_data_ptr_as_int(output)), n,
          num_bins, max_threads);
    case 3:
      return cpu_bucket_builder_typed(
          keys_ptr,
          reinterpret_cast<const uint64_t *>(get_ndarray_data_ptr_as_int(values)),
          offsets_ptr,
          reinterpret_cast<uint64_t *>(get_ndarray_data_ptr_as_int(output)), n,
          num_bins, max_threads);
    case 4:
      return cpu_bucket_builder_typed(
          keys_ptr,
          reinterpret_cast<const int64_t *>(get_ndarray_data_ptr_as_int(values)),
          offsets_ptr,
          reinterpret_cast<int64_t *>(get_ndarray_data_ptr_as_int(output)), n,
          num_bins, max_threads);
    case 5:
      return cpu_bucket_builder_typed(
          keys_ptr,
          reinterpret_cast<const double *>(get_ndarray_data_ptr_as_int(values)),
          offsets_ptr,
          reinterpret_cast<double *>(get_ndarray_data_ptr_as_int(output)), n,
          num_bins, max_threads);
    default:
      TI_ERROR("CPU native bucket builder received an unsupported value type.");
  }
  return 0;
}

std::size_t Program::cpu_bucket_builder_workspace_bytes() const {
  return 0;
}

bool Program::cpu_grouped_reduce_available() const {
  return arch_is_cpu(compile_config().arch);
}

std::size_t Program::cpu_grouped_reduce_i32_ndarray(Ndarray *keys,
                                                    Ndarray *values,
                                                    Ndarray *output,
                                                    int op) {
  return cpu_grouped_reduce_ndarray(keys, values, output, 0, op);
}

std::size_t Program::cpu_grouped_reduce_ndarray(Ndarray *keys,
                                                Ndarray *values,
                                                Ndarray *output,
                                                int value_type,
                                                int op) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native grouped reduce is only available on CPU backends.");
  TI_ERROR_IF(!keys || !values || !output,
              "CPU native grouped reduce received a null ndarray.");
  TI_ERROR_IF(keys->shape.size() != 1 || values->shape.size() != 1 ||
                  output->shape.size() != 1,
              "CPU native grouped reduce expects 1D ndarrays.");
  TI_ERROR_IF(keys->get_nelement() != values->get_nelement(),
              "CPU native grouped reduce keys and values sizes differ.");
  TI_ERROR_IF(output->get_nelement() == 0,
              "CPU native grouped reduce output must contain at least one group.");
  const std::size_t expected_size = primitive_value_type_size(value_type);
  TI_ERROR_IF(expected_size == 0,
              "CPU native grouped reduce received an unsupported value type.");
  TI_ERROR_IF(keys->get_element_size() != sizeof(int32_t) ||
                  values->get_element_size() != expected_size ||
                  output->get_element_size() != expected_size,
              "CPU native grouped reduce value type or i32 key size mismatch.");
  TI_ERROR_IF(op != 0, "CPU native grouped reduce currently supports only sum.");
  const std::size_t n = keys->get_nelement();
  const std::size_t num_groups = output->get_nelement();
  auto *keys_ptr =
      reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(keys));
  const int max_threads = std::max(1, compile_config().cpu_max_num_threads);
  const int target_threads =
      std::min(max_threads, std::max(1, static_cast<int>(n / 65536)));
  switch (value_type) {
    case 0:
      return cpu_grouped_reduce_typed(
          keys_ptr,
          reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(values)),
          reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(output)), n,
          num_groups, max_threads, target_threads);
    case 1:
      return cpu_grouped_reduce_typed(
          keys_ptr,
          reinterpret_cast<const float *>(get_ndarray_data_ptr_as_int(values)),
          reinterpret_cast<float *>(get_ndarray_data_ptr_as_int(output)), n,
          num_groups, max_threads, target_threads);
    case 2:
      return cpu_grouped_reduce_typed(
          keys_ptr,
          reinterpret_cast<const uint32_t *>(get_ndarray_data_ptr_as_int(values)),
          reinterpret_cast<uint32_t *>(get_ndarray_data_ptr_as_int(output)), n,
          num_groups, max_threads, target_threads);
    case 3:
      return cpu_grouped_reduce_typed(
          keys_ptr,
          reinterpret_cast<const uint64_t *>(get_ndarray_data_ptr_as_int(values)),
          reinterpret_cast<uint64_t *>(get_ndarray_data_ptr_as_int(output)), n,
          num_groups, max_threads, target_threads);
    case 4:
      return cpu_grouped_reduce_typed(
          keys_ptr,
          reinterpret_cast<const int64_t *>(get_ndarray_data_ptr_as_int(values)),
          reinterpret_cast<int64_t *>(get_ndarray_data_ptr_as_int(output)), n,
          num_groups, max_threads, target_threads);
    case 5:
      return cpu_grouped_reduce_typed(
          keys_ptr,
          reinterpret_cast<const double *>(get_ndarray_data_ptr_as_int(values)),
          reinterpret_cast<double *>(get_ndarray_data_ptr_as_int(output)), n,
          num_groups, max_threads, target_threads);
    default:
      TI_ERROR("CPU native grouped reduce received an unsupported value type.");
  }
  return 0;
}

std::size_t Program::cpu_grouped_reduce_member_ndarray(Ndarray *keys,
                                                       Ndarray *values,
                                                       Ndarray *output,
                                                       int value_type,
                                                       std::size_t offset,
                                                       std::size_t stride,
                                                       int op) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native strided grouped reduce is only available on CPU "
              "backends.");
  check_grouped_reduce_member_request("CPU native", keys, values, output,
                                      value_type, offset, stride, op);
  const std::size_t n = keys->get_nelement();
  const std::size_t num_groups = output->get_nelement();
  auto *keys_ptr =
      reinterpret_cast<const int32_t *>(get_ndarray_data_ptr_as_int(keys));
  auto *values_ptr =
      reinterpret_cast<const uint8_t *>(get_ndarray_data_ptr_as_int(values));
  TI_ERROR_IF(!keys_ptr || !values_ptr,
              "CPU native strided grouped reduce received a null data pointer.");
  const int max_threads = std::max(1, compile_config().cpu_max_num_threads);
  const int target_threads =
      std::min(max_threads, std::max(1, static_cast<int>(n / 65536)));
  switch (value_type) {
    case 0:
      return cpu_grouped_reduce_strided_typed<int32_t>(
          keys_ptr, values_ptr, offset, stride,
          reinterpret_cast<int32_t *>(get_ndarray_data_ptr_as_int(output)), n,
          num_groups, max_threads, target_threads);
    case 1:
      return cpu_grouped_reduce_strided_typed<float>(
          keys_ptr, values_ptr, offset, stride,
          reinterpret_cast<float *>(get_ndarray_data_ptr_as_int(output)), n,
          num_groups, max_threads, target_threads);
    case 2:
      return cpu_grouped_reduce_strided_typed<uint32_t>(
          keys_ptr, values_ptr, offset, stride,
          reinterpret_cast<uint32_t *>(get_ndarray_data_ptr_as_int(output)), n,
          num_groups, max_threads, target_threads);
    case 3:
      return cpu_grouped_reduce_strided_typed<uint64_t>(
          keys_ptr, values_ptr, offset, stride,
          reinterpret_cast<uint64_t *>(get_ndarray_data_ptr_as_int(output)), n,
          num_groups, max_threads, target_threads);
    case 4:
      return cpu_grouped_reduce_strided_typed<int64_t>(
          keys_ptr, values_ptr, offset, stride,
          reinterpret_cast<int64_t *>(get_ndarray_data_ptr_as_int(output)), n,
          num_groups, max_threads, target_threads);
    case 5:
      return cpu_grouped_reduce_strided_typed<double>(
          keys_ptr, values_ptr, offset, stride,
          reinterpret_cast<double *>(get_ndarray_data_ptr_as_int(output)), n,
          num_groups, max_threads, target_threads);
    default:
      TI_ERROR("CPU native strided grouped reduce received an unsupported "
               "value type.");
  }
  return 0;
}

std::size_t Program::cpu_grouped_reduce_strided_ndarray(
    Ndarray *keys,
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t output_offset,
    std::size_t output_stride,
    int op) {
  return cpu_grouped_reduce_strided_keys_ndarray(
      keys, values, output, value_type, 0, sizeof(int32_t), values_offset,
      values_stride, output_offset, output_stride, op);
}

std::size_t Program::cpu_grouped_reduce_strided_keys_ndarray(
    Ndarray *keys,
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t keys_offset,
    std::size_t keys_stride,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t output_offset,
    std::size_t output_stride,
    int op) {
  TI_ERROR_IF(!arch_is_cpu(compile_config().arch),
              "CPU native strided grouped reduce is only available on CPU "
              "backends.");
  check_grouped_reduce_strided_keys_request(
      "CPU native", keys, values, output, value_type, keys_offset, keys_stride,
      values_offset, values_stride, output_offset, output_stride, op);
  const std::size_t n = keys->get_nelement();
  const std::size_t num_groups = output->get_nelement();
  auto *keys_ptr =
      reinterpret_cast<const uint8_t *>(get_ndarray_data_ptr_as_int(keys));
  auto *values_ptr =
      reinterpret_cast<const uint8_t *>(get_ndarray_data_ptr_as_int(values));
  auto *output_ptr =
      reinterpret_cast<uint8_t *>(get_ndarray_data_ptr_as_int(output));
  TI_ERROR_IF(!keys_ptr || !values_ptr || !output_ptr,
              "CPU native strided grouped reduce received a null data pointer.");
  const int max_threads = std::max(1, compile_config().cpu_max_num_threads);
  const int target_threads =
      std::min(max_threads, std::max(1, static_cast<int>(n / 65536)));
  switch (value_type) {
    case 0:
      return cpu_grouped_reduce_strided_io_typed<int32_t>(
          keys_ptr, keys_offset, keys_stride, values_ptr, values_offset,
          values_stride, output_ptr, output_offset, output_stride, n,
          num_groups, max_threads, target_threads);
    case 1:
      return cpu_grouped_reduce_strided_io_typed<float>(
          keys_ptr, keys_offset, keys_stride, values_ptr, values_offset,
          values_stride, output_ptr, output_offset, output_stride, n,
          num_groups, max_threads, target_threads);
    case 2:
      return cpu_grouped_reduce_strided_io_typed<uint32_t>(
          keys_ptr, keys_offset, keys_stride, values_ptr, values_offset,
          values_stride, output_ptr, output_offset, output_stride, n,
          num_groups, max_threads, target_threads);
    case 3:
      return cpu_grouped_reduce_strided_io_typed<uint64_t>(
          keys_ptr, keys_offset, keys_stride, values_ptr, values_offset,
          values_stride, output_ptr, output_offset, output_stride, n,
          num_groups, max_threads, target_threads);
    case 4:
      return cpu_grouped_reduce_strided_io_typed<int64_t>(
          keys_ptr, keys_offset, keys_stride, values_ptr, values_offset,
          values_stride, output_ptr, output_offset, output_stride, n,
          num_groups, max_threads, target_threads);
    case 5:
      return cpu_grouped_reduce_strided_io_typed<double>(
          keys_ptr, keys_offset, keys_stride, values_ptr, values_offset,
          values_stride, output_ptr, output_offset, output_stride, n,
          num_groups, max_threads, target_threads);
    default:
      TI_ERROR(
          "CPU native strided grouped reduce received an unsupported value "
          "type.");
  }
  return 0;
}

std::size_t Program::cpu_grouped_reduce_workspace_bytes() const {
  return cpu_grouped_reduce_workspace_bytes_peak.load(std::memory_order_relaxed);
}

std::pair<const ArgPackType *, size_t>
Program::get_argpack_type_with_data_layout(const ArgPackType *old_ty,
                                           const std::string &layout) {
  // Convert to StructType
  auto *struct_type_old =
      TypeFactory::get_instance()
          .get_struct_type(old_ty->elements(), old_ty->get_layout())
          ->as<StructType>();
  // Call get_struct_type_with_data_layout
  auto [struct_type, size] = program_impl_->get_struct_type_with_data_layout(
      const_cast<StructType *>(struct_type_old), layout);
  // Convert back to ArgPackType
  auto *new_ty =
      TypeFactory::get_instance()
          .get_argpack_type(struct_type->elements(), struct_type->get_layout())
          ->as<ArgPackType>();
  return {new_ty, size};
}

std::pair<const StructType *, size_t> Program::get_struct_type_with_data_layout(
    const StructType *old_ty,
    const std::string &layout) {
  return program_impl_->get_struct_type_with_data_layout(old_ty, layout);
}

Program::~Program() {
  finalize();
}

DeviceCapabilityConfig translate_devcaps(const std::vector<std::string> &caps) {
  // Each device capability assignment is named like this:
  // - `spirv_version=1.3`
  // - `spirv_has_int8`
  DeviceCapabilityConfig cfg{};
  for (const std::string &cap : caps) {
    std::string_view key;
    uint32_t value;
    size_t ieq = cap.find('=');
    if (ieq == std::string::npos) {
      key = cap;
      value = 1;
    } else {
      key = std::string_view(cap.c_str(), ieq);
      value = (uint32_t)std::atol(cap.c_str() + ieq + 1);
    }
    DeviceCapability devcap = str2devcap(key);
    cfg.set(devcap, value);
  }

  // Assign default caps (that always present).
  if (!cfg.contains(DeviceCapability::spirv_version)) {
    cfg.set(DeviceCapability::spirv_version, 0x10300);
  }
  return cfg;
}

std::unique_ptr<AotModuleBuilder> Program::make_aot_module_builder(
    Arch arch,
    const std::vector<std::string> &caps) {
  DeviceCapabilityConfig cfg = translate_devcaps(caps);
  // FIXME: This couples the runtime backend with the target AOT backend. E.g.
  // If we want to build a Metal AOT module, we have to be on the macOS
  // platform. Consider decoupling this part
  if (arch_uses_llvm(compile_config().arch) ||
      compile_config().arch == Arch::metal ||
      compile_config().arch == Arch::vulkan ||
      compile_config().arch == Arch::opengl ||
      compile_config().arch == Arch::gles ||
      compile_config().arch == Arch::dx12) {
    return program_impl_->make_aot_module_builder(cfg);
  }
  return nullptr;
}

int Program::allocate_snode_tree_id() {
  if (free_snode_tree_ids_.empty()) {
    return snode_trees_.size();
  } else {
    int id = free_snode_tree_ids_.top();
    free_snode_tree_ids_.pop();
    return id;
  }
}

void Program::enqueue_compute_op_lambda(
    std::function<void(Device *device, CommandList *cmdlist)> op,
    const std::vector<ComputeOpImageRef> &image_refs) {
  program_impl_->enqueue_compute_op_lambda(op, image_refs);
}

}  // namespace taichi::lang
