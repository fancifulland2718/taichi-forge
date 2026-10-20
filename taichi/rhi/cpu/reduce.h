#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include "taichi/rhi/cpu/parallel_range.h"
#include "taichi/system/threading.h"

namespace taichi::lang::cpu {

// Preserve the selected partition order independently of the consumer's worker
// count. These are algorithm choices, not input-capacity limits.
inline bool use_parallel_aggregation(std::size_t n, int target_threads) {
  return n >= 262144 && target_threads >= 4;
}

inline int reduce_partitions(std::size_t n, int max_threads) {
  const auto chunks = n / 32768 + (n % 32768 != 0);
  const auto target = static_cast<int>(std::min<std::size_t>(
      chunks, static_cast<std::size_t>(std::max(1, max_threads))));
  return use_parallel_aggregation(n, target) ? target : 1;
}

template <typename T>
struct CpuReduceTaskContext {
  const T *values{nullptr};
  T *partial{nullptr};
  std::size_t n{0};
  int num_threads{1};
  int op{0};
};

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
T cpu_reduce_sum_contiguous_range(const T *values,
                                  std::size_t begin,
                                  std::size_t end) {
  if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
    using U = std::make_unsigned_t<T>;
    const auto *unsigned_values = reinterpret_cast<const U *>(values);
    U acc = 0;
    for (std::size_t i = begin; i < end; ++i) {
      acc += unsigned_values[i];
    }
    T result{};
    std::memcpy(&result, &acc, sizeof(T));
    return result;
  } else {
    T acc{};
    for (std::size_t i = begin; i < end; ++i) {
      acc += values[i];
    }
    return acc;
  }
}

template <typename T>
void cpu_reduce_task(void *raw_ctx, int /*thread_id*/, int task_id) {
  auto *ctx = static_cast<CpuReduceTaskContext<T> *>(raw_ctx);
  const int tid = task_id;
  const std::size_t begin =
      partition_boundary(ctx->n, ctx->num_threads, tid);
  const std::size_t end =
      partition_boundary(ctx->n, ctx->num_threads, tid + 1);
  if (ctx->op == 0) {
    ctx->partial[tid] = cpu_reduce_sum_contiguous_range(ctx->values, begin, end);
    return;
  }
  T acc = cpu_reduce_identity<T>(ctx->op);
  for (std::size_t i = begin; i < end; ++i) {
    acc = cpu_reduce_combine(acc, ctx->values[i], ctx->op);
  }
  ctx->partial[tid] = acc;
}

// Preparation supplies partial storage and the owner's existing scheduler.
// The mathematical loop is shared by JIT submission and fixed AOT recording.
template <typename T>
void reduce(const T *values, T *output, int op, std::size_t n,
            int partitions, T *partial, ThreadPool *pool) {
  if (!values || !output || !n || op < 0 || op > 2 || partitions < 1 ||
      std::size_t(partitions) > n || (partitions > 1 && (!partial || !pool)))
    throw std::invalid_argument("CPU reduction has invalid storage or scheduler");
  if (partitions > 1) {
    CpuReduceTaskContext<T> context{values, partial, n, partitions, op};
    pool->run(partitions, partitions, &context, cpu_reduce_task<T>);
    T result = cpu_reduce_identity<T>(op);
    for (int i = 0; i < partitions; ++i)
      result = cpu_reduce_combine(result, partial[i], op);
    output[0] = result;
  } else if (op == 0) {
    output[0] = cpu_reduce_sum_contiguous_range(values, 0, n);
  } else {
    T result = cpu_reduce_identity<T>(op);
    for (std::size_t i = 0; i < n; ++i)
      result = cpu_reduce_combine(result, values[i], op);
    output[0] = result;
  }
}

template <typename T>
struct CpuStridedReduceTaskContext {
  const std::uint8_t *values;
  T *partial;
  std::size_t n, offset, stride;
  int num_threads, op;
};

template <typename T>
T cpu_reduce_sum_strided_range(const std::uint8_t *values,
                               std::size_t offset, std::size_t stride,
                               std::size_t begin, std::size_t end) {
  if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
    using U = std::make_unsigned_t<T>;
    U acc = 0;
    for (std::size_t i = begin; i < end; ++i) {
      U bits;
      std::memcpy(&bits, values + offset + i * stride, sizeof(T));
      acc += bits;
    }
    T result;
    std::memcpy(&result, &acc, sizeof(T));
    return result;
  } else {
    T acc{};
    for (std::size_t i = begin; i < end; ++i)
      acc += *reinterpret_cast<const T *>(values + offset + i * stride);
    return acc;
  }
}

template <typename T>
void cpu_strided_reduce_task(void *raw_ctx, int /*thread_id*/, int task_id) {
  auto *ctx = static_cast<CpuStridedReduceTaskContext<T> *>(raw_ctx);
  const auto begin = partition_boundary(ctx->n, ctx->num_threads, task_id);
  const auto end = partition_boundary(ctx->n, ctx->num_threads, task_id + 1);
  if (ctx->op == 0) {
    ctx->partial[task_id] = cpu_reduce_sum_strided_range<T>(
        ctx->values, ctx->offset, ctx->stride, begin, end);
    return;
  }
  T acc = cpu_reduce_identity<T>(ctx->op);
  for (std::size_t i = begin; i < end; ++i)
    acc = cpu_reduce_combine(
        acc, *reinterpret_cast<const T *>(ctx->values + ctx->offset + i * ctx->stride),
        ctx->op);
  ctx->partial[task_id] = acc;
}

// The caller proves the physical window and alias separation. Both JIT and
// AOT preserve the same partition order; only the first output is written.
template <typename T>
void reduce_strided(const std::uint8_t *values, T *output, int op,
                     std::size_t n, std::size_t offset, std::size_t stride,
                     int partitions, T *partial, ThreadPool *pool) {
  if (!values || !output || !n || op < 0 || op > 2 || partitions < 1 ||
      std::size_t(partitions) > n || (partitions > 1 && (!partial || !pool)) ||
      stride < sizeof(T) || offset % alignof(T) || stride % alignof(T) ||
      offset > std::numeric_limits<std::size_t>::max() - sizeof(T) ||
      n - 1 > (std::numeric_limits<std::size_t>::max() - offset - sizeof(T)) / stride)
    throw std::invalid_argument("CPU strided reduction has invalid storage or scheduler");
  if (partitions > 1) {
    CpuStridedReduceTaskContext<T> context{values, partial, n, offset, stride, partitions, op};
    pool->run(partitions, partitions, &context, cpu_strided_reduce_task<T>);
    T result = cpu_reduce_identity<T>(op);
    for (int i = 0; i < partitions; ++i)
      result = cpu_reduce_combine(result, partial[i], op);
    output[0] = result;
  } else if (op == 0) {
    output[0] = cpu_reduce_sum_strided_range<T>(values, offset, stride, 0, n);
  } else {
    T result = cpu_reduce_identity<T>(op);
    for (std::size_t i = 0; i < n; ++i)
      result = cpu_reduce_combine(
          result, *reinterpret_cast<const T *>(values + offset + i * stride), op);
    output[0] = result;
  }
}

}  // namespace taichi::lang::cpu
