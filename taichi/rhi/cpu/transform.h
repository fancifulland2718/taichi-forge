#pragma once

#include <stdexcept>
#include <type_traits>
#include "taichi/rhi/cpu/parallel_range.h"
#include "taichi/system/threading.h"

namespace taichi::lang::cpu {

inline int transform_partitions(std::size_t scalar_count,
                                int max_threads,
                                bool packed_tensor) {
  if (!packed_tensor)
    return simple_loop_partitions(scalar_count, max_threads);
  // Preserve the existing packed-transform policy, which has a lower parallel
  // crossover than the scalar loop. This is a scheduling choice, not a limit.
  const auto chunks = scalar_count / 32768 + (scalar_count % 32768 != 0);
  const int target = static_cast<int>(std::min<std::size_t>(
      chunks, static_cast<std::size_t>(std::max(1, max_threads))));
  return scalar_count >= 65536 && target > 1 ? target : 1;
}

template <typename T>
struct TransformTaskContext {
  const T *src{nullptr};
  T *dst{nullptr};
  std::size_t n{0};
  T scale{};
  T bias{};
  int num_threads{1};
};

template <typename T>
void transform_task(void *raw_ctx, int /*thread_id*/, int task_id) {
  auto *ctx = static_cast<TransformTaskContext<T> *>(raw_ctx);
  const auto begin = partition_boundary(ctx->n, ctx->num_threads, task_id);
  const auto end = partition_boundary(ctx->n, ctx->num_threads, task_id + 1);
  for (std::size_t i = begin; i < end; ++i) {
    ctx->dst[i] = ctx->src[i] * ctx->scale + ctx->bias;
  }
}

// Signed integer arrays use their unsigned storage representation, preserving
// modular arithmetic. Coefficient conversion belongs to cold preparation. Exact
// in-place bindings and disjoint ranges are supported; the caller proves
// aliases.
template <typename T>
void transform(const T *src,
               T *dst,
               std::size_t n,
               T scale,
               T bias,
               int partitions,
               ThreadPool *pool) {
  static_assert(std::is_unsigned_v<T> || std::is_floating_point_v<T>);
  if (partitions < 1 || (n && (!src || !dst)) ||
      (n && partitions > 1 && !pool)) {
    throw std::invalid_argument(
        "Transform requires valid storage and scheduler");
  }
  if (!n)
    return;
  if (partitions > 1) {
    TransformTaskContext<T> ctx{src, dst, n, scale, bias, partitions};
    pool->run(partitions, partitions, &ctx, transform_task<T>);
    return;
  }
  for (std::size_t i = 0; i < n; ++i) {
    dst[i] = src[i] * scale + bias;
  }
}

template <typename T>
struct StridedTransformTaskContext {
  const std::uint8_t *src;
  std::uint8_t *dst;
  std::size_t n, src_offset, src_stride, dst_offset, dst_stride;
  T scale, bias;
  int num_threads;
};

template <typename T>
void strided_transform_task(void *raw, int, int task) {
  const auto &ctx = *static_cast<StridedTransformTaskContext<T> *>(raw);
  const auto begin = partition_boundary(ctx.n, ctx.num_threads, task);
  const auto end = partition_boundary(ctx.n, ctx.num_threads, task + 1);
  for (auto i = begin; i < end; ++i) {
    const auto value = *reinterpret_cast<const T *>(ctx.src + ctx.src_offset + i * ctx.src_stride);
    *reinterpret_cast<T *>(ctx.dst + ctx.dst_offset + i * ctx.dst_stride) =
        value * ctx.scale + ctx.bias;
  }
}

// The caller validates reachable spans, scalar alignment and aliasing once.
// Keep the JIT scalar scheduling policy and allocation-free serial path.
template <typename T>
void transform_strided(const std::uint8_t *src,
                       std::uint8_t *dst,
                       std::size_t n,
                       std::size_t src_offset,
                       std::size_t src_stride,
                       std::size_t dst_offset,
                       std::size_t dst_stride,
                       T scale, T bias, int partitions, ThreadPool *pool) {
  static_assert(std::is_unsigned_v<T> || std::is_floating_point_v<T>);
  if (partitions < 1 || (n && (!src || !dst)) || (n && partitions > 1 && !pool))
    throw std::invalid_argument("Strided transform requires storage and scheduler");
  if (!n)
    return;
  StridedTransformTaskContext<T> ctx{
      src, dst, n, src_offset, src_stride, dst_offset, dst_stride,
      scale, bias, partitions};
  if (partitions > 1)
    pool->run(partitions, partitions, &ctx, strided_transform_task<T>);
  else
    strided_transform_task<T>(&ctx, 0, 0);
}

}  // namespace taichi::lang::cpu
