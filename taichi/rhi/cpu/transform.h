#pragma once

#include <stdexcept>
#include <type_traits>
#include "taichi/rhi/cpu/parallel_range.h"
#include "taichi/system/threading.h"

namespace taichi::lang::cpu {

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

}  // namespace taichi::lang::cpu
