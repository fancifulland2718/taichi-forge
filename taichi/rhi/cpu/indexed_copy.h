#pragma once

#include <cstring>
#include <stdexcept>
#include "taichi/rhi/cpu/parallel_range.h"
#include "taichi/system/threading.h"

namespace taichi::lang::cpu {

// Preserve the existing worker selection, including the lower scatter worker
// count for large random writes. These are scheduling choices, not size limits.
inline int indexed_copy_target_threads(std::size_t n,
                                       int max_threads,
                                       bool scatter) {
  const auto chunks = n / 32768 + (n % 32768 != 0);
  int target = static_cast<int>(std::min<std::size_t>(
      chunks, static_cast<std::size_t>(std::max(1, max_threads))));
  if (scatter && n >= 262144)
    target = std::min(target, 4);
  return std::max(1, target);
}

inline int indexed_copy_partitions(std::size_t n,
                                   int max_threads,
                                   bool scatter) {
  return n >= 65536 ? indexed_copy_target_threads(n, max_threads, scatter) : 1;
}

struct IndexedCopyTaskContext {
  const uint8_t *src{nullptr};
  const int32_t *indices{nullptr};
  uint8_t *dst{nullptr};
  std::size_t n{0};
  std::size_t index_bound{0};
  std::size_t item_bytes{0};
  bool scatter{false};
  int num_threads{1};
};

inline void copy_indexed_payload(uint8_t *dst,
                                 const uint8_t *src,
                                 std::size_t item_bytes) {
  auto *dst_words = reinterpret_cast<uint32_t *>(dst);
  const auto *src_words = reinterpret_cast<const uint32_t *>(src);
  if (item_bytes == sizeof(uint32_t)) {
    dst_words[0] = src_words[0];
    return;
  }
  if (item_bytes == sizeof(uint64_t)) {
    dst_words[0] = src_words[0];
    dst_words[1] = src_words[1];
    return;
  }
  if (item_bytes == 4 * sizeof(uint32_t)) {
    dst_words[0] = src_words[0];
    dst_words[1] = src_words[1];
    dst_words[2] = src_words[2];
    dst_words[3] = src_words[3];
    return;
  }
  const std::size_t words = item_bytes / sizeof(uint32_t);
  for (std::size_t word = 0; word < words; ++word) {
    dst_words[word] = src_words[word];
  }
}

inline void zero_indexed_payload(uint8_t *dst, std::size_t item_bytes) {
  auto *dst_words = reinterpret_cast<uint32_t *>(dst);
  if (item_bytes == sizeof(uint32_t)) {
    dst_words[0] = 0;
    return;
  }
  if (item_bytes == sizeof(uint64_t)) {
    dst_words[0] = 0;
    dst_words[1] = 0;
    return;
  }
  if (item_bytes == 4 * sizeof(uint32_t)) {
    dst_words[0] = 0;
    dst_words[1] = 0;
    dst_words[2] = 0;
    dst_words[3] = 0;
    return;
  }
  const std::size_t words = item_bytes / sizeof(uint32_t);
  for (std::size_t word = 0; word < words; ++word) {
    dst_words[word] = 0;
  }
}

inline void indexed_copy_task(void *raw_ctx, int /*thread_id*/, int task_id) {
  auto *ctx = static_cast<IndexedCopyTaskContext *>(raw_ctx);
  const auto begin = partition_boundary(ctx->n, ctx->num_threads, task_id);
  const auto end = partition_boundary(ctx->n, ctx->num_threads, task_id + 1);
  if (ctx->scatter) {
    for (std::size_t i = begin; i < end; ++i) {
      const auto index = static_cast<std::size_t>(ctx->indices[i]);
      if (index < ctx->index_bound) {
        copy_indexed_payload(ctx->dst + index * ctx->item_bytes,
                             ctx->src + i * ctx->item_bytes, ctx->item_bytes);
      }
    }
  } else {
    for (std::size_t i = begin; i < end; ++i) {
      const auto index = static_cast<std::size_t>(ctx->indices[i]);
      if (index < ctx->index_bound) {
        copy_indexed_payload(ctx->dst + i * ctx->item_bytes,
                             ctx->src + index * ctx->item_bytes,
                             ctx->item_bytes);
      } else {
        zero_indexed_payload(ctx->dst + i * ctx->item_bytes, ctx->item_bytes);
      }
    }
  }
}

// Payloads are raw 4-byte words, preserving floating/integer bit patterns.
// The caller proves storage extents and disjoint writes. Scatter targets must
// already be validated as unique before any worker writes; this low-level core
// neither reads indices back from another device nor owns validation storage.
inline void indexed_copy(const uint8_t *src,
                         const int32_t *indices,
                         uint8_t *dst,
                         std::size_t n,
                         std::size_t index_bound,
                         std::size_t item_bytes,
                         bool scatter,
                         int partitions,
                         ThreadPool *pool) {
  if (partitions < 1 || item_bytes == 0 || item_bytes % sizeof(uint32_t) != 0 ||
      (n && (!src || !indices || !dst || (partitions > 1 && !pool))))
    throw std::invalid_argument(
        "Indexed copy requires valid storage and scheduler");
  if (!n)
    return;
  if (partitions > 1) {
    IndexedCopyTaskContext ctx{src,         indices,    dst,     n,
                               index_bound, item_bytes, scatter, partitions};
    pool->run(partitions, partitions, &ctx, indexed_copy_task);
    return;
  }
  if (scatter) {
    for (std::size_t i = 0; i < n; ++i) {
      const auto index = static_cast<std::size_t>(indices[i]);
      if (index < index_bound)
        std::memcpy(dst + index * item_bytes, src + i * item_bytes, item_bytes);
    }
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      const auto index = static_cast<std::size_t>(indices[i]);
      if (index < index_bound)
        std::memcpy(dst + i * item_bytes, src + index * item_bytes, item_bytes);
      else
        std::memset(dst + i * item_bytes, 0, item_bytes);
    }
  }
}

}  // namespace taichi::lang::cpu
