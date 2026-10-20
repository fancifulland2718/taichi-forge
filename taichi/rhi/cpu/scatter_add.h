#pragma once

#include "taichi/rhi/cpu/reduce.h"
#include "taichi/rhi/cpu/aggregation.h"

namespace taichi::lang::cpu {

// Allocation and retention policy belong to the caller. Only address-space
// overflow is rejected here; this is not a capacity or worker-count policy.
template <typename T>
std::size_t scatter_add_workspace_bytes(std::size_t dst_items, int partitions) {
  return aggregation_workspace_bytes<T>(dst_items, partitions);
}

inline std::size_t packed_scatter_add_scalars(std::size_t items,
                                            std::size_t lanes) {
  if (!lanes || items > std::numeric_limits<std::size_t>::max() / lanes)
    throw std::invalid_argument("Packed scatter-add payload exceeds address space");
  return items * lanes;
}

template <typename T>
int packed_scatter_add_partitions(std::size_t n,
                                 std::size_t dst_items,
                                 std::size_t lanes,
                                 int max_threads) {
  return scatter_add_workspace_partitions<T>(
      packed_scatter_add_scalars(dst_items, lanes),
      aggregation_target_threads(n, dst_items, max_threads));
}

template <typename T>
struct PackedScatterAddContext {
  const uint8_t *src;
  const int32_t *indices;
  T *partial;
  uint8_t *dst;
  std::size_t n, dst_items, lanes;
  std::size_t src_offset, src_stride, dst_offset, dst_stride;
  int partitions;
};

template <typename T>
void packed_scatter_add_count_task(void *raw, int /*worker*/, int task) {
  const auto &c = *static_cast<PackedScatterAddContext<T> *>(raw);
  T *local = c.partial + c.dst_items * c.lanes * std::size_t(task);
  const auto begin = partition_boundary(c.n, c.partitions, task);
  const auto end = partition_boundary(c.n, c.partitions, task + 1);
  for (auto i = begin; i < end; ++i) {
    const auto index = static_cast<std::size_t>(c.indices[i]);
    if (index < c.dst_items) {
      const auto *src = reinterpret_cast<const T *>(
          c.src + c.src_offset + i * c.src_stride);
      auto *out = local + index * c.lanes;
      for (std::size_t lane = 0; lane < c.lanes; ++lane)
        out[lane] = cpu_reduce_combine(out[lane], src[lane], 0);
    }
  }
}

template <typename T>
void packed_scatter_add_merge_task(void *raw, int /*worker*/, int task) {
  const auto &c = *static_cast<PackedScatterAddContext<T> *>(raw);
  const auto total = c.dst_items * c.lanes;
  const auto begin = partition_boundary(total, c.partitions, task);
  const auto end = partition_boundary(total, c.partitions, task + 1);
  for (auto scalar = begin; scalar < end; ++scalar) {
    T value{};
    for (int p = 0; p < c.partitions; ++p)
      value = cpu_reduce_combine(
          value, c.partial[total * std::size_t(p) + scalar], 0);
    auto *out = reinterpret_cast<T *>(
        c.dst + c.dst_offset + (scalar / c.lanes) * c.dst_stride);
    const auto lane = scalar % c.lanes;
    out[lane] = cpu_reduce_combine(out[lane], value, 0);
  }
}

// Shared packed Field algorithm: partition records, then merge scalar output
// lanes in the original order. The owner supplies zeroed reusable partials.
template <typename T>
void packed_scatter_add(const uint8_t *src,
                        std::size_t src_offset,
                        std::size_t src_stride,
                        const int32_t *indices,
                        uint8_t *dst,
                        std::size_t dst_offset,
                        std::size_t dst_stride,
                        std::size_t n,
                        std::size_t dst_items,
                        std::size_t lanes,
                        int partitions,
                        T *partial,
                        ThreadPool *pool) {
  const auto scalars = packed_scatter_add_scalars(dst_items, lanes);
  scatter_add_workspace_bytes<T>(scalars, partitions);
  if (!n || !dst_items)
    return;
  if (!src || !indices || !dst ||
      lanes > std::numeric_limits<std::size_t>::max() / sizeof(T) ||
      src_stride < lanes * sizeof(T) || dst_stride < lanes * sizeof(T) ||
      (partitions > 1 && (!partial || !pool)))
    throw std::invalid_argument(
        "CPU packed scatter-add has invalid storage or scheduler");
  if (partitions > 1) {
    PackedScatterAddContext<T> c{src, indices, partial, dst, n, dst_items,
                                 lanes, src_offset, src_stride, dst_offset,
                                 dst_stride, partitions};
    pool->run(partitions, partitions, &c, packed_scatter_add_count_task<T>);
    pool->run(partitions, partitions, &c, packed_scatter_add_merge_task<T>);
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      const auto index = static_cast<std::size_t>(indices[i]);
      if (index < dst_items) {
        const auto *value =
            reinterpret_cast<const T *>(src + src_offset + i * src_stride);
        auto *out =
            reinterpret_cast<T *>(dst + dst_offset + index * dst_stride);
        for (std::size_t lane = 0; lane < lanes; ++lane)
          out[lane] = cpu_reduce_combine(out[lane], value[lane], 0);
      }
    }
  }
}

template <typename T>
struct ScatterAddContext {
  const T *src;
  const int32_t *indices;
  T *partial;
  T *dst;
  std::size_t n, dst_items;
  int partitions;
};

template <typename T>
void scatter_add_count_task(void *raw, int /*worker*/, int task) {
  auto &c = *static_cast<ScatterAddContext<T> *>(raw);
  T *local = c.partial + c.dst_items * static_cast<std::size_t>(task);
  const auto begin = partition_boundary(c.n, c.partitions, task);
  const auto end = partition_boundary(c.n, c.partitions, task + 1);
  for (auto i = begin; i < end; ++i) {
    const auto index = static_cast<std::size_t>(c.indices[i]);
    if (index < c.dst_items)
      local[index] = cpu_reduce_combine(local[index], c.src[i], 0);
  }
}

template <typename T, bool accumulate>
void scatter_add_merge_task(void *raw, int /*worker*/, int task) {
  auto &c = *static_cast<ScatterAddContext<T> *>(raw);
  const auto begin = partition_boundary(c.dst_items, c.partitions, task);
  const auto end = partition_boundary(c.dst_items, c.partitions, task + 1);
  for (auto i = begin; i < end; ++i) {
    T value{};
    for (int p = 0; p < c.partitions; ++p)
      value = cpu_reduce_combine(
          value, c.partial[c.dst_items * static_cast<std::size_t>(p) + i], 0);
    if constexpr (accumulate)
      c.dst[i] = cpu_reduce_combine(c.dst[i], value, 0);
    else
      c.dst[i] = value;
  }
}

// Partial storage must be zeroed before every parallel call. This avoids a
// second clear when the JIT scratch arena already supplies zeroed storage.
// A fixed owner can retain the storage and clear it once per replay. Preserve
// partition order and modulo signed integer addition. Grouped reduction writes
// each parallel output once; only the serial path needs to clear it first.
template <typename T, bool accumulate>
void indexed_sum(const T *src,
                 const int32_t *indices,
                 T *dst,
                 std::size_t n,
                 std::size_t dst_items,
                 int partitions,
                 T *partial,
                 ThreadPool *pool) {
  if (!dst_items || (accumulate && !n))
    return;
  scatter_add_workspace_bytes<T>(dst_items, partitions);
  if (!dst ||
      (n && (!src || !indices || (partitions > 1 && (!partial || !pool)))))
    throw std::invalid_argument(
        "CPU indexed sum has invalid storage or scheduler");
  // The caller qualifies actual typed storage. Its addressable element count
  // cannot reach a negative i32 sign-extended to size_t (payloads are >=4B).
  // Keep the existing single unsigned comparison in the hot accumulation loop.
  if (n && partitions > 1) {
    ScatterAddContext<T> context{src, indices,   partial,   dst,
                                 n,   dst_items, partitions};
    pool->run(partitions, partitions, &context, scatter_add_count_task<T>);
    pool->run(partitions, partitions, &context,
              scatter_add_merge_task<T, accumulate>);
  } else {
    if constexpr (!accumulate)
      std::fill(dst, dst + dst_items, T{});
    for (std::size_t i = 0; i < n; ++i) {
      const auto index = static_cast<std::size_t>(indices[i]);
      if (index < dst_items)
        dst[index] = cpu_reduce_combine(dst[index], src[i], 0);
    }
  }
}

template <typename T>
void scatter_add(const T *src,
                 const int32_t *indices,
                 T *dst,
                 std::size_t n,
                 std::size_t dst_items,
                 int partitions,
                 T *partial,
                 ThreadPool *pool) {
  indexed_sum<T, true>(src, indices, dst, n, dst_items, partitions, partial,
                       pool);
}

template <typename T>
void grouped_reduce(const int32_t *keys,
                    const T *values,
                    T *output,
                    std::size_t n,
                    std::size_t groups,
                    int partitions,
                    T *partial,
                    ThreadPool *pool) {
  indexed_sum<T, false>(values, keys, output, n, groups, partitions, partial,
                        pool);
}

}  // namespace taichi::lang::cpu
