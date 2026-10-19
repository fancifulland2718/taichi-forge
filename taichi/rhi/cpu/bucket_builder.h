#pragma once

#include <cstring>

#include "taichi/rhi/cpu/aggregation.h"
#include "taichi/rhi/cpu/reduce.h"

namespace taichi::lang::cpu {

// Counts become scatter cursors in place after the prefix pass. The owner
// supplies zeroed storage for each replay; there is no second offsets table.
inline std::size_t bucket_builder_workspace_bytes(std::size_t bins,
                                                  int partitions) {
  if (partitions < 1 || bins > std::numeric_limits<std::size_t>::max() /
                                   sizeof(int32_t) /
                                   static_cast<std::size_t>(partitions))
    throw std::invalid_argument("CPU bucket workspace exceeds address space");
  return bins * static_cast<std::size_t>(partitions) * sizeof(int32_t);
}

// Keep the existing typed/raw partition policy, including its serial fallback.
// This is an execution strategy, not an input-capacity limit.
inline int bucket_builder_partitions(std::size_t n,
                                     std::size_t bins,
                                     int max_threads,
                                     std::size_t chunk_items) {
  if (!chunk_items)
    throw std::invalid_argument("CPU bucket partition chunk must be positive");
  const auto chunks = n / chunk_items + (n % chunk_items != 0);
  const auto target = static_cast<int>(std::max<std::size_t>(
      1, std::min<std::size_t>(chunks, std::max(1, max_threads))));
  return use_parallel_aggregation(n, target) &&
                 bins <= max_local_aggregation_workspace_bytes /
                             sizeof(int32_t) / 2 / target
             ? target
             : 1;
}

template <typename Copy>
struct BucketBuilderContext {
  const int32_t *keys;
  int32_t *scratch;
  std::size_t n, bins;
  int partitions;
  Copy copy;
};

template <typename Copy>
void bucket_count_task(void *raw, int /*worker*/, int task) {
  auto &c = *static_cast<BucketBuilderContext<Copy> *>(raw);
  auto *counts = c.scratch + c.bins * static_cast<std::size_t>(task);
  const auto begin = partition_boundary(c.n, c.partitions, task);
  const auto end = partition_boundary(c.n, c.partitions, task + 1);
  for (auto i = begin; i < end; ++i) {
    const auto key = c.keys[i];
    if (key >= 0 && static_cast<std::size_t>(key) < c.bins)
      ++counts[key];
  }
}

template <typename Copy>
void bucket_scatter_task(void *raw, int /*worker*/, int task) {
  auto &c = *static_cast<BucketBuilderContext<Copy> *>(raw);
  auto *cursor = c.scratch + c.bins * static_cast<std::size_t>(task);
  const auto begin = partition_boundary(c.n, c.partitions, task);
  const auto end = partition_boundary(c.n, c.partitions, task + 1);
  for (auto i = begin; i < end; ++i) {
    const auto key = c.keys[i];
    if (key >= 0 && static_cast<std::size_t>(key) < c.bins)
      c.copy(static_cast<std::size_t>(cursor[key]++), i);
  }
}

template <typename Copy>
void bucket_builder_impl(const int32_t *keys,
                         int32_t *offsets,
                         std::size_t n,
                         std::size_t bins,
                         int partitions,
                         int32_t *scratch,
                         ThreadPool *pool,
                         Copy copy) {
  bucket_builder_workspace_bytes(bins, partitions);
  // i32 offsets/counts are the existing buffer ABI. Check before writing or
  // launching tasks so that local counts and cursor increments cannot overflow.
  if (n > static_cast<std::size_t>(INT32_MAX) || !offsets ||
      bins >= std::numeric_limits<std::size_t>::max() / sizeof(int32_t) ||
      (n && bins && (!keys || !scratch || (partitions > 1 && !pool))))
    throw std::invalid_argument(
        "CPU bucket builder has invalid storage, scheduler or i32 item count");
  if (!n || !bins) {
    std::fill_n(offsets, bins + 1, int32_t{});
    return;
  }
  BucketBuilderContext<Copy> context{keys, scratch, n, bins, partitions, copy};
  if (partitions > 1)
    pool->run(partitions, partitions, &context, bucket_count_task<Copy>);
  else
    bucket_count_task<Copy>(&context, 0, 0);

  int32_t running = 0;
  offsets[0] = 0;
  for (std::size_t bin = 0; bin < bins; ++bin) {
    for (int p = 0; p < partitions; ++p) {
      auto &count = scratch[bins * static_cast<std::size_t>(p) + bin];
      const auto items = count;
      count = running;
      running += items;
    }
    offsets[bin + 1] = running;
  }
  // Partitions cover contiguous input slices and receive disjoint bucket
  // intervals in input order. Scheduling order cannot change payload order.
  if (partitions > 1)
    pool->run(partitions, partitions, &context, bucket_scatter_task<Copy>);
  else
    bucket_scatter_task<Copy>(&context, 0, 0);
}

template <typename T>
void bucket_builder(const int32_t *keys,
                    const T *values,
                    int32_t *offsets,
                    T *output,
                    std::size_t n,
                    std::size_t bins,
                    int partitions,
                    int32_t *scratch,
                    ThreadPool *pool) {
  if (n && (!values || !output ||
            n > std::numeric_limits<std::size_t>::max() / sizeof(T)))
    throw std::invalid_argument(
        "CPU bucket builder has invalid payload storage");
  bucket_builder_impl(keys, offsets, n, bins, partitions, scratch, pool,
                      [values, output](std::size_t dst, std::size_t src) {
                        output[dst] = values[src];
                      });
}

inline void bucket_builder_bytes(const int32_t *keys,
                                 const uint8_t *values,
                                 int32_t *offsets,
                                 uint8_t *output,
                                 std::size_t n,
                                 std::size_t bins,
                                 std::size_t item_bytes,
                                 int partitions,
                                 int32_t *scratch,
                                 ThreadPool *pool) {
  if (!item_bytes || n > std::numeric_limits<std::size_t>::max() / item_bytes ||
      (n && (!values || !output)))
    throw std::invalid_argument(
        "CPU bucket builder has invalid payload storage");
  bucket_builder_impl(
      keys, offsets, n, bins, partitions, scratch, pool,
      [values, output, item_bytes](std::size_t dst, std::size_t src) {
        std::memcpy(output + dst * item_bytes, values + src * item_bytes,
                    item_bytes);
      });
}

}  // namespace taichi::lang::cpu
