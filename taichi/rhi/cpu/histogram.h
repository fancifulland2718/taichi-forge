#pragma once

#include "taichi/rhi/cpu/aggregation.h"
#include "taichi/rhi/cpu/reduce.h"

namespace taichi::lang::cpu {

// Preserve the existing parallel strategy. Larger histograms use the serial
// algorithm; 4096 is not an input/admission limit.
inline int histogram_partitions(std::size_t n, std::size_t bins, int workers) {
  return bins <= 4096 ? reduce_partitions(n, workers) : 1;
}

template <typename ValueT>
bool histogram_valid_bin(ValueT bin, std::size_t num_bins) {
  if constexpr (std::is_signed_v<ValueT>) {
    if (bin < 0)
      return false;
  }
  return static_cast<std::size_t>(bin) < num_bins;
}

template <typename ValueT, typename CounterT>
struct HistogramContext {
  const ValueT *values;
  CounterT *partial;
  std::size_t n, num_bins;
  int partitions;
};

template <typename ValueT, typename CounterT>
void histogram_count_task(void *raw, int /*worker*/, int task) {
  auto &c = *static_cast<HistogramContext<ValueT, CounterT> *>(raw);
  auto *local = c.partial + static_cast<std::size_t>(task) * c.num_bins;
  const auto begin = partition_boundary(c.n, c.partitions, task);
  const auto end = partition_boundary(c.n, c.partitions, task + 1);
  for (auto i = begin; i < end; ++i) {
    const auto bin = c.values[i];
    if (histogram_valid_bin(bin, c.num_bins)) {
      auto &counter = local[static_cast<std::size_t>(bin)];
      counter = cpu_reduce_combine(counter, CounterT{1}, 0);
    }
  }
}

// Caller-owned partials are zeroed once before each parallel call. The merge
// overwrites every output bin, so clearing the output first is redundant.
// Serial calls overwrite the histogram too, including an empty input prefix.
template <typename ValueT, typename CounterT>
void histogram(const ValueT *values,
               CounterT *bins,
               std::size_t n,
               std::size_t num_bins,
               int partitions,
               CounterT *partial,
               ThreadPool *pool) {
  static_assert(std::is_integral_v<ValueT> && std::is_integral_v<CounterT>);
  aggregation_workspace_bytes<CounterT>(num_bins, partitions);
  if ((n && !values) || !bins || !num_bins ||
      (partitions > 1 && (!partial || !pool)))
    throw std::invalid_argument(
        "CPU histogram has invalid storage or scheduler");
  if (partitions > 1) {
    HistogramContext<ValueT, CounterT> c{values, partial, n, num_bins,
                                         partitions};
    pool->run(partitions, partitions, &c,
              histogram_count_task<ValueT, CounterT>);
    for (std::size_t bin = 0; bin < num_bins; ++bin) {
      CounterT total{};
      for (int p = 0; p < partitions; ++p)
        total = cpu_reduce_combine(
            total, partial[static_cast<std::size_t>(p) * num_bins + bin], 0);
      bins[bin] = total;
    }
  } else {
    std::fill(bins, bins + num_bins, CounterT{});
    for (std::size_t i = 0; i < n; ++i) {
      const auto bin = values[i];
      if (histogram_valid_bin(bin, num_bins)) {
        auto &counter = bins[static_cast<std::size_t>(bin)];
        counter = cpu_reduce_combine(counter, CounterT{1}, 0);
      }
    }
  }
}

}  // namespace taichi::lang::cpu
