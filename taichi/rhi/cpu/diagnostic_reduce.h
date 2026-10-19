#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include "taichi/rhi/cpu/parallel_range.h"
#include "taichi/system/threading.h"

namespace taichi::lang::cpu {

template <typename T>
struct DiagnosticPackedRead {
  using value_type = T;
  const T *data;
  T operator[](std::size_t i) const {
    return data[i];
  }
};

template <typename T>
struct DiagnosticStridedRead {
  using value_type = T;
  const uint8_t *data;
  std::size_t offset, stride;
  T operator[](std::size_t i) const {
    return *reinterpret_cast<const T *>(data + offset + i * stride);
  }
};

template <typename Reduction>
struct DiagnosticReductionContext {
  Reduction reduction;
  typename Reduction::value_type *partial;
  std::size_t n;
  int partitions;
};

template <typename Reduction>
void diagnostic_reduce_task(void *raw, int /*worker*/, int task) {
  auto &c = *static_cast<DiagnosticReductionContext<Reduction> *>(raw);
  c.partial[task] =
      c.reduction(partition_boundary(c.n, c.partitions, task),
                  partition_boundary(c.n, c.partitions, task + 1));
}

// Every task overwrites its partial, including empty ranges. The caller owns
// storage and scheduler; replay needs neither allocation nor pre-clearing.
template <typename Reduction>
typename Reduction::value_type diagnostic_reduce(
    Reduction reduction,
    std::size_t n,
    int partitions,
    typename Reduction::value_type *partial,
    ThreadPool *pool) {
  if (partitions < 1 || (partitions > 1 && (!partial || !pool)))
    throw std::invalid_argument(
        "CPU diagnostic reduction has an invalid scheduler or partial storage");
  if (partitions == 1)
    return reduction(0, n);
  DiagnosticReductionContext<Reduction> context{reduction, partial, n,
                                                partitions};
  pool->run(partitions, partitions, &context,
            diagnostic_reduce_task<Reduction>);
  typename Reduction::value_type result{};
  for (int i = 0; i < partitions; ++i)
    result = reduction.combine(result, partial[i]);
  return result;
}

template <typename T>
bool check_predicate(T value, int op, int lower, int upper) {
  switch (op) {
    case 0:
      return value != T{};
    case 1:
      return value == T{};
    case 2:
      if constexpr (std::is_floating_point_v<T>)
        return std::isnan(value);
      return false;
    case 3:
      if constexpr (std::is_floating_point_v<T>)
        return std::isinf(value);
      return false;
    case 4:
      if constexpr (std::is_floating_point_v<T>)
        return !std::isfinite(value);
      return false;
    case 5:
      if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
        return value < static_cast<T>(lower) || value >= static_cast<T>(upper);
      } else if constexpr (std::is_integral_v<T>) {
        return upper <= 0 || (lower > 0 && value < static_cast<T>(lower)) ||
               value >= static_cast<T>(upper);
      }
      return false;
  }
  return false;
}

template <typename Read>
struct CheckCountReduction {
  using value_type = uint64_t;
  Read values;
  int op, lower, upper;
  template <int Op, typename Counter>
  Counter count(std::size_t begin, std::size_t end) const {
    Counter result{};
    for (auto i = begin; i < end; ++i)
      result += check_predicate(values[i], Op, lower, upper) ? 1 : 0;
    return result;
  }
  template <int Op>
  uint64_t count_range(std::size_t begin, std::size_t end) const {
    // Preserve the ordinary 32-bit vectorized loop. Wide ranges and the merge
    // use exact wide counts; their maximum sum is the addressable input count.
    return end - begin <= UINT32_MAX ? count<Op, uint32_t>(begin, end)
                                     : count<Op, uint64_t>(begin, end);
  }
  uint64_t operator()(std::size_t begin, std::size_t end) const {
    // Select the predicate once per range, so the inner loop remains amenable
    // to vectorization even with the wide-count fallback present.
    switch (op) {
      case 0:
        return count_range<0>(begin, end);
      case 1:
        return count_range<1>(begin, end);
      case 2:
        return count_range<2>(begin, end);
      case 3:
        return count_range<3>(begin, end);
      case 4:
        return count_range<4>(begin, end);
      case 5:
        return count_range<5>(begin, end);
    }
    return 0;
  }
  static uint64_t combine(uint64_t a, uint64_t b) {
    return a + b;
  }
};

template <typename Read>
uint64_t check_count(Read values,
                     std::size_t n,
                     int op,
                     int lower,
                     int upper,
                     int partitions,
                     uint64_t *partial,
                     ThreadPool *pool) {
  if ((n && !values.data) || op < 0 || op > 5)
    throw std::invalid_argument("CPU check has invalid input or predicate");
  return diagnostic_reduce(CheckCountReduction<Read>{values, op, lower, upper},
                           n, partitions, partial, pool);
}

// Existing i32 device scalars reserve negative values for count overflow.
// This retains zero/nonzero truth without truncating a large count to zero.
inline int32_t encode_check_count_i32(uint64_t count) {
  return count <= INT32_MAX ? static_cast<int32_t>(count) : INT32_MIN;
}

template <typename T>
T diagnostic_abs(T value) {
  static_assert(std::is_floating_point_v<T>);
  if (std::isnan(value))
    return std::numeric_limits<T>::infinity();
  return value < T{} ? -value : value;
}

template <typename Read>
struct MetricReduction {
  using value_type = typename Read::value_type;
  Read values, other;
  int op;
  value_type operator()(std::size_t begin, std::size_t end) const {
    value_type result{};
    for (auto i = begin; i < end; ++i) {
      const auto value = op ? values[i] - other[i] : values[i];
      result = std::max(result, diagnostic_abs(value));
    }
    return result;
  }
  static value_type combine(value_type a, value_type b) {
    return std::max(a, b);
  }
};

template <typename Read>
typename Read::value_type metric_reduce(Read values,
                                        Read other,
                                        std::size_t n,
                                        int op,
                                        int partitions,
                                        typename Read::value_type *partial,
                                        ThreadPool *pool) {
  if ((n && (!values.data || (op == 1 && !other.data))) || op < 0 || op > 1)
    throw std::invalid_argument("CPU metric has invalid input or operation");
  return diagnostic_reduce(MetricReduction<Read>{values, other, op}, n,
                           partitions, partial, pool);
}

}  // namespace taichi::lang::cpu
