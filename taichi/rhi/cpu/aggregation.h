#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace taichi::lang::cpu {

// The owner chooses partitions and storage retention. This calculation only
// validates address-space arithmetic; it imposes no algorithm budget.
template <typename T>
std::size_t aggregation_workspace_bytes(std::size_t groups, int partitions) {
  if (partitions < 1)
    throw std::invalid_argument("CPU aggregation requires positive partitions");
  if (partitions == 1 || !groups)
    return 0;
  const auto count = static_cast<std::size_t>(partitions);
  if (groups > std::numeric_limits<std::size_t>::max() / sizeof(T) / count)
    throw std::invalid_argument(
        "CPU aggregation workspace exceeds address space");
  return groups * count * sizeof(T);
}

// Existing algorithm/retention choices, not limits on input capacity. Freeze
// the producer configuration when replay must preserve floating-point order.
inline constexpr std::size_t max_local_aggregation_workspace_bytes = 64ull
                                                                     << 20;

inline int aggregation_target_threads(std::size_t n,
                                      std::size_t groups,
                                      int max_threads) {
  auto target = std::max<std::size_t>(1, n / 65536);
  if (groups <= 1024 && n >= 262144)
    target = std::min<std::size_t>(16, std::max<std::size_t>(1, n / 32768));
  return static_cast<int>(
      std::min<std::size_t>(std::max(1, max_threads), target));
}

template <typename T>
int scatter_add_workspace_partitions(std::size_t groups, int target) {
  return target > 1 && groups <= max_local_aggregation_workspace_bytes /
                                     sizeof(T) / target
             ? target
             : 1;
}

template <typename T>
int scatter_add_partitions(std::size_t n, std::size_t groups, int max_threads) {
  return scatter_add_workspace_partitions<T>(
      groups, aggregation_target_threads(n, groups, max_threads));
}

}  // namespace taichi::lang::cpu
