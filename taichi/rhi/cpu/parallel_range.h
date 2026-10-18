#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace taichi::lang::cpu {

inline std::size_t partition_boundary(std::size_t n, int parts, int index) {
  return (n / parts) * index +
         (static_cast<std::uint64_t>(n % parts) * index) / parts;
}

// Algorithm selection, not input-capacity limits. Keep JIT and prepared native
// operations on the same policy; the owner supplies the actual worker pool.
inline bool use_parallel_simple_loop(std::size_t n, int target_threads) {
  return n >= 262144 && target_threads > 1;
}

inline int simple_loop_partitions(std::size_t n, int max_threads) {
  const auto chunks = n / 32768 + (n % 32768 != 0);
  const auto target = static_cast<int>(std::min<std::size_t>(
      chunks, static_cast<std::size_t>(std::max(1, max_threads))));
  return use_parallel_simple_loop(n, target) ? target : 1;
}

}  // namespace taichi::lang::cpu
