#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

namespace taichi::lang {

inline constexpr int kCpuMinChunkSize = 512;

// Number of productive chunks for make_cpu_multithreaded_range_for's i32
// partition. Keep its generic count if any original intermediate can overflow.
inline int cpu_range_chunk_count(int begin, int end, int num_threads) {
  if (num_threads <= 0)
    return num_threads;
  const int64_t extent = int64_t(end) - begin;
  const int64_t rounded = extent + num_threads - 1;
  constexpr auto lo = std::numeric_limits<int32_t>::min();
  constexpr auto hi = std::numeric_limits<int32_t>::max();
  if (extent < lo || extent > hi || rounded < lo || extent + num_threads > hi)
    return num_threads;
  const int64_t chunk =
      std::max<int64_t>(kCpuMinChunkSize, rounded / num_threads);
  if (chunk * num_threads > hi || int64_t(begin) + chunk * num_threads > hi)
    return num_threads;
  return extent <= 0 ? 0 : int((extent + chunk - 1) / chunk);
}

}  // namespace taichi::lang
