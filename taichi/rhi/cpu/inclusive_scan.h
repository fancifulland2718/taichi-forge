#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace taichi::lang::cpu {

// Shared by Program submission and fixed native recordings. Integer addition
// uses unsigned storage so overflow has the same modular semantics as on GPU.
inline void inclusive_scan(void *data, std::size_t size, int value_type) {
  const auto scan = [size](auto *values) {
    using T = std::remove_pointer_t<decltype(values)>;
    T sum{};
    for (std::size_t i = 0; i < size; ++i) {
      sum += values[i];
      values[i] = sum;
    }
  };
  switch (value_type) {
    case 0:
    case 2:
      scan(static_cast<std::uint32_t *>(data));
      break;
    case 1:
      scan(static_cast<float *>(data));
      break;
    case 3:
    case 4:
      scan(static_cast<std::uint64_t *>(data));
      break;
    case 5:
      scan(static_cast<double *>(data));
      break;
    default:
      throw std::invalid_argument("CPU scan requires a supported numeric type");
  }
}

}  // namespace taichi::lang::cpu
