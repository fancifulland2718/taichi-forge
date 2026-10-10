#pragma once

#include <cstddef>
#include <cstdint>

namespace taichi::lang {

// Shared by IR allocation, LLVM codegen and the runtime bitcode. Each entry
// stores a primal value followed by an adjoint of the same size.
struct AdStackLayout {
  using Counter = std::uint64_t;

  static constexpr std::size_t header_size = sizeof(Counter);
  static constexpr std::size_t alignment = alignof(Counter);

  static constexpr std::size_t entry_size(std::size_t element_size) {
    return 2 * element_size;
  }

  static constexpr std::size_t storage_size(std::size_t element_size,
                                          std::size_t capacity) {
    return header_size + entry_size(element_size) * capacity;
  }
};

}  // namespace taichi::lang
