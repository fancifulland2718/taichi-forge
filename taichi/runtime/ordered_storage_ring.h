#pragma once

#include <cstddef>
#include <stdexcept>

namespace taichi::lang {
// Each frame binds distinct scratch and runs on the same ordered backend queue.
// The enclosing prepared owner retains the scratch through retirement. This is
// a cold storage recipe, not a host completion-polling or concurrent-lane API.
struct OrderedStorageRing {
  std::size_t capacity{0};

  std::size_t frame_size(std::size_t operation_count) const {
    if (!capacity || !operation_count || operation_count % capacity)
      throw std::invalid_argument(
          "Ordered storage ring requires nonempty equal-sized frames");
    return operation_count / capacity;
  }
};
}  // namespace taichi::lang
