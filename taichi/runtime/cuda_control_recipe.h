#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace taichi::lang {

enum class CudaControlEncoding { device_update, masked };

// Selected flat masked or depth-2 encoding. Store one copy of physical
// dispatches, not expanded nodes or consumer device handles. Condition/body
// ranges include the selected tail conditions and index their concatenation.
template <typename Binding>
struct CudaControlRecipe {
  struct Inner {
    Binding predicate;
    std::size_t condition_begin{0};
    std::size_t body_begin{0};
    std::size_t end{0};
    std::uint64_t max_iterations{0};
  };
  std::vector<Inner> inners;
  CudaControlEncoding encoding{CudaControlEncoding::device_update};

  void validate(std::size_t initial,
                std::size_t dispatches,
                bool continue_while_nonzero = true) const {
    const bool flat_masked =
        encoding == CudaControlEncoding::masked && inners.empty();
    if (initial >= dispatches ||
        (!flat_masked && (!initial || inners.empty() || !continue_while_nonzero)) ||
        (encoding != CudaControlEncoding::device_update &&
         encoding != CudaControlEncoding::masked))
      throw std::invalid_argument("Invalid selected CUDA control");
    auto cursor = initial;
    for (const auto &inner : inners) {
      if (cursor > inner.condition_begin ||
          inner.condition_begin >= inner.body_begin ||
          inner.body_begin >= inner.end || inner.end >= dispatches)
        throw std::invalid_argument(
            "Invalid CUDA nested dispatch boundaries");
      cursor = inner.end;
    }
  }
};

}  // namespace taichi::lang
