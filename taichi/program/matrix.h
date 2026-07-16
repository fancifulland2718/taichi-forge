#pragma once

#include <array>

#include "taichi/ir/type.h"

namespace taichi::lang {

/**
 * Matrix wrapper used in AOT.
 */
class Matrix {
 public:
  explicit Matrix(const uint32_t &length,
                  const DataType dtype,
                  const intptr_t &data,
                  uint32_t ndim = 0,
                  uint32_t shape0 = 0,
                  uint32_t shape1 = 0)
      : length_(length),
        dtype_(dtype),
        data_(data),
        ndim_(ndim),
        shape_({shape0, shape1}) {
  }

  DataType dtype() const {
    return dtype_;
  }

  uint32_t length() const {
    // number of matrix elements
    return length_;
  }

  intptr_t data() const {
    return data_;
  }

  uint32_t ndim() const {
    return ndim_;
  }

  uint32_t shape(uint32_t axis) const {
    TI_ASSERT(axis < ndim_ && axis < shape_.size());
    return shape_[axis];
  }

 private:
  uint32_t length_{};
  DataType dtype_{};
  intptr_t data_{};
  uint32_t ndim_{};
  std::array<uint32_t, 2> shape_{};
};

}  // namespace taichi::lang
