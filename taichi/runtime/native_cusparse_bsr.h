#pragma once

#include <array>
#include "taichi/runtime/native_operation_description.h"

namespace taichi::lang {

// Immutable block topology with mutable numeric values at the same address.
// Blocks are square and row-major; the executor retains every bound buffer.
struct BsrF32Description {
  static constexpr const char *kind = "cusparse_bsr_f32";
  static constexpr std::uint32_t version = 1;
  std::int64_t block_rows{0}, block_columns{0}, block_size{0};
  std::array<NativeArrayBinding, 3> arrays;
};

class BsrF32Resource : public NativeResource {
 public:
  const BsrF32Description &description() const {
    return description_;
  }

 protected:
  explicit BsrF32Resource(BsrF32Description description)
      : description_(std::move(description)) {
  }
  BsrF32Description description_;
};

struct BsrSpmvF32Operation {
  static constexpr const char *kind = "cusparse_bsr_spmv_f32";
  static constexpr std::uint32_t version = 1;
  std::shared_ptr<BsrF32Resource> matrix;
  NativeArrayBinding input, output;
  float alpha{1}, beta{0};
};

}  // namespace taichi::lang
