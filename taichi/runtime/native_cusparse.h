#pragma once

#include <array>
#include "taichi/runtime/native_operation_description.h"

namespace taichi::lang {

// Index arrays are immutable for this resource's lifetime; coefficients may be
// updated in place between ordered executions. The executor retains storage.
struct CsrF32Description {
  static constexpr const char *kind = "cusparse_csr_f32";
  static constexpr std::uint32_t version = 1;
  std::int64_t rows{0}, columns{0};
  std::array<NativeArrayBinding, 3> arrays;  // offsets, column indices, values.
};

class CsrF32Resource : public NativeResource {
 public:
  const CsrF32Description &description() const {
    return description_;
  }

 protected:
  explicit CsrF32Resource(CsrF32Description description)
      : description_(std::move(description)) {
  }
  CsrF32Description description_;
};

struct CsrSpmvF32Operation {
  static constexpr const char *kind = "cusparse_spmv_f32";
  static constexpr std::uint32_t version = 1;
  std::shared_ptr<CsrF32Resource> matrix;
  NativeArrayBinding input, output;
  bool transpose{false};
  bool preprocess{true};
  std::uint32_t algorithm{1};
  float alpha{1}, beta{0};
};

}  // namespace taichi::lang
