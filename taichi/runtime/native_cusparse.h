#pragma once

#include <array>
#include <optional>
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

struct CsrSpmmF32Operation {
  static constexpr const char *kind = "cusparse_spmm_f32";
  static constexpr std::uint32_t version = 1;
  enum class Preprocess { none, optional, required };
  std::shared_ptr<CsrF32Resource> matrix;
  NativeArrayBinding input, output;
  std::int64_t rhs_count{0};
  std::uint32_t algorithm{2};
  Preprocess preprocess{Preprocess::none};
  float alpha{1}, beta{0};
  std::optional<std::uint64_t> expected_workspace_bytes;
};

}  // namespace taichi::lang
