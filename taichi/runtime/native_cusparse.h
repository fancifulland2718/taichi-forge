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

struct CsrTriangularF32Description {
  static constexpr const char *kind = "cusparse_triangular_plan_f32";
  static constexpr std::uint32_t version = 1;
  std::shared_ptr<CsrF32Resource> matrix;
  NativeArrayBinding input, output;
  std::int64_t rhs_count{1};
  bool upper{false}, unit_diagonal{false}, transpose{false};
  // Mutable numeric storage requires explicit refresh before using changed
  // values. Fixed plans retain their initial analysis with no warm refresh.
  bool mutable_values{false};
};

class CsrTriangularF32Resource : public NativeResource {
 public:
  const CsrTriangularF32Description &description() const {
    return description_;
  }

 protected:
  explicit CsrTriangularF32Resource(CsrTriangularF32Description description)
      : description_(std::move(description)) {
  }
  CsrTriangularF32Description description_;
};

template <bool Update>
struct CsrTriangularF32Operation {
  static constexpr const char *kind = Update ? "cusparse_triangular_update_f32"
                                             : "cusparse_triangular_solve_f32";
  static constexpr std::uint32_t version = 1;
  std::shared_ptr<CsrTriangularF32Resource> plan;
};

}  // namespace taichi::lang
