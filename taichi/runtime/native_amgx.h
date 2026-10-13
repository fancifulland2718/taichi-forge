#pragma once

#include <optional>
#include "taichi/runtime/native_operation_description.h"

namespace taichi::lang {
struct AmgxSolverDescription {
  static constexpr const char *kind = "amgx_solver_csr";
  static constexpr uint32_t version = 1;
  int32_t rows{0}, nonzeros{0};
  DataType dtype;
  std::string config;
  bool compute_residual{true};
};

class AmgxSolverResource : public NativeResource {
 public:
  const AmgxSolverDescription &description() const {
    return description_;
  }

 protected:
  explicit AmgxSolverResource(AmgxSolverDescription description)
      : description_(std::move(description)) {
  }
  AmgxSolverDescription description_;
};

struct AmgxSetupOperation {
  static constexpr const char *kind = "amgx_setup_csr";
  static constexpr uint32_t version = 1;
  std::shared_ptr<AmgxSolverResource> solver;
  NativeArrayBinding row_offsets, column_indices, values;
};
struct AmgxReplaceOperation {
  static constexpr const char *kind = "amgx_replace_coefficients";
  static constexpr uint32_t version = 1;
  std::shared_ptr<AmgxSolverResource> solver;
  NativeArrayBinding values;
};
struct AmgxSolveOperation {
  static constexpr const char *kind = "amgx_solve";
  static constexpr uint32_t version = 1;
  std::shared_ptr<AmgxSolverResource> solver;
  NativeArrayBinding rhs, solution;
  std::optional<NativeArrayBinding> status, iterations, residual;
  uint32_t initial_guess{1};  // 0 upload, 1 zero, 2 retained.
  bool retain_initial_guess{false};
};
}  // namespace taichi::lang
