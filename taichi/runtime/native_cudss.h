#pragma once

#include <array>
#include <optional>
#include "taichi/runtime/native_operation_description.h"

namespace taichi::lang {

// Immutable topology and preparation policy; the CUDA resource takes a private
// snapshot. Numerical updates belong to an explicit solve group, not this seed.
struct CudssPlanDescription {
  static constexpr const char *kind = "cudss_plan_f32";
  static constexpr uint32_t version = 1;
  uint64_t rows{0}, nonzeros{0};
  int matrix_type{0}, matrix_view{0};
  int phase{0}, reordering{0}, solve_algorithm{0};
  std::array<NativeArrayBinding, 3> arrays;
};

// Same portable CSR/configuration seed, but preparation only analyzes it.
// Each invocation supplies values and owns an ordinary refactor transaction.
struct CudssTransactionPlanDescription : CudssPlanDescription {
  static constexpr const char *kind = "cudss_transaction_plan_f32";
};

class CudssPlanResource : public NativeResource {
 public:
  const CudssPlanDescription &description() const {
    return description_;
  }
  bool transactional() const {
    return transactional_;
  }

 protected:
  explicit CudssPlanResource(CudssPlanDescription description,
                             bool transactional = false)
      : description_(std::move(description)), transactional_(transactional) {
  }
  CudssPlanDescription description_;
  bool transactional_{false};
};

// A numerical phase runs once, followed by all RHS solves. Splitting this into
// independent operations would repeat the factorization and parameter storage.
struct CudssSolveGroupOperation {
  static constexpr const char *kind = "cudss_solve_group_f32";
  static constexpr uint32_t version = 1;
  std::shared_ptr<CudssPlanResource> plan;
  std::optional<NativeArrayBinding> values;
  std::vector<std::pair<NativeArrayBinding, NativeArrayBinding>> rhs_outputs;
  // Captured JIT graphs have a stricter binding contract than explicit groups.
  bool disjoint_bindings{false};
};

struct CudssRefactorSolveOperation {
  static constexpr const char *kind = "cudss_refactor_solve_f32";
  static constexpr uint32_t version = 1;
  std::shared_ptr<CudssPlanResource> plan;
  NativeArrayBinding values, rhs, output;
};

}  // namespace taichi::lang
