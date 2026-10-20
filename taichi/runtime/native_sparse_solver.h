#pragma once

#include <array>
#include <optional>
#include "taichi/runtime/native_operation_description.h"

namespace taichi::lang {

// Frozen compressed topology and numerical policy. The consumer reconstructs
// its own factor; no process handle, Eigen object or workspace is serialized.
struct SparseDirectPlanDescription {
  static constexpr const char *kind = "sparse_direct_plan";
  static constexpr std::uint32_t version = 1;
  std::uint64_t rows{0}, nonzeros{0};
  DataType dtype{PrimitiveType::f32};
  std::string implementation{"eigen"};
  std::string method{"LLT"}, ordering{"AMD"}, format{"csc"};
  bool refactor{false};
  // Offsets/indices are i32; values follow their exact compressed order.
  std::array<NativeArrayBinding, 3> arrays;
};

class SparseDirectPlanResource : public NativeResource {
 public:
  const SparseDirectPlanDescription &description() const {
    return description_;
  }

 protected:
  explicit SparseDirectPlanResource(SparseDirectPlanDescription description)
      : description_(std::move(description)) {
  }
  SparseDirectPlanDescription description_;
};

// One optional numerical refactor followed by all RHS solves. Grouping avoids
// repeating the factorization for every RHS. Inputs may share read-only storage.
struct SparseDirectSolveOperation {
  static constexpr const char *kind = "sparse_direct_solve";
  static constexpr std::uint32_t version = 1;
  std::shared_ptr<SparseDirectPlanResource> plan;
  std::optional<NativeArrayBinding> values;
  std::vector<std::pair<NativeArrayBinding, NativeArrayBinding>> rhs_outputs;
};

// CUDA host numerical phases have a different composition contract from a
// fixed LLT solve. Both operations share the same reconstructed factor owner.
struct SparseDirectOrderedSolveOperation : SparseDirectSolveOperation {
  static constexpr const char *kind = "sparse_direct_ordered_solve";
  static constexpr std::uint32_t version = 1;
};

}  // namespace taichi::lang
