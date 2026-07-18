#pragma once

#include "taichi/program/conjugate_gradient.h"

namespace taichi::lang {

// Private f32 correctness plan for caller-owned fixed Vulkan CSR/BSR
// operators. The plan keeps recurrence scalars device-resident and submits a
// bounded masked sequence, with one synchronization and one packed state
// readback at the end of each solve. Per-iteration true-residual replacement
// is intentionally conservative and this probe makes no performance claim.
class VulkanSparseBiCGSTABPlan {
 public:
  VulkanSparseBiCGSTABPlan(Program *program,
                           SparseMatrix &matrix,
                           int max_iterations,
                           float absolute_tolerance,
                           bool verbose,
                           float relative_tolerance = 0.0f);
  ~VulkanSparseBiCGSTABPlan();

  void solve(Program *program, const Ndarray &x, const Ndarray &b);

  bool is_success() const {
    return status_ == SparseSolveStatus::kConverged;
  }

  int get_status() const {
    return static_cast<int>(status_);
  }

  int get_iterations() const {
    return iterations_;
  }

  double get_initial_residual_norm() const {
    return initial_residual_norm_;
  }

  double get_residual_norm() const {
    return residual_norm_;
  }

  SparseSolveResult get_last_result() const {
    return {status_, iterations_, initial_residual_norm_, residual_norm_,
            static_cast<double>(absolute_tolerance_),
            static_cast<double>(relative_tolerance_),
            relative_reference_norm_, effective_tolerance_};
  }

  SparseSolvePlanRuntimeStatistics debug_runtime_statistics() const;

 private:
  void validate_controls() const;
  void apply_operator(Program *program,
                      const Ndarray &input,
                      const Ndarray &output,
                      bool masked);
  void release_workspace();

  Program *program_{nullptr};
  SparseMatrix &matrix_;
  VulkanSparseMatrix *csr_matrix_{nullptr};
  VulkanSparseBsrMatrix *bsr_matrix_{nullptr};
  int max_iterations_{0};
  float absolute_tolerance_{0.0f};
  float relative_tolerance_{0.0f};
  bool verbose_{false};
  Ndarray *residual_{nullptr};
  Ndarray *shadow_residual_{nullptr};
  Ndarray *direction_{nullptr};
  Ndarray *operator_direction_{nullptr};
  Ndarray *intermediate_residual_{nullptr};
  Ndarray *operator_intermediate_{nullptr};
  Ndarray *true_residual_{nullptr};
  Ndarray *candidate_solution_{nullptr};
  Ndarray *state_{nullptr};
  mutable std::mutex solve_mutex_;
  SparseSolveStatus status_{SparseSolveStatus::kNotRun};
  int iterations_{0};
  double initial_residual_norm_{0.0};
  double residual_norm_{0.0};
  double relative_reference_norm_{0.0};
  double effective_tolerance_{0.0};
  bool has_solved_{false};
  std::uint64_t solve_calls_{0};
  std::uint64_t total_iterations_{0};
  std::uint64_t workspace_builds_{0};
  std::uint64_t workspace_reuses_{0};
  std::uint64_t operator_apply_calls_{0};
  std::uint64_t masked_operator_dispatches_{0};
  std::uint64_t device_scalar_operations_{0};
  std::uint64_t host_scalar_readbacks_{0};
  std::uint64_t host_synchronizations_{0};
  std::uint64_t device_to_device_bytes_{0};
  std::uint64_t device_to_host_bytes_{0};
  std::uint64_t last_solve_pattern_version_{0};
  std::uint64_t last_solve_numeric_version_{0};
};

std::unique_ptr<VulkanSparseBiCGSTABPlan>
make_vulkan_fixed_sparse_bicgstab_plan(
    Program *program,
    SparseMatrix &matrix,
    int max_iterations,
    float absolute_tolerance,
    bool verbose,
    float relative_tolerance = 0.0f);

}  // namespace taichi::lang
