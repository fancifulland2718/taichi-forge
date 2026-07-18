#pragma once

#include "taichi/program/conjugate_gradient.h"

namespace taichi::lang {

// Private f32 correctness plan for caller-owned fixed CUDA CSR/BSR
// operators. Scalar recurrence decisions remain on the host through the
// existing CUBLAS host-pointer API, so this plan intentionally does not claim
// a device-scalar or Graph-ready execution path.
class CudaSparseBiCGSTABPlan {
 public:
  CudaSparseBiCGSTABPlan(Program *program,
                         SparseMatrix &matrix,
                         int max_iterations,
                         float absolute_tolerance,
                         bool verbose,
                         float relative_tolerance = 0.0f);
  ~CudaSparseBiCGSTABPlan();

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
  void ensure_workspace(int size);
  void release_workspace();
  void apply_operator(std::uintptr_t input, std::uintptr_t output);
  void copy_vector(float *destination, const float *source, int size);
  float dot(const float *left, const float *right, int size);
  bool vector_is_finite(const float *vector, int size);
  float true_residual_squared(const float *x, const float *b, int size);
  bool refresh_true_residual(const float *x,
                             const float *b,
                             int size);
  void finish_solve();

  Program *program_{nullptr};
  SparseMatrix &matrix_;
  CuSparseMatrix *csr_matrix_{nullptr};
  CuSparseBsrMatrix *bsr_matrix_{nullptr};
  cublasHandle_t handle_{nullptr};
  int max_iterations_{0};
  float absolute_tolerance_{0.0f};
  float relative_tolerance_{0.0f};
  bool verbose_{false};
  float *residual_{nullptr};
  float *shadow_residual_{nullptr};
  float *direction_{nullptr};
  float *operator_direction_{nullptr};
  float *intermediate_residual_{nullptr};
  float *operator_intermediate_{nullptr};
  int workspace_size_{0};
  mutable std::mutex solve_mutex_;
  SparseSolveStatus status_{SparseSolveStatus::kNotRun};
  int iterations_{0};
  double initial_residual_norm_{0.0};
  double residual_norm_{0.0};
  double relative_reference_norm_{0.0};
  double effective_tolerance_{0.0};
  std::uint64_t solve_calls_{0};
  std::uint64_t total_iterations_{0};
  std::uint64_t workspace_builds_{0};
  std::uint64_t workspace_reuses_{0};
  std::uint64_t operator_apply_calls_{0};
  std::uint64_t host_scalar_reductions_{0};
  std::uint64_t host_scalar_readbacks_{0};
  std::uint64_t host_synchronizations_{0};
  std::uint64_t device_to_device_bytes_{0};
  std::uint64_t device_to_host_bytes_{0};
  std::uint64_t last_solve_pattern_version_{0};
  std::uint64_t last_solve_numeric_version_{0};
};

std::unique_ptr<CudaSparseBiCGSTABPlan>
make_cuda_fixed_sparse_bicgstab_plan(
    Program *program,
    SparseMatrix &matrix,
    int max_iterations,
    float absolute_tolerance,
    bool verbose,
    float relative_tolerance = 0.0f);

}  // namespace taichi::lang
