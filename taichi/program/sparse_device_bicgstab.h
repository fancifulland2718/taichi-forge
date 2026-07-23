#pragma once

#include "taichi/program/conjugate_gradient.h"

#include <cstdint>
#include <memory>
#include <mutex>

namespace taichi::lang {

class ExperimentalLinearOperatorHandle;
class OperatorPlan;
class OperatorPinnedAction;
class PreconditionerPlan;
struct DeviceBiCGSTABCudaReplayState;
struct DeviceBiCGSTABVulkanReplayState;

// Provider-neutral f32 BiCGSTAB for CUDA and Vulkan. The recurrence scalars
// and termination mask remain device-resident between explicit observation
// boundaries. A fixed LinearOperator may be consumed as a right
// preconditioner. Stored identity-preconditioned operators can replay a
// complete bounded iteration chunk; compiled actions retain direct submission.
class DeviceBiCGSTAB {
 public:
  DeviceBiCGSTAB(Program *program,
                 ExperimentalLinearOperatorHandle &operator_handle,
                 SparseMatrix *stored_matrix,
                 ExperimentalLinearOperatorHandle *preconditioner,
                 int max_iterations,
                 float absolute_tolerance,
                 float relative_tolerance);
  ~DeviceBiCGSTAB();

  DeviceBiCGSTAB(const DeviceBiCGSTAB &) = delete;
  DeviceBiCGSTAB &operator=(const DeviceBiCGSTAB &) = delete;

  void configure_execution_policy(SparseSolveExecutionPolicy policy,
                                  int host_check_interval);
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
  SparseSolveResult get_last_result() const;
  SparseSolvePlanRuntimeStatistics debug_runtime_statistics() const;

 private:
  void validate_controls() const;
  void allocate_workspace();
  void release_workspace();
  void initialize_cuda();
  void release_cuda();
  bool has_preconditioner() const;
  bool native_stored_provider() const;
  void apply_operator(const OperatorPinnedAction &generation,
                      const Ndarray &input,
                      const Ndarray &output,
                      void *stream,
                      bool native_capture);
  void apply_preconditioner(const OperatorPinnedAction &generation,
                            const Ndarray &input,
                            const Ndarray &output);
  void backend_dot(const Ndarray &left,
                   const Ndarray &right,
                   const Ndarray &output,
                   void *stream);
  void backend_true_residual(const OperatorPinnedAction &generation,
                             const Ndarray &x,
                             const Ndarray &b,
                             void *stream,
                             bool native_capture);
  void backend_scalar_stage(int stage,
                            bool limit_reached,
                            void *stream);
  void backend_direction(void *stream);
  void backend_intermediate(void *stream);
  void backend_commit(const Ndarray &x,
                      const Ndarray &solution_direction,
                      const Ndarray &solution_intermediate,
                      void *stream);
  void backend_reconcile(const Ndarray &x, void *stream);
  void issue_iteration(const OperatorPinnedAction &operator_generation,
                       const OperatorPinnedAction &preconditioner_generation,
                       const Ndarray &x,
                       void *stream,
                       bool native_capture);
  void issue_chunk(const OperatorPinnedAction &operator_generation,
                   const OperatorPinnedAction &preconditioner_generation,
                   const Ndarray &x,
                   const Ndarray &b,
                   int chunk_iterations,
                   bool limit_reached,
                   void *stream,
                   bool native_capture);
  bool try_submit_cuda_chunk(
      const OperatorPinnedAction &operator_generation,
      const OperatorPinnedAction &preconditioner_generation,
      const Ndarray &x,
      const Ndarray &b,
      int chunk_iterations,
      bool limit_reached);
  bool try_submit_vulkan_chunk(
      const OperatorPinnedAction &operator_generation,
      const OperatorPinnedAction &preconditioner_generation,
      const Ndarray &x,
      const Ndarray &b,
      int chunk_iterations,
      bool limit_reached,
      std::size_t slot_index);
  void read_state(bool synchronize);
  std::uintptr_t address(const Ndarray *array) const;

  Program *program_{nullptr};
  ExperimentalLinearOperatorHandle *operator_handle_{nullptr};
  SparseMatrix *stored_matrix_{nullptr};
  ExperimentalLinearOperatorHandle *operator_preconditioner_{nullptr};
  std::unique_ptr<OperatorPlan> operator_plan_;
  std::unique_ptr<PreconditionerPlan> preconditioner_plan_;
  int rows_{0};
  int max_iterations_{0};
  float absolute_tolerance_{0.0f};
  float relative_tolerance_{0.0f};
  SparseSolveExecutionPolicy execution_policy_{
      SparseSolveExecutionPolicy::host_check_every_k};
  int host_check_interval_{4};

  Ndarray *residual_{nullptr};
  Ndarray *shadow_residual_{nullptr};
  Ndarray *direction_{nullptr};
  Ndarray *operator_direction_{nullptr};
  Ndarray *intermediate_{nullptr};
  Ndarray *operator_intermediate_{nullptr};
  Ndarray *preconditioned_direction_{nullptr};
  Ndarray *preconditioned_intermediate_{nullptr};
  Ndarray *initial_residual_squared_{nullptr};
  Ndarray *rhs_squared_{nullptr};
  Ndarray *dot0_{nullptr};
  Ndarray *dot1_{nullptr};
  Ndarray *state_{nullptr};

  void *cublas_handle_{nullptr};
  void *solver_stream_{nullptr};
  bool cublas_stream_bound_{false};
  bool cublas_device_pointer_mode_{false};
  std::unique_ptr<DeviceBiCGSTABCudaReplayState> cuda_replay_;
  std::unique_ptr<DeviceBiCGSTABVulkanReplayState> vulkan_replay_;

  std::uint32_t host_state_[24]{};
  SparseSolveStatus status_{SparseSolveStatus::kNotRun};
  SparseSolveBreakdownReason breakdown_reason_{
      SparseSolveBreakdownReason::none};
  int iterations_{0};
  double initial_residual_norm_{0.0};
  double residual_norm_{0.0};
  double relative_reference_norm_{0.0};
  double effective_tolerance_{0.0};
  mutable std::mutex solve_mutex_;

  std::uint64_t solve_calls_{0};
  std::uint64_t total_iterations_{0};
  std::uint64_t workspace_reuses_{0};
  std::uint64_t operator_apply_calls_{0};
  std::uint64_t preconditioner_apply_calls_{0};
  std::uint64_t dot_product_calls_{0};
  std::uint64_t vector_update_calls_{0};
  std::uint64_t device_scalar_operations_{0};
  std::uint64_t host_scalar_readbacks_{0};
  std::uint64_t host_synchronizations_{0};
  std::uint64_t executed_iterations_{0};
  std::uint64_t solver_chunk_builds_{0};
  std::uint64_t solver_chunk_reuses_{0};
  std::uint64_t solver_chunk_direct_submissions_{0};
  std::uint64_t solver_chunk_replays_{0};
  std::uint64_t solver_chunk_rebinds_{0};
  std::uint64_t solver_chunk_invalidations_{0};
  std::uint64_t device_to_device_bytes_{0};
  std::uint64_t device_to_host_bytes_{0};
  std::uint64_t host_to_device_bytes_{0};
  std::uint64_t last_solve_pattern_version_{0};
  std::uint64_t last_solve_numeric_version_{0};
};

std::unique_ptr<DeviceBiCGSTAB> make_device_bicgstab_solver(
    Program *program,
    ExperimentalLinearOperatorHandle &operator_handle,
    SparseMatrix *stored_matrix,
    ExperimentalLinearOperatorHandle *preconditioner,
    int max_iterations,
    float absolute_tolerance,
    float relative_tolerance);

}  // namespace taichi::lang
