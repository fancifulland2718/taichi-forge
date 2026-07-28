#pragma once

#include "taichi/program/conjugate_gradient.h"

#include <cstdint>
#include <memory>
#include <mutex>

namespace taichi::lang {

class LinearOperatorHandle;
class OperatorPlan;
class OperatorPinnedAction;
class PreconditionerPlan;
struct DeviceGMRESCudaReplayState;
struct DeviceGMRESVulkanReplayState;

// Provider-neutral restarted GMRES(m)/FGMRES(m) for scalar f32 CUDA/Vulkan
// operators. FGMRES pins a finite variable-linear action table for one solve,
// selects it by solve-global scheduled inner slot, and keeps a separate Z
// basis.
class DeviceGMRES {
 public:
  DeviceGMRES(Program *program,
              LinearOperatorHandle &operator_handle,
              SparseMatrix *stored_matrix,
              LinearOperatorHandle *preconditioner,
              int max_iterations,
              int restart,
              float absolute_tolerance,
              float relative_tolerance,
              std::vector<LinearOperatorHandle *>
                  flexible_preconditioners = {});
  ~DeviceGMRES();

  DeviceGMRES(const DeviceGMRES &) = delete;
  DeviceGMRES &operator=(const DeviceGMRES &) = delete;

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
  bool flexible() const;
  bool native_stored_provider() const;
  std::uintptr_t address(const Ndarray *array) const;
  void apply_operator(const OperatorPinnedAction &generation,
                      const Ndarray &input,
                      const Ndarray &output,
                      void *stream,
                      bool native_capture);
  void apply_preconditioner(const OperatorPinnedAction &generation,
                            const Ndarray &input,
                            const Ndarray &output);
  void apply_preconditioner(PreconditionerPlan &plan,
                            const OperatorPinnedAction &generation,
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
                            int step,
                            bool limit_reached,
                            void *stream);
  void backend_basis(const Ndarray &source,
                     int row,
                     int mode,
                     void *stream);
  void backend_store_preconditioned_basis(int row, void *stream);
  void backend_multi_dot(int basis_count, void *stream);
  void backend_projection(int step, int pass, void *stream);
  void backend_combine(void *stream);
  void backend_add_update(const Ndarray &x,
                          const Ndarray &update,
                          void *stream);
  void issue_cycle(const OperatorPinnedAction &operator_generation,
                   const OperatorPinnedAction &preconditioner_generation,
                   const std::vector<OperatorPinnedAction>
                       &flexible_generations,
                   const Ndarray &x,
                   const Ndarray &b,
                   int cycle_steps,
                   int solve_iteration_offset,
                   bool limit_reached,
                   void *stream,
                   bool native_capture);
  bool try_submit_cuda_cycle(
      const OperatorPinnedAction &operator_generation,
      const OperatorPinnedAction &preconditioner_generation,
      const std::vector<OperatorPinnedAction> &flexible_generations,
      const Ndarray &x,
      const Ndarray &b,
      int cycle_steps,
      int solve_iteration_offset,
      bool limit_reached);
  bool try_submit_vulkan_cycle(
      const OperatorPinnedAction &operator_generation,
      const OperatorPinnedAction &preconditioner_generation,
      const std::vector<OperatorPinnedAction> &flexible_generations,
      const Ndarray &x,
      const Ndarray &b,
      int cycle_steps,
      int solve_iteration_offset,
      bool limit_reached,
      std::size_t slot_index);
  void read_state(bool synchronize);

  Program *program_{nullptr};
  LinearOperatorHandle *operator_handle_{nullptr};
  SparseMatrix *stored_matrix_{nullptr};
  LinearOperatorHandle *operator_preconditioner_{nullptr};
  std::vector<LinearOperatorHandle *>
      operator_flexible_preconditioners_;
  std::unique_ptr<OperatorPlan> operator_plan_;
  std::unique_ptr<PreconditionerPlan> preconditioner_plan_;
  std::vector<std::unique_ptr<PreconditionerPlan>>
      flexible_preconditioner_plans_;
  int rows_{0};
  int max_iterations_{0};
  int restart_{16};
  int multi_dot_groups_{0};
  float absolute_tolerance_{0.0f};
  float relative_tolerance_{0.0f};
  SparseSolveExecutionPolicy execution_policy_{
      SparseSolveExecutionPolicy::host_check_every_k};
  int host_check_interval_{16};

  Ndarray *basis_{nullptr};
  Ndarray *preconditioned_basis_{nullptr};
  Ndarray *residual_{nullptr};
  Ndarray *current_{nullptr};
  Ndarray *work_{nullptr};
  Ndarray *update_{nullptr};
  Ndarray *preconditioned_{nullptr};
  Ndarray *multi_dot_partials_{nullptr};
  Ndarray *projection_{nullptr};
  Ndarray *hessenberg_{nullptr};
  Ndarray *cosines_{nullptr};
  Ndarray *sines_{nullptr};
  Ndarray *least_squares_rhs_{nullptr};
  Ndarray *coefficients_{nullptr};
  Ndarray *initial_residual_squared_{nullptr};
  Ndarray *rhs_squared_{nullptr};
  Ndarray *dot0_{nullptr};
  Ndarray *dot1_{nullptr};
  Ndarray *state_{nullptr};

  void *cublas_handle_{nullptr};
  bool cublas_stream_bound_{false};
  bool cublas_device_pointer_mode_{false};
  std::unique_ptr<DeviceGMRESCudaReplayState> cuda_replay_;
  std::unique_ptr<DeviceGMRESVulkanReplayState> vulkan_replay_;

  std::uint32_t host_state_[32]{};
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
  std::uint64_t preconditioner_action_selections_{0};
  std::uint64_t preconditioner_schedule_wraps_{0};
  std::uint64_t dot_product_calls_{0};
  std::uint64_t multi_dot_calls_{0};
  std::uint64_t vector_update_calls_{0};
  std::uint64_t device_scalar_operations_{0};
  std::uint64_t host_scalar_readbacks_{0};
  std::uint64_t host_synchronizations_{0};
  std::uint64_t executed_iterations_{0};
  std::uint64_t restart_cycles_{0};
  std::uint64_t happy_breakdowns_{0};
  std::uint64_t solver_chunk_direct_submissions_{0};
  std::uint64_t solver_chunk_builds_{0};
  std::uint64_t solver_chunk_reuses_{0};
  std::uint64_t solver_chunk_replays_{0};
  std::uint64_t solver_chunk_rebinds_{0};
  std::uint64_t solver_chunk_invalidations_{0};
  std::uint64_t device_to_device_bytes_{0};
  std::uint64_t device_to_host_bytes_{0};
  std::uint64_t host_to_device_bytes_{0};
  std::uint64_t last_solve_pattern_version_{0};
  std::uint64_t last_solve_numeric_version_{0};
};

std::unique_ptr<DeviceGMRES> make_device_gmres_solver(
    Program *program,
    LinearOperatorHandle &operator_handle,
    SparseMatrix *stored_matrix,
    LinearOperatorHandle *preconditioner,
    int max_iterations,
    int restart,
    float absolute_tolerance,
    float relative_tolerance);
std::unique_ptr<DeviceGMRES> make_device_fgmres_solver(
    Program *program,
    LinearOperatorHandle &operator_handle,
    std::vector<LinearOperatorHandle *> preconditioners,
    int max_iterations,
    int restart,
    float absolute_tolerance,
    float relative_tolerance);

}  // namespace taichi::lang
