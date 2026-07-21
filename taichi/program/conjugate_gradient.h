#pragma once

#include "sparse_matrix.h"

#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"

#include "Eigen/IterativeLinearSolvers"

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstdint>
#include <mutex>

namespace taichi::lang {

class SparseJacobiPreconditionerPlan;
class SparseBlockJacobiPreconditionerPlan;
class CompiledKernelPreconditionerPlan;
class OperatorPlan;
class OperatorBinding;
class OperatorPinnedAction;

enum class SparseSolveStatus : int {
  kNotRun = -1,
  kMaxIterations = 0,
  kBreakdown = 1,
  kConverged = 2,
};

struct SparseSolveResult {
  SparseSolveStatus status{SparseSolveStatus::kNotRun};
  int iterations{0};
  double initial_residual_norm{0.0};
  double residual_norm{0.0};
  double absolute_tolerance{0.0};
  double relative_tolerance{0.0};
  double relative_reference_norm{0.0};
  double effective_tolerance{0.0};

  bool converged() const {
    return status == SparseSolveStatus::kConverged;
  }

  bool breakdown() const {
    return status == SparseSolveStatus::kBreakdown;
  }

  bool reached_max_iterations() const {
    return status == SparseSolveStatus::kMaxIterations;
  }
};

struct SparseSolvePlanRuntimeStatistics {
  std::string backend_family{"unknown"};
  std::string method{"cg"};
  std::string dtype{"unknown"};
  int rows{0};
  int cols{0};
  int max_iterations{0};
  double absolute_tolerance{0.0};
  double relative_tolerance{0.0};
  double last_relative_reference_norm{0.0};
  double last_effective_tolerance{0.0};

  std::uint64_t operator_pattern_version{0};
  std::uint64_t operator_numeric_version{0};
  std::uint64_t last_solve_pattern_version{0};
  std::uint64_t last_solve_numeric_version{0};
  bool operator_pattern_changed_since_last_solve{false};
  bool operator_numeric_changed_since_last_solve{false};

  std::uint64_t solve_calls{0};
  std::uint64_t total_iterations{0};
  std::uint64_t workspace_builds{0};
  std::uint64_t workspace_reuses{0};
  std::uint64_t operator_apply_calls{0};
  bool operator_apply_calls_available{false};
  std::uint64_t host_scalar_reductions{0};
  std::uint64_t device_scalar_operations{0};
  std::uint64_t host_scalar_readbacks{0};
  std::uint64_t host_synchronizations{0};
  bool fixed_iteration_only{false};
  bool bounded_masked_execution{false};
  std::string preconditioner_method{"identity"};
  std::uint64_t preconditioner_apply_calls{0};
  bool preconditioner_apply_calls_available{true};

  std::uint64_t persistent_vector_count{0};
  std::uint64_t persistent_vector_reserved_bytes{0};
  std::uint64_t persistent_scalar_count{0};
  std::uint64_t persistent_scalar_reserved_bytes{0};
  std::uint64_t cublas_handle_count{0};
  bool external_preconditioner{false};
  std::string preconditioner_ownership_scope{"none"};
  bool solver_state_rebuilt_each_solve{false};
  std::uint64_t transient_solver_workspace_bytes{0};
  bool transient_solver_workspace_bytes_available{false};

  std::uint64_t device_to_device_bytes{0};
  std::uint64_t device_to_host_bytes{0};
  std::uint64_t host_to_device_bytes{0};
};

template <typename EigenT, typename DT>
class CG {
 public:
  using EigenMatrix = Eigen::SparseMatrix<DT>;
  using EigenSolver = Eigen::ConjugateGradient<
      EigenMatrix,
      Eigen::Lower | Eigen::Upper>;

  CG(SparseMatrix &A,
     int max_iters,
     DT absolute_tolerance,
     bool verbose,
     DT relative_tolerance = static_cast<DT>(0))
      : A_(A),
        max_iters_(max_iters),
        absolute_tolerance_(absolute_tolerance),
        relative_tolerance_(relative_tolerance),
        verbose_(verbose) {
    const auto operator_stats = A_.debug_runtime_statistics();
    TI_ERROR_IF(
        operator_stats.backend_family != "cpu" ||
            (operator_stats.storage_format != "csr" &&
             operator_stats.storage_format != "csc"),
        "CPU SparseCG currently requires an Eigen CSR/CSC matrix.");
    TI_ERROR_IF(A_.num_rows() <= 0 || A_.num_rows() != A_.num_cols(),
                "CPU SparseCG requires a non-empty square matrix.");
    TI_ERROR_IF(max_iters_ < 0,
                "CPU SparseCG requires non-negative max iterations.");
    TI_ERROR_IF(!std::isfinite(absolute_tolerance_) ||
                    !std::isfinite(relative_tolerance_) ||
                    absolute_tolerance_ < static_cast<DT>(0) ||
                    relative_tolerance_ < static_cast<DT>(0) ||
                    (absolute_tolerance_ == static_cast<DT>(0) &&
                     relative_tolerance_ == static_cast<DT>(0)),
                "CPU SparseCG requires finite non-negative atol and rtol "
                "with at least one positive tolerance.");
    x_ = EigenT::Zero(A_.num_cols());
    b_ = EigenT::Zero(A_.num_rows());
    cg_.setMaxIterations(max_iters_);
  }

  void set_x(EigenT &x) {
    x_ = x;
  }

  void reset_x() {
    x_.setZero();
  }

  void set_b(EigenT &b) {
    b_ = b;
  }

  void set_x_ndarray(Program *prog, Ndarray &x) {
    size_t dX = prog->get_ndarray_data_ptr_as_int(&x);
    x_ = Eigen::Map<EigenT>((DT *)dX, A_.num_cols());
  }

  void set_b_ndarray(Program *prog, Ndarray &b) {
    size_t db = prog->get_ndarray_data_ptr_as_int(&b);
    b_ = Eigen::Map<EigenT>((DT *)db, A_.num_rows());
  }

  void solve() {
    const auto operator_stats = A_.debug_runtime_statistics();
    status_ = SparseSolveStatus::kNotRun;
    iterations_ = 0;
    initial_residual_norm_ = 0.0;
    residual_norm_ = 0.0;
    relative_reference_norm_ = 0.0;
    effective_tolerance_ = static_cast<double>(absolute_tolerance_);
    solve_calls_.fetch_add(1, std::memory_order_relaxed);
    last_solve_pattern_version_.store(operator_stats.pattern_version,
                                      std::memory_order_relaxed);
    last_solve_numeric_version_.store(operator_stats.numeric_version,
                                      std::memory_order_relaxed);
    EigenSparseMatrix<EigenMatrix> &A =
        static_cast<EigenSparseMatrix<EigenMatrix> &>(A_);
    EigenMatrix *A_eigen = (EigenMatrix *)A.get_matrix();
    initial_residual_norm_ = ((*A_eigen) * x_ - b_).norm();
    const auto b_norm = b_.norm();
    relative_reference_norm_ = static_cast<double>(b_norm);
    effective_tolerance_ = std::max(
        static_cast<double>(absolute_tolerance_),
        static_cast<double>(relative_tolerance_) *
            relative_reference_norm_);
    if (!std::isfinite(effective_tolerance_)) {
      residual_norm_ = initial_residual_norm_;
      status_ = SparseSolveStatus::kBreakdown;
      return;
    }
    cg_.setTolerance(
        b_norm > 0
            ? static_cast<DT>(effective_tolerance_ /
                              relative_reference_norm_)
            : relative_tolerance_);
    const bool solver_state_current =
        solver_state_initialized_ &&
        solver_state_pattern_version_ == operator_stats.pattern_version &&
        solver_state_numeric_version_ == operator_stats.numeric_version;
    if (solver_state_current) {
      workspace_reuses_.fetch_add(1, std::memory_order_relaxed);
    } else {
      cg_.compute(*A_eigen);
      workspace_builds_.fetch_add(1, std::memory_order_relaxed);
      solver_state_initialized_ = cg_.info() == Eigen::Success;
      if (solver_state_initialized_) {
        solver_state_pattern_version_ = operator_stats.pattern_version;
        solver_state_numeric_version_ = operator_stats.numeric_version;
      }
    }
    x_ = cg_.solveWithGuess(b_, x_);
    iterations_ = cg_.iterations();
    total_iterations_.fetch_add(static_cast<std::uint64_t>(iterations_),
                                std::memory_order_relaxed);
    residual_norm_ = ((*A_eigen) * x_ - b_).norm();
    if (verbose_) {
      std::cout << "#iterations:     " << iterations_ << std::endl;
      std::cout << "estimated error: " << cg_.error() << std::endl;
      std::cout << "residual norm:   " << residual_norm_ << std::endl;
    }
    const bool finite_residuals = std::isfinite(initial_residual_norm_) &&
                                  std::isfinite(residual_norm_);
    if (!finite_residuals || cg_.info() == Eigen::NumericalIssue ||
        cg_.info() == Eigen::InvalidInput) {
      status_ = SparseSolveStatus::kBreakdown;
    } else if (residual_norm_ <= effective_tolerance_) {
      status_ = SparseSolveStatus::kConverged;
    } else {
      status_ = SparseSolveStatus::kMaxIterations;
    }
  }

  EigenT &get_x() {
    return x_;
  }

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

  SparseSolvePlanRuntimeStatistics debug_runtime_statistics() const {
    const auto operator_stats = A_.debug_runtime_statistics();
    SparseSolvePlanRuntimeStatistics result;
    result.backend_family = "cpu";
    result.dtype = data_type_name(A_.get_data_type());
    result.rows = A_.num_rows();
    result.cols = A_.num_cols();
    result.max_iterations = max_iters_;
    result.absolute_tolerance =
        static_cast<double>(absolute_tolerance_);
    result.relative_tolerance =
        static_cast<double>(relative_tolerance_);
    result.last_relative_reference_norm = relative_reference_norm_;
    result.last_effective_tolerance = effective_tolerance_;
    result.operator_pattern_version = operator_stats.pattern_version;
    result.operator_numeric_version = operator_stats.numeric_version;
    result.last_solve_pattern_version =
        last_solve_pattern_version_.load(std::memory_order_relaxed);
    result.last_solve_numeric_version =
        last_solve_numeric_version_.load(std::memory_order_relaxed);
    result.solve_calls = solve_calls_.load(std::memory_order_relaxed);
    result.operator_pattern_changed_since_last_solve =
        result.solve_calls > 0 && result.operator_pattern_version !=
                                      result.last_solve_pattern_version;
    result.operator_numeric_changed_since_last_solve =
        result.solve_calls > 0 && result.operator_numeric_version !=
                                      result.last_solve_numeric_version;
    result.total_iterations =
        total_iterations_.load(std::memory_order_relaxed);
    result.workspace_builds =
        workspace_builds_.load(std::memory_order_relaxed);
    result.workspace_reuses =
        workspace_reuses_.load(std::memory_order_relaxed);
    result.preconditioner_method = "jacobi";
    result.preconditioner_apply_calls_available = false;
    result.preconditioner_ownership_scope = "provider_state";
    result.persistent_vector_count = 2;
    result.persistent_vector_reserved_bytes =
        (static_cast<std::uint64_t>(A_.num_cols()) +
         static_cast<std::uint64_t>(A_.num_rows())) *
        sizeof(DT);
    result.solver_state_rebuilt_each_solve = false;
    return result;
  }

 private:
  SparseMatrix &A_;
  EigenSolver cg_;
  EigenT x_;
  EigenT b_;
  int max_iters_{0};
  DT absolute_tolerance_{0.0f};
  DT relative_tolerance_{0.0f};
  bool verbose_{false};
  SparseSolveStatus status_{SparseSolveStatus::kNotRun};
  int iterations_{0};
  double initial_residual_norm_{0.0};
  double residual_norm_{0.0};
  double relative_reference_norm_{0.0};
  double effective_tolerance_{0.0};
  std::atomic<std::uint64_t> solve_calls_{0};
  std::atomic<std::uint64_t> total_iterations_{0};
  std::atomic<std::uint64_t> workspace_builds_{0};
  std::atomic<std::uint64_t> workspace_reuses_{0};
  std::atomic<std::uint64_t> last_solve_pattern_version_{0};
  std::atomic<std::uint64_t> last_solve_numeric_version_{0};
  bool solver_state_initialized_{false};
  std::uint64_t solver_state_pattern_version_{0};
  std::uint64_t solver_state_numeric_version_{0};
};

template <typename EigenT, typename DT>
std::unique_ptr<CG<EigenT, DT>> make_cg_solver(SparseMatrix &A,
                                               int max_iters,
                                               DT absolute_tolerance,
                                               bool verbose,
                                               DT relative_tolerance =
                                                   static_cast<DT>(0)) {
  return std::make_unique<CG<EigenT, DT>>(
      A, max_iters, absolute_tolerance, verbose, relative_tolerance);
}

class CUCG {
 public:
  CUCG(SparseMatrix &A,
       int max_iters,
       float absolute_tolerance,
       bool verbose,
       float relative_tolerance = 0.0f)
      : A_(A),
        max_iters_(max_iters),
        absolute_tolerance_(absolute_tolerance),
        relative_tolerance_(relative_tolerance),
        verbose_(verbose) {
    TI_ERROR_IF(dynamic_cast<CuSparseMatrix *>(&A_) == nullptr,
                "CUDA conjugate gradient currently requires a CSR matrix.");
    validate_controls();
    init_solver();
  }
  CUCG(Program *program,
       SparseMatrix &A,
       SparseJacobiPreconditionerPlan &preconditioner,
       int max_iters,
       float absolute_tolerance,
       bool verbose,
       float relative_tolerance = 0.0f);
  CUCG(Program *program,
       SparseMatrix &A,
       SparseBlockJacobiPreconditionerPlan &preconditioner,
       int max_iters,
       float absolute_tolerance,
       bool verbose,
       float relative_tolerance = 0.0f);
  CUCG(Program *program,
       CompiledKernelLinearOperator &A,
       CompiledKernelPreconditionerPlan *preconditioner,
       int max_iters,
       float absolute_tolerance,
       bool verbose,
       float relative_tolerance = 0.0f);

  ~CUCG();

  void solve(Program *prog, const Ndarray &x, const Ndarray &b);

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
  void init_solver();
  void validate_controls() const;
  void ensure_workspace(Program *program, int size);
  void release_workspace();
  bool has_preconditioner() const;
  void validate_preconditioner(Program *program) const;
  void apply_preconditioner(Program *program,
                            float *input,
                            float *output,
                            const Ndarray *input_array,
                            const Ndarray *output_array);
  void apply_operator(Program *program,
                      std::uintptr_t input,
                      std::uintptr_t output,
                      const Ndarray *input_array,
                      const Ndarray *output_array);

  cublasHandle_t handle_{nullptr};
  Program *program_{nullptr};
  SparseMatrix &A_;
  SparseJacobiPreconditionerPlan *preconditioner_{nullptr};
  SparseBlockJacobiPreconditionerPlan *block_preconditioner_{nullptr};
  CompiledKernelLinearOperator *compiled_kernel_operator_{nullptr};
  CompiledKernelPreconditionerPlan *compiled_kernel_preconditioner_{nullptr};
  int max_iters_{0};
  float absolute_tolerance_{0.0f};
  float relative_tolerance_{0.0f};
  bool verbose_{false};
  SparseSolveStatus status_{SparseSolveStatus::kNotRun};
  int iterations_{0};
  double initial_residual_norm_{0.0};
  double residual_norm_{0.0};
  double relative_reference_norm_{0.0};
  double effective_tolerance_{0.0};
  mutable std::mutex solve_mutex_;
  float *workspace_ax_{nullptr};
  float *workspace_r_{nullptr};
  float *workspace_p_{nullptr};
  float *workspace_z_{nullptr};
  Ndarray *workspace_ax_ndarray_{nullptr};
  Ndarray *workspace_r_ndarray_{nullptr};
  Ndarray *workspace_p_ndarray_{nullptr};
  Ndarray *workspace_z_ndarray_{nullptr};
  int workspace_size_{0};
  std::uint64_t solve_calls_{0};
  std::uint64_t total_iterations_{0};
  std::uint64_t workspace_builds_{0};
  std::uint64_t workspace_reuses_{0};
  std::uint64_t operator_apply_calls_{0};
  std::uint64_t preconditioner_apply_calls_{0};
  std::uint64_t host_scalar_reductions_{0};
  std::uint64_t device_to_device_bytes_{0};
  std::uint64_t last_solve_pattern_version_{0};
  std::uint64_t last_solve_numeric_version_{0};
};

std::unique_ptr<CUCG> make_cucg_solver(SparseMatrix &A,
                                       int max_iters,
                                       float absolute_tolerance,
                                       bool verbose,
                                       float relative_tolerance = 0.0f);

std::unique_ptr<CUCG> make_cuda_jacobi_pcg_solver(
    Program *program,
    SparseMatrix &A,
    SparseJacobiPreconditionerPlan &preconditioner,
    int max_iters,
    float absolute_tolerance,
    bool verbose,
    float relative_tolerance = 0.0f);

std::unique_ptr<CUCG> make_cuda_block_jacobi_pcg_solver(
    Program *program,
    SparseMatrix &A,
    SparseBlockJacobiPreconditionerPlan &preconditioner,
    int max_iters,
    float absolute_tolerance,
    bool verbose,
    float relative_tolerance = 0.0f);

std::unique_ptr<CUCG> make_cuda_compiled_kernel_cg_solver(
    Program *program,
    CompiledKernelLinearOperator &A,
    int max_iters,
    float absolute_tolerance,
    bool verbose,
    float relative_tolerance = 0.0f);

std::unique_ptr<CUCG> make_cuda_compiled_kernel_pcg_solver(
    Program *program,
    CompiledKernelLinearOperator &A,
    CompiledKernelPreconditionerPlan &preconditioner,
    int max_iters,
    float absolute_tolerance,
    bool verbose,
    float relative_tolerance = 0.0f);

class CpuSparseCGPlan {
 public:
  CpuSparseCGPlan(Program *program,
                  SparseMatrix &matrix,
                  int max_iterations,
                  double absolute_tolerance,
                  double relative_tolerance = 0.0);
  CpuSparseCGPlan(Program *program,
               SparseMatrix &matrix,
               SparseJacobiPreconditionerPlan &preconditioner,
               int max_iterations,
               double absolute_tolerance,
               double relative_tolerance = 0.0);
  CpuSparseCGPlan(Program *program,
               SparseMatrix &matrix,
               SparseBlockJacobiPreconditionerPlan &preconditioner,
               int max_iterations,
               double absolute_tolerance,
               double relative_tolerance = 0.0);
  CpuSparseCGPlan(Program *program,
                  CompiledKernelLinearOperator &matrix,
                  CompiledKernelPreconditionerPlan &preconditioner,
                  int max_iterations,
                  double absolute_tolerance,
                  double relative_tolerance = 0.0);
  ~CpuSparseCGPlan();

  void solve(Program *program, const Ndarray &x, const Ndarray &b);

  bool is_success() const {
    return status_ == SparseSolveStatus::kConverged;
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

  int get_status() const {
    return static_cast<int>(status_);
  }

  SparseSolveResult get_last_result() const {
    return {status_, iterations_, initial_residual_norm_, residual_norm_,
            absolute_tolerance_, relative_tolerance_,
            relative_reference_norm_, effective_tolerance_};
  }

  SparseSolvePlanRuntimeStatistics debug_runtime_statistics() const;

 private:
  struct PreconditionerBinding;

  template <typename T>
  void solve_typed(T *x,
                   const T *b,
                   const std::array<T *, 4> &workspace,
                   const Ndarray &solution_array,
                   const OperatorPinnedAction &operator_generation,
                   const OperatorPinnedAction *preconditioner_generation);
  CpuSparseCGPlan(Program *program,
                  OperatorBinding operator_binding,
                  std::unique_ptr<PreconditionerBinding> preconditioner,
                  int max_iterations,
                  double absolute_tolerance,
                  double relative_tolerance);
  static std::unique_ptr<PreconditionerBinding> bind_preconditioner(
      Program *program,
      SparseMatrix &matrix,
      SparseJacobiPreconditionerPlan &preconditioner);
  static std::unique_ptr<PreconditionerBinding> bind_preconditioner(
      Program *program,
      SparseMatrix &matrix,
      SparseBlockJacobiPreconditionerPlan &preconditioner);
  static std::unique_ptr<PreconditionerBinding> bind_preconditioner(
      Program *program,
      CompiledKernelLinearOperator &matrix,
      CompiledKernelPreconditionerPlan &preconditioner);
  void validate_preconditioner(Program *program) const;
  void apply_operator(const OperatorPinnedAction &generation,
                      const Ndarray &input,
                      const Ndarray &output);
  void apply_preconditioner(const OperatorPinnedAction &generation,
                            const Ndarray &input,
                            const Ndarray &output);
  void release_workspace();

  Program *program_{nullptr};
  std::unique_ptr<PreconditionerBinding> preconditioner_binding_;
  std::unique_ptr<OperatorPlan> operator_plan_;
  std::unique_ptr<OperatorPlan> preconditioner_plan_;
  DataType dtype_{PrimitiveType::f32};
  int rows_{0};
  int cols_{0};
  int max_iterations_{0};
  double absolute_tolerance_{0.0};
  double relative_tolerance_{0.0};
  std::array<Ndarray *, 4> workspace_{};
  mutable std::mutex solve_mutex_;
  bool has_solved_{false};
  SparseSolveStatus status_{SparseSolveStatus::kNotRun};
  int iterations_{0};
  double initial_residual_norm_{0.0};
  double residual_norm_{0.0};
  double relative_reference_norm_{0.0};
  double effective_tolerance_{0.0};
  std::uint64_t solve_calls_{0};
  std::uint64_t total_iterations_{0};
  std::uint64_t workspace_builds_{1};
  std::uint64_t workspace_reuses_{0};
  std::uint64_t operator_apply_calls_{0};
  std::uint64_t preconditioner_apply_calls_{0};
  std::uint64_t host_scalar_reductions_{0};
  std::uint64_t last_solve_pattern_version_{0};
  std::uint64_t last_solve_numeric_version_{0};
};

// Preserve the old private C++ name while the implementation now serves both
// fixed CSR/scalar-Jacobi and BSR/block-Jacobi operators.
using CpuBsrCGPlan = CpuSparseCGPlan;

std::unique_ptr<CpuSparseCGPlan> make_cpu_operator_cg_solver(
    Program *program,
    SparseMatrix &matrix,
    int max_iterations,
    double absolute_tolerance,
    double relative_tolerance = 0.0);

std::unique_ptr<CpuSparseCGPlan> make_cpu_jacobi_pcg_solver(
    Program *program,
    SparseMatrix &matrix,
    SparseJacobiPreconditionerPlan &preconditioner,
    int max_iterations,
    double absolute_tolerance,
    double relative_tolerance = 0.0);

std::unique_ptr<CpuSparseCGPlan> make_cpu_block_jacobi_pcg_solver(
    Program *program,
    SparseMatrix &matrix,
    SparseBlockJacobiPreconditionerPlan &preconditioner,
    int max_iterations,
    double absolute_tolerance,
    double relative_tolerance = 0.0);

std::unique_ptr<CpuSparseCGPlan> make_cpu_compiled_kernel_pcg_solver(
    Program *program,
    CompiledKernelLinearOperator &matrix,
    CompiledKernelPreconditionerPlan &preconditioner,
    int max_iterations,
    double absolute_tolerance,
    double relative_tolerance = 0.0);

class VulkanCGIterationPlan {
 public:
  VulkanCGIterationPlan(Program *program,
                        SparseMatrix &matrix,
                        int fixed_iterations);
  VulkanCGIterationPlan(Program *program,
                        SparseMatrix &matrix,
                        int max_iterations,
                        float absolute_tolerance);
  VulkanCGIterationPlan(
      Program *program,
      SparseMatrix &matrix,
      SparseJacobiPreconditionerPlan &preconditioner,
      int max_iterations,
      float absolute_tolerance);
  VulkanCGIterationPlan(
      Program *program,
      SparseMatrix &matrix,
      SparseBlockJacobiPreconditionerPlan &preconditioner,
      int max_iterations,
      float absolute_tolerance);
  VulkanCGIterationPlan(Program *program,
                        CompiledKernelLinearOperator &matrix,
                        int max_iterations,
                        float absolute_tolerance);
  VulkanCGIterationPlan(
      Program *program,
      CompiledKernelLinearOperator &matrix,
      CompiledKernelPreconditionerPlan &preconditioner,
      int max_iterations,
      float absolute_tolerance);
  ~VulkanCGIterationPlan();

  void solve(Program *program, const Ndarray &x, const Ndarray &b);

  bool is_success() const {
    return is_success_;
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

  int get_status() const {
    return status_;
  }

  SparseSolveResult get_last_result() const {
    return {static_cast<SparseSolveStatus>(status_), iterations_,
            initial_residual_norm_, residual_norm_,
            static_cast<double>(absolute_tolerance_), 0.0, 0.0,
            static_cast<double>(absolute_tolerance_)};
  }

  SparseSolvePlanRuntimeStatistics debug_runtime_statistics() const;

 private:
  VulkanCGIterationPlan(Program *program,
                        SparseMatrix &matrix,
                        int max_iterations,
                        float absolute_tolerance,
                        bool adaptive,
                        bool allow_compiled_kernel_operator,
                        SparseJacobiPreconditionerPlan *preconditioner,
                        SparseBlockJacobiPreconditionerPlan
                            *block_preconditioner,
                        CompiledKernelPreconditionerPlan
                            *compiled_kernel_preconditioner);
  bool has_preconditioner() const;
  void validate_preconditioner(Program *program) const;
  void apply_preconditioner(Program *program,
                            const Ndarray &input,
                            const Ndarray &output);
  void apply_operator(Program *program,
                      const Ndarray &input,
                      const Ndarray &output);
  void release_workspace();

  Program *program_{nullptr};
  SparseMatrix *matrix_{nullptr};
  VulkanSparseMatrix *csr_matrix_{nullptr};
  VulkanSparseBsrMatrix *bsr_matrix_{nullptr};
  SparseJacobiPreconditionerPlan *preconditioner_{nullptr};
  SparseBlockJacobiPreconditionerPlan *block_preconditioner_{nullptr};
  CompiledKernelPreconditionerPlan *compiled_kernel_preconditioner_{nullptr};
  int fixed_iterations_{0};
  float absolute_tolerance_{0.0f};
  bool adaptive_{false};
  CompiledKernelLinearOperator *compiled_kernel_operator_{nullptr};
  Ndarray *ap_{nullptr};
  Ndarray *residual_{nullptr};
  Ndarray *direction_{nullptr};
  Ndarray *preconditioned_residual_{nullptr};
  Ndarray *initial_rr_{nullptr};
  Ndarray *rr_a_{nullptr};
  Ndarray *rr_b_{nullptr};
  Ndarray *p_ap_{nullptr};
  Ndarray *alpha_{nullptr};
  Ndarray *beta_{nullptr};
  Ndarray *residual_norm_scalar_{nullptr};
  Ndarray *status_scalar_{nullptr};
  Ndarray *zero_status_scalar_{nullptr};
  Ndarray *completed_iterations_scalar_{nullptr};
  mutable std::mutex solve_mutex_;
  bool is_success_{false};
  bool has_solved_{false};
  int iterations_{0};
  int status_{static_cast<int>(SparseSolveStatus::kNotRun)};
  double initial_residual_norm_{0.0};
  double residual_norm_{0.0};
  std::uint64_t solve_calls_{0};
  std::uint64_t total_iterations_{0};
  std::uint64_t workspace_builds_{1};
  std::uint64_t workspace_reuses_{0};
  std::uint64_t operator_apply_calls_{0};
  std::uint64_t preconditioner_apply_calls_{0};
  std::uint64_t device_scalar_operations_{0};
  std::uint64_t host_scalar_readbacks_{0};
  std::uint64_t host_synchronizations_{0};
  std::uint64_t device_to_device_bytes_{0};
  std::uint64_t device_to_host_bytes_{0};
  std::uint64_t host_to_device_bytes_{sizeof(int32_t)};
  std::uint64_t last_solve_pattern_version_{0};
  std::uint64_t last_solve_numeric_version_{0};
};

std::unique_ptr<VulkanCGIterationPlan> make_vulkan_cg_iteration_plan(
    Program *program,
    SparseMatrix &matrix,
    int fixed_iterations);

std::unique_ptr<VulkanCGIterationPlan> make_vulkan_cg_convergence_plan(
    Program *program,
    SparseMatrix &matrix,
    int max_iterations,
    float absolute_tolerance);

std::unique_ptr<VulkanCGIterationPlan>
make_vulkan_jacobi_pcg_convergence_plan(
    Program *program,
    SparseMatrix &matrix,
    SparseJacobiPreconditionerPlan &preconditioner,
    int max_iterations,
    float absolute_tolerance);

std::unique_ptr<VulkanCGIterationPlan>
make_vulkan_block_jacobi_pcg_convergence_plan(
    Program *program,
    SparseMatrix &matrix,
    SparseBlockJacobiPreconditionerPlan &preconditioner,
    int max_iterations,
    float absolute_tolerance);

std::unique_ptr<VulkanCGIterationPlan>
make_vulkan_compiled_kernel_cg_convergence_plan(
    Program *program,
    CompiledKernelLinearOperator &matrix,
    int max_iterations,
    float absolute_tolerance);

std::unique_ptr<VulkanCGIterationPlan>
make_vulkan_compiled_kernel_pcg_convergence_plan(
    Program *program,
    CompiledKernelLinearOperator &matrix,
    CompiledKernelPreconditionerPlan &preconditioner,
    int max_iterations,
    float absolute_tolerance);
}  // namespace taichi::lang
