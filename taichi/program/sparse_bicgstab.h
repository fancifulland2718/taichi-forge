#pragma once

#include "taichi/program/conjugate_gradient.h"

#include <memory>

namespace taichi::lang {

// CPU provider wrapper for nonsymmetric explicit sparse systems. Keep this
// separate from CG because the matrix suitability and breakdown contracts are
// different, even though both expose the same result/telemetry schema.
template <typename EigenT, typename DT>
class SparseBiCGSTAB {
 public:
  using ColMatrix = Eigen::SparseMatrix<DT, Eigen::ColMajor>;
  using RowMatrix = Eigen::SparseMatrix<DT, Eigen::RowMajor>;
  using ColSolver = Eigen::BiCGSTAB<ColMatrix>;
  using RowSolver = Eigen::BiCGSTAB<RowMatrix>;

  SparseBiCGSTAB(SparseMatrix &matrix,
                 int max_iterations,
                 DT absolute_tolerance,
                 bool verbose,
                 DT relative_tolerance = static_cast<DT>(0))
      : matrix_(matrix),
        max_iterations_(max_iterations),
        absolute_tolerance_(absolute_tolerance),
        relative_tolerance_(relative_tolerance),
        verbose_(verbose) {
    const auto operator_stats = matrix_.debug_runtime_statistics();
    TI_ERROR_IF(
        operator_stats.backend_family != "cpu" ||
            operator_stats.provider_name != "eigen" ||
            (operator_stats.storage_format != "csr" &&
             operator_stats.storage_format != "csc"),
        "CPU SparseBiCGSTAB currently requires an Eigen CSR/CSC matrix.");
    TI_ERROR_IF(
        matrix_.num_rows() <= 0 || matrix_.num_rows() != matrix_.num_cols(),
        "CPU SparseBiCGSTAB requires a non-empty square matrix.");
    TI_ERROR_IF(max_iterations_ < 0,
                "CPU SparseBiCGSTAB requires non-negative max iterations.");
    TI_ERROR_IF(!std::isfinite(absolute_tolerance_) ||
                    !std::isfinite(relative_tolerance_) ||
                    absolute_tolerance_ < static_cast<DT>(0) ||
                    relative_tolerance_ < static_cast<DT>(0) ||
                    (absolute_tolerance_ == static_cast<DT>(0) &&
                     relative_tolerance_ == static_cast<DT>(0)),
                "CPU SparseBiCGSTAB requires finite non-negative atol and "
                "rtol with at least one positive tolerance.");

    if (auto *wrapper =
            dynamic_cast<EigenSparseMatrix<ColMatrix> *>(&matrix_)) {
      col_matrix_ = static_cast<const ColMatrix *>(wrapper->get_matrix());
      col_solver_ = std::make_unique<ColSolver>();
      col_solver_->setMaxIterations(max_iterations_);
    } else if (auto *wrapper =
                   dynamic_cast<EigenSparseMatrix<RowMatrix> *>(&matrix_)) {
      row_matrix_ = static_cast<const RowMatrix *>(wrapper->get_matrix());
      row_solver_ = std::make_unique<RowSolver>();
      row_solver_->setMaxIterations(max_iterations_);
    } else {
      TI_ERROR("CPU SparseBiCGSTAB matrix dtype/storage does not match the "
               "selected provider.");
    }
    x_ = EigenT::Zero(matrix_.num_cols());
    b_ = EigenT::Zero(matrix_.num_rows());
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

  void set_x_ndarray(Program *program, Ndarray &x) {
    const auto address = program->get_ndarray_data_ptr_as_int(&x);
    x_ = Eigen::Map<EigenT>(reinterpret_cast<DT *>(address),
                            matrix_.num_cols());
  }

  void set_b_ndarray(Program *program, Ndarray &b) {
    const auto address = program->get_ndarray_data_ptr_as_int(&b);
    b_ = Eigen::Map<EigenT>(reinterpret_cast<DT *>(address),
                            matrix_.num_rows());
  }

  void solve() {
    const auto operator_stats = matrix_.debug_runtime_statistics();
    reset_last_result(operator_stats);
    solve_calls_.fetch_add(1, std::memory_order_relaxed);

    initial_residual_norm_ = true_residual_norm();
    residual_norm_ = initial_residual_norm_;
    const double rhs_norm = static_cast<double>(b_.norm());
    relative_reference_norm_ = rhs_norm;
    effective_tolerance_ = std::max(
        static_cast<double>(absolute_tolerance_),
        static_cast<double>(relative_tolerance_) * rhs_norm);
    if (!x_.allFinite() || !b_.allFinite() ||
        !std::isfinite(initial_residual_norm_) ||
        !std::isfinite(rhs_norm) || !std::isfinite(effective_tolerance_)) {
      status_ = SparseSolveStatus::kBreakdown;
      return;
    }
    if (initial_residual_norm_ <= effective_tolerance_) {
      status_ = SparseSolveStatus::kConverged;
      return;
    }
    if (max_iterations_ == 0) {
      status_ = SparseSolveStatus::kMaxIterations;
      return;
    }
    // Eigen's BiCGSTAB zero-RHS branch sets x to zero but does not update its
    // iteration counter. Close the mathematically exact case here and keep the
    // public iteration/result contract deterministic.
    if (rhs_norm == 0.0) {
      x_.setZero();
      residual_norm_ = 0.0;
      status_ = SparseSolveStatus::kConverged;
      return;
    }

    const DT provider_tolerance =
        static_cast<DT>(effective_tolerance_ / rhs_norm);
    const bool solver_state_current =
        solver_state_initialized_ &&
        solver_state_pattern_version_ == operator_stats.pattern_version &&
        solver_state_numeric_version_ == operator_stats.numeric_version;
    if (solver_state_current) {
      workspace_reuses_.fetch_add(1, std::memory_order_relaxed);
    } else {
      compute_provider();
      workspace_builds_.fetch_add(1, std::memory_order_relaxed);
      solver_state_initialized_ = provider_info() == Eigen::Success;
      if (!solver_state_initialized_) {
        status_ = SparseSolveStatus::kBreakdown;
        return;
      }
      solver_state_pattern_version_ = operator_stats.pattern_version;
      solver_state_numeric_version_ = operator_stats.numeric_version;
    }

    set_provider_tolerance(provider_tolerance);
    solve_with_guess();
    iterations_ = provider_iterations();
    total_iterations_.fetch_add(static_cast<std::uint64_t>(iterations_),
                                std::memory_order_relaxed);
    residual_norm_ = true_residual_norm();
    if (verbose_) {
      std::cout << "#iterations:     " << iterations_ << std::endl;
      std::cout << "estimated error: " << provider_error() << std::endl;
      std::cout << "residual norm:   " << residual_norm_ << std::endl;
    }
    const auto info = provider_info();
    if (!x_.allFinite() || !std::isfinite(residual_norm_) ||
        info == Eigen::NumericalIssue || info == Eigen::InvalidInput) {
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
    const auto operator_stats = matrix_.debug_runtime_statistics();
    SparseSolvePlanRuntimeStatistics result;
    result.backend_family = "cpu";
    result.method = "bicgstab";
    result.dtype = data_type_name(matrix_.get_data_type());
    result.rows = matrix_.num_rows();
    result.cols = matrix_.num_cols();
    result.max_iterations = max_iterations_;
    result.absolute_tolerance = static_cast<double>(absolute_tolerance_);
    result.relative_tolerance = static_cast<double>(relative_tolerance_);
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
    result.operator_apply_calls_available = false;
    result.preconditioner_method = "jacobi";
    result.preconditioner_apply_calls_available = false;
    result.preconditioner_ownership_scope = "provider_state";
    result.persistent_vector_count = 2;
    result.persistent_vector_reserved_bytes =
        (static_cast<std::uint64_t>(matrix_.num_cols()) +
         static_cast<std::uint64_t>(matrix_.num_rows())) *
        sizeof(DT);
    result.solver_state_rebuilt_each_solve = false;
    return result;
  }

 private:
  void reset_last_result(const SparseMatrixRuntimeStatistics &stats) {
    status_ = SparseSolveStatus::kNotRun;
    iterations_ = 0;
    initial_residual_norm_ = 0.0;
    residual_norm_ = 0.0;
    relative_reference_norm_ = 0.0;
    effective_tolerance_ = static_cast<double>(absolute_tolerance_);
    last_solve_pattern_version_.store(stats.pattern_version,
                                      std::memory_order_relaxed);
    last_solve_numeric_version_.store(stats.numeric_version,
                                      std::memory_order_relaxed);
  }

  double true_residual_norm() const {
    if (col_matrix_) {
      return ((*col_matrix_) * x_ - b_).norm();
    }
    return ((*row_matrix_) * x_ - b_).norm();
  }

  void compute_provider() {
    if (col_solver_) {
      col_solver_->compute(*col_matrix_);
    } else {
      row_solver_->compute(*row_matrix_);
    }
  }

  void set_provider_tolerance(DT tolerance) {
    if (col_solver_) {
      col_solver_->setTolerance(tolerance);
    } else {
      row_solver_->setTolerance(tolerance);
    }
  }

  void solve_with_guess() {
    if (col_solver_) {
      x_ = col_solver_->solveWithGuess(b_, x_);
    } else {
      x_ = row_solver_->solveWithGuess(b_, x_);
    }
  }

  Eigen::ComputationInfo provider_info() const {
    return col_solver_ ? col_solver_->info() : row_solver_->info();
  }

  int provider_iterations() const {
    return static_cast<int>(col_solver_ ? col_solver_->iterations()
                                        : row_solver_->iterations());
  }

  double provider_error() const {
    return static_cast<double>(col_solver_ ? col_solver_->error()
                                           : row_solver_->error());
  }

  SparseMatrix &matrix_;
  const ColMatrix *col_matrix_{nullptr};
  const RowMatrix *row_matrix_{nullptr};
  std::unique_ptr<ColSolver> col_solver_;
  std::unique_ptr<RowSolver> row_solver_;
  EigenT x_;
  EigenT b_;
  int max_iterations_{0};
  DT absolute_tolerance_{0};
  DT relative_tolerance_{0};
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

}  // namespace taichi::lang
