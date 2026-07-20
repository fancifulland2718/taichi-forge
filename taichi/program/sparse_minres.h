#pragma once

#include "taichi/program/conjugate_gradient.h"

#include <limits>

namespace taichi::lang {

// Hardened identity-preconditioned MINRES for explicit CPU Eigen matrices.
// This intentionally does not instantiate Eigen's unsupported MINRES: that
// implementation normalizes before initial/happy-breakdown checks and cannot
// provide the structured failure contract required by the public API.
template <typename EigenT, typename DT>
class SparseMINRES {
 public:
  using ColMatrix = Eigen::SparseMatrix<DT, Eigen::ColMajor>;
  using RowMatrix = Eigen::SparseMatrix<DT, Eigen::RowMajor>;

  SparseMINRES(SparseMatrix &matrix,
               int max_iterations,
               DT absolute_tolerance,
               bool verbose,
               DT relative_tolerance = static_cast<DT>(0))
      : SparseMINRES(nullptr, matrix, max_iterations, absolute_tolerance,
                     verbose, relative_tolerance) {
  }

  SparseMINRES(Program *program,
               SparseMatrix &matrix,
               int max_iterations,
               DT absolute_tolerance,
               bool verbose,
               DT relative_tolerance = static_cast<DT>(0))
      : program_(program),
        matrix_(matrix),
        max_iterations_(max_iterations),
        absolute_tolerance_(absolute_tolerance),
        relative_tolerance_(relative_tolerance),
        verbose_(verbose) {
    const auto operator_stats = matrix_.debug_runtime_statistics();
    csr_matrix_ = dynamic_cast<CpuSparseCsrMatrix *>(&matrix_);
    bsr_matrix_ = dynamic_cast<CpuSparseBsrMatrix *>(&matrix_);
    const bool is_eigen =
        operator_stats.backend_family == "cpu" &&
        operator_stats.provider_name == "eigen" &&
        (operator_stats.storage_format == "csr" ||
         operator_stats.storage_format == "csc");
    const bool is_public_fixed =
        operator_stats.backend_family == "cpu" &&
        operator_stats.provider_name == "forge_cpu_native" &&
        operator_stats.pattern_storage_shared &&
        operator_stats.pattern_builds == 0 &&
        ((csr_matrix_ && operator_stats.storage_format == "csr") ||
         (bsr_matrix_ && operator_stats.storage_format == "bsr"));
    TI_ERROR_IF(!is_eigen && !is_public_fixed,
                "CPU SparseMINRES requires an Eigen CSR/CSC matrix or a "
                "caller-owned fixed CPU CSR/BSR pattern.");
    TI_ERROR_IF(is_public_fixed &&
                    (!program_ ||
                     !arch_is_cpu(program_->compile_config().arch)),
                "Fixed CPU SparseMINRES requires its owning CPU Program.");
    const DataType expected_dtype =
        std::is_same_v<DT, float64> ? DataType(PrimitiveType::f64)
                                    : DataType(PrimitiveType::f32);
    TI_ERROR_IF(matrix_.get_data_type() != expected_dtype,
                "CPU SparseMINRES matrix dtype does not match the selected "
                "f32/f64 recurrence.");
    TI_ERROR_IF(
        matrix_.num_rows() <= 0 || matrix_.num_rows() != matrix_.num_cols(),
        "CPU SparseMINRES requires a non-empty square matrix.");
    TI_ERROR_IF(max_iterations_ < 0,
                "CPU SparseMINRES requires non-negative max iterations.");
    TI_ERROR_IF(!std::isfinite(absolute_tolerance_) ||
                    !std::isfinite(relative_tolerance_) ||
                    absolute_tolerance_ < static_cast<DT>(0) ||
                    relative_tolerance_ < static_cast<DT>(0) ||
                    (absolute_tolerance_ == static_cast<DT>(0) &&
                     relative_tolerance_ == static_cast<DT>(0)),
                "CPU SparseMINRES requires finite non-negative atol and rtol "
                "with at least one positive tolerance.");

    if (auto *wrapper =
            dynamic_cast<EigenSparseMatrix<ColMatrix> *>(&matrix_)) {
      col_matrix_ = static_cast<const ColMatrix *>(wrapper->get_matrix());
    } else if (auto *wrapper =
                   dynamic_cast<EigenSparseMatrix<RowMatrix> *>(&matrix_)) {
      row_matrix_ = static_cast<const RowMatrix *>(wrapper->get_matrix());
    } else if (!is_public_fixed) {
      TI_ERROR("CPU SparseMINRES matrix dtype/storage does not match the "
               "selected provider.");
    }

    const int size = matrix_.num_rows();
    x_ = EigenT::Zero(size);
    b_ = EigenT::Zero(size);
    residual_ = EigenT::Zero(size);
    v_old_ = EigenT::Zero(size);
    v_ = EigenT::Zero(size);
    v_new_ = EigenT::Zero(size);
    p_older_ = EigenT::Zero(size);
    p_old_ = EigenT::Zero(size);
    p_ = EigenT::Zero(size);
  }

  void set_x(EigenT &x) {
    TI_ERROR_IF(x.size() != matrix_.num_cols(),
                "SparseMINRES initial guess must have {} entries, got {}.",
                matrix_.num_cols(), x.size());
    x_ = x;
  }

  void reset_x() {
    x_.setZero();
  }

  void set_b(EigenT &b) {
    TI_ERROR_IF(b.size() != matrix_.num_rows(),
                "SparseMINRES RHS must have {} entries, got {}.",
                matrix_.num_rows(), b.size());
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
    const auto previous_solve_calls =
        solve_calls_.fetch_add(1, std::memory_order_relaxed);
    if (previous_solve_calls > 0) {
      workspace_reuses_.fetch_add(1, std::memory_order_relaxed);
    }

    if (!x_.allFinite() || !b_.allFinite()) {
      initial_residual_norm_ = std::numeric_limits<double>::infinity();
      residual_norm_ = initial_residual_norm_;
      status_ = SparseSolveStatus::kBreakdown;
      finish_solve();
      return;
    }
    const double rhs_norm = static_cast<double>(b_.stableNorm());
    relative_reference_norm_ = rhs_norm;
    effective_tolerance_ = std::max(
        static_cast<double>(absolute_tolerance_),
        static_cast<double>(relative_tolerance_) * rhs_norm);
    initial_residual_norm_ = true_residual_norm();
    residual_norm_ = initial_residual_norm_;
    estimated_residual_norm_ = initial_residual_norm_;
    if (!std::isfinite(rhs_norm) ||
        !std::isfinite(effective_tolerance_) ||
        !std::isfinite(initial_residual_norm_)) {
      status_ = SparseSolveStatus::kBreakdown;
      finish_solve();
      return;
    }
    if (initial_residual_norm_ <= effective_tolerance_) {
      status_ = SparseSolveStatus::kConverged;
      finish_solve();
      return;
    }
    if (max_iterations_ == 0) {
      status_ = SparseSolveStatus::kMaxIterations;
      finish_solve();
      return;
    }
    if (rhs_norm == 0.0) {
      x_.setZero();
      residual_norm_ = 0.0;
      estimated_residual_norm_ = 0.0;
      status_ = SparseSolveStatus::kConverged;
      finish_solve();
      return;
    }

    DT beta = static_cast<DT>(initial_residual_norm_);
    const DT beta_one = beta;
    v_old_.setZero();
    v_ = residual_ / beta;
    v_new_.setZero();
    p_older_.setZero();
    p_old_.setZero();
    p_.setZero();
    DT c = static_cast<DT>(1);
    DT c_old = static_cast<DT>(1);
    DT s = static_cast<DT>(0);
    DT s_old = static_cast<DT>(0);
    DT eta = static_cast<DT>(1);
    bool terminated = false;

    for (int iteration = 0; iteration < max_iterations_; ++iteration) {
      apply_operator(v_, v_new_);
      v_new_ -= beta * v_old_;
      const DT alpha = v_new_.dot(v_);
      v_new_ -= alpha * v_;
      const DT beta_new = v_new_.stableNorm();
      if (!finite_scalar(alpha) || !finite_scalar(beta_new)) {
        status_ = SparseSolveStatus::kBreakdown;
        terminated = true;
        break;
      }

      const DT r2 = s * alpha + c * c_old * beta;
      const DT r3 = s_old * beta;
      const DT r1_hat = c * alpha - c_old * s * beta;
      const DT r1 = std::hypot(r1_hat, beta_new);
      if (!finite_scalar(r1) || r1 == static_cast<DT>(0)) {
        residual_norm_ = true_residual_norm();
        status_ = x_.allFinite() && std::isfinite(residual_norm_) &&
                          residual_norm_ <= effective_tolerance_
                      ? SparseSolveStatus::kConverged
                      : SparseSolveStatus::kBreakdown;
        terminated = true;
        break;
      }

      c_old = c;
      s_old = s;
      c = r1_hat / r1;
      s = beta_new / r1;
      p_older_ = p_old_;
      p_old_ = p_;
      p_ = (v_ - r2 * p_old_ - r3 * p_older_) / r1;
      x_ += beta_one * c * eta * p_;
      iterations_ = iteration + 1;
      estimated_residual_norm_ *= std::abs(static_cast<double>(s));
      if (!x_.allFinite() || !std::isfinite(estimated_residual_norm_)) {
        status_ = SparseSolveStatus::kBreakdown;
        terminated = true;
        break;
      }

      const bool krylov_space_closed = beta_new == static_cast<DT>(0);
      if (estimated_residual_norm_ <= effective_tolerance_ ||
          krylov_space_closed) {
        residual_norm_ = true_residual_norm();
        if (!std::isfinite(residual_norm_) || !x_.allFinite()) {
          status_ = SparseSolveStatus::kBreakdown;
          terminated = true;
          break;
        }
        if (residual_norm_ <= effective_tolerance_) {
          status_ = SparseSolveStatus::kConverged;
          terminated = true;
          break;
        }
        if (krylov_space_closed) {
          status_ = SparseSolveStatus::kBreakdown;
          terminated = true;
          break;
        }
      }

      eta = -s * eta;
      v_old_ = v_;
      v_ = v_new_ / beta_new;
      beta = beta_new;
    }

    if (!terminated) {
      residual_norm_ = true_residual_norm();
      if (!x_.allFinite() || !std::isfinite(residual_norm_)) {
        status_ = SparseSolveStatus::kBreakdown;
      } else if (residual_norm_ <= effective_tolerance_) {
        status_ = SparseSolveStatus::kConverged;
      } else {
        status_ = SparseSolveStatus::kMaxIterations;
      }
    }
    finish_solve();
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
    result.method = "minres";
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
    result.workspace_builds = 1;
    result.workspace_reuses =
        workspace_reuses_.load(std::memory_order_relaxed);
    result.operator_apply_calls =
        operator_apply_calls_.load(std::memory_order_relaxed);
    result.operator_apply_calls_available = true;
    result.preconditioner_method = "identity";
    result.preconditioner_apply_calls = 0;
    result.preconditioner_apply_calls_available = true;
    result.preconditioner_ownership_scope = "none";
    result.persistent_vector_count = 9;
    result.persistent_vector_reserved_bytes =
        static_cast<std::uint64_t>(9) *
        static_cast<std::uint64_t>(matrix_.num_rows()) * sizeof(DT);
    result.solver_state_rebuilt_each_solve = false;
    result.transient_solver_workspace_bytes = 0;
    result.transient_solver_workspace_bytes_available = true;
    return result;
  }

 private:
  static bool finite_scalar(DT value) {
    return std::isfinite(static_cast<double>(value));
  }

  void reset_last_result(const SparseMatrixRuntimeStatistics &stats) {
    status_ = SparseSolveStatus::kNotRun;
    iterations_ = 0;
    initial_residual_norm_ = 0.0;
    residual_norm_ = 0.0;
    estimated_residual_norm_ = 0.0;
    relative_reference_norm_ = 0.0;
    effective_tolerance_ = static_cast<double>(absolute_tolerance_);
    last_solve_pattern_version_.store(stats.pattern_version,
                                      std::memory_order_relaxed);
    last_solve_numeric_version_.store(stats.numeric_version,
                                      std::memory_order_relaxed);
  }

  void apply_operator(const EigenT &input, EigenT &output) {
    if (col_matrix_) {
      output.noalias() = (*col_matrix_) * input;
    } else if (row_matrix_) {
      output.noalias() = (*row_matrix_) * input;
    } else if (csr_matrix_) {
      csr_matrix_->spmv_cpu_raw(
          program_, reinterpret_cast<std::uintptr_t>(input.data()),
          reinterpret_cast<std::uintptr_t>(output.data()));
    } else {
      bsr_matrix_->spmv_cpu_raw(
          program_, reinterpret_cast<std::uintptr_t>(input.data()),
          reinterpret_cast<std::uintptr_t>(output.data()));
    }
    operator_apply_calls_.fetch_add(1, std::memory_order_relaxed);
  }

  double true_residual_norm() {
    apply_operator(x_, residual_);
    residual_ = b_ - residual_;
    return static_cast<double>(residual_.stableNorm());
  }

  void finish_solve() {
    total_iterations_.fetch_add(static_cast<std::uint64_t>(iterations_),
                                std::memory_order_relaxed);
    if (verbose_) {
      std::cout << "#iterations:       " << iterations_ << std::endl;
      std::cout << "estimated residual: " << estimated_residual_norm_
                << std::endl;
      std::cout << "residual norm:     " << residual_norm_ << std::endl;
    }
  }

  Program *program_{nullptr};
  SparseMatrix &matrix_;
  const ColMatrix *col_matrix_{nullptr};
  const RowMatrix *row_matrix_{nullptr};
  CpuSparseCsrMatrix *csr_matrix_{nullptr};
  CpuSparseBsrMatrix *bsr_matrix_{nullptr};
  EigenT x_;
  EigenT b_;
  EigenT residual_;
  EigenT v_old_;
  EigenT v_;
  EigenT v_new_;
  EigenT p_older_;
  EigenT p_old_;
  EigenT p_;
  int max_iterations_{0};
  DT absolute_tolerance_{0};
  DT relative_tolerance_{0};
  bool verbose_{false};
  SparseSolveStatus status_{SparseSolveStatus::kNotRun};
  int iterations_{0};
  double initial_residual_norm_{0.0};
  double residual_norm_{0.0};
  double estimated_residual_norm_{0.0};
  double relative_reference_norm_{0.0};
  double effective_tolerance_{0.0};
  std::atomic<std::uint64_t> solve_calls_{0};
  std::atomic<std::uint64_t> total_iterations_{0};
  std::atomic<std::uint64_t> workspace_reuses_{0};
  std::atomic<std::uint64_t> operator_apply_calls_{0};
  std::atomic<std::uint64_t> last_solve_pattern_version_{0};
  std::atomic<std::uint64_t> last_solve_numeric_version_{0};
};

}  // namespace taichi::lang
