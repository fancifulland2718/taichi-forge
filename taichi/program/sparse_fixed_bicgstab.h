#pragma once

#include "taichi/program/conjugate_gradient.h"

#include <limits>

namespace taichi::lang {

// Identity-preconditioned BiCGSTAB for caller-owned fixed CPU CSR/BSR
// operators. Mutable Eigen matrices keep using their existing provider path.
template <typename EigenT, typename DT>
class FixedSparseBiCGSTAB {
 public:
  FixedSparseBiCGSTAB(Program *program,
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
    TI_ERROR_IF(!program_ || !arch_is_cpu(program_->compile_config().arch),
                "Fixed SparseBiCGSTAB requires an owning CPU Program.");
    const auto operator_stats = matrix_.debug_runtime_statistics();
    csr_matrix_ = dynamic_cast<CpuSparseCsrMatrix *>(&matrix_);
    bsr_matrix_ = dynamic_cast<CpuSparseBsrMatrix *>(&matrix_);
    const bool is_public_fixed =
        operator_stats.backend_family == "cpu" &&
        operator_stats.provider_name == "forge_cpu_native" &&
        operator_stats.pattern_storage_shared &&
        operator_stats.pattern_builds == 0 &&
        ((csr_matrix_ && operator_stats.storage_format == "csr") ||
         (bsr_matrix_ && operator_stats.storage_format == "bsr"));
    TI_ERROR_IF(!is_public_fixed,
                "Fixed SparseBiCGSTAB requires a caller-owned fixed CPU "
                "CSR/BSR pattern.");
    TI_ERROR_IF(
        matrix_.num_rows() <= 0 || matrix_.num_rows() != matrix_.num_cols(),
        "Fixed SparseBiCGSTAB requires a non-empty square matrix.");
    const DataType expected_dtype =
        std::is_same_v<DT, float64> ? DataType(PrimitiveType::f64)
                                    : DataType(PrimitiveType::f32);
    TI_ERROR_IF(matrix_.get_data_type() != expected_dtype,
                "Fixed SparseBiCGSTAB matrix dtype does not match the "
                "selected f32/f64 recurrence.");
    TI_ERROR_IF(max_iterations_ < 0,
                "Fixed SparseBiCGSTAB requires non-negative max "
                "iterations.");
    TI_ERROR_IF(!std::isfinite(absolute_tolerance_) ||
                    !std::isfinite(relative_tolerance_) ||
                    absolute_tolerance_ < static_cast<DT>(0) ||
                    relative_tolerance_ < static_cast<DT>(0) ||
                    (absolute_tolerance_ == static_cast<DT>(0) &&
                     relative_tolerance_ == static_cast<DT>(0)),
                "Fixed SparseBiCGSTAB requires finite non-negative atol and "
                "rtol with at least one positive tolerance.");

    const int size = matrix_.num_rows();
    x_ = EigenT::Zero(size);
    b_ = EigenT::Zero(size);
    residual_ = EigenT::Zero(size);
    shadow_residual_ = EigenT::Zero(size);
    direction_ = EigenT::Zero(size);
    operator_direction_ = EigenT::Zero(size);
    intermediate_residual_ = EigenT::Zero(size);
    operator_intermediate_ = EigenT::Zero(size);
  }

  void set_x(EigenT &x) {
    TI_ERROR_IF(x.size() != matrix_.num_cols(),
                "SparseBiCGSTAB initial guess must have {} entries, got {}.",
                matrix_.num_cols(), x.size());
    x_ = x;
  }

  void reset_x() {
    x_.setZero();
  }

  void set_b(EigenT &b) {
    TI_ERROR_IF(b.size() != matrix_.num_rows(),
                "SparseBiCGSTAB RHS must have {} entries, got {}.",
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

    const double rhs_norm = vector_norm(b_);
    relative_reference_norm_ = rhs_norm;
    effective_tolerance_ = std::max(
        static_cast<double>(absolute_tolerance_),
        static_cast<double>(relative_tolerance_) * rhs_norm);
    initial_residual_norm_ = true_residual_norm();
    residual_norm_ = initial_residual_norm_;
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
      status_ = SparseSolveStatus::kConverged;
      finish_solve();
      return;
    }

    shadow_residual_ = residual_;
    direction_.setZero();
    operator_direction_.setZero();
    double rho_old = 1.0;
    double alpha = 1.0;
    double omega = 1.0;
    bool fresh_direction = true;
    bool terminated = false;

    for (int iteration = 0; iteration < max_iterations_; ++iteration) {
      double rho = dot(shadow_residual_, residual_);
      if (!std::isfinite(rho)) {
        status_ = SparseSolveStatus::kBreakdown;
        terminated = true;
        break;
      }
      if (rho == 0.0) {
        residual_norm_ = true_residual_norm();
        if (!x_.allFinite() || !std::isfinite(residual_norm_)) {
          status_ = SparseSolveStatus::kBreakdown;
          terminated = true;
          break;
        }
        if (residual_norm_ <= effective_tolerance_) {
          status_ = SparseSolveStatus::kConverged;
          terminated = true;
          break;
        }
        restart_from_true_residual(rho_old, alpha, omega,
                                   fresh_direction);
        rho = dot(shadow_residual_, residual_);
        if (!std::isfinite(rho) || rho <= 0.0) {
          status_ = SparseSolveStatus::kBreakdown;
          terminated = true;
          break;
        }
      }

      if (fresh_direction) {
        direction_ = residual_;
        fresh_direction = false;
      } else {
        if (rho_old == 0.0 || omega == 0.0) {
          status_ = SparseSolveStatus::kBreakdown;
          terminated = true;
          break;
        }
        const double beta = (rho / rho_old) * (alpha / omega);
        if (!std::isfinite(beta)) {
          status_ = SparseSolveStatus::kBreakdown;
          terminated = true;
          break;
        }
        for (int index = 0; index < matrix_.num_rows(); ++index) {
          direction_[index] = static_cast<DT>(
              static_cast<double>(residual_[index]) +
              beta * (static_cast<double>(direction_[index]) -
                      omega *
                          static_cast<double>(operator_direction_[index])));
        }
        if (!direction_.allFinite()) {
          status_ = SparseSolveStatus::kBreakdown;
          terminated = true;
          break;
        }
      }

      apply_operator(direction_, operator_direction_);
      const double alpha_denominator =
          dot(shadow_residual_, operator_direction_);
      if (!std::isfinite(alpha_denominator) || alpha_denominator == 0.0) {
        status_ = SparseSolveStatus::kBreakdown;
        terminated = true;
        break;
      }
      alpha = rho / alpha_denominator;
      if (!std::isfinite(alpha)) {
        status_ = SparseSolveStatus::kBreakdown;
        terminated = true;
        break;
      }
      for (int index = 0; index < matrix_.num_rows(); ++index) {
        intermediate_residual_[index] = static_cast<DT>(
            static_cast<double>(residual_[index]) -
            alpha * static_cast<double>(operator_direction_[index]));
      }
      const double intermediate_norm = vector_norm(intermediate_residual_);
      if (!std::isfinite(intermediate_norm)) {
        status_ = SparseSolveStatus::kBreakdown;
        terminated = true;
        break;
      }
      if (intermediate_norm <= effective_tolerance_) {
        update_solution(alpha, 0.0);
        iterations_ = iteration + 1;
        if (!x_.allFinite()) {
          status_ = SparseSolveStatus::kBreakdown;
          terminated = true;
          break;
        }
        residual_norm_ = true_residual_norm();
        if (std::isfinite(residual_norm_) &&
            residual_norm_ <= effective_tolerance_) {
          status_ = SparseSolveStatus::kConverged;
          terminated = true;
          break;
        }
        if (!std::isfinite(residual_norm_)) {
          status_ = SparseSolveStatus::kBreakdown;
          terminated = true;
          break;
        }
        restart_from_true_residual(rho_old, alpha, omega,
                                   fresh_direction);
        continue;
      }

      apply_operator(intermediate_residual_, operator_intermediate_);
      const double omega_denominator =
          dot(operator_intermediate_, operator_intermediate_);
      const double omega_numerator =
          dot(operator_intermediate_, intermediate_residual_);
      if (!std::isfinite(omega_denominator) ||
          !std::isfinite(omega_numerator)) {
        status_ = SparseSolveStatus::kBreakdown;
        terminated = true;
        break;
      }
      if (omega_denominator == 0.0) {
        update_solution(alpha, 0.0);
        iterations_ = iteration + 1;
        residual_norm_ = true_residual_norm();
        status_ = x_.allFinite() && std::isfinite(residual_norm_) &&
                          residual_norm_ <= effective_tolerance_
                      ? SparseSolveStatus::kConverged
                      : SparseSolveStatus::kBreakdown;
        terminated = true;
        break;
      }
      omega = omega_numerator / omega_denominator;
      if (!std::isfinite(omega)) {
        status_ = SparseSolveStatus::kBreakdown;
        terminated = true;
        break;
      }
      update_solution(alpha, omega);
      iterations_ = iteration + 1;
      if (!x_.allFinite()) {
        status_ = SparseSolveStatus::kBreakdown;
        terminated = true;
        break;
      }
      for (int index = 0; index < matrix_.num_rows(); ++index) {
        residual_[index] = static_cast<DT>(
            static_cast<double>(intermediate_residual_[index]) -
            omega * static_cast<double>(operator_intermediate_[index]));
      }
      if (!residual_.allFinite()) {
        status_ = SparseSolveStatus::kBreakdown;
        terminated = true;
        break;
      }
      const double recurrence_residual_norm = vector_norm(residual_);
      if (!std::isfinite(recurrence_residual_norm)) {
        status_ = SparseSolveStatus::kBreakdown;
        terminated = true;
        break;
      }
      if (recurrence_residual_norm <= effective_tolerance_) {
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
        restart_from_true_residual(rho_old, alpha, omega,
                                   fresh_direction);
        continue;
      }
      if (omega == 0.0) {
        residual_norm_ = true_residual_norm();
        status_ = x_.allFinite() && std::isfinite(residual_norm_) &&
                          residual_norm_ <= effective_tolerance_
                      ? SparseSolveStatus::kConverged
                      : SparseSolveStatus::kBreakdown;
        terminated = true;
        break;
      }
      rho_old = rho;
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
    } else if (status_ == SparseSolveStatus::kBreakdown) {
      if (x_.allFinite() && b_.allFinite()) {
        residual_norm_ = true_residual_norm();
      } else {
        residual_norm_ = std::numeric_limits<double>::infinity();
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
    result.workspace_builds = 1;
    result.workspace_reuses =
        workspace_reuses_.load(std::memory_order_relaxed);
    result.operator_apply_calls =
        operator_apply_calls_.load(std::memory_order_relaxed);
    result.operator_apply_calls_available = true;
    result.host_scalar_reductions =
        host_scalar_reductions_.load(std::memory_order_relaxed);
    result.preconditioner_method = "identity";
    result.preconditioner_apply_calls = 0;
    result.preconditioner_apply_calls_available = true;
    result.preconditioner_ownership_scope = "none";
    result.persistent_vector_count = 8;
    result.persistent_vector_reserved_bytes =
        static_cast<std::uint64_t>(8) *
        static_cast<std::uint64_t>(matrix_.num_rows()) * sizeof(DT);
    result.solver_state_rebuilt_each_solve = false;
    result.transient_solver_workspace_bytes = 0;
    result.transient_solver_workspace_bytes_available = true;
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

  double dot(const EigenT &left, const EigenT &right) {
    double result = 0.0;
    for (int index = 0; index < matrix_.num_rows(); ++index) {
      result += static_cast<double>(left[index]) *
                static_cast<double>(right[index]);
    }
    host_scalar_reductions_.fetch_add(1, std::memory_order_relaxed);
    return result;
  }

  double vector_norm(const EigenT &vector) {
    const double squared_norm = dot(vector, vector);
    return std::isfinite(squared_norm) && squared_norm >= 0.0
               ? std::sqrt(squared_norm)
               : std::numeric_limits<double>::quiet_NaN();
  }

  void apply_operator(const EigenT &input, EigenT &output) {
    if (csr_matrix_) {
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
    for (int index = 0; index < matrix_.num_rows(); ++index) {
      residual_[index] = b_[index] - residual_[index];
    }
    return vector_norm(residual_);
  }

  void update_solution(double alpha, double omega) {
    for (int index = 0; index < matrix_.num_rows(); ++index) {
      x_[index] = static_cast<DT>(
          static_cast<double>(x_[index]) +
          alpha * static_cast<double>(direction_[index]) +
          omega * static_cast<double>(intermediate_residual_[index]));
    }
  }

  void restart_from_true_residual(double &rho_old,
                                  double &alpha,
                                  double &omega,
                                  bool &fresh_direction) {
    shadow_residual_ = residual_;
    direction_.setZero();
    operator_direction_.setZero();
    rho_old = 1.0;
    alpha = 1.0;
    omega = 1.0;
    fresh_direction = true;
  }

  void finish_solve() {
    total_iterations_.fetch_add(static_cast<std::uint64_t>(iterations_),
                                std::memory_order_relaxed);
    if (verbose_) {
      std::cout << "#iterations:     " << iterations_ << std::endl;
      std::cout << "residual norm:   " << residual_norm_ << std::endl;
    }
  }

  Program *program_{nullptr};
  SparseMatrix &matrix_;
  CpuSparseCsrMatrix *csr_matrix_{nullptr};
  CpuSparseBsrMatrix *bsr_matrix_{nullptr};
  EigenT x_;
  EigenT b_;
  EigenT residual_;
  EigenT shadow_residual_;
  EigenT direction_;
  EigenT operator_direction_;
  EigenT intermediate_residual_;
  EigenT operator_intermediate_;
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
  std::atomic<std::uint64_t> workspace_reuses_{0};
  std::atomic<std::uint64_t> operator_apply_calls_{0};
  std::atomic<std::uint64_t> host_scalar_reductions_{0};
  std::atomic<std::uint64_t> last_solve_pattern_version_{0};
  std::atomic<std::uint64_t> last_solve_numeric_version_{0};
};

}  // namespace taichi::lang
