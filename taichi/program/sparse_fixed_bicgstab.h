#pragma once

#include "taichi/program/conjugate_gradient.h"
#include "taichi/program/linear_operator.h"

#include <limits>

namespace taichi::lang {

// BiCGSTAB over a provider-neutral OperatorPlan, with identity or one
// fixed-linear right preconditioner action.
// Public fixed CSR/BSR matrices enter through a construction-time
// compatibility binding; the recurrence only consumes one pinned action.
template <typename EigenT, typename DT>
class FixedSparseBiCGSTAB {
 public:
  FixedSparseBiCGSTAB(Program *program,
                      SparseMatrix &matrix,
                      int max_iterations,
                      DT absolute_tolerance,
                      bool verbose,
                      DT relative_tolerance = static_cast<DT>(0))
      : FixedSparseBiCGSTAB(
            program,
            make_cpu_fixed_sparse_operator_binding(program, matrix),
            max_iterations,
            absolute_tolerance,
            verbose,
            relative_tolerance) {
  }

  FixedSparseBiCGSTAB(Program *program,
                      OperatorBinding operator_binding,
                      int max_iterations,
                      DT absolute_tolerance,
                      bool verbose,
                      DT relative_tolerance = static_cast<DT>(0))
      : program_(program),
        max_iterations_(max_iterations),
        absolute_tolerance_(absolute_tolerance),
        relative_tolerance_(relative_tolerance),
        verbose_(verbose) {
    TI_ERROR_IF(program_ && !arch_is_cpu(program_->compile_config().arch),
                "Operator BiCGSTAB supports host-reference or CPU Program "
                "bindings only.");
    operator_plan_ = std::make_unique<OperatorPlan>(
        program_, std::move(operator_binding));
    const auto &descriptor = operator_plan_->descriptor();
    validate_operator_solver_compatibility(
        descriptor, operator_plan_->mathematical_traits(),
        OperatorSolverFamily::bicgstab);
    TI_ERROR_IF(descriptor.domain.scalar_extent >
                    static_cast<std::size_t>(
                        std::numeric_limits<int>::max()),
                "Operator BiCGSTAB extent exceeds the supported int range.");
    rows_ = static_cast<int>(descriptor.range.scalar_extent);
    cols_ = static_cast<int>(descriptor.domain.scalar_extent);
    dtype_ = descriptor.range.scalar_type;
    const DataType expected_dtype =
        std::is_same_v<DT, float64> ? DataType(PrimitiveType::f64)
                                    : DataType(PrimitiveType::f32);
    TI_ERROR_IF(dtype_ != expected_dtype,
                "Operator BiCGSTAB dtype does not match the "
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

    const int size = rows_;
    x_ = EigenT::Zero(size);
    b_ = EigenT::Zero(size);
    residual_ = EigenT::Zero(size);
    shadow_residual_ = EigenT::Zero(size);
    direction_ = EigenT::Zero(size);
    operator_direction_ = EigenT::Zero(size);
    intermediate_residual_ = EigenT::Zero(size);
    operator_intermediate_ = EigenT::Zero(size);
  }

  FixedSparseBiCGSTAB(Program *program,
                      OperatorBinding operator_binding,
                      ExperimentalLinearOperatorHandle &preconditioner,
                      int max_iterations,
                      DT absolute_tolerance,
                      bool verbose,
                      DT relative_tolerance = static_cast<DT>(0))
      : FixedSparseBiCGSTAB(program, std::move(operator_binding),
                            max_iterations, absolute_tolerance, verbose,
                            relative_tolerance) {
    preconditioner_plan_ = make_solver_right_preconditioner_plan(
        program_, *operator_plan_, preconditioner, "linear_operator");
    const int size = rows_;
    preconditioned_direction_ = EigenT::Zero(size);
    preconditioned_intermediate_ = EigenT::Zero(size);
  }

  void set_x(EigenT &x) {
    TI_ERROR_IF(x.size() != cols_,
                "SparseBiCGSTAB initial guess must have {} entries, got {}.",
                cols_, x.size());
    x_ = x;
  }

  void reset_x() {
    x_.setZero();
  }

  void set_b(EigenT &b) {
    TI_ERROR_IF(b.size() != rows_,
                "SparseBiCGSTAB RHS must have {} entries, got {}.",
                rows_, b.size());
    b_ = b;
  }

  void set_x_ndarray(Program *program, Ndarray &x) {
    const auto address = program->get_ndarray_data_ptr_as_int(&x);
    x_ = Eigen::Map<EigenT>(reinterpret_cast<DT *>(address), cols_);
  }

  void set_b_ndarray(Program *program, Ndarray &b) {
    const auto address = program->get_ndarray_data_ptr_as_int(&b);
    b_ = Eigen::Map<EigenT>(reinterpret_cast<DT *>(address), rows_);
  }

  void solve_ndarray(Program *program, Ndarray &x, Ndarray &b) {
    TI_ERROR_IF(program != program_,
                "Operator BiCGSTAB solve must use its construction Program.");
    set_x_ndarray(program, x);
    set_b_ndarray(program, b);
    solve();
    const auto address = program->get_ndarray_data_ptr_as_int(&x);
    Eigen::Map<EigenT>(reinterpret_cast<DT *>(address), cols_) = x_;
  }

  void solve() {
    auto operator_generation = operator_plan_->pin();
    OperatorPinnedAction preconditioner_generation;
    if (preconditioner_plan_) {
      preconditioner_generation =
          preconditioner_plan_->update_and_pin(operator_generation);
    }
    reset_last_result(operator_generation.resource_stamp());
    auto apply_operator = [&](const EigenT &input, EigenT &output) {
      this->apply_operator(operator_generation, input, output);
    };
    auto apply_preconditioner = [&](const EigenT &input, EigenT &output) {
      this->apply_preconditioner(preconditioner_generation, input, output);
    };
    auto true_residual_norm = [&] {
      return this->true_residual_norm(operator_generation);
    };
    const auto previous_solve_calls =
        solve_calls_.fetch_add(1, std::memory_order_relaxed);
    if (previous_solve_calls > 0) {
      workspace_reuses_.fetch_add(1, std::memory_order_relaxed);
    }
    if (!x_.allFinite() || !b_.allFinite()) {
      initial_residual_norm_ = std::numeric_limits<double>::infinity();
      residual_norm_ = initial_residual_norm_;
      set_breakdown(SparseSolveBreakdownReason::nonfinite);
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
      set_breakdown(SparseSolveBreakdownReason::nonfinite);
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
        set_breakdown(SparseSolveBreakdownReason::nonfinite);
        terminated = true;
        break;
      }
      if (rho == 0.0) {
        residual_norm_ = true_residual_norm();
        if (!x_.allFinite() || !std::isfinite(residual_norm_)) {
          set_breakdown(SparseSolveBreakdownReason::nonfinite);
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
        if (!std::isfinite(rho)) {
          set_breakdown(SparseSolveBreakdownReason::nonfinite);
          terminated = true;
          break;
        }
        if (rho == 0.0) {
          set_breakdown(SparseSolveBreakdownReason::rho);
          terminated = true;
          break;
        }
      }

      if (fresh_direction) {
        direction_ = residual_;
        fresh_direction = false;
      } else {
        if (rho_old == 0.0) {
          set_breakdown(SparseSolveBreakdownReason::rho);
          terminated = true;
          break;
        }
        if (omega == 0.0) {
          set_breakdown(SparseSolveBreakdownReason::omega);
          terminated = true;
          break;
        }
        const double beta = (rho / rho_old) * (alpha / omega);
        if (!std::isfinite(beta)) {
          set_breakdown(SparseSolveBreakdownReason::nonfinite);
          terminated = true;
          break;
        }
        for (int index = 0; index < rows_; ++index) {
          direction_[index] = static_cast<DT>(
              static_cast<double>(residual_[index]) +
              beta * (static_cast<double>(direction_[index]) -
                      omega *
                          static_cast<double>(operator_direction_[index])));
        }
        if (!direction_.allFinite()) {
          set_breakdown(SparseSolveBreakdownReason::nonfinite);
          terminated = true;
          break;
        }
      }

      const EigenT *operator_direction_input = &direction_;
      if (preconditioner_plan_) {
        apply_preconditioner(direction_, preconditioned_direction_);
        operator_direction_input = &preconditioned_direction_;
      }
      apply_operator(*operator_direction_input, operator_direction_);
      const double alpha_denominator =
          dot(shadow_residual_, operator_direction_);
      if (!std::isfinite(alpha_denominator)) {
        set_breakdown(SparseSolveBreakdownReason::nonfinite);
        terminated = true;
        break;
      }
      if (alpha_denominator == 0.0) {
        set_breakdown(SparseSolveBreakdownReason::alpha_denominator);
        terminated = true;
        break;
      }
      alpha = rho / alpha_denominator;
      if (!std::isfinite(alpha)) {
        set_breakdown(SparseSolveBreakdownReason::nonfinite);
        terminated = true;
        break;
      }
      for (int index = 0; index < rows_; ++index) {
        intermediate_residual_[index] = static_cast<DT>(
            static_cast<double>(residual_[index]) -
            alpha * static_cast<double>(operator_direction_[index]));
      }
      const double intermediate_norm = vector_norm(intermediate_residual_);
      if (!std::isfinite(intermediate_norm)) {
        set_breakdown(SparseSolveBreakdownReason::nonfinite);
        terminated = true;
        break;
      }
      if (intermediate_norm <= effective_tolerance_) {
        update_solution(alpha, 0.0);
        iterations_ = iteration + 1;
        if (!x_.allFinite()) {
          set_breakdown(SparseSolveBreakdownReason::nonfinite);
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
          set_breakdown(SparseSolveBreakdownReason::nonfinite);
          terminated = true;
          break;
        }
        restart_from_true_residual(rho_old, alpha, omega,
                                   fresh_direction);
        continue;
      }

      const EigenT *operator_intermediate_input = &intermediate_residual_;
      if (preconditioner_plan_) {
        apply_preconditioner(intermediate_residual_,
                             preconditioned_intermediate_);
        operator_intermediate_input = &preconditioned_intermediate_;
      }
      apply_operator(*operator_intermediate_input, operator_intermediate_);
      const double omega_denominator =
          dot(operator_intermediate_, operator_intermediate_);
      const double omega_numerator =
          dot(operator_intermediate_, intermediate_residual_);
      if (!std::isfinite(omega_denominator) ||
          !std::isfinite(omega_numerator)) {
        set_breakdown(SparseSolveBreakdownReason::nonfinite);
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
        if (status_ == SparseSolveStatus::kBreakdown) {
          breakdown_reason_ =
              x_.allFinite() && std::isfinite(residual_norm_)
                  ? SparseSolveBreakdownReason::omega_denominator
                  : SparseSolveBreakdownReason::nonfinite;
        }
        terminated = true;
        break;
      }
      omega = omega_numerator / omega_denominator;
      if (!std::isfinite(omega)) {
        set_breakdown(SparseSolveBreakdownReason::nonfinite);
        terminated = true;
        break;
      }
      update_solution(alpha, omega);
      iterations_ = iteration + 1;
      if (!x_.allFinite()) {
        set_breakdown(SparseSolveBreakdownReason::nonfinite);
        terminated = true;
        break;
      }
      for (int index = 0; index < rows_; ++index) {
        residual_[index] = static_cast<DT>(
            static_cast<double>(intermediate_residual_[index]) -
            omega * static_cast<double>(operator_intermediate_[index]));
      }
      if (!residual_.allFinite()) {
        set_breakdown(SparseSolveBreakdownReason::nonfinite);
        terminated = true;
        break;
      }
      const double recurrence_residual_norm = vector_norm(residual_);
      if (!std::isfinite(recurrence_residual_norm)) {
        set_breakdown(SparseSolveBreakdownReason::nonfinite);
        terminated = true;
        break;
      }
      if (recurrence_residual_norm <= effective_tolerance_) {
        residual_norm_ = true_residual_norm();
        if (!std::isfinite(residual_norm_) || !x_.allFinite()) {
          set_breakdown(SparseSolveBreakdownReason::nonfinite);
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
        if (status_ == SparseSolveStatus::kBreakdown) {
          breakdown_reason_ =
              x_.allFinite() && std::isfinite(residual_norm_)
                  ? SparseSolveBreakdownReason::omega
                  : SparseSolveBreakdownReason::nonfinite;
        }
        terminated = true;
        break;
      }
      rho_old = rho;
    }

    if (!terminated) {
      residual_norm_ = true_residual_norm();
      if (!x_.allFinite() || !std::isfinite(residual_norm_)) {
        set_breakdown(SparseSolveBreakdownReason::nonfinite);
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
            relative_reference_norm_, effective_tolerance_,
            breakdown_reason_};
  }

  SparseSolvePlanRuntimeStatistics debug_runtime_statistics() const {
    const auto operator_stamp = operator_plan_->resource_stamp();
    const auto plan_statistics = operator_plan_->debug_runtime_statistics();
    SparseSolvePlanRuntimeStatistics result;
    result.backend_family = "cpu";
    result.method = "bicgstab";
    result.dtype = data_type_name(dtype_);
    result.rows = rows_;
    result.cols = cols_;
    result.max_iterations = max_iterations_;
    result.absolute_tolerance = static_cast<double>(absolute_tolerance_);
    result.relative_tolerance = static_cast<double>(relative_tolerance_);
    result.last_relative_reference_norm = relative_reference_norm_;
    result.last_effective_tolerance = effective_tolerance_;
    result.last_breakdown_reason =
        sparse_solve_breakdown_reason_name(breakdown_reason_);
    result.operator_pattern_version = operator_stamp.topology_revision;
    result.operator_numeric_version = operator_stamp.numeric_revision;
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
    result.operator_action_provider = operator_plan_->provider_name();
    result.operator_asynchronous_submit =
        operator_plan_->capabilities().asynchronous_submit;
    result.operator_generation_pins = plan_statistics.generation_pins;
    result.operator_generation_changes = plan_statistics.generation_changes;
    result.operator_numeric_generation_changes =
        plan_statistics.numeric_generation_changes;
    result.operator_binding_generation_changes =
        plan_statistics.binding_generation_changes;
    result.operator_plan_invalidations = plan_statistics.invalidations;
    result.host_scalar_reductions =
        host_scalar_reductions_.load(std::memory_order_relaxed);
    result.preconditioner_method =
        preconditioner_plan_ ? preconditioner_plan_->method() : "identity";
    result.preconditioner_apply_calls =
        preconditioner_apply_calls_.load(std::memory_order_relaxed);
    result.preconditioner_apply_calls_available = true;
    result.preconditioning_side = preconditioner_plan_ ? "right" : "none";
    result.preconditioner_ownership_scope =
        preconditioner_plan_ ? "solver_plan" : "none";
    const std::uint64_t persistent_vectors =
        preconditioner_plan_ ? 10 : 8;
    result.persistent_vector_count = persistent_vectors;
    result.persistent_vector_reserved_bytes =
        persistent_vectors *
        static_cast<std::uint64_t>(rows_) * sizeof(DT);
    if (preconditioner_plan_) {
      append_solver_preconditioner_plan_statistics(*preconditioner_plan_,
                                                    result);
    }
    result.solver_state_rebuilt_each_solve = false;
    result.transient_solver_workspace_bytes = 0;
    result.transient_solver_workspace_bytes_available = true;
    return result;
  }

 private:
  void reset_last_result(const OperatorResourceStamp &stamp) {
    status_ = SparseSolveStatus::kNotRun;
    breakdown_reason_ = SparseSolveBreakdownReason::none;
    iterations_ = 0;
    initial_residual_norm_ = 0.0;
    residual_norm_ = 0.0;
    relative_reference_norm_ = 0.0;
    effective_tolerance_ = static_cast<double>(absolute_tolerance_);
    last_solve_pattern_version_.store(stamp.topology_revision,
                                      std::memory_order_relaxed);
    last_solve_numeric_version_.store(stamp.numeric_revision,
                                      std::memory_order_relaxed);
  }

  double dot(const EigenT &left, const EigenT &right) {
    double result = 0.0;
    for (int index = 0; index < rows_; ++index) {
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

  void apply_operator(const OperatorPinnedAction &generation,
                      const EigenT &input,
                      EigenT &output) {
    const auto &descriptor = operator_plan_->descriptor();
    const auto input_address =
        reinterpret_cast<std::uintptr_t>(input.data());
    const auto output_address =
        reinterpret_cast<std::uintptr_t>(output.data());
    const auto input_view =
        program_ ? OperatorVectorView::from_device_pointer(
                       program_, input_address, descriptor.domain, false)
                 : OperatorVectorView::from_const_host(input.data(),
                                                       descriptor.domain);
    const auto output_view =
        program_ ? OperatorVectorView::from_device_pointer(
                       program_, output_address, descriptor.range, true)
                 : OperatorVectorView::from_mutable_host(output.data(),
                                                         descriptor.range);
    operator_plan_->submit(
        generation,
        {OperatorApplyMode::forward, input_view, nullptr, output_view});
    operator_apply_calls_.fetch_add(1, std::memory_order_relaxed);
  }

  void apply_preconditioner(const OperatorPinnedAction &generation,
                            const EigenT &input,
                            EigenT &output) {
    TI_ASSERT(preconditioner_plan_ && generation);
    auto &plan = preconditioner_plan_->action();
    const auto &descriptor = plan.descriptor();
    const auto input_address =
        reinterpret_cast<std::uintptr_t>(input.data());
    const auto output_address =
        reinterpret_cast<std::uintptr_t>(output.data());
    const auto input_view =
        program_ ? OperatorVectorView::from_device_pointer(
                       program_, input_address, descriptor.domain, false)
                 : OperatorVectorView::from_const_host(input.data(),
                                                       descriptor.domain);
    const auto output_view =
        program_ ? OperatorVectorView::from_device_pointer(
                       program_, output_address, descriptor.range, true)
                 : OperatorVectorView::from_mutable_host(output.data(),
                                                         descriptor.range);
    plan.submit(generation,
                {OperatorApplyMode::forward, input_view, nullptr,
                 output_view});
    preconditioner_apply_calls_.fetch_add(1,
                                          std::memory_order_relaxed);
  }

  double true_residual_norm(const OperatorPinnedAction &generation) {
    apply_operator(generation, x_, residual_);
    for (int index = 0; index < rows_; ++index) {
      residual_[index] = b_[index] - residual_[index];
    }
    return vector_norm(residual_);
  }

  void update_solution(double alpha, double omega) {
    const auto &solution_direction =
        preconditioner_plan_ ? preconditioned_direction_ : direction_;
    const auto &solution_intermediate =
        preconditioner_plan_ ? preconditioned_intermediate_
                             : intermediate_residual_;
    for (int index = 0; index < rows_; ++index) {
      x_[index] = static_cast<DT>(
          static_cast<double>(x_[index]) +
          alpha * static_cast<double>(solution_direction[index]) +
          omega * static_cast<double>(solution_intermediate[index]));
    }
  }

  void set_breakdown(SparseSolveBreakdownReason reason) {
    status_ = SparseSolveStatus::kBreakdown;
    breakdown_reason_ = reason;
  }

  void restart_from_true_residual(double &rho_old,
                                  double &alpha,
                                  double &omega,
                                  bool &fresh_direction) {
    shadow_residual_ = residual_;
    direction_.setZero();
    operator_direction_.setZero();
    if (preconditioner_plan_) {
      preconditioned_direction_.setZero();
      preconditioned_intermediate_.setZero();
    }
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
  std::unique_ptr<OperatorPlan> operator_plan_;
  std::unique_ptr<PreconditionerPlan> preconditioner_plan_;
  DataType dtype_{PrimitiveType::f32};
  int rows_{0};
  int cols_{0};
  EigenT x_;
  EigenT b_;
  EigenT residual_;
  EigenT shadow_residual_;
  EigenT direction_;
  EigenT operator_direction_;
  EigenT intermediate_residual_;
  EigenT operator_intermediate_;
  EigenT preconditioned_direction_;
  EigenT preconditioned_intermediate_;
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
  SparseSolveBreakdownReason breakdown_reason_{
      SparseSolveBreakdownReason::none};
  std::atomic<std::uint64_t> solve_calls_{0};
  std::atomic<std::uint64_t> total_iterations_{0};
  std::atomic<std::uint64_t> workspace_reuses_{0};
  std::atomic<std::uint64_t> operator_apply_calls_{0};
  std::atomic<std::uint64_t> preconditioner_apply_calls_{0};
  std::atomic<std::uint64_t> host_scalar_reductions_{0};
  std::atomic<std::uint64_t> last_solve_pattern_version_{0};
  std::atomic<std::uint64_t> last_solve_numeric_version_{0};
};

}  // namespace taichi::lang
