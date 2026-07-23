#pragma once

#include "taichi/program/conjugate_gradient.h"
#include "taichi/program/linear_operator.h"

#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>

namespace taichi::lang {

// Provider-neutral restarted GMRES with fixed-linear right preconditioning.
// The fixed action permits reconstructing M^-1(V y) once per restart cycle,
// so ordinary GMRES stores V but not the Z basis required by FGMRES.
template <typename EigenT, typename DT>
class FixedSparseGMRES {
 public:
  using DenseMatrix =
      Eigen::Matrix<DT, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

  FixedSparseGMRES(Program *program,
                   SparseMatrix &matrix,
                   int max_iterations,
                   int restart,
                   DT absolute_tolerance,
                   bool verbose,
                   DT relative_tolerance = static_cast<DT>(0))
      : FixedSparseGMRES(
            program,
            make_cpu_fixed_sparse_operator_binding(program, matrix),
            max_iterations,
            restart,
            absolute_tolerance,
            verbose,
            relative_tolerance) {
  }

  FixedSparseGMRES(Program *program,
                   OperatorBinding operator_binding,
                   int max_iterations,
                   int restart,
                   DT absolute_tolerance,
                   bool verbose,
                   DT relative_tolerance = static_cast<DT>(0))
      : program_(program),
        max_iterations_(max_iterations),
        restart_(restart),
        absolute_tolerance_(absolute_tolerance),
        relative_tolerance_(relative_tolerance),
        verbose_(verbose) {
    TI_ERROR_IF(program_ && !arch_is_cpu(program_->compile_config().arch),
                "Operator GMRES supports host-reference or CPU Program "
                "bindings only.");
    TI_ERROR_IF(restart_ != 8 && restart_ != 16 && restart_ != 32,
                "GMRES restart must be one of 8, 16, or 32.");
    TI_ERROR_IF(max_iterations_ < 0,
                "GMRES requires non-negative max_iterations.");
    TI_ERROR_IF(!std::isfinite(absolute_tolerance_) ||
                    !std::isfinite(relative_tolerance_) ||
                    absolute_tolerance_ < static_cast<DT>(0) ||
                    relative_tolerance_ < static_cast<DT>(0) ||
                    (absolute_tolerance_ == static_cast<DT>(0) &&
                     relative_tolerance_ == static_cast<DT>(0)),
                "GMRES requires finite non-negative atol/rtol with at "
                "least one positive tolerance.");

    operator_plan_ = std::make_unique<OperatorPlan>(
        program_, std::move(operator_binding));
    const auto &descriptor = operator_plan_->descriptor();
    validate_operator_solver_compatibility(
        descriptor, operator_plan_->mathematical_traits(),
        OperatorSolverFamily::gmres);
    TI_ERROR_IF(descriptor.domain.scalar_extent >
                    static_cast<std::size_t>(
                        std::numeric_limits<int>::max()),
                "Operator GMRES extent exceeds the supported int range.");
    rows_ = static_cast<int>(descriptor.range.scalar_extent);
    cols_ = static_cast<int>(descriptor.domain.scalar_extent);
    dtype_ = descriptor.range.scalar_type;
    const DataType expected_dtype =
        std::is_same_v<DT, float64> ? DataType(PrimitiveType::f64)
                                    : DataType(PrimitiveType::f32);
    TI_ERROR_IF(dtype_ != expected_dtype,
                "Operator GMRES dtype does not match the selected "
                "f32/f64 recurrence.");

    allocate_workspace();
  }

  FixedSparseGMRES(Program *program,
                   OperatorBinding operator_binding,
                   ExperimentalLinearOperatorHandle &preconditioner,
                   int max_iterations,
                   int restart,
                   DT absolute_tolerance,
                   bool verbose,
                   DT relative_tolerance = static_cast<DT>(0))
      : FixedSparseGMRES(program, std::move(operator_binding),
                         max_iterations, restart, absolute_tolerance,
                         verbose, relative_tolerance) {
    preconditioner_plan_ = make_solver_right_preconditioner_plan(
        program_, *operator_plan_, preconditioner, "linear_operator");
    preconditioned_work_ = EigenT::Zero(rows_);
  }

  void set_x(EigenT &x) {
    TI_ERROR_IF(x.size() != cols_,
                "GMRES initial guess must have {} entries, got {}.", cols_,
                x.size());
    x_ = x;
  }

  void reset_x() {
    x_.setZero();
  }

  void set_b(EigenT &b) {
    TI_ERROR_IF(b.size() != rows_,
                "GMRES RHS must have {} entries, got {}.", rows_, b.size());
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
                "Operator GMRES solve must use its construction Program.");
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

    relative_reference_norm_ = vector_norm(b_);
    effective_tolerance_ = std::max(
        static_cast<double>(absolute_tolerance_),
        static_cast<double>(relative_tolerance_) *
            relative_reference_norm_);
    initial_residual_norm_ =
        true_residual_norm(operator_generation);
    residual_norm_ = initial_residual_norm_;
    if (!std::isfinite(relative_reference_norm_) ||
        !std::isfinite(effective_tolerance_) ||
        !std::isfinite(residual_norm_)) {
      set_breakdown(SparseSolveBreakdownReason::nonfinite);
      finish_solve();
      return;
    }
    if (residual_norm_ <= effective_tolerance_) {
      status_ = SparseSolveStatus::kConverged;
      finish_solve();
      return;
    }
    if (max_iterations_ == 0) {
      status_ = SparseSolveStatus::kMaxIterations;
      finish_solve();
      return;
    }

    while (iterations_ < max_iterations_ &&
           status_ == SparseSolveStatus::kNotRun) {
      const int cycle_limit =
          std::min(restart_, max_iterations_ - iterations_);
      hessenberg_.setZero();
      givens_cosine_.setZero();
      givens_sine_.setZero();
      least_squares_rhs_.setZero();
      coefficients_.setZero();
      projection_.setZero();

      const double beta = residual_norm_;
      if (!std::isfinite(beta) || beta <= 0.0) {
        set_breakdown(SparseSolveBreakdownReason::orthogonalization_failure);
        break;
      }
      basis_[0] = residual_ / static_cast<DT>(beta);
      vector_update_calls_.fetch_add(1, std::memory_order_relaxed);
      least_squares_rhs_[0] = static_cast<DT>(beta);

      int used = 0;
      bool happy = false;
      for (int j = 0; j < cycle_limit; ++j) {
        const EigenT *operator_input = &basis_[j];
        if (preconditioner_plan_) {
          apply_preconditioner(preconditioner_generation, basis_[j],
                               preconditioned_work_);
          operator_input = &preconditioned_work_;
        }
        apply_operator(operator_generation, *operator_input, work_);
        if (!work_.allFinite()) {
          set_breakdown(SparseSolveBreakdownReason::nonfinite);
          break;
        }
        const double preorthogonal_norm = vector_norm(work_);
        if (!std::isfinite(preorthogonal_norm)) {
          set_breakdown(SparseSolveBreakdownReason::nonfinite);
          break;
        }

        for (int pass = 0; pass < 2; ++pass) {
          multi_dot(work_, j + 1);
          for (int i = 0; i <= j; ++i) {
            hessenberg_(i, j) += projection_[i];
          }
          project(work_, j + 1);
        }

        const double next_norm = vector_norm(work_);
        if (!std::isfinite(next_norm)) {
          set_breakdown(SparseSolveBreakdownReason::nonfinite);
          break;
        }
        hessenberg_(j + 1, j) = static_cast<DT>(next_norm);
        const double happy_tolerance =
            64.0 * static_cast<double>(std::numeric_limits<DT>::epsilon()) *
            std::max(preorthogonal_norm,
                     static_cast<double>(std::numeric_limits<DT>::min()));
        happy = next_norm <= happy_tolerance;

        for (int i = 0; i < j; ++i) {
          const DT upper = hessenberg_(i, j);
          const DT lower = hessenberg_(i + 1, j);
          hessenberg_(i, j) =
              givens_cosine_[i] * upper + givens_sine_[i] * lower;
          hessenberg_(i + 1, j) =
              -givens_sine_[i] * upper + givens_cosine_[i] * lower;
        }
        const double diagonal = static_cast<double>(hessenberg_(j, j));
        const double subdiagonal =
            static_cast<double>(hessenberg_(j + 1, j));
        const double denominator = std::hypot(diagonal, subdiagonal);
        if (!std::isfinite(denominator)) {
          set_breakdown(SparseSolveBreakdownReason::nonfinite);
          break;
        }
        if (denominator <=
            static_cast<double>(std::numeric_limits<DT>::min())) {
          set_breakdown(SparseSolveBreakdownReason::hessenberg_singular);
          break;
        }
        givens_cosine_[j] = static_cast<DT>(diagonal / denominator);
        givens_sine_[j] = static_cast<DT>(subdiagonal / denominator);
        hessenberg_(j, j) = static_cast<DT>(denominator);
        hessenberg_(j + 1, j) = static_cast<DT>(0);
        const DT rhs_entry = least_squares_rhs_[j];
        least_squares_rhs_[j] = givens_cosine_[j] * rhs_entry;
        least_squares_rhs_[j + 1] = -givens_sine_[j] * rhs_entry;

        ++iterations_;
        used = j + 1;
        if (!happy) {
          basis_[j + 1] = work_ / static_cast<DT>(next_norm);
          vector_update_calls_.fetch_add(1, std::memory_order_relaxed);
        }
        const double estimated_residual =
            std::abs(static_cast<double>(least_squares_rhs_[j + 1]));
        if (!std::isfinite(estimated_residual)) {
          set_breakdown(SparseSolveBreakdownReason::nonfinite);
          break;
        }
        if (happy || estimated_residual <= effective_tolerance_ ||
            iterations_ >= max_iterations_) {
          break;
        }
      }

      if (status_ == SparseSolveStatus::kBreakdown) {
        break;
      }
      if (used <= 0 || !solve_hessenberg(used)) {
        if (status_ == SparseSolveStatus::kNotRun) {
          set_breakdown(SparseSolveBreakdownReason::hessenberg_singular);
        }
        break;
      }

      update_.setZero();
      for (int i = 0; i < used; ++i) {
        update_.noalias() += coefficients_[i] * basis_[i];
      }
      vector_update_calls_.fetch_add(1, std::memory_order_relaxed);
      if (preconditioner_plan_) {
        apply_preconditioner(preconditioner_generation, update_,
                             preconditioned_work_);
        x_ += preconditioned_work_;
      } else {
        x_ += update_;
      }
      vector_update_calls_.fetch_add(1, std::memory_order_relaxed);
      restart_cycles_.fetch_add(1, std::memory_order_relaxed);
      residual_norm_ = true_residual_norm(operator_generation);
      if (!std::isfinite(residual_norm_)) {
        set_breakdown(SparseSolveBreakdownReason::nonfinite);
        break;
      }
      if (residual_norm_ <= effective_tolerance_) {
        status_ = SparseSolveStatus::kConverged;
        if (happy) {
          happy_breakdowns_.fetch_add(1, std::memory_order_relaxed);
        }
        break;
      }
      if (happy) {
        set_breakdown(SparseSolveBreakdownReason::arnoldi_breakdown);
        break;
      }
    }

    if (status_ == SparseSolveStatus::kNotRun) {
      status_ = SparseSolveStatus::kMaxIterations;
    }
    finish_solve();
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

  EigenT get_x() const {
    return x_;
  }

  SparseSolveResult get_last_result() const {
    return {status_, iterations_, initial_residual_norm_, residual_norm_,
            absolute_tolerance_, relative_tolerance_,
            relative_reference_norm_, effective_tolerance_,
            breakdown_reason_};
  }

  SparseSolvePlanRuntimeStatistics debug_runtime_statistics() const {
    SparseSolvePlanRuntimeStatistics result;
    result.backend_family = program_ ? arch_name(program_->compile_config().arch)
                                     : "host_reference";
    result.method = "gmres";
    result.dtype = dtype_ == PrimitiveType::f64 ? "f64" : "f32";
    result.rows = rows_;
    result.cols = cols_;
    result.max_iterations = max_iterations_;
    result.absolute_tolerance = absolute_tolerance_;
    result.relative_tolerance = relative_tolerance_;
    result.last_relative_reference_norm = relative_reference_norm_;
    result.last_effective_tolerance = effective_tolerance_;
    result.last_breakdown_reason =
        sparse_solve_breakdown_reason_name(breakdown_reason_);
    const auto stamp = operator_plan_->resource_stamp();
    result.operator_pattern_version = stamp.topology_revision;
    result.operator_numeric_version = stamp.numeric_revision;
    result.last_solve_pattern_version =
        last_solve_pattern_version_.load(std::memory_order_relaxed);
    result.last_solve_numeric_version =
        last_solve_numeric_version_.load(std::memory_order_relaxed);
    result.operator_pattern_changed_since_last_solve =
        solve_calls_.load(std::memory_order_relaxed) > 0 &&
        stamp.topology_revision != result.last_solve_pattern_version;
    result.operator_numeric_changed_since_last_solve =
        solve_calls_.load(std::memory_order_relaxed) > 0 &&
        stamp.numeric_revision != result.last_solve_numeric_version;
    result.operator_action_provider = operator_plan_->provider_name();
    result.operator_asynchronous_submit =
        operator_plan_->capabilities().asynchronous_submit;
    const auto operator_stats = operator_plan_->debug_runtime_statistics();
    result.operator_generation_pins = operator_stats.generation_pins;
    result.operator_generation_changes = operator_stats.generation_changes;
    result.operator_numeric_generation_changes =
        operator_stats.numeric_generation_changes;
    result.operator_binding_generation_changes =
        operator_stats.binding_generation_changes;
    result.operator_plan_invalidations = operator_stats.invalidations;
    result.solve_calls = solve_calls_.load(std::memory_order_relaxed);
    result.total_iterations =
        total_iterations_.load(std::memory_order_relaxed);
    result.workspace_builds = 1;
    result.workspace_reuses =
        workspace_reuses_.load(std::memory_order_relaxed);
    result.operator_apply_calls =
        operator_apply_calls_.load(std::memory_order_relaxed);
    result.operator_apply_calls_available = true;
    result.dot_product_calls =
        dot_product_calls_.load(std::memory_order_relaxed);
    result.dot_product_calls_available = true;
    result.multi_dot_calls =
        multi_dot_calls_.load(std::memory_order_relaxed);
    result.multi_dot_calls_available = true;
    result.vector_update_calls =
        vector_update_calls_.load(std::memory_order_relaxed);
    result.vector_update_calls_available = true;
    result.host_scalar_reductions =
        host_scalar_reductions_.load(std::memory_order_relaxed);
    result.logical_iterations = result.total_iterations;
    result.executed_iterations = result.total_iterations;
    result.restart_cycles = restart_cycles_.load(std::memory_order_relaxed);
    result.happy_breakdowns =
        happy_breakdowns_.load(std::memory_order_relaxed);
    result.restart = restart_;
    result.orthogonalization_strategy = "cgs2_always_reorthogonalize";
    result.orthogonalization_passes = 2;
    result.requested_solver_execution_policy = "host_each_iteration";
    result.solver_execution_policy = "host_each_iteration";
    result.host_check_interval = 1;
    result.solver_scalar_location = "host";
    result.preconditioning_side = preconditioner_plan_ ? "right" : "none";
    result.preconditioner_method =
        preconditioner_plan_ ? preconditioner_plan_->method() : "identity";
    result.preconditioner_behavior =
        preconditioner_plan_ ? "fixed_linear" : "identity";
    result.preconditioner_apply_calls =
        preconditioner_apply_calls_.load(std::memory_order_relaxed);
    result.preconditioner_apply_calls_available = true;
    result.external_preconditioner = preconditioner_plan_ != nullptr;
    result.preconditioner_ownership_scope =
        preconditioner_plan_ ? "solve_plan" : "none";
    if (preconditioner_plan_) {
      append_solver_preconditioner_plan_statistics(*preconditioner_plan_,
                                                   result);
    }
    const std::uint64_t basis_count =
        static_cast<std::uint64_t>(restart_ + 1);
    const std::uint64_t auxiliary_count = preconditioner_plan_ ? 4u : 3u;
    result.persistent_vector_count = basis_count + auxiliary_count;
    result.persistent_vector_reserved_bytes =
        result.persistent_vector_count * static_cast<std::uint64_t>(rows_) *
        sizeof(DT);
    result.basis_vector_count = basis_count;
    result.basis_reserved_bytes =
        basis_count * static_cast<std::uint64_t>(rows_) * sizeof(DT);
    result.persistent_scalar_count =
        static_cast<std::uint64_t>(restart_) * restart_ +
        6u * static_cast<std::uint64_t>(restart_) + 1u;
    result.persistent_scalar_reserved_bytes =
        result.persistent_scalar_count * sizeof(DT);
    result.solver_state_rebuilt_each_solve = false;
    result.transient_solver_workspace_bytes = 0;
    result.transient_solver_workspace_bytes_available = true;
    return result;
  }

 private:
  void allocate_workspace() {
    x_ = EigenT::Zero(cols_);
    b_ = EigenT::Zero(rows_);
    residual_ = EigenT::Zero(rows_);
    work_ = EigenT::Zero(rows_);
    update_ = EigenT::Zero(rows_);
    basis_.reserve(static_cast<std::size_t>(restart_ + 1));
    for (int i = 0; i <= restart_; ++i) {
      basis_.push_back(EigenT::Zero(rows_));
    }
    hessenberg_ = DenseMatrix::Zero(restart_ + 1, restart_);
    givens_cosine_ = EigenT::Zero(restart_);
    givens_sine_ = EigenT::Zero(restart_);
    least_squares_rhs_ = EigenT::Zero(restart_ + 1);
    coefficients_ = EigenT::Zero(restart_);
    projection_ = EigenT::Zero(restart_);
  }

  void reset_last_result(const OperatorResourceStamp &stamp) {
    status_ = SparseSolveStatus::kNotRun;
    breakdown_reason_ = SparseSolveBreakdownReason::none;
    iterations_ = 0;
    initial_residual_norm_ = 0.0;
    residual_norm_ = 0.0;
    relative_reference_norm_ = 0.0;
    effective_tolerance_ = absolute_tolerance_;
    last_solve_pattern_version_.store(stamp.topology_revision,
                                      std::memory_order_relaxed);
    last_solve_numeric_version_.store(stamp.numeric_revision,
                                      std::memory_order_relaxed);
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

  double vector_norm(const EigenT &vector) {
    dot_product_calls_.fetch_add(1, std::memory_order_relaxed);
    host_scalar_reductions_.fetch_add(1, std::memory_order_relaxed);
    return static_cast<double>(vector.norm());
  }

  void multi_dot(const EigenT &vector, int count) {
    multi_dot_calls_.fetch_add(1, std::memory_order_relaxed);
    dot_product_calls_.fetch_add(static_cast<std::uint64_t>(count),
                                 std::memory_order_relaxed);
    host_scalar_reductions_.fetch_add(1, std::memory_order_relaxed);
    for (int i = 0; i < count; ++i) {
      projection_[i] = basis_[i].dot(vector);
    }
  }

  void project(EigenT &vector, int count) {
    for (int i = 0; i < count; ++i) {
      vector.noalias() -= projection_[i] * basis_[i];
    }
    vector_update_calls_.fetch_add(1, std::memory_order_relaxed);
  }

  double true_residual_norm(const OperatorPinnedAction &generation) {
    apply_operator(generation, x_, residual_);
    residual_ = b_ - residual_;
    vector_update_calls_.fetch_add(1, std::memory_order_relaxed);
    return vector_norm(residual_);
  }

  bool solve_hessenberg(int used) {
    DT scale = static_cast<DT>(0);
    for (int row = 0; row < used; ++row) {
      for (int col = row; col < used; ++col) {
        scale = std::max(scale, std::abs(hessenberg_(row, col)));
      }
    }
    const double pivot_tolerance =
        64.0 * static_cast<double>(std::numeric_limits<DT>::epsilon()) *
        std::max(static_cast<double>(scale),
                 static_cast<double>(std::numeric_limits<DT>::min()));
    for (int row = used - 1; row >= 0; --row) {
      double value = static_cast<double>(least_squares_rhs_[row]);
      for (int col = row + 1; col < used; ++col) {
        value -= static_cast<double>(hessenberg_(row, col)) *
                 static_cast<double>(coefficients_[col]);
      }
      const double pivot = static_cast<double>(hessenberg_(row, row));
      if (!std::isfinite(value) || !std::isfinite(pivot)) {
        set_breakdown(SparseSolveBreakdownReason::nonfinite);
        return false;
      }
      if (std::abs(pivot) <= pivot_tolerance) {
        set_breakdown(SparseSolveBreakdownReason::hessenberg_singular);
        return false;
      }
      coefficients_[row] = static_cast<DT>(value / pivot);
    }
    return coefficients_.head(used).allFinite();
  }

  void set_breakdown(SparseSolveBreakdownReason reason) {
    status_ = SparseSolveStatus::kBreakdown;
    breakdown_reason_ = reason;
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
  EigenT work_;
  EigenT update_;
  EigenT preconditioned_work_;
  std::vector<EigenT> basis_;
  DenseMatrix hessenberg_;
  EigenT givens_cosine_;
  EigenT givens_sine_;
  EigenT least_squares_rhs_;
  EigenT coefficients_;
  EigenT projection_;
  int max_iterations_{0};
  int restart_{0};
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
  std::atomic<std::uint64_t> dot_product_calls_{0};
  std::atomic<std::uint64_t> multi_dot_calls_{0};
  std::atomic<std::uint64_t> vector_update_calls_{0};
  std::atomic<std::uint64_t> host_scalar_reductions_{0};
  std::atomic<std::uint64_t> restart_cycles_{0};
  std::atomic<std::uint64_t> happy_breakdowns_{0};
  std::atomic<std::uint64_t> last_solve_pattern_version_{0};
  std::atomic<std::uint64_t> last_solve_numeric_version_{0};
};

}  // namespace taichi::lang
