#pragma once

#include "taichi/program/conjugate_gradient.h"
#include "taichi/program/linear_operator.h"

#include <cstring>
#include <limits>

namespace taichi::lang {

// Hardened identity-preconditioned MINRES over one provider-neutral
// OperatorPlan. The recurrence owns persistent host vectors and pins exactly
// one operator generation for the complete solve.
template <typename EigenT, typename DT>
class OperatorMINRES {
 public:
  OperatorMINRES(Program *program,
                 OperatorBinding operator_binding,
                 int max_iterations,
                 DT absolute_tolerance,
                 DT relative_tolerance = static_cast<DT>(0))
      : program_(program),
        max_iterations_(max_iterations),
        absolute_tolerance_(absolute_tolerance),
        relative_tolerance_(relative_tolerance) {
    TI_ERROR_IF(!program_ || !arch_is_cpu(program_->compile_config().arch),
                "Operator MINRES currently requires an owning CPU Program.");
    operator_plan_ = std::make_unique<OperatorPlan>(
        program_, std::move(operator_binding));
    const auto &descriptor = operator_plan_->descriptor();
    validate_operator_solver_compatibility(
        descriptor, operator_plan_->mathematical_traits(),
        OperatorSolverFamily::minres);
    TI_ERROR_IF(descriptor.domain.scalar_extent >
                    static_cast<std::size_t>(
                        std::numeric_limits<int>::max()),
                "Operator MINRES extent exceeds the supported int range.");
    rows_ = static_cast<int>(descriptor.range.scalar_extent);
    cols_ = static_cast<int>(descriptor.domain.scalar_extent);
    dtype_ = descriptor.range.scalar_type;
    const DataType expected_dtype =
        std::is_same_v<DT, float64> ? DataType(PrimitiveType::f64)
                                    : DataType(PrimitiveType::f32);
    TI_ERROR_IF(dtype_ != expected_dtype,
                "Operator MINRES dtype does not match the selected "
                "f32/f64 recurrence.");
    TI_ERROR_IF(max_iterations_ < 0,
                "Operator MINRES requires non-negative max iterations.");
    TI_ERROR_IF(!std::isfinite(absolute_tolerance_) ||
                    !std::isfinite(relative_tolerance_) ||
                    absolute_tolerance_ < static_cast<DT>(0) ||
                    relative_tolerance_ < static_cast<DT>(0) ||
                    (absolute_tolerance_ == static_cast<DT>(0) &&
                     relative_tolerance_ == static_cast<DT>(0)),
                "Operator MINRES requires finite non-negative atol and rtol "
                "with at least one positive tolerance.");

    x_ = EigenT::Zero(rows_);
    b_ = EigenT::Zero(rows_);
    residual_ = EigenT::Zero(rows_);
    v_old_ = EigenT::Zero(rows_);
    v_ = EigenT::Zero(rows_);
    v_new_ = EigenT::Zero(rows_);
    p_older_ = EigenT::Zero(rows_);
    p_old_ = EigenT::Zero(rows_);
    p_ = EigenT::Zero(rows_);
    try {
      operator_input_ = program_->create_ndarray(
          dtype_, {rows_}, ExternalArrayLayout::kNull, false);
      operator_output_ = program_->create_ndarray(
          dtype_, {rows_}, ExternalArrayLayout::kNull, false);
    } catch (...) {
      release_workspace();
      throw;
    }
  }

  ~OperatorMINRES() {
    release_workspace();
  }

  void solve_ndarray(Program *program, Ndarray &x, Ndarray &b) {
    TI_ERROR_IF(program != program_,
                "Operator MINRES solve must use its construction Program.");
    const auto x_address = program->get_ndarray_data_ptr_as_int(&x);
    const auto b_address = program->get_ndarray_data_ptr_as_int(&b);
    x_ = Eigen::Map<EigenT>(reinterpret_cast<DT *>(x_address), cols_);
    b_ = Eigen::Map<EigenT>(reinterpret_cast<DT *>(b_address), rows_);
    solve();
    Eigen::Map<EigenT>(reinterpret_cast<DT *>(x_address), cols_) = x_;
  }

  void solve() {
    auto operator_generation = operator_plan_->pin();
    reset_last_result(operator_generation.resource_stamp());
    const auto previous_solve_calls =
        solve_calls_.fetch_add(1, std::memory_order_relaxed);
    if (previous_solve_calls > 0) {
      workspace_reuses_.fetch_add(1, std::memory_order_relaxed);
    }

    if (!x_.allFinite() || !b_.allFinite()) {
      initial_residual_norm_ = std::numeric_limits<double>::infinity();
      residual_norm_ = initial_residual_norm_;
      estimated_residual_norm_ = initial_residual_norm_;
      status_ = SparseSolveStatus::kBreakdown;
      finish_solve();
      return;
    }
    const double rhs_norm = vector_norm(b_);
    relative_reference_norm_ = rhs_norm;
    effective_tolerance_ = std::max(
        static_cast<double>(absolute_tolerance_),
        static_cast<double>(relative_tolerance_) * rhs_norm);
    initial_residual_norm_ = true_residual_norm(operator_generation);
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
      apply_operator(operator_generation, v_, v_new_);
      v_new_ -= beta * v_old_;
      const DT alpha = static_cast<DT>(dot(v_new_, v_));
      v_new_ -= alpha * v_;
      const DT beta_new = static_cast<DT>(vector_norm(v_new_));
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
        residual_norm_ = true_residual_norm(operator_generation);
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
        residual_norm_ = true_residual_norm(operator_generation);
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
      residual_norm_ = true_residual_norm(operator_generation);
      if (!x_.allFinite() || !std::isfinite(residual_norm_)) {
        status_ = SparseSolveStatus::kBreakdown;
      } else if (residual_norm_ <= effective_tolerance_) {
        status_ = SparseSolveStatus::kConverged;
      } else {
        status_ = SparseSolveStatus::kMaxIterations;
      }
    } else if (status_ == SparseSolveStatus::kBreakdown && x_.allFinite() &&
               b_.allFinite()) {
      residual_norm_ = true_residual_norm(operator_generation);
    }
    finish_solve();
  }

  SparseSolveResult get_last_result() const {
    return {status_, iterations_, initial_residual_norm_, residual_norm_,
            static_cast<double>(absolute_tolerance_),
            static_cast<double>(relative_tolerance_),
            relative_reference_norm_, effective_tolerance_};
  }

  SparseSolvePlanRuntimeStatistics debug_runtime_statistics() const {
    const auto operator_stamp = operator_plan_->resource_stamp();
    const auto plan_statistics = operator_plan_->debug_runtime_statistics();
    SparseSolvePlanRuntimeStatistics result;
    result.backend_family = "cpu";
    result.method = "minres";
    result.dtype = data_type_name(dtype_);
    result.rows = rows_;
    result.cols = cols_;
    result.max_iterations = max_iterations_;
    result.absolute_tolerance = static_cast<double>(absolute_tolerance_);
    result.relative_tolerance = static_cast<double>(relative_tolerance_);
    result.last_relative_reference_norm = relative_reference_norm_;
    result.last_effective_tolerance = effective_tolerance_;
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
    result.preconditioner_method = "identity";
    result.preconditioner_apply_calls = 0;
    result.preconditioner_apply_calls_available = true;
    result.preconditioner_ownership_scope = "none";
    result.persistent_vector_count = 11;
    result.persistent_vector_reserved_bytes =
        static_cast<std::uint64_t>(11) *
        static_cast<std::uint64_t>(rows_) * sizeof(DT);
    result.solver_state_rebuilt_each_solve = false;
    result.transient_solver_workspace_bytes = 0;
    result.transient_solver_workspace_bytes_available = true;
    return result;
  }

 private:
  static bool finite_scalar(DT value) {
    return std::isfinite(static_cast<double>(value));
  }

  void reset_last_result(const OperatorResourceStamp &stamp) {
    status_ = SparseSolveStatus::kNotRun;
    iterations_ = 0;
    initial_residual_norm_ = 0.0;
    residual_norm_ = 0.0;
    estimated_residual_norm_ = 0.0;
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
        program_->get_ndarray_data_ptr_as_int(operator_input_);
    const auto output_address =
        program_->get_ndarray_data_ptr_as_int(operator_output_);
    std::memcpy(reinterpret_cast<void *>(input_address), input.data(),
                static_cast<std::size_t>(rows_) * sizeof(DT));
    operator_plan_->submit(
        generation,
        {OperatorApplyMode::forward,
         OperatorVectorView::from_ndarray(
             program_, *operator_input_, descriptor.domain, false),
         nullptr,
         OperatorVectorView::from_ndarray(
             program_, *operator_output_, descriptor.range, true)});
    std::memcpy(output.data(), reinterpret_cast<const void *>(output_address),
                static_cast<std::size_t>(rows_) * sizeof(DT));
    operator_apply_calls_.fetch_add(1, std::memory_order_relaxed);
  }

  void release_workspace() {
    if (operator_input_ && program_) {
      program_->delete_ndarray(operator_input_);
    }
    if (operator_output_ && program_) {
      program_->delete_ndarray(operator_output_);
    }
    operator_input_ = nullptr;
    operator_output_ = nullptr;
  }

  double true_residual_norm(const OperatorPinnedAction &generation) {
    apply_operator(generation, x_, residual_);
    for (int index = 0; index < rows_; ++index) {
      residual_[index] = b_[index] - residual_[index];
    }
    return vector_norm(residual_);
  }

  void finish_solve() {
    total_iterations_.fetch_add(static_cast<std::uint64_t>(iterations_),
                                std::memory_order_relaxed);
  }

  Program *program_{nullptr};
  std::unique_ptr<OperatorPlan> operator_plan_;
  Ndarray *operator_input_{nullptr};
  Ndarray *operator_output_{nullptr};
  DataType dtype_{PrimitiveType::f32};
  int rows_{0};
  int cols_{0};
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
  std::atomic<std::uint64_t> host_scalar_reductions_{0};
  std::atomic<std::uint64_t> last_solve_pattern_version_{0};
  std::atomic<std::uint64_t> last_solve_numeric_version_{0};
};

}  // namespace taichi::lang
