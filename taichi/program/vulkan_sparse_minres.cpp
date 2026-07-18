#include "taichi/program/vulkan_sparse_minres.h"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace taichi::lang {

namespace {

constexpr std::size_t kStateWordCount = 24;

enum StateWord : std::size_t {
  kInitialResidualSquared = 0,
  kRightHandSideSquared = 1,
  kSolutionSquared = 2,
  kTrueResidualSquared = 3,
  kToleranceSquared = 4,
  kAlpha = 7,
  kBetaNewSquared = 8,
  kStatus = 20,
  kCompletedIterations = 21,
};

enum ScalarStage : std::uint32_t {
  kInitialize = 0,
  kPrepareRotation = 1,
  kFinalizeIteration = 2,
  kFinish = 3,
};

float decode_float(std::uint32_t word) {
  float value = 0.0f;
  static_assert(sizeof(value) == sizeof(word));
  std::memcpy(&value, &word, sizeof(value));
  return value;
}

}  // namespace

VulkanSparseMINRESPlan::VulkanSparseMINRESPlan(
    Program *program,
    SparseMatrix &matrix,
    int max_iterations,
    float absolute_tolerance,
    bool verbose,
    float relative_tolerance)
    : program_(program),
      matrix_(matrix),
      max_iterations_(max_iterations),
      absolute_tolerance_(absolute_tolerance),
      relative_tolerance_(relative_tolerance),
      verbose_(verbose) {
#if defined(TI_WITH_VULKAN)
  TI_ERROR_IF(!program_ || program_->compile_config().arch != Arch::vulkan,
              "Vulkan fixed MINRES requires an active Vulkan Program.");
  const auto operator_stats = matrix_.debug_runtime_statistics();
  csr_matrix_ = dynamic_cast<VulkanSparseMatrix *>(&matrix_);
  bsr_matrix_ = dynamic_cast<VulkanSparseBsrMatrix *>(&matrix_);
  const bool is_public_fixed =
      operator_stats.backend_family == "vulkan" &&
      operator_stats.provider_name == "forge_vulkan_native" &&
      operator_stats.pattern_storage_shared &&
      operator_stats.pattern_builds == 0 &&
      ((csr_matrix_ && operator_stats.storage_format == "csr") ||
       (bsr_matrix_ && operator_stats.storage_format == "bsr"));
  TI_ERROR_IF(!is_public_fixed,
              "Vulkan fixed MINRES requires a caller-owned shared CSR/BSR "
              "pattern with pattern_builds=0.");
  TI_ERROR_IF(matrix_.num_rows() <= 0 ||
                  matrix_.num_rows() != matrix_.num_cols(),
              "Vulkan fixed MINRES requires a non-empty square matrix.");
  TI_ERROR_IF(matrix_.get_data_type() != PrimitiveType::f32,
              "Vulkan fixed MINRES currently requires f32 values.");
  validate_controls();

  const int n = matrix_.num_rows();
  auto create_vector = [&]() {
    return program_->create_ndarray(PrimitiveType::f32, {n},
                                    ExternalArrayLayout::kNull, false);
  };
  try {
    residual_ = create_vector();
    v_old_ = create_vector();
    v_ = create_vector();
    v_new_ = create_vector();
    p_older_ = create_vector();
    p_old_ = create_vector();
    p_ = create_vector();
    state_ = program_->create_ndarray(
        PrimitiveType::u32, {static_cast<int>(kStateWordCount)},
        ExternalArrayLayout::kNull, false);
  } catch (...) {
    release_workspace();
    throw;
  }
  workspace_builds_ = 1;
#else
  TI_NOT_IMPLEMENTED;
#endif
}

VulkanSparseMINRESPlan::~VulkanSparseMINRESPlan() {
  release_workspace();
}

void VulkanSparseMINRESPlan::validate_controls() const {
  TI_ERROR_IF(max_iterations_ < 0,
              "Vulkan fixed MINRES requires non-negative max iterations.");
  TI_ERROR_IF(!std::isfinite(absolute_tolerance_) ||
                  !std::isfinite(relative_tolerance_) ||
                  absolute_tolerance_ < 0.0f ||
                  relative_tolerance_ < 0.0f ||
                  (absolute_tolerance_ == 0.0f &&
                   relative_tolerance_ == 0.0f),
              "Vulkan fixed MINRES requires finite non-negative atol and "
              "rtol with at least one positive tolerance.");
}

void VulkanSparseMINRESPlan::apply_operator(Program *program,
                                            const Ndarray &input,
                                            const Ndarray &output,
                                            bool masked) {
  if (csr_matrix_) {
    if (masked) {
      csr_matrix_->nd_spmv_masked(
          program, input, output, *state_, kStatus);
    } else {
      csr_matrix_->nd_spmv(program, input, output);
    }
  } else if (bsr_matrix_) {
    if (masked) {
      bsr_matrix_->nd_spmv_masked(
          program, input, output, *state_, kStatus);
    } else {
      bsr_matrix_->nd_spmv(program, input, output);
    }
  } else {
    TI_ERROR("Vulkan fixed MINRES received an unsupported operator.");
  }
  operator_apply_calls_++;
  masked_operator_dispatches_ += masked ? 1 : 0;
}

void VulkanSparseMINRESPlan::solve(Program *program,
                                   const Ndarray &x,
                                   const Ndarray &b) {
#if defined(TI_WITH_VULKAN)
  TI_ERROR_IF(program != program_,
              "Vulkan fixed MINRES requires its construction Program.");
  const int n = matrix_.num_rows();
  auto check_vector = [&](const char *role, const Ndarray &array) {
    TI_ERROR_IF(array.shape.size() != 1 ||
                    array.get_element_data_type() != PrimitiveType::f32 ||
                    !array.get_element_shape().empty() ||
                    array.get_element_size() != sizeof(float32) ||
                    array.get_nelement() != static_cast<std::size_t>(n),
                "Vulkan fixed MINRES {} must contain exactly {} scalar f32 "
                "entries.",
                role, n);
  };
  check_vector("solution", x);
  check_vector("right-hand side", b);
  TI_ERROR_IF(x.get_device_allocation() == b.get_device_allocation(),
              "Vulkan fixed MINRES solution and RHS must not alias.");

  std::lock_guard<std::mutex> lock(solve_mutex_);
  auto submission_guard =
      program->acquire_runtime_resource_submission_guard();
  const std::uint64_t program_syncs_before =
      program->runtime_statistics().snapshot().synchronization.program_syncs;
  const Ndarray *resources[] = {&x,      &b,     residual_, v_old_,
                                v_,      v_new_, p_older_,  p_old_,
                                p_,      state_};
  program->retain_ndarrays_for_external_submission(
      resources, std::size(resources));
  const auto operator_stats = matrix_.debug_runtime_statistics();
  if (has_solved_) {
    workspace_reuses_++;
  } else {
    has_solved_ = true;
  }
  solve_calls_++;
  last_solve_pattern_version_ = operator_stats.pattern_version;
  last_solve_numeric_version_ = operator_stats.numeric_version;
  status_ = SparseSolveStatus::kNotRun;
  iterations_ = 0;
  initial_residual_norm_ = 0.0;
  residual_norm_ = 0.0;
  relative_reference_norm_ = 0.0;
  effective_tolerance_ = static_cast<double>(absolute_tolerance_);

  auto *mutable_x = const_cast<Ndarray *>(&x);
  auto *mutable_b = const_cast<Ndarray *>(&b);
  program->fill_ndarray_fast_u32(state_, 0);
  apply_operator(program, x, *v_new_, false);
  program->copy_ndarray_fast(residual_, mutable_b);
  program->vulkan_sparse_axpy(v_new_, residual_, n, -1.0f);
  program->vulkan_sparse_dot_to_state_slot(
      residual_, residual_, state_, kInitialResidualSquared, n);
  program->vulkan_sparse_dot_to_state_slot(
      mutable_b, mutable_b, state_, kRightHandSideSquared, n);
  program->vulkan_sparse_dot_to_state_slot(
      mutable_x, mutable_x, state_, kSolutionSquared, n);
  program->vulkan_sparse_minres_scalar(
      state_, kInitialize, 0, absolute_tolerance_, relative_tolerance_);
  program->vulkan_sparse_minres_init_vectors(
      state_, mutable_x, residual_, v_old_, v_, p_older_, p_old_, p_, n);

  for (int iteration = 0; iteration < max_iterations_; ++iteration) {
    apply_operator(program, *v_, *v_new_, true);
    program->vulkan_sparse_minres_lanczos_beta(
        state_, v_old_, v_new_, n);
    program->vulkan_sparse_dot_to_state_slot(
        v_new_, v_, state_, kAlpha, n);
    program->vulkan_sparse_minres_lanczos_alpha(
        state_, v_, v_new_, n);
    program->vulkan_sparse_dot_to_state_slot(
        v_new_, v_new_, state_, kBetaNewSquared, n);
    program->vulkan_sparse_minres_scalar(
        state_, kPrepareRotation, static_cast<std::uint32_t>(iteration),
        absolute_tolerance_, relative_tolerance_);
    program->vulkan_sparse_minres_direction(
        state_, v_, p_older_, p_old_, p_, mutable_x, n);
    program->vulkan_sparse_minres_shift(state_, v_old_, v_, v_new_, n);

    apply_operator(program, x, *v_new_, true);
    program->copy_ndarray_fast(residual_, mutable_b);
    program->vulkan_sparse_axpy(v_new_, residual_, n, -1.0f);
    program->vulkan_sparse_dot_to_state_slot(
        residual_, residual_, state_, kTrueResidualSquared, n);
    program->vulkan_sparse_dot_to_state_slot(
        mutable_x, mutable_x, state_, kSolutionSquared, n);
    program->vulkan_sparse_minres_scalar(
        state_, kFinalizeIteration, static_cast<std::uint32_t>(iteration),
        absolute_tolerance_, relative_tolerance_);
  }

  apply_operator(program, x, *v_new_, false);
  program->copy_ndarray_fast(residual_, mutable_b);
  program->vulkan_sparse_axpy(v_new_, residual_, n, -1.0f);
  program->vulkan_sparse_dot_to_state_slot(
      residual_, residual_, state_, kTrueResidualSquared, n);
  program->vulkan_sparse_dot_to_state_slot(
      mutable_x, mutable_x, state_, kSolutionSquared, n);
  program->vulkan_sparse_minres_scalar(
      state_, kFinish, static_cast<std::uint32_t>(max_iterations_),
      absolute_tolerance_, relative_tolerance_);

  std::array<std::uint32_t, kStateWordCount> state_host{};
  program->synchronize();
  program->copy_ndarray_to_host(state_, state_host.data(),
                                state_host.size() * sizeof(uint32_t));

  const float initial_squared =
      decode_float(state_host[kInitialResidualSquared]);
  const float rhs_squared =
      decode_float(state_host[kRightHandSideSquared]);
  const float final_squared =
      decode_float(state_host[kTrueResidualSquared]);
  const float tolerance_squared =
      decode_float(state_host[kToleranceSquared]);
  const std::uint32_t completed = state_host[kCompletedIterations];
  const std::int32_t status_value =
      static_cast<std::int32_t>(state_host[kStatus]);
  const bool valid_host_result =
      std::isfinite(initial_squared) && initial_squared >= 0.0f &&
      std::isfinite(rhs_squared) && rhs_squared >= 0.0f &&
      std::isfinite(final_squared) && final_squared >= 0.0f &&
      std::isfinite(tolerance_squared) && tolerance_squared >= 0.0f &&
      completed <= static_cast<std::uint32_t>(max_iterations_) &&
      status_value >= static_cast<int>(SparseSolveStatus::kMaxIterations) &&
      status_value <= static_cast<int>(SparseSolveStatus::kConverged);
  if (valid_host_result) {
    initial_residual_norm_ =
        std::sqrt(static_cast<double>(initial_squared));
    relative_reference_norm_ =
        std::sqrt(static_cast<double>(rhs_squared));
    residual_norm_ = std::sqrt(static_cast<double>(final_squared));
    effective_tolerance_ =
        std::sqrt(static_cast<double>(tolerance_squared));
    iterations_ = static_cast<int>(completed);
    status_ = static_cast<SparseSolveStatus>(status_value);
    if (status_ == SparseSolveStatus::kConverged &&
        residual_norm_ > effective_tolerance_) {
      status_ = SparseSolveStatus::kBreakdown;
    }
  } else {
    initial_residual_norm_ = std::numeric_limits<double>::infinity();
    residual_norm_ = initial_residual_norm_;
    status_ = SparseSolveStatus::kBreakdown;
    iterations_ = 0;
  }

  total_iterations_ += static_cast<std::uint64_t>(iterations_);
  device_scalar_operations_ +=
      7 + 6 * static_cast<std::uint64_t>(max_iterations_);
  host_scalar_readbacks_ += kStateWordCount;
  const std::uint64_t program_syncs_after =
      program->runtime_statistics().snapshot().synchronization.program_syncs;
  host_synchronizations_ += program_syncs_after - program_syncs_before;
  device_to_device_bytes_ +=
      static_cast<std::uint64_t>(max_iterations_ + 2) *
      static_cast<std::uint64_t>(n) * sizeof(float32);
  device_to_host_bytes_ += kStateWordCount * sizeof(uint32_t);
  if (verbose_) {
    fmt::print("#iterations:     {}\n", iterations_);
    fmt::print("residual norm:   {}\n", residual_norm_);
  }
#else
  TI_NOT_IMPLEMENTED;
#endif
}

SparseSolvePlanRuntimeStatistics
VulkanSparseMINRESPlan::debug_runtime_statistics() const {
  std::lock_guard<std::mutex> lock(solve_mutex_);
  const auto operator_stats = matrix_.debug_runtime_statistics();
  SparseSolvePlanRuntimeStatistics result;
  result.backend_family = "vulkan";
  result.method = "minres_identity_bounded_true_residual_probe";
  result.dtype = "f32";
  result.rows = matrix_.num_rows();
  result.cols = matrix_.num_cols();
  result.max_iterations = max_iterations_;
  result.absolute_tolerance = static_cast<double>(absolute_tolerance_);
  result.relative_tolerance = static_cast<double>(relative_tolerance_);
  result.last_relative_reference_norm = relative_reference_norm_;
  result.last_effective_tolerance = effective_tolerance_;
  result.operator_pattern_version = operator_stats.pattern_version;
  result.operator_numeric_version = operator_stats.numeric_version;
  result.last_solve_pattern_version = last_solve_pattern_version_;
  result.last_solve_numeric_version = last_solve_numeric_version_;
  result.operator_pattern_changed_since_last_solve =
      solve_calls_ > 0 && result.operator_pattern_version !=
                              result.last_solve_pattern_version;
  result.operator_numeric_changed_since_last_solve =
      solve_calls_ > 0 && result.operator_numeric_version !=
                              result.last_solve_numeric_version;
  result.solve_calls = solve_calls_;
  result.total_iterations = total_iterations_;
  result.workspace_builds = workspace_builds_;
  result.workspace_reuses = workspace_reuses_;
  result.operator_apply_calls = operator_apply_calls_;
  result.operator_apply_calls_available = true;
  result.operator_apply_call_scope = "scheduled_dispatches";
  result.masked_operator_dispatches = masked_operator_dispatches_;
  result.preconditioner_method = "identity";
  result.preconditioner_apply_calls = 0;
  result.preconditioner_apply_calls_available = true;
  result.preconditioner_ownership_scope = "none";
  result.host_scalar_reductions = 0;
  result.device_scalar_operations = device_scalar_operations_;
  result.host_scalar_readbacks = host_scalar_readbacks_;
  result.host_synchronizations = host_synchronizations_;
  result.host_synchronizations_exact = true;
  result.host_synchronization_scope = "program_syncs_during_solve";
  result.persistent_vector_count = 7;
  result.persistent_vector_reserved_bytes =
      7 * static_cast<std::uint64_t>(matrix_.num_rows()) * sizeof(float32);
  result.persistent_scalar_count = kStateWordCount;
  result.persistent_scalar_reserved_bytes =
      kStateWordCount * sizeof(uint32_t);
  result.cublas_handle_count = 0;
  result.solver_state_rebuilt_each_solve = false;
  result.transient_solver_workspace_bytes = 0;
  result.transient_solver_workspace_bytes_available = true;
  result.shared_primitive_workspace_bytes =
      program_->vulkan_sparse_algebra_workspace_bytes() +
      program_->vulkan_reduce_workspace_bytes();
  result.shared_primitive_workspace_bytes_available = true;
  result.shared_primitive_workspace_ownership_scope =
      "program_sparse_algebra_and_reduce_cache";
  result.fixed_iteration_only = false;
  result.bounded_masked_execution = true;
  result.device_to_device_bytes = device_to_device_bytes_;
  result.device_to_host_bytes = device_to_host_bytes_;
  return result;
}

void VulkanSparseMINRESPlan::release_workspace() {
#if defined(TI_WITH_VULKAN)
  if (!program_) {
    return;
  }
  auto release = [&](Ndarray *&array) {
    if (array) {
      program_->delete_ndarray(array);
      array = nullptr;
    }
  };
  release(state_);
  release(p_);
  release(p_old_);
  release(p_older_);
  release(v_new_);
  release(v_);
  release(v_old_);
  release(residual_);
#endif
}

std::unique_ptr<VulkanSparseMINRESPlan> make_vulkan_fixed_sparse_minres_plan(
    Program *program,
    SparseMatrix &matrix,
    int max_iterations,
    float absolute_tolerance,
    bool verbose,
    float relative_tolerance) {
  return std::make_unique<VulkanSparseMINRESPlan>(
      program, matrix, max_iterations, absolute_tolerance, verbose,
      relative_tolerance);
}

}  // namespace taichi::lang
