#include "taichi/program/vulkan_sparse_bicgstab.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace taichi::lang {

namespace {

constexpr std::size_t kStateWordCount = 20;

enum StateWord : std::size_t {
  kInitialResidualSquared = 0,
  kRightHandSideSquared = 1,
  kSolutionSquared = 2,
  kRho = 3,
  kResidualSquared = 4,
  kRhoOld = 5,
  kAlpha = 6,
  kOmega = 7,
  kBeta = 8,
  kAlphaDenominator = 9,
  kIntermediateSquared = 10,
  kOperatorIntermediateSquared = 11,
  kOperatorIntermediateDotIntermediate = 12,
  kTrueResidualSquared = 13,
  kToleranceSquared = 14,
  kStatus = 16,
  kCompletedIterations = 17,
};

enum ScalarStage : std::uint32_t {
  kInitialize = 0,
  kPrepareDirection = 1,
  kPrepareAlpha = 2,
  kPrepareOmega = 3,
  kFinalizeIteration = 4,
  kFinish = 5,
};

float decode_float(std::uint32_t word) {
  float value = 0.0f;
  static_assert(sizeof(value) == sizeof(word));
  std::memcpy(&value, &word, sizeof(value));
  return value;
}

}  // namespace

VulkanSparseBiCGSTABPlan::VulkanSparseBiCGSTABPlan(
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
              "Vulkan fixed BiCGSTAB requires an active Vulkan Program.");
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
              "Vulkan fixed BiCGSTAB requires a caller-owned shared CSR/BSR "
              "pattern with pattern_builds=0.");
  TI_ERROR_IF(matrix_.num_rows() <= 0 ||
                  matrix_.num_rows() != matrix_.num_cols(),
              "Vulkan fixed BiCGSTAB requires a non-empty square matrix.");
  TI_ERROR_IF(matrix_.get_data_type() != PrimitiveType::f32,
              "Vulkan fixed BiCGSTAB currently requires f32 values.");
  validate_controls();

  const int n = matrix_.num_rows();
  auto create_vector = [&]() {
    return program_->create_ndarray(PrimitiveType::f32, {n},
                                    ExternalArrayLayout::kNull, false);
  };
  try {
    residual_ = create_vector();
    shadow_residual_ = create_vector();
    direction_ = create_vector();
    operator_direction_ = create_vector();
    intermediate_residual_ = create_vector();
    operator_intermediate_ = create_vector();
    true_residual_ = create_vector();
    candidate_solution_ = create_vector();
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

VulkanSparseBiCGSTABPlan::~VulkanSparseBiCGSTABPlan() {
  release_workspace();
}

void VulkanSparseBiCGSTABPlan::validate_controls() const {
  TI_ERROR_IF(max_iterations_ < 0,
              "Vulkan fixed BiCGSTAB requires non-negative max "
              "iterations.");
  TI_ERROR_IF(!std::isfinite(absolute_tolerance_) ||
                  !std::isfinite(relative_tolerance_) ||
                  absolute_tolerance_ < 0.0f ||
                  relative_tolerance_ < 0.0f ||
                  (absolute_tolerance_ == 0.0f &&
                   relative_tolerance_ == 0.0f),
              "Vulkan fixed BiCGSTAB requires finite non-negative atol and "
              "rtol with at least one positive tolerance.");
}

void VulkanSparseBiCGSTABPlan::apply_operator(
    Program *program,
    const Ndarray &input,
    const Ndarray &output) {
  if (csr_matrix_) {
    csr_matrix_->nd_spmv(program, input, output);
  } else if (bsr_matrix_) {
    bsr_matrix_->nd_spmv(program, input, output);
  } else {
    TI_ERROR("Vulkan fixed BiCGSTAB received an unsupported operator.");
  }
  operator_apply_calls_++;
}

void VulkanSparseBiCGSTABPlan::solve(Program *program,
                                     const Ndarray &x,
                                     const Ndarray &b) {
#if defined(TI_WITH_VULKAN)
  TI_ERROR_IF(program != program_,
              "Vulkan fixed BiCGSTAB requires its construction Program.");
  const int n = matrix_.num_rows();
  auto check_vector = [&](const char *role, const Ndarray &array) {
    TI_ERROR_IF(array.shape.size() != 1 ||
                    array.get_element_data_type() != PrimitiveType::f32 ||
                    !array.get_element_shape().empty() ||
                    array.get_element_size() != sizeof(float32) ||
                    array.get_nelement() != static_cast<std::size_t>(n),
                "Vulkan fixed BiCGSTAB {} must contain exactly {} scalar "
                "f32 entries.",
                role, n);
  };
  check_vector("solution", x);
  check_vector("right-hand side", b);
  TI_ERROR_IF(x.get_device_allocation() == b.get_device_allocation(),
              "Vulkan fixed BiCGSTAB solution and RHS must not alias.");

  std::lock_guard<std::mutex> lock(solve_mutex_);
  auto submission_guard =
      program->acquire_runtime_resource_submission_guard();
  const Ndarray *resources[] = {
      &x,
      &b,
      residual_,
      shadow_residual_,
      direction_,
      operator_direction_,
      intermediate_residual_,
      operator_intermediate_,
      true_residual_,
      candidate_solution_,
      state_};
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
  program->copy_ndarray_fast(residual_, mutable_b);
  apply_operator(program, x, *operator_intermediate_);
  program->vulkan_sparse_axpy(operator_intermediate_, residual_, n, -1.0f);
  program->vulkan_sparse_dot_to_state_slot(
      residual_, residual_, state_, kInitialResidualSquared, n);
  program->vulkan_sparse_dot_to_state_slot(
      mutable_b, mutable_b, state_, kRightHandSideSquared, n);
  program->vulkan_sparse_dot_to_state_slot(
      mutable_x, mutable_x, state_, kSolutionSquared, n);
  program->vulkan_sparse_bicgstab_scalar(
      state_, kInitialize, 0, absolute_tolerance_, relative_tolerance_);
  program->vulkan_sparse_bicgstab_init_vectors(
      state_, mutable_x, residual_, shadow_residual_, direction_,
      intermediate_residual_, candidate_solution_, n);

  for (int iteration = 0; iteration < max_iterations_; ++iteration) {
    program->vulkan_sparse_dot_to_state_slot(
        shadow_residual_, residual_, state_, kRho, n);
    program->vulkan_sparse_dot_to_state_slot(
        residual_, residual_, state_, kResidualSquared, n);
    program->vulkan_sparse_bicgstab_scalar(
        state_, kPrepareDirection, static_cast<std::uint32_t>(iteration),
        absolute_tolerance_, relative_tolerance_);
    program->vulkan_sparse_bicgstab_direction(
        state_, residual_, shadow_residual_, direction_,
        operator_direction_, n);
    apply_operator(program, *direction_, *operator_direction_);
    program->vulkan_sparse_dot_to_state_slot(
        shadow_residual_, operator_direction_, state_, kAlphaDenominator, n);
    program->vulkan_sparse_bicgstab_scalar(
        state_, kPrepareAlpha, static_cast<std::uint32_t>(iteration),
        absolute_tolerance_, relative_tolerance_);
    program->vulkan_sparse_bicgstab_alpha_vectors(
        state_, residual_, direction_, operator_direction_, mutable_x,
        candidate_solution_, intermediate_residual_, n);
    program->vulkan_sparse_dot_to_state_slot(
        intermediate_residual_, intermediate_residual_, state_,
        kIntermediateSquared, n);
    program->vulkan_sparse_dot_to_state_slot(
        candidate_solution_, candidate_solution_, state_, kSolutionSquared,
        n);

    apply_operator(program, *candidate_solution_, *operator_intermediate_);
    program->copy_ndarray_fast(true_residual_, mutable_b);
    program->vulkan_sparse_axpy(
        operator_intermediate_, true_residual_, n, -1.0f);
    program->vulkan_sparse_dot_to_state_slot(
        true_residual_, true_residual_, state_, kTrueResidualSquared, n);

    apply_operator(program, *intermediate_residual_,
                   *operator_intermediate_);
    program->vulkan_sparse_dot_to_state_slot(
        operator_intermediate_, operator_intermediate_, state_,
        kOperatorIntermediateSquared, n);
    program->vulkan_sparse_dot_to_state_slot(
        operator_intermediate_, intermediate_residual_, state_,
        kOperatorIntermediateDotIntermediate, n);
    program->vulkan_sparse_bicgstab_scalar(
        state_, kPrepareOmega, static_cast<std::uint32_t>(iteration),
        absolute_tolerance_, relative_tolerance_);
    program->vulkan_sparse_bicgstab_omega_vectors(
        state_, intermediate_residual_, operator_intermediate_,
        true_residual_, candidate_solution_, mutable_x, residual_,
        shadow_residual_, n);

    apply_operator(program, x, *operator_intermediate_);
    program->copy_ndarray_fast(true_residual_, mutable_b);
    program->vulkan_sparse_axpy(
        operator_intermediate_, true_residual_, n, -1.0f);
    program->vulkan_sparse_dot_to_state_slot(
        true_residual_, true_residual_, state_, kTrueResidualSquared, n);
    program->vulkan_sparse_dot_to_state_slot(
        mutable_x, mutable_x, state_, kSolutionSquared, n);
    program->vulkan_sparse_bicgstab_scalar(
        state_, kFinalizeIteration, static_cast<std::uint32_t>(iteration),
        absolute_tolerance_, relative_tolerance_);
    program->vulkan_sparse_bicgstab_replace_residual(
        state_, true_residual_, residual_, n);
  }

  apply_operator(program, x, *operator_intermediate_);
  program->copy_ndarray_fast(true_residual_, mutable_b);
  program->vulkan_sparse_axpy(
      operator_intermediate_, true_residual_, n, -1.0f);
  program->vulkan_sparse_dot_to_state_slot(
      true_residual_, true_residual_, state_, kTrueResidualSquared, n);
  program->vulkan_sparse_dot_to_state_slot(
      mutable_x, mutable_x, state_, kSolutionSquared, n);
  program->vulkan_sparse_bicgstab_scalar(
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
      7 + 14 * static_cast<std::uint64_t>(max_iterations_);
  host_scalar_readbacks_ += kStateWordCount;
  host_synchronizations_ += 1;
  device_to_device_bytes_ +=
      static_cast<std::uint64_t>(2 * max_iterations_ + 2) *
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
VulkanSparseBiCGSTABPlan::debug_runtime_statistics() const {
  std::lock_guard<std::mutex> lock(solve_mutex_);
  const auto operator_stats = matrix_.debug_runtime_statistics();
  SparseSolvePlanRuntimeStatistics result;
  result.backend_family = "vulkan";
  result.method = "bicgstab_identity_bounded_true_residual_probe";
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
  result.preconditioner_method = "identity";
  result.preconditioner_apply_calls = 0;
  result.preconditioner_apply_calls_available = true;
  result.preconditioner_ownership_scope = "none";
  result.host_scalar_reductions = 0;
  result.device_scalar_operations = device_scalar_operations_;
  result.host_scalar_readbacks = host_scalar_readbacks_;
  result.host_synchronizations = host_synchronizations_;
  result.host_synchronizations_exact = false;
  result.host_synchronization_scope = "explicit_plan_only";
  result.persistent_vector_count = 8;
  result.persistent_vector_reserved_bytes =
      8 * static_cast<std::uint64_t>(matrix_.num_rows()) * sizeof(float32);
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

void VulkanSparseBiCGSTABPlan::release_workspace() {
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
  release(candidate_solution_);
  release(true_residual_);
  release(operator_intermediate_);
  release(intermediate_residual_);
  release(operator_direction_);
  release(direction_);
  release(shadow_residual_);
  release(residual_);
#endif
}

std::unique_ptr<VulkanSparseBiCGSTABPlan>
make_vulkan_fixed_sparse_bicgstab_plan(
    Program *program,
    SparseMatrix &matrix,
    int max_iterations,
    float absolute_tolerance,
    bool verbose,
    float relative_tolerance) {
  return std::make_unique<VulkanSparseBiCGSTABPlan>(
      program, matrix, max_iterations, absolute_tolerance, verbose,
      relative_tolerance);
}

}  // namespace taichi::lang
