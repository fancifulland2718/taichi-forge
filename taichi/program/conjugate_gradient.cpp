#include "conjugate_gradient.h"
#include "linear_operator.h"
#include "sparse_preconditioner.h"

#include <algorithm>
#include <functional>
#include <limits>

namespace taichi::lang {
namespace {

OperatorBinding bind_preconditioner_action(
    Program *program,
    SparseMatrix &matrix,
    SparseJacobiPreconditionerPlan &preconditioner,
    std::string provider_name);
OperatorBinding bind_preconditioner_action(
    Program *program,
    SparseMatrix &matrix,
    SparseBlockJacobiPreconditionerPlan &preconditioner,
    std::string provider_name);
OperatorBinding bind_preconditioner_action(
    Program *program,
    CompiledKernelLinearOperator &matrix,
    CompiledKernelPreconditionerPlan &preconditioner,
    std::string provider_name);
void validate_preconditioner_generation(
    const OperatorPinnedAction &operator_generation,
    const OperatorPinnedAction &preconditioner_generation);
void append_operator_plan_statistics(
    const OperatorPlan &plan,
    bool preconditioner,
    SparseSolvePlanRuntimeStatistics &statistics);

}  // namespace

CUCG::CUCG(SparseMatrix &A,
           int max_iters,
           float absolute_tolerance,
           bool verbose,
           float relative_tolerance)
    : A_(A),
      cuda_csr_operator_(dynamic_cast<CuSparseMatrix *>(&A)),
      max_iters_(max_iters),
      absolute_tolerance_(absolute_tolerance),
      relative_tolerance_(relative_tolerance),
      verbose_(verbose) {
  TI_ERROR_IF(!cuda_csr_operator_,
              "CUDA conjugate gradient currently requires a CSR matrix.");
  validate_controls();
  init_solver();
}

CUCG::CUCG(Program *program,
           SparseMatrix &A,
           SparseJacobiPreconditionerPlan &preconditioner,
           int max_iters,
           float absolute_tolerance,
           bool verbose,
           float relative_tolerance)
    : program_(program),
      A_(A),
      cuda_csr_operator_(dynamic_cast<CuSparseMatrix *>(&A)),
      preconditioner_(&preconditioner),
      max_iters_(max_iters),
      absolute_tolerance_(absolute_tolerance),
      relative_tolerance_(relative_tolerance),
      verbose_(verbose) {
  TI_ERROR_IF(!cuda_csr_operator_,
              "CUDA Jacobi-PCG currently requires a CSR matrix.");
  operator_plan_ = std::make_unique<OperatorPlan>(
      program_, make_cuda_csr_operator_binding(program_,
                                               *cuda_csr_operator_));
  validate_controls();
  preconditioner_->validate_compatible(program_, A_);
  preconditioner_plan_ = std::make_unique<OperatorPlan>(
      program_, bind_preconditioner_action(program_, A_, preconditioner,
                                           "cuda_jacobi"));
  init_solver();
}

CUCG::CUCG(Program *program,
           SparseMatrix &A,
           SparseBlockJacobiPreconditionerPlan &preconditioner,
           int max_iters,
           float absolute_tolerance,
           bool verbose,
           float relative_tolerance)
    : program_(program),
      A_(A),
      cuda_bsr_operator_(dynamic_cast<CuSparseBsrMatrix *>(&A)),
      block_preconditioner_(&preconditioner),
      max_iters_(max_iters),
      absolute_tolerance_(absolute_tolerance),
      relative_tolerance_(relative_tolerance),
      verbose_(verbose) {
  TI_ERROR_IF(!cuda_bsr_operator_,
              "CUDA block-Jacobi PCG requires an internal BSR matrix.");
  operator_plan_ = std::make_unique<OperatorPlan>(
      program_, make_cuda_bsr_operator_binding(program_,
                                               *cuda_bsr_operator_));
  validate_controls();
  block_preconditioner_->validate_compatible(program_, A_);
  preconditioner_plan_ = std::make_unique<OperatorPlan>(
      program_, bind_preconditioner_action(program_, A_, preconditioner,
                                           "cuda_block_jacobi"));
  init_solver();
}

CUCG::CUCG(Program *program,
           CompiledKernelLinearOperator &A,
           CompiledKernelPreconditionerPlan *preconditioner,
           int max_iters,
           float absolute_tolerance,
           bool verbose,
           float relative_tolerance)
    : program_(program),
      A_(A),
      compiled_kernel_operator_(&A),
      compiled_kernel_preconditioner_(preconditioner),
      max_iters_(max_iters),
      absolute_tolerance_(absolute_tolerance),
      relative_tolerance_(relative_tolerance),
      verbose_(verbose) {
  TI_ERROR_IF(!program_ || program_->compile_config().arch != Arch::cuda ||
                  A.owning_program() != program_,
              "CUDA compiled-kernel CG requires its owning CUDA Program.");
  validate_controls();
  if (compiled_kernel_preconditioner_) {
    compiled_kernel_preconditioner_->validate_compatible(program_, A);
  }
  operator_plan_ = std::make_unique<OperatorPlan>(
      program_, make_cuda_program_kernel_operator_binding(program_, A));
  if (compiled_kernel_preconditioner_) {
    preconditioner_plan_ = std::make_unique<OperatorPlan>(
        program_, bind_preconditioner_action(
                      program_, A, *compiled_kernel_preconditioner_,
                      "cuda_compiled_inverse_apply"));
  }
  init_solver();
}

bool CUCG::has_preconditioner() const {
  return preconditioner_ != nullptr || block_preconditioner_ != nullptr ||
         compiled_kernel_preconditioner_ != nullptr;
}

void CUCG::validate_controls() const {
  TI_ERROR_IF(max_iters_ < 0,
              "CUDA SparseCG requires non-negative max iterations.");
  TI_ERROR_IF(!std::isfinite(absolute_tolerance_) ||
                  !std::isfinite(relative_tolerance_) ||
                  absolute_tolerance_ < 0.0f ||
                  relative_tolerance_ < 0.0f ||
                  (absolute_tolerance_ == 0.0f &&
                   relative_tolerance_ == 0.0f),
              "CUDA SparseCG requires finite non-negative atol and rtol "
              "with at least one positive tolerance.");
}

void CUCG::validate_preconditioner(Program *program) const {
  if (preconditioner_) {
    preconditioner_->validate_compatible(program, A_);
  } else if (block_preconditioner_) {
    block_preconditioner_->validate_compatible(program, A_);
  } else if (compiled_kernel_preconditioner_) {
    compiled_kernel_preconditioner_->validate_compatible(
        program, *compiled_kernel_operator_);
  }
}

void CUCG::apply_preconditioner(Program *program,
                                const OperatorPinnedAction &generation,
                                float *input,
                                float *output,
                                const Ndarray *input_array,
                                const Ndarray *output_array) {
  TI_ERROR_IF(!preconditioner_plan_,
              "CUDA CG preconditioner plan is not initialized.");
  const auto &descriptor = preconditioner_plan_->descriptor();
  const auto input_view =
      input_array
          ? OperatorVectorView::from_ndarray(
                program, *input_array, descriptor.domain, false)
          : OperatorVectorView::from_device_pointer(
                program, reinterpret_cast<std::uintptr_t>(input),
                descriptor.domain, false);
  const auto output_view =
      output_array
          ? OperatorVectorView::from_ndarray(
                program, *output_array, descriptor.range, true)
          : OperatorVectorView::from_device_pointer(
                program, reinterpret_cast<std::uintptr_t>(output),
                descriptor.range, true);
  preconditioner_plan_->submit(
      generation,
      {OperatorApplyMode::forward, input_view, nullptr, output_view});
}

void CUCG::apply_operator(Program *program,
                          const OperatorPinnedAction &generation,
                          std::uintptr_t input,
                          std::uintptr_t output,
                          const Ndarray *input_array,
                          const Ndarray *output_array) {
  TI_ERROR_IF(!operator_plan_,
              "CUDA CG operator plan is not initialized.");
  const auto &descriptor = operator_plan_->descriptor();
  const auto input_view =
      input_array
          ? OperatorVectorView::from_ndarray(
                program, *input_array, descriptor.domain, false)
          : OperatorVectorView::from_device_pointer(
                program, input, descriptor.domain, false);
  const auto output_view =
      output_array
          ? OperatorVectorView::from_ndarray(
                program, *output_array, descriptor.range, true)
          : OperatorVectorView::from_device_pointer(
                program, output, descriptor.range, true);
  operator_plan_->submit(
      generation,
      {OperatorApplyMode::forward, input_view, nullptr, output_view});
}

void CUCG::ensure_operator_plan(Program *program) {
  if (operator_plan_) {
    TI_ERROR_IF(program_ && program != program_,
                "CUDA CG must keep using its construction Program.");
    return;
  }
  TI_ERROR_IF(!program || program->compile_config().arch != Arch::cuda ||
                  !cuda_csr_operator_,
              "CUDA CG compatibility binding requires an active CUDA "
              "Program and CSR operator.");
  program_ = program;
  operator_plan_ = std::make_unique<OperatorPlan>(
      program_, make_cuda_csr_operator_binding(program_,
                                               *cuda_csr_operator_));
}

void CUCG::init_solver() {
#if defined(TI_WITH_CUDA)
  if (!CUBLASDriver::get_instance().is_loaded()) {
    bool load_success = CUBLASDriver::get_instance().load_cublas();
    if (!load_success) {
      TI_ERROR("Failed to load cublas library!");
    }
  }
  CUBLASDriver::get_instance().cubCreate(&handle_);
  int version;
  CUBLASDriver::get_instance().cubGetVersion(handle_, &version);
  TI_TRACE("CUBLAS version: {}\n", version);
#endif
}

CUCG::~CUCG() {
#if defined(TI_WITH_CUDA)
  release_workspace();
  if (handle_) {
    CUBLASDriver::get_instance().cubDestroy(handle_);
  }
#endif
}

void CUCG::ensure_workspace(Program *program, int size) {
#if defined(TI_WITH_CUDA)
  if (workspace_size_ == size && workspace_ax_ && workspace_r_ &&
      workspace_p_ && (!has_preconditioner() || workspace_z_)) {
    workspace_reuses_++;
    return;
  }
  release_workspace();
  if (size <= 0) {
    return;
  }
  if (compiled_kernel_operator_) {
    TI_ERROR_IF(program != program_,
                "CUDA compiled-kernel CG workspace requires its owning "
                "Program.");
    auto create_vector = [&]() {
      return program->create_ndarray(PrimitiveType::f32, {size},
                                     ExternalArrayLayout::kNull, false);
    };
    try {
      workspace_ax_ndarray_ = create_vector();
      workspace_r_ndarray_ = create_vector();
      workspace_p_ndarray_ = create_vector();
      if (has_preconditioner()) {
        workspace_z_ndarray_ = create_vector();
      }
      workspace_ax_ = reinterpret_cast<float *>(
          program->get_ndarray_data_ptr_as_int(workspace_ax_ndarray_));
      workspace_r_ = reinterpret_cast<float *>(
          program->get_ndarray_data_ptr_as_int(workspace_r_ndarray_));
      workspace_p_ = reinterpret_cast<float *>(
          program->get_ndarray_data_ptr_as_int(workspace_p_ndarray_));
      if (workspace_z_ndarray_) {
        workspace_z_ = reinterpret_cast<float *>(
            program->get_ndarray_data_ptr_as_int(workspace_z_ndarray_));
      }
    } catch (...) {
      release_workspace();
      throw;
    }
    workspace_size_ = size;
    workspace_builds_++;
    return;
  }
  CUDADriver::get_instance().malloc((void **)&workspace_ax_,
                                    sizeof(float) * size);
  CUDADriver::get_instance().malloc((void **)&workspace_r_,
                                    sizeof(float) * size);
  CUDADriver::get_instance().malloc((void **)&workspace_p_,
                                    sizeof(float) * size);
  if (has_preconditioner()) {
    CUDADriver::get_instance().malloc((void **)&workspace_z_,
                                      sizeof(float) * size);
  }
  workspace_size_ = size;
  workspace_builds_++;
#endif
}

void CUCG::release_workspace() {
#if defined(TI_WITH_CUDA)
  if (workspace_ax_ndarray_ && program_)
    program_->delete_ndarray(workspace_ax_ndarray_);
  else if (workspace_ax_)
    CUDADriver::get_instance().mem_free(workspace_ax_);
  if (workspace_r_ndarray_ && program_)
    program_->delete_ndarray(workspace_r_ndarray_);
  else if (workspace_r_)
    CUDADriver::get_instance().mem_free(workspace_r_);
  if (workspace_p_ndarray_ && program_)
    program_->delete_ndarray(workspace_p_ndarray_);
  else if (workspace_p_)
    CUDADriver::get_instance().mem_free(workspace_p_);
  if (workspace_z_ndarray_ && program_)
    program_->delete_ndarray(workspace_z_ndarray_);
  else if (workspace_z_)
    CUDADriver::get_instance().mem_free(workspace_z_);
  workspace_ax_ndarray_ = nullptr;
  workspace_r_ndarray_ = nullptr;
  workspace_p_ndarray_ = nullptr;
  workspace_z_ndarray_ = nullptr;
  workspace_ax_ = nullptr;
  workspace_r_ = nullptr;
  workspace_p_ = nullptr;
  workspace_z_ = nullptr;
  workspace_size_ = 0;
#endif
}

void CUCG::solve(Program *prog, const Ndarray &x, const Ndarray &b) {
#if defined(TI_WITH_CUDA)
  std::lock_guard<std::mutex> lock(solve_mutex_);
  TI_ERROR_IF(
      compiled_kernel_operator_ &&
          (prog != program_ || x.owning_program() != program_ ||
           b.owning_program() != program_),
      "CUDA compiled-kernel CG requires solution and RHS ndarrays owned by "
      "its construction Program.");
  ensure_operator_plan(prog);
  auto operator_generation = operator_plan_->pin();
  auto preconditioner_generation =
      preconditioner_plan_ ? preconditioner_plan_->pin()
                           : OperatorPinnedAction{};
  if (has_preconditioner()) {
    TI_ERROR_IF(prog != program_,
                "CUDA preconditioned CG must be solved by its construction "
                "Program.");
    validate_preconditioner(prog);
    validate_preconditioner_generation(operator_generation,
                                       preconditioner_generation);
  }
  const auto operator_stamp = operator_generation.resource_stamp();
  solve_calls_++;
  last_solve_pattern_version_ = operator_stamp.topology_revision;
  last_solve_numeric_version_ = operator_stamp.numeric_revision;
  status_ = SparseSolveStatus::kNotRun;
  iterations_ = 0;
  initial_residual_norm_ = 0.0;
  residual_norm_ = 0.0;
  relative_reference_norm_ = 0.0;
  effective_tolerance_ =
      static_cast<double>(absolute_tolerance_);

  size_t dX = prog->get_ndarray_data_ptr_as_int(&x);
  size_t db = prog->get_ndarray_data_ptr_as_int(&b);
  int m = A_.num_rows();

  ensure_workspace(prog, m);
  float *d_Ax = workspace_ax_;
  float *d_r = workspace_r_;
  float *d_p = workspace_p_;
  float *d_z = workspace_z_;

  // r = b
  CUDADriver::get_instance().memcpy_device_to_device((void *)d_r, (void *)db,
                                                     sizeof(float) * m);
  device_to_device_bytes_ += sizeof(float) * m;

  // Ax = A @ x
  apply_operator(prog, operator_generation, dX, size_t(d_Ax), &x,
                 workspace_ax_ndarray_);
  operator_apply_calls_++;

  // r = r - Ax = b - Ax
  float alpham1 = -1.0f;
  CUBLASDriver::get_instance().cubSaxpy(handle_, m, &alpham1, d_Ax, 1, d_r, 1);

  float r1 = 0.0f;
  CUBLASDriver::get_instance().cubSdot(handle_, m, d_r, 1, d_r, 1, &r1);
  host_scalar_reductions_++;
  initial_residual_norm_ = std::sqrt(std::max(r1, 0.0f));

  bool breakdown = !std::isfinite(r1) || r1 < 0.0f ||
                   !std::isfinite(initial_residual_norm_);
  if (!breakdown && relative_tolerance_ > 0.0f) {
    float rhs_squared_norm = 0.0f;
    const auto *rhs = reinterpret_cast<const float *>(db);
    CUBLASDriver::get_instance().cubSdot(
        handle_, m, rhs, 1, rhs, 1, &rhs_squared_norm);
    host_scalar_reductions_++;
    if (!std::isfinite(rhs_squared_norm) || rhs_squared_norm < 0.0f) {
      breakdown = true;
    } else {
      relative_reference_norm_ =
          std::sqrt(static_cast<double>(rhs_squared_norm));
      effective_tolerance_ = std::max(
          static_cast<double>(absolute_tolerance_),
          static_cast<double>(relative_tolerance_) *
              relative_reference_norm_);
      breakdown = !std::isfinite(effective_tolerance_);
    }
  }

  float alpha = 1.0f;
  float beta = 0.0f;
  float r0 = 0.0f;
  float rho = r1;
  float rho0 = 0.0f;
  float dot = 0.0f;
  const double tolerance_squared =
      effective_tolerance_ * effective_tolerance_;
  if (!breakdown && has_preconditioner() && r1 > tolerance_squared &&
      max_iters_ > 0) {
    apply_preconditioner(prog, preconditioner_generation, d_r, d_z,
                         workspace_r_ndarray_, workspace_z_ndarray_);
    preconditioner_apply_calls_++;
    CUBLASDriver::get_instance().cubSdot(handle_, m, d_r, 1, d_z, 1,
                                         &rho);
    host_scalar_reductions_++;
    breakdown = !std::isfinite(rho) || rho <= 0.0f;
  }

  while (!breakdown && r1 > tolerance_squared &&
         iterations_ < max_iters_) {
    if (iterations_ > 0) {
      beta = has_preconditioner() ? rho / rho0 : r1 / r0;
      if (has_preconditioner() && !std::isfinite(beta)) {
        breakdown = true;
        break;
      }
      // p = z + beta * p for PCG, or r + beta * p for identity CG.
      CUBLASDriver::get_instance().cubSscal(handle_, m, &beta, d_p, 1);
      CUBLASDriver::get_instance().cubSaxpy(
          handle_, m, &alpha, has_preconditioner() ? d_z : d_r, 1, d_p, 1);
    } else {
      // p = z for PCG, or r for identity CG.
      CUDADriver::get_instance().memcpy_device_to_device(
          (void *)d_p, (void *)(has_preconditioner() ? d_z : d_r),
          sizeof(float) * m);
      device_to_device_bytes_ += sizeof(float) * m;
    }

    // Ap = A @ p
    apply_operator(prog, operator_generation, size_t(d_p), size_t(d_Ax),
                   workspace_p_ndarray_, workspace_ax_ndarray_);
    operator_apply_calls_++;
    // dot = p @ Ap
    CUBLASDriver::get_instance().cubSdot(handle_, m, d_p, 1, d_Ax, 1, &dot);
    host_scalar_reductions_++;
    if (!std::isfinite(dot) || dot <= 0.0f) {
      breakdown = true;
      break;
    }
    const float numerator = has_preconditioner() ? rho : r1;
    float a = numerator / dot;
    if (has_preconditioner() && !std::isfinite(a)) {
      breakdown = true;
      break;
    }
    // x = x + a * p
    CUBLASDriver::get_instance().cubSaxpy(handle_, m, &a, d_p, 1, (float *)dX,
                                          1);
    // r = r - a * Ap
    float na = -a;
    CUBLASDriver::get_instance().cubSaxpy(handle_, m, &na, d_Ax, 1, d_r, 1);
    r0 = r1;
    if (has_preconditioner()) {
      rho0 = rho;
    }
    // r1 = r @ r
    CUBLASDriver::get_instance().cubSdot(handle_, m, d_r, 1, d_r, 1, &r1);
    host_scalar_reductions_++;
    iterations_++;
    if (has_preconditioner() && std::isfinite(r1) &&
        r1 > tolerance_squared && iterations_ < max_iters_) {
      apply_preconditioner(prog, preconditioner_generation, d_r, d_z,
                           workspace_r_ndarray_, workspace_z_ndarray_);
      preconditioner_apply_calls_++;
      CUBLASDriver::get_instance().cubSdot(handle_, m, d_r, 1, d_z, 1,
                                           &rho);
      host_scalar_reductions_++;
      if (!std::isfinite(rho) || rho <= 0.0f) {
        breakdown = true;
      }
    }
    if (verbose_)
      fmt::print("iter: {}, r1: {}\n", iterations_, r1);
  }
  residual_norm_ = std::sqrt(std::max(r1, 0.0f));
  total_iterations_ += static_cast<std::uint64_t>(iterations_);
  if (breakdown || !std::isfinite(r1) ||
      !std::isfinite(initial_residual_norm_) ||
      !std::isfinite(residual_norm_)) {
    status_ = SparseSolveStatus::kBreakdown;
  } else if (residual_norm_ <= effective_tolerance_) {
    status_ = SparseSolveStatus::kConverged;
  } else {
    status_ = SparseSolveStatus::kMaxIterations;
  }

#endif
}

SparseSolvePlanRuntimeStatistics CUCG::debug_runtime_statistics() const {
  std::lock_guard<std::mutex> lock(solve_mutex_);
  const auto operator_stats = A_.debug_runtime_statistics();
  SparseSolvePlanRuntimeStatistics result;
  result.backend_family = "cuda";
  if (compiled_kernel_operator_) {
    result.method = has_preconditioner() ? "pcg_compiled_kernel"
                                         : "cg_compiled_kernel";
    result.preconditioner_method = compiled_kernel_preconditioner_
                                       ? compiled_kernel_preconditioner_
                                             ->debug_runtime_statistics()
                                             .method
                                       : "identity";
    result.external_preconditioner = has_preconditioner();
    result.preconditioner_ownership_scope =
        has_preconditioner() ? "external_plan" : "none";
  } else if (has_preconditioner()) {
    result.method =
        block_preconditioner_ ? "pcg_block_jacobi" : "pcg_jacobi";
    result.preconditioner_method =
        block_preconditioner_ ? "block_jacobi" : "jacobi";
    result.external_preconditioner = true;
    result.preconditioner_ownership_scope = "external_plan";
  }
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
  result.preconditioner_apply_calls = preconditioner_apply_calls_;
  result.host_scalar_reductions = host_scalar_reductions_;
  result.persistent_vector_count =
      workspace_ax_ != nullptr && workspace_r_ != nullptr &&
              workspace_p_ != nullptr &&
              (!has_preconditioner() || workspace_z_ != nullptr)
          ? (has_preconditioner() ? 4 : 3)
          : 0;
  result.persistent_vector_reserved_bytes =
      result.persistent_vector_count == 0
          ? 0
          : result.persistent_vector_count *
                static_cast<std::uint64_t>(workspace_size_) * sizeof(float);
  result.cublas_handle_count = handle_ != nullptr ? 1 : 0;
  result.device_to_device_bytes = device_to_device_bytes_;
  append_operator_plan_statistics(*operator_plan_, false, result);
  if (preconditioner_plan_) {
    append_operator_plan_statistics(*preconditioner_plan_, true, result);
  }
  return result;
}

std::unique_ptr<CUCG> make_cucg_solver(SparseMatrix &A,
                                       int max_iters,
                                       float absolute_tolerance,
                                       bool verbose,
                                       float relative_tolerance) {
  return std::make_unique<CUCG>(A, max_iters, absolute_tolerance,
                                verbose, relative_tolerance);
}

std::unique_ptr<CUCG> make_cuda_jacobi_pcg_solver(
    Program *program,
    SparseMatrix &A,
    SparseJacobiPreconditionerPlan &preconditioner,
    int max_iters,
    float absolute_tolerance,
    bool verbose,
    float relative_tolerance) {
  return std::make_unique<CUCG>(program, A, preconditioner, max_iters,
                                absolute_tolerance, verbose,
                                relative_tolerance);
}

std::unique_ptr<CUCG> make_cuda_block_jacobi_pcg_solver(
    Program *program,
    SparseMatrix &A,
    SparseBlockJacobiPreconditionerPlan &preconditioner,
    int max_iters,
    float absolute_tolerance,
    bool verbose,
    float relative_tolerance) {
  return std::make_unique<CUCG>(program, A, preconditioner, max_iters,
                                absolute_tolerance, verbose,
                                relative_tolerance);
}

std::unique_ptr<CUCG> make_cuda_compiled_kernel_cg_solver(
    Program *program,
    CompiledKernelLinearOperator &A,
    int max_iters,
    float absolute_tolerance,
    bool verbose,
    float relative_tolerance) {
  return std::make_unique<CUCG>(program, A, nullptr, max_iters,
                                absolute_tolerance, verbose,
                                relative_tolerance);
}

std::unique_ptr<CUCG> make_cuda_compiled_kernel_pcg_solver(
    Program *program,
    CompiledKernelLinearOperator &A,
    CompiledKernelPreconditionerPlan &preconditioner,
    int max_iters,
    float absolute_tolerance,
    bool verbose,
    float relative_tolerance) {
  return std::make_unique<CUCG>(program, A, &preconditioner, max_iters,
                                absolute_tolerance, verbose,
                                relative_tolerance);
}

struct CpuSparseCGPlan::PreconditionerBinding {
  OperatorBinding binding;
  std::function<void(Program *)> validate;
  std::string method;
};

namespace {

OperatorDescriptor square_operator_descriptor(const SparseMatrix &matrix) {
  const OperatorSpaceDesc space{matrix.get_data_type(),
                                static_cast<std::size_t>(matrix.num_rows())};
  return {space, space};
}

OperatorBinding bind_cpu_operator_compatibility(Program *program,
                                                SparseMatrix &matrix) {
  if (auto *csr = dynamic_cast<CpuSparseCsrMatrix *>(&matrix)) {
    return make_cpu_csr_operator_binding(program, *csr);
  }
  if (auto *bsr = dynamic_cast<CpuSparseBsrMatrix *>(&matrix)) {
    return make_cpu_bsr_operator_binding(program, *bsr);
  }
  if (auto *kernel =
          dynamic_cast<CompiledKernelLinearOperator *>(&matrix)) {
    return make_cpu_program_kernel_operator_binding(program, *kernel);
  }
  const auto statistics = matrix.debug_runtime_statistics();
  TI_ERROR(
      "CPU operator CG compatibility factory does not support backend '{}' "
      "with storage format '{}' (provider '{}'); no fallback was performed.",
      statistics.backend_family, statistics.storage_format,
      statistics.provider_name);
}

OperatorResourceStamp preconditioner_stamp(
    Program *program,
    const SparsePreconditionerPlanRuntimeStatistics &statistics,
    const void *identity) {
  return {
      reinterpret_cast<std::uintptr_t>(program),
      program->runtime_program_generation(),
      1,
      statistics.operator_pattern_version_current,
      statistics.operator_numeric_version_current,
      reinterpret_cast<std::uintptr_t>(identity),
  };
}

OperatorBinding bind_preconditioner_action(
    Program *program,
    SparseMatrix &matrix,
    SparseJacobiPreconditionerPlan &preconditioner,
    std::string provider_name) {
  OperatorCapabilities capabilities;
  capabilities.asynchronous_submit =
      !arch_is_cpu(program->compile_config().arch);
  auto action = OperatorAction(
      square_operator_descriptor(matrix), capabilities,
      std::move(provider_name),
      [program, &preconditioner] {
        return preconditioner_stamp(program,
                                    preconditioner.debug_runtime_statistics(),
                                    &preconditioner);
      },
      [program, &preconditioner](OperatorApplyMode mode,
                                 const OperatorVectorView &input,
                                 const OperatorVectorView &output) {
        TI_ERROR_IF(mode != OperatorApplyMode::forward,
                    "Jacobi preconditioner action supports forward apply "
                    "only.");
        const auto arch = program->compile_config().arch;
        if (arch_is_cuda(arch)) {
          preconditioner.apply_cuda_raw(program, input.data, output.data);
          return;
        }
        if (arch == Arch::vulkan) {
          TI_ERROR_IF(!input.ndarray || !output.ndarray,
                      "Vulkan Jacobi action requires ndarray views.");
          preconditioner.apply(program, *input.ndarray, *output.ndarray);
          return;
        }
        TI_ERROR_IF(!arch_is_cpu(arch),
                    "Jacobi action supports CPU, CUDA, and Vulkan only.");
        preconditioner.apply_cpu_raw(program, input.data, output.data);
      });
  return OperatorBinding(
      std::move(action),
      [&preconditioner] { return preconditioner.acquire_resource_lease(); });
}

OperatorBinding bind_preconditioner_action(
    Program *program,
    SparseMatrix &matrix,
    SparseBlockJacobiPreconditionerPlan &preconditioner,
    std::string provider_name) {
  OperatorCapabilities capabilities;
  capabilities.asynchronous_submit =
      !arch_is_cpu(program->compile_config().arch);
  auto action = OperatorAction(
      square_operator_descriptor(matrix), capabilities,
      std::move(provider_name),
      [program, &preconditioner] {
        return preconditioner_stamp(program,
                                    preconditioner.debug_runtime_statistics(),
                                    &preconditioner);
      },
      [program, &preconditioner](OperatorApplyMode mode,
                                 const OperatorVectorView &input,
                                 const OperatorVectorView &output) {
        TI_ERROR_IF(mode != OperatorApplyMode::forward,
                    "Block-Jacobi preconditioner action supports forward "
                    "apply only.");
        const auto arch = program->compile_config().arch;
        if (arch_is_cuda(arch)) {
          preconditioner.apply_cuda_raw(program, input.data, output.data);
          return;
        }
        if (arch == Arch::vulkan) {
          TI_ERROR_IF(!input.ndarray || !output.ndarray,
                      "Vulkan block-Jacobi action requires ndarray views.");
          preconditioner.apply(program, *input.ndarray, *output.ndarray);
          return;
        }
        TI_ERROR_IF(
            !arch_is_cpu(arch),
            "Block-Jacobi action supports CPU, CUDA, and Vulkan only.");
        preconditioner.apply_cpu_raw(program, input.data, output.data);
      });
  return OperatorBinding(
      std::move(action),
      [&preconditioner] { return preconditioner.acquire_resource_lease(); });
}

OperatorBinding bind_preconditioner_action(
    Program *program,
    CompiledKernelLinearOperator &matrix,
    CompiledKernelPreconditionerPlan &preconditioner,
    std::string provider_name) {
  OperatorCapabilities capabilities;
  capabilities.asynchronous_submit =
      !arch_is_cpu(program->compile_config().arch);
  auto action = OperatorAction(
      square_operator_descriptor(matrix), capabilities,
      std::move(provider_name),
      [program, &preconditioner] {
        return preconditioner_stamp(program,
                                    preconditioner.debug_runtime_statistics(),
                                    &preconditioner);
      },
      [program, &matrix, &preconditioner](
          OperatorApplyMode mode, const OperatorVectorView &input,
          const OperatorVectorView &output) {
        TI_ERROR_IF(mode != OperatorApplyMode::forward || !input.ndarray ||
                        !output.ndarray,
                    "Compiled inverse action requires forward ndarray "
                    "views.");
        preconditioner.apply(program, matrix, *input.ndarray,
                             *output.ndarray);
      });
  return OperatorBinding(
      std::move(action),
      [&preconditioner] { return preconditioner.acquire_resource_lease(); });
}

void validate_preconditioner_generation(
    const OperatorPinnedAction &operator_generation,
    const OperatorPinnedAction &preconditioner_generation) {
  const auto operator_stamp = operator_generation.resource_stamp();
  const auto preconditioner_stamp =
      preconditioner_generation.resource_stamp();
  TI_ERROR_IF(
      operator_stamp.program_identity !=
              preconditioner_stamp.program_identity ||
          operator_stamp.program_generation !=
              preconditioner_stamp.program_generation ||
          operator_stamp.topology_revision !=
              preconditioner_stamp.topology_revision ||
          operator_stamp.numeric_revision !=
              preconditioner_stamp.numeric_revision,
      "Pinned preconditioner generation does not match the pinned target "
      "operator generation.");
}

void append_operator_plan_statistics(
    const OperatorPlan &plan,
    bool preconditioner,
    SparseSolvePlanRuntimeStatistics &statistics) {
  const auto plan_statistics = plan.debug_runtime_statistics();
  if (preconditioner) {
    statistics.preconditioner_action_provider = plan.provider_name();
    statistics.preconditioner_asynchronous_submit =
        plan.capabilities().asynchronous_submit;
    statistics.preconditioner_generation_pins =
        plan_statistics.generation_pins;
    statistics.preconditioner_generation_changes =
        plan_statistics.generation_changes;
    statistics.preconditioner_numeric_generation_changes =
        plan_statistics.numeric_generation_changes;
    statistics.preconditioner_binding_generation_changes =
        plan_statistics.binding_generation_changes;
    statistics.preconditioner_plan_invalidations =
        plan_statistics.invalidations;
    return;
  }
  statistics.operator_action_provider = plan.provider_name();
  statistics.operator_asynchronous_submit =
      plan.capabilities().asynchronous_submit;
  statistics.operator_generation_pins = plan_statistics.generation_pins;
  statistics.operator_generation_changes =
      plan_statistics.generation_changes;
  statistics.operator_numeric_generation_changes =
      plan_statistics.numeric_generation_changes;
  statistics.operator_binding_generation_changes =
      plan_statistics.binding_generation_changes;
  statistics.operator_plan_invalidations = plan_statistics.invalidations;
}

}  // namespace

std::unique_ptr<CpuSparseCGPlan::PreconditionerBinding>
CpuSparseCGPlan::bind_preconditioner(
    Program *program,
    SparseMatrix &matrix,
    SparseJacobiPreconditionerPlan &preconditioner) {
  return std::make_unique<PreconditionerBinding>(PreconditionerBinding{
      bind_preconditioner_action(program, matrix, preconditioner,
                                 "cpu_jacobi"),
      [&matrix, &preconditioner](Program *candidate) {
        preconditioner.validate_compatible(candidate, matrix);
      },
      "jacobi"});
}

std::unique_ptr<CpuSparseCGPlan::PreconditionerBinding>
CpuSparseCGPlan::bind_preconditioner(
    Program *program,
    SparseMatrix &matrix,
    SparseBlockJacobiPreconditionerPlan &preconditioner) {
  return std::make_unique<PreconditionerBinding>(PreconditionerBinding{
      bind_preconditioner_action(program, matrix, preconditioner,
                                 "cpu_block_jacobi"),
      [&matrix, &preconditioner](Program *candidate) {
        preconditioner.validate_compatible(candidate, matrix);
      },
      "block_jacobi"});
}

std::unique_ptr<CpuSparseCGPlan::PreconditionerBinding>
CpuSparseCGPlan::bind_preconditioner(
    Program *program,
    CompiledKernelLinearOperator &matrix,
    CompiledKernelPreconditionerPlan &preconditioner) {
  return std::make_unique<PreconditionerBinding>(PreconditionerBinding{
      bind_preconditioner_action(program, matrix, preconditioner,
                                 "compiled_kernel_inverse_apply"),
      [&matrix, &preconditioner](Program *candidate) {
        preconditioner.validate_compatible(candidate, matrix);
      },
      "compiled_kernel_inverse_apply"});
}

CpuSparseCGPlan::CpuSparseCGPlan(Program *program,
                                 SparseMatrix &matrix,
                                 int max_iterations,
                                 double absolute_tolerance,
                                 double relative_tolerance)
    : CpuSparseCGPlan(program,
                      bind_cpu_operator_compatibility(program, matrix),
                      nullptr,
                      max_iterations,
                      absolute_tolerance,
                      relative_tolerance) {
}

CpuSparseCGPlan::CpuSparseCGPlan(Program *program,
                                 SparseMatrix &matrix,
                                 SparseJacobiPreconditionerPlan &preconditioner,
                                 int max_iterations,
                                 double absolute_tolerance,
                                 double relative_tolerance)
    : CpuSparseCGPlan(program,
                      bind_cpu_operator_compatibility(program, matrix),
                      bind_preconditioner(program, matrix, preconditioner),
                      max_iterations,
                      absolute_tolerance,
                      relative_tolerance) {
}

CpuSparseCGPlan::CpuSparseCGPlan(
    Program *program,
    SparseMatrix &matrix,
    SparseBlockJacobiPreconditionerPlan &preconditioner,
    int max_iterations,
    double absolute_tolerance,
    double relative_tolerance)
    : CpuSparseCGPlan(program,
                      bind_cpu_operator_compatibility(program, matrix),
                      bind_preconditioner(program, matrix, preconditioner),
                      max_iterations,
                      absolute_tolerance,
                      relative_tolerance) {
}

CpuSparseCGPlan::CpuSparseCGPlan(
    Program *program,
    CompiledKernelLinearOperator &matrix,
    CompiledKernelPreconditionerPlan &preconditioner,
    int max_iterations,
    double absolute_tolerance,
    double relative_tolerance)
    : CpuSparseCGPlan(program,
                      bind_cpu_operator_compatibility(program, matrix),
                      bind_preconditioner(program, matrix, preconditioner),
                      max_iterations,
                      absolute_tolerance,
                      relative_tolerance) {
}

CpuSparseCGPlan::CpuSparseCGPlan(
    Program *program,
    OperatorBinding operator_binding,
    std::unique_ptr<PreconditionerBinding> preconditioner,
    int max_iterations,
    double absolute_tolerance,
    double relative_tolerance)
    : program_(program),
      preconditioner_binding_(std::move(preconditioner)),
      max_iterations_(max_iterations),
      absolute_tolerance_(absolute_tolerance),
      relative_tolerance_(relative_tolerance) {
  TI_ERROR_IF(!program_ || !arch_is_cpu(program_->compile_config().arch),
              "CPU operator CG/PCG requires an active CPU Program.");
  const auto &descriptor = operator_binding.action().descriptor();
  TI_ERROR_IF(descriptor.domain.scalar_type !=
                      descriptor.range.scalar_type ||
                  descriptor.domain.scalar_extent !=
                      descriptor.range.scalar_extent ||
                  descriptor.range.scalar_extent >
                      static_cast<std::size_t>(
                          std::numeric_limits<int>::max()),
              "CPU operator CG/PCG requires a non-empty square operator.");
  dtype_ = descriptor.range.scalar_type;
  rows_ = static_cast<int>(descriptor.range.scalar_extent);
  cols_ = static_cast<int>(descriptor.domain.scalar_extent);
  TI_ERROR_IF(dtype_ != PrimitiveType::f32 && dtype_ != PrimitiveType::f64,
              "CPU operator CG/PCG requires f32 or f64 values.");
  TI_ERROR_IF(max_iterations_ < 0,
              "CPU operator CG/PCG requires non-negative max iterations.");
  TI_ERROR_IF(!std::isfinite(absolute_tolerance_) ||
                  !std::isfinite(relative_tolerance_) ||
                  absolute_tolerance_ < 0.0 || relative_tolerance_ < 0.0 ||
                  (absolute_tolerance_ == 0.0 && relative_tolerance_ == 0.0),
              "CPU operator CG/PCG requires finite non-negative atol and rtol "
              "with at least one positive tolerance.");
  operator_plan_ = std::make_unique<OperatorPlan>(
      program_, std::move(operator_binding));
  const auto initial_stamp = operator_plan_->resource_stamp();
  TI_ERROR_IF(
      initial_stamp.program_identity !=
          reinterpret_cast<std::uintptr_t>(program_),
      "CPU operator CG/PCG binding belongs to a different Program.");
  if (preconditioner_binding_) {
    preconditioner_plan_ = std::make_unique<OperatorPlan>(
        program_, preconditioner_binding_->binding);
  }
  validate_preconditioner(program_);
  try {
    for (auto &vector : workspace_) {
      vector = program_->create_ndarray(dtype_, {rows_},
                                        ExternalArrayLayout::kNull, false);
    }
  } catch (...) {
    release_workspace();
    throw;
  }
}

CpuSparseCGPlan::~CpuSparseCGPlan() {
  release_workspace();
}

void CpuSparseCGPlan::release_workspace() {
  if (!program_) {
    return;
  }
  for (auto &vector : workspace_) {
    if (vector) {
      program_->delete_ndarray(vector);
      vector = nullptr;
    }
  }
}

void CpuSparseCGPlan::validate_preconditioner(Program *program) const {
  if (preconditioner_binding_) {
    preconditioner_binding_->validate(program);
  }
}

void CpuSparseCGPlan::apply_operator(
    const OperatorPinnedAction &generation,
    const Ndarray &input,
    const Ndarray &output) {
  const auto &descriptor = operator_plan_->descriptor();
  operator_plan_->submit(
      generation,
      {OperatorApplyMode::forward,
       OperatorVectorView::from_ndarray(
           program_, input, descriptor.domain, false),
       nullptr,
       OperatorVectorView::from_ndarray(
           program_, output, descriptor.range, true)});
}

void CpuSparseCGPlan::apply_preconditioner(
    const OperatorPinnedAction &generation,
    const Ndarray &input,
    const Ndarray &output) {
  TI_ERROR_IF(!preconditioner_plan_,
              "Identity CG has no preconditioner action.");
  const auto &descriptor = preconditioner_plan_->descriptor();
  preconditioner_plan_->submit(
      generation,
      {OperatorApplyMode::forward,
       OperatorVectorView::from_ndarray(
           program_, input, descriptor.domain, false),
       nullptr,
       OperatorVectorView::from_ndarray(
           program_, output, descriptor.range, true)});
}

template <typename T>
void CpuSparseCGPlan::solve_typed(T *x,
                                  const T *b,
                                  const std::array<T *, 4> &workspace,
                                  const Ndarray &solution_array,
                                  const OperatorPinnedAction
                                      &operator_generation,
                                  const OperatorPinnedAction
                                      *preconditioner_generation) {
  const int rows = rows_;
  T *ax = workspace[0];
  T *residual = workspace[1];
  T *direction = workspace[2];
  T *preconditioned_residual = workspace[3];
  auto dot = [&](const T *lhs, const T *rhs) {
    double result = 0.0;
    for (int index = 0; index < rows; ++index) {
      result +=
          static_cast<double>(lhs[index]) * static_cast<double>(rhs[index]);
    }
    host_scalar_reductions_++;
    return result;
  };

  relative_reference_norm_ = 0.0;
  effective_tolerance_ = absolute_tolerance_;
  if (relative_tolerance_ > 0.0) {
    const double rhs_squared_norm = dot(b, b);
    if (!std::isfinite(rhs_squared_norm) || rhs_squared_norm < 0.0) {
      status_ = SparseSolveStatus::kBreakdown;
      return;
    }
    relative_reference_norm_ = std::sqrt(rhs_squared_norm);
    effective_tolerance_ = std::max(
        absolute_tolerance_, relative_tolerance_ * relative_reference_norm_);
    if (!std::isfinite(effective_tolerance_)) {
      status_ = SparseSolveStatus::kBreakdown;
      return;
    }
  }

  apply_operator(operator_generation, solution_array, *workspace_[0]);
  operator_apply_calls_++;
  for (int index = 0; index < rows; ++index) {
    residual[index] = b[index] - ax[index];
  }
  double rr = dot(residual, residual);
  initial_residual_norm_ = std::isfinite(rr) && rr >= 0.0
                               ? std::sqrt(rr)
                               : std::numeric_limits<double>::quiet_NaN();
  residual_norm_ = initial_residual_norm_;
  const double tolerance_squared = effective_tolerance_ * effective_tolerance_;
  if (!std::isfinite(rr) || rr < 0.0) {
    status_ = SparseSolveStatus::kBreakdown;
    return;
  }
  if (rr <= tolerance_squared) {
    status_ = SparseSolveStatus::kConverged;
    return;
  }

  double rho = rr;
  if (preconditioner_plan_) {
    apply_preconditioner(*preconditioner_generation, *workspace_[1],
                         *workspace_[3]);
    preconditioner_apply_calls_++;
    rho = dot(residual, preconditioned_residual);
  }
  if (!std::isfinite(rho) || rho <= 0.0) {
    status_ = SparseSolveStatus::kBreakdown;
    return;
  }
  const T *initial_direction =
      preconditioner_plan_ ? preconditioned_residual : residual;
  std::copy(initial_direction, initial_direction + rows, direction);

  bool breakdown = false;
  while (iterations_ < max_iterations_ && rr > tolerance_squared) {
    apply_operator(operator_generation, *workspace_[2], *workspace_[0]);
    operator_apply_calls_++;
    const double p_ap = dot(direction, ax);
    if (!std::isfinite(p_ap) || p_ap <= 0.0) {
      breakdown = true;
      break;
    }
    const double alpha = rho / p_ap;
    if (!std::isfinite(alpha)) {
      breakdown = true;
      break;
    }
    for (int index = 0; index < rows; ++index) {
      x[index] = static_cast<T>(static_cast<double>(x[index]) +
                                alpha * static_cast<double>(direction[index]));
      residual[index] = static_cast<T>(static_cast<double>(residual[index]) -
                                       alpha * static_cast<double>(ax[index]));
    }
    rr = dot(residual, residual);
    iterations_++;
    if (!std::isfinite(rr) || rr < 0.0) {
      breakdown = true;
      break;
    }
    if (rr <= tolerance_squared || iterations_ >= max_iterations_) {
      break;
    }

    double next_rho = rr;
    if (preconditioner_plan_) {
      apply_preconditioner(*preconditioner_generation, *workspace_[1],
                           *workspace_[3]);
      preconditioner_apply_calls_++;
      next_rho = dot(residual, preconditioned_residual);
    }
    if (!std::isfinite(next_rho) || next_rho <= 0.0) {
      breakdown = true;
      break;
    }
    const double beta = next_rho / rho;
    if (!std::isfinite(beta)) {
      breakdown = true;
      break;
    }
    const T *updated_direction =
        preconditioner_plan_ ? preconditioned_residual : residual;
    for (int index = 0; index < rows; ++index) {
      direction[index] =
          static_cast<T>(static_cast<double>(updated_direction[index]) +
                         beta * static_cast<double>(direction[index]));
    }
    rho = next_rho;
  }

  residual_norm_ = std::isfinite(rr) && rr >= 0.0
                       ? std::sqrt(rr)
                       : std::numeric_limits<double>::quiet_NaN();
  if (breakdown || !std::isfinite(residual_norm_)) {
    status_ = SparseSolveStatus::kBreakdown;
  } else if (residual_norm_ <= effective_tolerance_) {
    status_ = SparseSolveStatus::kConverged;
  } else {
    status_ = SparseSolveStatus::kMaxIterations;
  }
}

void CpuSparseCGPlan::solve(Program *program,
                            const Ndarray &x,
                            const Ndarray &b) {
  TI_ERROR_IF(program != program_,
              "CPU operator CG/PCG must be solved by its construction "
              "Program.");
  const int rows = rows_;
  auto validate_vector = [&](const char *role, const Ndarray &array) {
    TI_ERROR_IF(array.get_element_data_type() != dtype_ ||
                    !array.get_element_shape().empty() ||
                    array.shape.size() != 1 ||
                    array.get_nelement() != static_cast<std::size_t>(rows) ||
                    array.get_element_size() != data_type_size(dtype_),
                "CPU operator CG/PCG {} must contain exactly {} scalar {} "
                "entries.",
                role, rows, data_type_name(dtype_));
  };
  validate_vector("solution", x);
  validate_vector("right-hand side", b);
  TI_ERROR_IF(x.owning_program() != program_ || b.owning_program() != program_,
              "CPU operator CG/PCG requires solution and RHS owned by "
              "its construction Program.");
  const auto solution = program_->get_ndarray_data_ptr_as_int(&x);
  const auto rhs = program_->get_ndarray_data_ptr_as_int(&b);
  TI_ERROR_IF(solution == rhs,
              "CPU operator CG/PCG solution and RHS must not alias.");

  std::lock_guard<std::mutex> lock(solve_mutex_);
  auto operator_generation = operator_plan_->pin();
  auto preconditioner_generation =
      preconditioner_plan_ ? preconditioner_plan_->pin()
                           : OperatorPinnedAction{};
  validate_preconditioner(program);
  if (preconditioner_plan_) {
    validate_preconditioner_generation(operator_generation,
                                       preconditioner_generation);
  }
  const auto operator_stamp = operator_generation.resource_stamp();
  if (has_solved_) {
    workspace_reuses_++;
  } else {
    has_solved_ = true;
  }
  solve_calls_++;
  last_solve_pattern_version_ = operator_stamp.topology_revision;
  last_solve_numeric_version_ = operator_stamp.numeric_revision;
  status_ = SparseSolveStatus::kNotRun;
  iterations_ = 0;
  initial_residual_norm_ = 0.0;
  residual_norm_ = 0.0;
  relative_reference_norm_ = 0.0;
  effective_tolerance_ = absolute_tolerance_;
  if (dtype_ == PrimitiveType::f32) {
    std::array<float32 *, 4> workspace{};
    for (int index = 0; index < 4; ++index) {
      workspace[index] = reinterpret_cast<float32 *>(
          program_->get_ndarray_data_ptr_as_int(workspace_[index]));
    }
    solve_typed(reinterpret_cast<float32 *>(solution),
                reinterpret_cast<const float32 *>(rhs), workspace, x,
                operator_generation,
                preconditioner_plan_ ? &preconditioner_generation : nullptr);
  } else {
    std::array<float64 *, 4> workspace{};
    for (int index = 0; index < 4; ++index) {
      workspace[index] = reinterpret_cast<float64 *>(
          program_->get_ndarray_data_ptr_as_int(workspace_[index]));
    }
    solve_typed(reinterpret_cast<float64 *>(solution),
                reinterpret_cast<const float64 *>(rhs), workspace, x,
                operator_generation,
                preconditioner_plan_ ? &preconditioner_generation : nullptr);
  }
  total_iterations_ += static_cast<std::uint64_t>(iterations_);
}

SparseSolvePlanRuntimeStatistics CpuSparseCGPlan::debug_runtime_statistics()
    const {
  std::lock_guard<std::mutex> lock(solve_mutex_);
  const auto operator_stamp = operator_plan_->resource_stamp();
  SparseSolvePlanRuntimeStatistics result;
  result.backend_family = "cpu";
  if (!preconditioner_binding_) {
    result.method = "cg_operator_action";
  } else if (preconditioner_binding_->method ==
             "compiled_kernel_inverse_apply") {
    result.method = "pcg_compiled_kernel";
  } else {
    result.method = "pcg_" + preconditioner_binding_->method;
  }
  result.dtype = data_type_name(dtype_);
  result.rows = rows_;
  result.cols = cols_;
  result.max_iterations = max_iterations_;
  result.absolute_tolerance = absolute_tolerance_;
  result.relative_tolerance = relative_tolerance_;
  result.last_relative_reference_norm = relative_reference_norm_;
  result.last_effective_tolerance = effective_tolerance_;
  result.operator_pattern_version = operator_stamp.topology_revision;
  result.operator_numeric_version = operator_stamp.numeric_revision;
  result.last_solve_pattern_version = last_solve_pattern_version_;
  result.last_solve_numeric_version = last_solve_numeric_version_;
  result.operator_pattern_changed_since_last_solve =
      solve_calls_ > 0 &&
      result.operator_pattern_version != result.last_solve_pattern_version;
  result.operator_numeric_changed_since_last_solve =
      solve_calls_ > 0 &&
      result.operator_numeric_version != result.last_solve_numeric_version;
  result.solve_calls = solve_calls_;
  result.total_iterations = total_iterations_;
  result.workspace_builds = workspace_builds_;
  result.workspace_reuses = workspace_reuses_;
  result.operator_apply_calls = operator_apply_calls_;
  result.operator_apply_calls_available = true;
  result.preconditioner_method =
      preconditioner_binding_ ? preconditioner_binding_->method : "identity";
  result.preconditioner_apply_calls = preconditioner_apply_calls_;
  result.host_scalar_reductions = host_scalar_reductions_;
  result.fixed_iteration_only = false;
  result.bounded_masked_execution = false;
  result.persistent_vector_count = 4;
  result.persistent_vector_reserved_bytes =
      4 * static_cast<std::uint64_t>(rows_) *
      data_type_size(dtype_);
  result.external_preconditioner = preconditioner_binding_ != nullptr;
  result.preconditioner_ownership_scope =
      preconditioner_binding_ ? "external_plan" : "none";
  result.solver_state_rebuilt_each_solve = false;
  append_operator_plan_statistics(*operator_plan_, false, result);
  if (preconditioner_plan_) {
    append_operator_plan_statistics(*preconditioner_plan_, true, result);
  }
  return result;
}

std::unique_ptr<CpuSparseCGPlan> make_cpu_operator_cg_solver(
    Program *program,
    SparseMatrix &matrix,
    int max_iterations,
    double absolute_tolerance,
    double relative_tolerance) {
  return std::make_unique<CpuSparseCGPlan>(
      program, matrix, max_iterations, absolute_tolerance, relative_tolerance);
}

std::unique_ptr<CpuSparseCGPlan> make_cpu_jacobi_pcg_solver(
    Program *program,
    SparseMatrix &matrix,
    SparseJacobiPreconditionerPlan &preconditioner,
    int max_iterations,
    double absolute_tolerance,
    double relative_tolerance) {
  return std::make_unique<CpuSparseCGPlan>(
      program, matrix, preconditioner, max_iterations, absolute_tolerance,
      relative_tolerance);
}

std::unique_ptr<CpuSparseCGPlan> make_cpu_block_jacobi_pcg_solver(
    Program *program,
    SparseMatrix &matrix,
    SparseBlockJacobiPreconditionerPlan &preconditioner,
    int max_iterations,
    double absolute_tolerance,
    double relative_tolerance) {
  return std::make_unique<CpuSparseCGPlan>(
      program, matrix, preconditioner, max_iterations,
      absolute_tolerance, relative_tolerance);
}

std::unique_ptr<CpuSparseCGPlan> make_cpu_compiled_kernel_pcg_solver(
    Program *program,
    CompiledKernelLinearOperator &matrix,
    CompiledKernelPreconditionerPlan &preconditioner,
    int max_iterations,
    double absolute_tolerance,
    double relative_tolerance) {
  return std::make_unique<CpuSparseCGPlan>(
      program, matrix, preconditioner, max_iterations,
      absolute_tolerance, relative_tolerance);
}

VulkanCGIterationPlan::VulkanCGIterationPlan(Program *program,
    SparseMatrix &matrix,
    int fixed_iterations)
    : VulkanCGIterationPlan(program, matrix, fixed_iterations, 0.0f, false,
                            false, nullptr, nullptr, nullptr) {
}

VulkanCGIterationPlan::VulkanCGIterationPlan(Program *program,
                                             SparseMatrix &matrix,
                                             int max_iterations,
    float absolute_tolerance)
    : VulkanCGIterationPlan(program, matrix, max_iterations,
                            absolute_tolerance, true, false, nullptr,
                            nullptr, nullptr) {
}

VulkanCGIterationPlan::VulkanCGIterationPlan(
    Program *program,
    SparseMatrix &matrix,
    SparseJacobiPreconditionerPlan &preconditioner,
    int max_iterations,
    float absolute_tolerance)
    : VulkanCGIterationPlan(program, matrix, max_iterations,
                            absolute_tolerance, true, false,
                            &preconditioner, nullptr, nullptr) {
}

VulkanCGIterationPlan::VulkanCGIterationPlan(
    Program *program,
    SparseMatrix &matrix,
    SparseBlockJacobiPreconditionerPlan &preconditioner,
    int max_iterations,
    float absolute_tolerance)
    : VulkanCGIterationPlan(program, matrix, max_iterations,
                            absolute_tolerance, true, false, nullptr,
                            &preconditioner, nullptr) {
}

VulkanCGIterationPlan::VulkanCGIterationPlan(
    Program *program,
    CompiledKernelLinearOperator &matrix,
    int max_iterations,
    float absolute_tolerance)
    : VulkanCGIterationPlan(program, matrix, max_iterations,
                            absolute_tolerance, true, true, nullptr, nullptr,
                            nullptr) {
}

VulkanCGIterationPlan::VulkanCGIterationPlan(
    Program *program,
    CompiledKernelLinearOperator &matrix,
    CompiledKernelPreconditionerPlan &preconditioner,
    int max_iterations,
    float absolute_tolerance)
    : VulkanCGIterationPlan(program, matrix, max_iterations,
                            absolute_tolerance, true, true, nullptr, nullptr,
                            &preconditioner) {
}

VulkanCGIterationPlan::VulkanCGIterationPlan(Program *program,
                                             SparseMatrix &matrix,
                                             int max_iterations,
                                             float absolute_tolerance,
                                             bool adaptive,
                                             bool allow_compiled_kernel_operator,
                                             SparseJacobiPreconditionerPlan
                                                 *preconditioner,
                                             SparseBlockJacobiPreconditionerPlan
                                                 *block_preconditioner,
                                             CompiledKernelPreconditionerPlan
                                                 *compiled_kernel_preconditioner) {
#if defined(TI_WITH_VULKAN)
  TI_ERROR_IF(!program || program->compile_config().arch != Arch::vulkan,
              "Vulkan CG iteration plans require an active Vulkan Program.");
  auto *vulkan_csr = dynamic_cast<VulkanSparseMatrix *>(&matrix);
  auto *vulkan_bsr = dynamic_cast<VulkanSparseBsrMatrix *>(&matrix);
  auto *compiled_kernel_operator =
      dynamic_cast<CompiledKernelLinearOperator *>(&matrix);
  const int preconditioner_count = (preconditioner ? 1 : 0) +
                                   (block_preconditioner ? 1 : 0) +
                                   (compiled_kernel_preconditioner ? 1 : 0);
  TI_ERROR_IF(preconditioner_count > 1,
              "Vulkan CG iteration plans accept at most one "
              "preconditioner.");
  if (allow_compiled_kernel_operator) {
    TI_ERROR_IF(!compiled_kernel_operator || preconditioner ||
                    block_preconditioner ||
                    compiled_kernel_operator->owning_program() != program,
                "Private Vulkan compiled-kernel CG requires its owning "
                "compiled-kernel operator and either identity or a "
                "compiled-kernel preconditioner.");
  } else if (block_preconditioner) {
    TI_ERROR_IF(!vulkan_bsr,
                "Vulkan block-Jacobi PCG requires an internal Vulkan BSR "
                "matrix.");
  } else {
    TI_ERROR_IF(!vulkan_csr,
                "Vulkan CG iteration plans require an internal Vulkan CSR "
                "matrix.");
  }
  TI_ERROR_IF(matrix.num_rows() <= 0 ||
                  matrix.num_rows() != matrix.num_cols(),
              "Vulkan CG iteration plans require a non-empty square matrix.");
  TI_ERROR_IF(matrix.get_data_type() != PrimitiveType::f32,
              "Vulkan CG iteration plans currently require f32 values.");
  TI_ERROR_IF(max_iterations <= 0,
              "Vulkan CG iteration plans require positive max iterations.");
  TI_ERROR_IF(adaptive &&
                  (!std::isfinite(absolute_tolerance) ||
                   absolute_tolerance <= 0.0f ||
                   !std::isfinite(absolute_tolerance * absolute_tolerance)),
              "Adaptive Vulkan CG plans require a finite positive absolute "
              "tolerance.");
  program_ = program;
  matrix_ = &matrix;
  csr_matrix_ = vulkan_csr;
  bsr_matrix_ = vulkan_bsr;
  preconditioner_ = preconditioner;
  block_preconditioner_ = block_preconditioner;
  compiled_kernel_preconditioner_ = compiled_kernel_preconditioner;
  compiled_kernel_operator_ =
      allow_compiled_kernel_operator ? compiled_kernel_operator : nullptr;
  if (has_preconditioner()) {
    validate_preconditioner(program_);
  }
  fixed_iterations_ = max_iterations;
  absolute_tolerance_ = absolute_tolerance;
  adaptive_ = adaptive;
  if (compiled_kernel_operator_) {
    operator_plan_ = std::make_unique<OperatorPlan>(
        program_, make_vulkan_program_kernel_operator_binding(
                      program_, *compiled_kernel_operator_));
  } else if (bsr_matrix_) {
    operator_plan_ = std::make_unique<OperatorPlan>(
        program_, make_vulkan_bsr_operator_binding(program_, *bsr_matrix_));
  } else {
    operator_plan_ = std::make_unique<OperatorPlan>(
        program_, make_vulkan_csr_operator_binding(program_, *csr_matrix_));
  }
  if (preconditioner_) {
    preconditioner_plan_ = std::make_unique<OperatorPlan>(
        program_, bind_preconditioner_action(
                      program_, matrix, *preconditioner_,
                      "vulkan_jacobi"));
  } else if (block_preconditioner_) {
    preconditioner_plan_ = std::make_unique<OperatorPlan>(
        program_, bind_preconditioner_action(
                      program_, matrix, *block_preconditioner_,
                      "vulkan_block_jacobi"));
  } else if (compiled_kernel_preconditioner_) {
    preconditioner_plan_ = std::make_unique<OperatorPlan>(
        program_, bind_preconditioner_action(
                      program_, *compiled_kernel_operator_,
                      *compiled_kernel_preconditioner_,
                      "vulkan_compiled_inverse_apply"));
  }
  const int n = matrix.num_rows();
  auto create_vector = [&]() {
    return program->create_ndarray(PrimitiveType::f32, {n},
                                   ExternalArrayLayout::kNull, false);
  };
  auto create_f32_scalar = [&]() {
    return program->create_ndarray(PrimitiveType::f32, {1},
                                   ExternalArrayLayout::kNull, false);
  };
  auto create_i32_scalar = [&]() {
    return program->create_ndarray(PrimitiveType::i32, {1},
                                   ExternalArrayLayout::kNull, false);
  };
  try {
    ap_ = create_vector();
    residual_ = create_vector();
    direction_ = create_vector();
    if (has_preconditioner()) {
      preconditioned_residual_ = create_vector();
    }
    initial_rr_ = create_f32_scalar();
    rr_a_ = create_f32_scalar();
    rr_b_ = create_f32_scalar();
    p_ap_ = create_f32_scalar();
    alpha_ = create_f32_scalar();
    beta_ = create_f32_scalar();
    residual_norm_scalar_ = create_f32_scalar();
    status_scalar_ = create_i32_scalar();
    zero_status_scalar_ = create_i32_scalar();
    if (adaptive_) {
      completed_iterations_scalar_ = create_i32_scalar();
    }
    const int32_t zero = 0;
    program->copy_ndarray_from_host(zero_status_scalar_, &zero,
                                    sizeof(zero));
  } catch (...) {
    release_workspace();
    throw;
  }
#else
  TI_NOT_IMPLEMENTED;
#endif
}

bool VulkanCGIterationPlan::has_preconditioner() const {
  return preconditioner_ != nullptr || block_preconditioner_ != nullptr ||
         compiled_kernel_preconditioner_ != nullptr;
}

void VulkanCGIterationPlan::validate_preconditioner(Program *program) const {
  if (preconditioner_) {
    preconditioner_->validate_compatible(program, *matrix_);
  } else if (block_preconditioner_) {
    block_preconditioner_->validate_compatible(program, *matrix_);
  } else if (compiled_kernel_preconditioner_) {
    TI_ERROR_IF(!compiled_kernel_operator_,
                "Compiled-kernel preconditioning requires its typed target "
                "binding.");
    compiled_kernel_preconditioner_->validate_compatible(
        program, *compiled_kernel_operator_);
  }
}

void VulkanCGIterationPlan::apply_preconditioner(
    Program *program,
    const OperatorPinnedAction &generation,
    const Ndarray &input,
    const Ndarray &output) {
  TI_ERROR_IF(!preconditioner_plan_,
              "Vulkan CG preconditioner plan is not initialized.");
  const auto &descriptor = preconditioner_plan_->descriptor();
  preconditioner_plan_->submit(
      generation,
      {OperatorApplyMode::forward,
       OperatorVectorView::from_ndarray(
           program, input, descriptor.domain, false),
       nullptr,
       OperatorVectorView::from_ndarray(
           program, output, descriptor.range, true)});
}

void VulkanCGIterationPlan::apply_operator(Program *program,
                                           const OperatorPinnedAction
                                               &generation,
                                           const Ndarray &input,
                                           const Ndarray &output) {
  TI_ERROR_IF(!operator_plan_,
              "Vulkan CG operator plan is not initialized.");
  const auto &descriptor = operator_plan_->descriptor();
  operator_plan_->submit(
      generation,
      {OperatorApplyMode::forward,
       OperatorVectorView::from_ndarray(
           program, input, descriptor.domain, false),
       nullptr,
       OperatorVectorView::from_ndarray(
           program, output, descriptor.range, true)});
}

VulkanCGIterationPlan::~VulkanCGIterationPlan() {
  release_workspace();
}

void VulkanCGIterationPlan::solve(Program *program,
                                  const Ndarray &x,
                                  const Ndarray &b) {
#if defined(TI_WITH_VULKAN)
  TI_ERROR_IF(program != program_,
              "Vulkan CG iteration plan requires its owning Program.");
  const int n = matrix_->num_rows();
  auto check_vector = [&](const char *role, const Ndarray &array) {
    TI_ERROR_IF(array.shape.size() != 1 ||
                    array.get_element_data_type() != PrimitiveType::f32 ||
                    !array.get_element_shape().empty() ||
                    array.get_element_size() != sizeof(float32) ||
                    array.get_nelement() != static_cast<std::size_t>(n),
                "Vulkan CG iteration plan {} must contain exactly {} "
                "scalar f32 entries.",
                role, n);
  };
  check_vector("solution", x);
  check_vector("right-hand side", b);
  TI_ERROR_IF(x.get_device_allocation() == b.get_device_allocation(),
              "Vulkan CG iteration plan solution and RHS must not alias.");

  std::lock_guard<std::mutex> lock(solve_mutex_);
  auto operator_generation = operator_plan_->pin();
  auto preconditioner_generation =
      preconditioner_plan_ ? preconditioner_plan_->pin()
                           : OperatorPinnedAction{};
  if (has_preconditioner()) {
    validate_preconditioner(program);
    validate_preconditioner_generation(operator_generation,
                                       preconditioner_generation);
  }
  auto submission_guard =
      program->acquire_runtime_resource_submission_guard();
  const Ndarray *resources[] = {
      &x,          &b,          ap_,          residual_,
      direction_,  preconditioned_residual_,  initial_rr_, rr_a_, rr_b_,
      p_ap_,       alpha_,      beta_,        residual_norm_scalar_,
      status_scalar_,           zero_status_scalar_,
      completed_iterations_scalar_};
  program->retain_ndarrays_for_external_submission(
      resources, std::size(resources));
  const auto operator_stamp = operator_generation.resource_stamp();
  if (has_solved_) {
    workspace_reuses_++;
  } else {
    has_solved_ = true;
  }
  solve_calls_++;
  last_solve_pattern_version_ = operator_stamp.topology_revision;
  last_solve_numeric_version_ = operator_stamp.numeric_revision;
  is_success_ = false;
  iterations_ = 0;
  status_ = static_cast<int>(SparseSolveStatus::kMaxIterations);
  initial_residual_norm_ = 0.0;
  residual_norm_ = 0.0;

  auto *mutable_x = const_cast<Ndarray *>(&x);
  auto *mutable_b = const_cast<Ndarray *>(&b);
  program->copy_ndarray_fast(status_scalar_, zero_status_scalar_);
  if (adaptive_) {
    program->copy_ndarray_fast(completed_iterations_scalar_,
                               zero_status_scalar_);
  }
  program->copy_ndarray_fast(residual_, mutable_b);
  apply_operator(program, operator_generation, x, *ap_);
  program->vulkan_sparse_axpy(ap_, residual_, n, -1.0f);
  program->vulkan_sparse_dot(residual_, residual_, initial_rr_, n);
  if (adaptive_) {
    const float tolerance_squared =
        absolute_tolerance_ * absolute_tolerance_;
    program->vulkan_sparse_convergence(
        initial_rr_, status_scalar_, completed_iterations_scalar_,
        tolerance_squared, 0);
  }
  if (has_preconditioner()) {
    apply_preconditioner(program, preconditioner_generation, *residual_,
                         *preconditioned_residual_);
    program->vulkan_sparse_dot(residual_, preconditioned_residual_, rr_a_,
                               n);
    program->copy_ndarray_fast(direction_, preconditioned_residual_);
  } else {
    program->copy_ndarray_fast(rr_a_, initial_rr_);
    program->copy_ndarray_fast(direction_, residual_);
  }

  Ndarray *current_rr = rr_a_;
  Ndarray *next_rr = rr_b_;
  for (int iteration = 0; iteration < fixed_iterations_; ++iteration) {
    apply_operator(program, operator_generation, *direction_, *ap_);
    program->vulkan_sparse_dot(direction_, ap_, p_ap_, n);
    program->vulkan_sparse_scalar_divide(current_rr, p_ap_, alpha_,
                                         status_scalar_);
    program->vulkan_sparse_cg_update(direction_, ap_, alpha_, mutable_x,
                                     residual_, n);
    if (has_preconditioner()) {
      program->vulkan_sparse_dot(residual_, residual_,
                                 residual_norm_scalar_, n);
      if (adaptive_) {
        const float tolerance_squared =
            absolute_tolerance_ * absolute_tolerance_;
        program->vulkan_sparse_convergence(
            residual_norm_scalar_, status_scalar_,
            completed_iterations_scalar_, tolerance_squared,
            static_cast<std::uint32_t>(iteration + 1));
      }
      apply_preconditioner(program, preconditioner_generation, *residual_,
                           *preconditioned_residual_);
      program->vulkan_sparse_dot(residual_, preconditioned_residual_,
                                 next_rr, n);
    } else {
      program->vulkan_sparse_dot(residual_, residual_, next_rr, n);
      if (adaptive_) {
        const float tolerance_squared =
            absolute_tolerance_ * absolute_tolerance_;
        program->vulkan_sparse_convergence(
            next_rr, status_scalar_, completed_iterations_scalar_,
            tolerance_squared,
            static_cast<std::uint32_t>(iteration + 1));
      }
    }
    if (iteration + 1 < fixed_iterations_) {
      program->vulkan_sparse_scalar_divide(next_rr, current_rr, beta_,
                                           status_scalar_);
      program->vulkan_sparse_cg_direction(
          has_preconditioner() ? preconditioned_residual_ : residual_, beta_,
          direction_, n);
      std::swap(current_rr, next_rr);
    }
  }
  program->vulkan_sparse_norm(residual_, residual_norm_scalar_, n);

  float initial_rr_host = 0.0f;
  float residual_norm_host = 0.0f;
  int32_t status_host = 0;
  int32_t completed_iterations_host = fixed_iterations_;
  program->synchronize();
  program->copy_ndarray_to_host(initial_rr_, &initial_rr_host,
                                sizeof(initial_rr_host));
  program->copy_ndarray_to_host(residual_norm_scalar_, &residual_norm_host,
                                sizeof(residual_norm_host));
  program->copy_ndarray_to_host(status_scalar_, &status_host,
                                sizeof(status_host));
  if (adaptive_) {
    program->copy_ndarray_to_host(completed_iterations_scalar_,
                                  &completed_iterations_host,
                                  sizeof(completed_iterations_host));
  }
  initial_residual_norm_ =
      std::isfinite(initial_rr_host) && initial_rr_host >= 0.0f
          ? std::sqrt(static_cast<double>(initial_rr_host))
          : std::numeric_limits<double>::quiet_NaN();
  residual_norm_ = static_cast<double>(residual_norm_host);
  status_ = status_host;
  iterations_ = adaptive_ ? completed_iterations_host : fixed_iterations_;
  const bool finite_residuals = std::isfinite(initial_residual_norm_) &&
                                std::isfinite(residual_norm_);
  is_success_ =
      adaptive_
          ? status_ == static_cast<int>(SparseSolveStatus::kConverged) &&
                finite_residuals && residual_norm_ <= absolute_tolerance_
          : status_ ==
                    static_cast<int>(SparseSolveStatus::kMaxIterations) &&
                finite_residuals;
  total_iterations_ += static_cast<std::uint64_t>(iterations_);
  operator_apply_calls_ +=
      static_cast<std::uint64_t>(fixed_iterations_ + 1);
  if (has_preconditioner()) {
    preconditioner_apply_calls_ +=
        static_cast<std::uint64_t>(fixed_iterations_ + 1);
  }
  device_scalar_operations_ += static_cast<std::uint64_t>(
      4 * fixed_iterations_ - 2 + (adaptive_ ? fixed_iterations_ + 1 : 0) +
      (has_preconditioner() ? fixed_iterations_ + 1 : 0));
  host_scalar_readbacks_ += adaptive_ ? 4 : 3;
  host_synchronizations_ += 1;
  device_to_device_bytes_ +=
      2 * static_cast<std::uint64_t>(n) * sizeof(float32) +
      (adaptive_ ? 3 : 2) * sizeof(uint32_t);
  device_to_host_bytes_ +=
      2 * sizeof(float32) + (adaptive_ ? 2 : 1) * sizeof(int32_t);
#else
  TI_NOT_IMPLEMENTED;
#endif
}

SparseSolvePlanRuntimeStatistics
VulkanCGIterationPlan::debug_runtime_statistics() const {
  std::lock_guard<std::mutex> lock(solve_mutex_);
  const auto operator_stats = matrix_->debug_runtime_statistics();
  SparseSolvePlanRuntimeStatistics result;
  result.backend_family = "vulkan";
  if (has_preconditioner()) {
    if (compiled_kernel_preconditioner_) {
      result.method =
          adaptive_ ? "pcg_compiled_kernel_bounded_masked_probe"
                    : "pcg_compiled_kernel_fixed_iteration_probe";
      result.preconditioner_method =
          compiled_kernel_preconditioner_->debug_runtime_statistics().method;
    } else if (block_preconditioner_) {
      result.method =
          adaptive_ ? "pcg_block_jacobi_bounded_masked_probe"
                    : "pcg_block_jacobi_fixed_iteration_probe";
      result.preconditioner_method = "block_jacobi";
    } else {
      result.method = adaptive_ ? "pcg_jacobi_bounded_masked_probe"
                                : "pcg_jacobi_fixed_iteration_probe";
      result.preconditioner_method = "jacobi";
    }
  } else {
    if (compiled_kernel_operator_) {
      result.method = adaptive_
                          ? "cg_compiled_kernel_bounded_masked_probe"
                          : "cg_compiled_kernel_fixed_iteration_probe";
    } else {
      result.method = adaptive_ ? "cg_bounded_masked_probe"
                                : "cg_fixed_iteration_probe";
    }
  }
  result.dtype = "f32";
  result.rows = matrix_->num_rows();
  result.cols = matrix_->num_cols();
  result.max_iterations = fixed_iterations_;
  result.absolute_tolerance = static_cast<double>(absolute_tolerance_);
  result.relative_tolerance = 0.0;
  result.last_relative_reference_norm = 0.0;
  result.last_effective_tolerance =
      static_cast<double>(absolute_tolerance_);
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
  result.preconditioner_apply_calls = preconditioner_apply_calls_;
  result.host_scalar_reductions = 0;
  result.device_scalar_operations = device_scalar_operations_;
  result.host_scalar_readbacks = host_scalar_readbacks_;
  result.host_synchronizations = host_synchronizations_;
  result.fixed_iteration_only = !adaptive_;
  result.bounded_masked_execution = adaptive_;
  result.persistent_vector_count = has_preconditioner() ? 4 : 3;
  result.persistent_vector_reserved_bytes =
      result.persistent_vector_count *
      static_cast<std::uint64_t>(matrix_->num_rows()) * sizeof(float32);
  result.persistent_scalar_count = adaptive_ ? 10 : 9;
  result.persistent_scalar_reserved_bytes =
      7 * sizeof(float32) + (adaptive_ ? 3 : 2) * sizeof(int32_t);
  result.external_preconditioner = has_preconditioner();
  result.preconditioner_ownership_scope =
      has_preconditioner() ? "external_plan" : "none";
  result.solver_state_rebuilt_each_solve = false;
  result.device_to_device_bytes = device_to_device_bytes_;
  result.device_to_host_bytes = device_to_host_bytes_;
  result.host_to_device_bytes = host_to_device_bytes_;
  append_operator_plan_statistics(*operator_plan_, false, result);
  if (preconditioner_plan_) {
    append_operator_plan_statistics(*preconditioner_plan_, true, result);
  }
  return result;
}

void VulkanCGIterationPlan::release_workspace() {
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
  release(zero_status_scalar_);
  release(completed_iterations_scalar_);
  release(status_scalar_);
  release(residual_norm_scalar_);
  release(beta_);
  release(alpha_);
  release(p_ap_);
  release(rr_b_);
  release(rr_a_);
  release(initial_rr_);
  release(preconditioned_residual_);
  release(direction_);
  release(residual_);
  release(ap_);
#endif
}

std::unique_ptr<VulkanCGIterationPlan> make_vulkan_cg_iteration_plan(
    Program *program,
    SparseMatrix &matrix,
    int fixed_iterations) {
  return std::make_unique<VulkanCGIterationPlan>(program, matrix,
                                                 fixed_iterations);
}

std::unique_ptr<VulkanCGIterationPlan> make_vulkan_cg_convergence_plan(
    Program *program,
    SparseMatrix &matrix,
    int max_iterations,
    float absolute_tolerance) {
  return std::make_unique<VulkanCGIterationPlan>(
      program, matrix, max_iterations, absolute_tolerance);
}

std::unique_ptr<VulkanCGIterationPlan>
make_vulkan_jacobi_pcg_convergence_plan(
    Program *program,
    SparseMatrix &matrix,
    SparseJacobiPreconditionerPlan &preconditioner,
    int max_iterations,
    float absolute_tolerance) {
  return std::make_unique<VulkanCGIterationPlan>(
      program, matrix, preconditioner, max_iterations,
      absolute_tolerance);
}

std::unique_ptr<VulkanCGIterationPlan>
make_vulkan_block_jacobi_pcg_convergence_plan(
    Program *program,
    SparseMatrix &matrix,
    SparseBlockJacobiPreconditionerPlan &preconditioner,
    int max_iterations,
    float absolute_tolerance) {
  return std::make_unique<VulkanCGIterationPlan>(
      program, matrix, preconditioner, max_iterations,
      absolute_tolerance);
}

std::unique_ptr<VulkanCGIterationPlan>
make_vulkan_compiled_kernel_cg_convergence_plan(
    Program *program,
    CompiledKernelLinearOperator &matrix,
    int max_iterations,
    float absolute_tolerance) {
  return std::make_unique<VulkanCGIterationPlan>(
      program, matrix, max_iterations, absolute_tolerance);
}

std::unique_ptr<VulkanCGIterationPlan>
make_vulkan_compiled_kernel_pcg_convergence_plan(
    Program *program,
    CompiledKernelLinearOperator &matrix,
    CompiledKernelPreconditionerPlan &preconditioner,
    int max_iterations,
    float absolute_tolerance) {
  return std::make_unique<VulkanCGIterationPlan>(
      program, matrix, preconditioner, max_iterations, absolute_tolerance);
}
}  // namespace taichi::lang
