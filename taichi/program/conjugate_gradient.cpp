#include "conjugate_gradient.h"

#include <algorithm>

namespace taichi::lang {
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

void CUCG::ensure_workspace(int size) {
#if defined(TI_WITH_CUDA)
  if (workspace_size_ == size && workspace_ax_ && workspace_r_ &&
      workspace_p_) {
    workspace_reuses_++;
    return;
  }
  release_workspace();
  if (size <= 0) {
    return;
  }
  CUDADriver::get_instance().malloc((void **)&workspace_ax_,
                                    sizeof(float) * size);
  CUDADriver::get_instance().malloc((void **)&workspace_r_,
                                    sizeof(float) * size);
  CUDADriver::get_instance().malloc((void **)&workspace_p_,
                                    sizeof(float) * size);
  workspace_size_ = size;
  workspace_builds_++;
#endif
}

void CUCG::release_workspace() {
#if defined(TI_WITH_CUDA)
  if (workspace_ax_)
    CUDADriver::get_instance().mem_free(workspace_ax_);
  if (workspace_r_)
    CUDADriver::get_instance().mem_free(workspace_r_);
  if (workspace_p_)
    CUDADriver::get_instance().mem_free(workspace_p_);
  workspace_ax_ = nullptr;
  workspace_r_ = nullptr;
  workspace_p_ = nullptr;
  workspace_size_ = 0;
#endif
}

void CUCG::solve(Program *prog, const Ndarray &x, const Ndarray &b) {
#if defined(TI_WITH_CUDA)
  std::lock_guard<std::mutex> lock(solve_mutex_);
  const auto operator_stats = A_.debug_runtime_statistics();
  solve_calls_++;
  last_solve_pattern_version_ = operator_stats.pattern_version;
  last_solve_numeric_version_ = operator_stats.numeric_version;
  is_success_ = false;
  iterations_ = 0;
  initial_residual_norm_ = 0.0;
  residual_norm_ = 0.0;

  CuSparseMatrix &A = static_cast<CuSparseMatrix &>(A_);
  size_t dX = prog->get_ndarray_data_ptr_as_int(&x);
  size_t db = prog->get_ndarray_data_ptr_as_int(&b);
  int m = A.num_rows();

  ensure_workspace(m);
  float *d_Ax = workspace_ax_;
  float *d_r = workspace_r_;
  float *d_p = workspace_p_;

  // r = b
  CUDADriver::get_instance().memcpy_device_to_device((void *)d_r, (void *)db,
                                                     sizeof(float) * m);
  device_to_device_bytes_ += sizeof(float) * m;

  // Ax = A @ x
  A.spmv(dX, size_t(d_Ax));
  operator_apply_calls_++;

  // r = r - Ax = b - Ax
  float alpham1 = -1.0f;
  CUBLASDriver::get_instance().cubSaxpy(handle_, m, &alpham1, d_Ax, 1, d_r, 1);

  float r1 = 0.0f;
  CUBLASDriver::get_instance().cubSdot(handle_, m, d_r, 1, d_r, 1, &r1);
  host_scalar_reductions_++;
  initial_residual_norm_ = std::sqrt(std::max(r1, 0.0f));

  float alpha = 1.0, beta = 0.0, r0 = 0.0, dot = 0.0;
  bool breakdown = false;

  while (r1 > tol_ * tol_ && iterations_ < max_iters_) {
    if (iterations_ > 0) {
      // beta = r'_{k+1} @ r_{k+1} / r'_k @ r_k
      beta = r1 / r0;
      // p = r + beta * p
      CUBLASDriver::get_instance().cubSscal(handle_, m, &beta, d_p, 1);
      CUBLASDriver::get_instance().cubSaxpy(handle_, m, &alpha, d_r, 1, d_p, 1);
    } else {
      // p = r
      CUDADriver::get_instance().memcpy_device_to_device(
          (void *)d_p, (void *)d_r, sizeof(float) * m);
      device_to_device_bytes_ += sizeof(float) * m;
    }

    // Ap = A @ p
    A.spmv(size_t(d_p), size_t(d_Ax));
    operator_apply_calls_++;
    // dot = p @ Ap
    CUBLASDriver::get_instance().cubSdot(handle_, m, d_p, 1, d_Ax, 1, &dot);
    host_scalar_reductions_++;
    if (!std::isfinite(dot) || dot <= 0.0f) {
      breakdown = true;
      break;
    }
    float a = r1 / dot;
    // x = x + a * p
    CUBLASDriver::get_instance().cubSaxpy(handle_, m, &a, d_p, 1, (float *)dX,
                                          1);
    // r = r - a * Ap
    float na = -a;
    CUBLASDriver::get_instance().cubSaxpy(handle_, m, &na, d_Ax, 1, d_r, 1);
    r0 = r1;
    // r1 = r @ r
    CUBLASDriver::get_instance().cubSdot(handle_, m, d_r, 1, d_r, 1, &r1);
    host_scalar_reductions_++;
    iterations_++;
    if (verbose_)
      fmt::print("iter: {}, r1: {}\n", iterations_, r1);
  }
  residual_norm_ = std::sqrt(std::max(r1, 0.0f));
  total_iterations_ += static_cast<std::uint64_t>(iterations_);
  is_success_ = !breakdown && std::isfinite(r1) &&
                residual_norm_ <= static_cast<double>(tol_);

#endif
}

SparseSolvePlanRuntimeStatistics CUCG::debug_runtime_statistics() const {
  std::lock_guard<std::mutex> lock(solve_mutex_);
  const auto operator_stats = A_.debug_runtime_statistics();
  SparseSolvePlanRuntimeStatistics result;
  result.backend_family = "cuda";
  result.dtype = data_type_name(A_.get_data_type());
  result.rows = A_.num_rows();
  result.cols = A_.num_cols();
  result.max_iterations = max_iters_;
  result.absolute_tolerance = static_cast<double>(tol_);
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
  result.host_scalar_reductions = host_scalar_reductions_;
  result.persistent_vector_count =
      workspace_ax_ != nullptr && workspace_r_ != nullptr &&
              workspace_p_ != nullptr
          ? 3
          : 0;
  result.persistent_vector_reserved_bytes =
      result.persistent_vector_count == 0
          ? 0
          : 3 * static_cast<std::uint64_t>(workspace_size_) * sizeof(float);
  result.cublas_handle_count = handle_ != nullptr ? 1 : 0;
  result.device_to_device_bytes = device_to_device_bytes_;
  return result;
}

std::unique_ptr<CUCG> make_cucg_solver(SparseMatrix &A,
                                       int max_iters,
                                       float tol,
                                       bool verbose) {
  return std::make_unique<CUCG>(A, max_iters, tol, verbose);
}
}  // namespace taichi::lang
