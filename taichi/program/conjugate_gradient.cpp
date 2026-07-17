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
  if (handle_) {
    CUBLASDriver::get_instance().cubDestroy(handle_);
  }
#endif
}

void CUCG::solve(Program *prog, const Ndarray &x, const Ndarray &b) {
#if defined(TI_WITH_CUDA)
  is_success_ = false;
  iterations_ = 0;
  initial_residual_norm_ = 0.0;
  residual_norm_ = 0.0;

  CuSparseMatrix &A = static_cast<CuSparseMatrix &>(A_);
  size_t dX = prog->get_ndarray_data_ptr_as_int(&x);
  size_t db = prog->get_ndarray_data_ptr_as_int(&b);
  int m = A.num_rows();

  float *d_Ax = nullptr;
  float *d_r = nullptr;
  float *d_p = nullptr;
  CUDADriver::get_instance().malloc((void **)&d_Ax, sizeof(float) * m);
  CUDADriver::get_instance().malloc((void **)&d_r, sizeof(float) * m);
  CUDADriver::get_instance().malloc((void **)&d_p, sizeof(float) * m);

  // r = b
  CUDADriver::get_instance().memcpy_device_to_device((void *)d_r, (void *)db,
                                                     sizeof(float) * m);

  // Ax = A @ x
  A.spmv(dX, size_t(d_Ax));

  // r = r - Ax = b - Ax
  float alpham1 = -1.0f;
  CUBLASDriver::get_instance().cubSaxpy(handle_, m, &alpham1, d_Ax, 1, d_r, 1);

  float r1 = 0.0f;
  CUBLASDriver::get_instance().cubSdot(handle_, m, d_r, 1, d_r, 1, &r1);
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
    }

    // Ap = A @ p
    A.spmv(size_t(d_p), size_t(d_Ax));
    // dot = p @ Ap
    CUBLASDriver::get_instance().cubSdot(handle_, m, d_p, 1, d_Ax, 1, &dot);
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
    iterations_++;
    if (verbose_)
      fmt::print("iter: {}, r1: {}\n", iterations_, r1);
  }
  residual_norm_ = std::sqrt(std::max(r1, 0.0f));
  is_success_ = !breakdown && std::isfinite(r1) &&
                residual_norm_ <= static_cast<double>(tol_);

  CUDADriver::get_instance().mem_free(d_Ax);
  CUDADriver::get_instance().mem_free(d_r);
  CUDADriver::get_instance().mem_free(d_p);
#endif
}

std::unique_ptr<CUCG> make_cucg_solver(SparseMatrix &A,
                                       int max_iters,
                                       float tol,
                                       bool verbose) {
  return std::make_unique<CUCG>(A, max_iters, tol, verbose);
}
}  // namespace taichi::lang
