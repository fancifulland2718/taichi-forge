#include "sparse_matrix.h"

#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"

#include "Eigen/IterativeLinearSolvers"

#include <cmath>
#include <mutex>

namespace taichi::lang {
template <typename EigenT, typename DT>
class CG {
 public:
  CG(SparseMatrix &A, int max_iters, DT tol, bool verbose)
      : A_(A), max_iters_(max_iters), tol_(tol), verbose_(verbose) {
    x_ = EigenT::Zero(A_.num_cols());
    b_ = EigenT::Zero(A_.num_rows());
  }

  void set_x(EigenT &x) {
    x_ = x;
  }

  void set_b(EigenT &b) {
    b_ = b;
  }

  void set_x_ndarray(Program *prog, Ndarray &x) {
    size_t dX = prog->get_ndarray_data_ptr_as_int(&x);
    x_ = Eigen::Map<EigenT>((DT *)dX, A_.num_cols());
  }

  void set_b_ndarray(Program *prog, Ndarray &b) {
    size_t db = prog->get_ndarray_data_ptr_as_int(&b);
    b_ = Eigen::Map<EigenT>((DT *)db, A_.num_rows());
  }

  void solve() {
    Eigen::ConjugateGradient<Eigen::SparseMatrix<DT>,
                             Eigen::Lower | Eigen::Upper>
        cg;
    cg.setMaxIterations(max_iters_);
    EigenSparseMatrix<Eigen::SparseMatrix<DT>> &A =
        static_cast<EigenSparseMatrix<Eigen::SparseMatrix<DT>> &>(A_);
    Eigen::SparseMatrix<DT> *A_eigen =
        (Eigen::SparseMatrix<DT> *)A.get_matrix();
    initial_residual_norm_ = ((*A_eigen) * x_ - b_).norm();
    const auto b_norm = b_.norm();
    cg.setTolerance(b_norm > 0 ? tol_ / b_norm : tol_);
    cg.compute(*A_eigen);
    x_ = cg.solveWithGuess(b_, x_);
    iterations_ = cg.iterations();
    residual_norm_ = ((*A_eigen) * x_ - b_).norm();
    if (verbose_) {
      std::cout << "#iterations:     " << iterations_ << std::endl;
      std::cout << "estimated error: " << cg.error() << std::endl;
      std::cout << "residual norm:   " << residual_norm_ << std::endl;
    }
    is_success_ = cg.info() == Eigen::Success &&
                  std::isfinite(residual_norm_) && residual_norm_ <= tol_;
  }

  EigenT &get_x() {
    return x_;
  }

  bool is_success() const {
    return is_success_;
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

 private:
  SparseMatrix &A_;
  EigenT x_;
  EigenT b_;
  int max_iters_{0};
  DT tol_{0.0f};
  bool verbose_{false};
  bool is_success_{false};
  int iterations_{0};
  double initial_residual_norm_{0.0};
  double residual_norm_{0.0};
};

template <typename EigenT, typename DT>
std::unique_ptr<CG<EigenT, DT>> make_cg_solver(SparseMatrix &A,
                                               int max_iters,
                                               DT tol,
                                               bool verbose) {
  return std::make_unique<CG<EigenT, DT>>(A, max_iters, tol, verbose);
}

class CUCG {
 public:
  CUCG(SparseMatrix &A, int max_iters, float tol, bool verbose)
      : A_(A), max_iters_(max_iters), tol_(tol), verbose_(verbose) {
    init_solver();
  }

  ~CUCG();

  void solve(Program *prog, const Ndarray &x, const Ndarray &b);

  bool is_success() const {
    return is_success_;
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

 private:
  void init_solver();
  void ensure_workspace(int size);
  void release_workspace();

  cublasHandle_t handle_{nullptr};
  SparseMatrix &A_;
  int max_iters_{0};
  float tol_{0.0f};
  bool verbose_{false};
  bool is_success_{false};
  int iterations_{0};
  double initial_residual_norm_{0.0};
  double residual_norm_{0.0};
  std::mutex solve_mutex_;
  float *workspace_ax_{nullptr};
  float *workspace_r_{nullptr};
  float *workspace_p_{nullptr};
  int workspace_size_{0};
};

std::unique_ptr<CUCG> make_cucg_solver(SparseMatrix &A,
                                       int max_iters,
                                       float tol,
                                       bool verbose);
}  // namespace taichi::lang
