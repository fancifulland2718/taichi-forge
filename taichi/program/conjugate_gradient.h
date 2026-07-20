#include "sparse_matrix.h"

#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"

#include "Eigen/IterativeLinearSolvers"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <mutex>

namespace taichi::lang {

struct SparseSolvePlanRuntimeStatistics {
  std::string backend_family{"unknown"};
  std::string method{"cg"};
  std::string dtype{"unknown"};
  int rows{0};
  int cols{0};
  int max_iterations{0};
  double absolute_tolerance{0.0};

  std::uint64_t operator_pattern_version{0};
  std::uint64_t operator_numeric_version{0};
  std::uint64_t last_solve_pattern_version{0};
  std::uint64_t last_solve_numeric_version{0};
  bool operator_pattern_changed_since_last_solve{false};
  bool operator_numeric_changed_since_last_solve{false};

  std::uint64_t solve_calls{0};
  std::uint64_t total_iterations{0};
  std::uint64_t workspace_builds{0};
  std::uint64_t workspace_reuses{0};
  std::uint64_t operator_apply_calls{0};
  bool operator_apply_calls_available{false};
  std::uint64_t host_scalar_reductions{0};

  std::uint64_t persistent_vector_count{0};
  std::uint64_t persistent_vector_reserved_bytes{0};
  std::uint64_t cublas_handle_count{0};
  bool solver_state_rebuilt_each_solve{false};
  std::uint64_t transient_solver_workspace_bytes{0};
  bool transient_solver_workspace_bytes_available{false};

  std::uint64_t device_to_device_bytes{0};
};

template <typename EigenT, typename DT>
class CG {
 public:
  using EigenMatrix = Eigen::SparseMatrix<DT>;
  using EigenSolver = Eigen::ConjugateGradient<
      EigenMatrix,
      Eigen::Lower | Eigen::Upper>;

  CG(SparseMatrix &A, int max_iters, DT tol, bool verbose)
      : A_(A), max_iters_(max_iters), tol_(tol), verbose_(verbose) {
    x_ = EigenT::Zero(A_.num_cols());
    b_ = EigenT::Zero(A_.num_rows());
    cg_.setMaxIterations(max_iters_);
  }

  void set_x(EigenT &x) {
    x_ = x;
  }

  void reset_x() {
    x_.setZero();
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
    const auto operator_stats = A_.debug_runtime_statistics();
    solve_calls_.fetch_add(1, std::memory_order_relaxed);
    last_solve_pattern_version_.store(operator_stats.pattern_version,
                                      std::memory_order_relaxed);
    last_solve_numeric_version_.store(operator_stats.numeric_version,
                                      std::memory_order_relaxed);
    EigenSparseMatrix<EigenMatrix> &A =
        static_cast<EigenSparseMatrix<EigenMatrix> &>(A_);
    EigenMatrix *A_eigen = (EigenMatrix *)A.get_matrix();
    initial_residual_norm_ = ((*A_eigen) * x_ - b_).norm();
    const auto b_norm = b_.norm();
    cg_.setTolerance(b_norm > 0 ? tol_ / b_norm : tol_);
    const bool solver_state_current =
        solver_state_initialized_ &&
        solver_state_pattern_version_ == operator_stats.pattern_version &&
        solver_state_numeric_version_ == operator_stats.numeric_version;
    if (solver_state_current) {
      workspace_reuses_.fetch_add(1, std::memory_order_relaxed);
    } else {
      cg_.compute(*A_eigen);
      workspace_builds_.fetch_add(1, std::memory_order_relaxed);
      solver_state_initialized_ = cg_.info() == Eigen::Success;
      if (solver_state_initialized_) {
        solver_state_pattern_version_ = operator_stats.pattern_version;
        solver_state_numeric_version_ = operator_stats.numeric_version;
      }
    }
    x_ = cg_.solveWithGuess(b_, x_);
    iterations_ = cg_.iterations();
    total_iterations_.fetch_add(static_cast<std::uint64_t>(iterations_),
                                std::memory_order_relaxed);
    residual_norm_ = ((*A_eigen) * x_ - b_).norm();
    if (verbose_) {
      std::cout << "#iterations:     " << iterations_ << std::endl;
      std::cout << "estimated error: " << cg_.error() << std::endl;
      std::cout << "residual norm:   " << residual_norm_ << std::endl;
    }
    is_success_ = cg_.info() == Eigen::Success &&
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

  SparseSolvePlanRuntimeStatistics debug_runtime_statistics() const {
    const auto operator_stats = A_.debug_runtime_statistics();
    SparseSolvePlanRuntimeStatistics result;
    result.backend_family = "cpu";
    result.dtype = data_type_name(A_.get_data_type());
    result.rows = A_.num_rows();
    result.cols = A_.num_cols();
    result.max_iterations = max_iters_;
    result.absolute_tolerance = static_cast<double>(tol_);
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
    result.workspace_builds =
        workspace_builds_.load(std::memory_order_relaxed);
    result.workspace_reuses =
        workspace_reuses_.load(std::memory_order_relaxed);
    result.persistent_vector_count = 2;
    result.persistent_vector_reserved_bytes =
        (static_cast<std::uint64_t>(A_.num_cols()) +
         static_cast<std::uint64_t>(A_.num_rows())) *
        sizeof(DT);
    result.solver_state_rebuilt_each_solve = false;
    return result;
  }

 private:
  SparseMatrix &A_;
  EigenSolver cg_;
  EigenT x_;
  EigenT b_;
  int max_iters_{0};
  DT tol_{0.0f};
  bool verbose_{false};
  bool is_success_{false};
  int iterations_{0};
  double initial_residual_norm_{0.0};
  double residual_norm_{0.0};
  std::atomic<std::uint64_t> solve_calls_{0};
  std::atomic<std::uint64_t> total_iterations_{0};
  std::atomic<std::uint64_t> workspace_builds_{0};
  std::atomic<std::uint64_t> workspace_reuses_{0};
  std::atomic<std::uint64_t> last_solve_pattern_version_{0};
  std::atomic<std::uint64_t> last_solve_numeric_version_{0};
  bool solver_state_initialized_{false};
  std::uint64_t solver_state_pattern_version_{0};
  std::uint64_t solver_state_numeric_version_{0};
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

  SparseSolvePlanRuntimeStatistics debug_runtime_statistics() const;

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
  mutable std::mutex solve_mutex_;
  float *workspace_ax_{nullptr};
  float *workspace_r_{nullptr};
  float *workspace_p_{nullptr};
  int workspace_size_{0};
  std::uint64_t solve_calls_{0};
  std::uint64_t total_iterations_{0};
  std::uint64_t workspace_builds_{0};
  std::uint64_t workspace_reuses_{0};
  std::uint64_t operator_apply_calls_{0};
  std::uint64_t host_scalar_reductions_{0};
  std::uint64_t device_to_device_bytes_{0};
  std::uint64_t last_solve_pattern_version_{0};
  std::uint64_t last_solve_numeric_version_{0};
};

std::unique_ptr<CUCG> make_cucg_solver(SparseMatrix &A,
                                       int max_iters,
                                       float tol,
                                       bool verbose);
}  // namespace taichi::lang
