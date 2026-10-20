#pragma once

#include "sparse_matrix.h"

#include "taichi/ir/type.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/program/program.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#define DECLARE_EIGEN_LLT_SOLVER(dt, type, order)                    \
  typedef EigenSparseSolver<                                         \
      Eigen::Simplicial##type<Eigen::SparseMatrix<dt>, Eigen::Lower, \
                              Eigen::order##Ordering<int>>,          \
      Eigen::SparseMatrix<dt>>                                       \
      EigenSparseSolver##dt##type##order;

#define DECLARE_EIGEN_LU_SOLVER(dt, type, order)                              \
  typedef EigenSparseSolver<Eigen::Sparse##type<Eigen::SparseMatrix<dt>,      \
                                                Eigen::order##Ordering<int>>, \
                            Eigen::SparseMatrix<dt>>                          \
      EigenSparseSolver##dt##type##order;

namespace taichi::lang {

// Contiguous scalar vectors in the solver's address space. The caller owns
// their storage through completion; no Program or Ndarray is needed here.
struct SparseSolverConstVectorView {
  const void *data;
  DataType dtype;
  std::size_t size;
};

struct SparseSolverVectorView {
  void *data;
  DataType dtype;
  std::size_t size;
};

class SparseSolver {
 protected:
  int rows_{0};
  int cols_{0};
  DataType dtype_{PrimitiveType::f32};
  bool is_initialized_{false};
  bool factorization_recorded_{false};
  std::uint64_t factorized_matrix_id_{0};
  std::uint64_t factorized_pattern_version_{0};
  std::uint64_t factorized_numeric_version_{0};

  void invalidate_factorization();
  void record_factorization(const SparseMatrix &sm);
  void validate_dense_vectors(const SparseSolverConstVectorView &rhs,
                              const SparseSolverVectorView &solution) const;

 public:
  virtual ~SparseSolver() = default;
  void init_solver(const int rows, const int cols, const DataType dtype) {
    rows_ = rows;
    cols_ = cols;
    dtype_ = dtype;
    is_initialized_ = true;
  }
  virtual bool compute(const SparseMatrix &sm) = 0;
  virtual void analyze_pattern(const SparseMatrix &sm) = 0;
  virtual void factorize(const SparseMatrix &sm) = 0;
  virtual bool info() = 0;
  virtual void solve_dense_vector(const SparseMatrix &matrix,
                                 const SparseSolverConstVectorView &rhs,
                                 const SparseSolverVectorView &solution);
  void validate_factorization(const SparseMatrix &sm) const;
};

template <class EigenSolver, class EigenMatrix>
class EigenSparseSolver : public SparseSolver {
 private:
  EigenSolver solver_;
  bool pattern_analyzed_{false};
  std::vector<std::pair<int, int>> analyzed_pattern_;

  void remember_pattern(const EigenMatrix &matrix);
  void require_same_pattern(const EigenMatrix &matrix) const;
  void solve_dense_values(const SparseSolverConstVectorView &rhs,
                          const SparseSolverVectorView &solution);

 public:
  ~EigenSparseSolver() override = default;
  bool compute(const SparseMatrix &sm) override;
  void analyze_pattern(const SparseMatrix &sm) override;
  void factorize(const SparseMatrix &sm) override;
  template <typename T>
  T solve(const T &b);

  template <typename T, typename V>
  void solve_rf(Program *prog,
                const SparseMatrix &sm,
                const Ndarray &b,
                const Ndarray &x);
  bool info() override;
  void solve_dense_vector(const SparseMatrix &matrix,
                          const SparseSolverConstVectorView &rhs,
                          const SparseSolverVectorView &solution) override;
};

DECLARE_EIGEN_LLT_SOLVER(float32, LLT, AMD);
DECLARE_EIGEN_LLT_SOLVER(float32, LLT, COLAMD);
DECLARE_EIGEN_LLT_SOLVER(float32, LDLT, AMD);
DECLARE_EIGEN_LLT_SOLVER(float32, LDLT, COLAMD);
DECLARE_EIGEN_LU_SOLVER(float32, LU, AMD);
DECLARE_EIGEN_LU_SOLVER(float32, LU, COLAMD);
DECLARE_EIGEN_LLT_SOLVER(float64, LLT, AMD);
DECLARE_EIGEN_LLT_SOLVER(float64, LLT, COLAMD);
DECLARE_EIGEN_LLT_SOLVER(float64, LDLT, AMD);
DECLARE_EIGEN_LLT_SOLVER(float64, LDLT, COLAMD);
DECLARE_EIGEN_LU_SOLVER(float64, LU, AMD);
DECLARE_EIGEN_LU_SOLVER(float64, LU, COLAMD);

class CuSparseSolver : public SparseSolver {
 public:
  enum class SolverType { Cholesky, LU };

 private:
  SolverType solver_type_{SolverType::Cholesky};
  std::string ordering_{"AMD"};
  csrcholInfo_t info_{nullptr};
  csrluInfoHost_t lu_info_{nullptr};
  cusolverSpHandle_t cusolver_handle_{nullptr};
  cusparseHandle_t cusparse_handel_{nullptr};
  cusparseMatDescr_t descr_{nullptr};
  void *gpu_buffer_{nullptr};
  void *cpu_buffer_{nullptr};
  std::size_t gpu_buffer_bytes_{0};
  std::size_t cpu_buffer_bytes_{0};
  bool factor_workspace_prepared_{false};
  void *permuted_rhs_{nullptr};
  void *permuted_solution_{nullptr};
  cusparseDnVecDescr_t rhs_descriptor_{nullptr};
  cusparseDnVecDescr_t solution_descriptor_{nullptr};
  cusparseSpVecDescr_t permuted_rhs_descriptor_{nullptr};
  cusparseSpVecDescr_t permuted_solution_descriptor_{nullptr};
  std::vector<float> host_rhs_, host_permuted_rhs_;
  std::vector<float> host_solution_, host_permuted_solution_;
  std::uint64_t workspace_allocations_{0};
  std::uint64_t descriptor_creations_{0};
  std::uint64_t solve_calls_{0};
  std::uint64_t rhs_readback_bytes_{0};
  bool is_analyzed_{false};
  bool is_factorized_{false};

  int *h_Q_{
      nullptr}; /* <int> n,  B = Q*A*Q' or B = A(Q,Q) by MATLAB notation */
  int *d_Q_{nullptr};
  int *h_csrRowPtrB_{nullptr}; /* <int> n+1 */
  int *h_csrColIndB_{nullptr}; /* <int> nnzA */
  float *h_csrValB_{nullptr};  /* <float> nnzA */
  int *h_mapBfromA_{nullptr};  /* <int> nnzA */
  int *d_csrRowPtrB_{nullptr}; /* <int> n+1 */
  int *d_csrColIndB_{nullptr}; /* <int> nnzA */
  float *d_csrValB_{nullptr};  /* <float> nnzA */
  std::vector<int> analyzed_csr_row_ptr_;
  std::vector<int> analyzed_csr_col_ind_;
  std::uint64_t analyzed_matrix_id_{0};
  std::uint64_t analyzed_pattern_version_{0};
  std::uint64_t analyzed_shared_pattern_id_{0};
  std::uint64_t loaded_matrix_id_{0};
  std::uint64_t loaded_numeric_version_{0};

 public:
  CuSparseSolver();
  explicit CuSparseSolver(SolverType solver_type,
                          std::string ordering = "AMD")
      : solver_type_(solver_type), ordering_(std::move(ordering)) {
    init_solver();
  }
  ~CuSparseSolver() override;
  bool compute(const SparseMatrix &sm) override;
  void analyze_pattern(const SparseMatrix &sm) override;

  void factorize(const SparseMatrix &sm) override;
  void solve_rf(Program *prog,
                const SparseMatrix &sm,
                const Ndarray &b,
                const Ndarray &x);

  bool info() override {
    return factorization_recorded_ && is_factorized_;
  }
  // Ordered on Forge's legacy default CUDA stream. Storage must remain alive
  // until that stream completes. LU includes a synchronous host solve.
  void solve_dense_vector(const SparseMatrix &matrix,
                          const SparseSolverConstVectorView &rhs,
                          const SparseSolverVectorView &solution) override;
  std::map<std::string, std::uint64_t> runtime_statistics() const;
  std::vector<int> ordering_permutation() const;

 private:
  void init_solver();
  void clear_analysis_resources();
  void reorder(const CuSparseMatrix &sm);
  void validate_analyzed_pattern(const CuSparseMatrix &sm) const;
  void refresh_factorization_values(const CuSparseMatrix &sm);
  void analyze_pattern_cholesky(const SparseMatrix &sm);
  void analyze_pattern_lu(const SparseMatrix &sm);
  void factorize_cholesky(const SparseMatrix &sm);
  void factorize_lu(const SparseMatrix &sm);
  void prepare_solve_workspace();
  void solve_cholesky(const void *rhs, void *solution);
  void solve_lu(const void *rhs, void *solution);
};

std::unique_ptr<SparseSolver> make_sparse_solver(DataType dt,
                                                 const std::string &solver_type,
                                                 const std::string &ordering);

std::unique_ptr<SparseSolver> make_cusparse_solver(
    DataType dt,
    const std::string &solver_type,
    const std::string &ordering);
}  // namespace taichi::lang
