#include "taichi/ir/type_utils.h"

#include "sparse_solver.h"

#include <cstring>
#include <limits>
#include <optional>
#include <unordered_map>

#if defined(TI_WITH_CUDA)
#include "taichi/rhi/cuda/cuda_context.h"
#endif

namespace taichi::lang {
#define EIGEN_LLT_SOLVER_INSTANTIATION(dt, type, order)              \
  template class EigenSparseSolver<                                  \
      Eigen::Simplicial##type<Eigen::SparseMatrix<dt>, Eigen::Lower, \
                              Eigen::order##Ordering<int>>,          \
      Eigen::SparseMatrix<dt>>;
#define EIGEN_LU_SOLVER_INSTANTIATION(dt, type, order)  \
  template class EigenSparseSolver<                     \
      Eigen::Sparse##type<Eigen::SparseMatrix<dt>,      \
                          Eigen::order##Ordering<int>>, \
      Eigen::SparseMatrix<dt>>;
// Explicit instantiation of EigenSparseSolver
EIGEN_LLT_SOLVER_INSTANTIATION(float32, LLT, AMD);
EIGEN_LLT_SOLVER_INSTANTIATION(float32, LLT, COLAMD);
EIGEN_LLT_SOLVER_INSTANTIATION(float32, LDLT, AMD);
EIGEN_LLT_SOLVER_INSTANTIATION(float32, LDLT, COLAMD);
EIGEN_LU_SOLVER_INSTANTIATION(float32, LU, AMD);
EIGEN_LU_SOLVER_INSTANTIATION(float32, LU, COLAMD);
EIGEN_LLT_SOLVER_INSTANTIATION(float64, LLT, AMD);
EIGEN_LLT_SOLVER_INSTANTIATION(float64, LLT, COLAMD);
EIGEN_LLT_SOLVER_INSTANTIATION(float64, LDLT, AMD);
EIGEN_LLT_SOLVER_INSTANTIATION(float64, LDLT, COLAMD);
EIGEN_LU_SOLVER_INSTANTIATION(float64, LU, AMD);
EIGEN_LU_SOLVER_INSTANTIATION(float64, LU, COLAMD);
}  // namespace taichi::lang

// Explicit instantiation of the template class EigenSparseSolver::solve
#define EIGEN_LLT_SOLVE_INSTANTIATION(dt, type, order, df)               \
  using T##dt = Eigen::VectorX##df;                                      \
  using S##dt##type##order =                                             \
      Eigen::Simplicial##type<Eigen::SparseMatrix<dt>, Eigen::Lower,     \
                              Eigen::order##Ordering<int>>;              \
  template T##dt                                                         \
  EigenSparseSolver<S##dt##type##order, Eigen::SparseMatrix<dt>>::solve( \
      const T##dt &b);
#define EIGEN_LU_SOLVE_INSTANTIATION(dt, type, order, df)                  \
  using LUT##dt = Eigen::VectorX##df;                                      \
  using LUS##dt##type##order =                                             \
      Eigen::Sparse##type<Eigen::SparseMatrix<dt>,                         \
                          Eigen::order##Ordering<int>>;                    \
  template LUT##dt                                                         \
  EigenSparseSolver<LUS##dt##type##order, Eigen::SparseMatrix<dt>>::solve( \
      const LUT##dt &b);

// Explicit instantiation of the template class EigenSparseSolver::solve_rf
#define INSTANTIATE_LLT_SOLVE_RF(dt, type, order, df)                     \
  using llt##dt##type##order =                                            \
      Eigen::Simplicial##type<Eigen::SparseMatrix<dt>, Eigen::Lower,      \
                              Eigen::order##Ordering<int>>;               \
  template void EigenSparseSolver<llt##dt##type##order,                   \
                                  Eigen::SparseMatrix<dt>>::solve_rf<df,  \
                                                                     dt>( \
      Program * prog, const SparseMatrix &sm, const Ndarray &b,           \
      const Ndarray &x);

#define INSTANTIATE_LU_SOLVE_RF(dt, type, order, df)                      \
  using lu##dt##type##order =                                             \
      Eigen::Sparse##type<Eigen::SparseMatrix<dt>,                        \
                          Eigen::order##Ordering<int>>;                   \
  template void EigenSparseSolver<lu##dt##type##order,                    \
                                  Eigen::SparseMatrix<dt>>::solve_rf<df,  \
                                                                     dt>( \
      Program * prog, const SparseMatrix &sm, const Ndarray &b,           \
      const Ndarray &x);

#define MAKE_EIGEN_SOLVER(dt, type, order) \
  std::make_unique<EigenSparseSolver##dt##type##order>()

#define MAKE_SOLVER(dt, type, order)                             \
  {{#dt, #type, #order}, []() -> std::unique_ptr<SparseSolver> { \
     return MAKE_EIGEN_SOLVER(dt, type, order);                  \
   }}

using Triplets = std::tuple<std::string, std::string, std::string>;
namespace {
struct key_hash {
  std::size_t operator()(const Triplets &k) const {
    auto h1 = std::hash<std::string>{}(std::get<0>(k));
    auto h2 = std::hash<std::string>{}(std::get<1>(k));
    auto h3 = std::hash<std::string>{}(std::get<2>(k));
    return h1 ^ h2 ^ h3;
  }
};
}  // namespace

namespace taichi::lang {

void SparseSolver::validate_dense_vectors(
    const SparseSolverConstVectorView &rhs,
    const SparseSolverVectorView &solution) const {
  TI_ERROR_IF(!factorization_recorded_,
              "SparseSolver has no successful numeric factorization.");
  TI_ERROR_IF(rhs.dtype != dtype_ || solution.dtype != dtype_,
              "SparseSolver vectors must have the factorization dtype {}.",
              data_type_name(dtype_));
  TI_ERROR_IF(rhs.size != static_cast<std::size_t>(rows_) ||
                  solution.size != static_cast<std::size_t>(cols_),
              "SparseSolver expects {} RHS and {} solution scalars.", rows_,
              cols_);
  TI_ERROR_IF(!rhs.data || !solution.data,
              "SparseSolver vectors require non-null storage.");
  const auto width = static_cast<std::size_t>(data_type_size(dtype_));
  const auto limit = std::numeric_limits<std::uintptr_t>::max();
  TI_ERROR_IF(rhs.size > limit / width || solution.size > limit / width,
              "SparseSolver vector byte extent overflows the address space.");
  const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
  const auto solution_begin = reinterpret_cast<std::uintptr_t>(solution.data);
  const auto rhs_bytes = rhs.size * width;
  const auto solution_bytes = solution.size * width;
  TI_ERROR_IF(rhs_begin % width || solution_begin % width ||
                  rhs_begin > limit - rhs_bytes ||
                  solution_begin > limit - solution_bytes,
              "SparseSolver vectors require aligned, representable storage.");
  TI_ERROR_IF(rhs_begin < solution_begin + solution_bytes &&
                  solution_begin < rhs_begin + rhs_bytes,
              "SparseSolver RHS and solution storage must not overlap.");
}

void SparseSolver::solve_dense_vector(
    const SparseMatrix &matrix,
    const SparseSolverConstVectorView &rhs,
    const SparseSolverVectorView &solution) {
  TI_ERROR("This sparse solver does not expose a raw dense-vector entry.");
}

void SparseSolver::invalidate_factorization() {
  factorization_recorded_ = false;
  factorized_matrix_id_ = 0;
  factorized_pattern_version_ = 0;
  factorized_numeric_version_ = 0;
}

void SparseSolver::record_factorization(const SparseMatrix &sm) {
  factorization_recorded_ = true;
  factorized_matrix_id_ = sm.matrix_id();
  factorized_pattern_version_ = sm.pattern_version();
  factorized_numeric_version_ = sm.numeric_version();
}

void SparseSolver::validate_factorization(const SparseMatrix &sm) const {
  TI_ERROR_IF(
      !factorization_recorded_,
      "SparseSolver has no successful numeric factorization. Call compute() "
      "or factorize() before solve().");
  TI_ERROR_IF(factorized_matrix_id_ != sm.matrix_id() ||
                  factorized_pattern_version_ != sm.pattern_version() ||
                  factorized_numeric_version_ != sm.numeric_version(),
              "SparseSolver factorization is stale because the matrix or its "
              "pattern/numeric version changed. Call factorize() again before "
              "solve().");
}

namespace {

const void *require_cpu_eigen_matrix(const SparseMatrix &matrix) {
  const auto stats = matrix.debug_runtime_statistics();
  TI_ERROR_IF(
      stats.backend_family != "cpu" || stats.provider_name != "eigen" ||
          (stats.storage_format != "csr" && stats.storage_format != "csc"),
      "CPU sparse direct solvers currently require an Eigen "
      "CSR/CSC matrix.");
  const void *provider_matrix = matrix.get_matrix();
  TI_ERROR_IF(provider_matrix == nullptr,
              "CPU sparse direct solver received an unavailable Eigen "
              "matrix provider.");
  return provider_matrix;
}

const CuSparseMatrix &require_cuda_csr_matrix(const SparseMatrix &matrix) {
  const auto *csr = dynamic_cast<const CuSparseMatrix *>(&matrix);
  TI_ERROR_IF(csr == nullptr,
              "CUDA sparse direct solvers currently require a CSR matrix.");
  return *csr;
}

}  // namespace

#define GET_EM(sm)                                                        \
  const auto em_stats = sm.debug_runtime_statistics();                    \
  const void *provider_matrix = require_cpu_eigen_matrix(sm);             \
  using RowMajorEigenMatrix =                                             \
      Eigen::SparseMatrix<typename EigenMatrix::Scalar, Eigen::RowMajor>; \
  std::optional<EigenMatrix> canonical_matrix;                            \
  const EigenMatrix *mat = nullptr;                                       \
  if (em_stats.storage_format == "csc") {                                 \
    mat = static_cast<const EigenMatrix *>(provider_matrix);              \
  } else {                                                                \
    canonical_matrix.emplace(                                             \
        *static_cast<const RowMajorEigenMatrix *>(provider_matrix));      \
    mat = &*canonical_matrix;                                             \
  }

template <class EigenSolver, class EigenMatrix>
void EigenSparseSolver<EigenSolver, EigenMatrix>::remember_pattern(
    const EigenMatrix &matrix) {
  analyzed_pattern_.clear();
  analyzed_pattern_.reserve(static_cast<std::size_t>(matrix.nonZeros()));
  for (int outer = 0; outer < matrix.outerSize(); ++outer) {
    for (typename EigenMatrix::InnerIterator it(matrix, outer); it; ++it) {
      analyzed_pattern_.emplace_back(it.row(), it.col());
    }
  }
  pattern_analyzed_ = true;
}

template <class EigenSolver, class EigenMatrix>
void EigenSparseSolver<EigenSolver, EigenMatrix>::require_same_pattern(
    const EigenMatrix &matrix) const {
  TI_ERROR_IF(!pattern_analyzed_,
              "SparseSolver factorize() requires analyze_pattern() first.");
  TI_ERROR_IF(
      static_cast<std::size_t>(matrix.nonZeros()) != analyzed_pattern_.size(),
      "SparseSolver factorize() requires the same sparse pattern that was "
      "passed to analyze_pattern(); the nonzero count changed.");
  std::size_t ordinal = 0;
  for (int outer = 0; outer < matrix.outerSize(); ++outer) {
    for (typename EigenMatrix::InnerIterator it(matrix, outer); it; ++it) {
      const auto &analyzed_index = analyzed_pattern_[ordinal];
      TI_ERROR_IF(
          analyzed_index.first != it.row() || analyzed_index.second != it.col(),
          "SparseSolver factorize() requires the same sparse pattern that "
          "was passed to analyze_pattern(); an index changed.");
      ++ordinal;
    }
  }
}

template <class EigenSolver, class EigenMatrix>
bool EigenSparseSolver<EigenSolver, EigenMatrix>::compute(
    const SparseMatrix &sm) {
  invalidate_factorization();
  SparseSolver::init_solver(sm.num_rows(), sm.num_cols(), sm.get_data_type());
  GET_EM(sm);
  remember_pattern(*mat);
  solver_.compute(*mat);
  if (solver_.info() != Eigen::Success) {
    return false;
  } else {
    record_factorization(sm);
    return true;
  }
}
template <class EigenSolver, class EigenMatrix>
void EigenSparseSolver<EigenSolver, EigenMatrix>::analyze_pattern(
    const SparseMatrix &sm) {
  invalidate_factorization();
  SparseSolver::init_solver(sm.num_rows(), sm.num_cols(), sm.get_data_type());
  GET_EM(sm);
  remember_pattern(*mat);
  solver_.analyzePattern(*mat);
}

template <class EigenSolver, class EigenMatrix>
void EigenSparseSolver<EigenSolver, EigenMatrix>::factorize(
    const SparseMatrix &sm) {
  GET_EM(sm);
  require_same_pattern(*mat);
  invalidate_factorization();
  solver_.factorize(*mat);
  if (solver_.info() == Eigen::Success) {
    record_factorization(sm);
  }
}

template <class EigenSolver, class EigenMatrix>
template <typename T>
T EigenSparseSolver<EigenSolver, EigenMatrix>::solve(const T &b) {
  T result(cols_);
  solve_dense_values({b.data(), dtype_, static_cast<std::size_t>(b.size())},
                     {result.data(), dtype_,
                      static_cast<std::size_t>(result.size())});
  return result;
}

template <class EigenSolver, class EigenMatrix>
void EigenSparseSolver<EigenSolver, EigenMatrix>::solve_dense_values(
    const SparseSolverConstVectorView &rhs,
    const SparseSolverVectorView &solution) {
  validate_dense_vectors(rhs, solution);
  using Scalar = typename EigenMatrix::Scalar;
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  Eigen::Map<Vector>(static_cast<Scalar *>(solution.data), cols_) =
      solver_.solve(Eigen::Map<const Vector>(
          static_cast<const Scalar *>(rhs.data), rows_));
}

template <class EigenSolver, class EigenMatrix>
void EigenSparseSolver<EigenSolver, EigenMatrix>::solve_dense_vector(
    const SparseMatrix &matrix,
    const SparseSolverConstVectorView &rhs,
    const SparseSolverVectorView &solution) {
  validate_factorization(matrix);
  solve_dense_values(rhs, solution);
}

EIGEN_LLT_SOLVE_INSTANTIATION(float32, LLT, AMD, f);
EIGEN_LLT_SOLVE_INSTANTIATION(float32, LLT, COLAMD, f);
EIGEN_LLT_SOLVE_INSTANTIATION(float32, LDLT, AMD, f);
EIGEN_LLT_SOLVE_INSTANTIATION(float32, LDLT, COLAMD, f);
EIGEN_LU_SOLVE_INSTANTIATION(float32, LU, AMD, f);
EIGEN_LU_SOLVE_INSTANTIATION(float32, LU, COLAMD, f);
EIGEN_LLT_SOLVE_INSTANTIATION(float64, LLT, AMD, d);
EIGEN_LLT_SOLVE_INSTANTIATION(float64, LLT, COLAMD, d);
EIGEN_LLT_SOLVE_INSTANTIATION(float64, LDLT, AMD, d);
EIGEN_LLT_SOLVE_INSTANTIATION(float64, LDLT, COLAMD, d);
EIGEN_LU_SOLVE_INSTANTIATION(float64, LU, AMD, d);
EIGEN_LU_SOLVE_INSTANTIATION(float64, LU, COLAMD, d);

template <class EigenSolver, class EigenMatrix>
bool EigenSparseSolver<EigenSolver, EigenMatrix>::info() {
  // Eigen's info() requires initialization; symbolic analysis alone also
  // does not establish a usable numeric factorization.
  return factorization_recorded_ && solver_.info() == Eigen::Success;
}

template <class EigenSolver, class EigenMatrix>
template <typename T, typename V>
void EigenSparseSolver<EigenSolver, EigenMatrix>::solve_rf(
    Program *prog,
    const SparseMatrix &sm,
    const Ndarray &b,
    const Ndarray &x) {
  TI_ERROR_IF(b.shape.size() != 1 || x.shape.size() != 1 ||
                  !b.get_element_shape().empty() ||
                  !x.get_element_shape().empty(),
              "SparseSolver expects one-dimensional scalar ndarrays.");
  size_t db = prog->get_ndarray_data_ptr_as_int(&b);
  size_t dX = prog->get_ndarray_data_ptr_as_int(&x);
  solve_dense_vector(
      sm, {reinterpret_cast<const void *>(db), b.get_element_data_type(),
           b.get_nelement()},
      {reinterpret_cast<void *>(dX), x.get_element_data_type(),
       x.get_nelement()});
}

INSTANTIATE_LLT_SOLVE_RF(float32, LLT, COLAMD, Eigen::VectorXf)
INSTANTIATE_LLT_SOLVE_RF(float32, LDLT, COLAMD, Eigen::VectorXf)
INSTANTIATE_LLT_SOLVE_RF(float32, LLT, AMD, Eigen::VectorXf)
INSTANTIATE_LLT_SOLVE_RF(float32, LDLT, AMD, Eigen::VectorXf)
INSTANTIATE_LU_SOLVE_RF(float32, LU, AMD, Eigen::VectorXf)
INSTANTIATE_LU_SOLVE_RF(float32, LU, COLAMD, Eigen::VectorXf)
INSTANTIATE_LLT_SOLVE_RF(float64, LLT, COLAMD, Eigen::VectorXd)
INSTANTIATE_LLT_SOLVE_RF(float64, LDLT, COLAMD, Eigen::VectorXd)
INSTANTIATE_LLT_SOLVE_RF(float64, LLT, AMD, Eigen::VectorXd)
INSTANTIATE_LLT_SOLVE_RF(float64, LDLT, AMD, Eigen::VectorXd)
INSTANTIATE_LU_SOLVE_RF(float64, LU, AMD, Eigen::VectorXd)
INSTANTIATE_LU_SOLVE_RF(float64, LU, COLAMD, Eigen::VectorXd)

CuSparseSolver::CuSparseSolver() {
  init_solver();
}

void CuSparseSolver::init_solver() {
#if defined(TI_WITH_CUDA)
  auto context_guard = CUDAContext::get_instance().get_guard();
  if (!CUSPARSEDriver::get_instance().is_loaded()) {
    bool load_success = CUSPARSEDriver::get_instance().load_cusparse();
    if (!load_success) {
      TI_ERROR("Failed to load cusparse library!");
    }
  }
  if (!CUSOLVERDriver::get_instance().is_loaded()) {
    bool load_success = CUSOLVERDriver::get_instance().load_cusolver();
    if (!load_success) {
      TI_ERROR("Failed to load cusolver library!");
    }
  }
  TI_ERROR_IF(solver_type_ == SolverType::Cholesky &&
                  !CUSPARSEDriver::get_instance().cpDnVecSetValues.available(),
              "CUDA SparseSolver requires cusparseDnVecSetValues.");
#endif
}

void CuSparseSolver::clear_analysis_resources() {
#if defined(TI_WITH_CUDA)
  if (rhs_descriptor_)
    CUSPARSEDriver::get_instance().cpDestroyDnVec(rhs_descriptor_);
  if (solution_descriptor_)
    CUSPARSEDriver::get_instance().cpDestroyDnVec(solution_descriptor_);
  if (permuted_rhs_descriptor_)
    CUSPARSEDriver::get_instance().cpDestroySpVec(permuted_rhs_descriptor_);
  if (permuted_solution_descriptor_)
    CUSPARSEDriver::get_instance().cpDestroySpVec(permuted_solution_descriptor_);
  if (permuted_rhs_)
    CUDADriver::get_instance().mem_free(permuted_rhs_);
  if (permuted_solution_)
    CUDADriver::get_instance().mem_free(permuted_solution_);
  if (h_Q_ != nullptr)
    free(h_Q_);
  if (h_csrRowPtrB_ != nullptr)
    free(h_csrRowPtrB_);
  if (h_csrColIndB_ != nullptr)
    free(h_csrColIndB_);
  if (h_csrValB_ != nullptr)
    free(h_csrValB_);
  if (h_mapBfromA_ != nullptr)
    free(h_mapBfromA_);
  if (cpu_buffer_ != nullptr)
    free(cpu_buffer_);
  if (info_ != nullptr)
    CUSOLVERDriver::get_instance().csSpDestroyCsrcholInfo(info_);
  if (lu_info_ != nullptr)
    CUSOLVERDriver::get_instance().csSpDestroyCsrluInfoHost(lu_info_);
  if (cusolver_handle_ != nullptr)
    CUSOLVERDriver::get_instance().csSpDestory(cusolver_handle_);
  if (cusparse_handel_ != nullptr)
    CUSPARSEDriver::get_instance().cpDestroy(cusparse_handel_);
  if (descr_ != nullptr)
    CUSPARSEDriver::get_instance().cpDestroyMatDescr(descr_);
  if (gpu_buffer_ != nullptr)
    CUDADriver::get_instance().mem_free(gpu_buffer_);
  if (d_Q_ != nullptr)
    CUDADriver::get_instance().mem_free(d_Q_);
  if (d_csrRowPtrB_ != nullptr)
    CUDADriver::get_instance().mem_free(d_csrRowPtrB_);
  if (d_csrColIndB_ != nullptr)
    CUDADriver::get_instance().mem_free(d_csrColIndB_);
  if (d_csrValB_ != nullptr)
    CUDADriver::get_instance().mem_free(d_csrValB_);
#endif
  rhs_descriptor_ = nullptr;
  solution_descriptor_ = nullptr;
  permuted_rhs_descriptor_ = nullptr;
  permuted_solution_descriptor_ = nullptr;
  permuted_rhs_ = nullptr;
  permuted_solution_ = nullptr;
  host_rhs_.clear();
  host_permuted_rhs_.clear();
  host_solution_.clear();
  host_permuted_solution_.clear();
  gpu_buffer_bytes_ = 0;
  cpu_buffer_bytes_ = 0;
  factor_workspace_prepared_ = false;
  h_Q_ = nullptr;
  h_csrRowPtrB_ = nullptr;
  h_csrColIndB_ = nullptr;
  h_csrValB_ = nullptr;
  h_mapBfromA_ = nullptr;
  cpu_buffer_ = nullptr;
  info_ = nullptr;
  lu_info_ = nullptr;
  cusolver_handle_ = nullptr;
  cusparse_handel_ = nullptr;
  descr_ = nullptr;
  gpu_buffer_ = nullptr;
  d_Q_ = nullptr;
  d_csrRowPtrB_ = nullptr;
  d_csrColIndB_ = nullptr;
  d_csrValB_ = nullptr;
  analyzed_csr_row_ptr_.clear();
  analyzed_csr_col_ind_.clear();
  analyzed_matrix_id_ = 0;
  analyzed_pattern_version_ = 0;
  analyzed_shared_pattern_id_ = 0;
  loaded_matrix_id_ = 0;
  loaded_numeric_version_ = 0;
  is_analyzed_ = false;
  is_factorized_ = false;
  invalidate_factorization();
}

void CuSparseSolver::reorder(const CuSparseMatrix &A) {
#if defined(TI_WITH_CUDA)
  size_t rowsA = A.num_rows();
  size_t colsA = A.num_cols();
  size_t nnzA = A.get_nnz();
  void *d_csrRowPtrA = A.get_row_ptr();
  void *d_csrColIndA = A.get_col_ind();
  void *d_csrValA = A.get_val_ptr();
  CUSOLVERDriver::get_instance().csSpCreate(&cusolver_handle_);
  CUSPARSEDriver::get_instance().cpCreate(&cusparse_handel_);
  CUSPARSEDriver::get_instance().cpCreateMatDescr(&descr_);
  CUSPARSEDriver::get_instance().cpSetMatType(descr_,
                                              CUSPARSE_MATRIX_TYPE_GENERAL);
  CUSPARSEDriver::get_instance().cpSetMatIndexBase(descr_,
                                                   CUSPARSE_INDEX_BASE_ZERO);
  h_Q_ = (int *)malloc(sizeof(int) * colsA);
  h_csrRowPtrB_ = (int *)malloc(sizeof(int) * (rowsA + 1));
  h_csrColIndB_ = (int *)malloc(sizeof(int) * std::max(nnzA, size_t{1}));
  h_csrValB_ = (float *)malloc(sizeof(float) * std::max(nnzA, size_t{1}));
  h_mapBfromA_ = (int *)malloc(sizeof(int) * std::max(nnzA, size_t{1}));
  TI_ERROR_IF(!h_Q_ || !h_csrRowPtrB_ || !h_csrColIndB_ || !h_csrValB_ ||
                  !h_mapBfromA_,
              "CUDA SparseSolver host pattern allocation failed.");

  analyzed_csr_row_ptr_.resize(rowsA + 1);
  analyzed_csr_col_ind_.resize(nnzA);
  std::vector<float> h_csr_val_a(nnzA);
  CUDADriver::get_instance().memcpy_device_to_host(
      analyzed_csr_row_ptr_.data(), d_csrRowPtrA, sizeof(int) * (rowsA + 1));
  CUDADriver::get_instance().memcpy_device_to_host(
      analyzed_csr_col_ind_.data(), d_csrColIndA, sizeof(int) * nnzA);
  CUDADriver::get_instance().memcpy_device_to_host(
      h_csr_val_a.data(), d_csrValA, sizeof(float) * nnzA);
  std::memcpy(h_csrRowPtrB_, analyzed_csr_row_ptr_.data(),
              sizeof(int) * (rowsA + 1));
  std::memcpy(h_csrColIndB_, analyzed_csr_col_ind_.data(), sizeof(int) * nnzA);

  if (ordering_ == "AMD") {
    CUSOLVERDriver::get_instance().csSpXcsrsymamdHost(
        cusolver_handle_, rowsA, nnzA, descr_, h_csrRowPtrB_, h_csrColIndB_,
        h_Q_);
  } else {
    // cuSOLVERSp has no COLAMD entry. Reuse Eigen's cold pattern ordering,
    // then use cuSOLVERSp for the same symmetric permutation B = A(Q,Q).
    using RowMatrix = Eigen::SparseMatrix<float, Eigen::RowMajor>;
    Eigen::Map<const RowMatrix> source(rowsA, colsA, nnzA, h_csrRowPtrB_,
                                       h_csrColIndB_, h_csr_val_a.data());
    Eigen::SparseMatrix<float> column_pattern = source;
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> permutation;
    Eigen::COLAMDOrdering<int>()(column_pattern, permutation);
    for (int old_column = 0; old_column < cols_; ++old_column) {
      h_Q_[permutation.indices()[old_column]] = old_column;
    }
  }
  CUDADriver::get_instance().malloc((void **)&d_Q_, sizeof(int) * colsA);
  CUDADriver::get_instance().memcpy_host_to_device((void *)d_Q_, (void *)h_Q_,
                                                   sizeof(int) * (colsA));
  size_t size_perm = 0;
  CUSOLVERDriver::get_instance().csSpXcsrperm_bufferSizeHost(
      cusolver_handle_, rowsA, colsA, nnzA, descr_, h_csrRowPtrB_,
      h_csrColIndB_, h_Q_, h_Q_, &size_perm);
  std::vector<char> buffer_cpu(std::max(size_perm, size_t{1}));
  for (int j = 0; j < nnzA; j++) {
    h_mapBfromA_[j] = j;
  }
  CUSOLVERDriver::get_instance().csSpXcsrpermHost(
      cusolver_handle_, rowsA, colsA, nnzA, descr_, h_csrRowPtrB_,
      h_csrColIndB_, h_Q_, h_Q_, h_mapBfromA_, buffer_cpu.data());
  // B = A( mapBfromA )
  for (int j = 0; j < nnzA; j++) {
    h_csrValB_[j] = h_csr_val_a[h_mapBfromA_[j]];
  }
  CUDADriver::get_instance().malloc((void **)&d_csrRowPtrB_,
                                    sizeof(int) * (rowsA + 1));
  CUDADriver::get_instance().malloc((void **)&d_csrColIndB_,
                                    sizeof(int) * std::max(nnzA, size_t{1}));
  CUDADriver::get_instance().malloc(
      (void **)&d_csrValB_, sizeof(float) * std::max(nnzA, size_t{1}));
  CUDADriver::get_instance().memcpy_host_to_device(
      (void *)d_csrRowPtrB_, (void *)h_csrRowPtrB_, sizeof(int) * (rowsA + 1));
  CUDADriver::get_instance().memcpy_host_to_device(
      (void *)d_csrColIndB_, (void *)h_csrColIndB_, sizeof(int) * nnzA);
  CUDADriver::get_instance().memcpy_host_to_device(
      (void *)d_csrValB_, (void *)h_csrValB_, sizeof(float) * nnzA);
  const auto stats = A.debug_runtime_statistics();
  analyzed_matrix_id_ = A.matrix_id();
  analyzed_pattern_version_ = A.pattern_version();
  analyzed_shared_pattern_id_ = stats.shared_pattern_id;
  loaded_matrix_id_ = A.matrix_id();
  loaded_numeric_version_ = A.numeric_version();
#endif
}

void CuSparseSolver::validate_analyzed_pattern(const CuSparseMatrix &A) const {
#if defined(TI_WITH_CUDA)
  TI_ERROR_IF(!is_analyzed_,
              "SparseSolver factorize() requires analyze_pattern() first.");
  const auto stats = A.debug_runtime_statistics();
  TI_ERROR_IF(A.num_rows() != rows_ || A.num_cols() != cols_ ||
                  A.get_data_type() != dtype_ ||
                  A.get_nnz() != static_cast<int>(analyzed_csr_col_ind_.size()),
              "SparseSolver factorize() requires the same sparse pattern "
              "that was passed to analyze_pattern(); shape, dtype, or "
              "nonzero count changed.");
  if (A.matrix_id() == analyzed_matrix_id_ &&
      A.pattern_version() == analyzed_pattern_version_) {
    return;
  }
  if (analyzed_shared_pattern_id_ != 0 &&
      stats.shared_pattern_id == analyzed_shared_pattern_id_) {
    return;
  }
  std::vector<int> row_ptr(analyzed_csr_row_ptr_.size());
  std::vector<int> col_ind(analyzed_csr_col_ind_.size());
  CUDADriver::get_instance().memcpy_device_to_host(
      row_ptr.data(), A.get_row_ptr(), sizeof(int) * row_ptr.size());
  CUDADriver::get_instance().memcpy_device_to_host(
      col_ind.data(), A.get_col_ind(), sizeof(int) * col_ind.size());
  TI_ERROR_IF(
      row_ptr != analyzed_csr_row_ptr_ || col_ind != analyzed_csr_col_ind_,
      "SparseSolver factorize() requires the same sparse pattern that was "
      "passed to analyze_pattern(); a CSR index changed.");
#else
  TI_NOT_IMPLEMENTED
#endif
}

void CuSparseSolver::refresh_factorization_values(const CuSparseMatrix &A) {
#if defined(TI_WITH_CUDA)
  if (loaded_matrix_id_ == A.matrix_id() &&
      loaded_numeric_version_ == A.numeric_version()) {
    return;
  }
  const auto nnz = static_cast<std::size_t>(A.get_nnz());
  std::vector<float> values(nnz);
  CUDADriver::get_instance().memcpy_device_to_host(
      values.data(), A.get_val_ptr(), sizeof(float) * nnz);
  for (std::size_t j = 0; j < nnz; ++j) {
    h_csrValB_[j] = values[h_mapBfromA_[j]];
  }
  if (solver_type_ == SolverType::Cholesky) {
    CUDADriver::get_instance().memcpy_host_to_device(d_csrValB_, h_csrValB_,
                                                     sizeof(float) * nnz);
  }
  loaded_matrix_id_ = A.matrix_id();
  loaded_numeric_version_ = A.numeric_version();
#else
  TI_NOT_IMPLEMENTED
#endif
}

// Reference:
// https://github.com/NVIDIA/cuda-samples/blob/master/Samples/4_CUDA_Libraries/cuSolverSp_LowlevelCholesky/cuSolverSp_LowlevelCholesky.cpp
void CuSparseSolver::analyze_pattern(const SparseMatrix &sm) {
#if defined(TI_WITH_CUDA)
  auto submission_guard =
      CUDAContext::get_instance().get_submission_lock_guard();
  auto context_guard = CUDAContext::get_instance().get_guard();
  require_cuda_csr_matrix(sm);
  TI_ERROR_IF(sm.get_data_type() != PrimitiveType::f32 ||
                  sm.num_rows() <= 0 || sm.num_rows() != sm.num_cols(),
              "CUDA SparseSolver requires a nonempty square f32 CSR matrix.");
  // Retire the previous factor and its reusable solve workspace before
  // replacing them. Warm solves only enqueue work on this same stream.
  CUDADriver::get_instance().stream_synchronize(nullptr);
  clear_analysis_resources();
  SparseSolver::init_solver(sm.num_rows(), sm.num_cols(), sm.get_data_type());
  try {
    switch (solver_type_) {
      case SolverType::Cholesky:
        analyze_pattern_cholesky(sm);
        break;
      case SolverType::LU:
        analyze_pattern_lu(sm);
        break;
      default:
        TI_NOT_IMPLEMENTED
    }
    prepare_solve_workspace();
  } catch (...) {
    clear_analysis_resources();
    throw;
  }
#else
  TI_NOT_IMPLEMENTED
#endif
}
void CuSparseSolver::analyze_pattern_cholesky(const SparseMatrix &sm) {
#if defined(TI_WITH_CUDA)
  // Retrive the info of the sparse matrix
  const CuSparseMatrix &A = require_cuda_csr_matrix(sm);

  // step 1: reorder the sparse matrix
  reorder(A);

  // step 2: create opaque info structure
  CUSOLVERDriver::get_instance().csSpCreateCsrcholInfo(&info_);

  // step 3: analyze chol(A) to know structure of L
  size_t rowsA = A.num_rows();
  size_t nnzA = A.get_nnz();
  CUSOLVERDriver::get_instance().csSpXcsrcholAnalysis(
      cusolver_handle_, rowsA, nnzA, descr_, d_csrRowPtrB_, d_csrColIndB_,
      info_);
  is_analyzed_ = true;
#else
  TI_NOT_IMPLEMENTED
#endif
}
void CuSparseSolver::analyze_pattern_lu(const SparseMatrix &sm) {
#if defined(TI_WITH_CUDA)
  // Retrive the info of the sparse matrix
  const CuSparseMatrix &A = require_cuda_csr_matrix(sm);

  // step 1: reorder the sparse matrix
  reorder(A);

  // step 2: create opaque info structure
  CUSOLVERDriver::get_instance().csSpCreateCsrluInfoHost(&lu_info_);

  // step 3: analyze LU(B) to know structure of Q and R, and upper bound for
  // nnz(L+U)
  size_t rowsA = A.num_rows();
  size_t nnzA = A.get_nnz();
  CUSOLVERDriver::get_instance().csSpXcsrluAnalysisHost(
      cusolver_handle_, rowsA, nnzA, descr_, h_csrRowPtrB_, h_csrColIndB_,
      lu_info_);
  is_analyzed_ = true;
#else
  TI_NOT_IMPLEMENTED
#endif
}
void CuSparseSolver::factorize(const SparseMatrix &sm) {
#if defined(TI_WITH_CUDA)
  auto submission_guard =
      CUDAContext::get_instance().get_submission_lock_guard();
  auto context_guard = CUDAContext::get_instance().get_guard();
#endif
  const CuSparseMatrix &A = require_cuda_csr_matrix(sm);
  validate_analyzed_pattern(A);
  invalidate_factorization();
  is_factorized_ = false;
#if defined(TI_WITH_CUDA)
  CUDADriver::get_instance().stream_synchronize(nullptr);
#endif
  refresh_factorization_values(A);
  switch (solver_type_) {
    case SolverType::Cholesky:
      factorize_cholesky(sm);
      break;
    case SolverType::LU:
      factorize_lu(sm);
      break;
    default:
      TI_NOT_IMPLEMENTED
  }
  if (is_factorized_) {
    record_factorization(sm);
  }
}
void CuSparseSolver::factorize_cholesky(const SparseMatrix &sm) {
#if defined(TI_WITH_CUDA)
  // Retrive the info of the sparse matrix
  const CuSparseMatrix &A = require_cuda_csr_matrix(sm);
  size_t rowsA = A.num_rows();
  size_t nnzA = A.get_nnz();

  if (!factor_workspace_prepared_) {
    size_t size_internal = 0;
    size_t size_chol = 0;
    CUSOLVERDriver::get_instance().csSpScsrcholBufferInfo(
        cusolver_handle_, rowsA, nnzA, descr_, d_csrValB_, d_csrRowPtrB_,
        d_csrColIndB_, info_, &size_internal, &size_chol);
    if (size_chol > 0) {
      CUDADriver::get_instance().malloc(&gpu_buffer_, size_chol);
      ++workspace_allocations_;
    }
    gpu_buffer_bytes_ = size_chol;
    factor_workspace_prepared_ = true;
  }

  // step 2: compute A = L*L^T
  CUSOLVERDriver::get_instance().csSpScsrcholFactor(
      cusolver_handle_, rowsA, nnzA, descr_, d_csrValB_, d_csrRowPtrB_,
      d_csrColIndB_, info_, gpu_buffer_);
  // step 3: check if the matrix is singular
  const float tol = 1.e-14;
  int singularity = 0;
  CUSOLVERDriver::get_instance().csSpScsrcholZeroPivot(cusolver_handle_, info_,
                                                       tol, &singularity);
  is_factorized_ = singularity == -1;
#else
  TI_NOT_IMPLEMENTED
#endif
}
void CuSparseSolver::factorize_lu(const SparseMatrix &sm) {
#if defined(TI_WITH_CUDA)
  // Retrive the info of the sparse matrix
  const CuSparseMatrix &A = require_cuda_csr_matrix(sm);
  size_t rowsA = A.num_rows();
  size_t nnzA = A.get_nnz();
  // BufferInfoHost also prepares vendor-owned factor storage. It is an
  // analysis-stage operation, not a repeatable numeric refactor query.
  if (!factor_workspace_prepared_) {
    size_t size_lu = 0;
    size_t buffer_size = 0;
    CUSOLVERDriver::get_instance().csSpScsrluBufferInfoHost(
        cusolver_handle_, rowsA, nnzA, descr_, h_csrValB_, h_csrRowPtrB_,
        h_csrColIndB_, lu_info_, &buffer_size, &size_lu);
    if (size_lu > 0) {
      cpu_buffer_ = malloc(size_lu);
      TI_ERROR_IF(!cpu_buffer_,
                  "CUDA SparseSolver LU workspace allocation failed.");
      ++workspace_allocations_;
    }
    cpu_buffer_bytes_ = size_lu;
    factor_workspace_prepared_ = true;
  }

  // step 5: compute Ppivot * B = L * U
  CUSOLVERDriver::get_instance().csSpScsrluFactorHost(
      cusolver_handle_, rowsA, nnzA, descr_, h_csrValB_, h_csrRowPtrB_,
      h_csrColIndB_, lu_info_, 1.0f, cpu_buffer_);

  // step 6: check singularity by tol
  int singularity = 0;
  const float tol = 1.e-6;
  CUSOLVERDriver::get_instance().csSpScsrluZeroPivotHost(
      cusolver_handle_, lu_info_, tol, &singularity);
  is_factorized_ = singularity == -1;
#else
  TI_NOT_IMPLEMENTED
#endif
}
bool CuSparseSolver::compute(const SparseMatrix &sm) {
  analyze_pattern(sm);
  factorize(sm);
  return info();
}

void CuSparseSolver::solve_rf(Program *prog,
                              const SparseMatrix &sm,
                              const Ndarray &b,
                              const Ndarray &x) {
  TI_ERROR_IF(b.shape.size() != 1 || x.shape.size() != 1 ||
                  !b.get_element_shape().empty() ||
                  !x.get_element_shape().empty(),
              "SparseSolver expects one-dimensional scalar ndarrays.");
  solve_dense_vector(
      sm, {reinterpret_cast<const void *>(
               prog->get_ndarray_data_ptr_as_int(&b)),
           b.get_element_data_type(), b.get_nelement()},
      {reinterpret_cast<void *>(prog->get_ndarray_data_ptr_as_int(&x)),
       x.get_element_data_type(), x.get_nelement()});
}

void CuSparseSolver::solve_dense_vector(
    const SparseMatrix &matrix,
    const SparseSolverConstVectorView &rhs,
    const SparseSolverVectorView &solution) {
#if defined(TI_WITH_CUDA)
  auto submission_guard =
      CUDAContext::get_instance().get_submission_lock_guard();
  auto context_guard = CUDAContext::get_instance().get_guard();
  validate_factorization(matrix);
  validate_dense_vectors(rhs, solution);
  switch (solver_type_) {
    case SolverType::Cholesky:
      solve_cholesky(rhs.data, solution.data);
      break;
    case SolverType::LU:
      solve_lu(rhs.data, solution.data);
      break;
    default:
      TI_NOT_IMPLEMENTED
  }
  ++solve_calls_;
#else
  TI_NOT_IMPLEMENTED
#endif
}

void CuSparseSolver::prepare_solve_workspace() {
#if defined(TI_WITH_CUDA)
  if (solver_type_ == SolverType::LU) {
    auto resize = [&](std::vector<float> &values, std::size_t size) {
      if (values.capacity() < size)
        ++workspace_allocations_;
      values.resize(size);
    };
    resize(host_rhs_, rows_);
    resize(host_permuted_rhs_, rows_);
    resize(host_solution_, cols_);
    resize(host_permuted_solution_, cols_);
    return;
  }
  auto &cuda = CUDADriver::get_instance();
  auto &sparse = CUSPARSEDriver::get_instance();
  cuda.malloc(&permuted_rhs_, sizeof(float) * rows_);
  ++workspace_allocations_;
  cuda.malloc(&permuted_solution_, sizeof(float) * cols_);
  ++workspace_allocations_;
  // Dense descriptor pointers are rebound at dispatch. The initial pointers
  // refer to owned storage, so construction needs no borrowed RHS/output.
  sparse.cpCreateDnVec(&rhs_descriptor_, rows_, permuted_rhs_, CUDA_R_32F);
  ++descriptor_creations_;
  sparse.cpCreateDnVec(&solution_descriptor_, cols_, permuted_solution_,
                       CUDA_R_32F);
  ++descriptor_creations_;
  sparse.cpCreateSpVec(&permuted_rhs_descriptor_, rows_, rows_, d_Q_,
                       permuted_rhs_, CUSPARSE_INDEX_32I,
                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  ++descriptor_creations_;
  sparse.cpCreateSpVec(&permuted_solution_descriptor_, cols_, cols_, d_Q_,
                       permuted_solution_, CUSPARSE_INDEX_32I,
                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  ++descriptor_creations_;
#else
  TI_NOT_IMPLEMENTED
#endif
}

void CuSparseSolver::solve_cholesky(const void *rhs, void *solution) {
#if defined(TI_WITH_CUDA)
  auto &sparse = CUSPARSEDriver::get_instance();
  sparse.cpDnVecSetValues(rhs_descriptor_, const_cast<void *>(rhs));
  sparse.cpDnVecSetValues(solution_descriptor_, solution);
  sparse.cpGather(cusparse_handel_, rhs_descriptor_,
                  permuted_rhs_descriptor_);
  CUSOLVERDriver::get_instance().csSpScsrcholSolve(
      cusolver_handle_, rows_, permuted_rhs_, permuted_solution_, info_,
      gpu_buffer_);
  sparse.cpScatter(cusparse_handel_, permuted_solution_descriptor_,
                   solution_descriptor_);
#else
  TI_NOT_IMPLEMENTED
#endif
}

void CuSparseSolver::solve_lu(const void *rhs, void *solution) {
#if defined(TI_WITH_CUDA)
  // This legacy provider's LU factor and solve are host operations. Download
  // only the RHS; every output element is overwritten after the host solve.
  CUDADriver::get_instance().memcpy_device_to_host(
      host_rhs_.data(), const_cast<void *>(rhs), sizeof(float) * rows_);
  rhs_readback_bytes_ += sizeof(float) * rows_;
  for (int j = 0; j < rows_; ++j) {
    host_permuted_rhs_[j] = host_rhs_[h_Q_[j]];
  }
  CUSOLVERDriver::get_instance().csSpScsrluSolveHost(
      cusolver_handle_, rows_, host_permuted_rhs_.data(),
      host_permuted_solution_.data(), lu_info_, cpu_buffer_);
  for (int j = 0; j < cols_; ++j) {
    host_solution_[h_Q_[j]] = host_permuted_solution_[j];
  }
  CUDADriver::get_instance().memcpy_host_to_device(
      solution, host_solution_.data(), sizeof(float) * cols_);
#else
  TI_NOT_IMPLEMENTED
#endif
}

std::map<std::string, std::uint64_t>
CuSparseSolver::runtime_statistics() const {
  return {{"workspace_allocations", workspace_allocations_},
          {"descriptor_creations", descriptor_creations_},
          {"solve_calls", solve_calls_},
          {"rhs_readback_bytes", rhs_readback_bytes_},
          {"workspace_bytes",
           gpu_buffer_bytes_ + cpu_buffer_bytes_ +
               (permuted_rhs_ ? sizeof(float) * rows_ : 0) +
               (permuted_solution_ ? sizeof(float) * cols_ : 0) +
               sizeof(float) *
                   (host_rhs_.capacity() + host_permuted_rhs_.capacity() +
                    host_solution_.capacity() +
                    host_permuted_solution_.capacity())}};
}

std::vector<int> CuSparseSolver::ordering_permutation() const {
  if (!is_analyzed_)
    return {};
  return {h_Q_, h_Q_ + rows_};
}

std::unique_ptr<SparseSolver> make_sparse_solver(DataType dt,
                                                 const std::string &solver_type,
                                                 const std::string &ordering) {
  using key_type = Triplets;
  using func_type = std::unique_ptr<SparseSolver> (*)();
  static const std::unordered_map<key_type, func_type, key_hash>
      solver_factory = {
          MAKE_SOLVER(float32, LLT, AMD),  MAKE_SOLVER(float32, LLT, COLAMD),
          MAKE_SOLVER(float32, LDLT, AMD), MAKE_SOLVER(float32, LDLT, COLAMD),
          MAKE_SOLVER(float32, LU, AMD),   MAKE_SOLVER(float32, LU, COLAMD),
          MAKE_SOLVER(float64, LLT, AMD),  MAKE_SOLVER(float64, LLT, COLAMD),
          MAKE_SOLVER(float64, LDLT, AMD), MAKE_SOLVER(float64, LDLT, COLAMD),
          MAKE_SOLVER(float64, LU, AMD),   MAKE_SOLVER(float64, LU, COLAMD)};
  static const std::unordered_map<std::string, std::string> dt_map = {
      {"f32", "float32"}, {"f64", "float64"}};
  auto it = dt_map.find(taichi::lang::data_type_name(dt));
  if (it == dt_map.end())
    TI_ERROR("Not supported sparse solver data type: {}",
             taichi::lang::data_type_name(dt));

  Triplets solver_key = std::make_tuple(it->second, solver_type, ordering);
  if (solver_factory.find(solver_key) != solver_factory.end()) {
    auto solver_func = solver_factory.at(solver_key);
    return solver_func();
  } else
    TI_ERROR("Not supported sparse solver type: {}", solver_type);
}

CuSparseSolver::~CuSparseSolver() {
#if defined(TI_WITH_CUDA)
  auto submission_guard =
      CUDAContext::get_instance().get_submission_lock_guard();
  auto context_guard = CUDAContext::get_instance().get_guard();
  if (cusolver_handle_)
    CUDADriver::get_instance().stream_synchronize(nullptr);
#endif
  clear_analysis_resources();
}
std::unique_ptr<SparseSolver> make_cusparse_solver(
    DataType dt,
    const std::string &solver_type,
    const std::string &ordering) {
  TI_ERROR_IF(dt != PrimitiveType::f32,
              "cuSOLVERSp SparseSolver supports f32 only.");
  TI_ERROR_IF(ordering != "AMD" && ordering != "COLAMD",
              "cuSOLVERSp SparseSolver supports AMD or COLAMD ordering.");
  TI_ERROR_IF(solver_type == "LDLT",
              "cuSOLVERSp SparseSolver does not implement LDLT. Use LLT "
              "for positive-definite matrices or explicitly select a provider "
              "with symmetric-indefinite support, such as cuDSS.");
  if (solver_type == "LLT") {
    return std::make_unique<CuSparseSolver>(
        CuSparseSolver::SolverType::Cholesky, ordering);
  } else if (solver_type == "LU") {
    return std::make_unique<CuSparseSolver>(CuSparseSolver::SolverType::LU,
                                          ordering);
  } else {
    TI_ERROR("Not supported sparse solver type: {}", solver_type);
  }
}
}  // namespace taichi::lang
