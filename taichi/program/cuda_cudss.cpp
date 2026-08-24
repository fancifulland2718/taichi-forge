#include "taichi/program/program.h"

#include <cstdint>
#include <utility>
#include <vector>

#if defined(TI_WITH_CUDA)
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_driver.h"

namespace taichi::lang {
namespace {

constexpr int kCudssStatusSuccess = 0;
constexpr int kCudssPhaseAnalysis = 3;
constexpr int kCudssPhaseFactorization = 4;
constexpr int kCudssPhaseRefactorization = 8;
constexpr int kCudssPhaseSolve = 1008;
constexpr int kCudssDataTypeF32 = 0;
constexpr int kCudssDataTypeI32 = 10;
constexpr int kCudssMatrixGeneral = 0;
constexpr int kCudssMatrixSymmetric = 1;
constexpr int kCudssMatrixSpd = 3;
constexpr int kCudssViewFull = 0;
constexpr int kCudssViewLower = 1;
constexpr int kCudssViewUpper = 2;
constexpr int kCudssBaseZero = 0;
constexpr int kCudssLayoutColumnMajor = 0;

void require_cudss_success(std::uint32_t status, const char *operation) {
  TI_ERROR_IF(status != kCudssStatusSuccess,
              "CUDA cuDSS {} failed (status {}).", operation, status);
}

const CuSparseMatrix &require_cudss_matrix(SparseMatrix *matrix,
                                           Program *program) {
  TI_ERROR_IF(!matrix, "CUDA cuDSS received a null sparse matrix.");
  const auto *csr = dynamic_cast<const CuSparseMatrix *>(matrix);
  TI_ERROR_IF(!csr,
              "CUDA cuDSS requires a CUDA scalar CSR SparseMatrix; no "
              "format conversion or host fallback was performed.");
  TI_ERROR_IF(csr->num_rows() <= 0 || csr->num_rows() != csr->num_cols(),
              "CUDA cuDSS requires a non-empty square sparse matrix.");
  TI_ERROR_IF(csr->get_data_type() != PrimitiveType::f32,
              "The first CUDA cuDSS provider slice supports f32 only.");
  TI_ERROR_IF(csr->get_nnz() <= 0 || !csr->get_row_ptr() ||
                  !csr->get_col_ind() || !csr->get_val_ptr(),
              "CUDA cuDSS requires a materialized non-empty CSR matrix.");
  (void)program;
  return *csr;
}

void validate_cudss_matrix_contract(int matrix_type, int matrix_view) {
  TI_ERROR_IF(matrix_type != kCudssMatrixGeneral &&
                  matrix_type != kCudssMatrixSymmetric &&
                  matrix_type != kCudssMatrixSpd,
              "CUDA cuDSS matrix_type must be general, symmetric, or spd.");
  TI_ERROR_IF(matrix_view != kCudssViewFull &&
                  matrix_view != kCudssViewLower &&
                  matrix_view != kCudssViewUpper,
              "CUDA cuDSS matrix_view must be full, lower, or upper.");
  TI_ERROR_IF(matrix_type == kCudssMatrixGeneral &&
                  matrix_view != kCudssViewFull,
              "CUDA cuDSS general matrices require the full matrix view.");
}

void validate_cudss_vector(Ndarray *array,
                           std::size_t expected_elements,
                           const char *name,
                           Program *program) {
  TI_ERROR_IF(!array, "CUDA cuDSS {} received a null ndarray.", name);
  TI_ERROR_IF(!array->get_element_shape().empty() ||
                  array->get_element_data_type() != PrimitiveType::f32 ||
                  array->get_nelement() != expected_elements ||
                  array->get_element_size() != sizeof(float32),
              "CUDA cuDSS {} must be a compact scalar f32 ndarray with {} "
              "entries.",
              name, expected_elements);
  TI_ERROR_IF(array->owning_program() != program,
              "CUDA cuDSS {} must belong to the active runtime.", name);
}

}  // namespace

class CudaCudssPlan {
 public:
  CudaCudssPlan(const CuSparseMatrix &matrix,
                int matrix_type,
                int matrix_view,
                const std::string &library_path)
      : rows_(static_cast<std::size_t>(matrix.num_rows())) {
    validate_cudss_matrix_contract(matrix_type, matrix_view);
    auto &driver = CUDSSDriver::get_instance();
    TI_ERROR_IF(!driver.load_cudss(library_path),
                "CUDA cuDSS could not load the tested 0.8.x ABI and its "
                "required symbols. Install a matching user-managed cuDSS "
                "package or pass its shared-library path explicitly.");
    try {
      require_cudss_success(driver.create.call(&context_), "handle creation");
      TI_ERROR_IF(!context_, "CUDA cuDSS returned a null handle.");
      require_cudss_success(driver.set_stream.call(context_, nullptr),
                            "runtime stream binding");
      require_cudss_success(driver.config_create.call(&config_),
                            "configuration creation");
      require_cudss_success(driver.data_create.call(context_, &data_),
                            "solver-data creation");
      auto *row_start = matrix.get_row_ptr();
      // cuDSS accepts the canonical three-array CSR form when rowEnd is null.
      // Passing rowOffsets + 1 selects its unsupported four-array CSR form.
      require_cudss_success(
          driver.matrix_create_csr.call(
              &matrix_, static_cast<std::int64_t>(matrix.num_rows()),
              static_cast<std::int64_t>(matrix.num_cols()),
              static_cast<std::int64_t>(matrix.get_nnz()), row_start, nullptr,
              matrix.get_col_ind(), matrix.get_val_ptr(), kCudssDataTypeI32,
              kCudssDataTypeI32, kCudssDataTypeF32, matrix_type, matrix_view,
              kCudssBaseZero),
          "CSR descriptor creation");
    } catch (...) {
      destroy(true);
      throw;
    }
  }

  void analyze(const CuSparseMatrix &matrix) {
    std::lock_guard<std::mutex> lock(mutex_);
    TI_ERROR_IF(closed_, "CUDA cuDSS plan is closed.");
    TI_ERROR_IF(static_cast<std::size_t>(matrix.num_rows()) != rows_ ||
                    matrix.num_rows() != matrix.num_cols() ||
                    matrix.get_data_type() != PrimitiveType::f32,
                "CUDA cuDSS analyze received a matrix that does not match "
                "the plan shape or dtype.");
    auto &driver = CUDSSDriver::get_instance();
    require_cudss_success(
        driver.execute.call(context_, kCudssPhaseAnalysis, config_, data_,
                            matrix_, nullptr, nullptr),
        "analysis");
    analyzed_csr_row_ptr_.resize(rows_ + 1);
    analyzed_csr_col_ind_.resize(matrix.get_nnz());
    CUDADriver::get_instance().memcpy_device_to_host(
        analyzed_csr_row_ptr_.data(), matrix.get_row_ptr(),
        sizeof(int) * analyzed_csr_row_ptr_.size());
    CUDADriver::get_instance().memcpy_device_to_host(
        analyzed_csr_col_ind_.data(), matrix.get_col_ind(),
        sizeof(int) * analyzed_csr_col_ind_.size());
    const auto stats = matrix.debug_runtime_statistics();
    analyzed_matrix_id_ = matrix.matrix_id();
    analyzed_pattern_version_ = matrix.pattern_version();
    analyzed_shared_pattern_id_ = stats.shared_pattern_id;
    analyzed_ = true;
    factorized_ = false;
  }

  void factorize(const CuSparseMatrix &matrix, bool refactorize) {
    std::lock_guard<std::mutex> lock(mutex_);
    TI_ERROR_IF(closed_, "CUDA cuDSS plan is closed.");
    TI_ERROR_IF(!analyzed_,
                "CUDA cuDSS factorization requires analyze() first.");
    TI_ERROR_IF(refactorize && !factorized_,
                "CUDA cuDSS refactorization requires a prior successful "
                "factorization.");
    validate_analyzed_pattern(matrix);
    auto &driver = CUDSSDriver::get_instance();
    require_cudss_success(
        driver.matrix_set_csr_pointers.call(
            matrix_, matrix.get_row_ptr(), nullptr, matrix.get_col_ind(),
            matrix.get_val_ptr()),
        "CSR descriptor rebinding");
    factorized_ = false;
    require_cudss_success(
        driver.execute.call(context_,
                            refactorize ? kCudssPhaseRefactorization
                                        : kCudssPhaseFactorization,
                            config_, data_, matrix_, nullptr, nullptr),
        refactorize ? "refactorization" : "factorization");
    factorized_ = true;
    factorized_matrix_id_ = matrix.matrix_id();
    factorized_pattern_version_ = matrix.pattern_version();
    factorized_numeric_version_ = matrix.numeric_version();
  }

  void solve(const CuSparseMatrix &matrix, void *rhs, void *solution) {
    std::lock_guard<std::mutex> lock(mutex_);
    TI_ERROR_IF(closed_, "CUDA cuDSS plan is closed.");
    TI_ERROR_IF(!factorized_,
                "CUDA cuDSS solve requires a successful factorization.");
    TI_ERROR_IF(
        factorized_matrix_id_ != matrix.matrix_id() ||
            factorized_pattern_version_ != matrix.pattern_version() ||
            factorized_numeric_version_ != matrix.numeric_version(),
        "CUDA cuDSS factorization is stale because the matrix or its "
        "pattern/numeric version changed. Call factorize() again before "
        "solve().");
    auto &driver = CUDSSDriver::get_instance();
    if (!rhs_) {
      try {
        require_cudss_success(
            driver.matrix_create_dn.call(
                &rhs_, static_cast<std::int64_t>(rows_), 1,
                static_cast<std::int64_t>(rows_), rhs, kCudssDataTypeF32,
                kCudssLayoutColumnMajor),
            "right-hand-side descriptor creation");
        require_cudss_success(
            driver.matrix_create_dn.call(
                &solution_, static_cast<std::int64_t>(rows_), 1,
                static_cast<std::int64_t>(rows_), solution,
                kCudssDataTypeF32, kCudssLayoutColumnMajor),
            "solution descriptor creation");
      } catch (...) {
        if (rhs_) {
          driver.matrix_destroy.call_with_warning(rhs_);
          rhs_ = nullptr;
        }
        throw;
      }
    } else {
      require_cudss_success(driver.matrix_set_values.call(rhs_, rhs),
                            "right-hand-side rebinding");
      require_cudss_success(
          driver.matrix_set_values.call(solution_, solution),
          "solution rebinding");
    }
    require_cudss_success(
        driver.execute.call(context_, kCudssPhaseSolve, config_, data_,
                            matrix_, solution_, rhs_),
        "solve");
  }

  std::unordered_map<std::string, std::uint64_t> statistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {{"rows", static_cast<std::uint64_t>(rows_)},
            {"analyzed", analyzed_ ? 1u : 0u},
            {"factorized", factorized_ ? 1u : 0u},
            {"closed", closed_ ? 1u : 0u}};
  }

  void destroy(bool provider_calls_safe) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
      return;
    }
    closed_ = true;
    if (!provider_calls_safe || !CUDSSDriver::get_instance().is_loaded()) {
      return;
    }
    auto &driver = CUDSSDriver::get_instance();
    if (solution_) {
      driver.matrix_destroy.call_with_warning(solution_);
      solution_ = nullptr;
    }
    if (rhs_) {
      driver.matrix_destroy.call_with_warning(rhs_);
      rhs_ = nullptr;
    }
    if (matrix_) {
      driver.matrix_destroy.call_with_warning(matrix_);
      matrix_ = nullptr;
    }
    if (data_) {
      driver.data_destroy.call_with_warning(context_, data_);
      data_ = nullptr;
    }
    if (config_) {
      driver.config_destroy.call_with_warning(config_);
      config_ = nullptr;
    }
    if (context_) {
      driver.destroy.call_with_warning(context_);
      context_ = nullptr;
    }
  }

 private:
  void validate_analyzed_pattern(const CuSparseMatrix &matrix) const {
    TI_ERROR_IF(static_cast<std::size_t>(matrix.num_rows()) != rows_ ||
                    matrix.num_rows() != matrix.num_cols() ||
                    matrix.get_data_type() != PrimitiveType::f32 ||
                    matrix.get_nnz() !=
                        static_cast<int>(analyzed_csr_col_ind_.size()),
                "CUDA cuDSS factorize() requires the same sparse pattern "
                "that was passed to analyze(); shape, dtype, or nonzero "
                "count changed.");
    const auto stats = matrix.debug_runtime_statistics();
    if ((matrix.matrix_id() == analyzed_matrix_id_ &&
         matrix.pattern_version() == analyzed_pattern_version_) ||
        (analyzed_shared_pattern_id_ != 0 &&
         stats.shared_pattern_id == analyzed_shared_pattern_id_)) {
      return;
    }
    std::vector<int> row_ptr(analyzed_csr_row_ptr_.size());
    std::vector<int> col_ind(analyzed_csr_col_ind_.size());
    CUDADriver::get_instance().memcpy_device_to_host(
        row_ptr.data(), matrix.get_row_ptr(), sizeof(int) * row_ptr.size());
    CUDADriver::get_instance().memcpy_device_to_host(
        col_ind.data(), matrix.get_col_ind(), sizeof(int) * col_ind.size());
    TI_ERROR_IF(row_ptr != analyzed_csr_row_ptr_ ||
                    col_ind != analyzed_csr_col_ind_,
                "CUDA cuDSS factorize() requires the same sparse pattern "
                "that was passed to analyze(); a CSR index changed.");
  }

  std::size_t rows_{0};
  void *context_{nullptr};
  void *config_{nullptr};
  void *data_{nullptr};
  void *matrix_{nullptr};
  void *rhs_{nullptr};
  void *solution_{nullptr};
  bool analyzed_{false};
  bool factorized_{false};
  bool closed_{false};
  std::vector<int> analyzed_csr_row_ptr_;
  std::vector<int> analyzed_csr_col_ind_;
  std::uint64_t analyzed_matrix_id_{0};
  std::uint64_t analyzed_pattern_version_{0};
  std::uint64_t analyzed_shared_pattern_id_{0};
  std::uint64_t factorized_matrix_id_{0};
  std::uint64_t factorized_pattern_version_{0};
  std::uint64_t factorized_numeric_version_{0};
  mutable std::mutex mutex_;
};

std::uint64_t Program::create_cuda_cudss_plan(
    SparseMatrix *matrix,
    int matrix_type,
    int matrix_view,
    const std::string &library_path) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA cuDSS plans require the CUDA backend.");
  TI_ERROR_IF(!CUDADriver::get_instance_without_context()
                   .nvidia_extensions_available(),
              "CUDA cuDSS requires the NVIDIA CUDA provider.");
  const auto &csr = require_cudss_matrix(matrix, this);
  auto cuda_submission_guard =
      CUDAContext::get_instance().get_submission_lock_guard();
  auto context_guard = CUDAContext::get_instance().get_guard();
  auto plan = std::make_shared<CudaCudssPlan>(csr, matrix_type, matrix_view,
                                              library_path);
  std::lock_guard<std::mutex> lock(cuda_cudss_plan_mutex_);
  TI_ERROR_IF(next_cuda_cudss_plan_handle_ == 0,
              "CUDA cuDSS plan handle space exhausted.");
  const auto handle = next_cuda_cudss_plan_handle_++;
  cuda_cudss_plans_.emplace(handle, std::move(plan));
  return handle;
}

void Program::cuda_cudss_analyze(std::uint64_t handle, SparseMatrix *matrix) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  std::shared_ptr<CudaCudssPlan> plan;
  {
    std::lock_guard<std::mutex> lock(cuda_cudss_plan_mutex_);
    const auto found = cuda_cudss_plans_.find(handle);
    TI_ERROR_IF(found == cuda_cudss_plans_.end(),
                "CUDA cuDSS plan handle is stale or closed.");
    plan = found->second;
  }
  auto cuda_submission_guard =
      CUDAContext::get_instance().get_submission_lock_guard();
  auto context_guard = CUDAContext::get_instance().get_guard();
  const auto &csr = require_cudss_matrix(matrix, this);
  plan->analyze(csr);
  mark_runtime_submission_pending();
}

void Program::cuda_cudss_factorize(std::uint64_t handle,
                                   SparseMatrix *matrix,
                                   bool refactorize) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  std::shared_ptr<CudaCudssPlan> plan;
  {
    std::lock_guard<std::mutex> lock(cuda_cudss_plan_mutex_);
    const auto found = cuda_cudss_plans_.find(handle);
    TI_ERROR_IF(found == cuda_cudss_plans_.end(),
                "CUDA cuDSS plan handle is stale or closed.");
    plan = found->second;
  }
  auto cuda_submission_guard =
      CUDAContext::get_instance().get_submission_lock_guard();
  auto context_guard = CUDAContext::get_instance().get_guard();
  const auto &csr = require_cudss_matrix(matrix, this);
  plan->factorize(csr, refactorize);
  mark_runtime_submission_pending();
}

std::size_t Program::cuda_cudss_solve(std::uint64_t handle,
                                      SparseMatrix *matrix,
                                      Ndarray *rhs,
                                      Ndarray *solution) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA cuDSS solve requires the CUDA backend.");
  std::shared_ptr<CudaCudssPlan> plan;
  {
    std::lock_guard<std::mutex> lock(cuda_cudss_plan_mutex_);
    const auto found = cuda_cudss_plans_.find(handle);
    TI_ERROR_IF(found == cuda_cudss_plans_.end(),
                "CUDA cuDSS plan handle is stale or closed.");
    plan = found->second;
  }
  const auto stats = plan->statistics();
  const auto rows = static_cast<std::size_t>(stats.at("rows"));
  validate_cudss_vector(rhs, rows, "right-hand side", this);
  validate_cudss_vector(solution, rows, "solution", this);
  TI_ERROR_IF(rhs->get_device_allocation() == solution->get_device_allocation(),
              "The first CUDA cuDSS slice requires distinct right-hand-side "
              "and solution allocations.");
  auto cuda_submission_guard =
      CUDAContext::get_instance().get_submission_lock_guard();
  auto context_guard = CUDAContext::get_instance().get_guard();
  const auto &csr = require_cudss_matrix(matrix, this);
  auto *rhs_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(rhs));
  auto *solution_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(solution));
  TI_ERROR_IF(!rhs_ptr || !solution_ptr,
              "CUDA cuDSS received a null dense-vector device pointer.");
  plan->solve(csr, rhs_ptr, solution_ptr);
  pin_ndarray_launch_leases(acquire_ndarray_leases({rhs, solution}));
  mark_runtime_submission_pending();
  return 0;
}

std::unordered_map<std::string, std::uint64_t>
Program::cuda_cudss_plan_statistics(std::uint64_t handle) {
  std::lock_guard<std::mutex> lock(cuda_cudss_plan_mutex_);
  const auto found = cuda_cudss_plans_.find(handle);
  TI_ERROR_IF(found == cuda_cudss_plans_.end(),
              "CUDA cuDSS plan handle is stale or closed.");
  return found->second->statistics();
}

void Program::destroy_cuda_cudss_plan(std::uint64_t handle) {
  std::shared_ptr<CudaCudssPlan> plan;
  {
    auto submission_guard = acquire_runtime_resource_submission_guard();
    std::lock_guard<std::mutex> lock(cuda_cudss_plan_mutex_);
    const auto found = cuda_cudss_plans_.find(handle);
    if (found == cuda_cudss_plans_.end()) {
      return;
    }
    plan = std::move(found->second);
    cuda_cudss_plans_.erase(found);
  }
  if (!runtime_has_fatal_fault()) {
    synchronize();
    auto cuda_submission_guard =
        CUDAContext::get_instance().get_submission_lock_guard();
    auto context_guard = CUDAContext::get_instance().get_guard();
    plan->destroy(true);
  } else {
    plan->destroy(false);
  }
}

void Program::cuda_clear_cudss_plans() {
  std::vector<std::shared_ptr<CudaCudssPlan>> plans;
  {
    std::lock_guard<std::mutex> lock(cuda_cudss_plan_mutex_);
    plans.reserve(cuda_cudss_plans_.size());
    for (auto &[handle, plan] : cuda_cudss_plans_) {
      plans.push_back(std::move(plan));
    }
    cuda_cudss_plans_.clear();
  }
  const bool provider_calls_safe = !runtime_has_fatal_fault();
  if (provider_calls_safe && !plans.empty()) {
    auto cuda_submission_guard =
        CUDAContext::get_instance().get_submission_lock_guard();
    auto context_guard = CUDAContext::get_instance().get_guard();
    for (auto &plan : plans) {
      plan->destroy(true);
    }
  } else {
    for (auto &plan : plans) {
      plan->destroy(false);
    }
  }
}

}  // namespace taichi::lang

#else

namespace taichi::lang {

std::uint64_t Program::create_cuda_cudss_plan(SparseMatrix *,
                                              int,
                                              int,
                                              const std::string &) {
  TI_ERROR("CUDA cuDSS requires TI_WITH_CUDA=ON.");
}

void Program::cuda_cudss_analyze(std::uint64_t, SparseMatrix *) {
  TI_ERROR("CUDA cuDSS requires TI_WITH_CUDA=ON.");
}

void Program::cuda_cudss_factorize(std::uint64_t, SparseMatrix *, bool) {
  TI_ERROR("CUDA cuDSS requires TI_WITH_CUDA=ON.");
}

std::size_t Program::cuda_cudss_solve(std::uint64_t,
                                      SparseMatrix *,
                                      Ndarray *,
                                      Ndarray *) {
  TI_ERROR("CUDA cuDSS requires TI_WITH_CUDA=ON.");
}

std::unordered_map<std::string, std::uint64_t>
Program::cuda_cudss_plan_statistics(std::uint64_t) {
  TI_ERROR("CUDA cuDSS requires TI_WITH_CUDA=ON.");
}

void Program::destroy_cuda_cudss_plan(std::uint64_t) {
}

void Program::cuda_clear_cudss_plans() {
}

}  // namespace taichi::lang

#endif
