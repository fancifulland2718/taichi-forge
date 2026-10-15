#include "taichi/rhi/cuda/cusparse_plan.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include "taichi/rhi/cuda/cuda_driver.h"

namespace taichi::lang {
namespace {
cusparseOperation_t operation(const CudaCsrSpmvDescription &description) {
  return description.transpose ? CUSPARSE_OPERATION_TRANSPOSE
                               : CUSPARSE_OPERATION_NON_TRANSPOSE;
}

std::size_t index_bytes(cusparseIndexType_t type) {
  if (type == CUSPARSE_INDEX_32I)
    return sizeof(std::int32_t);
  if (type == CUSPARSE_INDEX_64I)
    return sizeof(std::int64_t);
  throw std::invalid_argument("CUDA CSR requires signed 32/64-bit indices");
}

void aligned(const void *pointer, std::size_t alignment, bool required) {
  if ((required && !pointer) ||
      reinterpret_cast<std::uintptr_t>(pointer) % alignment)
    throw std::invalid_argument("CUDA CSR requires aligned live storage");
}
}  // namespace

CudaCsrSpmvPlan::CudaCsrSpmvPlan(CudaCsrSpmvDescription description)
    : description_(description) {
  const auto &d = description_;
  if (d.rows < 0 || d.columns < 0 || d.nonzeros < 0 ||
      !std::isfinite(d.alpha) || !std::isfinite(d.beta) ||
      (d.algorithm != CUSPARSE_SPMV_CSR_ALG1 &&
       d.algorithm != CUSPARSE_SPMV_CSR_ALG2))
    throw std::invalid_argument("Invalid CUDA CSR SpMV description");
  aligned(d.row_offsets, index_bytes(d.row_type), true);
  aligned(d.column_indices, index_bytes(d.column_type), d.nonzeros != 0);
  if (d.row_type != d.column_type)
    throw std::invalid_argument(
        "CUDA CSR row and column indices require matching widths");
  aligned(d.values, sizeof(float), d.nonzeros != 0);
  auto &driver = CUSPARSEDriver::get_instance();
  try {
    driver.cpCreate(&handle_);
    driver.cpSetPointerMode(handle_, CUSPARSE_POINTER_MODE_HOST);
    driver.cpCreateCsr(&matrix_, d.rows, d.columns, d.nonzeros, d.row_offsets,
                       d.column_indices, d.values, d.row_type, d.column_type,
                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  } catch (...) {
    clear();
    throw;
  }
}

CudaCsrSpmvPlan::~CudaCsrSpmvPlan() {
  clear();
}

void CudaCsrSpmvPlan::clear() {
  auto &driver = CUSPARSEDriver::get_instance();
  if (input_)
    driver.cpDestroyDnVec.call_with_warning(input_);
  if (output_)
    driver.cpDestroyDnVec.call_with_warning(output_);
  if (matrix_)
    driver.cpDestroySpMat.call_with_warning(matrix_);
  if (handle_)
    driver.cpDestroy.call_with_warning(handle_);
  if (workspace_)
    CUDADriver::get_instance().mem_free.call_with_warning(workspace_);
  input_ = output_ = nullptr;
  matrix_ = nullptr;
  handle_ = nullptr;
  workspace_ = nullptr;
}

void CudaCsrSpmvPlan::bind_stream(CUstream stream) {
  if (stream_ == stream)
    return;
  auto &driver = CUSPARSEDriver::get_instance();
  TI_ERROR_IF(
      !driver.cpSetStream.available() || !driver.cpGetStream.available(),
      "CUDA CSR SpMV cannot bind the requested execution stream.");
  driver.cpSetStream(handle_, stream);
  CUstream bound = reinterpret_cast<CUstream>(1);
  driver.cpGetStream(handle_, &bound);
  TI_ERROR_IF(bound != stream,
              "CUDA CSR SpMV stream binding could not be verified.");
  stream_ = stream;
}

CudaCsrSpmvPlan::Preparation CudaCsrSpmvPlan::prepare(void *input,
                                                      void *output,
                                                      CUstream stream,
                                                      bool enable_preprocess) {
  const auto input_size =
      description_.transpose ? description_.rows : description_.columns;
  const auto output_size =
      description_.transpose ? description_.columns : description_.rows;
  aligned(input, sizeof(float), input_size != 0);
  aligned(output, sizeof(float), output_size != 0);
  bind_stream(stream);
  auto &driver = CUSPARSEDriver::get_instance();
  Preparation result;
  auto bind_vector = [&](cusparseDnVecDescr_t &descriptor, void *&previous,
                         void *address, std::int64_t size) {
    if (descriptor && previous == address)
      return;
    cusparseDnVecDescr_t created = nullptr;
    driver.cpCreateDnVec(&created, size, address, CUDA_R_32F);
    if (descriptor) {
      driver.cpDestroyDnVec.call_with_warning(descriptor);
      ++result.descriptor_rebinds;
    }
    descriptor = created;
    previous = address;
    ++result.descriptor_creations;
  };
  bind_vector(input_, input_address_, input, input_size);
  bind_vector(output_, output_address_, output, output_size);
  if (!initialized_) {
    driver.cpSpMV_bufferSize(handle_, operation(description_),
                             &description_.alpha, matrix_, input_,
                             &description_.beta, output_, CUDA_R_32F,
                             description_.algorithm, &workspace_bytes_);
    if (workspace_bytes_) {
      CUDADriver::get_instance().malloc(&workspace_, workspace_bytes_);
      result.workspace_allocated = true;
    }
    initialized_ = result.built = true;
  }
  if (enable_preprocess && driver.cpSpMVPreprocess.available() &&
      !preprocess_disabled_) {
    if (!statistics_.preprocessed) {
      statistics_.preprocess_error = driver.cpSpMVPreprocess.call(
          handle_, operation(description_), &description_.alpha, matrix_,
          input_, &description_.beta, output_, CUDA_R_32F,
          description_.algorithm, workspace_);
      if (!statistics_.preprocess_error) {
        statistics_.preprocessed = true;
        ++statistics_.preprocess_builds;
      } else {
        preprocess_disabled_ = true;
        ++statistics_.preprocess_fallbacks;
      }
    } else {
      ++statistics_.preprocess_reuses;
    }
  } else {
    ++statistics_.preprocess_fallbacks;
  }
  return result;
}

void CudaCsrSpmvPlan::record(CUstream stream) {
  TI_ERROR_IF(!initialized_,
              "CUDA CSR SpMV must be prepared before recording.");
  bind_stream(stream);
  CUSPARSEDriver::get_instance().cpSpMV(handle_, operation(description_),
                                        &description_.alpha, matrix_, input_,
                                        &description_.beta, output_, CUDA_R_32F,
                                        description_.algorithm, workspace_);
}

CudaCsrSpmmPlan::CudaCsrSpmmPlan(CudaCsrSpmmDescription description)
    : description_(description) {
  const auto &d = description_;
  if (d.rows < 0 || d.columns < 0 || d.nonzeros < 0 || d.rhs_count <= 0 ||
      !std::isfinite(d.alpha) || !std::isfinite(d.beta) ||
      (d.algorithm != CUSPARSE_SPMM_CSR_ALG2 &&
       d.algorithm != CUSPARSE_SPMM_CSR_ALG3) ||
      (d.preprocess != CudaSpmmPreprocess::none &&
       d.preprocess != CudaSpmmPreprocess::optional &&
       d.preprocess != CudaSpmmPreprocess::required))
    throw std::invalid_argument("Invalid CUDA CSR SpMM description");
  for (auto size : {d.rows, d.columns})
    if (std::uint64_t(size) > std::numeric_limits<std::size_t>::max() /
                                  sizeof(float) / std::uint64_t(d.rhs_count))
      throw std::invalid_argument("CUDA CSR SpMM dense extent overflows");
  aligned(d.row_offsets, index_bytes(d.row_type), true);
  aligned(d.column_indices, index_bytes(d.column_type), d.nonzeros != 0);
  aligned(d.values, sizeof(float), d.nonzeros != 0);
  if (d.row_type != d.column_type)
    throw std::invalid_argument(
        "CUDA CSR row and column indices require matching widths");
  auto &driver = CUSPARSEDriver::get_instance();
  TI_ERROR_IF(!driver.capabilities().spmm_f32_available,
              "The loaded cuSPARSE provider does not expose f32 SpMM.");
  TI_ERROR_IF(d.preprocess == CudaSpmmPreprocess::required &&
                  !driver.cpSpMMPreprocess.available(),
              "The selected SpMM plan requires cuSPARSE preprocessing.");
  try {
    driver.cpCreate(&handle_);
    driver.cpSetPointerMode(handle_, CUSPARSE_POINTER_MODE_HOST);
    driver.cpCreateCsr(&matrix_, d.rows, d.columns, d.nonzeros, d.row_offsets,
                       d.column_indices, d.values, d.row_type, d.column_type,
                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  } catch (...) {
    clear();
    throw;
  }
}

CudaCsrSpmmPlan::~CudaCsrSpmmPlan() {
  clear();
}

void CudaCsrSpmmPlan::clear() {
  auto &driver = CUSPARSEDriver::get_instance();
  if (input_)
    driver.cpDestroyDnMat.call_with_warning(input_);
  if (output_)
    driver.cpDestroyDnMat.call_with_warning(output_);
  if (matrix_)
    driver.cpDestroySpMat.call_with_warning(matrix_);
  if (handle_)
    driver.cpDestroy.call_with_warning(handle_);
  if (workspace_)
    CUDADriver::get_instance().mem_free.call_with_warning(workspace_);
  input_ = output_ = nullptr;
  matrix_ = nullptr;
  handle_ = nullptr;
  workspace_ = nullptr;
  initialized_ = false;
}

void CudaCsrSpmmPlan::bind_stream(CUstream stream) {
  if (stream_bound_ && stream_ == stream)
    return;
  auto &driver = CUSPARSEDriver::get_instance();
  TI_ERROR_IF(
      !driver.cpSetStream.available() || !driver.cpGetStream.available(),
      "CUDA CSR SpMM cannot bind the requested execution stream.");
  driver.cpSetStream(handle_, stream);
  CUstream bound = reinterpret_cast<CUstream>(1);
  driver.cpGetStream(handle_, &bound);
  TI_ERROR_IF(bound != stream,
              "CUDA CSR SpMM stream binding could not be verified.");
  stream_ = stream;
  stream_bound_ = true;
}

CudaCsrSpmmPlan::Preparation CudaCsrSpmmPlan::prepare(void *input,
                                                      void *output,
                                                      CUstream stream) {
  const auto &d = description_;
  aligned(input, sizeof(float), d.columns != 0);
  aligned(output, sizeof(float), d.rows != 0);
  TI_ERROR_IF(!handle_, "CUDA CSR SpMM preparation failed; rebuild the plan.");
  bind_stream(stream);
  auto &driver = CUSPARSEDriver::get_instance();
  Preparation result;
  if (!initialized_) {
    try {
      driver.cpCreateDnMat(&input_, d.columns, d.rhs_count, d.rhs_count, input,
                           CUDA_R_32F, CUSPARSE_ORDER_ROW);
      driver.cpCreateDnMat(&output_, d.rows, d.rhs_count, d.rhs_count, output,
                           CUDA_R_32F, CUSPARSE_ORDER_ROW);
      driver.cpSpMMBufferSize(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                              CUSPARSE_OPERATION_NON_TRANSPOSE, &d.alpha,
                              matrix_, input_, &d.beta, output_, CUDA_R_32F,
                              d.algorithm, &workspace_bytes_);
      if (workspace_bytes_) {
        CUDADriver::get_instance().malloc(&workspace_, workspace_bytes_);
        result.workspace_allocated = true;
      }
      if (d.preprocess != CudaSpmmPreprocess::none &&
          driver.cpSpMMPreprocess.available()) {
        statistics_.preprocess_attempted = true;
        statistics_.preprocess_error = driver.cpSpMMPreprocess.call(
            handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &d.alpha, matrix_, input_,
            &d.beta, output_, CUDA_R_32F, d.algorithm, workspace_);
        TI_ERROR_IF(d.preprocess == CudaSpmmPreprocess::required &&
                        statistics_.preprocess_error,
                    "The selected SpMM preprocessing failed with status {}; no "
                    "different physical plan was substituted.",
                    statistics_.preprocess_error);
        statistics_.preprocessed = statistics_.preprocess_error == 0;
      }
      input_address_ = input;
      output_address_ = output;
      result.built = initialized_ = true;
    } catch (...) {
      clear();
      throw;
    }
  } else {
    auto rebind = [&](cusparseDnMatDescr_t descriptor, void *&previous,
                      void *address) {
      if (previous != address) {
        driver.cpDnMatSetValues(descriptor, address);
        previous = address;
        ++result.descriptor_rebinds;
      }
    };
    rebind(input_, input_address_, input);
    rebind(output_, output_address_, output);
  }
  return result;
}

void CudaCsrSpmmPlan::record(CUstream stream) {
  TI_ERROR_IF(!initialized_,
              "CUDA CSR SpMM must be prepared before recording.");
  bind_stream(stream);
  CUSPARSEDriver::get_instance().cpSpMM(
      handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &description_.alpha, matrix_, input_,
      &description_.beta, output_, CUDA_R_32F, description_.algorithm,
      workspace_);
}
CudaCsrTriangularPlan::CudaCsrTriangularPlan(
    CudaCsrTriangularDescription description)
    : description_(description) {
  const auto &d = description_;
  if (d.rows < 0 || d.nonzeros < 0 || d.rhs_count < 1 ||
      std::uint64_t(d.rows) > std::numeric_limits<std::size_t>::max() /
                                  sizeof(float) / std::uint64_t(d.rhs_count))
    throw std::invalid_argument("Invalid CUDA CSR triangular description");
  aligned(d.row_offsets, index_bytes(d.index_type), true);
  aligned(d.column_indices, index_bytes(d.index_type), d.nonzeros != 0);
  aligned(d.values, sizeof(float), d.nonzeros != 0);
  auto &driver = CUSPARSEDriver::get_instance();
  const auto capabilities = driver.capabilities();
  TI_ERROR_IF(d.rhs_count == 1 ? !capabilities.spsv_f32_available
                               : !capabilities.spsm_f32_available,
              "The loaded cuSPARSE provider does not expose triangular solve.");
  try {
    driver.cpCreate(&handle_);
    driver.cpSetPointerMode(handle_, CUSPARSE_POINTER_MODE_HOST);
    driver.cpCreateCsr(&matrix_, d.rows, d.rows, d.nonzeros, d.row_offsets,
                       d.column_indices, d.values, d.index_type, d.index_type,
                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    const auto fill =
        d.upper ? CUSPARSE_FILL_MODE_UPPER : CUSPARSE_FILL_MODE_LOWER;
    const auto diagonal =
        d.unit_diagonal ? CUSPARSE_DIAG_TYPE_UNIT : CUSPARSE_DIAG_TYPE_NON_UNIT;
    driver.cpSpMatSetAttribute(matrix_, CUSPARSE_SPMAT_FILL_MODE, &fill,
                               sizeof(fill));
    driver.cpSpMatSetAttribute(matrix_, CUSPARSE_SPMAT_DIAG_TYPE, &diagonal,
                               sizeof(diagonal));
  } catch (...) {
    clear();
    throw;
  }
}

CudaCsrTriangularPlan::~CudaCsrTriangularPlan() {
  clear();
}

void CudaCsrTriangularPlan::clear() {
  auto &driver = CUSPARSEDriver::get_instance();
  if (vector_solve_)
    driver.cpSpSVDestroyDescr.call_with_warning(vector_solve_);
  if (matrix_solve_)
    driver.cpSpSMDestroyDescr.call_with_warning(matrix_solve_);
  if (input_vector_)
    driver.cpDestroyDnVec.call_with_warning(input_vector_);
  if (output_vector_)
    driver.cpDestroyDnVec.call_with_warning(output_vector_);
  if (input_matrix_)
    driver.cpDestroyDnMat.call_with_warning(input_matrix_);
  if (output_matrix_)
    driver.cpDestroyDnMat.call_with_warning(output_matrix_);
  if (matrix_)
    driver.cpDestroySpMat.call_with_warning(matrix_);
  if (handle_)
    driver.cpDestroy.call_with_warning(handle_);
  if (workspace_)
    CUDADriver::get_instance().mem_free.call_with_warning(workspace_);
  vector_solve_ = nullptr;
  matrix_solve_ = nullptr;
  input_vector_ = output_vector_ = nullptr;
  input_matrix_ = output_matrix_ = nullptr;
  matrix_ = nullptr;
  handle_ = nullptr;
  workspace_ = nullptr;
  workspace_bytes_ = 0;
  initialized_ = stream_bound_ = false;
}

cusparseOperation_t CudaCsrTriangularPlan::operation() const {
  return description_.transpose ? CUSPARSE_OPERATION_TRANSPOSE
                                : CUSPARSE_OPERATION_NON_TRANSPOSE;
}

void CudaCsrTriangularPlan::bind_stream(CUstream stream) {
  if (stream_bound_ && stream_ == stream)
    return;
  auto &driver = CUSPARSEDriver::get_instance();
  TI_ERROR_IF(
      !driver.cpSetStream.available() || !driver.cpGetStream.available(),
      "CUDA CSR triangular plan cannot bind the requested stream.");
  driver.cpSetStream(handle_, stream);
  CUstream bound = reinterpret_cast<CUstream>(1);
  driver.cpGetStream(handle_, &bound);
  TI_ERROR_IF(bound != stream,
              "CUDA CSR triangular stream binding could not be verified.");
  stream_ = stream;
  stream_bound_ = true;
}

CudaCsrTriangularPlan::Preparation
CudaCsrTriangularPlan::prepare(void *input, void *output, CUstream stream) {
  const auto &d = description_;
  aligned(input, sizeof(float), d.rows != 0);
  aligned(output, sizeof(float), d.rows != 0);
  TI_ERROR_IF(d.rows != 0 && input == output,
              "CUDA CSR triangular input and output must be distinct.");
  TI_ERROR_IF(!handle_,
              "CUDA CSR triangular preparation failed; rebuild the plan.");
  bind_stream(stream);
  auto &driver = CUSPARSEDriver::get_instance();
  Preparation result;
  const float alpha = 1;
  if (!initialized_) {
    try {
      if (d.rhs_count == 1) {
        driver.cpCreateDnVec(&input_vector_, d.rows, input, CUDA_R_32F);
        driver.cpCreateDnVec(&output_vector_, d.rows, output, CUDA_R_32F);
        driver.cpSpSVCreateDescr(&vector_solve_);
        driver.cpSpSVBufferSize(handle_, operation(), &alpha, matrix_,
                                input_vector_, output_vector_, CUDA_R_32F,
                                CUSPARSE_SPSV_ALG_DEFAULT, vector_solve_,
                                &workspace_bytes_);
      } else {
        driver.cpCreateDnMat(&input_matrix_, d.rows, d.rhs_count, d.rhs_count,
                             input, CUDA_R_32F, CUSPARSE_ORDER_ROW);
        driver.cpCreateDnMat(&output_matrix_, d.rows, d.rhs_count, d.rhs_count,
                             output, CUDA_R_32F, CUSPARSE_ORDER_ROW);
        driver.cpSpSMCreateDescr(&matrix_solve_);
        driver.cpSpSMBufferSize(
            handle_, operation(), CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
            matrix_, input_matrix_, output_matrix_, CUDA_R_32F,
            CUSPARSE_SPSM_ALG_DEFAULT, matrix_solve_, &workspace_bytes_);
      }
      if (workspace_bytes_)
        CUDADriver::get_instance().malloc(&workspace_, workspace_bytes_);
      if (d.rhs_count == 1) {
        driver.cpSpSVAnalysis(handle_, operation(), &alpha, matrix_,
                              input_vector_, output_vector_, CUDA_R_32F,
                              CUSPARSE_SPSV_ALG_DEFAULT, vector_solve_,
                              workspace_);
      } else {
        driver.cpSpSMAnalysis(
            handle_, operation(), CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
            matrix_, input_matrix_, output_matrix_, CUDA_R_32F,
            CUSPARSE_SPSM_ALG_DEFAULT, matrix_solve_, workspace_);
      }
      input_address_ = input;
      output_address_ = output;
      initialized_ = result.built = true;
    } catch (...) {
      clear();
      throw;
    }
  } else {
    auto rebind = [&](bool is_input, void *address) {
      auto &previous = is_input ? input_address_ : output_address_;
      if (previous == address)
        return;
      if (d.rhs_count == 1) {
        driver.cpDnVecSetValues(is_input ? input_vector_ : output_vector_,
                                address);
      } else {
        driver.cpDnMatSetValues(is_input ? input_matrix_ : output_matrix_,
                                address);
      }
      previous = address;
      ++result.descriptor_rebinds;
    };
    rebind(true, input);
    rebind(false, output);
  }
  return result;
}

void CudaCsrTriangularPlan::record(CUstream stream) {
  TI_ERROR_IF(!initialized_,
              "CUDA CSR triangular plan must be prepared before recording.");
  bind_stream(stream);
  auto &driver = CUSPARSEDriver::get_instance();
  const float alpha = 1;
  if (description_.rhs_count == 1) {
    driver.cpSpSVSolve(handle_, operation(), &alpha, matrix_, input_vector_,
                       output_vector_, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT,
                       vector_solve_);
  } else {
    driver.cpSpSMSolve(handle_, operation(), CUSPARSE_OPERATION_NON_TRANSPOSE,
                       &alpha, matrix_, input_matrix_, output_matrix_,
                       CUDA_R_32F, CUSPARSE_SPSM_ALG_DEFAULT, matrix_solve_);
  }
}

void CudaCsrTriangularPlan::update_values() {
  update_values(stream_);
}

void CudaCsrTriangularPlan::update_values(CUstream stream) {
  TI_ERROR_IF(!initialized_,
              "CUDA CSR triangular values require a prepared plan.");
  bind_stream(stream);
  auto &driver = CUSPARSEDriver::get_instance();
  if (description_.rhs_count == 1) {
    TI_ERROR_IF(!driver.cpSpSVUpdateMatrix.available(),
                "The cuSPARSE provider cannot update retained SpSV values.");
    driver.cpSpSVUpdateMatrix(handle_, vector_solve_, description_.values,
                              CUSPARSE_SPSV_UPDATE_GENERAL);
  } else {
    TI_ERROR_IF(!driver.cpSpSMUpdateMatrix.available(),
                "The cuSPARSE provider cannot update retained SpSM values.");
    driver.cpSpSMUpdateMatrix(handle_, matrix_solve_, description_.values,
                              CUSPARSE_SPSM_UPDATE_GENERAL);
  }
}
}  // namespace taichi::lang
