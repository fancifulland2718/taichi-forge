#include "taichi/program/sparse_preconditioner.h"

#include "taichi/program/linear_operator.h"
#include "taichi/program/program.h"

#if defined(TI_WITH_CUDA)
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/cuda/primitives/hierarchical_ptx.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

namespace taichi::lang {
namespace {

void validate_vector(const char *role,
                     const Ndarray &array,
                     int rows,
                     DataType dtype = PrimitiveType::f32) {
  TI_ERROR_IF(array.get_element_data_type() != dtype ||
                  !array.get_element_shape().empty() ||
                  array.shape.size() != 1 ||
                  array.get_nelement() != static_cast<std::size_t>(rows) ||
                  array.get_element_size() != data_type_size(dtype),
              "Sparse Jacobi {} must contain exactly {} scalar {} entries.",
              role, rows, data_type_name(dtype));
}

template <typename T>
T invert_diagonal_value(T value, int row) {
  TI_ERROR_IF(!std::isfinite(value),
              "Sparse Jacobi diagonal at row {} is not finite.", row);
  TI_ERROR_IF(value == static_cast<T>(0),
              "Sparse Jacobi diagonal at row {} is zero.", row);
  const T inverse = static_cast<T>(1) / value;
  TI_ERROR_IF(!std::isfinite(inverse),
              "Sparse Jacobi inverse diagonal at row {} is not finite.",
              row);
  return inverse;
}

template <typename T>
std::vector<T> extract_csr_inverse(
    const std::vector<int32_t> &row_offsets,
    const std::vector<int32_t> &column_indices,
    const T *values,
    std::size_t value_count,
    int rows,
    std::vector<int32_t> *diagonal_offsets = nullptr) {
  TI_ERROR_IF(row_offsets.size() != static_cast<std::size_t>(rows) + 1 ||
                  column_indices.size() != value_count || !values,
              "Sparse Jacobi received inconsistent CSR host storage.");
  TI_ERROR_IF(row_offsets.front() != 0 ||
                  row_offsets.back() !=
                      static_cast<int32_t>(column_indices.size()),
              "Sparse Jacobi CSR row offsets must start at zero and end at "
              "nnz.");
  std::vector<T> inverse(static_cast<std::size_t>(rows));
  if (diagonal_offsets) {
    diagonal_offsets->assign(static_cast<std::size_t>(rows), -1);
  }
  for (int row = 0; row < rows; ++row) {
    const int32_t begin = row_offsets[row];
    const int32_t end = row_offsets[row + 1];
    TI_ERROR_IF(begin < 0 || end < begin ||
                    end > static_cast<int32_t>(column_indices.size()),
                "Sparse Jacobi CSR row offsets are not monotone at row {}.",
                row);
    bool found = false;
    T diagonal = static_cast<T>(0);
    for (int32_t offset = begin; offset < end; ++offset) {
      const int32_t column = column_indices[offset];
      TI_ERROR_IF(column < 0 || column >= rows,
                  "Sparse Jacobi CSR column {} at offset {} is outside [0, "
                  "{}).",
                  column, offset, rows);
      if (column == row) {
        TI_ERROR_IF(found,
                    "Sparse Jacobi CSR row {} contains duplicate diagonal "
                    "entries.",
                    row);
        found = true;
        diagonal = values[offset];
        if (diagonal_offsets) {
          (*diagonal_offsets)[row] = offset;
        }
      }
    }
    TI_ERROR_IF(!found,
                "Sparse Jacobi CSR row {} has no stored diagonal entry.",
                row);
    inverse[row] = invert_diagonal_value(diagonal, row);
  }
  return inverse;
}

template <typename T>
std::vector<T> extract_host_csr_inverse(
    const std::vector<int32_t> &row_offsets,
    const std::vector<int32_t> &column_indices,
    const std::vector<T> &values,
    int rows,
    std::vector<int32_t> *diagonal_offsets = nullptr) {
  return extract_csr_inverse(row_offsets, column_indices, values.data(),
                             values.size(), rows, diagonal_offsets);
}

template <typename T>
std::vector<T> extract_cpu_csr_inverse(
    SparseMatrix &matrix,
    int rows,
    std::vector<int32_t> *diagonal_offsets = nullptr) {
  if (auto *fixed = dynamic_cast<CpuSparseCsrMatrix *>(&matrix)) {
    return extract_csr_inverse(
        fixed->get_row_offsets(), fixed->get_column_indices(),
        static_cast<const T *>(fixed->get_values()),
        static_cast<std::size_t>(fixed->num_nonzero()), rows,
        diagonal_offsets);
  }

  using RowMajor = Eigen::SparseMatrix<T, Eigen::RowMajor>;
  auto *typed = dynamic_cast<EigenSparseMatrix<RowMajor> *>(&matrix);
  TI_ERROR_IF(!typed,
              "CPU Sparse Jacobi plans require an f32/f64 row-major Eigen "
              "CSR or fixed CSR matrix.");
  const auto *eigen =
      static_cast<const RowMajor *>(typed->get_matrix());
  std::vector<T> inverse(static_cast<std::size_t>(rows));
  if (diagonal_offsets) {
    diagonal_offsets->assign(static_cast<std::size_t>(rows), -1);
  }
  for (int row = 0; row < rows; ++row) {
    bool found = false;
    T diagonal = static_cast<T>(0);
    for (typename RowMajor::InnerIterator entry(*eigen, row); entry;
         ++entry) {
      if (entry.col() == row) {
        found = true;
        diagonal = entry.value();
        break;
      }
    }
    TI_ERROR_IF(!found,
                "Sparse Jacobi CSR row {} has no stored diagonal entry.",
                row);
    inverse[row] = invert_diagonal_value(diagonal, row);
  }
  return inverse;
}

template <typename T>
std::vector<T> refresh_cpu_csr_inverse(
    SparseMatrix &matrix,
    int rows,
    const std::vector<int32_t> &diagonal_offsets) {
  if (auto *fixed = dynamic_cast<CpuSparseCsrMatrix *>(&matrix)) {
    TI_ERROR_IF(
        diagonal_offsets.size() != static_cast<std::size_t>(rows),
        "CPU Sparse Jacobi numeric refresh has invalid fixed-CSR metadata.");
    const int nnz = fixed->num_nonzero();
    const auto *values = static_cast<const T *>(fixed->get_values());
    std::vector<T> inverse(static_cast<std::size_t>(rows));
    for (int row = 0; row < rows; ++row) {
      const int32_t offset = diagonal_offsets[row];
      TI_ERROR_IF(offset < 0 || offset >= nnz,
                  "Sparse Jacobi diagonal offset for row {} is outside the "
                  "fixed values array.",
                  row);
      inverse[row] = invert_diagonal_value(values[offset], row);
    }
    return inverse;
  }
  return extract_cpu_csr_inverse<T>(matrix, rows);
}

template <typename T>
std::vector<T> invert_dense_block(const T *block,
                                  int block_size,
                                  int block_row) {
  const int augmented_columns = 2 * block_size;
  std::vector<double> augmented(
      static_cast<std::size_t>(block_size * augmented_columns), 0.0);
  for (int row = 0; row < block_size; ++row) {
    for (int column = 0; column < block_size; ++column) {
      const T value = block[row * block_size + column];
      TI_ERROR_IF(!std::isfinite(value),
                  "Sparse block-Jacobi diagonal block {} contains a "
                  "non-finite value.",
                  block_row);
      augmented[row * augmented_columns + column] = value;
    }
    augmented[row * augmented_columns + block_size + row] = 1.0;
  }

  for (int pivot = 0; pivot < block_size; ++pivot) {
    int selected_row = pivot;
    double selected_magnitude =
        std::abs(augmented[pivot * augmented_columns + pivot]);
    for (int row = pivot + 1; row < block_size; ++row) {
      const double magnitude =
          std::abs(augmented[row * augmented_columns + pivot]);
      if (magnitude > selected_magnitude) {
        selected_magnitude = magnitude;
        selected_row = row;
      }
    }
    TI_ERROR_IF(selected_magnitude == 0.0 ||
                    !std::isfinite(selected_magnitude),
                "Sparse block-Jacobi diagonal block {} is singular.",
                block_row);
    if (selected_row != pivot) {
      for (int column = 0; column < augmented_columns; ++column) {
        std::swap(augmented[pivot * augmented_columns + column],
                  augmented[selected_row * augmented_columns + column]);
      }
    }
    const double pivot_value =
        augmented[pivot * augmented_columns + pivot];
    for (int column = 0; column < augmented_columns; ++column) {
      augmented[pivot * augmented_columns + column] /= pivot_value;
    }
    for (int row = 0; row < block_size; ++row) {
      if (row == pivot) {
        continue;
      }
      const double scale = augmented[row * augmented_columns + pivot];
      for (int column = 0; column < augmented_columns; ++column) {
        augmented[row * augmented_columns + column] -=
            scale * augmented[pivot * augmented_columns + column];
      }
    }
  }

  std::vector<T> inverse(
      static_cast<std::size_t>(block_size * block_size));
  for (int row = 0; row < block_size; ++row) {
    for (int column = 0; column < block_size; ++column) {
      const double value = augmented[
          row * augmented_columns + block_size + column];
      const T stored = static_cast<T>(value);
      TI_ERROR_IF(!std::isfinite(value) || !std::isfinite(stored),
                  "Sparse block-Jacobi inverse block {} is not finite.",
                  block_row);
      inverse[row * block_size + column] = stored;
    }
  }
  return inverse;
}

template <typename T>
std::vector<T> extract_host_bsr_inverse_blocks(
    const std::vector<int32_t> &row_offsets,
    const std::vector<int32_t> &column_indices,
    const std::vector<T> &values,
    int block_rows,
    int block_size,
    std::vector<int32_t> *diagonal_offsets = nullptr) {
  const std::size_t block_width =
      static_cast<std::size_t>(block_size * block_size);
  TI_ERROR_IF(
      row_offsets.size() != static_cast<std::size_t>(block_rows) + 1 ||
          values.size() != column_indices.size() * block_width,
      "Sparse block-Jacobi received inconsistent BSR host storage.");
  TI_ERROR_IF(row_offsets.front() != 0 ||
                  row_offsets.back() !=
                      static_cast<int32_t>(column_indices.size()),
              "Sparse block-Jacobi BSR row offsets must start at zero and "
              "end at block nnz.");
  std::vector<T> inverse_blocks(
      static_cast<std::size_t>(block_rows) * block_width);
  if (diagonal_offsets) {
    diagonal_offsets->assign(static_cast<std::size_t>(block_rows), -1);
  }
  for (int block_row = 0; block_row < block_rows; ++block_row) {
    const int32_t begin = row_offsets[block_row];
    const int32_t end = row_offsets[block_row + 1];
    TI_ERROR_IF(begin < 0 || end < begin ||
                    end > static_cast<int32_t>(column_indices.size()),
                "Sparse block-Jacobi BSR row offsets are not monotone at "
                "block row {}.",
                block_row);
    int32_t diagonal_offset = -1;
    for (int32_t offset = begin; offset < end; ++offset) {
      const int32_t column = column_indices[offset];
      TI_ERROR_IF(column < 0 || column >= block_rows,
                  "Sparse block-Jacobi BSR column {} at offset {} is outside "
                  "[0, {}).",
                  column, offset, block_rows);
      if (column == block_row) {
        TI_ERROR_IF(diagonal_offset >= 0,
                    "Sparse block-Jacobi block row {} contains duplicate "
                    "diagonal blocks.",
                    block_row);
        diagonal_offset = offset;
      }
    }
    TI_ERROR_IF(diagonal_offset < 0,
                "Sparse block-Jacobi block row {} has no stored diagonal "
                "block.",
                block_row);
    if (diagonal_offsets) {
      (*diagonal_offsets)[block_row] = diagonal_offset;
    }
    auto inverse = invert_dense_block(
        values.data() + static_cast<std::size_t>(diagonal_offset) *
                            block_width,
        block_size, block_row);
    std::copy(inverse.begin(), inverse.end(),
              inverse_blocks.begin() +
                  static_cast<std::size_t>(block_row) * block_width);
  }
  return inverse_blocks;
}

template <typename T>
std::vector<T> invert_bsr_diagonal_blocks(
    const T *values,
    int block_nnz,
    const std::vector<int32_t> &diagonal_block_offsets,
    int block_rows,
    int block_size) {
  const std::size_t block_width =
      static_cast<std::size_t>(block_size * block_size);
  TI_ERROR_IF(
      block_nnz <= 0 ||
          diagonal_block_offsets.size() !=
              static_cast<std::size_t>(block_rows),
      "Sparse block-Jacobi numeric refresh has invalid fixed-BSR "
      "metadata.");
  std::vector<T> inverse_blocks(
      static_cast<std::size_t>(block_rows) * block_width);
  for (int block_row = 0; block_row < block_rows; ++block_row) {
    const int32_t offset = diagonal_block_offsets[block_row];
    TI_ERROR_IF(offset < 0 || offset >= block_nnz,
                "Sparse block-Jacobi diagonal block offset for row {} is "
                "outside the fixed values array.",
                block_row);
    auto inverse = invert_dense_block(
        values + static_cast<std::size_t>(offset) * block_width,
        block_size, block_row);
    std::copy(inverse.begin(), inverse.end(),
              inverse_blocks.begin() +
                  static_cast<std::size_t>(block_row) * block_width);
  }
  return inverse_blocks;
}

template <typename T>
void cpu_block_diagonal_apply(const T *inverse_blocks,
                              const T *input,
                              T *output,
                              int block_rows,
                              int block_size) {
  const std::size_t block_width =
      static_cast<std::size_t>(block_size * block_size);
  for (int block_row = 0; block_row < block_rows; ++block_row) {
    T local_input[12];
    const std::size_t base =
        static_cast<std::size_t>(block_row) * block_size;
    for (int local_row = 0; local_row < block_size; ++local_row) {
      local_input[local_row] = input[base + local_row];
    }
    const T *inverse =
        inverse_blocks + static_cast<std::size_t>(block_row) * block_width;
    for (int local_row = 0; local_row < block_size; ++local_row) {
      T sum = static_cast<T>(0);
      for (int local_column = 0; local_column < block_size;
           ++local_column) {
        sum += inverse[local_row * block_size + local_column] *
               local_input[local_column];
      }
      output[base + local_row] = sum;
    }
  }
}

}  // namespace

SparseJacobiPreconditionerPlan::SparseJacobiPreconditionerPlan(
    Program *program,
    SparseMatrix &matrix)
    : program_(program), matrix_(&matrix) {
  TI_ERROR_IF(!program_,
              "Sparse Jacobi plans require an active Taichi Program.");
  const auto operator_stats = matrix.debug_runtime_statistics();
  backend_family_ = operator_stats.backend_family;
  dtype_ = matrix.get_data_type();
  TI_ERROR_IF(matrix.num_rows() <= 0 ||
                  matrix.num_rows() != matrix.num_cols(),
              "Sparse Jacobi plans require a positive square matrix, got {} "
              "x {}.",
              matrix.num_rows(), matrix.num_cols());
  TI_ERROR_IF(operator_stats.storage_format != "csr",
              "Sparse Jacobi plans currently require fixed CSR storage, got "
              "{}.",
              operator_stats.storage_format);

  rows_ = matrix.num_rows();
  pattern_version_at_build_ = operator_stats.pattern_version;
  numeric_version_at_build_ = operator_stats.numeric_version;

  if (backend_family_ == "cpu") {
    TI_ERROR_IF(program_->compile_config().arch != Arch::x64 &&
                    program_->compile_config().arch != Arch::arm64,
                "CPU Sparse Jacobi matrix and Program backend do not match.");
    TI_ERROR_IF(dtype_ != PrimitiveType::f32 &&
                    dtype_ != PrimitiveType::f64,
                "CPU Sparse Jacobi plans require f32 or f64 matrices, got "
                "{}.",
                data_type_name(dtype_));
    if (dtype_ == PrimitiveType::f32) {
      host_inverse_f32_ = extract_cpu_csr_inverse<float32>(
          matrix, rows_, &diagonal_offsets_);
    } else {
      host_inverse_f64_ = extract_cpu_csr_inverse<float64>(
          matrix, rows_, &diagonal_offsets_);
    }
    return;
  }

  TI_ERROR_IF(dtype_ != PrimitiveType::f32,
              "{} Sparse Jacobi plans currently require f32 matrices, got "
              "{}.",
              backend_family_, data_type_name(dtype_));
  const int nnz = matrix.num_nonzero();
  TI_ERROR_IF(nnz <= 0,
              "Sparse Jacobi plans require at least one stored value.");
  std::vector<int32_t> row_offsets(static_cast<std::size_t>(rows_) + 1);
  std::vector<int32_t> column_indices(static_cast<std::size_t>(nnz));
  std::vector<float32> values(static_cast<std::size_t>(nnz));
  const std::size_t row_bytes = row_offsets.size() * sizeof(int32_t);
  const std::size_t column_bytes = column_indices.size() * sizeof(int32_t);
  const std::size_t value_bytes = values.size() * sizeof(float32);

  if (backend_family_ == "cuda") {
#if defined(TI_WITH_CUDA)
    TI_ERROR_IF(program_->compile_config().arch != Arch::cuda,
                "CUDA Sparse Jacobi matrix and Program backend do not "
                "match.");
    auto *cuda_matrix = dynamic_cast<CuSparseMatrix *>(&matrix);
    TI_ERROR_IF(!cuda_matrix,
                "CUDA Sparse Jacobi plans currently require scalar CSR "
                "matrices.");
    auto &driver = CUDADriver::get_instance();
    driver.memcpy_device_to_host(row_offsets.data(),
                                 cuda_matrix->get_row_ptr(), row_bytes);
    driver.memcpy_device_to_host(column_indices.data(),
                                 cuda_matrix->get_col_ind(), column_bytes);
    driver.memcpy_device_to_host(values.data(), cuda_matrix->get_val_ptr(),
                                 value_bytes);
#else
    TI_NOT_IMPLEMENTED;
#endif
  } else if (backend_family_ == "vulkan") {
#if defined(TI_WITH_VULKAN)
    TI_ERROR_IF(program_->compile_config().arch != Arch::vulkan,
                "Vulkan Sparse Jacobi matrix and Program backend do not "
                "match.");
    auto *vulkan_matrix = dynamic_cast<VulkanSparseMatrix *>(&matrix);
    TI_ERROR_IF(!vulkan_matrix,
                "Vulkan Sparse Jacobi plans require internal fixed CSR "
                "matrices.");
    // Value-only updates are queued D2D copies. Submit and complete them once
    // before the construction-only host validation snapshot.
    program_->synchronize();
    construction_host_synchronizations_ = 1;
    program_->copy_ndarray_to_host(
        const_cast<Ndarray *>(vulkan_matrix->get_row_offsets()),
        row_offsets.data(), row_bytes);
    program_->copy_ndarray_to_host(
        const_cast<Ndarray *>(vulkan_matrix->get_column_indices()),
        column_indices.data(), column_bytes);
    program_->copy_ndarray_to_host(
        const_cast<Ndarray *>(vulkan_matrix->get_values()), values.data(),
        value_bytes);
#else
    TI_NOT_IMPLEMENTED;
#endif
  } else {
    TI_ERROR("Sparse Jacobi plans do not support backend {}.",
             backend_family_);
  }

  construction_device_to_host_bytes_ =
      row_bytes + column_bytes + value_bytes;
  auto inverse = extract_host_csr_inverse(
      row_offsets, column_indices, values, rows_, &diagonal_offsets_);
  try {
    device_inverse_ = program_->create_ndarray(
        PrimitiveType::f32, {rows_}, ExternalArrayLayout::kNull, false);
    program_->copy_ndarray_from_host(
        device_inverse_, inverse.data(), inverse.size() * sizeof(float32));
  } catch (...) {
    release_resources();
    throw;
  }
  construction_host_to_device_bytes_ = inverse.size() * sizeof(float32);
}

SparseJacobiPreconditionerPlan::~SparseJacobiPreconditionerPlan() {
  release_resources();
}

void SparseJacobiPreconditionerPlan::validate_compatible(
    Program *program,
    const SparseMatrix &matrix) const {
  auto matrix_guard = matrix_->acquire_numeric_access_guard();
  std::lock_guard<std::recursive_mutex> lock(apply_mutex_);
  TI_ERROR_IF(program != program_,
              "Sparse Jacobi plan must be used by its construction "
              "Program.");
  TI_ERROR_IF(&matrix != matrix_,
              "Sparse Jacobi plan and solve plan must reference the same "
              "sparse operator.");
  const auto current = matrix_->debug_runtime_statistics();
  TI_ERROR_IF(
      current.pattern_version != pattern_version_at_build_ ||
          current.numeric_version != numeric_version_at_build_,
      "Sparse Jacobi plan is stale: operator version changed from ({}, {}) "
      "to ({}, {}); rebuild the preconditioner.",
      pattern_version_at_build_, numeric_version_at_build_,
      current.pattern_version, current.numeric_version);
}

void SparseJacobiPreconditionerPlan::refresh_numeric(Program *program) {
  auto matrix_guard = matrix_->acquire_numeric_access_guard();
  std::lock_guard<std::recursive_mutex> lock(apply_mutex_);
  numeric_refresh_calls_++;
  try {
    TI_ERROR_IF(program != program_,
                "Sparse Jacobi numeric refresh must use its construction "
                "Program.");
    const auto current = matrix_->debug_runtime_statistics();
    TI_ERROR_IF(
        current.pattern_version != pattern_version_at_build_,
        "Sparse Jacobi numeric refresh requires an unchanged pattern; "
        "pattern version changed from {} to {}.",
        pattern_version_at_build_, current.pattern_version);
    if (current.numeric_version == numeric_version_at_build_) {
      numeric_refresh_noops_++;
      return;
    }

    if (backend_family_ == "cpu") {
      const auto inverse_bytes = static_cast<std::uint64_t>(rows_) *
                                 data_type_size(dtype_);
      refresh_peak_temporary_host_bytes_ = std::max(
          refresh_peak_temporary_host_bytes_, inverse_bytes);
      if (dtype_ == PrimitiveType::f32) {
        auto inverse = refresh_cpu_csr_inverse<float32>(
            *matrix_, rows_, diagonal_offsets_);
        host_inverse_f32_.swap(inverse);
      } else {
        auto inverse = refresh_cpu_csr_inverse<float64>(
            *matrix_, rows_, diagonal_offsets_);
        host_inverse_f64_.swap(inverse);
      }
      numeric_version_at_build_ = current.numeric_version;
      numeric_refresh_successes_++;
      return;
    }

    std::vector<float32> inverse(static_cast<std::size_t>(rows_));
    const int nnz = matrix_->num_nonzero();
    TI_ERROR_IF(
        nnz <= 0 ||
            diagonal_offsets_.size() != static_cast<std::size_t>(rows_),
        "Sparse Jacobi numeric refresh has invalid fixed-CSR metadata.");
    std::vector<float32> values(static_cast<std::size_t>(nnz));
    const std::size_t value_bytes = values.size() * sizeof(float32);
    refresh_peak_temporary_host_bytes_ = std::max(
        refresh_peak_temporary_host_bytes_,
        static_cast<std::uint64_t>(value_bytes +
                                   inverse.size() * sizeof(float32)));
    if (backend_family_ == "cuda") {
#if defined(TI_WITH_CUDA)
      auto *cuda_matrix = dynamic_cast<CuSparseMatrix *>(matrix_);
      TI_ERROR_IF(!cuda_matrix,
                  "CUDA Sparse Jacobi numeric refresh requires scalar "
                  "CSR storage.");
      CUDADriver::get_instance().memcpy_device_to_host(
          values.data(), cuda_matrix->get_val_ptr(), value_bytes);
      refresh_device_to_host_bytes_ += value_bytes;
#else
      TI_NOT_IMPLEMENTED;
#endif
    } else if (backend_family_ == "vulkan") {
#if defined(TI_WITH_VULKAN)
      auto *vulkan_matrix = dynamic_cast<VulkanSparseMatrix *>(matrix_);
      TI_ERROR_IF(!vulkan_matrix,
                  "Vulkan Sparse Jacobi numeric refresh requires internal "
                  "fixed CSR storage.");
      program_->synchronize();
      refresh_host_synchronizations_++;
      program_->copy_ndarray_to_host(
          const_cast<Ndarray *>(vulkan_matrix->get_values()), values.data(),
          value_bytes);
      refresh_device_to_host_bytes_ += value_bytes;
#else
      TI_NOT_IMPLEMENTED;
#endif
    } else {
      TI_ERROR("Sparse Jacobi numeric refresh does not support backend {}.",
               backend_family_);
    }
    for (int row = 0; row < rows_; ++row) {
      const int32_t offset = diagonal_offsets_[row];
      TI_ERROR_IF(offset < 0 || offset >= nnz,
                  "Sparse Jacobi diagonal offset for row {} is outside the "
                  "fixed values array.",
                  row);
      inverse[row] = invert_diagonal_value(values[offset], row);
    }

    Ndarray *replacement = nullptr;
    try {
      replacement = program_->create_ndarray(
          PrimitiveType::f32, {rows_}, ExternalArrayLayout::kNull, false);
      refresh_peak_temporary_device_bytes_ = std::max(
          refresh_peak_temporary_device_bytes_,
          static_cast<std::uint64_t>(inverse.size() * sizeof(float32)));
      program_->copy_ndarray_from_host(
          replacement, inverse.data(), inverse.size() * sizeof(float32));
    } catch (...) {
      if (replacement) {
        program_->delete_ndarray(replacement);
      }
      throw;
    }
    Ndarray *old_inverse = device_inverse_;
    device_inverse_ = replacement;
    if (old_inverse) {
      program_->delete_ndarray(old_inverse);
    }
    refresh_host_to_device_bytes_ += inverse.size() * sizeof(float32);
    numeric_version_at_build_ = current.numeric_version;
    numeric_refresh_successes_++;
  } catch (...) {
    numeric_refresh_failures_++;
    throw;
  }
}

void SparseJacobiPreconditionerPlan::release_resources() {
  if (device_inverse_ && program_) {
    program_->delete_ndarray(device_inverse_);
    device_inverse_ = nullptr;
  }
}

void SparseJacobiPreconditionerPlan::apply(Program *program,
                                           const Ndarray &input,
                                           const Ndarray &output) {
  TI_ERROR_IF(program != program_,
              "Sparse Jacobi plan must be applied by its construction "
              "Program.");
  validate_vector("input", input, rows_, dtype_);
  validate_vector("output", output, rows_, dtype_);
  if (backend_family_ == "cpu") {
    apply_cpu_raw(program, program_->get_ndarray_data_ptr_as_int(&input),
                  program_->get_ndarray_data_ptr_as_int(&output));
    return;
  }
  auto matrix_guard = matrix_->acquire_numeric_access_guard();
  std::lock_guard<std::recursive_mutex> lock(apply_mutex_);
  const auto current = matrix_->debug_runtime_statistics();
  TI_ERROR_IF(
      current.pattern_version != pattern_version_at_build_ ||
          current.numeric_version != numeric_version_at_build_,
      "Sparse Jacobi plan is stale: operator version changed from ({}, {}) "
      "to ({}, {}); rebuild the preconditioner.",
      pattern_version_at_build_, numeric_version_at_build_,
      current.pattern_version, current.numeric_version);

  if (backend_family_ == "cuda") {
#if defined(TI_WITH_CUDA)
    auto submission_guard =
        program_->acquire_runtime_resource_submission_guard();
    const Ndarray *resources[] = {device_inverse_, &input, &output};
    program_->retain_ndarrays_for_external_submission(resources,
                                                       std::size(resources));
    cuda::driver_sparse_diagonal_apply_f32(
        reinterpret_cast<void *>(
            program_->get_ndarray_data_ptr_as_int(device_inverse_)),
        reinterpret_cast<void *>(
            program_->get_ndarray_data_ptr_as_int(&input)),
        reinterpret_cast<void *>(
            program_->get_ndarray_data_ptr_as_int(&output)),
        rows_, nullptr);
#else
    TI_NOT_IMPLEMENTED;
#endif
  } else if (backend_family_ == "vulkan") {
    program_->vulkan_sparse_diagonal_apply(
        device_inverse_, const_cast<Ndarray *>(&input),
        const_cast<Ndarray *>(&output), rows_);
  } else {
    TI_ERROR("Sparse Jacobi plan has unsupported backend {}.",
             backend_family_);
  }
  apply_calls_++;
}

void SparseJacobiPreconditionerPlan::apply_cpu_raw(
    Program *program,
    std::uintptr_t input,
    std::uintptr_t output) {
  TI_ERROR_IF(program != program_,
              "Sparse Jacobi plan must be applied by its construction "
              "Program.");
  TI_ERROR_IF(backend_family_ != "cpu" ||
                  !arch_is_cpu(program->compile_config().arch) || input == 0 ||
                  output == 0,
              "CPU Sparse Jacobi raw apply requires CPU storage and non-null "
              "input/output pointers.");
  auto matrix_guard = matrix_->acquire_numeric_access_guard();
  std::lock_guard<std::recursive_mutex> lock(apply_mutex_);
  const auto current = matrix_->debug_runtime_statistics();
  TI_ERROR_IF(
      current.pattern_version != pattern_version_at_build_ ||
          current.numeric_version != numeric_version_at_build_,
      "Sparse Jacobi plan is stale: operator version changed from ({}, {}) "
      "to ({}, {}); rebuild or refresh the preconditioner.",
      pattern_version_at_build_, numeric_version_at_build_,
      current.pattern_version, current.numeric_version);
  if (dtype_ == PrimitiveType::f32) {
    TI_ERROR_IF(host_inverse_f32_.size() != static_cast<std::size_t>(rows_),
                "CPU Sparse Jacobi f32 inverse storage is incomplete.");
    const auto *source = reinterpret_cast<const float32 *>(input);
    auto *destination = reinterpret_cast<float32 *>(output);
    for (int row = 0; row < rows_; ++row) {
      destination[row] = host_inverse_f32_[row] * source[row];
    }
  } else {
    TI_ERROR_IF(host_inverse_f64_.size() != static_cast<std::size_t>(rows_),
                "CPU Sparse Jacobi f64 inverse storage is incomplete.");
    const auto *source = reinterpret_cast<const float64 *>(input);
    auto *destination = reinterpret_cast<float64 *>(output);
    for (int row = 0; row < rows_; ++row) {
      destination[row] = host_inverse_f64_[row] * source[row];
    }
  }
  apply_calls_++;
}

void SparseJacobiPreconditionerPlan::apply_cuda_raw(
    Program *program,
    std::uintptr_t input,
    std::uintptr_t output,
    CUstream stream) {
  TI_ERROR_IF(program != program_,
              "Sparse Jacobi plan must be applied by its construction "
              "Program.");
  TI_ERROR_IF(backend_family_ != "cuda" || input == 0 || output == 0,
              "CUDA Sparse Jacobi raw apply requires CUDA storage and "
              "non-null input/output pointers.");
  auto matrix_guard = matrix_->acquire_numeric_access_guard();
  std::lock_guard<std::recursive_mutex> lock(apply_mutex_);
  const auto current = matrix_->debug_runtime_statistics();
  TI_ERROR_IF(
      current.pattern_version != pattern_version_at_build_ ||
          current.numeric_version != numeric_version_at_build_,
      "Sparse Jacobi plan is stale: operator version changed from ({}, {}) "
      "to ({}, {}); rebuild the preconditioner.",
      pattern_version_at_build_, numeric_version_at_build_,
      current.pattern_version, current.numeric_version);
#if defined(TI_WITH_CUDA)
  auto submission_guard =
      program_->acquire_runtime_resource_submission_guard();
  const Ndarray *resources[] = {device_inverse_};
  program_->retain_ndarrays_for_external_submission(resources,
                                                     std::size(resources));
  cuda::driver_sparse_diagonal_apply_f32(
      reinterpret_cast<void *>(
          program_->get_ndarray_data_ptr_as_int(device_inverse_)),
      reinterpret_cast<void *>(input), reinterpret_cast<void *>(output),
      rows_, stream);
#else
  TI_NOT_IMPLEMENTED;
#endif
  apply_calls_++;
}

void SparseJacobiPreconditionerPlan::record_replayed_apply_calls(
    std::uint64_t count) {
  std::lock_guard<std::recursive_mutex> lock(apply_mutex_);
  apply_calls_ += count;
}

OperatorResourceLease
SparseJacobiPreconditionerPlan::acquire_resource_lease() const {
  struct ResourcePin {
    SparseMatrix::NumericAccessGuard matrix;
    std::unique_lock<std::recursive_mutex> preconditioner;
  };
  return OperatorResourceLease::hold(ResourcePin{
      matrix_->acquire_numeric_access_guard(),
      std::unique_lock<std::recursive_mutex>(apply_mutex_)});
}

SparsePreconditionerPlanRuntimeStatistics
SparseJacobiPreconditionerPlan::debug_runtime_statistics() const {
  auto matrix_guard = matrix_->acquire_numeric_access_guard();
  std::lock_guard<std::recursive_mutex> lock(apply_mutex_);
  const auto current = matrix_->debug_runtime_statistics();
  SparsePreconditionerPlanRuntimeStatistics result;
  result.backend_family = backend_family_;
  result.dtype = data_type_name(dtype_);
  result.rows = rows_;
  result.operator_pattern_version_at_build = pattern_version_at_build_;
  result.operator_numeric_version_at_build = numeric_version_at_build_;
  result.operator_pattern_version_current = current.pattern_version;
  result.operator_numeric_version_current = current.numeric_version;
  result.operator_stale =
      current.pattern_version != pattern_version_at_build_ ||
      current.numeric_version != numeric_version_at_build_;
  result.apply_calls = apply_calls_;
  result.persistent_inverse_count = 1;
  result.persistent_inverse_reserved_bytes =
      static_cast<std::uint64_t>(rows_) * data_type_size(dtype_);
  result.construction_device_to_host_bytes =
      construction_device_to_host_bytes_;
  result.construction_host_to_device_bytes =
      construction_host_to_device_bytes_;
  result.construction_host_synchronizations =
      construction_host_synchronizations_;
  result.numeric_refresh_calls = numeric_refresh_calls_;
  result.numeric_refresh_successes = numeric_refresh_successes_;
  result.numeric_refresh_noops = numeric_refresh_noops_;
  result.numeric_refresh_failures = numeric_refresh_failures_;
  result.refresh_device_to_host_bytes = refresh_device_to_host_bytes_;
  result.refresh_host_to_device_bytes = refresh_host_to_device_bytes_;
  result.refresh_host_synchronizations =
      refresh_host_synchronizations_;
  result.refresh_peak_temporary_host_bytes =
      refresh_peak_temporary_host_bytes_;
  result.refresh_peak_temporary_device_bytes =
      refresh_peak_temporary_device_bytes_;
  result.numeric_refresh_supported = true;
  return result;
}

std::unique_ptr<SparseJacobiPreconditionerPlan>
make_sparse_jacobi_preconditioner_plan(Program *program,
                                       SparseMatrix &matrix) {
  return std::make_unique<SparseJacobiPreconditionerPlan>(program, matrix);
}

SparseBlockJacobiPreconditionerPlan::SparseBlockJacobiPreconditionerPlan(
    Program *program,
    SparseMatrix &matrix)
    : program_(program), matrix_(&matrix) {
  TI_ERROR_IF(!program_,
              "Sparse block-Jacobi plans require an active Program.");
  const auto operator_stats = matrix_->debug_runtime_statistics();
  backend_family_ = operator_stats.backend_family;
  dtype_ = matrix_->get_data_type();
  TI_ERROR_IF(operator_stats.storage_format != "bsr" ||
                  (dtype_ != PrimitiveType::f32 &&
                   dtype_ != PrimitiveType::f64),
              "Sparse block-Jacobi plans require f32 or f64 BSR storage.");
  if (backend_family_ == "cpu") {
    TI_ERROR_IF(!arch_is_cpu(program_->compile_config().arch),
                "CPU Sparse block-Jacobi matrix and Program backend do not "
                "match.");
    auto *typed = dynamic_cast<CpuSparseBsrMatrix *>(matrix_);
    TI_ERROR_IF(!typed,
                "CPU Sparse block-Jacobi plans require internal CPU BSR "
                "storage.");
    block_rows_ = typed->get_block_rows();
    block_size_ = typed->get_block_size();
    block_nnz_ = typed->get_block_nnz();
  } else if (backend_family_ == "cuda") {
#if defined(TI_WITH_CUDA)
    TI_ERROR_IF(dtype_ != PrimitiveType::f32,
                "CUDA Sparse block-Jacobi plans currently require f32 "
                "storage.");
    TI_ERROR_IF(program_->compile_config().arch != Arch::cuda,
                "CUDA Sparse block-Jacobi matrix and Program backend do "
                "not match.");
    auto *typed = dynamic_cast<CuSparseBsrMatrix *>(matrix_);
    TI_ERROR_IF(!typed,
                "CUDA Sparse block-Jacobi plans require internal CUDA BSR "
                "storage.");
    block_rows_ = typed->get_block_rows();
    block_size_ = typed->get_block_size();
    block_nnz_ = typed->get_block_nnz();
#else
    TI_NOT_IMPLEMENTED;
#endif
  } else if (backend_family_ == "vulkan") {
#if defined(TI_WITH_VULKAN)
    TI_ERROR_IF(dtype_ != PrimitiveType::f32,
                "Vulkan Sparse block-Jacobi plans currently require f32 "
                "storage.");
    TI_ERROR_IF(program_->compile_config().arch != Arch::vulkan,
                "Vulkan Sparse block-Jacobi matrix and Program backend do "
                "not match.");
    auto *typed = dynamic_cast<VulkanSparseBsrMatrix *>(matrix_);
    TI_ERROR_IF(!typed,
                "Vulkan Sparse block-Jacobi plans require internal Vulkan "
                "BSR storage.");
    block_rows_ = typed->get_block_rows();
    block_size_ = typed->get_block_size();
    block_nnz_ = typed->get_block_nnz();
#else
    TI_NOT_IMPLEMENTED;
#endif
  } else {
    TI_ERROR("Sparse block-Jacobi plans do not support backend {}.",
             backend_family_);
  }
  rows_ = matrix_->num_rows();
  TI_ERROR_IF(block_rows_ <= 0 ||
                  block_nnz_ <= 0 ||
                  (block_size_ != 2 && block_size_ != 3 &&
                   block_size_ != 6 && block_size_ != 12) ||
                  rows_ != block_rows_ * block_size_ ||
                  matrix_->num_cols() != rows_,
              "Sparse block-Jacobi received invalid BSR geometry.");
  pattern_version_at_build_ = operator_stats.pattern_version;
  numeric_version_at_build_ = operator_stats.numeric_version;

  const std::size_t block_width =
      static_cast<std::size_t>(block_size_ * block_size_);
  const std::size_t value_count =
      static_cast<std::size_t>(block_nnz_) * block_width;
  std::vector<int32_t> row_offsets(
      static_cast<std::size_t>(block_rows_) + 1);
  std::vector<int32_t> column_indices(
      static_cast<std::size_t>(block_nnz_));
  std::vector<float32> values_f32;
  std::vector<float64> values_f64;
  if (dtype_ == PrimitiveType::f32) {
    values_f32.resize(value_count);
  } else {
    values_f64.resize(value_count);
  }
  const std::size_t row_bytes = row_offsets.size() * sizeof(int32_t);
  const std::size_t column_bytes = column_indices.size() * sizeof(int32_t);
  const std::size_t value_bytes = value_count * data_type_size(dtype_);
  if (backend_family_ == "cpu") {
    auto *typed = static_cast<CpuSparseBsrMatrix *>(matrix_);
    row_offsets = typed->get_block_row_offsets();
    column_indices = typed->get_block_column_indices();
    if (dtype_ == PrimitiveType::f32) {
      std::memcpy(values_f32.data(), typed->get_block_values(),
                  value_bytes);
    } else {
      std::memcpy(values_f64.data(), typed->get_block_values(),
                  value_bytes);
    }
  } else if (backend_family_ == "cuda") {
#if defined(TI_WITH_CUDA)
    auto *typed = static_cast<CuSparseBsrMatrix *>(matrix_);
    auto &driver = CUDADriver::get_instance();
    driver.memcpy_device_to_host(row_offsets.data(),
                                 typed->get_block_row_offsets(), row_bytes);
    driver.memcpy_device_to_host(
        column_indices.data(), typed->get_block_column_indices(),
        column_bytes);
    driver.memcpy_device_to_host(values_f32.data(),
                                 typed->get_block_values(), value_bytes);
#else
    TI_NOT_IMPLEMENTED;
#endif
  } else {
#if defined(TI_WITH_VULKAN)
    auto *typed = static_cast<VulkanSparseBsrMatrix *>(matrix_);
    program_->synchronize();
    construction_host_synchronizations_ = 1;
    program_->copy_ndarray_to_host(
        const_cast<Ndarray *>(typed->get_block_row_offsets()),
        row_offsets.data(), row_bytes);
    program_->copy_ndarray_to_host(
        const_cast<Ndarray *>(typed->get_block_column_indices()),
        column_indices.data(), column_bytes);
    program_->copy_ndarray_to_host(
        const_cast<Ndarray *>(typed->get_block_values()), values_f32.data(),
        value_bytes);
#else
    TI_NOT_IMPLEMENTED;
#endif
  }
  if (backend_family_ != "cpu") {
    construction_device_to_host_bytes_ =
        row_bytes + column_bytes + value_bytes;
  }
  if (dtype_ == PrimitiveType::f32) {
    auto inverse_blocks = extract_host_bsr_inverse_blocks(
        row_offsets, column_indices, values_f32, block_rows_, block_size_,
        &diagonal_block_offsets_);
    if (backend_family_ == "cpu") {
      host_inverse_blocks_f32_ = std::move(inverse_blocks);
      return;
    }
    try {
      device_inverse_blocks_ = program_->create_ndarray(
          PrimitiveType::f32,
          {static_cast<int>(inverse_blocks.size())},
          ExternalArrayLayout::kNull, false);
      program_->copy_ndarray_from_host(
          device_inverse_blocks_, inverse_blocks.data(),
          inverse_blocks.size() * sizeof(float32));
    } catch (...) {
      release_resources();
      throw;
    }
    construction_host_to_device_bytes_ =
        inverse_blocks.size() * sizeof(float32);
    return;
  }
  host_inverse_blocks_f64_ = extract_host_bsr_inverse_blocks(
      row_offsets, column_indices, values_f64, block_rows_, block_size_,
      &diagonal_block_offsets_);
}

SparseBlockJacobiPreconditionerPlan::~SparseBlockJacobiPreconditionerPlan() {
  release_resources();
}

void SparseBlockJacobiPreconditionerPlan::validate_compatible(
    Program *program,
    const SparseMatrix &matrix) const {
  auto matrix_guard = matrix_->acquire_numeric_access_guard();
  std::lock_guard<std::recursive_mutex> lock(apply_mutex_);
  TI_ERROR_IF(program != program_,
              "Sparse block-Jacobi plan must be used by its construction "
              "Program.");
  TI_ERROR_IF(&matrix != matrix_,
              "Sparse block-Jacobi plan and solve plan must reference the "
              "same sparse operator.");
  const auto current = matrix_->debug_runtime_statistics();
  TI_ERROR_IF(
      current.pattern_version != pattern_version_at_build_ ||
          current.numeric_version != numeric_version_at_build_,
      "Sparse block-Jacobi plan is stale: operator version changed from "
      "({}, {}) to ({}, {}); rebuild the preconditioner.",
      pattern_version_at_build_, numeric_version_at_build_,
      current.pattern_version, current.numeric_version);
}

void SparseBlockJacobiPreconditionerPlan::refresh_numeric(
    Program *program) {
  auto matrix_guard = matrix_->acquire_numeric_access_guard();
  std::lock_guard<std::recursive_mutex> lock(apply_mutex_);
  numeric_refresh_calls_++;
  try {
    TI_ERROR_IF(
        program != program_,
        "Sparse block-Jacobi numeric refresh must use its construction "
        "Program.");
    const auto current = matrix_->debug_runtime_statistics();
    TI_ERROR_IF(
        current.pattern_version != pattern_version_at_build_,
        "Sparse block-Jacobi numeric refresh requires an unchanged "
        "pattern; pattern version changed from {} to {}.",
        pattern_version_at_build_, current.pattern_version);
    if (current.numeric_version == numeric_version_at_build_) {
      numeric_refresh_noops_++;
      return;
    }

    const int block_nnz = block_nnz_;
    const std::size_t block_width =
        static_cast<std::size_t>(block_size_ * block_size_);
    TI_ERROR_IF(
        block_nnz <= 0 ||
            diagonal_block_offsets_.size() !=
                static_cast<std::size_t>(block_rows_),
        "Sparse block-Jacobi numeric refresh has invalid fixed-BSR "
        "metadata.");
    if (backend_family_ == "cpu") {
      auto *typed = static_cast<CpuSparseBsrMatrix *>(matrix_);
      const std::size_t inverse_count =
          static_cast<std::size_t>(block_rows_) * block_width;
      const std::size_t inverse_bytes =
          inverse_count * data_type_size(dtype_);
      refresh_peak_temporary_host_bytes_ = std::max(
          refresh_peak_temporary_host_bytes_,
          static_cast<std::uint64_t>(inverse_bytes));
      if (dtype_ == PrimitiveType::f32) {
        auto inverse_blocks = invert_bsr_diagonal_blocks(
            static_cast<const float32 *>(typed->get_block_values()),
            block_nnz, diagonal_block_offsets_, block_rows_, block_size_);
        host_inverse_blocks_f32_.swap(inverse_blocks);
      } else {
        auto inverse_blocks = invert_bsr_diagonal_blocks(
            static_cast<const float64 *>(typed->get_block_values()),
            block_nnz, diagonal_block_offsets_, block_rows_, block_size_);
        host_inverse_blocks_f64_.swap(inverse_blocks);
      }
      numeric_version_at_build_ = current.numeric_version;
      numeric_refresh_successes_++;
      return;
    }
    std::vector<float32> values(
        static_cast<std::size_t>(block_nnz) * block_width);
    std::vector<float32> inverse_blocks(
        static_cast<std::size_t>(block_rows_) * block_width);
    const std::size_t value_bytes = values.size() * sizeof(float32);
    const std::size_t inverse_bytes =
        inverse_blocks.size() * sizeof(float32);
    refresh_peak_temporary_host_bytes_ = std::max(
        refresh_peak_temporary_host_bytes_,
        static_cast<std::uint64_t>(value_bytes + inverse_bytes));
    if (backend_family_ == "cuda") {
#if defined(TI_WITH_CUDA)
      auto *typed = static_cast<CuSparseBsrMatrix *>(matrix_);
      CUDADriver::get_instance().memcpy_device_to_host(
          values.data(), typed->get_block_values(), value_bytes);
#else
      TI_NOT_IMPLEMENTED;
#endif
    } else if (backend_family_ == "vulkan") {
#if defined(TI_WITH_VULKAN)
      auto *typed = static_cast<VulkanSparseBsrMatrix *>(matrix_);
      program_->synchronize();
      refresh_host_synchronizations_++;
      program_->copy_ndarray_to_host(
          const_cast<Ndarray *>(typed->get_block_values()), values.data(),
          value_bytes);
#else
      TI_NOT_IMPLEMENTED;
#endif
    } else {
      TI_ERROR("Sparse block-Jacobi numeric refresh does not support "
               "backend {}.",
               backend_family_);
    }
    refresh_device_to_host_bytes_ += value_bytes;

    for (int block_row = 0; block_row < block_rows_; ++block_row) {
      const int32_t offset = diagonal_block_offsets_[block_row];
      TI_ERROR_IF(offset < 0 || offset >= block_nnz,
                  "Sparse block-Jacobi diagonal block offset for row {} "
                  "is outside the fixed values array.",
                  block_row);
      auto inverse = invert_dense_block(
          values.data() + static_cast<std::size_t>(offset) * block_width,
          block_size_, block_row);
      std::copy(inverse.begin(), inverse.end(),
                inverse_blocks.begin() +
                    static_cast<std::size_t>(block_row) * block_width);
    }

    Ndarray *replacement = nullptr;
    try {
      replacement = program_->create_ndarray(
          PrimitiveType::f32,
          {static_cast<int>(inverse_blocks.size())},
          ExternalArrayLayout::kNull, false);
      refresh_peak_temporary_device_bytes_ = std::max(
          refresh_peak_temporary_device_bytes_,
          static_cast<std::uint64_t>(inverse_bytes));
      program_->copy_ndarray_from_host(replacement, inverse_blocks.data(),
                                       inverse_bytes);
    } catch (...) {
      if (replacement) {
        program_->delete_ndarray(replacement);
      }
      throw;
    }
    Ndarray *old_inverse = device_inverse_blocks_;
    device_inverse_blocks_ = replacement;
    if (old_inverse) {
      program_->delete_ndarray(old_inverse);
    }
    refresh_host_to_device_bytes_ += inverse_bytes;
    numeric_version_at_build_ = current.numeric_version;
    numeric_refresh_successes_++;
  } catch (...) {
    numeric_refresh_failures_++;
    throw;
  }
}

void SparseBlockJacobiPreconditionerPlan::release_resources() {
  if (device_inverse_blocks_ && program_) {
    program_->delete_ndarray(device_inverse_blocks_);
    device_inverse_blocks_ = nullptr;
  }
}

void SparseBlockJacobiPreconditionerPlan::apply(
    Program *program,
    const Ndarray &input,
    const Ndarray &output) {
  TI_ERROR_IF(program != program_,
              "Sparse block-Jacobi plan must be applied by its construction "
              "Program.");
  validate_vector("block-Jacobi input", input, rows_, dtype_);
  validate_vector("block-Jacobi output", output, rows_, dtype_);
  if (backend_family_ == "cpu") {
    apply_cpu_raw(program, program_->get_ndarray_data_ptr_as_int(&input),
                  program_->get_ndarray_data_ptr_as_int(&output));
    return;
  }
  auto matrix_guard = matrix_->acquire_numeric_access_guard();
  std::lock_guard<std::recursive_mutex> lock(apply_mutex_);
  const auto current = matrix_->debug_runtime_statistics();
  TI_ERROR_IF(
      current.pattern_version != pattern_version_at_build_ ||
          current.numeric_version != numeric_version_at_build_,
      "Sparse block-Jacobi plan is stale: operator version changed from ({}, "
      "{}) to ({}, {}); rebuild the preconditioner.",
      pattern_version_at_build_, numeric_version_at_build_,
      current.pattern_version, current.numeric_version);
  if (backend_family_ == "cuda") {
#if defined(TI_WITH_CUDA)
    auto submission_guard =
        program_->acquire_runtime_resource_submission_guard();
    const Ndarray *resources[] = {device_inverse_blocks_, &input, &output};
    program_->retain_ndarrays_for_external_submission(resources,
                                                       std::size(resources));
    cuda::driver_sparse_block_diagonal_apply_f32(
        reinterpret_cast<void *>(
            program_->get_ndarray_data_ptr_as_int(device_inverse_blocks_)),
        reinterpret_cast<void *>(
            program_->get_ndarray_data_ptr_as_int(&input)),
        reinterpret_cast<void *>(
            program_->get_ndarray_data_ptr_as_int(&output)),
        block_rows_, block_size_, nullptr);
#else
    TI_NOT_IMPLEMENTED;
#endif
  } else if (backend_family_ == "vulkan") {
    program_->vulkan_sparse_block_diagonal_apply(
        device_inverse_blocks_, const_cast<Ndarray *>(&input),
        const_cast<Ndarray *>(&output), block_rows_, block_size_);
  } else {
    TI_ERROR("Sparse block-Jacobi apply does not support backend {}.",
             backend_family_);
  }
  apply_calls_++;
}

void SparseBlockJacobiPreconditionerPlan::apply_cpu_raw(
    Program *program,
    std::uintptr_t input,
    std::uintptr_t output) {
  TI_ERROR_IF(program != program_,
              "Sparse block-Jacobi plan must be applied by its "
              "construction Program.");
  TI_ERROR_IF(backend_family_ != "cpu" || input == 0 || output == 0,
              "CPU Sparse block-Jacobi raw apply requires non-null "
              "input/output pointers and CPU storage.");
  auto matrix_guard = matrix_->acquire_numeric_access_guard();
  std::lock_guard<std::recursive_mutex> lock(apply_mutex_);
  const auto current = matrix_->debug_runtime_statistics();
  TI_ERROR_IF(
      current.pattern_version != pattern_version_at_build_ ||
          current.numeric_version != numeric_version_at_build_,
      "Sparse block-Jacobi plan is stale: operator version changed from "
      "({}, {}) to ({}, {}); rebuild the preconditioner.",
      pattern_version_at_build_, numeric_version_at_build_,
      current.pattern_version, current.numeric_version);
  if (dtype_ == PrimitiveType::f32) {
    TI_ERROR_IF(host_inverse_blocks_f32_.empty(),
                "CPU Sparse block-Jacobi f32 inverse storage is empty.");
    cpu_block_diagonal_apply(
        host_inverse_blocks_f32_.data(),
        reinterpret_cast<const float32 *>(input),
        reinterpret_cast<float32 *>(output), block_rows_, block_size_);
  } else {
    TI_ERROR_IF(host_inverse_blocks_f64_.empty(),
                "CPU Sparse block-Jacobi f64 inverse storage is empty.");
    cpu_block_diagonal_apply(
        host_inverse_blocks_f64_.data(),
        reinterpret_cast<const float64 *>(input),
        reinterpret_cast<float64 *>(output), block_rows_, block_size_);
  }
  apply_calls_++;
}

void SparseBlockJacobiPreconditionerPlan::apply_cuda_raw(
    Program *program,
    std::uintptr_t input,
    std::uintptr_t output,
    CUstream stream) {
  TI_ERROR_IF(program != program_,
              "Sparse block-Jacobi plan must be applied by its "
              "construction Program.");
  TI_ERROR_IF(backend_family_ != "cuda" || input == 0 || output == 0,
              "CUDA Sparse block-Jacobi raw apply requires non-null "
              "input/output pointers and CUDA storage.");
  auto matrix_guard = matrix_->acquire_numeric_access_guard();
  std::lock_guard<std::recursive_mutex> lock(apply_mutex_);
  const auto current = matrix_->debug_runtime_statistics();
  TI_ERROR_IF(
      current.pattern_version != pattern_version_at_build_ ||
          current.numeric_version != numeric_version_at_build_,
      "Sparse block-Jacobi plan is stale: operator version changed from "
      "({}, {}) to ({}, {}); rebuild the preconditioner.",
      pattern_version_at_build_, numeric_version_at_build_,
      current.pattern_version, current.numeric_version);
#if defined(TI_WITH_CUDA)
  auto submission_guard =
      program_->acquire_runtime_resource_submission_guard();
  const Ndarray *resources[] = {device_inverse_blocks_};
  program_->retain_ndarrays_for_external_submission(resources,
                                                     std::size(resources));
  cuda::driver_sparse_block_diagonal_apply_f32(
      reinterpret_cast<void *>(
          program_->get_ndarray_data_ptr_as_int(device_inverse_blocks_)),
      reinterpret_cast<void *>(input), reinterpret_cast<void *>(output),
      block_rows_, block_size_, stream);
#else
  TI_NOT_IMPLEMENTED;
#endif
  apply_calls_++;
}

void SparseBlockJacobiPreconditionerPlan::record_replayed_apply_calls(
    std::uint64_t count) {
  std::lock_guard<std::recursive_mutex> lock(apply_mutex_);
  apply_calls_ += count;
}

OperatorResourceLease
SparseBlockJacobiPreconditionerPlan::acquire_resource_lease() const {
  struct ResourcePin {
    SparseMatrix::NumericAccessGuard matrix;
    std::unique_lock<std::recursive_mutex> preconditioner;
  };
  return OperatorResourceLease::hold(ResourcePin{
      matrix_->acquire_numeric_access_guard(),
      std::unique_lock<std::recursive_mutex>(apply_mutex_)});
}

SparsePreconditionerPlanRuntimeStatistics
SparseBlockJacobiPreconditionerPlan::debug_runtime_statistics() const {
  auto matrix_guard = matrix_->acquire_numeric_access_guard();
  std::lock_guard<std::recursive_mutex> lock(apply_mutex_);
  const auto current = matrix_->debug_runtime_statistics();
  SparsePreconditionerPlanRuntimeStatistics result;
  result.backend_family = backend_family_;
  result.method = "block_jacobi";
  result.dtype = data_type_name(dtype_);
  result.rows = rows_;
  result.block_rows = block_rows_;
  result.block_size = block_size_;
  result.operator_pattern_version_at_build = pattern_version_at_build_;
  result.operator_numeric_version_at_build = numeric_version_at_build_;
  result.operator_pattern_version_current = current.pattern_version;
  result.operator_numeric_version_current = current.numeric_version;
  result.operator_stale =
      current.pattern_version != pattern_version_at_build_ ||
      current.numeric_version != numeric_version_at_build_;
  result.apply_calls = apply_calls_;
  result.persistent_inverse_count = block_rows_;
  result.persistent_inverse_reserved_bytes =
      static_cast<std::uint64_t>(block_rows_) * block_size_ * block_size_ *
      data_type_size(dtype_);
  result.construction_device_to_host_bytes =
      construction_device_to_host_bytes_;
  result.construction_host_to_device_bytes =
      construction_host_to_device_bytes_;
  result.construction_host_synchronizations =
      construction_host_synchronizations_;
  result.numeric_refresh_calls = numeric_refresh_calls_;
  result.numeric_refresh_successes = numeric_refresh_successes_;
  result.numeric_refresh_noops = numeric_refresh_noops_;
  result.numeric_refresh_failures = numeric_refresh_failures_;
  result.refresh_device_to_host_bytes = refresh_device_to_host_bytes_;
  result.refresh_host_to_device_bytes = refresh_host_to_device_bytes_;
  result.refresh_host_synchronizations =
      refresh_host_synchronizations_;
  result.refresh_peak_temporary_host_bytes =
      refresh_peak_temporary_host_bytes_;
  result.refresh_peak_temporary_device_bytes =
      refresh_peak_temporary_device_bytes_;
  result.numeric_refresh_supported = true;
  return result;
}

std::unique_ptr<SparseBlockJacobiPreconditionerPlan>
make_sparse_block_jacobi_preconditioner_plan(Program *program,
                                             SparseMatrix &matrix) {
  return std::make_unique<SparseBlockJacobiPreconditionerPlan>(program,
                                                               matrix);
}

CompiledKernelPreconditionerPlan::CompiledKernelPreconditionerPlan(
    Program *program,
    CompiledKernelLinearOperator &target_operator,
    SparseMatrix &inverse_apply_operator,
    bool assume_symmetric_positive_definite) {
  TI_ERROR_IF(!program || !assume_symmetric_positive_definite,
              "Compiled-kernel preconditioners require an owning Program "
              "and an explicit symmetric-positive-definite contract.");
  auto *compiled_kernel_inverse =
      dynamic_cast<CompiledKernelLinearOperator *>(&inverse_apply_operator);
  auto *compiled_graph_inverse =
      dynamic_cast<CompiledGraphLinearOperator *>(&inverse_apply_operator);
  TI_ERROR_IF(!compiled_kernel_inverse && !compiled_graph_inverse,
              "Compiled-kernel preconditioners require an inverse-apply "
              "provider backed by a compiled kernel or compiled Graph.");
  Program *inverse_program = compiled_kernel_inverse
                                 ? compiled_kernel_inverse->owning_program()
                                 : compiled_graph_inverse->owning_program();
  TI_ERROR_IF(static_cast<SparseMatrix *>(&target_operator) ==
                  &inverse_apply_operator,
              "Compiled-kernel preconditioners require a distinct "
              "inverse-apply operator.");
  TI_ERROR_IF(target_operator.owning_program() != program ||
                  inverse_program != program,
              "Compiled-kernel target and inverse-apply providers must "
              "belong to the same Program.");
  TI_ERROR_IF(target_operator.num_rows() != inverse_apply_operator.num_rows() ||
                  target_operator.num_cols() !=
                      inverse_apply_operator.num_cols() ||
                  target_operator.get_data_type() != PrimitiveType::f32 ||
                  inverse_apply_operator.get_data_type() !=
                      PrimitiveType::f32,
              "Compiled-kernel preconditioners require matching non-empty "
              "square f32 target and inverse-apply operators.");

  auto target_guard = target_operator.acquire_numeric_access_guard();
  auto inverse_guard = inverse_apply_operator.acquire_numeric_access_guard();
  const auto target_stats = target_operator.debug_runtime_statistics();
  const auto inverse_stats = inverse_apply_operator.debug_runtime_statistics();
  program_ = program;
  target_operator_ = &target_operator;
  inverse_apply_operator_ = &inverse_apply_operator;
  target_pattern_version_at_build_ = target_stats.pattern_version;
  target_numeric_version_at_build_ = target_stats.numeric_version;
  inverse_pattern_version_at_build_ = inverse_stats.pattern_version;
  inverse_numeric_version_at_build_ = inverse_stats.numeric_version;
}

void CompiledKernelPreconditionerPlan::validate_compatible_locked(
    Program *program,
    const CompiledKernelLinearOperator &target_operator) const {
  TI_ERROR_IF(program != program_ || &target_operator != target_operator_,
              "Compiled-kernel preconditioner must be used with its "
              "construction Program and target operator.");
  const auto target_stats = target_operator_->debug_runtime_statistics();
  const auto inverse_stats =
      inverse_apply_operator_->debug_runtime_statistics();
  TI_ERROR_IF(
      target_stats.pattern_version != target_pattern_version_at_build_ ||
          target_stats.numeric_version != target_numeric_version_at_build_ ||
          inverse_stats.pattern_version != inverse_pattern_version_at_build_ ||
          inverse_stats.numeric_version != inverse_numeric_version_at_build_,
      "Compiled-kernel preconditioner is stale: target version changed from "
      "({}, {}) to ({}, {}) or inverse-apply version changed from ({}, {}) "
      "to ({}, {}); rebuild the preconditioner plan.",
      target_pattern_version_at_build_, target_numeric_version_at_build_,
      target_stats.pattern_version, target_stats.numeric_version,
      inverse_pattern_version_at_build_, inverse_numeric_version_at_build_,
      inverse_stats.pattern_version, inverse_stats.numeric_version);
}

void CompiledKernelPreconditionerPlan::validate_compatible(
    Program *program,
    const CompiledKernelLinearOperator &target_operator) const {
  auto target_guard = target_operator_->acquire_numeric_access_guard();
  auto inverse_guard = inverse_apply_operator_->acquire_numeric_access_guard();
  std::lock_guard<std::recursive_mutex> lock(apply_mutex_);
  validate_compatible_locked(program, target_operator);
}

void CompiledKernelPreconditionerPlan::apply(
    Program *program,
    const CompiledKernelLinearOperator &target_operator,
    const Ndarray &input,
    const Ndarray &output) {
  auto target_guard = target_operator_->acquire_numeric_access_guard();
  auto inverse_guard = inverse_apply_operator_->acquire_numeric_access_guard();
  std::lock_guard<std::recursive_mutex> lock(apply_mutex_);
  validate_compatible_locked(program, target_operator);
  inverse_apply_operator_->nd_spmv(program, input, output);
  apply_calls_++;
}

OperatorResourceLease
CompiledKernelPreconditionerPlan::acquire_resource_lease() const {
  struct ResourcePin {
    SparseMatrix::NumericAccessGuard target;
    SparseMatrix::NumericAccessGuard inverse;
    std::unique_lock<std::recursive_mutex> preconditioner;
  };
  return OperatorResourceLease::hold(ResourcePin{
      target_operator_->acquire_numeric_access_guard(),
      inverse_apply_operator_->acquire_numeric_access_guard(),
      std::unique_lock<std::recursive_mutex>(apply_mutex_)});
}

SparsePreconditionerPlanRuntimeStatistics
CompiledKernelPreconditionerPlan::debug_runtime_statistics() const {
  auto target_guard = target_operator_->acquire_numeric_access_guard();
  auto inverse_guard = inverse_apply_operator_->acquire_numeric_access_guard();
  std::lock_guard<std::recursive_mutex> lock(apply_mutex_);
  const auto target_stats = target_operator_->debug_runtime_statistics();
  const auto inverse_stats =
      inverse_apply_operator_->debug_runtime_statistics();
  SparsePreconditionerPlanRuntimeStatistics result;
  result.backend_family = target_stats.backend_family;
  result.method =
      dynamic_cast<CompiledGraphLinearOperator *>(inverse_apply_operator_)
          ? "compiled_graph_inverse_apply"
          : "compiled_kernel_inverse_apply";
  result.dtype = "f32";
  result.rows = target_operator_->num_rows();
  result.operator_pattern_version_at_build =
      target_pattern_version_at_build_;
  result.operator_numeric_version_at_build =
      target_numeric_version_at_build_;
  result.operator_pattern_version_current = target_stats.pattern_version;
  result.operator_numeric_version_current = target_stats.numeric_version;
  result.operator_stale =
      target_stats.pattern_version != target_pattern_version_at_build_ ||
      target_stats.numeric_version != target_numeric_version_at_build_;
  result.preconditioner_pattern_version_at_build =
      inverse_pattern_version_at_build_;
  result.preconditioner_numeric_version_at_build =
      inverse_numeric_version_at_build_;
  result.preconditioner_pattern_version_current =
      inverse_stats.pattern_version;
  result.preconditioner_numeric_version_current =
      inverse_stats.numeric_version;
  result.preconditioner_stale =
      inverse_stats.pattern_version != inverse_pattern_version_at_build_ ||
      inverse_stats.numeric_version != inverse_numeric_version_at_build_;
  result.apply_calls = apply_calls_;
  result.persistent_inverse_count = 0;
  result.persistent_inverse_reserved_bytes = 0;
  result.numeric_refresh_supported = false;
  result.in_place_apply_supported = false;
  return result;
}

std::unique_ptr<CompiledKernelPreconditionerPlan>
make_compiled_kernel_preconditioner_plan(
    Program *program,
    CompiledKernelLinearOperator &target_operator,
    SparseMatrix &inverse_apply_operator,
    bool assume_symmetric_positive_definite) {
  return std::make_unique<CompiledKernelPreconditionerPlan>(
      program, target_operator, inverse_apply_operator,
      assume_symmetric_positive_definite);
}

}  // namespace taichi::lang
