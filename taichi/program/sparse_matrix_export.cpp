#include "taichi/program/sparse_matrix_export.h"

#include "taichi/program/program.h"
#include "taichi/program/sparse_matrix.h"
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#endif

namespace taichi::lang {
namespace {
#ifdef TI_WITH_CUDA
template <typename Read>
auto snapshot_locked(Program &program, SparseMatrix &matrix, Read read) {
  // Updates can enter the numeric gate before Program storage resolution;
  // Graphs enter through Program/CUDA first. Obtain unlocked guards and use
  // deadlock-avoiding group acquisition rather than introducing either order.
  auto numeric = matrix.acquire_numeric_access_guard();
  numeric.unlock();
  auto submission = program.acquire_runtime_resource_submission_guard();
  submission.unlock();
  auto &cuda = CUDAContext::get_instance();
  auto cuda_submission = cuda.get_submission_lock_guard();
  cuda_submission.unlock();
  std::lock(numeric, submission, cuda_submission);
  auto context = cuda.get_guard();
  program.synchronize();
  return read();
}

template <typename Snapshot>
void read_arrays(SparseMatrix &matrix,
                 Snapshot &result,
                 void *offsets,
                 void *columns,
                 void *values) {
  std::uint64_t bytes = 0;
  auto read = [&](auto &host, void *source) {
    const auto size = host.size() * sizeof(host[0]);
    if (size)
      CUDADriver::get_instance().memcpy_device_to_host(host.data(), source,
                                                       size);
    bytes += size;
  };
  read(result.row_offsets, offsets);
  read(result.column_indices, columns);
  read(result.values, values);
  matrix.record_transfer_bytes(0, bytes, 0);
}
#endif
}  // namespace

SparseCsrHostSnapshot snapshot_cuda_csr(Program &program,
                                        SparseMatrix &matrix,
                                        bool include_values) {
#ifdef TI_WITH_CUDA
  auto *csr = dynamic_cast<CuSparseMatrix *>(&matrix);
  TI_ERROR_IF(
      program.compile_config().arch != Arch::cuda || !csr ||
          matrix.get_data_type() != PrimitiveType::f32,
      "AOT sparse snapshots require a live scalar f32 CUDA CSR matrix.");
  return snapshot_locked(program, matrix, [&] {
    TI_ERROR_IF(
        csr->num_rows() < 0 || csr->num_cols() < 0 || csr->get_nnz() < 0 ||
            !csr->get_row_ptr() ||
            (csr->get_nnz() && (!csr->get_col_ind() || !csr->get_val_ptr())),
        "AOT sparse snapshots require initialized CSR storage.");
    SparseCsrHostSnapshot result;
    result.rows = csr->num_rows();
    result.columns = csr->num_cols();
    result.row_offsets.resize(std::size_t(result.rows) + 1);
    result.column_indices.resize(csr->get_nnz());
    if (include_values)
      result.values.resize(csr->get_nnz());
    read_arrays(matrix, result, csr->get_row_ptr(), csr->get_col_ind(),
                csr->get_val_ptr());
    return result;
  });
#else
  TI_ERROR("AOT CUDA CSR snapshots require CUDA support.");
#endif
}

SparseBsrHostSnapshot snapshot_cuda_bsr(Program &program,
                                        SparseMatrix &matrix,
                                        bool include_values) {
#ifdef TI_WITH_CUDA
  auto *bsr = dynamic_cast<CuSparseBsrMatrix *>(&matrix);
  TI_ERROR_IF(program.compile_config().arch != Arch::cuda || !bsr ||
                  matrix.get_data_type() != PrimitiveType::f32,
              "AOT sparse snapshots require a live f32 CUDA BSR matrix.");
  return snapshot_locked(program, matrix, [&] {
    TI_ERROR_IF(
        bsr->get_block_rows() <= 0 || bsr->get_block_cols() <= 0 ||
            bsr->get_block_size() <= 0 || bsr->get_block_nnz() < 0 ||
            !bsr->get_block_row_offsets() ||
            (bsr->get_block_nnz() &&
             (!bsr->get_block_column_indices() || !bsr->get_block_values())),
        "AOT sparse snapshots require initialized BSR storage.");
    SparseBsrHostSnapshot result;
    result.block_rows = bsr->get_block_rows();
    result.block_columns = bsr->get_block_cols();
    result.block_size = bsr->get_block_size();
    result.row_offsets.resize(std::size_t(result.block_rows) + 1);
    result.column_indices.resize(bsr->get_block_nnz());
    if (include_values)
      result.values.resize(std::size_t(bsr->get_block_nnz()) *
                           result.block_size * result.block_size);
    read_arrays(matrix, result, bsr->get_block_row_offsets(),
                bsr->get_block_column_indices(), bsr->get_block_values());
    return result;
  });
#else
  TI_ERROR("AOT CUDA BSR snapshots require CUDA support.");
#endif
}
}  // namespace taichi::lang
