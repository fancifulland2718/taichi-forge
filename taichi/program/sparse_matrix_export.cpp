#include "taichi/program/sparse_matrix_export.h"

#include "taichi/program/program.h"
#include "taichi/program/sparse_matrix.h"
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#endif

namespace taichi::lang {
SparseCsrHostSnapshot snapshot_cuda_csr(Program &program,
                                        SparseMatrix &matrix) {
#ifdef TI_WITH_CUDA
  auto *csr = dynamic_cast<CuSparseMatrix *>(&matrix);
  TI_ERROR_IF(
      program.compile_config().arch != Arch::cuda || !csr ||
          matrix.get_data_type() != PrimitiveType::f32,
      "AOT sparse snapshots require a live scalar f32 CUDA CSR matrix.");
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
  TI_ERROR_IF(
      csr->num_rows() < 0 || csr->num_cols() < 0 || csr->get_nnz() < 0 ||
          !csr->get_row_ptr() ||
          (csr->get_nnz() && (!csr->get_col_ind() || !csr->get_val_ptr())),
      "AOT sparse snapshots require initialized CSR storage.");
  program.synchronize();
  SparseCsrHostSnapshot result;
  result.rows = csr->num_rows();
  result.columns = csr->num_cols();
  result.row_offsets.resize(std::size_t(result.rows) + 1);
  result.column_indices.resize(csr->get_nnz());
  result.values.resize(csr->get_nnz());
  std::uint64_t bytes = 0;
  auto read = [&](auto &host, void *source) {
    const auto size = host.size() * sizeof(host[0]);
    if (size)
      CUDADriver::get_instance().memcpy_device_to_host(host.data(), source,
                                                       size);
    bytes += size;
  };
  read(result.row_offsets, csr->get_row_ptr());
  read(result.column_indices, csr->get_col_ind());
  read(result.values, csr->get_val_ptr());
  matrix.record_transfer_bytes(0, bytes, 0);
  return result;
#else
  TI_ERROR("AOT CUDA CSR snapshots require CUDA support.");
#endif
}
}  // namespace taichi::lang
