#pragma once

#include <cstdint>
#include <vector>

namespace taichi::lang {
class Program;
class SparseMatrix;

// Host-only export data. No descriptor, device address or vendor workspace is
// serialized. Source matrices retain their normal live-value behavior.
struct SparseCsrHostSnapshot {
  int rows{0}, columns{0};
  std::vector<std::int32_t> row_offsets, column_indices;
  std::vector<float> values;
};

SparseCsrHostSnapshot snapshot_cuda_csr(Program &program, SparseMatrix &matrix);

struct SparseBsrHostSnapshot {
  int block_rows{0}, block_columns{0}, block_size{0};
  std::vector<std::int32_t> row_offsets, column_indices;
  std::vector<float> values;
};

SparseBsrHostSnapshot snapshot_cuda_bsr(Program &program, SparseMatrix &matrix);
}  // namespace taichi::lang
