#pragma once

#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace taichi::lang {

// Cold, zero-based CSR validation shared by sparse providers. Unsorted rows are
// allowed; duplicate entries are not. Already sorted rows need no candidate
// set.
template <typename Offset, typename Index>
void validate_native_csr_indices(const std::vector<Offset> &offsets,
                                 const std::vector<Index> &columns,
                                 std::uint64_t column_count) {
  if (offsets.empty() || offsets.front() != 0 || offsets.back() < 0 ||
      std::uint64_t(offsets.back()) != columns.size())
    throw std::invalid_argument(
        "CSR offsets must start at zero and end at nnz");
  std::unordered_set<Index> seen;
  for (std::size_t row = 0; row + 1 < offsets.size(); ++row) {
    const auto begin = offsets[row], end = offsets[row + 1];
    if (begin < 0 || end < begin || std::uint64_t(end) > columns.size())
      throw std::invalid_argument("CSR offsets must be monotonic within nnz");
    bool unordered = false;
    for (auto i = begin; i < end; ++i) {
      const auto column = columns[i];
      if (column < 0 || std::uint64_t(column) >= column_count)
        throw std::invalid_argument("CSR column index is out of bounds");
      if (i > begin) {
        if (columns[i - 1] == column)
          throw std::invalid_argument("CSR duplicate column within a row");
        unordered |= columns[i - 1] > column;
      }
    }
    if (unordered) {
      seen.clear();
      for (auto i = begin; i < end; ++i)
        if (!seen.insert(columns[i]).second)
          throw std::invalid_argument("CSR duplicate column within a row");
    }
  }
}

}  // namespace taichi::lang
