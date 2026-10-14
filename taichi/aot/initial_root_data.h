#pragma once

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "taichi/common/core.h"
#include "taichi/common/serialization.h"

namespace taichi::lang::aot {

// Dense root bytes use the artifact's backend layout and module-local tree ID.
// Storage belongs to the module, not to an individual prepared execution.
struct InitialRootData {
  int tree_id{-1};
  std::uint64_t size{0};
  std::string sha256;

  TI_IO_DEF(tree_id, size, sha256);
};

inline void validate_initial_root_data(const InitialRootData &entry) {
  TI_ERROR_IF(entry.tree_id < 0 || entry.sha256.size() != 64 ||
                  entry.sha256.find_first_not_of("0123456789abcdef") !=
                      std::string::npos,
              "Invalid AOT initial root data descriptor.");
}

inline void validate_initial_root_data(
    const std::vector<InitialRootData> &entries,
    const std::vector<size_t> &root_sizes) {
  std::unordered_set<int> seen;
  for (const auto &entry : entries) {
    validate_initial_root_data(entry);
    TI_ERROR_IF(static_cast<size_t>(entry.tree_id) >= root_sizes.size() ||
                    !seen.insert(entry.tree_id).second,
                "Invalid or duplicate AOT initial root tree {}.", entry.tree_id);
    TI_ERROR_IF(entry.size != root_sizes[entry.tree_id],
                "AOT initial root data size does not match tree {}.",
                entry.tree_id);
  }
}

}  // namespace taichi::lang::aot
