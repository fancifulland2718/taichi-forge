#pragma once

#include <map>
#include <string>
#include <vector>
#include "taichi/common/json.h"
#include "taichi/common/serialization.h"
#include "taichi/common/virtual_dir.h"

namespace taichi::lang::storage {
struct DenseStorageProperties;
}

namespace taichi::lang::aot {

// An affine array inside module-owned dense root storage. Names survive root-ID
// relocation; no producer addresses, copies or separate allocation are stored.
struct RootStorageView {
  int tree_id{-1};
  std::uint64_t byte_offset{0};
  std::string dtype;
  std::vector<std::uint32_t> shape;
  std::vector<std::uint32_t> element_shape;
  std::vector<std::int64_t> index_strides_bytes;
  std::vector<std::int64_t> element_strides_bytes;
  TI_IO_DEF(tree_id, byte_offset, dtype, shape, element_shape,
            index_strides_bytes, element_strides_bytes);

  storage::DenseStorageProperties properties() const;
  // Logical payload, distinct from the reachable physical window.
  std::uint64_t bytes() const;
  bool operator==(const RootStorageView &other) const;
};

using RootStorageViews = std::map<std::string, RootStorageView>;
RootStorageViews parse_root_storage_views(const liong::json::JsonValue &json);
RootStorageViews load_root_storage_views(const io::VirtualDir &directory);
void write_root_storage_views(const RootStorageViews &views,
                              const std::map<int, int> &tree_mapping,
                              const std::string &directory);

}  // namespace taichi::lang::aot
