#pragma once

#include "taichi/runtime/native_operation_description.h"

namespace taichi::lang {

struct IndexedCopyOperation {
  static constexpr const char *kind = "indexed_copy";
  static constexpr std::uint32_t version = 1;
  std::size_t size{0}, src_items{0}, dst_items{0};
  DataType dtype;
  std::vector<std::uint64_t> element_shape;
  bool scatter{false};
  NativeArrayBinding src, indices, dst;
};

std::size_t indexed_copy_item_bytes(const IndexedCopyOperation &operation);
void validate_indexed_copy_storage(Arch arch,
                                   const IndexedCopyOperation &operation);

class NativeOperation;
namespace gfx {
std::shared_ptr<NativeOperation> prepare_vulkan_indexed_copy_operation(
    Device &device,
    const IndexedCopyOperation &operation);
}
}  // namespace taichi::lang
