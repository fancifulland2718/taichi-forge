#pragma once

#include "taichi/runtime/native_operation_description.h"

namespace taichi::lang {

struct ScatterAddOperation {
  static constexpr const char *kind = "scatter_add";
  static constexpr std::uint32_t version = 1;
  std::size_t size{0}, dst_items{0};
  int value_type{0};
  int cpu_max_threads{0};
  NativeArrayBinding src, indices, dst;
};

void validate_scatter_add_storage(Arch arch,
                                  const ScatterAddOperation &operation);
class NativeOperation;
namespace gfx {
std::shared_ptr<NativeOperation> prepare_vulkan_scatter_add_operation(
    Device &device,
    const ScatterAddOperation &operation);
}
}  // namespace taichi::lang
