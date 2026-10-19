#pragma once

#include <tuple>
#include "taichi/runtime/native_operation.h"

namespace taichi::lang {

// Internal preparation metadata, not a persistent application description.
// Exact matches avoid changing a recorded pointer or backend workspace layout.
struct SortWorkspaceDescription {
  static constexpr const char *kind = "sort_workspace";
  static constexpr std::uint32_t version = 1;
  using Signature =
      std::tuple<std::size_t, NativeSortKeyType, std::size_t, std::size_t>;
  std::size_t size;
  NativeSortKeyType key_type;
  std::size_t value_bytes;
  std::size_t value_scalar_bytes;

  static SortWorkspaceDescription from(const SortOperation &operation) {
    return {operation.size, operation.key_type, operation.value_bytes,
            operation.value_scalar_bytes};
  }
  Signature signature() const {
    return {size, key_type, value_bytes, value_scalar_bytes};
  }
};

// A mutable scratch resource. It may be shared by sequential operations, never
// implicitly between parallel branches or independently prepared frames.
class NativeSortWorkspace : public NativeResource {
 public:
  Arch arch() const final {
    return arch_;
  }
  Device &device() const final {
    return device_;
  }
  bool matches(const SortOperation &operation) const {
    return description_.signature() ==
           SortWorkspaceDescription::from(operation).signature();
  }

 protected:
  NativeSortWorkspace(Arch arch,
                      Device &device,
                      const SortWorkspaceDescription &description)
      : arch_(arch), device_(device), description_(description) {
  }
  Arch arch_;
  Device &device_;
  SortWorkspaceDescription description_;
};

std::shared_ptr<NativeSortWorkspace> prepare_sort_workspace(
    Arch arch,
    Device &device,
    const SortWorkspaceDescription &description);

namespace gfx {
std::shared_ptr<NativeSortWorkspace> prepare_vulkan_sort_workspace(
    Device &device,
    const SortWorkspaceDescription &description);
}
}  // namespace taichi::lang
