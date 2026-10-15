#pragma once

#include "taichi/runtime/native_operation_description.h"

namespace taichi::lang {

// Cold facts only. The caller must retain the device and allocation owners.
// memory == 0 means the underlying Vulkan memory identity is unknown; equal
// buffer handles can still prove disjoint subranges within that buffer.
struct NativeStorageRange {
  Device *device{nullptr};
  std::uint64_t memory{0};
  std::uint64_t memory_offset{0};
  std::uint64_t buffer{0};
  std::uint64_t buffer_offset{0};
  std::uint64_t bytes{0};

  bool proven_disjoint(const NativeStorageRange &other) const noexcept;
  bool proven_same_range(const NativeStorageRange &other) const noexcept;
};

NativeStorageRange resolve_native_storage_range(Arch arch,
                                                DevicePtr pointer,
                                                std::uint64_t bytes);

// Requirements on a packed scalar/AoS array. Adapters must qualify packed
// storage first; shape metadata alone is not proof that an affine view is
// packed.
struct NativeArrayLayoutRequirement {
  std::vector<std::uint64_t> minimum_shape;
  std::vector<std::uint64_t> element_shape;
  std::uint64_t record_bytes{0};
  std::uint64_t alignment{0};
  // minimum_shape becomes an equality requirement when selected explicitly.
  bool exact_shape{false};
  // Empty preserves a layout-only certificate; typed producers freeze dtype.
  std::string dtype;
};

void validate_native_array_layout(const NativeArrayBinding &array,
                                  const NativeArrayLayoutRequirement &required);

}  // namespace taichi::lang
