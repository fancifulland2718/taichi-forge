#pragma once

#include <optional>
#include "taichi/program/storage_view.h"
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

// Vulkan kernel ndarrays need physical storage-buffer usage. CPU/CUDA use
// untyped allocations. Transfer and raster-only arrays must not use this gate.
void validate_native_shader_storage(Arch arch, DeviceAllocation allocation);

struct NativeStorageWrite {
  NativeStorageRange range;
  // Only this read port may overlap, and only as the exact same range. The
  // caller must additionally qualify matching dtype/shape for an in-place view.
  std::optional<std::size_t> inplace_read;
};

// Read/read sharing is legal. Writes cannot overlap another write or any read
// except their explicitly paired in-place port. Uses O(N) storage and O(N log
// N) cold validation, including imported Vulkan buffers with unknown memory
// IDs.
void validate_native_disjoint_writes(
    const std::vector<NativeStorageRange> &reads,
    const std::vector<NativeStorageWrite> &writes);

// Prove only cross-group separation. Repeated/overlapping ranges within either
// group are legal here; this is not a certificate for write/write ordering.
// Shares the owner, imported-memory and O(N log N) range validation above.
void validate_native_read_write_disjoint(
    const std::vector<NativeStorageRange> &reads,
    const std::vector<NativeStorageRange> &writes);

// Conservative allocation-wide independence, for fixed queue partitions whose
// kernel metadata does not describe exact accessed subranges. Resolve each
// wrapper once per group; distinct wrappers are not proof of physical separation.
void validate_native_allocation_groups_disjoint(
    Arch arch,
    const std::vector<DeviceAllocation> &first,
    const std::vector<DeviceAllocation> &second);

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

// Owner-independent, cold validation. Does not authorize access to a device or
// certify unique writes; an operation must qualify its supported mapping.
struct NativeDenseArrayLayout {
  storage::DenseStorageLayoutSpec layout;
  storage::DenseStorageProperties properties;
};

NativeDenseArrayLayout validate_native_dense_array_layout(
    const NativeArrayBinding &array);

// Validated physical view with the same borrowed lifetime as NativeStorageRange.
// No allocation or lease is created. Comparing views is independent of element
// count; uncertain overlap remains unproven, including unknown imported memory.
class NativeDenseArrayView {
 public:
  const NativeDenseArrayLayout &layout() const {
    return layout_;
  }
  const NativeStorageRange &range() const {
    return range_;
  }
  bool proven_disjoint(const NativeDenseArrayView &other) const noexcept;
  // Exact coordinate mapping, including dtype and shape, not just equal bounds.
  // This is not a certificate for internal uniqueness of a writable mapping.
  bool proven_same_mapping(const NativeDenseArrayView &other) const noexcept;

 private:
  friend NativeDenseArrayView resolve_native_dense_array(
      Arch arch, const NativeArrayBinding &array);
  NativeDenseArrayView(NativeDenseArrayLayout layout,
                       NativeStorageRange storage);
  NativeDenseArrayLayout layout_;
  NativeStorageRange origin_;
  NativeStorageRange range_;
  std::uint64_t stride_gcd_{0};
};

NativeDenseArrayView resolve_native_dense_array(
    Arch arch, const NativeArrayBinding &array);

}  // namespace taichi::lang
