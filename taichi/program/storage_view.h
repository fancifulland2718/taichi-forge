#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "taichi/inc/constants.h"
#include "taichi/ir/type.h"
#include "taichi/program/runtime_resource_registry.h"
#include "taichi/rhi/public_device.h"
#include "taichi/struct/snode_tree.h"

namespace taichi::lang {

class Ndarray;
class Program;
class SNode;

namespace storage {

constexpr std::size_t kMaxDenseStorageRank = taichi_max_num_indices;

enum class StorageOwnerKind : std::uint8_t {
  kInvalid,
  kProgramNdarray,
  kSNodePayload,
  kExternalManaged,
  kSubmissionScopedHost,
};

enum class StorageSourceKind : std::uint8_t {
  kUnknown,
  kNdarray,
  kDenseScalarField,
  kDensePackedField,
  kStructScalarMember,
  kStructTensorMember,
  kExternalDense,
};

enum class StorageAccess : std::uint8_t {
  kReadOnly,
  kWriteOnly,
  kReadWrite,
};

enum class StorageMappingUniqueness : std::uint8_t {
  kProvenUnique,
  kProvenOverlapping,
  kUnknown,
};

enum class StorageAliasRelation : std::uint8_t {
  kProvenDisjoint,
  kProvenOverlap,
  kUnknown,
};

enum class StorageExecutionMode : std::uint8_t {
  kDirectContiguous,
  kDirectAffine,
  kComponentBundle,
  kIndexedProjection,
  kMaterialized,
  kUnsupported,
};

enum class RuntimeStorageConsumer : std::uint8_t {
  kOrdinaryKernel,
  kGraphReplay,
  kGraphCapture,
  kNativeConsumer,
  kExternalInterop,
};

enum class RuntimeStorageMode : std::uint8_t {
  kOrdinary,
  kReplay,
  kCapture,
};

struct RuntimeStorageCapabilities {
  bool describable{false};
  bool bindable{false};
  bool replayable{false};
  bool capturable{false};
  bool zero_copy_qualified{false};
};

enum class StorageArrayLayout : std::uint8_t {
  kNone,
  kScalar,
  kAos,
  kSoa,
};

enum class StorageFailureReason : std::uint8_t {
  kNone,
  kInvalidOwner,
  kInvalidDtype,
  kInvalidRank,
  kInvalidShape,
  kInvalidOffset,
  kArithmeticOverflow,
  kStaleOwner,
  kRetiredGeneration,
  kDifferentProgram,
  kUnsupportedStorageKind,
  kUnsupportedBackend,
  kUnsupportedDtype,
  kUnsupportedRank,
  kUnsupportedLayout,
  kUnsupportedStride,
  kMisalignedRange,
  kInternalOverlap,
  kAliasUnknown,
  kWriteAlias,
  kReadOnlySource,
  kExternalOwnerNotAccepted,
  kExternalSyncUnavailable,
  kGraphIdentityUnstable,
  kCopyForbidden,
  kMaterializationUnavailable,
};

struct StorageOwnerRef {
  StorageOwnerKind kind{StorageOwnerKind::kInvalid};
  std::uint64_t program_domain{0};
  RuntimeResourceHandle ndarray_handle;
  SNodeTreeDependency tree;
  int anchor_snode_id{-1};
  std::vector<int> component_snode_ids;
  std::uint64_t external_owner_domain{0};
  std::uint32_t external_slot{0};
  std::uint32_t external_generation{0};

  static StorageOwnerRef program_ndarray(std::uint64_t program_domain,
                                         RuntimeResourceHandle handle);
  static StorageOwnerRef snode_payload(std::uint64_t program_domain,
                                       SNodeTreeDependency tree,
                                       int anchor_snode_id,
                                       std::vector<int> component_snode_ids);
  static StorageOwnerRef external_managed(std::uint64_t owner_domain,
                                          std::uint32_t slot,
                                          std::uint32_t generation);
  static StorageOwnerRef submission_scoped_host(std::uint64_t owner_domain);

  bool valid() const noexcept;
  bool same_logical_owner(const StorageOwnerRef &other) const noexcept;
  bool same_physical_owner(const StorageOwnerRef &other) const noexcept;
};

struct DenseStorageLayoutSpec {
  DataType scalar_type;
  std::vector<std::int64_t> index_shape;
  std::vector<std::int64_t> index_strides_bytes;
  std::vector<std::int64_t> element_shape;
  std::vector<std::int64_t> element_strides_bytes;
  std::int64_t byte_offset{0};
  StorageAccess access{StorageAccess::kReadWrite};
};

struct DenseStorageProperties {
  bool empty{false};
  bool aligned{false};
  bool compact_contiguous{false};
  bool element_contiguous{false};
  bool canonical_aos{false};
  bool canonical_soa{false};
  bool ndarray_abi_compatible{false};
  bool single_record_stride_compatible{false};
  bool has_negative_stride{false};
  std::uint64_t scalar_size{0};
  std::uint64_t scalar_alignment{0};
  std::uint64_t scalar_count{0};
  std::uint64_t item_count{0};
  std::int64_t reachable_begin{0};
  std::int64_t reachable_end{0};
  std::int64_t record_stride{0};
  StorageArrayLayout array_layout{StorageArrayLayout::kNone};
  StorageMappingUniqueness uniqueness{StorageMappingUniqueness::kUnknown};
};

// Submission-scoped device capability derived from an immutable descriptor
// only after its owner generation has been validated. Descriptors deliberately
// do not contain DeviceAllocation or raw-pointer state; backends may consume
// this value only while Program owns the corresponding launch transaction.
struct ResolvedDenseBinding {
  DeviceAllocation allocation{kDeviceNullAllocation};
  std::uint64_t byte_offset{0};
  std::uint64_t byte_size{0};
  std::uint64_t runtime_signature{0};
  std::uint64_t synchronization_domain_identity{0};
  RuntimeStorageCapabilities capabilities;
  bool valid{false};

  DevicePtr device_ptr() const {
    return allocation.get_ptr(byte_offset);
  }
};

class TI_DLL_EXPORT DenseStorageDescriptor {
 public:
  const StorageOwnerRef &owner() const noexcept {
    return owner_;
  }
  StorageSourceKind source_kind() const noexcept {
    return source_kind_;
  }
  DataType scalar_type() const noexcept {
    return scalar_type_;
  }
  StorageAccess access() const noexcept {
    return access_;
  }
  std::size_t index_rank() const noexcept {
    return index_rank_;
  }
  std::size_t element_rank() const noexcept {
    return element_rank_;
  }
  std::size_t total_rank() const noexcept {
    return index_rank_ + element_rank_;
  }
  std::int64_t index_extent(std::size_t axis) const;
  std::int64_t index_stride_bytes(std::size_t axis) const;
  std::int64_t element_extent(std::size_t axis) const;
  std::int64_t element_stride_bytes(std::size_t axis) const;
  std::vector<std::int64_t> index_shape() const;
  std::vector<std::int64_t> index_strides_bytes() const;
  std::vector<std::int64_t> element_shape() const;
  std::vector<std::int64_t> element_strides_bytes() const;
  std::int64_t byte_offset() const noexcept {
    return byte_offset_;
  }
  std::uint64_t fingerprint() const noexcept {
    return fingerprint_;
  }
  const DenseStorageProperties &properties() const noexcept {
    return properties_;
  }

 private:
  friend struct DenseStorageBuildResult;
  friend struct DenseStorageDescriptorAccess;

  DenseStorageDescriptor(
      StorageOwnerRef owner,
      StorageSourceKind source_kind,
      DataType scalar_type,
      StorageAccess access,
      std::uint8_t index_rank,
      std::uint8_t element_rank,
      std::array<std::int64_t, kMaxDenseStorageRank> extents,
      std::array<std::int64_t, kMaxDenseStorageRank> strides_bytes,
      std::int64_t byte_offset,
      DenseStorageProperties properties,
      std::uint64_t fingerprint);

  StorageOwnerRef owner_;
  StorageSourceKind source_kind_{StorageSourceKind::kUnknown};
  DataType scalar_type_;
  StorageAccess access_{StorageAccess::kReadWrite};
  std::uint8_t index_rank_{0};
  std::uint8_t element_rank_{0};
  std::array<std::int64_t, kMaxDenseStorageRank> extents_{};
  std::array<std::int64_t, kMaxDenseStorageRank> strides_bytes_{};
  std::int64_t byte_offset_{0};
  DenseStorageProperties properties_;
  std::uint64_t fingerprint_{0};
};

struct TI_DLL_EXPORT DenseStorageBuildResult {
  StorageFailureReason reason{StorageFailureReason::kNone};
  std::optional<DenseStorageDescriptor> descriptor;

  explicit operator bool() const noexcept {
    return reason == StorageFailureReason::kNone && descriptor.has_value();
  }
};

struct DenseStorageRequirement {
  bool require_scalar_type{false};
  DataType scalar_type;
  std::size_t min_index_rank{0};
  std::size_t max_index_rank{kMaxDenseStorageRank};
  std::size_t max_element_rank{kMaxDenseStorageRank};
  bool require_ndarray_abi{false};
  bool accept_compact_subrange{true};
  bool accept_single_record_stride{false};
  bool accept_general_affine{false};
  bool require_unique_mapping{false};
  bool require_writable{false};
  bool accept_external_owner{false};
  bool allow_materialization{false};
};

struct StorageQualification {
  bool supported{false};
  StorageExecutionMode execution_mode{StorageExecutionMode::kUnsupported};
  StorageFailureReason reason{StorageFailureReason::kNone};
  bool requires_materialization{false};
  std::uint64_t estimated_copy_bytes{0};
};

struct RuntimeStorageRequirement {
  DenseStorageRequirement dense;
  RuntimeStorageConsumer consumer{RuntimeStorageConsumer::kOrdinaryKernel};
  RuntimeStorageMode mode{RuntimeStorageMode::kOrdinary};
  Arch backend{Arch::x64};
  bool require_external_sync{false};
};

struct RuntimeStorageQualification {
  StorageQualification dense;
  RuntimeStorageCapabilities capabilities;
  StorageFailureReason reason{StorageFailureReason::kNone};
  std::uint64_t stable_signature{0};
};

// Immutable logical argument passed between Python adapters, Graph preparation,
// ordinary launch contexts, and native/external consumers. It owns descriptor
// metadata but never a raw pointer, DeviceAllocation, OS handle, or runtime
// lease. Those remain submission-scoped in ResolvedDenseBinding.
class TI_DLL_EXPORT RuntimeStorageArgument {
 public:
  RuntimeStorageArgument(DenseStorageDescriptor descriptor,
                         RuntimeStorageRequirement requirement,
                         std::uint64_t synchronization_domain_identity = 0);

  const DenseStorageDescriptor &descriptor() const noexcept {
    return descriptor_;
  }
  const RuntimeStorageRequirement &requirement() const noexcept {
    return requirement_;
  }
  std::uint64_t synchronization_domain_identity() const noexcept {
    return synchronization_domain_identity_;
  }
  const RuntimeStorageQualification &qualification() const noexcept {
    return qualification_;
  }
  std::uint64_t stable_signature() const noexcept {
    return qualification_.stable_signature;
  }

 private:
  DenseStorageDescriptor descriptor_;
  RuntimeStorageRequirement requirement_;
  std::uint64_t synchronization_domain_identity_{0};
  RuntimeStorageQualification qualification_;
};

TI_DLL_EXPORT DenseStorageBuildResult
build_dense_storage_descriptor(StorageOwnerRef owner,
                               StorageSourceKind source_kind,
                               const DenseStorageLayoutSpec &layout);

TI_DLL_EXPORT StorageQualification
qualify_dense_storage(const DenseStorageDescriptor &descriptor,
                      const DenseStorageRequirement &requirement);

TI_DLL_EXPORT RuntimeStorageQualification
qualify_runtime_storage(const DenseStorageDescriptor &descriptor,
                        const RuntimeStorageRequirement &requirement,
                        std::uint64_t synchronization_domain_identity = 0);

TI_DLL_EXPORT StorageAliasRelation
analyze_logical_storage_alias(const DenseStorageDescriptor &lhs,
                              const DenseStorageDescriptor &rhs) noexcept;

TI_DLL_EXPORT DenseStorageBuildResult
describe_ndarray_storage(const Ndarray &array,
                         StorageAccess access = StorageAccess::kReadWrite);

// Reinterpret a proven compact dense byte range as a scalar 1-D vector.
// This changes only logical shape/stride metadata: ownership, byte offset,
// access, and storage source remain unchanged and no allocation or copy is
// performed. Non-compact or non-unique mappings fail closed.
TI_DLL_EXPORT DenseStorageBuildResult
flatten_dense_storage_to_scalar_vector(
    const DenseStorageDescriptor &descriptor);

// Compose a zero-copy positive-stride subview over the logical index axes.
// Starts, lengths, and steps are already normalized by the caller. The owner,
// access mode, element layout, and lifecycle generation are preserved.
TI_DLL_EXPORT DenseStorageBuildResult slice_dense_storage(
    const DenseStorageDescriptor &descriptor,
    const std::vector<std::int64_t> &starts,
    const std::vector<std::int64_t> &lengths,
    const std::vector<std::int64_t> &steps);

TI_DLL_EXPORT DenseStorageBuildResult describe_struct_member_storage(
    const Ndarray &base,
    DataType scalar_type,
    const std::vector<std::int64_t> &index_shape,
    const std::vector<std::int64_t> &element_shape,
    std::int64_t byte_offset,
    std::int64_t record_stride,
    StorageSourceKind source_kind,
    StorageAccess access = StorageAccess::kReadWrite);

TI_DLL_EXPORT DenseStorageBuildResult
describe_dense_field_storage(Program &program,
                             SNode *anchor,
                             const std::vector<SNode *> &components,
                             DataType scalar_type,
                             const std::vector<std::int64_t> &index_shape,
                             const std::vector<std::int64_t> &element_shape,
                             StorageAccess access = StorageAccess::kReadWrite);

TI_DLL_EXPORT StorageFailureReason
validate_storage_owner(Program &program,
                       const DenseStorageDescriptor &descriptor) noexcept;

TI_DLL_EXPORT const char *to_string(StorageOwnerKind value) noexcept;
TI_DLL_EXPORT const char *to_string(StorageSourceKind value) noexcept;
TI_DLL_EXPORT const char *to_string(StorageAccess value) noexcept;
TI_DLL_EXPORT const char *to_string(StorageMappingUniqueness value) noexcept;
TI_DLL_EXPORT const char *to_string(StorageAliasRelation value) noexcept;
TI_DLL_EXPORT const char *to_string(StorageExecutionMode value) noexcept;
TI_DLL_EXPORT const char *to_string(RuntimeStorageConsumer value) noexcept;
TI_DLL_EXPORT const char *to_string(RuntimeStorageMode value) noexcept;
TI_DLL_EXPORT const char *to_string(StorageArrayLayout value) noexcept;
TI_DLL_EXPORT const char *to_string(StorageFailureReason value) noexcept;

}  // namespace storage
}  // namespace taichi::lang
