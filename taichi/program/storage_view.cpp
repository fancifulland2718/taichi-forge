#include "taichi/program/storage_view.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "taichi/ir/snode.h"
#include "taichi/ir/type_utils.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"

namespace taichi::lang::storage {
namespace {

constexpr std::uint64_t kFnvOffsetBasis = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

void fingerprint_byte(std::uint64_t &value, std::uint8_t byte) noexcept {
  value ^= byte;
  value *= kFnvPrime;
}

template <typename T>
void fingerprint_integer(std::uint64_t &value, T input) noexcept {
  using Unsigned = std::make_unsigned_t<T>;
  Unsigned bits = static_cast<Unsigned>(input);
  for (std::size_t i = 0; i < sizeof(Unsigned); ++i) {
    fingerprint_byte(value,
                     static_cast<std::uint8_t>((bits >> (i * 8)) & 0xffu));
  }
}

void fingerprint_owner(std::uint64_t &value,
                       const StorageOwnerRef &owner) noexcept {
  fingerprint_integer(value, owner.kind);
  fingerprint_integer(value, owner.program_domain);
  switch (owner.kind) {
    case StorageOwnerKind::kProgramNdarray:
      fingerprint_integer(value, owner.ndarray_handle.domain);
      fingerprint_integer(value, owner.ndarray_handle.kind);
      fingerprint_integer(value, owner.ndarray_handle.index);
      fingerprint_integer(value, owner.ndarray_handle.generation);
      break;
    case StorageOwnerKind::kSNodePayload:
      fingerprint_integer(value, owner.tree.tree_id);
      fingerprint_integer(value, owner.tree.generation);
      fingerprint_integer(value, owner.tree.layout_fingerprint);
      fingerprint_integer(value, owner.anchor_snode_id);
      fingerprint_integer(value, owner.component_snode_ids.size());
      for (int component : owner.component_snode_ids) {
        fingerprint_integer(value, component);
      }
      break;
    case StorageOwnerKind::kExternalManaged:
      fingerprint_integer(value, owner.external_owner_domain);
      fingerprint_integer(value, owner.external_slot);
      fingerprint_integer(value, owner.external_generation);
      break;
    case StorageOwnerKind::kSubmissionScopedHost:
      fingerprint_integer(value, owner.external_owner_domain);
      break;
    case StorageOwnerKind::kInvalid:
      break;
  }
}

bool checked_add(std::int64_t lhs,
                 std::int64_t rhs,
                 std::int64_t *result) noexcept {
  if ((rhs > 0 && lhs > (std::numeric_limits<std::int64_t>::max)() - rhs) ||
      (rhs < 0 && lhs < (std::numeric_limits<std::int64_t>::min)() - rhs)) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

bool checked_mul_nonnegative(std::int64_t value,
                             std::int64_t count,
                             std::int64_t *result) noexcept {
  if (count < 0) {
    return false;
  }
  if (value == 0 || count == 0) {
    *result = 0;
    return true;
  }
  if (value > 0) {
    if (value > (std::numeric_limits<std::int64_t>::max)() / count) {
      return false;
    }
  } else if (value < (std::numeric_limits<std::int64_t>::min)() / count) {
    return false;
  }
  *result = value * count;
  return true;
}

bool checked_mul_u64(std::uint64_t lhs,
                     std::uint64_t rhs,
                     std::uint64_t *result) noexcept {
  if (lhs != 0 && rhs > (std::numeric_limits<std::uint64_t>::max)() / lhs) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

bool checked_abs(std::int64_t value, std::uint64_t *result) noexcept {
  if (value == (std::numeric_limits<std::int64_t>::min)()) {
    return false;
  }
  *result = static_cast<std::uint64_t>(value < 0 ? -value : value);
  return true;
}

bool access_is_writable(StorageAccess access) noexcept {
  return access == StorageAccess::kWriteOnly ||
         access == StorageAccess::kReadWrite;
}

bool scalar_type_info(DataType type,
                      std::uint64_t *size,
                      std::uint64_t *alignment) noexcept {
  try {
    if (!type->is<PrimitiveType>() || is_quant(type) ||
        type->is_primitive(PrimitiveTypeID::unknown)) {
      return false;
    }
    const int byte_size = data_type_size(type);
    const int byte_alignment = data_type_alignment(type);
    if (byte_size <= 0 || byte_alignment <= 0) {
      return false;
    }
    *size = static_cast<std::uint64_t>(byte_size);
    *alignment = static_cast<std::uint64_t>(byte_alignment);
    return true;
  } catch (...) {
    return false;
  }
}

bool canonical_strides(const std::vector<std::int64_t> &shape,
                       std::int64_t scalar_stride,
                       std::vector<std::int64_t> *strides) noexcept {
  strides->assign(shape.size(), 0);
  std::int64_t stride = scalar_stride;
  for (std::size_t i = shape.size(); i > 0; --i) {
    (*strides)[i - 1] = stride;
    if (shape[i - 1] < 0 ||
        !checked_mul_nonnegative(stride, shape[i - 1], &stride)) {
      return false;
    }
  }
  return true;
}

bool strides_equal(
    const std::array<std::int64_t, kMaxDenseStorageRank> &actual,
    const std::array<std::int64_t, kMaxDenseStorageRank> &extents,
    std::size_t offset,
    const std::vector<std::int64_t> &expected) noexcept {
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (extents[offset + i] > 1 && actual[offset + i] != expected[i]) {
      return false;
    }
  }
  return true;
}

StorageMappingUniqueness derive_uniqueness(
    const std::array<std::int64_t, kMaxDenseStorageRank> &extents,
    const std::array<std::int64_t, kMaxDenseStorageRank> &strides,
    std::size_t rank,
    std::uint64_t scalar_size,
    bool empty,
    bool *arithmetic_valid) noexcept {
  *arithmetic_valid = true;
  if (empty) {
    return StorageMappingUniqueness::kProvenUnique;
  }

  std::vector<std::pair<std::uint64_t, std::int64_t>> axes;
  axes.reserve(rank);
  for (std::size_t axis = 0; axis < rank; ++axis) {
    if (extents[axis] <= 1) {
      continue;
    }
    if (strides[axis] == 0) {
      return StorageMappingUniqueness::kProvenOverlapping;
    }
    std::uint64_t magnitude = 0;
    if (!checked_abs(strides[axis], &magnitude)) {
      *arithmetic_valid = false;
      return StorageMappingUniqueness::kUnknown;
    }
    if (magnitude < scalar_size) {
      return StorageMappingUniqueness::kProvenOverlapping;
    }
    axes.emplace_back(magnitude, extents[axis]);
  }
  std::sort(axes.begin(), axes.end());

  std::uint64_t covered = scalar_size;
  for (const auto &[stride, extent] : axes) {
    if (stride < covered) {
      return StorageMappingUniqueness::kUnknown;
    }
    std::uint64_t delta = 0;
    if (!checked_mul_u64(stride, static_cast<std::uint64_t>(extent - 1),
                         &delta) ||
        covered > (std::numeric_limits<std::uint64_t>::max)() - delta) {
      *arithmetic_valid = false;
      return StorageMappingUniqueness::kUnknown;
    }
    covered += delta;
  }
  return StorageMappingUniqueness::kProvenUnique;
}

bool interval_disjoint(std::int64_t lhs_begin,
                       std::int64_t lhs_end,
                       std::int64_t rhs_begin,
                       std::int64_t rhs_end) noexcept {
  return lhs_end <= rhs_begin || rhs_end <= lhs_begin;
}

std::uint64_t descriptor_fingerprint(
    const StorageOwnerRef &owner,
    StorageSourceKind source_kind,
    DataType scalar_type,
    StorageAccess access,
    std::size_t index_rank,
    std::size_t element_rank,
    const std::array<std::int64_t, kMaxDenseStorageRank> &extents,
    const std::array<std::int64_t, kMaxDenseStorageRank> &strides,
    std::int64_t byte_offset) noexcept {
  std::uint64_t value = kFnvOffsetBasis;
  fingerprint_owner(value, owner);
  fingerprint_integer(value, source_kind);
  fingerprint_integer(value, scalar_type.hash());
  fingerprint_integer(value, access);
  fingerprint_integer(value, index_rank);
  fingerprint_integer(value, element_rank);
  for (std::size_t axis = 0; axis < index_rank + element_rank; ++axis) {
    fingerprint_integer(value, extents[axis]);
    fingerprint_integer(value, strides[axis]);
  }
  fingerprint_integer(value, byte_offset);
  return value;
}

std::vector<std::int64_t> as_i64(const std::vector<int> &values) {
  return std::vector<std::int64_t>(values.begin(), values.end());
}

bool make_ndarray_layout(const Ndarray &array, DenseStorageLayoutSpec *layout) {
  layout->scalar_type = array.get_element_data_type();
  layout->index_shape = as_i64(array.shape);
  layout->element_shape = as_i64(array.get_element_shape());

  std::uint64_t scalar_size = 0;
  std::uint64_t scalar_alignment = 0;
  if (!scalar_type_info(layout->scalar_type, &scalar_size, &scalar_alignment) ||
      scalar_size > static_cast<std::uint64_t>(
                        (std::numeric_limits<std::int64_t>::max)())) {
    return false;
  }

  const auto scalar_stride = static_cast<std::int64_t>(scalar_size);
  const bool soa = array.layout == ExternalArrayLayout::kSOA;
  std::vector<std::int64_t> physical_shape;
  if (soa) {
    physical_shape = layout->element_shape;
    physical_shape.insert(physical_shape.end(), layout->index_shape.begin(),
                          layout->index_shape.end());
  } else {
    physical_shape = layout->index_shape;
    physical_shape.insert(physical_shape.end(), layout->element_shape.begin(),
                          layout->element_shape.end());
  }
  std::vector<std::int64_t> physical_strides;
  if (!canonical_strides(physical_shape, scalar_stride, &physical_strides)) {
    return false;
  }

  if (soa) {
    const std::size_t element_rank = layout->element_shape.size();
    layout->element_strides_bytes.assign(
        physical_strides.begin(), physical_strides.begin() + element_rank);
    layout->index_strides_bytes.assign(physical_strides.begin() + element_rank,
                                       physical_strides.end());
  } else {
    const std::size_t index_rank = layout->index_shape.size();
    layout->index_strides_bytes.assign(physical_strides.begin(),
                                       physical_strides.begin() + index_rank);
    layout->element_strides_bytes.assign(physical_strides.begin() + index_rank,
                                         physical_strides.end());
  }
  return true;
}

bool is_program_ndarray_owner(const StorageOwnerRef &owner) noexcept {
  return owner.kind == StorageOwnerKind::kProgramNdarray;
}

}  // namespace

StorageOwnerRef StorageOwnerRef::program_ndarray(std::uint64_t program_domain,
                                                 RuntimeResourceHandle handle) {
  StorageOwnerRef owner;
  owner.kind = StorageOwnerKind::kProgramNdarray;
  owner.program_domain = program_domain;
  owner.ndarray_handle = handle;
  return owner;
}

StorageOwnerRef StorageOwnerRef::snode_payload(
    std::uint64_t program_domain,
    SNodeTreeDependency tree,
    int anchor_snode_id,
    std::vector<int> component_snode_ids) {
  StorageOwnerRef owner;
  owner.kind = StorageOwnerKind::kSNodePayload;
  owner.program_domain = program_domain;
  owner.tree = tree;
  owner.anchor_snode_id = anchor_snode_id;
  owner.component_snode_ids = std::move(component_snode_ids);
  return owner;
}

StorageOwnerRef StorageOwnerRef::external_managed(std::uint64_t owner_domain,
                                                  std::uint32_t slot,
                                                  std::uint32_t generation) {
  StorageOwnerRef owner;
  owner.kind = StorageOwnerKind::kExternalManaged;
  owner.external_owner_domain = owner_domain;
  owner.external_slot = slot;
  owner.external_generation = generation;
  return owner;
}

StorageOwnerRef StorageOwnerRef::submission_scoped_host(
    std::uint64_t owner_domain) {
  StorageOwnerRef owner;
  owner.kind = StorageOwnerKind::kSubmissionScopedHost;
  owner.external_owner_domain = owner_domain;
  return owner;
}

bool StorageOwnerRef::valid() const noexcept {
  switch (kind) {
    case StorageOwnerKind::kProgramNdarray:
      return program_domain != 0 && ndarray_handle;
    case StorageOwnerKind::kSNodePayload:
      return program_domain != 0 && tree.tree_id >= 0 && tree.generation != 0 &&
             anchor_snode_id >= 0;
    case StorageOwnerKind::kExternalManaged:
      return external_owner_domain != 0 && external_generation != 0;
    case StorageOwnerKind::kSubmissionScopedHost:
      return external_owner_domain != 0;
    case StorageOwnerKind::kInvalid:
      return false;
  }
  return false;
}

bool StorageOwnerRef::same_logical_owner(
    const StorageOwnerRef &other) const noexcept {
  if (kind != other.kind) {
    return false;
  }
  switch (kind) {
    case StorageOwnerKind::kProgramNdarray:
      return ndarray_handle == other.ndarray_handle;
    case StorageOwnerKind::kSNodePayload:
      return program_domain == other.program_domain && tree == other.tree &&
             anchor_snode_id == other.anchor_snode_id &&
             component_snode_ids == other.component_snode_ids;
    case StorageOwnerKind::kExternalManaged:
      return external_owner_domain == other.external_owner_domain &&
             external_slot == other.external_slot &&
             external_generation == other.external_generation;
    case StorageOwnerKind::kSubmissionScopedHost:
      return external_owner_domain == other.external_owner_domain;
    case StorageOwnerKind::kInvalid:
      return false;
  }
  return false;
}

bool StorageOwnerRef::same_physical_owner(
    const StorageOwnerRef &other) const noexcept {
  if (kind != other.kind) {
    return false;
  }
  switch (kind) {
    case StorageOwnerKind::kProgramNdarray:
      return ndarray_handle == other.ndarray_handle;
    case StorageOwnerKind::kSNodePayload:
      return program_domain == other.program_domain && tree == other.tree;
    case StorageOwnerKind::kExternalManaged:
      return external_owner_domain == other.external_owner_domain &&
             external_slot == other.external_slot &&
             external_generation == other.external_generation;
    case StorageOwnerKind::kSubmissionScopedHost:
      return external_owner_domain == other.external_owner_domain;
    case StorageOwnerKind::kInvalid:
      return false;
  }
  return false;
}

DenseStorageDescriptor::DenseStorageDescriptor(
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
    std::uint64_t fingerprint)
    : owner_(std::move(owner)),
      source_kind_(source_kind),
      scalar_type_(scalar_type),
      access_(access),
      index_rank_(index_rank),
      element_rank_(element_rank),
      extents_(extents),
      strides_bytes_(strides_bytes),
      byte_offset_(byte_offset),
      properties_(properties),
      fingerprint_(fingerprint) {
}

std::int64_t DenseStorageDescriptor::index_extent(std::size_t axis) const {
  if (axis >= index_rank_) {
    throw std::out_of_range("dense storage index axis is out of range");
  }
  return extents_[axis];
}

std::int64_t DenseStorageDescriptor::index_stride_bytes(
    std::size_t axis) const {
  if (axis >= index_rank_) {
    throw std::out_of_range("dense storage index axis is out of range");
  }
  return strides_bytes_[axis];
}

std::int64_t DenseStorageDescriptor::element_extent(std::size_t axis) const {
  if (axis >= element_rank_) {
    throw std::out_of_range("dense storage element axis is out of range");
  }
  return extents_[index_rank_ + axis];
}

std::int64_t DenseStorageDescriptor::element_stride_bytes(
    std::size_t axis) const {
  if (axis >= element_rank_) {
    throw std::out_of_range("dense storage element axis is out of range");
  }
  return strides_bytes_[index_rank_ + axis];
}

std::vector<std::int64_t> DenseStorageDescriptor::index_shape() const {
  return {extents_.begin(), extents_.begin() + index_rank_};
}

std::vector<std::int64_t> DenseStorageDescriptor::index_strides_bytes() const {
  return {strides_bytes_.begin(), strides_bytes_.begin() + index_rank_};
}

std::vector<std::int64_t> DenseStorageDescriptor::element_shape() const {
  return {extents_.begin() + index_rank_,
          extents_.begin() + index_rank_ + element_rank_};
}

std::vector<std::int64_t> DenseStorageDescriptor::element_strides_bytes()
    const {
  return {strides_bytes_.begin() + index_rank_,
          strides_bytes_.begin() + index_rank_ + element_rank_};
}

DenseStorageBuildResult build_dense_storage_descriptor(
    StorageOwnerRef owner,
    StorageSourceKind source_kind,
    const DenseStorageLayoutSpec &layout) {
  DenseStorageBuildResult result;
  if (!owner.valid()) {
    result.reason = StorageFailureReason::kInvalidOwner;
    return result;
  }
  if (source_kind == StorageSourceKind::kUnknown) {
    result.reason = StorageFailureReason::kUnsupportedStorageKind;
    return result;
  }
  std::uint64_t scalar_size = 0;
  std::uint64_t scalar_alignment = 0;
  if (!scalar_type_info(layout.scalar_type, &scalar_size, &scalar_alignment)) {
    result.reason = StorageFailureReason::kInvalidDtype;
    return result;
  }
  if (layout.index_shape.size() != layout.index_strides_bytes.size() ||
      layout.element_shape.size() != layout.element_strides_bytes.size() ||
      layout.index_shape.size() + layout.element_shape.size() >
          kMaxDenseStorageRank) {
    result.reason = StorageFailureReason::kInvalidRank;
    return result;
  }
  if (layout.byte_offset < 0) {
    result.reason = StorageFailureReason::kInvalidOffset;
    return result;
  }

  const std::size_t index_rank = layout.index_shape.size();
  const std::size_t element_rank = layout.element_shape.size();
  const std::size_t total_rank = index_rank + element_rank;
  std::array<std::int64_t, kMaxDenseStorageRank> extents{};
  std::array<std::int64_t, kMaxDenseStorageRank> strides{};
  for (std::size_t i = 0; i < index_rank; ++i) {
    extents[i] = layout.index_shape[i];
    strides[i] = layout.index_strides_bytes[i];
  }
  for (std::size_t i = 0; i < element_rank; ++i) {
    extents[index_rank + i] = layout.element_shape[i];
    strides[index_rank + i] = layout.element_strides_bytes[i];
  }

  DenseStorageProperties properties;
  properties.scalar_size = scalar_size;
  properties.scalar_alignment = scalar_alignment;
  properties.scalar_count = 1;
  properties.item_count = 1;
  for (std::size_t axis = 0; axis < total_rank; ++axis) {
    if (extents[axis] < 0) {
      result.reason = StorageFailureReason::kInvalidShape;
      return result;
    }
    properties.empty |= extents[axis] == 0;
    std::uint64_t product = 0;
    if (!checked_mul_u64(properties.scalar_count,
                         static_cast<std::uint64_t>(extents[axis]), &product)) {
      result.reason = StorageFailureReason::kArithmeticOverflow;
      return result;
    }
    properties.scalar_count = product;
    if (axis < index_rank) {
      if (!checked_mul_u64(properties.item_count,
                           static_cast<std::uint64_t>(extents[axis]),
                           &product)) {
        result.reason = StorageFailureReason::kArithmeticOverflow;
        return result;
      }
      properties.item_count = product;
    }
    properties.has_negative_stride |= strides[axis] < 0;
  }
  if (properties.empty) {
    properties.scalar_count = 0;
  }

  properties.reachable_begin = layout.byte_offset;
  properties.reachable_end = layout.byte_offset;
  if (!properties.empty) {
    std::int64_t min_offset = layout.byte_offset;
    std::int64_t max_offset = layout.byte_offset;
    for (std::size_t axis = 0; axis < total_rank; ++axis) {
      if (extents[axis] <= 1) {
        continue;
      }
      std::int64_t delta = 0;
      if (!checked_mul_nonnegative(strides[axis], extents[axis] - 1, &delta)) {
        result.reason = StorageFailureReason::kArithmeticOverflow;
        return result;
      }
      if (delta < 0) {
        if (!checked_add(min_offset, delta, &min_offset)) {
          result.reason = StorageFailureReason::kArithmeticOverflow;
          return result;
        }
      } else if (!checked_add(max_offset, delta, &max_offset)) {
        result.reason = StorageFailureReason::kArithmeticOverflow;
        return result;
      }
    }
    if (min_offset < 0) {
      result.reason = StorageFailureReason::kInvalidOffset;
      return result;
    }
    if (scalar_size > static_cast<std::uint64_t>(
                          (std::numeric_limits<std::int64_t>::max)()) ||
        !checked_add(max_offset, static_cast<std::int64_t>(scalar_size),
                     &properties.reachable_end)) {
      result.reason = StorageFailureReason::kArithmeticOverflow;
      return result;
    }
    properties.reachable_begin = min_offset;
  }

  properties.aligned =
      layout.byte_offset % static_cast<std::int64_t>(scalar_alignment) == 0;
  for (std::size_t axis = 0; axis < total_rank && properties.aligned; ++axis) {
    if (extents[axis] > 1 &&
        strides[axis] % static_cast<std::int64_t>(scalar_alignment) != 0) {
      properties.aligned = false;
    }
  }

  bool uniqueness_arithmetic_valid = true;
  properties.uniqueness =
      derive_uniqueness(extents, strides, total_rank, scalar_size,
                        properties.empty, &uniqueness_arithmetic_valid);
  if (!uniqueness_arithmetic_valid) {
    result.reason = StorageFailureReason::kArithmeticOverflow;
    return result;
  }

  std::vector<std::int64_t> aos_shape = layout.index_shape;
  aos_shape.insert(aos_shape.end(), layout.element_shape.begin(),
                   layout.element_shape.end());
  std::vector<std::int64_t> expected_aos;
  if (!canonical_strides(aos_shape, static_cast<std::int64_t>(scalar_size),
                         &expected_aos)) {
    result.reason = StorageFailureReason::kArithmeticOverflow;
    return result;
  }
  properties.canonical_aos = strides_equal(strides, extents, 0, expected_aos);

  std::vector<std::int64_t> soa_shape = layout.element_shape;
  soa_shape.insert(soa_shape.end(), layout.index_shape.begin(),
                   layout.index_shape.end());
  std::vector<std::int64_t> expected_soa;
  if (!canonical_strides(soa_shape, static_cast<std::int64_t>(scalar_size),
                         &expected_soa)) {
    result.reason = StorageFailureReason::kArithmeticOverflow;
    return result;
  }
  bool canonical_soa = true;
  for (std::size_t axis = 0; axis < index_rank; ++axis) {
    if (extents[axis] > 1) {
      canonical_soa &= strides[axis] == expected_soa[element_rank + axis];
    }
  }
  for (std::size_t axis = 0; axis < element_rank; ++axis) {
    if (extents[index_rank + axis] > 1) {
      canonical_soa &= strides[index_rank + axis] == expected_soa[axis];
    }
  }
  properties.canonical_soa = canonical_soa;

  if (element_rank == 0) {
    properties.array_layout = StorageArrayLayout::kScalar;
  } else if (properties.canonical_aos) {
    properties.array_layout = StorageArrayLayout::kAos;
  } else if (properties.canonical_soa) {
    properties.array_layout = StorageArrayLayout::kSoa;
  }
  properties.ndarray_abi_compatible =
      properties.canonical_aos || properties.canonical_soa;

  std::vector<std::int64_t> expected_element;
  if (!canonical_strides(layout.element_shape,
                         static_cast<std::int64_t>(scalar_size),
                         &expected_element)) {
    result.reason = StorageFailureReason::kArithmeticOverflow;
    return result;
  }
  properties.element_contiguous =
      strides_equal(strides, extents, index_rank, expected_element);

  std::uint64_t element_scalar_count = 1;
  for (std::int64_t extent : layout.element_shape) {
    if (!checked_mul_u64(element_scalar_count,
                         static_cast<std::uint64_t>(extent),
                         &element_scalar_count)) {
      result.reason = StorageFailureReason::kArithmeticOverflow;
      return result;
    }
  }
  std::uint64_t item_bytes = 0;
  if (!checked_mul_u64(element_scalar_count, scalar_size, &item_bytes) ||
      item_bytes > static_cast<std::uint64_t>(
                       (std::numeric_limits<std::int64_t>::max)())) {
    result.reason = StorageFailureReason::kArithmeticOverflow;
    return result;
  }
  properties.record_stride = static_cast<std::int64_t>(item_bytes);
  if (index_rank > 0 && strides[index_rank - 1] > 0) {
    properties.record_stride = strides[index_rank - 1];
  }
  std::vector<std::int64_t> expected_index;
  properties.single_record_stride_compatible =
      properties.element_contiguous &&
      canonical_strides(layout.index_shape, properties.record_stride,
                        &expected_index) &&
      strides_equal(strides, extents, 0, expected_index);

  if (properties.empty) {
    properties.compact_contiguous = true;
  } else {
    std::uint64_t compact_bytes = 0;
    properties.compact_contiguous =
        properties.uniqueness == StorageMappingUniqueness::kProvenUnique &&
        !properties.has_negative_stride &&
        checked_mul_u64(properties.scalar_count, scalar_size, &compact_bytes) &&
        properties.reachable_end >= properties.reachable_begin &&
        static_cast<std::uint64_t>(properties.reachable_end -
                                   properties.reachable_begin) == compact_bytes;
  }

  const std::uint64_t fingerprint = descriptor_fingerprint(
      owner, source_kind, layout.scalar_type, layout.access, index_rank,
      element_rank, extents, strides, layout.byte_offset);
  result.descriptor = DenseStorageDescriptor(
      std::move(owner), source_kind, layout.scalar_type, layout.access,
      static_cast<std::uint8_t>(index_rank),
      static_cast<std::uint8_t>(element_rank), extents, strides,
      layout.byte_offset, properties, fingerprint);
  return result;
}

StorageQualification qualify_dense_storage(
    const DenseStorageDescriptor &descriptor,
    const DenseStorageRequirement &requirement) {
  StorageQualification result;
  const DenseStorageProperties &properties = descriptor.properties();
  auto reject = [&](StorageFailureReason reason) {
    result.reason = reason;
    if (requirement.allow_materialization) {
      result.supported = true;
      result.execution_mode = StorageExecutionMode::kMaterialized;
      result.requires_materialization = true;
      checked_mul_u64(properties.scalar_count, properties.scalar_size,
                      &result.estimated_copy_bytes);
    }
    return result;
  };

  if (descriptor.owner().kind == StorageOwnerKind::kExternalManaged &&
      !requirement.accept_external_owner) {
    return reject(StorageFailureReason::kExternalOwnerNotAccepted);
  }
  if (requirement.require_scalar_type &&
      descriptor.scalar_type() != requirement.scalar_type) {
    return reject(StorageFailureReason::kUnsupportedDtype);
  }
  if (descriptor.index_rank() < requirement.min_index_rank ||
      descriptor.index_rank() > requirement.max_index_rank ||
      descriptor.element_rank() > requirement.max_element_rank) {
    return reject(StorageFailureReason::kUnsupportedRank);
  }
  if (requirement.require_writable &&
      !access_is_writable(descriptor.access())) {
    return reject(StorageFailureReason::kReadOnlySource);
  }
  if (!properties.aligned) {
    return reject(StorageFailureReason::kMisalignedRange);
  }
  if (requirement.require_unique_mapping) {
    if (properties.uniqueness == StorageMappingUniqueness::kProvenOverlapping) {
      return reject(StorageFailureReason::kInternalOverlap);
    }
    if (properties.uniqueness == StorageMappingUniqueness::kUnknown) {
      return reject(StorageFailureReason::kAliasUnknown);
    }
  }
  if (requirement.require_ndarray_abi && !properties.ndarray_abi_compatible) {
    return reject(StorageFailureReason::kUnsupportedLayout);
  }
  if (!requirement.accept_compact_subrange && descriptor.byte_offset() != 0) {
    return reject(StorageFailureReason::kUnsupportedLayout);
  }
  if (properties.compact_contiguous && properties.ndarray_abi_compatible) {
    result.supported = true;
    result.execution_mode = StorageExecutionMode::kDirectContiguous;
    return result;
  }
  if (properties.single_record_stride_compatible &&
      requirement.accept_single_record_stride) {
    result.supported = true;
    result.execution_mode = StorageExecutionMode::kDirectAffine;
    return result;
  }
  if (requirement.accept_general_affine) {
    result.supported = true;
    result.execution_mode = StorageExecutionMode::kDirectAffine;
    return result;
  }
  return reject(StorageFailureReason::kUnsupportedStride);
}

StorageAliasRelation analyze_logical_storage_alias(
    const DenseStorageDescriptor &lhs,
    const DenseStorageDescriptor &rhs) noexcept {
  const DenseStorageProperties &lhs_props = lhs.properties();
  const DenseStorageProperties &rhs_props = rhs.properties();
  if (lhs_props.empty || rhs_props.empty) {
    return StorageAliasRelation::kProvenDisjoint;
  }
  if (!lhs.owner().same_physical_owner(rhs.owner())) {
    if (is_program_ndarray_owner(lhs.owner()) &&
        is_program_ndarray_owner(rhs.owner()) &&
        lhs.owner().program_domain == rhs.owner().program_domain) {
      return StorageAliasRelation::kProvenDisjoint;
    }
    return StorageAliasRelation::kUnknown;
  }
  if (interval_disjoint(lhs_props.reachable_begin, lhs_props.reachable_end,
                        rhs_props.reachable_begin, rhs_props.reachable_end)) {
    return StorageAliasRelation::kProvenDisjoint;
  }
  if (lhs.fingerprint() == rhs.fingerprint() ||
      (lhs_props.compact_contiguous && rhs_props.compact_contiguous)) {
    return StorageAliasRelation::kProvenOverlap;
  }
  return StorageAliasRelation::kUnknown;
}

DenseStorageBuildResult describe_ndarray_storage(const Ndarray &array,
                                                 StorageAccess access) {
  DenseStorageBuildResult invalid;
  Program *program = array.owning_program();
  const RuntimeResourceHandle handle = array.runtime_resource_handle();
  if (program == nullptr || !handle) {
    invalid.reason = StorageFailureReason::kInvalidOwner;
    return invalid;
  }
  DenseStorageLayoutSpec layout;
  if (!make_ndarray_layout(array, &layout)) {
    invalid.reason = StorageFailureReason::kInvalidDtype;
    return invalid;
  }
  layout.access = access;
  return build_dense_storage_descriptor(
      StorageOwnerRef::program_ndarray(program->runtime_program_generation(),
                                       handle),
      StorageSourceKind::kNdarray, layout);
}

DenseStorageBuildResult flatten_dense_storage_to_scalar_vector(
    const DenseStorageDescriptor &descriptor) {
  DenseStorageBuildResult invalid;
  const DenseStorageProperties &properties = descriptor.properties();
  if (!properties.compact_contiguous ||
      properties.uniqueness != StorageMappingUniqueness::kProvenUnique ||
      properties.has_negative_stride || properties.reachable_begin < 0 ||
      properties.reachable_begin != descriptor.byte_offset() ||
      properties.scalar_size == 0 ||
      properties.scalar_count > static_cast<std::uint64_t>(
                                    (std::numeric_limits<std::int64_t>::max)()) ||
      properties.scalar_size > static_cast<std::uint64_t>(
                                   (std::numeric_limits<std::int64_t>::max)())) {
    invalid.reason = StorageFailureReason::kUnsupportedLayout;
    return invalid;
  }

  DenseStorageLayoutSpec layout;
  layout.scalar_type = descriptor.scalar_type();
  layout.index_shape = {
      static_cast<std::int64_t>(properties.scalar_count)};
  layout.index_strides_bytes = {
      static_cast<std::int64_t>(properties.scalar_size)};
  layout.byte_offset = descriptor.byte_offset();
  layout.access = descriptor.access();
  return build_dense_storage_descriptor(
      descriptor.owner(), descriptor.source_kind(), layout);
}

DenseStorageBuildResult describe_struct_member_storage(
    const Ndarray &base,
    DataType scalar_type,
    const std::vector<std::int64_t> &index_shape,
    const std::vector<std::int64_t> &element_shape,
    std::int64_t byte_offset,
    std::int64_t record_stride,
    StorageSourceKind source_kind,
    StorageAccess access) {
  DenseStorageBuildResult invalid;
  Program *program = base.owning_program();
  const RuntimeResourceHandle handle = base.runtime_resource_handle();
  if (program == nullptr || !handle) {
    invalid.reason = StorageFailureReason::kInvalidOwner;
    return invalid;
  }
  if (record_stride <= 0 || byte_offset < 0 ||
      static_cast<std::uint64_t>(record_stride) > base.get_element_size()) {
    invalid.reason = StorageFailureReason::kInvalidOffset;
    return invalid;
  }

  std::uint64_t scalar_size = 0;
  std::uint64_t scalar_alignment = 0;
  if (!scalar_type_info(scalar_type, &scalar_size, &scalar_alignment)) {
    invalid.reason = StorageFailureReason::kInvalidDtype;
    return invalid;
  }
  std::uint64_t lane_count = 1;
  for (std::int64_t extent : element_shape) {
    if (extent < 0 ||
        !checked_mul_u64(lane_count, static_cast<std::uint64_t>(extent),
                         &lane_count)) {
      invalid.reason = extent < 0 ? StorageFailureReason::kInvalidShape
                                  : StorageFailureReason::kArithmeticOverflow;
      return invalid;
    }
  }
  std::uint64_t member_bytes = 0;
  if (!checked_mul_u64(lane_count, scalar_size, &member_bytes) ||
      static_cast<std::uint64_t>(byte_offset) + member_bytes >
          base.get_element_size()) {
    invalid.reason = StorageFailureReason::kInvalidOffset;
    return invalid;
  }

  DenseStorageLayoutSpec layout;
  layout.scalar_type = scalar_type;
  layout.index_shape = index_shape;
  layout.element_shape = element_shape;
  layout.byte_offset = byte_offset;
  layout.access = access;
  if (!canonical_strides(index_shape, record_stride,
                         &layout.index_strides_bytes) ||
      !canonical_strides(element_shape, static_cast<std::int64_t>(scalar_size),
                         &layout.element_strides_bytes)) {
    invalid.reason = StorageFailureReason::kArithmeticOverflow;
    return invalid;
  }
  return build_dense_storage_descriptor(
      StorageOwnerRef::program_ndarray(program->runtime_program_generation(),
                                       handle),
      source_kind, layout);
}

DenseStorageBuildResult describe_dense_field_storage(
    Program &program,
    SNode *anchor,
    const std::vector<SNode *> &components,
    DataType scalar_type,
    const std::vector<std::int64_t> &index_shape,
    const std::vector<std::int64_t> &element_shape,
    StorageAccess access) {
  DenseStorageBuildResult invalid;
  if (anchor == nullptr || anchor->type != SNodeType::place ||
      components.empty()) {
    invalid.reason = StorageFailureReason::kUnsupportedLayout;
    return invalid;
  }
  std::uint64_t scalar_size = 0;
  std::uint64_t scalar_alignment = 0;
  if (!scalar_type_info(scalar_type, &scalar_size, &scalar_alignment)) {
    invalid.reason = StorageFailureReason::kInvalidDtype;
    return invalid;
  }

  SNode *parent = anchor->parent;
  SNode *root_child = nullptr;
  std::int64_t record_stride = static_cast<std::int64_t>(scalar_size);
  if (parent != nullptr && parent->type == SNodeType::root) {
    root_child = anchor;
  } else if (parent != nullptr && parent->type == SNodeType::dense &&
             parent->parent != nullptr &&
             parent->parent->type == SNodeType::root) {
    root_child = parent;
    if (parent->cell_size_bytes >
        static_cast<std::size_t>((std::numeric_limits<std::int64_t>::max)())) {
      invalid.reason = StorageFailureReason::kArithmeticOverflow;
      return invalid;
    }
    record_stride = static_cast<std::int64_t>(parent->cell_size_bytes);
  } else {
    invalid.reason = StorageFailureReason::kUnsupportedLayout;
    return invalid;
  }

  const int tree_id = root_child->parent->get_snode_tree_id();
  std::vector<int> component_ids;
  component_ids.reserve(components.size());
  for (std::size_t lane = 0; lane < components.size(); ++lane) {
    SNode *component = components[lane];
    if (component == nullptr || component->type != SNodeType::place ||
        component->parent != parent ||
        component->get_snode_tree_id() != tree_id) {
      invalid.reason = StorageFailureReason::kUnsupportedLayout;
      return invalid;
    }
    if (components.size() > 1 &&
        component->offset_bytes_in_parent_cell != lane * scalar_size) {
      invalid.reason = StorageFailureReason::kUnsupportedLayout;
      return invalid;
    }
    component_ids.push_back(component->id);
  }
  std::uint64_t lane_count = 1;
  for (std::int64_t extent : element_shape) {
    if (extent < 0 ||
        !checked_mul_u64(lane_count, static_cast<std::uint64_t>(extent),
                         &lane_count)) {
      invalid.reason = extent < 0 ? StorageFailureReason::kInvalidShape
                                  : StorageFailureReason::kArithmeticOverflow;
      return invalid;
    }
  }
  if (lane_count != components.size() ||
      (components.size() > 1 && static_cast<std::uint64_t>(record_stride) !=
                                    scalar_size * components.size())) {
    invalid.reason = StorageFailureReason::kUnsupportedLayout;
    return invalid;
  }

  std::vector<SNodeTreeDependency> dependencies;
  try {
    dependencies = program.snapshot_snode_tree_dependencies({tree_id});
  } catch (...) {
    invalid.reason = StorageFailureReason::kStaleOwner;
    return invalid;
  }
  if (dependencies.size() != 1) {
    invalid.reason = StorageFailureReason::kStaleOwner;
    return invalid;
  }
  std::size_t field_offset = 0;
  try {
    field_offset = program.get_dense_field_device_ptr(anchor).offset;
  } catch (...) {
    invalid.reason = StorageFailureReason::kUnsupportedLayout;
    return invalid;
  }
  if (field_offset >
      static_cast<std::size_t>((std::numeric_limits<std::int64_t>::max)())) {
    invalid.reason = StorageFailureReason::kArithmeticOverflow;
    return invalid;
  }

  DenseStorageLayoutSpec layout;
  layout.scalar_type = scalar_type;
  layout.index_shape = index_shape;
  layout.element_shape = element_shape;
  layout.byte_offset = static_cast<std::int64_t>(field_offset);
  layout.access = access;
  if (!canonical_strides(index_shape, record_stride,
                         &layout.index_strides_bytes) ||
      !canonical_strides(element_shape, static_cast<std::int64_t>(scalar_size),
                         &layout.element_strides_bytes)) {
    invalid.reason = StorageFailureReason::kArithmeticOverflow;
    return invalid;
  }
  const StorageSourceKind source_kind =
      components.size() == 1 ? StorageSourceKind::kDenseScalarField
                             : StorageSourceKind::kDensePackedField;
  return build_dense_storage_descriptor(
      StorageOwnerRef::snode_payload(program.runtime_program_generation(),
                                     dependencies.front(), anchor->id,
                                     std::move(component_ids)),
      source_kind, layout);
}

StorageFailureReason validate_storage_owner(
    Program &program,
    const DenseStorageDescriptor &descriptor) noexcept {
  const StorageOwnerRef &owner = descriptor.owner();
  if (!owner.valid()) {
    return StorageFailureReason::kInvalidOwner;
  }
  if ((owner.kind == StorageOwnerKind::kProgramNdarray ||
       owner.kind == StorageOwnerKind::kSNodePayload) &&
      owner.program_domain != program.runtime_program_generation()) {
    return StorageFailureReason::kDifferentProgram;
  }
  try {
    if (owner.kind == StorageOwnerKind::kProgramNdarray) {
      auto lease = program.acquire_ndarray_external_lease(owner.ndarray_handle);
      return lease ? StorageFailureReason::kNone
                   : StorageFailureReason::kStaleOwner;
    }
    if (owner.kind == StorageOwnerKind::kSNodePayload) {
      const auto current =
          program.snapshot_snode_tree_dependencies({owner.tree.tree_id});
      if (current.size() != 1) {
        return StorageFailureReason::kStaleOwner;
      }
      if (current.front().generation != owner.tree.generation) {
        return StorageFailureReason::kRetiredGeneration;
      }
      if (current.front().layout_fingerprint != owner.tree.layout_fingerprint) {
        return StorageFailureReason::kStaleOwner;
      }
    }
    if (owner.kind == StorageOwnerKind::kExternalManaged) {
      return program.validate_external_dense_storage_owner(owner)
                 ? StorageFailureReason::kNone
                 : StorageFailureReason::kStaleOwner;
    }
    return StorageFailureReason::kNone;
  } catch (...) {
    return StorageFailureReason::kStaleOwner;
  }
}

#define STORAGE_ENUM_TO_STRING_CASE(name) \
  case decltype(value)::name:             \
    return #name

const char *to_string(StorageOwnerKind value) noexcept {
  switch (value) {
    STORAGE_ENUM_TO_STRING_CASE(kInvalid);
    STORAGE_ENUM_TO_STRING_CASE(kProgramNdarray);
    STORAGE_ENUM_TO_STRING_CASE(kSNodePayload);
    STORAGE_ENUM_TO_STRING_CASE(kExternalManaged);
    STORAGE_ENUM_TO_STRING_CASE(kSubmissionScopedHost);
  }
  return "kInvalid";
}

const char *to_string(StorageSourceKind value) noexcept {
  switch (value) {
    STORAGE_ENUM_TO_STRING_CASE(kUnknown);
    STORAGE_ENUM_TO_STRING_CASE(kNdarray);
    STORAGE_ENUM_TO_STRING_CASE(kDenseScalarField);
    STORAGE_ENUM_TO_STRING_CASE(kDensePackedField);
    STORAGE_ENUM_TO_STRING_CASE(kStructScalarMember);
    STORAGE_ENUM_TO_STRING_CASE(kStructTensorMember);
    STORAGE_ENUM_TO_STRING_CASE(kExternalDense);
  }
  return "kUnknown";
}

const char *to_string(StorageAccess value) noexcept {
  switch (value) {
    STORAGE_ENUM_TO_STRING_CASE(kReadOnly);
    STORAGE_ENUM_TO_STRING_CASE(kWriteOnly);
    STORAGE_ENUM_TO_STRING_CASE(kReadWrite);
  }
  return "kReadWrite";
}

const char *to_string(StorageMappingUniqueness value) noexcept {
  switch (value) {
    STORAGE_ENUM_TO_STRING_CASE(kProvenUnique);
    STORAGE_ENUM_TO_STRING_CASE(kProvenOverlapping);
    STORAGE_ENUM_TO_STRING_CASE(kUnknown);
  }
  return "kUnknown";
}

const char *to_string(StorageAliasRelation value) noexcept {
  switch (value) {
    STORAGE_ENUM_TO_STRING_CASE(kProvenDisjoint);
    STORAGE_ENUM_TO_STRING_CASE(kProvenOverlap);
    STORAGE_ENUM_TO_STRING_CASE(kUnknown);
  }
  return "kUnknown";
}

const char *to_string(StorageExecutionMode value) noexcept {
  switch (value) {
    STORAGE_ENUM_TO_STRING_CASE(kDirectContiguous);
    STORAGE_ENUM_TO_STRING_CASE(kDirectAffine);
    STORAGE_ENUM_TO_STRING_CASE(kComponentBundle);
    STORAGE_ENUM_TO_STRING_CASE(kIndexedProjection);
    STORAGE_ENUM_TO_STRING_CASE(kMaterialized);
    STORAGE_ENUM_TO_STRING_CASE(kUnsupported);
  }
  return "kUnsupported";
}

const char *to_string(StorageArrayLayout value) noexcept {
  switch (value) {
    STORAGE_ENUM_TO_STRING_CASE(kNone);
    STORAGE_ENUM_TO_STRING_CASE(kScalar);
    STORAGE_ENUM_TO_STRING_CASE(kAos);
    STORAGE_ENUM_TO_STRING_CASE(kSoa);
  }
  return "kNone";
}

const char *to_string(StorageFailureReason value) noexcept {
  switch (value) {
    STORAGE_ENUM_TO_STRING_CASE(kNone);
    STORAGE_ENUM_TO_STRING_CASE(kInvalidOwner);
    STORAGE_ENUM_TO_STRING_CASE(kInvalidDtype);
    STORAGE_ENUM_TO_STRING_CASE(kInvalidRank);
    STORAGE_ENUM_TO_STRING_CASE(kInvalidShape);
    STORAGE_ENUM_TO_STRING_CASE(kInvalidOffset);
    STORAGE_ENUM_TO_STRING_CASE(kArithmeticOverflow);
    STORAGE_ENUM_TO_STRING_CASE(kStaleOwner);
    STORAGE_ENUM_TO_STRING_CASE(kRetiredGeneration);
    STORAGE_ENUM_TO_STRING_CASE(kDifferentProgram);
    STORAGE_ENUM_TO_STRING_CASE(kUnsupportedStorageKind);
    STORAGE_ENUM_TO_STRING_CASE(kUnsupportedBackend);
    STORAGE_ENUM_TO_STRING_CASE(kUnsupportedDtype);
    STORAGE_ENUM_TO_STRING_CASE(kUnsupportedRank);
    STORAGE_ENUM_TO_STRING_CASE(kUnsupportedLayout);
    STORAGE_ENUM_TO_STRING_CASE(kUnsupportedStride);
    STORAGE_ENUM_TO_STRING_CASE(kMisalignedRange);
    STORAGE_ENUM_TO_STRING_CASE(kInternalOverlap);
    STORAGE_ENUM_TO_STRING_CASE(kAliasUnknown);
    STORAGE_ENUM_TO_STRING_CASE(kWriteAlias);
    STORAGE_ENUM_TO_STRING_CASE(kReadOnlySource);
    STORAGE_ENUM_TO_STRING_CASE(kExternalOwnerNotAccepted);
    STORAGE_ENUM_TO_STRING_CASE(kExternalSyncUnavailable);
    STORAGE_ENUM_TO_STRING_CASE(kGraphIdentityUnstable);
    STORAGE_ENUM_TO_STRING_CASE(kCopyForbidden);
    STORAGE_ENUM_TO_STRING_CASE(kMaterializationUnavailable);
  }
  return "kUnsupportedLayout";
}

#undef STORAGE_ENUM_TO_STRING_CASE

}  // namespace taichi::lang::storage
