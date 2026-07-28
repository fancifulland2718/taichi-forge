#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <set>
#include <vector>

#include "taichi/program/storage_view.h"

namespace taichi::lang::storage {
namespace {

StorageOwnerRef host_owner(std::uint64_t domain = 1) {
  return StorageOwnerRef::submission_scoped_host(domain);
}

DenseStorageBuildResult build_f32(
    std::vector<std::int64_t> index_shape,
    std::vector<std::int64_t> index_strides,
    std::vector<std::int64_t> element_shape = {},
    std::vector<std::int64_t> element_strides = {},
    std::int64_t byte_offset = 0,
    StorageOwnerRef owner = host_owner(),
    StorageSourceKind source = StorageSourceKind::kExternalDense,
    StorageAccess access = StorageAccess::kReadWrite) {
  DenseStorageLayoutSpec spec;
  spec.scalar_type = PrimitiveType::f32;
  spec.index_shape = std::move(index_shape);
  spec.index_strides_bytes = std::move(index_strides);
  spec.element_shape = std::move(element_shape);
  spec.element_strides_bytes = std::move(element_strides);
  spec.byte_offset = byte_offset;
  spec.access = access;
  return build_dense_storage_descriptor(std::move(owner), source, spec);
}

TEST(StorageViewTest, CanonicalAosAndSoaRemainDistinct) {
  auto aos = build_f32({3, 5}, {40, 8}, {2}, {4});
  ASSERT_TRUE(aos);
  const auto &aos_properties = aos.descriptor->properties();
  EXPECT_TRUE(aos_properties.aligned);
  EXPECT_TRUE(aos_properties.compact_contiguous);
  EXPECT_TRUE(aos_properties.element_contiguous);
  EXPECT_TRUE(aos_properties.canonical_aos);
  EXPECT_FALSE(aos_properties.canonical_soa);
  EXPECT_TRUE(aos_properties.ndarray_abi_compatible);
  EXPECT_TRUE(aos_properties.single_record_stride_compatible);
  EXPECT_EQ(aos_properties.array_layout, StorageArrayLayout::kAos);
  EXPECT_EQ(aos_properties.scalar_count, 30u);
  EXPECT_EQ(aos_properties.item_count, 15u);
  EXPECT_EQ(aos_properties.reachable_begin, 0);
  EXPECT_EQ(aos_properties.reachable_end, 120);
  EXPECT_EQ(aos_properties.record_stride, 8);

  auto soa = build_f32({3}, {4}, {2}, {12});
  ASSERT_TRUE(soa);
  const auto &soa_properties = soa.descriptor->properties();
  EXPECT_TRUE(soa_properties.compact_contiguous);
  EXPECT_FALSE(soa_properties.element_contiguous);
  EXPECT_FALSE(soa_properties.canonical_aos);
  EXPECT_TRUE(soa_properties.canonical_soa);
  EXPECT_TRUE(soa_properties.ndarray_abi_compatible);
  EXPECT_FALSE(soa_properties.single_record_stride_compatible);
  EXPECT_EQ(soa_properties.array_layout, StorageArrayLayout::kSoa);
  EXPECT_EQ(soa_properties.reachable_end, 24);
}

TEST(StorageViewTest, RecordStrideIsQualifiedWithoutPretendingGeneralAffine) {
  auto member = build_f32({7}, {16}, {}, {}, 4, host_owner(3),
                          StorageSourceKind::kStructScalarMember);
  ASSERT_TRUE(member);
  const auto &properties = member.descriptor->properties();
  EXPECT_FALSE(properties.compact_contiguous);
  EXPECT_TRUE(properties.element_contiguous);
  EXPECT_FALSE(properties.ndarray_abi_compatible);
  EXPECT_TRUE(properties.single_record_stride_compatible);
  EXPECT_EQ(properties.record_stride, 16);
  EXPECT_EQ(properties.reachable_begin, 4);
  EXPECT_EQ(properties.reachable_end, 104);

  DenseStorageRequirement contiguous_only;
  contiguous_only.require_unique_mapping = true;
  auto rejected = qualify_dense_storage(*member.descriptor, contiguous_only);
  EXPECT_FALSE(rejected.supported);
  EXPECT_EQ(rejected.reason, StorageFailureReason::kUnsupportedStride);

  DenseStorageRequirement record_stride;
  record_stride.require_unique_mapping = true;
  record_stride.accept_single_record_stride = true;
  auto accepted = qualify_dense_storage(*member.descriptor, record_stride);
  EXPECT_TRUE(accepted.supported);
  EXPECT_EQ(accepted.execution_mode, StorageExecutionMode::kDirectAffine);
  EXPECT_FALSE(accepted.requires_materialization);
}

TEST(StorageViewTest, WritableQualificationRejectsOverlapAndMisalignment) {
  auto broadcast = build_f32({4}, {0});
  ASSERT_TRUE(broadcast);
  EXPECT_EQ(broadcast.descriptor->properties().uniqueness,
            StorageMappingUniqueness::kProvenOverlapping);

  DenseStorageRequirement writable;
  writable.require_unique_mapping = true;
  writable.require_writable = true;
  writable.accept_general_affine = true;
  auto overlap = qualify_dense_storage(*broadcast.descriptor, writable);
  EXPECT_FALSE(overlap.supported);
  EXPECT_EQ(overlap.reason, StorageFailureReason::kInternalOverlap);

  auto misaligned = build_f32({4}, {4}, {}, {}, 2);
  ASSERT_TRUE(misaligned);
  EXPECT_FALSE(misaligned.descriptor->properties().aligned);
  auto alignment = qualify_dense_storage(*misaligned.descriptor, writable);
  EXPECT_FALSE(alignment.supported);
  EXPECT_EQ(alignment.reason, StorageFailureReason::kMisalignedRange);

  DenseStorageRequirement materialize = writable;
  materialize.allow_materialization = true;
  auto fallback = qualify_dense_storage(*broadcast.descriptor, materialize);
  EXPECT_TRUE(fallback.supported);
  EXPECT_EQ(fallback.execution_mode, StorageExecutionMode::kMaterialized);
  EXPECT_TRUE(fallback.requires_materialization);
  EXPECT_EQ(fallback.estimated_copy_bytes, 16u);
}

TEST(StorageViewTest, NegativeStrideIsDescribedButNotCalledContiguous) {
  auto reversed = build_f32({3}, {-4}, {}, {}, 8);
  ASSERT_TRUE(reversed);
  const auto &properties = reversed.descriptor->properties();
  EXPECT_TRUE(properties.has_negative_stride);
  EXPECT_EQ(properties.reachable_begin, 0);
  EXPECT_EQ(properties.reachable_end, 12);
  EXPECT_EQ(properties.uniqueness, StorageMappingUniqueness::kProvenUnique);
  EXPECT_FALSE(properties.ndarray_abi_compatible);
  EXPECT_FALSE(properties.compact_contiguous);

  DenseStorageRequirement general;
  general.require_unique_mapping = true;
  general.accept_general_affine = true;
  auto qualification = qualify_dense_storage(*reversed.descriptor, general);
  EXPECT_TRUE(qualification.supported);
  EXPECT_EQ(qualification.execution_mode, StorageExecutionMode::kDirectAffine);
}

TEST(StorageViewTest, CheckedArithmeticRejectsInvalidMappings) {
  auto invalid_owner = build_f32({4}, {4}, {}, {}, 0, StorageOwnerRef{});
  EXPECT_FALSE(invalid_owner);
  EXPECT_EQ(invalid_owner.reason, StorageFailureReason::kInvalidOwner);

  auto invalid_rank = build_f32({2}, {});
  EXPECT_FALSE(invalid_rank);
  EXPECT_EQ(invalid_rank.reason, StorageFailureReason::kInvalidRank);

  auto invalid_shape = build_f32({-1}, {4});
  EXPECT_FALSE(invalid_shape);
  EXPECT_EQ(invalid_shape.reason, StorageFailureReason::kInvalidShape);

  auto invalid_offset = build_f32({2}, {-4}, {}, {}, 0);
  EXPECT_FALSE(invalid_offset);
  EXPECT_EQ(invalid_offset.reason, StorageFailureReason::kInvalidOffset);

  auto overflow = build_f32({3}, {(std::numeric_limits<std::int64_t>::max)()});
  EXPECT_FALSE(overflow);
  EXPECT_EQ(overflow.reason, StorageFailureReason::kArithmeticOverflow);
}

TEST(StorageViewTest, FingerprintContainsStableOwnerAndLogicalMapping) {
  auto first = build_f32({4, 5}, {20, 4}, {}, {}, 0, host_owner(11));
  auto same = build_f32({4, 5}, {20, 4}, {}, {}, 0, host_owner(11));
  auto different_offset = build_f32({4, 5}, {20, 4}, {}, {}, 4, host_owner(11));
  auto different_owner = build_f32({4, 5}, {20, 4}, {}, {}, 0, host_owner(12));
  ASSERT_TRUE(first);
  ASSERT_TRUE(same);
  ASSERT_TRUE(different_offset);
  ASSERT_TRUE(different_owner);
  EXPECT_EQ(first.descriptor->fingerprint(), same.descriptor->fingerprint());
  EXPECT_NE(first.descriptor->fingerprint(),
            different_offset.descriptor->fingerprint());
  EXPECT_NE(first.descriptor->fingerprint(),
            different_owner.descriptor->fingerprint());
}

TEST(StorageViewTest, LogicalAliasAnalysisIsTriState) {
  RuntimeResourceHandle handle{51, 7, 0, 1};
  RuntimeResourceHandle other_handle{51, 7, 1, 1};
  auto owner = StorageOwnerRef::program_ndarray(61, handle);
  auto other_owner = StorageOwnerRef::program_ndarray(61, other_handle);

  auto first =
      build_f32({4}, {4}, {}, {}, 0, owner, StorageSourceKind::kNdarray);
  auto adjacent =
      build_f32({4}, {4}, {}, {}, 16, owner, StorageSourceKind::kNdarray);
  auto overlapping =
      build_f32({4}, {4}, {}, {}, 8, owner, StorageSourceKind::kNdarray);
  auto other =
      build_f32({4}, {4}, {}, {}, 0, other_owner, StorageSourceKind::kNdarray);
  ASSERT_TRUE(first);
  ASSERT_TRUE(adjacent);
  ASSERT_TRUE(overlapping);
  ASSERT_TRUE(other);

  EXPECT_EQ(
      analyze_logical_storage_alias(*first.descriptor, *adjacent.descriptor),
      StorageAliasRelation::kProvenDisjoint);
  EXPECT_EQ(
      analyze_logical_storage_alias(*first.descriptor, *overlapping.descriptor),
      StorageAliasRelation::kProvenOverlap);
  EXPECT_EQ(analyze_logical_storage_alias(*first.descriptor, *other.descriptor),
            StorageAliasRelation::kProvenDisjoint);

  auto external_a = build_f32({4}, {4}, {}, {}, 0, host_owner(1));
  auto external_b = build_f32({4}, {4}, {}, {}, 0, host_owner(2));
  EXPECT_EQ(analyze_logical_storage_alias(*external_a.descriptor,
                                          *external_b.descriptor),
            StorageAliasRelation::kUnknown);
}

TEST(StorageViewTest, RandomizedPropertiesNeverOverclaimAddressMapping) {
  std::mt19937_64 random(0x5a17c0de);
  std::uniform_int_distribution<int> rank_distribution(1, 3);
  std::uniform_int_distribution<int> extent_distribution(1, 4);
  std::uniform_int_distribution<int> stride_distribution(0, 12);

  for (int sample = 0; sample < 2000; ++sample) {
    const int rank = rank_distribution(random);
    std::vector<std::int64_t> shape(rank);
    std::vector<std::int64_t> strides(rank);
    for (int axis = 0; axis < rank; ++axis) {
      shape[axis] = extent_distribution(random);
      strides[axis] = 4 * stride_distribution(random);
    }
    auto built = build_f32(shape, strides);
    ASSERT_TRUE(built);
    const auto &descriptor = *built.descriptor;
    const auto &properties = descriptor.properties();

    std::vector<std::int64_t> addresses{descriptor.byte_offset()};
    for (int axis = 0; axis < rank; ++axis) {
      std::vector<std::int64_t> expanded;
      expanded.reserve(addresses.size() *
                       static_cast<std::size_t>(shape[axis]));
      for (std::int64_t address : addresses) {
        for (std::int64_t index = 0; index < shape[axis]; ++index) {
          expanded.push_back(address + index * strides[axis]);
        }
      }
      addresses = std::move(expanded);
    }
    const auto [minimum, maximum] =
        std::minmax_element(addresses.begin(), addresses.end());
    EXPECT_EQ(properties.reachable_begin, *minimum);
    EXPECT_EQ(properties.reachable_end, *maximum + 4);

    const std::set<std::int64_t> unique(addresses.begin(), addresses.end());
    const bool oracle_unique = unique.size() == addresses.size();
    if (properties.uniqueness == StorageMappingUniqueness::kProvenUnique) {
      EXPECT_TRUE(oracle_unique);
    } else if (properties.uniqueness ==
               StorageMappingUniqueness::kProvenOverlapping) {
      EXPECT_FALSE(oracle_unique);
    }
    if (properties.compact_contiguous) {
      EXPECT_TRUE(oracle_unique);
      EXPECT_EQ(properties.reachable_end - properties.reachable_begin,
                static_cast<std::int64_t>(addresses.size() * 4));
    }
  }
}

TEST(StorageViewTest, RuntimeArgumentSeparatesBindingReplayAndCapture) {
  RuntimeResourceHandle handle{71, 4, 2, 9};
  auto built = build_f32({8}, {4}, {}, {}, 0,
                         StorageOwnerRef::program_ndarray(81, handle),
                         StorageSourceKind::kNdarray);
  ASSERT_TRUE(built);

  RuntimeStorageRequirement ordinary;
  ordinary.backend = Arch::cuda;
  ordinary.dense.require_ndarray_abi = true;
  ordinary.dense.require_unique_mapping = true;
  ordinary.dense.require_writable = true;
  RuntimeStorageArgument ordinary_argument(*built.descriptor, ordinary);
  const auto &ordinary_caps = ordinary_argument.qualification().capabilities;
  EXPECT_TRUE(ordinary_caps.describable);
  EXPECT_TRUE(ordinary_caps.bindable);
  EXPECT_TRUE(ordinary_caps.replayable);
  EXPECT_FALSE(ordinary_caps.capturable);
  EXPECT_TRUE(ordinary_caps.zero_copy_qualified);
  EXPECT_EQ(ordinary_argument.qualification().reason,
            StorageFailureReason::kNone);

  RuntimeStorageRequirement capture = ordinary;
  capture.consumer = RuntimeStorageConsumer::kGraphCapture;
  capture.mode = RuntimeStorageMode::kCapture;
  RuntimeStorageArgument capture_argument(*built.descriptor, capture);
  EXPECT_TRUE(capture_argument.qualification().capabilities.capturable);
  EXPECT_NE(capture_argument.stable_signature(),
            ordinary_argument.stable_signature());

  RuntimeStorageArgument same_capture(*built.descriptor, capture);
  EXPECT_EQ(same_capture.stable_signature(),
            capture_argument.stable_signature());
}

TEST(StorageViewTest, RuntimeArgumentFailsClosedForIdentityAndSync) {
  auto host = build_f32({4}, {4});
  ASSERT_TRUE(host);
  RuntimeStorageRequirement replay;
  replay.backend = Arch::vulkan;
  replay.consumer = RuntimeStorageConsumer::kGraphReplay;
  replay.mode = RuntimeStorageMode::kReplay;
  replay.dense.require_ndarray_abi = true;
  replay.dense.require_unique_mapping = true;
  RuntimeStorageArgument host_argument(*host.descriptor, replay);
  EXPECT_TRUE(host_argument.qualification().capabilities.bindable);
  EXPECT_FALSE(host_argument.qualification().capabilities.replayable);
  EXPECT_EQ(host_argument.qualification().reason,
            StorageFailureReason::kGraphIdentityUnstable);

  auto external = build_f32({4}, {4}, {}, {}, 0,
                            StorageOwnerRef::external_managed(91, 3, 7),
                            StorageSourceKind::kExternalDense);
  ASSERT_TRUE(external);
  RuntimeStorageRequirement external_replay = replay;
  external_replay.dense.accept_external_owner = true;
  external_replay.require_external_sync = true;
  RuntimeStorageArgument missing_sync(*external.descriptor, external_replay);
  EXPECT_FALSE(missing_sync.qualification().capabilities.bindable);
  EXPECT_FALSE(missing_sync.qualification().capabilities.zero_copy_qualified);
  EXPECT_EQ(missing_sync.qualification().reason,
            StorageFailureReason::kExternalSyncUnavailable);

  RuntimeStorageArgument synchronized(*external.descriptor, external_replay,
                                      0x1234);
  EXPECT_TRUE(synchronized.qualification().capabilities.bindable);
  EXPECT_TRUE(synchronized.qualification().capabilities.replayable);
  EXPECT_TRUE(synchronized.qualification().capabilities.zero_copy_qualified);
  EXPECT_EQ(synchronized.qualification().reason, StorageFailureReason::kNone);
  EXPECT_NE(synchronized.stable_signature(), missing_sync.stable_signature());
}

TEST(StorageViewTest, DescriptorMetadataStaysInlineAndBounded) {
  EXPECT_LE(sizeof(DenseStorageDescriptor), 512u);
  EXPECT_LE(sizeof(RuntimeStorageArgument), 640u);
}

}  // namespace
}  // namespace taichi::lang::storage
