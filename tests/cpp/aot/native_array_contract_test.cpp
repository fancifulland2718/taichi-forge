#include "gtest/gtest.h"

#include <limits>
#include <random>
#include <set>
#include "taichi/runtime/native_array_contract.h"
#ifdef TI_WITH_LLVM
#include "taichi/rhi/cpu/cpu_device.h"
#endif
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#endif

namespace taichi::lang {
namespace {
NativeArrayBinding member(std::uint64_t count = 17,
                          std::int64_t stride = 20,
                          std::int64_t offset = 4) {
  NativeArrayBinding array;
  array.scalar_type = PrimitiveType::f32;
  array.shape = {count};
  array.bytes = 340;
  array.strides = std::make_shared<const NativeArrayStrides>(
      NativeArrayStrides{{stride}, {}, offset});
  return array;
}

TEST(NativeArrayContract, AffineBindingsRequireExplicitProviderOptIn) {
  auto array = member();
  NativeBinding binding(array);
  EXPECT_THROW(binding.array(), std::invalid_argument);
  EXPECT_EQ(binding.dense_array().strides, array.strides);
  EXPECT_THROW(validate_native_array_layout(
                   array, {{17}, {}, 4, 4, true, "f32"}),
               std::invalid_argument);
  // Even an explicitly packed layout must pass an affine-aware provider.
  array = member(17, 4, 0);
  EXPECT_THROW(NativeBinding(array).array(), std::invalid_argument);
  array.strides.reset();
  EXPECT_NO_THROW(NativeBinding(array).array());
  EXPECT_NO_THROW(validate_native_array_layout(
      array, {{17}, {}, 4, 4, true, "f32"}));
}

TEST(NativeArrayContract, LogicalPayloadDoesNotReplaceAccessibleSpan) {
  auto array = member();
  auto facts = validate_native_dense_array_layout(array);
  EXPECT_EQ(facts.properties.scalar_count, 17);
  EXPECT_EQ(facts.properties.record_stride, 20);
  EXPECT_EQ(facts.properties.reachable_begin, 4);
  EXPECT_EQ(facts.properties.reachable_end, 328);
  EXPECT_FALSE(facts.properties.compact_contiguous);
  array.bytes = 68;  // Logical payload is not its physical span.
  EXPECT_THROW(validate_native_dense_array_layout(array), std::invalid_argument);
  array.bytes = 328;
  EXPECT_NO_THROW(validate_native_dense_array_layout(array));
}

TEST(NativeArrayContract, ScalarPrefixFlattensUniformRecordsWithoutRepacking) {
  auto array = member();
  array.shape = {3, 5};
  array.strides = std::make_shared<const NativeArrayStrides>(
      NativeArrayStrides{{100, 20}, {}, 4});
  const auto prefix = native_scalar_prefix(array, 11, PrimitiveType::f32);
  EXPECT_EQ(prefix.shape, (std::vector<std::uint64_t>{11}));
  EXPECT_EQ(prefix.strides->index_bytes, (std::vector<std::int64_t>{20}));
  EXPECT_EQ(prefix.strides->byte_offset, 4);
  EXPECT_EQ(prefix.bytes, array.bytes);
  EXPECT_EQ(prefix.pointer, array.pointer);
  EXPECT_EQ(validate_native_dense_array_layout(prefix).properties.reachable_end, 208);
  EXPECT_THROW(native_scalar_prefix(array, 16, PrimitiveType::f32), std::invalid_argument);
  EXPECT_THROW(native_scalar_prefix(array, 0, PrimitiveType::f32), std::invalid_argument);
  EXPECT_THROW(native_scalar_prefix(array, 11, PrimitiveType::i32), std::invalid_argument);
  array.shape = {};
  array.strides = std::make_shared<const NativeArrayStrides>(
      NativeArrayStrides{{}, {}, 8});
  EXPECT_EQ(native_scalar_prefix(array, 1, PrimitiveType::f32).strides->index_bytes,
            (std::vector<std::int64_t>{4}));
  array = member(17, 0, 4);
  EXPECT_THROW(native_scalar_prefix(array, 1, PrimitiveType::f32), std::invalid_argument);
  array = member(17, -20, 324);
  EXPECT_THROW(native_scalar_prefix(array, 11, PrimitiveType::f32), std::invalid_argument);
}

TEST(NativeArrayContract, RejectsMalformedLayoutsBeforeDeviceAccess) {
  auto array = member();
  array.shape = {std::numeric_limits<std::uint64_t>::max()};
  EXPECT_THROW(validate_native_dense_array_layout(array), std::invalid_argument);
  array = member();
  array.element_shape = {0};
  EXPECT_THROW(validate_native_dense_array_layout(array), std::invalid_argument);
  array = member();
  array.shape = {2, 3};
  EXPECT_THROW(validate_native_dense_array_layout(array), std::invalid_argument);
  for (auto bad : {member(17, 20, -1), member(17, 20, 2), member(17, 21, 4),
                   member(17, -20, 4),
                   member(17, std::numeric_limits<std::int64_t>::max(), 4)}) {
    EXPECT_THROW(validate_native_dense_array_layout(bad), std::invalid_argument);
  }
  array = member();
  array.scalar_type = PrimitiveType::unknown;
  EXPECT_THROW(validate_native_dense_array_layout(array), std::invalid_argument);
  array = member();
  array.pointer.offset = 1;
  EXPECT_THROW(validate_native_dense_array_layout(array), std::invalid_argument);
  array.base_offset = std::numeric_limits<std::uint64_t>::max();
  EXPECT_NO_THROW(validate_native_dense_array_layout(array));  // Modular sum.
  array = member(17, 20, 3);
  array.pointer.offset = 1;
  EXPECT_TRUE(validate_native_dense_array_layout(array).properties.aligned);
  array = member();
  array.strides.reset();
  array.shape = {std::uint64_t(std::numeric_limits<std::int64_t>::max())};
  EXPECT_THROW(validate_native_dense_array_layout(array), std::invalid_argument);
}

TEST(NativeArrayContract, LayoutWorkDependsOnRankRatherThanRecordCount) {
  auto array = member(1000000000000ull);
  array.bytes = 20000000000000ull;
  const auto facts = validate_native_dense_array_layout(array);
  EXPECT_EQ(facts.properties.item_count, 1000000000000ull);
  EXPECT_EQ(facts.properties.reachable_end, 19999999999988ll);
}

TEST(NativeArrayContract, PackedTensorsEmptyAndReversedViewsShareLayoutMath) {
  auto array = member();
  array.strides.reset();
  array.shape = {2, 3};
  array.element_shape = {2, 2};
  array.bytes = 96;
  auto facts = validate_native_dense_array_layout(array);
  EXPECT_EQ(facts.layout.index_strides_bytes, (std::vector<std::int64_t>{48, 16}));
  EXPECT_EQ(facts.layout.element_strides_bytes, (std::vector<std::int64_t>{8, 4}));
  EXPECT_TRUE(facts.properties.compact_contiguous);
  array = member(0, 20, 0);
  array.bytes = 0;
  EXPECT_TRUE(validate_native_dense_array_layout(array).properties.empty);
  array = member(17, -20, 324);
  facts = validate_native_dense_array_layout(array);
  EXPECT_TRUE(facts.properties.has_negative_stride);
  EXPECT_EQ(facts.properties.reachable_begin, 4);
  EXPECT_EQ(facts.properties.reachable_end, 328);
  array = member(17, 0, 4);
  EXPECT_EQ(validate_native_dense_array_layout(array).properties.uniqueness,
            storage::StorageMappingUniqueness::kProvenOverlapping);
}

#ifdef TI_WITH_LLVM
TEST(NativeArrayContract, ResolvesLiveOwnerWindowAndCoordinateMapping) {
  cpu::CpuDevice device, other_device;
  auto [allocation, status] = device.allocate_memory_unique({340});
  auto [foreign, foreign_status] = other_device.allocate_memory_unique({340});
  ASSERT_EQ(status, RhiResult::success);
  ASSERT_EQ(foreign_status, RhiResult::success);
  auto array = member();
  array.pointer = allocation->get_ptr();
  auto a = resolve_native_dense_array(Arch::x64, array);
  EXPECT_EQ(a.range().bytes, 324);
  auto shifted = member(17, 20, 0);
  shifted.pointer = allocation->get_ptr(4);
  shifted.bytes = 336;
  auto b = resolve_native_dense_array(Arch::x64, shifted);
  EXPECT_TRUE(a.proven_same_mapping(b));
  EXPECT_FALSE(a.proven_disjoint(b));
  shifted = member(17, 20, 3);
  shifted.pointer = allocation->get_ptr(1);
  shifted.bytes = 339;
  EXPECT_TRUE(a.proven_same_mapping(
      resolve_native_dense_array(Arch::x64, shifted)));
  shifted.scalar_type = PrimitiveType::i32;
  EXPECT_FALSE(a.proven_same_mapping(
      resolve_native_dense_array(Arch::x64, shifted)));
  auto reversed = member(17, -20, 324);
  reversed.pointer = array.pointer;
  auto r = resolve_native_dense_array(Arch::x64, reversed);
  EXPECT_FALSE(a.proven_same_mapping(r));
  EXPECT_FALSE(a.proven_disjoint(r));
  array.pointer = foreign->get_ptr();
  auto f = resolve_native_dense_array(Arch::x64, array);
  EXPECT_FALSE(a.proven_disjoint(f));
  EXPECT_FALSE(a.proven_same_mapping(f));
  array.pointer = allocation->get_ptr();
  array.bytes = 341;
  EXPECT_THROW(resolve_native_dense_array(Arch::x64, array), std::invalid_argument);
  array.bytes = 328;
  array.pointer.offset = 20;
  EXPECT_THROW(resolve_native_dense_array(Arch::x64, array), std::invalid_argument);
}

TEST(NativeArrayContract, DisjointMembersCanHaveOverlappingBoundingRanges) {
  cpu::CpuDevice device;
  auto [allocation, status] = device.allocate_memory_unique({340});
  ASSERT_EQ(status, RhiResult::success);
  auto x = member(), y = member(17, 20, 8);
  x.pointer = y.pointer = allocation->get_ptr();
  auto a = resolve_native_dense_array(Arch::x64, x);
  auto b = resolve_native_dense_array(Arch::x64, y);
  EXPECT_FALSE(a.range().proven_disjoint(b.range()));
  EXPECT_TRUE(a.proven_disjoint(b));
  EXPECT_TRUE(b.proven_disjoint(a));
  y.scalar_type = PrimitiveType::u64;
  y.strides = std::make_shared<const NativeArrayStrides>(
      NativeArrayStrides{{24}, {}, 0});
  y.shape = {14};
  b = resolve_native_dense_array(Arch::x64, y);
  EXPECT_FALSE(a.proven_disjoint(b));
  // Empty mappings and physically separate allocations use range proof.
  y = member(0, 20, 340);
  y.pointer = allocation->get_ptr();
  EXPECT_TRUE(a.proven_disjoint(resolve_native_dense_array(Arch::x64, y)));
  auto [separate, separate_status] = device.allocate_memory_unique({340});
  ASSERT_EQ(separate_status, RhiResult::success);
  x.pointer = separate->get_ptr();
  EXPECT_TRUE(a.proven_disjoint(resolve_native_dense_array(Arch::x64, x)));
}

TEST(NativeArrayContract, AffineSeparationMatchesIndependentTinyAddressOracle) {
  cpu::CpuDevice device;
  auto [allocation, status] = device.allocate_memory_unique({4096});
  ASSERT_EQ(status, RhiResult::success);
  std::mt19937 random(888);
  auto sample = [&]() {
    auto array = member();
    array.bytes = 4096;
    array.pointer = allocation->get_ptr();
    array.scalar_type = random() % 2 ? PrimitiveType::u32 : PrimitiveType::u8;
    const auto width = data_type_size(array.scalar_type);
    array.shape = {random() % 4, 1 + random() % 4};
    array.element_shape = {1 + random() % 3};
    array.strides = std::make_shared<const NativeArrayStrides>(
        NativeArrayStrides{
            {std::int64_t(random() % 17) * width - 8 * width,
             std::int64_t(random() % 9) * width - 4 * width},
            {std::int64_t(random() % 5) * width - 2 * width},
            512 + std::int64_t(random() % 17) * width});
    return array;
  };
  auto addresses = [](const NativeArrayBinding &array) {
    std::set<std::int64_t> result;
    const auto &s = *array.strides;
    for (std::uint64_t i = 0; i < array.shape[0]; ++i)
      for (std::uint64_t j = 0; j < array.shape[1]; ++j)
        for (std::uint64_t k = 0; k < array.element_shape[0]; ++k)
          for (int byte = 0; byte < data_type_size(array.scalar_type); ++byte)
            result.insert(s.byte_offset + std::int64_t(i) * s.index_bytes[0] +
                          std::int64_t(j) * s.index_bytes[1] +
                          std::int64_t(k) * s.element_bytes[0] + byte);
    return result;
  };
  int proved = 0, overlapping = 0;
  for (int trial = 0; trial < 1000; ++trial) {
    auto x = sample(), y = sample();
    const auto a = resolve_native_dense_array(Arch::x64, x);
    const auto b = resolve_native_dense_array(Arch::x64, y);
    const auto ax = addresses(x), ay = addresses(y);
    bool intersects = false;
    for (auto address : ax)
      intersects |= ay.count(address) != 0;
    overlapping += intersects;
    EXPECT_EQ(a.proven_disjoint(b), b.proven_disjoint(a));
    if (a.proven_disjoint(b)) {
      ++proved;
      EXPECT_FALSE(intersects) << trial;
    }
    EXPECT_TRUE(a.proven_same_mapping(a));
  }
  EXPECT_GT(proved, 0);
  EXPECT_GT(overlapping, 0);
}
#endif

#ifdef TI_WITH_CUDA
TEST(NativeArrayContract, CudaMemberViewsResolvePhysicalStorageWithoutCommands) {
  if (!CUDADriver::get_instance_without_context().detected())
    GTEST_SKIP();
  cuda::CudaDevice device;
  auto guard = CUDAContext::get_instance().get_guard();
  auto [allocation, status] = device.allocate_memory_unique({340});
  ASSERT_EQ(status, RhiResult::success);
  auto x = member(), y = member(17, 20, 8);
  x.pointer = y.pointer = allocation->get_ptr();
  auto a = resolve_native_dense_array(Arch::cuda, x);
  auto b = resolve_native_dense_array(Arch::cuda, y);
  EXPECT_FALSE(a.range().proven_disjoint(b.range()));
  EXPECT_TRUE(a.proven_disjoint(b));
  EXPECT_FALSE(a.proven_same_mapping(b));
  y = member(17, 20, 0);
  y.pointer = allocation->get_ptr(4);
  y.bytes = 336;
  EXPECT_TRUE(a.proven_same_mapping(resolve_native_dense_array(Arch::cuda, y)));
  y.bytes = 337;
  EXPECT_THROW(resolve_native_dense_array(Arch::cuda, y), std::invalid_argument);
}
#endif
}  // namespace
}  // namespace taichi::lang
